"""级联判断编排层的行为测试。"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
from nano_jev import (  # noqa: E402
    BudgetExceededError,
    HybridBudget,
    HybridPolicy,
    HybridResult,
    Label,
    classify,
    judge_hybrid,
    judge_many,
)


def _llm_label(comment_id: str) -> Label:
    return Label(
        comment_id=comment_id,
        is_relevant=True,
        sentiment="positive",
        sentiment_score=0.8,
        confidence=0.9,
        intent="praise",
        aspects=["capability"],
        emotion=["认可"],
    )


def test_confident_sample_stays_on_deterministic_path():
    result = judge_hybrid(
        "c0",
        "这个大模型回答准确，能力很强，真的非常推荐",
        llm_judge=lambda *_: pytest.fail("LLM should not be called"),
    )

    assert isinstance(result, HybridResult)
    assert result.source == "nano-jev"
    assert result.escalated is False
    assert result.escalation_reasons == ()
    assert result.label.meta["hybrid"]["source"] == "nano-jev"


def test_low_heuristic_confidence_escalates_and_records_provenance():
    nano_label = classify("c1", "这个模型还行")
    received: list[Label] = []

    def llm_judge(content: str, hint: Label) -> Label:
        assert content == "这个模型还行"
        received.append(hint)
        return _llm_label(hint.comment_id)

    result = judge_hybrid("c1", "这个模型还行", llm_judge=llm_judge)

    assert result.source == "llm"
    assert result.escalated is True
    assert result.escalation_reasons == ("low_heuristic_confidence",)
    assert result.label.sentiment == "positive"
    assert result.label.meta["hybrid"]["escalated"] is True
    assert received[0] == nano_label


def test_mixed_sentiment_escalates_even_above_confidence_threshold():
    result = judge_hybrid(
        "c2",
        "这个模型回答准确，但价格太贵，能力不错但速度很慢",
        llm_judge=lambda _content, hint: _llm_label(hint.comment_id),
        policy=HybridPolicy(minimum_confidence=0.0),
    )

    assert result.escalated is True
    assert "mixed_sentiment" in result.escalation_reasons


def test_escalation_without_model_callback_fails_explicitly():
    with pytest.raises(RuntimeError, match="no llm_judge callback"):
        judge_hybrid("c0", "这个模型还行")


def test_invalid_model_output_is_rejected():
    with pytest.raises(TypeError, match="must return"):
        judge_hybrid("c0", "这个模型还行", llm_judge=lambda *_: "positive")  # type: ignore[arg-type]


def test_model_cannot_change_comment_identity():
    with pytest.raises(ValueError, match="expected 'c0'"):
        judge_hybrid("c0", "这个模型还行", llm_judge=lambda *_: _llm_label("wrong"))


def test_policy_validates_confidence_threshold():
    with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
        HybridPolicy(minimum_confidence=1.1)


def test_policy_validates_other_route_settings():
    with pytest.raises(ValueError, match="minimum_content_length"):
        HybridPolicy(minimum_content_length=-1)
    with pytest.raises(ValueError, match="on_budget_exhausted"):
        HybridPolicy(on_budget_exhausted="ignore")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="on_llm_error"):
        HybridPolicy(on_llm_error="ignore")  # type: ignore[arg-type]


def test_model_output_is_normalized_to_controlled_vocabulary():
    def llm_judge(_content: str, hint: Label) -> Label:
        return Label(
            comment_id=hint.comment_id,
            is_relevant=True,
            sentiment="Positive",
            sentiment_score=2.0,
            confidence=1.4,
            intent="praise!!",
            aspects=["capability", "not-a-real-aspect"],
            emotion=["curiosity"],
            evidence_quote="not in the source text",
        )

    result = judge_hybrid("c0", "这个模型还行", llm_judge=llm_judge)

    assert result.label.sentiment == "positive"
    assert result.label.intent == "other"
    assert result.label.aspects == ["capability"]
    assert result.label.emotion == ["期待"]
    assert result.label.sentiment_score == 1.0
    assert result.label.confidence == 1.0
    assert result.label.evidence_quote == "这个模型还行"
    assert set(result.normalizations) >= {
        "sentiment",
        "intent",
        "aspects",
        "sentiment_score",
        "confidence",
        "evidence_quote",
    }
    assert result.coarse_label is not None
    assert result.coarse_label.to_raw()["sentiment"] == classify("c0", "这个模型还行").sentiment
    assert result.llm_duration_ms is not None
    assert result.llm_duration_ms >= 0.0


def test_non_numeric_score_is_rejected_as_contract_error():
    def llm_judge(_content: str, hint: Label) -> Label:
        label = _llm_label(hint.comment_id)
        label.confidence = "high"  # type: ignore[assignment]
        return label

    with pytest.raises(ValueError, match="non-numeric confidence"):
        judge_hybrid("c0", "这个模型还行", llm_judge=llm_judge)


def test_llm_error_propagates_by_default():
    def boom(_content: str, _hint: Label) -> Label:
        raise TimeoutError("slow model")

    with pytest.raises(TimeoutError, match="slow model"):
        judge_hybrid("c0", "这个模型还行", llm_judge=boom)


def test_llm_error_can_fall_back_to_coarse_result():
    def boom(_content: str, _hint: Label) -> Label:
        raise TimeoutError("slow model")

    result = judge_hybrid(
        "c0",
        "这个模型还行",
        llm_judge=boom,
        policy=HybridPolicy(on_llm_error="nano"),
    )

    assert result.source == "nano-jev"
    assert result.escalated is False
    assert result.llm_error == "TimeoutError"
    assert "llm_error" in result.escalation_reasons
    assert result.label.meta["hybrid"]["llm_error"] == "TimeoutError"


def test_budget_downgrades_after_limit():
    budget = HybridBudget(limit=1)
    callback = lambda _content, hint: _llm_label(hint.comment_id)  # noqa: E731

    first = judge_hybrid("c0", "这个模型还行", llm_judge=callback, budget=budget)
    second = judge_hybrid("c1", "这个模型还行", llm_judge=callback, budget=budget)

    assert first.source == "llm"
    assert second.source == "nano-jev"
    assert "budget_exhausted" in second.escalation_reasons
    assert second.label.meta["hybrid"]["budget_exhausted"] is True
    assert budget.used == 1
    assert budget.remaining == 0


def test_budget_can_fail_loudly():
    budget = HybridBudget(limit=0)

    with pytest.raises(BudgetExceededError, match="budget exhausted"):
        judge_hybrid(
            "c0",
            "这个模型还行",
            llm_judge=lambda _content, hint: _llm_label(hint.comment_id),
            policy=HybridPolicy(on_budget_exhausted="raise"),
            budget=budget,
        )


def test_short_and_irrelevant_routes_add_reasons():
    short = judge_hybrid(
        "c0",
        "还行",
        llm_judge=lambda _content, hint: _llm_label(hint.comment_id),
        policy=HybridPolicy(minimum_content_length=10),
    )
    irrelevant = judge_hybrid(
        "c1",
        "😂😂",
        llm_judge=lambda _content, hint: _llm_label(hint.comment_id),
        policy=HybridPolicy(minimum_confidence=0.0, escalate_irrelevant=True),
    )

    assert "short_content" in short.escalation_reasons
    assert "irrelevant" in irrelevant.escalation_reasons


def test_missing_aspects_can_trigger_escalation():
    result = judge_hybrid(
        "c0",
        "这个模型还行",
        llm_judge=lambda _content, hint: _llm_label(hint.comment_id),
        policy=HybridPolicy(minimum_confidence=0.0, escalate_without_aspects=True),
    )

    assert "no_aspects" in result.escalation_reasons


def test_judge_many_preserves_order_and_shares_budget():
    comments = [
        {"comment_id": "c0", "content": "这个模型还行"},
        {"comment_id": "c1", "content": "这个模型还行"},
    ]

    results = judge_many(
        comments,
        llm_judge=lambda _content, hint: _llm_label(hint.comment_id),
        budget=HybridBudget(limit=1),
    )

    assert [result.label.comment_id for result in results] == ["c0", "c1"]
    assert [result.source for result in results] == ["llm", "nano-jev"]


def test_budget_validates_limits():
    with pytest.raises(ValueError, match="limit"):
        HybridBudget(limit=-1)
    with pytest.raises(ValueError, match="used"):
        HybridBudget(limit=1, used=-1)
