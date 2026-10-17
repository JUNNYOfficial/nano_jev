"""字段级仲裁的行为测试。"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
from nano_jev.nj_arbitrate import (  # noqa: E402
    FIELDS,
    arbitration_fields,
    field_agreement,
    label_fields,
    merge_labels,
)
from nano_jev.nj_engine import Label  # noqa: E402


def _label(
    *,
    sentiment: str = "neutral",
    intent: str = "information",
    aspects: list[str] | None = None,
    confidence: float = 0.6,
) -> Label:
    return Label(
        comment_id="c0",
        is_relevant=True,
        sentiment=sentiment,
        sentiment_score=0.0,
        confidence=confidence,
        intent=intent,
        aspects=["capability"] if aspects is None else aspects,
        emotion=[],
        evidence_quote="原文",
    )


def test_agreement_treats_aspects_as_a_set():
    left = _label(aspects=["capability", "price"])
    right = _label(aspects=["price", "capability"])

    agreement = field_agreement(left, right)

    assert agreement["aspects"] is True
    assert agreement["sentiment"] is True


def test_merge_takes_only_requested_fields():
    coarse = _label(sentiment="neutral", intent="information")
    model = _label(sentiment="positive", intent="praise", confidence=0.95)

    merged = merge_labels(coarse, model, ("sentiment",))

    assert merged.sentiment == "positive"
    assert merged.intent == "information"  # 未请求，保留粗筛
    assert merged.confidence == coarse.confidence  # 未整条采纳，不继承模型置信度
    assert arbitration_fields(merged) == ("sentiment",)


def test_merge_records_conflicts():
    coarse = _label(sentiment="neutral")
    model = _label(sentiment="positive")

    merged = merge_labels(coarse, model, ("sentiment", "intent"))

    record = merged.meta["arbitration"]
    assert record["conflicts"] == ["sentiment"]
    assert record["adopted"] == ["sentiment", "intent"]


def test_conflict_policy_can_keep_coarse_value():
    coarse = _label(sentiment="neutral")
    model = _label(sentiment="positive")

    merged = merge_labels(
        coarse, model, ("sentiment",), conflict="prefer_coarse"
    )

    assert merged.sentiment == "neutral"
    assert merged.meta["arbitration"]["conflicts"] == ["sentiment"]


def test_full_merge_adopts_model_confidence():
    coarse = _label(confidence=0.4)
    model = _label(confidence=0.9, sentiment="positive")

    merged = merge_labels(coarse, model)

    assert merged.sentiment == "positive"
    assert merged.confidence == 0.9
    assert arbitration_fields(merged) == FIELDS


def test_merge_copies_mutable_containers():
    coarse = _label()
    model = _label(aspects=["price"])

    merged = merge_labels(coarse, model, ("aspects",))
    merged.aspects.append("trust")

    assert "trust" not in coarse.aspects
    assert model.aspects == ["price"]


def test_merge_rejects_unknown_fields_and_policy():
    with pytest.raises(ValueError, match="unknown field"):
        merge_labels(_label(), _label(), ("nope",))
    with pytest.raises(ValueError, match="conflict must be"):
        merge_labels(_label(), _label(), ("sentiment",), conflict="ignore")  # type: ignore[arg-type]


def test_label_fields_excludes_meta():
    fields = label_fields()

    assert "meta" not in fields
    assert "sentiment" in fields


def test_arbitration_fields_tolerates_missing_or_malformed_meta():
    assert arbitration_fields(_label()) == ()
    broken = _label()
    broken.meta["arbitration"] = {"adopted": "sentiment"}
    assert arbitration_fields(broken) == ()
    broken.meta["arbitration"] = "nope"
    assert arbitration_fields(broken) == ()
