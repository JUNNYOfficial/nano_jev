"""标签契约校验的行为测试。"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
from nano_jev.nj_engine import Label  # noqa: E402
from nano_jev.nj_validate import (  # noqa: E402
    is_valid_label,
    require_valid_label,
    validate_label,
    validate_label_mapping,
)


def _label(**overrides: object) -> Label:
    """构造测试标签；``overrides`` 允许写出非法值（校验测试需要）。"""
    values: dict[str, object] = {
        "comment_id": "c0",
        "is_relevant": True,
        "sentiment": "positive",
        "sentiment_score": 0.5,
        "confidence": 0.8,
        "intent": "praise",
        "aspects": ["capability"],
        "emotion": ["认可"],
        "evidence_quote": "很好用",
        "meta": {},
    }
    values.update(overrides)
    return Label(**values)  # type: ignore[arg-type]


def test_valid_label_has_no_issues():
    assert validate_label(_label()) == ()
    assert is_valid_label(_label()) is True
    assert require_valid_label(_label()) == _label()


def test_out_of_vocabulary_values_are_reported():
    issues = validate_label(
        _label(sentiment="Positive", intent="praise!!", aspects=["nope"])
    )
    fields = {issue.field for issue in issues}

    assert "sentiment" in fields
    assert "intent" in fields
    assert "aspects" in fields
    assert all("controlled vocabulary" in issue.reason or "one of" in issue.reason
               for issue in issues if issue.field in fields)


def test_numeric_ranges_are_enforced():
    issues = validate_label(_label(sentiment_score=2.0, confidence=-0.1))
    fields = {issue.field for issue in issues}

    assert fields == {"sentiment_score", "confidence"}


def test_non_finite_and_non_numeric_scores_are_reported():
    issues = validate_label(_label(confidence=float("nan"), sentiment_score="high"))

    assert {issue.field for issue in issues} == {"confidence", "sentiment_score"}


def test_duplicate_and_excess_aspects_are_reported():
    duplicates = validate_label(_label(aspects=["capability", "capability"]))
    assert any("duplicate" in issue.reason for issue in duplicates)

    too_many = validate_label(_label(aspects=["capability", "price", "quality",
                                              "usability", "reliability", "trust"]))
    assert any("at most" in issue.reason for issue in too_many)


def test_lists_must_be_lists():
    issues = validate_label(_label(aspects="capability", emotion=None))
    assert {issue.field for issue in issues} == {"aspects", "emotion"}


def test_comment_id_must_be_a_non_empty_string():
    assert validate_label(_label(comment_id="")) != ()
    assert validate_label(_label(comment_id=123)) != ()  # type: ignore[arg-type]


def test_is_relevant_must_be_bool():
    issues = validate_label(_label(is_relevant=1))
    assert [issue.field for issue in issues] == ["is_relevant"]


def test_quote_must_be_a_verbatim_substring():
    text = "这个模型回答很准确，速度快"
    assert validate_label(_label(evidence_quote="回答很准确"), source_text=text) == ()

    issues = validate_label(_label(evidence_quote="回答非常准确"), source_text=text)
    assert [issue.field for issue in issues] == ["evidence_quote"]
    assert "verbatim" in issues[0].reason


def test_strict_quote_rejects_empty_evidence():
    assert validate_label(_label(evidence_quote=""), source_text="原文") == ()

    issues = validate_label(_label(evidence_quote=""), source_text="原文", strict_quote=True)
    assert [issue.field for issue in issues] == ["evidence_quote"]
    assert "empty" in issues[0].reason


def test_meta_must_be_a_mapping():
    issues = validate_label(_label(meta=["not", "a", "dict"]))
    assert [issue.field for issue in issues] == ["meta"]


def test_require_valid_label_raises_with_detail():
    with pytest.raises(ValueError, match=r"invalid label:.*sentiment"):
        require_valid_label(_label(sentiment="positive!!"))


def test_issue_serializes_to_json_friendly_dict():
    issue = validate_label(_label(confidence=5.0))[0]
    payload = issue.to_dict()

    assert payload == {
        "field": "confidence",
        "reason": "must be within [0, 1]",
        "value": 5.0,
    }
    import json

    json.dumps(payload)


def test_mapping_validation_does_not_normalize():
    """回归：数据集里的 ``Positive`` / 词表外维度必须被拒绝，而不是被悄悄改掉。"""
    issues = validate_label_mapping(
        {"sentiment": "Positive", "intent": "praise", "aspects": ["nope"]}
    )
    fields = {issue.field for issue in issues}

    assert fields == {"sentiment", "aspects"}


def test_mapping_validation_accepts_partial_gold():
    assert validate_label_mapping({"sentiment": "positive"}) == ()
    assert validate_label_mapping({"aspects": ["capability", "price"]}) == ()


def test_mapping_validation_rejects_unknown_fields():
    issues = validate_label_mapping({"sentimen": "positive"})
    assert issues[0].field == ""
    assert "unknown field" in issues[0].reason


def test_mapping_validation_can_require_all_fields():
    issues = validate_label_mapping({"sentiment": "positive"}, allow_partial=False)
    assert any("missing required field" in issue.reason for issue in issues)


def test_mapping_validation_checks_numeric_ranges_and_types():
    issues = validate_label_mapping(
        {"sentiment_score": 5, "confidence": "high", "is_relevant": 1}
    )
    fields = {issue.field for issue in issues}

    assert fields == {"sentiment_score", "confidence", "is_relevant"}


def test_mapping_validation_rejects_non_mapping():
    issues = validate_label_mapping(["sentiment"])
    assert issues[0].field == ""
    assert "mapping" in issues[0].reason


def test_mapping_validation_detects_duplicates_and_limits():
    assert any(
        "duplicate" in issue.reason
        for issue in validate_label_mapping({"aspects": ["price", "price"]})
    )
