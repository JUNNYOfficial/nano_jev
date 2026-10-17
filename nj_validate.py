"""标签契约校验：把"什么算合法标签"写成可执行的规则。

归一化（:mod:`nano_jev.nj_hybrid`）解决的是"模型给了不合规输出怎么办"；本模块
回答另一个问题——"**这份标签到底合不合契约**"。两者用途不同：

* 归一化用于运行时：尽量把模型输出收敛到词表内，让请求继续；
* 校验用于边界：入口、评测数据集、离线产物、跨服务传输。这里**不修数据**，
  只列出问题，由调用方决定是拒绝、告警还是修正。

这样"校验"不会悄悄把错误数据改掉——那会让问题在更下游才暴露。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .nj_engine import Label
from .nj_vocab import ASPECTS, EMOTIONS, INTENTS, MAX_ASPECTS, MAX_EMOTIONS, SENTIMENTS


@dataclass(frozen=True)
class LabelIssue:
    """一处契约问题；``field`` 为空表示跨字段/整体问题。"""

    field: str
    reason: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        """转成可写日志/JSON 的扁平字典。"""
        return {"field": self.field, "reason": self.reason, "value": self.value}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_label(
    label: Label,
    *,
    source_text: str | None = None,
    strict_quote: bool = False,
) -> tuple[LabelIssue, ...]:
    """校验一个标签是否满足受控词表与字段约束；返回问题列表（空即可用）。

    ``source_text`` 给出时校验 ``evidence_quote`` 是否为原文逐字子串；
    ``strict_quote=True`` 时**空引文也算问题**（用于要求"每条判断都要有证据"的
    场景），默认允许空引文（例如纯表情样本）。
    """
    issues: list[LabelIssue] = []

    if not isinstance(label.comment_id, str) or not label.comment_id:
        issues.append(LabelIssue("comment_id", "must be a non-empty string", label.comment_id))

    if not isinstance(label.is_relevant, bool):
        issues.append(LabelIssue("is_relevant", "must be a bool", label.is_relevant))

    if label.sentiment not in SENTIMENTS:
        issues.append(
            LabelIssue("sentiment", f"must be one of {list(SENTIMENTS)}", label.sentiment)
        )
    if label.intent not in INTENTS:
        issues.append(LabelIssue("intent", f"must be one of {list(INTENTS)}", label.intent))

    if not _finite(label.sentiment_score):
        issues.append(LabelIssue("sentiment_score", "must be a finite number", label.sentiment_score))
    elif not -1.0 <= float(label.sentiment_score) <= 1.0:
        issues.append(
            LabelIssue("sentiment_score", "must be within [-1, 1]", label.sentiment_score)
        )

    if not _finite(label.confidence):
        issues.append(LabelIssue("confidence", "must be a finite number", label.confidence))
    elif not 0.0 <= float(label.confidence) <= 1.0:
        issues.append(LabelIssue("confidence", "must be within [0, 1]", label.confidence))

    issues.extend(_validate_list("aspects", label.aspects, ASPECTS, MAX_ASPECTS))
    issues.extend(_validate_list("emotion", label.emotion, EMOTIONS, MAX_EMOTIONS))

    if not isinstance(label.evidence_quote, str):
        issues.append(LabelIssue("evidence_quote", "must be a string", label.evidence_quote))
    else:
        quote = label.evidence_quote
        if strict_quote and not quote.strip():
            issues.append(LabelIssue("evidence_quote", "must not be empty", quote))
        if quote and source_text is not None and quote not in source_text:
            issues.append(
                LabelIssue("evidence_quote", "must be a verbatim substring of the source", quote)
            )

    if not isinstance(label.meta, Mapping):
        issues.append(LabelIssue("meta", "must be a mapping", type(label.meta).__name__))

    return tuple(issues)


def _validate_list(
    field: str, values: Any, allowed: Sequence[str] | Mapping[str, Any], limit: int
) -> list[LabelIssue]:
    """校验枚举列表：元素合法、无重复、不超上限。"""
    issues: list[LabelIssue] = []
    if not isinstance(values, (list, tuple)):
        return [LabelIssue(field, "must be a list", values)]
    allowed_set = set(allowed)
    seen: set[str] = set()
    for item in values:
        if item not in allowed_set:
            issues.append(LabelIssue(field, "value outside the controlled vocabulary", item))
        elif item in seen:
            issues.append(LabelIssue(field, "duplicate value", item))
        else:
            seen.add(item)
    if len(values) > limit:
        issues.append(LabelIssue(field, f"must contain at most {limit} values", len(values)))
    return issues


def is_valid_label(
    label: Label, *, source_text: str | None = None, strict_quote: bool = False
) -> bool:
    """``validate_label`` 的布尔快捷方式。"""
    return not validate_label(label, source_text=source_text, strict_quote=strict_quote)


def require_valid_label(
    label: Label, *, source_text: str | None = None, strict_quote: bool = False
) -> Label:
    """不合法时抛 ``ValueError``（带全部问题），合法则原样返回。"""
    issues = validate_label(label, source_text=source_text, strict_quote=strict_quote)
    if issues:
        detail = "; ".join(f"{issue.field}: {issue.reason}" for issue in issues)
        raise ValueError(f"invalid label: {detail}")
    return label


#: 原始标签字典允许出现的字段（含 ``meta``；``comment_id`` 由调用方决定是否携带）。
LABEL_FIELDS: tuple[str, ...] = (
    "comment_id",
    "is_relevant",
    "sentiment",
    "sentiment_score",
    "confidence",
    "intent",
    "aspects",
    "emotion",
    "evidence_quote",
    "meta",
)


def validate_label_mapping(data: Any, *, allow_partial: bool = True) -> tuple[LabelIssue, ...]:
    """直接校验**原始字典**，不经过归一化。

    这是校验与归一化的关键差别：``label_from_mapping`` 会把 ``"Positive"`` 改成
    ``"positive"``、把词表外维度丢掉，于是"原始输入是否合法"就再也看不出来了。
    数据集、跨服务响应这类**基准数据**必须用本函数校验。

    ``allow_partial=True`` 时只校验出现的字段（数据集里 ``gold`` 常只标其中几列）；
    设为 ``False`` 则要求 ``LABEL_FIELDS`` 里的关键字段齐全。
    """
    if not isinstance(data, Mapping):
        return (LabelIssue("", "label must be a mapping", type(data).__name__),)

    issues: list[LabelIssue] = []
    unknown = sorted(set(data) - set(LABEL_FIELDS))
    if unknown:
        issues.append(LabelIssue("", f"unknown field(s) {unknown}", unknown))

    if not allow_partial:
        missing = [
            name
            for name in ("comment_id", "is_relevant", "sentiment", "intent")
            if name not in data
        ]
        if missing:
            issues.append(LabelIssue("", f"missing required field(s) {missing}", missing))

    _check_enum(issues, data, "sentiment", SENTIMENTS)
    _check_enum(issues, data, "intent", INTENTS)
    _check_list(issues, data, "aspects", ASPECTS, MAX_ASPECTS)
    _check_list(issues, data, "emotion", EMOTIONS, MAX_EMOTIONS)

    if "is_relevant" in data and not isinstance(data["is_relevant"], bool):
        issues.append(LabelIssue("is_relevant", "must be a bool", data["is_relevant"]))

    for name, low, high in (("sentiment_score", -1.0, 1.0), ("confidence", 0.0, 1.0)):
        if name not in data:
            continue
        value = data[name]
        if not _finite(value):
            issues.append(LabelIssue(name, "must be a finite number", value))
        elif not low <= float(value) <= high:
            issues.append(LabelIssue(name, f"must be within [{low}, {high}]", value))

    if "evidence_quote" in data and not isinstance(data["evidence_quote"], str):
        issues.append(LabelIssue("evidence_quote", "must be a string", data["evidence_quote"]))
    if "meta" in data and not isinstance(data["meta"], Mapping):
        issues.append(LabelIssue("meta", "must be a mapping", type(data["meta"]).__name__))

    return tuple(issues)


def _check_enum(
    issues: list[LabelIssue], data: Mapping[str, Any], field: str, allowed: Sequence[str]
) -> None:
    if field not in data:
        return
    value = data[field]
    if not isinstance(value, str) or value not in allowed:
        issues.append(LabelIssue(field, f"must be one of {list(allowed)}", value))


def _check_list(
    issues: list[LabelIssue],
    data: Mapping[str, Any],
    field: str,
    allowed: Sequence[str] | Mapping[str, Any],
    limit: int,
) -> None:
    if field not in data:
        return
    issues.extend(_validate_list(field, data[field], allowed, limit))
