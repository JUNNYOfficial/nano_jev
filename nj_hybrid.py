"""轻量粗筛与大模型之间的级联判断编排层。

本模块不绑定任何模型供应商或 SDK。调用方注入一个把模型结构化输出映射为
``Label`` 的回调；低风险样本保留 nano_jev 的确定性结果，疑难样本才升级。

模型输出一律按 :mod:`nano_jev.nj_vocab` 的受控词表校验：越界枚举、越界分数、
伪造引文和词表外维度不会原样进入下游。升级失败或预算耗尽时，策略显式决定是
退回确定性粗筛结果，还是直接抛错，不做静默兜底。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Literal, Protocol

from .nj_engine import Label, classify
from .nj_vocab import INTENTS, SENTIMENTS, to_aspects, to_emotions, to_enum


class LLMJudge(Protocol):
    """将评论和粗筛线索交给大模型，并返回受控结构化标签。"""

    def __call__(self, content: str, nano_label: Label, /) -> Label: ...


class BudgetExceededError(RuntimeError):
    """升级预算已耗尽，且策略要求直接抛错。"""


@dataclass
class HybridBudget:
    """跨请求复用的升级次数预算。

    只统计真正尝试调用大模型的样本；被粗筛直接接受的样本不占用预算。
    预算耗尽时若策略选择 ``"nano"``，结果会带 ``budget_exhausted`` 原因，
    便于离线统计被降级的样本。
    """

    limit: int
    used: int = 0

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("limit must be >= 0")
        if self.used < 0:
            raise ValueError("used must be >= 0")

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def consume(self) -> bool:
        """尝试占用一次升级额度；成功返回 ``True``。"""
        if self.exhausted:
            return False
        self.used += 1
        return True


@dataclass(frozen=True)
class HybridPolicy:
    """级联路由规则。阈值是启发式路由门槛，不是校准后的概率。"""

    minimum_confidence: float = 0.65
    escalate_mixed: bool = True
    minimum_content_length: int = 0
    escalate_irrelevant: bool = False
    escalate_without_aspects: bool = False
    on_budget_exhausted: Literal["nano", "raise"] = "nano"
    on_llm_error: Literal["raise", "nano"] = "raise"

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0.0 and 1.0")
        if self.minimum_content_length < 0:
            raise ValueError("minimum_content_length must be >= 0")
        if self.on_budget_exhausted not in ("nano", "raise"):
            raise ValueError("on_budget_exhausted must be 'nano' or 'raise'")
        if self.on_llm_error not in ("raise", "nano"):
            raise ValueError("on_llm_error must be 'raise' or 'nano'")

    def escalation_reasons(
        self, label: Label, content_length: int | None = None
    ) -> tuple[str, ...]:
        """返回触发升级的原因；空元组表示直接采用粗筛结果。"""
        reasons: list[str] = []
        if label.confidence < self.minimum_confidence:
            reasons.append("low_heuristic_confidence")
        if self.escalate_mixed and label.sentiment == "mixed":
            reasons.append("mixed_sentiment")
        if content_length is not None and content_length < self.minimum_content_length:
            reasons.append("short_content")
        if self.escalate_irrelevant and not label.is_relevant:
            reasons.append("irrelevant")
        if self.escalate_without_aspects and not label.aspects:
            reasons.append("no_aspects")
        return tuple(reasons)


@dataclass(frozen=True)
class HybridResult:
    """级联判断结果及其决策来源。"""

    label: Label
    source: Literal["nano-jev", "llm"]
    escalated: bool
    escalation_reasons: tuple[str, ...]
    coarse_label: Label | None = None
    llm_duration_ms: float | None = None
    normalizations: tuple[str, ...] = ()
    llm_error: str | None = None


def _copy_label(label: Label) -> Label:
    """复制可变字段，避免路由回调改写粗筛器的原始标签。"""
    return replace(
        label,
        aspects=list(label.aspects),
        emotion=list(label.emotion),
        meta=dict(label.meta),
    )


def _number(value: Any, name: str, low: float, high: float) -> tuple[float, bool]:
    """把模型返回的分数收敛到有限区间；非数值直接判为契约错误。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"llm_judge returned a non-numeric {name}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"llm_judge returned a non-finite {name}: {value!r}")
    clamped = min(high, max(low, number))
    return clamped, clamped != number


def _normalize_model_label(
    model_label: Label, nano_label: Label, body: str
) -> tuple[Label, tuple[str, ...]]:
    """把模型标签收敛到受控词表，并记录发生过的规范化。"""
    normalizations: list[str] = []
    if str(model_label.comment_id) != nano_label.comment_id:
        raise ValueError(
            "llm_judge returned a Label with comment_id "
            f"{model_label.comment_id!r}; expected {nano_label.comment_id!r}"
        )
    if not isinstance(model_label.comment_id, str):
        normalizations.append("comment_id")

    sentiment, changed = to_enum(
        model_label.sentiment, SENTIMENTS, fallback="neutral"
    )
    if changed:
        normalizations.append("sentiment")

    intent, changed = to_enum(model_label.intent, INTENTS, fallback="other")
    if changed:
        normalizations.append("intent")

    aspects, dropped, truncated = to_aspects(model_label.aspects)
    if dropped or truncated or not isinstance(model_label.aspects, (list, tuple)):
        normalizations.append("aspects")

    emotion, dropped, truncated = to_emotions(model_label.emotion)
    if dropped or truncated or not isinstance(model_label.emotion, (list, tuple)):
        normalizations.append("emotion")

    sentiment_score, changed = _number(
        model_label.sentiment_score, "sentiment_score", -1.0, 1.0
    )
    if changed:
        normalizations.append("sentiment_score")

    confidence, changed = _number(model_label.confidence, "confidence", 0.0, 1.0)
    if changed:
        normalizations.append("confidence")

    relevant = model_label.is_relevant
    if not isinstance(relevant, bool):
        relevant = bool(relevant)
        normalizations.append("is_relevant")

    quote = model_label.evidence_quote
    if not isinstance(quote, str):
        quote = str(quote)
        normalizations.append("evidence_quote")
    if quote and quote not in body:
        quote = nano_label.evidence_quote
        normalizations.append("evidence_quote")

    if isinstance(model_label.meta, Mapping):
        meta = dict(model_label.meta)
    else:
        meta = {}
        normalizations.append("meta")

    normalized = replace(
        model_label,
        comment_id=nano_label.comment_id,
        is_relevant=relevant,
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        confidence=confidence,
        intent=intent,
        aspects=aspects,
        emotion=emotion,
        evidence_quote=quote,
        meta=meta,
    )
    return normalized, tuple(normalizations)


def _nano_result(
    nano_label: Label,
    reasons: tuple[str, ...],
    *,
    llm_error: str | None = None,
    budget_exhausted: bool = False,
) -> HybridResult:
    accepted = _copy_label(nano_label)
    hybrid_meta: dict[str, Any] = {
        "source": "nano-jev",
        "escalated": False,
        "escalation_reasons": list(reasons),
    }
    if llm_error is not None:
        hybrid_meta["llm_error"] = llm_error
    if budget_exhausted:
        hybrid_meta["budget_exhausted"] = True
    accepted.meta["hybrid"] = hybrid_meta
    return HybridResult(
        accepted,
        "nano-jev",
        False,
        reasons,
        _copy_label(nano_label),
        llm_error=llm_error,
    )


def judge_hybrid(
    comment_id: object,
    content: object,
    topic_title: object = "",
    like_count: object = 0,
    topics: Sequence[str] | None = None,
    *,
    llm_judge: LLMJudge | None = None,
    policy: HybridPolicy | None = None,
    budget: HybridBudget | None = None,
) -> HybridResult:
    """先运行 nano_jev，只有路由策略要求升级时才调用大模型。

    若样本需要升级但没有提供 ``llm_judge``，显式抛出 ``RuntimeError``，不把
    未复核的粗筛结果伪装成大模型结论。模型异常默认向上抛；设置
    ``on_llm_error="nano"`` 才退回粗筛结果，并记录异常类型。
    """
    body = "" if content is None else str(content)
    nano_label = classify(
        comment_id,
        body,
        topic_title=topic_title,
        like_count=like_count,
        topics=topics,
    )
    active_policy = policy if policy is not None else HybridPolicy()
    reasons = active_policy.escalation_reasons(nano_label, len(body.strip()))

    if not reasons:
        return _nano_result(nano_label, reasons)

    if llm_judge is None:
        raise RuntimeError(
            "nano_jev requested LLM escalation "
            f"({', '.join(reasons)}), but no llm_judge callback was provided"
        )

    if budget is not None and not budget.consume():
        if active_policy.on_budget_exhausted == "raise":
            raise BudgetExceededError(
                f"LLM escalation budget exhausted after {budget.used} call(s)"
            )
        return _nano_result(
            nano_label,
            (*reasons, "budget_exhausted"),
            budget_exhausted=True,
        )

    started = perf_counter()
    try:
        model_label = llm_judge(body, _copy_label(nano_label))
    except Exception as exc:
        if active_policy.on_llm_error == "raise":
            raise
        return _nano_result(
            nano_label,
            (*reasons, "llm_error"),
            llm_error=type(exc).__name__,
        )
    duration_ms = (perf_counter() - started) * 1000.0

    if not isinstance(model_label, Label):
        raise TypeError("llm_judge must return a nano_jev Label")

    normalized, normalizations = _normalize_model_label(model_label, nano_label, body)
    normalized.meta["hybrid"] = {
        "source": "llm",
        "escalated": True,
        "escalation_reasons": list(reasons),
        "normalizations": list(normalizations),
    }
    return HybridResult(
        normalized,
        "llm",
        True,
        reasons,
        _copy_label(nano_label),
        duration_ms,
        normalizations,
    )


def judge_many(
    comments: Sequence[Mapping[str, object]],
    *,
    topics: Sequence[str] | None = None,
    llm_judge: LLMJudge | None = None,
    policy: HybridPolicy | None = None,
    budget: HybridBudget | None = None,
) -> list[HybridResult]:
    """按输入顺序批量判断；``budget`` 在所有条目之间共享。"""
    return [
        judge_hybrid(
            item.get("comment_id", index),
            item.get("content", ""),
            topic_title=item.get("topic_title", ""),
            like_count=item.get("like_count", 0),
            topics=topics,
            llm_judge=llm_judge,
            policy=policy,
            budget=budget,
        )
        for index, item in enumerate(comments)
    ]
