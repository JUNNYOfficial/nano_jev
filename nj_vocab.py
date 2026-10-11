"""受控词表：与 jev-arena 的 ``src/vocab.mjs`` 逐项对齐。

**为什么镜像一份**：本子系统要作为 Jev 的**平替**接进 jev-arena 的对比场，两边
产出的标签必须落在**同一套枚举**上，否则报告里「Jev vs 平替」无法逐字段对比。
上游改词表时改这里，并跑 ``test_nj_contract.py`` 的对齐守门。

对齐口径来自 jev-arena 仓库（2026-10-11 校对）：``SENTIMENTS`` / ``STANCES`` /
``INTENTS`` / ``ASPECTS`` / ``EMOTIONS`` 与两个上限常量均取自其 ``vocab.mjs``。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

#: 整体态度（单选）。
SENTIMENTS: tuple[str, ...] = ("positive", "negative", "neutral", "mixed")

#: 立场（单选）。jev-arena 保留但当前未在主流程使用，这里同样保留以对齐词表。
STANCES: tuple[str, ...] = ("support", "oppose", "neutral")

#: 意图（单选）。
INTENTS: tuple[str, ...] = (
    "praise",
    "complaint",
    "question",
    "suggestion",
    "correction",
    "agreement",
    "disagreement",
    "joke",
    "information",
    "other",
)

#: 分析维度：``key``（进 JSON）→ 中文说明（给人看）。
ASPECTS: dict[str, str] = {
    "capability": "能力与效果",
    "quality": "质量",
    "usability": "易用性",
    "performance": "速度与性能",
    "reliability": "稳定性",
    "price": "价格与性价比",
    "service": "服务与体验",
    "content": "内容与选题",
    "presentation": "表达与结构",
    "production": "制作质量",
    "trust": "可信度",
    "comparison": "竞品比较",
    "other": "其他",
}

#: 情绪闭集（中文词即取值）。自由词表会让「好奇/curiosity/疑惑」变成三个标签。
EMOTIONS: tuple[str, ...] = ("认可", "惊喜", "期待", "质疑", "失望", "担忧", "调侃", "愤怒")

#: 维度与情绪的上限，与上游一致。
MAX_ASPECTS = 5
MAX_EMOTIONS = 3

#: 只收录语义等价、映射无歧义的别名；拿不准一律落兜底值。
_INTENT_ALIASES: dict[str, str] = {
    "comparison": "information",
    "compare": "information",
    "informational": "information",
    "对比": "information",
    "比较": "information",
    "赞扬": "praise",
    "表扬": "praise",
    "抱怨": "complaint",
    "吐槽": "complaint",
    "提问": "question",
    "询问": "question",
    "建议": "suggestion",
    "纠错": "correction",
    "赞同": "agreement",
    "反对": "disagreement",
    "玩梗": "joke",
    "调侃": "joke",
    "信息": "information",
    "分享": "information",
    "观望": "other",
    "中立": "other",
    "陈述": "information",
    "描述": "information",
}

_SENTIMENT_ALIASES: dict[str, str] = {
    "mixed-positive": "mixed",
    "positive-negative": "mixed",
    "正面": "positive",
    "负面": "negative",
    "中性": "neutral",
    "混合": "mixed",
    "积极": "positive",
    "消极": "negative",
    "客观": "neutral",
}

_ASPECT_ALIASES: dict[str, str] = {
    "能力": "capability",
    "能力与效果": "capability",
    "效果": "capability",
    "质量": "quality",
    "易用性": "usability",
    "易用": "usability",
    "速度": "performance",
    "性能": "performance",
    "速度与性能": "performance",
    "延迟": "performance",
    "稳定性": "reliability",
    "稳定": "reliability",
    "可靠性": "reliability",
    "价格": "price",
    "性价比": "price",
    "价格与性价比": "price",
    "费用": "price",
    "服务": "service",
    "服务与体验": "service",
    "体验": "service",
    "内容": "content",
    "内容与选题": "content",
    "选题": "content",
    "表达": "presentation",
    "表达与结构": "presentation",
    "结构": "presentation",
    "制作": "production",
    "制作质量": "production",
    "可信度": "trust",
    "信任": "trust",
    "竞品": "comparison",
    "竞品比较": "comparison",
    "对比": "comparison",
    "其他": "other",
}

_EMOTION_ALIASES: dict[str, str] = {
    "curiosity": "期待",
    "好奇": "期待",
    "认可": "认可",
    "赞同": "认可",
    "surprise": "惊喜",
    "惊讶": "惊喜",
    "expectancy": "期待",
    "anticipation": "期待",
    "doubt": "质疑",
    "skepticism": "质疑",
    "disappointment": "失望",
    "concern": "担忧",
    "worry": "担忧",
    "mockery": "调侃",
    "teasing": "调侃",
    "anger": "愤怒",
    "angry": "愤怒",
    "愤怒": "愤怒",
}


def _fold(value: Any) -> str:
    """把任意值收敛成去空白字符串（``None`` → 空串）。"""
    return "" if value is None else str(value).strip()


def _as_items(raw: Any) -> list[Any]:
    """把入参安全收敛成列表（非 ``list``/``tuple`` 一律当空）。

    ``isinstance`` 收窄后类型检查器只知道是 ``list``/``tuple`` 而推不出元素类型，
    故显式 ``cast``，避免 ``reportUnknownArgumentType``。
    """
    if isinstance(raw, (list, tuple)):
        return list(cast("Iterable[Any]", raw))
    return []


def to_enum(
    raw: Any,
    allowed: Iterable[str],
    aliases: dict[str, str] | None = None,
    fallback: str = "other",
) -> tuple[str, bool]:
    """把枚举值收敛到词表内，返回 ``(值, 是否发生过改写)``。

    与上游 ``toEnum`` 同口径：先试小写、再试原文、再查别名，都不中落 ``fallback``。
    """
    allowed_set = set(allowed)
    text = _fold(raw)
    lower = text.lower()
    if lower in allowed_set:
        return lower, lower != text
    if text in allowed_set:
        return text, False
    if aliases:
        mapped = aliases.get(lower) or aliases.get(text)
        if mapped and mapped in allowed_set:
            return mapped, True
    return fallback, True


def to_aspects(raw: Any) -> tuple[list[str], int, int]:
    """收敛维度列表，返回 ``(保留项, 丢弃数, 截断数)``。

    **先丢未知项再截断**：否则一个词表外的值会把合法项挤出上限。支持上游实测会
    出现的对象数组写法（``[{"aspect": ...}]``）。
    """
    items: list[Any] = _as_items(raw)
    known: list[str] = []
    dropped = 0
    for item in items:
        key: Any = item
        if isinstance(item, dict):
            fields = cast("dict[str, Any]", item)
            key = fields.get("aspect") or fields.get("key") or fields.get("name")
        text = _fold(key)
        if not text:
            dropped += 1
            continue
        lower = text.lower()
        resolved: str | None = (
            lower if lower in ASPECTS else _ASPECT_ALIASES.get(lower) or _ASPECT_ALIASES.get(text)
        )
        if not resolved or resolved not in ASPECTS:
            dropped += 1
            continue
        if resolved not in known:
            known.append(resolved)
    truncated = max(0, len(known) - MAX_ASPECTS)
    return known[:MAX_ASPECTS], dropped, truncated


def to_emotions(raw: Any) -> tuple[list[str], int, int]:
    """收敛情绪列表，返回 ``(保留项, 丢弃数, 截断数)``；口径同 :func:`to_aspects`。"""
    items: list[Any] = _as_items(raw)
    kept: list[str] = []
    dropped = 0
    for item in items:
        key: Any = item
        if isinstance(item, dict):
            fields = cast("dict[str, Any]", item)
            key = fields.get("emotion") or fields.get("label") or fields.get("name")
        text = _fold(key)
        if not text:
            dropped += 1
            continue
        resolved: str | None = (
            text if text in EMOTIONS else _EMOTION_ALIASES.get(text.lower()) or _EMOTION_ALIASES.get(text)
        )
        if not resolved or resolved not in EMOTIONS:
            dropped += 1
            continue
        if resolved not in kept:
            kept.append(resolved)
    truncated = max(0, len(kept) - MAX_EMOTIONS)
    return kept[:MAX_EMOTIONS], dropped, truncated
