"""粗筛引擎：把一条文本确定性地判成一套结构化标签。

**核心契约**：同样的输入**永远**得到同样的输出（无随机、无网络、无模型）。这是本
子系统能被 jev-arena 记录、被回放、被逐条复核的前提——与主项目「确定性可复核 vs
探索性」的分工一致。

**它不是语义模型**：判断完全由 :mod:`nano_jev.nj_lexicon` 的词典命中数驱动，只做
粗粒度分档。准确率上限有限，定位是「便宜地先筛一遍」。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from . import nj_lexicon as lx
from .nj_vocab import ASPECTS, EMOTIONS, INTENTS, MAX_ASPECTS, MAX_EMOTIONS

#: 中文字符（含扩展区与全角标点）计约 1 token，其余字符约 0.5 token。
_CJK_RE = re.compile(r"[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef\u3000-\u303f]")

#: 判断「这段文本有没有实际内容」用的字母/数字/汉字判据。
_ALNUM_OR_CJK_RE = re.compile(r"[\w\u4e00-\u9fff]")

#: 断句用的句末标点（中英混排）。
_SENTENCE_RE = re.compile(r"[^。！？!?；;…\n]+[。！？!?；;…]*")


def clamp(value: float, low: float, high: float) -> float:
    """把 ``value`` 夹到 ``[low, high]``。"""
    return max(low, min(high, value))


def estimate_tokens(text: Any) -> int:
    """粗估 token 数，仅用于响应里的用量口径，**不参与计费**。

    口径与 jev-arena 的 ``estimateTokens`` 一致：中文 1.05 token/字，其余 0.5
    token/字符，向上取整。
    """
    cjk = 0
    other = 0
    for ch in str(text):
        if _CJK_RE.match(ch):
            cjk += 1
        else:
            other += 1
    return int(-(-(cjk * 1.05 + other * 0.5) // 1))


def _count(text: str, terms: Sequence[str]) -> int:
    """``terms`` 里有多少个词条出现在 ``text`` 中（每个词条至多计一次）。"""
    return sum(1 for term in terms if term in text)


def longest_quote(content: str, max_len: int = 200) -> str:
    """从原文机械摘出最长的一句引文，作为 ``evidence_quote``。

    **必须是原文逐字子串**：截断取自原串本身，不做任何改写/压缩空白，故结果仍是
    子串，可通过 jev-arena 的逐字校验。
    """
    text = str(content or "")
    if not text.strip():
        return ""
    sentences = _SENTENCE_RE.findall(text) or [text]
    best = ""
    for sentence in sentences:
        trimmed = sentence.strip()
        if len(trimmed) > len(best):
            best = trimmed
    return best[:max_len] if len(best) > max_len else best


@dataclass
class Label:
    """一条内容的结构化标签，字段口径与 jev-arena 的统一 ``Label`` 对齐。"""

    comment_id: str
    is_relevant: bool
    sentiment: str
    sentiment_score: float
    confidence: float
    intent: str
    aspects: list[str] = field(default_factory=list)
    emotion: list[str] = field(default_factory=list)
    evidence_quote: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_raw(self) -> dict[str, Any]:
        """转成 jev-arena 在 ``normalizeLabel`` 前期望的原始字段字典。"""
        return {
            "comment_id": self.comment_id,
            "is_relevant": self.is_relevant,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "intent": self.intent,
            "aspects": list(self.aspects),
            "emotion": list(self.emotion),
            "evidence_quote": self.evidence_quote,
        }


def _polarity(text: str) -> tuple[float, int]:
    """返回 ``(极性分, 命中总数)``；极性分落在 ``[-1, 1]``，0 表示无褒贬。"""
    pos = _count(text, lx.POSITIVE) + 2 * _count(text, lx.STRONG_POSITIVE)
    neg = _count(text, lx.NEGATIVE) + 2 * _count(text, lx.STRONG_NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0, 0
    return (pos - neg) / total, total


def _sentiment_of(score: float, pos_hits: int, neg_hits: int) -> str:
    """由极性分与双向命中数定档；双向都明显时判为 ``mixed``。"""
    if pos_hits > 0 and neg_hits > 0 and min(pos_hits, neg_hits) >= 0.5 * max(pos_hits, neg_hits):
        return "mixed"
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


def _best_by_cues(text: str, cues: dict[str, Sequence[str]], order: Iterable[str]) -> tuple[str, int]:
    """按命中数取线索字典里的最好的 key；并列时按 ``order`` 的先后决定。"""
    order_list = list(order)
    best_key = ""
    best_hits = 0
    for key in order_list:
        hits = _count(text, cues.get(key, ()))
        if hits > best_hits:
            best_key = key
            best_hits = hits
    return best_key, best_hits


def _ranked_by_cues(text: str, cues: dict[str, Sequence[str]], order: Iterable[str], limit: int) -> list[str]:
    """按命中数降序取前 ``limit`` 个 key；命中数为 0 的不计入。"""
    scored: list[tuple[str, int, int]] = []
    for rank, key in enumerate(order):
        hits = _count(text, cues.get(key, ()))
        if hits > 0:
            scored.append((key, hits, rank))
    scored.sort(key=lambda item: (-item[1], item[2]))
    return [key for key, _hits, _rank in scored[:limit]]


def _is_relevant(content: str, title: str, topics: Sequence[str]) -> bool:
    """判相关性：命中主题词，或「有足够评估信号且长度达标」。纯噪声判否。"""
    raw = content.strip()
    if not raw or not _ALNUM_OR_CJK_RE.search(raw):
        return False
    text = f"{title} {content}".lower()
    if any(topic in text for topic in topics):
        return True
    opinion = _count(text, lx.POSITIVE) + _count(text, lx.STRONG_POSITIVE)
    opinion += _count(text, lx.NEGATIVE) + _count(text, lx.STRONG_NEGATIVE)
    aspect_hits = sum(_count(text, cues) for cues in lx.ASPECT_CUES.values())
    intent_hits = sum(_count(text, cues) for cues in lx.INTENT_CUES.values())
    return len(raw) >= 12 and (opinion + aspect_hits + intent_hits) >= 2


def classify(
    comment_id: Any,
    content: Any,
    topic_title: Any = "",
    like_count: Any = 0,
    topics: Sequence[str] | None = None,
) -> Label:
    """把一条内容判成 :class:`Label`。**纯函数，确定性**。"""
    cid = str(comment_id)
    body = "" if content is None else str(content)
    title = "" if topic_title is None else str(topic_title)
    topic_list = tuple(topics) if topics else lx.DEFAULT_TOPICS
    text = body.lower()

    score, polarity_total = _polarity(text)
    pos_hits = _count(text, lx.POSITIVE) + _count(text, lx.STRONG_POSITIVE)
    neg_hits = _count(text, lx.NEGATIVE) + _count(text, lx.STRONG_NEGATIVE)
    sentiment = _sentiment_of(score, pos_hits, neg_hits)

    relevance = _is_relevant(body, title, topic_list)

    intent, intent_hits = _best_by_cues(text, lx.INTENT_CUES, INTENTS)
    if not intent:
        if relevance and abs(score) < 0.15:
            intent = "information"
        elif score > 0.15:
            intent = "praise"
        elif score < -0.15:
            intent = "complaint"
        else:
            intent = "other"

    aspects = _ranked_by_cues(text, lx.ASPECT_CUES, ASPECTS.keys(), MAX_ASPECTS)
    emotions = _ranked_by_cues(text, lx.EMOTION_CUES, EMOTIONS, MAX_EMOTIONS)

    if polarity_total == 0:
        confidence = 0.3
    else:
        confidence = clamp(0.35 + 0.45 * abs(score) + 0.03 * min(polarity_total, 8), 0.0, 0.95)

    return Label(
        comment_id=cid,
        is_relevant=relevance,
        sentiment=sentiment,
        sentiment_score=round(score, 3),
        confidence=round(confidence, 3),
        intent=intent,
        aspects=aspects,
        emotion=emotions,
        evidence_quote=longest_quote(body),
        meta={
            "backend": "nano-jev",
            "likeCount": int(like_count) if str(like_count).lstrip("-").isdigit() else 0,
            "intentHits": intent_hits,
            "polarityHits": polarity_total,
        },
    )
