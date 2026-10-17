"""粗筛引擎：把一条文本确定性地判成一套结构化标签。

**核心契约**：同样的输入**永远**得到同样的输出（无随机、无网络、无模型）。这是本
子系统能被 jev-arena 记录、被回放、被逐条复核的前提——与主项目「确定性可复核 vs
探索性」的分工一致。

**它不是语义模型**：判断完全由 :mod:`nano_jev.nj_lexicon` 的词典命中数驱动，只做
粗粒度分档。准确率上限有限，定位是「便宜地先筛一遍」。

**词条怎么匹配**：中文没有词边界、本子系统又刻意不引入分词依赖，故中文词条仍是
子串匹配；而英文子串匹配会持续制造系统性误命中（``ai`` ⊂ ``chairman``/``rain``、
``app`` ⊂ ``apple``、``bug`` ⊂ ``debug``），故**纯 ASCII 词条改按英文词边界匹配**，
详见 :func:`_compile_term`。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import nj_lexicon as lx
from .nj_vocab import ASPECTS, EMOTIONS, INTENTS, MAX_ASPECTS, MAX_EMOTIONS

#: 中文字符（含扩展区与全角标点）计约 1 token，其余字符约 0.5 token。
_CJK_RE = re.compile(r"[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef\u3000-\u303f]")

#: 判断「这段文本有没有实际内容」用的字母/数字/汉字判据。
_ALNUM_OR_CJK_RE = re.compile(r"[\w\u4e00-\u9fff]")

#: 断句用的句末标点（中英混排）。
_SENTENCE_RE = re.compile(r"[^。！？!?；;…\n]+[。！？!?；;…]*")

#: 只有「纯 ASCII 字母，单词之间单空格」的词条才切英文词边界；其余一律走子串。
_ASCII_TERM_RE = re.compile(r"^[a-z]+(?: [a-z]+)*$")

#: 英文词尾屈折：复数/三单、过去式、进行时、比较级/最高级。
#: 允许它们是为了不把 ``models`` / ``crashed`` / ``slower`` 这些真命中一起切掉。
_INFLECTIONS = r"(?:s|es|d|ed|ing|er|est)?"


def clamp(value: float, low: float, high: float) -> float:
    """把 ``value`` 夹到 ``[low, high]``。"""
    return max(low, min(high, value))


def estimate_tokens(text: Any) -> int:
    """粗估 token 数，仅用于响应里的用量口径，**不参与计费**。

    口径与 jev-arena 的 ``estimateTokens`` 一致：中文 1.05 token/字，其余 0.5
    token/字符，向上取整。
    """
    raw = str(text)
    cjk = len(_CJK_RE.findall(raw))
    other = len(raw) - cjk
    return int(-(-(cjk * 1.05 + other * 0.5) // 1))


def _compile_term(term: str) -> re.Pattern[str] | None:
    """把词条编成「按英文词边界匹配」的正则；返回 ``None`` 表示该词条仍走子串匹配。

    只有纯 ASCII 字母词条才切边界：中文没有词边界，对 ``卡`` / ``好`` 这类单字词条
    切边界会把它们整体切没，而本子系统刻意不做分词（保持零依赖）。英文则必须切——
    子串匹配会让 ``ai`` 命中 ``chairman``/``rain``/``waited``、``app`` 命中
    ``apple``、``bug`` 命中 ``debug``、``ui`` 命中 ``build``，且这类误命中会直接
    把整条无关评论判成「与主题相关」。

    两侧刻意用「非字母」而不是 ``\\b``：``\\b`` 把数字也算词内字符，会让 ``gpt``
    匹配不上 ``gpt4``、``llama`` 匹配不上 ``llama3``。词尾再放一个可选的屈折后缀
    （见 :data:`_INFLECTIONS`），以保住 ``models`` / ``prompts`` / ``crashed``。
    """
    if not _ASCII_TERM_RE.match(term):
        return None
    return re.compile(rf"(?<![a-z]){re.escape(term)}{_INFLECTIONS}(?![a-z])")


#: 需要按英文词边界匹配的词条 → 预编译模式。**不在表里 = 走子串匹配**，故中文词条
#: 无需登记，只有 ASCII 词条才进这张表。
_ASCII_PATTERNS: dict[str, re.Pattern[str]] = {}

#: 已登记过的词条组（按元组值去重）：自定义主题词只需在首次出现时登记一次。
_REGISTERED_GROUPS: set[tuple[str, ...]] = set()


def _register_terms(terms: tuple[str, ...]) -> None:
    """登记一组词条的匹配模式；同一组只做一次（幂等）。

    词典里的静态词条在导入时登记（见 :func:`_warm_ascii_patterns`）；``--topics``
    传进来的自定义主题词由 :func:`classify` 按组登记。**登记被按组去重**，故热路径上
    的代价只是一次元组哈希，而不是为组内每个词条各查一次表。

    ``_REGISTERED_GROUPS.add()`` 必须留在循环**之后**：服务是多线程的，先登记组再填
    词条会让另一个线程看到「已登记」却读到填了一半的表。放在最后则最坏只是重复登记
    一遍（写入的值相同，幂等）。
    """
    if terms in _REGISTERED_GROUPS:
        return
    for term in terms:
        pattern = _compile_term(term)
        if pattern is not None:
            _ASCII_PATTERNS[term] = pattern
    _REGISTERED_GROUPS.add(terms)


def _warm_ascii_patterns() -> None:
    """导入时把 :mod:`nano_jev.nj_lexicon` 的静态英文词条一次性预编译好。"""
    groups: list[tuple[str, ...]] = [
        lx.POSITIVE,
        lx.STRONG_POSITIVE,
        lx.NEGATIVE,
        lx.STRONG_NEGATIVE,
        lx.DEFAULT_TOPICS,
        lx.NOISE_MARKERS,
        *lx.INTENT_CUES.values(),
        *lx.ASPECT_CUES.values(),
        *lx.EMOTION_CUES.values(),
    ]
    for group in groups:
        _register_terms(group)


_warm_ascii_patterns()


def _count(text: str, terms: Sequence[str]) -> int:
    """``terms`` 里有多少个词条出现在 ``text`` 中（每个词条至多计一次）。

    ``text`` 须已小写（调用方 :func:`classify` / :func:`_is_relevant` 的口径）。
    匹配规则：ASCII 词条按英文词边界，其余按子串，见 :func:`_compile_term`。
    """
    hits = 0
    table = _ASCII_PATTERNS
    for term in terms:
        # 先用纯 C 的子串探测当快筛。这是合法的必要条件：英文词条的屈折后缀只追加在
        # 词尾，「正则命中」必然蕴含「子串命中」，对中文词条同样成立。占绝大多数的
        # 未命中词条到这里就结束——既不必查表，更不必跑正则。
        if term not in text:
            continue
        pattern = table.get(term)
        if pattern is not None and pattern.search(text) is None:
            continue
        hits += 1
    return hits


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
    aspects: list[str] = field(default_factory=list[str])
    emotion: list[str] = field(default_factory=list[str])
    evidence_quote: str = ""
    meta: dict[str, Any] = field(default_factory=dict[str, Any])

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


def _polarity(pos: int, neg: int) -> tuple[float, int]:
    """由**已加权**的双向命中数算 ``(极性分, 总命中)``；总命中为 0 时回 0 分。

    纯函数：计数在 :func:`classify` 里一次完成，这里不再碰词条、不再重复扫描。
    """
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


def _best_by_cues(
    text: str, cues: Mapping[str, Sequence[str]], order: Iterable[str]
) -> tuple[str, int, int]:
    """按命中数取线索字典里的最好的 key；并列时按 ``order`` 的先后决定。

    返回 ``(best_key, best_hits, total_hits)``：``total_hits`` 是这一遍扫描顺带
    累计的命中总数，供相关性判断复用——省得再单独扫一遍同一批词条。
    """
    order_list = list(order)
    best_key = ""
    best_hits = 0
    total = 0
    for key in order_list:
        hits = _count(text, cues.get(key, ()))
        total += hits
        if hits > best_hits:
            best_key = key
            best_hits = hits
    return best_key, best_hits, total


def _ranked_by_cues(
    text: str, cues: Mapping[str, Sequence[str]], order: Iterable[str], limit: int
) -> tuple[list[str], int]:
    """按命中数降序取前 ``limit`` 个 key；命中数为 0 的不计入。

    返回 ``(keys, total_hits)``：``total_hits`` 同样顺带累计，供相关性判断复用。
    """
    scored: list[tuple[str, int, int]] = []
    total = 0
    for rank, key in enumerate(order):
        hits = _count(text, cues.get(key, ()))
        total += hits
        if hits > 0:
            scored.append((key, hits, rank))
    scored.sort(key=lambda item: (-item[1], item[2]))
    return [key for key, _hits, _rank in scored[:limit]], total


def _is_relevant(
    content: str,
    title: str,
    topics: Sequence[str],
    opinion_hits: int,
    aspect_hits: int,
    intent_hits: int,
) -> bool:
    """判相关性：命中主题词，或「有足够评估信号且长度达标」。纯噪声判否。

    ``opinion_hits`` / ``aspect_hits`` / ``intent_hits`` 由 :func:`classify` 里
    **唯一一次**的词典扫描算好传入——这里不再重复扫，只做纯逻辑判断。
    """
    raw = content.strip()
    if not raw or not _ALNUM_OR_CJK_RE.search(raw):
        return False
    text = f"{title} {content}".lower()
    if _count(text, topics):
        return True
    return len(raw) >= 12 and (opinion_hits + aspect_hits + intent_hits) >= 2


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
    # 自定义主题词（``--topics``）也要按英文词边界判，故首次出现时登记一次；
    # 登记按组去重，热路径上只是一次元组哈希。
    _register_terms(topic_list)
    text = body.lower()

    # —— 每个词条组只扫一遍，命中总数一路带下去复用 ——
    # 旧实现把极性词条扫了两遍（_polarity 一遍、pos/neg_hits 一遍），主题词没命中时
    # 又在 _is_relevant 里把 opinion/aspect/intent 各扫一遍。这里是「纯函数判断器」的
    # 热路径：词条扫描次数直接决定单条决策耗时，故收紧成每组建一次。
    pos_base = _count(text, lx.POSITIVE)
    pos_strong = _count(text, lx.STRONG_POSITIVE)
    neg_base = _count(text, lx.NEGATIVE)
    neg_strong = _count(text, lx.STRONG_NEGATIVE)
    pos_hits = pos_base + pos_strong
    neg_hits = neg_base + neg_strong
    # 加权命中数（强词 ×2）进极性分；双向命中数（不强词）进 sentiment 的 mixed 判定。
    score, polarity_total = _polarity(pos_base + 2 * pos_strong, neg_base + 2 * neg_strong)
    sentiment = _sentiment_of(score, pos_hits, neg_hits)

    intent, intent_hits, intent_total = _best_by_cues(text, lx.INTENT_CUES, INTENTS)
    aspects, aspect_total = _ranked_by_cues(text, lx.ASPECT_CUES, ASPECTS.keys(), MAX_ASPECTS)
    emotions, _emotion_total = _ranked_by_cues(text, lx.EMOTION_CUES, EMOTIONS, MAX_EMOTIONS)

    relevance = _is_relevant(
        body,
        title,
        topic_list,
        opinion_hits=pos_hits + neg_hits,
        aspect_hits=aspect_total,
        intent_hits=intent_total,
    )

    if not intent:
        if relevance and abs(score) < 0.15:
            intent = "information"
        elif score > 0.15:
            intent = "praise"
        elif score < -0.15:
            intent = "complaint"
        else:
            intent = "other"

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


#: 逐条判断函数：接收评论字典（``comment_id`` / ``content`` / ``topic_title`` /
#: ``like_count``），返回标签。确定性粗筛与级联判断都满足这个形状，故契约层
#: 与服务层只需依赖它，不必知道背后是词典还是大模型。
Labeler = Callable[[Mapping[str, Any]], Label]
