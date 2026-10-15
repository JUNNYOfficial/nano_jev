"""``nano_jev.nj_engine`` 的行为测试。

锁住的重点是**确定性**与**粗筛判据的可解释性**：同样的输入必须给同样的标签；
褒贬/意图/维度/情绪的判据要能在测试里说清楚，而不是"跑出来是什么就是什么"。
"""

# 下面的 nano_jev.* 导入依赖运行时注入的 sys.path（见 REPO_ROOT），静态检查器
# 无法跟踪这种动态路径，故在本文件内关闭 reportMissingImports。
# pyright: reportMissingImports=false

from __future__ import annotations

import pathlib
import sys

# 本文件位于 nano_jev/tests/ 下，故要往上三层才回到"能看到 nano_jev 包"的仓库父目录。
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nano_jev import nj_engine  # noqa: E402
from nano_jev import nj_lexicon as lx  # noqa: E402
from nano_jev.nj_engine import _count, clamp, classify, estimate_tokens, longest_quote  # noqa: E402
from nano_jev.nj_vocab import MAX_ASPECTS, MAX_EMOTIONS  # noqa: E402


def test_classification_is_deterministic():
    """同一输入两次判定的结果必须逐字段一致（可回放、可复核的前提）。"""
    first = classify("c0", "这个模型很好用，速度快，推荐！", topic_title="评测")
    second = classify("c0", "这个模型很好用，速度快，推荐！", topic_title="评测")
    assert first == second


def test_classify_scans_each_lexicon_group_once(monkeypatch):
    """回归：热路径不得重复扫描——每个词典组只该被扫一遍。

    这是「快速决策」的结构性保证。旧实现把极性词条扫了两遍、主题词未命中时又在
    相关性判断里把 opinion/aspect/intent 各重扫一遍，实际检查量会远超「每组一遍」
    的上界。这里用 monkeypatch 统计真实词条检查量，断言不超过上界——纯计数、不依赖
    计时，故不会在慢机器上抖动。
    """
    groups = [
        lx.POSITIVE,
        lx.STRONG_POSITIVE,
        lx.NEGATIVE,
        lx.STRONG_NEGATIVE,
        lx.DEFAULT_TOPICS,
        *lx.INTENT_CUES.values(),
        *lx.ASPECT_CUES.values(),
        *lx.EMOTION_CUES.values(),
    ]
    once = sum(len(group) for group in groups)

    checked = {"terms": 0}
    real_count = nj_engine._count

    def counting(text: str, terms) -> int:
        checked["terms"] += len(terms)
        return real_count(text, terms)

    monkeypatch.setattr(nj_engine, "_count", counting)

    # 覆盖主题命中（提前返回）与不命中（走相关性 fallback，旧实现最慢）两条路径。
    classify("c0", "这个模型回答准确，速度快，推荐")
    classify("c1", "I waited for the rain and the failure was painful")

    assert checked["terms"] <= once * 2, f"词条检查 {checked['terms']} 超出每组一遍的上界 {once * 2}"


def test_positive_and_negative_polarity():
    positive = classify("c0", "这个模型太厉害了，回答准确，速度快，非常推荐")
    negative = classify("c1", "太差了，经常崩溃，速度慢还死贵，不推荐")
    assert positive.sentiment == "positive"
    assert positive.sentiment_score > 0
    assert negative.sentiment == "negative"
    assert negative.sentiment_score < 0


def test_mixed_when_both_opinions_present():
    """双向都有明显信号时判 ``mixed``，而不是被数量多的一方盖过。"""
    label = classify("c0", "回答很准确，但是价格太贵了，稳定性也差")
    assert label.sentiment == "mixed"


def test_pure_emoji_is_not_relevant():
    """纯表情/符号不构成"在讨论或评价"，判不相关。"""
    assert classify("c0", "😂😂😂").is_relevant is False
    assert classify("c1", "").is_relevant is False


def test_topic_mention_is_relevant():
    assert classify("c0", "DeepSeek 这次更新挺有意思").is_relevant is True


def test_intent_question_and_suggestion():
    assert classify("c0", "请问这个模型支持中文吗？").intent == "question"
    assert classify("c1", "建议增加一个导出功能").intent == "suggestion"


def test_aspects_and_emotions_are_capped():
    text = "速度慢、价格贵、客服差、界面难用、内容少、还不稳定，很失望，也很担心，太生气了"
    label = classify("c0", text)
    assert len(label.aspects) <= MAX_ASPECTS
    assert len(label.emotion) <= MAX_EMOTIONS
    assert "失望" in label.emotion


def test_evidence_quote_is_verbatim_substring():
    """引文必须是原文逐字子串（jev-arena 会做逐字校验）。"""
    content = "第一句很短。第二句是这个评论里最长的一句话，用来作为引文候选。"
    quote = longest_quote(content)
    assert quote
    assert quote in content


def test_longest_quote_truncates_but_stays_substring():
    content = "啊" * 500
    quote = longest_quote(content, max_len=200)
    assert len(quote) == 200
    assert quote in content


def test_confidence_is_bounded():
    label = classify("c0", "好")
    assert 0.0 <= label.confidence <= 1.0


def test_clamp_and_estimate_tokens():
    assert clamp(5.0, 0.0, 1.0) == 1.0
    assert clamp(-3.0, 0.0, 1.0) == 0.0
    # 中文按 1.05 token/字，向上取整；10 个汉字 → 11。
    assert estimate_tokens("十个中文字符测试工具") == 11
    assert estimate_tokens("") == 0


def test_custom_topics_change_relevance():
    """换一套主题词即可把"相关性"判据搬到别的评论域。"""
    text = "这款相机拍夜景很不错"
    assert classify("c0", text).is_relevant is False
    assert classify("c0", text, topics=("相机",)).is_relevant is True


def test_ascii_terms_match_only_at_word_boundaries():
    """回归：英文词条只认词边界，不再把 ``ai``/``app``/``bug`` 当成词内子串。"""
    for text, term in [
        ("the chairman said", "ai"),
        ("i waited for the rain", "ai"),
        ("an apple a day", "app"),
        ("let us debug this", "bug"),
        ("the build failed", "ui"),
    ]:
        assert _count(text, (term,)) == 0, f"{term!r} 不该命中 {text!r}"


def test_ascii_terms_tolerate_inflections():
    """回归：切词边界不能把 ``models``/``crashed``/``slower`` 这些真命中一起切掉。"""
    for text, term in [
        ("both models are good", "model"),
        ("it crashed twice", "crash"),
        ("it crashes a lot", "crash"),
        ("the slower path", "slow"),
        ("i recommended it", "recommend"),
        ("i liked the answer", "like"),
    ]:
        assert _count(text, (term,)) == 1, f"{term!r} 应命中 {text!r}"


def test_ascii_terms_allow_digit_suffix():
    """``gpt4`` / ``llama3`` 这类带版本号的写法仍算命中，但 ``chatgpt`` 不算。"""
    assert _count("gpt4 is fast", ("gpt",)) == 1
    assert _count("llama3 rocks", ("llama",)) == 1
    assert _count("not chatgpt", ("gpt",)) == 0


def test_chinese_terms_keep_substring_matching():
    """中文没有词边界，仍走子串匹配——这是刻意保留的代价，README 局限节已写明。"""
    assert _count("这个模型很好", ("模型",)) == 1
    assert _count("这是一张卡片", ("卡",)) == 1


def test_custom_ascii_topics_also_use_word_boundaries():
    """``--topics`` 传进来的自定义英文主题词也要切边界，不能漏回子串匹配。"""
    assert classify("c0", "The chairman spoke", topics=("ai",)).is_relevant is False
    assert classify("c0", "The AI model is good", topics=("ai",)).is_relevant is True


def test_off_domain_english_is_not_relevant():
    """回归：整条与主题无关的英文句子，不该因 ``ai``/``app`` 这类子串误命中被判相关。"""
    for text in [
        "The chairman said he will remain in charge",
        "I waited for the rain and the failure was painful",
        "An apple a day keeps the doctor away",
        "Let's debug the legacy codebase tomorrow",
        "The aisle was crowded with tourists",
    ]:
        assert classify("c0", text).is_relevant is False, text
