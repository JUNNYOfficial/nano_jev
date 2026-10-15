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

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nano_jev.nj_engine import clamp, classify, estimate_tokens, longest_quote  # noqa: E402
from nano_jev.nj_vocab import MAX_ASPECTS, MAX_EMOTIONS  # noqa: E402


def test_classification_is_deterministic():
    """同一输入两次判定的结果必须逐字段一致（可回放、可复核的前提）。"""
    first = classify("c0", "这个模型很好用，速度快，推荐！", topic_title="评测")
    second = classify("c0", "这个模型很好用，速度快，推荐！", topic_title="评测")
    assert first == second


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
