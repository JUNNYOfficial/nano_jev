"""``nano_jev.nj_contract`` 与受控词表的测试。

两类重点：① **契约形状**——回给 jev-arena 的 ``answers`` 必须能被它的
``normalizeLabel`` 消费；② **词表对齐**——枚举必须与上游 ``vocab.mjs`` 逐项一致，
否则两边标签无法对比。上游改词表而这里没跟，本文件应当报红。
"""

from __future__ import annotations

import pathlib
import sys

import pytest

# 本文件位于 nano_jev/tests/ 下，故要往上三层才回到"能看到 nano_jev 包"的仓库父目录。
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nano_jev.nj_contract import (  # noqa: E402
    build_answers,
    build_decisions_response,
    detect_role,
    handle_decisions,
    label_comments,
    parse_decisions_request,
)
from nano_jev.nj_engine import estimate_tokens  # noqa: E402
from nano_jev.nj_lexicon import DEFAULT_TOPICS as LEXICON_DEFAULT_TOPICS  # noqa: E402
from nano_jev.nj_vocab import (  # noqa: E402
    ASPECTS,
    EMOTIONS,
    INTENTS,
    MAX_ASPECTS,
    MAX_EMOTIONS,
    SENTIMENTS,
    STANCES,
    to_aspects,
    to_emotions,
    to_enum,
)


def _arena_request() -> dict:
    """一份与 jev-arena ``backend/jev.mjs`` 同形状的请求（节选关键问题）。"""
    return {
        "model": "typesafe/jev-1.13",
        "state": {
            "comments": [
                {
                    "comment_id": "c0",
                    "content": "太差了，经常崩溃，客服也不理人",
                    "like_count": 3,
                    "topic_title": "评测",
                }
            ]
        },
        "questions": {
            "c0_relevant": {"type": "noul", "criteria": {"true": "是", "false": "否"}},
            "c0_sentiment": {
                "type": "choice",
                "criteria": dict.fromkeys(SENTIMENTS),
            },
            "c0_intent": {"type": "choice", "criteria": dict.fromkeys(INTENTS)},
            "c0_score": {
                "type": "score",
                "criteria": ["强烈负面", "偏负面", "中性/无法判断", "偏正面", "强烈正面"],
            },
            "c0_aspect_reliability": {
                "type": "noul",
                "criteria": {"true": "是", "false": "否"},
            },
            "c0_emotion_失望": {"type": "noul", "criteria": {"true": "是", "false": "否"}},
        },
    }


def test_module_exports_are_complete():
    """``__all__`` 必须逐项可取；``VERSION`` / ``DEFAULT_TOPICS`` 也在公开面内。"""
    import nano_jev

    for name in nano_jev.__all__:
        assert hasattr(nano_jev, name), name
    assert nano_jev.VERSION
    assert tuple(LEXICON_DEFAULT_TOPICS) == nano_jev.DEFAULT_TOPICS


def test_estimate_tokens_edge_cases():
    """``estimate_tokens`` 改写为一次扫描后，边界与混排仍按老口径。"""
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 2  # str(None) == "None"，4 字符 × 0.5 = 2.0
    assert estimate_tokens("😀") == 1
    assert estimate_tokens("ab") == 1
    assert estimate_tokens("啊" * 100) == 105


def test_vocab_matches_jev_arena():
    """受控词表必须与 jev-arena ``src/vocab.mjs`` 逐项一致（对齐守门）。"""
    assert SENTIMENTS == ("positive", "negative", "neutral", "mixed")
    assert STANCES == ("support", "oppose", "neutral")
    assert INTENTS == (
        "praise", "complaint", "question", "suggestion", "correction",
        "agreement", "disagreement", "joke", "information", "other",
    )
    assert tuple(ASPECTS.keys()) == (
        "capability", "quality", "usability", "performance", "reliability",
        "price", "service", "content", "presentation", "production",
        "trust", "comparison", "other",
    )
    assert EMOTIONS == ("认可", "惊喜", "期待", "质疑", "失望", "担忧", "调侃", "愤怒")
    assert (MAX_ASPECTS, MAX_EMOTIONS) == (5, 3)


def test_parse_request_from_arena_shape():
    request = parse_decisions_request(_arena_request())
    assert request.model == "typesafe/jev-1.13"
    assert len(request.comments) == 1
    assert request.comments[0]["comment_id"] == "c0"
    assert set(request.questions) == set(_arena_request()["questions"])


def test_parse_request_without_comments_falls_back_to_single():
    request = parse_decisions_request({"state": "一句被直接当评论的文本", "questions": {}})
    assert request.comments == [
        {"comment_id": "c0", "content": "一句被直接当评论的文本", "topic_title": "", "like_count": 0}
    ]


def test_detect_role_by_suffix_and_definition():
    assert detect_role("c3_relevant", {"type": "noul"}) == ("relevant", None)
    assert detect_role("c3_aspect_capability", {"type": "noul"}) == ("aspect", "capability")
    assert detect_role("c3_emotion_失望", {"type": "noul"}) == ("emotion", "失望")
    # 没有语义后缀时按 criteria 反推。
    role, _key = detect_role("q7", {"type": "choice", "criteria": dict.fromkeys(SENTIMENTS)})
    assert role == "sentiment"


def test_build_answers_shape_and_probabilities():
    request = parse_decisions_request(_arena_request())
    labels = label_comments(request)
    answers = build_answers(request, labels)

    assert set(answers) == set(request.questions)
    relevant = answers["c0_relevant"]
    assert relevant["type"] == "noul"
    assert 0.0 <= relevant["noul"] <= 1.0

    sentiment = answers["c0_sentiment"]
    assert sentiment["type"] == "choice"
    assert sentiment["choice"] in SENTIMENTS
    assert sum(sentiment["probabilities"].values()) == pytest.approx(1.0, abs=0.02)

    score = answers["c0_score"]
    assert score["type"] == "score"
    assert 0.0 <= score["score"] <= 4.0
    assert set(score["probabilities"]) == {"0", "1", "2", "3", "4"}
    assert score["legend"]["2"] == "中性/无法判断"


def test_build_answers_unknown_question_still_well_formed():
    """认不出的问题也要给出形状合法的兜底答案，而不是崩掉。"""
    request = parse_decisions_request(
        {
            "state": {"comments": [{"comment_id": "c0", "content": "随便一句"}]},
            "questions": {"c0_weird": {"type": "noul", "criteria": {"true": "是", "false": "否"}}},
        }
    )
    answers = build_answers(request, label_comments(request))
    assert answers["c0_weird"]["type"] == "noul"
    assert answers["c0_weird"]["noul"] == 0.5


def test_decisions_response_shape_and_cost():
    request = parse_decisions_request(_arena_request())
    response = build_decisions_response(request, label_comments(request))
    assert set(response) == {"model", "answers", "usage", "id", "provider"}
    assert response["provider"] == "NanoJev"
    # 本地服务不产生供应商费用：0 是"免费"，不是"未知"。
    assert response["usage"]["cost"] == 0.0
    assert response["usage"]["input_tokens"] > 0


def test_handle_decisions_round_trip():
    response = handle_decisions(_arena_request())
    assert response["answers"]["c0_relevant"]["noul"] > 0.5
    assert response["answers"]["c0_aspect_reliability"]["noul"] > 0.5


def test_to_enum_and_list_normalizers():
    assert to_enum("Positive", SENTIMENTS) == ("positive", True)
    assert to_enum("正面", SENTIMENTS, {"正面": "positive"}) == ("positive", True)
    assert to_enum("完全没见过", INTENTS) == ("other", True)
    # 别名指向词表外的值时，必须回落 fallback，绝不能把表外值透出。
    assert to_enum("weird", SENTIMENTS, {"weird": "not_a_sentiment"}) == ("other", True)

    kept, dropped, truncated = to_aspects(["capability", "未知维度", "price"])
    assert kept == ["capability", "price"]
    assert (dropped, truncated) == (1, 0)

    too_many = [k for k in ASPECTS if k != "other"]
    _kept, _dropped, truncated2 = to_aspects(too_many)
    assert truncated2 == len(too_many) - MAX_ASPECTS

    emotions, _dropped, _truncated = to_emotions([{"emotion": "失望"}, "curiosity"])
    assert emotions == ["失望", "期待"]
