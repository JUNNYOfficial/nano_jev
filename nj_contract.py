"""Jev 原生 ``/alpha/decisions`` 契约的解析与响应构造。

jev-arena 的左道按下面这份**已验证**的形状发请求（对照其 ``src/backends/jev.mjs``）::

    POST {base}/alpha/decisions          # 注意：没有 /v1
    { "model": "typesafe/jev-1.13",
      "state": {"comments": [{comment_id, content, like_count, topic_title}, ...]},
      "questions": { "c0_relevant": {...}, "c0_sentiment": {...}, ... } }

本模块把该请求解析成 :class:`DecisionsRequest`，把逐条 :class:`~nano_jev.nj_engine.Label`
**翻译回逐问题的 ``answers``**，并拼出与 Jev 同形状的响应。这样 jev-arena 无需改一
行代码即可把左道指向本服务——它自己会从 ``answers`` 里 ``normalizeLabel``。

**答案不是写死给那 25 个问题的**：``build_answers`` 先按问题 id 的语义后缀识别角色，
识别不出再退回按 ``type``/``criteria`` 推断，最后才对任意问题给出兜底答案。故换上
另一套问题清单也能用。
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from . import nj_lexicon as lx
from .nj_engine import Label, classify, estimate_tokens
from .nj_vocab import ASPECTS, EMOTIONS, INTENTS, SENTIMENTS

#: 形如 ``c3_sentiment`` / ``c12_aspect_capability`` 的问题 id。
_COMMENT_INDEX_RE = re.compile(r"^c(\d+)_(.+)$")

#: 打分档位分布的高斯宽度（单位：档）。
_SCORE_SIGMA = 0.6


@dataclass
class DecisionsRequest:
    """一份 ``/alpha/decisions`` 请求解析后的结构化表示。"""

    model: str
    comments: list[dict[str, Any]]
    questions: dict[str, dict[str, Any]]


def _as_dict(value: Any) -> dict[str, Any]:
    """把任意值收敛成 ``dict[str, Any]``（非 dict 一律空）。

    ``Any`` 经 ``isinstance(x, dict)`` 收窄后会变成 ``dict[Unknown, Unknown]``，
    之后每次 ``.get()``/``.keys()`` 都是"部分未知"。这里显式 ``cast`` 成
    ``dict[str, Any]``，让取值落回 ``Any``（Pylance 认 ``Any``，不认 ``Unknown``）。
    """
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    return {}


def _as_list(value: Any) -> list[Any]:
    """把任意值收敛成 ``list[Any]``（非 list/tuple 一律空）。"""
    if isinstance(value, (list, tuple)):
        return list(cast("Iterable[Any]", value))
    return []


def _coerce_comments(state: Any) -> list[dict[str, Any]]:
    """把 ``state`` 收敛成统一的评论列表；``state`` 不是标准形状时尽力兜底。"""
    raw: list[Any] | None = None
    candidate = _as_dict(state).get("comments")
    if isinstance(candidate, list):
        raw = _as_list(candidate)
    elif isinstance(state, list):
        raw = _as_list(state)

    if raw is None:
        content = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        return [{"comment_id": "c0", "content": content, "topic_title": "", "like_count": 0}]

    comments: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            row = _as_dict(item)
            comments.append(
                {
                    "comment_id": str(row.get("comment_id", row.get("id", index))),
                    "content": str(row.get("content", "")),
                    "topic_title": str(row.get("topic_title", "")),
                    "like_count": row.get("like_count", 0),
                }
            )
        else:
            comments.append(
                {"comment_id": str(index), "content": str(item), "topic_title": "", "like_count": 0}
            )
    return comments


def parse_decisions_request(body: Any) -> DecisionsRequest:
    """解析 ``/alpha/decisions`` 的请求体。缺字段时给安全默认值，**不抛异常**。"""
    data = _as_dict(body)
    model = str(data.get("model") or "nano-jev")
    comments = _coerce_comments(data.get("state"))
    questions: dict[str, dict[str, Any]] = {}
    for qid, question in _as_dict(data.get("questions")).items():
        questions[str(qid)] = _as_dict(question) if isinstance(question, dict) else {"type": "noul"}
    return DecisionsRequest(model=model, comments=comments, questions=questions)


def label_comments(request: DecisionsRequest, topics: Sequence[str] | None = None) -> list[Label]:
    """对请求里的每条评论跑一次粗筛，顺序与 ``comments`` 严格一致。"""
    return [
        classify(
            comment["comment_id"],
            comment["content"],
            topic_title=comment.get("topic_title", ""),
            like_count=comment.get("like_count", 0),
            topics=topics,
        )
        for comment in request.comments
    ]


def _comment_index(question_id: str, n_comments: int) -> int | None:
    """从问题 id 里取出它属于第几条评论；取不到且只有一条评论时归到 0。"""
    match = _COMMENT_INDEX_RE.match(question_id)
    if match:
        return int(match.group(1))
    return 0 if n_comments == 1 else None


def _role_from_suffix(suffix: str) -> tuple[str, str | None]:
    """按问题 id 的语义后缀识别角色（jev-arena 的命名约定）。"""
    low = suffix.lower()
    if low == "relevant":
        return "relevant", None
    if low == "sentiment":
        return "sentiment", None
    if low == "intent":
        return "intent", None
    if low == "score":
        return "score", None
    if low.startswith("aspect_"):
        return "aspect", suffix[len("aspect_") :]
    if low.startswith("emotion_"):
        return "emotion", suffix[len("emotion_") :]
    return "", None


def _role_from_question(question: dict[str, Any]) -> tuple[str, str | None]:
    """识别不出后缀时，退回按 ``type`` 与 ``criteria``/``instructions`` 推断角色。"""
    qtype = str(question.get("type") or "").lower()
    criteria = question.get("criteria")
    instructions = str(question.get("instructions") or "")
    if qtype == "choice" and isinstance(criteria, dict):
        keys = set(_as_dict(criteria).keys())
        if keys and keys <= set(SENTIMENTS):
            return "sentiment", None
        if keys and keys <= set(INTENTS):
            return "intent", None
    if qtype == "score":
        return "score", None
    if qtype in {"noul", "boolean"}:
        for key, zh in ASPECTS.items():
            if zh in instructions:
                return "aspect", key
        for word in EMOTIONS:
            if word in instructions:
                return "emotion", word
        if "相关" in instructions or "relevant" in instructions.lower():
            return "relevant", None
    return "", None


def detect_role(question_id: str, question: dict[str, Any]) -> tuple[str, str | None]:
    """综合 id 后缀与问题定义，判出该问题的角色。"""
    match = _COMMENT_INDEX_RE.match(question_id)
    suffix = match.group(2) if match else question_id
    role, key = _role_from_suffix(suffix)
    if role:
        return role, key
    return _role_from_question(question)


def _choice_answer(value: str, allowed: Sequence[str], confidence: float) -> dict[str, Any]:
    """构造 ``choice`` 答案：把 ``confidence`` 给中选项，其余均摊剩余概率。"""
    conf = round(float(confidence), 3)
    probabilities: dict[str, float] = {}
    if value in allowed:
        others = [item for item in allowed if item != value]
        probabilities[value] = round(conf, 4)
        share = (1.0 - conf) / len(others) if others else 0.0
        for item in others:
            probabilities[item] = round(share, 4)
    else:
        for item in allowed:
            probabilities[item] = round(1.0 / len(allowed), 4)
    return {"type": "choice", "choice": value, "probabilities": probabilities, "confidence": conf}


def _score_answer(label: Label, criteria: Sequence[Any]) -> dict[str, Any]:
    """构造 ``score`` 答案：档位下标由 ``sentiment_score`` 线性映射而来。"""
    n = len(criteria) if criteria else 5
    index = (label.sentiment_score + 1.0) / 2.0 * (n - 1)
    weights: list[float] = []
    for i in range(n):
        weights.append(2.718281828459045 ** (-((i - index) ** 2) / (2.0 * _SCORE_SIGMA**2)))
    total = sum(weights) or 1.0
    probabilities = {str(i): round(weights[i] / total, 4) for i in range(n)}
    legend = {str(i): (criteria[i] if i < len(criteria) else "") for i in range(n)}
    return {
        "type": "score",
        "score": round(index, 3),
        "legend": legend,
        "probabilities": probabilities,
        "confidence": round(label.confidence, 3),
    }


def _noul_answer(probability: float) -> dict[str, Any]:
    """构造 ``noul``（真值概率）答案；Jev 侧无 ``confidence`` 字段。"""
    return {"type": "noul", "noul": round(float(probability), 4)}


def _fallback_answer(question: dict[str, Any]) -> dict[str, Any]:
    """角色未知时的兜底答案：形状合法、置信度低，不假装知道。"""
    qtype = str(question.get("type") or "").lower()
    criteria = question.get("criteria")
    if qtype in {"choice"} and isinstance(criteria, dict) and criteria:
        keys = list(_as_dict(criteria).keys())
        return _choice_answer(keys[0], keys, 0.2)
    if qtype == "score":
        labels: list[Any] = _as_list(criteria) if isinstance(criteria, list) else ["0", "1", "2", "3", "4"]
        return _score_answer(Label("", False, "neutral", 0.0, 0.2, "other"), labels)
    return _noul_answer(0.5)


def _resolve_key(key: str | None, allowed: Sequence[str]) -> str:
    """把问题 id 里带出的 key 收敛到词表内（大小写不敏感）；不中返回空串。"""
    if not key:
        return ""
    if key in allowed:
        return key
    low = key.lower()
    for candidate in allowed:
        if candidate.lower() == low:
            return candidate
    return ""


def _answer_for(
    role: str,
    key: str | None,
    question: dict[str, Any],
    label: Label | None,
) -> dict[str, Any]:
    """把角色与标签合成一个答案；无对应标签时退回低置信兜底。"""
    if label is None:
        return _fallback_answer(question)
    if role == "relevant":
        return _noul_answer(0.9 if label.is_relevant else 0.12)
    if role == "sentiment":
        return _choice_answer(label.sentiment, SENTIMENTS, label.confidence)
    if role == "intent":
        return _choice_answer(label.intent, INTENTS, label.confidence)
    if role == "score":
        criteria = question.get("criteria")
        return _score_answer(label, _as_list(criteria))
    if role == "aspect":
        resolved = _resolve_key(key, list(ASPECTS.keys()))
        return _noul_answer(0.85 if resolved and resolved in label.aspects else 0.12)
    if role == "emotion":
        resolved = _resolve_key(key, list(EMOTIONS))
        return _noul_answer(0.8 if resolved and resolved in label.emotion else 0.12)
    return _fallback_answer(question)


def build_answers(request: DecisionsRequest, labels: Sequence[Label]) -> dict[str, dict[str, Any]]:
    """把逐条标签翻译成逐问题的 ``answers``（键与原 ``questions`` 一一对应）。"""
    answers: dict[str, dict[str, Any]] = {}
    n_comments = len(request.comments)
    for question_id, question in request.questions.items():
        role, key = detect_role(question_id, question)
        index = _comment_index(question_id, n_comments)
        label = labels[index] if index is not None and 0 <= index < len(labels) else None
        answers[question_id] = _answer_for(role, key, question, label)
    return answers


def build_decisions_response(
    request: DecisionsRequest,
    labels: Sequence[Label],
    model: str | None = None,
) -> dict[str, Any]:
    """拼出与 Jev 同形状的响应体（``answers`` / ``usage`` / ``id`` / ``provider``）。"""
    answers = build_answers(request, labels)
    payload_in: dict[str, Any] = {
        "state": {"comments": request.comments},
        "questions": request.questions,
    }
    input_tokens = estimate_tokens(json.dumps(payload_in, ensure_ascii=False))
    output_tokens = estimate_tokens(json.dumps(answers, ensure_ascii=False))
    return {
        "model": model or request.model,
        "answers": answers,
        # cost=0：本服务在本地跑，不产生供应商费用。汇总里的 0 是「免费」而非「未知」。
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost": 0.0},
        "id": "gen-dec-nano-" + uuid.uuid4().hex[:12],
        "provider": "NanoJev",
    }


def handle_decisions(body: Any, topics: Sequence[str] | None = None) -> dict[str, Any]:
    """一站式入口：解析请求 → 粗筛 → 构造响应。"""
    request = parse_decisions_request(body)
    labels = label_comments(request, topics=topics)
    return build_decisions_response(request, labels)


#: 暴露给调用方的默认主题词，便于文档与 CLI 引用。
DEFAULT_TOPICS = lx.DEFAULT_TOPICS
