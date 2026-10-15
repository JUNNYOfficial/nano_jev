"""OpenAI 兼容的 ``/chat/completions`` 适配层。

jev-arena 的本地开源版除 Jev 协议外还支持 ``openai`` 协议（右侧 DeepSeek 走的就是
它）。本模块让本服务也能挂在 ``openai`` 那一侧：接收 ``{model, messages, ...}``，
从用户消息里抽出评论，返回一个标准形状的 chat completion，其 ``content`` 是
``{"labels": [...]}``。

字段口径对照其 ``src/backends/deepseek.mjs``：它读 ``choices[0].message.content``、
``choices[0].finish_reason`` 与 ``usage.prompt_tokens`` / ``usage.completion_tokens``。

除默认的整包响应外，本层也支持 OpenAI 的两种常见用法：``n``（一条请求要多个候选，
对确定性后端即同一答案重复 ``n`` 份）与 ``stream``（``text/event-stream`` 的
``chat.completion.chunk`` 增量帧，末帧以 ``data: [DONE]`` 收尾）。
"""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, cast

from .nj_engine import Label, classify, estimate_tokens

#: 从用户消息里抽评论时，认这个 JSON 键。
_COMMENTS_KEY = "comments"

#: ``n`` 的上限：确定性后端下多候选取值相同，只做形状兼容，故不放大计算量。
_MAX_N = 8


def _as_dict(value: Any) -> dict[str, object]:
    """把任意值收敛成 ``dict[str, object]``（非 dict 一律空）。

    ``Any`` 经 ``isinstance(x, dict)`` 收窄后会变成 ``dict[Unknown, Unknown]``，
    之后每次 ``.get()`` 都是"部分未知"。这里显式 ``cast`` 成 ``dict[str, object]``，
    让取键结果落回已知类型 ``object``（配合下方各处的 ``isinstance`` 再收窄）。
    """
    if isinstance(value, dict):
        return cast("dict[str, object]", value)
    return {}


def _as_list(value: Any) -> list[object]:
    """把任意值收敛成 ``list[object]``（非 list/tuple 一律空）。"""
    if isinstance(value, (list, tuple)):
        return list(cast("Iterable[object]", value))
    return []


def _message_text(messages: object) -> str:
    """取最后一条 ``user`` 消息的纯文本（多模态 content 只拼文本片段）。"""
    text = ""
    for message in _as_list(messages):
        msg = _as_dict(message)
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                str(part.get("text", ""))
                for part in (_as_dict(item) for item in _as_list(content))
                if part.get("type") == "text"
            )
    return text


def _find_comments_object(text: str) -> dict[str, object] | None:
    """在任意文本里找出**最后一个**含 ``comments`` 数组的 JSON 对象。

    对每个 ``{`` 都试一次 ``raw_decode``。曾试图改成从后往前扫并命中即返回，实测
    在 jev-arena 形状的 payload 上**零收益**（43.1us → 42.9us）：目标对象是最外层
    那个，下标最小，逆序反而要走过全部嵌套花括号才轮到它。该 payload 下整个函数只
    占单次请求的极小头，故保留这个更好懂的版本。
    """
    decoder = json.JSONDecoder()
    best: dict[str, object] | None = None
    index = text.find("{")
    while index != -1:
        decoded: object = None
        with contextlib.suppress(ValueError):
            decoded, _end = decoder.raw_decode(text, index)
        obj = _as_dict(decoded)
        if isinstance(obj.get(_COMMENTS_KEY), list):
            best = obj
        index = text.find("{", index + 1)
    return best


def _coerce_comment(item: object, index: int) -> dict[str, Any]:
    """把数组里的一个元素收敛成标准评论字典。"""
    if isinstance(item, dict):
        obj = _as_dict(item)
        return {
            "comment_id": str(obj.get("comment_id", obj.get("id", index))),
            "content": str(obj.get("content", "")),
            "topic_title": str(obj.get("topic_title", "")),
            "like_count": obj.get("like_count", 0),
        }
    return {"comment_id": str(index), "content": str(item), "topic_title": "", "like_count": 0}


def extract_comments(messages: object) -> list[dict[str, Any]]:
    """从消息里抽出评论列表；抽不到就把整段用户文本当成单条评论。"""
    text = _message_text(messages)
    obj = _find_comments_object(text)
    if obj is not None:
        raw = _as_list(obj.get(_COMMENTS_KEY))
        return [_coerce_comment(item, index) for index, item in enumerate(raw)]
    if text.strip():
        return [{"comment_id": "0", "content": text, "topic_title": "", "like_count": 0}]
    return []


def coerce_n(value: Any) -> int:
    """把请求里的 ``n`` 收敛成 ``[1, _MAX_N]`` 的整数；非法值回落到 1。"""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, min(_MAX_N, parsed))


def wants_stream(body: object) -> bool:
    """请求是否显式要求 SSE 增量返回（``"stream": true``）。"""
    return _as_dict(body).get("stream") is True


def labels_content(labels: Sequence[Label]) -> str:
    """把逐条标签序列化成 chat completion 的 ``content`` 文本。"""
    return json.dumps({"labels": [label.to_raw() for label in labels]}, ensure_ascii=False)


def build_chat_completion(
    labels: Sequence[Label],
    model: str,
    prompt_tokens: int = 0,
    n: int = 1,
) -> dict[str, Any]:
    """把逐条标签拼成标准 chat completion 响应；``n`` 份候选取同一答案。"""
    content = labels_content(labels)
    completion_tokens = estimate_tokens(content)
    return {
        "id": "chatcmpl-nano-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
            for index in range(coerce_n(n))
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def iter_stream_chunks(labels: Sequence[Label], model: str, n: int = 1) -> Iterator[str]:
    """产出 ``text/event-stream`` 格式的增量帧（每帧已是完整的 SSE 消息）。

    帧序列遵循 OpenAI 的 ``chat.completion.chunk`` 约定：先一个只带 ``role`` 的帧，
    再一个承载全文 ``content`` 的帧，最后以 ``finish_reason`` 与 ``[DONE]`` 收尾。
    服务端逐帧写出、不缓冲整包，故客户端能立刻拿到首字节。
    """
    created = int(time.time())
    completion_id = "chatcmpl-nano-" + uuid.uuid4().hex[:12]
    count = coerce_n(n)
    content = labels_content(labels)

    def frame(deltas: list[dict[str, Any]], finish: str | None) -> str:
        payload: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": index, "delta": deltas[index], "finish_reason": finish}
                for index in range(count)
            ],
        }
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    yield frame([{"role": "assistant", "content": ""} for _ in range(count)], None)
    yield frame([{"content": content} for _ in range(count)], None)
    yield frame([{} for _ in range(count)], "stop")
    yield "data: [DONE]\n\n"


def _classify_messages(messages: object, topics: Sequence[str] | None) -> list[Label]:
    """抽出评论并逐条粗筛，供整包与流式两条路径共用。"""
    return [
        classify(
            comment["comment_id"],
            comment["content"],
            topic_title=comment["topic_title"],
            like_count=comment["like_count"],
            topics=topics,
        )
        for comment in extract_comments(messages)
    ]


def handle_chat(body: Any, topics: Sequence[str] | None = None) -> dict[str, Any]:
    """一站式入口：解析 chat 请求 → 抽评论 → 粗筛 → 拼整包响应。"""
    data = _as_dict(body)
    model = str(data.get("model") or "nano-jev")
    messages = data.get("messages")
    labels = _classify_messages(messages, topics)
    prompt_tokens = estimate_tokens(json.dumps(messages, ensure_ascii=False))
    return build_chat_completion(labels, model, prompt_tokens, n=coerce_n(data.get("n", 1)))


def handle_chat_stream(body: Any, topics: Sequence[str] | None = None) -> Iterator[str]:
    """一站式流式入口：解析 chat 请求 → 粗筛 → 产出 SSE 帧。"""
    data = _as_dict(body)
    model = str(data.get("model") or "nano-jev")
    labels = _classify_messages(data.get("messages"), topics)
    return iter_stream_chunks(labels, model, n=coerce_n(data.get("n", 1)))
