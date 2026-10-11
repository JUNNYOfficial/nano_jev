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

import json
import time
import uuid
from typing import Any, Iterator, Sequence

from .nj_engine import Label, classify, estimate_tokens

#: 从用户消息里抽评论时，认这个 JSON 键。
_COMMENTS_KEY = "comments"

#: ``n`` 的上限：确定性后端下多候选取值相同，只做形状兼容，故不放大计算量。
_MAX_N = 8


def _message_text(messages: Any) -> str:
    """取最后一条 ``user`` 消息的纯文本（多模态 content 只拼文本片段）。"""
    text = ""
    if not isinstance(messages, list):
        return text
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            text = "".join(parts)
    return text


def _find_comments_object(text: str) -> dict[str, Any] | None:
    """在任意文本里找出**最后一个**含 ``comments`` 数组的 JSON 对象。"""
    decoder = json.JSONDecoder()
    best: dict[str, Any] | None = None
    index = text.find("{")
    while index != -1:
        try:
            obj, _end = decoder.raw_decode(text, index)
        except ValueError:
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get(_COMMENTS_KEY), list):
            best = obj
        index = text.find("{", index + 1)
    return best


def _coerce_comment(item: Any, index: int) -> dict[str, Any]:
    """把数组里的一个元素收敛成标准评论字典。"""
    if isinstance(item, dict):
        return {
            "comment_id": str(item.get("comment_id", item.get("id", index))),
            "content": str(item.get("content", "")),
            "topic_title": str(item.get("topic_title", "")),
            "like_count": item.get("like_count", 0),
        }
    return {"comment_id": str(index), "content": str(item), "topic_title": "", "like_count": 0}


def extract_comments(messages: Any) -> list[dict[str, Any]]:
    """从消息里抽出评论列表；抽不到就把整段用户文本当成单条评论。"""
    text = _message_text(messages)
    obj = _find_comments_object(text)
    if obj is not None:
        raw = obj[_COMMENTS_KEY]
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


def wants_stream(body: Any) -> bool:
    """请求是否显式要求 SSE 增量返回（``"stream": true``）。"""
    return isinstance(body, dict) and body.get("stream") is True


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
        payload = {
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


def _classify_messages(messages: Any, topics: Sequence[str] | None) -> list[Label]:
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
    if not isinstance(body, dict):
        body = {}
    model = str(body.get("model") or "nano-jev")
    messages = body.get("messages")
    labels = _classify_messages(messages, topics)
    prompt_tokens = estimate_tokens(json.dumps(messages, ensure_ascii=False))
    return build_chat_completion(labels, model, prompt_tokens, n=body.get("n", 1))


def handle_chat_stream(body: Any, topics: Sequence[str] | None = None) -> Iterator[str]:
    """一站式流式入口：解析 chat 请求 → 粗筛 → 产出 SSE 帧。"""
    if not isinstance(body, dict):
        body = {}
    model = str(body.get("model") or "nano-jev")
    labels = _classify_messages(body.get("messages"), topics)
    return iter_stream_chunks(labels, model, n=body.get("n", 1))
