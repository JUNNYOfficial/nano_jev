"""``nano_jev.nj_openai`` 的单元测试：聚焦 OpenAI 协议的两处形状约定。

只测纯函数与生成器，不起服务——端到端的那一段留给 ``test_nj_server.py``。
覆盖：``n`` 收敛、``stream`` 判定、整包响应的候选数、以及 SSE 帧序列的完整性。
"""

from __future__ import annotations

import json
import pathlib
import sys

# 本文件位于 nano_jev/tests/ 下，故要往上三层才回到"能看到 nano_jev 包"的仓库父目录。
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nano_jev import nj_openai  # noqa: E402
from nano_jev.nj_engine import classify  # noqa: E402


def _labels() -> list:
    return [classify("c0", "太差了，经常崩溃"), classify("c1", "很好，推荐")]


def test_coerce_n_bounds_and_defaults():
    assert nj_openai.coerce_n(3) == 3
    assert nj_openai.coerce_n("2") == 2
    assert nj_openai.coerce_n(0) == 1
    assert nj_openai.coerce_n(99) == nj_openai._MAX_N
    assert nj_openai.coerce_n(None) == 1
    assert nj_openai.coerce_n("abc") == 1


def test_wants_stream_only_on_explicit_true():
    assert nj_openai.wants_stream({"stream": True}) is True
    assert nj_openai.wants_stream({"stream": False}) is False
    assert nj_openai.wants_stream({"stream": "true"}) is False
    assert nj_openai.wants_stream("not-a-dict") is False


def test_build_chat_completion_repeats_choice_n_times():
    payload = nj_openai.build_chat_completion(_labels(), "nano-jev", prompt_tokens=7, n=3)
    assert payload["object"] == "chat.completion"
    assert [choice["index"] for choice in payload["choices"]] == [0, 1, 2]
    contents = {choice["message"]["content"] for choice in payload["choices"]}
    assert len(contents) == 1  # 确定性后端：n 个候选内容一致
    labels = json.loads(contents.pop())["labels"]
    assert [item["comment_id"] for item in labels] == ["c0", "c1"]
    assert payload["usage"]["prompt_tokens"] == 7


def _sse_payloads(chunks: list[str]) -> list:
    """把 SSE 文本帧还原成对象列表（丢弃 ``[DONE]`` 哨兵）。"""
    objects = []
    for chunk in chunks:
        assert chunk.startswith("data: ") and chunk.endswith("\n\n")
        body = chunk[len("data: ") : -2]
        if body != "[DONE]":
            objects.append(json.loads(body))
    return objects


def test_iter_stream_chunks_frame_sequence():
    chunks = list(nj_openai.iter_stream_chunks(_labels(), "nano-jev"))
    assert chunks[-1] == "data: [DONE]\n\n"
    objects = _sse_payloads(chunks)

    assert all(obj["object"] == "chat.completion.chunk" for obj in objects)
    assert len({obj["id"] for obj in objects}) == 1  # 同一 completion id

    role_frame, content_frame, stop_frame = objects
    assert role_frame["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert role_frame["choices"][0]["finish_reason"] is None

    labels = json.loads(content_frame["choices"][0]["delta"]["content"])["labels"]
    assert [item["comment_id"] for item in labels] == ["c0", "c1"]

    assert stop_frame["choices"][0]["finish_reason"] == "stop"


def test_handle_chat_stream_ignores_n_for_shared_content():
    body = {
        "model": "nano-jev",
        "n": 2,
        "messages": [{"role": "user", "content": json.dumps({"comments": []})}],
    }
    chunks = list(nj_openai.handle_chat_stream(body))
    objects = _sse_payloads(chunks)
    assert [choice["index"] for choice in objects[0]["choices"]] == [0, 1]
