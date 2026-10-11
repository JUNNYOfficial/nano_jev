"""``nano_jev.nj_server`` 的集成测试：真起服务、真发 HTTP 请求。

不 mock、不打桩——服务端跑真实 socket，请求走真实 ``urllib``。这样"能不能被
jev-arena 指着用"这件事才算被端到端验证过（契约测试只测了函数，测不到路由与
序列化）。
"""

from __future__ import annotations

import json
import pathlib
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nano_jev import nj_server  # noqa: E402

# 本机可能配了系统级代理（macOS 的 ``_scproxy`` 会让 urllib 把 localhost 也走代理，
# 实测会拿到 502）。测试只连本地服务，故显式禁用代理。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@pytest.fixture()
def server_url() -> Iterator[str]:
    """在随机端口起一个真实服务，测试结束再关掉。"""
    httpd = nj_server.make_server("127.0.0.1", 0)
    host, port = httpd.server_address[0], httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _post(url: str, payload: dict) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with _OPENER.open(request, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _get(url: str) -> tuple[int, dict]:
    with _OPENER.open(url, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _post_text(url: str, payload: dict) -> tuple[int, dict, str]:
    """POST 并保留原始文本与响应头（流式响应不是单个 JSON 对象，需原样取回）。"""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with _OPENER.open(request, timeout=5) as response:
        return response.status, dict(response.headers), response.read().decode("utf-8")


def _decisions_body() -> dict:
    return {
        "model": "typesafe/jev-1.13",
        "state": {
            "comments": [
                {"comment_id": "c0", "content": "回答很准确，速度快，推荐！", "topic_title": "评测"}
            ]
        },
        "questions": {
            "c0_relevant": {"type": "noul", "criteria": {"true": "是", "false": "否"}},
            "c0_sentiment": {
                "type": "choice",
                "criteria": {"positive": None, "negative": None, "neutral": None, "mixed": None},
            },
            "c0_score": {"type": "score", "criteria": ["强烈负面", "偏负面", "中性", "偏正面", "强烈正面"]},
        },
    }


def test_health_and_models(server_url: str):
    status, payload = _get(server_url + "/health")
    assert status == 200
    assert payload["status"] == "ok"

    status, payload = _get(server_url + "/v1/models")
    assert status == 200
    assert payload["data"][0]["id"] == "nano-jev"


def test_decisions_endpoint(server_url: str):
    status, payload = _post(server_url + "/alpha/decisions", _decisions_body())
    assert status == 200
    assert payload["provider"] == "NanoJev"
    assert payload["answers"]["c0_relevant"]["noul"] > 0.5
    assert payload["answers"]["c0_sentiment"]["choice"] == "positive"


def test_chat_completions_endpoint(server_url: str):
    body = {
        "model": "nano-jev",
        "messages": [
            {"role": "system", "content": "你是标注器"},
            {
                "role": "user",
                "content": "标注下面这批评论：\n"
                + json.dumps(
                    {"comments": [{"comment_id": "c0", "content": "太差了，经常崩溃"}]},
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    status, payload = _post(server_url + "/v1/chat/completions", body)
    assert status == 200
    assert payload["choices"][0]["finish_reason"] == "stop"
    content = json.loads(payload["choices"][0]["message"]["content"])
    assert [item["comment_id"] for item in content["labels"]] == ["c0"]
    label = content["labels"][0]
    assert label["sentiment"] == "negative"
    # 引文必须仍是原文逐字子串（jev-arena 会校验）。
    assert label["evidence_quote"] in "太差了，经常崩溃"
    assert payload["usage"]["prompt_tokens"] > 0


def test_chat_completions_n_choices(server_url: str):
    body = {
        "model": "nano-jev",
        "n": 3,
        "messages": [{"role": "user", "content": json.dumps({"comments": []})}],
    }
    status, payload = _post(server_url + "/v1/chat/completions", body)
    assert status == 200
    assert [choice["index"] for choice in payload["choices"]] == [0, 1, 2]


def test_chat_completions_streaming(server_url: str):
    body = {
        "model": "nano-jev",
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {"comments": [{"comment_id": "c0", "content": "太差了，经常崩溃"}]},
                    ensure_ascii=False,
                ),
            }
        ],
    }
    status, headers, text = _post_text(server_url + "/v1/chat/completions", body)
    assert status == 200
    assert headers["Content-Type"].startswith("text/event-stream")

    frames = [line for line in text.splitlines() if line.startswith("data: ")]
    assert frames[-1] == "data: [DONE]"
    chunks = [json.loads(frame[len("data: ") :]) for frame in frames[:-1]]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    labels = json.loads(chunks[1]["choices"][0]["delta"]["content"])["labels"]
    assert labels[0]["sentiment"] == "negative"


def test_models_by_id(server_url: str):
    status, payload = _get(server_url + "/v1/models/nano-jev")
    assert status == 200
    assert payload["id"] == "nano-jev"

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(server_url + "/v1/models/does-not-exist")
    assert excinfo.value.code == 404
    error = json.loads(excinfo.value.read().decode("utf-8"))
    assert error["error"]["code"] == 404


def test_unknown_path_returns_404(server_url: str):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(server_url + "/nope", {})
    assert excinfo.value.code == 404
    payload = json.loads(excinfo.value.read().decode("utf-8"))
    assert payload["error"]["code"] == 404


def test_bad_json_returns_400(server_url: str):
    data = b"{ not json"
    request = urllib.request.Request(
        server_url + "/alpha/decisions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _OPENER.open(request, timeout=5)
    assert excinfo.value.code == 400
