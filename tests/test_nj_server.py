"""``nano_jev.nj_server`` 的集成测试：真起服务、真发 HTTP 请求。

不 mock、不打桩——服务端跑真实 socket，请求走真实 ``urllib``。这样"能不能被
jev-arena 指着用"这件事才算被端到端验证过（契约测试只测了函数，测不到路由与
序列化）。
"""

from __future__ import annotations

import http.client
import json
import pathlib
import socket
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import pytest

# 本文件位于 nano_jev/tests/ 下，故要往上三层才回到"能看到 nano_jev 包"的仓库父目录。
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
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
    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture()
def secured_server_url() -> Iterator[str]:
    """带访问令牌的服务；用于验证鉴权，以及 401 后 keep-alive 连接不被污染。"""
    previous = nj_server._token
    nj_server._token = "s3cret"
    httpd = nj_server.make_server("127.0.0.1", 0)
    host, port = httpd.server_address[0], httpd.server_address[1]
    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        nj_server._token = previous


def _auth_headers(token: str) -> dict[str, str]:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}


def _post(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with _OPENER.open(request, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _get(url: str) -> tuple[int, dict[str, Any]]:
    with _OPENER.open(url, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _post_text(url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any], str]:
    """POST 并保留原始文本与响应头（流式响应不是单个 JSON 对象，需原样取回）。"""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with _OPENER.open(request, timeout=5) as response:
        return response.status, dict(response.headers), response.read().decode("utf-8")


def _host_port(server_url: str) -> tuple[str, int]:
    """把 ``http://host:port`` 拆成 ``(host, port)``。"""
    host, _, port = server_url[len("http://") :].partition(":")
    return host, int(port)


def _raw_exchange(server_url: str, request: bytes) -> bytes:
    """发原始字节并读到 EOF，返回完整响应。

    用于构造 urlopen 发不出的畸形请求（如非法/超大的 ``Content-Length``），以及
    逐字节核对分块编码的结尾。
    """
    host, port = _host_port(server_url)
    conn = socket.create_connection((host, port), timeout=5)
    try:
        conn.sendall(request)
        pieces: list[bytes] = []
        while True:
            try:
                piece = conn.recv(65536)
            except TimeoutError:
                break
            if not piece:
                break
            pieces.append(piece)
        return b"".join(pieces)
    finally:
        conn.close()


def _raw_post(server_url: str, path: str, body: bytes, headers: str = "") -> bytes:
    host, _port = _host_port(server_url)
    head = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        "Content-Type: application/json\r\n"
        f"{headers}"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    )
    return _raw_exchange(server_url, head.encode("ascii") + body)


def _decisions_body() -> dict[str, Any]:
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


def test_get_on_post_only_path_returns_405(server_url: str):
    """回归：``GET /alpha/decisions`` 之前会以 200 回一个和请求无关的服务索引。"""
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(server_url + "/alpha/decisions")
    assert excinfo.value.code == 405
    assert excinfo.value.headers["Allow"] == "POST, OPTIONS"
    payload = json.loads(excinfo.value.read().decode("utf-8"))
    assert payload["error"]["code"] == 405


def test_post_on_get_only_path_returns_405(server_url: str):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(server_url + "/health", {})
    assert excinfo.value.code == 405
    assert excinfo.value.headers["Allow"] == "GET, OPTIONS"


def test_unsupported_methods_return_json_not_html_501(server_url: str):
    """回归：PY 默认的 501 是 HTML，违反「契约错误一律 JSON」的约定。"""
    host, _port = _host_port(server_url)
    for method in ("PUT", "DELETE", "PATCH"):
        response = _raw_exchange(
            server_url,
            f"{method} /alpha/decisions HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode(),
        )
        assert response.startswith(b"HTTP/1.1 405 "), (method, response[:40])
        assert b"application/json" in response
        assert b'"code": 405' in response

    # 未知路径仍是 404。
    response = _raw_exchange(
        server_url,
        f"PUT /nope HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode(),
    )
    assert response.startswith(b"HTTP/1.1 404 ")


def test_options_returns_allow_header(server_url: str):
    host, _port = _host_port(server_url)
    response = _raw_exchange(
        server_url,
        f"OPTIONS /alpha/decisions HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode(),
    )
    assert response.startswith(b"HTTP/1.1 204 ")
    assert b"Allow: POST, OPTIONS" in response


def test_model_subpath_methods_are_consistent(server_url: str):
    """回归：``/v1/models/{model}`` 单靠后缀抓不到，OPTIONS/错误方法会掉到 404。

    列表路径回 405/204，子路径却回 404，是行为不一致。
    """
    host, _port = _host_port(server_url)

    response = _raw_exchange(
        server_url,
        f"OPTIONS /v1/models/nano-jev HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode(),
    )
    assert response.startswith(b"HTTP/1.1 204 "), response[:40]
    assert b"Allow: GET, OPTIONS" in response

    response = _raw_exchange(
        server_url,
        (
            f"POST /v1/models/nano-jev HTTP/1.1\r\nHost: {host}\r\n"
            "Content-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"
        ).encode(),
    )
    assert response.startswith(b"HTTP/1.1 405 "), response[:40]


def test_internal_error_is_not_leaked_to_client(server_url: str, monkeypatch: pytest.MonkeyPatch):
    """回归：500 之前会把异常原文回给客户端，等于对外泄露内部细节。"""

    def boom(_body: Any, _topics: Any = None) -> Any:
        raise RuntimeError("secret内部路径/var/tmp/x.py")

    monkeypatch.setattr(nj_server, "handle_decisions", boom)
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(server_url + "/alpha/decisions", _decisions_body())
    assert excinfo.value.code == 500
    body = excinfo.value.read().decode("utf-8")
    assert "secret" not in body
    assert "内部错误" in body


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


def test_auth_token_enforced(secured_server_url: str):
    # 未带令牌 → 401
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(secured_server_url + "/alpha/decisions", _decisions_body())
    assert excinfo.value.code == 401

    # 正确令牌 → 200
    data = json.dumps(_decisions_body()).encode("utf-8")
    request = urllib.request.Request(
        secured_server_url + "/alpha/decisions",
        data=data,
        headers=_auth_headers("s3cret"),
        method="POST",
    )
    with _OPENER.open(request, timeout=5) as response:
        assert response.status == 200


def test_unauthorized_keeps_connection_usable(secured_server_url: str):
    """回归：401 若未读完请求体，HTTP/1.1 keep-alive 的下一个请求会被污染。"""
    host, _, port = secured_server_url[len("http://") :].partition(":")
    body = json.dumps(_decisions_body())
    conn = http.client.HTTPConnection(host, int(port), timeout=5)
    try:
        conn.request("POST", "/alpha/decisions", body=body, headers=_auth_headers("wrong"))
        first = conn.getresponse()
        first.read()
        assert first.status == 401

        conn.request("POST", "/alpha/decisions", body=body, headers=_auth_headers("s3cret"))
        second = conn.getresponse()
        payload = json.loads(second.read().decode("utf-8"))
        assert second.status == 200
        assert payload["provider"] == "NanoJev"
    finally:
        conn.close()


def test_parse_topics_precedence(monkeypatch: pytest.MonkeyPatch):
    """主题词取值优先级：入参 → NANO_JV_TOPICS → 默认词表。"""
    monkeypatch.setenv("NANO_JV_TOPICS", "恒星, 行星")
    assert nj_server.parse_topics(None) == ("恒星", "行星")
    assert nj_server.parse_topics("模型") == ("模型",)

    monkeypatch.delenv("NANO_JV_TOPICS")
    assert nj_server.parse_topics(None) == tuple(nj_server.DEFAULT_TOPICS)
    assert nj_server.parse_topics(",") == tuple(nj_server.DEFAULT_TOPICS)


@pytest.fixture()
def env_configured_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """按环境变量配置起的服务；用于验证 NANO_JV_TOPICS / NANO_JV_TOKEN 真的生效。"""
    monkeypatch.setenv("NANO_JV_TOPICS", "恒星,行星")
    monkeypatch.setenv("NANO_JV_TOKEN", "env-token")
    previous = (nj_server._topics, nj_server._token)
    nj_server.apply_config()
    assert nj_server._topics == ("恒星", "行星")
    assert nj_server._token == "env-token"

    httpd = nj_server.make_server("127.0.0.1", 0)
    host, port = httpd.server_address[0], httpd.server_address[1]
    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
        nj_server._topics, nj_server._token = previous


def test_env_token_is_enforced(env_configured_server: str):
    """回归：文档承诺的 NANO_JV_TOKEN 必须真的开启鉴权。"""
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(env_configured_server + "/alpha/decisions", _decisions_body())
    assert excinfo.value.code == 401


def test_env_topics_change_relevance(env_configured_server: str):
    """回归：文档承诺的 NANO_JV_TOPICS 必须真的换掉相关性判据。

    「恒星的一生如何演化」在默认词表下判不相关，换成 ``恒星`` 后必须判相关。
    """
    body = {
        "model": "nano-jev",
        "state": {
            "comments": [{"comment_id": "c0", "content": "恒星的一生如何演化", "topic_title": ""}]
        },
        "questions": {"c0_relevant": {"type": "noul", "criteria": {"true": "是", "false": "否"}}},
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        env_configured_server + "/alpha/decisions",
        data=data,
        headers=_auth_headers("env-token"),
        method="POST",
    )
    with _OPENER.open(request, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    assert payload["answers"]["c0_relevant"]["noul"] > 0.5


def test_negative_content_length_does_not_hang(server_url: str):
    """回归：``Content-Length: -1`` 会让 ``rfile.read(-1)`` 读到 EOF，永久挂住服务线程。"""
    host, _port = _host_port(server_url)
    response = _raw_exchange(
        server_url,
        (
            "POST /alpha/decisions HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: -1\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii"),
    )
    assert response.startswith(b"HTTP/1.1 400 ")
    assert b"Content-Length" in response
    # 服务必须还活着（挂死的话这里会超时失败）。
    status, _payload = _post(server_url + "/alpha/decisions", _decisions_body())
    assert status == 200


def test_non_numeric_content_length_is_rejected(server_url: str):
    host, _port = _host_port(server_url)
    response = _raw_exchange(
        server_url,
        (
            "POST /alpha/decisions HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: notanumber\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii"),
    )
    assert response.startswith(b"HTTP/1.1 400 ")


def test_oversized_content_length_is_rejected(server_url: str):
    """回归：超大 ``Content-Length`` 不该让服务按声明长度去分配/读取。"""
    host, _port = _host_port(server_url)
    response = _raw_exchange(
        server_url,
        (
            "POST /alpha/decisions HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {nj_server._MAX_BODY_BYTES + 1}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii"),
    )
    assert response.startswith(b"HTTP/1.1 413 ")


def test_rejected_body_length_closes_connection(server_url: str):
    """长度越界的请求不能被复用：请求体没读走，残留字节会污染同一条连接的下个请求。"""
    response = _raw_exchange(
        server_url,
        (
            "POST /alpha/decisions HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: -1\r\n\r\n"
        ).encode("ascii"),
    )
    assert response.startswith(b"HTTP/1.1 400 ")
    assert b"Connection: close" in response


def test_event_stream_terminates_when_chunk_iteration_fails(
    server_url: str, monkeypatch: pytest.MonkeyPatch
):
    """回归：分块流中途失败也必须补上终止块，否则客户端会一直等 chunked 结束。"""

    def broken_stream(_body: Any, _topics: Any = None) -> Iterator[str]:
        yield 'data: {"partial": true}\n\n'
        raise BrokenPipeError("客户端提前断开")

    monkeypatch.setattr(nj_server, "handle_chat_stream", broken_stream)
    body = json.dumps({"model": "nano-jev", "stream": True, "messages": []}).encode("utf-8")
    response = _raw_post(server_url, "/v1/chat/completions", body)
    assert b"text/event-stream" in response
    assert b"partial" in response
    assert response.endswith(b"0\r\n\r\n"), "缺少分块编码终止块"


def test_rejected_method_with_body_does_not_pollute_connection(server_url: str):
    """回归：405 分支不读请求体，带 body 的 PUT 会把残留下一个请求（同 401 那类 bug）。"""
    host, port = _host_port(server_url)
    conn = socket.create_connection((host, port), timeout=5)
    try:
        body = json.dumps(_decisions_body())
        conn.sendall(
            (
                "PUT /alpha/decisions HTTP/1.1\r\n"
                f"Host: {host}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(body.encode())}\r\n"
                "\r\n"
                + body
            ).encode()
        )
        conn.settimeout(5)
        chunks = []
        while True:
            try:
                piece = conn.recv(65536)
            except OSError:
                break
            if not piece:
                break
            chunks.append(piece)
        response = b"".join(chunks).decode("latin1")
        assert response.startswith("HTTP/1.1 405 "), response[:60]
        # 必须声明断开，否则残留 body 会被当成同一条连接上的下一个请求来解析。
        assert "Connection: close" in response
    finally:
        conn.close()

    # 新连接仍然可用。
    status, _payload = _post(server_url + "/alpha/decisions", _decisions_body())
    assert status == 200


def test_loopback_detection():
    for host in ("127.0.0.1", "127.0.0.5", "localhost", "::1"):
        assert nj_server._is_loopback(host) is True, host
    for host in ("0.0.0.0", "::", "192.168.1.10", ""):
        assert nj_server._is_loopback(host) is False, host


def test_cli_version_exits_zero(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as excinfo:
        nj_server.main(["--version"])
    assert excinfo.value.code == 0
    assert f"nano-jev {nj_server.VERSION}" in capsys.readouterr().out


def test_cli_label_rejects_text_with_stdin(capsys: pytest.CaptureFixture[str]):
    """回归：位置参数与 ``--stdin`` 同时给时，旧行为是静默以 stdin 为准。"""
    assert nj_server.main(["label", "一段文本", "--stdin"]) == 2
    assert "只能给一个" in capsys.readouterr().err


def test_cli_label_by_flag_and_positional(capsys: pytest.CaptureFixture[str]):
    assert nj_server.main(["label", "这个模型很好用，推荐"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["is_relevant"] is True
    assert payload["sentiment"] == "positive"


def test_cli_no_subcommand_exits_nonzero(capsys: pytest.CaptureFixture[str]):
    """回归：无子命令时只打印 help 却回 0，会把「没做任何事」误判成成功。"""
    assert nj_server.main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_decide_missing_file_is_clean_error(capsys: pytest.CaptureFixture[str]):
    """回归：``decide --file`` 指向不存在的文件时，之前是整段 traceback。"""
    assert nj_server.main(["decide", "--file", "/nonexistent/nope.json"]) == 1
    assert "无法读取" in capsys.readouterr().err
    assert "Traceback" not in capsys.readouterr().err


def test_cli_decide_bad_json_is_clean_error(capsys: pytest.CaptureFixture[str], tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    assert nj_server.main(["decide", "--file", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "不是合法 JSON" in err
    assert "Traceback" not in err


def test_non_loopback_without_token_warns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """绑定在非回环地址且没设令牌时必须给出警告，否则会静默把服务开到局域网。"""
    monkeypatch.delenv("NANO_JV_TOKEN", raising=False)
    previous = (nj_server._topics, nj_server._token)

    def stop_at_make_server(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(nj_server, "make_server", stop_at_make_server)
    try:
        with pytest.raises(KeyboardInterrupt):
            nj_server.serve("0.0.0.0", 0, token=None)
        err = capsys.readouterr().err
        assert "警告" in err
        assert "NANO_JV_TOKEN" in err
    finally:
        nj_server._topics, nj_server._token = previous
