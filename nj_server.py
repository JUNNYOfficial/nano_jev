"""零依赖 HTTP 服务与命令行入口（``python -m nano_jev.nj_server``）。

只用标准库 ``http.server``，故**可整包复制到任何有 Python 的机器上直接跑**，不需要
装 Flask/FastAPI。暴露的路径：

* ``POST …/alpha/decisions`` —— Jev 原生契约（左道）。
* ``POST …/chat/completions`` —— OpenAI 兼容契约（右道亦可挂）；带 ``"stream": true``
  时回 ``text/event-stream`` 的增量帧。
* ``GET …/health`` —— 健康检查与版本。
* ``GET …/models`` / ``GET …/models/{model}`` —— OpenAI 风格的模型清单与单项查询（便于
  某些客户端探测）。

**鉴权**：设 ``NANO_JV_TOKEN``（或 ``--token``）后，请求须带匹配的
``Authorization: Bearer <token>``；未设则放行（本地使用）。
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

from . import VERSION
from .nj_contract import DEFAULT_TOPICS, handle_decisions
from .nj_engine import classify
from .nj_openai import handle_chat, handle_chat_stream, wants_stream

#: 相关性判据用的主题词（可被 CLI/环境变量覆盖）。
_TOPICS: tuple[str, ...] = tuple(DEFAULT_TOPICS)

#: 可选的访问令牌；为空表示不鉴权。
_TOKEN: str | None = None

#: 对外暴露的模型对象（OpenAI 风格），``/v1/models`` 与其子路径共用。
_MODEL: dict[str, Any] = {"id": "nano-jev", "object": "model", "owned_by": "nano_jev"}


def parse_topics(value: str | None) -> tuple[str, ...]:
    """把逗号分隔的主题词解析成元组；空值回落到默认主题词。"""
    if not value:
        return tuple(DEFAULT_TOPICS)
    parts = [part.strip() for part in value.split(",")]
    return tuple(part for part in parts if part) or tuple(DEFAULT_TOPICS)


class _Handler(BaseHTTPRequestHandler):
    """请求处理器。契约错误一律回 ``{"error": {"message", "code"}}``。"""

    server_version = "nano-jev/" + VERSION
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """静音默认的逐请求日志（打点交给调用方，避免污染输出）。"""

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, code: int, status: int) -> None:
        self._send_json({"error": {"message": message, "code": code}}, status=status)

    def _send_event_stream(self, chunks: Iterable[str]) -> None:
        """以 HTTP/1.1 分块编码逐帧写出 SSE；末块终止，无需 ``Content-Length``。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for chunk in chunks:
            data = chunk.encode("utf-8")
            self.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _read_body(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _authorized(self) -> bool:
        if not _TOKEN:
            return True
        header = self.headers.get("Authorization") or ""
        return header.strip() == f"Bearer {_TOKEN}"

    def _path(self) -> str:
        return urlparse(self.path).path.rstrip("/") or "/"

    def do_GET(self) -> None:  # noqa: N802  （BaseHTTPRequestHandler 约定名）
        path = self._path()
        if path.endswith("/health"):
            self._send_json({"status": "ok", "version": VERSION, "backend": "nano-jev"})
            return
        if path.endswith("/models"):
            self._send_json({"object": "list", "data": [dict(_MODEL)]})
            return
        if "/models/" in path:
            model_id = path.rsplit("/", 1)[-1]
            if model_id == _MODEL["id"]:
                self._send_json(dict(_MODEL))
            else:
                self._send_error_json(f"模型不存在：{model_id}", 404, 404)
            return
        self._send_json(
            {
                "service": "nano_jev",
                "version": VERSION,
                "endpoints": [
                    "POST /alpha/decisions",
                    "POST /v1/chat/completions",
                    "GET /health",
                    "GET /v1/models",
                    "GET /v1/models/{model}",
                ],
            }
        )

    def do_POST(self) -> None:  # noqa: N802  （BaseHTTPRequestHandler 约定名）
        if not self._authorized():
            self._send_error_json("未授权", 401, 401)
            return
        try:
            body = self._read_body()
        except (ValueError, UnicodeDecodeError) as exc:
            self._send_error_json(f"请求体不是合法 JSON：{exc}", 400, 400)
            return

        path = self._path()
        payload: Any = None
        stream_chunks: Iterable[str] | None = None
        try:
            if path.endswith("/alpha/decisions"):
                payload = handle_decisions(body, _TOPICS)
            elif path.endswith("/chat/completions"):
                if wants_stream(body):
                    # 分类在 handle_chat_stream() 内同步完成，故错误仍能被下面捕获；
                    # 逐帧写出放到 try 之外，避免半途写失败后再补发 JSON 头。
                    stream_chunks = handle_chat_stream(body, _TOPICS)
                else:
                    payload = handle_chat(body, _TOPICS)
            else:
                self._send_error_json(f"未知路径：{path}", 404, 404)
                return
        except Exception as exc:  # noqa: BLE001  （服务端兜底：任何异常都转成契约错误）
            self._send_error_json(f"内部错误：{exc}", 500, 500)
            return
        if stream_chunks is not None:
            self._send_event_stream(stream_chunks)
            return
        self._send_json(payload)


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    """构造（但不启动）HTTP 服务；``port=0`` 时由系统分配端口，便于测试。"""
    return ThreadingHTTPServer((host, port), _Handler)


def serve(host: str, port: int, topics: Sequence[str] | None = None, token: str | None = None) -> None:
    """启动服务并阻塞运行。"""
    global _TOPICS, _TOKEN
    if topics:
        _TOPICS = tuple(topics)
    _TOKEN = token
    httpd = make_server(host, port)
    bound = httpd.server_address
    print(f"nano_jev 服务已启动：http://{bound[0]}:{bound[1]}（版本 {VERSION}）", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("收到中断，正在退出……", file=sys.stderr)
    finally:
        httpd.server_close()


def _run_label(args: argparse.Namespace) -> int:
    """``label`` 子命令：对一段文本做一次粗筛，打印标签 JSON。"""
    text = sys.stdin.read() if args.stdin else args.text
    label = classify(args.id, text, topics=parse_topics(args.topics))
    print(json.dumps(label.to_raw(), ensure_ascii=False, indent=2))
    return 0


def _run_decide(args: argparse.Namespace) -> int:
    """``decide`` 子命令：读入一份 decisions 请求，输出响应 JSON。"""
    raw = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    body = json.loads(raw)
    response = handle_decisions(body, parse_topics(args.topics))
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(prog="nano_jev", description="Nano Jev 平替：确定性粗筛服务")
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="启动 HTTP 服务")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--topics", default=None, help="逗号分隔的主题词，覆盖默认相关性领域")
    serve_parser.add_argument("--token", default=None, help="可选的访问令牌")

    label_parser = sub.add_parser("label", help="对一段文本做一次粗筛")
    label_parser.add_argument("text", nargs="?", default="")
    label_parser.add_argument("--id", default="0")
    label_parser.add_argument("--stdin", action="store_true", help="从标准输入读文本")
    label_parser.add_argument("--topics", default=None)

    decide_parser = sub.add_parser("decide", help="读入 decisions 请求，输出响应 JSON")
    decide_parser.add_argument("--file", default="-", help="请求 JSON 文件；默认从标准输入读")
    decide_parser.add_argument("--topics", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        serve(args.host, args.port, parse_topics(args.topics), args.token)
        return 0
    if args.command == "label":
        return _run_label(args)
    if args.command == "decide":
        return _run_decide(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
