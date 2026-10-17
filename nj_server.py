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
import contextlib
import hmac
import json
import os
import sys
import traceback
from collections.abc import Iterable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from . import VERSION
from .nj_cache import HybridCache
from .nj_contract import DEFAULT_TOPICS, handle_decisions
from .nj_engine import Labeler, classify
from .nj_guard import GuardPolicy
from .nj_hybrid import HybridPolicy
from .nj_llm import OpenAICompatJudge, make_hybrid_labeler
from .nj_openai import handle_chat, handle_chat_stream, wants_stream

#: 主题词与访问令牌的环境变量名（命令行参数优先于它们）。
_TOPICS_ENV = "NANO_JV_TOPICS"
_TOKEN_ENV = "NANO_JV_TOKEN"

#: 级联后端的可选环境变量（命令行参数优先）。
_LLM_ENDPOINT_ENV = "NANO_JV_LLM_ENDPOINT"
_LLM_MODEL_ENV = "NANO_JV_LLM_MODEL"
_LLM_KEY_ENV = "NANO_JV_LLM_API_KEY"

#: 相关性判据用的主题词（可被 CLI/环境变量覆盖，见 :func:`apply_config`）。
_topics: tuple[str, ...] = tuple(DEFAULT_TOPICS)

#: 可选的访问令牌；为空表示不鉴权。
_token: str | None = None

#: 可选的逐条判断函数。``None``（默认）表示纯确定性粗筛——服务不联网、不调模型。
_labeler: Labeler | None = None

#: 级联模式下的缓存句柄（关闭时为空缓存，用于在 ``/health`` 报 0 命中）。
_cache: HybridCache = HybridCache(namespace="")

#: 对外暴露的模型对象（OpenAI 风格），``/v1/models`` 与其子路径共用。
_MODEL: dict[str, Any] = {"id": "nano-jev", "object": "model", "owned_by": "nano_jev"}

#: 请求体字节上限。jev-arena 一批评论的请求上限约 32k token（约百 KB 量级），
#: 这里留足两个数量级的余量；超线直接拒掉，不给「用超大 Content-Length 拖垮进程」的机会。
_MAX_BODY_BYTES = 32 * 1024 * 1024

#: 只接受 POST 的路径（按后缀匹配，与分发逻辑一致）与只接受 GET 的路径。
_POST_ONLY_SUFFIXES = ("/alpha/decisions", "/chat/completions")
_GET_ONLY_SUFFIXES = ("/health", "/models")

#: 绑到这几种地址时视为「仅本机可见」，不带令牌也不算暴露。
_LOOPBACK_HOSTS = frozenset({"localhost", "::1", "[::1]"})


def _is_loopback(host: str) -> bool:
    """``host`` 是否只对本机可见（``127.0.0.0/8`` / ``::1`` / ``localhost``）。"""
    if host in _LOOPBACK_HOSTS:
        return True
    return host.startswith("127.")


def parse_topics(value: str | None) -> tuple[str, ...]:
    """把逗号分隔的主题词解析成元组。

    取值优先级：入参 → 环境变量 ``NANO_JV_TOPICS`` → 默认主题词。故 ``None`` 表示
    「没在命令行上给」，此时才去读环境变量。
    """
    raw = value if value else os.environ.get(_TOPICS_ENV)
    if not raw:
        return tuple(DEFAULT_TOPICS)
    parts = [part.strip() for part in raw.split(",")]
    return tuple(part for part in parts if part) or tuple(DEFAULT_TOPICS)


def _env_token() -> str | None:
    """环境变量 ``NANO_JV_TOKEN`` 提供的访问令牌；未设或为空串均视为未设。"""
    return os.environ.get(_TOKEN_ENV) or None


def configure_hybrid(
    judge: object | None = None,
    *,
    policy: HybridPolicy | None = None,
    guard: GuardPolicy | None = None,
    cache_namespace: str = "",
    cache_size: int = 1024,
    cache_ttl_s: float | None = None,
    topics: Sequence[str] | None = None,
) -> HybridCache:
    """启用/关闭服务的级联后端；返回生效的缓存句柄。

    ``judge`` 为 ``None`` 时关闭级联，服务退回纯确定性粗筛（不联网、不调模型）。
    否则逐条评论先粗筛，只有策略要求升级时才调用该回调。

    缓存默认关闭：``cache_namespace`` 留空即不缓存。开启时请用
    ``"<模型 id>/<提示词版本>"`` 作为命名空间，避免换模型后读到旧结论。
    """
    global _labeler, _cache
    if judge is None:
        _labeler = None
        _cache = HybridCache(namespace="")
        return _cache

    _cache = HybridCache(
        namespace=cache_namespace, maxsize=cache_size, ttl_s=cache_ttl_s
    )
    _labeler = make_hybrid_labeler(
        judge,
        policy=policy if policy is not None else HybridPolicy(),
        guard=guard,
        cache=_cache if _cache.enabled else None,
        topics=topics if topics is not None else _topics,
    )
    return _cache


def hybrid_status() -> dict[str, object]:
    """级联后端的只读状态，供 ``/health`` 与排障使用。"""
    return {"hybrid": _labeler is not None, "cache": _cache.stats()}


def apply_config(topics: Sequence[str] | None = None, token: str | None = None) -> None:
    """把配置写入模块级状态，供 :func:`serve` 与测试共用。

    ``topics`` / ``token`` 为 ``None`` 时分别回落到环境变量 ``NANO_JV_TOPICS`` /
    ``NANO_JV_TOKEN``；再没有才用默认主题词、不开鉴权。
    """
    global _topics, _token
    _topics = tuple(topics) if topics else parse_topics(None)
    _token = token if token is not None else _env_token()


class _ContractError(Exception):
    """请求体不合法：带上对外的错误码与 HTTP 状态，供 :meth:`_Handler.do_POST` 转换成响应。"""

    def __init__(self, message: str, code: int, status: int) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


class _Handler(BaseHTTPRequestHandler):
    """请求处理器。契约错误一律回 ``{"error": {"message", "code"}}``。"""

    server_version = "nano-jev/" + VERSION
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """静音默认的逐请求日志（打点交给调用方，避免污染输出）。"""

    def _send_json(
        self,
        payload: Any,
        status: int = 200,
        close: bool = False,
        headers: Sequence[tuple[str, str]] = (),
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers:
            self.send_header(name, value)
        if close:
            # 请求体没读完（或压根没读）时不能再复用连接：残留字节会被当成下一个请求，
            # 污染同一条连接。显式声明并断开，让客户端另开连接重发。
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(
        self,
        message: str,
        code: int,
        status: int,
        close: bool = False,
        headers: Sequence[tuple[str, str]] = (),
    ) -> None:
        self._send_json(
            {"error": {"message": message, "code": code}},
            status=status,
            close=close,
            headers=headers,
        )

    def _allowed_methods(self, path: str) -> tuple[str, ...]:
        """该路径接受的 HTTP 方法；空元组表示路径未知。

        顺序要与 ``do_GET`` / ``do_POST`` 的分发保持一致：``/v1/models/{model}``
        单靠 ``endswith`` 抓不到（它不以 ``/models`` 结尾），故单独按 ``/models/``
        前缀判断。否则 ``OPTIONS`` / 方法不对的请求会对这个子路径回 404，而列表
        路径却回 405/204，行为不一致。
        """
        if path.endswith(_POST_ONLY_SUFFIXES):
            return ("POST", "OPTIONS")
        if path.endswith(_GET_ONLY_SUFFIXES):
            return ("GET", "OPTIONS")
        if "/models/" in path:
            return ("GET", "OPTIONS")
        return ()

    def _send_event_stream(self, chunks: Iterable[str]) -> None:
        """以 HTTP/1.1 分块编码逐帧写出 SSE；末块终止，无需 ``Content-Length``。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for chunk in chunks:
                data = chunk.encode("utf-8")
                self.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
                self.wfile.flush()
        except OSError:
            # 客户端提前断开时终止块也写不出去：断开连接即可，别把异常抛给 socketserver。
            self.close_connection = True
        finally:
            # 无论正常收尾还是中途失败都要写终止块：否则客户端等 chunked 流结束会一直挂到超时。
            with contextlib.suppress(OSError):
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

    def _read_body(self) -> Any:
        """按 ``Content-Length`` 读并解析请求体。

        越界的长度必须在下读之前拦掉：``rfile.read(-1)`` 会读到 EOF 把线程永久挂住，
        而超大长度会直接吃内存——两者都是「一个请求打死一个线程」。这两种情况都
        **不读**请求体，故调用方须 ``close=True`` 断开连接。
        """
        raw_length = self.headers.get("Content-Length")
        length = 0
        if raw_length:
            try:
                length = int(raw_length.strip())
            except ValueError as exc:
                raise _ContractError(f"Content-Length 不是整数：{raw_length}", 400, 400) from exc
            if length < 0:
                raise _ContractError(f"Content-Length 不能为负：{length}", 400, 400)
            if length > _MAX_BODY_BYTES:
                raise _ContractError(
                    f"请求体过大：{length} 字节，上限 {_MAX_BODY_BYTES} 字节", 413, 413
                )
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _authorized(self) -> bool:
        if not _token:
            return True
        header = self.headers.get("Authorization") or ""
        # 常量时间比较，避免用响应时间逐字节试出令牌（--host 0.0.0.0 时尤其值得）。
        # 各自编成 bytes：compare_digest 不吃非 ASCII 的 str，而令牌可能含中文。
        return hmac.compare_digest(header.strip().encode("utf-8"), f"Bearer {_token}".encode())

    def _path(self) -> str:
        return urlparse(self.path).path.rstrip("/") or "/"

    def _reject_method(self) -> None:
        """已知路径但方法不对：回 405 + ``Allow``；路径不认识则回 404。

        以 JSON 报错、而不是走 ``BaseHTTPRequestHandler`` 默认的 HTML 501，是为了守住
        「契约错误一律回 ``{"error": {...}}``」这条约定——非 JSON 的报错会让按契约解析
        响应体的客户端拿到解析失败而不是可读的错误。

        这些分支**没有读请求体**，故一律 ``close=True``：否则（如带 body 的 ``PUT``）
        残留字节会被当成同一条 keep-alive 连接上的下一个请求——和 401 分支当初那个
        bug 是同一类。
        """
        path = self._path()
        methods = self._allowed_methods(path)
        if not methods:
            self._send_error_json(f"未知路径：{path}", 404, 404, close=True)
            return
        self._send_error_json(
            f"方法不允许：{self.command} {path}，可用 {', '.join(methods)}",
            405,
            405,
            close=True,
            headers=[("Allow", ", ".join(methods))],
        )

    def do_OPTIONS(self) -> None:
        """回 ``Allow``，不返回响应体（204），便于客户端探测可用方法。"""
        path = self._path()
        methods = self._allowed_methods(path)
        if not methods:
            self._send_error_json(f"未知路径：{path}", 404, 404, close=True)
            return
        # 同样没读请求体，故断开复用（OPTIONS 通常无 body，但不能假设调用方守规矩）。
        self.send_response(204)
        self.send_header("Allow", ", ".join(methods))
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

    def do_PUT(self) -> None:
        self._reject_method()

    def do_PATCH(self) -> None:
        self._reject_method()

    def do_DELETE(self) -> None:
        self._reject_method()

    def do_HEAD(self) -> None:
        self._reject_method()

    def do_GET(self) -> None:
        path = self._path()
        if path.endswith(_POST_ONLY_SUFFIXES):
            # 之前这里会落到下面的服务索引、以 200 回一个和请求无关的 JSON。
            self._reject_method()
            return
        if path.endswith("/health"):
            self._send_json(
                {
                    "status": "ok",
                    "version": VERSION,
                    "backend": "nano-jev",
                    **hybrid_status(),
                }
            )
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

    def do_POST(self) -> None:
        # 先读完请求体：否则在 HTTP/1.1 keep-alive 下，返回错误响应时残留在缓冲区
        # 的请求体会被当成下一个请求来解析，污染同一条连接（原 401 分支就有此 bug）。
        try:
            body = self._read_body()
        except _ContractError as exc:
            # 异常里带的是「没读过请求体」的情形，故必须断开连接。
            self._send_error_json(exc.message, exc.code, exc.status, close=True)
            return
        except (ValueError, UnicodeDecodeError) as exc:
            self._send_error_json(f"请求体不是合法 JSON：{exc}", 400, 400)
            return
        if not self._authorized():
            self._send_error_json("未授权", 401, 401)
            return

        path = self._path()
        payload: Any = None
        stream_chunks: Iterable[str] | None = None
        try:
            if path.endswith("/alpha/decisions"):
                payload = handle_decisions(body, _topics, _labeler)
            elif path.endswith("/chat/completions"):
                if wants_stream(body):
                    # 分类在 handle_chat_stream() 内同步完成，故错误仍能被下面捕获；
                    # 逐帧写出放到 try 之外，避免半途写失败后再补发 JSON 头。
                    stream_chunks = handle_chat_stream(body, _topics, _labeler)
                else:
                    payload = handle_chat(body, _topics, _labeler)
            elif path.endswith(_GET_ONLY_SUFFIXES) or "/models/" in path:
                self._reject_method()
                return
            else:
                self._send_error_json(f"未知路径：{path}", 404, 404)
                return
        except Exception:
            # 异常细节只进服务端日志：对外回泛化文案，避免把内部结构/路径泄给调用方。
            # log_message 被静音了，故这里显式打到 stderr，否则排障时一片空白。
            traceback.print_exc(file=sys.stderr)
            self._send_error_json("内部错误，详见服务端日志", 500, 500)
            return
        if stream_chunks is not None:
            self._send_event_stream(stream_chunks)
            return
        self._send_json(payload)


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    """构造（但不启动）HTTP 服务；``port=0`` 时由系统分配端口，便于测试。"""
    return ThreadingHTTPServer((host, port), _Handler)


def serve(host: str, port: int, topics: Sequence[str] | None = None, token: str | None = None) -> None:
    """启动服务并阻塞运行；``topics`` / ``token`` 为 ``None`` 时回落到对应环境变量。"""
    apply_config(topics, token)
    if not _token and not _is_loopback(host):
        print(
            f"警告：绑定在 {host} 且未设访问令牌，服务对本机以外的机器完全开放。"
            f"如需鉴权请设 {_TOKEN_ENV} 或加 --token。",
            file=sys.stderr,
        )
    httpd = make_server(host, port)
    bound_host, bound_port = httpd.server_address[0], httpd.server_address[1]
    if isinstance(bound_host, bytes):
        bound_host = bound_host.decode("utf-8", "replace")
    print(f"nano_jev 服务已启动：http://{bound_host}:{bound_port}（版本 {VERSION}）", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("收到中断，正在退出……", file=sys.stderr)
    finally:
        httpd.server_close()


def _run_label(args: argparse.Namespace) -> int:
    """``label`` 子命令：对一段文本做一次粗筛，打印标签 JSON。"""
    if args.stdin and args.text:
        # 两者同时给时旧行为是静默以 stdin 为准，调用方很容易误以为用的是位置参数。
        print("错误：位置参数与 --stdin 只能给一个。", file=sys.stderr)
        return 2
    text = sys.stdin.read() if args.stdin else args.text
    label = classify(args.id, text, topics=parse_topics(args.topics))
    print(json.dumps(label.to_raw(), ensure_ascii=False, indent=2))
    return 0


def _run_decide(args: argparse.Namespace) -> int:
    """``decide`` 子命令：读入一份 decisions 请求，输出响应 JSON。"""
    source = args.file if args.file != "-" else "标准输入"
    try:
        if args.file == "-":
            raw = sys.stdin.read()
        else:
            with open(args.file, encoding="utf-8") as handle:
                raw = handle.read()
        body = json.loads(raw)
    except OSError as exc:
        # 文件不存在/打不开：给一行可读的错误，而不是整段 traceback。
        print(f"错误：无法读取 {source}：{exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"错误：{source} 不是合法 JSON：{exc}", file=sys.stderr)
        return 1
    response = handle_decisions(body, parse_topics(args.topics))
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(prog="nano_jev", description="Nano Jev 平替：确定性粗筛服务")
    parser.add_argument("--version", action="version", version=f"nano-jev {VERSION}")
    sub = parser.add_subparsers(dest="command")

    serve_parser = sub.add_parser("serve", help="启动 HTTP 服务")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument(
        "--topics", default=None, help="逗号分隔的主题词，覆盖默认相关性领域；未给时读 NANO_JV_TOPICS"
    )
    serve_parser.add_argument("--token", default=None, help="可选的访问令牌；未给时读 NANO_JV_TOKEN")
    serve_parser.add_argument(
        "--hybrid-endpoint",
        default=None,
        help="启用级联：OpenAI 兼容 base URL（如 http://host/v1）；未给时读 "
        f"{_LLM_ENDPOINT_ENV}，都没有则只跑确定性粗筛",
    )
    serve_parser.add_argument(
        "--hybrid-model", default=None, help=f"级联模型名；未给时读 {_LLM_MODEL_ENV}"
    )
    serve_parser.add_argument(
        "--hybrid-api-key", default=None, help=f"级联模型密钥；未给时读 {_LLM_KEY_ENV}"
    )
    serve_parser.add_argument(
        "--hybrid-min-confidence",
        type=float,
        default=0.65,
        help="粗筛置信度低于该值时升级（启发式门槛，不是校准概率）",
    )
    serve_parser.add_argument(
        "--hybrid-guard",
        choices=("off", "flag", "block", "raise"),
        default="flag",
        help="输入防护：off 关闭；flag 只标记；block 命中注入时不升级；raise 直接报错",
    )
    serve_parser.add_argument(
        "--hybrid-cache-namespace",
        default="",
        help="启用缓存所需的命名空间（建议写成 <模型>/<提示词版本>）；留空不缓存",
    )
    serve_parser.add_argument(
        "--hybrid-cache-size", type=int, default=1024, help="缓存条目上限"
    )
    serve_parser.add_argument(
        "--hybrid-cache-ttl", type=float, default=None, help="缓存条目存活秒数；默认不过期"
    )

    label_parser = sub.add_parser("label", help="对一段文本做一次粗筛")
    label_parser.add_argument("text", nargs="?", default="")
    label_parser.add_argument("--id", default="0")
    label_parser.add_argument("--stdin", action="store_true", help="从标准输入读文本")
    label_parser.add_argument(
        "--topics", default=None, help="逗号分隔的主题词；未给时读 NANO_JV_TOPICS"
    )

    decide_parser = sub.add_parser("decide", help="读入 decisions 请求，输出响应 JSON")
    decide_parser.add_argument("--file", default="-", help="请求 JSON 文件；默认从标准输入读")
    decide_parser.add_argument(
        "--topics", default=None, help="逗号分隔的主题词；未给时读 NANO_JV_TOPICS"
    )
    return parser


def _configure_hybrid_from_args(args: argparse.Namespace) -> None:
    """按命令行/环境变量启用级联；未配端点则显式关闭（保持纯确定性服务）。"""
    endpoint = args.hybrid_endpoint or os.environ.get(_LLM_ENDPOINT_ENV)
    if not endpoint:
        configure_hybrid(None)
        return
    model = args.hybrid_model or os.environ.get(_LLM_MODEL_ENV) or "nano-jev"
    api_key = args.hybrid_api_key or os.environ.get(_LLM_KEY_ENV) or ""
    guard = None if args.hybrid_guard == "off" else GuardPolicy(
        injection_action=args.hybrid_guard
    )
    configure_hybrid(
        OpenAICompatJudge(endpoint=endpoint, model=model, api_key=api_key),
        policy=HybridPolicy(minimum_confidence=args.hybrid_min_confidence),
        guard=guard,
        cache_namespace=args.hybrid_cache_namespace,
        cache_size=args.hybrid_cache_size,
        cache_ttl_s=args.hybrid_cache_ttl,
        topics=parse_topics(args.topics),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        _configure_hybrid_from_args(args)
        serve(args.host, args.port, parse_topics(args.topics), args.token)
        return 0
    if args.command == "label":
        return _run_label(args)
    if args.command == "decide":
        return _run_decide(args)
    # 没给子命令：打印用法并以「用法错误」的退出码 2 结束（argparse 惯例），
    # 而不是 0——0 会让脚本把「什么都没做」误判成成功。
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
