"""服务级限流：并发上限、全局升级配额与按客户端配额。

`nano_jev` 的 HTTP 服务默认是"每个请求一个线程"。级联模式下每个升级样本都会
阻塞等一次模型调用；没有上限时，模型一慢，线程数就会随请求数线性增长，最后连
健康检查都排不上。本模块把三条限制集中在 `ServiceLimits`：

* **并发**：同时在飞的模型调用数，超出立即拒绝（不排队——排队只会让尾延迟更长）；
* **全局配额**：整台机器上一共允许多少次升级；
* **按客户端配额**：防止单个调用方把配额吃光。

三条都是**显式拒绝**（`LimitExceededError`），由服务层映射成 429 + `Retry-After`；
不做静默降级，否则调用方会以为结果经过模型复核，其实没有。

线程安全：所有计数在一把锁内完成；并发槽用信号量，`finally` 释放。
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


class LimitExceededError(RuntimeError):
    """触发了并发或配额限制；``retry_after_s`` 供服务层写 ``Retry-After``。"""

    def __init__(self, reason: str, retry_after_s: float = 1.0) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after_s = retry_after_s


@dataclass
class ServiceLimits:
    """服务级限制。``0`` / ``None`` 表示不限制该维度。

    ``max_concurrent`` 只约束**真正调用模型**的那段时间；粗筛不占槽位。
    """

    max_concurrent: int = 0
    global_budget: int | None = None
    per_client_budget: int | None = None
    retry_after_s: float = 1.0
    in_flight: int = field(default=0, init=False, repr=False)
    used: int = field(default=0, init=False, repr=False)
    rejected: int = field(default=0, init=False, repr=False)
    _sem: threading.BoundedSemaphore | None = field(
        default=None, init=False, repr=False
    )
    _clients: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_concurrent < 0:
            raise ValueError("max_concurrent must be >= 0")
        if self.global_budget is not None and self.global_budget < 0:
            raise ValueError("global_budget must be >= 0 or None")
        if self.per_client_budget is not None and self.per_client_budget < 0:
            raise ValueError("per_client_budget must be >= 0 or None")
        if self.retry_after_s <= 0:
            raise ValueError("retry_after_s must be > 0")
        if self.max_concurrent:
            self._sem = threading.BoundedSemaphore(self.max_concurrent)

    @contextmanager
    def model_call(self, client: str = "") -> Iterator[None]:
        """占一个并发槽并按客户端记账；超限抛 :class:`LimitExceededError`。

        配额在**尝试调用**时扣除，失败也不退还：模型已经收到请求，成本已经产生。
        """
        if self._sem is not None and not self._sem.acquire(blocking=False):
            with self._lock:
                self.rejected += 1
            raise LimitExceededError("concurrency_limit", self.retry_after_s)

        granted = False
        try:
            with self._lock:
                if self.global_budget is not None and self.used >= self.global_budget:
                    self.rejected += 1
                    raise LimitExceededError("global_budget", self.retry_after_s)
                client_used = self._clients.get(client, 0)
                if (
                    self.per_client_budget is not None
                    and client_used >= self.per_client_budget
                ):
                    self.rejected += 1
                    raise LimitExceededError("client_budget", self.retry_after_s)
                self.used += 1
                self._clients[client] = client_used + 1
                self.in_flight += 1
                granted = True
            yield
        finally:
            if granted:
                with self._lock:
                    self.in_flight -= 1
            if self._sem is not None:
                self._sem.release()

    def snapshot(self) -> dict[str, object]:
        """只读状态，供 ``/health`` 输出。"""
        with self._lock:
            return {
                "max_concurrent": self.max_concurrent,
                "in_flight": self.in_flight,
                "used": self.used,
                "global_budget": self.global_budget,
                "per_client_budget": self.per_client_budget,
                "rejected": self.rejected,
                "clients": len(self._clients),
            }
