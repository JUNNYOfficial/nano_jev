"""级联结果的缓存：同样的文本 + 同样的策略，不重复付一次模型调用。

**为什么必须显式给命名空间**：缓存键只能看见文本与策略，看不见"背后是哪个模型、
哪版提示词"。若默认共享，换模型或改提示词后会继续读到旧结论，且没有任何迹象。
故 ``namespace`` 由调用方提供，约定写成 ``"<模型 id>/<提示词版本>"``；留空则
整个缓存不生效（只影响升级后的结果，粗筛本身是确定性的，缓存无收益）。

线程安全：内部用一把锁保护 LRU 与计数；缓存命中返回**深拷贝**，调用方改标签
不会污染缓存里的副本。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查时需要
    from .nj_hybrid import HybridResult

#: 缓存键：命名空间 + 文本 + 主题 + 点赞数 + 主题词 + 策略指纹。
CacheKey = tuple[Any, ...]


def copy_result(result: HybridResult) -> HybridResult:
    """复制结果及其可变容器，避免缓存与调用方共享同一份标签。"""
    return replace(
        result,
        label=replace(
            result.label,
            aspects=list(result.label.aspects),
            emotion=list(result.label.emotion),
            meta=dict(result.label.meta),
        ),
        coarse_label=(
            None
            if result.coarse_label is None
            else replace(
                result.coarse_label,
                aspects=list(result.coarse_label.aspects),
                emotion=list(result.coarse_label.emotion),
                meta=dict(result.coarse_label.meta),
            )
        ),
        escalation_reasons=tuple(result.escalation_reasons),
        normalizations=tuple(result.normalizations),
    )


def fingerprint(value: object) -> str:
    """把 dataclass 策略压成稳定字符串，作为缓存键的一部分。

    只读取**构造期字段**（``init=True``）。``init=False`` 的字段是运行时状态
    （锁、计数、上一次决策），若参与指纹，缓存键会随每次调用变化、永远无法命中——
    实测 ``BrainInspiredPolicy.last`` 就会造成这种漂移。
    """
    from dataclasses import fields as dataclass_fields
    from dataclasses import is_dataclass

    if not is_dataclass(value) or isinstance(value, type):
        return repr(value)
    parts: list[str] = []
    for item in dataclass_fields(value):
        if not item.init:
            continue
        parts.append(f"{item.name}={getattr(value, item.name)!r}")
    return ";".join(parts)


@dataclass
class HybridCache:
    """带 TTL 的线程安全 LRU 缓存。

    ``namespace`` 为空字符串时 :meth:`enabled` 为 ``False``，调用方应跳过缓存。
    """

    namespace: str = ""
    maxsize: int = 1024
    ttl_s: float | None = None
    hits: int = field(default=0, init=False, repr=False)
    misses: int = field(default=0, init=False, repr=False)
    evictions: int = field(default=0, init=False, repr=False)
    coalesced: int = field(default=0, init=False, repr=False)
    _entries: OrderedDict[CacheKey, tuple[float, HybridResult]] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _flights: dict[CacheKey, threading.Event] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        if self.ttl_s is not None and self.ttl_s <= 0:
            raise ValueError("ttl_s must be > 0 or None")

    @property
    def enabled(self) -> bool:
        """是否真的参与缓存（命名空间为空即关闭）。"""
        return bool(self.namespace)

    def make_key(
        self,
        content: str,
        topic_title: str,
        like_count: object,
        topics: Sequence[str] | None,
        policy: object,
        guard: object | None = None,
    ) -> CacheKey:
        """拼出缓存键；``like_count`` 也参与，因为它会写进 ``meta``。"""
        return (
            self.namespace,
            content,
            topic_title,
            str(like_count),
            tuple(topics) if topics else None,
            fingerprint(policy),
            fingerprint(guard) if guard is not None else "",
        )

    def get(self, key: CacheKey) -> HybridResult | None:
        """取一条缓存；过期或不存在返回 ``None``。"""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, value = entry
            if self.ttl_s is not None and time.monotonic() - stored_at > self.ttl_s:
                del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return copy_result(value)

    def put(self, key: CacheKey, value: HybridResult) -> None:
        """写入一条缓存；超出 ``maxsize`` 时淘汰最久未用的条目。"""
        with self._lock:
            self._entries[key] = (time.monotonic(), copy_result(value))
            self._entries.move_to_end(key)
            while len(self._entries) > self.maxsize:
                self._entries.popitem(last=False)
                self.evictions += 1

    def begin(
        self, key: CacheKey, *, flight_timeout: float = 120.0
    ) -> tuple[HybridResult | None, threading.Event | None]:
        """查缓存并合并并发请求；返回 ``(命中结果, 计算完成事件)``。

        命中时返回 ``(结果, None)``。未命中时：
        * 第一个调用者成为 **leader**，拿到 ``(None, event)``，计算完必须调用
          :meth:`finish`（``finally`` 里也要调用，否则跟随者会一直等）；
        * 其余调用者等待 leader 发布结果，命中则直接返回；leader 失败时它们会
          重新竞争成为 leader。

        这样 8 个相同的并发请求只付一次模型调用，而不是 8 次。
        """
        for _attempt in range(3):
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None:
                    stored_at, value = entry
                    fresh = (
                        self.ttl_s is None
                        or time.monotonic() - stored_at <= self.ttl_s
                    )
                    if fresh:
                        self._entries.move_to_end(key)
                        self.hits += 1
                        return copy_result(value), None
                    del self._entries[key]

                event = self._flights.get(key)
                if event is None:
                    event = threading.Event()
                    self._flights[key] = event
                    self.misses += 1
                    return None, event
                self.coalesced += 1

            event.wait(flight_timeout)
            # 醒来后回到循环顶部重新查缓存；leader 已写入就直接命中。
        # 连续多轮都没拿到结果（leader 反复失败/超时）：退化为自行计算，不再等待。
        self.misses += 1
        return None, None

    def finish(self, key: CacheKey, event: threading.Event) -> None:
        """结束一次 leader 计算：移除航班标记并唤醒跟随者。"""
        with self._lock:
            if self._flights.get(key) is event:
                del self._flights[key]
        event.set()

    def clear(self) -> None:
        """清空内容（计数保留，便于观察长期命中率）。"""
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, Any]:
        """缓存计数快照，供健康检查/日志使用。"""
        with self._lock:
            total = self.hits + self.misses
            return {
                "enabled": self.enabled,
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "coalesced": self.coalesced,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }
