"""缓存层的行为测试：命中、过期、淘汰、线程安全与拷贝隔离。"""

from __future__ import annotations

import pathlib
import sys
import threading
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
from nano_jev.nj_cache import HybridCache, fingerprint  # noqa: E402
from nano_jev.nj_engine import Label  # noqa: E402
from nano_jev.nj_hybrid import HybridPolicy, HybridResult  # noqa: E402


def _result(sentiment: str = "positive") -> HybridResult:
    label = Label(
        "c0", True, sentiment, 0.5, 0.9, "praise", ["capability"], ["认可"], "原文"
    )
    return HybridResult(label, "llm", True, ("low",), None, 1.0, ())


def test_cache_requires_namespace():
    assert HybridCache(namespace="").enabled is False
    assert HybridCache(namespace="m/v1").enabled is True


def test_get_put_roundtrip_and_stats():
    cache = HybridCache(namespace="m/v1")
    key = cache.make_key("文本", "", 0, None, HybridPolicy())

    assert cache.get(key) is None
    cache.put(key, _result())
    hit = cache.get(key)

    assert hit is not None
    assert hit.label.sentiment == "positive"
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["entries"] == 1
    assert stats["hit_rate"] == 0.5


def test_cache_returns_copies_so_callers_cannot_mutate_state():
    cache = HybridCache(namespace="m/v1")
    key = cache.make_key("文本", "", 0, None, HybridPolicy())
    cache.put(key, _result())

    first = cache.get(key)
    assert first is not None
    first.label.aspects.append("price")
    first.label.meta["x"] = 1

    second = cache.get(key)
    assert second is not None
    assert second.label.aspects == ["capability"]
    assert "x" not in second.label.meta


def test_ttl_expires_entries():
    cache = HybridCache(namespace="m/v1", ttl_s=0.01)
    key = cache.make_key("文本", "", 0, None, HybridPolicy())
    cache.put(key, _result())
    time.sleep(0.02)

    assert cache.get(key) is None
    assert cache.stats()["entries"] == 0


def test_lru_eviction():
    cache = HybridCache(namespace="m/v1", maxsize=2)
    keys = [cache.make_key(f"文本{i}", "", 0, None, HybridPolicy()) for i in range(3)]

    cache.put(keys[0], _result())
    cache.put(keys[1], _result())
    cache.get(keys[0])  # keys[0] 变成最近使用
    cache.put(keys[2], _result())

    assert cache.get(keys[1]) is None  # 被淘汰
    assert cache.get(keys[0]) is not None
    assert cache.stats()["evictions"] == 1


def test_key_depends_on_policy_and_guard():
    cache = HybridCache(namespace="m/v1")
    base = cache.make_key("文本", "", 0, None, HybridPolicy())
    other_policy = cache.make_key(
        "文本", "", 0, None, HybridPolicy(minimum_confidence=0.5)
    )
    other_guard = cache.make_key("文本", "", 0, None, HybridPolicy(), object())

    assert base != other_policy
    assert base != other_guard


def test_fingerprint_ignores_runtime_handles():
    left = HybridCache(namespace="a")
    right = HybridCache(namespace="a")

    assert fingerprint(left) == fingerprint(right)


def test_concurrent_writes_do_not_lose_entries():
    cache = HybridCache(namespace="m/v1", maxsize=1000)

    def worker(index: int) -> None:
        key = cache.make_key(f"文本{index}", "", 0, None, HybridPolicy())
        cache.put(key, _result())
        cache.get(key)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert cache.stats()["entries"] == 50


def test_validation():
    with pytest.raises(ValueError, match="maxsize"):
        HybridCache(namespace="m", maxsize=0)
    with pytest.raises(ValueError, match="ttl_s"):
        HybridCache(namespace="m", ttl_s=0)
