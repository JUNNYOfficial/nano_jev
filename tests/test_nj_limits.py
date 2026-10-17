"""服务级限流的行为测试：并发槽、全局配额与按客户端配额。"""

from __future__ import annotations

import pathlib
import sys
import threading

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
from nano_jev.nj_limits import LimitExceededError, ServiceLimits  # noqa: E402


def test_concurrency_limit_rejects_instead_of_queueing():
    limits = ServiceLimits(max_concurrent=1)
    entered = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with limits.model_call("a"):
            entered.set()
            release.wait(5)

    thread = threading.Thread(target=holder)
    thread.start()
    assert entered.wait(5)

    with pytest.raises(LimitExceededError) as excinfo, limits.model_call("b"):
        pytest.fail("should never enter")
    assert excinfo.value.reason == "concurrency_limit"
    assert limits.snapshot()["rejected"] == 1

    release.set()
    thread.join(5)
    assert limits.snapshot()["in_flight"] == 0


def test_slot_is_released_after_an_exception():
    limits = ServiceLimits(max_concurrent=1)

    with pytest.raises(RuntimeError), limits.model_call("a"):
        raise RuntimeError("model failed")

    with limits.model_call("a"):
        pass
    assert limits.snapshot()["in_flight"] == 0


def test_global_budget_is_a_hard_cap():
    limits = ServiceLimits(global_budget=2)

    for _ in range(2):
        with limits.model_call("a"):
            pass

    with pytest.raises(LimitExceededError) as excinfo, limits.model_call("a"):
        pytest.fail("budget exhausted")
    assert excinfo.value.reason == "global_budget"
    assert limits.snapshot()["used"] == 2


def test_per_client_budget_isolated_between_clients():
    limits = ServiceLimits(per_client_budget=1)

    with limits.model_call("alice"):
        pass
    with limits.model_call("bob"):
        pass

    with pytest.raises(LimitExceededError) as excinfo, limits.model_call("alice"):
        pytest.fail("alice over budget")
    assert excinfo.value.reason == "client_budget"
    assert limits.snapshot()["clients"] == 2


def test_budget_is_consumed_even_when_the_call_fails():
    """模型已收到请求就该计费；失败退还配额会鼓励无限重试。"""
    limits = ServiceLimits(global_budget=1)

    with pytest.raises(RuntimeError), limits.model_call("a"):
        raise RuntimeError("boom")

    with pytest.raises(LimitExceededError), limits.model_call("a"):
        pass
    assert limits.snapshot()["used"] == 1


def test_unlimited_by_default():
    limits = ServiceLimits()
    for _ in range(50):
        with limits.model_call("a"):
            pass

    snapshot = limits.snapshot()
    assert snapshot["used"] == 50
    assert snapshot["max_concurrent"] == 0
    assert snapshot["global_budget"] is None


def test_limits_are_thread_safe():
    limits = ServiceLimits(global_budget=25)
    granted: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            with limits.model_call("a"):
                pass
            ok = 1
        except LimitExceededError:
            ok = 0
        with lock:
            granted.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(60)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(granted) == 25
    assert limits.snapshot()["used"] == 25


def test_validation():
    with pytest.raises(ValueError, match="max_concurrent"):
        ServiceLimits(max_concurrent=-1)
    with pytest.raises(ValueError, match="global_budget"):
        ServiceLimits(global_budget=-1)
    with pytest.raises(ValueError, match="per_client_budget"):
        ServiceLimits(per_client_budget=-1)
    with pytest.raises(ValueError, match="retry_after_s"):
        ServiceLimits(retry_after_s=0)
