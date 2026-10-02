"""Offline tests for the disk-aware bounded prefetcher."""

from __future__ import annotations

import threading
import time

from benchmark.runner.capacity.prefetch import Prefetcher
from benchmark.runner.capacity.types import GIB, ImageSize, ResourceConfig


def _sizes(refs: list[str], *, bytes_each: int = GIB, cached: set[str] | None = None):
    cached = cached or set()
    return [
        ImageSize(ref, ref in cached, None, bytes_each, "local" if ref in cached else "registry")
        for ref in refs
    ]


class _Recorder:
    """Thread-safe pull recorder that also tracks peak concurrency."""

    def __init__(self, *, delay: float = 0.02) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.peak = 0
        self.pulled: list[str] = []
        self._delay = delay

    def __call__(self, ref: str) -> bool:
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
        time.sleep(self._delay)
        with self._lock:
            self.pulled.append(ref)
            self._active -= 1
        return True


def _run(refs: list[str], *, config: ResourceConfig, free: int, sizes, pull) -> Prefetcher:
    prefetch = Prefetcher(refs, config=config, sizes=sizes, pull_fn=pull, free_fn=lambda: free)
    prefetch.start()
    for ref in refs:
        prefetch.wait_ready(ref, timeout=5.0)
    prefetch.close()
    return prefetch


def test_prefetch_respects_worker_bound() -> None:
    refs = ["r0", "r1", "r2", "r3"]
    config = ResourceConfig(prefetch_enabled=True, prefetch_workers=2, disk_reserve_gib=0.0)
    pull = _Recorder()
    _run(refs, config=config, free=100 * GIB, sizes=_sizes(refs), pull=pull)
    assert pull.peak == 2
    assert sorted(pull.pulled) == sorted(refs)


def test_prefetch_reserve_limits_outstanding_pulls() -> None:
    refs = ["r0", "r1", "r2"]
    # 3 GiB free - 2 GiB reserve leaves 1 GiB, so only one 1 GiB pull may be outstanding.
    config = ResourceConfig(
        prefetch_enabled=True, prefetch_workers=3, disk_reserve_gib=2.0, disk_safety_factor=1.0
    )
    pull = _Recorder()
    _run(refs, config=config, free=3 * GIB, sizes=_sizes(refs), pull=pull)
    assert pull.peak == 1
    assert sorted(pull.pulled) == sorted(refs)


def test_prefetch_skips_cached_images() -> None:
    refs = ["cached", "fresh"]
    config = ResourceConfig(prefetch_enabled=True, prefetch_workers=2, disk_reserve_gib=0.0)
    pull = _Recorder()
    prefetch = Prefetcher(
        refs,
        config=config,
        sizes=_sizes(refs, cached={"cached"}),
        pull_fn=pull,
        free_fn=lambda: 100 * GIB,
    )
    prefetch.start()
    assert prefetch.wait_ready("cached", timeout=1.0) is True
    assert prefetch.wait_ready("fresh", timeout=5.0) is True
    prefetch.close()
    assert pull.pulled == ["fresh"]


def test_prefetch_off_by_default() -> None:
    refs = ["r0"]
    config = ResourceConfig(prefetch_enabled=False)
    pull = _Recorder()
    prefetch = Prefetcher(
        refs, config=config, sizes=_sizes(refs), pull_fn=pull, free_fn=lambda: 100 * GIB
    )
    prefetch.start()
    prefetch.close()
    assert pull.pulled == []
    assert prefetch.wait_ready("r0", timeout=1.0) is False


def test_falsy_pull_is_recorded_failed() -> None:
    refs = ["bad"]
    config = ResourceConfig(prefetch_enabled=True, prefetch_workers=1, disk_reserve_gib=0.0)
    prefetch = Prefetcher(
        refs,
        config=config,
        sizes=_sizes(refs),
        pull_fn=lambda ref: False,
        free_fn=lambda: 100 * GIB,
    )
    prefetch.start()
    assert prefetch.wait_ready("bad", timeout=5.0) is False
    prefetch.close()


def test_no_space_wait_is_bounded_by_the_config_timeout() -> None:
    refs = ["r0"]
    # 0 free - 10 GiB reserve can never fit the image, so the scheduler only ever waits.
    config = ResourceConfig(
        prefetch_enabled=True,
        prefetch_workers=1,
        disk_reserve_gib=10.0,
        prefetch_wait_timeout_s=0.05,
    )
    calls: list[str] = []

    def pull(ref: str) -> bool:
        calls.append(ref)
        return True

    prefetch = Prefetcher(refs, config=config, sizes=_sizes(refs), pull_fn=pull, free_fn=lambda: 0)
    prefetch.start()
    started = time.monotonic()
    assert prefetch.wait_ready("r0") is False  # None -> the config bound, never forever
    assert time.monotonic() - started < 5.0
    prefetch.close()
    assert calls == []  # never fit, so never pulled
