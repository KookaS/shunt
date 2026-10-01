"""Disk-aware bounded prefetcher: pull in order within the free-space budget."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Final

from benchmark.runner.capacity.types import GIB, ImageSize, ResourceConfig

PullFn = Callable[[str], None]
FreeFn = Callable[[], int]
OnPulledFn = Callable[[str], None]

_DEFAULT_ESTIMATE_GIB: Final[float] = 3.0


class Prefetcher:
    """Pull a ref set in order with a bounded pool and a strict disk budget.

    Off unless ``config.prefetch_enabled``. The scheduler only starts a pull when
    the free space left after the reserve can absorb the image's estimate, so
    outstanding pulls never eat the reserve.
    """

    def __init__(
        self,
        refs: Iterable[str],
        *,
        config: ResourceConfig,
        sizes: Iterable[ImageSize],
        pull_fn: PullFn,
        free_fn: FreeFn,
        on_pulled: OnPulledFn | None = None,
    ) -> None:
        self._config = config
        self._refs = list(refs)
        self._sizes = {size.ref: size for size in sizes}
        self._pull_fn = pull_fn
        self._free_fn = free_fn
        self._on_pulled = on_pulled
        self._reserve = int(config.disk_reserve_gib * GIB)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: deque[str] = deque()
        self._inflight = 0
        self._reserved = 0
        self._ready: set[str] = set()
        self._failed: set[str] = set()
        self._closed = False
        self._pool: ThreadPoolExecutor | None = None
        self._scheduler: threading.Thread | None = None

    def start(self) -> None:
        """Begin scheduling pulls; a no-op unless prefetch is enabled."""
        if not self._config.prefetch_enabled:
            return
        with self._cond:
            for ref in self._refs:
                size = self._sizes.get(ref)
                if size is not None and size.cached:
                    self._ready.add(ref)
                else:
                    self._pending.append(ref)
        self._pool = ThreadPoolExecutor(max_workers=max(1, self._config.prefetch_workers))
        self._scheduler = threading.Thread(
            target=self._schedule, name="capacity-prefetch", daemon=True
        )
        self._scheduler.start()

    def wait_ready(self, ref: str, timeout: float | None = None) -> bool:
        """Block until ``ref`` has been pulled (or attempted); True when it is local."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            if self._pool is None and self._scheduler is None:
                return ref in self._ready
            while ref not in self._ready and ref not in self._failed:
                if self._closed:
                    return ref in self._ready
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return ref in self._ready
                self._cond.wait(remaining)
            return ref in self._ready

    def close(self) -> None:
        """Stop scheduling and join the scheduler and worker pool."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        if self._scheduler is not None:
            self._scheduler.join()
        if self._pool is not None:
            self._pool.shutdown(wait=True)

    def _estimate(self, ref: str) -> int:
        """Bytes to reserve for a ref; the configured fallback when unknown."""
        size = self._sizes.get(ref)
        if size is None:
            return int(self._config.assumed_image_gib * GIB)
        return size.uncompressed_estimate_bytes

    def _fits(self, estimate: int) -> bool:
        """True when the estimate fits in free space after the reserve."""
        return self._free_fn() - self._reserve - (self._reserved + estimate) >= 0

    def _next_ref(self) -> str | None:
        """Block until the next ref may be pulled, or return None when done/closed."""
        while True:
            with self._cond:
                if self._closed:
                    return None
                if not self._pending:
                    if self._inflight == 0:
                        return None
                    self._cond.wait()
                    continue
                if self._inflight >= self._config.prefetch_workers:
                    self._cond.wait()
                    continue
                ref = self._pending[0]
                estimate = self._estimate(ref)
                if not self._fits(estimate):
                    self._cond.wait()
                    continue
                self._pending.popleft()
                self._inflight += 1
                self._reserved += estimate
                return ref

    def _schedule(self) -> None:
        """Submit pulls in order until the queue drains or is closed."""
        while True:
            ref = self._next_ref()
            if ref is None:
                return
            assert self._pool is not None
            self._pool.submit(self._run, ref)

    def _run(self, ref: str) -> None:
        """Pull one ref, then release its reservation and record the outcome."""
        estimate = self._estimate(ref)
        ok = False
        try:
            self._pull_fn(ref)
            ok = True
        except Exception:  # noqa: BLE001 - one failed prefetch must not kill the pool
            ok = False
        with self._cond:
            self._inflight -= 1
            self._reserved = max(0, self._reserved - estimate)
            if ok:
                self._ready.add(ref)
            else:
                self._failed.add(ref)
            self._cond.notify_all()
        if ok and self._on_pulled is not None:
            self._on_pulled(ref)
