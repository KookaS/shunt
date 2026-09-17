"""Worker-utilization budgets and the saturation gate for the benchmark stages.

How many workers a stage may run at once is a property of the WORKLOAD, not a single
global number: a container replay pins roughly one core and is capped well below the core
count by memory, a live cell is I/O-bound (Docker + the provider's LLM) so cores are not the
binding constraint, and a figure job carries the embedder and the corpus. This module owns
those numbers once, so the stage defaults cannot drift apart.

The constants are HOST-SPECIFIC and CALIBRATABLE. They are conservative on purpose: one
worker too few costs wall-clock, one too many can OOM a container and turn a real result
into an infra failure. Raise them on a RAM-richer host.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

_GIB: Final[int] = 1024**3

# Calibrated on the 15.9 GB / 16-core benchmark host, where MemAvailable sits near 9.6 GB
# during a run. A container replay runs whole test files (sympy, matplotlib, sphinx, django)
# in the hundreds-of-MB-to-GB range; 1.5 GiB each leaves the parent, the docker daemon and
# page cache clear headroom. A figure job loads the real fastembed embedder plus the corpus,
# so it is budgeted more. Both are host-specific — edit them for a different machine.
PER_CONTAINER_BYTES: Final[int] = 3 * _GIB // 2  # 1.5 GiB
PER_FIGURE_BYTES: Final[int] = 2 * _GIB  # 2 GiB
MAX_STAMP_WORKERS: Final[int] = 6
_RESERVED_CORES: Final[int] = 2

WORKLOADS: Final[tuple[str, ...]] = ("free", "paid", "stamp", "figures", "grading")
_IO_BOUND: Final[frozenset[str]] = frozenset({"free", "paid"})


def worker_budget(workload: str, nproc: int, mem_available_bytes: int) -> int:
    """A workload's concurrent-worker ceiling from the host's cores and available memory.

    Live collection (`free`/`paid`) is I/O-bound, so the whole host is budgeted and the
    caller's runnable-lane count is the real cap — apply `cap_by_pending` before use. The
    container-bound workloads (stamp/grading) reserve two cores for the parent and the
    docker daemon and never exceed `MAX_STAMP_WORKERS`; figures reserve one core. Every
    path floors at one worker so an unknown or memory-less host still makes progress.
    """
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload {workload!r}; expected one of {list(WORKLOADS)}")
    if workload in _IO_BOUND:
        return max(1, min(nproc, mem_available_bytes // PER_CONTAINER_BYTES))
    if workload == "stamp":
        by_mem = mem_available_bytes // PER_CONTAINER_BYTES
        return max(1, min(MAX_STAMP_WORKERS, nproc - _RESERVED_CORES, by_mem))
    if workload == "figures":
        by_mem = mem_available_bytes // PER_FIGURE_BYTES
        return max(1, min(nproc - 1, by_mem))
    # grading
    by_mem = mem_available_bytes // PER_CONTAINER_BYTES
    return max(1, min(nproc - _RESERVED_CORES, by_mem))


def cap_by_pending(budget: int, pending: int) -> int:
    """The usable workers when only *pending* lanes are runnable — zero when none are."""
    return max(0, min(budget, pending))


def _read_meminfo(field: str) -> int | None:
    """One /proc/meminfo field in bytes, or None when the file or field is unreadable."""
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if not line.startswith(f"{field}:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            return int(parts[1]) * 1024
        except ValueError:
            return None
    return None


def mem_total_bytes() -> int:
    """MemTotal in bytes, or 0 when /proc/meminfo is unavailable."""
    return _read_meminfo("MemTotal") or 0


def mem_available_bytes() -> int:
    """MemAvailable in bytes; falls back to MemTotal, then 0 (budgets then floor at 1)."""
    return _read_meminfo("MemAvailable") or mem_total_bytes()


class UtilizationRecorder:
    """Records how fully a workload used its worker budget, for the saturation gate.

    Thread-safe: every worker thread calls `track()` (or `enter`/`exit`) around its unit of
    work, the orchestrator calls `sample()` to log the load, and `note_idle` attributes time
    a worker sat with nothing to do. `write` emits the record as JSON.
    """

    def __init__(
        self,
        workload: str,
        *,
        nproc: int | None = None,
        mem_total: int | None = None,
        budget: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.workload = workload
        self.nproc = (os.cpu_count() or 1) if nproc is None else nproc
        self.mem_total_bytes = mem_total_bytes() if mem_total is None else mem_total
        self.budget = (
            worker_budget(workload, self.nproc, mem_available_bytes()) if budget is None else budget
        )
        self._clock = clock
        self._lock = threading.Lock()
        self._in_flight = 0
        self._peak = 0
        self._idle_seconds = 0.0
        self._idle_reason = ""
        self._load_samples: list[float] = []

    @property
    def peak_workers_in_flight(self) -> int:
        """The most workers that were in flight at the same instant."""
        with self._lock:
            return self._peak

    @property
    def idle_seconds(self) -> float:
        """Accumulated seconds a worker sat idle."""
        with self._lock:
            return self._idle_seconds

    def enter(self) -> None:
        """Mark one worker in flight; updates the peak."""
        with self._lock:
            self._in_flight += 1
            self._peak = max(self._peak, self._in_flight)

    def exit(self) -> None:
        """Mark one worker no longer in flight."""
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    def note_in_flight(self, n: int) -> None:
        """Set the in-flight count directly (an external scheduler's view); updates the peak."""
        with self._lock:
            self._in_flight = max(0, n)
            self._peak = max(self._peak, self._in_flight)

    @contextmanager
    def track(self) -> Iterator[None]:
        """Context manager wrapping one unit of work: `with rec.track(): ...`."""
        self.enter()
        try:
            yield
        finally:
            self.exit()

    def sample(self) -> float | None:
        """Sample the 1-minute load average, appending it; None when unreadable."""
        try:
            text = Path("/proc/loadavg").read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            value = float(text.split()[0])
        except (IndexError, ValueError):
            return None
        with self._lock:
            self._load_samples.append(value)
        return value

    def note_idle(self, seconds: float, reason: str) -> None:
        """Accumulate idle time and the reason for it (the last non-empty reason wins)."""
        with self._lock:
            self._idle_seconds += max(0.0, seconds)
            if reason:
                self._idle_reason = reason

    def payload(self) -> dict[str, object]:
        """The record as a JSON-serialisable dict."""
        with self._lock:
            return {
                "workload": self.workload,
                "nproc": self.nproc,
                "mem_total_bytes": self.mem_total_bytes,
                "budget": self.budget,
                "peak_workers_in_flight": self._peak,
                "load_samples": list(self._load_samples),
                "idle_seconds": round(self._idle_seconds, 3),
                "idle_reason": self._idle_reason,
            }

    def write(self, path: Path) -> Path:
        """Write the record to *path* (parent directories created) and return it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path


def assert_saturated(recorder: UtilizationRecorder, *, pending: int) -> None:
    """HARD gate: raise when runnable work existed but the budget went unused.

    A run that reports fewer peers than `min(budget, pending)` never reached the parallelism
    it was budgeted for, so it is not a valid utilization measurement. Returns None when
    there was nothing to run, or when the budget was saturated.
    """
    if pending <= 0:
        return
    required = min(recorder.budget, pending)
    if recorder.peak_workers_in_flight < required:
        raise RuntimeError(
            f"worker saturation gate failed for {recorder.workload!r}: peak "
            f"{recorder.peak_workers_in_flight} in flight < required {required} "
            f"(budget {recorder.budget}, pending {pending})"
        )
