"""Self-contained RAM guard: meminfo parsing, the cap arithmetic, and the hard refusal.

All offline — the procfs seam is injected, no container starts, no live host is touched.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from benchmark.runner import memory_guard
from benchmark.runner.memory_guard import InsufficientMemoryError, MemoryConfig, MemoryGuard

GIB = 1024**3
# 16384000 kB ≈ 15.6 GiB host; MemAvailable is 8388608 kB = exactly 8 GiB.
MEMINFO = (
    "MemTotal:       16384000 kB\n"
    "MemFree:         1000000 kB\n"
    "MemAvailable:    8388608 kB\n"
    "Buffers:          500000 kB\n"
)


# --- meminfo parsing -------------------------------------------------------------------


def test_reads_mem_available_in_bytes() -> None:
    assert memory_guard.read_mem_available_bytes(read_text=lambda _p: MEMINFO) == 8 * GIB


def test_reads_from_an_injected_path(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(MEMINFO)
    assert memory_guard.read_mem_available_bytes(meminfo) == 8 * GIB


def test_a_missing_mem_available_line_refuses() -> None:
    with pytest.raises(ValueError, match="MemAvailable"):
        memory_guard.read_mem_available_bytes(read_text=lambda _p: "MemTotal: 10 kB\n")


# --- the cap arithmetic ----------------------------------------------------------------


def test_requested_above_fit_is_capped() -> None:
    # headroom 8 GiB / default 1.5 GiB bound = 5.33 -> 5, below the requested 12.
    assert memory_guard.cap_workers(12, config=MemoryConfig(), available_bytes=10 * GIB) == 5


def test_requested_below_fit_is_honoured() -> None:
    assert memory_guard.cap_workers(2, config=MemoryConfig(), available_bytes=10 * GIB) == 2


def test_zero_observations_uses_the_default_container_size() -> None:
    # 6.5 GiB - 2 GiB reserve = 4.5 GiB; / 1.5 = 3.
    available = 6 * GIB + GIB // 2
    assert memory_guard.cap_workers(8, config=MemoryConfig(), available_bytes=available) == 3


def test_observations_raise_the_bound_and_lower_the_cap() -> None:
    observed = (4.0, 4.0, 4.0)  # p90 = 4 GiB, +20% margin = 4.8 GiB per container
    with_obs = memory_guard.cap_workers(
        8, config=MemoryConfig(), available_bytes=10 * GIB, observed_peak_gib=observed
    )
    without = memory_guard.cap_workers(8, config=MemoryConfig(), available_bytes=10 * GIB)
    assert with_obs == 1
    assert with_obs < without


def test_p90_is_used_not_the_mean() -> None:
    # [1, 1, 4]: linear p90 = 3.4 (mean is 2.0). p90 bound = 4.08; mean bound would be 2.4.
    observed = (1.0, 1.0, 4.0)
    # With 9 GiB headroom: p90 -> floor(9/4.08) = 2; the mean would have allowed 3.
    assert (
        memory_guard.cap_workers(
            8, config=MemoryConfig(), available_bytes=11 * GIB, observed_peak_gib=observed
        )
        == 2
    )


def test_only_the_first_sample_containers_shape_the_bound() -> None:
    # A later outlier is ignored: sampling is the first `sample_containers` (default 3), so
    # the bound stays at the opening wave's p90 of 4 GiB rather than the 100 GiB outlier.
    observed = (4.0, 4.0, 4.0, 100.0, 100.0)
    assert (
        memory_guard.cap_workers(
            8, config=MemoryConfig(), available_bytes=10 * GIB, observed_peak_gib=observed
        )
        == 1
    )


# --- the hard refusal ------------------------------------------------------------------


def test_nothing_fits_refuses_instead_of_clamping_to_zero() -> None:
    # 1 GiB available, 2 GiB reserve: negative headroom, so not even one container fits.
    with pytest.raises(InsufficientMemoryError) as excinfo:
        memory_guard.cap_workers(4, config=MemoryConfig(), available_bytes=1 * GIB)
    err = excinfo.value
    assert (err.requested, err.effective) == (4, 0)
    assert err.available_bytes == 1 * GIB
    assert err.bound_gib == pytest.approx(1.5)


def test_memory_guard_refuses_before_starting_any_container() -> None:
    guard = MemoryGuard(
        config=MemoryConfig(), requested_workers=4, available_bytes_fn=lambda: 1 * GIB
    )
    with pytest.raises(InsufficientMemoryError):
        guard.check_or_raise()
    with pytest.raises(InsufficientMemoryError):
        _ = guard.effective_workers


def test_memory_guard_caps_when_memory_allows() -> None:
    guard = MemoryGuard(
        config=MemoryConfig(), requested_workers=12, available_bytes_fn=lambda: 10 * GIB
    )
    assert guard.check_or_raise() == 5
    assert guard.effective_workers == 5


# --- thread-safe record ----------------------------------------------------------------


def test_record_is_thread_safe() -> None:
    guard = MemoryGuard(
        config=MemoryConfig(), requested_workers=8, available_bytes_fn=lambda: 64 * GIB
    )
    threads_n, per_thread = 8, 250
    barrier = threading.Barrier(threads_n)

    def worker() -> None:
        barrier.wait()
        for _ in range(per_thread):
            guard.record(1.0)

    threads = [threading.Thread(target=worker) for _ in range(threads_n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(guard.observed_peak_gib) == threads_n * per_thread
