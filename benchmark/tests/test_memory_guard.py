"""Self-contained RAM guard: meminfo parsing, the cap arithmetic, and the hard refusal.

All offline — the procfs seam is injected, no container starts, no live host is touched.
"""

from __future__ import annotations

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


def test_default_container_size_is_used() -> None:
    # 6.5 GiB - 2 GiB reserve = 4.5 GiB; / 1.5 = 3.
    available = 6 * GIB + GIB // 2
    assert memory_guard.cap_workers(8, config=MemoryConfig(), available_bytes=available) == 3


def test_a_larger_container_size_lowers_the_cap() -> None:
    # 8 GiB headroom / 4 GiB per container = 2, below the default bound's 5.
    assert (
        memory_guard.cap_workers(
            8, config=MemoryConfig(container_gib=4.0), available_bytes=10 * GIB
        )
        == 2
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


# --- strict config parsing -------------------------------------------------------------


def test_from_mapping_reads_the_memory_keys() -> None:
    cfg = MemoryConfig.from_mapping({"container_memory_gib": 3.0, "memory_reserve_gib": 1.0})
    assert (cfg.container_gib, cfg.reserve_gib) == (3.0, 1.0)


def test_from_mapping_ignores_none_and_empty() -> None:
    base = MemoryConfig(container_gib=2.5)
    assert MemoryConfig.from_mapping({}, base=base) == base
    assert MemoryConfig.from_mapping(None, base=base) == base


@pytest.mark.parametrize("bad", ["lots", [1], True])
def test_from_mapping_rejects_non_numeric(bad: object) -> None:
    with pytest.raises(ValueError, match="container_memory_gib"):
        MemoryConfig.from_mapping({"container_memory_gib": bad})


def test_from_mapping_rejects_negative() -> None:
    with pytest.raises(ValueError, match="memory_reserve_gib"):
        MemoryConfig.from_mapping({"memory_reserve_gib": -1})
