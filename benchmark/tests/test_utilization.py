"""Unit tests for worker-utilization budgets and the saturation gate.

Pure: no network, no Docker, no subprocess. The budget math and the hard gate are exercised
directly; the recorder's /proc reads are guarded so the suite runs off-host too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark import utilization
from benchmark.utilization import (
    PER_CONTAINER_BYTES,
    PER_FIGURE_BYTES,
    UtilizationRecorder,
    assert_saturated,
    cap_by_pending,
    worker_budget,
)

_HUGE = 100 * 1024**3


def test_unknown_workload_is_rejected() -> None:
    with pytest.raises(ValueError):
        worker_budget("nonsense", 8, _HUGE)


@pytest.mark.parametrize("workload", utilization.WORKLOADS)
def test_every_workload_keeps_at_least_one_worker(workload: str) -> None:
    # A host with no cores reported and no memory still has to make progress.
    assert worker_budget(workload, 1, 0) >= 1
    assert worker_budget(workload, 0, 0) >= 1


def test_figures_reserves_one_core() -> None:
    assert worker_budget("figures", 8, _HUGE) == 7


def test_stamp_reserves_two_cores_and_caps_at_six() -> None:
    assert worker_budget("stamp", 16, _HUGE) == 6


def test_grading_reserves_two_cores_without_the_stamp_cap() -> None:
    assert worker_budget("grading", 16, _HUGE) == 14


def test_fewer_than_three_cores_still_runs_one_worker() -> None:
    assert worker_budget("stamp", 2, 0) == 1
    assert worker_budget("grading", 1, 0) == 1
    assert worker_budget("figures", 2, 0) == 1


def test_available_memory_caps_the_core_derived_budget() -> None:
    assert worker_budget("figures", 16, PER_FIGURE_BYTES * 3) == 3
    assert worker_budget("stamp", 16, PER_CONTAINER_BYTES * 4) == 4
    assert worker_budget("grading", 16, PER_CONTAINER_BYTES * 2) == 2


def test_live_collection_budgets_the_whole_host_then_memory_caps_it() -> None:
    assert worker_budget("free", 8, _HUGE) == 8
    assert worker_budget("paid", 8, _HUGE) == 8
    assert worker_budget("free", 8, PER_CONTAINER_BYTES) == 1


def test_cap_by_pending_clamps_to_runnable_lanes() -> None:
    assert cap_by_pending(8, 3) == 3
    assert cap_by_pending(2, 5) == 2
    assert cap_by_pending(8, 0) == 0
    assert cap_by_pending(0, 5) == 0


def _recorder(*, budget: int = 4) -> UtilizationRecorder:
    return UtilizationRecorder("figures", nproc=8, mem_total=_HUGE, budget=budget)


def test_recorder_tracks_peak_in_flight_from_enter_exit() -> None:
    rec = _recorder()
    rec.enter()
    rec.enter()
    rec.enter()
    assert rec.peak_workers_in_flight == 3
    rec.exit()
    assert rec.peak_workers_in_flight == 3
    rec.enter()
    rec.enter()
    assert rec.peak_workers_in_flight == 4
    for _ in range(6):
        rec.exit()


def test_recorder_context_manager_and_note_in_flight() -> None:
    rec = _recorder()
    with rec.track(), rec.track():
        assert rec.peak_workers_in_flight == 2
    assert rec.peak_workers_in_flight == 2
    rec.note_in_flight(5)
    assert rec.peak_workers_in_flight == 5


def test_recorder_counts_idle_and_samples_load() -> None:
    rec = _recorder(budget=2)
    rec.note_idle(1.5, "waiting on the last producer")
    rec.note_idle(0.5, "")
    assert rec.idle_seconds == 2.0
    rec.sample()
    payload = rec.payload()
    assert payload["idle_reason"] == "waiting on the last producer"
    if Path("/proc/loadavg").exists():
        assert payload["load_samples"]


def test_recorder_write_round_trips_the_documented_keys(tmp_path: Path) -> None:
    rec = UtilizationRecorder("stamp", nproc=8, mem_total=123, budget=3)
    with rec.track():
        pass
    path = rec.write(tmp_path / "nested" / "utilization.json")
    payload = json.loads(path.read_text())
    assert set(payload) == {
        "workload",
        "nproc",
        "mem_total_bytes",
        "budget",
        "peak_workers_in_flight",
        "load_samples",
        "idle_seconds",
        "idle_reason",
    }
    assert payload["workload"] == "stamp"
    assert payload["nproc"] == 8
    assert payload["mem_total_bytes"] == 123
    assert payload["budget"] == 3
    assert payload["peak_workers_in_flight"] == 1


def test_saturation_gate_passes_when_budget_is_met() -> None:
    rec = _recorder()
    rec.note_in_flight(4)
    assert_saturated(rec, pending=10)


def test_saturation_gate_passes_when_pending_is_below_budget() -> None:
    rec = _recorder()
    rec.note_in_flight(3)
    assert_saturated(rec, pending=3)


def test_saturation_gate_ignores_an_empty_queue() -> None:
    rec = _recorder()
    assert_saturated(rec, pending=0)


def test_saturation_gate_fails_when_budget_went_unused() -> None:
    rec = _recorder()
    rec.note_in_flight(2)
    with pytest.raises(RuntimeError, match="saturation gate failed"):
        assert_saturated(rec, pending=10)
