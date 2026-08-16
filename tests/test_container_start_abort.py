"""Consecutive container-start failures abort the live runner instead of skip-storming.

A missing/throttled image (docker exit 125 / 429) once made every cell skip and hammer the
registry; now ``--max-start-failures`` consecutive start-failures abort fast.
"""

from __future__ import annotations

from typing import Any, Final

import pytest

from benchmark.runner import run_matrix

SF: Final = run_matrix._START_FAILURE
_OK: Final[dict[str, Any]] = {"real_cost": 0.0}


def _inject_results(monkeypatch, results: list[Any]) -> list[tuple[str, str, str]]:
    """Make ``_run_one_cell`` yield the next injected result; return the per-cell call log."""
    calls: list[tuple[str, str, str]] = []
    seq = iter(results)

    def fake_run_one_cell(cell: tuple[str, str, str], _ctx: Any) -> Any:
        calls.append(cell)
        return next(seq)

    monkeypatch.setattr(run_matrix, "_run_one_cell", fake_run_one_cell)
    return calls


def _cells(n: int) -> list[tuple[str, str, str]]:
    """One-cell-per-challenge list (per-cell == per-challenge for these)."""
    return [(f"repo__task-{i}", "m", "default") for i in range(1, n + 1)]


# --- classification of the raised exception ------------------------------------------


@pytest.mark.parametrize(
    "msg",
    [
        "Command '[...]' returned non-zero exit status 125.",
        "toomanyrequests: 429 Too Many Requests",
        "error Starting container for instance foo",
        "harness produced no valid report for x/y (report=None, rc=125); leaving cell MISSING",
    ],
)
def test_container_start_failure_classified(msg: str) -> None:
    assert run_matrix._is_container_start_failure(Exception(msg))


@pytest.mark.parametrize(
    "msg",
    ["No module named 'swebench'", "agent produced an empty patch", "spec hash mismatch"],
)
def test_non_start_failure_not_classified(msg: str) -> None:
    assert not run_matrix._is_container_start_failure(Exception(msg))


# --- serial abort logic --------------------------------------------------------------


def test_consecutive_start_failures_abort(monkeypatch) -> None:
    # 5 consecutive start-failures (cap 5) abort at the 5th; cells 6-10 never start.
    calls = _inject_results(monkeypatch, [SF] * 10)
    with pytest.raises(run_matrix.ContainerStartAbortError):
        run_matrix._run_cells_serial(_cells(10), ctx=None, max_cost=None, max_start_failures=5)  # type: ignore[arg-type]
    assert len(calls) == 5


def test_success_resets_start_failure_counter(monkeypatch) -> None:
    # Start-failures interspersed with successes never reach 5 CONSECUTIVE → no abort.
    results = [SF] * 4 + [_OK] + [SF] * 4 + [_OK] + [SF] * 4
    calls = _inject_results(monkeypatch, results)
    rows = run_matrix._run_cells_serial(
        _cells(len(results)),
        ctx=None,
        max_cost=None,
        max_start_failures=5,  # type: ignore[arg-type]
    )
    assert len(calls) == len(results)  # ran every cell, never aborted
    assert len(rows) == 2  # the two successful cells


def test_non_start_error_skips_without_counting(monkeypatch) -> None:
    # Regular per-cell skips (None) never increment the counter — even under a tight cap.
    calls = _inject_results(monkeypatch, [None] * 10)
    rows = run_matrix._run_cells_serial(_cells(10), ctx=None, max_cost=None, max_start_failures=2)  # type: ignore[arg-type]
    assert len(calls) == 10  # every cell tried; nothing aborted
    assert rows == []


def test_no_threshold_never_aborts(monkeypatch) -> None:
    # max_start_failures=None (the library default) restores the pure skip-and-continue path.
    calls = _inject_results(monkeypatch, [SF] * 6)
    rows = run_matrix._run_cells_serial(_cells(6), ctx=None, max_cost=None, max_start_failures=None)  # type: ignore[arg-type]
    assert len(calls) == 6
    assert rows == []


# --- parallel path (per-batch accounting) --------------------------------------------


def test_parallel_consecutive_start_failures_abort(monkeypatch) -> None:
    # Each cell is its own challenge/batch, so per-batch accounting == per-cell here.
    calls = _inject_results(monkeypatch, [SF] * 10)
    with pytest.raises(run_matrix.ContainerStartAbortError):
        run_matrix._run_cells_parallel(
            _cells(10),
            ctx=None,
            workers=2,
            max_cost=None,
            max_start_failures=3,  # type: ignore[arg-type]
        )
    assert len(calls) >= 3  # aborted once 3 consecutive start-failures accrued
