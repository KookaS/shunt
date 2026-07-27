"""Run-level abort: an unusable API (dead key / no balance) or N consecutive failures abort
the WHOLE run instead of marching through hundreds of cells writing garbage.

All stubbed (no live/paid/Docker): ``_run_one_cell`` is monkeypatched to yield sentinels.
"""

from __future__ import annotations

from typing import Any, Final

import pytest

from benchmark.runner import run_matrix

SF: Final = run_matrix._START_FAILURE
_OK: Final[dict[str, Any]] = {"real_cost": 0.0}


def _unusable(reason: str = "no balance") -> run_matrix._ApiUnusable:
    return run_matrix._ApiUnusable(reason)


def _inject(monkeypatch, results: list[Any]) -> list[tuple[str, str, str]]:
    """Make ``_run_one_cell`` yield the next injected result; return the per-cell call log."""
    calls: list[tuple[str, str, str]] = []
    seq = iter(results)

    def fake_run_one_cell(cell: tuple[str, str, str], _ctx: Any) -> Any:
        calls.append(cell)
        return next(seq)

    monkeypatch.setattr(run_matrix, "_run_one_cell", fake_run_one_cell)
    return calls


def _cells(n: int) -> list[tuple[str, str, str]]:
    """One-cell-per-challenge list (per-cell == per-challenge here)."""
    return [(f"repo__task-{i}", "m", "default") for i in range(1, n + 1)]


# --- immediate abort on the first unambiguous API-unusable cell -----------------------


def test_api_unusable_aborts_immediately_serial(monkeypatch) -> None:
    # First cell is API-unusable → abort at once; cells 2..10 never start (retrying a dead key
    # is pointless). The abort message names the cause.
    calls = _inject(monkeypatch, [_unusable("insufficient balance")] + [_OK] * 9)
    with pytest.raises(run_matrix.RunAbortError, match="insufficient balance"):
        run_matrix._run_cells_serial(_cells(10), ctx=None, max_cost=None)  # type: ignore[arg-type]
    assert len(calls) == 1  # aborted on the very first unusable cell


def test_api_unusable_is_never_recorded_as_a_row(monkeypatch) -> None:
    # A successful cell, THEN an unusable one: the good row is kept (checkpointed), the unusable
    # cell aborts — it is never turned into a fake pass=False row.
    calls = _inject(monkeypatch, [{"real_cost": 1.0}, _unusable(), _OK])
    with pytest.raises(run_matrix.RunAbortError):
        run_matrix._run_cells_serial(_cells(3), ctx=None, max_cost=None)  # type: ignore[arg-type]
    assert len(calls) == 2  # ran the good cell, hit the unusable one, aborted


def test_api_unusable_aborts_parallel(monkeypatch) -> None:
    calls = _inject(monkeypatch, [_unusable()] * 10)
    with pytest.raises(run_matrix.RunAbortError):
        run_matrix._run_cells_parallel(
            _cells(10),
            ctx=None,  # type: ignore[arg-type]
            workers=2,
            max_cost=None,
        )
    assert len(calls) >= 1


# --- consecutive-any-failure catch-all ------------------------------------------------


def test_consecutive_any_failures_abort(monkeypatch) -> None:
    # 5 consecutive None-skips (rate-limit / dep error) with a catch-all cap of 5 → abort at the
    # 5th; the dedicated container-start counter is untouched (these aren't start failures).
    calls = _inject(monkeypatch, [None] * 10)
    with pytest.raises(run_matrix.RunAbortError, match="consecutive cell failures"):
        run_matrix._run_cells_serial(
            _cells(10),
            ctx=None,  # type: ignore[arg-type]
            max_cost=None,
            max_start_failures=None,
            max_consecutive_failures=5,
        )
    assert len(calls) == 5


def test_success_resets_consecutive_counter(monkeypatch) -> None:
    # Failures interspersed with a success never reach 5 CONSECUTIVE → no abort.
    results = [None] * 4 + [_OK] + [None] * 4 + [_OK] + [None] * 4
    calls = _inject(monkeypatch, results)
    rows = run_matrix._run_cells_serial(
        _cells(len(results)),
        ctx=None,  # type: ignore[arg-type]
        max_cost=None,
        max_consecutive_failures=5,
    )
    assert len(calls) == len(results)  # ran every cell, never aborted
    assert len(rows) == 2  # the two successful cells


def test_start_failures_count_toward_consecutive_catch_all(monkeypatch) -> None:
    # Container-start failures also count toward the catch-all: with no start-failure cap but a
    # consecutive cap of 3, three start-failures in a row still abort the run.
    calls = _inject(monkeypatch, [SF] * 6)
    with pytest.raises(run_matrix.RunAbortError, match="consecutive cell failures"):
        run_matrix._run_cells_serial(
            _cells(6),
            ctx=None,  # type: ignore[arg-type]
            max_cost=None,
            max_start_failures=None,
            max_consecutive_failures=3,
        )
    assert len(calls) == 3


def test_no_consecutive_cap_never_aborts_on_skips(monkeypatch) -> None:
    # Default (None) restores pure skip-and-continue: 10 None-skips, no abort.
    calls = _inject(monkeypatch, [None] * 10)
    rows = run_matrix._run_cells_serial(_cells(10), ctx=None, max_cost=None)  # type: ignore[arg-type]
    assert len(calls) == 10 and rows == []


def test_consecutive_abort_parallel(monkeypatch) -> None:
    calls = _inject(monkeypatch, [None] * 10)
    with pytest.raises(run_matrix.RunAbortError, match="consecutive cell failures"):
        run_matrix._run_cells_parallel(
            _cells(10),
            ctx=None,  # type: ignore[arg-type]
            workers=2,
            max_cost=None,
            max_start_failures=None,
            max_consecutive_failures=3,
        )
    assert len(calls) >= 3
