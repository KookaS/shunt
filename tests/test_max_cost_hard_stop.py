"""HARD per-cell --max-cost stop for the serial live runner.
Regression guard: the old per-challenge-boundary check overran caps;
now every cell checks before it starts.
"""

from __future__ import annotations

from typing import Any

from benchmark.runner import run_matrix


def _inject_costs(monkeypatch, costs: list[float]) -> list[tuple[str, str, str]]:
    """Make ``_run_one_cell`` return a row with the next injected cost; count invocations.

    Returns the shared ``calls`` list (one entry per cell actually started) so a test
    can assert exactly how many cells the loop began before the cap stopped it.
    """
    calls: list[tuple[str, str, str]] = []
    seq = iter(costs)

    def fake_run_one_cell(cell: tuple[str, str, str], _ctx: Any) -> dict[str, Any]:
        calls.append(cell)
        return {"real_cost": next(seq)}

    monkeypatch.setattr(run_matrix, "_run_one_cell", fake_run_one_cell)
    return calls


def _cells(n: int) -> list[tuple[str, str, str]]:
    """One-cell-per-challenge list (per-cell == per-challenge for these)."""
    return [(f"repo__task-{i}", "m", "default") for i in range(1, n + 1)]


def _one_challenge(n_cells: int) -> list[tuple[str, str, str]]:
    """A SINGLE challenge with ``n_cells`` model cells — the case the old bug overran."""
    return [("repo__task-1", f"m{i}", "default") for i in range(n_cells)]


def test_stops_at_first_cell_that_crosses_cap(monkeypatch):
    # Cumulative: 0.10, 0.25, 0.45, ... cap 0.40. Cell 3 pushes to 0.45 >= 0.40, so the
    # pre-check before cell 4 stops the run: exactly 3 cells run, cells 4 and 5 never start.
    costs = [0.10, 0.15, 0.20, 0.10, 0.10]
    calls = _inject_costs(monkeypatch, costs)
    rows = run_matrix._run_cells_serial(_cells(5), ctx=None, max_cost=0.40)  # type: ignore[arg-type]

    assert len(calls) == 3  # cells 4 and 5 were NEVER started
    assert len(rows) == 3
    assert sum(float(r["real_cost"]) for r in rows) == 0.45  # crossed the cap, then stopped


def test_multi_cell_challenge_stops_mid_challenge(monkeypatch):
    # The exact old-bug shape: ONE challenge with 8 model cells @ $0.10, cap $0.25. The old
    # per-challenge-boundary check ran all 8 ($0.80); the per-cell check stops after 3
    # ($0.30 >= $0.25), leaving the challenge partially covered.
    calls = _inject_costs(monkeypatch, [0.10] * 8)
    rows = run_matrix._run_cells_serial(_one_challenge(8), ctx=None, max_cost=0.25)  # type: ignore[arg-type]

    assert len(calls) == 3  # stopped mid-challenge — did NOT run the whole started batch
    assert len(rows) == 3


def test_no_cap_runs_every_cell(monkeypatch):
    calls = _inject_costs(monkeypatch, [1.0] * 5)
    rows = run_matrix._run_cells_serial(_cells(5), ctx=None, max_cost=None)  # type: ignore[arg-type]

    assert len(calls) == 5 == len(rows)


def test_zero_cap_starts_no_cell(monkeypatch):
    # spent == max_cost == 0 trips the pre-check before cell 1: nothing is started, so no
    # budget is spent on a zero cap.
    calls = _inject_costs(monkeypatch, [1.0] * 5)
    rows = run_matrix._run_cells_serial(_cells(5), ctx=None, max_cost=0.0)  # type: ignore[arg-type]

    assert calls == []
    assert rows == []
