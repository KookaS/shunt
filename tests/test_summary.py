"""BUG 1 (summary variant): a strategy selecting a (task, model) pair with no row in
the results matrix must treat that task as UNSCORABLE — excluded from the metrics and
counted as ``n_unscorable`` — not recorded as a real fail@$0."""

from __future__ import annotations

from benchmark.routing.strategies.fixed import AlwaysFrontier
from benchmark.routing.summary import compute_strategy_rows


def _matrix() -> dict:
    return {
        "models": {
            "cheap-model": {"input_price": 0.10, "output_price": 0.10},
            "frontier-model": {"input_price": 5.00, "output_price": 5.00},
        },
        "tasks": {"t1": {}, "t2": {}},
        "results": {
            "t1": {
                "cheap-model": {"pass": True, "cost": 1.0},
                "frontier-model": {"pass": True, "cost": 10.0},
            },
            # frontier-model NOT measured here -> AlwaysFrontier lands on a gap.
            "t2": {"cheap-model": {"pass": True, "cost": 1.0}},
        },
    }


def _matrix_with_empty_middle() -> dict:
    return {
        "models": {
            "cheap-model": {"input_price": 0.10, "output_price": 0.10},
            "frontier-model": {"input_price": 5.00, "output_price": 5.00},
        },
        "tasks": {"a": {}, "b": {}, "c": {}},
        "results": {
            "a": {
                "cheap-model": {"pass": True, "cost": 1.0},
                "frontier-model": {"pass": True, "cost": 10.0},
            },
            # Entirely empty -> the ORACLE itself returns unscorable (non-empty
            # oracle_unscorable), AND it sits in the MIDDLE so a re-index that
            # dropped alignment would mispair the survivors.
            "b": {},
            "c": {
                "cheap-model": {"pass": True, "cost": 2.0},
                "frontier-model": {"pass": True, "cost": 20.0},
            },
        },
    }


def test_oracle_unscorable_drops_task_in_lockstep():
    matrix = _matrix_with_empty_middle()
    rows = compute_strategy_rows(
        matrix, ["a", "b", "c"], [AlwaysFrontier()], gamma=0.1, bootstrap=50, seed=1
    )
    row = next(r for r in rows if r["strategy"] == "Always-Frontier")
    # 'b' excluded via the oracle_unscorable | strat_unscorable union; survivors
    # 'a' and 'c' scored on the aligned remainder (frontier costs 10 + 20).
    assert row["n_unscorable"] == 1
    assert row["n_tasks"] == 2
    assert row["TotalCost"] == 30.0


def test_unscorable_task_excluded_from_metrics():
    matrix = _matrix()
    rows = compute_strategy_rows(
        matrix, ["t1", "t2"], [AlwaysFrontier()], gamma=0.1, bootstrap=50, seed=1
    )
    row = next(r for r in rows if r["strategy"] == "Always-Frontier")
    # t2 has no frontier cell -> unscorable; only t1 counts.
    assert row["n_unscorable"] == 1
    assert row["n_tasks"] == 1
    # Cost reflects only the measured t1 cell ($10), NOT a phantom $0 for t2.
    assert row["TotalCost"] == 10.0
