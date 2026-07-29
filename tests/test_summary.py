"""BUG 1 (summary variant): a strategy selecting a (task, model) pair with no row in
the results matrix must treat that task as UNSCORABLE — excluded from the metrics and
counted as ``n_unscorable`` — not recorded as a real fail@$0."""

from __future__ import annotations

import pytest

from benchmark import config
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


def _matrix_with_never_executed_cell() -> dict:
    return {
        "models": {
            "cheap-model": {"input_price": 0.10, "output_price": 0.10},
            "frontier-model": {"input_price": 5.00, "output_price": 5.00},
        },
        "tasks": {"t1": {}, "t2": {}},
        "results": {
            "t1": {
                "cheap-model": {"pass": True, "cost": 1.0, "real_cost": 1.0, "calls": 1},
                "frontier-model": {"pass": True, "cost": 10.0, "real_cost": 10.0, "calls": 1},
            },
            # A row EXISTS but nothing ever ran: zero priced calls and $0 real spend. The
            # `pass: False` here is the absence of a run, not an observed failure — scoring it
            # as a measured fail@$0 invents a free win for whichever strategy picks it.
            "t2": {
                "cheap-model": {"pass": True, "cost": 1.0, "real_cost": 1.0, "calls": 1},
                "frontier-model": {"pass": False, "cost": 0.0, "real_cost": 0.0, "calls": 0},
            },
        },
    }


@pytest.mark.parametrize("impute_enabled", [True, False])
def test_a_never_executed_cell_is_unscorable_under_either_impute_setting(
    monkeypatch, impute_enabled
):
    # The non-observation predicate must hold on the SCORING path, so flipping the supported
    # `impute.enabled` key cannot silently promote never-executed cells into measured failures
    # at $0 — they feed the kill gate, concentrated on the baseline control model.
    monkeypatch.setattr(config, "impute_config", lambda: {"enabled": impute_enabled})
    rows = compute_strategy_rows(
        _matrix_with_never_executed_cell(),
        ["t1", "t2"],
        [AlwaysFrontier()],
        gamma=0.1,
        bootstrap=50,
        seed=1,
    )
    row = next(r for r in rows if r["strategy"] == "Always-Frontier")
    assert row["n_unscorable"] == 1
    assert row["n_tasks"] == 1
    assert row["TotalCost"] == 10.0


class _CascadeToucherOfAnUnmeasuredCell:
    """A cascade whose PATH crossed an unmeasured cell, though its FINAL cell is measured."""

    name = "Fake-Cascade"

    def __init__(self) -> None:
        self.cascade_total_cost = 0.0
        self.cascade_scorable = True

    def select(self, tid, task_meta, matrix):
        # t2's path crosses a cell the matrix never measured, billed at $0 — exactly what
        # `cascade_scorable` exists to report. The returned cell itself IS measured, so the
        # single-cell scorability test cannot see the understated cost.
        self.cascade_scorable = tid != "t2"
        self.cascade_total_cost = 1.0
        return "cheap-model"


def test_a_cascade_that_crossed_an_unmeasured_cell_is_unscorable():
    # `cascade_scorable` is authoritative on the SCORING path, not only in the kill gate:
    # otherwise a cascade silently bills a never-measured intermediate at $0 and the
    # published total understates real spend.
    matrix = {
        "models": {"cheap-model": {"input_price": 0.1, "output_price": 0.1}},
        "tasks": {"t1": {}, "t2": {}},
        "results": {
            "t1": {"cheap-model": {"pass": True, "cost": 1.0, "real_cost": 1.0, "calls": 1}},
            "t2": {"cheap-model": {"pass": True, "cost": 1.0, "real_cost": 1.0, "calls": 1}},
        },
    }
    rows = compute_strategy_rows(
        matrix, ["t1", "t2"], [_CascadeToucherOfAnUnmeasuredCell()], gamma=0.1, bootstrap=50, seed=1
    )
    row = next(r for r in rows if r["strategy"] == "Fake-Cascade")
    assert row["n_unscorable"] == 1
    assert row["n_tasks"] == 1


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
