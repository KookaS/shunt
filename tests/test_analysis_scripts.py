"""Guard tests for the routing analysis/reporting periphery — clean exit on an
empty results.csv, and report.py's regret plot including the shipped
kNN / kNN-cascade routers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pytest

from benchmark import config
from benchmark.routing.scripts import (
    compute_costs,
    embedding_compare,
    plot_exploration,
    plot_strategies,
    plot_timing,
    threshold_sweep,
    viz_knn,
)

# Absolute config path so the test is independent of the process CWD.
CONFIG_PATH = str(Path(config.__file__).resolve().parent / "benchmark.yaml")

# The analysis scripts that must exit cleanly on a header-only results.csv.
_GUARDED_SCRIPTS: Final = [
    compute_costs.main,
    embedding_compare.main,
    plot_exploration.main,
    plot_strategies.main,
    plot_timing.main,
    threshold_sweep.main,
    viz_knn.main,
]


@pytest.mark.parametrize("script_main", _GUARDED_SCRIPTS, ids=lambda m: m.__module__)
def test_script_exits_cleanly_on_empty_matrix(script_main, monkeypatch, capsys, tmp_path):
    """Each analysis script must early-return with a message, not raise."""
    monkeypatch.setattr("sys.argv", ["prog"])
    monkeypatch.setattr(config, "results_csv_path", lambda: tmp_path / "empty.csv")
    # Must not raise (ZeroDivisionError / KeyError / ValueError before the fix).
    assert script_main(CONFIG_PATH) is None
    out = capsys.readouterr().out
    assert "No results yet" in out


class TestEmbeddingCompareNeighbors:
    """compute_overlap must not crash when fewer challenges than k are populated."""

    def test_fewer_tasks_than_k_does_not_crash(self):
        # 6 populated challenges, k=10 (the shipped k) — the partial-population state.
        feats = np.eye(6, dtype=float)
        neighbors = embedding_compare.compute_overlap(feats, k=10)
        # One row per task; width clamped to the available non-self neighbors.
        assert neighbors.shape[0] == 6
        assert 1 <= neighbors.shape[1] <= 5

    def test_single_task_does_not_crash(self):
        feats = np.ones((1, 4), dtype=float)
        neighbors = embedding_compare.compute_overlap(feats, k=10)
        assert neighbors.shape[0] == 1


class TestReportRegretFactories:
    """report.py must not silently drop the headline kNN routers."""

    # The old ImportError-degradation path this class used to also cover was
    # dropped: _build_strategy_factories now single-sources the enabled
    # strategy set from run_eval.get_strategies, and fastembed/hnswlib are
    # unconditional base dependencies (pyproject.toml [project.dependencies],
    # not a benchmark-only extra) — "embedding deps unavailable" is not a
    # reachable state to degrade for. See
    # test_report_plots.py::TestStrategyFactoriesMatchEnabledSet for the
    # config-enabled-set coverage the refactor introduced.

    def test_knn_strategies_present_in_factory_map(self):
        from benchmark.routing.report import _build_strategy_factories

        factories = _build_strategy_factories(gamma=0.1)
        # Keys must equal the strategies' .name so they match results.csv rows.
        assert "kNN" in factories
        assert "kNN-cascade" in factories
        assert factories["kNN"]().name == "kNN"
        assert factories["kNN-cascade"]().name == "kNN-cascade"


class TestRegretExcludesUnscorable:
    """F1: cumulative_regret must EXCLUDE a coverage-gap decision (chosen model
    unmeasured on a task), not score it fail@$0 into the regret series."""

    def test_evaluate_strategies_flags_unmeasured_cell(self):
        from benchmark.routing.report import _evaluate_strategies

        class _Fake:
            name = "Fake"

            def select(self, tid, meta, matrix):  # noqa: ANN001, ANN201, ARG002
                return "frontier-model"

        matrix = {
            "tasks": {"t1": {}, "t2": {}},
            "results": {
                "t1": {"frontier-model": {"pass": True, "cost": 10.0}},
                # t2 has NO frontier-model cell -> a coverage gap.
                "t2": {"cheap-model": {"pass": True, "cost": 1.0}},
            },
        }
        evaluated = _evaluate_strategies({"Fake": _Fake}, matrix, ["t1", "t2"])
        decisions, unscorable = evaluated["Fake"]
        assert unscorable == {"t2"}
        # The gap cell is still present positionally but must be excluded downstream.
        assert decisions[1] == ("t2", "frontier-model", False, 0.0)

    def test_compute_per_task_regret_drops_excluded_task(self):
        from benchmark.routing.report import _compute_per_task_regret

        strat = [("t1", "m", True, 1.0), ("t2", "m", False, 0.0), ("t3", "m", True, 1.0)]
        oracle = [("t1", "o", True, 1.0), ("t2", "o", True, 2.0), ("t3", "o", True, 1.0)]
        # t2 is the coverage gap: dropped, so the series covers 2 tasks not 3, and the
        # phantom fail@$0 regret it would have contributed never enters the curve.
        excluded = _compute_per_task_regret(strat, oracle, 0.1, {"t2"})
        assert len(excluded) == 2
        imputed = _compute_per_task_regret(strat, oracle, 0.1, None)
        assert len(imputed) == 3
        # The dropped task carried real regret under imputation -> curves differ.
        assert float(excluded[-1]) != float(imputed[-1])


class TestThresholdSweepExcludesUnscorable:
    """F2: evaluate_params must EXCLUDE a coverage-gap escalation (swept kNN rule
    lands on a model unmeasured on the task), not impute fail@$0."""

    def test_unmeasured_chosen_cell_excluded_from_aggregation(self):
        config.load(CONFIG_PATH)
        models = {
            "cheap-model": {"input_price": 0.1, "output_price": 0.1},
            "frontier-model": {"input_price": 5.0, "output_price": 5.0},
        }
        matrix = {"models": models}
        results_map = {
            "t1": {"cheap-model": {"pass": True, "cost": 1.0}},
            "t2": {"cheap-model": {"pass": True, "cost": 1.0}},
            # t3 measured ONLY on frontier -> neighbours vote cheap, so the chosen
            # cheap cell is missing here: a coverage gap that must be excluded.
            "t3": {"frontier-model": {"pass": True, "cost": 10.0}},
        }
        task_ids = ["t1", "t2", "t3"]
        # Near-identical embeddings so every task's neighbourhood is the other two.
        features = np.array([[1.0, 0.0], [1.0, 0.01], [0.99, 0.0]])

        chosen, _passed, _cost, scored = threshold_sweep.knn_select(
            2,
            task_ids,
            task_ids,
            features,
            results_map,
            matrix,
            k=2,
            success_rate_thresh=0.5,
            min_samples=1,
        )
        assert chosen == "cheap-model"
        assert scored is False  # unmeasured chosen cell -> unscorable

        row = threshold_sweep.evaluate_params(
            task_ids,
            task_ids,
            features,
            results_map,
            matrix,
            k=2,
            success_rate_thresh=0.5,
            min_samples=1,
        )
        assert row["n_excluded"] == 1
        assert row["n_scored"] == 2
        # AvgPerf% is over the 2 SCORED tasks (both pass), not diluted to 66% by a
        # phantom fail on t3.
        assert row["AvgPerf%"] == 100.0


class TestPlotTimingStrategyCalls:
    """plot_timing._strategy_calls must unpack _evaluate_strategies' (decisions,
    unscorable) pair — a populated matrix, since the empty-matrix guard early-returns
    before reaching it (this is the shape the guard could not catch).
    """

    def test_strategy_calls_on_populated_matrix(self):
        config.load(CONFIG_PATH)
        matrix = {
            "tasks": {"t1": {}, "t2": {}},
            "results": {
                "t1": {"kNN": {"pass": True, "cost": 1.0, "calls": 3}},
                "t2": {"kNN": {"pass": True, "cost": 1.0, "calls": 5}},
            },
        }

        class _Fake:
            name = "kNN"

            def select(self, tid, meta, matrix):  # noqa: ANN001, ANN201, ARG002
                return "kNN"

        # Patch the factory builder so the replay uses a measured model on every task.
        import benchmark.routing.report as report_mod

        orig = report_mod._build_strategy_factories
        report_mod._build_strategy_factories = lambda gamma: {"kNN": _Fake}
        try:
            out = plot_timing._strategy_calls(matrix, ["t1", "t2"], gamma=0.1)
        finally:
            report_mod._build_strategy_factories = orig
        assert out == {"kNN": [3, 5]}

    def test_strategy_calls_excludes_coverage_gap(self):
        config.load(CONFIG_PATH)
        matrix = {
            "tasks": {"t1": {}, "t2": {}},
            "results": {
                "t1": {"kNN": {"pass": True, "cost": 1.0, "calls": 4}},
                # t2 has no kNN cell -> unscorable; its calls must not be counted.
                "t2": {"other": {"pass": True, "cost": 1.0, "calls": 9}},
            },
        }

        class _Fake:
            name = "kNN"

            def select(self, tid, meta, matrix):  # noqa: ANN001, ANN201, ARG002
                return "kNN"

        import benchmark.routing.report as report_mod

        orig = report_mod._build_strategy_factories
        report_mod._build_strategy_factories = lambda gamma: {"kNN": _Fake}
        try:
            out = plot_timing._strategy_calls(matrix, ["t1", "t2"], gamma=0.1)
        finally:
            report_mod._build_strategy_factories = orig
        assert out == {"kNN": [4]}


class TestZeroEvidenceRows:
    """A strategy with no scorable task must never be certified Pareto-optimal,
    and a degenerate row set must fail loudly instead of crashing mid-report."""

    def test_empty_decisions_yield_shaped_zero_metrics(self):
        from benchmark.routing.metrics import compute_metrics

        m = compute_metrics([])
        # Every key report.py/plot_strategies.py index must exist (was: {} -> KeyError).
        assert m["n_tasks"] == 0
        assert m["TotalCost"] == 0.0
        assert m["AvgPerf%"] == 0.0

    def test_zero_task_strategy_is_not_pareto(self):
        from benchmark.routing.report import _is_pareto

        assert _is_pareto({"strategy": "kNN", "n_tasks": 0, "Pareto": True}) is False
        assert _is_pareto({"strategy": "kNN", "n_tasks": "", "Pareto": "True"}) is False
        assert _is_pareto({"strategy": "kNN", "n_tasks": 3, "Pareto": True}) is True

    def test_thin_rows_are_rejected_with_a_reason(self):
        from benchmark.routing.report import _validate_rows

        missing = _validate_rows([{"strategy": "kNN", "n_tasks": 3}])
        assert missing is not None and "TotalCost" in missing

        no_evidence = _validate_rows(
            [{"strategy": "kNN", "n_tasks": 0, "TotalCost": 0.0, "AvgPerf%": 0.0}]
        )
        assert no_evidence is not None and "scorable" in no_evidence

        assert (
            _validate_rows([{"strategy": "kNN", "n_tasks": 2, "TotalCost": 1.0, "AvgPerf%": 50.0}])
            is None
        )
