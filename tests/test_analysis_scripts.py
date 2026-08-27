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
    ladder_evidence,
    plot_exploration,
    plot_knn_nulls,
    threshold_sweep,
    viz_knn,
)

# Absolute config path so the test is independent of the process CWD.
CONFIG_PATH = str(Path(config.__file__).resolve().parent / "benchmark.yaml")

# The analysis scripts that must exit cleanly on a header-only results.csv.
_GUARDED_SCRIPTS: Final = [
    compute_costs.main,
    ladder_evidence.main,
    plot_exploration.main,
    plot_knn_nulls.main,
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
        assert "kNN-semantic" in factories
        assert "kNN-semantic-cascade" in factories
        assert factories["kNN-semantic"]().name == "kNN-semantic"
        assert factories["kNN-semantic-cascade"]().name == "kNN-semantic-cascade"


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

    def test_thin_rows_are_rejected_with_a_reason(self):
        from benchmark.routing.report import _validate_rows

        missing = _validate_rows([{"strategy": "kNN-semantic", "n_tasks": 3}])
        assert missing is not None and "TotalCost" in missing

        no_evidence = _validate_rows(
            [{"strategy": "kNN-semantic", "n_tasks": 0, "TotalCost": 0.0, "AvgPerf%": 0.0}]
        )
        assert no_evidence is not None and "scorable" in no_evidence

        assert (
            _validate_rows(
                [{"strategy": "kNN-semantic", "n_tasks": 2, "TotalCost": 1.0, "AvgPerf%": 50.0}]
            )
            is None
        )


class TestSweepSelectionIsNotRewardArgmax:
    """gamma=0.1 makes reward-argmax degenerate to escalate-everything."""

    def test_cost_at_equal_quality_prefers_the_cheap_equivalent(self):
        rows = [
            # Statistically indistinguishable pass rates, wildly different cost.
            {"k": 54, "n_scored": 177, "AvgPerf%": 96.0, "TotalCost": 81.2, "Reward": 161.9},
            {"k": 24, "n_scored": 177, "AvgPerf%": 92.1, "TotalCost": 62.0, "Reward": 155.0},
        ]
        best = threshold_sweep.cost_at_equal_quality(rows)
        assert best["k"] == 24  # cheaper, and not significantly worse

    def test_a_significantly_worse_cell_is_not_selected(self):
        rows = [
            {"k": 54, "n_scored": 177, "AvgPerf%": 96.0, "TotalCost": 81.2, "Reward": 161.9},
            {"k": 2, "n_scored": 177, "AvgPerf%": 40.0, "TotalCost": 1.0, "Reward": 70.0},
        ]
        best = threshold_sweep.cost_at_equal_quality(rows)
        assert best["k"] == 54  # the cheap cell is far below the quality floor

    def test_zero_evidence_rows_are_never_selected(self):
        rows = [
            {"k": 9, "n_scored": 0, "AvgPerf%": 0.0, "TotalCost": 0.0, "Reward": 0.0},
            {"k": 24, "n_scored": 177, "AvgPerf%": 92.1, "TotalCost": 62.0, "Reward": 155.0},
        ]
        assert threshold_sweep.cost_at_equal_quality(rows)["k"] == 24

    def test_empty_rows_return_none(self):
        assert threshold_sweep.cost_at_equal_quality([]) is None

    def test_folds_partition_every_task(self):
        folds = threshold_sweep.fold_assignment(177, 5)
        assert len(folds) == 177
        assert set(folds.tolist()) == {0, 1, 2, 3, 4}
        # Balanced to within one task, so no fold dominates the index.
        counts = [int((folds == f).sum()) for f in range(5)]
        assert max(counts) - min(counts) <= 1

    def test_fold_assignment_is_deterministic(self):
        a = threshold_sweep.fold_assignment(50, 5)
        b = threshold_sweep.fold_assignment(50, 5)
        assert a.tolist() == b.tolist()


class TestSweepHeldOutRestrictsTheIndex:
    """A task must never vote over its own fold — that is what makes the sweep held-out."""

    def test_allowed_restricts_the_neighbourhood(self):
        config.load(CONFIG_PATH)
        models = {
            "cheap-model": {"input_price": 0.1, "output_price": 0.1},
            "frontier-model": {"input_price": 5.0, "output_price": 5.0},
        }
        matrix = {"models": models}
        # Tasks 0,1 pass on cheap; task 2 (the only one allowed as a neighbour) fails it.
        results_map = {
            "t0": {"cheap-model": {"pass": True, "cost": 1.0}},
            "t1": {"cheap-model": {"pass": True, "cost": 1.0}},
            "t2": {
                "cheap-model": {"pass": False, "cost": 1.0},
                "frontier-model": {"pass": True, "cost": 10.0},
            },
        }
        task_ids = ["t0", "t1", "t2"]
        features = np.array([[1.0, 0.0], [1.0, 0.01], [0.99, 0.0]])

        # Voting over everything: t0's neighbours include the passing t1 -> cheap qualifies.
        chosen_all, _p, _c, _s = threshold_sweep.knn_select(
            0,
            task_ids,
            task_ids,
            features,
            results_map,
            matrix,
            k=2,
            success_rate_thresh=0.5,
            min_samples=1,
        )
        assert chosen_all == "cheap-model"

        # Restricted to t2 only: the sole neighbour fails cheap, so the rule escalates.
        chosen_held, _p, _c, _s = threshold_sweep.knn_select(
            0,
            task_ids,
            task_ids,
            features,
            results_map,
            matrix,
            k=2,
            success_rate_thresh=0.5,
            min_samples=1,
            allowed=np.array([2]),
        )
        assert chosen_held == "frontier-model"

    def test_precomputed_sims_row_matches_the_on_the_fly_computation(self):
        config.load(CONFIG_PATH)
        matrix = {"models": {"m": {"input_price": 1.0, "output_price": 1.0}}}
        results_map = {f"t{i}": {"m": {"pass": i % 2 == 0, "cost": 1.0}} for i in range(6)}
        task_ids = sorted(results_map)
        rng = np.random.default_rng(0)
        feats = rng.normal(size=(6, 4))
        unit = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        sims = unit @ unit.T
        for i in range(6):
            direct = threshold_sweep.knn_select(
                i,
                task_ids,
                task_ids,
                feats,
                results_map,
                matrix,
                k=3,
                success_rate_thresh=0.5,
                min_samples=1,
            )
            cached = threshold_sweep.knn_select(
                i,
                task_ids,
                task_ids,
                feats,
                results_map,
                matrix,
                k=3,
                success_rate_thresh=0.5,
                min_samples=1,
                sims_row=sims[i],
            )
            assert direct == cached

    def test_evaluate_params_reports_allocation_degeneracy(self):
        config.load(CONFIG_PATH)
        matrix = {
            "models": {
                "cheap-model": {"input_price": 0.1, "output_price": 0.1},
                "frontier-model": {"input_price": 5.0, "output_price": 5.0},
            }
        }
        # Nothing clears a 0.99 threshold on cheap, so every task escalates.
        results_map = {
            f"t{i}": {
                "cheap-model": {"pass": False, "cost": 1.0},
                "frontier-model": {"pass": True, "cost": 10.0},
            }
            for i in range(6)
        }
        task_ids = sorted(results_map)
        rng = np.random.default_rng(1)
        feats = rng.normal(size=(6, 4))
        row = threshold_sweep.evaluate_params(
            task_ids,
            task_ids,
            feats,
            results_map,
            matrix,
            k=3,
            success_rate_thresh=0.99,
            min_samples=1,
            frontier_model="frontier-model",
        )
        assert row["frontier_share"] == 1.0
        assert row["n_models_used"] == 1  # a "router" that uses one model is a fixed policy
