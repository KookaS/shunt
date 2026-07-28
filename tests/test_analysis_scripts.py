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
    plot_knn_nulls,
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
    plot_knn_nulls.main,
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
        # report._is_pareto (which read the summary's own Pareto column) was removed:
        # that column ranks hindsight oracles alongside routers, so on the cost-quality
        # plane it painted a real router "dominated". _deployable_pareto is the single
        # definition now, and it keeps the same no-evidence guard.
        from benchmark.routing.report import _deployable_pareto

        assert _deployable_pareto(["kNN"], [0.0], [0.0], [0]) == set()
        assert _deployable_pareto(["kNN"], [1.0], [50.0], [3]) == {"kNN"}

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


class TestNeighborhoodPurity:
    """viz_knn's purity must exclude the query from its own neighbourhood and
    normalise by the neighbours that exist, not by the requested k."""

    MODELS: Final = ["deepseek-v4-flash", "kimi-k3"]

    def _fixture(self, n: int):
        """n unit-norm vectors plus a pass matrix where the cheapest model always passes."""
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(n, 16))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        vecs = np.zeros((n, len(self.MODELS) * 4))
        vecs[:, 0] = 1.0
        return emb, vecs

    def test_k_above_task_count_reports_true_purity(self):
        # 8 tasks all routed to one model: purity is 1.00 by definition. Dividing by
        # the configured k=20 and counting the query itself as a neighbour reported 0.40.
        config.load(CONFIG_PATH)
        n = 8
        emb, vecs = self._fixture(n)
        task_ids = [f"t{i}" for i in range(n)]
        purity, selections = viz_knn.compute_neighborhood_purity(
            emb @ emb.T, task_ids, vecs, emb, self.MODELS, k=20
        )
        assert len(set(selections.values())) == 1
        assert purity.tolist() == [1.0] * n

    def test_query_is_never_its_own_neighbour(self):
        n = 8
        emb, _ = self._fixture(n)
        sim = emb @ emb.T
        for i in range(n):
            neighbors = viz_knn._nearest_neighbors(sim[i], i, 20).tolist()
            assert i not in neighbors
            assert len(neighbors) == n - 1

    def test_single_task_matrix_does_not_divide_by_zero(self):
        config.load(CONFIG_PATH)
        emb, vecs = self._fixture(1)
        purity, _ = viz_knn.compute_neighborhood_purity(
            emb @ emb.T, ["t0"], vecs, emb, self.MODELS, k=10
        )
        assert purity.tolist() == [0.0]


class TestPlotStrategiesReadsTheSummary:
    """The figure must plot strategy_summary.csv, never a second derivation of it."""

    # The committed PNG once disagreed with that CSV on all seven points (Tier-Classifier
    # by 174% on cost) because it re-derived the rows itself. It also loaded three ONNX
    # embedders to do so and was OOM-killed at >4 GB.

    def _write(self, path: Path) -> None:
        path.write_text(
            "strategy,n_tasks,n_unscorable,n_pass,AvgPerf%,AvgPerf_ci_lower,AvgPerf_ci_upper,"
            "TotalCost,AvgCost,Reward,CumReg,CumReg_ci_lower,CumReg_ci_upper,rAcc,Pareto\n"
            "Oracle,177,23,171,96.61,93.79,98.87,13.5896,0.076778,169.641,0.0,0.0,0.0,1.0,True\n"
            "kNN,174,26,136,78.16,71.84,83.91,9.5279,0.054758,135.05,31.6,22.3,41.3,0.6,False\n"
        )

    def test_rows_are_typed_and_match_the_csv(self, tmp_path):
        csv_path = tmp_path / "strategy_summary.csv"
        self._write(csv_path)
        rows = plot_strategies.load_summary_rows(csv_path)
        assert [r["strategy"] for r in rows] == ["Oracle", "kNN"]
        assert rows[0]["AvgPerf%"] == 96.61
        assert rows[0]["TotalCost"] == 13.5896
        # The CIs the old figure discarded must survive the read.
        assert rows[1]["AvgPerf_ci_lower"] == 71.84
        assert rows[1]["AvgPerf_ci_upper"] == 83.91
        assert rows[0]["Pareto"] is True
        assert rows[1]["Pareto"] is False

    def test_no_embedding_strategy_is_instantiated(self, monkeypatch, tmp_path):
        """Reading the CSV must not touch the strategy factories that load embedders."""
        import benchmark.routing.report as report_mod

        def _boom(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
            raise AssertionError("plot_strategies must not re-derive rows")

        monkeypatch.setattr(report_mod, "derive_rows", _boom, raising=False)
        csv_path = tmp_path / "strategy_summary.csv"
        self._write(csv_path)
        assert len(plot_strategies.load_summary_rows(csv_path)) == 2

    def test_stale_summary_is_flagged(self, tmp_path):
        summary_csv = tmp_path / "strategy_summary.csv"
        results_csv = tmp_path / "results.csv"
        self._write(summary_csv)
        results_csv.write_text("x\n")
        import os

        # Summary predates results -> the figure would describe an earlier results set.
        os.utime(summary_csv, (1_000_000, 1_000_000))
        os.utime(results_csv, (2_000_000, 2_000_000))
        note = plot_strategies._staleness_limit(summary_csv, results_csv)
        assert note is not None and "STALE INPUT" in note

    def test_fresh_summary_is_not_flagged(self, tmp_path):
        summary_csv = tmp_path / "strategy_summary.csv"
        results_csv = tmp_path / "results.csv"
        self._write(summary_csv)
        results_csv.write_text("x\n")
        import os

        os.utime(results_csv, (1_000_000, 1_000_000))
        os.utime(summary_csv, (2_000_000, 2_000_000))
        assert plot_strategies._staleness_limit(summary_csv, results_csv) is None

    def test_missing_summary_exits_cleanly(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr("sys.argv", ["prog", "--summary", str(tmp_path / "absent.csv")])
        assert plot_strategies.main(CONFIG_PATH) is None
        assert "No results yet" in capsys.readouterr().out


class TestLabelDeclutter:
    """Nudging overlapping labels apart must stay bounded."""

    # The nudge has no natural stop: every unresolved pass moves a label further, so without
    # a ceiling that only applies to below-marker labels AND a clamp on the axes box, a dense
    # cluster walks its labels clean off the figure.

    @staticmethod
    def _stack(dy: float, n: int = 6):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([0.5] * n, [0.5] * n)
        anns = [
            ax.annotate(
                f"label {i}\n99%, $9.99",
                (0.5, 0.5),
                fontsize=8,
                textcoords="offset points",
                xytext=(10, dy),
                va="bottom" if dy > 0 else "top",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none"},
            )
            for i in range(n)
        ]
        return fig, ax, anns

    def test_labels_stay_inside_the_axes(self):
        fig, ax, anns = self._stack(dy=-13.0)
        plot_strategies._declutter(fig, ax, anns)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = ax.get_window_extent(renderer)
        for ann in anns:
            box = plot_strategies._label_box(ann, renderer)
            assert box.y0 >= bounds.y0 - 1.0
            assert box.y1 <= bounds.y1 + 1.0

    def test_a_label_above_its_marker_may_move_up(self):
        # The ceiling exists so a BELOW-marker label never rises into the title. Applying it
        # to above-marker labels too left room == 0 on every first pass, so labels could only
        # ever move DOWN and the whole cluster drifted one way.
        fig, ax, anns = self._stack(dy=11.0)
        plot_strategies._declutter(fig, ax, anns)
        assert max(ann.xyann[1] for ann in anns) > 11.0

    def test_a_label_below_its_marker_never_rises_above_its_start(self):
        fig, ax, anns = self._stack(dy=-13.0)
        plot_strategies._declutter(fig, ax, anns)
        assert max(ann.xyann[1] for ann in anns) <= -13.0


class TestTimingExcludesCensoredZeroCallRows:
    """15 rows recorded 0 calls; `if not calls` let them through because "0" is truthy."""

    HEADER: Final = "model,calls,pass,timeout_flag,stop_reason\n"

    def _csv(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "results.csv"
        path.write_text(self.HEADER + body)
        return path

    def test_zero_call_rows_are_dropped_not_averaged(self, tmp_path):
        csv_path = self._csv(
            tmp_path,
            "kimi-k3,40,True,False,solved\n"
            "kimi-k3,30,True,False,solved\n"
            "kimi-k3,0,False,True,\n",  # censored before the first round-trip
        )
        calls, zero, _cens = plot_timing._model_calls(csv_path)
        assert calls["kimi-k3"] == [40, 30]
        assert zero == {"kimi-k3": 1}
        # Counting the 0 would have reported 23.3 instead of 35.0.
        assert sum(calls["kimi-k3"]) / len(calls["kimi-k3"]) == 35.0

    def test_censored_rows_that_made_calls_are_kept_and_counted(self, tmp_path):
        csv_path = self._csv(
            tmp_path,
            "kimi-k3,40,False,False,step_limit\nkimi-k3,30,True,False,solved\n",
        )
        calls, zero, censored = plot_timing._model_calls(csv_path)
        assert calls["kimi-k3"] == [40, 30]
        assert zero == {}
        assert censored == 1  # truncated run -> the bar is a lower bound

    def test_csv_string_booleans_are_coerced(self):
        # bool("False") is True, so a raw row must never be handed to censoring.is_censored.
        assert plot_timing._as_bool("False") is False
        assert plot_timing._as_bool("True") is True
        assert plot_timing._is_censored_row({"pass": "False", "timeout_flag": "True"}) is True
        assert plot_timing._is_censored_row({"pass": "True", "timeout_flag": "False"}) is False

    def test_blank_calls_column_is_skipped(self, tmp_path):
        csv_path = self._csv(tmp_path, "kimi-k3,,False,False,\nkimi-k3,12,True,False,solved\n")
        calls, zero, _c = plot_timing._model_calls(csv_path)
        assert calls["kimi-k3"] == [12]
        assert zero == {}


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
