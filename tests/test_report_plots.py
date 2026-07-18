"""Tests for the report plot helpers — the phantom-frontier guard in particular."""

from __future__ import annotations

from benchmark import config
from benchmark.routing import report, run_eval


class TestStrategyFactoriesMatchEnabledSet:
    """The regret plot's strategy set must derive from the same config-enabled
    source every other plot uses (run_eval.get_strategies) — no headline
    strategy silently dropped, no strategy added that isn't config-enabled."""

    def test_every_enabled_strategy_has_a_factory(self):
        config.load("benchmark/config.yaml")
        enabled_names = {s.name for s in run_eval.get_strategies()}
        factories = report._build_strategy_factories(config.gamma())
        assert enabled_names <= factories.keys()

    def test_external_prior_is_not_silently_dropped(self):
        # external_prior is enabled in config.yaml's strategies.enabled — a
        # Pareto-front headline strategy that must appear on the regret plot.
        config.load("benchmark/config.yaml")
        factories = report._build_strategy_factories(config.gamma())
        assert "External-Prior" in factories

    def test_oracle_reward_always_present_as_internal_reference(self):
        # Oracle-reward is the regret plot's baseline every strategy is scored
        # against — required even when config.yaml comments it out of `enabled`.
        config.load("benchmark/config.yaml")
        assert "oracle_reward" not in config.strategies().get("enabled", [])
        factories = report._build_strategy_factories(config.gamma())
        assert "Oracle-reward" in factories

    def test_no_strategy_added_beyond_enabled_plus_oracle_reward(self):
        config.load("benchmark/config.yaml")
        enabled_names = {s.name for s in run_eval.get_strategies()}
        factories = report._build_strategy_factories(config.gamma())
        assert factories.keys() == enabled_names | {"Oracle-reward"}


class TestArmSizeLegend:
    """N4 (plot_arm_cloud) must explain size=arm-rank in-figure — a viewer
    can't otherwise decode marker size (a self-sufficiency gap)."""

    def test_degenerate_single_rank_has_one_handle(self):
        handles = report._arm_size_legend_handles(0)
        assert len(handles) == 1
        assert "0" in handles[0].get_label()

    def test_multi_rank_spans_endpoints(self):
        handles = report._arm_size_legend_handles(2)
        labels = " ".join(h.get_label() for h in handles)
        assert "rank 0" in labels
        assert "rank 2" in labels

    def test_marker_size_grows_with_rank(self):
        handles = report._arm_size_legend_handles(2)
        sizes = [h.get_markersize() for h in handles]
        assert sizes == sorted(sizes)
        assert sizes[0] < sizes[-1]

    def _raw_multi_arm(self):
        return {
            "t1": {
                "m1": {"none": {"pass": True, "cost": 0.01}, "high": {"pass": True, "cost": 0.05}},
            },
            "t2": {
                "m1": {"none": {"pass": False, "cost": 0.01}, "high": {"pass": True, "cost": 0.05}},
            },
        }

    def _raw_single_arm(self):
        return {"t1": {"m1": {"default": {"pass": True, "cost": 0.01}}}}

    def test_plot_arm_cloud_renders_legend_on_multi_arm_data(self, tmp_path):
        model_colors = {"m1": "#0072B2"}
        arm_ranks = {("m1", "none"): 0, ("m1", "high"): 1}
        out = report.plot_arm_cloud(self._raw_multi_arm(), tmp_path, model_colors, arm_ranks)
        assert out is not None and out.exists() and out.stat().st_size > 0

    def test_plot_arm_cloud_renders_legend_on_single_arm_data(self, tmp_path):
        model_colors = {"m1": "#0072B2"}
        arm_ranks: dict[tuple[str, str], int] = {}
        out = report.plot_arm_cloud(self._raw_single_arm(), tmp_path, model_colors, arm_ranks)
        assert out is not None and out.exists() and out.stat().st_size > 0


class TestDisabledModelExcluded:
    def test_disabled_model_cannot_leak_via_stray_row(self):
        # A disabled model (opus) with a stray results row must never re-enter the
        # matrix — otherwise it silently re-promotes to frontier (opus $30 > k3 $18).
        config.load("benchmark/config.yaml")
        stray = {
            "t1": {
                "deepseek-v4-flash": {"pass": True, "cost": 0.01},
                "claude-opus-4-6": {"pass": True, "cost": 0.2},  # disabled in config
            }
        }
        assert "claude-opus-4-6" not in config.models_matrix(stray)


def _matrix(results: dict) -> dict:
    # Two models; opus is far more expensive so it is the "frontier" pick.
    return {
        "models": {
            "cheap": {"input_price": 0.1, "output_price": 0.2},
            "opus": {"input_price": 5.0, "output_price": 25.0},
        },
        "results": results,
    }


class TestFrontierCoverage:
    def test_flags_partial_frontier(self):
        # opus present on 1 of 3 tasks -> phantom baseline.
        m = _matrix(
            {
                "t1": {"cheap": {"pass": True, "cost": 0.01}, "opus": {"pass": True, "cost": 0.2}},
                "t2": {"cheap": {"pass": True, "cost": 0.01}},
                "t3": {"cheap": {"pass": False, "cost": 0.01}},
            }
        )
        assert report._frontier_coverage(m) == ("opus", 1, 3)

    def test_full_coverage_is_not_phantom(self):
        m = _matrix(
            {
                "t1": {"cheap": {"pass": True, "cost": 0.01}, "opus": {"pass": True, "cost": 0.2}},
                "t2": {"cheap": {"pass": True, "cost": 0.01}, "opus": {"pass": True, "cost": 0.2}},
            }
        )
        frontier, covered, total = report._frontier_coverage(m)
        assert (frontier, covered, total) == ("opus", 2, 2)

    def test_none_matrix_returns_none(self):
        assert report._frontier_coverage(None) is None
        assert report._frontier_coverage({"models": {}, "results": {}}) is None


class TestParetoPhantomGuard:
    """K1 (plot_pareto) must exclude a phantom Always-Frontier from the hull/AIQ
    computation, mirroring the guard N1 and cost_savings already apply."""

    def test_phantom_frontier_excluded_from_hull_indices(self):
        names = ["Always-Cheap", "Always-Frontier"]
        pareto_map = {"Always-Cheap": True, "Always-Frontier": True}
        assert report._hull_pareto_indices(names, pareto_map, phantom=True) == [0]

    def test_non_phantom_frontier_kept_in_hull_indices(self):
        names = ["Always-Cheap", "Always-Frontier"]
        pareto_map = {"Always-Cheap": True, "Always-Frontier": True}
        assert report._hull_pareto_indices(names, pareto_map, phantom=False) == [0, 1]

    def test_non_pareto_strategy_excluded_regardless_of_phantom(self):
        names = ["Always-Cheap", "Random", "Always-Frontier"]
        pareto_map = {"Always-Cheap": True, "Random": False, "Always-Frontier": True}
        assert report._hull_pareto_indices(names, pareto_map, phantom=False) == [0, 2]

    def test_plot_pareto_renders_and_warns_on_phantom(self, tmp_path):
        rows = [
            {
                "strategy": "Always-Cheap",
                "TotalCost": "0.03",
                "AvgPerf%": "80",
                "Pareto": True,
                "n_tasks": "2",
                "n_pass": "2",
            },
            {
                "strategy": "Always-Frontier",
                "TotalCost": "0.0",
                "AvgPerf%": "0.0",
                "Pareto": True,
                "n_tasks": "0",
                "n_pass": "0",
            },
        ]
        m = _matrix(
            {
                "t1": {"cheap": {"pass": True, "cost": 0.01}},
                "t2": {"cheap": {"pass": True, "cost": 0.01}},
            }
        )
        # opus (the frontier) has zero coverage here -> a textbook phantom baseline.
        orig_close = report.plt.close
        report.plt.close = lambda *a, **k: None  # type: ignore[assignment]
        try:
            out = report.plot_pareto(rows, tmp_path, matrix=m)
            fig = report.plt.gcf()
            texts = " ".join(t.get_text() for t in fig.axes[0].texts)
        finally:
            report.plt.close = orig_close  # type: ignore[assignment]
            report.plt.close("all")
        assert out.exists() and out.stat().st_size > 0
        assert "phantom baseline" in texts

    def test_plot_pareto_no_warning_on_full_coverage(self, tmp_path):
        rows = [
            {
                "strategy": "Always-Cheap",
                "TotalCost": "0.03",
                "AvgPerf%": "80",
                "Pareto": True,
                "n_tasks": "2",
                "n_pass": "2",
            },
            {
                "strategy": "Always-Frontier",
                "TotalCost": "0.20",
                "AvgPerf%": "100",
                "Pareto": True,
                "n_tasks": "2",
                "n_pass": "2",
            },
        ]
        m = _matrix(
            {
                "t1": {"cheap": {"pass": True, "cost": 0.01}, "opus": {"pass": True, "cost": 0.2}},
                "t2": {"cheap": {"pass": True, "cost": 0.01}, "opus": {"pass": True, "cost": 0.2}},
            }
        )
        orig_close = report.plt.close
        report.plt.close = lambda *a, **k: None  # type: ignore[assignment]
        try:
            out = report.plot_pareto(rows, tmp_path, matrix=m)
            fig = report.plt.gcf()
            texts = " ".join(t.get_text() for t in fig.axes[0].texts)
        finally:
            report.plt.close = orig_close  # type: ignore[assignment]
            report.plt.close("all")
        assert out.exists() and out.stat().st_size > 0
        assert "phantom baseline" not in texts


class TestPlotsRender:
    def test_cost_savings_renders_with_phantom(self, tmp_path):
        rows = [
            {"strategy": "Always-Cheap", "TotalCost": "0.03", "AvgPerf%": "80"},
            {"strategy": "Always-Frontier", "TotalCost": "0.20", "AvgPerf%": "10"},
        ]
        m = _matrix(
            {
                "t1": {"cheap": {"pass": True, "cost": 0.01}, "opus": {"pass": True, "cost": 0.2}},
                "t2": {"cheap": {"pass": True, "cost": 0.01}},
            }
        )
        out = report.plot_cost_savings(rows, tmp_path, m)
        assert out.exists() and out.stat().st_size > 0

    def test_heatmap_renders_task_by_model(self, tmp_path):
        m = _matrix(
            {
                "proj__t1": {
                    "cheap": {"pass": True, "cost": 0.01},
                    "opus": {"pass": True, "cost": 0.2},
                },
                "proj__t2": {"cheap": {"pass": False, "cost": 0.01}},
            }
        )
        challenges = tmp_path / "challenges.json"
        challenges.write_text("{}")
        # plot_heatmap reloads via config.load_matrix; drive it through the public
        # path by monkeypatching load_matrix to return our in-memory matrix.
        orig = report.load_matrix
        report.load_matrix = lambda _p: m  # type: ignore[assignment]
        try:
            out = report.plot_heatmap(challenges, tmp_path)
        finally:
            report.load_matrix = orig  # type: ignore[assignment]
        assert out.exists() and out.stat().st_size > 0
