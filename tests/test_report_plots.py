"""Tests for the report plot helpers — the phantom-frontier guard in particular."""

from __future__ import annotations

from benchmark import config
from benchmark.routing import report, run_eval
from benchmark.routing.impute import ImputedMatrix


class TestStrategyFactoriesMatchEnabledSet:
    """The regret plot's strategy set must derive from the same config-enabled
    source every other plot uses (run_eval.get_strategies) — no headline
    strategy silently dropped, no strategy added that isn't config-enabled."""

    def test_every_enabled_strategy_has_a_factory(self):
        config.load("benchmark/benchmark.yaml")
        enabled_names = {s.name for s in run_eval.get_strategies()}
        factories = report._build_strategy_factories(config.gamma())
        assert enabled_names <= factories.keys()

    def test_shipped_knn_is_not_silently_dropped(self):
        # knn is enabled in benchmark.yaml's strategies.enabled — the shipped
        # algorithm, and the headline strategy that must appear on the regret plot.
        config.load("benchmark/benchmark.yaml")
        factories = report._build_strategy_factories(config.gamma())
        assert "kNN" in factories

    def test_oracle_reward_always_present_as_internal_reference(self):
        # Oracle-reward is the regret plot's baseline every strategy is scored
        # against — required even when benchmark.yaml comments it out of `enabled`.
        config.load("benchmark/benchmark.yaml")
        assert "oracle_reward" not in config.strategies().get("enabled", [])
        factories = report._build_strategy_factories(config.gamma())
        assert "Oracle-reward" in factories

    def test_no_strategy_added_beyond_enabled_plus_oracle_reward(self):
        config.load("benchmark/benchmark.yaml")
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
        config.load("benchmark/benchmark.yaml")
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


class TestHullParetoIndices:
    """K1 (plot_pareto): the hull takes the DEPLOYABLE Pareto-flagged rows. A hindsight
    oracle anchored the hull apex, so the shaded 'achievable region' reached a height
    no live router can buy."""

    def test_pareto_flagged_rows_enter_hull(self):
        names = ["Always-Cheap", "Always-Frontier"]
        pareto_map = {"Always-Cheap": True, "Always-Frontier": True}
        assert report._hull_pareto_indices(names, pareto_map) == [0, 1]

    def test_non_pareto_strategy_excluded(self):
        names = ["Always-Cheap", "Random", "Always-Frontier"]
        pareto_map = {"Always-Cheap": True, "Random": False, "Always-Frontier": True}
        assert report._hull_pareto_indices(names, pareto_map) == [0, 2]

    def test_hindsight_oracle_never_anchors_the_hull(self):
        names = ["Always-Cheap", "Oracle", "Oracle-reward", "Always-Frontier"]
        pareto_map = dict.fromkeys(names, True)
        assert report._hull_pareto_indices(names, pareto_map) == [0, 3]


class TestDisclosureBanner:
    """The equal-coverage disclosure rides in the plot frame's footer NOTE section."""

    def test_plot_pareto_draws_banner(self, tmp_path):
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
        banner = "equal-coverage via monotone imputation — 40% of frontier cells imputed"
        orig_close = report.plt.close
        report.plt.close = lambda *a, **k: None  # type: ignore[assignment]
        try:
            out = report.plot_pareto(rows, tmp_path, banner=banner)
            fig = report.plt.gcf()
            # The footer is word-wrapped across fig.text lines, so compare on
            # whitespace-normalized text rather than the raw rendered strings.
            texts = " ".join(" ".join(t.get_text().split()) for t in fig.texts)
        finally:
            report.plt.close = orig_close  # type: ignore[assignment]
            report.plt.close("all")
        assert out.exists() and out.stat().st_size > 0
        assert "equal-coverage via monotone imputation" in texts

    def test_no_banner_when_none(self, tmp_path):
        rows = [
            {
                "strategy": "Always-Cheap",
                "TotalCost": "0.03",
                "AvgPerf%": "80",
                "Pareto": True,
                "n_tasks": "2",
                "n_pass": "2",
            },
        ]
        out = report.plot_pareto(rows, tmp_path, banner=None)
        assert out.exists() and out.stat().st_size > 0


def _imputed(n_multi_observed: int, violations: list | None = None) -> ImputedMatrix:
    return ImputedMatrix(
        matrix={},
        violations=list(violations or []),
        n_real=0,
        n_imputed=0,
        n_unknown=0,
        tau={},
        n_multi_observed=n_multi_observed,
    )


class TestNoEvidenceRowsDoNotBecomeClaims:
    """A strategy parked at $0 / n_tasks=0 is metrics.py's explicit no-scorable-task
    signal (see _deployable_pareto) — no footer sentence may count it as evidence."""

    def test_coverage_note_excludes_zero_n_rows_from_its_subject(self):
        ann = report._row_coverage_annotations([10, 10, 0])
        note = " ".join(ann.notes)
        assert "Every plotted strategy" not in note
        assert note == "Every scored strategy (2 of 3 plotted) is scored on the same 10 task(s)."

    def test_coverage_note_keeps_the_strong_claim_when_every_row_is_scored(self):
        ann = report._row_coverage_annotations([10, 10])
        assert ann.notes == ("Every plotted strategy is scored on the same 10 task(s).",)

    def test_cheapest_claim_ignores_the_no_evidence_row(self):
        # 8/10 vs the frontier's 9/10 -> intervals overlap, so Always-Cheap qualifies.
        ann = report._cost_savings_annotations(
            names=["Always-Cheap", "Broken", "Always-Frontier"],
            ns=[10, 0, 10],
            n_pass=[8, 0, 9],
            costs=[3.0, 0.0, 9.0],
            ref=9.0,
            banner=None,
        )
        cheapest = next(n for n in ann.notes if "Cheapest" in n)
        assert "$3.0000" in cheapest
        assert "$0.0000" not in cheapest

    def test_cheapest_claim_skips_a_strategy_that_is_not_equal_quality(self):
        # The GOAL is equal-quality cost. 1/10 vs 10/10 does not overlap, so naming
        # the cheap bar as the cheapest strategy would violate the figure's own goal.
        ann = report._cost_savings_annotations(
            names=["Always-Cheap", "Always-Frontier"],
            ns=[10, 10],
            n_pass=[1, 10],
            costs=[3.0, 9.0],
            ref=9.0,
            banner=None,
        )
        footer = " ".join(ann.notes)
        assert "no equal-quality cost claim" in footer
        assert "$3.0000" not in footer

    def test_coverage_and_cheapest_notes_do_not_contradict_each_other(self):
        ann = report._cost_savings_annotations(
            names=["Always-Cheap", "Broken"],
            ns=[10, 0],
            n_pass=[8, 0],
            costs=[3.0, 0.0],
            ref=9.0,
            banner=None,
        )
        footer = " ".join((*ann.notes, *ann.limitations))
        assert "no scorable task" in footer
        assert "Every plotted strategy" not in footer


class TestSyntheticArmCacheMakesNoArmClaim:
    """_synthesize_raw collapses every model to one 'default' arm when no per-arm
    cache exists — that is missing data, not a measured single-arm sweep."""

    _single = {"t1": {"m1": {"default": {"pass": True, "cost": 0.01}}}}

    def test_synthesized_cache_suppresses_the_single_arm_claim(self):
        assert report._single_arm_limits(self._single, synthesized=True) == ()

    def test_loaded_single_arm_cache_still_makes_the_claim(self):
        limits = report._single_arm_limits(self._single, synthesized=False)
        assert len(limits) == 1
        assert "exactly one sampled arm" in limits[0]

    def test_heatmap_footer_on_synthesized_data_claims_no_arm_sweep(self):
        import numpy as np

        ann = report._heatmap_annotations(
            grid=np.array([[1.0, 0.0]]),
            coverage=np.array([1, 1]),
            glyphs=True,
            raw=self._single,
            synthesized=True,
        )
        footer = " ".join(ann.limitations)
        assert "synthesized" in footer
        assert "exactly one sampled arm" not in footer


class TestMonotonicityUnmeasuredIsNotMeasured:
    """violation_ci returns (0, 0, 0) at n=0 — a vacuous denominator, not a perfect run."""

    def test_zero_multi_observed_makes_no_measured_claim(self):
        line = report._violation_line(_imputed(0))
        assert "UNMEASURED" in line
        assert "measured, not assumed" not in line
        assert "v̂=0.000" not in line

    def test_real_denominator_still_reports_the_measured_rate(self):
        line = report._violation_line(_imputed(8))
        assert "v̂=0.000" in line
        assert "8 multi-observed tasks" in line
        assert "measured, not assumed" in line

    def test_capability_footer_does_not_assert_100_percent_on_zero_observations(self):
        rank = config.CapabilityRank(ordered=[], evidence={})
        ann = report._capability_annotations(
            _imputed(0), rank, {}, n_bands=2, total=4, unsolvable=1
        )
        footer = " ".join(ann.notes)
        assert "UNMEASURED" in footer
        assert "measured, not assumed" not in footer

    def test_disclosure_banner_does_not_assert_100_percent_on_zero_observations(self):
        banner = report._disclosure_banner(_imputed(0), [{"n_tasks": 3}])
        assert banner is not None
        assert "UNVERIFIED" in banner
        assert "measured, not assumed" not in banner


class TestCapabilityFooterCounts:
    def test_zero_tasks_reports_zero_not_the_clamped_denominator(self):
        rank = config.CapabilityRank(ordered=[], evidence={})
        ann = report._capability_annotations(
            _imputed(2), rank, {}, n_bands=2, total=0, unsolvable=0
        )
        assert "0 task(s) across 2 capability band(s)" in ann.notes[0]
        assert "1 task(s)" not in ann.notes[0]

    def test_single_band_axis_is_disclosed(self):
        rank = config.CapabilityRank(ordered=[], evidence={})
        ann = report._capability_annotations(
            _imputed(2), rank, {}, n_bands=1, total=4, unsolvable=0
        )
        assert any("COLLAPSED to a single capability band" in n for n in ann.notes)

    def test_multi_band_axis_carries_no_collapse_note(self):
        rank = config.CapabilityRank(ordered=[], evidence={})
        ann = report._capability_annotations(
            _imputed(2), rank, {}, n_bands=3, total=4, unsolvable=0
        )
        assert not any("COLLAPSED" in n for n in ann.notes)


class TestPlotsRender:
    def test_cost_savings_renders(self, tmp_path):
        rows = [
            {"strategy": "Always-Cheap", "TotalCost": "0.03", "AvgPerf%": "80"},
            {"strategy": "Always-Frontier", "TotalCost": "0.20", "AvgPerf%": "10"},
        ]
        out = report.plot_cost_savings(rows, tmp_path)
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
