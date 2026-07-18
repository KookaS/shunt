"""Unit tests for the shared arm-aware plotting infra (benchmark/routing/plot_style.py)."""

from __future__ import annotations

import math

from benchmark.routing import plot_style as ps


class TestWilsonInterval:
    def test_zero_n_returns_zero_zero(self):
        assert ps.wilson_interval(0, 0) == (0.0, 0.0)

    def test_known_value_p_half_n_100(self):
        # Textbook Wilson CI for 50/100 at z=1.96 is approximately (0.404, 0.596).
        lo, hi = ps.wilson_interval(50, 100)
        assert math.isclose(lo, 0.404, abs_tol=0.01)
        assert math.isclose(hi, 0.596, abs_tol=0.01)

    def test_bounds_stay_within_0_1(self):
        lo, hi = ps.wilson_interval(0, 5)
        assert 0.0 <= lo <= hi <= 1.0
        lo, hi = ps.wilson_interval(5, 5)
        assert 0.0 <= lo <= hi <= 1.0

    def test_narrower_at_large_n_same_rate(self):
        lo_small, hi_small = ps.wilson_interval(5, 10)
        lo_big, hi_big = ps.wilson_interval(500, 1000)
        assert (hi_big - lo_big) < (hi_small - lo_small)

    def test_ci_yerr_nonnegative_and_bounds(self):
        rate, lo, hi = 0.5, 0.4, 0.6
        down, up = ps.ci_yerr(rate, lo, hi)
        assert math.isclose(down, 0.1)
        assert math.isclose(up, 0.1)


class TestProvisional:
    def test_below_threshold_is_provisional(self):
        assert ps.is_provisional(3, min_n=10) is True

    def test_at_or_above_threshold_is_not(self):
        assert ps.is_provisional(10, min_n=10) is False
        assert ps.is_provisional(50, min_n=10) is False


class TestCiFooter:
    def test_states_method_and_level(self):
        footer = ps.ci_footer()
        assert "Wilson" in footer
        assert "95%" in footer


class TestModelColorMap:
    def test_assigns_distinct_hues_in_order(self):
        models = ["a", "b", "c"]
        colors = ps.model_color_map(models)
        assert colors["a"] == ps.OKABE_ITO[0]
        assert colors["b"] == ps.OKABE_ITO[1]
        assert colors["c"] == ps.OKABE_ITO[2]
        assert len(set(colors.values())) == 3

    def test_stable_across_filtered_subsets(self):
        full_order = ["a", "b", "c", "d"]
        colors = ps.model_color_map(full_order)
        # A model's color must not change when a caller filters to a subset —
        # simulated here by re-using the SAME map for a filtered view.
        subset_colors = {m: colors[m] for m in ["b", "d"]}
        assert subset_colors["b"] == colors["b"]
        assert subset_colors["d"] == colors["d"]

    def test_cycles_past_eight_models(self):
        models = [f"m{i}" for i in range(10)]
        colors = ps.model_color_map(models)
        assert colors["m0"] == colors["m8"]


class TestArmMarkerSize:
    def test_single_arm_model_uses_base_size(self):
        assert ps.arm_marker_size(0, max_rank=0) == ps._ARM_BASE_SIZE

    def test_size_increases_with_rank(self):
        sizes = [ps.arm_marker_size(r, max_rank=2) for r in range(3)]
        assert sizes == sorted(sizes)
        assert sizes[0] < sizes[-1]

    def test_legend_values_span_all_ranks(self):
        values = ps.arm_size_legend_values(2)
        assert [r for r, _ in values] == [0, 1, 2]


class TestParetoAndFrontier:
    def test_pareto_prune_removes_dominated(self):
        # (2, 50) is dominated by (1, 60): cheaper AND better.
        points = [(1.0, 60.0), (2.0, 50.0), (3.0, 90.0)]
        kept = ps.pareto_prune(points)
        assert (2.0, 50.0) not in kept
        assert (1.0, 60.0) in kept
        assert (3.0, 90.0) in kept

    def test_upper_hull_excludes_concave_point(self):
        # (2,1) sits below the chord from (1,2) to (3,3) -> excluded from the hull.
        points = [(0.0, 0.0), (1.0, 2.0), (2.0, 1.0), (3.0, 3.0)]
        hull = ps.upper_hull(points)
        assert hull == [(0.0, 0.0), (1.0, 2.0), (3.0, 3.0)]

    def test_upper_hull_handles_small_input(self):
        assert ps.upper_hull([]) == []
        assert ps.upper_hull([(1.0, 2.0)]) == [(1.0, 2.0)]

    def test_area_under_frontier_rectangle_case(self):
        # A flat frontier at 100% pass from cost 0 to 2 -> full rectangle -> AIQ = 1.0.
        hull = [(0.0, 100.0), (2.0, 100.0)]
        assert math.isclose(ps.area_under_frontier(hull), 1.0)

    def test_area_under_frontier_half_rectangle(self):
        # Triangle from (0,0) to (2,100): half the bounding rectangle -> AIQ = 0.5.
        hull = [(0.0, 0.0), (2.0, 100.0)]
        assert math.isclose(ps.area_under_frontier(hull), 0.5)

    def test_area_under_frontier_empty(self):
        assert ps.area_under_frontier([]) == 0.0


class TestArmColumnsAndStats:
    def _raw(self) -> ps.RawResults:
        return {
            "t1": {
                "m1": {"none": {"pass": True, "cost": 0.01}, "high": {"pass": True, "cost": 0.05}},
                "m2": {"max": {"pass": False, "cost": 0.2}},
            },
            "t2": {
                "m1": {"none": {"pass": False, "cost": 0.01}},
            },
        }

    def test_arm_columns_lists_every_pair(self):
        cols = ps.arm_columns(self._raw())
        assert set(cols) == {("m1", "none"), ("m1", "high"), ("m2", "max")}

    def test_arm_stats_aggregates_across_tasks(self):
        stats = ps.arm_stats(self._raw(), "m1", "none")
        assert stats.n == 2
        assert stats.passes == 1
        assert stats.pass_rate == 0.5
        assert math.isclose(stats.total_cost, 0.02)
        assert math.isclose(stats.avg_cost, 0.01)

    def test_arm_stats_missing_column_is_zero(self):
        stats = ps.arm_stats(self._raw(), "nope", "none")
        assert stats.n == 0
        assert stats.pass_rate == 0.0
        assert stats.wilson == (0.0, 0.0)

    def test_provisional_flag_on_small_n(self):
        stats = ps.arm_stats(self._raw(), "m2", "max")
        assert stats.n == 1
        assert stats.provisional is True

    def test_is_single_arm_true_for_default_only_data(self):
        raw = {"t1": {"m1": {"default": {"pass": True, "cost": 0.01}}}}
        assert ps.is_single_arm(raw) is True

    def test_is_single_arm_false_when_any_model_has_two_arms(self):
        assert ps.is_single_arm(self._raw()) is False


class TestLabelPointsWithLeaders:
    def test_empty_points_is_a_noop(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ps.label_points_with_leaders(ax, [])
        assert len(ax.texts) == 0
        plt.close(fig)

    def test_one_annotation_per_point_even_with_duplicates(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        # Two identical (x, y) points (the Oracle == Always-Cheap collision case)
        # must still each get their own, distinctly-positioned label.
        points = [(1.0, 80.0, "Oracle"), (1.0, 80.0, "Always-Cheap"), (2.0, 50.0, "kNN")]
        ps.label_points_with_leaders(ax, points)
        assert len(ax.texts) == 3
        names = {t.get_text() for t in ax.texts}
        assert names == {"Oracle", "Always-Cheap", "kNN"}
        plt.close(fig)
