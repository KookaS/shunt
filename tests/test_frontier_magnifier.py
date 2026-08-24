"""The cost-quality plane's magnified panel: it must appear on a crowd and vanish without one."""

# The defect this guards is not a crowded label — it is a HARDCODED zoom window. Costs move on
# every re-run and every new strategy, so a written-down box eventually points at empty canvas or
# crops the crowd it exists to explain, and nothing fails. Every assertion below is about the
# panel following the data rather than a constant.

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from benchmark import plot_frame  # noqa: E402
from benchmark.routing import plot_style  # noqa: E402
from benchmark.routing.figures import context as ctxmod  # noqa: E402
from benchmark.routing.figures import cost_quality_frontier as fig  # noqa: E402


def _row(name: str, cost: float, perf: float, **extra: object) -> dict:
    row = {
        "strategy": name,
        "n_tasks": "100",
        "n_pass": str(int(perf)),
        "AvgPerf%": f"{perf}",
        "TotalCost": f"{cost}",
        "TotalCost_cacheaware": f"{cost}",
        "TotalCost_ci_lower": f"{cost * 0.9}",
        "TotalCost_ci_upper": f"{cost * 1.1}",
        "context_cost_alpha_01": "0",
        "context_cost_alpha_03": "0",
        "context_cost_alpha_10": "0",
        "context_cost_n": "0",
    }
    row.update({k: str(v) for k, v in extra.items()})
    return row


def _draw(rows: list[dict]) -> tuple[plt.Figure, plt.Axes, tuple[str, ...]]:
    ctx = ctxmod.RoutingContext(
        out_dir=Path("."),
        manifest=Path("figures.json"),
        matrix={},
        completed={},
        imputed=None,
        tasks=[],
        rows=rows,
        raw=None,
        models_by_price=[],
        banner=None,
        by_strategy={},
        digest="test",
    )
    figure = plot_frame.new_figure(plot_frame.SINGLE_TALL)
    ax = figure.subplots()
    pareto = fig._live_pareto(rows)
    hull = plot_style.upper_hull(fig._live_points(rows))
    notes = fig._draw(ax, ctx, pareto, hull)
    return figure, ax, notes


class TestThePanelFollowsTheData:
    def test_spread_out_strategies_get_no_panel_at_all(self):
        # The property that makes this durable: when the data spreads out, the figure
        # degrades back to a plain plane rather than magnifying an empty region.
        rows = [
            _row("Always-Cheap", 1.5, 70.0),
            _row("kNN", 12.0, 80.0),
            _row("Session-Cascade", 30.0, 90.0),
            _row("Always-Frontier", 96.0, 96.0),
        ]
        figure, ax, notes = _draw(rows)
        # `inset_axes` attaches to the PARENT axes, not to `figure.axes`.
        assert ax.child_axes == []
        assert notes == ()
        plt.close(figure)

    def test_a_real_crowd_earns_exactly_one_panel(self):
        rows = [
            _row("Always-Cheap", 1.5, 70.0),
            _row("Price-Cascade", 27.11, 96.74),
            _row("Session-Cascade", 28.71, 96.74),
            _row("kNN-cascade (within-task)", 30.44, 96.74),
            _row("kNN-cascade", 38.26, 96.74),
            _row("Always-Frontier", 96.0, 95.11),
        ]
        figure, ax, notes = _draw(rows)
        assert len(ax.child_axes) == 1
        assert any(n.startswith("layout:") for n in notes)
        plt.close(figure)

    def test_the_panel_window_moves_with_the_costs(self):
        # Same crowd, ten times the money: the window must follow, which a written-down
        # box could not do.
        cheap = [
            _row("Always-Cheap", 1.5, 70.0),
            _row("Price-Cascade", 27.11, 96.74),
            _row("Session-Cascade", 28.71, 96.74),
            _row("kNN-cascade", 38.26, 96.74),
            _row("Always-Frontier", 96.0, 95.11),
        ]
        dear = [
            _row(r["strategy"], float(r["TotalCost_cacheaware"]) * 10, float(r["AvgPerf%"]))
            for r in cheap
        ]
        windows = []
        for rows in (cheap, dear):
            figure, ax, _notes = _draw(rows)
            windows.append(ax.child_axes[0].get_xlim())
            plt.close(figure)
        assert windows[1][0] > windows[0][1]

    def test_every_crowded_name_is_labelled_once_across_plane_and_panel(self):
        rows = [
            _row("Always-Cheap", 1.5, 70.0),
            _row("Price-Cascade", 27.11, 96.74),
            _row("Session-Cascade", 28.71, 96.74),
            _row("kNN-cascade (within-task)", 30.44, 96.74),
            _row("kNN-cascade", 38.26, 96.74),
            _row("Always-Frontier", 96.0, 95.11),
        ]
        figure, ax, _notes = _draw(rows)
        drawn = {t.get_text() for a in [ax, *ax.child_axes] for t in a.texts}
        for row in rows:
            assert row["strategy"] in drawn, row["strategy"]
        plt.close(figure)


class TestTheBracketIsDrawnOnDeployableEscalatingRowsOnly:
    def test_a_blocked_within_task_cascade_is_never_bracketed(self):
        # It carries alpha columns and is still excluded: the model prices a SESSION
        # boundary handoff, and a within-task cascade has none to price.
        rows = [
            _row("Price-Cascade", 27.11, 96.74, context_cost_alpha_10=30.52),
            _row("kNN-cascade (within-task)", 30.44, 96.74, context_cost_alpha_10=34.55),
            _row("Session-Cascade", 28.71, 96.74, context_cost_alpha_10=32.81),
            _row("kNN-cascade", 38.26, 96.74, context_cost_alpha_10=42.49),
        ]
        names = [str(r["strategy"]) for r in fig._bracket_rows(rows)]
        assert names == ["Session-Cascade", "kNN-cascade"]

    def test_a_strategy_that_never_escalates_carries_no_bracket(self):
        rows = [_row("Always-Frontier", 96.0, 95.11, context_cost_alpha_10=96.0)]
        assert fig._bracket_rows(rows) == []

    def test_the_window_stretches_to_contain_the_longest_bracket(self):
        rows = [
            _row("Always-Cheap", 1.5, 70.0),
            _row("Price-Cascade", 27.11, 96.74),
            _row("Session-Cascade", 28.71, 96.74, context_cost_alpha_10=32.81),
            _row("kNN-cascade", 38.26, 96.74, context_cost_alpha_10=200.0),
            _row("Always-Frontier", 96.0, 95.11),
        ]
        assert fig._bracket_extent(rows, {"kNN-cascade"}) == 200.0
