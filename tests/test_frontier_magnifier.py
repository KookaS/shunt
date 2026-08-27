"""The cost-quality plane's magnification stack: A, then A's box, then B's box."""

# The defect this guards is not a crowded label — it is a HARDCODED zoom window. Costs move on
# every re-run and every new strategy, so a written-down box eventually points at empty canvas or
# crops the crowd it exists to explain, and nothing fails. Every assertion below is about the
# levels following the data rather than a constant, and about the figure dropping a level
# HONESTLY — saying so on the record — when the data gives it nothing to magnify.

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle  # noqa: E402

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


def _context(rows: list[dict], out: Path) -> ctxmod.RoutingContext:
    return ctxmod.RoutingContext(
        out_dir=out,
        manifest=out / "figures.json",
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


def _compose(rows: list[dict]) -> tuple[plt.Figure, list[plt.Axes], tuple[str, ...]]:
    """Lay the figure out exactly as `render` does, but keep the axes open to inspect."""
    ctx = _context(rows, Path("."))
    pareto = fig._live_pareto(rows)
    hull = plot_style.upper_hull(fig._live_points(rows))
    aiq = plot_style.area_under_frontier(hull)
    figure, _size, notes = fig._compose(ctx, pareto, hull, aiq)
    return figure, list(figure.axes), notes


def _levels(notes: tuple[str, ...]) -> str:
    return next(n for n in notes if n.startswith("layout: ") and " level" in n)


def _magnified(notes: tuple[str, ...]) -> bool:
    """Did the figure spend a level on markers that overlap?"""
    return "the group whose markers overlap" in _levels(notes)


def _texts(axes: list[plt.Axes]) -> set[str]:
    return {t.get_text() for ax in axes for t in ax.texts}


def _intervals(ax: plt.Axes) -> int:
    """How many error bars this panel really draws — the key's empty handles do not count."""
    return sum(
        len(collection.get_segments())
        for container in ax.containers
        for collection in container.lines[2]
    )


def _crowd() -> list[dict]:
    """Four cascades whose markers overlap, plus rows that spread out around them."""
    return [
        _row("Always-Cheap", 1.5, 70.0),
        _row("kNN-semantic", 11.79, 77.72),
        _row("Oracle", 18.33, 96.74),
        _row("Session-Cascade", 28.71, 96.74),
        _row("kNN-difficulty-cascade", 29.01, 96.74),
        _row("Difficulty-Band-cascade", 29.92, 96.74),
        _row("kNN-semantic-cascade (within-task)", 30.44, 96.74),
        _row("kNN-semantic-cascade", 38.49, 96.74),
        _row("Always-Frontier", 96.02, 95.11),
    ]


class TestTheMagnifiedLevelFollowsTheData:
    def test_spread_out_strategies_get_no_magnified_level_at_all(self):
        # The property that makes this durable: when the data spreads out, the figure
        # degrades back to a plain plane rather than magnifying an empty region.
        rows = [
            _row("Always-Cheap", 1.5, 70.0),
            _row("kNN-semantic", 12.0, 80.0),
            _row("Session-Cascade", 30.0, 90.0),
            _row("Always-Frontier", 96.0, 96.0),
        ]
        figure, axes, notes = _compose(rows)
        assert len(axes) == 1
        assert not _magnified(notes)
        assert "ONE level" in _levels(notes)
        plt.close(figure)

    def test_a_real_crowd_earns_a_magnified_level(self):
        figure, axes, notes = _compose(_crowd())
        assert len(axes) >= 2
        assert _magnified(notes)
        plt.close(figure)

    def test_the_magnified_window_moves_with_the_costs(self):
        # Same crowd, ten times the money: the window must follow, which a written-down
        # box could not do.
        dear = [
            _row(
                str(r["strategy"]),
                float(r["TotalCost_cacheaware"]) * 10,
                float(r["AvgPerf%"]),
            )
            for r in _crowd()
        ]
        windows = []
        for rows in (_crowd(), dear):
            figure, axes, notes = _compose(rows)
            assert _magnified(notes)
            windows.append(axes[-1].get_xlim())
            plt.close(figure)
        assert windows[1][0] > windows[0][1]

    def test_every_name_is_printed_somewhere_across_the_stack(self):
        # Nothing dropped silently: a marker with no name anywhere is the one outcome this
        # figure must never ship, whichever level it ends up on.
        rows = _crowd()
        figure, axes, _notes = _compose(rows)
        drawn = _texts(axes)
        for row in rows:
            assert row["strategy"] in drawn, row["strategy"]
        plt.close(figure)


class TestTheBracketIsDrawnOnDeployableEscalatingRowsOnly:
    def test_a_blocked_within_task_cascade_is_never_bracketed(self):
        # It carries alpha columns and is still excluded: the model prices a SESSION
        # boundary handoff, and a within-task cascade has none to price.
        rows = [
            _row("Price-Cascade", 27.11, 96.74, context_cost_alpha_10=30.52),
            _row("kNN-semantic-cascade (within-task)", 30.44, 96.74, context_cost_alpha_10=34.55),
            _row("Session-Cascade", 28.71, 96.74, context_cost_alpha_10=32.81),
            _row("kNN-semantic-cascade", 38.26, 96.74, context_cost_alpha_10=42.49),
        ]
        names = [str(r["strategy"]) for r in fig._bracket_rows(rows)]
        assert names == ["Session-Cascade", "kNN-semantic-cascade"]

    def test_a_strategy_that_never_escalates_carries_no_bracket(self):
        rows = [_row("Always-Frontier", 96.0, 95.11, context_cost_alpha_10=96.0)]
        assert fig._bracket_rows(rows) == []

    def test_the_window_stretches_to_contain_the_longest_bracket(self):
        rows = [
            _row("Always-Cheap", 1.5, 70.0),
            _row("Price-Cascade", 27.11, 96.74),
            _row("Session-Cascade", 28.71, 96.74, context_cost_alpha_10=32.81),
            _row("kNN-semantic-cascade", 38.26, 96.74, context_cost_alpha_10=200.0),
            _row("Always-Frontier", 96.0, 95.11),
        ]
        assert fig._bracket_extent(rows, {"kNN-semantic-cascade"}) == 200.0


def _render(rows: list[dict], out: Path) -> dict:
    """Render the whole figure into `out` and return the manifest row it recorded."""
    assert fig.render(_context(rows, out)) is not None
    return json.loads((out / "figures.json").read_text())["figures"]["cost_quality_frontier.png"]


def _spread() -> list[dict]:
    """A top band the detail window keeps, and three rows far below it that it must exclude."""
    top = [
        _row("Oracle", 18.33, 96.74),
        _row("Price-Cascade", 27.11, 96.74),
        _row("Session-Cascade", 28.71, 96.74),
        _row("kNN-semantic-cascade", 38.49, 96.74),
        _row("Always-Frontier", 96.02, 95.11),
    ]
    return [
        *top,
        _row("kNN-semantic", 11.79, 77.72),
        _row("Always-Cheap", 1.5, 75.54),
        _row("kNN-semantic-tier", 11.53, 65.76),
    ]


class TestTheDescentIsDrawn:
    """A box on the parent and connectors to the child, at every level below the first."""

    def test_three_levels_nest_and_each_child_is_marked_on_its_parent(self):
        rows = _crowd() + [_row("kNN-semantic-tier", 11.53, 65.76)]
        figure, axes, notes = _compose(rows)
        assert len(axes) == 3, _levels(notes)
        assert "THREE levels" in _levels(notes)
        for parent, child in zip(axes, axes[1:], strict=False):
            px, cx = parent.get_xlim(), child.get_xlim()
            # A magnification is strictly inside its parent, and strictly narrower.
            assert px[0] <= cx[0] < cx[1] <= px[1]
            assert cx[1] / cx[0] < px[1] / px[0]
            boxes = [p for p in parent.patches if isinstance(p, Rectangle) and p.get_width()]
            marked = [
                b
                for b in boxes
                if abs(b.get_x() - cx[0]) < 1e-6 and abs(b.get_x() + b.get_width() - cx[1]) < 1e-6
            ]
            assert marked, "the child's window is not drawn on its parent"
            links = [a for a in parent.get_children() if isinstance(a, ConnectionPatch)]
            assert len(links) == 2, "a magnification needs both of its connectors"
        plt.close(figure)

    def test_the_panels_are_captioned_in_magnification_order(self):
        figure, axes, _notes = _compose(_crowd() + [_row("kNN-semantic-tier", 11.53, 65.76)])
        captions = [ax.get_title(loc="left") for ax in axes]
        assert captions[0].startswith("A · every strategy, full cost range")
        assert captions[1].startswith("B · magnified from A")
        assert captions[2].startswith("C · magnified from B")
        # Where the intervals live is said on the canvas, not only in the prose.
        assert "with the pass-rate and cost intervals" in captions[1]
        assert "no intervals at this scale" in captions[2]
        plt.close(figure)

    def test_only_the_panels_above_the_magnified_one_carry_intervals(self):
        figure, axes, notes = _compose(_crowd() + [_row("kNN-semantic-tier", 11.53, 65.76)])
        assert _magnified(notes)
        assert all(_intervals(ax) > 0 for ax in axes[:-1]), "an upper panel lost its intervals"
        assert _intervals(axes[-1]) == 0, "the magnified panel must not redraw the intervals"
        plt.close(figure)


class TestTheFocusAndContextSplit:
    """The full-range panel exists to carry what the detail window leaves out — and name it."""

    def test_a_row_outside_the_window_is_named_on_the_full_range_panel(self, tmp_path):
        # The one outcome this figure must never ship is a truncation that reads as full
        # coverage. Every excluded row has to be BOTH on the canvas and named on it.
        rows = _spread()
        record = _render(rows, tmp_path)
        assert record["size"] == "wide_tall"
        figure, axes, notes = _compose(rows)
        assert len(axes) >= 2
        detail = axes[1].get_xlim()
        outside = [r for r in rows if not (detail[0] <= float(r["TotalCost_cacheaware"]))]
        assert outside, "this fixture is meant to leave rows outside the detail window"
        named = {t.get_text() for t in axes[0].texts}
        for row in outside:
            assert row["strategy"] in named, row["strategy"]
        plt.close(figure)

    def test_the_window_is_derived_from_the_rows_not_written_down(self, tmp_path):
        # Ten times the money, the same shape: a hardcoded window could not follow, and this
        # is the defect the whole derivation exists to prevent. It must hold at EVERY level,
        # so the check is on each panel's own limits, not only on the note.
        cheap = _crowd() + [_row("kNN-semantic-tier", 11.53, 65.76)]
        dear = [
            _row(
                str(r["strategy"]),
                float(r["TotalCost_cacheaware"]) * 10,
                float(r["AvgPerf%"]),
            )
            for r in cheap
        ]
        limits = []
        for index, rows in enumerate((cheap, dear)):
            out = tmp_path / str(index)
            out.mkdir()
            note = next(n for n in _render(rows, out)["notes"] if n.startswith("layout: THREE"))
            figure, axes, _notes = _compose(rows)
            limits.append([ax.get_xlim() for ax in axes])
            assert "THREE levels" in note
            plt.close(figure)
        assert len(limits[0]) == len(limits[1]) == 3
        # Both DERIVED windows moved with the money — the detail window and the magnified one.
        # Panel A is the full data range and is excluded on purpose: it is not a window.
        for lean, rich in zip(limits[0][1:], limits[1][1:], strict=True):
            assert rich[0] > lean[1]

    def test_no_row_outside_the_band_means_no_detail_level(self, tmp_path):
        # Degrade honestly: with nothing for a detail panel to add, the figure is the single
        # plane it has always been, on the single-plane canvas.
        rows = [
            _row("Oracle", 18.33, 96.74),
            _row("Session-Cascade", 28.71, 96.74),
            _row("Always-Frontier", 96.02, 95.11),
        ]
        record = _render(rows, tmp_path)
        assert record["size"] == "single_tall"
        levels = next(n for n in record["notes"] if n.startswith("layout: "))
        assert "ONE level" in levels
        assert "a detail panel would redraw the whole of panel A" in levels

    def test_a_crowd_with_no_row_outside_the_band_still_earns_a_magnified_level(self, tmp_path):
        # The other degrade axis: nothing to put in a detail panel, but markers that overlap.
        rows = [
            _row("Oracle", 18.33, 96.74),
            _row("Session-Cascade", 28.71, 96.74),
            _row("kNN-difficulty-cascade", 29.01, 96.74),
            _row("Difficulty-Band-cascade", 29.92, 96.74),
            _row("Always-Frontier", 96.02, 95.11),
        ]
        record = _render(rows, tmp_path)
        levels = next(n for n in record["notes"] if n.startswith("layout: "))
        assert "TWO levels" in levels
        assert "the group whose markers overlap" in levels
        assert "a detail panel would redraw the whole of panel A" in levels

    def test_a_split_with_no_overlapping_markers_stops_at_two_levels(self, tmp_path):
        rows = [
            _row("Always-Cheap", 1.5, 70.0),
            _row("kNN-semantic", 12.0, 74.0),
            _row("Oracle", 18.33, 96.74),
            _row("Session-Cascade", 30.0, 96.0),
            _row("kNN-semantic-cascade", 55.0, 95.5),
            _row("Always-Frontier", 96.0, 95.11),
        ]
        record = _render(rows, tmp_path)
        levels = next(n for n in record["notes"] if n.startswith("layout: "))
        assert "TWO levels" in levels
        assert "nothing left to magnify" in levels
        assert "the detail window" in levels


class TestWhichRemedyACrowdEarns:
    """A level of magnification is for markers that collide; a ladder is for names that do."""

    def test_names_that_collide_over_separable_markers_are_laddered_not_magnified(self):
        # Far enough apart in cost that no two markers touch, close enough that the names
        # cannot be printed beside them. Magnifying this costs a second copy of the points
        # and buys nothing, so the names go on levels instead.
        rows = [
            _row("Always-Cheap", 1.5, 70.0),
            _row("kNN-semantic-cascade (within-task)", 24.0, 96.74),
            _row("Difficulty-Band-cascade", 33.0, 96.74),
            _row("kNN-difficulty-cascade", 45.0, 96.74),
            _row("Always-Frontier", 96.0, 95.11),
        ]
        figure, axes, notes = _compose(rows)
        assert not _magnified(notes)
        assert any("stacked on levels" in n for n in notes), notes
        drawn = _texts(axes)
        for row in rows:
            assert row["strategy"] in drawn, row["strategy"]
        plt.close(figure)

    def test_a_marker_blob_is_magnified_and_a_separable_neighbour_is_not_dragged_in(self):
        # The deepest window is the BLOB's, not the whole label cluster's: a neighbour whose
        # marker is already separable would widen the window and undo the magnification.
        rows = [
            _row("Always-Cheap", 1.5, 70.0),
            _row("Session-Cascade", 28.71, 96.74),
            _row("kNN-difficulty-cascade", 29.01, 96.74),
            _row("Difficulty-Band-cascade", 29.92, 96.74),
            _row("kNN-semantic-cascade", 38.49, 96.74),
            _row("Always-Frontier", 96.0, 95.11),
        ]
        figure, axes, notes = _compose(rows)
        assert _magnified(notes)
        assert axes[-1].get_xlim()[1] < 38.49
        drawn = _texts(axes)
        for row in rows:
            assert row["strategy"] in drawn, row["strategy"]
        plt.close(figure)
