"""The universe figures' edge cases: dot-not-bar cost, floor markers, provisional hatching."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from benchmark import plot_frame  # noqa: E402
from benchmark.routing.figures import context as ctxmod  # noqa: E402
from benchmark.routing.figures import universe as uni  # noqa: E402
from benchmark.routing.model_universe import Performance, UniverseRow  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


def _invalid_row(name: str, *, covered: int, cells: int, channel: str = "free") -> UniverseRow:
    return UniverseRow(
        canonical=name,
        identity=name,
        providers=(),
        channel=channel,
        listings=(),
        cells=cells,
        covered=covered,
        corpus=100,
        capability="not-ranked",
        triage="not-triaged",
        valid=False,
        reason="collection-only free channel",
        collection_only=True,
    )


def _ctx(out_dir: Path) -> ctxmod.RoutingContext:
    """The minimal context these two canvases read: an output dir and a provenance stamp."""
    return ctxmod.RoutingContext(
        out_dir=out_dir,
        manifest=out_dir / "figures.json",
        matrix={},
        completed={},
        imputed=None,
        tasks=[],
        rows=[],
        raw=None,
        models_by_price=[],
        banner=None,
        by_strategy={},
        digest="test",
    )


class TestCostKind:
    """The kind is the ENTITLEMENT, not the number: a paid $0 is sub-floor, never free."""

    def test_a_free_channel_is_free_whatever_the_number(self) -> None:
        assert uni._cost_kind("free", 0.0) == "free"
        assert uni._cost_kind("free", 0.5) == "free"

    def test_a_paid_sub_floor_mean_is_sub_micro_not_free(self) -> None:
        # F13: a paid mean that rounds below the log floor must not draw as a genuine $0.
        assert uni._cost_kind("paid", 1e-9) == "sub_micro"
        assert uni._cost_kind("paid", 0.0) == "sub_micro"

    def test_the_floor_itself_is_priced(self) -> None:
        assert uni._cost_kind("paid", uni._COST_FLOOR) == "priced"
        assert uni._cost_kind("paid", 0.008) == "priced"

    def test_the_two_floor_labels_name_the_entitlement(self) -> None:
        assert uni._floor_text("free") != uni._floor_text("sub_micro")
        assert "free" in uni._floor_text("free")
        assert "sub-floor" in uni._floor_text("sub_micro")
        assert "unpriced" in uni._floor_text("sub_micro")

    def test_economics_floor_keys_use_the_plotted_channel_colour(self) -> None:
        # The plotted floor marker is `facecolor=_colour(row)` — orange for a free channel, blue
        # for a paid sub-floor mean. The legend swatches used a grey stand-in for both.
        free = _invalid_row("free-model", covered=0, cells=0, channel="free")
        paid = _invalid_row("paid-model", covered=0, cells=0, channel="paid")
        rows = (free, paid)
        perf = {r.identity: Performance(r.identity, 0, 0, 0.0) for r in rows}
        fig = plot_frame.new_figure(plot_frame.SINGLE)
        try:
            ax = fig.subplots()
            uni._draw_economics_cost(ax, rows, perf)
            handles = {h.get_label(): h for h in ax.get_legend().legend_handles}
            free_fill = to_rgb(handles[uni._floor_text("free")].get_facecolor())
            paid_fill = to_rgb(handles[uni._floor_text("sub_micro")].get_facecolor())
            assert free_fill == to_rgb(uni._FREE)
            assert paid_fill == to_rgb(uni._PAID)
        finally:
            plt.close(fig)


class TestInsufficientRows:
    """A one-cell rate draws a visible stub and says so, never a 0%/100% bar."""

    def test_invalid_quality_draws_a_stub_and_labels_insufficient_n(self) -> None:
        row = _invalid_row("one-cell", covered=1, cells=1)
        perf = {row.identity: Performance(row.identity, 1, 1, 0.0)}
        fig = plot_frame.new_figure(plot_frame.SINGLE)
        try:
            ax = fig.subplots()
            uni._draw_invalid_quality(ax, (row,), perf)
            assert any("insufficient n" in t.get_text() for t in ax.texts)
            widths = [p.get_width() for p in ax.patches]
            assert widths and all(w <= uni._INSUFFICIENT_STUB + 1e-9 for w in widths)
        finally:
            plt.close(fig)

    def test_invalid_quality_draws_no_second_legend(self) -> None:
        # The duplicate panel-B key printed over the step-3.7-flash bar; panel A carries it.
        row = _invalid_row("m", covered=10, cells=10)
        perf = {row.identity: Performance(row.identity, 10, 5, 0.0)}
        fig = plot_frame.new_figure(plot_frame.SINGLE)
        try:
            ax = fig.subplots()
            uni._draw_invalid_quality(ax, (row,), perf)
            assert ax.get_legend() is None
        finally:
            plt.close(fig)

    def test_economics_quality_labels_insufficient_n(self) -> None:
        row = _invalid_row("one-cell", covered=1, cells=1)
        perf = {row.identity: Performance(row.identity, 1, 0, 0.0)}
        fig = plot_frame.new_figure(plot_frame.SINGLE)
        try:
            ax = fig.subplots()
            uni._draw_economics_quality(ax, (row,), perf)
            assert any("insufficient n" in t.get_text() for t in ax.texts)
        finally:
            plt.close(fig)

    def test_floor_marker_is_inset_from_the_spine(self) -> None:
        row = _invalid_row("m", covered=0, cells=0)
        fig = plot_frame.new_figure(plot_frame.SINGLE)
        try:
            ax = fig.subplots()
            uni._floor_marker(ax, 0.0, row, "free")
            rectangles = [p for p in ax.patches if isinstance(p, Rectangle)]
            assert rectangles
            assert rectangles[0].get_x() == uni._FLOOR_LEFT > 0.0
        finally:
            plt.close(fig)

    def test_free_row_labels_use_the_darkened_free_ink(self) -> None:
        free_row = _invalid_row("free", covered=1, cells=1, channel="free")
        paid_row = _invalid_row("paid", covered=1, cells=1, channel="paid")
        assert uni._row_label_colour(free_row) == uni._FREE_INK
        assert uni._row_label_colour(paid_row) == uni._INK


class TestCoverageTop:
    def test_all_zero_coverage_does_not_collapse_the_axis(self) -> None:
        # F12: `max(covered) if covered else 1` returned 0 for a non-empty all-zero roster.
        assert uni._coverage_top([0, 0, 0]) == 1

    def test_the_maximum_wins(self) -> None:
        assert uni._coverage_top([0, 7, 3]) == 7


class TestAllZeroCoverage:
    def test_an_all_zero_roster_keeps_an_axis_and_hatches_a_stub(self) -> None:
        # F12: with every covered count zero the old axis top was 0 and every bar was a bare
        # rule. The top must be 1 and each zero row must carry a hatched stub.
        rows = (
            _invalid_row("a", covered=0, cells=0),
            _invalid_row("b", covered=0, cells=0),
        )
        perf = {row.identity: Performance(row.identity, 0, 0, 0.0) for row in rows}
        fig = plot_frame.new_figure(plot_frame.SINGLE)
        ax = fig.subplots()
        uni._draw_invalid_coverage(ax, rows, perf, 100)
        assert ax.get_xlim() == (0.0, 1.42)
        assert any(p.get_hatch() == "////" for p in ax.patches)


class TestProvisionalKey:
    def test_the_key_names_the_threshold_only_when_a_row_is_provisional(self) -> None:
        from benchmark.routing import plot_style

        labels = [h.get_label() for h in uni._provisional_key(True)]
        assert any(
            "provisional" in label and str(plot_style.MIN_N_PROVISIONAL) in label
            for label in labels
        )
        assert all("provisional" not in h.get_label() for h in uni._provisional_key(False))


class TestCommittedProducersRenderUnderTheLayoutGate:
    def test_universe_economics_draws_without_a_layout_violation(self, tmp_path: Path) -> None:
        path = uni.render_economics(_ctx(tmp_path))
        assert path is not None and path.exists()

    def test_universe_invalid_draws_without_a_layout_violation(self, tmp_path: Path) -> None:
        path = uni.render_invalid(_ctx(tmp_path))
        assert path is not None and path.exists()
