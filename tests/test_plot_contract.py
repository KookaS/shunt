"""The layout audit: it must catch the defects that actually shipped, and only those."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import Final  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from benchmark import plot_contract, plot_frame  # noqa: E402

_SPEC = plot_frame.FigureSpec(
    title="A claim about the data",
    subtitle="175 tasks · 95% Wilson bands",
    reading="x = cost, y = pass rate.",
    goal="Aim top-left.",
)
_ROWS: Final = [[str(i), "10", "726", "0.421"] for i in range(30)]
_HEADER: Final = ["n", "stale", "fired", "P(fail)"]


def _kinds(fig, band_top_px=None) -> set[str]:
    return {v.kind for v in plot_contract.audit(fig, band_top_px=band_top_px)}


def _clipping_figure():
    """An axes filling the whole canvas, so its tick labels fall off the edge."""
    # The pre-`new_figure` shape: a bare `plt.figure` with no layout engine. Constrained
    # layout CANNOT produce this — given a long label it shrinks the axes instead, which is
    # the engine working correctly and must not be asserted against.
    fig = plt.figure(figsize=(4.0, 3.0), dpi=plot_frame.DPI)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.plot([0, 1], [0, 1])
    return fig


class TestCleanFiguresPass:
    """A false positive here is worse than a miss: it blocks every legitimate figure."""

    def test_single_axes(self, tmp_path):
        plot_frame.render(tmp_path / "a.png", _SPEC, lambda ax: ax.plot([1, 2], [1, 2]) and None)

    def test_two_panels_with_a_colorbar(self, tmp_path):
        fig, axes = plot_frame.subplots(plot_frame.WIDE, 1, 2)
        image = axes[0].imshow(np.arange(80).reshape(8, 10))
        fig.colorbar(image, ax=axes[0])
        axes[1].bar(["alpha", "beta"], [1, 2])
        plot_frame.panel_label(axes[0], "held-out pass rate")
        plot_frame.save(fig, tmp_path / "b.png", _SPEC, size=plot_frame.WIDE)

    def test_thirty_row_table_bound_to_its_axes(self, tmp_path):
        size = plot_frame.table_size(30)
        fig = plot_frame.new_figure(size)
        ax = fig.subplots()
        ax.axis("off")
        table = ax.table(cellText=_ROWS, colLabels=_HEADER, bbox=[0, 0, 1, 1], cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        plot_frame.save(fig, tmp_path / "c.png", _SPEC, size=size)

    def test_axis_off_panel_reports_no_clipped_ticks(self):
        """`ax.axis('off')` leaves tick-label artists behind that are never drawn."""
        fig = plot_frame.new_figure(plot_frame.SINGLE)
        fig.subplots().axis("off")
        assert _kinds(fig) == set()

    def test_ticks_outside_the_view_limits_are_not_clipping(self):
        """A locator keeps ticks at -0.2 and 1.2 on a [0,1] axis; neither is rendered."""
        fig = plot_frame.new_figure(plot_frame.SINGLE)
        ax = fig.subplots()
        ax.plot([0, 1], [0, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        assert "outside_canvas" not in _kinds(fig)


class TestRealDefectsAreCaught:
    def test_table_scale_overflowing_its_axes(self):
        """The sweep_table bug: `Table.scale` grows past the axes and nothing clips it."""
        fig = plot_frame.new_figure(plot_frame.SINGLE)
        ax = fig.subplots()
        ax.axis("off")
        ax.table(cellText=_ROWS, colLabels=_HEADER, loc="center").scale(1.16, 1.55)
        assert {"table_overflow", "outside_canvas"} & _kinds(fig)

    def test_axes_reaching_into_the_title_band(self):
        """The other half of the sweep_table bug: the title drawn over the content."""
        fig = plot_frame.new_figure(plot_frame.SINGLE)
        ax = fig.subplots()
        ax.plot([1, 2], [1, 2])
        fig.canvas.draw()
        assert "band_intruded" in _kinds(fig, band_top_px=10.0)

    def test_tick_labels_clipped_off_the_canvas(self):
        """The heatmap bug: the leftmost cell label rendered as '.78' instead of '0.78'."""
        fig = _clipping_figure()
        assert "outside_canvas" in _kinds(fig)

    def test_assert_clean_raises_with_the_figure_name_and_every_violation(self):
        with pytest.raises(plot_contract.LayoutError, match=r"broken\.png"):
            plot_contract.assert_clean(_clipping_figure(), "broken.png")


class TestStrictModeIsWired:
    def test_save_refuses_a_broken_figure_under_strict_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SHUNT_PLOT_STRICT", "1")
        with pytest.raises(plot_contract.LayoutError):
            plot_frame.save(_clipping_figure(), tmp_path / "x.png", _SPEC)
        assert not (tmp_path / "x.png").exists()

    def test_conftest_turns_strict_mode_on_for_the_whole_suite(self):
        import os

        assert os.environ.get("SHUNT_PLOT_STRICT") == "1"
