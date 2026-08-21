"""The shared figure frame: what reaches the canvas, what reaches the manifest."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from benchmark import plot_frame  # noqa: E402

# The watermark is deliberately NOT re-exported through `benchmark.plot_frame`: the two
# benchmark halves are measurement halves and have no business reaching it.
from shunt.inspect import plot_frame as shipped_frame  # noqa: E402

_SPEC = plot_frame.FigureSpec(
    title="Cheap-first routing does not beat always-frontier at equal quality",
    subtitle="175 tasks · total USD per strategy · 95% Wilson bands",
    reading="x = cost in USD, y = pass rate in %; each dot is one strategy.",
    goal="Aim top-left: highest pass rate for the least money.",
)


def _draw(ax) -> None:
    ax.plot([1, 2], [1, 2])
    ax.set_xlabel("cost (USD)")
    ax.set_ylabel("pass rate (%)")


class TestMandatorySections:
    """`reading` and `goal` are the docs record — a figure cannot opt out of being explained."""

    def test_blank_reading_rejected(self):
        with pytest.raises(ValueError):
            plot_frame.FigureSpec(reading="   ", goal="Aim top-left.")

    def test_blank_goal_rejected(self):
        with pytest.raises(ValueError):
            plot_frame.FigureSpec(reading="x = cost.", goal="")

    def test_over_long_title_rejected(self):
        with pytest.raises(ValueError, match="max 90"):
            plot_frame.FigureSpec(title="x" * 91, reading="x = cost.", goal="Aim top-left.")


class TestCaveat:
    """One red line, or none. A truncated caveat ships half a sentence."""

    def test_over_long_caveat_raises_rather_than_truncating(self):
        with pytest.raises(ValueError, match="max 120"):
            plot_frame.FigureSpec(caveat="x" * 121, reading="x = cost.", goal="Aim top-left.")

    def test_blank_caveat_rejected(self):
        with pytest.raises(ValueError, match="pass None"):
            plot_frame.FigureSpec(caveat="  ", reading="x = cost.", goal="Aim top-left.")

    def test_multiline_caveat_rejected(self):
        with pytest.raises(ValueError, match="single line"):
            plot_frame.FigureSpec(caveat="a\nb", reading="x = cost.", goal="Aim top-left.")

    def test_runtime_caveat_only_fills_an_empty_one(self):
        assert _SPEC.merged(plot_frame.Annotations(caveat="corpus failed")).caveat == (
            "corpus failed"
        )

    def test_static_caveat_outranks_the_runtime_one(self):
        spec = plot_frame.FigureSpec(
            caveat="imputed cells included", reading="x = cost.", goal="Aim top-left."
        )
        merged = spec.merged(plot_frame.Annotations(caveat="corpus failed"))
        assert merged.caveat == "imputed cells included"


class TestRuntimeMerge:
    """Data-derived content comes through the callback, never as stale prose."""

    def test_runtime_limitations_append(self):
        spec = _SPEC.merged(plot_frame.Annotations(limitations=("7 of 188 tasks uncovered.",)))
        assert spec.limitations == ("7 of 188 tasks uncovered.",)

    def test_duplicate_text_collapses(self):
        base = plot_frame.FigureSpec(
            reading="x = cost.", goal="Aim top-left.", notes=("single-arm data",)
        )
        merged = base.merged(plot_frame.Annotations(notes=("single-arm data",)))
        assert merged.notes == ("single-arm data",)

    def test_merge_without_extra_is_identity(self):
        assert _SPEC.merged(None) is _SPEC

    def test_subtitle_facts_append_to_the_static_subtitle(self):
        merged = _SPEC.merged(plot_frame.Annotations(subtitle_facts=("status=OK",)))
        assert merged.subtitle.endswith("· status=OK")

    def test_static_definition_wins_on_conflict(self):
        base = plot_frame.FigureSpec(
            reading="x = cost.", goal="Aim top-left.", definitions=(("arm", "static meaning"),)
        )
        merged = base.merged(plot_frame.Annotations(definitions=(("arm", "runtime meaning"),)))
        assert merged.definitions == (("arm", "static meaning"),)


class TestCanvasCarriesOnlyTheClaim:
    """READ / GOAL / TERMS / NOTE / LIMITS are records, not ink."""

    def test_the_explanatory_sections_are_never_drawn(self, tmp_path):
        spec = plot_frame.FigureSpec(
            title="A claim",
            subtitle="n=175",
            reading="THIS_MUST_NOT_BE_DRAWN",
            goal="NOR_THIS",
            definitions=(("term", "NOR_THIS_EITHER"),),
            notes=("NOR_THIS_NOTE",),
            limitations=("NOR_THIS_LIMIT",),
        )
        fig = plot_frame.new_figure(plot_frame.SINGLE)
        _draw(fig.subplots())
        plot_frame.attach_band(fig, spec)
        drawn = " ".join(t.get_text() for t in fig.texts)
        plt.close(fig)
        assert "A claim" in drawn and "n=175" in drawn
        assert "MUST_NOT" not in drawn and "NOR_THIS" not in drawn

    def test_band_grows_with_a_caveat(self):
        without = plot_frame.band_height_inches(_SPEC, 10.0)
        with_caveat = plot_frame.band_height_inches(
            plot_frame.FigureSpec(
                title=_SPEC.title,
                subtitle=_SPEC.subtitle,
                caveat="Imputed cells are included in every height on this chart.",
                reading=_SPEC.reading,
                goal=_SPEC.goal,
            ),
            10.0,
        )
        assert with_caveat > without > 0.0


class TestRendering:
    def test_render_writes_a_png(self, tmp_path):
        out = plot_frame.render(tmp_path / "sub" / "fig.png", _SPEC, _draw)
        assert out.exists() and out.stat().st_size > 0

    def test_png_pixels_are_exactly_the_named_size(self, tmp_path):
        """`bbox_inches='tight'` is gone, so the canvas is the size we asked for."""
        out = plot_frame.render(tmp_path / "fig.png", _SPEC, _draw, size=plot_frame.WIDE)
        assert Image.open(out).size == (
            round(plot_frame.WIDE.width_in * plot_frame.DPI),
            round(plot_frame.WIDE.height_in * plot_frame.DPI),
        )

    def test_render_closes_the_figure(self, tmp_path):
        plt.close("all")
        plot_frame.render(tmp_path / "fig.png", _SPEC, _draw)
        assert not plt.get_fignums()

    def test_table_size_grows_with_rows(self):
        assert table_h(4) < table_h(30) <= 13.0

    def test_every_named_size_is_registered(self):
        assert set(plot_frame.SIZES) == {"single", "single_tall", "square", "wide", "wide_tall"}


def table_h(rows: int) -> float:
    return plot_frame.table_size(rows).height_in


class TestManifest:
    """The record the canvas no longer carries has to land somewhere checkable."""

    def _prov(self, tmp_path) -> plot_frame.Provenance:
        return plot_frame.Provenance(
            generator="benchmark.routing.report",
            data_digest="deadbeef",
            manifest=tmp_path / "figures.json",
        )

    def test_save_records_the_full_spec(self, tmp_path):
        plot_frame.render(
            tmp_path / "fig.png",
            _SPEC,
            _draw,
            provenance=self._prov(tmp_path),
        )
        row = json.loads((tmp_path / "figures.json").read_text())["figures"]["fig.png"]
        assert row["reading"] == _SPEC.reading
        assert row["goal"] == _SPEC.goal
        assert row["title"] == _SPEC.title
        assert row["generator"] == "benchmark.routing.report"

    def test_upsert_leaves_other_entries_alone(self, tmp_path):
        """Eight processes write routing figures — a truncating write would lose seven."""
        prov = self._prov(tmp_path)
        plot_frame.render(tmp_path / "a.png", _SPEC, _draw, provenance=prov)
        plot_frame.render(tmp_path / "b.png", _SPEC, _draw, provenance=prov)
        figures = json.loads((tmp_path / "figures.json").read_text())["figures"]
        assert set(figures) == {"a.png", "b.png"}

    def test_no_timestamp_makes_regeneration_a_no_op_diff(self, tmp_path):
        prov = self._prov(tmp_path)
        plot_frame.render(tmp_path / "a.png", _SPEC, _draw, provenance=prov)
        first = (tmp_path / "figures.json").read_text()
        plot_frame.render(tmp_path / "a.png", _SPEC, _draw, provenance=prov)
        assert (tmp_path / "figures.json").read_text() == first

    def test_prune_drops_rows_whose_png_is_gone(self, tmp_path):
        prov = self._prov(tmp_path)
        plot_frame.render(tmp_path / "a.png", _SPEC, _draw, provenance=prov)
        plot_frame.render(tmp_path / "b.png", _SPEC, _draw, provenance=prov)
        assert plot_frame.prune(tmp_path / "figures.json", ["a.png"]) == ["b.png"]
        figures = json.loads((tmp_path / "figures.json").read_text())["figures"]
        assert set(figures) == {"a.png"}

    def test_no_manifest_written_without_provenance(self, tmp_path):
        plot_frame.render(tmp_path / "a.png", _SPEC, _draw)
        assert not (tmp_path / "figures.json").exists()


class TestFailureIsClean:
    """A driver that renders dozens of figures must not leak one on a failure."""

    def test_draw_exception_closes_the_figure(self, tmp_path):
        plt.close("all")

        def boom(_ax):
            raise RuntimeError("draw failed")

        with pytest.raises(RuntimeError):
            plot_frame.render(tmp_path / "fig.png", _SPEC, boom)
        assert not plt.get_fignums()

    def test_save_failure_closes_the_figure(self, tmp_path):
        plt.close("all")
        fig = plot_frame.new_figure(plot_frame.SINGLE)
        _draw(fig.subplots())
        target = tmp_path / "ro"
        target.mkdir()
        target.chmod(0o500)
        try:
            with pytest.raises(OSError):
                plot_frame.save(fig, target / "fig.png", _SPEC)
            assert not plt.get_fignums()
        finally:
            target.chmod(0o700)


class TestBandNeverOverflowsTheCanvas:
    """The wrap column is an estimate, and a wrong estimate runs text off the PNG."""

    _WORST = (
        ("narrow glyphs", ("x" * 9 + " ") * 30),
        (
            "real subtitle",
            "escalate_after_n=2, stale_window=10 · base 0.421 · fires 727/727 · 727/799 scored",
        ),
        (
            "capitalised",
            "MEASURED ONLY NO IMPUTED CELL ON EITHER SIDE N=87 WILSON BANDS MCNEMAR EXACT " * 4,
        ),
    )

    @pytest.mark.parametrize("label,subtitle", _WORST)
    @pytest.mark.parametrize(
        "size",
        [
            plot_frame.SINGLE,
            plot_frame.SINGLE_TALL,
            plot_frame.SQUARE,
            plot_frame.WIDE,
            plot_frame.WIDE_TALL,
            plot_frame.table_size(30),
            plot_frame.table_size(30, width_in=16.0),
        ],
        ids=lambda s: s.name,
    )
    def test_worst_case_band_stays_inside_every_size(self, size, label, subtitle):
        from benchmark import plot_contract

        spec = plot_frame.FigureSpec(
            title="A CLAIM ABOUT THE DATA THAT IS FAIRLY LONG AND CAPITALISED HERE",
            subtitle=subtitle[:260],
            # Bold, and 12% wider per glyph than the regular subtitle above it — the case a
            # single weight-blind advance constant silently overflowed.
            caveat="MEASURED ONLY - EVERY IMPUTED CELL IS PASS=TRUE SO EVERY HEIGHT IS BIASED UP",
            reading="x is cost.",
            goal="Aim top-left.",
        )
        fig = plot_frame.new_figure(size)
        fig.subplots().plot([1, 2], [1, 2])
        try:
            band_top = plot_frame.attach_band(fig, spec)
            assert plot_contract.audit(fig, band_top_px=band_top) == []
        finally:
            plt.close(fig)

    def test_bold_wraps_narrower_than_regular(self):
        """Bold is measurably wider, so it must get fewer columns at the same size."""
        assert plot_frame._GLYPH_ADVANCE_BOLD > plot_frame._GLYPH_ADVANCE
        text = "w " * 400
        regular = max(len(line) for line in plot_frame._wrap(text, 10.0, 9.0))
        bold = max(len(line) for line in plot_frame._wrap(text, 10.0, 9.0, bold=True))
        assert bold < regular

    def test_wrap_leaves_a_margin_on_both_sides(self):
        """Sizing the wrap to the FULL width was the original overflow bug."""
        columns = len(plot_frame._wrap("w " * 400, 10.0, 9.0)[0])
        assert columns < int(10.0 * 72.0 / (9.0 * plot_frame._GLYPH_ADVANCE))


# --- the watermark: applied at the one door a figure can leave by --------------------


def _pixels(path) -> set[tuple[int, int, int]]:
    return {px for px in Image.open(path).convert("RGB").getdata()}


def test_no_watermark_by_default(tmp_path) -> None:
    path = plot_frame.render(tmp_path / "plain.png", _SPEC, lambda ax: _draw(ax))
    assert path.exists()
    # The mark is the only artist that paints CAVEAT_RED into the plotting area of a figure
    # whose spec carries no caveat, so its colour is the signal.
    assert not _watermark_ink(path)


def test_every_save_inside_the_block_is_stamped(tmp_path) -> None:
    # The point of the test: the draw callback below knows nothing about the watermark and is
    # not asked about it. A family opts in once, and every figure it produces carries the mark.
    with shipped_frame.watermarked("SYNTHETIC — NOT MEASURED"):
        first = plot_frame.render(tmp_path / "a.png", _SPEC, lambda ax: _draw(ax))
        second = plot_frame.render(tmp_path / "b.png", _SPEC, lambda ax: _draw(ax))
    assert _watermark_ink(first)
    assert _watermark_ink(second)


def test_the_block_does_not_leak(tmp_path) -> None:
    with shipped_frame.watermarked("SYNTHETIC — NOT MEASURED"):
        plot_frame.render(tmp_path / "inside.png", _SPEC, lambda ax: _draw(ax))
    after = plot_frame.render(tmp_path / "after.png", _SPEC, lambda ax: _draw(ax))
    assert not _watermark_ink(after)


def test_a_none_watermark_is_a_no_op(tmp_path) -> None:
    plain = plot_frame.render(tmp_path / "plain.png", _SPEC, lambda ax: _draw(ax))
    with shipped_frame.watermarked(None):
        wrapped = plot_frame.render(tmp_path / "wrapped.png", _SPEC, lambda ax: _draw(ax))
    assert wrapped.read_bytes() == plain.read_bytes()


def _watermark_ink(path) -> bool:
    """Is the mark's own colour on the canvas — CAVEAT_RED composited onto white at its alpha?

    Derived from the module's constants rather than hardcoded, so a change to either the
    colour or the alpha moves this test with it instead of silently invalidating it.
    """
    ink = tuple(int(shipped_frame.CAVEAT_RED[i : i + 2], 16) for i in (1, 3, 5))
    alpha = shipped_frame._WATERMARK_ALPHA
    want = tuple(255.0 - alpha * (255 - c) for c in ink)
    return any(all(abs(px[i] - want[i]) <= 2 for i in range(3)) for px in _pixels(path))
