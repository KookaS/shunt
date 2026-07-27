"""The shared figure frame: mandatory sections, runtime merge, and layout safety."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from benchmark import plot_frame  # noqa: E402

_SPEC = plot_frame.FigureSpec(
    reading="x = cost in USD, y = pass rate in %; each dot is one strategy.",
    goal="Aim top-left: highest pass rate for the least money.",
)


class TestMandatorySections:
    """READ and GOAL are the contract — a figure cannot opt out of being readable."""

    def test_blank_reading_rejected(self):
        with pytest.raises(ValueError):
            plot_frame.FigureSpec(reading="   ", goal="Aim top-left.")

    def test_blank_goal_rejected(self):
        with pytest.raises(ValueError):
            plot_frame.FigureSpec(reading="x = cost.", goal="")

    def test_minimal_spec_renders_two_blocks(self):
        labels = [label for label, _body, _color in plot_frame.footer_blocks(_SPEC)]
        assert labels == ["READ", "GOAL"]


class TestOptionalSections:
    def test_sections_render_in_fixed_order_when_present(self):
        spec = plot_frame.FigureSpec(
            reading="x = cost.",
            goal="Aim top-left.",
            definitions=(("regret", "reward lost vs the best possible choice"),),
            notes=("Scored on the completed matrix.",),
            limitations=("Only 43 tasks.",),
        )
        labels = [label for label, _body, _color in plot_frame.footer_blocks(spec)]
        assert labels == ["READ", "GOAL", "TERMS", "NOTE", "LIMITS"]

    def test_limitations_are_red(self):
        spec = plot_frame.FigureSpec(
            reading="x = cost.", goal="Aim top-left.", limitations=("Only 43 tasks.",)
        )
        colors = {label: color for label, _body, color in plot_frame.footer_blocks(spec)}
        assert colors["LIMITS"] == plot_frame.LIMIT_RED


class TestRuntimeMerge:
    """Data-derived caveats come through the callback, never as stale prose."""

    def test_runtime_annotations_append(self):
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


class TestRendering:
    def test_render_writes_a_png(self, tmp_path):
        out = plot_frame.render(
            tmp_path / "sub" / "fig.png", _SPEC, lambda ax: ax.plot([1, 2], [1, 2]) and None
        )
        assert out.exists() and out.stat().st_size > 0

    def test_render_closes_the_figure(self, tmp_path):
        plt.close("all")
        plot_frame.render(tmp_path / "fig.png", _SPEC, lambda ax: None)
        assert not plt.get_fignums()

    def test_narrow_figure_shrinks_font_but_not_below_floor(self):
        assert plot_frame._fontsize(4.0) == plot_frame._MIN_FONTSIZE
        assert plot_frame._fontsize(10.0) == plot_frame._BASE_FONTSIZE

    def test_wrap_columns_scale_with_width(self):
        narrow = plot_frame._wrap_columns(6.0, 7.5)
        wide = plot_frame._wrap_columns(14.0, 7.5)
        assert 40 <= narrow < wide

    def test_long_unbroken_body_still_wraps_to_multiple_lines(self):
        body = "word " * 200
        lines = plot_frame._wrapped_lines("READ", body.strip(), 120)
        assert len(lines) > 1
        assert all(len(line) <= 122 for line in lines)


class TestRuntimeDefinitions:
    """A term that only exists on some renders can be defined at runtime."""

    def test_runtime_definition_appears(self):
        merged = _SPEC.merged(
            plot_frame.Annotations(definitions=(("cascade", "try cheap, verify, escalate"),))
        )
        assert ("cascade", "try cheap, verify, escalate") in merged.definitions

    def test_static_definition_wins_on_conflict(self):
        base = plot_frame.FigureSpec(
            reading="x = cost.", goal="Aim top-left.", definitions=(("arm", "static meaning"),)
        )
        merged = base.merged(plot_frame.Annotations(definitions=(("arm", "runtime meaning"),)))
        assert merged.definitions == (("arm", "static meaning"),)


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
        fig, _ = plt.subplots()
        target = tmp_path / "ro"
        target.mkdir()
        target.chmod(0o500)
        try:
            with pytest.raises(OSError):
                plot_frame.save(fig, target / "fig.png", _SPEC)
            assert not plt.get_fignums()
        finally:
            target.chmod(0o700)


class TestFooterCannotStretchTheCanvas:
    """bbox_inches='tight' sizes the PNG to the widest artist — including the footer."""

    def test_long_unbreakable_token_is_broken(self):
        lines = plot_frame._wrapped_lines("READ", "x" * 400, 120)
        assert max(len(line) for line in lines) <= 122
