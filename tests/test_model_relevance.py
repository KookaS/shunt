"""The model-relevance figure and the canonical catalogue: status taxonomy, one census, gate.

The figure answers the owner's scale/relevance question — which models clear the bar and why
the rest fall short — without hue-per-model. These tests pin the parts that can silently lie:
the status precedence, the funnel/status counts being READ from the census rather than
hardcoded, and the canvas clearing the layout gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from benchmark import plot_frame  # noqa: E402
from benchmark.routing import model_catalog, model_validity  # noqa: E402
from benchmark.routing.figures import context as ctxmod  # noqa: E402
from benchmark.routing.figures import model_relevance  # noqa: E402
from benchmark.routing.model_catalog import (  # noqa: E402
    FREE_ONLY,
    INSUFFICIENT,
    NO_EVIDENCE,
    VALID,
)
from benchmark.routing.model_validity import FREE, PAID, ModelValidity  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


def _row(
    model: str,
    *,
    channel: str = PAID,
    cells: int = 25,
    triage: str = "KEEP",
    capability: str = "measured",
    live: bool = True,
) -> ModelValidity:
    valid = (
        live and triage == "KEEP" and capability == "measured" and channel == PAID and cells >= 25
    )
    return ModelValidity(
        model=model,
        channel=channel,
        providers=("p",),
        listings=(model,),
        collection_only=False,
        live=live,
        enabled=live,
        triage=triage,
        capability=capability,
        cells=cells,
        covered=cells,
        corpus=100,
        valid=valid,
        reason="inference-valid" if valid else "not in the live pool",
    )


class TestStatusPrecedence:
    """One row, one status; the precedence is documented so it cannot drift silently."""

    def test_a_valid_row_is_valid(self) -> None:
        assert model_catalog.status_of(_row("m")) == VALID

    def test_no_measured_cell_is_no_evidence(self) -> None:
        assert model_catalog.status_of(_row("m", cells=0)) == NO_EVIDENCE

    def test_a_free_channel_is_free_only_even_with_many_cells(self) -> None:
        assert model_catalog.status_of(_row("m", channel=FREE, cells=99)) == FREE_ONLY

    def test_a_paid_identity_below_the_floor_is_insufficient(self) -> None:
        assert model_catalog.status_of(_row("m", cells=1)) == INSUFFICIENT

    def test_a_covered_paid_identity_names_its_first_failure(self) -> None:
        assert model_catalog.status_of(_row("m", live=False)) == "invalid:live"
        assert model_catalog.status_of(_row("m", triage="DROP")) == "invalid:triage"
        assert model_catalog.status_of(_row("m", capability="not-ranked")) == "invalid:capability"


class TestFunnelIsReadFromTheCensus:
    """The funnel stages come from the census, so a figure cannot quote a stale count."""

    def _data(self) -> model_relevance._Data:
        rows = model_validity.validity_census()
        return model_relevance._Data(
            rows=tuple(rows), perf={}, floor=model_validity.cell_floor(), corpus=rows[0].corpus
        )

    def test_named_matches_the_roster(self) -> None:
        data = self._data()
        assert data.named == len(data.rows)

    def test_stage_counts_match_the_rows(self) -> None:
        data = self._data()
        counts = dict(data.funnel)
        assert counts["named (canonical identities)"] == len(data.rows)
        assert counts["evidenced (≥1 measured cell)"] == sum(
            1 for r in data.rows if model_validity.is_evidenced(r)
        )
        assert counts["paid channel (of all named)"] == sum(
            1 for r in data.rows if r.channel == PAID
        )
        assert counts["inference-valid"] == sum(1 for r in data.rows if r.valid)

    def test_near_miss_are_evidenced_at_or_above_the_floor_and_invalid(self) -> None:
        data = self._data()
        for row in data.near_miss():
            assert model_validity.is_evidenced(row)
            assert not row.valid
            assert row.cells >= data.floor

    def test_the_paid_milestone_is_not_the_paid_subset_of_evidenced(self) -> None:
        # The paid-channel bar counts every named identity; the paid subset of the evidenced
        # set is smaller, and the canvas must be able to say so.
        data = self._data()
        assert data.paid_evidenced == sum(
            1 for r in data.rows if model_validity.is_evidenced(r) and r.channel == PAID
        )
        assert data.paid_evidenced <= len(data.evidenced)


class TestScatterEncoding:
    """The legend is a legend of the data: same glyphs, same fills, one shape per status."""

    def test_every_status_has_a_distinct_marker_form(self) -> None:
        forms = list(model_relevance._MARKERS.values())
        assert len(forms) == len(set(forms))

    def test_free_only_does_not_borrow_another_status_glyph(self) -> None:
        # The free-only channel keeps the diamond; no other status may share it, and
        # insufficient evidence no longer shares the down-triangle with coverage-below-K.
        assert (
            list(model_relevance._MARKERS.values()).count(model_relevance._MARKERS[FREE_ONLY]) == 1
        )
        assert (
            model_relevance._MARKERS[INSUFFICIENT] != model_relevance._MARKERS["invalid:coverage"]
        )

    def test_no_channel_key_reuses_a_status_form(self) -> None:
        # Hue is the channel and form is the status, so a channel key must not borrow a status
        # form. The free channel IS the free-only diamond (its one plotted form, merged), so the
        # only legal overlap is that documented pair.
        overlaps = set(model_relevance._CHANNEL_MARKERS.values()) & set(
            model_relevance._MARKERS.values()
        )
        assert overlaps == {model_relevance._MARKERS[FREE_ONLY]}

    def test_every_legend_swatch_is_a_unique_marker_fill_pair(self) -> None:
        # The bug this pins: the paid channel key (`o`, blue) and the `invalid:live` status
        # (`o`, blue) were the same swatch naming two different things.
        data = model_relevance._Data(
            rows=(
                _row("valid-model"),
                _row("not-live", live=False),
                _row("free-model", channel=FREE),
            ),
            perf={},
            floor=25,
            corpus=100,
        )
        pairs = [
            (h.get_marker(), to_rgb(h.get_markerfacecolor()))
            for h in model_relevance._legend_handles(data)
        ]
        assert len(pairs) == len(set(pairs))

    def test_legend_fills_match_the_plotted_channel_colour(self) -> None:
        data = model_relevance._Data(
            rows=(_row("valid-model"), _row("free-model", channel=FREE)),
            perf={},
            floor=25,
            corpus=100,
        )
        handles = {h.get_label(): h for h in model_relevance._legend_handles(data)}
        free_swatch = to_rgb(handles["free channel (free-only)"].get_markerfacecolor())
        paid_swatch = to_rgb(handles["inference-valid"].get_markerfacecolor())
        assert free_swatch == to_rgb(model_relevance._FREE)
        assert paid_swatch == to_rgb(model_relevance._PAID)
        assert free_swatch != paid_swatch

    def test_the_free_key_draws_the_exact_free_glyph(self) -> None:
        # The free channel's only plotted form IS the free-only diamond, so the key must not
        # borrow a circle no free row is ever drawn with.
        data = model_relevance._Data(
            rows=(_row("free-model", channel=FREE),), perf={}, floor=25, corpus=100
        )
        handles = {h.get_label(): h for h in model_relevance._legend_handles(data)}
        free_glyph = handles["free channel (free-only)"].get_marker()
        assert free_glyph == model_relevance._MARKERS[FREE_ONLY]

    def test_the_valid_key_prints_exactly_one_star(self) -> None:
        # A star marker beside a label that ALSO begins with a star read as two glyphs.
        data = model_relevance._Data(rows=(_row("valid-model"),), perf={}, floor=25, corpus=100)
        handles = {h.get_label(): h for h in model_relevance._legend_handles(data)}
        assert handles["inference-valid"].get_marker() == "*"
        assert "★" not in handles["inference-valid"].get_label()

    def test_goal_names_the_derived_starred_count(self) -> None:
        # The goal used to say "the four starred points" — a census count typed into prose.
        assert "the 4 starred points" in model_relevance._goal_text(4)
        assert "four" not in model_relevance._goal_text(4)

    def test_coincident_anchors_collide_and_separated_ones_do_not(self) -> None:
        # The pile of free n=1/0% identities and the two n=2/100% points must be nudged; a
        # point a decade away on x or a few points away on y must not be.
        assert model_relevance._near_anchor((2.0, 100.0), (2.0, 100.0))
        assert model_relevance._near_anchor((10.0, 80.0), (10.0, 80.0))
        assert not model_relevance._near_anchor((10.0, 80.0), (20.0, 80.0))
        assert not model_relevance._near_anchor((10.0, 0.0), (10.0, 80.0))

    def test_per_criterion_axis_states_non_exclusive(self) -> None:
        data = model_relevance._Data(rows=(_row("a"),), perf={}, floor=25, corpus=100)
        fig = plot_frame.new_figure(plot_frame.SINGLE)
        try:
            ax = fig.subplots()
            model_relevance._draw_exclusions(ax, data)
            assert "non-exclusive" in ax.get_xlabel()
        finally:
            plt.close(fig)


class TestCatalogueAndRender:
    """The CSV and the canvas read the same census and both clear the layout gate."""

    def test_catalogue_has_one_row_per_named_identity(self) -> None:
        rows = model_catalog.catalog_rows()
        census = model_validity.validity_census()
        assert [r.model for r in rows] == [r.model for r in census]

    def test_catalogue_wilson_brackets_the_measured_rate(self) -> None:
        for row in model_catalog.catalog_rows():
            assert row.wilson_lo <= row.pass_rate + 1e-9
            assert row.pass_rate <= row.wilson_hi + 1e-9

    def test_render_draws_without_a_layout_violation(self, tmp_path: Path) -> None:
        validity = model_validity.validity_census()
        ctx = ctxmod.RoutingContext(
            out_dir=tmp_path,
            manifest=tmp_path / "figures.json",
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
            validity=validity,
        )
        path = model_relevance.render(ctx)
        assert path is not None and path.exists()
