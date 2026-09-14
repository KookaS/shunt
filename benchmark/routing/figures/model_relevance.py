"""model_relevance.png — which models clear the bar, and why the rest fall short.

The owner's scale/relevance question — *which* models are measured to resolve tasks, and *why*
the others are excluded — answers better on two axes than as a roster. Panel A places every
EVIDENCED canonical weights identity by its measured default-arm cell count (x) against its
measured pass rate with a 95% Wilson interval (y), so the scale of the evidence and the quality
it supports are read together. A vertical rule marks the K-cell floor: points to its left are
below the evidence bar whatever their rate. Panel B is the validity funnel — named → evidenced
→ paid channel → inference-valid — and a per-criterion exclusion count for every identity that
falls out.

ENCODINGS ARE NOT HUE-PER-MODEL. Hue is the channel (paid/free), because the channel is a
serving fact a reader must not have to decode from a name; the marker FORM carries the
validity status (inference-valid, each first-failing criterion, or insufficient evidence).
Direct labels name the inference-valid models, the near-misses and the thin sub-floor points —
every evidenced identity — and any label the placement ladder cannot seat without overprinting
is routed to the notes rather than drawn on a neighbour.

The census, the predicate, the K floor and the first-failing reasons all come from
`benchmark.routing.model_validity`; the measured pass rate and mean cost from
`benchmark.routing.model_universe`. Nothing here re-derives a second predicate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, cast

from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import model_universe, model_validity, plot_style
from benchmark.routing.figures import context as ctxmod
from benchmark.routing.model_catalog import (
    FREE_ONLY,
    INSUFFICIENT,
    status_of,
)
from benchmark.routing.model_universe import Performance, display_name
from benchmark.routing.model_validity import ModelValidity
from shunt.inspect import model_grid as grid_drawer

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.transforms import Transform

_PAID: Final[str] = "#0072B2"
_FREE: Final[str] = "#E69F00"
_INK: Final[str] = "#1a1a1a"
_RULE: Final[str] = "#C62828"
_GRID: Final[str] = "#EEEEEE"
# A sub-floor point below this measured rate is not worth a direct label; its name is in the
# notes. A wall of n=1 0% labels under the axis was the alternative.
_LABEL_RATE_MIN: Final[float] = 50.0

# Two markers whose cell counts and rates are this close would render as ONE glyph, so the
# later one is nudged in display space and tied back to its true coordinate by a leader. The x
# half is a RATIO because the axis is logarithmic; the y half is percentage points. Sized to
# the glyph, not the label: the pile of six free n=1/0% identities at one cell is the case
# this exists for, and its members must read as six marks rather than one.
_NEAR_POINTS: Final[float] = 3.0
_NEAR_DECADES: Final[float] = 0.03
_NUDGE_POINTS: Final[float] = 15.0
# A 0% marker centred on the bottom spine was half-buried in it. The axis floor sits a few
# points below the 0 tick so every measured zero is drawn whole; no tick is placed there.
_Y_FLOOR: Final[float] = -5.0

# THE ONE "evidenced" definition, stated on the canvas because the funnel and the scatter both
# count through it: at least one measured default-arm cell in either channel. The census roster
# is wider (live slots and collection-only listings with no committed measurement).
_EVIDENCED_TERM: Final[tuple[str, str]] = (
    "evidenced",
    "a canonical identity with at least one measured default-arm cell in either channel "
    "(paid or free); a named identity with no measured cell is a coverage gap, not a weak model",
)

# The marker forms, one per status the taxonomy emits. NO hue-per-model: hue is the channel.
# Every status gets a DISTINCT form: `invalid:coverage` and `insufficient` used to share the
# down-triangle, and the free-only channel's diamond read as the free channel's own key.
_MARKERS: Final[dict[str, str]] = {
    "valid": "*",
    "invalid:live": "X",
    "invalid:triage": "s",
    "invalid:capability": "^",
    "invalid:coverage": "v",
    INSUFFICIENT: "P",
    FREE_ONLY: "D",
}
# The hue keys, separate from the status forms: the paid channel is a neutral filled circle and
# the free channel IS the free-only diamond (every free row is collection-only, so the status
# form and the channel key are the one glyph, merged rather than repeated). A channel key must
# never borrow another status form — the paid circle used to be `o`, the same glyph as
# `invalid:live`, so the key named two different things with one swatch.
_CHANNEL_MARKERS: Final[dict[str, str]] = {
    "paid": "o",
    "free": _MARKERS[FREE_ONLY],
}
_MARKER_LABELS: Final[dict[str, str]] = {
    "valid": "inference-valid",
    "invalid:live": "not in live pool",
    "invalid:triage": "triage not pass",
    "invalid:capability": "capability is a price prior",
    "invalid:coverage": "coverage below K",
    INSUFFICIENT: "insufficient (<K cells)",
    FREE_ONLY: "free-only channel",
}

# The per-criterion exclusion legend, in `ModelValidity.criteria` order.
_EXCLUSION_LABELS: Final[tuple[str, ...]] = (
    "not in live pool",
    "triage not pass",
    "capability is a price prior",
    "coverage below K cells",
    "no paid channel",
)


@dataclass(frozen=True)
class _Data:
    """The census and the measured outcomes the two panels share, read once per render."""

    rows: tuple[ModelValidity, ...]
    perf: dict[str, Performance]
    floor: int
    corpus: int

    @property
    def evidenced(self) -> tuple[ModelValidity, ...]:
        return tuple(r for r in self.rows if model_validity.is_evidenced(r))

    @property
    def named(self) -> int:
        return len(self.rows)

    @property
    def valid(self) -> int:
        return sum(1 for r in self.rows if r.valid)

    @property
    def paid(self) -> int:
        return sum(1 for r in self.rows if r.channel == model_validity.PAID)

    @property
    def paid_evidenced(self) -> int:
        """The paid subset of the evidenced set — the funnel milestone is NOT this count."""
        return sum(1 for r in self.evidenced if r.channel == model_validity.PAID)

    @property
    def free(self) -> int:
        return sum(1 for r in self.rows if r.channel == model_validity.FREE)

    @property
    def funnel(self) -> tuple[tuple[str, int], ...]:
        """The validity milestones: named → evidenced → paid channel → inference-valid.

        These are MILESTONES, not a nested filter chain: the paid-channel count is over every
        named identity, so it is not the paid subset of the evidenced ones (which is smaller).
        The per-criterion panel and the notes carry the intersection.
        """
        return (
            ("named (canonical identities)", self.named),
            ("evidenced (≥1 measured cell)", len(self.evidenced)),
            ("paid channel (of all named)", self.paid),
            ("inference-valid", self.valid),
        )

    def exclusions(self) -> tuple[tuple[str, int], ...]:
        """Per-criterion counts over every named identity (a criterion can fail several ways)."""
        return tuple(
            (label, sum(1 for r in self.rows if not r.criteria[index]))
            for index, label in enumerate(_EXCLUSION_LABELS)
        )

    def near_miss(self) -> tuple[ModelValidity, ...]:
        """Evidenced identities at or above the K floor that are not inference-valid."""
        return tuple(r for r in self.evidenced if not r.valid and r.cells >= self.floor)


def _load(ctx: ctxmod.RoutingContext) -> _Data | None:
    rows = ctx.validity if ctx.validity is not None else model_validity.validity_census()
    if not rows:
        return None
    return _Data(
        rows=tuple(rows),
        perf=model_universe.performance(),
        floor=model_validity.cell_floor(),
        corpus=rows[0].corpus,
    )


def _colour(row: ModelValidity) -> str:
    return _FREE if row.channel == model_validity.FREE else _PAID


def _marker(row: ModelValidity) -> str:
    """The marker form for a row's status; a no-evidence row is not drawn on the scatter."""
    return _MARKERS.get(status_of(row), "x")


def _near_anchor(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Would two markers at these data anchors render on top of each other?"""
    if abs(a[1] - b[1]) >= _NEAR_POINTS:
        return False
    # A zero cell count cannot be placed on the log axis; treat it as the same column.
    if a[0] <= 0 or b[0] <= 0:
        return True
    return abs(math.log10(a[0] / b[0])) < _NEAR_DECADES


def _place_label(
    ax: Axes,
    row: ModelValidity,
    rate: float,
    hi: float,
    placed: list[tuple[tuple[float, float], float]],
    ceiling: float,
    overflow: list[str],
    transform: Transform | None = None,
) -> None:
    """One direct label, or the row's name routed to the notes when the ladder is full."""
    anchor = (float(row.cells), rate)
    offset = grid_drawer.label_offset(anchor, placed, forced_below=hi >= ceiling)
    if offset is None:
        overflow.append(display_name(row.model))
        return
    placed.append((anchor, offset))
    moved = offset != grid_drawer._LABEL_OFFSETS[0]
    ax.annotate(
        display_name(row.model),
        anchor,
        textcoords="offset points",
        xycoords=transform if transform is not None else ax.transData,
        xytext=(0, offset),
        ha="center",
        va="top" if offset < 0 else "baseline",
        fontsize=6.4,
        color=_INK,
        arrowprops=(
            {"arrowstyle": "-", "color": _RULE, "lw": 0.6, "shrinkA": 1.0, "shrinkB": 3.0}
            if moved
            else None
        ),
    )


def _wilson_whisker(
    ax: Axes, x: float, rate: float, passes: int, n: int, transform: Transform | None = None
) -> None:
    """A capped, symmetric Wilson whisker around the point.

    The Wilson bounds are asymmetric near 0 and 1. Drawn as the true interval the lower arm
    collapsed onto the axis and the whisker read as a bar growing from zero; the whisker is
    therefore symmetric about the estimate with half-width = the larger Wilson arm, capped at
    both ends, so a wide interval reads as uncertainty rather than as a filled bar.
    """
    lo, hi = plot_style.wilson_interval(passes, n)
    lo_p, hi_p = lo * 100.0, hi * 100.0
    half = max(rate - lo_p, hi_p - rate)
    ax.errorbar(
        [x],
        [rate],
        yerr=[[min(half, rate)], [min(half, 100.0 - rate)]],
        fmt="none",
        ecolor="#9a9a9a",
        elinewidth=0.9,
        capsize=2.6,
        capthick=0.9,
        zorder=2,
        transform=transform,
    )


def _draw_scatter(ax: Axes, data: _Data) -> tuple[str, ...]:
    """The measured-cells vs pass-rate plane, with direct labels for the relevant models."""
    rows = data.evidenced
    near_miss = {r.model for r in data.near_miss()}
    placed: list[tuple[tuple[float, float], float]] = []
    overflow: list[str] = []
    # The true anchors already drawn, so a later point at the same coordinate is nudged rather
    # than printed over the earlier glyph (the model_grid drawer's contract). The whisker is
    # drawn once per coincidence: the colliding rows share it, and repeating it would stack six
    # identical bars on top of each other.
    drawn: list[tuple[float, float]] = []
    for row in rows:
        stats = data.perf.get(row.model)
        if stats is None or stats.n <= 0:
            continue
        rate = stats.pass_rate * 100.0
        anchor = (float(row.cells), rate)
        collisions = sum(1 for other in drawn if _near_anchor(anchor, other))
        transform = (
            offset_copy(
                ax.transData,
                fig=cast("Figure", ax.figure),
                x=collisions * _NUDGE_POINTS,
                units="points",
            )
            if collisions
            else None
        )
        lo, hi = plot_style.wilson_interval(stats.passes, stats.n)
        if not collisions:
            _wilson_whisker(ax, float(row.cells), rate, stats.passes, stats.n)
        ax.scatter(
            [row.cells],
            [rate],
            s=210 if row.valid else 62,
            marker=_marker(row),
            c=_colour(row),
            edgecolors="white",
            linewidths=0.7,
            zorder=4 if row.valid else 3,
            transform=transform,
        )
        if transform is not None:
            # A short leader ties the displaced glyph back to its true coordinate, so the
            # offset is disclosed rather than silently moving the measured point.
            ax.annotate(
                "",
                xy=anchor,
                xycoords=transform,
                xytext=anchor,
                textcoords=ax.transData,
                arrowprops={
                    "arrowstyle": "-",
                    "color": _RULE,
                    "lw": 0.6,
                    "shrinkA": 0.0,
                    "shrinkB": 1.0,
                },
                zorder=2,
            )
        # The models that matter are named: inference-valid, near-miss, and the sub-floor points
        # with a measured rate worth reading. A thin n=1 point at 0% is noise; its name is in
        # the notes rather than crammed under the axis. A name the ladder cannot seat is also
        # routed to the notes rather than drawn on a neighbour.
        if row.valid or row.model in near_miss or rate >= _LABEL_RATE_MIN:
            _place_label(ax, row, rate, hi * 100, placed, 98.0, overflow, transform)
        drawn.append(anchor)
    # The K rule is the point of the x axis: left of it the rate is not yet evidence.
    ax.axvline(data.floor, color=_RULE, linestyle=(0, (5, 3)), linewidth=1.2, zorder=1)
    ax.text(
        data.floor * 1.06,
        1.5,
        f"K={data.floor} cell floor",
        color=_RULE,
        fontsize=7.0,
        rotation=90,
        va="bottom",
    )
    ax.set_xscale("log")
    cells = [r.cells for r in rows if r.cells > 0] or [1]
    ax.set_xlim(0.8, max(cells) * 2.0)
    # The floor sits below the 0 tick so a measured 0% mark is drawn whole rather than half
    # buried in the bottom spine; no tick is placed at the floor.
    ax.set_ylim(_Y_FLOOR, 104)
    ax.set_xlabel("measured default-arm cells (log), paid + free channel", fontsize=8.5)
    ax.set_ylabel("measured pass rate % (95% Wilson)", fontsize=8.5)
    ax.grid(axis="y", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5)
    plot_frame.panel_label(ax, "A · does the measured evidence support the model?")
    return tuple(overflow)


def _draw_funnel(ax: Axes, data: _Data) -> None:
    """The cumulative validity funnel as a horizontal bar chart."""
    stages = data.funnel
    ys = list(range(len(stages)))[::-1]
    values = [value for _label, value in stages]
    ax.barh(ys, values, height=0.6, color="#90A4AE", alpha=0.9, zorder=2)
    for y, value in zip(ys, values, strict=True):
        ax.text(value + max(values) * 0.02, y, str(value), va="center", fontsize=8.0, color=_INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _value in stages], fontsize=7.2)
    # Only enough headroom for the printed count: the old 1.18 factor left a third of the axis
    # empty past the longest bar.
    ax.set_xlim(0, (max(values) or 1) * 1.08 if values else 1.0)
    ax.set_xlabel("canonical identities", fontsize=8.5)
    ax.grid(axis="x", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.2)
    plot_frame.panel_label(ax, "B · validity funnel")


def _draw_exclusions(ax: Axes, data: _Data) -> None:
    """How many named identities fail each criterion (non-cumulative, so categories overlap)."""
    counts = data.exclusions()
    ys = list(range(len(counts)))[::-1]
    values = [value for _label, value in counts]
    ax.barh(ys, values, height=0.6, color="#C62828", alpha=0.85, zorder=2)
    for y, value in zip(ys, values, strict=True):
        ax.text(value + max(values) * 0.03, y, str(value), va="center", fontsize=7.5, color=_INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _value in counts], fontsize=7.0)
    ax.set_xlim(0, (max(values) or 1) * 1.12 if values else 1.0)
    # The criteria are NOT nested filters: one identity can fail several, so the bars are
    # labelled non-exclusive rather than read as shares of a whole.
    ax.set_xlabel(
        "named identities failing the criterion (non-exclusive; a model can fail several)",
        fontsize=8.5,
    )
    ax.grid(axis="x", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.0)
    plot_frame.panel_label(ax, "B′ · per-criterion exclusions")


def _legend_handles(data: _Data) -> list[Line2D]:
    """Hue's channel key plus one marker entry per status actually drawn.

    Every swatch is the EXACT glyph and fill the scatter plots, so the key is a legend of the
    data rather than a second, greyed-out encoding. The free channel's own glyph IS the
    free-only diamond — every free row is collection-only, so no other free form exists — and
    the two are merged into one key rather than repeated. The paid channel key uses a form no
    status borrows, so no two swatches are the same (marker, fill). The inference-valid key
    draws one star: its LABEL carries no star of its own, which used to print a second glyph
    beside the swatch.
    """
    handles: list[Line2D] = [
        Line2D(
            [],
            [],
            color="none",
            marker=_CHANNEL_MARKERS["paid"],
            markersize=7,
            markerfacecolor=_PAID,
            markeredgecolor="white",
            label="paid channel",
        ),
        Line2D(
            [],
            [],
            color="none",
            marker=_CHANNEL_MARKERS["free"],
            markersize=7,
            markerfacecolor=_FREE,
            markeredgecolor="white",
            label="free channel (free-only)",
        ),
    ]
    statuses = {status_of(r) for r in data.evidenced}
    for status in (
        "valid",
        "invalid:live",
        "invalid:triage",
        "invalid:capability",
        "invalid:coverage",
        INSUFFICIENT,
    ):
        if status in statuses:
            handles.append(
                Line2D(
                    [],
                    [],
                    color="none",
                    marker=_MARKERS[status],
                    markersize=9 if status == "valid" else 6.5,
                    markerfacecolor=_PAID,
                    markeredgecolor="white",
                    label=_MARKER_LABELS[status],
                )
            )
    return handles


def _annotations(data: _Data, overflow: tuple[str, ...]) -> Annotations:
    notes = [
        f"panel B milestones: {data.named} named, {len(data.evidenced)} evidenced, "
        f"{data.paid} with a paid channel (of all named), {data.valid} inference-valid; "
        f"{data.paid_evidenced} of the evidenced identities are paid"
    ]
    notes += [f"{display_name(r.model)}: {_status_text(r, data)}" for r in data.rows]
    if overflow:
        notes.append(
            "direct labels the placement ladder could not seat without overprinting (their "
            "markers are drawn, unlabelled): " + ", ".join(overflow)
        )
    # DERIVED, not typed: the milestone and the evidenced subset are the two counts the reader
    # would otherwise have to reconcile by hand, and both move with the census.
    limitations = (
        "Panel B's paid-channel milestone is the count of named identities with a paid "
        f"channel ({data.paid}); it is NOT the paid subset of the {len(data.evidenced)} "
        f"evidenced identities, which is smaller ({data.paid_evidenced}). The milestones are "
        "separate filters, not a nested chain.",
    )
    return Annotations(
        subtitle_facts=(
            f"{data.named} named · {len(data.evidenced)} evidenced ({data.paid_evidenced} paid) · "
            f"{data.paid} paid channel · {data.free} free · {data.valid} inference-valid",
            f"coverage floor K={data.floor} measured default-arm cells · "
            f"{data.corpus} verified challenges",
        ),
        caveat=(
            "a pass rate left of the K rule is not yet evidence; wide intervals are thin coverage"
        ),
        notes=tuple(notes),
        limitations=limitations,
        counts=(
            ("named", data.named),
            ("evidenced", len(data.evidenced)),
            ("paid", data.paid),
            ("free", data.free),
            ("valid", data.valid),
        ),
    )


def _status_text(row: ModelValidity, data: _Data) -> str:
    """One census row's status, evidence and outcome, for the manifest notes."""
    stats = data.perf.get(row.model)
    outcome = (
        f"{stats.pass_rate:.1%} on n={stats.n}, ${stats.mean_cost:.4f}/task"
        if stats and stats.n
        else "no measured outcome"
    )
    return (
        f"{row.channel}, status {status_of(row)} — {outcome}; "
        f"{row.cells} cells, {row.covered}/{data.corpus} challenges; {row.reason}"
    )


def _goal_text(valid: int | None) -> str:
    """The goal prose, with the count of inference-valid (starred) identities derived.

    `None` is the module-level fallback: it names the starred set without a number so the
    constant SPEC cannot go stale before `render` binds the census count.
    """
    starred = "the starred points" if valid is None else f"the {valid} starred points"
    return (
        f"Read {starred} as the models the router may actually serve, then read the points at "
        "and above K that are NOT valid: those are the near-misses, models with enough measured "
        "evidence that a policy change (a channel, a triage verdict) could admit. A high pass "
        "rate far left of the K rule is a promising but under-measured model, not a rejected one."
    )


SPEC = FigureSpec(
    title="Which models clear the evidence bar — and why the rest fall short",
    subtitle=(
        "canonical weights identity · hue = channel (paid/free) · marker form = validity status "
        "· no hue-per-model"
    ),
    caveat=("a pass rate left of the K rule is not yet evidence; wide intervals are thin coverage"),
    reading=(
        "Panel A: one point per EVIDENCED canonical weights identity — x is its measured "
        "default-arm cell count on a log axis, y its measured pass rate with a 95% Wilson "
        "interval. The dashed vertical rule is the K-cell floor; a point to its left has too "
        "little measured evidence to clear the bar however high its rate. Hue is the CHANNEL "
        "(blue paid, orange free) — never a hue per model — and the marker FORM is the validity "
        "status: a star is inference-valid, and each other form is a first-failing criterion or "
        "an insufficient sample. Inference-valid and near-miss models are direct-labelled; a "
        "name the placement ladder cannot seat without overprinting is listed in the notes "
        "rather than drawn on a neighbour, and two markers at the same measured coordinate are "
        "separated with a short leader back to the true point. Panel B shows four milestones — "
        "named, evidenced, "
        "paid-channel, inference-valid — and panel B′ counts, per criterion, how many named "
        "identities fail it (the categories overlap, so they do not sum to the milestones). The "
        "paid-channel bar counts ALL named identities with a paid channel, not only the "
        "evidenced ones, so it is a milestone and not a nested filter: the paid subset of the "
        "evidenced set is smaller, and the notes state it."
    ),
    goal=_goal_text(None),
    definitions=(
        (
            "canonical identity",
            "the weights slug (`model_version`), so the same weights served by several "
            "providers are ONE point and a provider prefix or `-free` marker is never a name",
        ),
        (
            "inference-valid",
            "in the live pool AND triage pass (KEEP/EXCEPTION) AND capability rank measured "
            "AND coverage >= K measured default-arm cells AND a paid channel",
        ),
        _EVIDENCED_TERM,
        (
            "near-miss",
            "an evidenced identity at or above the K cell floor that is not inference-valid — "
            "enough measured evidence to be a candidate, kept out by a criterion rather than by "
            "a thin sample",
        ),
        (
            "measured pass rate",
            "passes over measured default-arm cells in either channel; the whisker is a 95% "
            "Wilson interval drawn symmetrically about the estimate, half-width = the larger "
            "Wilson arm, so it reads as uncertainty rather than as a bar from zero",
        ),
    ),
    notes=(
        "The status taxonomy is READ from benchmark.routing.model_validity: hue is the channel "
        "and marker form the first-failing criterion or the evidence floor, so no reader has to "
        "decode a model from a colour.",
    ),
    limitations=(
        "The x axis counts measured cells, not comparable task sets: models were not run on "
        "identical tasks, so two points at the same height are not a paired comparison.",
        "Free-channel points were collected under a different campaign from the paid corpus's, "
        "so their coverage is not comparable cell-for-cell and their intervals are wide from "
        "thin coverage, not from measured weakness.",
        "A named identity with no measured cell cannot be placed on panel A at all; panel B's "
        "first milestone accounts for it as the drop from named to evidenced.",
        # The paid-channel milestone limitation is DERIVED in `_annotations` from the census
        # counts, so it cannot go stale against the rows this render actually read.
    ),
)


def build(ctx: ctxmod.RoutingContext) -> _Data | None:
    """The census the report already computed once, or a fresh read for a standalone caller."""
    return _load(ctx)


def render(ctx: ctxmod.RoutingContext) -> Path | None:
    """Draw model_relevance.png; None when no model has been evidenced."""
    data = build(ctx)
    if data is None:
        return None
    size = plot_frame.FigureSize("model_relevance", 14.0, 10.5)
    fig = plot_frame.new_figure(size)
    axd = fig.subplot_mosaic(
        [["scatter", "scatter"], ["funnel", "exclude"]],
        height_ratios=(2.25, 1.0),
    )
    overflow = _draw_scatter(axd["scatter"], data)
    _draw_funnel(axd["funnel"], data)
    _draw_exclusions(axd["exclude"], data)
    axd["scatter"].legend(
        handles=_legend_handles(data),
        loc="lower right",
        fontsize=6.8,
        frameon=False,
        ncol=2,
        handlelength=1.4,
    )
    # Direct labels sit a few points from their own marker, and the placement ladder avoids
    # overprint by construction; this opt-in check turns any residual collision into a failed
    # render rather than a shipped collage (the same backstop model_grid uses).
    from shunt.inspect import plot_contract  # noqa: PLC0415

    plot_contract.request_annotation_audit(fig)
    return plot_frame.save(
        fig,
        ctx.out_dir / "model_relevance.png",
        replace(SPEC, goal=_goal_text(data.valid)),
        extra=_annotations(data, overflow),
        provenance=ctx.provenance(__name__),
        size=size,
    )
