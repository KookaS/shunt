"""The model grid — one drawer, two data sources: what a rung costs, weighs, and delivers."""

# WHY THIS LIVES IN THE WHEEL. The benchmark half draws it from `results.csv` and the
# inference half from the live outcome store, and those are the same three panels over the
# same three questions. A drawer that shipped only inside `benchmark/` could not be reached
# from the rig container, and a second copy under `src/` would drift from it; `render` here
# takes an already-computed `GridData` and knows nothing about where the rows came from,
# which is the pattern `render_inference_figures.py` already proves for the seven live
# figures.
#
# THE ZERO PROBLEM, and why panel A is split rather than nudged. A log axis cannot show zero,
# and a locally-served rung costs exactly zero dollars per token. The three usual escapes are
# all lies of different sizes: plotting $0.001 invents a price, symlog alone still draws a
# linear neighbourhood around zero that reads as "nearly free rather than free", and dropping
# the row hides the cheapest thing on the ladder. So the axis BREAKS: a narrow categorical
# column at the left holds every $0 row, the log region holds every priced row, and the gap
# between them is drawn as a break rather than as distance. A reader can see that the two
# regions are not one ruler.
#
# HUE IS A CLUSTER, NOT A MODEL. `plot_style.model_color_map` raises past its palette, and a
# ladder heading for ~20 rungs would trip it on the first render. More to the point, twenty
# hues carry no reading. So hue encodes the SIZE CLASS — the coarse grouping panel B then
# gives exact numbers for — which turns panel A into a question worth asking: does sparsity
# class predict where a rung lands on the price/quality plane? Serving mode rides on the
# marker EDGE, so the two encodings never compete for the same channel.
#
# A ROW FROM OUTSIDE THE CORPUS IS DAGGERED, NOT ASSIMILATED. A rung can be measured before it
# is in the corpus — a different harness, a different task draw — and dropping it would hide a
# real result while merging it silently would let a reader compare two heights that were never
# comparable. So such a row carries `provenance_note`, draws with a dagger on both panels that
# name it, and puts its harness, its sample and its verdict ceiling under the canvas. The
# panel's own limitation list says what the dagger costs a reader.
#
# UNDISCLOSED IS DRAWN, NOT DROPPED. Every closed API tier publishes no parameter count. Those
# rows keep their place on panel A at a fixed reference marker with a distinct shape, and
# panel B prints the word instead of a bar. Sizing them by a guessed parameter count is the
# one thing this figure must never do.

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from shunt.inspect import plot_frame
from shunt.inspect.plot_frame import Annotations, FigureSpec
from shunt.inspect.plot_style import ci_yerr, wilson_interval

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.axes import Axes

    from shunt.inspect.plot_frame import Provenance

# ------------------------------------------------------------------ size classes

# Four bands, chosen so each names a serving reality rather than a round number: a rung above
# 1T total cannot be served anywhere but a datacentre, 100B-1T is the frontier-MoE band, below
# 100B is what fits on one machine, and UNDISCLOSED is every closed API tier.
_TRILLION: Final[int] = 1_000_000_000_000
_HUNDRED_B: Final[int] = 100_000_000_000

CLUSTER_ORDER: Final[tuple[str, ...]] = (
    "≥1T total",
    "100B–1T total",
    "<100B total",
    "size UNDISCLOSED",
)
# Okabe-Ito members chosen for contrast against the near-white canvas, in the band order
# above; UNDISCLOSED is the neutral, because "we do not know" must not look like a category
# with an opinion.
CLUSTER_COLOR: Final[dict[str, str]] = {
    "≥1T total": "#0072B2",
    "100B–1T total": "#009E73",
    "<100B total": "#D55E00",
    "size UNDISCLOSED": "#767676",
}

_HOSTED_EDGE: Final[str] = "#1a1a1a"
_LOCAL_EDGE: Final[str] = "#B15928"
_RULE: Final[str] = "#9a9a9a"
_ZERO_LABEL: Final[str] = "$0 · local"

# Marker area in points², from `sqrt(active_params)` scaled into a readable band. The square
# root is applied to the PARAMETER count and the result mapped to AREA, so perceived size
# tracks the fourth root of parameters — a 2.8T rung beside a 3B one is otherwise either a
# blot or a dot.
_AREA_MIN: Final[float] = 46.0
_AREA_MAX: Final[float] = 470.0
# Every UNDISCLOSED row draws at one fixed size that is deliberately NOT on the ramp, so it
# cannot be read off the size legend as if it were a measurement.
_AREA_UNKNOWN: Final[float] = 78.0

_EMPTY_WRAP: Final[int] = 30

# How close two panel-A labels have to be before one of them is moved. The x half is a RATIO
# because the axis is logarithmic — two rungs 10% apart in price sit on top of each other
# wherever they are on the decade — and the y half is in percentage points. Sized to the
# 6.6pt label: a pair inside both bounds overprints, a pair outside either one does not.
_LABEL_NEAR_DECADES: Final[float] = 0.09
_LABEL_NEAR_POINTS: Final[float] = 7.0
# The offsets a crowded label may take, in points, tried in this order: above the marker, then
# below it, then one row further out on each side. A LADDER rather than a two-way flip because
# the ladder is growing — three rungs at one price and one rate is the expected case, not a
# corner — and a flip can only separate two of them. Every rung on the ladder is one label
# height apart, so two labels that end on different rungs cannot overprint. THREE rungs per
# SIDE, not per ladder: the ceiling case may only go downwards, and a ladder with two negative
# rungs would collapse a cluster of three ceiling-height rows back onto each other.
_LABEL_OFFSETS: Final[tuple[float, ...]] = (11.0, -18.0, 24.0, -31.0, 37.0, -44.0)


@dataclass(frozen=True)
class GridRow:
    """One model on the grid: its price, its weight, its measured outcome, its latency."""

    name: str
    # The panel-A cost quantity, in dollars, on whatever basis `GridData.x_label` names. None
    # means the weights are served locally, so there is no per-token list price and no billed
    # session total to place — the category column, never a small number on the log axis. It
    # is a statement about the AXIS QUANTITY being absent, NOT a claim that running the rung
    # is free: a local rung's cost per solved task is UNDEFINED, and any row that needs to say
    # so says it in `provenance_note`.
    x: float | None
    serving_mode: str
    n: int
    passes: int
    # None is UNDISCLOSED: the vendor publishes no figure and none is invented here.
    total_params: int | None
    active_params: int | None
    # Per-call latency observations in seconds. Empty means MISSING — never zero, and never a
    # panel drawn from no samples.
    latency_s: tuple[float, ...] = ()
    # Set ONLY on a row whose outcome was measured outside the corpus named in
    # `GridData.source` — a different harness, a different task draw, or both. Such a row is
    # not cell-for-cell comparable with the rest of the panel, so it is drawn with a dagger on
    # its label and this sentence is printed beside it. None means the row came from the
    # corpus and needs no qualification.
    provenance_note: str | None = None

    @property
    def is_external(self) -> bool:
        return self.provenance_note is not None

    @property
    def rate(self) -> float:
        return self.passes / self.n if self.n else 0.0

    @property
    def wilson(self) -> tuple[float, float]:
        return wilson_interval(self.passes, self.n)

    @property
    def cluster(self) -> str:
        if self.total_params is None:
            return CLUSTER_ORDER[3]
        if self.total_params >= _TRILLION:
            return CLUSTER_ORDER[0]
        if self.total_params >= _HUNDRED_B:
            return CLUSTER_ORDER[1]
        return CLUSTER_ORDER[2]

    @property
    def is_local(self) -> bool:
        return self.serving_mode == "local"


@dataclass(frozen=True)
class GridData:
    """Everything the three panels read, plus the sentence that makes the x axis auditable."""

    rows: tuple[GridRow, ...]
    # Panel A's x axis label. The two halves buy on DIFFERENT currencies — the benchmark half
    # has token counts and prices a blended $/Mtok, the live store records only dollars and so
    # plots a measured $/session — and a hardcoded label would state one half's basis on both
    # canvases. The quantity itself always lives in `GridRow.x`.
    x_label: str
    # How that x was arrived at — printed in the subtitle, because a price axis without its
    # basis is not a checkable number.
    price_basis: str
    # What produced these rows (a corpus name), so the two halves are never confused.
    source: str
    # THE PANEL-A LIMITATION, PER HALF — because the two halves plot different quantities and
    # a hardcoded sentence states one half's truth on both canvases. The benchmark half plots
    # a LIST PRICE and must say so; the live half plots a MEASURED BILL and the list-price
    # sentence is simply false there. No default: a new caller must decide which it is rather
    # than inherit the wrong one silently.
    x_limitation: str


# ------------------------------------------------------------------ shared geometry


def _frame(ax: Axes) -> None:
    """A dashed outline for a panel carrying prose instead of data."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(
        Rectangle(
            (0.02, 0.02),
            0.96,
            0.96,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor="#cccccc",
            linestyle=(0, (6, 6)),
            linewidth=1.0,
        )
    )


def _empty(ax: Axes, message: str) -> None:
    """Name what is absent. A blank axes reads as a failed render, not as an answer."""
    _frame(ax)
    ax.text(
        0.5,
        0.5,
        "\n".join(textwrap.wrap(message, _EMPTY_WRAP)),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        color=plot_frame.MUTED,
        style="italic",
    )


def marker_area(active: int | None, span: tuple[int, int] | None) -> float:
    """Marker area for one row: on the sqrt ramp when sized, at the fixed mark when not."""
    if active is None or span is None:
        return _AREA_UNKNOWN
    lo, hi = span
    if hi <= lo:
        return (_AREA_MIN + _AREA_MAX) / 2
    frac = (active**0.5 - lo**0.5) / (hi**0.5 - lo**0.5)
    return float(_AREA_MIN + frac * (_AREA_MAX - _AREA_MIN))


def _active_span(rows: Sequence[GridRow]) -> tuple[int, int] | None:
    sized = [r.active_params for r in rows if r.active_params is not None]
    return (min(sized), max(sized)) if sized else None


def _edge(row: GridRow) -> tuple[str, float]:
    return (_LOCAL_EDGE, 1.9) if row.is_local else (_HOSTED_EDGE, 1.1)


def _near(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Would labels anchored at these two points overprint each other?"""
    if abs(a[1] - b[1]) >= _LABEL_NEAR_POINTS:
        return False
    # The $0 column has one x for every row, so proximity there is the y test alone.
    if a[0] <= 0 or b[0] <= 0:
        return True
    return abs(math.log10(a[0] / b[0])) < _LABEL_NEAR_DECADES


def label_offset(
    anchor: tuple[float, float],
    placed: Sequence[tuple[tuple[float, float], float]],
    *,
    forced_below: bool,
) -> float:
    """The first offset on the ladder that no already-placed nearby label is using."""
    # `forced_below` is the ceiling case: a label above a marker whose Wilson bar already
    # reaches the top of the axis is drawn off-canvas, and raising the ceiling to hold it
    # would invent headroom the measurement does not have.
    ladder = [o for o in _LABEL_OFFSETS if o < 0] if forced_below else list(_LABEL_OFFSETS)
    taken = {offset for other, offset in placed if _near(anchor, other)}
    return next((o for o in ladder if o not in taken), ladder[-1])


def _label(row: GridRow) -> str:
    """The row's name, daggered when its outcome came from outside the panel's corpus."""
    # A TEXT channel on purpose. Hue, marker edge, marker area and marker shape are all
    # already spoken for, and a fifth visual encoding would compete with one of them for the
    # same glance. The dagger is inert until a reader looks for it, and the note under the
    # canvas says exactly what it means.
    return f"{row.name} †" if row.is_external else row.name


# ------------------------------------------------------------------ panel A


def _draw_operating_frontier(
    ax_zero: Axes, ax_log: Axes, rows: Sequence[GridRow], x_label: str
) -> tuple[float, float]:
    """The price/quality plane, with the $0 column and the log region on one shared y."""
    span = _active_span(rows)
    priced = [r for r in rows if r.x is not None]
    free = [r for r in rows if r.x is None]
    ceiling = min(100.0, max(r.wilson[1] * 100 for r in rows) + 9.0) - 6.0

    for ax, subset, xs in (
        (ax_zero, free, [0.0] * len(free)),
        (ax_log, priced, [r.x or 0.0 for r in priced]),
    ):
        # Labels are placed left to right so the choice is deterministic: the same corpus
        # draws the same canvas, and each row dodges every label already placed near it.
        placed: list[tuple[tuple[float, float], float]] = []
        for row, x in sorted(zip(subset, xs, strict=True), key=lambda p: (p[1], p[0].name)):
            edge, width = _edge(row)
            lo, hi = row.wilson
            down, up = ci_yerr(row.rate * 100, lo * 100, hi * 100)
            ax.errorbar(
                [x],
                [row.rate * 100],
                yerr=[[down], [up]],
                fmt="none",
                ecolor=_RULE,
                elinewidth=1.0,
                capsize=2.5,
                zorder=2,
            )
            ax.scatter(
                [x],
                [row.rate * 100],
                s=marker_area(row.active_params, span),
                c=CLUSTER_COLOR[row.cluster],
                marker="D" if row.total_params is None else "o",
                edgecolors=edge,
                linewidths=width,
                alpha=0.92,
                zorder=3,
            )
            anchor = (x, row.rate * 100)
            offset = label_offset(anchor, placed, forced_below=row.wilson[1] * 100 >= ceiling)
            placed.append((anchor, offset))
            ax.annotate(
                _label(row),
                anchor,
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                va="top" if offset < 0 else "baseline",
                fontsize=6.6,
                color=plot_frame.INK,
            )

    ax_zero.set_xlim(-0.6, 0.6)
    ax_zero.set_xticks([0.0])
    ax_zero.set_xticklabels([_ZERO_LABEL], fontsize=7.5)
    ax_zero.set_ylabel("verified pass rate (%)", fontsize=8.5)
    if not free:
        # NOT `_empty`: this axes carries the shared y scale for the whole panel, and the
        # prose frame strips exactly the ticks and spines panel A is read against. An empty
        # category column says so in its own space and keeps its ruler.
        ax_zero.text(
            0.5,
            0.5,
            "none\nyet",
            transform=ax_zero.transAxes,
            ha="center",
            va="center",
            fontsize=7.5,
            color=plot_frame.MUTED,
            style="italic",
        )

    if priced:
        values = [r.x or 0.0 for r in priced]
        ax_log.set_xscale("log")
        ax_log.set_xlim(min(values) / 2.2, max(values) * 2.2)
    # The MIX is named in the subtitle, not here: the two halves blend differently (the
    # benchmark half measures its corpus's own split; the live store records no tokens), and a
    # hardcoded label would state one half's basis on both canvases.
    ax_log.set_xlabel(x_label, fontsize=8.5)
    ax_log.tick_params(labelleft=False)
    for ax in (ax_zero, ax_log):
        ax.grid(visible=True, axis="y", alpha=0.22, linewidth=0.6)
        ax.tick_params(labelsize=7.5)
    # The break marks: the two regions are not one ruler and must not read as one.
    ax_zero.spines["right"].set_visible(False)
    ax_log.spines["left"].set_visible(False)
    for ax, xpos in ((ax_zero, 1.0), (ax_log, 0.0)):
        ax.plot(
            [xpos, xpos],
            [0.0, 1.0],
            transform=ax.transAxes,
            color="#bbbbbb",
            linestyle=(0, (2, 3)),
            linewidth=1.0,
            clip_on=False,
            zorder=1,
        )
    rates = [r.rate * 100 for r in rows]
    lows = [r.wilson[0] * 100 for r in rows]
    highs = [r.wilson[1] * 100 for r in rows]
    # A deliberate strip of empty axis below the lowest whisker, so the key sits on blank
    # canvas instead of over a confidence bar. Empty space is honest; a hidden bar is not.
    lo = max(0.0, min(lows) - 13.0)
    hi = min(100.0, max(highs + rates) + 9.0)
    return lo, hi


def _panel_a_key(ax: Axes, rows: Sequence[GridRow]) -> None:
    """Two legends on one panel: what the hue means, and what the edge means."""
    present = [c for c in CLUSTER_ORDER if any(r.cluster == c for r in rows)]
    hue = [
        Line2D(
            [],
            [],
            marker="D" if c == CLUSTER_ORDER[3] else "o",
            linestyle="none",
            markersize=6,
            markerfacecolor=CLUSTER_COLOR[c],
            markeredgecolor=_HOSTED_EDGE,
            label=c,
        )
        for c in present
    ]
    modes = sorted({r.serving_mode for r in rows})
    edge = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor=_LOCAL_EDGE if m == "local" else _HOSTED_EDGE,
            markeredgewidth=1.9 if m == "local" else 1.1,
            label=f"{m} (edge)",
        )
        for m in modes
    ]
    span = _active_span(rows)
    size_key: list[Line2D] = []
    if span is not None and span[0] != span[1]:
        # Two anchors, not a ramp: the encoding is a fourth root of parameters, so a
        # graduated key would invite reading intermediate areas off it as measurements.
        for value in (span[0], span[1]):
            size_key.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markersize=(marker_area(value, span) / 3.1416) ** 0.5,
                    markerfacecolor="#dddddd",
                    markeredgecolor=_HOSTED_EDGE,
                    label=f"{value / 1e9:.0f}B active",
                )
            )
    # ONE legend, three columns. Three separate legends had to be placed in three free
    # regions, and the only region big enough for the size key sat on top of the
    # highest-scoring marker — a key that hides the point it explains.
    ax.legend(
        handles=[*hue, *edge, *size_key],
        fontsize=6.2,
        loc="lower right",
        ncols=2,
        framealpha=0.94,
        labelspacing=0.9,
        handletextpad=0.9,
        columnspacing=1.1,
    )


# ------------------------------------------------------------------ panel B


def _draw_size_ladder(ax: Axes, rows: Sequence[GridRow]) -> int:
    """Total hollow, active filled, joined by a rule whose length IS the sparsity gap."""
    sized = [r for r in rows if r.total_params is not None and r.active_params is not None]
    unknown = [r for r in rows if r.total_params is None or r.active_params is None]
    ordered = sorted(sized, key=lambda r: r.total_params or 0)
    labels: list[str] = []
    drawn = 0
    # UNDISCLOSED rows sit at the BOTTOM of the ladder. Above the sized rows they would read
    # as the largest models, which is a claim about them nobody made.
    for row in sorted(unknown, key=lambda r: r.name):
        ax.text(
            0.02,
            len(labels),
            "UNDISCLOSED — vendor publishes no count",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=6.6,
            color=plot_frame.MUTED,
            style="italic",
        )
        labels.append(_label(row))
    for y, row in enumerate(ordered, start=len(labels)):
        total = row.total_params or 0
        active = row.active_params or 0
        colour = CLUSTER_COLOR[row.cluster]
        ax.plot([active, total], [y, y], color=_RULE, linewidth=1.4, zorder=1)
        ax.scatter(
            [total], [y], s=64, facecolors="none", edgecolors=colour, linewidths=1.7, zorder=3
        )
        ax.scatter([active], [y], s=54, c=colour, zorder=3)
        labels.append(_label(row))
        drawn += 1
    if not labels:
        _empty(ax, "no model declares a size")
        return 0
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    if drawn:
        ax.set_xscale("log")
    ax.set_xlabel("parameters (log) — hollow: total · filled: active", fontsize=8.5)
    ax.grid(visible=True, axis="x", alpha=0.22, linewidth=0.6)
    ax.tick_params(labelsize=7.5)
    return drawn


# ------------------------------------------------------------------ panel C


def _draw_latency(ax: Axes, rows: Sequence[GridRow], mode: str) -> int:
    """One serving population's per-call latency. Hosted and local never share an axis."""
    subset = [r for r in rows if r.serving_mode == mode and r.latency_s]
    if not subset:
        _empty(ax, f"no {mode} latency measured — the column is blank, not zero")
        return 0
    ordered = sorted(subset, key=lambda r: sum(r.latency_s) / len(r.latency_s))
    ax.boxplot(
        [list(r.latency_s) for r in ordered],
        vert=False,
        widths=0.55,
        showfliers=False,
    )
    ax.set_yticks(range(1, len(ordered) + 1))
    ax.set_yticklabels([r.name for r in ordered], fontsize=7.2)
    ax.set_xlabel(f"{mode} latency per call (s)", fontsize=8.5)
    ax.grid(visible=True, axis="x", alpha=0.22, linewidth=0.6)
    ax.tick_params(labelsize=7.5)
    return len(ordered)


# ------------------------------------------------------------------ assembly


def grid_annotations(data: GridData, *, sized: int, hosted: int, local: int) -> Annotations:
    """Subtitle facts and notes derived from the rows actually drawn."""
    rows = data.rows
    free = [r for r in rows if r.x is None]
    undisclosed = [r for r in rows if r.total_params is None]
    external = [r for r in rows if r.is_external]
    ns = [r.n for r in rows]
    facts = [
        f"{len(rows)} models over {data.source}",
        f"{len(free)} at $0 (local) · {len(rows) - len(free)} priced",
    ]
    if ns:
        # The n spread is a subtitle fact, not a footnote: the widest and narrowest rows
        # differ by an order of magnitude here, and a reader comparing two heights on this
        # panel is not comparing two equally-observed things.
        facts.append(f"n per model {min(ns)}–{max(ns)}, unpaired task sets")
    facts.append(data.price_basis)
    notes = [
        f"{_label(r)}: {r.rate * 100:.1f}% on n={r.n} "
        + ("· $0 (local)" if r.x is None else f"· ${r.x:.4f} ({data.x_label})")
        + (
            " · size UNDISCLOSED"
            if r.total_params is None
            else f" · {r.total_params / 1e9:.0f}B total / "
            f"{(r.active_params or 0) / 1e9:.0f}B active"
        )
        for r in sorted(rows, key=lambda r: r.name)
    ]
    if undisclosed:
        notes.append(
            "drawn at a fixed reference marker because no parameter count is published: "
            + ", ".join(sorted(r.name for r in undisclosed))
        )
    for row in sorted(external, key=lambda r: r.name):
        notes.append(f"† {row.name}: {row.provenance_note}")
    limitations = [
        data.x_limitation,
        "The $0 column and the log region are not one ruler. The gap between them is a break, "
        "and no distance across it is meaningful.",
        "Hue is a coarse size band, not a capability measurement — panel B carries the exact "
        "counts, and a band is not evidence that its members behave alike.",
    ]
    if external:
        # The dagger's meaning, stated as a LIMITATION rather than a footnote: an undaggered
        # row and a daggered one differ by more than which tasks they saw, and a reader who
        # compares their heights without knowing that is being misled by the panel.
        limitations.append(
            "A DAGGERED row (†) was measured outside this corpus, under a different harness. "
            "Its height is not comparable cell-for-cell with the rest of the panel — read it "
            "as a separate measurement plotted on the same axes, never as one more corpus "
            "row. Its note below states the harness, the sample, how far that sample overlaps "
            "this corpus, and the verdict ceiling."
        )
    if hosted == 0 and local == 0:
        limitations.append(
            "Panels C and D are empty: no latency has been instrumented on any row. The "
            "column is MISSING, and nothing here should be read as a speed claim."
        )
    return Annotations(
        subtitle_facts=tuple(facts),
        notes=tuple(notes),
        limitations=tuple(limitations),
        counts=(
            ("models", len(rows)),
            ("sized", sized),
            ("undisclosed", len(undisclosed)),
            ("external", len(external)),
            ("latency_hosted", hosted),
            ("latency_local", local),
        ),
    )


def render(
    path: Path, data: GridData, spec: FigureSpec, provenance: Provenance | None = None
) -> Path:
    """Draw the three-panel model grid at *path*. An empty corpus draws empty panels."""
    # EMPTINESS IS A RESULT, not a skip. A corpus with no labelled model draws four framed
    # panels saying so; returning None here would leave the docs section pointing at a file
    # that does not exist, which SH009 reports as a missing figure rather than as the honest
    # "nothing has been measured yet" the canvas can state for itself.
    size = plot_frame.WIDE_TALL
    fig = plot_frame.new_figure(size)
    axd = fig.subplot_mosaic(
        [["a_zero", "a_log", "b"], ["c_hosted", "c_hosted", "c_local"]],
        width_ratios=(0.42, 2.55, 2.35),
        # A visible gutter between the $0 column and the log region: the break marks say the
        # two are not one ruler, and the gap is what makes that legible at a glance.
        gridspec_kw={"wspace": 0.22},
        height_ratios=(2.1, 0.85),
    )
    axd["a_zero"].sharey(axd["a_log"])
    plot_frame.panel_label(axd["a_zero"], "A · operating frontier")
    plot_frame.panel_label(axd["b"], "B · size ladder")
    plot_frame.panel_label(axd["c_hosted"], "C · latency, hosted")
    plot_frame.panel_label(axd["c_local"], "D · latency, local")

    if data.rows:
        ylo, yhi = _draw_operating_frontier(axd["a_zero"], axd["a_log"], data.rows, data.x_label)
        axd["a_log"].set_ylim(ylo, yhi)
        _panel_a_key(axd["a_log"], data.rows)
    else:
        _empty(axd["a_zero"], "")
        _empty(axd["a_log"], "no model in this corpus carries a verified outcome")
    sized = _draw_size_ladder(axd["b"], data.rows)
    hosted = _draw_latency(axd["c_hosted"], data.rows, "hosted")
    local = _draw_latency(axd["c_local"], data.rows, "local")

    return plot_frame.save(
        fig,
        path,
        spec,
        extra=grid_annotations(data, sized=sized, hosted=hosted, local=local),
        provenance=provenance,
        size=size,
    )
