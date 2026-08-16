"""complementarity.png — the routing premise, and the LOWER bound on what any router can win."""

# THIS IS THE ONE FIGURE THAT HOLDS THE TRI-STATE GRID (pass / fail / never sampled), and
# retiring it would have deleted the premise the whole product rests on. A router can only
# win on a task where the models DISAGREE: a task every model solves is free, a task none
# solves is lost, and only the split tasks are contestable. That count appears in no other
# figure and in no CSV.
#
# It is a LOWER bound, not a ceiling, and the grid itself is why. "Solved by every column"
# means "solved by every column that was SAMPLED", and at this sampling density EVERY such task
# still has unsampled columns — so any of them turns contestable the moment one of those columns
# is run and fails. That is what puts the gain-side ceiling at split + solved-by-all rather than
# at split.
#
# The solved-by-none rows are under-sampled too, so in principle they could also join the
# contestable set — but only by an unsampled column PASSING, which no measurement on this corpus
# supports, and a task no model solves is not value a router can capture. They are therefore
# excluded from the upper bound and reported separately rather than silently dropped.
#
# The red sampling caveat is the REASON for the interval, not an unrelated aside.
#
# It also carries the coverage audit. Two columns are only fairly compared on rows where
# BOTH are non-grey, which is a different denominator from the one the Pareto-style figures
# use, and the per-column sampling range is wide enough that forgetting it is a real error.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import plot_style
from benchmark.routing.figures import context as ctxmod
from benchmark.routing.plot_style import RawResults

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

_SPLIT = "#0072B2"
_ALL = "#2E7D32"
_NONE = "#C62828"

# A row label needs this many points of pitch to stay legible; below it the axis is
# thinned rather than allowed to overprint. `bbox_inches` no longer grows the canvas to
# absorb an over-full axis, so this is a hard constraint, not a preference.
_MIN_ROW_PITCH_PT: float = 6.0
_MIN_ROW_LABEL_PT: float = 4.0
_MAX_ROW_LABEL_PT: float = 7.0

SPEC = FigureSpec(
    title="The split count is a LOWER bound on contestable tasks — unsampled cells can only add",
    reading=(
        "Left: every task (row) against every measured (model, arm) column. Green is a "
        "pass, red a fail, grey never sampled — grey is NOT a failure. Middle: how many "
        "tasks each column was actually run on, so the uneven denominators are visible. "
        "Right: the task census. A split task — at least one pass and at least one fail — "
        "can already be won or lost by a routing decision. A task solved by every SAMPLED "
        "column has not been shown uncontestable, only untested: each one still has unsampled "
        "columns, and one failing column is enough to move it into the split slice."
    ),
    goal=(
        "Read the right panel as a FLOOR, not a ceiling. The split count is what is contestable "
        "on the evidence in hand; the solved-by-all slice sits above it as tasks no column has "
        "yet been seen to fail, so the true count lies somewhere between the split count and "
        "split-plus-solved-by-all. Then read the middle panel before comparing any two columns "
        "anywhere else in this set — two columns are only fairly compared on the rows where "
        "both are non-grey, which is a different denominator from the pooled pass rates."
    ),
    definitions=(
        ("split task", "at least one column passed and at least one failed"),
        (
            "sampling density",
            "sampled cells over total cells. The matrix is sparse BY DESIGN — p(arm|model) "
            "sampling — so a low density is a budget decision, not missing data.",
        ),
        (
            "contestable floor",
            "the measured split count. It can only rise as unsampled cells are filled: a "
            "split task stays split, while a solved-by-all task joins it as soon as any "
            "unsampled column fails.",
        ),
    ),
    notes=(
        "Columns are ordered by model price, then by within-model reasoning rank, so the "
        "grid reads left-to-right as cheap-to-expensive.",
        "The upper end of the interval counts the solved-by-all slice only. Solved-by-none "
        "rows are under-sampled too, but they could join only by an unsampled column PASSING, "
        "which nothing measured here supports, and a task no model solves is not value a "
        "router can capture.",
    ),
    limitations=(
        "The grid is drawn from the RAW measured cache, not the imputed matrix, so a grey "
        "cell is genuinely unmeasured rather than filled. Every other figure in this set "
        "that quotes a pass rate scores the imputed matrix instead.",
        "At this sampling density 'solved by every sampled column' is NOT 'solved by every "
        "column', so the contestable count is an interval rather than a number. Only filling "
        "the grey cells pins it down.",
    ),
)


@dataclass(frozen=True)
class Spread:
    """How many columns a slice of rows never sampled — the width of its unmeasured margin."""

    n_rows: int
    n_with_unsampled: int
    lo: int
    median: float
    hi: int


def _spread(grid: np.ndarray, rows: list[int]) -> Spread:
    counts = [int(np.sum(np.isnan(grid[i]))) for i in rows]
    if not counts:
        return Spread(n_rows=0, n_with_unsampled=0, lo=0, median=0.0, hi=0)
    return Spread(
        n_rows=len(counts),
        n_with_unsampled=sum(1 for c in counts if c > 0),
        lo=min(counts),
        median=float(np.median(counts)),
        hi=max(counts),
    )


@dataclass(frozen=True)
class Census:
    """The contestability FLOOR, the interval above it, and the coverage both rest on."""

    n_tasks: int
    n_cols: int
    n_sampled: int
    split: int
    all_pass: int
    none_pass: int
    coverage: np.ndarray
    columns: list[tuple[str, str]]
    grid: np.ndarray
    all_pass_spread: Spread
    none_pass_spread: Spread

    @property
    def density(self) -> float:
        return (
            self.n_sampled / (self.n_tasks * self.n_cols) if self.n_tasks and self.n_cols else 0.0
        )

    @property
    def contestable_floor(self) -> int:
        """Tasks already shown contestable. Filling grey cells can only raise this."""
        return self.split

    @property
    def contestable_ceiling(self) -> int:
        """The floor plus every task no sampled column has yet been seen to fail."""
        return self.split + self.all_pass


def build_census(raw: RawResults, columns: list[tuple[str, str]]) -> Census:
    """The tri-state grid plus the three counts the ceiling is read from."""
    tasks = sorted(raw)
    grid = np.full((len(tasks), len(columns)), np.nan, dtype=float)
    for i, tid in enumerate(tasks):
        per_model = raw.get(tid, {})
        for j, (model, arm) in enumerate(columns):
            row = per_model.get(model, {}).get(arm)
            if row is not None:
                grid[i, j] = 1.0 if row.get("pass") else 0.0
    split = 0
    all_rows: list[int] = []
    none_rows: list[int] = []
    for i in range(len(tasks)):
        seen = grid[i, ~np.isnan(grid[i])]
        if seen.size == 0:
            continue
        if seen.min() != seen.max():
            split += 1
        elif seen.max() == 1.0:
            all_rows.append(i)
        else:
            none_rows.append(i)
    return Census(
        n_tasks=len(tasks),
        n_cols=len(columns),
        n_sampled=int(np.sum(~np.isnan(grid))),
        split=split,
        all_pass=len(all_rows),
        none_pass=len(none_rows),
        coverage=np.sum(~np.isnan(grid), axis=0),
        columns=columns,
        grid=grid,
        all_pass_spread=_spread(grid, all_rows),
        none_pass_spread=_spread(grid, none_rows),
    )


def row_label_step(n_tasks: int, axes_height_in: float) -> tuple[int, float]:
    """(step, fontsize) that labels as many rows as the axes can physically hold."""
    if n_tasks <= 0 or axes_height_in <= 0:
        return (1, _MAX_ROW_LABEL_PT)
    capacity = max(1, int(axes_height_in * 72.0 / _MIN_ROW_PITCH_PT))
    step = max(1, -(-n_tasks // capacity))
    pitch = axes_height_in * 72.0 / (n_tasks / step)
    return (step, float(min(_MAX_ROW_LABEL_PT, max(_MIN_ROW_LABEL_PT, pitch * 0.75))))


def _draw_grid(ax: Axes, census: Census, axes_height_in: float) -> None:
    cmap = ListedColormap([plot_style.TRISTATE_FAIL, plot_style.TRISTATE_PASS]).with_extremes(
        bad=plot_style.TRISTATE_UNSAMPLED
    )
    ax.imshow(census.grid, cmap=cmap, aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    # No column tick labels. Thirteen (model, arm) names across four inches only fit
    # rotated, and rotated ticks at 7pt are precisely what made the previous version
    # unreadable. Panel B names every column, in the same order, at a readable size.
    ax.set_xticks([])
    step, size = row_label_step(census.n_tasks, axes_height_in)
    ticks = list(range(0, census.n_tasks, step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([""] * len(ticks), fontsize=size)
    ax.tick_params(axis="y", length=2, pad=1.5)
    ax.set_ylabel(f"{census.n_tasks} tasks, one row each", fontsize=8)
    # Below the axes, in the strip the removed tick labels freed — a legend inside the
    # grid would sit on top of real cells, and every cell here is data.
    ax.legend(
        handles=[
            Patch(color=plot_style.TRISTATE_PASS, label="pass"),
            Patch(color=plot_style.TRISTATE_FAIL, label="fail"),
            Patch(color=plot_style.TRISTATE_UNSAMPLED, label="never sampled"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        fontsize=7,
        frameon=False,
    )
    ax.set_xlabel("columns, cheapest → priciest (named in panel B)", fontsize=8, labelpad=16)
    plot_frame.panel_label(ax, "A · task × (model, arm), tri-state")


def _draw_coverage(ax: Axes, census: Census) -> None:
    ys = list(range(census.n_cols))[::-1]
    for y, (col, n) in zip(ys, zip(census.columns, census.coverage, strict=True), strict=True):
        ax.barh(y, int(n), height=0.62, color="#9ec4e2", zorder=2)
        ax.text(
            int(n) + census.n_tasks * 0.015,
            y,
            f"{int(n)}/{census.n_tasks}",
            fontsize=7,
            va="center",
            color="#333333",
        )
        _ = col
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{m}\n{a}" for m, a in census.columns], fontsize=6.5)
    ax.set_xlim(0, census.n_tasks * 1.3)
    ax.set_xlabel("tasks this column was measured on", fontsize=9)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "B · uneven denominators, by design")


def _draw_census(ax: Axes, census: Census) -> None:
    parts = [
        ("not yet contested\n(every sampled column passed)", census.all_pass, _ALL),
        ("SPLIT — contestable", census.split, _SPLIT),
        ("solved by none", census.none_pass, _NONE),
    ]
    bottom = 0.0
    for label, value, colour in parts:
        ax.bar(0, value, bottom=bottom, width=0.5, color=colour, zorder=2)
        ax.text(
            0.30,
            bottom + value / 2.0,
            f"{label}\n{value} of {census.n_tasks}  ({value / max(census.n_tasks, 1):.0%})",
            fontsize=8,
            va="center",
            ha="left",
            color=colour,
        )
        bottom += value
    # Wide enough for the longest slice label ("every sampled column passed") to end inside
    # the axes. At 1.55 its closing bracket printed on top of the right spine.
    ax.set_xlim(-0.42, 2.05)
    ax.set_ylim(0, census.n_tasks * 1.05)
    ax.set_xticks([])
    ax.set_ylabel("tasks", fontsize=9)
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "C · contestable FLOOR, not ceiling")


def _annotations(census: Census) -> Annotations:
    lo, hi = int(census.coverage.min()), int(census.coverage.max())
    lo_col = census.columns[int(np.argmin(census.coverage))]
    hi_col = census.columns[int(np.argmax(census.coverage))]
    facts = [
        f"{census.n_tasks} tasks x {census.n_cols} columns, {census.n_sampled} of "
        f"{census.n_tasks * census.n_cols} cells sampled ({census.density:.1%})",
        f"coverage {lo}/{census.n_tasks} ({lo_col[0]}) to {hi}/{census.n_tasks} ({hi_col[0]})",
        f"contestable tasks in [{census.contestable_floor}, {census.contestable_ceiling}] of "
        f"{census.n_tasks} — {census.split} split is a FLOOR, not a ceiling",
    ]
    spread = census.all_pass_spread
    return Annotations(
        subtitle_facts=tuple(facts),
        # The sampling caveat IS the reason the count is an interval, so it says so here rather
        # than leaving the reader to connect a density number to a bound three lines away.
        caveat=(
            f"Sampling is {census.density:.0%} of the grid: grey is unmeasured, never a fail — "
            f"which is why {census.split} is a lower bound, not the ceiling."
            if census.density < 0.5
            else None
        ),
        notes=(
            f"solved by all: {census.all_pass}; split: {census.split}; solved by none: "
            f"{census.none_pass}",
            f"all {spread.n_with_unsampled} of {spread.n_rows} solved-by-all tasks still have "
            f"unsampled columns ({spread.lo} to {spread.hi} of {census.n_cols}, median "
            f"{spread.median:.0f}) — any one of them becomes contestable if an unsampled column "
            f"fails",
            f"the {census.none_pass_spread.n_rows} solved-by-none tasks are under-sampled too "
            f"({census.none_pass_spread.lo} to {census.none_pass_spread.hi} columns), but could "
            f"join only by an unsampled column PASSING, so they are excluded from the interval",
        ),
        counts=(
            ("tasks", census.n_tasks),
            ("columns", census.n_cols),
            ("sampled_cells", census.n_sampled),
            ("split_tasks", census.split),
            ("contestable_floor", census.contestable_floor),
            ("contestable_ceiling", census.contestable_ceiling),
        ),
    )


def render(ctx: ctxmod.RoutingContext, columns: list[tuple[str, str]]) -> Path | None:
    """Draw complementarity.png from the RAW measured cache."""
    if ctx.raw is None or not columns:
        return None
    census = build_census(ctx.raw, columns)
    if census.n_tasks == 0:
        return None
    size = plot_frame.WIDE_TALL
    fig, axes = plot_frame.subplots(size, 1, 3, width_ratios=(1.0, 1.15, 0.75))
    _draw_grid(axes[0], census, size.height_in - 2.0)
    _draw_coverage(axes[1], census)
    _draw_census(axes[2], census)
    for ax in (axes[1], axes[2]):
        plot_style.fit_end_labels(ax)
    return plot_frame.save(
        fig,
        ctx.out_dir / "complementarity.png",
        SPEC,
        extra=_annotations(census),
        provenance=ctx.provenance(__name__),
        size=size,
    )
