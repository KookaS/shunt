"""routing_decision_audit.png — the router's error budget: over- vs under-provisioning."""

# "167 of 175 decisions coincide with the cheapest sufficient model" is a sentence, and a
# sentence cannot say WHICH way the eight went. Over-provisioning (a cheaper model would
# also have solved it) costs money at no quality loss; under-provisioning (the pick failed
# where a dearer model would have worked) costs a solved task. They are different defects
# with different fixes, and pooling them into one accuracy number hides both.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Patch, Rectangle

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import metrics
from benchmark.routing.figures import context as ctxmod
from benchmark.routing.model_universe import canonical_label

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

# CVD-SAFE BUDGET HUES. The old pair put green (exact) directly beside red (under-provisioned),
# the one adjacency red-green colour blindness collapses; blue/amber/magenta stay separate under
# deuteranopia, protanopia and tritanopia, and every segment carries a text label besides.
_EXACT = "#0072B2"
_OVER = "#E69F00"
_UNDER = "#CC0066"
_FREE = "#9E9E9E"

SPEC = FigureSpec(
    title="The kNN selection rule's errors go both ways — it loses tasks, not just money",
    reading=(
        "Left: rows are the model the router chose, columns the cheapest model that actually "
        "solved the task. The diagonal is an exact hit. BELOW it the router paid for a model "
        "it did not need; above it the router under-provisioned and the task was lost. Right: "
        "the same decisions as an error budget — exact, over-provisioned, under-provisioned, "
        "and the tasks no model solved, which no decision could have won."
    ),
    goal=(
        "Read the two error columns against each other. Over-provisioning is the bill for "
        "guessing high and costs only money; under-provisioning costs a task that some "
        "dearer model would have solved, and no threshold recovers it after the fact. The "
        "rule plotted here is a single-shot kNN prediction with no verify-and-escalate step, "
        "so both are reachable — an earlier draft of this figure read the empty "
        "under-provisioned column of a CASCADE as a property of the router itself."
    ),
    definitions=(
        (
            "cheapest sufficient",
            "the cheapest measured model that passed this task — the router's correct answer. "
            "Undefined when no model passed.",
        ),
        (
            "over-provisioned",
            "the chosen model was dearer than the cheapest that would have passed.",
        ),
        ("under-provisioned", "the chosen model failed a task some dearer model solved."),
        (
            "outside the inference-valid pool",
            "the router's pick is a benchmark-only or collection-only model, so it has no axis "
            "on the pool-only grid. Counted in the decision denominator, never drawn.",
        ),
    ),
    notes=(
        "Both axes are in price order and rows are the CHOSEN model, so a cell below the "
        "diagonal is over-provisioning by construction rather than by convention.",
        "The grid is scoped to the inference-VALID models — the live pool clear of the "
        "coverage/triage/capability floor. Benchmark-only and collection-only models are named "
        "on model_validity.png and invalid_models.png; a choice outside this pool is not drawn, "
        "but it is counted and its count is printed in the subtitle.",
    ),
    limitations=(
        "Cheapest-sufficient is read off the coverage-completed matrix, so a task whose "
        "cheap cell was imputed pass=True yields a cheaper 'correct answer' than measurement "
        "alone supports — the over-provisioning count is an upper bound.",
    ),
)


@dataclass(frozen=True)
class Audit:
    """The confusion grid plus the four-way error budget.

    `outside` counts picks outside the inference-valid pool. They are NOT drawn (the grid
    axis is the pool) but they are part of the decision denominator, so the annotation must
    state them rather than silently shrinking the base.
    """

    models: list[str]
    grid: np.ndarray
    exact: int
    over: int
    under: int
    unwinnable: int
    outside: int

    @property
    def decided(self) -> int:
        return self.exact + self.over + self.under


def build_audit(chosen: dict[str, str], results: dict, models_by_price: list[str]) -> Audit:
    """Cross the router's pick with the cheapest model that solved each task."""
    index = {m: i for i, m in enumerate(models_by_price)}
    grid = np.zeros((len(models_by_price), len(models_by_price)), dtype=float)
    exact = over = under = unwinnable = outside = 0
    for tid, pick in chosen.items():
        per_model = results.get(tid, {})
        best = metrics.cheapest_sufficient(per_model, models_by_price)
        if best is None:
            unwinnable += 1
            continue
        if pick not in index:
            # A winnable task the router served from outside the pool: not drawable on a grid
            # whose axes are the pool, but still one of its decisions. Count it; never drop it.
            outside += 1
            continue
        grid[index[pick], index[best]] += 1
        if pick == best:
            exact += 1
        elif index[pick] > index[best]:
            over += 1
        else:
            under += 1
    return Audit(models_by_price, grid, exact, over, under, unwinnable, outside)


def _draw_grid(ax: Axes, audit: Audit, panel_width_in: float) -> None:
    # LOG norm, and zeros masked to white. On a linear scale the 126-count diagonal cell
    # absorbs the whole ramp and every other populated cell renders as blank paper — a
    # reader sees "one cell" where there are twelve. Log keeps the ordering visible.
    peak = float(audit.grid.max()) or 1.0
    ax.imshow(
        np.ma.masked_where(audit.grid <= 0, audit.grid),
        # Blues from 18% up: its bottom stop is white, so a count of 1 at the log floor
        # was indistinguishable from an empty cell.
        cmap=ListedColormap(colormaps["Blues"](np.linspace(0.18, 1.0, 256))).with_extremes(
            bad="white"
        ),
        norm=LogNorm(vmin=1.0, vmax=max(peak, 2.0)),
        aspect="auto",
        interpolation="nearest",
    )
    n = len(audit.models)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # Full names. Splitting on the first hyphen collapsed kimi-k2.5 and kimi-k3 into two
    # identical "kimi" ticks, which is worse than a long label. The label is the canonical
    # bare identity; `audit.models` stays the join key for the grid indices.
    labels = [canonical_label(m) for m in audit.models]
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("cheapest model that solved it", fontsize=9)
    ax.set_ylabel("model the router chose", fontsize=9)
    # A cell may carry a printed count only when it is physically wide enough to hold one.
    # Below the floor the grid stays colour-only and the counts live in the manifest.
    if n and panel_width_in / n >= 0.35:
        for i in range(n):
            for j in range(n):
                value = int(audit.grid[i, j])
                if value == 0:
                    continue
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                    # Threshold on the LOG ramp, not a linear fraction of the peak: at
                    # 0.55*126 every printed count would be dark-on-dark or dark-on-light
                    # by accident rather than by contrast.
                    color="white" if audit.grid[i, j] >= peak**0.62 else "#333333",
                )
    # A model the router never chose leaves an all-white row, which reads as missing data
    # rather than as a measured zero. Shade it and say so.
    for i in range(n):
        if audit.grid[i].sum() > 0:
            continue
        ax.add_patch(
            Rectangle(
                (-0.5, i - 0.5),
                n,
                1.0,
                facecolor="#E0E0E0",
                edgecolor="white",
                linewidth=1.0,
                alpha=0.9,
                zorder=2,
            )
        )
        ax.text(
            (n - 1) / 2,
            i,
            "never chosen (0 picks)",
            ha="center",
            va="center",
            fontsize=7.0,
            color="#555555",
            style="italic",
            zorder=3,
        )
    plot_frame.panel_label(ax, "A · chosen × cheapest sufficient")


def _draw_budget(ax: Axes, audit: Audit) -> None:
    """The four-way budget as ONE full-width, normalized horizontal bar.

    A vertical stack on a 0..160 axis spent ~98% of the panel on whitespace and forced thin
    segments into a label scrum. Normalized and horizontal, the segments are shares of one
    ruler and the key can carry the counts that no longer fit on a 3%-wide slice.
    """
    parts = [
        ("exact hit", audit.exact, _EXACT),
        ("over-provisioned (paid too much)", audit.over, _OVER),
        ("under-provisioned (task lost)", audit.under, _UNDER),
        ("no model solved it (unwinnable)", audit.unwinnable, _FREE),
    ]
    total = sum(p[1] for p in parts) or 1
    left = 0.0
    handles: list[Patch] = []
    for label, value, colour in parts:
        if value == 0:
            continue
        share = value / total
        ax.barh(
            0.5,
            share,
            left=left,
            height=0.44,
            color=colour,
            edgecolor="white",
            linewidth=1.2,
            zorder=2,
        )
        if share >= 0.10:
            ax.text(
                left + share / 2.0,
                0.5,
                f"{share:.0%}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
                zorder=3,
            )
        handles.append(
            Patch(facecolor=colour, edgecolor="white", label=f"{label} — {value} ({share:.0%})")
        )
        left += share
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([f"{int(round(t * 100))}%" for t in (0.0, 0.25, 0.5, 0.75, 1.0)])
    # The bar's base is decided + unwinnable ONLY: the outside-pool picks are counted in the
    # decision total but have no cell on the pool-only grid, so they must not be read into this
    # ruler. The label states the base rather than saying "scored", which included them.
    ax.set_xlabel(
        f"share of {audit.decided + audit.unwinnable} outcomes: "
        f"{audit.decided} decidable + {audit.unwinnable} unwinnable "
        f"({audit.outside} outside-pool picks excluded)",
        fontsize=9,
    )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.34),
        ncol=2,
        fontsize=7.6,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.6,
    )
    plot_frame.panel_label(ax, "B · the error budget")


def _annotations(audit: Audit) -> Annotations:
    total = audit.decided + audit.unwinnable + audit.outside
    facts = [
        f"{audit.decided} decidable decisions, {audit.unwinnable} tasks no model solved",
        f"{audit.outside} picks outside the inference-valid pool (not drawn)",
        f"{audit.exact} exact / {audit.over} over-provisioned / {audit.under} under-provisioned",
    ]
    caveat = None
    if audit.under:
        caveat = (
            f"{audit.under} task(s) were lost to under-provisioning — those are quality, not cost."
        )
    return Annotations(
        subtitle_facts=tuple(facts),
        caveat=caveat,
        notes=(
            f"exact-hit rate {audit.exact / max(audit.decided, 1):.1%} over the decidable set; "
            f"over-provisioning is {audit.over / max(audit.decided, 1):.1%}",
            f"denominator: {total} scored decisions = {audit.decided} decidable + "
            f"{audit.unwinnable} unwinnable + {audit.outside} outside the inference-valid pool. "
            "The outside-pool picks are counted but not drawn on the pool-only grid.",
            # The budget ruler's base is smaller than the decision denominator, so its exact
            # segment (66%-ish) is not the 68.2% exact-hit rate over the decidable set. State
            # both bases and the arithmetic that separates them.
            f"the budget bar's base is {audit.decided + audit.unwinnable} outcomes "
            f"({audit.decided} decidable + {audit.unwinnable} unwinnable), so its exact segment "
            f"reads {audit.exact / max(audit.decided + audit.unwinnable, 1):.1%}; the exact-hit "
            f"rate over the decidable set alone is {audit.exact / max(audit.decided, 1):.1%}. "
            f"The {audit.outside} outside-pool picks enter neither base.",
        ),
        counts=(
            ("decisions", total),
            ("exact", audit.exact),
            ("over", audit.over),
            ("under", audit.under),
            ("outside_pool", audit.outside),
        ),
    )


def render(ctx: ctxmod.RoutingContext, chosen: dict[str, str]) -> Path | None:
    """Draw routing_decision_audit.png from the kNN selection rule's per-task picks."""
    if not chosen:
        return None
    # INFERENCE-FACING: the comparison runs over the inference-valid pool only. A benchmark-only
    # model (qwen3.7-plus, gpt-5-mini, kimi-k2.5) is not one the live router can pick, so
    # including it here would compare a serving decision against a model that can never serve.
    models = ctx.inference_valid_models
    audit = build_audit(chosen, ctx.completed.get("results", {}), models)
    if audit.decided == 0:
        return None
    size = plot_frame.WIDE
    fig = plot_frame.new_figure(size)
    axd = fig.subplot_mosaic(
        [["grid"], ["budget"]],
        height_ratios=(2.5, 1.0),
    )
    # Panel A keeps the full width for a legible grid; the normalized bar runs the same width
    # below it instead of a tall narrow strip that was mostly whitespace.
    _draw_grid(axd["grid"], audit, size.width_in * 0.9)
    _draw_budget(axd["budget"], audit)
    return plot_frame.save(
        fig,
        ctx.out_dir / "routing_decision_audit.png",
        SPEC,
        extra=_annotations(audit),
        provenance=ctx.provenance(__name__),
        size=size,
    )
