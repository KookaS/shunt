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

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import metrics, plot_style
from benchmark.routing.figures import context as ctxmod

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

_EXACT = "#2E7D32"
_OVER = "#E69F00"
_UNDER = "#C62828"
_FREE = "#BDBDBD"

SPEC = FigureSpec(
    title="The shipped router's errors go both ways — it loses tasks, not just money",
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
        "shipped router is a single-shot kNN prediction with no verify-and-escalate step, "
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
    ),
    notes=(
        "Both axes are in price order and rows are the CHOSEN model, so a cell below the "
        "diagonal is over-provisioning by construction rather than by convention.",
    ),
    limitations=(
        "Cheapest-sufficient is read off the coverage-completed matrix, so a task whose "
        "cheap cell was imputed pass=True yields a cheaper 'correct answer' than measurement "
        "alone supports — the over-provisioning count is an upper bound.",
    ),
)


@dataclass(frozen=True)
class Audit:
    """The confusion grid plus the four-way error budget."""

    models: list[str]
    grid: np.ndarray
    exact: int
    over: int
    under: int
    unwinnable: int

    @property
    def decided(self) -> int:
        return self.exact + self.over + self.under


def build_audit(chosen: dict[str, str], results: dict, models_by_price: list[str]) -> Audit:
    """Cross the router's pick with the cheapest model that solved each task."""
    index = {m: i for i, m in enumerate(models_by_price)}
    grid = np.zeros((len(models_by_price), len(models_by_price)), dtype=float)
    exact = over = under = unwinnable = 0
    for tid, pick in chosen.items():
        per_model = results.get(tid, {})
        best = metrics.cheapest_sufficient(per_model, models_by_price)
        if best is None:
            unwinnable += 1
            continue
        if pick not in index:
            continue
        grid[index[pick], index[best]] += 1
        if pick == best:
            exact += 1
        elif index[pick] > index[best]:
            over += 1
        else:
            under += 1
    return Audit(models_by_price, grid, exact, over, under, unwinnable)


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
    # identical "kimi" ticks, which is worse than a long label.
    ax.set_xticklabels(audit.models, fontsize=6.5)
    ax.set_yticklabels(audit.models, fontsize=6.5)
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
    plot_frame.panel_label(ax, "A · chosen × cheapest sufficient")


def _draw_budget(ax: Axes, audit: Audit) -> None:
    parts = [
        ("exact hit", audit.exact, _EXACT),
        ("over-provisioned\n(paid too much)", audit.over, _OVER),
        ("under-provisioned\n(task lost)", audit.under, _UNDER),
        ("no model solved it\n(unwinnable)", audit.unwinnable, _FREE),
    ]
    total = sum(p[1] for p in parts) or 1
    bottom = 0.0
    drawn: list[tuple[str, int, str, float]] = []
    for label, value, colour in parts:
        ax.bar(0, value, bottom=bottom, width=0.5, color=colour, zorder=2)
        if value:
            drawn.append((label, value, colour, bottom + value / 2.0))
        bottom += value
    # A thin segment's label would sit on its neighbour's, so labels are pushed apart to a
    # minimum pitch and connected to their segment by a short leader. Walk TOP-DOWN — the
    # segments are stacked bottom-up, so iterating them in bar order pushed each label past
    # the one below it and left the topmost slice labelled at the bottom of the axes.
    pitch = total * 0.14
    next_free = float(total)
    for label, value, colour, centre in sorted(drawn, key=lambda d: -d[3]):
        label_y = min(centre, next_free)
        next_free = label_y - pitch
        ax.plot([0.26, 0.32], [centre, label_y], color="#bbbbbb", lw=0.7, zorder=3)
        ax.text(
            0.34,
            label_y,
            f"{label}\n{value}  ({value / total:.0%})",
            fontsize=8,
            va="center",
            ha="left",
            color=colour if colour != _FREE else "#666666",
        )
    ax.set_xlim(-0.42, 1.9)
    ax.set_ylim(0, total * 1.04)
    ax.set_xticks([])
    ax.set_ylabel("decisions", fontsize=9)
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "B · the error budget")


def _annotations(audit: Audit) -> Annotations:
    total = audit.decided + audit.unwinnable
    facts = [
        f"{audit.decided} decidable decisions, {audit.unwinnable} tasks no model solved",
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
        ),
        counts=(
            ("decisions", total),
            ("exact", audit.exact),
            ("over", audit.over),
            ("under", audit.under),
        ),
    )


def render(ctx: ctxmod.RoutingContext, chosen: dict[str, str]) -> Path | None:
    """Draw routing_decision_audit.png from the shipped router's per-task picks."""
    if not chosen:
        return None
    audit = build_audit(chosen, ctx.completed.get("results", {}), ctx.models_by_price)
    if audit.decided == 0:
        return None
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 2, width_ratios=(1.25, 0.85))
    _draw_grid(axes[0], audit, size.width_in * 0.55)
    _draw_budget(axes[1], audit)
    plot_style.fit_end_labels(axes[1])
    return plot_frame.save(
        fig,
        ctx.out_dir / "routing_decision_audit.png",
        SPEC,
        extra=_annotations(audit),
        provenance=ctx.provenance(__name__),
        size=size,
    )
