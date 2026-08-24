"""task_difficulty.png — how hard the corpus is, and whether the router tracks it."""

# Merges `capability_distribution` (which capability band each task actually needs) with
# `chosen_arm_vs_difficulty` (what the router picked against how many models solved the
# task). Drawn together they answer one question rather than two halves of it: the corpus
# has a difficulty gradient, and the allocation either follows it or it does not.

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from matplotlib.patches import Patch

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import plot_style
from benchmark.routing.figures import context as ctxmod

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

_BAND = "#0072B2"
_UNSOLVED = "#C62828"

SPEC = FigureSpec(
    title="The kNN selection rule sends most of every difficulty bucket to the cheapest model",
    reading=(
        "Left: how many tasks each capability band is the cheapest sufficient answer for, "
        "weakest band on the left, plus the tasks no enabled model solved. Right: for each "
        "count of solving models — the corpus's own difficulty measure — the share of tasks "
        "the kNN selection rule sent to each model, as stacked bars with the task count "
        "above."
    ),
    goal=(
        "Compare the stacks across the right panel's buckets. The rule plotted here is kNN: it "
        "predicts ONCE from the neighbourhood and does not escalate, so a stack that barely "
        "moves from the hardest bucket to the easiest means the prediction is barely "
        "conditioning on difficulty at all. Read embedding_signal.png for why — the input it "
        "predicts from carries almost no routable signal."
    ),
    definitions=(
        (
            "capability band",
            "models grouped by derived capability rank; a task's band is the weakest band "
            "containing a model that solved it.",
        ),
        (
            "solving models",
            "how many enabled models solved the task. Zero means unwinnable, all means free.",
        ),
    ),
    notes=(
        "Bands and solving-model counts are read off the coverage-completed matrix, the same "
        "matrix every strategy is scored on.",
    ),
    limitations=(
        "An imputed cell is always a pass, so a task's band is a LOWER bound on the "
        "capability it truly needs and the solving-model count is an upper bound.",
        "The right panel is NOT circular for the rule plotted — kNN decides before any "
        "outcome for this task exists — but it is not independent either: the neighbours it "
        "reads and the solving-model count it is plotted against come from one matrix.",
        "'No enabled model solved it' counts the six models at their DEFAULT arms. "
        "complementarity.png counts every sampled (model, arm) column instead, so its "
        "solved-by-none figure is smaller — a different denominator, not a disagreement.",
    ),
)


def band_histogram(
    results: dict, tasks: list[str], bands: dict[str, int], models_by_price: list[str]
) -> tuple[dict[int, int], int, int]:
    """(band -> tasks whose cheapest solver sits in it, no-solver tasks, tasks not scored)."""
    counts: Counter[int] = Counter()
    unsolved = 0
    unscored = 0
    for tid in tasks:
        per_model = results.get(tid, {})
        # A task the completion dropped (an incomplete challenge) has NO cells at all.
        # Counting it as "no model solved it" turned 25 excluded challenges into a
        # capability claim and inflated the unsolved bar from 6 to 31.
        if not any(m in per_model for m in models_by_price):
            unscored += 1
            continue
        solved = [m for m in models_by_price if per_model.get(m, {}).get("pass")]
        if not solved:
            unsolved += 1
            continue
        cheapest = solved[0]
        band = bands.get(cheapest)
        if band is not None:
            counts[band] += 1
    return dict(counts), unsolved, unscored


def allocation_by_difficulty(
    chosen: dict[str, str], results: dict, models_by_price: list[str]
) -> dict[int, Counter[str]]:
    """solving-model count -> Counter of the model the router picked."""
    out: dict[int, Counter[str]] = {}
    for tid, pick in chosen.items():
        per_model = results.get(tid, {})
        n_solved = sum(1 for m in models_by_price if per_model.get(m, {}).get("pass"))
        out.setdefault(n_solved, Counter())[pick] += 1
    return out


def _draw_bands(ax: Axes, counts: dict[int, int], unsolved: int) -> None:
    order = sorted(counts)
    xs = list(range(len(order) + (1 if unsolved else 0)))
    values = [counts[b] for b in order] + ([unsolved] if unsolved else [])
    colours = [_BAND] * len(order) + ([_UNSOLVED] if unsolved else [])
    labels = [f"band {b}" for b in order] + (["no enabled model\nsolved it"] if unsolved else [])
    total = sum(values) or 1
    for x, value, colour in zip(xs, values, colours, strict=True):
        ax.bar(x, value, width=0.62, color=colour, zorder=2)
        ax.text(
            x,
            value + total * 0.015,
            f"{value}\n{value / total:.0%}",
            fontsize=7.5,
            ha="center",
            va="bottom",
            color=colour,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max(values) * 1.28)
    ax.set_ylabel("tasks", fontsize=9)
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "A · cheapest band that solves the task")


def _draw_allocation(
    ax: Axes, alloc: dict[int, Counter[str]], models_by_price: list[str], colours: dict[str, str]
) -> None:
    order = sorted(alloc)
    xs = list(range(len(order)))
    for x, n_solved in zip(xs, order, strict=True):
        counter = alloc[n_solved]
        total = sum(counter.values()) or 1
        bottom = 0.0
        for model in models_by_price:
            share = counter.get(model, 0) / total
            if share <= 0:
                continue
            ax.bar(
                x, share, bottom=bottom, width=0.62, color=colours.get(model, "#9E9E9E"), zorder=2
            )
            bottom += share
        ax.text(x, 1.02, f"n={total}", fontsize=7.5, ha="center", va="bottom", color="#555555")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in order], fontsize=8)
    ax.set_xlabel("models that solved the task (harder ← → easier)", fontsize=9)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("share of the router's picks", fontsize=9)
    # Patch handles, not empty `bar` calls: an empty bar draws nothing, so matplotlib
    # gave every legend entry the default colour and the key contradicted the stacks.
    handles = [
        Patch(color=colours.get(model, "#9E9E9E"), label=model)
        for model in models_by_price
        if any(model in counter for counter in alloc.values())
    ]
    ax.legend(
        handles=handles,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=3,
        frameon=False,
    )
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "B · what the router picked, by difficulty")


def _annotations(
    counts: dict[int, int], unsolved: int, unscored: int, alloc: dict[int, Counter[str]]
) -> Annotations:
    total = sum(counts.values()) + unsolved
    spread = ""
    if alloc:
        hardest = min(alloc)
        easiest = max(alloc)
        h_top = alloc[hardest].most_common(1)
        e_top = alloc[easiest].most_common(1)
        if h_top and e_top:
            spread = (
                f"hardest bucket ({hardest} solvers) mostly {h_top[0][0]}, easiest "
                f"({easiest} solvers) mostly {e_top[0][0]}"
            )
    facts = [
        f"{total} scored tasks ({unscored} incomplete challenges excluded); "
        f"{unsolved} solved by no enabled model",
        f"{len(counts)} capability bands populated",
    ]
    if spread:
        facts.append(spread)
    return Annotations(
        subtitle_facts=tuple(facts),
        notes=tuple(f"band {b}: {n} tasks" for b, n in sorted(counts.items()))
        + tuple(f"{n} solvers: {dict(sorted(c.items()))}" for n, c in sorted(alloc.items())),
        counts=(("tasks", total), ("unsolved", unsolved), ("excluded", unscored)),
    )


def render(
    ctx: ctxmod.RoutingContext, bands: dict[str, int], chosen: dict[str, str]
) -> Path | None:
    """Draw task_difficulty.png from the band assignment and the router's picks."""
    results = ctx.completed.get("results", {})
    counts, unsolved, unscored = band_histogram(results, ctx.tasks, bands, ctx.models_by_price)
    if not counts:
        return None
    alloc = allocation_by_difficulty(chosen, results, ctx.models_by_price)
    colours = plot_style.model_color_map(ctx.models_by_price)
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 2, width_ratios=(0.9, 1.15))
    _draw_bands(axes[0], counts, unsolved)
    if alloc:
        _draw_allocation(axes[1], alloc, ctx.models_by_price, colours)
    return plot_frame.save(
        fig,
        ctx.out_dir / "task_difficulty.png",
        SPEC,
        extra=_annotations(counts, unsolved, unscored, alloc),
        provenance=ctx.provenance(__name__),
        size=size,
    )
