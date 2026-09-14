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
from benchmark.routing.model_universe import canonical_label

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

_BAND = "#0072B2"
_UNSOLVED = "#C62828"
# Aggregated share of picks that fall outside the inference-valid pool. The individual
# outside-pool models are NOT named here (they are named on model_validity / invalid_models).
_OUTSIDE = "#9E9E9E"
_OUTSIDE_LABEL = "outside inference-valid pool (not named)"

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
        "'No enabled model solved it' counts the inference-VALID benchmark models at their "
        "DEFAULT arms (benchmark-only and collection-only models are excluded, because the live "
        "router cannot pick them). complementarity.png counts every sampled (model, arm) column "
        "instead, so its solved-by-none figure is smaller — a different denominator, not a "
        "disagreement.",
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
    """solving-model count -> Counter of the model the router picked.

    The same `not any(...)` guard as `band_histogram`: a task the completion dropped (no
    cells at all) is UNSCORED, not unwinnable, so it must not enter the n=0 bucket and
    conflate with a task that genuinely had cells and no solver. The two panels then share
    one denominator — tasks with at least one inference-valid cell.
    """
    out: dict[int, Counter[str]] = {}
    for tid, pick in chosen.items():
        per_model = results.get(tid, {})
        if not any(m in per_model for m in models_by_price):
            continue
        n_solved = sum(1 for m in models_by_price if per_model.get(m, {}).get("pass"))
        out.setdefault(n_solved, Counter())[pick] += 1
    return out


def outside_pool_picks(alloc: dict[int, Counter[str]], models_by_price: list[str]) -> int:
    """How many picks name a model outside the inference-valid pool.

    These picks are still in the denominator (the router made them) but are drawn as ONE
    grey segment rather than named, so the stack reaches 1.0 without advertising a model
    the live router cannot choose.
    """
    pool = set(models_by_price)
    return sum(
        value for counter in alloc.values() for model, value in counter.items() if model not in pool
    )


def _allocation_note(n_solved: int, counter: Counter[str], models_by_price: list[str]) -> str:
    """One manifest line per bucket, with outside-pool picks aggregated (never named)."""
    named = {canonical_label(m): v for m, v in sorted(counter.items()) if m in set(models_by_price)}
    outside = sum(v for m, v in counter.items() if m not in set(models_by_price))
    if outside:
        named[_OUTSIDE_LABEL] = outside
    return f"{n_solved} solvers: {named}"


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
    # 1.28 reserved a fifth of the panel for a two-line label that needs about a
    # sixteenth of it.
    ax.set_ylim(0, max(values) * 1.12)
    ax.set_ylabel("tasks", fontsize=9)
    ax.legend(
        handles=[
            Patch(color=_BAND, label="solved by an enabled model"),
            Patch(color=_UNSOLVED, label="no enabled model solved it"),
        ],
        fontsize=7,
        loc="upper right",
        frameon=False,
    )
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
        # Every remaining pick named a model OUTSIDE the pool (benchmark-only or
        # collection-only). Counting them in the denominator but refusing to draw them left
        # the stack short of 1.0 — the bug this segment fixes. They are aggregated and grey
        # so the stack is complete without putting an unroutable model's name on the canvas.
        outside = 1.0 - bottom
        if outside > 1e-9:
            ax.bar(x, outside, bottom=bottom, width=0.62, color=_OUTSIDE, zorder=2)
        ax.text(x, 1.02, f"n={total}", fontsize=7.5, ha="center", va="bottom", color="#555555")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in order], fontsize=8)
    ax.set_xlabel("models that solved the task (harder ← → easier)", fontsize=9)
    # Every bar now reaches exactly 1.0 (pool shares plus the aggregated outside segment);
    # the headroom is only for the one-line n= label at 1.02.
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of the router's picks", fontsize=9)
    # Patch handles, not empty `bar` calls: an empty bar draws nothing, so matplotlib
    # gave every legend entry the default colour and the key contradicted the stacks.
    handles = [
        Patch(color=colours.get(model, "#9E9E9E"), label=canonical_label(model))
        for model in models_by_price
        if any(model in counter for counter in alloc.values())
    ]
    if outside_pool_picks(alloc, models_by_price):
        handles.append(Patch(color=_OUTSIDE, label=_OUTSIDE_LABEL))
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


def _modal_name(counter: Counter[str], models_by_price: list[str]) -> str | None:
    """The bucket's modal pick, labelled for the manifest.

    An outside-pool mode is aggregated to the same grey label the stack uses, never named:
    the individual benchmark-only/collection-only models are named on model_validity.png and
    invalid_models.png, and the canvas limitation says these picks are not named.
    """
    top = counter.most_common(1)
    if not top:
        return None
    model = top[0][0]
    return canonical_label(model) if model in set(models_by_price) else _OUTSIDE_LABEL


def _annotations(
    counts: dict[int, int],
    unsolved: int,
    unscored: int,
    alloc: dict[int, Counter[str]],
    models_by_price: list[str],
) -> Annotations:
    total = sum(counts.values()) + unsolved
    outside = outside_pool_picks(alloc, models_by_price)
    spread = ""
    if alloc:
        hardest = min(alloc)
        easiest = max(alloc)
        h_name = _modal_name(alloc[hardest], models_by_price)
        e_name = _modal_name(alloc[easiest], models_by_price)
        if h_name and e_name:
            spread = (
                f"hardest bucket ({hardest} solvers) mostly {h_name}, "
                f"easiest ({easiest} solvers) mostly {e_name}"
            )
    facts = [
        f"{total} scored tasks ({unscored} incomplete challenges excluded); "
        f"{unsolved} solved by no enabled model",
        f"{len(counts)} capability bands populated",
    ]
    if outside:
        facts.append(
            f"{outside} of panel B's picks fall outside the inference-valid pool "
            "(drawn as one grey segment, not named)"
        )
    if spread:
        facts.append(spread)
    return Annotations(
        subtitle_facts=tuple(facts),
        notes=tuple(f"band {b}: {n} tasks" for b, n in sorted(counts.items()))
        + tuple(
            _allocation_note(n, counter, models_by_price) for n, counter in sorted(alloc.items())
        )
        + (
            "Panels A and B share one denominator: tasks with at least one inference-valid "
            "cell. A task the completion dropped (no cells at all) is excluded from both, "
            "never counted as unwinnable.",
        ),
        limitations=(
            "Picks naming a model outside the inference-valid pool are counted in the "
            "denominator but aggregated into one grey segment and NOT named; the individual "
            "models are named on model_validity.png and invalid_models.png.",
        ),
        counts=(
            ("tasks", total),
            ("unsolved", unsolved),
            ("excluded", unscored),
            ("outside_pool", outside),
        ),
    )


def render(
    ctx: ctxmod.RoutingContext, bands: dict[str, int], chosen: dict[str, str]
) -> Path | None:
    """Draw task_difficulty.png from the band assignment and the router's picks."""
    results = ctx.completed.get("results", {})
    # INFERENCE-FACING: bands, allocation and the solved-by-none count run over the
    # inference-valid pool only. A benchmark-only model is not one the live router can pick.
    models = ctx.inference_valid_models
    counts, unsolved, unscored = band_histogram(results, ctx.tasks, bands, models)
    if not counts:
        return None
    alloc = allocation_by_difficulty(chosen, results, models)
    colours = plot_style.model_color_map(models)
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 2, width_ratios=(0.9, 1.15))
    _draw_bands(axes[0], counts, unsolved)
    if alloc:
        _draw_allocation(axes[1], alloc, models, colours)
    return plot_frame.save(
        fig,
        ctx.out_dir / "task_difficulty.png",
        SPEC,
        extra=_annotations(counts, unsolved, unscored, alloc, models),
        provenance=ctx.provenance(__name__),
        size=size,
    )
