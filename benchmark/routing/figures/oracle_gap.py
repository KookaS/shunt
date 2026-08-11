"""oracle_gap.png — what is left to win, where the win comes from, and who fails to take it."""

# Three findings live here and in no CSV.
#
# 1. THE WIN IS MECHANISM, NOT PREDICTION. `metrics.compute_cost_decomposition` (an
#    Oaxaca-Blinder split that nothing drew) separates the saving into a PRICE effect —
#    the same work billed at a cheaper model's rate — from a VOLUME effect and their
#    interaction. Almost all of it is price: the router is buying a cheaper tariff, not
#    predicting which task needs which model.
# 2. ARM-LEVEL HINDSIGHT BUYS NOTHING. The arm oracle, allowed to pick the best REALISED
#    reasoning arm per task, lands a hair below the model-level oracle. Choosing the
#    reasoning effort is worth approximately zero on this corpus.
# 3. AN ONLINE LEARNER LOSES TO A FIXED CHEAP POLICY. The inline optimistic-greedy bandit
#    ends with MORE regret than always-cheapest. That is a negative result worth keeping.
#
# The gamma panel is why the regret ladder is admissible at all. `docs/routing.md` rejects
# any particular dollars-per-pass exchange rate as indefensible — so the figure does not
# claim one. It sweeps gamma across three orders of magnitude and shows the ORDERING does
# not move, which is a claim about the ranking rather than about the rate.

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from matplotlib.patches import Patch

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import plot_style
from benchmark.routing.figures import context as ctxmod
from benchmark.routing.plot_style import usd
from benchmark.routing.strategy_class import StrategyClass, classify

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from matplotlib.axes import Axes

Decisions = list[tuple[str, str, bool, float]]

_PRICE = "#0072B2"
_VOLUME = "#56B4E9"
_IXN = "#CC79A7"
_TOTAL = "#333333"
_BOUND = "#9E9E9E"
_BLOCKED = "#56B4E9"
_CONTROL = "#E69F00"
_ROUTER = "#009E73"
_WORSE = "#C62828"

# Panels B and C carry every class, so the colour has to say which — a two-colour
# grey/green key made "not grey" read as "you could run this", which is false for the
# blocked series and for the control. Same palette as cost_quality_frontier.png.
_CLASS_COLOUR: Final[Mapping[StrategyClass, str]] = MappingProxyType(
    {
        StrategyClass.BOUND: _BOUND,
        StrategyClass.BLOCKED: _BLOCKED,
        StrategyClass.CONTROL: _CONTROL,
        StrategyClass.LIVE: _ROUTER,
    }
)
_CLASS_LABEL: Final[Mapping[StrategyClass, str]] = MappingProxyType(
    {
        StrategyClass.BOUND: "bound — unreachable by design",
        StrategyClass.BLOCKED: "blocked — not deployable today",
        StrategyClass.CONTROL: "control — never shippable",
        StrategyClass.LIVE: "live — runs in production today",
    }
)

# The exchange rate the regret metric is quoted at is indefensible as a point value, so the
# ranking is checked across this range instead. The endpoints bracket "cost is noise" and
# "cost dominates quality".
GAMMA_GRID: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.33)

SPEC = FigureSpec(
    title="The saving is a cheaper tariff, not a better prediction",
    reading=(
        "Left: the cost saving of the router against fixed-frontier, split by "
        "Oaxaca-Blinder into a price effect (cheaper tokens), a volume effect (fewer "
        "tokens) and their interaction, over the tasks where BOTH arms passed. Middle: "
        "cumulative regret against the hindsight oracle, lower is better, with 95% "
        "bootstrap intervals where the summary carries them; bar colour is the strategy's "
        "class — green runs live today, blue is blocked, orange is a control that must "
        "never ship, grey is a bound no router can reach — and a red outline marks a bar a "
        "fixed always-cheapest policy already beats. Right: the same ranking recomputed "
        "across three orders of magnitude of the cost/quality exchange rate, coloured the "
        "same way."
    ),
    goal=(
        "Read the left panel for what routing is actually doing — if price dominates, the "
        "value is in the price list, and a fixed cheap policy captures most of it without "
        "any prediction. Then read the right panel: a flat set of lines means the middle "
        "panel's ordering does not depend on the exchange rate nobody can defend."
    ),
    definitions=(
        (
            "price effect",
            "the saving from billing the SAME token volume at a cheaper model's rate.",
        ),
        ("volume effect", "the saving from producing FEWER tokens at the same rate."),
        (
            "regret",
            "reward the hindsight oracle collected that this strategy did not; reward is "
            "1 for a pass, 0 for a fail, minus gamma x cost in dollars.",
        ),
        (
            "arm oracle",
            "hindsight over the reasoning ARM as well as the model — the ceiling for "
            "reasoning-effort routing given the arms actually sampled.",
        ),
    ),
    notes=(
        "The decomposition is computed only over tasks where both arms were measured "
        "(scorable — never a coverage-gap, censored, or imputed fill) and both passed, so it "
        "is a cost comparison at genuinely equal quality on those tasks.",
    ),
    limitations=(
        "The bandit is an illustrative inline learner drawn for this figure only, not a "
        "shipped routing strategy. It shows that a naive learner loses here; it does not "
        "show that every learner would.",
        "The arm series exist only where more than one arm per model was sampled; the "
        "coverage is sparse by design.",
    ),
)


def _reward(passed: bool, cost: float, gamma: float) -> float:
    return (1.0 if passed else 0.0) - gamma * cost


def total_regret(series: Decisions, oracle: Decisions, gamma: float, excluded: set[str]) -> float:
    """Sum over scorable tasks of (oracle reward - strategy reward) at this gamma."""
    return float(
        sum(
            _reward(od[2], od[3], gamma) - _reward(sd[2], sd[3], gamma)
            for sd, od in zip(series, oracle, strict=True)
            if sd[0] not in excluded
        )
    )


def _draw_waterfall(ax: Axes, dec: dict[str, float]) -> None:
    parts = [
        ("price\neffect", dec["price_savings"], _PRICE),
        ("volume\neffect", dec["volume_savings"], _VOLUME),
        ("inter-\naction", dec["interaction"], _IXN),
    ]
    total = sum(value for _label, value, _c in parts)
    running = 0.0
    levels = [0.0]
    for i, (_label, value, colour) in enumerate(parts):
        ax.bar(i, value, bottom=running, width=0.62, color=colour, zorder=2)
        running += value
        levels.append(running)
        ax.plot([i - 0.31, i + 0.31], [running, running], color="#999999", lw=0.8, zorder=3)
    ax.bar(len(parts), total, width=0.62, color=_TOTAL, zorder=2)
    levels.append(total)
    ax.axhline(0.0, color="#999999", lw=0.9, zorder=1)

    # Headroom is computed from the RUNNING extremes, not from the total: a negative
    # interaction term takes the waterfall above the final bar, and a limit set from the
    # total alone pushed those labels off the canvas and into the title band.
    lo, hi = min(levels), max(levels)
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - 0.14 * span, hi + 0.26 * span)
    running = 0.0
    for i, (_label, value, colour) in enumerate(parts):
        ax.text(
            i,
            max(running, running + value) + span * 0.03,
            f"{usd(value, 2)}\n{value / total:.0%}",
            fontsize=7.5,
            ha="center",
            va="bottom",
            color=colour,
        )
        running += value
    ax.text(
        len(parts),
        total + span * 0.03,
        f"{usd(total, 2)}\ntotal",
        fontsize=7.5,
        ha="center",
        va="bottom",
        color=_TOTAL,
    )
    ax.set_xticks(range(len(parts) + 1))
    ax.set_xticklabels([p[0] for p in parts] + ["decomposed\nsaving"], fontsize=8)
    ax.set_ylabel("dollars saved vs fixed-frontier", fontsize=9)
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "A · where the saving comes from")


def _loses_to_cheap(name: str, value: float, cheap: float | None) -> bool:
    """True for a shippable-in-principle strategy that a fixed cheap policy beats."""
    # A BOUND is excluded: it reads the answer, so "it beat always-cheapest" is not news.
    return classify(name).cls is not StrategyClass.BOUND and cheap is not None and value > cheap


def _regret_legend(ax: Axes, names: list[str], finals: dict[str, float]) -> None:
    """One patch per class actually drawn, plus the red outline only where it fires."""
    cheap = finals.get("Always-Cheap")
    drawn = {classify(n).cls for n in names}
    handles = [
        Patch(color=_CLASS_COLOUR[cls], label=_CLASS_LABEL[cls])
        for cls in StrategyClass
        if cls in drawn
    ]
    if any(_loses_to_cheap(n, finals[n], cheap) for n in names):
        # Outline, not fill: the fill is reserved for the class, so a warning that
        # recoloured the bar would delete the one thing the colour is there to say.
        handles.append(
            Patch(
                facecolor="white",
                edgecolor=_WORSE,
                linewidth=1.6,
                label="red outline: worse than always-cheapest",
            )
        )
    ax.legend(handles=handles, fontsize=7, loc="center right", frameon=True, framealpha=0.92)


def _draw_regret(ax: Axes, finals: dict[str, float], cis: dict[str, tuple[float, float]]) -> None:
    names = sorted(finals, key=lambda n: finals[n])
    ys = list(range(len(names)))[::-1]
    cheap = finals.get("Always-Cheap")
    for y, name in zip(ys, names, strict=True):
        value = finals[name]
        warn = _loses_to_cheap(name, value, cheap)
        ax.barh(
            y,
            value,
            height=0.6,
            color=_CLASS_COLOUR[classify(name).cls],
            edgecolor=_WORSE if warn else "none",
            linewidth=1.6 if warn else 0.0,
            zorder=2,
        )
        interval = cis.get(name)
        if interval:
            ax.plot([interval[0], interval[1]], [y, y], color="#444444", lw=1.2, zorder=4)
        # Past the CI whisker, not past the bar: a label at the bar end sat on top of the
        # interval it is meant to be read alongside.
        ax.text(
            max(value, interval[1] if interval else value)
            + (max(finals.values()) - min(finals.values())) * 0.02,
            y,
            f"{value:.2f}",
            fontsize=7.5,
            va="center",
            color="#333333",
        )
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=7.5)
    lo, hi = min(finals.values()), max(finals.values())
    ax.set_xlim(min(lo * 1.2, -0.4), hi * 1.22)
    ax.axvline(0.0, color="#bbbbbb", lw=0.8, zorder=1)
    ax.set_xlabel("cumulative regret vs the hindsight oracle (lower is better)", fontsize=9)
    _regret_legend(ax, names, finals)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "B · regret ladder")


def _draw_gamma(ax: Axes, ranks: dict[str, list[int]]) -> None:
    xs = list(GAMMA_GRID)
    for name, series in sorted(ranks.items(), key=lambda kv: kv[1][0]):
        colour = _CLASS_COLOUR[classify(name).cls]
        ax.plot(xs, series, "o-", color=colour, lw=1.4, ms=4, zorder=3, alpha=0.9)
        ax.annotate(
            name,
            xy=(xs[-1], series[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=7,
            va="center",
            ha="left",
            color=colour,
        )
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{g:g}" for g in xs], fontsize=7.5)
    ax.minorticks_off()
    ax.set_xlim(xs[0] * 0.7, xs[-1] * 6.0)
    ax.invert_yaxis()
    ax.set_yticks(sorted({r for s in ranks.values() for r in s}))
    ax.set_xlabel("gamma — dollars per unit of quality", fontsize=9)
    ax.set_ylabel("rank by regret (1 = best)", fontsize=9)
    ax.grid(color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "C · rank by regret, across gamma")


def _annotations(
    dec: dict[str, float], finals: dict[str, float], ranks: dict[str, list[int]], gamma: float
) -> Annotations:
    mechanism = dec["price_pct"] + dec["interaction_pct"]
    stable = all(len(set(series)) == 1 for series in ranks.values())
    facts = [
        f"price {dec['price_pct']:.1f}% + interaction {dec['interaction_pct']:.1f}% = "
        f"{mechanism:.1f}% mechanism, volume {dec['volume_pct']:.1f}% "
        f"({dec['n_eq_pass']} both-pass tasks)",
        f"regret quoted at gamma={gamma:g}; ordering "
        f"{'IDENTICAL' if stable else 'MOVES'} across gamma {GAMMA_GRID[0]:g}-{GAMMA_GRID[-1]:g}",
    ]
    notes = [f"{name}: regret {value:.4f}" for name, value in sorted(finals.items())]
    if stable:
        notes.append(
            "Every strategy holds the same rank at every gamma on the grid, so the ladder's "
            "ordering is a statement about quality-at-cost and not about the exchange rate."
        )
    caveat = None
    cheap = finals.get("Always-Cheap")
    # BOUNDs are excluded by CLASS, not by a substring of their name: "is this an oracle"
    # was answered by `"oracle" in name`, which would have silently admitted any future
    # bound whose name lacks the word.
    losers = [n for n, v in finals.items() if _loses_to_cheap(n, v, cheap)]
    if losers:
        # Agrees in number: a list of two read "Arm-bandit, Tier-Classifier carries MORE",
        # which is the kind of seam a generated caption shows on a canvas nobody re-reads.
        named = ", ".join(sorted(losers))
        verb = "carries" if len(losers) == 1 else "carry"
        caveat = f"{named} {verb} MORE regret than always-cheapest."
    return Annotations(
        subtitle_facts=tuple(facts),
        caveat=caveat,
        notes=tuple(notes),
        counts=(("both_pass_tasks", int(dec["n_eq_pass"])), ("series", len(finals))),
    )


def render(  # noqa: PLR0913 (one argument per already-computed series; see report.main)
    ctx: ctxmod.RoutingContext,
    decomposition: dict[str, float],
    series: dict[str, Decisions],
    oracle: Decisions,
    excluded: set[str],
    gamma: float,
) -> Path | None:
    """Draw oracle_gap.png from the cost decomposition and the regret series."""
    if not decomposition.get("n_eq_pass") or not series:
        return None
    cis = {
        str(r["strategy"]): (float(r["CumReg_ci_lower"]), float(r["CumReg_ci_upper"]))
        for r in ctx.rows
        if r.get("CumReg_ci_lower") not in (None, "")
    }
    finals = {name: total_regret(d, oracle, gamma, excluded) for name, d in series.items()}
    ranks: dict[str, list[int]] = {name: [] for name in series}
    for g in GAMMA_GRID:
        at_g = {name: total_regret(d, oracle, g, excluded) for name, d in series.items()}
        order = sorted(at_g, key=lambda n: at_g[n])
        for position, name in enumerate(order, start=1):
            ranks[name].append(position)
    size = plot_frame.WIDE_TALL
    fig, axes = plot_frame.subplots(size, 1, 3, width_ratios=(0.8, 1.15, 1.0))
    _draw_waterfall(axes[0], decomposition)
    _draw_regret(axes[1], finals, cis)
    _draw_gamma(axes[2], ranks)
    for ax in (axes[1], axes[2]):
        plot_style.fit_end_labels(ax)
    return plot_frame.save(
        fig,
        ctx.out_dir / "oracle_gap.png",
        SPEC,
        extra=_annotations(decomposition, finals, ranks, gamma),
        provenance=ctx.provenance(__name__),
        size=size,
    )
