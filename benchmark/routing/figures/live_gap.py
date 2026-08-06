"""live_gap.png — how much of the headroom above the shipped router is buyable today."""

# Every other figure answers "which strategy wins". This one answers the question the
# four-class taxonomy exists for: of the distance between what SHIPS and what the corpus
# proves is possible, how much is engineering work and how much is unreachable in
# principle. The three pieces are:
#
#   A. At the bound's quality, what does each CLASS charge? The cheapest LIVE way to buy
#      the bound's pass rate, versus the blocked strategies, versus the bound itself. The
#      live-to-blocked span is wiring work; the blocked-to-bound span is not buyable at all.
#   B. What is each class FOR? A bound brackets the result from above, a control proves
#      the measurement is about something, a blocked strategy is a costed to-do. None of
#      them may anchor a headline — which is exactly why they are drawn apart from the
#      frontier rather than dropped.
#
# It deliberately does NOT re-draw the cost-quality plane; cost_quality_frontier.png owns
# that, and four figures once drew it at once.

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing.figures import context as ctxmod
from benchmark.routing.plot_style import usd
from benchmark.routing.strategy_class import StrategyClass, classify

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from matplotlib.axes import Axes

_LIVE = "#009E73"
_BLOCKED = "#56B4E9"
_CONTROL = "#E69F00"
_BOUND = "#9E9E9E"
_SPAN_OPEN = "#0072B2"
_SPAN_SHUT = "#B71C1C"

_CLASS_COLOUR: Final[Mapping[StrategyClass, str]] = MappingProxyType(
    {
        StrategyClass.LIVE: _LIVE,
        StrategyClass.BLOCKED: _BLOCKED,
        StrategyClass.CONTROL: _CONTROL,
        StrategyClass.BOUND: _BOUND,
    }
)

_ROLE: Final[Mapping[StrategyClass, str]] = MappingProxyType(
    {
        StrategyClass.LIVE: "you can run it",
        StrategyClass.BLOCKED: "a costed to-do",
        StrategyClass.CONTROL: "keeps the result interpretable",
        StrategyClass.BOUND: "brackets the result from above",
    }
)

# A strategy counts as "at the bound's quality" within this many percentage points of the
# best bound. Wider and a genuinely worse policy sneaks into the equal-quality ladder;
# narrower and floating-point ties on identical pass counts drop out of it.
_BAND_PP: Final[float] = 1.0


def _rows(ctx: ctxmod.RoutingContext) -> list[tuple[str, StrategyClass, float, float]]:
    """(name, class, total cost, pass rate) for every scored, priced strategy row."""
    out = []
    for row in ctx.rows:
        name = str(row["strategy"])
        cost, perf = float(row["TotalCost"]), float(row["AvgPerf%"])
        if cost > 0 and int(float(row.get("n_tasks", 0) or 0)) > 0:
            out.append((name, classify(name).cls, cost, perf))
    return out


def _at_bound_quality(
    rows: list[tuple[str, StrategyClass, float, float]],
) -> tuple[float, list[tuple[str, StrategyClass, float, float]]]:
    """The bound's pass rate, and every strategy within _BAND_PP of it, cheapest first."""
    bounds = [r for r in rows if r[1] is StrategyClass.BOUND]
    if not bounds:
        return 0.0, []
    ceiling = max(r[3] for r in bounds)
    band = [r for r in rows if r[3] >= ceiling - _BAND_PP]
    return ceiling, sorted(band, key=lambda r: r[2])


def _cheapest(band: list[tuple[str, StrategyClass, float, float]], cls: StrategyClass) -> float:
    """Cheapest total cost in `band` for one class, or 0.0 when the class is absent."""
    costs = [r[2] for r in band if r[1] is cls]
    return min(costs) if costs else 0.0


SPEC = FigureSpec(
    title="Most of the headroom above the shipped router is blocked, not impossible",
    reading=(
        "Left: every strategy that reaches the bound's pass rate within one percentage "
        "point, as total spend on a log axis, cheapest at the bottom, coloured by class. "
        "The blue bracket is the span between the cheapest LIVE way to buy that quality and "
        "the cheapest BLOCKED one — engineering work, not physics. The red bracket is the "
        "span from there down to the bound, which no strategy of any class can cross. "
        "Right: how many strategies each class contributes and the best pass rate it "
        "reaches, with the reason that class is kept in the corpus."
    ),
    goal=(
        "Read the two brackets against each other. A large blue span and a small red one "
        "means the shipped router's deficit is a backlog item; the reverse would mean the "
        "corpus has been squeezed and the remaining distance is a property of the models, "
        "not of the routing. Neither bracket is a result you can deploy — the whole point "
        "of separating the classes is that only the green bars are purchasable."
    ),
    definitions=(
        (
            "bound",
            "a strategy that reads the query task's own realised outcome. Unreachable BY "
            "DESIGN; it exists to say how much is left, never to be shipped.",
        ),
        (
            "blocked",
            "realizable in principle, with a named blocker and a path to live recorded in "
            "benchmark/routing/strategy_class.py. A costed to-do, not a result.",
        ),
        (
            "control",
            "exists so the other numbers mean something — a strategy the measurement is "
            "compared against, which must never ship.",
        ),
        (
            "at the bound's quality",
            "pass rate within 1.0pp of the best bound's. A cost comparison across the band "
            "is therefore an equal-quality comparison to within that tolerance.",
        ),
    ),
    notes=(
        "Costs are the same naive per-task totals every other routing figure uses, so the "
        "brackets are comparable with cost_quality_frontier.png.",
    ),
    limitations=(
        "The blue bracket is what the BLOCKED strategies measured here would buy IF their "
        "blockers were removed. Their blockers are structural (cache-safety, an offline-fit "
        "input), so the live mechanism that replaces them may land nowhere near this span.",
        "Only strategies inside the quality band appear on the left panel. A cheap strategy "
        "that gives up quality is not a smaller version of this gap — read the frontier "
        "figure for that trade.",
        "The bound reads realised outcomes on the SAME corpus it is measured on, so it is a "
        "ceiling for this task set, not a general one.",
    ),
)


def _draw_band(ax: Axes, band: list[tuple[str, StrategyClass, float, float]]) -> None:
    ys = list(range(len(band)))
    for y, (_name, cls, cost, perf) in zip(ys, band, strict=True):
        ax.barh(y, cost, height=0.6, color=_CLASS_COLOUR[cls], zorder=2)
        ax.text(
            cost * 1.06,
            y,
            f"{usd(cost)} · {perf:.2f}%",
            fontsize=7.5,
            va="center",
            ha="left",
            color="#333333",
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in band], fontsize=8)
    ax.set_xscale("log")
    costs = [r[2] for r in band]
    ax.set_xlim(min(costs) * 0.45, max(costs) * 4.2)
    # Two bracket rows plus their labels live below the bars; -1.25 is what keeps the
    # lower label inside the canvas instead of clipped by the axis.
    ax.set_ylim(-1.25, len(band) - 0.25)
    ax.set_xlabel("total spend at the bound's quality (USD, log)", fontsize=9)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "A · what the bound's quality costs, by class")


def _draw_spans(ax: Axes, band: list[tuple[str, StrategyClass, float, float]]) -> None:
    """The two brackets: live→blocked is wiring work, blocked→bound is not buyable."""
    live = _cheapest(band, StrategyClass.LIVE)
    blocked = _cheapest(band, StrategyClass.BLOCKED)
    bound = _cheapest(band, StrategyClass.BOUND)
    # Drawn below the bars, not across them: a bracket at bar height overlapped the value
    # labels and SHUNT_PLOT_STRICT refuses to write a figure with an overlapping artist.
    spans = [
        (blocked, live, _SPAN_OPEN, -0.45, "unlocked by wiring the blocked ones"),
        (bound, blocked, _SPAN_SHUT, -0.85, "not buyable by any strategy"),
    ]
    for lo, hi, colour, y, label in spans:
        if lo <= 0 or hi <= 0 or hi <= lo:
            continue
        ax.plot([lo, hi], [y, y], color=colour, lw=1.6, zorder=5)
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.07, y + 0.07], color=colour, lw=1.6, zorder=5)
        ax.text(
            (lo * hi) ** 0.5,
            y - 0.10,
            f"{usd(hi - lo)} — {label}",
            fontsize=7.5,
            ha="center",
            va="top",
            color=colour,
        )


def _draw_census(ax: Axes, rows: list[tuple[str, StrategyClass, float, float]]) -> None:
    present = [cls for cls in StrategyClass if any(r[1] is cls for r in rows)]
    ys = list(range(len(present)))[::-1]
    widest = 0
    for y, cls in zip(ys, present, strict=True):
        members = [r for r in rows if r[1] is cls]
        widest = max(widest, len(members))
        ax.barh(y, len(members), height=0.55, color=_CLASS_COLOUR[cls], zorder=2)
        ax.text(
            len(members) + 0.12,
            y,
            f"best {max(r[3] for r in members):.2f}% · {_ROLE[cls]}",
            fontsize=7.5,
            va="center",
            ha="left",
            color="#333333",
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([cls.value for cls in present], fontsize=8)
    # Room for the label beside the longest bar, scaled to that bar rather than to the
    # strategy total — an axis sized by the total left the bars stranded at the left edge.
    ax.set_xlim(0, max(widest, 1) * 2.9)
    ax.set_ylim(-0.6, len(present) - 0.4)
    ax.set_xlabel("strategies in this class", fontsize=9)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "B · what each class is for")


def _annotations(
    rows: list[tuple[str, StrategyClass, float, float]],
    band: list[tuple[str, StrategyClass, float, float]],
    ceiling: float,
) -> Annotations:
    live = _cheapest(band, StrategyClass.LIVE)
    blocked = _cheapest(band, StrategyClass.BLOCKED)
    bound = _cheapest(band, StrategyClass.BOUND)
    total = live - bound
    facts = [
        f"{len(band)} of {len(rows)} strategies reach {ceiling:.2f}% ± {_BAND_PP:.0f}pp",
        f"cheapest live {usd(live)} · cheapest blocked {usd(blocked)} · bound {usd(bound)}",
    ]
    caveat = None
    if total > 0 and blocked > 0:
        share = (live - blocked) / total
        facts.append(f"blocked strategies hold {share:.0%} of the live-to-bound headroom")
        caveat = (
            f"{share:.0%} of the headroom sits behind a blocker, so it is a to-do, not a "
            "measured saving."
        )
    notes = [f"{name}: {usd(cost)}, {perf:.2f}% ({cls.value})" for name, cls, cost, perf in band]
    return Annotations(
        subtitle_facts=tuple(facts),
        caveat=caveat,
        notes=tuple(notes),
        counts=(("in_band", len(band)), ("strategies", len(rows))),
    )


def render(ctx: ctxmod.RoutingContext) -> Path | None:
    """Draw live_gap.png, or None when no bound row prices the headroom."""
    rows = _rows(ctx)
    ceiling, band = _at_bound_quality(rows)
    if len(band) < 2:
        return None
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 2, width_ratios=(1.35, 1.0))
    _draw_band(axes[0], band)
    _draw_spans(axes[0], band)
    _draw_census(axes[1], rows)
    return plot_frame.save(
        fig,
        ctx.out_dir / "live_gap.png",
        SPEC,
        extra=_annotations(rows, band, ceiling),
        provenance=ctx.provenance(__name__),
        size=size,
    )
