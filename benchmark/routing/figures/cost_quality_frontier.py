"""cost_quality_frontier.png — one cost-quality plane, replacing four that drew the same one."""

# `pareto_scatter`, `cost_savings`, `cost_quality_equal` and `strategy_comparison` were four
# renderings of a single (total cost, pass rate) scatter over the same strategy rows. They
# disagreed on the Pareto definition, on whether hindsight oracles were peers, and — worst —
# `cost_savings` headlined a saving ratio taken at UNEQUAL quality. One plane, one Pareto
# definition, one hull.
#
# Two deliberate departures from the figures this replaces. The x axis is LOG, because the
# strategies span two decades and a linear axis crushed five of them into the left margin.
# The y axis starts below the worst point rather than at zero, because a 20-point band
# rendered on a 0-100 axis is a flat line — the FIGURE is the comparison, and the axis is
# labelled so nobody reads the range as a full scale.

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import numpy as np

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import plot_style
from benchmark.routing.figures import context as ctxmod
from benchmark.routing.plot_style import LabelPoint, usd
from benchmark.routing.strategy_class import StrategyClass, classify, is_live

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from matplotlib.axes import Axes

_PARETO = "#009E73"
_FRONTIER = "#D55E00"
_OTHER = "#9E9E9E"
_BLOCKED = "#56B4E9"
_HULL = "#009E73"

SPEC = FigureSpec(
    title="Every strategy on one plane — the frontier is drawn over live ones only",
    reading=(
        "x is total dollars spent over the scored task set, on a log axis because the "
        "strategies span two decades. y is pass rate in percent with a 95% Wilson interval. "
        "Marker SHAPE carries what a strategy is: circles and the orange diamond can run in "
        "production today, blue squares are blocked (realizable but not cache-safe, or not "
        "wired), an X is a control that must never ship, and a star is a bound that is "
        "unreachable by design. Only the live points enter the Pareto test and the shaded "
        "mixture region — a frontier anchored on a strategy the router rejects at boot "
        "describes an operating point nobody can buy."
    ),
    goal=(
        "A router earns its existence only by sitting ABOVE the shaded mixture region — "
        "below or on it, the same cost-quality point is reachable by flipping a weighted "
        "coin between two fixed policies."
    ),
    definitions=(
        (
            "live",
            "the router may be configured with this strategy today — the set is derived from "
            "the product's own LIVE_STRATEGIES, not restated here.",
        ),
        (
            "live Pareto",
            "no other LIVE strategy is both cheaper and at least as good. Bounds, controls "
            "and blocked strategies are excluded — none of them can be bought.",
        ),
        (
            "mixture region",
            "the upper convex hull of the LIVE points. Any point under it is reachable by "
            "probabilistically mixing two fixed policies.",
        ),
        (
            "blocked",
            "realizable in principle but not wired or not cache-safe; each carries a named "
            "blocker and a path to live in benchmark/routing/strategy_class.py.",
        ),
    ),
    notes=(
        "The y axis is clipped to the data range, not to 0-100. The axis label says so; the "
        "alternative was a figure on which every strategy is the same flat line.",
    ),
    limitations=(
        "A non-live point's number is still a real measurement — it is the CONCLUSION that is "
        "limited: no operator can buy it, so it may not anchor a frontier or a headline.",
        "Total cost is naive per-task cost. The gate's criterion is cache-aware cost — see "
        "cache_economics.png.",
        "Pass rates are scored on the coverage-completed matrix, whose imputed cells are all "
        "pass=True — see evidence_basis.png for how much of each strategy's number that is.",
    ),
)


def _live_pareto(rows: list[dict]) -> set[str]:
    """Names no OTHER LIVE strategy beats on both cost and pass rate."""
    # Live-only is the whole point: a frontier that admits a strategy the router rejects
    # at boot describes an operating point nobody can buy. This used to admit them, which
    # made Price-Cascade a "deployable Pareto" point and anchored the published hull on it.
    live = [
        (str(r["strategy"]), float(r["TotalCost"]), float(r["AvgPerf%"]))
        for r in rows
        if is_live(str(r["strategy"])) and int(float(r.get("n_tasks", 0) or 0)) > 0
    ]
    return {
        name
        for name, cost, perf in live
        if not any(
            oc <= cost and op >= perf and (oc < cost or op > perf)
            for on, oc, op in live
            if on != name
        )
    }


def _legend(ax: Axes, plotted: set[str], pareto: set[str]) -> None:
    """One row per marker actually drawn — an absent class must not appear in the key."""
    classes = {classify(n).cls for n in plotted}
    rows: list[tuple[str, int, str, str]] = []
    if any(is_live(n) and n in pareto and n != ctxmod.BASELINE_STRATEGY for n in plotted):
        rows.append((_PARETO, 90, "o", "live, on the frontier"))
    if ctxmod.BASELINE_STRATEGY in plotted:
        rows.append((_FRONTIER, 90, "D", "fixed-frontier baseline (live)"))
    if any(is_live(n) and n not in pareto for n in plotted):
        rows.append((_OTHER, 70, "o", "live, but dominated"))
    if StrategyClass.BLOCKED in classes:
        rows.append((_BLOCKED, 80, "s", "NOT deployable — blocked (see docs/routing.md)"))
    if StrategyClass.CONTROL in classes:
        rows.append((_OTHER, 80, "X", "control — never shippable by design"))
    if StrategyClass.BOUND in classes:
        rows.append((_OTHER, 110, "*", "bound — unreachable by design"))
    for colour, size, marker, label in rows:
        ax.scatter([], [], s=size, c=colour, marker=marker, label=label)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)


def _encode(name: str, pareto: set[str]) -> tuple[str, int, str]:
    """(colour, size, marker) for one strategy — shape carries CLASS, colour carries rank."""
    cls = classify(name).cls
    if cls is StrategyClass.BOUND:
        return _OTHER, 150, "*"
    if cls is StrategyClass.BLOCKED:
        return _BLOCKED, 95, "s"
    if cls is StrategyClass.CONTROL:
        return _OTHER, 80, "X"
    if name == ctxmod.BASELINE_STRATEGY:
        return _FRONTIER, 130, "D"
    return (_PARETO, 110, "o") if name in pareto else (_OTHER, 70, "o")


def _draw(
    ax: Axes, ctx: ctxmod.RoutingContext
) -> tuple[set[str], list[tuple[float, float]], float]:
    rows = ctx.rows
    pareto = _live_pareto(rows)
    labels: list[LabelPoint] = []
    for row in rows:
        name = str(row["strategy"])
        cost, perf = float(row["TotalCost"]), float(row["AvgPerf%"])
        n = int(float(row.get("n_tasks", 0) or 0))
        n_pass = int(float(row.get("n_pass", 0) or 0))
        if cost <= 0:
            continue
        colour, size, marker = _encode(name, pareto)
        down = up = 0.0
        if n > 0:
            lo, hi = plot_style.wilson_interval(n_pass, n)
            down, up = plot_style.ci_yerr(perf / 100.0, lo, hi)
            ax.errorbar(
                cost,
                perf,
                yerr=[[down * 100], [up * 100]],
                fmt="none",
                ecolor="#666666",
                elinewidth=1.0,
                capsize=3,
                zorder=3,
            )
        ax.scatter(
            [cost],
            [perf],
            s=size,
            c=colour,
            marker=marker,
            zorder=4,
            edgecolors="white",
            linewidths=0.6,
        )
        labels.append(LabelPoint(cost, perf, name, down * 100, up * 100))

    # Hull over LIVE points only. A mixture frontier is the cost-quality a coin-flip
    # between two REAL policies buys; anchoring it on a strategy that cannot be deployed
    # describes a mixture nobody can construct.
    live_points = [
        (float(r["TotalCost"]), float(r["AvgPerf%"]))
        for r in rows
        if is_live(str(r["strategy"])) and float(r["TotalCost"]) > 0
    ]
    hull = plot_style.upper_hull(live_points)
    aiq = plot_style.area_under_frontier(hull)
    if len(hull) >= 2:
        hx = [p[0] for p in hull]
        hy = [p[1] for p in hull]
        ax.plot(hx, hy, color=_HULL, lw=1.6, zorder=2, label="mixture frontier (no router needed)")
        ax.fill_between(hx, min(hy) - 40, hy, color=_HULL, alpha=0.07, zorder=1)

    perfs = [float(r["AvgPerf%"]) for r in rows if float(r["TotalCost"]) > 0]
    ax.set_ylim(min(perfs) - 3.0, 100.0)
    costs = [float(r["TotalCost"]) for r in rows if float(r["TotalCost"]) > 0]
    ax.set_xscale("log")
    ax.set_xlim(min(costs) * 0.55, max(costs) * 2.6)
    ax.set_xlabel("total spend over the scored task set (USD, log)", fontsize=9)
    ax.set_ylabel(f"pass rate (%) — axis clipped to [{min(perfs) - 3:.0f}, 100]", fontsize=9)
    ax.grid(color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    _legend(ax, {str(r["strategy"]) for r in rows if float(r["TotalCost"]) > 0}, pareto)
    plot_style.place_labels(ax, labels, fontsize=7.5)
    return pareto, hull, aiq


_STANDING: Final[Mapping[StrategyClass, str]] = MappingProxyType(
    {
        StrategyClass.BOUND: "bound — unreachable by design",
        StrategyClass.CONTROL: "control — never shippable",
        StrategyClass.BLOCKED: "blocked — not deployable",
    }
)


def _standing(name: str, pareto: set[str]) -> str:
    """This row's CLASS, and for a live row whether the frontier passes through it."""
    cls = classify(name).cls
    if cls is not StrategyClass.LIVE:
        return _STANDING[cls]
    return "live, on the frontier" if name in pareto else "live, but dominated"


def _caveat(shown: list[dict]) -> str | None:
    """Derived, never hardcoded: how much of this plane a reader may not act on."""
    live = [r for r in shown if is_live(str(r["strategy"]))]
    non_live = len(shown) - len(live)
    if not non_live:
        return None
    return (
        f"{non_live} of {len(shown)} strategies shown cannot run live; the frontier uses "
        f"the {len(live)} live ones only."
    )


def _annotations(ctx: ctxmod.RoutingContext, pareto: set[str], aiq: float) -> Annotations:
    router = ctx.row(ctxmod.ROUTER_STRATEGY)
    baseline = ctx.row(ctxmod.BASELINE_STRATEGY)
    ns = {int(float(r.get("n_tasks", 0) or 0)) for r in ctx.rows}
    facts = [f"{len(ctx.rows)} strategies over {max(ns) if ns else 0} scored tasks"]
    if router and baseline:
        ratio = float(router["TotalCost"]) / float(baseline["TotalCost"])
        facts.append(
            f"{ctxmod.ROUTER_STRATEGY} {usd(float(router['TotalCost']))} at "
            f"{float(router['AvgPerf%']):.1f}% vs baseline {usd(float(baseline['TotalCost']))} at "
            f"{float(baseline['AvgPerf%']):.1f}% ({ratio:.0%} of the bill)"
        )
    facts.append(f"area under the live frontier {aiq:.3f}")
    notes = [
        f"{r['strategy']}: {usd(float(r['TotalCost']))}, {float(r['AvgPerf%']):.2f}% "
        f"(n={int(float(r.get('n_tasks', 0) or 0))}, {_standing(str(r['strategy']), pareto)})"
        for r in ctx.rows
    ]
    if ctx.banner:
        notes.append(ctx.banner)
    return Annotations(
        subtitle_facts=tuple(facts),
        caveat=_caveat([r for r in ctx.rows if float(r.get("TotalCost", 0) or 0) > 0]),
        notes=tuple(notes),
        counts=(("strategies", len(ctx.rows)), ("tasks", max(ns) if ns else 0)),
    )


def render(ctx: ctxmod.RoutingContext) -> Path | None:
    """Draw cost_quality_frontier.png, or None when no strategy row carries cost."""
    if not ctx.rows or not any(float(r.get("TotalCost", 0) or 0) > 0 for r in ctx.rows):
        return None
    size = plot_frame.SINGLE_TALL
    fig = plot_frame.new_figure(size)
    ax = fig.subplots()
    pareto, _hull, aiq = _draw(ax, ctx)
    _ = np  # numpy is imported for the hull maths in plot_style; keep the dependency explicit
    return plot_frame.save(
        fig,
        ctx.out_dir / "cost_quality_frontier.png",
        SPEC,
        extra=_annotations(ctx, pareto, aiq),
        provenance=ctx.provenance(__name__),
        size=size,
    )
