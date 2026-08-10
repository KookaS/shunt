"""cost_quality_frontier.png — one cost-quality plane, replacing four that drew the same one."""

# `pareto_scatter`, `cost_savings`, `cost_quality_equal` and `strategy_comparison` were four
# renderings of a single (total cost, pass rate) scatter over the same strategy rows. They
# disagreed on the Pareto definition, on whether hindsight oracles were peers, and — worst —
# `cost_savings` headlined a saving ratio taken at UNEQUAL quality. One plane, one Pareto
# definition, one hull.
#
# ONE COST MODEL, TOO. The plane used to rank on raw `TotalCost` while strategy_summary.csv's
# `Pareto` column had already moved to cache-aware cost. Membership agreed on one corpus,
# which is a coincidence and not a structure — the two costs part company exactly on cascades,
# because a cascade re-serves one model on consecutive attempts and banks the cached prefix.
# Both the marker and the hull now use the cache-aware column the summary decides with, and
# the naive sum stays on canvas as the far end of a drawn bracket so the choice is auditable.
#
# Two deliberate departures from the figures this replaces. The x axis is LOG, because the
# strategies span two decades and a linear axis crushed five of them into the left margin.
# The y axis starts below the worst point rather than at zero, because a 20-point band
# rendered on a 0-100 axis is a flat line — the FIGURE is the comparison, and the axis is
# labelled so nobody reads the range as a full scale.

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import plot_style
from benchmark.routing.figures import context as ctxmod
from benchmark.routing.plot_style import LabelPoint, usd
from benchmark.routing.strategy_class import (
    StrategyClass,
    classify,
    is_live,
    shipped_mechanism,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from matplotlib.axes import Axes

_PARETO = "#009E73"
_FRONTIER = "#D55E00"
# #9E9E9E and #56B4E9 both fell below 3:1 against the near-white canvas, so a dominated
# point and a blocked square were the two faintest marks on a figure about which strategy
# to buy. Darkened to the Okabe-Ito blue and a grey that clears the contrast floor; class
# is still carried by SHAPE, never by colour alone.
_OTHER = "#757575"
_BLOCKED = "#0072B2"
_HULL = "#009E73"
_UNCERTAINTY = "#666666"

# The x axis is the cost a deployment is BILLED, which is the cache-aware column the summary
# now decides `Pareto` on. `TotalCost` stays in the picture as the other end of a two-ended
# cost mark, so the reader can see how far the cost-model choice moves each strategy instead
# of having to trust that it does not.
_COST_KEY: Final[str] = "TotalCost_cacheaware"
_NAIVE_KEY: Final[str] = "TotalCost"

SPEC = FigureSpec(
    title="The live frontier at cache-aware cost — cost is the uncertain axis",
    reading=(
        "x is total dollars spent over the scored task set, on a log axis because the "
        "strategies span two decades. The marker sits at CACHE-AWARE cost — what a "
        "deployment is billed once a repeat of the same model banks its cached prefix — and "
        "the thin line running from it ends at an open tick at the NAIVE per-call sum, so "
        "the gap between the two cost models is a drawn distance. The capped bar at that "
        "tick is the 95% bootstrap interval on the naive total; it is drawn on the naive "
        "statistic because that is the one it was computed for. Both cost marks appear on "
        "LIVE strategies only. y is pass rate in percent "
        "with a 95% Wilson interval. Marker SHAPE carries what a strategy is: circles and "
        "the orange diamond can run in production today, blue squares are blocked — no "
        "router.strategy value names them — an X is a control that must never ship, and a "
        "star is a bound that is unreachable by design. Marker FILL splits the blue squares, "
        "and it is the distinction most likely to be misread: a SOLID square is blocked and "
        "nothing equivalent runs today, while a HOLLOW one marks a mechanism that already "
        "ships and is on by default under another config surface, where the only thing "
        "blocked is the strategy NAME. Only the live points "
        "enter the Pareto test and the shaded mixture region — a frontier anchored on a "
        "strategy the router rejects at boot describes an operating point nobody can buy."
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
            "no other LIVE strategy is both cheaper and at least as good, on CACHE-AWARE "
            "cost. Bounds, controls and blocked strategies are excluded — none of them is a "
            "setting an operator can choose. This is the same cost model "
            "strategy_summary.csv's Pareto column uses, so the plane and the table cannot "
            "disagree.",
        ),
        (
            "cache-aware cost",
            "a repeat of the same model on a consecutive attempt is billed at the provider's "
            "cache-read rate rather than at full input price. Cascades re-serve one model by "
            "construction, so this is exactly where the two cost models part company.",
        ),
        (
            "mixture region",
            "the upper convex hull of the LIVE points. Any point under it is reachable by "
            "probabilistically mixing two fixed policies.",
        ),
        (
            "blocked",
            "no router.strategy value names it. This says nothing about whether the "
            "MECHANISM runs: a solid square cannot be run today, a hollow one already runs "
            "in every default install under the config surface named in the legend. Both are "
            "excluded from the frontier, because the frontier ranks settings an operator can "
            "choose. Each row's blocker and path to live are in "
            "benchmark/routing/strategy_class.py.",
        ),
    ),
    notes=(
        "The y axis is clipped to the data range, not to 0-100. The axis label says so; the "
        "alternative was a figure on which every strategy is the same flat line.",
        "The marker and the hull use the same cache-aware cost column strategy_summary.csv "
        "decides Pareto on, so the plane and the table cannot rank strategies differently.",
    ),
    limitations=(
        "A non-live point's number is still a real measurement — it is the CONCLUSION that is "
        "limited: no router.strategy setting reproduces it, so it may not anchor the frontier "
        "or a headline. It does NOT follow that the underlying capability is unavailable — a "
        "blocked row may measure a mechanism that ships in a different layer — and the "
        "per-strategy blocker in benchmark/routing/strategy_class.py says which case it is.",
        "The cache-aware x position rests on an ASSUMED cache hit rate; only the per-model "
        "discount and input share are measured — see cache_economics.png for the range that "
        "assumption spans.",
        "The horizontal interval belongs to the NAIVE total and is not transplanted onto the "
        "cache-aware marker. No bootstrap interval on cache-aware cost is published, so the "
        "sampling uncertainty of the plotted x is bracketed by the naive one, not measured.",
        "Pass rates are scored on the coverage-completed matrix, whose imputed cells are all "
        "pass=True — see evidence_basis.png for how much of each strategy's number that is.",
        "The scored set is chosen by coverage, not at random: the collector runs the "
        "expensive tier only on the discriminating slice, so both axes describe a "
        "difficulty-biased sample. The subtitle carries the measured gap.",
    ),
)


def _num(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    """One numeric cell, tolerating a column the summary has not written yet."""
    # The summary grew nine columns; a figure that KeyErrors on an older CSV cannot be run
    # against an archived result set, which is the one time you most want to redraw it.
    try:
        return float(row.get(key) or default)
    except (TypeError, ValueError):
        return default


def _cost(row: Mapping[str, str]) -> float:
    """The billed cost this figure plots: cache-aware, falling back to the naive sum."""
    return _num(row, _COST_KEY) or _num(row, _NAIVE_KEY)


def _is_cache_aware(rows: list[dict]) -> bool:
    """Whether the x axis really is the cache-aware column, or fell back to the naive sum."""
    # An archived summary predates the cache-aware column. Saying "cache-aware" on a canvas
    # that fell back to raw sums would be the exact defect this figure was redrawn to fix.
    return any(_num(r, _COST_KEY) > 0 for r in rows)


def _live_pareto(rows: list[dict]) -> set[str]:
    """Names no OTHER LIVE strategy beats on both cache-aware cost and pass rate."""
    # Live-only is the whole point: a frontier that admits a strategy the router rejects
    # at boot describes an operating point nobody can buy. This used to admit them, which
    # made Price-Cascade a "deployable Pareto" point and anchored the published hull on it.
    #
    # The cost key is cache-aware for a second reason: it used to be the raw sum while
    # strategy_summary.csv's own `Pareto` column had moved to cache-aware. Membership
    # happened to agree on one corpus, which is a coincidence and not a structure — and the
    # two costs diverge MOST on cascades, exactly where the next strategy will land.
    live = [
        (str(r["strategy"]), _cost(r), float(r["AvgPerf%"]))
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
    blocked = [n for n in plotted if classify(n).cls is StrategyClass.BLOCKED]
    rows: list[tuple[str, int, str, bool, str]] = []
    if any(is_live(n) and n in pareto and n != ctxmod.BASELINE_STRATEGY for n in plotted):
        rows.append((_PARETO, 90, "o", True, "live, on the frontier"))
    if ctxmod.BASELINE_STRATEGY in plotted:
        rows.append((_FRONTIER, 90, "D", True, "fixed-frontier baseline (live)"))
    if any(is_live(n) and n not in pareto for n in plotted):
        rows.append((_OTHER, 70, "o", True, "live, but dominated"))
    # Two rows, not one, and only the ones actually drawn. A single "NOT deployable"
    # row over both kinds of blocked strategy told a reader that a mechanism shipping in
    # every default install cannot be run — the mirror image of the defect this figure was
    # redrawn to remove, and the more dangerous half, because it argues against building on
    # something that already works.
    if any(shipped_mechanism(n) is None for n in blocked):
        rows.append((_BLOCKED, 80, "s", True, "not deployable — blocked (see docs/routing.md)"))
    for surface in sorted({s for n in blocked if (s := shipped_mechanism(n)) is not None}):
        rows.append(
            (_BLOCKED, 80, "s", False, f"mechanism ships as {surface}; only the NAME is blocked")
        )
    if StrategyClass.CONTROL in classes:
        rows.append((_OTHER, 80, "X", True, "control — never shippable by design"))
    if StrategyClass.BOUND in classes:
        rows.append((_OTHER, 110, "*", True, "bound — unreachable by design"))
    for colour, size, marker, filled, label in rows:
        ax.scatter(
            [],
            [],
            s=size,
            marker=marker,
            facecolors=colour if filled else "none",
            edgecolors=colour,
            linewidths=0.6 if filled else 1.6,
            label=label,
        )
    # The cost marks need their own key: without it the bar reads as a confidence interval
    # on the plotted (cache-aware) x, which is exactly what it is not.
    ax.plot(
        [],
        [],
        color=_UNCERTAINTY,
        lw=0.9,
        marker="|",
        ms=7,
        mew=1.2,
        alpha=0.8,
        label="→ naive per-call cost (cost-model gap)",
    )
    ax.errorbar(
        [],
        [],
        xerr=[[0], [0]],
        fmt="none",
        ecolor=_UNCERTAINTY,
        elinewidth=1.0,
        capsize=3,
        alpha=0.7,
        label="95% bootstrap CI — on the NAIVE total",
    )
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)


def _encode(name: str, pareto: set[str]) -> tuple[str, int, str, bool]:
    """(colour, size, marker, filled) — shape carries CLASS, fill carries does-it-ship."""
    # The FILL is the fourth channel and it carries exactly one bit: whether the mechanism
    # this row measures runs in production today. A hollow blue square is still a blue
    # square — same class, same exclusion from the hull — but it no longer tells a reader
    # that a default-on shipped mechanism cannot be run.
    cls = classify(name).cls
    if cls is StrategyClass.BOUND:
        return _OTHER, 150, "*", True
    if cls is StrategyClass.BLOCKED:
        return _BLOCKED, 95, "s", shipped_mechanism(name) is None
    if cls is StrategyClass.CONTROL:
        return _OTHER, 80, "X", True
    if name == ctxmod.BASELINE_STRATEGY:
        return _FRONTIER, 130, "D", True
    return (_PARETO, 110, "o", True) if name in pareto else (_OTHER, 70, "o", True)


def _extent_xs(row: Mapping[str, str]) -> list[float]:
    """Every x this row draws — marker, naive tick and CI caps — for the axis limits."""
    xs = [_cost(row), _num(row, _NAIVE_KEY), _num(row, "TotalCost_ci_lower")]
    xs.append(_num(row, "TotalCost_ci_upper"))
    return [x for x in xs if x > 0]


def _draw_cost_uncertainty(
    ax: Axes, row: Mapping[str, str], cost: float, perf: float
) -> tuple[float, float]:
    """The two things missing from x — the cost-model gap and the bootstrap CI — as (lo, hi)."""
    # y carried a Wilson interval while x carried a bare point, so the plane advertised a
    # precision on cost that does not exist. Both marks below are drawn where they belong:
    # the bracket spans two MEASURED costs, and the interval sits on the naive statistic it
    # was bootstrapped from rather than being transplanted onto the cache-aware marker.
    #
    # LIVE rows only. Five strategies sit within a point of each other in pass rate, so
    # drawing every one's bar stacks five overlapping rules across the top of the plane —
    # and a cost interval on a strategy nobody can deploy is precision about a purchase that
    # is not on offer, which this figure already refuses to headline.
    if not is_live(str(row.get("strategy", ""))):
        return (0.0, 0.0)
    naive = _num(row, _NAIVE_KEY)
    if naive <= 0:
        return (0.0, 0.0)
    if naive != cost:
        ax.plot([cost, naive], [perf, perf], color=_UNCERTAINTY, lw=0.9, alpha=0.7, zorder=2)
        ax.plot([naive], [perf], "|", color=_UNCERTAINTY, ms=7, mew=1.2, alpha=0.9, zorder=2)
    lo, hi = _num(row, "TotalCost_ci_lower"), _num(row, "TotalCost_ci_upper")
    if 0 < lo <= naive <= hi:
        ax.errorbar(
            naive,
            perf,
            xerr=[[naive - lo], [hi - naive]],
            fmt="none",
            ecolor=_UNCERTAINTY,
            elinewidth=1.0,
            capsize=3,
            alpha=0.7,
            zorder=2,
        )
        return (max(cost - lo, 0.0), max(hi - cost, 0.0))
    return (max(cost - naive, 0.0), max(naive - cost, 0.0))


def _draw(
    ax: Axes, ctx: ctxmod.RoutingContext
) -> tuple[set[str], list[tuple[float, float]], float]:
    rows = ctx.rows
    pareto = _live_pareto(rows)
    labels: list[LabelPoint] = []
    for row in rows:
        name = str(row["strategy"])
        cost, perf = _cost(row), float(row["AvgPerf%"])
        n = int(float(row.get("n_tasks", 0) or 0))
        n_pass = int(float(row.get("n_pass", 0) or 0))
        if cost <= 0:
            continue
        colour, size, marker, filled = _encode(name, pareto)
        xerr_lo, xerr_hi = _draw_cost_uncertainty(ax, row, cost, perf)
        down = up = 0.0
        if n > 0:
            lo, hi = plot_style.wilson_interval(n_pass, n)
            down, up = plot_style.ci_yerr(perf / 100.0, lo, hi)
            ax.errorbar(
                cost,
                perf,
                yerr=[[down * 100], [up * 100]],
                fmt="none",
                ecolor=_UNCERTAINTY,
                elinewidth=1.0,
                capsize=3,
                zorder=3,
            )
        ax.scatter(
            [cost],
            [perf],
            s=size,
            marker=marker,
            zorder=4,
            facecolors=colour if filled else "none",
            edgecolors="white" if filled else colour,
            linewidths=0.6 if filled else 1.6,
        )
        labels.append(
            LabelPoint(cost, perf, name, down * 100, up * 100, xerr_lo=xerr_lo, xerr_hi=xerr_hi)
        )

    # Hull over LIVE points only. A mixture frontier is the cost-quality a coin-flip
    # between two REAL policies buys; anchoring it on a strategy that cannot be deployed
    # describes a mixture nobody can construct.
    live_points = [
        (_cost(r), float(r["AvgPerf%"]))
        for r in rows
        if is_live(str(r["strategy"])) and _cost(r) > 0
    ]
    hull = plot_style.upper_hull(live_points)
    aiq = plot_style.area_under_frontier(hull)
    if len(hull) >= 2:
        hx = [p[0] for p in hull]
        hy = [p[1] for p in hull]
        ax.plot(hx, hy, color=_HULL, lw=1.6, zorder=2, label="mixture frontier (no router needed)")
        ax.fill_between(hx, min(hy) - 40, hy, color=_HULL, alpha=0.07, zorder=1)

    axis_model = "cache-aware" if _is_cache_aware(rows) else "naive per-call"
    perfs = [float(r["AvgPerf%"]) for r in rows if _cost(r) > 0]
    ax.set_ylim(min(perfs) - 3.0, 100.0)
    # The extent must cover the CI caps too, or a strategy whose cost interval runs off the
    # right edge is silently clipped — which is precisely the strategy the reader needs.
    costs = [x for r in rows if _cost(r) > 0 for x in _extent_xs(r)]
    ax.set_xscale("log")
    ax.set_xlim(min(costs) * 0.55, max(costs) * 2.6)
    ax.set_xlabel(
        f"total spend over the scored task set — {axis_model} (USD, log)",
        fontsize=9,
    )
    ax.set_ylabel(f"pass rate (%) — axis clipped to [{min(perfs) - 3:.0f}, 100]", fontsize=9)
    ax.grid(color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    _legend(ax, {str(r["strategy"]) for r in rows if _cost(r) > 0}, pareto)
    plot_style.place_labels(ax, labels, fontsize=7.5)
    return pareto, hull, aiq


_STANDING: Final[Mapping[StrategyClass, str]] = MappingProxyType(
    {
        StrategyClass.BOUND: "bound — unreachable by design",
        StrategyClass.CONTROL: "control — never shippable",
        StrategyClass.BLOCKED: "blocked — no router.strategy names it",
    }
)


def _standing(name: str, pareto: set[str]) -> str:
    """This row's CLASS, and for a live row whether the frontier passes through it."""
    cls = classify(name).cls
    if cls is StrategyClass.BLOCKED:
        surface = shipped_mechanism(name)
        if surface is not None:
            return f"blocked as a NAME only — the mechanism ships as {surface}"
    if cls is not StrategyClass.LIVE:
        return _STANDING[cls]
    return "live, on the frontier" if name in pareto else "live, but dominated"


def _subset_clause(shown: list[dict]) -> str:
    """The selection warning, from the summary's own subset_note — '' when it reports none."""
    # This outranks the live/non-live count for the red line: non-liveness is already carried
    # on canvas by marker shape and the legend, whereas nothing on the plane says the scored
    # set was chosen by coverage. That selection moves BOTH axes for EVERY point at once.
    flagged = [r for r in shown if str(r.get("subset_note") or "")]
    if not flagged:
        return ""
    scored = {int(float(r.get("n_tasks", 0) or 0)) for r in flagged}
    return f"scored on {max(scored) if scored else 0} coverage-selected tasks (dropped are harder)"


def _caveat(shown: list[dict]) -> str | None:
    """Derived, never hardcoded: how much of this plane a reader may not act on."""
    live = [r for r in shown if is_live(str(r["strategy"]))]
    non_live = len(shown) - len(live)
    parts = [p for p in (_subset_clause(shown),) if p]
    if non_live:
        parts.append(f"{non_live} of {len(shown)} are not selectable as router.strategy")
    if not parts:
        return None
    return "; ".join(parts) + "."


def _pareto_crosscheck(rows: list[dict], pareto: set[str]) -> list[str]:
    """Say out loud whether the drawn frontier and the summary's Pareto column agree."""
    # The two were computed on DIFFERENT cost models and happened to agree, which is the
    # shape of a bug that stays invisible until it matters. They are now computed on the
    # same one, so a disagreement can only mean the live filter — and it is recorded rather
    # than left for someone to notice.
    claimed = {str(r["strategy"]) for r in rows if str(r.get("Pareto", "")).lower() == "true"}
    if not claimed:
        return []
    extra = sorted(pareto - claimed)
    if extra:
        return [
            "frontier vs strategy_summary.csv Pareto: drawn-only "
            + ", ".join(extra)
            + " (both on cache-aware cost; a difference here is the live-only filter)"
        ]
    return ["frontier membership agrees with strategy_summary.csv's cache-aware Pareto column"]


def _annotations(ctx: ctxmod.RoutingContext, pareto: set[str], aiq: float) -> Annotations:
    router = ctx.row(ctxmod.ROUTER_STRATEGY)
    baseline = ctx.row(ctxmod.BASELINE_STRATEGY)
    ns = {int(float(r.get("n_tasks", 0) or 0)) for r in ctx.rows}
    model = "cache-aware" if _is_cache_aware(ctx.rows) else "naive — no cache-aware column"
    facts = [f"{len(ctx.rows)} strategies over {max(ns) if ns else 0} scored tasks"]
    if router and baseline and _cost(baseline) > 0:
        ratio = _cost(router) / _cost(baseline)
        facts.append(
            f"{ctxmod.ROUTER_STRATEGY} {usd(_cost(router))} at "
            f"{float(router['AvgPerf%']):.1f}% vs baseline {usd(_cost(baseline))} at "
            f"{float(baseline['AvgPerf%']):.1f}% ({ratio:.0%} of the bill, {model})"
        )
    if router:
        lo, hi = _num(router, "TotalCost_ci_lower"), _num(router, "TotalCost_ci_upper")
        if hi > lo > 0:
            facts.append(
                f"cost is the wide axis: {ctxmod.ROUTER_STRATEGY}'s naive total spans "
                f"{usd(lo)}–{usd(hi)} (95% bootstrap)"
            )
    facts.append(f"area under the live frontier {aiq:.3f}")
    notes = [
        f"{r['strategy']}: {usd(_cost(r))} cache-aware / {usd(_num(r, _NAIVE_KEY))} naive, "
        f"{float(r['AvgPerf%']):.2f}% "
        f"(n={int(float(r.get('n_tasks', 0) or 0))}, {_standing(str(r['strategy']), pareto)})"
        for r in ctx.rows
    ]
    notes.extend(_pareto_crosscheck(ctx.rows, pareto))
    subset = next((str(r.get("subset_note") or "") for r in ctx.rows if r.get("subset_note")), "")
    if subset:
        notes.append(f"selection: {subset}")
    if ctx.banner:
        notes.append(ctx.banner)
    return Annotations(
        subtitle_facts=tuple(facts),
        caveat=_caveat([r for r in ctx.rows if _cost(r) > 0]),
        notes=tuple(notes),
        counts=(("strategies", len(ctx.rows)), ("tasks", max(ns) if ns else 0)),
    )


def render(ctx: ctxmod.RoutingContext) -> Path | None:
    """Draw cost_quality_frontier.png, or None when no strategy row carries cost."""
    if not ctx.rows or not any(_cost(r) > 0 for r in ctx.rows):
        return None
    size = plot_frame.SINGLE_TALL
    fig = plot_frame.new_figure(size)
    ax = fig.subplots()
    pareto, _hull, aiq = _draw(ax, ctx)
    return plot_frame.save(
        fig,
        ctx.out_dir / "cost_quality_frontier.png",
        SPEC,
        extra=_annotations(ctx, pareto, aiq),
        provenance=ctx.provenance(__name__),
        size=size,
    )
