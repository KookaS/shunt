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

import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from matplotlib import ticker

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import plot_style
from benchmark.routing.figures import context as ctxmod
from benchmark.routing.plot_style import LabelCluster, LabelPoint, usd
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
    from matplotlib.figure import Figure

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
# The context-transfer cost model's columns, written by summary.compute_strategy_rows. The
# figure READS them and never computes the model: a plot that re-derives a cost is a second
# accounting path, and the one thing this figure exists to prevent is two of those.
_ALPHA_01_KEY: Final[str] = "context_cost_alpha_01"
_ALPHA_03_KEY: Final[str] = "context_cost_alpha_03"
_ALPHA_10_KEY: Final[str] = "context_cost_alpha_10"
_ALPHA_N_KEY: Final[str] = "context_cost_n"
# Distinguished from the naive-cost bracket by LINESTYLE, never by colour: a colour-only
# encoding already failed a contrast audit on this canvas (see the _OTHER/_BLOCKED note above),
# and two grey rules at the same weight would be one mark to a reader who cannot separate hues.
_CONTEXT_LS: Final[tuple[int, tuple[int, int]]] = (0, (3, 2))

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
        "LIVE strategies only. "
        "y is pass rate in percent "
        "with a 95% Wilson interval. Marker SHAPE carries what a strategy is: circles and "
        "the orange diamond can run in production today, blue squares are blocked — no "
        "router.strategy value names them — an X is a control that must never ship, and a "
        "star is a bound that is unreachable by design. Marker FILL splits the blue squares: "
        "a SOLID square is blocked and nothing equivalent runs today, while a HOLLOW one "
        "would mark a mechanism that ships under another config surface with only its NAME "
        "blocked. NO ROW IS HOLLOW TODAY — the one that was, Session-Cascade, is now live as "
        "router.strategy: session_cascade, and every remaining blue square is solid, i.e. "
        "genuinely unrunnable. Only the live points "
        "enter the Pareto test and the shaded mixture region — a frontier anchored on a "
        "strategy the router rejects at boot describes an operating point nobody can buy. "
        "WHERE SEVERAL STRATEGIES COLLAPSE INTO A FEW PIXELS a MAGNIFIED PANEL redraws that "
        "region at larger scale; the outlined rectangle on the plane and the two lines running "
        "to the panel say which region it is. The panel is a MAGNIFICATION, not extra data: "
        "every mark in it is one of the points already on the plane, at the same cost and the "
        "same pass rate, and it is drawn only when the names in a region genuinely cannot be "
        "placed there — when the strategies spread out, no panel appears. It leaves the "
        "intervals behind, because at that scale they run off the panel; they stay on the "
        "plane. Inside the panel the two DEPLOYABLE escalating strategies each carry a "
        "context-transfer cost model, drawn as a dashed rule hanging just under the marker it "
        "belongs to and joined to it by a thin line: the MARKER is what the benchmark "
        "measures, a fresh context on every rung, which is not a setting anyone can select; "
        "the shaded segment is `context_transfer: summary`, a band rather than a tick because "
        "how far a summariser compresses is not a constant; and the tick at the right end is "
        "`context_transfer: full`, which is what ships today. That rule asserts NO pass rate "
        "— it is horizontal, and its height carries no meaning at all."
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
            "context transfer",
            "what a deployment is billed once the model an escalation moves to is resent the "
            "conversation. The router config names the two settings the panel labels: "
            "`summary` resends a summarised prefix, `full` resends all of it, and `full` is "
            "the shipped default. The resent prefix is a cache MISS by construction — a new "
            "model receiving a prefix it has never seen — so it is charged at the full input "
            "rate, never at the cache-read rate. Context size is estimated as "
            "t = 2 x in_tok / calls. It is a cost model, it asserts no pass rate, and it rests "
            "on the token-complete subset named in the notes.",
        ),
        (
            "magnified panel",
            "an inset that redraws one crowded region of the plane at larger scale. Same "
            "points, same numbers, no extra data — and no intervals, which stay on the plane. "
            "Which region it covers is decided from the rendered figure, not written down: the "
            "labels are measured against each other, and a panel is drawn only where they "
            "cannot all be placed. Its bounds are the crowd's own bounding box widened to "
            "contain everything the panel draws, including the full length of a context "
            "bracket.",
        ),
        (
            "mixture region",
            "the upper convex hull of the LIVE points. Any point under it is reachable by "
            "probabilistically mixing two fixed policies.",
        ),
        (
            "blocked",
            "no router.strategy value names it, and on this corpus that now means genuinely "
            "unrunnable: the two remaining blocked cascades verify INSIDE one task, which "
            "breaks the one-decision-per-session cache-safety spine and is excluded by "
            "design rather than pending. They are excluded from the frontier, because the "
            "frontier ranks settings an operator can choose, and they are kept because they "
            "price what session cadence costs. Each row's blocker and path to live are in "
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
        "A cascade point prices the LADDER's cost, not its per-rung quality. The shipped "
        "ladder's cheap intermediate rungs are measured separately against the base model on "
        "this same corpus, and are null or net-harmful there — see ladder_rungs.png.",
        "The cache-aware x position rests on an ASSUMED cache hit rate; only the per-model "
        "discount and input share are measured — see cache_economics.png for the range that "
        "assumption spans.",
        "The horizontal interval belongs to the NAIVE total and is not transplanted onto the "
        "cache-aware marker. The cache-aware ratio's own 90% bootstrap CI is published by the "
        "kill gate (cache cost is scoped per task, so a whole-task resample preserves it) — "
        "not transplanted onto this plot.",
        "Pass rates are scored on the coverage-completed matrix, whose imputed cells are all "
        "pass=True — see evidence_basis.png for how much of each strategy's number that is.",
        "The scored set is chosen by coverage, not at random: the collector runs the "
        "expensive tier only on the discriminating slice, so both axes describe a "
        "difficulty-biased sample. The subtitle carries the measured gap.",
        "The dashed context bracket is a COST MODEL, not a measurement. It re-prices the "
        "marker when the context an attempt ends holding is resent to the model escalation "
        "moves to — a cache MISS by construction, so it is charged at full input rate. It "
        "asserts NO pass rate: the bracket is horizontal because nothing here measures what "
        "carrying context does to quality.",
        "The canvas labels the bracket in CONFIG vocabulary; the cost model underneath is "
        "parameterised by alpha, the share of the context an attempt ends holding that is "
        "resent. The mapping is exact: `context_transfer: summary` is the alpha 0.1-0.3 band "
        "(a band, because a summariser's compression ratio is not a constant and one tick "
        "would assert a precision this model does not have), `context_transfer: full` is "
        "alpha = 1.0, and the marker itself is alpha = 0 — a fresh context on every rung, "
        "which is what the OFFLINE benchmark replays and what live inference never does. "
        "alpha = 0 is deliberately not offered as a config value: `none` is not a "
        "context_transfer setting, so the marker is the offline/live divergence made visible "
        "rather than a third option.",
        "A bracket is drawn on the DEPLOYABLE escalating strategies only. The summary table "
        "also carries alpha columns for the two within-task cascades, and they are "
        "deliberately not drawn: the model prices a SESSION-BOUNDARY handoff, and those two "
        "rows are blocked precisely because they retry inside one task, so they have no "
        "boundary to hand off at. A strategy that never escalates carries nothing and "
        "correctly shows no bracket at all.",
        "The bracket's context size is estimated as t = 2 x in_tok / calls, which assumes the "
        "prefix grows LINEARLY across a task's calls. Tool output and file reads do not arrive "
        "at a constant rate, so the error is one-sided in an unknown direction, and the bracket "
        "is an ordering of magnitudes rather than a quotable dollar amount.",
        "The bracket is computed on the token-complete subset — the tasks where every attempt "
        "on the realized path landed on a measured, token-bearing cell — which is strictly "
        "smaller than the scored set, because an imputed cell carries no token columns at all. "
        "What transfers to the plotted marker is the dimensionless surcharge FACTOR, not the "
        "subset's own dollars, and that transfer ASSUMES the subset carries context per dollar "
        "the way the scored set does. The subset is not a random sample of it — it is the tasks "
        "the collector happened to measure on every rung this strategy walked — so if those "
        "tasks escalate differently from the rest, the bracket is biased in the direction of "
        "that difference and nothing here corrects it. The subset size is published as the n in "
        "the bracket note row.",
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
        label="95% bootstrap CI (naive total)",
    )
    # The context-transfer bracket USED to earn a row here. It no longer does, because it is
    # no longer drawn on the plane: it lives in the magnified panel, labelled in place with the
    # config words it prices. A key for a mark the canvas does not carry is worse than no key.
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
    """Every x this row reaches — marker, naive tick, CI caps and bracket — for the limits."""
    # The alpha=1.0 end is still in here now that the bracket has moved into the magnified
    # panel, and for a sharper reason than before: the panel's window is derived to CONTAIN the
    # bracket, and the rectangle marking that window is drawn on the PLANE. An x limit computed
    # without the bracket would put part of that rectangle outside the axes.
    xs = [_cost(row), _num(row, _NAIVE_KEY), _num(row, "TotalCost_ci_lower")]
    xs.append(_num(row, "TotalCost_ci_upper"))
    xs.append(_num(row, _ALPHA_10_KEY))
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


def _bracket_rows(rows: list[dict]) -> list[dict]:
    """The rows a context bracket may be drawn on: DEPLOYABLE strategies that escalate."""
    # TWO CONDITIONS, AND THE FIRST IS NOT A CONVENIENCE. The alpha model prices a SESSION
    # BOUNDARY handoff — the CLI resending a conversation to a model that has never seen it.
    # `strategy_summary.csv` also carries alpha columns for Price-Cascade and kNN-cascade
    # (within-task), and bracketing them would be wrong however available the numbers are:
    # those two are blocked precisely BECAUSE they retry inside one task (strategy_class.py's
    # _CASCADE_BLOCKER), so they have no session boundary to hand off at and a bracket on them
    # prices a deployment that structurally cannot exist. `is_live` is what encodes that here —
    # the within-task cadence is excluded by design, not pending — so do not "fix" this by
    # widening it to every row with an alpha column.
    #
    # The second condition keeps Always-Cheap and Always-Frontier out: a strategy that never
    # escalates carries no context across a handoff, and its surcharge is exactly zero.
    return [
        r for r in rows if is_live(str(r["strategy"])) and _num(r, _ALPHA_10_KEY) > _cost(r) > 0
    ]


def _bracket_extent(rows: list[dict], names: set[str]) -> float:
    """The rightmost dollar any bracket among `names` reaches — 0 when none is drawn."""
    # The panel has to CONTAIN the brackets it draws, so the window is derived from this
    # rather than from the markers alone. A longer bracket widens the window on its own.
    return max(
        (_num(r, _ALPHA_10_KEY) for r in _bracket_rows(rows) if str(r["strategy"]) in names),
        default=0.0,
    )


def _draw_bracket(
    inset: Axes, row: Mapping[str, str], cost: float, perf: float, level: float
) -> float:
    """One row's context-transfer cost model, in the panel. Returns the x it reaches."""
    # WHAT IT IS. A COST MODEL — the marker's cache-aware total re-priced when a share of each
    # attempt's ending context is resent, uncached, to the model escalation moves to. It asserts
    # NO pass rate, which is exactly why it may be drawn a hair BELOW the pass-rate line: three
    # other strategies share that line inside this window, and a rule through them would read as
    # a measurement crossing their markers. A thin connector says which marker it belongs to.
    #
    # ONLY IN THE PANEL. On the plane these rules land in the same ninety pixels as five markers
    # and their intervals, which is the congestion the panel exists to relieve.
    #
    # LABELLED IN CONFIG VOCABULARY, NOT GREEK. `summary` and `full` are what a reader types into
    # router.yaml; alpha is the model's parameter and lives in the figure's limitations. The
    # marker end is what the BENCHMARK measures — a fresh context per rung — and is deliberately
    # not offered as a config value, because it is not one.
    lo, mid, hi = (_num(row, k) for k in (_ALPHA_01_KEY, _ALPHA_03_KEY, _ALPHA_10_KEY))
    inset.plot([cost, cost], [perf, level], color=_UNCERTAINTY, lw=0.6, alpha=0.6, zorder=2)
    inset.plot([cost, hi], [level, level], color=_UNCERTAINTY, lw=0.9, ls=_CONTEXT_LS, zorder=2)
    if 0 < lo <= mid:
        # A BAND, not a tick: how far a summariser compresses is not a constant, and one tick
        # asserted a precision the model does not have.
        inset.plot(
            [lo, mid],
            [level, level],
            color=_UNCERTAINTY,
            lw=5.0,
            alpha=0.28,
            solid_capstyle="butt",
            zorder=2,
        )
        inset.annotate(
            "summary",
            xy=(math.sqrt(lo * mid), level),
            xytext=(0, -4),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=_BRACKET_FONT,
            color=_UNCERTAINTY,
        )
    inset.plot([hi], [level], "|", color=_UNCERTAINTY, ms=7, mew=1.2, alpha=0.9, zorder=2)
    inset.annotate(
        "full",
        xy=(hi, level),
        xytext=(3, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=_BRACKET_FONT,
        color=_UNCERTAINTY,
    )
    return hi


def _context_note_row(rows: list[dict]) -> list[str]:
    """The bracket's numbers as a derived note — the n it rests on is never separated from it."""
    out: list[str] = []
    for row in _bracket_rows(rows):
        lo, mid, hi = (_num(row, k) for k in (_ALPHA_01_KEY, _ALPHA_03_KEY, _ALPHA_10_KEY))
        out.append(
            f"context transfer on {row['strategy']}: {usd(_cost(row))} as the benchmark "
            f"measures it (a fresh context per rung — not a config value), "
            f"{usd(lo)}–{usd(mid)} at context_transfer: summary, {usd(hi)} at "
            f"context_transfer: full (the shipped default) — a cost model over measured tokens "
            f"and registry input prices, computed on the token-complete subset "
            f"(n={int(_num(row, _ALPHA_N_KEY))}), asserting no pass rate"
        )
    return out


def _live_points(rows: list[dict]) -> list[tuple[float, float]]:
    """(cache-aware cost, pass rate) for every LIVE row that carries a cost."""
    # Hull over LIVE points only. A mixture frontier is the cost-quality a coin-flip
    # between two REAL policies buys; anchoring it on a strategy that cannot be deployed
    # describes a mixture nobody can construct.
    return [
        (_cost(r), float(r["AvgPerf%"]))
        for r in rows
        if is_live(str(r["strategy"])) and _cost(r) > 0
    ]


def _plot_row(
    ax: Axes, row: Mapping[str, str], pareto: set[str], *, uncertainty: bool
) -> LabelPoint | None:
    """One strategy's mark on one axes. `uncertainty` draws the intervals and cost models."""
    # Split out of `_draw` so the SAME marker code serves the plane and a magnified inset:
    # a panel that re-encoded its markers would be a second encoding of the same rows, and
    # the first time a class changed shape the two would disagree.
    name = str(row["strategy"])
    cost, perf = _cost(row), float(row["AvgPerf%"])
    if cost <= 0:
        return None
    colour, size, marker, filled = _encode(name, pareto)
    xerr_lo = xerr_hi = down = up = 0.0
    if uncertainty:
        xerr_lo, xerr_hi = _draw_cost_uncertainty(ax, row, cost, perf)
        n = int(float(row.get("n_tasks", 0) or 0))
        if n > 0:
            lo, hi = plot_style.wilson_interval(int(float(row.get("n_pass", 0) or 0)), n)
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
    # The label carries its MARKER's colour, so the pairing survives a crowd even before
    # the inset below moves the crowd somewhere with room. Redundant with shape by design
    # — never colour alone.
    return LabelPoint(
        cost, perf, name, down * 100, up * 100, xerr_lo=xerr_lo, xerr_hi=xerr_hi, color=colour
    )


def _draw_hull(ax: Axes, hull: list[tuple[float, float]], *, fill: bool) -> None:
    """The mixture frontier, and the region a coin-flip between two policies already reaches."""
    if len(hull) < 2:
        return
    hx = [p[0] for p in hull]
    hy = [p[1] for p in hull]
    label = "mixture frontier (no router needed)" if fill else None
    ax.plot(hx, hy, color=_HULL, lw=1.6, zorder=2, label=label)
    if fill:
        ax.fill_between(hx, min(hy) - 40, hy, color=_HULL, alpha=0.07, zorder=1)


def _draw(
    ax: Axes, ctx: ctxmod.RoutingContext, pareto: set[str], hull: list[tuple[float, float]]
) -> tuple[str, ...]:
    """Draw the plane; return any note the LAYOUT itself has to declare."""
    rows = ctx.rows
    labels = [lp for r in rows if (lp := _plot_row(ax, r, pareto, uncertainty=True)) is not None]
    _draw_hull(ax, hull, fill=True)

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
    return _magnify(ax, ctx, pareto, labels, hull)


# ---------------------------------------------------------------------------
# The magnified inset.
#
# Five strategies share ONE pass rate inside a third of a decade of cost, which on this
# canvas is about ninety pixels — into which five names of up to twenty-five characters
# must fit. `place_labels` searches eight directions at three radii and then falls back to
# leader slots, and here the FALLBACK SLOTS collide too: the committed plane printed
# "Session-Cascade" and "kNN-cascade" on the same pixels, so two of the five names were not
# merely crowded but unreadable. Tuning the offsets cannot fix that; the room is not there.
#
# Everything below is DERIVED from the rendered geometry, never from a written-down window.
# A hardcoded zoom box is the artefact that rots into a lie: costs move on every re-run and
# every new strategy, and a fixed box would keep pointing at an empty region — or crop the
# very crowd it exists to explain — with nothing failing to say so. So: cluster the points
# in pixels against their own label extents, magnify only a crowd that actually exists,
# place the panel in the emptiest region the canvas has (computed, not a chosen corner),
# and when the data spreads out and no crowd remains, draw NO panel at all and leave the
# plane exactly as it is today. A crowd that finds no free region is NAMED in the figure's
# notes rather than dropped.
# ---------------------------------------------------------------------------

_LABEL_FONT: Final[float] = 7.5
_INSET_FONT: Final[float] = 7.0
_INSET_EDGE: Final[str] = "#555555"
_BRACKET_FONT: Final[float] = 6.0
# Breathing room around every obstacle, as a share of the axes. Small enough that the panel
# can use a genuinely tight gap, large enough that "free" never means "touching".
_CLEAR_FRAC: Final[float] = 0.012
# A panel may take at most this share of the plane: past it the inset stops being an aside
# and starts competing with the figure it explains.
_INSET_MAX_W: Final[float] = 0.52
_INSET_MAX_H: Final[float] = 0.42
# The floor a panel may shrink to: below it the ladder has no room for a second level, and a
# panel that cannot hold its own crowd is worse than none.
_INSET_MIN_H: Final[float] = 0.16


def _frac(ax: Axes, x: float, y: float) -> tuple[float, float]:
    """One data point in AXES FRACTION — the one space a panel position is valid in."""
    # Axes fraction, not pixels: `inset_axes` is placed in it, and it is invariant to
    # wherever the layout engine finally puts the axes.
    px, py = ax.transData.transform((x, y))
    fx, fy = ax.transAxes.inverted().transform((px, py))
    return float(fx), float(fy)


def _to_px(ax: Axes, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """An axes-fraction rect in display pixels, which is what `place_labels` reasons in."""
    x0, y0 = ax.transAxes.transform((box[0], box[1]))
    x1, y1 = ax.transAxes.transform((box[2], box[3]))
    return (float(x0), float(y0), float(x1), float(y1))


def _fill_columns(
    ax: Axes, hull: list[tuple[float, float]], samples: int = 40
) -> list[tuple[float, float, float, float]]:
    """The shaded mixture region as a staircase of columns, in axes fraction."""
    # A rect per hull SEGMENT (the obvious encoding) is useless here: the frontier climbs
    # twenty points across two decades, so one rect at the segment's high end declares the
    # whole plane occupied and no panel ever fits. Sampling the drawn line instead — linear
    # in log-x, which is how matplotlib draws it on this axis — hugs the fill closely enough
    # that the triangle of free canvas ABOVE the frontier stays available, which is exactly
    # where a panel belongs.
    if len(hull) < 2:
        return []
    lo, hi = math.log10(hull[0][0]), math.log10(hull[-1][0])
    knots = [(math.log10(x), y) for x, y in hull]
    edges = [lo + (hi - lo) * i / samples for i in range(samples + 1)]
    out: list[tuple[float, float, float, float]] = []
    for left, right in zip(edges, edges[1:], strict=False):
        top = max(_hull_y(knots, left), _hull_y(knots, right))
        fx0, _ = _frac(ax, 10.0**left, top)
        fx1, fy1 = _frac(ax, 10.0**right, top)
        out.append((min(fx0, fx1), 0.0, max(fx0, fx1), fy1 + _CLEAR_FRAC))
    return out


def _hull_y(knots: list[tuple[float, float]], log_x: float) -> float:
    """The frontier's height at one log-x, interpolated the way the line is drawn."""
    for (lx0, y0), (lx1, y1) in zip(knots, knots[1:], strict=False):
        if lx0 <= log_x <= lx1 and lx1 > lx0:
            return y0 + (y1 - y0) * (log_x - lx0) / (lx1 - lx0)
    return max(y for _lx, y in knots)


def _occupied(
    ax: Axes, labels: list[LabelPoint], hull: list[tuple[float, float]]
) -> list[tuple[float, float, float, float]]:
    """Every part of the plane already spoken for, in axes fraction."""
    lo_x = ax.get_xlim()[0]
    boxes: list[tuple[float, float, float, float]] = []
    for p in labels:
        # A point owns its marker AND its intervals: the naive-cost rule and the bootstrap
        # caps are ink a panel must not land on. The context bracket is not in here because
        # it is no longer drawn on the plane — it is drawn inside the panel.
        fx0, fy0 = _frac(ax, max(p.x - p.xerr_lo, lo_x), p.y - p.yerr_lo)
        fx1, fy1 = _frac(ax, max(p.x + p.xerr_hi, lo_x), p.y + p.yerr_hi)
        boxes.append(
            (
                min(fx0, fx1) - _CLEAR_FRAC,
                min(fy0, fy1) - _CLEAR_FRAC,
                max(fx0, fx1) + _CLEAR_FRAC,
                max(fy0, fy1) + _CLEAR_FRAC,
            )
        )
    boxes.extend(_fill_columns(ax, hull))
    legend = ax.get_legend()
    if legend is not None:
        inv = ax.transAxes.inverted()
        extent = legend.get_window_extent()
        lx0, ly0 = inv.transform((extent.x0, extent.y0))
        lx1, ly1 = inv.transform((extent.x1, extent.y1))
        boxes.append((lx0 - _CLEAR_FRAC, ly0 - _CLEAR_FRAC, lx1 + _CLEAR_FRAC, ly1 + _CLEAR_FRAC))
    return boxes


def _pad(lo: float, hi: float, others: list[float], frac: float, floor: float) -> float:
    """Half the margin a zoom window needs on one axis: label room, floored, then capped."""
    # Three terms, each doing one job. `frac` is the share of the PANEL the label text needs,
    # so the visual margin is constant wherever the crowd sits on a log axis. `floor` opens a
    # window at all when the crowd has no extent on this axis — five points at one pass rate
    # have a zero-height bounding box. The cap is the honest one: never swallow a point that
    # is not in the crowd, because a magnified region must show everything inside it.
    span = max(hi - lo, 0.0)
    want = max(span * frac / max(1.0 - 2.0 * frac, 0.2), floor)
    gaps = [lo - o for o in others if o < lo] + [o - hi for o in others if o > hi]
    room = min(gaps) * 0.5 if gaps else want
    return max(min(want, room), floor * 0.25)


def _window(
    ax: Axes,
    cluster: LabelCluster,
    labels: list[LabelPoint],
    panel: tuple[float, float],
    reach: float = 0.0,
) -> tuple[float, float, float, float]:
    """The data bounds the panel shows: the crowd's own box, padded from PIXEL measurements."""
    # `reach` is the rightmost dollar the panel will DRAW — a context bracket runs past the
    # marker it belongs to, and a window derived from markers alone would crop it. It is
    # measured, not assumed: lengthen a bracket and the window follows it.
    outside = [p for i, p in enumerate(labels) if i not in cluster.members]
    box = ax.get_window_extent()
    wide = max(plot_style.label_extent(labels[i].text, _INSET_FONT)[0] for i in cluster.members)
    row = plot_style.label_extent("", _INSET_FONT)[1]
    fig = ax.get_figure(root=True)
    scale = (fig.dpi if fig is not None else 72.0) / 72.0
    frac_x = min(0.5 * wide * scale / (panel[0] * box.width), 0.22)
    frac_y = min(2.0 * row * scale / (panel[1] * box.height), 0.30)

    lx0, lx1 = math.log10(cluster.x0), math.log10(max(cluster.x1, reach))
    x_lim = ax.get_xlim()
    pad_x = _pad(
        lx0,
        lx1,
        [math.log10(p.x) for p in outside if p.x > 0],
        frac_x,
        0.02 * abs(math.log10(x_lim[1]) - math.log10(x_lim[0])),
    )
    y_lim = ax.get_ylim()
    pad_y = _pad(
        cluster.y0, cluster.y1, [p.y for p in outside], frac_y, 0.02 * abs(y_lim[1] - y_lim[0])
    )
    return (10.0 ** (lx0 - pad_x), 10.0 ** (lx1 + pad_x), cluster.y0 - pad_y, cluster.y1 + pad_y)


def _fit_panel(
    ax: Axes,
    cluster: LabelCluster,
    labels: list[LabelPoint],
    obstacles: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    """The emptiest place a panel big enough for this crowd fits — or None if nowhere does."""
    # Size is asked for by the NAMES the panel must hold, not chosen: the widest one has to
    # fit with margin, and the set has to lay out in about two rows.
    box = ax.get_window_extent()
    fig = ax.get_figure(root=True)
    scale = (fig.dpi if fig is not None else 72.0) / 72.0
    extents = [plot_style.label_extent(labels[i].text, _INSET_FONT) for i in cluster.members]
    widths = [w * scale for w, _h in extents]
    row = extents[0][1] * scale
    want_w = min(max(max(widths) * 1.6, sum(widths) * 0.62) / box.width, _INSET_MAX_W)
    want_h = min(max(7.0 * row / box.height, _INSET_MIN_H), _INSET_MAX_H)
    # HEIGHT FIRST, then width. The panel labels its crowd as a ladder — one level per name —
    # so vertical room is what decides whether every name gets placed, while width only has to
    # hold the widest of them. Searching tallest-first and returning the first fit is that
    # preference made explicit; `want_*` are floors, not targets, so a canvas with little free
    # room still gets a panel rather than none.
    steps = 6
    for i in range(steps + 1):
        height = _INSET_MAX_H + (want_h - _INSET_MAX_H) * i / steps
        for j in range(steps + 1):
            width = _INSET_MAX_W + (want_w - _INSET_MAX_W) * j / steps
            found = plot_style.free_region(width, height, obstacles, margin=0.03, steps=28)
            if found is not None:
                return found
    return None


_NICE: Final[tuple[float, ...]] = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)


def _nice_log_ticks(lo: float, hi: float, count: int = 4) -> list[float]:
    """Round dollar ticks across a NARROW log window, where a decade locator gives one."""
    # `LogLocator` ticks decades and their subdivisions, and a window a third of a decade wide
    # gets one or two ticks out of it — a magnified panel whose x axis says "$30" and nothing
    # else. These are placed by dividing the window geometrically and snapping each to a round
    # number, so the ticks follow wherever the window lands.
    out: list[float] = []
    for i in range(count):
        raw = lo * (hi / lo) ** ((i + 0.5) / count)
        step = 10.0 ** math.floor(math.log10(raw))
        value = min((step * m for m in _NICE), key=lambda v: abs(v - raw))
        if lo <= value <= hi and value not in out:
            out.append(value)
    return out


def _style_inset(inset: Axes) -> None:
    """A panel reads as an aside: its own quiet frame, small ticks, dollars on x."""
    inset.set_facecolor("#ffffff")
    inset.patch.set_alpha(0.94)
    for spine in inset.spines.values():
        spine.set_color(_INSET_EDGE)
        spine.set_linewidth(0.8)
    inset.tick_params(labelsize=6.2, length=2.5, width=0.6, pad=1.5, colors="#555555")
    inset.xaxis.set_major_locator(ticker.FixedLocator(_nice_log_ticks(*inset.get_xlim())))
    inset.xaxis.set_minor_locator(ticker.NullLocator())
    inset.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _p: usd(v, 0)))
    inset.yaxis.set_major_locator(ticker.MaxNLocator(3))
    inset.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _p: f"{v:.1f}%"))
    inset.grid(color="#f0f0f0", lw=0.5)
    inset.set_axisbelow(True)


def _draw_inset(
    ax: Axes,
    ctx: ctxmod.RoutingContext,
    pareto: set[str],
    hull: list[tuple[float, float]],
    box: tuple[float, float, float, float],
    window: tuple[float, float, float, float],
) -> set[str]:
    """Redraw the crowd at magnification. Returns the names the panel labels."""
    # The panel reuses `_plot_row`, so every marker in it carries the same shape and fill the
    # plane gives it — one encoding, not two. What it does NOT redraw is the uncertainty: the
    # intervals span far more than the window, so inside it they would clip to five vertical
    # rules through five markers, which is the picket fence this panel exists to escape. The
    # panel magnifies POSITION and prices CONTEXT; the plane keeps the intervals.
    inset = ax.inset_axes((box[0], box[1], box[2] - box[0], box[3] - box[1]))
    inset.set_xscale("log")
    points = [
        lp for r in ctx.rows if (lp := _plot_row(inset, r, pareto, uncertainty=False)) is not None
    ]
    _draw_hull(inset, hull, fill=False)
    inset.set_xlim(window[0], window[1])
    inset.set_ylim(window[2], window[3])
    _style_inset(inset)
    # Whatever falls INSIDE the window gets its name here — membership of the crowd decided
    # the window, but the window decides the labels, so a point the box happens to contain is
    # never drawn anonymously.
    inside = [p for p in points if window[0] <= p.x <= window[1] and window[2] <= p.y <= window[3]]
    named = {p.text for p in inside}
    # A LADDER, not the plane's offset search. Even magnified these markers are closer together
    # than their names are wide, and the offset search answers that by handing the name to the
    # leader fallback, which pointed "Session-Cascade" at the blue square next to it. Stacking
    # each name over its own marker on a vertical leader cannot mispair, and vertical room is
    # what a panel has.
    obstacles = _draw_brackets(inset, ctx.rows, named, window)
    obstacles.append(_caption(inset))
    missed = plot_style.stack_labels(
        inset,
        inside,
        fontsize=_INSET_FONT,
        obstacles=obstacles,
    )
    ax.indicate_inset_zoom(inset, edgecolor=_INSET_EDGE, alpha=0.9, lw=0.8)
    # A name the ladder could not fit is NOT labelled here, so it must go back to the plane
    # rather than be quietly lost between the two.
    return named - set(missed)


def _caption(inset: Axes) -> tuple[float, float, float, float]:
    """Say on the canvas what the panel is. Returns its pixel rect, so a name cannot land on it."""
    # The rectangle and connectors are the conventional idiom for a magnification, and a reader
    # who does not know the idiom reads a second, contradictory scatter. Six words fix that, and
    # they are the six that matter: the panel adds no data and drops the intervals.
    text = inset.text(
        0.985,
        0.965,
        "magnified — same points, no intervals",
        transform=inset.transAxes,
        ha="right",
        va="top",
        fontsize=_BRACKET_FONT,
        color=_UNCERTAINTY,
    )
    fig = inset.get_figure(root=True)
    if fig is not None:
        fig.draw_without_rendering()
    box = text.get_window_extent()
    return (float(box.x0), float(box.y0), float(box.x1), float(box.y1))


def _draw_brackets(
    inset: Axes, rows: list[dict], named: set[str], window: tuple[float, float, float, float]
) -> list[tuple[float, float, float, float]]:
    """Every bracket the panel draws, returned as pixel rects the labeller must avoid."""
    obstacles: list[tuple[float, float, float, float]] = []
    for row in _bracket_rows(rows):
        name = str(row["strategy"])
        if name not in named:
            continue
        cost, perf = _cost(row), float(row["AvgPerf%"])
        # Halfway between the row's own pass rate and the panel's floor: derived, so a taller
        # window drops the rule further and a bracket always has its label room below it. The
        # rule asserts no pass rate, which is what makes moving it off the line legitimate.
        level = (perf + window[2]) / 2.0
        hi = _draw_bracket(inset, row, cost, perf, level)
        x0, y0 = inset.transData.transform((cost, level))
        x1, y1 = inset.transData.transform((max(hi, cost), perf))
        # The rule is not the whole obstacle: `summary` hangs BELOW it and `full` runs past its
        # right end, and a strategy name placed onto either is the collision this panel exists
        # to remove — moved from the plane into the panel, which would be no fix at all.
        fig = inset.get_figure(root=True)
        scale = (fig.dpi if fig is not None else 72.0) / 72.0
        below = (plot_style.label_extent("summary", _BRACKET_FONT)[1] + 4.0) * scale
        right = plot_style.label_extent("full", _BRACKET_FONT)[0] * scale
        obstacles.append((float(x0), float(y0) - below, float(x1) + right, float(y1)))
    return obstacles


def _magnify(
    ax: Axes,
    ctx: ctxmod.RoutingContext,
    pareto: set[str],
    labels: list[LabelPoint],
    hull: list[tuple[float, float]],
) -> tuple[str, ...]:
    """Label the plane, moving any crowd that cannot be labelled where it sits into a panel."""
    fig: Figure | None = ax.get_figure(root=True)
    if fig is not None:
        fig.draw_without_rendering()
    clusters = plot_style.label_clusters(ax, labels, fontsize=_LABEL_FONT)
    obstacles = _occupied(ax, labels, hull)
    panels: list[tuple[float, float, float, float]] = []
    notes: list[str] = []
    named: set[str] = set()
    for cluster in clusters:
        box = _fit_panel(ax, cluster, labels, obstacles)
        crowd = [labels[i].text for i in cluster.members]
        if box is None:
            # NEVER silently: a crowd left on the plane keeps its leader lines, and the
            # figure's own notes say which one the canvas had no room to magnify.
            notes.append(
                "layout: no region of the canvas was free enough to magnify the cluster "
                + ", ".join(sorted(crowd))
                + " — those names stay on the plane with leader lines"
            )
            continue
        window = _window(
            ax,
            cluster,
            labels,
            (box[2] - box[0], box[3] - box[1]),
            _bracket_extent(ctx.rows, set(crowd)),
        )
        named |= _draw_inset(ax, ctx, pareto, hull, box, window)
        obstacles.append(box)
        panels.append(_to_px(ax, box))
        # The rectangle `indicate_inset_zoom` draws around the magnified region is ink on the
        # PLANE, and a name placed across it is cut in half by its edge.
        rx0, ry0 = _frac(ax, window[0], window[2])
        rx1, ry1 = _frac(ax, window[1], window[3])
        panels.append(_to_px(ax, (rx0, ry0, rx1, ry1)))
        notes.append(
            f"layout: {len(crowd)} strategies within {usd(window[0])}–{usd(window[1])} at "
            f"{window[2]:.1f}–{window[3]:.1f}% are magnified in an inset panel — the SAME "
            f"points at larger scale, not extra data, and without their intervals, which "
            f"stay on the plane"
        )
    legend = ax.get_legend()
    if legend is not None:
        extent = legend.get_window_extent()
        panels.append((extent.x0, extent.y0, extent.x1, extent.y1))
    plot_style.place_labels(
        ax,
        [p for p in labels if p.text not in named],
        fontsize=_LABEL_FONT,
        obstacles=panels,
    )
    return tuple(notes)


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


def per_strategy_note_rows(rows: list[dict]) -> list[str]:
    """The one per-strategy note row per summary row — DERIVED from the results table,
    never hand-written, so a note row cannot drift from the numbers it quotes."""
    pareto = _live_pareto(rows)
    return [
        f"{r['strategy']}: {usd(_cost(r))} cache-aware / {usd(_num(r, _NAIVE_KEY))} naive, "
        f"{float(r['AvgPerf%']):.2f}% "
        f"(n={int(float(r.get('n_tasks', 0) or 0))}, {_standing(str(r['strategy']), pareto)})"
        for r in rows
    ]


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


def _annotations(
    ctx: ctxmod.RoutingContext,
    pareto: set[str],
    aiq: float,
    layout_notes: tuple[str, ...] = (),
) -> Annotations:
    # The SHIPPED DEFAULT, not the kill gate's pre-registered arm. `_draw_cost_uncertainty`
    # draws its bracket and CI bar on LIVE rows only, and the kNN row is a control — so a
    # subtitle quoting kNN's bootstrap span would name a range that appears nowhere on the
    # canvas. The row the subtitle quotes has to be a row the plane draws in full.
    name = ctxmod.DEFAULT_STRATEGY
    router = ctx.row(name)
    baseline = ctx.row(ctxmod.BASELINE_STRATEGY)
    ns = {int(float(r.get("n_tasks", 0) or 0)) for r in ctx.rows}
    model = "cache-aware" if _is_cache_aware(ctx.rows) else "naive — no cache-aware column"
    facts = [f"{len(ctx.rows)} strategies over {max(ns) if ns else 0} scored tasks"]
    if router and baseline and _cost(baseline) > 0:
        ratio = _cost(router) / _cost(baseline)
        facts.append(
            f"{name} {usd(_cost(router))} at "
            f"{float(router['AvgPerf%']):.1f}% vs baseline {usd(_cost(baseline))} at "
            f"{float(baseline['AvgPerf%']):.1f}% ({ratio:.0%} of the bill, {model})"
        )
    if router:
        lo, hi = _num(router, "TotalCost_ci_lower"), _num(router, "TotalCost_ci_upper")
        if hi > lo > 0:
            facts.append(
                f"cost is the wide axis: {name}'s naive total spans "
                f"{usd(lo)}–{usd(hi)} (95% bootstrap)"
            )
    facts.append(f"area under the live frontier {aiq:.3f}")
    notes = per_strategy_note_rows(ctx.rows)
    notes.extend(_context_note_row(ctx.rows))
    notes.extend(_pareto_crosscheck(ctx.rows, pareto))
    subset = next((str(r.get("subset_note") or "") for r in ctx.rows if r.get("subset_note")), "")
    if subset:
        notes.append(f"selection: {subset}")
    if ctx.banner:
        notes.append(ctx.banner)
    # Whatever the LAYOUT had to decide for itself — which crowd it magnified, and any it
    # could not fit. A truncation that reads as full coverage is the one outcome this
    # figure must never ship, so the panel's own limits are published, not implied.
    notes.extend(layout_notes)
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
    # The frontier, the hull and the area scalar are pure functions of the rows, so they are
    # resolved BEFORE anything is drawn — which is what lets the title band reserve its room
    # first. The band takes about a fifth of the canvas height, and `_magnify` measures the
    # axes to decide where a crowd is and where the canvas is empty: measured before the band
    # is reserved, every one of those decisions would be taken on an axes box that never
    # exists. `plot_frame.save` reserves the same rect again; reserving it is idempotent.
    pareto = _live_pareto(ctx.rows)
    hull = plot_style.upper_hull(_live_points(ctx.rows))
    aiq = plot_style.area_under_frontier(hull)
    plot_frame.reserve_band(fig, SPEC.merged(_annotations(ctx, pareto, aiq)))
    layout_notes = _draw(ax, ctx, pareto, hull)
    return plot_frame.save(
        fig,
        ctx.out_dir / "cost_quality_frontier.png",
        SPEC,
        extra=_annotations(ctx, pareto, aiq, layout_notes),
        provenance=ctx.provenance(__name__),
        size=size,
    )
