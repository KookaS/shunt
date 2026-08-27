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

from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.patches import ConnectionPatch, Rectangle

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
        "THREE PANELS, THREE MAGNIFICATIONS, ONE PLANE. Panel A is the whole set — every "
        "strategy that carries a cost, over the full cost range, at full label, interval and "
        "hull treatment. Panel B redraws the outlined box on A at the same full width; panel "
        "C redraws the outlined box on B. Each box is joined to the panel that magnifies it "
        "by two dashed lines running from its lower corners to that panel's upper corners, so "
        "the A-to-B-to-C descent is drawn rather than asserted. The three x axes are three "
        "DIFFERENT scales, each ticked in round dollars across its own window, and only the "
        "bottom panel carries the axis label. In all three, x is total dollars spent over the "
        "scored task set on a log axis, because the strategies span two decades, and y is "
        "pass rate in percent. "
        "NEITHER BOX IS WRITTEN DOWN. B's is the DETAIL WINDOW, derived from the rows every "
        "time the figure is drawn: every strategy whose pass rate clears the best measured "
        "row's own Wilson lower bound, plus the fixed-frontier baseline and the oracle bound, "
        "widened by the room the names need. C's is the group whose MARKERS overlap inside B "
        "— measured on the rendered canvas, not chosen — widened to contain the full length "
        "of every context bracket it draws. "
        "EVERY PANEL NAMES EVERY STRATEGY ITS OWN WINDOW HOLDS, so a name repeats down the "
        "stack at each scale — that is what a magnification is, one point named twice, not "
        "two measurements. A strategy outside the detail window is therefore named on A, on "
        "the plane, at its own cost and pass rate: it is not part of the comparison B is for, "
        "and it is not a footnote either. The ONE exception is the group panel C magnifies. "
        "Above C its markers are a single blob, so a name printed there would land on a mark "
        "nobody can resolve, and those names appear only on C. Any name that could be placed "
        "on no panel at all would be listed in the notes; today there is none. "
        "THE FIGURE DEGRADES ON BOTH AXES, and says which case it is in. With nothing outside "
        "the detail window there is nothing for B to add, and the figure is A plus the "
        "magnified panel. With no overlapping markers there is nothing to magnify, and the "
        "figure is A plus the detail panel. With neither, it is the single plane it has "
        "always been. The layout note states how many levels the figure has. "
        "WHERE THE INTERVALS LIVE. Panels A and B carry them; panel C carries none. The "
        "vertical rule through each marker is its 95% Wilson interval on the pass rate, drawn "
        "BEHIND the markers and faintly: several strategies share one pass rate here, so "
        "their intervals are literally the same rule drawn several times, and at full weight "
        "they read as a picket fence of separate measurements. The thin horizontal line "
        "running from a marker ends at an open tick at the NAIVE per-call sum, so the gap "
        "between the two cost models is a drawn distance; the capped bar at that tick is the "
        "95% bootstrap interval on the naive total, drawn on the naive statistic because that "
        "is the one it was computed for. Both cost marks appear on LIVE strategies only. At "
        "C's scale every one of these runs off the panel, so C leaves them behind and the "
        "panel caption says so. "
        "IN EVERY PANEL the marker sits at CACHE-AWARE cost — what a deployment is billed "
        "once a repeat of the same model banks its cached prefix. "
        "Marker SHAPE carries what a strategy is: circles and the orange diamond can run in "
        "production today, blue squares are blocked — no router.strategy value names them — "
        "an X is a control that must never ship, and a star is a bound that is unreachable by "
        "design. Marker FILL splits the blue squares: a SOLID square is blocked and nothing "
        "equivalent runs today, while a HOLLOW one would mark a mechanism that ships under "
        "another config surface with only its NAME blocked. NO ROW IS HOLLOW TODAY — the one "
        "that was, Session-Cascade, is now live as router.strategy: session_cascade, and "
        "every remaining blue square is solid, i.e. genuinely unrunnable. Only the live "
        "points enter the Pareto test and the shaded mixture region — a frontier anchored on "
        "a strategy the router rejects at boot describes an operating point nobody can buy. "
        "HOW A NAME IS PLACED SAYS WHAT WAS CROWDED. A name printed beside its marker had "
        "room there. A name lifted onto a level above or below the plane, on a vertical "
        "leader down to its own marker, belongs to a group whose NAMES could not all be "
        "printed where they sit — the leader is vertical and lands on the marker it names, so "
        "it cannot be misread onto a neighbour. Panel C labels its whole crowd that way, "
        "because at that scale nothing else can. "
        "Inside panel C, a DEPLOYABLE escalating strategy carries a context-transfer cost "
        "model, drawn as a dashed rule hanging under the marker it belongs to and joined to "
        "it by a thin line: the MARKER is what the benchmark measures, a fresh context on "
        "every rung, which is not a setting anyone can select; the shaded segment is "
        "`context_transfer: summary`, a band rather than a tick because how far a summariser "
        "compresses is not a constant; and the tick at the right end is `context_transfer: "
        "full`, which is what ships today. That rule asserts NO pass rate — it is horizontal, "
        "and its height carries no meaning at all. A deployable escalating row panel C does "
        "not contain carries no bracket on the canvas, and the notes name it and publish its "
        "numbers. The key is drawn ONCE for the whole figure, below the bottom panel, from "
        "every class the figure actually carries."
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
            "detail window",
            "the region of the plane panel B redraws at full width, and the box drawn on "
            "panel A. DERIVED from the rows every time the figure is drawn — the strategies "
            "whose pass rate clears the best measured row's own Wilson lower bound, unioned "
            "with the fixed-frontier baseline and the oracle bound so the two reference "
            "points can never fall out of the picture, then widened by measured label "
            "extents. Never a written-down box: costs move on every re-run, and a fixed one "
            "would end up pointing at empty canvas or cropping the comparison it exists for, "
            "with nothing failing to say so.",
        ),
        (
            "magnified window",
            "the region panel C redraws at full width, and the box drawn on panel B. Same "
            "points, same numbers, no extra data — and no intervals, which stay in the panel "
            "above. Which region it covers is decided from the RENDERED figure, not written "
            "down: it is the group whose MARKERS overlap on the parent panel, which is the "
            "one crowd no label placement can fix. A group whose names merely collide over "
            "separable markers is stacked on levels where it sits and earns no panel. The "
            "window is the overlapping group's own bounding box widened to contain everything "
            "panel C draws, including the full length of a context bracket.",
        ),
        (
            "level",
            "one full-width panel, and one step of magnification. The figure has three when "
            "something falls outside the detail window AND a group of markers overlaps inside "
            "it, two when only one of those holds, and one when neither does. The count is "
            "derived from the rows and the rendered geometry, and the layout note states it.",
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
        "The y axis is clipped to the data range, not to 0-100, and each panel is clipped to "
        "its own window. Every axis label carries the range it shows; the alternative was a "
        "figure on which every strategy is the same flat line.",
        "Each panel below the first shows one REGION of the panel above it, and the region is "
        "derived rather than chosen: the layout note records the windows it landed on, and "
        "the panel above always carries every strategy the window leaves out — named there, "
        "on the plane. A strategy is never dropped; the worst that happens is that it is "
        "compared at a coarser scale than the crowd below it.",
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
        "The difficulty rows' x position includes the MEASURED per-task judge label cost "
        "(gpt-5.6-terra, ~$0.0016/task measured — folded into both cost columns and published as "
        "judge_label_cost in strategy_summary.csv). A judge call is one per task and never "
        "cached, so it is identical under both cost models; every other row carries no judge "
        "bill at all.",
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
    # The frontier's own row, drawn here rather than carried on the hull artist: the hull is
    # drawn on every panel and the key exists once, so the two cannot be the same artist.
    ax.plot([], [], color=_HULL, lw=1.6, label="mixture frontier (no router needed)")
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
    # BELOW THE AXES, NOT INSIDE IT. Ten keyed marks stacked in a corner took roughly a
    # quarter of the plane — and it was the quarter with no data in it, which is the only room
    # a magnified panel can ever be placed in. A key is not data; it does not have to compete
    # with the plot for the plot's own canvas. Two rows under the x label cost a little height
    # and give the whole lower-right back.
    # The drop is measured in INCHES, not in axes fraction. A fraction that clears the tick
    # labels under a full-height plane is a sliver under a third-height panel, and the key
    # would print through the axis it sits below.
    fig = ax.get_figure(root=True)
    if fig is not None:
        fig.draw_without_rendering()
    height = ax.get_window_extent().height or 1.0
    drop = -_LEGEND_GAP_IN * (fig.dpi if fig is not None else 72.0) / height
    ax.legend(
        fontsize=7.5,
        loc="upper left",
        bbox_to_anchor=(0.0, drop),
        ncol=math.ceil((len(rows) + 3) / 2),
        frameon=False,
        borderaxespad=0.0,
        columnspacing=1.4,
        handletextpad=0.5,
    )


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
    # `strategy_summary.csv` also carries alpha columns for Price-Cascade and kNN-semantic-cascade
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
            # BEHIND the markers and quiet. Six strategies share one pass rate on this
            # corpus, so six IDENTICAL Wilson rules stack into a picket fence that reads as
            # six different measurements and is unattributable to any one marker. Nothing is
            # dropped — the interval is still drawn on every point, and the y label says the
            # axis is a window — but it recedes to the background layer it belongs in, and
            # the names are allowed to cross it.
            ax.errorbar(
                cost,
                perf,
                yerr=[[down * 100], [up * 100]],
                fmt="none",
                ecolor=_UNCERTAINTY,
                elinewidth=0.8,
                capsize=2.5,
                alpha=0.45,
                zorder=1.5,
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
    # NO legend label here. The frontier is drawn on every panel, and the key is drawn once for
    # the whole figure; labelling it per panel made the key depend on which panel happened to
    # carry the fill, which is how the deepest level silently dropped the frontier row.
    if len(hull) < 2:
        return
    hx = [p[0] for p in hull]
    hy = [p[1] for p in hull]
    ax.plot(hx, hy, color=_HULL, lw=1.6, zorder=2)
    if fill:
        ax.fill_between(hx, min(hy) - 40, hy, color=_HULL, alpha=0.07, zorder=1)


# ---------------------------------------------------------------------------
# Three levels of magnification, stacked full width.
#
# The plane has to serve two readers at once. One wants to know what the whole strategy set
# costs, across two decades. The other wants to compare the strategies that are actually a
# purchase decision — the ones clustered near the top of the quality range — and those sit in
# a third of a decade where a dozen names cannot be printed. Two earlier answers failed for
# the same reason: an inset that SCAVENGED free space inside a packed plane magnified whatever
# region the canvas happened to leave empty, and a thin marker-only locator strip bought the
# detail axes its room by demoting every excluded strategy to a caption.
#
# The canvas is split into full-width panels instead, one per magnification level, each a real
# plot with its own scale, its own ticks and its own names. A row excluded from the window
# below is named where it sits, on the plane, which is the honest place for it. A child's
# window is drawn as a rectangle on its parent and joined to it by two connectors, which is
# the conventional idiom for a magnification and the only thing that makes the descent
# unambiguous.
#
# BOTH WINDOWS ARE DERIVED, never written down (a hardcoded box rots into a lie the moment a
# cost moves): the detail window is every row whose pass rate clears the best measured row's
# own Wilson lower bound, unioned with the baseline and the bound so the two reference points
# can never fall out of the picture; the magnified window is the group whose markers overlap
# on the rendered parent panel. When either has nothing to say the figure drops that level and
# says how many are left.
# ---------------------------------------------------------------------------

_ZOOM: Final[str] = "#D55E00"
# The connectors are dashed so they cannot be mistaken for the frontier or for a cost mark,
# which are the only other long thin lines on this canvas.
_ZOOM_LS: Final[tuple[int, tuple[int, int]]] = (0, (4, 3))
# Height shares of the plotting area, one per level. The panel that carries the comparison —
# the last one, whichever it is — gets the most room, and panel A the least, because A's job
# is position over two decades and it needs no room for a ladder.
_PANEL_RATIOS: Final[Mapping[int, tuple[float, ...]]] = MappingProxyType(
    {1: (1.0,), 2: (1.0, 1.15), 3: (1.05, 1.15, 0.9)}
)
# Room reserved under the bottom panel for the key, in inches. As a share of the axes it would
# shrink with every extra level, and the key does not.
_LEGEND_GAP_IN: Final[float] = 0.62


def _wilson_floor(rows: list[dict]) -> float:
    """The pass rate a row must clear to be indistinguishable from the best measured one."""
    # DERIVED, and from the interval the figure already draws on every point rather than from
    # a chosen number of percentage points. A band written as "within 2pp" is a constant that
    # knows nothing about how many tasks were scored; this one widens on a small corpus and
    # tightens on a large one, which is the behaviour a "region of interest" should have.
    scored = [
        (float(r["AvgPerf%"]), r)
        for r in rows
        if _cost(r) > 0 and int(float(r.get("n_tasks", 0) or 0)) > 0
    ]
    if not scored:
        return 0.0
    top = max(perf for perf, _r in scored)
    lows = [
        plot_style.wilson_interval(int(float(r.get("n_pass", 0) or 0)), int(float(r["n_tasks"])))[0]
        * 100.0
        for perf, r in scored
        if perf >= top
    ]
    return min(lows) if lows else top


def _detail_names(rows: list[dict]) -> set[str]:
    """The rows the DETAIL axes shows: the top band, plus the baseline and the bound."""
    floor = _wilson_floor(rows)
    priced = [r for r in rows if _cost(r) > 0]
    names = {str(r["strategy"]) for r in priced if float(r["AvgPerf%"]) >= floor}
    # Unioned in, not assumed present in the band: the kill gate is read against the fixed
    # frontier baseline and the oracle bound, so a window that cropped either would answer the
    # figure's own question with two of its three anchors missing.
    names |= {
        str(r["strategy"])
        for r in priced
        if classify(str(r["strategy"])).cls is StrategyClass.BOUND
        or str(r["strategy"]) == ctxmod.BASELINE_STRATEGY
    }
    return names


def _in_window(row: Mapping[str, str], window: tuple[float, float, float, float] | None) -> bool:
    """Is this row's marker inside the detail window? (Everything is, when there is none.)"""
    cost = _cost(row)
    if cost <= 0:
        return False
    if window is None:
        return True
    return window[0] <= cost <= window[1] and window[2] <= float(row["AvgPerf%"]) <= window[3]


def _perf_extent(row: Mapping[str, str]) -> list[float]:
    """Every y this row reaches — the marker and both Wilson caps."""
    perf = float(row["AvgPerf%"])
    n = int(float(row.get("n_tasks", 0) or 0))
    if n <= 0:
        return [perf]
    lo, hi = plot_style.wilson_interval(int(float(row.get("n_pass", 0) or 0)), n)
    return [perf, lo * 100.0, hi * 100.0]


def _detail_window(
    ax: Axes, rows: list[dict], names: set[str]
) -> tuple[float, float, float, float]:
    """The data bounds the detail axes shows, padded from MEASURED label extents."""
    # Same construction as the inset's `_window`: the crowd's own bounding box — here the
    # bounding box of the rows the band selected, including everything they DRAW, so no
    # interval cap and no naive-cost tick is cropped — widened by the room the widest name
    # needs at the font it will be drawn in. Measured, so a longer strategy name opens the
    # window rather than colliding inside it.
    inside = [r for r in rows if str(r["strategy"]) in names and _cost(r) > 0]
    xs = [x for r in inside for x in _extent_xs(r)]
    ys = [y for r in inside for y in _perf_extent(r)]
    fig = ax.get_figure(root=True)
    if fig is not None:
        fig.draw_without_rendering()
    box = ax.get_window_extent()
    scale = (fig.dpi if fig is not None else 72.0) / 72.0
    wide = max(plot_style.label_extent(n, _LABEL_FONT)[0] for n in names) * scale
    row_h = plot_style.label_extent("", _LABEL_FONT)[1] * scale
    # Capped, and the y cap is the tighter one: names now go on levels in the panel BELOW, so
    # the detail panel needs room for its own labels, not for a ladder. At the uncapped rate a
    # third-height panel reserved half its own span for padding and drew the comparison into
    # the middle third of the axes.
    frac_x = min(0.5 * wide / box.width, 0.22)
    frac_y = min(3.0 * row_h / box.height, 0.14)
    lx0, lx1 = math.log10(min(xs)), math.log10(max(xs))
    pad_x = max(lx1 - lx0, 0.05) * frac_x / max(1.0 - 2.0 * frac_x, 0.2)
    pad_y = max(max(ys) - min(ys), 0.5) * frac_y / max(1.0 - 2.0 * frac_y, 0.2)
    return (
        10.0 ** (lx0 - pad_x),
        10.0 ** (lx1 + pad_x),
        min(ys) - pad_y,
        min(max(ys) + pad_y, 100.0),
    )


def _panel_marks(
    ax: Axes,
    ctx: ctxmod.RoutingContext,
    pareto: set[str],
    hull: list[tuple[float, float]],
    window: tuple[float, float, float, float] | None,
    *,
    uncertainty: bool,
    fill: bool,
) -> list[LabelPoint]:
    """Every mark one panel carries: the rows its own window contains, and the frontier."""
    # One encoding for all three levels, because all three call `_plot_row`. A panel that
    # re-encoded its markers would be a second encoding of the same rows, and the first time a
    # class changed shape the levels would disagree about what a strategy is.
    shown = [r for r in ctx.rows if _in_window(r, window)]
    points = [
        lp for r in shown if (lp := _plot_row(ax, r, pareto, uncertainty=uncertainty)) is not None
    ]
    _draw_hull(ax, hull, fill=fill)
    return points


def _panel_limits(
    ax: Axes, rows: list[dict], window: tuple[float, float, float, float] | None
) -> tuple[float, float]:
    """Apply one panel's limits and return the (y floor, y ceiling) its label must quote."""
    # Two cases, one code path. With no window the panel is the whole data range, which is what
    # panel A always shows. With one, the limits ARE the window — derived in `_detail_window`
    # or `_zoom_window`, never written down here.
    ax.set_xscale("log")
    if window is not None:
        ax.set_xlim(window[0], window[1])
        ax.set_ylim(window[2], window[3])
        return (window[2], window[3])
    perfs = [float(r["AvgPerf%"]) for r in rows if _cost(r) > 0]
    # The extent must cover the CI caps too, or a strategy whose cost interval runs off the
    # right edge is silently clipped — which is precisely the strategy the reader needs.
    costs = [x for r in rows if _cost(r) > 0 for x in _extent_xs(r)]
    ax.set_xlim(min(costs) * 0.55, max(costs) * 2.6)
    ax.set_ylim(min(perfs) - 3.0, 100.0)
    return (min(perfs) - 3.0, 100.0)


def _style_panel(ax: Axes, span: tuple[float, float], *, flat_at: float | None) -> None:
    """Round dollars across THIS panel's own window, and a y precision its span earns."""
    # `LogLocator` ticks decades, and a window a third of a decade wide gets one or two ticks
    # out of it — an axis a reader has no position to read against. `_nice_log_ticks` divides
    # the panel's own window geometrically and snaps each sample to a round number, so all
    # three levels are ticked in the same vocabulary at three different scales.
    ticks = _nice_log_ticks(*ax.get_xlim(), count=6)
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    gaps = [b - a for a, b in zip(ticks, ticks[1:], strict=False)]
    dollars = 0 if not gaps or min(gaps) >= 1.0 else (1 if min(gaps) >= 0.1 else 2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _p: usd(v, dollars)))
    if flat_at is not None:
        # A crowd that shares ONE pass rate has no y story, and a window padded around a
        # zero-height box ticks three values that differ in the second decimal — three
        # different-looking numbers for one measurement. ONE tick, at the rate they all sit
        # at, is the honest axis: it says the height is a constant and names the constant.
        ax.set_yticks([flat_at])
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    places = 0 if span[1] - span[0] >= _COARSE_Y_SPAN else 1
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _p: f"{v:.{places}f}%"))
    ax.tick_params(labelsize=7.0, length=2.5, width=0.6, pad=1.5, colors="#555555")
    ax.grid(color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# A y window at least this many points tall reads fine in whole percent; below it the ticks
# would repeat the same integer, so the panel earns a decimal.
_COARSE_Y_SPAN: Final[float] = 8.0


def _indicate(parent: Axes, child: Axes, window: tuple[float, float, float, float]) -> None:
    """Draw the child's window on its parent and join the two — the descent, drawn."""
    # The rectangle alone is ambiguous on a stack of full-width panels: three plots of the same
    # width read as three independent scatters unless something says which is inside which.
    # The connectors are the conventional `indicate_inset_zoom` idiom, run across the gap
    # between two sibling axes instead of into a floating inset, and they land on the child's
    # top corners — which ARE the window's upper corners, because the window is the child's
    # limits. They are drawn as FIGURE artists so neither axes' tight bbox grows to hold them.
    parent.add_patch(
        Rectangle(
            (window[0], window[2]),
            window[1] - window[0],
            window[3] - window[2],
            facecolor=_ZOOM,
            edgecolor=_ZOOM,
            alpha=0.10,
            lw=1.2,
            zorder=1.1,
        )
    )
    floor = parent.get_ylim()[0]
    for x, corner in ((window[0], 0.0), (window[1], 1.0)):
        # TWO SEGMENTS, AND NEITHER CROSSES A MARK. A single corner-to-corner line is the
        # `indicate_inset_zoom` idiom, and on full-width siblings it becomes a diagonal across
        # the whole parent and out into the margin — it crossed the frontier, the intervals and
        # three panels' worth of canvas. The first segment is VERTICAL, at the box's own edge,
        # so it reads as that edge continued and stays in one column of the plane. The second
        # runs from the axis floor to the child's top corner, which lives entirely in the gap
        # between the two panels.
        parent.plot(
            [x, x], [window[2], floor], color=_ZOOM, lw=0.9, ls=_ZOOM_LS, alpha=0.45, zorder=1.05
        )
        link = ConnectionPatch(
            xyA=(x, floor),
            coordsA=parent.transData,
            xyB=(corner, 1.0),
            coordsB=child.transAxes,
            color=_ZOOM,
            lw=0.9,
            ls=_ZOOM_LS,
            alpha=0.55,
        )
        # ADDED TO THE PARENT, not to the figure, and that is the whole trick. Drawn as one of
        # the parent's own artists it is painted during the parent's pass, so the child's own
        # opaque background covers whatever would otherwise run across the child's data, and
        # the visible line stops exactly at the corner it points to. `set_in_layout(False)`
        # keeps a patch that deliberately leaves the axes out of the tight bbox the layout
        # engine and the clipping audit both measure.
        link.set_zorder(1.05)
        link.set_in_layout(False)
        parent.add_artist(link)


def _place_names(ax: Axes, points: list[LabelPoint], own: set[str]) -> tuple[set[str], list[str]]:
    """Label the names THIS panel owns — a ladder for a crowd, the offset search for the rest."""
    # A name belongs to exactly one panel: the deepest one whose window contains it. That is
    # what keeps a strategy from being printed twice at two scales, and it is also what makes
    # panel A the place the excluded rows are named — they are in no deeper window.
    mine = [p for p in points if p.text in own]
    if not mine:
        return set(), []
    fig = ax.get_figure(root=True)
    if fig is not None:
        fig.draw_without_rendering()
    obstacles = _ink_boxes(ax, points)
    placed: set[str] = set()
    notes: list[str] = []
    for cluster in plot_style.label_clusters(ax, mine, fontsize=_LABEL_FONT):
        stacked, note = _ladder(ax, mine, cluster, obstacles)
        placed |= stacked
        notes.append(note)
    # `place_labels` always places — a point with no free slot earns a leader — so anything the
    # ladder withdrew is still named, and the returned set is the whole of `own`.
    plot_style.place_labels(
        ax,
        [p for p in mine if p.text not in placed],
        fontsize=_LABEL_FONT,
        obstacles=obstacles,
    )
    return {p.text for p in mine}, notes


def _draw_level(
    ax: Axes,
    ctx: ctxmod.RoutingContext,
    pareto: set[str],
    hull: list[tuple[float, float]],
    window: tuple[float, float, float, float] | None,
    own: set[str],
) -> tuple[set[str], list[str]]:
    """One un-magnified level: markers, intervals, the hull and its fill, and its own names."""
    points = _panel_marks(ax, ctx, pareto, hull, window, uncertainty=True, fill=True)
    span = _panel_limits(ax, ctx.rows, window)
    _style_panel(ax, span, flat_at=None)
    # SHORT. A rotated y label is as tall as it is long, and at a third of the canvas each
    # panel has room for about twenty characters — past that the three labels overprint each
    # other. What the window IS goes in the panel caption, which runs horizontally.
    ax.set_ylabel(f"pass rate % [{span[0]:.0f}, {span[1]:.0f}]", fontsize=8)
    return _place_names(ax, points, own)


def _draw_magnified(
    ax: Axes,
    ctx: ctxmod.RoutingContext,
    pareto: set[str],
    hull: list[tuple[float, float]],
    window: tuple[float, float, float, float],
) -> tuple[set[str], list[str]]:
    """The deepest level: the overlapping group at full width, laddered, with no intervals."""
    # What it does NOT redraw is the uncertainty: the intervals span far more than this window,
    # so inside it they would clip to a set of vertical rules through markers that all sit at
    # one pass rate — the picket fence this level exists to escape. It magnifies POSITION and
    # prices CONTEXT; the panel above keeps the intervals.
    points = _panel_marks(ax, ctx, pareto, hull, window, uncertainty=False, fill=False)
    span = _panel_limits(ax, ctx.rows, window)
    named = {p.text for p in points}
    levels = {round(p.y, 6) for p in points}
    _style_panel(ax, span, flat_at=next(iter(levels)) if len(levels) == 1 else None)
    ax.set_ylabel(f"pass rate % [{span[0]:.1f}, {span[1]:.1f}]", fontsize=8)
    obstacles = _draw_brackets(ax, ctx.rows, named, window)
    # The panel's FLOOR is an obstacle too. `stack_labels` only requires a rung to sit inside
    # the axes, so the lowest one can land flush with the bottom spine — a hair above the
    # panel's own dollar ticks, which then read as one block of text with the name.
    guard = 0.5 * plot_style.label_extent("", _LABEL_FONT)[1] * _panel_scale(ax)
    obstacles.append((ax.bbox.x0, ax.bbox.y0, ax.bbox.x1, ax.bbox.y0 + guard))
    # A LADDER, not the offset search. Even magnified these markers are closer together than
    # their names are wide, and the offset search answers that by handing the name to the
    # leader fallback, which pointed "Session-Cascade" at the blue square next to it. Stacking
    # each name over its own marker on a vertical leader cannot mispair.
    missed = plot_style.stack_labels(ax, points, fontsize=_LABEL_FONT, obstacles=obstacles)
    return named - set(missed), _bracket_absence_note(ctx.rows, named)


def _bracket_absence_note(rows: list[dict], drawn: set[str]) -> list[str]:
    """Name every deployable escalating row whose context bracket is NOT on the canvas."""
    # The bracket lives on the deepest panel only — on the plane above it lands in the same
    # ninety pixels as five markers and their intervals. A row that panel does not contain
    # therefore shows none, and that has to be said rather than left as an absence.
    absent = sorted(
        str(r["strategy"]) for r in _bracket_rows(rows) if str(r["strategy"]) not in drawn
    )
    if not absent:
        return []
    return [
        "layout: the context-transfer bracket is drawn only where the markers are magnified, so "
        + ", ".join(absent)
        + " carries none on the canvas; its numbers are in the context-transfer note rows above"
    ]


# ---------------------------------------------------------------------------
# Naming the crowd.
#
# Six strategies share ONE pass rate inside a third of a decade of cost, which even at full
# canvas width is about three hundred pixels — into which six names of up to twenty-five
# characters must fit. `place_labels` searches eight directions at three radii and then falls
# back to leader slots, and here the FALLBACK SLOTS collide too: the committed plane printed
# "Session-Cascade" and "kNN-semantic-cascade" on the same pixels, so two of the names were not
# merely crowded but unreadable. Tuning the offsets cannot fix that; the room is not there.
#
# Two different failures, two different remedies. Names that collide over SEPARABLE markers are
# a labelling problem, and the ladder solves it outright: one name per level over a vertical
# leader to its own marker. MARKERS that collide are a resolution problem — a leader lands
# honestly on a blob nobody can see as four marks — and only magnification fixes it, which is
# what the deepest level is for. Everything below is measured on the RENDERED geometry, never
# from a written-down window.
# ---------------------------------------------------------------------------

_LABEL_FONT: Final[float] = 7.5
_BRACKET_FONT: Final[float] = 6.0
# Clear air two markers need between their centres, in marker diameters, to read as two.
_MARK_GAP: Final[float] = 1.5


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
    reach: float = 0.0,
) -> tuple[float, float, float, float]:
    """The data bounds the deepest panel shows: the crowd's own box, padded from PIXELS."""
    # `reach` is the rightmost dollar that panel will DRAW — a context bracket runs past the
    # marker it belongs to, and a window derived from markers alone would crop it. It is
    # measured, not assumed: lengthen a bracket and the window follows it.
    #
    # The label room is measured against the PARENT panel's box because the child is the same
    # width and a comparable height — which is exactly what changed when the magnified region
    # stopped being a floating inset and became a full-width panel of its own.
    outside = [p for i, p in enumerate(labels) if i not in cluster.members]
    box = ax.get_window_extent()
    wide = max(plot_style.label_extent(labels[i].text, _LABEL_FONT)[0] for i in cluster.members)
    row = plot_style.label_extent("", _LABEL_FONT)[1]
    fig = ax.get_figure(root=True)
    scale = (fig.dpi if fig is not None else 72.0) / 72.0
    frac_x = min(0.5 * wide * scale / box.width, 0.22)
    frac_y = min(2.0 * row * scale / box.height, 0.30)

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
    if len(out) >= 2:
        return out
    # A window narrower than the gap between two round numbers snaps every sample to the SAME
    # one, and an axis with a single tick gives a reader no position to read against. Below
    # that width the geometric spacing has nothing left to say, so the fallback steps linearly
    # across the window on a round increment derived from its own span. The ticks still sit at
    # their true log positions; only the choice of WHICH dollars to label changes.
    raw_step = (hi - lo) / count
    if raw_step <= 0:
        return out
    mag = 10.0 ** math.floor(math.log10(raw_step))
    step = min((mag * m for m in (1.0, 2.0, 2.5, 5.0, 10.0)), key=lambda v: abs(v - raw_step))
    first = math.ceil(lo / step) * step
    return [first + i * step for i in range(count + 1) if lo <= first + i * step <= hi]


def _panel_scale(ax: Axes) -> float:
    """Points-to-pixels for the figure this axes belongs to."""
    fig = ax.get_figure(root=True)
    return (fig.dpi if fig is not None else 72.0) / 72.0


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


def _ink_boxes(ax: Axes, labels: list[LabelPoint]) -> list[tuple[float, float, float, float]]:
    """Every mark a name on this panel must not land on, in display pixels."""
    # Real ink only. The shaded mixture region is a 7%-alpha tint carrying no marks — it is the
    # emptiest canvas a panel has — and treating it as an obstacle leaves a ladder nowhere to
    # go. What a name must clear is markers, horizontal cost marks and the key.
    out: list[tuple[float, float, float, float]] = []
    fig = ax.get_figure(root=True)
    scale = (fig.dpi if fig is not None else 72.0) / 72.0
    # The MARKER and the HORIZONTAL cost marks are ink a name must clear. The VERTICAL Wilson
    # whisker deliberately is not: it is drawn behind the markers at low alpha, it is taller
    # than the whole strategy spread, and on this corpus six of them are the SAME interval — so
    # honouring it as an obstacle walls off the one band of empty canvas the ladder exists to
    # use, and a crowd that could be labelled cleanly gets leader lines instead. A name
    # crossing a faint background rule stays readable; a name printed on another name does not.
    pad = _LABEL_FONT * scale
    for p in labels:
        x0, _y0 = ax.transData.transform((p.x - p.xerr_lo, p.y))
        x1, py = ax.transData.transform((p.x + p.xerr_hi, p.y))
        out.append((float(min(x0, x1)), float(py) - pad, float(max(x0, x1)), float(py) + pad))
    legend = ax.get_legend()
    if legend is not None:
        extent = legend.get_window_extent()
        out.append((extent.x0, extent.y0, extent.x1, extent.y1))
    # THE PANEL CAPTION, and every pixel above the axes. `place_labels` offsets a name off its
    # marker with no idea where the axes ends, so a point near the top of a panel gets its name
    # printed over that panel's caption — which on a stack of three is the one string a reader
    # uses to work out what they are looking at. The band is measured in label rows off the
    # axes' own top edge rather than off the caption artist, because a left-aligned title is
    # not the artist `Axes.get_title()` returns and reading the private one would rot.
    out.append((ax.bbox.x0, ax.bbox.y1, ax.bbox.x1, ax.bbox.y1 + _CAPTION_ROWS * pad))
    return out


# How far above a panel is spoken for by its caption, in label rows.
_CAPTION_ROWS: Final[float] = 4.0


def _collide_group(
    ax: Axes, labels: list[LabelPoint], cluster: LabelCluster
) -> LabelCluster | None:
    """The sub-crowd inside this cluster whose MARKERS overlap — the only part magnification
    can help. None when every marker in it is already separable."""
    # `label_clusters` groups on either overlapping names OR overlapping markers, and the two
    # want opposite remedies. Names that collide over separable markers are a labelling
    # problem the ladder solves outright. MARKERS that collide are a resolution problem: a
    # leader lands honestly on a blob nobody can see as four marks, and only magnification
    # fixes it — but only if the window is the BLOB's, not the whole cluster's. Widening the
    # window to a member whose marker was never in the blob costs exactly the magnification
    # the panel was drawn for: on this corpus one such member stretched it from a twentieth
    # of a decade to a fifth, and the four markers stayed touching inside the panel.
    fig = ax.get_figure(root=True)
    scale = (fig.dpi if fig is not None else 72.0) / 72.0
    idx = list(cluster.members)
    pts = [ax.transData.transform((labels[i].x, labels[i].y)) for i in idx]
    # `_encode` sizes markers in points-squared, so the square root of the largest one this
    # crowd draws is the diameter two of them have to clear to read as two marks.
    # A marker's own diameter is the floor, not the test: two squares whose edges just touch
    # are one blob to a reader. `_MARK_GAP` is the clear air two marks need between them to
    # read as two, expressed in marker diameters so it follows the encoding rather than the
    # canvas.
    span = max(math.sqrt(_encode(labels[i].text, set())[1]) for i in idx) * scale * _MARK_GAP
    parent = list(range(len(idx)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if abs(pts[a][0] - pts[b][0]) < span and abs(pts[a][1] - pts[b][1]) < span:
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for a in range(len(idx)):
        groups.setdefault(find(a), []).append(idx[a])
    best = max(groups.values(), key=len)
    return _sub_cluster(labels, best) if len(best) >= 2 else None


def _sub_cluster(labels: list[LabelPoint], members: list[int]) -> LabelCluster:
    """A cluster over part of another one, with its own data box recomputed."""
    return LabelCluster(
        members=tuple(sorted(members)),
        x0=min(labels[i].x for i in members),
        x1=max(labels[i].x for i in members),
        y0=min(labels[i].y for i in members),
        y1=max(labels[i].y for i in members),
    )


def _ladder(
    ax: Axes,
    labels: list[LabelPoint],
    cluster: LabelCluster,
    obstacles: list[tuple[float, float, float, float]],
) -> tuple[set[str], str]:
    """Stack this crowd's names on levels over their own markers. Returns (placed, note)."""
    # ALL OR NOTHING. A partial ladder would hand the survivors back to `place_labels`, whose
    # offset search has already failed on this crowd by construction — that is what made it a
    # cluster — so they would land on the leader fallback and print across the rungs. The
    # rungs are withdrawn instead, and the whole crowd goes to one remedy or the other.
    crowd = {labels[i].text for i in cluster.members}
    rungs = len(ax.texts)
    missed = plot_style.stack_labels(
        ax,
        [labels[i] for i in cluster.members],
        fontsize=_LABEL_FONT,
        obstacles=obstacles,
        marker_pad_pt=10.0,
    )
    if missed:
        for artist in list(ax.texts[rungs:]):
            artist.remove()
        return set(), (
            "layout: the cluster "
            + ", ".join(sorted(crowd))
            + " had no room for a ladder on this panel — too little vertical space for one "
            "level per name — so those names are placed beside their markers with leader "
            "lines instead"
        )
    return crowd, (
        f"layout: {len(crowd)} strategies within {usd(cluster.x0)}–{usd(cluster.x1)} at "
        f"{cluster.y0:.1f}–{cluster.y1:.1f}% have separable markers but names too wide to "
        "print beside them, so the names are stacked on levels, each on a vertical leader to "
        "its own marker — no second copy of the points is drawn"
    )


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


_LETTERS: Final[str] = "ABC"
_LEVEL_WORDS: Final[tuple[str, ...]] = ("", "ONE", "TWO", "THREE")


def _axes(fig: Figure, levels: int) -> list[Axes]:
    """One full-width axes per magnification level, stacked top to bottom."""
    if levels == 1:
        return [fig.subplots()]
    return list(fig.subplots(levels, 1, height_ratios=_PANEL_RATIOS[levels]))


def _zoom_window(
    ax: Axes, ctx: ctxmod.RoutingContext, points: list[LabelPoint]
) -> tuple[float, float, float, float] | None:
    """The deepest panel's window: the densest group whose MARKERS overlap here, or None."""
    # `label_clusters` returns densest first, so the first blob found is the worst crowd on the
    # panel — which is the one a level of magnification is worth spending on.
    for cluster in plot_style.label_clusters(ax, points, fontsize=_LABEL_FONT):
        blob = _collide_group(ax, points, cluster)
        if blob is None:
            continue
        crowd = {points[i].text for i in blob.members}
        return _window(ax, blob, points, _bracket_extent(ctx.rows, crowd))
    return None


def _probe_zoom(
    ctx: ctxmod.RoutingContext,
    pareto: set[str],
    hull: list[tuple[float, float]],
    aiq: float,
    names: set[str],
    *,
    split: bool,
) -> tuple[float, float, float, float] | None:
    """Derive the magnified window by drawing its PARENT at the geometry it would have."""
    # THE LOOP THIS BREAKS. How many panels the figure has depends on whether a group of
    # markers overlaps, and that can only be measured on a rendered canvas whose geometry
    # depends on how many panels there are. So the parent is drawn once, throwaway, at the
    # DEEPER layout: if a blob is found the real figure has exactly the probed geometry, and if
    # none is found the real figure is one panel SHORTER by count and therefore taller per
    # panel, where the markers are further apart in pixels and so cannot have begun to overlap.
    fig = plot_frame.new_figure(plot_frame.WIDE_TALL)
    axes = _axes(fig, 3 if split else 2)
    plot_frame.reserve_band(fig, SPEC.merged(_annotations(ctx, pareto, aiq)))
    parent = axes[-2]
    window = _detail_window(parent, ctx.rows, names) if split else None
    points = _panel_marks(parent, ctx, pareto, hull, window, uncertainty=True, fill=True)
    _panel_limits(parent, ctx.rows, window)
    fig.draw_without_rendering()
    found = _zoom_window(parent, ctx, points)
    plt.close(fig)
    return found


def _panel_caption(
    index: int, windows: list[tuple[float, float, float, float] | None], *, magnified: bool
) -> str:
    """Name one panel, say what it magnifies, and say whether it carries the intervals."""
    window = windows[index]
    if index == 0 or window is None:
        head = f"{_LETTERS[index]} · every strategy, full cost range"
    else:
        head = (
            f"{_LETTERS[index]} · magnified from {_LETTERS[index - 1]}: "
            f"{usd(window[0])}–{usd(window[1])} at {window[2]:.1f}–{window[3]:.1f}%"
        )
    if index + 1 < len(windows):
        head += f" — the outlined box is panel {_LETTERS[index + 1]}"
    if magnified:
        return f"{head} — no intervals at this scale; they stay in panel {_LETTERS[index - 1]}"
    return f"{head} — with the pass-rate and cost intervals"


def _levels_note(
    windows: list[tuple[float, float, float, float] | None], *, magnified: bool
) -> str:
    """The record of how many levels the figure has and what each one is — one note, always."""
    # Non-optional, in every case, including the one-level fallback: a figure that silently
    # drops a level reads exactly like a figure that never had one.
    levels = len(windows)
    parts = [
        f"layout: {_LEVEL_WORDS[levels]} "
        + ("levels of magnification, stacked full width" if levels > 1 else "level, a single plane")
        + " — panel A carries every strategy over the full cost range"
    ]
    for index, window in enumerate(windows[1:], start=1):
        if window is None:
            continue
        derived = (
            "the group whose markers overlap in the panel above, measured on the rendered canvas"
            if magnified and index == levels - 1
            else "the detail window — every strategy whose pass rate clears the best measured "
            "row's own Wilson lower bound, plus the fixed-frontier baseline and the oracle bound"
        )
        parts.append(
            f"panel {_LETTERS[index]} redraws panel {_LETTERS[index - 1]}'s box at full width "
            f"({usd(window[0])}–{usd(window[1])} at {window[2]:.1f}–{window[3]:.1f}%), which is "
            f"{derived}, never written down"
        )
    if not magnified:
        parts.append(
            "no group of markers overlaps at the deepest scale drawn, so there is nothing left "
            "to magnify and the figure stops here"
        )
    if levels == 1 or (levels == 2 and magnified):
        parts.append(
            "every strategy clears the detail band, so a detail panel would redraw the whole of "
            "panel A and none is drawn"
        )
    carriers = ", ".join(_LETTERS[i] for i in range(levels - (1 if magnified else 0)))
    parts.append(
        f"the pass-rate Wilson interval and the two cost marks are drawn in panel(s) {carriers}"
        + (
            f", and panel {_LETTERS[levels - 1]} carries neither — at that scale they run off "
            "the panel — carrying the context-transfer brackets instead"
            if magnified
            else ""
        )
    )
    parts.append(
        "every panel names every strategy its own window holds, so a name repeats down the "
        "stack at each scale"
        + (
            f"; the exception is the group panel {_LETTERS[levels - 1]} magnifies, whose "
            "markers are one blob above it and which is therefore named only there"
            if magnified
            else ""
        )
    )
    return ". ".join(parts) + "."


def _draw_levels(  # noqa: PLR0913 (one argument per already-resolved input; see `_compose`)
    fig: Figure,
    axes: list[Axes],
    ctx: ctxmod.RoutingContext,
    pareto: set[str],
    hull: list[tuple[float, float]],
    windows: list[tuple[float, float, float, float] | None],
    *,
    magnified: bool,
) -> tuple[str, ...]:
    """Draw every level, mark each child's window on its parent, and name each row once."""
    priced = {str(r["strategy"]) for r in ctx.rows if _cost(r) > 0}
    last = len(axes) - 1
    # The key is drawn from EVERY row the figure carries, not just the bottom panel's: a class
    # that appears only in panel A is still on the canvas, and a key that omitted it would leave
    # a marker shape the reader has no way to decode. It goes in FIRST so it is already an
    # obstacle when the names below are placed.
    axis_model = "cache-aware" if _is_cache_aware(ctx.rows) else "naive per-call"
    axes[last].set_xlabel(
        f"total spend over the scored task set — {axis_model} (USD, log) — "
        "each panel has its own scale",
        fontsize=9,
    )
    _legend(axes[last], priced, pareto)
    members = [
        {str(r["strategy"]) for r in ctx.rows if _cost(r) > 0 and _in_window(r, w)} for w in windows
    ]
    # WHO NAMES WHAT. Every panel names every row its own window holds — a magnification names
    # the same point at each scale, the way a map inset does, and a reader who finds "Oracle"
    # in panel A and again in panel B is being told those are one measurement at two scales,
    # not two. The single exception is the magnified group: at the scales above it its markers
    # are one blob, so a name printed there lands on a mark nobody can resolve, and it is
    # printed only where the markers separate. The union over panels is every priced row.
    deep = members[last] if magnified else set()
    owned = [members[i] - (deep if i < last else set()) for i in range(len(members))]
    notes: list[str] = [_levels_note(windows, magnified=magnified)]
    named: set[str] = set()
    for index, (ax, window) in enumerate(zip(axes, windows, strict=True)):
        deepest = magnified and index == last
        # BEFORE the marks and the names: `_ink_boxes` reads the caption's rect off the axes,
        # and a caption set afterwards is a rect that did not exist when the names were placed.
        plot_frame.panel_label(ax, _panel_caption(index, windows, magnified=deepest))
        if deepest and window is not None:
            got, extra = _draw_magnified(ax, ctx, pareto, hull, window)
        else:
            got, extra = _draw_level(ax, ctx, pareto, hull, window, owned[index])
        named |= got
        notes.extend(extra)
        if index and window is not None:
            _indicate(axes[index - 1], ax, window)
    if not magnified:
        notes.extend(_bracket_absence_note(ctx.rows, set()))
    missing = sorted(priced - named)
    if missing:
        # NOTHING IS DROPPED SILENTLY. Every priced row is owned by exactly one panel, so this
        # can only fire if a panel's labeller withdrew a name it owned — and that has to be on
        # the record rather than left for a reader to notice a marker with no name.
        notes.append(
            "layout: no panel could print " + ", ".join(missing) + " — their markers are drawn "
            "and their numbers are in the per-strategy note rows above"
        )
    return tuple(notes)


def _compose(
    ctx: ctxmod.RoutingContext, pareto: set[str], hull: list[tuple[float, float]], aiq: float
) -> tuple[Figure, plot_frame.FigureSize, tuple[str, ...]]:
    """Lay the figure out at one, two or three levels, and draw all of them."""
    names = _detail_names(ctx.rows)
    outside = [r for r in ctx.rows if _cost(r) > 0 and str(r["strategy"]) not in names]
    # DEGRADE HONESTLY, ON BOTH AXES. A detail level earns its panel only when the band
    # actually separates the set: with nothing outside the window it would redraw the whole of
    # panel A, and with fewer than two rows inside it there is no comparison to draw. A
    # magnified level earns its panel only when markers actually overlap.
    split = bool(outside) and len(names) >= 2
    zoom = _probe_zoom(ctx, pareto, hull, aiq, names, split=split)
    levels = 1 + int(split) + int(zoom is not None)
    size = plot_frame.WIDE_TALL if levels > 1 else plot_frame.SINGLE_TALL
    fig = plot_frame.new_figure(size)
    axes = _axes(fig, levels)
    # The band takes about a fifth of the canvas height, and every layout decision below
    # measures an axes box. Taken before the band is reserved, each would be decided on an axes
    # that never exists. `plot_frame.save` reserves the same rect again; idempotent.
    plot_frame.reserve_band(fig, SPEC.merged(_annotations(ctx, pareto, aiq)))
    windows: list[tuple[float, float, float, float] | None] = [None]
    if split:
        windows.append(_detail_window(axes[1], ctx.rows, names))
    if zoom is not None:
        windows.append(zoom)
    notes = _draw_levels(fig, axes, ctx, pareto, hull, windows, magnified=zoom is not None)
    return fig, size, notes


def render(ctx: ctxmod.RoutingContext) -> Path | None:
    """Draw cost_quality_frontier.png, or None when no strategy row carries cost."""
    # The frontier, the hull and the area scalar are pure functions of the rows, so they are
    # resolved BEFORE anything is drawn — the layout needs them to decide how many levels it
    # has, and the title band needs them to reserve its room.
    if not ctx.rows or not any(_cost(r) > 0 for r in ctx.rows):
        return None
    pareto = _live_pareto(ctx.rows)
    hull = plot_style.upper_hull(_live_points(ctx.rows))
    aiq = plot_style.area_under_frontier(hull)
    fig, size, layout_notes = _compose(ctx, pareto, hull, aiq)
    return plot_frame.save(
        fig,
        ctx.out_dir / "cost_quality_frontier.png",
        SPEC,
        extra=_annotations(ctx, pareto, aiq, layout_notes),
        provenance=ctx.provenance(__name__),
        size=size,
    )
