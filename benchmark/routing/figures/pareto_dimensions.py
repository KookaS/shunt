"""pareto_dimensions.png — the frontier is not one set; it changes with the axis you buy on."""

# ONE FIGURE, NOT FIVE. The SH009 bijection byte-locks a figure's title, subtitle, caveat and
# notes between the manifest and its docs section, so five separate PNGs would mean five
# byte-locked note blocks for five renderings of ONE plane — the exact failure
# cost_quality_frontier.py was written to undo, where four figures of one scatter disagreed on
# the Pareto definition. The report also peaks near 5 GB holding the kNN family's ONNX
# embedders, and every extra figure is drawn inside that peak.
#
# FIVE X AXES, ONE Y. Cost is the axis the whole set already argues about; the other four are
# the ones an operator actually feels and nothing here has ever plotted — provider calls and
# output tokens (round trips and generation, the two latency proxies), the p95 session tail
# (how many sessions the WORST tasks cost, which is what a cascade spends its quality on), and
# the cost CV (whether the bill is predictable per task or carried by a few expensive ones).
# None of the four is a measured latency; they are the countable quantities latency is made of,
# and the limitations say so rather than letting the panel titles imply a stopwatch.
#
# LIVE ROWS ONLY ENTER A FRONTIER. Same rule as cost_quality_frontier: a frontier anchored on a
# hindsight oracle or a strategy the router rejects at boot describes an operating point nobody
# can buy. Bounds, controls and blocked rows are still DRAWN — dropping them would hide measured
# evidence — and carry their class by SHAPE so a reader who cannot separate the hues still sees
# that they are a different kind of thing.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from matplotlib import ticker
from matplotlib.patches import Rectangle

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import plot_style
from benchmark.routing.figures import context as ctxmod
from benchmark.routing.metrics import Axis, pareto_front
from benchmark.routing.plot_style import LabelPoint, usd
from benchmark.routing.strategy_class import StrategyClass, classify, is_live

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from matplotlib.axes import Axes

# The same three inks cost_quality_frontier.py settled on after a contrast audit failed once:
# #9E9E9E and #56B4E9 both fell below 3:1 against the near-white canvas. Class is carried by
# SHAPE and membership by FILL, so neither survives on hue alone.
_PARETO: Final[str] = "#009E73"
_OTHER: Final[str] = "#757575"
_BLOCKED: Final[str] = "#0072B2"

_QUALITY: Final[str] = "AvgPerf%"


@dataclass(frozen=True)
class Dimension:
    """One x axis: the summary column, how it is drawn, and how panel F names it."""

    column: str
    panel: str
    xlabel: str
    name: str
    short: str
    log: bool = False
    scale: float = 1.0


# Every one is a MINIMISED axis against maximised quality, which is what makes the five
# comparable at all: each panel asks the same question, "what does the next unit of pass rate
# cost me in this currency?".
DIMENSIONS: Final[tuple[Dimension, ...]] = (
    Dimension(
        "TotalCost_cacheaware",
        "A · dollars",
        "cache-aware bill over the corpus (log)",
        "billed cost",
        "cost",
        log=True,
    ),
    # PER TASK, OVER THE COUNTED SUBSET -- never the corpus total. `TotalCalls` looks like the
    # right column and is not: an imputed cell carries no `calls` key, both `BilledAttempt`
    # constructors read it through `int(... or 0)`, and the total therefore counts an unrun
    # cell as zero work. The deflation is uneven -- Always-Frontier's path is counted on 95 of
    # 184 tasks against Always-Cheap's 174 -- so ranking the totals against each other ranked
    # partly on who was measured least. `summary._counted_tasks` publishes the honest form.
    Dimension(
        "CallsPerTask",
        "B · round trips",
        "provider calls per task (counted paths only)",
        "provider calls",
        "calls",
    ),
    Dimension(
        "sessions_p95",
        "C · tail sessions",
        "sessions per task, p95",
        "session tail (p95)",
        "tail",
    ),
    Dimension(
        "OutTokPerTask",
        "D · generation",
        "output tokens per task, thousands (counted paths only)",
        "output tokens",
        "tokens",
        scale=1e-3,
    ),
    Dimension(
        "cost_cv",
        "E · predictability",
        # TWO COST MODELS ON ONE CANVAS, LABELLED RATHER THAN HIDDEN. Panel A is the
        # cache-aware total; this panel's CV is the dispersion of the NAIVE per-task
        # cost, because `summary.py` publishes the cache-aware discount only as a row
        # TOTAL -- there is no per-task cache-aware series to take a CV of. Putting both
        # panels on one basis would mean inventing that series; naming the difference on
        # the axis is what the figure can actually support.
        "cost CV across tasks (naive per-task cost)",
        "cost CV",
        "CV",
    ),
)

SPEC = FigureSpec(
    title="The frontier is not one set — it changes with the axis you buy on",
    reading=(
        "FIVE PANELS, ONE PLANE EACH, AND A MATRIX. Panels A-E share the same y axis — pass "
        "rate over the scored corpus — and differ only in what x costs: dollars (log, because "
        "the strategies span two decades), provider calls, the p95 session tail, output "
        "tokens, and the coefficient of variation of per-task cost. Panels B and D are PER "
        "TASK and are counted only over the paths that carry real counts, because an imputed "
        "cell records no calls and no tokens; panel C is counted on every scored task. THE "
        "TWO DOLLAR AXES ARE "
        "NOT ONE COST MODEL: panel A is the cache-aware total, panel E is the dispersion "
        "of the NAIVE per-task cost, and the panel labels say which is which. The green line "
        "in each "
        "panel joins that panel's Pareto frontier, computed on that panel's own two axes. "
        "MARKER SHAPE CARRIES CLASS, never colour alone: a circle can be configured today, a "
        "blue square is blocked, an X is a control that must never ship, a star is a bound "
        "unreachable by design. Only circles enter a frontier, so a hindsight oracle is drawn "
        "beside the strategies it bounds without ever being ranked as one of them. Only the "
        "configurable rows are named on the plane; the rest are read off their shape and the "
        "key. FILL, not hue, carries membership: a marker is filled only when it is on that "
        "panel's frontier, and only a circle can be — every other mark is an outline, which "
        "is also what keeps a frontier circle visible where two rows land on the same point. "
        "PANEL F IS THE FINDING. One row per configurable strategy, one column per dimension, "
        "and a filled cell where that strategy is on that dimension's frontier. Read a ROW to "
        "see how much of a strategy's optimality survives changing the currency; read a COLUMN "
        "to see who you would buy if that column were the only thing you cared about. The "
        "count at the right of each row is how many of the five it holds. THE DOTTED RULE "
        "across A-E is the pass rate most of the drawn rows share EXACTLY; it is named in "
        "panel C's right margin and explained in the notes, and along it the panels rank on x "
        "rather than trading quality for it."
    ),
    goal=(
        "Look for a row of panel F that is filled all the way across — a strategy that is "
        "optimal whatever you are buying. There is none. Then read the row of whichever "
        "column you actually pay in, because that, and not the cost panel alone, is the "
        "frontier your deployment sits on."
    ),
    definitions=(
        (
            "frontier (per dimension)",
            "no other LIVE strategy is at least as good on BOTH that panel's axes and "
            "strictly better on one. Computed per panel by the same routine the summary's "
            "Pareto column uses, over that panel's axis pair rather than a fixed one.",
        ),
        (
            "live",
            "the router may be configured with this strategy today — derived from the "
            "product's own LIVE_STRATEGIES, never restated in the benchmark.",
        ),
        (
            "session tail (p95)",
            "the 95th percentile of sessions spent per task. A cascade buys its pass rate by "
            "re-attempting, and this is what the worst tasks pay for that.",
        ),
        (
            "cost CV",
            "standard deviation of per-task cost divided by the mean. Low means a "
            "predictable bill; high means the total is carried by a few expensive tasks.",
        ),
        (
            "excluded",
            "a row missing a value on a panel's axis is dropped from THAT panel's frontier "
            "rather than read as zero — a missing value coerced to zero is un-dominated by "
            "construction, which certifies 'measured nothing' as optimal.",
        ),
    ),
    notes=(
        "Every panel ranks the CONFIGURABLE rows only. Bounds, controls and blocked rows are "
        "drawn at their measured position and never enter a frontier, because a frontier "
        "anchored on a point no operator can select describes an operating point nobody can "
        "buy.",
        "Calls, output tokens and the session tail are the countable quantities latency is "
        "made of, not a measured latency. Nothing in this corpus times a request.",
        "Rows can land on EXACTLY the same point on these axes — a blocked cascade shares the "
        "shipped default's call count, session tail and output-token total — so a marker may "
        "carry more than one strategy. Every name is lifted onto a level above its own marker "
        "on a vertical leader, which is what keeps a shared point from being read as one row.",
    ),
    limitations=(
        # The prose "not independent" claim this tuple used to carry is now MEASURED and
        # emitted at render time by `_dependence_limit`, from the same rows the panels plot.
        "TWO COST MODELS SHARE THIS CANVAS. Panel A's dollars are the cache-aware total "
        "(`TotalCost_cacheaware`, the column cost_quality_frontier.png ranks on); panel E's "
        "CV is the dispersion of the NAIVE per-task cost (`TotalCost`), because the "
        "cache-aware discount is published as a row total and there is no per-task "
        "cache-aware series to take a CV of. The panel labels carry the difference; a "
        "reader must not read A and E as two views of one bill.",
        "The y axis is the same imputation-biased pass rate every figure in this set uses: "
        "every filled cell is a pass, so all five frontiers sit on quality numbers biased "
        "upward by the share evidence_basis.png publishes.",
        "The dollar axis rests on an ASSUMED cache hit rate, as cost_quality_frontier.png's "
        "does. THE OTHER FOUR ARE COUNTED, NOT MODELLED — but two of them are counted over a "
        "SUBSET, not over the corpus: an imputed cell records no calls and no tokens at all, "
        "so panels B and D are per-task rates over each row's counted paths only, and the "
        "note below gives how much of each row that is. Panel C is counted on every scored "
        "task, because a session count comes from the strategy's own ladder rather than from "
        "a cell. Nothing here is scaled up to a corpus total: an unrun cell contributes "
        "nothing, never a zero.",
    ),
)


def _num(row: Mapping[str, str], key: str) -> float | None:
    """One numeric cell, or None when the column is absent, unparseable or NON-FINITE."""
    # None, never 0.0: `pareto_front` excludes a missing value and publishes the count, and a
    # zero here would instead make the row un-dominated on that axis by construction.
    # A non-finite cell is folded into the SAME None, so `excluded_counts` reports it: `float`
    # happily parses "nan"/"inf", and a NaN that reached the frontier once evicted a real
    # strategy from a published panel while the excluded count still read 0.
    raw = row.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _tie(rows: list[dict[str, str]]) -> tuple[float, int, int]:
    """``(pass rate, how many rows sit on it, passes behind it)`` for the y-axis plateau."""
    # Four of the five panels are a RANKING, not a trade-off, and nothing on the canvas said
    # so: most of the drawn rows share one pass rate exactly, so along that line only x
    # separates them. Measured here rather than asserted, so the annotation disappears if the
    # plateau does.
    quality = [q for r in rows if (q := _num(r, _QUALITY)) is not None]
    if not quality:
        return 0.0, 0, 0
    top = max(quality)
    tied = [r for r in rows if _num(r, _QUALITY) == top]
    n_tasks = max((int(float(r.get("n_tasks", 0) or 0)) for r in tied), default=0)
    return top, len(tied), round(top / 100.0 * n_tasks)


def _tie_edge_label(rows: list[dict[str, str]]) -> str | None:
    """The MARGIN label for the plateau rule — the rule is data, its words are not."""
    # Short enough to live in the right margin of panel C, where the figure edge is the only
    # neighbour. The sentence that explains it is prose and lives in `notes`; a box on the
    # plane occluded the panel it explained.
    top, n_tied, _passes = _tie(rows)
    if n_tied < 2:
        return None
    return f"{top:.2f}% · {n_tied} of {len(rows)} tie"


def _tie_note(rows: list[dict[str, str]]) -> str | None:
    """The plateau sentence, for the NOTES band — off-canvas, over no mark."""
    top, n_tied, passes = _tie(rows)
    if n_tied < 2:
        return None
    n_tasks = max((int(float(r.get("n_tasks", 0) or 0)) for r in rows), default=0)
    bound_ties = any(
        classify(str(r["strategy"])).cls is StrategyClass.BOUND and _num(r, _QUALITY) == top
        for r in rows
    )
    tail = (
        " — the hindsight bound stops here too, so the misses are tasks no model in the "
        "matrix solved"
        if bound_ties
        else ""
    )
    return (
        f"TIE: {n_tied} of the {len(rows)} drawn rows sit at exactly {top:.2f}% "
        f"({passes} passes of {n_tasks}){tail}. Along this line the panels RANK on x; "
        f"they do not trade quality for it."
    )


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation over the paired values — the tie-corrected rho scipy computes."""
    from scipy.stats import spearmanr

    return float(spearmanr(xs, ys).statistic)


def _dependence_limit(
    rows: list[dict[str, str]], front: dict[str, dict[str, bool]], order: list[str]
) -> str:
    """'The axes are not independent' as a MEASUREMENT, not a hedge."""
    # The static prose this replaced left a reader to guess how far from independent five axes
    # that all rise with re-attempts actually are. Both halves are derived from the rows the
    # panels plot: the strongest rank correlations, and the columns of panel F that turn out to
    # carry one identical front.
    pairs: list[tuple[float, str]] = []
    for i, a in enumerate(DIMENSIONS):
        for b in DIMENSIONS[i + 1 :]:
            paired = [
                (x, y)
                for r in rows
                if (x := _num(r, a.column)) is not None and (y := _num(r, b.column)) is not None
            ]
            if len(paired) > 2:
                rho = _spearman([x for x, _ in paired], [y for _, y in paired])
                if math.isfinite(rho):
                    pairs.append((rho, f"{a.name}/{b.name} {rho:+.3f}"))
    strongest = [text for _rho, text in sorted(pairs, key=lambda p: -abs(p[0]))[:3]]
    # Which columns of panel F are the SAME column wearing two names.
    groups: dict[tuple[str, ...], list[str]] = {}
    for dim in DIMENSIONS:
        members = tuple(sorted(n for n in order if front[dim.column].get(n, False)))
        groups.setdefault(members, []).append(dim.name)
    shared = [
        f"{', '.join(names)} select the IDENTICAL front ({', '.join(members) or 'none'})"
        for members, names in groups.items()
        if len(names) > 1
    ]
    measured = (
        f"strongest Spearman rho over the {len(rows)} drawn rows: {'; '.join(strongest)}"
        if strongest
        else "too few rows to correlate the axes"
    )
    duplicate = f" {'; '.join(shared)}." if shared else ""
    return (
        "The five dimensions are NOT independent, and this is the size of it — "
        f"{measured}. A cascade that re-attempts spends more calls, more output tokens and "
        f"more sessions at once, so those axes move together.{duplicate} The figure claims "
        "only that MEMBERSHIP differs across them, which is a statement about the ordering, "
        "not a claim that the axes measure five separate things."
    )


def _plotted(ctx: ctxmod.RoutingContext) -> list[dict[str, str]]:
    """The rows carrying evidence and a pass rate — everything this figure may draw."""
    return [
        r
        for r in ctx.rows
        if _num(r, _QUALITY) is not None and int(float(r.get("n_tasks", 0) or 0)) > 0
    ]


def membership(rows: list[dict[str, str]]) -> dict[str, dict[str, bool]]:
    """dimension column -> {live strategy -> on that dimension's frontier}."""
    live = {str(r["strategy"]): r for r in rows if is_live(str(r["strategy"]))}
    out: dict[str, dict[str, bool]] = {}
    for dim in DIMENSIONS:
        data = {
            name: {_QUALITY: _num(r, _QUALITY), dim.column: _num(r, dim.column)}
            for name, r in live.items()
        }
        result = pareto_front(data, (Axis(_QUALITY, "max"), Axis(dim.column, "min")))
        out[dim.column] = {**{name: False for name in result.excluded}, **result.members}
    return out


def excluded_counts(rows: list[dict[str, str]]) -> dict[str, tuple[str, ...]]:
    """dimension column -> the live strategies dropped from that panel for a missing value."""
    live = {str(r["strategy"]): r for r in rows if is_live(str(r["strategy"]))}
    return {
        dim.column: tuple(sorted(name for name, r in live.items() if _num(r, dim.column) is None))
        for dim in DIMENSIONS
    }


def _counted_coverage(rows: list[dict[str, str]]) -> list[tuple[str, int, int]]:
    """``(strategy, counted tasks, scored tasks)`` for every drawn row that reports it."""
    out: list[tuple[str, int, int]] = []
    for row in rows:
        counted, scored = _num(row, "counted_n"), _num(row, "n_tasks")
        if counted is not None and scored:
            out.append((str(row["strategy"]), int(counted), int(scored)))
    return sorted(out)


def _encode(name: str, on_front: bool) -> tuple[str, str, float, bool, int]:
    """colour, marker, size, filled, zorder — class by SHAPE, membership by FILL."""
    # NOTHING BUT A FRONTIER MEMBER IS FILLED, and the non-live rows sit UNDER the live ones.
    # Both rules exist for the same measured reason: several rows share a value exactly on
    # these axes (one blocked cascade has the same call count, session tail and output-token
    # total as the shipped default), so a solid square drawn last erased a frontier circle
    # completely and the panel showed a green line arriving at a blue square.
    cls = classify(name).cls
    if cls is StrategyClass.BOUND:
        return _OTHER, "*", 120.0, False, 3
    if cls is StrategyClass.CONTROL:
        return _OTHER, "X", 62.0, False, 3
    if cls is StrategyClass.BLOCKED:
        return _BLOCKED, "s", 50.0, False, 3
    return (_PARETO, "o", 135.0, True, 5) if on_front else (_OTHER, "o", 100.0, False, 4)


def _draw_panel(
    ax: Axes,
    dim: Dimension,
    rows: list[dict[str, str]],
    front: dict[str, bool],
    ylim: tuple[float, float],
    tie: float | None,
) -> list[str]:
    """One quality-vs-dimension plane, with that dimension's own frontier drawn on it."""
    labels: list[LabelPoint] = []
    for row in rows:
        name = str(row["strategy"])
        raw = _num(row, dim.column)
        quality = _num(row, _QUALITY)
        if raw is None or quality is None:
            continue
        x = raw * dim.scale
        colour, marker, size, filled, depth = _encode(name, front.get(name, False))
        ax.scatter(
            [x],
            [quality],
            s=size,
            marker=marker,
            facecolors=colour if filled else "none",
            edgecolors=colour,
            linewidths=0.8 if filled else 1.4,
            zorder=depth,
        )
        if is_live(name):
            labels.append(LabelPoint(x=x, y=quality, text=name))

    hull = sorted(
        ((_num(r, dim.column) or 0.0) * dim.scale, _num(r, _QUALITY) or 0.0)
        for r in rows
        if front.get(str(r["strategy"]), False) and _num(r, dim.column) is not None
    )
    if len(hull) > 1:
        ax.plot(
            [p[0] for p in hull],
            [p[1] for p in hull],
            color=_PARETO,
            lw=1.5,
            alpha=0.85,
            zorder=2,
        )

    if dim.log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _p: usd(v, 0)))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_ylim(*ylim)
    if tie is not None:
        # A reading guide, never an encoding: dotted, behind everything, no legend entry. It
        # marks the pass rate several rows share EXACTLY, which is what turns those panels
        # into a ranking on x. The words live off-canvas: a short edge label in panel C's
        # right margin, and the full sentence in the notes band.
        ax.axhline(tie, color=_OTHER, lw=0.7, ls=":", alpha=0.55, zorder=1)
    ax.set_xlabel(dim.xlabel, fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.grid(color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    ax.margins(x=0.18)
    # `margins` only REQUESTS an autoscale; the view limits stay stale until something
    # unstales them, and `transData` is not one of those things. The label ladder measures in
    # display pixels, so without this it places every name against the pre-margin transform —
    # which put one name off the panel edge and dropped it entirely.
    ax.autoscale_view()
    plot_frame.panel_label(ax, dim.panel)
    # `stack_labels`, not the offset search: several strategies share one pass rate here and
    # two of them are EXACTLY coincident on panel C, where every free slot is nearer some other
    # marker and the offset search's leader fallback puts a name confidently beside a
    # neighbour. A ladder cannot mispair — each name sits over its own marker on a vertical
    # leader — and it returns what it could not place rather than dropping it silently.
    return plot_style.stack_labels(ax, labels, fontsize=6.8, marker_pad_pt=9.0)


def _draw_matrix(ax: Axes, order: list[str], front: dict[str, dict[str, bool]]) -> None:
    """Panel F: one filled cell per (strategy, dimension) pair that is on the frontier."""
    ax.set_xlim(-0.6, len(DIMENSIONS) + 0.45)
    ax.set_ylim(-0.95, len(order) - 0.05)
    for yi, name in enumerate(order):
        y = len(order) - 1 - yi
        held = 0
        for xi, dim in enumerate(DIMENSIONS):
            on = front[dim.column].get(name, False)
            held += int(on)
            ax.add_patch(
                Rectangle(
                    (xi - 0.4, y - 0.3),
                    0.8,
                    0.6,
                    facecolor=_PARETO if on else "none",
                    edgecolor=_PARETO if on else _OTHER,
                    linewidth=1.0,
                    zorder=2,
                )
            )
            ax.text(
                xi,
                y,
                "✓" if on else "·",
                ha="center",
                va="center",
                fontsize=10 if on else 9,
                color="#ffffff" if on else _OTHER,
                zorder=3,
            )
        ax.text(
            len(DIMENSIONS) + 0.05,
            y,
            f"{held}/{len(DIMENSIONS)}",
            ha="left",
            va="center",
            fontsize=8,
            color=plot_frame.INK,
        )
    ax.set_yticks([len(order) - 1 - i for i in range(len(order))])
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xticks(range(len(DIMENSIONS)))
    ax.set_xticklabels([d.short for d in DIMENSIONS], fontsize=8)
    # The header belongs ABOVE the matrix: below it, the five column names shared a strip with
    # the fill key and the two rows of text ran together.
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        -0.55,
        -0.75,
        "filled = on that dimension's frontier   ·   open = dominated on it",
        fontsize=7.5,
        color=plot_frame.MUTED,
        ha="left",
        va="center",
    )
    plot_frame.panel_label(ax, "F · membership, per dimension")


def _key(ax: Axes, drawn: set[str]) -> None:
    """One key for the whole figure, built from the classes actually on the canvas."""
    classes = {classify(n).cls for n in drawn}
    entries: list[tuple[str, str, float, bool, str]] = [
        (_PARETO, "o", 70, True, "configurable, on this panel's frontier"),
        (_OTHER, "o", 55, False, "configurable, dominated here"),
    ]
    if StrategyClass.BLOCKED in classes:
        entries.append((_BLOCKED, "s", 55, False, "blocked — not configurable"))
    if StrategyClass.CONTROL in classes:
        entries.append((_OTHER, "X", 60, False, "control — never shippable"))
    if StrategyClass.BOUND in classes:
        entries.append((_OTHER, "*", 95, False, "bound — unreachable by design"))
    for colour, marker, size, filled, label in entries:
        ax.scatter(
            [],
            [],
            s=size,
            marker=marker,
            facecolors=colour if filled else "none",
            edgecolors=colour,
            linewidths=0.8 if filled else 1.4,
            label=label,
        )
    ax.legend(fontsize=6.8, loc="lower right", frameon=True, framealpha=0.92, handletextpad=0.4)


def _annotations(
    rows: list[dict[str, str]],
    order: list[str],
    front: dict[str, dict[str, bool]],
    dropped: dict[str, tuple[str, ...]],
    n_drawn: int,
    unplaced: Sequence[str],
) -> Annotations:
    """Subtitle facts and note rows derived from the membership actually computed."""
    held = {name: sum(front[d.column].get(name, False) for d in DIMENSIONS) for name in order}
    best = max(held.values(), default=0)
    total_excluded = sum(len(v) for v in dropped.values())
    facts = [
        f"{len(order)} configurable strategies over {len(DIMENSIONS)} dimensions, "
        f"{n_drawn} rows drawn",
        f"best row holds {best} of {len(DIMENSIONS)}"
        + ("" if best == len(DIMENSIONS) else " — no strategy holds all five"),
    ]
    default_held = held.get(ctxmod.DEFAULT_STRATEGY)
    if default_held is not None:
        facts.append(
            f"the shipped default ({ctxmod.DEFAULT_STRATEGY}) holds "
            f"{default_held} of {len(DIMENSIONS)}"
        )
    facts.append(
        "no row excluded for a missing value"
        if total_excluded == 0
        else f"{total_excluded} row-dimension pair(s) excluded for a missing value"
    )
    # THE PLATEAU SENTENCE LIVES HERE, not on the plane. It is prose about the whole canvas,
    # so it belongs in the band SH009 byte-locks to the docs section; the dotted rule and its
    # margin label carry the same fact on the figure without covering a mark. First, because
    # without it four of the five panels read as trade-offs when they are rankings.
    tie_note = _tie_note(rows)
    notes = ([tie_note] if tie_note is not None else []) + [
        f"{dim.name}: frontier = "
        + (", ".join(sorted(n for n in order if front[dim.column].get(n, False))) or "none")
        + (
            ""
            if not dropped[dim.column]
            else f"; excluded for a missing value: {', '.join(dropped[dim.column])}"
        )
        for dim in DIMENSIONS
    ]
    notes += [f"{name}: on {held[name]} of {len(DIMENSIONS)} frontiers" for name in order]
    # HOW MUCH OF EACH ROW PANELS B AND D ACTUALLY COUNT. Published per row rather than as one
    # corpus figure because the whole defect this replaced was that the shares are UNEQUAL —
    # a single average would hide exactly the thing that made ranking the totals wrong.
    notes += [
        f"counted paths for panels B and D — {name}: {counted} of {scored} scored task(s)"
        for name, counted, scored in _counted_coverage(rows)
    ]
    if unplaced:
        # A crowd the ladder could not finish is PUBLISHED rather than silently half-labelled.
        notes.append("names the label ladder could not place: " + ", ".join(sorted(set(unplaced))))
    return Annotations(
        subtitle_facts=tuple(facts),
        notes=tuple(notes),
        limitations=(_dependence_limit(rows, front, order),),
        caveat=(
            "Only configurable strategies enter a frontier; bounds, controls and blocked "
            "rows are drawn, never ranked."
        ),
        counts=(
            ("configurable_strategies", len(order)),
            ("dimensions", len(DIMENSIONS)),
            ("excluded_pairs", total_excluded),
        ),
    )


def render(ctx: ctxmod.RoutingContext) -> Path | None:
    """Draw pareto_dimensions.png — five quality-vs-cost-of-something planes plus the matrix."""
    rows = _plotted(ctx)
    live = [str(r["strategy"]) for r in rows if is_live(str(r["strategy"]))]
    if len(rows) < 2 or not live:
        return None
    front = membership(rows)
    dropped = excluded_counts(rows)
    order = sorted(
        live,
        key=lambda n: (-sum(front[d.column].get(n, False) for d in DIMENSIONS), n),
    )
    quality = [q for r in rows if (q := _num(r, _QUALITY)) is not None]
    span = max(quality) - min(quality)
    # Asymmetric headroom on purpose: five strategies share one pass rate at the top of this
    # corpus, so the label ladder climbs from there and needs the room above it. The axis is
    # clipped to the data range rather than 0-100 for the reason cost_quality_frontier gives —
    # a 20-point band on a full scale is a flat line — and every panel carries the same limits
    # so a reader comparing panels is comparing one y axis.
    # Capped at 100: a pass rate cannot exceed it, and an axis running to 105 to hold a label
    # ladder invents headroom the measurement does not have.
    ylim = (min(quality) - max(span * 0.10, 0.5), min(100.0, max(quality) + max(span * 0.34, 1.5)))

    size = plot_frame.WIDE_TALL
    fig, axes = plot_frame.subplots(size, 2, 3)
    flat = list(axes.flat)
    unplaced: list[str] = []
    tie_value, n_tied, _passes = _tie(rows)
    tie = tie_value if n_tied > 1 else None
    for ax, dim in zip(flat, DIMENSIONS, strict=False):
        unplaced += _draw_panel(ax, dim, rows, front[dim.column], ylim, tie)
    edge = _tie_edge_label(rows)
    if edge is not None and tie is not None:
        # THE RULE IS DATA, ITS LABEL IS NOT. The dotted line stays on the plane because seven
        # strategies really do sit on it; the words move to the margin, which costs no data
        # area. Panel C is the one scatter whose right edge is the FIGURE edge — A and B abut a
        # neighbour's tick labels, D and E sit under them — so a right-hand tick there collides
        # with nothing, and constrained layout reserves the width for it.
        edge_ax = flat[2].secondary_yaxis("right")
        edge_ax.set_yticks([tie])
        edge_ax.set_yticklabels([edge], fontsize=6.8, color=_OTHER)
        edge_ax.tick_params(length=2.5, width=0.6, colors=_OTHER)
        edge_ax.spines["right"].set_visible(False)
    for ax in (flat[0], flat[3]):
        ax.set_ylabel("pass rate (%)", fontsize=8.5)
    _key(flat[0], {str(r["strategy"]) for r in rows})
    _draw_matrix(flat[5], order, front)
    return plot_frame.save(
        fig,
        ctx.out_dir / "pareto_dimensions.png",
        SPEC,
        extra=_annotations(rows, order, front, dropped, len(rows), unplaced),
        provenance=ctx.provenance(__name__),
        size=size,
    )
