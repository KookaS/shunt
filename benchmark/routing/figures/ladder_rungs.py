"""ladder_rungs.png — what each escalation rung is measured to buy, beside what the ladder buys."""

# WHY THIS FIGURE EXISTS. Eight figures in this half price a cascade whose rungs are ordered by
# LIST PRICE, and price order is asserted nowhere to be capability order. `ladder_evidence` answers
# that empirically, per rung, off the committed results.csv — and its answer is that the shipped
# default pays for a rung measured worse than the base model and then steps over the cheapest rung
# measured better than it. Until now that answer lived only in a gitignored JSON report.
#
# The figure recomputes the rows through `ladder_evidence.build_evidence` rather than reading
# `reports/ladder_evidence.json`: that report is gitignored, and a figure whose input is not
# committed cannot be regenerated from a fresh clone.
#
# The ladder walk in panel B is DERIVED — the shortlist AND the live pool are read from the
# packaged router.yaml (models list + rank_shortlist) and stepped with the product's own
# `next_rung_rank` — so a config change moves the drawing instead of silently invalidating
# it. A benchmark target absent from the live pool is drawn as NOT-LIVE rather than as a
# skipped rung, because the shipped router can no longer route to it. The evidence in panel
# A reads no router module at all, which is what keeps the measurement independent of the
# policy it is used to judge.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing._live_pool import packaged_live_pool, packaged_rank_shortlist
from benchmark.routing.scripts import ladder_evidence

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

    from benchmark.routing.figures import context as ctxmod

_HELPFUL = "#2E7D32"
_HARMFUL = "#C62828"
_NEUTRAL = "#8a8a8a"
_NULL_BAND = "#BDBDBD"
_JUMP = "#6A1B9A"
_ZERO = "#bbbbbb"

_VERDICT_COLOR: Final[dict[str, str]] = {
    "NET-HELPFUL": _HELPFUL,
    "NET-HARMFUL": _HARMFUL,
    "INDISTINGUISHABLE": _NEUTRAL,
}

# Below this, the exact McNemar tail underflows to 0.0 in double precision. A figure that prints
# `p = 0` states an impossible thing, so the tail is rendered as an inequality instead.
_P_FLOOR: Final[float] = 1e-6

# Panel B's columns, in axes coordinates. The rung column sits left, the jump arrow bows out to
# its right, and the two text columns are read as a table beside them.
_X_MARK: Final[float] = 0.10
_X_STATUS: Final[float] = 0.34
_X_VERDICT: Final[float] = 0.58

SPEC = FigureSpec(
    title="The evidence-backed pool, and the rung the price-ranked ladder still skips",
    subtitle="paired on the overlap of scored default-arm runs · exact paired-exchangeability null",
    caveat=(
        "Observational overlap per pair, not a ladder replay: no logged session walked "
        "these rungs in sequence."
    ),
    reading=(
        "Left: for each candidate escalation target, the paired difference in resolve rate "
        "against the cheap base model, computed only on challenges where BOTH models have a "
        "scored default-arm outcome. The dot is the point estimate, the dark whisker the paired "
        "percentile bootstrap over challenges, and the pale whisker behind it the exact paired-"
        "exchangeability null band, so a dot inside the pale band is indistinguishable from "
        "chance. Rows are ordered by list price, cheapest at the bottom, which is the same order "
        "the ladder ranks by. Right: the same rows against the SHIPPED LIVE POOL (read from "
        "src/shunt/config/router.yaml's models list) — a filled marker is a rung the ladder "
        "actually visits, a hollow one a live rung the shortlist jump skips, and a hollow square "
        "is a benchmark target the shipped router no longer routes to (it stays measured, never "
        "served). The visit sequence is drawn as a stepped path and the shortlist's jump as a "
        "single long arrow."
    ),
    goal=(
        "Read panel A first and ignore the ladder: only two targets' intervals clear zero on the "
        "helpful side, and one of them is the most expensive rung measured. Then read panel B on "
        "the same rows: the shipped pool no longer holds the flat-to-harmful rungs (they are "
        "drawn NOT-LIVE), so the ladder's bought rungs are now the ones the evidence supports — "
        "and the remaining defect is visible on the canvas: the price-ranked walk can still jump "
        "over a net-helpful rung when a pricier frontier model's slot falls inside the shortlist. "
        "A row whose dark interval overlaps its own pale null band is unmeasured at this n, not "
        "shown to be neutral."
    ),
    definitions=(
        ("helps", "base failed the challenge, target resolved it"),
        ("hurts", "base resolved the challenge, target failed it"),
        (
            "delta",
            "target resolve rate minus base resolve rate on the shared challenges == "
            "(helps - hurts) / n",
        ),
        (
            "exact null",
            "the two-sided paired randomization test, in closed form — no Monte Carlo, no seed",
        ),
        (
            "rung",
            "a model the ladder can step to; the shortlist walks the cheapest ranks one at a "
            "time and then jumps to the top rank",
        ),
        (
            "not live",
            "a benchmark target absent from router.yaml's models: list — measured for evidence, "
            "never chosen for live inference",
        ),
    ),
    limitations=(
        "Overlap only: each row is scored on the challenges both models were run on, and those "
        "sets differ by row, so the rows are not scored on one common set and their deltas are "
        "not directly comparable to each other.",
        "Coverage is opportunistic, not assigned: which challenges each model was run on was not "
        "randomized, so a target measured on an easier overlap looks better for free.",
        "Default reasoning arm only. A rung the ladder reaches at a raised effort arm is not this "
        "row.",
        "This measures TARGETS, not the ladder: a real ladder pays for a rung only after a "
        "verified recurrence, so the cost of a harmful rung is not the whole of its price.",
        "One base, one corpus. A rung that is net-harmful here is net-harmful on this corpus's "
        "task mix, which is SWE-bench-derived and not your workload.",
        "The live pool's price order — and therefore which rung the shortlist jump skips — "
        "depends on frontier rows whose prices are research-estimated, not live Requesty listings.",
    ),
)


@dataclass(frozen=True)
class Rung:
    """One candidate target: what it is measured to buy, and whether the ladder visits it."""

    target: str
    price_multiple: float
    n: int
    helps: int
    hurts: int
    delta: float
    ci95: tuple[float, float]
    null_ci95: tuple[float, float]
    p_value: float
    verdict: str
    visited: bool
    # Whether the shipped router still routes to this target (router.yaml models: list).
    live: bool
    # The target's rank in the price-ordered LIVE pool (0 = cheapest, the base). Step vs
    # jump is decided on these ranks, NOT on the drawn-row index — the benchmark rows are a
    # subset of the pool. None for a NOT-LIVE target, which holds no live rank.
    live_rank: int | None

    @property
    def colour(self) -> str:
        return _VERDICT_COLOR.get(self.verdict, _NEUTRAL)


def p_text(p_value: float) -> str:
    """The exact tail as text; it underflows to 0.0, and `p = 0` is an impossible claim."""
    return f"< {_P_FLOOR:g}" if p_value < _P_FLOOR else f"{p_value:.2g}"


def shipped_walk(n_models: int) -> tuple[tuple[int, ...], int]:
    """Ranks the shipped ladder visits over an n-model price order, and its shortlist size."""
    # The product's own arithmetic and the packaged config, never a restatement of either: a
    # `rank_shortlist` edit must move this figure rather than leave it describing an old default.
    from shunt.router.escalation import next_rung_rank  # noqa: PLC0415
    from shunt.router.policy import load_router_policy, packaged_policy_path  # noqa: PLC0415

    shortlist = load_router_policy(packaged_policy_path()).escalation.rank_shortlist
    top = n_models - 1
    visits: list[int] = []
    current = 0
    while current < top:
        current = min(next_rung_rank(current, top, shortlist), top)
        visits.append(current)
    return (tuple(visits), shortlist)


def rungs(payload: dict[str, Any], live_pool: list[str] | None = None) -> list[Rung]:
    """The evidence rows in price order, tagged with the shipped ladder's live-pool visits.

    ``live_pool`` is the price-ordered routable set (default: the packaged router.yaml's);
    a benchmark target outside it is tagged ``live=False`` — measured, never served.
    """
    rows = payload["targets"]
    if live_pool is None:
        live_pool = packaged_live_pool()
    # A row's rank is its index in the price-ordered LIVE pool, not its index among the
    # benchmark rows: the drawn rows are a SUBSET of the pool (zai is row 3 here but live
    # rank 1), and a dominated target drawn below the live rungs holds no live rank at all.
    visited_ranks, _shortlist = shipped_walk(len(live_pool))
    rank_of = {name: rank for rank, name in enumerate(live_pool)}
    return [
        Rung(
            target=row["target"],
            price_multiple=row["price_multiple"],
            n=row["n"],
            helps=row["helps"],
            hurts=row["hurts"],
            delta=row["delta"],
            ci95=(row["ci95"][0], row["ci95"][1]),
            null_ci95=(row["null_ci95"][0], row["null_ci95"][1]),
            p_value=row["p_value"],
            verdict=row["verdict"],
            visited=rank_of.get(row["target"]) in visited_ranks,
            live=row["target"] in rank_of,
            live_rank=rank_of.get(row["target"]),
        )
        for row in rows
    ]


def _draw_evidence(ax: Axes, rows: list[Rung], base: str) -> None:
    """Panel A: the paired delta per rung, over its own exact null band."""
    ax.axvline(0.0, color=_ZERO, lw=1.0, zorder=1)
    for y, rung in enumerate(rows):
        ax.plot(
            list(rung.null_ci95),
            [y, y],
            color=_NULL_BAND,
            lw=9.0,
            alpha=0.5,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.plot(
            list(rung.ci95), [y, y], color=rung.colour, lw=2.6, solid_capstyle="round", zorder=3
        )
        ax.plot([rung.delta], [y], "o", color=rung.colour, ms=8, zorder=4)
    ax.set_yticks(list(range(len(rows))))
    ax.set_yticklabels(
        [f"{r.target}\n{r.price_multiple:.1f}x base · n={r.n}" for r in rows], fontsize=8
    )
    ax.text(
        0.0,
        -1.0,
        f"base {base} — every row is the paired difference against it",
        fontsize=7.5,
        color="#666666",
        ha="center",
        va="center",
    )
    ax.set_xlabel("paired difference in resolve rate, target − base", fontsize=9)
    ax.plot([], [], color=_NULL_BAND, lw=9.0, alpha=0.5, label="exact null 95%")
    ax.plot([], [], color=_NEUTRAL, lw=2.6, label="paired bootstrap 95%")
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "A · what each rung is measured to buy")


def _draw_path(ax: Axes, rows: list[Rung], base: str, top_rank: int) -> None:
    """Panel B: the shipped ladder's visit sequence over the same rows, jump drawn as an arrow."""
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([])
    # `sharey` keeps panel A's tick MARKS on this panel, where there is no scale to tick.
    ax.tick_params(left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.plot([_X_MARK], [-1.0], "s", color="#455A64", ms=8)
    ax.text(_X_STATUS, -1.0, "BASE", fontsize=8, va="center", color="#455A64")
    ax.text(_X_VERDICT, -1.0, base, fontsize=8, va="center", color="#666666")
    for y, rung in enumerate(rows):
        if not rung.live:
            # Not a skipped rung — a target the shipped router no longer routes to. Hollow
            # SQUARE to distinguish from a live-but-skipped hollow circle below.
            ax.plot(
                [_X_MARK],
                [y],
                "s",
                ms=7,
                markerfacecolor="white",
                markeredgecolor=_NEUTRAL,
                markeredgewidth=1.2,
            )
            ax.text(_X_STATUS, y, "not live", fontsize=8, va="center", color=_NEUTRAL)
            ax.text(_X_VERDICT, y, rung.verdict, fontsize=8, va="center", color=_NEUTRAL)
            continue
        ax.plot(
            [_X_MARK],
            [y],
            "o",
            ms=9,
            markerfacecolor=rung.colour if rung.visited else "white",
            markeredgecolor=rung.colour,
            markeredgewidth=1.6,
        )
        ax.text(
            _X_STATUS,
            y,
            "VISITED" if rung.visited else "skipped",
            fontsize=8,
            va="center",
            fontweight="bold" if rung.visited else "normal",
            color="#1a1a1a" if rung.visited else _NEUTRAL,
        )
        ax.text(_X_VERDICT, y, rung.verdict, fontsize=8, va="center", color=rung.colour)
    _draw_steps(ax, rows, top_rank)
    plot_frame.panel_label(ax, "B · what the shipped ladder does with them")


def _draw_steps(ax: Axes, rows: list[Rung], top_rank: int) -> None:
    """The walk over LIVE-POOL ranks: a step between adjacent ranks, one jump to the top."""
    # The drawn rows are a SUBSET of the live pool, so a row's index is NOT its rank: zai is
    # row 3 but rank 1, adjacent to the base at rank 0, so the base -> zai segment is a STEP
    # even though the NOT-LIVE rows the benchmark still draws below zai sit between them on
    # the canvas. The walk's single jump (shortlist -> top rank) is the only arrow: it arcs
    # UP over the rows above the last visited rung, never over the NOT-LIVE rows below.
    path = [(-1, 0)] + [
        (y, rung.live_rank)
        for y, rung in enumerate(rows)
        if rung.visited and rung.live_rank is not None
    ]
    for (start_y, start_rank), (end_y, end_rank) in zip(path, path[1:], strict=False):
        if end_rank == start_rank + 1:
            ax.plot([_X_MARK, _X_MARK], [start_y, end_y], color="#455A64", lw=1.4, zorder=1)
            continue
        _draw_jump(ax, start_y, end_y)
    if path and path[-1][1] != top_rank:
        # The top-rank model is not a benchmark row, so it is not drawn: the jump from the
        # last visited rung lands just past the top row, passing over the skipped live rungs
        # above it (kimi-k3) rather than ending at an invisible rung.
        _draw_jump(ax, path[-1][0], len(rows) - 0.5)


def _draw_jump(ax: Axes, start_y: float, end_y: float) -> None:
    """The walk's jump as a bowed arrow passing over the rows between *start_y* and *end_y*."""
    # The jump is POLICY, not measurement, so it is the only mark of its kind on the canvas:
    # a bowed arrow, drawn wide enough of the marker column to pass visibly OVER the rows it
    # skips rather than through their markers.
    ax.annotate(
        "",
        xy=(_X_MARK, end_y),
        xytext=(_X_MARK, start_y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": _JUMP,
            "lw": 1.8,
            "shrinkA": 6,
            "shrinkB": 6,
            "connectionstyle": "arc3,rad=-0.45",
        },
    )
    ax.text(
        _X_MARK + 0.10,
        (start_y + end_y) / 2.0,
        "jump to top rank",
        fontsize=7.5,
        color=_JUMP,
        va="center",
        ha="left",
    )


def _annotations(rows: list[Rung], payload: dict[str, Any], shortlist: int) -> Annotations:
    """Every fact on this figure derived from the rows — no target, price or verdict retyped."""
    visited = [r.target for r in rows if r.visited]
    skipped = [r.target for r in rows if r.live and not r.visited]
    not_live = [r.target for r in rows if not r.live]
    helpful_skipped = [
        r.target for r in rows if r.live and not r.visited and r.verdict == "NET-HELPFUL"
    ]
    facts = [
        f"base {payload['base_model']} · rank_shortlist={shortlist} visits "
        f"{len(visited)} of {len(skipped) + len(visited)} live targets",
        "visited: " + ", ".join(f"{r.target} ({r.delta:+.3f})" for r in rows if r.visited),
        "skipped: "
        + ", ".join(f"{r.target} ({r.delta:+.3f})" for r in rows if r.live and not r.visited),
    ]
    if not_live:
        facts.append("not live (registry only): " + ", ".join(not_live))
    notes = tuple(
        f"{r.target} at {r.price_multiple:.1f}x base: n={r.n}, helps {r.helps}, hurts "
        f"{r.hurts}, delta {r.delta:+.4f} [{r.ci95[0]:+.4f}, {r.ci95[1]:+.4f}], exact null "
        f"[{r.null_ci95[0]:+.4f}, {r.null_ci95[1]:+.4f}], p {p_text(r.p_value)}, {r.verdict}"
        for r in rows
    )
    if helpful_skipped:
        notes += (
            "the shortlist jumps over "
            + ", ".join(helpful_skipped)
            + ", a target whose interval clears zero on this corpus",
        )
    return Annotations(
        subtitle_facts=tuple(facts),
        notes=notes,
        counts=(
            ("targets", len(rows)),
            ("visited_rungs", len(visited)),
            ("paired_challenges", sum(r.n for r in rows)),
        ),
    )


def render(ctx: ctxmod.RoutingContext) -> Path | None:
    """Draw ladder_rungs.png, or return None when results.csv prices no comparable target."""
    payload = ladder_evidence.build_evidence()
    if payload is None or not payload["targets"]:
        return None
    live_pool = packaged_live_pool()
    shortlist = packaged_rank_shortlist()
    rows = rungs(payload, live_pool)
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 2, width_ratios=(1.75, 1.0), sharey=True)
    _draw_evidence(axes[0], rows, payload["base_model"])
    _draw_path(axes[1], rows, payload["base_model"], len(live_pool) - 1)
    for ax in axes:
        ax.set_ylim(-1.6, len(rows) - 0.4)
    return plot_frame.save(
        fig,
        ctx.out_dir / "ladder_rungs.png",
        SPEC,
        extra=_annotations(rows, payload, shortlist),
        provenance=ctx.provenance(__name__),
        size=size,
    )
