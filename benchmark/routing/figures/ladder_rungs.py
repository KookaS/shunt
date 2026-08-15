"""ladder_rungs.png — what each escalation rung is measured to buy, beside what the ladder buys."""

# WHY THIS FIGURE EXISTS. Eight figures in this half price a cascade whose rungs are ordered by
# LIST PRICE, and price order is asserted nowhere to be capability order. `ladder_evidence` answers
# that empirically, per rung, off the committed results.csv — and its answer is that the shipped
# default pays for a rung measured worse than the base model and then steps over the cheapest rung
# measured better than it. Until now that answer lived only in a gitignored JSON report.
#
# The figure recomputes the rows through `ladder_evidence.build_evidence` rather than reading
# `reports/ladder_evidence.json`: `reports/` is gitignored, and a figure whose input is not
# committed cannot be regenerated from a fresh clone.
#
# The ladder walk in panel B is DERIVED — the shortlist is read from the packaged router.yaml and
# stepped with the product's own `next_rung_rank` — so a config change moves the drawing instead of
# silently invalidating it. The evidence in panel A reads no router module at all, which is what
# keeps the measurement independent of the policy it is used to judge.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from benchmark import plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
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
    title="The ladder buys the rungs that do not help and jumps over the one that does",
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
        "the ladder ranks by. Right: the same rows, showing which of them the shipped "
        "rank_shortlist actually visits — a filled marker is a rung the ladder buys, a hollow one "
        "is a rung the jump skips — with the visit sequence drawn as a stepped path and the "
        "shortlist's jump drawn as a single long arrow."
    ),
    goal=(
        "Read panel A first and ignore the ladder: only two targets' intervals clear zero on the "
        "helpful side, and both sit far above the base price — while one visited rung's interval "
        "clears zero on the HARMFUL side. Then read panel B on the same rows — every filled "
        "marker below the jump is a rung the shipped default pays for, and the arrow passes over "
        "the cheapest target that measurably helps. A row whose dark interval overlaps its own "
        "pale null band is unmeasured at this n, not shown to be neutral."
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


def rungs(payload: dict[str, Any]) -> list[Rung]:
    """The evidence rows in ascending price order, tagged with the shipped ladder's visits."""
    rows = payload["targets"]
    # Rank 0 is the base; the targets follow it in the price order `build_evidence` sorted them
    # into, so a row's index+1 IS its rank in the pool this corpus prices.
    visited, _shortlist = shipped_walk(len(rows) + 1)
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
            visited=(index + 1) in visited,
        )
        for index, row in enumerate(rows)
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


def _draw_path(ax: Axes, rows: list[Rung], base: str) -> None:
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
    _draw_steps(ax, rows)
    plot_frame.panel_label(ax, "B · what the shipped ladder does with them")


def _draw_steps(ax: Axes, rows: list[Rung]) -> None:
    """The visit sequence: adjacent rungs as a stepped line, the shortlist's jump as an arrow."""
    path = [-1] + [y for y, rung in enumerate(rows) if rung.visited]
    for start, end in zip(path, path[1:], strict=False):
        if end == start + 1:
            ax.plot([_X_MARK, _X_MARK], [start, end], color="#455A64", lw=1.4, zorder=1)
            continue
        # The jump is POLICY, not measurement, so it is the only mark of its kind on the canvas:
        # a bowed arrow, drawn wide enough of the marker column to pass visibly OVER the rows it
        # skips rather than through their markers.
        ax.annotate(
            "",
            xy=(_X_MARK, end),
            xytext=(_X_MARK, start),
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
            (start + end) / 2.0,
            f"shortlist jump\nover {end - start - 1} rungs"
            if end - start - 1 != 1
            else "shortlist jump\nover 1 rung",
            fontsize=7.5,
            color=_JUMP,
            va="center",
            ha="left",
        )


def _annotations(rows: list[Rung], payload: dict[str, Any], shortlist: int) -> Annotations:
    """Every fact on this figure derived from the rows — no target, price or verdict retyped."""
    visited = [r.target for r in rows if r.visited]
    skipped = [r for r in rows if not r.visited]
    helpful_skipped = [r.target for r in skipped if r.verdict == "NET-HELPFUL"]
    return Annotations(
        subtitle_facts=(
            f"base {payload['base_model']} · rank_shortlist={shortlist} visits "
            f"{len(visited)} of {len(rows)} targets",
            "visited: " + ", ".join(f"{r.target} ({r.delta:+.3f})" for r in rows if r.visited),
            "skipped: " + ", ".join(f"{r.target} ({r.delta:+.3f})" for r in skipped),
        ),
        notes=tuple(
            f"{r.target} at {r.price_multiple:.1f}x base: n={r.n}, helps {r.helps}, hurts "
            f"{r.hurts}, delta {r.delta:+.4f} [{r.ci95[0]:+.4f}, {r.ci95[1]:+.4f}], exact null "
            f"[{r.null_ci95[0]:+.4f}, {r.null_ci95[1]:+.4f}], p {p_text(r.p_value)}, {r.verdict}"
            for r in rows
        )
        + (
            (
                "the shortlist jumps over "
                + ", ".join(helpful_skipped)
                + ", the cheapest target whose interval clears zero on this corpus",
            )
            if helpful_skipped
            else ()
        ),
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
    rows = rungs(payload)
    _visited, shortlist = shipped_walk(len(rows) + 1)
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 2, width_ratios=(1.75, 1.0), sharey=True)
    _draw_evidence(axes[0], rows, payload["base_model"])
    _draw_path(axes[1], rows, payload["base_model"])
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
