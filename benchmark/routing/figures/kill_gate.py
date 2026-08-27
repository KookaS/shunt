"""kill_gate.png — the pre-registered non-inferiority test, drawn for the first time."""

# The gate asks "is the router cheaper at EQUAL quality". `cost_quality_equal.png` used to
# answer the quality half by colouring points whose confidence intervals overlapped, which
# is a failure to reject a difference — not evidence of equivalence. The test that WAS
# pre-registered for this (delta = 5pp paired McNemar, `frontier_estimate.
# mcnemar_noninferiority`, `benchmark.yaml:noninferiority_margin`) has existed unused since
# it was written. This figure runs it, on three evidence bases at once, and states in the
# subtitle what n can actually resolve — which at this scale is less than the margin.

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Final

from benchmark import config, plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import frontier_estimate, plot_style
from benchmark.routing.figures import context as ctxmod
from benchmark.routing.plot_style import usd

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

_NORMAL = NormalDist()
_POWER_Z = _NORMAL.inv_cdf(0.80)  # one-sided 80% power
_ALPHA_Z = _NORMAL.inv_cdf(0.95)  # one-sided 5%

_NON_INFERIOR = "#2E7D32"
_INFERIOR = "#C62828"
_INCONCLUSIVE = "#8a8a8a"
_MARGIN = "#B71C1C"
_BASELINE_C = "#D55E00"
_ROUTER_C = "#0072B2"

# Below this many discordant pairs a rejection is arithmetically valid and evidentially
# thin, and the figure says so on the canvas rather than letting the green stand alone.
_THIN_DISCORDANCE: int = 5

_DECISION_COLOR: Final[dict[str, str]] = {
    "non_inferior": _NON_INFERIOR,
    "inferior": _INFERIOR,
    "inconclusive": _INCONCLUSIVE,
}

SPEC = FigureSpec(
    title="The pre-registered arm misses the 5pp bar on every basis; the shipped default clears it",
    reading=(
        "Left: one row per evidence basis. The dot is the paired pass-rate difference "
        "(the kNN selection rule minus fixed-frontier) in percentage points, the whisker its "
        "95% paired interval, and the dashed red line the pre-registered non-inferiority "
        "margin of -5pp. A row is green only when the Tango score test rejects H0 at that "
        "margin, red when the router is proven WORSE by more than the margin, grey when the "
        "data cannot tell. Right: the same tasks' total spend, baseline dot to router dot; a "
        "leftward arrow is a saving."
    ),
    goal=(
        "Read both panels together, in that order. The left panel is the gate: a saving on "
        "the right is only admissible once the left one is green. On the pre-registered rows "
        "it is not — that arm's quality deficit is several times the margin and the whisker "
        "excludes it on every basis, so the spend reduction beside it is bought at a loss that "
        "was pre-registered as unacceptable rather than at equal quality. The bottom row is a "
        "different arm and a different verdict: the shipped default clears the bar, at four "
        "times the pre-registered arm's bill and still under half the baseline's. It was not "
        "pre-registered, so read it as an observation, not as the gate being met."
    ),
    definitions=(
        (
            "the two router rows",
            # Read off ctxmod.DEFAULT_STRATEGY rather than spelled out: this string named
            # `kNN-cascade` for a release after the shipped default moved, so the terms
            # block described a row the canvas no longer draws. The label and the prose now
            # cannot disagree.
            f"The {ctxmod.ROUTER_STRATEGY} row is the selection rule with the escalation "
            "ladder removed — the pre-registered verdict arm, and not a value "
            f"router.strategy accepts. The {ctxmod.DEFAULT_STRATEGY} row is what a default "
            "install runs: one decision per session, cheapest-first, with the ladder on "
            "top, published without pre-registration.",
        ),
        (
            "non-inferiority",
            "H0: router quality <= baseline - delta, tested by the Tango score statistic on "
            "the discordant pairs. Rejecting it is positive evidence of equivalence; an "
            "overlapping confidence interval is not.",
        ),
        (
            "evidence basis",
            "Which tasks enter. `completed` includes monotone-imputed cells; `measured` "
            "keeps only tasks where neither arm billed a projected cell; `gate sample` is "
            "the subset benchmark.runner.kill_gate itself scores at its default N.",
        ),
        (
            "MDE",
            "The smallest true difference the design detects at 80% power, one-sided. For a "
            "paired test it is driven by the DISCORDANT rate, not by n alone, so it is quoted "
            "both at the observed discordance and at a reference 10% discordance.",
        ),
    ),
    notes=(
        "The margin is read from benchmark.yaml:collect.noninferiority_margin, so the bar "
        "on the canvas is the one that was pre-registered rather than one chosen after "
        "seeing the result.",
        "The pre-registration named the kNN selection rule as the verdict arm, and it is kept "
        "there: repointing it after seeing the data would rewrite the registered test. But "
        "router.strategy defaults to session_cascade, so the shipped default is drawn beside it on "
        "its own row, labelled NOT pre-registered. The gap is a pre-existing defect the rename "
        "exposed, not one it created — the pre-registered arm adjudicates a configuration no "
        "operator can select.",
    ),
    limitations=(
        "The cost panel is naive per-task cost. The gate's real criterion is cache-aware "
        "cost, which the gate bootstraps per task — cache cost is scoped per task (one "
        "task is one session), so a whole-task resample preserves within-task adjacency — "
        "and publishes as a 90% CI in the tracked verdict artifact. See cache_economics.png "
        "for how far the assumed hit rate moves that ratio.",
    ),
)


@dataclass(frozen=True)
class Basis:
    """One evidence basis: the paired outcomes, the test, and the money on those tasks."""

    label: str
    n: int
    diff_pp: float
    lo_pp: float
    hi_pp: float
    decision: str
    b: int
    c: int
    router_cost: float
    baseline_cost: float

    def mde_pp(self, discordance: float | None = None) -> float:
        """Smallest detectable paired difference at 80% power, in percentage points."""
        # McNemar's power is driven by the DISCORDANT rate, not by n alone: concordant
        # pairs carry no information about the difference. Quoting it at the OBSERVED
        # discordance alone would be circular — the same data both estimated the effect
        # and declared the design powerful — so callers pass a reference rate as well.
        if self.n <= 0:
            return float("nan")
        rate = (self.b + self.c) / self.n if discordance is None else discordance
        return 100.0 * (_ALPHA_Z + _POWER_Z) * math.sqrt(max(rate, 1e-9) / self.n)


def _paired_interval(b: int, c: int, n: int) -> tuple[float, float, float]:
    """(difference, lo, hi) for a paired binary contrast, as fractions."""
    if n <= 0:
        return (0.0, 0.0, 0.0)
    diff = (b - c) / n
    var = max((b + c) - (b - c) ** 2 / n, 0.0) / (n * n)
    half = _NORMAL.inv_cdf(0.975) * math.sqrt(var)
    return (diff, diff - half, diff + half)


def _basis(
    label: str,
    router: dict[str, int],
    baseline: dict[str, int],
    router_cost: dict[str, float],
    baseline_cost: dict[str, float],
    margin: float,
) -> Basis | None:
    shared = sorted(set(router) & set(baseline))
    if not shared:
        return None
    r = {t: router[t] for t in shared}
    b_map = {t: baseline[t] for t in shared}
    result = frontier_estimate.mcnemar_noninferiority(r, b_map, margin=margin)
    diff, lo, hi = _paired_interval(result.b, result.c, len(shared))
    return Basis(
        label=label,
        n=len(shared),
        diff_pp=100.0 * diff,
        lo_pp=100.0 * lo,
        hi_pp=100.0 * hi,
        decision=result.decision,
        b=result.b,
        c=result.c,
        router_cost=sum(router_cost.get(t, 0.0) for t in shared),
        baseline_cost=sum(baseline_cost.get(t, 0.0) for t in shared),
    )


def evidence_bases(ctx: ctxmod.RoutingContext, margin: float) -> list[Basis]:
    """The three nested populations the gate can be decided on, widest first."""
    router, baseline = ctxmod.ROUTER_STRATEGY, ctxmod.BASELINE_STRATEGY
    gate_n = int(config.benchmark_params().get("n_default", 20))
    bases: list[Basis] = []
    for label, measured_only in (("completed (imputed)", False), ("measured only", True)):
        found = _basis(
            label,
            ctx.pass_map(router, measured_only=measured_only),
            ctx.pass_map(baseline, measured_only=measured_only),
            ctx.cost_map(router, measured_only=measured_only),
            ctx.cost_map(baseline, measured_only=measured_only),
            margin,
        )
        if found is not None:
            bases.append(found)
    # The gate's own N, drawn from the same task order the runner samples, so the row is
    # the number `make kill-gate` prints rather than a differently-sampled lookalike.
    r_all, b_all = ctx.pass_map(router), ctx.pass_map(baseline)
    sample = sorted(set(r_all) & set(b_all))[:gate_n]
    found = _basis(
        f"gate sample (N={gate_n})",
        {t: r_all[t] for t in sample},
        {t: b_all[t] for t in sample},
        ctx.cost_map(router),
        ctx.cost_map(baseline),
        margin,
    )
    if found is not None:
        bases.append(found)
    bases.extend(_default_basis(ctx, margin))
    return bases


def _default_basis(ctx: ctxmod.RoutingContext, margin: float) -> list[Basis]:
    """The SHIPPED DEFAULT's row, marked as not pre-registered, or [] when it was not scored."""
    # The gate's verdict arm stays the kNN row it was pre-registered on. But `router.strategy`
    # defaults to the cascade, so a reader who takes the pre-registered row home has read a
    # verdict about a configuration nobody runs. Both are drawn, and the label — not a footnote —
    # says which one the pre-registration covers.
    default = ctxmod.DEFAULT_STRATEGY
    found = _basis(
        f"{default} — shipped default, NOT pre-registered",
        ctx.pass_map(default),
        ctx.pass_map(ctxmod.BASELINE_STRATEGY),
        ctx.cost_map(default),
        ctx.cost_map(ctxmod.BASELINE_STRATEGY),
        margin,
    )
    return [found] if found is not None else []


def _draw_forest(ax: Axes, bases: list[Basis], margin: float) -> None:
    ys = list(range(len(bases)))[::-1]
    ax.axvline(0.0, color="#bbbbbb", lw=0.8, zorder=1)
    ax.axvline(
        -100.0 * margin,
        color=_MARGIN,
        lw=1.4,
        ls="--",
        zorder=1,
        label=f"pre-registered margin −{100 * margin:.0f}pp",
    )
    for y, basis in zip(ys, bases, strict=True):
        colour = _DECISION_COLOR.get(basis.decision, _INCONCLUSIVE)
        ax.plot(
            [basis.lo_pp, basis.hi_pp],
            [y, y],
            color=colour,
            lw=2.4,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot([basis.diff_pp], [y], "o", color=colour, ms=8, zorder=3)
        # Start the row label clear of BOTH its own interval and the dashed margin line. An
        # `inferior` row's interval ends left of the margin, so the plain `hi_pp + 0.9` anchor
        # laid the text across the line — the one element a reader must be able to see.
        ax.text(
            max(basis.hi_pp, -100.0 * margin) + 0.9,
            y,
            f"{basis.decision.replace('_', '-')}  (b={basis.b}, c={basis.c})",
            fontsize=7.5,
            va="center",
            ha="left",
            color=colour,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{b.label}\nn={b.n}" for b in bases], fontsize=8)
    ax.set_xlabel("paired pass-rate difference, router − fixed-frontier (pp)", fontsize=9)
    # Tight on the DATA, then `fit_end_labels` widens by exactly what the row texts need.
    # The old `span * 1.75` right limit was a guess laid on top of that measurement, and it
    # left a third of the panel empty: nothing is plotted past +5pp.
    lo = min([b.lo_pp for b in bases] + [-100.0 * margin])
    hi = max([b.hi_pp for b in bases] + [0.0])
    span = max(hi - lo, 6.0)
    ax.set_xlim(lo - span * 0.06, hi + span * 0.06)
    ax.set_ylim(-0.7, len(bases) - 0.3)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "A · quality: is the router within the pre-registered margin?")


def _draw_cost(ax: Axes, bases: list[Basis]) -> None:
    ys = list(range(len(bases)))[::-1]
    for y, basis in zip(ys, bases, strict=True):
        ax.plot(
            [basis.baseline_cost, basis.router_cost],
            [y, y],
            color="#cccccc",
            lw=2.0,
            zorder=1,
            solid_capstyle="round",
        )
        ax.plot([basis.baseline_cost], [y], "D", color=_BASELINE_C, ms=7, zorder=3)
        ax.plot([basis.router_cost], [y], "o", color=_ROUTER_C, ms=7, zorder=3)
        share = basis.router_cost / basis.baseline_cost if basis.baseline_cost else float("nan")
        ax.text(
            max(basis.baseline_cost, basis.router_cost) * 1.35,
            y,
            f"{share:.0%} of baseline",
            fontsize=7.5,
            va="center",
            ha="left",
            color="#444444",
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    ax.set_xscale("log")
    ax.set_xlabel("total spend on that basis (USD, log)", fontsize=9)
    lo = min(min(b.router_cost, b.baseline_cost) for b in bases)
    hi = max(max(b.router_cost, b.baseline_cost) for b in bases)
    ax.set_xlim(max(lo * 0.45, 1e-4), hi * 4.5)
    ax.set_ylim(-0.7, len(bases) - 0.3)
    ax.plot([], [], "D", color=_BASELINE_C, ms=6, label="fixed-frontier")
    ax.plot([], [], "o", color=_ROUTER_C, ms=6, label="that row's router arm")
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "B · cost: the same tasks' bill, baseline → router")


def _annotations(bases: list[Basis], margin: float, banner: str | None) -> Annotations:
    widest = max(bases, key=lambda b: b.n)
    discordant = widest.b + widest.c
    facts = [
        f"paired Tango score at the pre-registered δ={100 * margin:.0f}pp",
        f"n={'/'.join(str(b.n) for b in bases)}",
        (
            f"arms disagree on {discordant} of {widest.n}: MDE ±{widest.mde_pp():.1f}pp there, "
            f"±{widest.mde_pp(0.10):.1f}pp at 10% discordance"
        ),
    ]
    undecided = [b for b in bases if b.decision == "inconclusive"]
    # An INFERIOR verdict outranks both other caveats: a reader who takes the cost panel
    # home without it has read a saving that was never bought at equal quality.
    inferior = [b for b in bases if b.decision == "inferior"]
    caveat = None
    if inferior:
        caveat = (
            f"{len(inferior)} of {len(bases)} rows: WORSE by more than the margin. Those "
            "rows' savings are not at equal quality."
        )
    elif discordant <= _THIN_DISCORDANCE:
        caveat = (
            f"Non-inferiority rests on {discordant} discordant pair(s) — the bar is cleared "
            "on very little evidence."
        )
    elif undecided:
        caveat = (
            f"{len(undecided)} of {len(bases)} bases cannot decide — that saving is at "
            "UNKNOWN, not equal, quality."
        )
    notes = [
        f"{b.label}: Δ={b.diff_pp:+.1f}pp [{b.lo_pp:+.1f}, {b.hi_pp:+.1f}], {b.decision}, "
        f"b={b.b} c={b.c}, router {usd(b.router_cost)} vs baseline {usd(b.baseline_cost)}, "
        f"MDE ±{b.mde_pp():.1f}pp (±{b.mde_pp(0.10):.1f}pp at 10% discordance)"
        for b in bases
    ]
    if banner:
        notes.append(banner)
    return Annotations(
        subtitle_facts=tuple(facts),
        caveat=caveat,
        notes=tuple(notes),
        counts=tuple((b.label, b.n) for b in bases),
    )


def render(ctx: ctxmod.RoutingContext) -> Path | None:
    """Draw kill_gate.png, or return None when neither arm has scored selections."""
    margin = float(config.get().get("collect", {}).get("noninferiority_margin", 0.05))
    bases = evidence_bases(ctx, margin)
    if not bases:
        return None
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 2, width_ratios=(1.45, 1.0))
    _draw_forest(axes[0], bases, margin)
    _draw_cost(axes[1], bases)
    for ax in axes:
        plot_style.fit_end_labels(ax)
    return plot_frame.save(
        fig,
        ctx.out_dir / "kill_gate.png",
        SPEC,
        extra=_annotations(bases, margin, ctx.banner),
        provenance=ctx.provenance(__name__),
        size=size,
    )


def summary_rows(ctx: ctxmod.RoutingContext) -> list[dict[str, Any]]:
    """The figure's numbers, for a truthfulness test that must not re-derive them."""
    margin = float(config.get().get("collect", {}).get("noninferiority_margin", 0.05))
    return [
        {
            "basis": b.label,
            "n": b.n,
            "diff_pp": b.diff_pp,
            "decision": b.decision,
            "mde_pp": b.mde_pp(),
            "router_cost": b.router_cost,
            "baseline_cost": b.baseline_cost,
        }
        for b in evidence_bases(ctx, margin)
    ]
