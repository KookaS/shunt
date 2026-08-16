#!/usr/bin/env python3
"""exploration_cost.png — what exploration costs, and what it buys."""

# Replays the SHIPPED exploration policy over the committed results matrix (no live calls)
# and answers one question in two panels: A puts both arms in the cost/quality plane with
# their intervals, so the price of exploring and the quality it bought are one geometry;
# B shows where the budget went — the running exploratory share of decisions against the
# router's own confidence-weighted counter and the configured cap.
#
# WHAT IS NOT DRAWN AND WHY. The matrix is static, so an exploratory pull can never improve
# a later decision: this is exploration's cost with its learning benefit pinned to zero, the
# pessimistic half of the ledger. That is the figure's one red caveat rather than a footnote.
# The intervals are percentile-bootstrap over tasks, not Wilson: the exploring arm's per-task
# pass is a mean over stochastic seeds, not a Bernoulli count, so a Wilson interval on it
# would be a binomial claim about a number that is not binomial. On the baseline arm (which
# IS binary) the two agree to within a point, and the bootstrap is what the paired deltas
# are computed with, so one interval family is used for both arms.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Final

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

from benchmark import config, plot_frame  # noqa: E402
from benchmark.plot_frame import Annotations, FigureSpec  # noqa: E402
from benchmark.routing import plot_style, summary  # noqa: E402
from benchmark.routing.exploration_replay import ReplayReport, evaluate  # noqa: E402
from benchmark.routing.figures import context as ctxmod  # noqa: E402
from shunt.router.policy import load_router_policy  # noqa: E402

# `python -m` sets __name__ to "__main__", which would land in the figure
# manifest instead of the module that drew it.
_GENERATOR = "benchmark.routing.scripts.plot_exploration"

if TYPE_CHECKING:
    from matplotlib.axes import Axes

SPEC = FigureSpec(
    title="Exploration costs more, buys no pass rate, and its learning benefit is unmeasurable",
    subtitle=(
        "offline Direct-Method replay of the shipped policy — recorded outcomes, no live calls"
    ),
    caveat=(
        "Static matrix: an exploratory pull can never inform a later decision — cost only, "
        "learning benefit pinned to zero."
    ),
    reading=(
        "A: the cost/quality plane. Each arm is one point — mean cost per task on x, pass "
        "rate on y — with 95% bootstrap intervals on both axes; the arrow runs from the "
        "exploit-only arm to the exploring one, and the box states the PAIRED difference, "
        "which is what this slice has the power to resolve. B: where the budget went. The "
        "orange curve is the running share of decisions that were exploratory as the replay "
        "proceeds, the dotted line is the router's own confidence-weighted explore counter "
        "at the end of the run, and the dashed line is the configured cap it is measured "
        "against."
    ),
    goal=(
        "Read the boxed paired difference in A, not the two overlapping marginal intervals: "
        "look for the cost delta and whether the pass delta clears zero. Then read B for "
        "whether the policy is spending its budget at all — a counter far under its cap "
        "means the measured overhead is not the overhead of a saturated budget."
    ),
    definitions=(
        ("exploration", "occasionally routing to a non-preferred model to learn its outcome"),
        ("exploit-only", "the same shipped policy with exploration switched off"),
        ("paired difference", "per-task gap between the arms, so shared task noise cancels"),
        (
            "explore_budget_frac",
            "cap on the router's own confidence-weighted explore counter — neighbourhood "
            "costs, not realized spend, so it is not comparable to the measured spend ratio",
        ),
    ),
    notes=(
        "The replay is EXACT, not simulated: on a fully dense sub-grid the recorded outcome "
        "is looked up and no request is sent.",
        "The realized spend ratio exceeding the cap is expected, not a bug — the cap counts "
        "the router's confidence-weighted neighbourhood costs, not realized spend.",
        "Unscorable cells are skipped and counted, never guessed.",
        "Intervals are percentile-bootstrap over tasks rather than Wilson: the exploring "
        "arm's per-task pass is a mean over stochastic seeds, not a Bernoulli count.",
    ),
    limitations=(
        "The outcome matrix is static, so an exploratory pull can never improve a later "
        "decision: this measures exploration's COST with its learning benefit set to zero, "
        "the pessimistic half of the ledger — not a verdict on whether exploration pays.",
        "The dense slice is found greedily, not optimally, and comes from a single workload.",
        "How much of the corpus exploration left un-probed is NOT drawn: the replay report "
        "carries aggregate decisions, not the per-(task, model) probe record that question "
        "needs.",
    ),
)

# Documented data-viz palette (light mode): slot-1 blue vs slot-2 orange for the two
# arms (worst-adjacent CVD clears the target), plus muted ink for the cap reference.
BASELINE_COLOR: Final = "#2a78d6"
EXPLORE_COLOR: Final = "#eb6834"
CAP_COLOR: Final = "#898781"
_INK: Final = "#0b0b0b"
_INK2: Final = "#52514e"
_GRID: Final = "#e1e0d9"

FIGURE_NAME: Final = "exploration_cost.png"
DEFAULT_OUT: Final = (
    Path(__file__).resolve().parents[3] / "docs/assets/figures/routing" / FIGURE_NAME
)


def _paired_tasks(report: ReplayReport) -> int:
    """Tasks both arms actually scored — the n behind every paired number on the canvas."""
    return len(set(report.per_task_baseline_pass) & set(report.per_task_exploration_pass))


def _panel_cost_quality(ax: Axes, report: ReplayReport) -> None:
    """Both arms in the cost/quality plane, with the paired difference stated."""
    arms = (
        ("exploit-only", report.baseline_cost, report.baseline_pass_rate, BASELINE_COLOR),
        ("exploration on", report.exploration_cost, report.exploration_pass_rate, EXPLORE_COLOR),
    )
    ax.set_axisbelow(True)
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.annotate(
        "",
        xy=(report.exploration_cost.value, report.exploration_pass_rate.value),
        xytext=(report.baseline_cost.value, report.baseline_pass_rate.value),
        arrowprops={"arrowstyle": "-|>", "color": "#b8b6ae", "lw": 1.6, "shrinkA": 9, "shrinkB": 9},
        zorder=2,
    )
    for label, cost, quality, colour in arms:
        ax.errorbar(
            [cost.value],
            [quality.value],
            xerr=[[cost.value - cost.lo], [cost.hi - cost.value]],
            yerr=[[quality.value - quality.lo], [quality.hi - quality.value]],
            fmt="o",
            ms=9,
            color=colour,
            ecolor=colour,
            elinewidth=1.4,
            capsize=4,
            zorder=4,
        )
        # Above the upper whisker, never below: a label under a point sits on its own
        # lower error bar.
        ax.annotate(
            f"{label}\n${cost.value:.4f}/task · pass {quality.value:.1%}",
            xy=(cost.value, quality.hi),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=colour,
            fontweight="bold",
        )

    costs = [report.baseline_cost, report.exploration_cost]
    rates = [report.baseline_pass_rate, report.exploration_pass_rate]
    x_lo, x_hi = min(c.lo for c in costs), max(c.hi for c in costs)
    y_lo, y_hi = min(r.lo for r in rates), max(r.hi for r in rates)
    ax.set_xlim(x_lo - 0.28 * (x_hi - x_lo), x_hi + 0.28 * (x_hi - x_lo))
    # Headroom above for the point labels, below for the paired-difference box.
    ax.set_ylim(y_lo - 0.13 * (y_hi - y_lo) - 0.045, y_hi + 0.42 * (y_hi - y_lo))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"${v:.4f}"))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0%}"))
    ax.set_xlabel("mean cost per task (USD)", fontsize=9, color=_INK)
    ax.set_ylabel("pass rate", fontsize=9, color=_INK)
    # The marginal intervals overlap heavily; the PAIRED difference is what this slice can
    # actually resolve, so it is stated rather than left to the eye.
    ax.annotate(
        f"paired difference (exploration − exploit-only), n={_paired_tasks(report)} tasks\n"
        f"pass {100 * report.pass_delta.value:+.1f}pp "
        f"[{100 * report.pass_delta.lo:+.1f}, {100 * report.pass_delta.hi:+.1f}]     "
        f"cost {report.cost_delta.value:+.5f} "
        f"[{report.cost_delta.lo:+.5f}, {report.cost_delta.hi:+.5f}]     "
        f"bill {report.cost_multiple:.2f}×",
        xy=(0.5, 0.015),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=8,
        color=_INK,
        bbox={"boxstyle": "round,pad=0.4", "fc": "#f4f4f2", "ec": _GRID},
        zorder=5,
    )
    plot_frame.panel_label(ax, "A · what it costs, what it buys")


def _panel_budget(ax: Axes, report: ReplayReport, budget_frac: float) -> None:
    """Where the exploration budget went: share of decisions, counter, and cap."""
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.6)
    shares = report.explore_share_by_round
    share_max = max(shares) if shares else 0.0
    if shares:
        ax.plot(
            np.arange(1, len(shares) + 1),
            shares,
            color=EXPLORE_COLOR,
            lw=2,
            zorder=4,
            label=f"exploratory share of decisions (ends at {shares[-1]:.1%})",
        )
        ax.set_xlim(1, len(shares))
    used = report.budget_explore_ratio / budget_frac if budget_frac else float("nan")
    ax.axhline(
        report.budget_explore_ratio,
        color=BASELINE_COLOR,
        ls=":",
        lw=1.6,
        zorder=3,
        label=(
            f"router's own explore counter {report.budget_explore_ratio:.3f} "
            f"({used:.0%} of the cap consumed)"
        ),
    )
    ax.axhline(
        budget_frac,
        color=CAP_COLOR,
        ls="--",
        lw=1.4,
        zorder=3,
        label=f"explore_budget_frac cap = {budget_frac:g}",
    )
    ax.set_xlabel("decisions routed (replay round)", fontsize=9, color=_INK)
    ax.set_ylabel("fraction of decisions / of the budget counter", fontsize=9, color=_INK)
    # Just enough headroom above the cap line to keep it off the frame; the worst-seed
    # numbers are text in the manifest, so they must not stretch the axis into whitespace.
    ax.set_ylim(0, max(budget_frac, report.budget_explore_ratio, share_max) * 1.30)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.2f}"))
    leg = ax.legend(fontsize=7.5, loc="lower right", framealpha=0.95)
    leg.get_frame().set_edgecolor(_GRID)
    plot_frame.panel_label(ax, "B · where the exploration budget went")


def _missing_concentration_limit(report: ReplayReport) -> list[str]:
    """Say so when the dropped baseline cells sit in one model rather than spreading."""
    by_model = report.baseline_missing_by_model
    if not by_model:
        return []
    top_model, top_n = max(by_model.items(), key=lambda kv: kv[1])
    if top_n < report.baseline_missing:
        return []
    where = "outside the dense slice" if top_model not in report.slice_.models else "in the slice"
    return [
        f"THE DROPPED BASELINE CELLS ARE NOT A RANDOM SAMPLE: all {report.baseline_missing} "
        f"unscorable exploit-only cells are {top_model}, a model {where} — so the "
        "exploit-only arm is systematically missing that model's tasks, not a random "
        "subset. The overhead is therefore reported PAIRED, over only the tasks both "
        "arms scored"
    ]


def _frontier_limit(report: ReplayReport) -> list[str]:
    """The dense slice maximises cells, so it can land on cheap models only."""
    # When it does, the measured "exploration overhead" is the overhead of exploring
    # between cheap models — not the shipped policy's, where an exploratory pull can land
    # on the frontier model at ~40x the price.
    slice_ = report.slice_
    pricing = config.enabled_pricing()
    priced = [(m, config.cost_per_1m(m, pricing)) for m in slice_.models]
    enabled = config.enabled_models() or slice_.models
    all_enabled = [(m, config.cost_per_1m(m, pricing)) for m in enabled]
    top_slice = max((c for _m, c in priced), default=0.0)
    top_enabled = max((c for _m, c in all_enabled), default=0.0)
    if not (priced and all_enabled and top_slice < top_enabled):
        return []
    absent = [m for m, _c in sorted(all_enabled, key=lambda t: -t[1]) if m not in slice_.models]
    return [
        f"NO FRONTIER ARM IN THIS SLICE: the dense sub-grid covers only "
        f"{', '.join(slice_.models)} — the priciest model here is "
        f"{plot_style.usd(top_slice)}/Mtok against {plot_style.usd(top_enabled)} across all "
        f"enabled models ({', '.join(absent)} are absent). The exploration overhead measured "
        f"here is between CHEAP models and is a LOWER BOUND on the shipped policy's, where "
        f"an exploratory pull can land on the frontier model"
    ]


def _annotations(report: ReplayReport, budget_frac: float) -> Annotations:
    """Runtime-derived canvas facts plus the manifest record: slice, seeds, skipped cells."""
    slice_ = report.slice_
    base, expl = report.baseline_pass_rate, report.exploration_pass_rate
    n_paired = _paired_tasks(report)
    limits: list[str] = []
    if base.lo <= expl.hi and expl.lo <= base.hi:
        limits.append(
            f"The two marginal pass-rate CIs overlap ([{base.lo:.0%}, {base.hi:.0%}] vs "
            f"[{expl.lo:.0%}, {expl.hi:.0%}]) — at {n_paired} paired tasks only the paired "
            "difference separates the arms"
        )
    limits.extend(_frontier_limit(report))
    limits.extend(_missing_concentration_limit(report))
    return Annotations(
        subtitle_facts=(
            f"dense slice {len(slice_.tasks)} tasks × {len(slice_.models)} models "
            f"({', '.join(slice_.models)}), {n_paired} scored by both arms, "
            f"{report.n_seeds} seeds",
            f"exploration bills {report.cost_multiple:.2f}× the exploit-only run "
            f"(worst seed {report.cost_multiple_worst_seed:.2f}×)",
            "95% percentile-bootstrap CIs over tasks",
        ),
        notes=(
            f"Direct-Method replay on the fully-dense slice: {slice_.n_cells} measured cells "
            f"(full matrix {slice_.matrix_density:.1%} dense)",
            f"Cells skipped as unscorable: {report.baseline_missing} baseline, "
            f"{report.exploration_missing_per_seed:.1f}/seed exploration",
            f"Realized explore/exploit SPEND {report.explore_ratio:.3f} (worst seed "
            f"{report.explore_ratio_worst_seed:.3f}); the router's own counter reached "
            f"{report.budget_explore_ratio:.3f} of its {budget_frac:g} cap",
        ),
        limitations=tuple(limits),
        counts=(
            ("slice tasks", len(slice_.tasks)),
            ("slice models", len(slice_.models)),
            ("paired tasks", n_paired),
            ("seeds", report.n_seeds),
        ),
    )


def plot(report: ReplayReport, out_path: Path, budget_frac: float, digest: str) -> None:
    """Draw exploration_cost.png through the shared frame."""
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 2, width_ratios=(1.0, 1.05))
    _panel_cost_quality(axes[0], report)
    _panel_budget(axes[1], report, budget_frac)
    plot_frame.save(
        fig,
        out_path,
        SPEC,
        extra=_annotations(report, budget_frac),
        provenance=plot_frame.Provenance(_GENERATOR, digest, ctxmod.MANIFEST),
        size=size,
    )


def _measured_only(matrix: dict) -> dict:
    """Keep only REAL (non-imputed) cells — the replay looks up recorded outcomes, never guessed."""
    results = {
        tid: {m: c for m, c in cells.items() if not c.get("imputed")}
        for tid, cells in matrix.get("results", {}).items()
    }
    results = {tid: cells for tid, cells in results.items() if cells}
    return {**matrix, "results": results}


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
    config.load(config_path)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=config_path, help="Path to config YAML")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, default=20, help="replay passes (TS is stochastic)")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.config != config_path:
        config.load(args.config)

    # The replay is a Direct-Method estimator: it looks up RECORDED outcomes on a dense
    # sub-grid ("never guessed"), so it must run on MEASURED cells only. Default to the
    # VALID set (complete challenges, censored + incomplete excluded), then drop the
    # imputed cells the completion added — leaving real measured outcomes on valid
    # challenges for dense_slice. (Unlike the other analytical plots, imputed cells are
    # excluded here because presenting them as recorded outcomes would break the replay.)
    matrix = _measured_only(summary.load_scored_matrix())
    if not matrix.get("results"):
        # Same clean early-exit contract as the other analysis scripts: an empty
        # cache is "nothing measured yet", not a crash in the dense-slice search.
        print(
            "No results yet — results.csv holds no rows. "
            "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
        )
        return

    policy = load_router_policy()
    report = evaluate(
        matrix,
        knn=policy.policy,
        exploration=policy.exploration,
        n_seeds=args.seeds,
        seed=args.seed,
    )
    slice_ = report.slice_
    print(
        f"dense slice: {len(slice_.tasks)} tasks x {len(slice_.models)} models "
        f"= {slice_.n_cells} cells (full matrix {slice_.matrix_density:.1%} dense)"
    )
    print(
        f"  exploit-only : pass {report.baseline_pass_rate.value:.1%} "
        f"[{report.baseline_pass_rate.lo:.1%}, {report.baseline_pass_rate.hi:.1%}]  "
        f"cost ${report.baseline_cost.value:.4f}/task"
    )
    print(
        f"  + exploration: pass {report.exploration_pass_rate.value:.1%} "
        f"[{report.exploration_pass_rate.lo:.1%}, {report.exploration_pass_rate.hi:.1%}]  "
        f"cost ${report.exploration_cost.value:.4f}/task"
    )
    print(
        f"  paired diff : pass {report.pass_delta.value:+.1%} "
        f"[{report.pass_delta.lo:+.1%}, {report.pass_delta.hi:+.1%}]  "
        f"cost {report.cost_delta.value:+.5f}/task "
        f"[{report.cost_delta.lo:+.5f}, {report.cost_delta.hi:+.5f}]"
    )
    print(
        f"  overhead    : {report.cost_multiple:.2f}x the exploration-off bill "
        f"(worst seed {report.cost_multiple_worst_seed:.2f}x); "
        f"explore/exploit spend {report.explore_ratio:.3f} "
        f"(worst seed {report.explore_ratio_worst_seed:.3f}, "
        f"router's own counter {report.budget_explore_ratio:.3f}, "
        f"cap {policy.exploration.explore_budget_frac:g})"
    )
    print(
        f"  skipped cells: {report.baseline_missing} baseline, "
        f"{report.exploration_missing_per_seed:.1f}/seed exploration"
    )
    plot(
        report,
        args.out,
        policy.exploration.explore_budget_frac,
        ctxmod.corpus_digest(matrix, slice_.tasks),
    )
    print(f"Plot saved to {args.out}")


if __name__ == "__main__":
    main()
