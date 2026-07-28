#!/usr/bin/env python3
"""Exploration-policy replay visualization — cost/quality vs exploration share."""

# Replays the SHIPPED exploration policy over the committed results matrix (no live
# calls) and plots two panels: exploit-only vs exploit+exploration on cost and pass rate
# with bootstrap CIs plus the paired difference (left), and the running exploratory share
# of decisions against the configured budget cap (right).

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark import config, plot_frame  # noqa: E402
from benchmark.plot_frame import Annotations, FigureSpec  # noqa: E402
from benchmark.routing import plot_style, summary  # noqa: E402
from benchmark.routing.exploration_replay import ReplayReport, evaluate  # noqa: E402
from shunt.router.policy import load_router_policy  # noqa: E402

SPEC = FigureSpec(
    reading=(
        "Left: mean cost per task with exploration OFF (blue) and ON (orange), whiskers = "
        "95% bootstrap CI; each bar is labelled with its pass rate and that rate's CI, and "
        "the boxed line states the PAIRED difference between the two arms. Right: the share "
        "of decisions that were exploratory as the replay proceeds (orange), against the "
        "configured budget cap (grey dashed) and the realized explore/exploit spend ratio "
        "(blue dotted). Expect exploration to cost more and to move pass rate very little."
    ),
    goal=(
        "Read the boxed paired difference, not the two overlapping marginal CIs: look for a "
        "small cost delta and a pass delta whose CI clears zero."
    ),
    definitions=(
        ("exploration", "occasionally routing to a non-preferred model to learn its outcome"),
        ("exploit-only", "the same shipped policy with exploration switched off"),
        ("paired difference", "per-task gap between the arms, so shared task noise cancels"),
        ("explore_budget_frac", "cap on the router's own confidence-weighted explore counter"),
    ),
    notes=(
        "The replay is EXACT, not simulated: on a fully dense sub-grid the recorded outcome "
        "is looked up and no request is sent.",
        "The realized spend ratio exceeding the cap is expected, not a bug — the cap counts "
        "the router's confidence-weighted neighbourhood costs, not realized spend.",
        "Unscorable cells are skipped and counted, never guessed.",
    ),
    limitations=(
        "The outcome matrix is static, so an exploratory pull can never improve a later "
        "decision: this measures exploration's COST with its learning benefit set to zero, "
        "the pessimistic half of the ledger — not a verdict on whether exploration pays.",
        "The dense slice is found greedily, not optimally, and comes from a single workload.",
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
_SURFACE: Final = "#fcfcfb"

DEFAULT_OUT: Final = Path(__file__).resolve().parents[1] / "reports" / "exploration_replay.png"


def _panel_cost_quality(ax, report: ReplayReport) -> None:
    """Grouped bars: mean cost/task and pass rate, each with its bootstrap CI."""
    labels = ["exploit-only\n(exploration off)", "exploit + exploration\n(shipped default)"]
    colors = [BASELINE_COLOR, EXPLORE_COLOR]
    costs = [report.baseline_cost, report.exploration_cost]
    passes = [report.baseline_pass_rate, report.exploration_pass_rate]

    x = np.arange(len(labels))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.6)
    ax.bar(
        x,
        [e.value for e in costs],
        color=colors,
        width=0.55,
        edgecolor=_SURFACE,
        linewidth=1.5,
        yerr=[
            [e.value - e.lo for e in costs],
            [e.hi - e.value for e in costs],
        ],
        capsize=6,
        error_kw={"ecolor": _INK2, "lw": 1.2},
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=_INK)
    ax.set_ylabel("mean cost per task (USD)", color=_INK)
    ax.set_title("Short-run cost and quality of exploration", fontsize=11, color=_INK)
    top = max(e.hi for e in costs)
    ax.set_ylim(0, top * 1.85)
    for xi, cost, quality in zip(x, costs, passes, strict=True):
        ax.annotate(
            f"${cost.value:.4f}\npass {quality.value:.1%}\n[{quality.lo:.0%}, {quality.hi:.0%}]",
            xy=(xi, cost.hi),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=_INK,
        )
    # The marginal CIs above overlap heavily; the PAIRED difference is what this slice
    # can actually resolve, so it is stated rather than left to the eye.
    ax.annotate(
        f"paired difference (exploration - exploit-only):\n"
        f"pass {report.pass_delta.value:+.1%} "
        f"[{report.pass_delta.lo:+.1%}, {report.pass_delta.hi:+.1%}]    "
        f"cost {report.cost_delta.value:+.4f} "
        f"[{report.cost_delta.lo:+.4f}, {report.cost_delta.hi:+.4f}]",
        xy=(0.5, 0.965),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=8.5,
        color=_INK,
        bbox={"boxstyle": "round,pad=0.4", "fc": "#f2f2f2", "ec": _GRID},
    )
    ax.margins(x=0.15)


def _panel_explore_share(ax, report: ReplayReport, budget_frac: float) -> None:
    """Running exploratory share of decisions, averaged over replay seeds."""
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=_GRID, linewidth=0.6)
    shares = report.explore_share_by_round
    share_max = max(shares) if shares else 0.0
    if shares:
        rounds = np.arange(1, len(shares) + 1)
        ax.plot(
            rounds,
            shares,
            color=EXPLORE_COLOR,
            lw=2,
            zorder=4,
            label="exploratory share of decisions",
        )
        ax.set_xlim(1, len(shares))
    ax.axhline(
        budget_frac,
        color=CAP_COLOR,
        ls="--",
        lw=1.4,
        zorder=3,
        label=f"explore_budget_frac={budget_frac:g} (cap on the router's OWN counter)",
    )
    ax.axhline(
        report.explore_ratio,
        color=BASELINE_COLOR,
        ls=":",
        lw=1.6,
        zorder=3,
        label=(
            f"realized explore/exploit SPEND: mean {report.explore_ratio:.2f}, "
            f"worst seed {report.explore_ratio_worst_seed:.2f}"
        ),
    )
    ax.set_xlabel("decisions routed (replay round)", color=_INK)
    ax.set_ylabel("fraction (of decisions, and of exploit spend)", color=_INK)
    # Fit the plotted marks (share line, cap, realized-spend mean) with room for the
    # legend; the worst-seed spend ratio is text-only in the legend, so it must NOT
    # stretch the axis and leave the panel half-empty.
    ax.set_ylim(0, max(0.6, budget_frac * 1.5, report.explore_ratio * 1.5, share_max * 1.5))
    ax.set_title(
        f"Exploration overhead: {report.cost_multiple:.2f}x the exploration-off bill "
        f"(worst seed {report.cost_multiple_worst_seed:.2f}x)",
        fontsize=11,
        color=_INK,
    )
    leg = ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    leg.get_frame().set_edgecolor(_GRID)


def _annotations(report: ReplayReport) -> Annotations:
    """Footer content derived from the replay: slice size, seeds, skipped cells."""
    slice_ = report.slice_
    base, expl = report.baseline_pass_rate, report.exploration_pass_rate
    limits: list[str] = []
    if base.lo <= expl.hi and expl.lo <= base.hi:
        limits.append(
            f"The two marginal pass-rate CIs overlap ([{base.lo:.0%}, {base.hi:.0%}] vs "
            f"[{expl.lo:.0%}, {expl.hi:.0%}]) — at {len(slice_.tasks)} tasks only the paired "
            "difference separates the arms"
        )
    # The dense-slice search maximises CELLS, so it prefers many tasks x few models. When the
    # models it lands on are all cheap, the measured "exploration overhead" is the overhead of
    # exploring between cheap models — not the shipped policy's, where an exploratory pull can
    # land on the frontier model at ~40x the price. That has to be said on the figure.
    priced = [(m, config.cost_per_1m(m, config.enabled_pricing())) for m in slice_.models]
    all_enabled = [
        (m, config.cost_per_1m(m, config.enabled_pricing()))
        for m in (config.enabled_models() or slice_.models)
    ]
    top_slice = max((c for _m, c in priced), default=0.0)
    top_enabled = max((c for _m, c in all_enabled), default=0.0)
    if priced and all_enabled and top_slice < top_enabled:
        absent = [m for m, _c in sorted(all_enabled, key=lambda t: -t[1]) if m not in slice_.models]
        limits.append(
            f"NO FRONTIER ARM IN THIS SLICE: the dense sub-grid covers only "
            f"{', '.join(slice_.models)} — the priciest model here is "
            f"{plot_style.usd(top_slice)}/Mtok against {plot_style.usd(top_enabled)} across "
            f"all enabled models ({', '.join(absent)} are "
            f"absent). The exploration overhead measured here is between CHEAP models and is a "
            f"LOWER BOUND on the shipped policy's, where an exploratory pull can land on the "
            f"frontier model"
        )
    return Annotations(
        notes=(
            f"Direct-Method replay on the fully-dense slice: {len(slice_.tasks)} tasks x "
            f"{len(slice_.models)} models = {slice_.n_cells} measured cells (full matrix "
            f"{slice_.matrix_density:.1%} dense); {report.n_seeds} seeds; 95% "
            "percentile-bootstrap CIs over tasks",
            f"Cells skipped as unscorable: {report.baseline_missing} baseline, "
            f"{report.exploration_missing_per_seed:.1f}/seed exploration",
        ),
        limitations=tuple(limits),
    )


def plot(report: ReplayReport, out_path: Path, budget_frac: float) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    fig.patch.set_facecolor(_SURFACE)
    for ax in (ax_left, ax_right):
        ax.set_facecolor(_SURFACE)
        ax.tick_params(colors=_INK2)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)
    _panel_cost_quality(ax_left, report)
    _panel_explore_share(ax_right, report, budget_frac)
    fig.suptitle(
        "Offline matrix replay of the production exploration policy "
        "(measured outcomes, no live calls)",
        fontsize=13,
        color=_INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plot_frame.save(fig, out_path, SPEC, extra=_annotations(report))


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
    plot(report, args.out, policy.exploration.explore_budget_frac)
    print(f"Plot saved to {args.out}")


if __name__ == "__main__":
    main()
