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

from benchmark import config  # noqa: E402
from benchmark.routing.exploration_replay import ReplayReport, evaluate  # noqa: E402
from shunt.router.policy import load_router_policy  # noqa: E402

# Okabe-Ito, matching plot_strategies.CB_PALETTE's convention.
BASELINE_COLOR: Final = "#0072B2"
EXPLORE_COLOR: Final = "#D55E00"
CAP_COLOR: Final = "#999999"

DEFAULT_OUT: Final = Path(__file__).resolve().parents[1] / "reports" / "exploration_replay.png"


def _panel_cost_quality(ax, report: ReplayReport) -> None:
    """Grouped bars: mean cost/task and pass rate, each with its bootstrap CI."""
    labels = ["exploit-only\n(exploration off)", "exploit + exploration\n(shipped default)"]
    colors = [BASELINE_COLOR, EXPLORE_COLOR]
    costs = [report.baseline_cost, report.exploration_cost]
    passes = [report.baseline_pass_rate, report.exploration_pass_rate]

    x = np.arange(len(labels))
    ax.bar(
        x,
        [e.value for e in costs],
        color=colors,
        width=0.55,
        edgecolor="white",
        yerr=[
            [e.value - e.lo for e in costs],
            [e.hi - e.value for e in costs],
        ],
        capsize=6,
        error_kw={"ecolor": "#333333", "lw": 1.2},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mean cost per task (USD)")
    ax.set_title("Short-run cost and quality of exploration", fontsize=11)
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
        bbox={"boxstyle": "round,pad=0.4", "fc": "#f2f2f2", "ec": "#cccccc"},
    )
    ax.margins(x=0.15)


def _panel_explore_share(ax, report: ReplayReport, budget_frac: float) -> None:
    """Running exploratory share of decisions, averaged over replay seeds."""
    shares = report.explore_share_by_round
    if shares:
        rounds = np.arange(1, len(shares) + 1)
        ax.plot(rounds, shares, color=EXPLORE_COLOR, lw=2, label="exploratory share of decisions")
        ax.set_xlim(1, len(shares))
    ax.axhline(
        budget_frac,
        color=CAP_COLOR,
        ls="--",
        lw=1.4,
        label=f"explore_budget_frac={budget_frac:g} (cap on the router's OWN counter)",
    )
    ax.axhline(
        report.explore_ratio,
        color=BASELINE_COLOR,
        ls=":",
        lw=1.4,
        label=(
            f"realized explore/exploit SPEND: mean {report.explore_ratio:.2f}, "
            f"worst seed {report.explore_ratio_worst_seed:.2f}"
        ),
    )
    ax.set_xlabel("decisions routed (replay round)")
    ax.set_ylabel("fraction exploratory")
    ax.set_ylim(0, max(0.6, budget_frac * 1.5, report.explore_ratio_worst_seed * 1.15))
    ax.set_title(
        f"Exploration overhead: {report.cost_multiple:.2f}x the exploration-off bill "
        f"(worst seed {report.cost_multiple_worst_seed:.2f}x)",
        fontsize=11,
    )
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)


def plot(report: ReplayReport, out_path: Path, budget_frac: float) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    _panel_cost_quality(ax_left, report)
    _panel_explore_share(ax_right, report, budget_frac)
    slice_ = report.slice_
    fig.suptitle(
        "Offline matrix replay of the production exploration policy "
        "(measured outcomes, no live calls)",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.012,
        f"Direct-Method replay on the fully-dense slice: {len(slice_.tasks)} tasks x "
        f"{len(slice_.models)} models = {slice_.n_cells} measured cells "
        f"(full matrix {slice_.matrix_density:.1%} dense); {report.n_seeds} seeds;\n"
        f"95% percentile-bootstrap CIs over tasks; unscorable cells skipped: "
        f"{report.baseline_missing} baseline, "
        f"{report.exploration_missing_per_seed:.1f}/seed exploration.\n"
        "The outcome matrix is static, so an exploratory pull can never improve a later decision:\n"
        "this is exploration's COST with its learning benefit set to zero, "
        "not a verdict on whether exploration pays.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.155, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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

    matrix = config.load_matrix()
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
