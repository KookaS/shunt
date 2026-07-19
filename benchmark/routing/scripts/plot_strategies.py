#!/usr/bin/env python3
"""Strategy comparison visualization — Pareto scatter plot.

Directly imports strategy classes and evaluates them against the benchmark
matrix, then produces a publication-quality Pareto frontier plot.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark import config  # noqa: E402
from benchmark.routing.metrics import compute_metrics, compute_pareto  # noqa: E402
from benchmark.routing.strategies import Strategy  # noqa: E402
from benchmark.routing.strategies.external_prior import ExternalPriorCascade  # noqa: E402
from benchmark.routing.strategies.fixed import (  # noqa: E402
    AlwaysCheap,
    AlwaysFrontier,
    Random,
)
from benchmark.routing.strategies.knn import kNNStrategy  # noqa: E402
from benchmark.routing.strategies.knn_blended import kNNBlended  # noqa: E402
from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy  # noqa: E402
from benchmark.routing.strategies.oracle import Oracle, OracleRewardAware  # noqa: E402


def load_matrix(path: Path) -> dict:
    return config.load_matrix(path)


def evaluate(strategy, matrix: dict, tasks: list[str]) -> list[tuple[str, str, bool, float]]:
    decisions: list[tuple[str, str, bool, float]] = []
    for tid in tasks:
        task_meta = matrix["tasks"].get(tid, {})
        model = strategy.select(tid, task_meta, matrix)
        outcome = matrix["results"].get(tid, {}).get(model, {})
        passed = outcome.get("pass", False)
        if hasattr(strategy, "cascade_total_cost") and strategy.cascade_total_cost is not None:
            cost = strategy.cascade_total_cost
        else:
            cost = outcome.get("cost", 0.0)
        decisions.append((tid, model, passed, cost))
    return decisions


def get_strategies() -> list[tuple[str, Strategy]]:
    """Read strategy list and params from config, return (name, instance) pairs."""
    g = config.gamma()
    strat_cfg = config.strategies()
    enabled = strat_cfg.get("enabled", [])
    if not enabled:
        enabled = [
            "oracle",
            "oracle_reward",
            "always_cheap",
            "always_frontier",
            "random",
            "knn",
            "knn_cascade",
        ]

    knn_p = dict(strat_cfg.get("knn", {}))
    cascade_p = dict(strat_cfg.get("knn", {}))
    cascade_p.update(strat_cfg.get("knn_cascade", {}))
    ext_p = dict(strat_cfg.get("external_prior", {}))
    blend_p = dict(strat_cfg.get("knn_blended", {}))

    registry: dict[str, Callable[[], Strategy]] = {
        "oracle": lambda: Oracle(),
        "oracle_reward": lambda: OracleRewardAware(gamma=g),
        "always_cheap": lambda: AlwaysCheap(),
        "always_frontier": lambda: AlwaysFrontier(),
        "random": lambda: Random(seed=42),
        "knn": lambda: kNNStrategy(**knn_p),
        "knn_cascade": lambda: kNNCascadeStrategy(**cascade_p),
        "external_prior": lambda: ExternalPriorCascade(**ext_p),
        "knn_blended": lambda: kNNBlended(**blend_p),
    }

    return [(name, registry[name]()) for name in enabled if name in registry]


def plot_pareto(
    results: list[dict],
    out_path: Path,
    display_name_key: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    markers: dict[str, str] | None = None,
    warning: str | None = None,
) -> None:
    if display_name_key is None:
        display_name_key = {
            "Oracle": "oracle",
            "Oracle-reward": "oracle_reward",
            "Always-Cheap": "always_cheap",
            "Always-Frontier": "always_frontier",
            "Random": "random",
            "kNN": "knn",
            "kNN-cascade": "knn_cascade",
            "External-Prior": "external_prior",
            "kNN-blended": "knn_blended",
        }
    if colors is None:
        colors = {
            "oracle": "#0072B2",
            "oracle_reward": "#56B4E9",
            "always_cheap": "#009E73",
            "always_frontier": "#D55E00",
            "random": "#CC79A7",
            "knn": "#E69F00",
            "knn_cascade": "#F0E442",
            "external_prior": "#882255",
            "knn_blended": "#44AA99",
        }
    if markers is None:
        markers = {
            "oracle": "D",
            "oracle_reward": "s",
            "always_cheap": "^",
            "always_frontier": "v",
            "random": "o",
            "knn": "P",
            "knn_cascade": "X",
            "external_prior": "*",
            "knn_blended": "h",
        }
    names = [r["strategy"] for r in results]
    costs = np.array([float(r["TotalCost"]) for r in results], dtype=float)
    perfs = np.array([float(r["AvgPerf%"]) for r in results], dtype=float)

    pareto_map = compute_pareto({r["strategy"]: r for r in results})

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, name in enumerate(names):
        key = display_name_key.get(name, "random")
        color = colors.get(key, "#999999")
        marker = markers.get(key, "o")
        is_pareto = pareto_map.get(name, False)

        ax.scatter(
            costs[i],
            perfs[i],
            c=color,
            s=180 if is_pareto else 100,
            marker=marker,
            zorder=5,
            edgecolors="white",
            linewidth=0.8,
            label=name,
        )

        label = f"{name}\n({perfs[i]:.0f}%, ${costs[i]:.2f})"
        ax.annotate(
            label,
            (costs[i], perfs[i]),
            fontsize=8,
            xytext=(8, 8),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
        )

    # Pareto frontier
    pareto_idx = [i for i, name in enumerate(names) if pareto_map.get(name, False)]
    if pareto_idx:
        pts = sorted([(costs[i], perfs[i]) for i in pareto_idx], key=lambda p: p[0])
        fx: list[float] = [pts[0][0]]
        fy: list[float] = [pts[0][1]]
        for i in range(1, len(pts)):
            if pts[i][1] > fy[-1]:
                fx.append(pts[i][0])
                fy.append(pts[i][1])
        if fx[0] > 0:
            fx = [0.0] + fx
            fy = [fy[0]] + fy
        ax.step(
            fx,
            fy,
            where="post",
            color="#333333",
            linewidth=2,
            linestyle="--",
            label="Pareto frontier",
        )
        ax.fill_between(fx, 0, fy, step="post", alpha=0.05, color="#333333")

    # Reference line at frontier pass rate (highest pass rate among all)
    best_perf = max(perfs)
    ax.axhline(
        y=best_perf,
        color="#666666",
        linestyle=":",
        linewidth=1,
        alpha=0.6,
        label=f"Best pass rate ({best_perf:.0f}%)",
    )

    # Sweet spot annotation
    ax.annotate(
        "SWEET SPOT\nhigh pass, low cost",
        xy=(0.15, 0.85),
        xycoords="axes fraction",
        fontsize=10,
        fontweight="bold",
        color="#009E73",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#009E73", alpha=0.8),
    )

    ax.set_xscale("log")
    ax.set_xlabel("Total Cost ($, log scale)", fontsize=11)
    ax.set_ylabel("Average Performance (%)", fontsize=11)
    ax.set_title("Strategy Comparison — Pass Rate vs Cost", fontsize=13, fontweight="bold")
    ax.set_xlim(left=costs.min() * 0.7)
    ax.legend(
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        ncol=1 if len(names) <= 4 else 2,
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.set_axisbelow(True)

    if warning:
        # Sparse-frontier honesty: a fixed-frontier strategy auto-fails every task
        # its model never ran on, dragging its pass rate to a phantom value.
        ax.annotate(
            warning,
            xy=(0.5, 1.09),  # clear of the bold title (~1.06) and the lower-right legend
            xycoords="axes fraction",
            fontsize=9,
            fontweight="bold",
            color="#B00020",
            ha="center",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="#FDECEA", edgecolor="#B00020", alpha=0.9
            ),
        )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(config_path: str = "benchmark/config.yaml") -> None:
    config.load(config_path)

    import argparse

    ap = argparse.ArgumentParser(description="Strategy comparison Pareto plot")
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--matrix", default=None, help="Path to matrix JSON (default: challenges.json)")
    ap.add_argument(
        "--output",
        default="benchmark/routing/reports/strategy_comparison.png",
        help="Output plot path",
    )
    args = ap.parse_args()

    if args.config != config_path:
        config.load(args.config)

    matrix_path = Path(args.matrix) if args.matrix else config.challenges_path()
    out_path = Path(args.output)

    matrix = load_matrix(matrix_path)
    tasks = sorted(matrix["results"].keys())

    if not tasks:
        print(
            "No results yet — results.csv holds no rows. "
            "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
        )
        return

    strategies = get_strategies()
    if not strategies:
        print("No strategies enabled — nothing to plot.")
        return

    print(f"Evaluating {len(strategies)} strategies on {len(tasks)} tasks")
    gamma = config.gamma()

    results: list[dict] = []
    for _name, strategy in strategies:
        decisions = evaluate(strategy, matrix, tasks)
        metrics = compute_metrics(decisions, gamma=gamma)
        results.append(
            {
                "strategy": strategy.name,
                "TotalCost": metrics["TotalCost"],
                "AvgPerf%": metrics["AvgPerf%"],
            }
        )
        print(
            f"  {strategy.name:25}  pass={metrics['AvgPerf%']:>5.2f}%  "
            f"cost=${metrics['TotalCost']:<8.4f}"
        )

    # Phantom-frontier guard (matches report.py::_frontier_coverage): the fixed
    # Always-Frontier strategy auto-fails every task its model never ran on, so a
    # sparsely-covered frontier model produces a misleadingly low pass rate.
    pricing = config.enabled_pricing()
    warning = None
    if pricing:
        frontier_model = max(pricing, key=lambda m: config.cost_per_1m(m, pricing))
        covered = sum(1 for tid in tasks if frontier_model in matrix["results"].get(tid, {}))
        if covered < len(tasks):
            warning = (
                f"⚠ PHANTOM BASELINE — Always-Frontier ({frontier_model}) ran on "
                f"{covered}/{len(tasks)} tasks; the rest auto-fail, so its pass rate is not real"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_pareto(results, out_path, warning=warning)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
