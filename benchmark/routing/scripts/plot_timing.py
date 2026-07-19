#!/usr/bin/env python3
"""Timing visualization — API calls as a compute-time proxy. Two-panel figure:
average calls per model (left) and per strategy (right).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from benchmark import config
from benchmark.routing.strategies import Strategy
from benchmark.routing.strategies.external_prior import ExternalPriorCascade
from benchmark.routing.strategies.fixed import AlwaysCheap, AlwaysFrontier, Random
from benchmark.routing.strategies.knn import kNNStrategy
from benchmark.routing.strategies.knn_blended import kNNBlended
from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy
from benchmark.routing.strategies.oracle import Oracle, OracleRewardAware

matplotlib.use("Agg")

CB_PALETTE: Final = {
    # Current enabled pool (6 models). Unlisted models fall back to gray via
    # CB_PALETTE.get(m, "#999999"); keep this in step with default_config.yaml.
    "deepseek-v4-flash": "#0072B2",
    "qwen3.7-plus": "#56B4E9",
    "gpt-5-mini": "#009E73",
    "kimi-k2.5": "#D55E00",
    "zai-glm-5.2": "#CC79A7",
    "kimi-k3": "#E69F00",
    "oracle": "#0072B2",
    "oracle_reward": "#56B4E9",
    "always_cheap": "#009E73",
    "always_frontier": "#D55E00",
    "random": "#CC79A7",
    "knn": "#E69F00",
    "knn_cascade": "#F0E442",
}


def load_matrix(path: Path) -> dict:
    return config.load_matrix(path)


def evaluate(
    strategy,
    matrix: dict,
    tasks: list[str],
) -> list[tuple[str, str, bool, float, int, int, int]]:
    decisions: list[tuple[str, str, bool, float, int, int, int]] = []
    for tid in tasks:
        task_meta = matrix["tasks"].get(tid, {})
        model = strategy.select(tid, task_meta, matrix)
        outcome = matrix["results"].get(tid, {}).get(model, {})
        passed = outcome.get("pass", False)
        if hasattr(strategy, "cascade_total_cost") and strategy.cascade_total_cost is not None:
            cost = strategy.cascade_total_cost
        else:
            cost = outcome.get("cost", 0.0)
        in_tok = outcome.get("in_tok", 0)
        out_tok = outcome.get("out_tok", 0)
        calls = outcome.get("calls", 0)
        decisions.append((tid, model, passed, cost, in_tok, out_tok, calls))
    return decisions


def get_strategies() -> list[tuple[str, Strategy]]:
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


def compute_avg_calls_per_model(
    strategies_decisions: dict[str, list[tuple]],
) -> dict[str, float]:
    model_calls: dict[str, float] = {}
    model_counts: dict[str, int] = {}
    for _sname, decisions in strategies_decisions.items():
        for d in decisions:
            model = d[1]
            calls = d[6]
            model_calls[model] = model_calls.get(model, 0.0) + calls
            model_counts[model] = model_counts.get(model, 0) + 1
    return {m: round(model_calls[m] / model_counts[m], 1) for m in sorted(model_calls)}


def compute_avg_calls_per_strategy(
    strategies_decisions: dict[str, list[tuple]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for sname, decisions in strategies_decisions.items():
        if not decisions:
            continue
        total = sum(d[6] for d in decisions)
        result[sname] = round(total / len(decisions), 1)
    return result


def plot_timing(
    avg_per_model: dict[str, float],
    avg_per_strategy: dict[str, float],
    out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: Avg calls per model ---
    models = list(avg_per_model.keys())
    values = [avg_per_model[m] for m in models]
    colors_m = [CB_PALETTE.get(m, "#999999") for m in models]
    x1 = np.arange(len(models))
    bars1 = ax1.bar(x1, values, color=colors_m, edgecolor="white", width=0.6)
    for bar, val in zip(bars1, values, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_xticks(x1)
    ax1.set_xticklabels(models, fontsize=9, rotation=20, ha="right")
    ax1.set_ylabel("Avg API Calls per Task", fontsize=11)
    ax1.set_title("Avg Calls per Model", fontsize=13, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_axisbelow(True)

    # --- Right: Avg calls per strategy ---
    strategies = list(avg_per_strategy.keys())
    strat_vals = [avg_per_strategy[s] for s in strategies]
    colors_s = [CB_PALETTE.get(s, "#999999") for s in strategies]
    x2 = np.arange(len(strategies))
    bars2 = ax2.bar(x2, strat_vals, color=colors_s, edgecolor="white", width=0.6)
    for bar, val in zip(bars2, strat_vals, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(strat_vals) * 0.02,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(strategies, fontsize=9, rotation=20, ha="right")
    ax2.set_ylabel("Avg API Calls per Task", fontsize=11)
    ax2.set_title("Avg Calls per Strategy", fontsize=13, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    fig.suptitle(
        "Timing Proxy: API Calls per Task",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(config_path: str = "benchmark/config.yaml") -> None:
    config.load(config_path)

    import argparse

    ap = argparse.ArgumentParser(description="Timing visualization (API calls as time proxy)")
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument(
        "--matrix",
        default=None,
        help="Path to matrix JSON (default: challenges.json)",
    )
    ap.add_argument(
        "--output",
        default="benchmark/routing/reports/timing_comparison.png",
        help="Output plot path",
    )
    args = ap.parse_args()

    if args.config != config_path:
        config.load(args.config)

    matrix_path = Path(args.matrix) if args.matrix else config.challenges_path()
    out_path = Path(args.output)

    matrix = load_matrix(matrix_path)
    tasks = sorted(matrix["results"].keys())

    strategies = get_strategies()
    if not strategies:
        print("No strategies enabled — nothing to plot.")
        return

    print(f"Evaluating {len(strategies)} strategies on {len(tasks)} tasks")

    strategies_decisions: dict[str, list[tuple]] = {}
    for _name, strategy in strategies:
        decisions = evaluate(strategy, matrix, tasks)
        strategies_decisions[strategy.name] = decisions
        total_calls = sum(d[6] for d in decisions)
        avg_calls = total_calls / len(decisions) if decisions else 0
        print(f"  {strategy.name:25}  total_calls={total_calls:>4d}  avg_calls={avg_calls:>5.1f}")

    avg_per_model = compute_avg_calls_per_model(strategies_decisions)
    avg_per_strategy = compute_avg_calls_per_strategy(strategies_decisions)

    print(f"\nAvg calls per model:  {avg_per_model}")
    print(f"Avg calls per strategy: {avg_per_strategy}")

    plot_timing(avg_per_model, avg_per_strategy, out_path)
    print(f"\nTiming plot saved to {out_path}")


if __name__ == "__main__":
    main()
