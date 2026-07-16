#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark import config
from benchmark.routing import integrity, summary
from benchmark.routing.metrics import _reward
from benchmark.routing.strategies import Strategy
from benchmark.routing.strategies.fixed import AlwaysCheap, AlwaysFrontier, Random
from benchmark.routing.strategies.oracle import Oracle, OracleRewardAware


def _build_strategy_factories(gamma: float) -> dict[str, Callable[[], Strategy]]:
    factories: dict[str, Callable[[], Strategy]] = {
        "Oracle": lambda: Oracle(),
        "Oracle-reward": lambda: OracleRewardAware(gamma=gamma),
        "Always-Frontier": lambda: AlwaysFrontier(),
        "Always-Cheap": lambda: AlwaysCheap(),
        "Random": lambda: Random(seed=42),
    }
    _add_knn_factories(factories)
    return factories


def _add_knn_factories(factories: dict[str, Callable[[], Strategy]]) -> None:
    """Register the headline kNN / kNN-cascade factories. Import lazily and warn
    if the heavy optional embedding deps (fastembed/hnswlib) are absent, rather
    than omit them silently from the per-task regret plot.
    """
    try:
        from benchmark.routing.strategies.knn import kNNStrategy
        from benchmark.routing.strategies.knn_cascade import kNNCascadeStrategy
    except ImportError as exc:
        print(
            "Warning: kNN/kNN-cascade omitted from per-task regret plot "
            f"(embedding deps unavailable: {exc})",
            file=sys.stderr,
        )
        return

    strat_cfg = config.strategies()
    knn_p = dict(strat_cfg.get("knn", {}))
    cascade_p = dict(strat_cfg.get("knn", {}))
    cascade_p.update(strat_cfg.get("knn_cascade", {}))
    factories["kNN"] = lambda: kNNStrategy(**knn_p)
    factories["kNN-cascade"] = lambda: kNNCascadeStrategy(**cascade_p)


def load_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_matrix(path: Path) -> dict | None:
    try:
        return config.load_matrix(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _is_pareto(row: dict) -> bool:
    """Pareto flag, tolerant of bool (in-memory rows) or str (CSV override)."""
    return row.get("Pareto") in (True, "True")


def print_summary(results: list[dict[str, str]]) -> None:
    """Print the strategy ranking to stdout. Rows are derived in-memory from the
    results.csv source of truth (no committed derived CSV).
    """
    rows = sorted(results, key=lambda r: float(r.get("Reward", 0)), reverse=True)
    cols = ["strategy", "n_pass", "AvgPerf%", "TotalCost", "AvgCost", "Reward", "Pareto"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols))


def plot_pareto(results: list[dict[str, str]], out_dir: Path) -> Path:
    names = [r["strategy"] for r in results]
    costs = np.array([float(r["TotalCost"]) for r in results], dtype=float)
    perfs = np.array([float(r["AvgPerf%"]) for r in results], dtype=float)
    pareto_map = {r["strategy"]: _is_pareto(r) for r in results}

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, name in enumerate(names):
        is_frontier = name == "Always-Frontier"
        is_pareto = pareto_map.get(name, False)
        if is_frontier:
            color, size, marker = "#F44336", 140, "D"
        elif is_pareto:
            color, size, marker = "#4CAF50", 100, "o"
        else:
            color, size, marker = "#9E9E9E", 70, "o"
        ax.scatter(
            costs[i],
            perfs[i],
            c=color,
            s=size,
            marker=marker,
            zorder=5,
            edgecolors="white",
            linewidth=0.5,
        )
        ax.annotate(
            name,
            (costs[i], perfs[i]),
            fontsize=9,
            xytext=(6, 6),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

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
            color="#4CAF50",
            linewidth=2,
            linestyle="--",
            label="Pareto frontier",
        )
        ax.fill_between(fx, 0, fy, step="post", alpha=0.05, color="#4CAF50")

    ax.set_xlabel("Total Cost ($)")
    ax.set_ylabel("Average Performance (%)")
    ax.set_title("Cost vs Quality — Strategy Pareto Frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = out_dir / "pareto_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _evaluate_strategies(
    factory_map: dict[str, Callable[[], Strategy]],
    matrix: dict,
    tasks: list[str],
) -> dict[str, list[tuple[str, str, bool, float]]]:
    evaluated: dict[str, list[tuple[str, str, bool, float]]] = {}
    for name, factory in factory_map.items():
        strategy = factory()
        decisions: list[tuple[str, str, bool, float]] = []
        for tid in tasks:
            task_meta = matrix.get("tasks", {}).get(tid, {})
            model = strategy.select(tid, task_meta, matrix)
            outcome = matrix.get("results", {}).get(tid, {}).get(model, {})
            passed = outcome.get("pass", False)
            cost = outcome.get("cost", 0.0)
            decisions.append((tid, model, passed, cost))
        evaluated[name] = decisions
    return evaluated


def _compute_per_task_regret(
    strategy_decisions: list[tuple[str, str, bool, float]],
    oracle_decisions: list[tuple[str, str, bool, float]],
    gamma: float = 0.1,
) -> np.ndarray:
    regrets: list[float] = []
    for sd, od in zip(strategy_decisions, oracle_decisions, strict=True):
        sr = _reward(sd[2], sd[3], gamma)
        or_ = _reward(od[2], od[3], gamma)
        regrets.append(or_ - sr)
    return np.cumsum(regrets)


def plot_cumulative_regret(
    results: list[dict[str, str]],
    out_dir: Path,
    matrix_path: Path | None = None,
    gamma: float = 0.1,
    strategy_factories: dict[str, Callable[[], Strategy]] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))

    if matrix_path is not None and strategy_factories is not None:
        matrix = load_matrix(matrix_path)
        if matrix is not None:
            tasks = sorted(matrix.get("results", {}).keys())
            if tasks:
                all_decisions = _evaluate_strategies(strategy_factories, matrix, tasks)
                oracle_decisions = all_decisions.get("Oracle-reward")
                if oracle_decisions:
                    for name in [r["strategy"] for r in results]:
                        decisions = all_decisions.get(name)
                        if decisions and name != "Oracle-reward":
                            cumreg = _compute_per_task_regret(decisions, oracle_decisions, gamma)
                            label = name + f" (total={cumreg[-1]:.2f})"
                            ax.plot(range(1, len(cumreg) + 1), cumreg, label=label, lw=1.5)

                    ax.set_xlabel("Task")
                    ax.set_ylabel("Cumulative Regret")
                    ax.set_title("Cumulative Regret vs Oracle (Per-Task)")
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)

                    path = out_dir / "cumulative_regret.png"
                    fig.savefig(path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    return path

    # Fallback: bar chart from aggregate CumReg
    names = [r["strategy"] for r in results]
    cumregs = [float(r["CumReg"]) for r in results]

    color_map = {
        "Oracle": "#4CAF50",
        "Oracle-reward": "#4CAF50",
        "Always-Frontier": "#F44336",
        "Random": "#FF9800",
        "Always-Cheap": "#2196F3",
    }
    colors = [color_map.get(n, "#9E9E9E") for n in names]

    bars = ax.bar(names, cumregs, color=colors, edgecolor="white")
    for bar, val in zip(bars, cumregs, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Cumulative Regret vs Oracle (Aggregate)")
    ax.grid(True, axis="y", alpha=0.3)

    path = out_dir / "cumulative_regret.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cost_savings(results: list[dict[str, str]], out_dir: Path) -> Path:
    names = [r["strategy"] for r in results]
    costs = [float(r["TotalCost"]) for r in results]
    perfs = [float(r["AvgPerf%"]) for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = {
        "Oracle": "#4CAF50",
        "Oracle-reward": "#4CAF50",
        "Always-Frontier": "#F44336",
        "Always-Cheap": "#2196F3",
        "Random": "#FF9800",
    }
    colors = [color_map.get(n, "#9E9E9E") for n in names]

    bars = ax.bar(names, costs, color=colors, edgecolor="white")
    for bar, cost, perf in zip(bars, costs, perfs, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(costs) * 0.01,
            f"${cost:.4f}  ({perf:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Total Cost ($)")
    ax.set_title("Cost Comparison by Strategy (Cost annotated with pass rate)")
    ax.grid(True, axis="y", alpha=0.3)

    if len(names) >= 2:
        ref_name = "Always-Frontier"
        ref_cost = None
        for r in results:
            if r["strategy"] == ref_name:
                ref_cost = float(r["TotalCost"])
                break
        if ref_cost is not None:
            ax.axhline(
                y=ref_cost,
                color="#F44336",
                linestyle=":",
                linewidth=1,
                alpha=0.7,
                label=f"{ref_name} cost = ${ref_cost:.4f}",
            )
            ax.legend()

    path = out_dir / "cost_savings.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_heatmap(matrix_path: Path, out_dir: Path) -> Path:
    matrix = load_matrix(matrix_path)
    if matrix is None:
        raise FileNotFoundError(f"Cannot load matrix from {matrix_path}")

    models = list(matrix["models"].keys())
    results_data = matrix.get("results", {})

    tasks_by_language: dict[str, list[str]] = {}
    for tid, meta in matrix.get("tasks", {}).items():
        lang = meta.get("language", "unknown")
        tasks_by_language.setdefault(lang, []).append(tid)

    languages = sorted(tasks_by_language.keys())
    n_models = len(models)
    n_langs = len(languages)

    if n_models == 0 or n_langs == 0:
        raise ValueError("No models or languages found in matrix")

    heat = np.zeros((n_models, n_langs), dtype=float)
    counts = np.zeros((n_models, n_langs), dtype=int)

    for j, lang in enumerate(languages):
        for tid in tasks_by_language[lang]:
            task_results = results_data.get(tid, {})
            for i, model in enumerate(models):
                outcome = task_results.get(model, {})
                if "pass" in outcome:
                    heat[i, j] += int(outcome["pass"])
                    counts[i, j] += 1

    mask = counts > 0
    heat[mask] = heat[mask] / counts[mask]
    heat[~mask] = np.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.colormaps["YlGn"]
    cmap.set_bad("lightgray")
    im = ax.imshow(heat, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    lang_labels = [f"{lang}\n(n={len(tasks_by_language[lang])})" for lang in languages]
    ax.set_xticks(range(n_langs))
    ax.set_xticklabels(lang_labels, fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(models, fontsize=9)

    for i in range(n_models):
        for j in range(n_langs):
            if not np.isnan(heat[i, j]):
                cell = f"{heat[i, j]:.0%}" if heat[i, j] > 0 else "0%"
                color = "white" if heat[i, j] > 0.5 else "black"
                ax.text(j, i, cell, ha="center", va="center", fontsize=8, color=color)

    ax.set_title("Model Complementarity — Pass Rate by Language")
    fig.colorbar(im, ax=ax, label="Pass Rate", fraction=0.046, pad=0.04)

    path = out_dir / "model_complementarity_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def derive_tasks(seed: int) -> list[str]:
    """All challenge specs sampled by ``sample_size``, matching run_matrix and check_integrity."""
    return config.sample_tasks(sorted(integrity.all_hashes().keys()), seed=seed)


def derive_rows(matrix: dict, tasks: list[str]) -> list[dict]:
    """Compute per-strategy summary rows in-memory from the results.csv cache —
    the single source of truth. Mirrors run_matrix.refresh_summary exactly (same
    strategies, gamma, bootstrap, seed, and task set — see derive_tasks).
    """
    from benchmark.routing import run_eval

    bm = config.benchmark_params()
    strategies = run_eval.get_strategies()
    return summary.compute_strategy_rows(
        matrix,
        tasks,
        strategies,
        gamma=config.gamma(),
        bootstrap=bm.get("bootstrap_iterations", 1000),
        seed=bm.get("seed", 42),
    )


def main(config_path: str = "benchmark/config.yaml") -> None:
    config.load(config_path)

    ap = argparse.ArgumentParser(
        description="Generate comparison reports from routing benchmark results"
    )
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument(
        "--results",
        default=None,
        help="Optional: read a pre-derived strategy-summary CSV instead of deriving "
        "in-memory from results.csv (default: derive from the source of truth)",
    )
    ap.add_argument(
        "--out-dir",
        default="benchmark/routing/reports",
        help="Output directory for reports (default: benchmark/routing/reports/)",
    )
    ap.add_argument(
        "--matrix",
        default=None,
        help="Path to task matrix JSON (enables per-task regret lines + heatmap)",
    )
    args = ap.parse_args()

    if args.config != config_path:
        config.load(args.config)

    matrix_path = Path(args.matrix) if args.matrix else config.challenges_path()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.results:
        results = load_results(Path(args.results))
        source = args.results
    else:
        matrix = load_matrix(matrix_path)
        if matrix is None or not matrix.get("results"):
            print(
                "No results yet — results.csv holds no rows. "
                "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
            )
            return
        tasks = derive_tasks(config.benchmark_params().get("seed", 42))
        results = derive_rows(matrix, tasks)
        # Write a human-readable copy to reports/ (gitignored) — never committed.
        summary.write_summary_csv(results, out_dir / "strategy_summary.csv")
        source = "results.csv (derived in-memory)"

    if not results:
        print("No strategy rows to plot.", file=sys.stderr)
        return

    print(f"Loaded {len(results)} strategies from {source}")

    print()
    print("Strategy ranking (derived from the results.csv source of truth):")
    print_summary(results)
    print()

    p1 = plot_pareto(results, out_dir)
    print(f"  Pareto       : {p1}")

    g = config.gamma()
    factories = _build_strategy_factories(g)
    p2 = plot_cumulative_regret(
        results, out_dir, matrix_path, gamma=g, strategy_factories=factories
    )
    print(f"  Regret       : {p2}")

    p3 = plot_cost_savings(results, out_dir)
    print(f"  Cost savings : {p3}")

    if matrix_path is not None:
        if matrix_path.exists():
            try:
                p4 = plot_heatmap(matrix_path, out_dir)
                print(f"  Heatmap      : {p4}")
            except (FileNotFoundError, ValueError, KeyError) as exc:
                print(f"  Heatmap      : skipped ({exc})")
        else:
            print(f"  Heatmap      : matrix file {matrix_path} not found, skipping")

    plt.close("all")


if __name__ == "__main__":
    main()
