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


def _frontier_coverage(matrix: dict | None) -> tuple[str, int, int] | None:
    """(frontier_model, tasks_with_data, total_tasks) for the most expensive
    evaluated model — the Always-Frontier / kill-gate baseline. ``None`` if no
    matrix. Exposes the sparse-frontier artifact that makes that baseline invalid.
    """
    if not matrix or not matrix.get("models"):
        return None
    results_data = matrix.get("results", {})
    total = len(results_data)
    if total == 0:
        return None
    frontier = max(matrix["models"], key=lambda m: _model_total_price(matrix["models"], m))
    covered = sum(1 for tr in results_data.values() if frontier in tr)
    return frontier, covered, total


def plot_cost_savings(
    results: list[dict[str, str]], out_dir: Path, matrix: dict | None = None
) -> Path:
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

    cov = _frontier_coverage(matrix)
    phantom = cov is not None and cov[1] < cov[2]
    if len(names) >= 2 and not phantom:
        # Only draw the frontier reference line when the frontier baseline is
        # actually measured on every task — otherwise it is not a valid reference.
        ref_cost = next(
            (float(r["TotalCost"]) for r in results if r["strategy"] == "Always-Frontier"), None
        )
        if ref_cost is not None:
            ax.axhline(
                y=ref_cost,
                color="#F44336",
                linestyle=":",
                linewidth=1,
                alpha=0.7,
                label=f"Always-Frontier cost = ${ref_cost:.4f}",
            )
            ax.legend()

    if phantom:
        assert cov is not None
        frontier, covered, total = cov
        if "Always-Frontier" in names:
            i = names.index("Always-Frontier")
            ax.annotate(
                f"⚠ PHANTOM BASELINE\n{frontier} evaluated on {covered}/{total} tasks —\n"
                f"the other {total - covered} count as free $0 failures.\n"
                "Not comparable; the kill-gate is unmeasurable here.",
                xy=(i, costs[i]),
                xytext=(0.5, 0.72),
                textcoords="axes fraction",
                ha="center",
                fontsize=8,
                color="#B71C1C",
                arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.35", fc="#FFEBEE", ec="#B71C1C", alpha=0.95),
            )

    path = out_dir / "cost_savings.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _cap_names(names: list[str], k: int = 6) -> str:
    """Join names, capping the list so titles can't explode at 500 tasks."""
    if not names:
        return "none"
    if len(names) <= k:
        return ", ".join(names)
    return ", ".join(names[:k]) + f" (+{len(names) - k} more)"


def _model_total_price(models: dict, name: str) -> float:
    info = models.get(name, {})
    return float(info.get("input_price", 0)) + float(info.get("output_price", 0))


def plot_heatmap(matrix_path: Path, out_dir: Path) -> Path:
    """Task × model outcome matrix — shows every cell so genuine
    complementarity and sparse coverage are visible.
    """
    matrix = load_matrix(matrix_path)
    if matrix is None:
        raise FileNotFoundError(f"Cannot load matrix from {matrix_path}")

    models_meta = matrix["models"]
    results_data = matrix.get("results", {})
    tasks = sorted(results_data.keys())
    models = sorted(models_meta.keys(), key=lambda m: _model_total_price(models_meta, m))
    if not models or not tasks:
        raise ValueError("No models or tasks found in matrix")

    # 1 = pass, 0 = fail, NaN = not evaluated (distinct from a real failure).
    grid = np.full((len(tasks), len(models)), np.nan, dtype=float)
    for i, tid in enumerate(tasks):
        task_results = results_data.get(tid, {})
        for j, model in enumerate(models):
            outcome = task_results.get(model, {})
            if "pass" in outcome:
                grid[i, j] = 1.0 if outcome["pass"] else 0.0

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    n_tasks, n_models = len(tasks), len(models)
    # Dynamic canvas: grows with the matrix but capped so it stays a usable image
    # at 500+ tasks / 20+ models rather than a fixed 9x7 that crushes everything.
    fig_w = min(24.0, max(8.0, 1.1 * n_models + 3.0))
    fig_h = min(32.0, max(5.0, 0.32 * n_tasks + 2.0))

    # 0=fail red, 1=pass green, NaN(not evaluated)=gray
    cmap = ListedColormap(["#C62828", "#2E7D32"]).with_extremes(bad="#E0E0E0")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    coverage = np.sum(~np.isnan(grid), axis=0)
    passes = np.nansum(grid, axis=0)
    col_labels = [f"{m}\n{int(passes[j])}/{int(coverage[j])} pass" for j, m in enumerate(models)]
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=30, ha="right")

    # Thin task labels when there are too many to read; show ~40 evenly spaced.
    ystep = max(1, int(np.ceil(n_tasks / 40)))
    yticks = list(range(0, n_tasks, ystep))
    ax.set_yticks(yticks)
    ax.set_yticklabels([tasks[i].split("__")[-1] for i in yticks], fontsize=7)

    # Per-cell glyphs only when cells are big enough to read; else colour-only + legend.
    glyphs = n_tasks <= 40 and n_models <= 20
    if glyphs:
        for i in range(n_tasks):
            for j in range(n_models):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#616161")
                else:
                    mark = "✓" if grid[i, j] == 1.0 else "✗"
                    ax.text(j, i, mark, ha="center", va="center", fontsize=10, color="white")
    else:
        ax.legend(
            handles=[
                Patch(color="#2E7D32", label="pass"),
                Patch(color="#C62828", label="fail"),
                Patch(color="#E0E0E0", label="not evaluated"),
            ],
            loc="upper left",
            bbox_to_anchor=(1.005, 1.0),
            fontsize=8,
            frameon=False,
        )

    # Data-driven subtitle so it stays correct on every regen: which tasks split the
    # field (models disagree), which none solve, and the frontier model's coverage.
    # Names are capped so the title can't explode at 500 tasks.
    split, none_solve = [], []
    for i, tid in enumerate(tasks):
        seen = grid[i, ~np.isnan(grid[i])]
        if seen.size and seen.min() != seen.max():
            split.append(tid.split("__")[-1])
        elif seen.size and seen.max() == 0.0:
            none_solve.append(tid.split("__")[-1])
    frontier = models[-1]  # most expensive (models are price-sorted)
    glyph_note = "" if glyphs else "  (colour-only at scale — see legend)"
    ax.set_title(
        f"Task × model outcomes — ✓ pass · ✗ fail · gray = not evaluated{glyph_note}\n"
        f"models disagree on {len(split)} task(s): {_cap_names(split)} · "
        f"solved by none: {_cap_names(none_solve)} · "
        f"frontier {frontier} on {int(coverage[-1])}/{n_tasks}",
        fontsize=9,
    )
    fig.tight_layout()

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

    matrix_for_plots = load_matrix(matrix_path) if matrix_path and matrix_path.exists() else None
    p3 = plot_cost_savings(results, out_dir, matrix_for_plots)
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
