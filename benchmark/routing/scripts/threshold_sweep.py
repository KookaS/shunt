#!/usr/bin/env python3
"""Sweep kNN hyperparameters (k, success_rate_threshold, min_samples) over the
challenges matrix, reporting cost-at-equal-quality sensitivity. Output: CSV +
heatmap in reports/.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from benchmark import config

matplotlib.use("Agg")


def load_matrix(path: Path) -> dict:
    return config.load_matrix(path)


def oracle(results: dict) -> str:
    candidates: list[tuple[str, float]] = []
    for model, outcome in results.items():
        if outcome.get("pass"):
            cost = outcome.get("cost", 0.0)
            candidates.append((model, cost))
    if not candidates:
        if results:
            return min(results, key=lambda m: results[m].get("cost", 0.0))
        return ""
    return min(candidates, key=lambda x: x[1])[0]


def model_cost(model_name: str, matrix: dict) -> float:
    models = matrix["models"]
    if model_name in models:
        m = models[model_name]
        return m.get("input_price", 0) + m.get("output_price", 0)
    return 0.0


def knn_select(
    query_idx: int,
    task_ids: list[str],
    task_descs: list[str],
    features: np.ndarray,
    results_map: dict[str, dict],
    matrix: dict,
    k: int,
    success_rate_thresh: float,
    min_samples: int,
) -> tuple[str, bool, float]:
    sims = cosine_similarity(features[query_idx : query_idx + 1], features).flatten()
    sims[query_idx] = -1.0

    nearest = np.argsort(sims)[::-1][:k]

    model_observations: dict[str, list[bool]] = {}
    for nidx in nearest:
        tid = task_ids[nidx]
        task_results = results_map[tid]
        for model_name, outcome in task_results.items():
            if model_name not in model_observations:
                model_observations[model_name] = []
            model_observations[model_name].append(outcome.get("pass", False))

    scored: list[tuple[str, float, int]] = []
    for model_name, passes in model_observations.items():
        if len(passes) < min_samples:
            continue
        success_rate = sum(passes) / len(passes)
        if success_rate >= success_rate_thresh:
            cost = model_cost(model_name, matrix)
            scored.append((model_name, cost, len(passes)))

    if not scored:
        cheapest_model = min(matrix["models"].keys(), key=lambda m: model_cost(m, matrix))
        chosen = cheapest_model
    else:
        scored.sort(key=lambda x: (x[1], -x[2]))
        chosen = scored[0][0]

    outcome = results_map[task_ids[query_idx]].get(chosen, {"pass": False, "cost": 0.0})
    return chosen, outcome.get("pass", False), outcome.get("cost", 0.0)


def compute_reward(passed: bool, cost: float, gamma: float = 0.1) -> float:
    return 1.0 - gamma * cost if passed else 0.0 - gamma * cost


def evaluate_params(
    task_ids: list[str],
    task_descs: list[str],
    features: np.ndarray,
    results_map: dict[str, dict],
    matrix: dict,
    k: int,
    success_rate_thresh: float,
    min_samples: int,
) -> dict:
    g = config.gamma()
    passes = 0
    total_cost = 0.0
    total_reward = 0.0
    n = len(task_ids)

    for i in range(n):
        _, passed, cost = knn_select(
            i,
            task_ids,
            task_descs,
            features,
            results_map,
            matrix,
            k,
            success_rate_thresh,
            min_samples,
        )
        passes += 1 if passed else 0
        total_cost += cost
        total_reward += compute_reward(passed, cost, g)

    return {
        "k": k,
        "success_rate_thresh": success_rate_thresh,
        "min_samples": min_samples,
        "AvgPerf%": round(passes / n * 100, 2),
        "TotalCost": round(total_cost, 6),
        "Reward": round(total_reward, 6),
    }


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
    config.load(config_path)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--output", default="benchmark/routing/reports/threshold_sweep_results.csv")
    ap.add_argument("--plot", default="benchmark/routing/reports/threshold_sweep_heatmap.png")
    ap.add_argument("--ks", nargs="+", type=int, default=None, help="k values to sweep")
    ap.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=None,
        help="success rate thresholds to sweep",
    )
    ap.add_argument(
        "--min-samples-list",
        nargs="+",
        type=int,
        default=None,
        help="min_samples values to sweep",
    )
    args = ap.parse_args()

    if args.config != config_path:
        config.load(args.config)

    matrix_path = Path(args.matrix) if args.matrix else config.challenges_path()
    matrix = load_matrix(matrix_path)
    task_ids = sorted(matrix["results"].keys())

    if not task_ids:
        print(
            "No results yet — results.csv holds no rows. "
            "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
        )
        return

    task_descs = [matrix["tasks"][tid]["description"] for tid in task_ids]
    results_map = matrix["results"]

    print(f"Loaded {len(task_ids)} tasks from {matrix_path}")

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=500)
    features = vectorizer.fit_transform(task_descs).toarray()
    print(f"Feature matrix: {features.shape}")

    # Sweep k up to the largest usable neighbourhood (n-1 neighbours, self excluded)
    # rather than a hardcoded 20: the reward optimum sat at k=18/20, the edge of the
    # old window, so the sweep never bracketed its own maximum.
    k_max = max(2, len(task_ids) - 1)
    ks = args.ks if args.ks is not None else [k for k in range(2, k_max + 1, 2)] or [2]
    thresholds = (
        args.thresholds
        if args.thresholds is not None
        else [round(0.5 + i * 0.1, 1) for i in range(5)]
    )
    mins_samples = args.min_samples_list if args.min_samples_list is not None else [1, 2, 3, 4, 5]

    print("Parameters sweeps:")
    print(f"  k = {ks}")
    print(f"  success_rate_thresh = {thresholds}")
    print(f"  min_samples = {mins_samples}")
    print(f"  Total combos: {len(ks) * len(thresholds) * len(mins_samples)}")

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    total = len(ks) * len(thresholds) * len(mins_samples)
    count = 0

    for k in ks:
        for thresh in thresholds:
            for ms in mins_samples:
                result = evaluate_params(
                    task_ids,
                    task_descs,
                    features,
                    results_map,
                    matrix,
                    k,
                    thresh,
                    ms,
                )
                all_results.append(result)
                count += 1
                if count % 20 == 0 or count == total:
                    print(f"  [{count}/{total}] k={k}, thresh={thresh}, min_samp={ms}")

    all_results.sort(key=lambda r: r["Reward"], reverse=True)

    with open(args.output, "w", newline="") as f:
        fields = ["k", "success_rate_thresh", "min_samples", "AvgPerf%", "TotalCost", "Reward"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in all_results:
            w.writerow(row)

    print(f"\nResults written to {args.output}")

    best = all_results[0]
    print("\n=== BEST PARAMETER COMBINATION ===")
    print(f"  k                   = {best['k']}")
    print(f"  success_rate_thresh = {best['success_rate_thresh']}")
    print(f"  min_samples         = {best['min_samples']}")
    print(f"  AvgPerf%            = {best['AvgPerf%']}")
    print(f"  TotalCost           = ${best['TotalCost']}")
    print(f"  Reward              = {best['Reward']}")
    print()

    print("=== TOP 10 COMBOS ===")
    print(f"{'k':>3} {'thresh':>7} {'min_samp':>8} {'AvgPerf%':>8} {'TotalCost':>10} {'Reward':>8}")
    for r in all_results[:10]:
        print(
            f"{r['k']:>3} {r['success_rate_thresh']:>7.1f} {r['min_samples']:>8d} "
            f"{r['AvgPerf%']:>8.2f} {r['TotalCost']:>10.6f} {r['Reward']:>8.4f}"
        )

    results_arr = np.array(
        [(r["k"], r["success_rate_thresh"], r["min_samples"], r["Reward"]) for r in all_results]
    )

    param_names = ["k", "success_rate_thresh", "min_samples"]
    param_values = [ks, thresholds, mins_samples]

    eta_sq = _variance_explained(results_arr, param_values)
    most_sensitive = param_names[int(np.argmax(eta_sq))]
    print("\n=== SENSITIVITY ANALYSIS ===")
    print("  eta^2 = fraction of Reward variance explained by that parameter (0-1)")
    for pn, pv in zip(param_names, eta_sq, strict=True):
        print(f"  {pn}: variance explained (eta^2) = {pv:.4f}")
    print(f"  Most sensitive parameter: {most_sensitive}")

    if most_sensitive in ("success_rate_thresh", "k"):
        fixed_name, fixed_val = "min_samples", best["min_samples"]
        y_name = "success_rate_thresh"
    else:
        fixed_name, fixed_val = "success_rate_thresh", best["success_rate_thresh"]
        y_name = "min_samples"
    filtered = [r for r in all_results if abs(r[fixed_name] - fixed_val) < 0.1]

    _plot_sweep_heatmap(
        filtered,
        Path(args.plot),
        y_name=y_name,
        fixed=(fixed_name, fixed_val),
        swept_ks=ks,
    )
    print(f"\nHeatmap saved to {args.plot}")


def _variance_explained(results_arr: np.ndarray, param_values: list[list]) -> list[float]:
    """Per-parameter eta^2: the fraction (0-1) of Reward variance between its levels.

    The previous ``np.var(group_means)`` was an UNNORMALIZED variance of group
    means — it printed values above 1 while labelled "variance explained".
    """
    rewards = results_arr[:, 3]
    total_ss = float(np.sum((rewards - rewards.mean()) ** 2))
    out: list[float] = []
    for pi, pvals in enumerate(param_values):
        between_ss = 0.0
        for v in pvals:
            mask = np.isclose(results_arr[:, pi], v)
            n_g = int(mask.sum())
            if n_g:
                between_ss += n_g * (float(np.mean(rewards[mask])) - float(rewards.mean())) ** 2
        out.append(between_ss / total_ss if total_ss > 0 else 0.0)
    return out


def _plot_sweep_heatmap(
    filtered: list[dict],
    plot_path: Path,
    *,
    y_name: str,
    fixed: tuple[str, float],
    swept_ks: list[int],
) -> None:
    """Reward heatmap over (k, y_name) at a fixed third parameter."""
    x_name = "k"
    x_vals = sorted({r[x_name] for r in filtered})
    y_vals = sorted({r[y_name] for r in filtered})
    heat_data = np.full((len(y_vals), len(x_vals)), np.nan)
    for r in filtered:
        heat_data[y_vals.index(r[y_name]), x_vals.index(r[x_name])] = r["Reward"]

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")
    im = ax.imshow(heat_data, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(y_vals)

    cmid = (np.nanmax(heat_data) + np.nanmin(heat_data)) / 2
    # Shrink the in-cell labels as the k sweep widens, or they run into each other.
    fontsize = 8 if len(x_vals) <= 12 else 5.0
    for yi in range(len(y_vals)):
        for xi in range(len(x_vals)):
            val = heat_data[yi, xi]
            if not np.isnan(val):
                color = "white" if val > cmid else "black"
                ax.text(
                    xi, yi, f"{val:.2f}", ha="center", va="center", fontsize=fontsize, color=color
                )

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    fixed_name, fixed_val = fixed
    ax.set_title(f"Reward by {x_name} and {y_name}\n(fixed {fixed_name} = {fixed_val})")

    # Bracketing honesty: an optimum sitting on the first/last swept k is not a
    # located maximum — the true optimum may lie outside the swept window.
    best_row = max(filtered, key=lambda r: r["Reward"])
    if swept_ks and best_row["k"] in (min(swept_ks), max(swept_ks)):
        ax.text(
            0.5,
            -0.13,
            f"⚠ max Reward ({best_row['Reward']:.2f}) sits at k={best_row['k']}, the EDGE of the "
            f"swept range k∈[{min(swept_ks)}, {max(swept_ks)}] — the optimum is not bracketed",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#B00020",
        )

    cb = fig.colorbar(im, ax=ax, label="Reward")
    cb.ax.tick_params(labelsize=9)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
