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


def oracle(model_map: dict, results: dict) -> str:
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


def main(config_path: str = "benchmark/config.yaml") -> None:
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

    ks = args.ks if args.ks is not None else list(range(2, 21, 2))
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

    variances = []
    for pi, (_pname, pvals) in enumerate(zip(param_names, param_values, strict=True)):
        group_means = []
        for v in pvals:
            mask = np.isclose(results_arr[:, pi], v)
            group_means.append(np.mean(results_arr[mask, 3]))
        variances.append(np.var(group_means))

    most_sensitive = param_names[np.argmax(variances)]
    third_idx = np.argsort(variances)[::-1][2]
    print("\n=== SENSITIVITY ANALYSIS ===")
    for pn, pv in zip(param_names, variances, strict=True):
        print(f"  {pn}: variance explained = {pv:.4f}")
    print(f"  Most sensitive parameter: {most_sensitive}")

    p3_idx = third_idx
    p3_name = param_names[p3_idx]

    if most_sensitive in ("success_rate_thresh", "k"):
        fixed_val = best["min_samples"]
        filtered = [r for r in all_results if abs(r["min_samples"] - fixed_val) < 0.1]
        x_vals = sorted(set(r["k"] for r in filtered))
        y_vals = sorted(set(r["success_rate_thresh"] for r in filtered))
        x_name, y_name = "k", "success_rate_thresh"
    else:
        fixed_val = best["success_rate_thresh"]
        filtered = [r for r in all_results if abs(r["success_rate_thresh"] - fixed_val) < 0.1]
        x_vals = sorted(set(r["k"] for r in filtered))
        y_vals = sorted(set(r["min_samples"] for r in filtered))
        x_name, y_name = "k", "min_samples"

    heat_data = np.full((len(y_vals), len(x_vals)), np.nan)
    for r in filtered:
        xi = x_vals.index(r[x_name])
        yi = y_vals.index(r[y_name])
        heat_data[yi, xi] = r["Reward"]

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")

    im = ax.imshow(heat_data, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(y_vals)

    for yi in range(len(y_vals)):
        for xi in range(len(x_vals)):
            val = heat_data[yi, xi]
            if not np.isnan(val):
                cmid = (np.nanmax(heat_data) + np.nanmin(heat_data)) / 2
                color = "white" if val > cmid else "black"
                ax.text(xi, yi, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"Reward by {x_name} and {y_name}\n(fixed {p3_name})")

    cb = fig.colorbar(im, ax=ax, label="Reward")
    cb.ax.tick_params(labelsize=9)

    fig.tight_layout()
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.plot, dpi=150)
    print(f"\nHeatmap saved to {args.plot}")


if __name__ == "__main__":
    main()
