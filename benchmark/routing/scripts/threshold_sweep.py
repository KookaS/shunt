#!/usr/bin/env python3
"""Sweep kNN hyperparameters (k, success_rate_threshold, min_samples) over the
challenges matrix, reporting cost-at-equal-quality sensitivity. Output: CSV +
heatmap in reports/.
"""

# Neighbourhoods use the real shipped jina embedder (the same ``Embedder`` the router
# runs), never a TF-IDF proxy.

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from sklearn.metrics.pairwise import cosine_similarity

from benchmark import config
from benchmark.routing.strategies.knn import _embed_texts

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
) -> tuple[str, bool, float, bool]:
    """Returns ``(chosen, passed, cost, scored)``. ``scored`` is False when the
    chosen model has no measured cell on the query task — a coverage gap the caller
    EXCLUDES from aggregation, never imputes as fail@$0."""
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

    outcome = results_map[task_ids[query_idx]].get(chosen)
    if outcome is None:
        return chosen, False, 0.0, False
    return chosen, outcome.get("pass", False), outcome.get("cost", 0.0), True


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
    scored = 0

    for i in range(n):
        _, passed, cost, ok = knn_select(
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
        # A coverage-gap escalation (chosen model unmeasured on this task) is
        # UNSCORABLE — excluded from the aggregation, not imputed fail@$0.
        if not ok:
            continue
        scored += 1
        passes += 1 if passed else 0
        total_cost += cost
        total_reward += compute_reward(passed, cost, g)

    return {
        "k": k,
        "success_rate_thresh": success_rate_thresh,
        "min_samples": min_samples,
        "n_scored": scored,
        "n_excluded": n - scored,
        "AvgPerf%": round(passes / scored * 100, 2) if scored else 0.0,
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

    features = _embed_texts(task_descs)
    print(f"Real jina-embedding matrix: {features.shape}")

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
        fields = [
            "k",
            "success_rate_thresh",
            "min_samples",
            "n_scored",
            "n_excluded",
            "AvgPerf%",
            "TotalCost",
            "Reward",
        ]
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
    print(f"  AvgPerf%            = {best['AvgPerf%']} (over {best['n_scored']} scored tasks)")
    print(f"  TotalCost           = ${best['TotalCost']}")
    print(f"  Reward              = {best['Reward']}")
    print(f"  excluded (coverage-gap escalations, unmeasured chosen cell) = {best['n_excluded']}")
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

    sensitivity = dict(zip(param_names, eta_sq, strict=True))
    excluded_max = max((int(r["n_excluded"]) for r in all_results), default=0)
    _plot_sweep_heatmap(
        filtered,
        Path(args.plot),
        y_name=y_name,
        fixed=(fixed_name, fixed_val),
        swept_ks=ks,
        sensitivity=sensitivity,
        excluded_max=excluded_max,
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


def _annotate_cells(
    ax: Axes,
    heat_data: np.ndarray,
    *,
    best_yx: tuple[int, int],
    label_step: int,
    fontsize: float,
) -> None:
    """In-cell reward labels: every ``label_step``-th column plus the optimum.

    Viridis is bright at HIGH values, so high cells take dark text and low cells
    light text (the reverse trips readers up); the split is at the colour midpoint.
    """
    cmid = (np.nanmax(heat_data) + np.nanmin(heat_data)) / 2
    best_yi, best_xi = best_yx
    for yi in range(heat_data.shape[0]):
        for xi in range(heat_data.shape[1]):
            val = heat_data[yi, xi]
            if np.isnan(val):
                continue
            is_best = yi == best_yi and xi == best_xi
            if xi % label_step != 0 and not is_best:
                continue
            ax.text(
                xi,
                yi,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=fontsize + 1 if is_best else fontsize,
                fontweight="bold" if is_best else "normal",
                color="black" if val > cmid else "white",
            )


def _plot_sweep_heatmap(
    filtered: list[dict],
    plot_path: Path,
    *,
    y_name: str,
    fixed: tuple[str, float],
    swept_ks: list[int],
    sensitivity: dict[str, float],
    excluded_max: int = 0,
) -> None:
    """Reward heatmap over (k, y_name) at a fixed third parameter."""
    x_name = "k"
    x_vals = sorted({r[x_name] for r in filtered})
    y_vals = sorted({r[y_name] for r in filtered})
    heat_data = np.full((len(y_vals), len(x_vals)), np.nan)
    for r in filtered:
        heat_data[y_vals.index(r[y_name]), x_vals.index(r[x_name])] = r["Reward"]

    best_row = max(filtered, key=lambda r: r["Reward"])
    best_yx = (y_vals.index(best_row[y_name]), x_vals.index(best_row[x_name]))

    # Width scales with the (data-driven) k sweep so cells stay legible; a wide k
    # range would otherwise crush the columns together.
    fig_w = min(20.0, max(11.0, 0.55 * len(x_vals) + 3.0))
    fig, ax = plt.subplots(figsize=(fig_w, 6.8))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")
    im = ax.imshow(heat_data, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(y_vals)

    # Many columns → label every 2nd one (plus the optimum) so text never collides;
    # the colourbar carries the full gradient, the CSV the exact values.
    label_step = 1 if len(x_vals) <= 12 else 2
    _annotate_cells(ax, heat_data, best_yx=best_yx, label_step=label_step, fontsize=7.5)

    # Ring the optimum so the brightest cell is unmistakable, not just "somewhere yellow".
    ax.add_patch(
        Rectangle(
            (best_yx[1] - 0.5, best_yx[0] - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#d03b3b",
            linewidth=2.2,
        )
    )

    # Human-readable axis labels: the swept params are kNN-strategy knobs, so spell
    # out what each one is rather than printing the bare config key.
    axis_labels = {
        "k": "k  (number of nearest neighbours)",
        "success_rate_thresh": (
            "success_rate_thresh\n"
            "(neighbour pass-rate below which the router escalates off the cheap model)"
        ),
        "min_samples": (
            "min_samples\n(min neighbours with a recorded outcome before the router trusts them)"
        ),
    }
    fixed_name, fixed_val = fixed
    fixed_eta = sensitivity.get(fixed_name)
    fixed_note = f", η²={fixed_eta:.2f} — negligible effect" if fixed_eta is not None else ""
    ax.set_xlabel(axis_labels.get(x_name, x_name))
    ax.set_ylabel(axis_labels.get(y_name, y_name), fontsize=9)
    ax.set_title(
        f"kNN routing reward over (k × {y_name})  —  brighter = higher reward = better\n"
        f"real jina embeddings · reward = tasks passed − γ·cost;  "
        f"fixed {fixed_name} = {fixed_val} (its best{fixed_note})",
        fontsize=12,
    )

    # "What good looks like" + bracketing honesty, stacked below the x-axis so
    # neither overlaps the labels. Data-driven from the sensitivity (η²) analysis.
    y_eta, k_eta = sensitivity.get(y_name), sensitivity.get(x_name)
    if y_eta is not None and k_eta is not None:
        ax.text(
            0.5,
            -0.19,
            f"What good looks like: settle in the brightest band. Reward is driven by "
            f"{y_name} (η²={y_eta:.2f}); k barely moves it (η²={k_eta:.2f}) — "
            f"pick k for stability, not reward.",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#333333",
        )

    lo, hi = min(swept_ks), max(swept_ks)
    if swept_ks and best_row["k"] in (lo, hi):
        # An optimum on the first/last swept k is not located — the true peak may
        # lie outside the window.
        note, colour = (
            f"⚠ max reward ({best_row['Reward']:.1f}) sits at k={best_row['k']}, the EDGE of the "
            f"swept range k∈[{lo}, {hi}] — the optimum is NOT bracketed (may lie beyond)",
            "#d03b3b",
        )
    else:
        note, colour = (
            f"✓ optimum k={best_row['k']} lies inside the swept range k∈[{lo}, {hi}] "
            f"— the maximum is bracketed",
            "#0ca30c",
        )
    ax.text(
        0.5,
        -0.26,
        note,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=colour,
    )

    if excluded_max > 0:
        ax.text(
            0.5,
            -0.33,
            f"Sweep EXCLUDES coverage-gap escalations (chosen model unmeasured on a task) "
            f"per combo — up to {excluded_max} task(s) dropped; never imputed fail@$0.",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#B71C1C",
            style="italic",
        )

    cb = fig.colorbar(
        im, ax=ax, label="Routing reward  =  tasks passed − γ·cost  (higher = better)"
    )
    cb.ax.tick_params(labelsize=9)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
