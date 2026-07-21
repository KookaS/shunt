#!/usr/bin/env python3
"""Outcome-vector NN proxy — clustering visualization for the challenges matrix."""

# Neighbours are found over MODEL-OUTCOME vectors (pass/cost/tokens/calls) as a
# stand-in for task-embedding similarity, then a simplified nearest-neighbour
# classifier picks a model. This is NOT the routed kNN strategy that
# plot_strategies.py evaluates (routing.strategies.knn.kNNStrategy, which embeds
# task text and gates on success_rate_threshold/min_samples), so its costs are
# not comparable to strategy_comparison.png. Every reader-facing label here says
# "outcome-vector NN proxy" for that reason.

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from benchmark import config  # noqa: E402


def build_feature_vectors(results, models_order):
    task_ids = sorted(results.keys())
    n = len(task_ids)
    vecs = np.zeros((n, len(models_order) * 4))

    for i, tid in enumerate(task_ids):
        for j, model in enumerate(models_order):
            r = results[tid].get(model, {})
            passed = 1.0 if r.get("pass", False) else 0.0
            cost = r.get("cost", 0.0)
            total_tok = r.get("in_tok", 0) + r.get("out_tok", 0)
            calls = r.get("calls", 0)
            base = j * 4
            vecs[i, base] = passed
            vecs[i, base + 1] = np.log10(cost + 1e-9) + 5
            vecs[i, base + 2] = np.log10(total_tok + 1)
            vecs[i, base + 3] = np.log10(calls + 1)

    return task_ids, vecs


def compute_task_similarity(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = vecs / norms
    return normalized @ normalized.T


def knn_select(vecs, query_idx, models_order, k=10):
    similarities = vecs @ vecs[query_idx]
    similarities[query_idx] = -np.inf
    nearest = np.argsort(similarities)[-k:][::-1]

    model_scores = {}
    for model_idx, model in enumerate(models_order):
        model_pass_col = model_idx * 4
        pass_rates = [vecs[n, model_pass_col] for n in nearest]
        avg_pass = np.mean(pass_rates)
        model_scores[model] = avg_pass

    max_pass = max(model_scores.values())
    tied = [m for m, s in model_scores.items() if s == max_pass]

    price_order = sorted(
        models_order, key=lambda m: config.cost_per_1m(m, config.enabled_pricing())
    )
    for m in price_order:
        if m in tied:
            return m
    return tied[0] if tied else models_order[0]


def compute_neighborhood_purity(similarity_matrix, task_ids, vecs, models_order, k=10):
    n = len(task_ids)
    selections = {}
    for i in range(n):
        selections[task_ids[i]] = knn_select(vecs, i, models_order, k=k)

    purity = np.zeros(n)
    for i in range(n):
        sims = similarity_matrix[i].copy()
        sims[i] = -np.inf
        nearest = np.argsort(sims)[-k:][::-1]
        selected = selections[task_ids[i]]
        match_count = sum(1 for n_idx in nearest if selections[task_ids[n_idx]] == selected)
        purity[i] = match_count / k

    return purity, selections


def _default_model_colors(models: list[str]) -> dict[str, str]:
    palette = [
        "#2ecc71",
        "#3498db",
        "#9b59b6",
        "#f39c12",
        "#e74c3c",
        "#1a1a2e",
        "#e67e22",
        "#1abc9c",
    ]
    return {m: palette[i % len(palette)] for i, m in enumerate(models)}


def main(config_path: str = "benchmark/benchmark.yaml"):
    config.load(config_path)
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--output-dir", default="benchmark/routing/reports")
    # Default k tracks the configured kNN strategy (benchmark.yaml
    # strategies.knn.k) so the proxy at least uses the same neighbourhood size
    # as the routed strategy — the neighbour SPACE still differs (see docstring).
    ap.add_argument(
        "--k",
        type=int,
        default=None,
        help="neighbourhood size for the proxy (default: configured strategies.knn.k)",
    )
    args = ap.parse_args()

    if args.config != config_path:
        config.load(args.config)

    matrix_path = Path(args.matrix) if args.matrix else config.challenges_path()
    matrix = config.load_matrix(matrix_path)

    models_order = config.enabled_models()
    if not models_order:
        models_order = list(matrix.get("models", {}).keys())

    k_neighbors = args.k if args.k is not None else int(config.knn_params().get("k", 10))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = matrix["results"]

    if not results:
        print(
            "No results yet — results.csv holds no rows. "
            "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
        )
        return

    task_ids, vecs = build_feature_vectors(results, models_order)
    n = len(task_ids)
    print(f"Loaded {n} tasks, feature vectors: {vecs.shape}")

    model_colors = _default_model_colors(models_order)

    selections = {}
    for i in range(n):
        selections[task_ids[i]] = knn_select(vecs, i, models_order, k=k_neighbors)

    unique, counts = np.unique(list(selections.values()), return_counts=True)
    print("outcome-vector NN proxy — model allocation:")
    for m, c in sorted(zip(unique, counts, strict=True), key=lambda x: -x[1]):
        print(f"  {m}: {c} tasks ({c / n * 100:.1f}%)")

    similarity = compute_task_similarity(vecs)
    purity, _ = compute_neighborhood_purity(similarity, task_ids, vecs, models_order, k=k_neighbors)
    print(f"Mean neighborhood purity: {np.mean(purity):.3f}")

    # 1. PCA Scatter
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vecs)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [model_colors.get(selections[tid], "#9E9E9E") for tid in task_ids]
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=40,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.3,
    )

    legend_handles = []
    for m in models_order:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=model_colors.get(m, "#9E9E9E"),
                markersize=10,
                label=m,
            )
        )
    ax.legend(handles=legend_handles, title="Proxy-selected model", fontsize=8)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.set_title("Outcome-Vector NN Proxy: PCA Projection of Task Features")
    fig.tight_layout()
    fig.savefig(str(output_dir / "knn_pca_scatter.png"), dpi=150)
    print("Saved knn_pca_scatter.png")
    plt.close(fig)

    # 2. Neighborhood Purity Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sort_order = sorted(
        range(n),
        key=lambda i: (
            list(models_order).index(selections[task_ids[i]])
            if selections[task_ids[i]] in models_order
            else 999
        ),
    )
    sorted_sim = similarity[sort_order][:, sort_order]

    im = axes[0].imshow(sorted_sim, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Task Similarity Matrix (grouped by selected model)")
    axes[0].set_xlabel("Task index (sorted)")
    axes[0].set_ylabel("Task index (sorted)")
    cbar = fig.colorbar(im, ax=axes[0], shrink=0.8)
    cbar.set_label("Cosine similarity")

    sep_positions = [0]
    current = selections[task_ids[sort_order[0]]]
    for i in range(1, n):
        m = selections[task_ids[sort_order[i]]]
        if m != current:
            sep_positions.append(i)
            current = m
    sep_positions.append(n)
    for sp in sep_positions[:-1]:
        axes[0].axhline(sp - 0.5, color="white", linestyle="--", linewidth=0.5, alpha=0.5)
        axes[0].axvline(sp - 0.5, color="white", linestyle="--", linewidth=0.5, alpha=0.5)

    y_centers = []
    for j in range(len(sep_positions) - 1):
        y_centers.append((sep_positions[j] + sep_positions[j + 1]) / 2)
    tick_models = []
    for j in range(len(sep_positions) - 1):
        mid = sort_order[sep_positions[j]]
        tick_models.append(selections[task_ids[mid]])
    axes[0].set_yticks(y_centers)
    axes[0].set_yticklabels(tick_models, fontsize=8)
    axes[0].set_xticks([])

    purity_by_model: dict[str, list[float]] = {m: [] for m in models_order}
    for i, tid in enumerate(task_ids):
        sel = selections[tid]
        if sel in purity_by_model:
            purity_by_model[sel].append(purity[i])

    x_pos = np.arange(len(models_order))
    mean_purities = [np.mean(purity_by_model[m]) if purity_by_model[m] else 0 for m in models_order]
    bar_colors = [model_colors.get(m, "#9E9E9E") for m in models_order]
    bars = axes[1].bar(x_pos, mean_purities, color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(models_order, rotation=45, ha="right", fontsize=9)
    axes[1].set_ylabel("Mean neighborhood purity")
    axes[1].set_title("Outcome-Vector NN Proxy — Neighborhood Purity by Selected Model")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(
        y=0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.7, label="0.5 baseline"
    )
    axes[1].legend(fontsize=8)

    for bar, val in zip(bars, mean_purities, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(str(output_dir / "neighborhood_purity.png"), dpi=150)
    print("Saved neighborhood_purity.png")
    plt.close(fig)

    # 3. Model Allocation Bar
    fig, ax = plt.subplots(figsize=(10, 6))
    model_counts = [counts[list(unique).index(m)] if m in unique else 0 for m in models_order]
    bars = ax.bar(
        range(len(models_order)),
        model_counts,
        color=[model_colors.get(m, "#9E9E9E") for m in models_order],
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xticks(range(len(models_order)))
    ax.set_xticklabels(models_order, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Number of tasks routed")
    ax.set_title(f"Outcome-Vector NN Proxy — Model Allocation Across {n} Tasks (k={k_neighbors})")

    for bar, cnt in zip(bars, model_counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(cnt),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylim(0, max(model_counts) * 1.15)
    fig.tight_layout()
    fig.savefig(str(output_dir / "model_allocation.png"), dpi=150)
    print("Saved model_allocation.png")
    plt.close(fig)

    # 4. Cost comparison plot (replaces the old knn_viz_report.txt)
    pricing = config.enabled_pricing()
    cheapest_model = min(pricing, key=lambda m: config.cost_per_1m(m, pricing)) if pricing else ""
    frontier_model = max(pricing, key=lambda m: config.cost_per_1m(m, pricing)) if pricing else ""

    cheap_cost = (
        sum(results[tid].get(cheapest_model, {}).get("cost", 0) for tid in task_ids)
        if cheapest_model
        else 0.0
    )
    frontier_cost = (
        sum(results[tid].get(frontier_model, {}).get("cost", 0) for tid in task_ids)
        if frontier_model
        else 0.0
    )
    knn_total = sum(results[tid].get(selections[tid], {}).get("cost", 0) for tid in task_ids)
    knn_pass = sum(
        1 for tid in task_ids if results[tid].get(selections[tid], {}).get("pass", False)
    )
    # Phantom-baseline guard (matches report.py::_frontier_coverage): the frontier
    # bar sums cost over only the tasks the frontier model actually ran. If that is
    # a subset of n, "saves X% vs frontier" compares kNN-over-n against
    # frontier-over-few — apples to oranges. Suppress the claim and flag it.
    frontier_covered = (
        sum(1 for tid in task_ids if frontier_model in results[tid]) if frontier_model else 0
    )
    phantom_frontier = bool(frontier_model) and frontier_covered < n
    savings_pct = (1 - knn_total / frontier_cost) * 100 if frontier_cost else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [
        f"cheapest\n({cheapest_model})",
        "outcome-vector\nNN proxy",
        f"frontier\n({frontier_model})",
    ]
    values = [cheap_cost, knn_total, frontier_cost]
    bars = ax.bar(labels, values, color=["#2196F3", "#E69F00", "#F44336"], edgecolor="black")
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(values) or 1) * 0.01,
            f"${val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylabel("Total cost across tasks ($)")
    if phantom_frontier:
        title_tail = f"pass {knn_pass}/{n} ({knn_pass / n * 100:.0f}%) — frontier savings N/A"
    else:
        title_tail = (
            f"pass {knn_pass}/{n} ({knn_pass / n * 100:.0f}%), saves {savings_pct:.0f}% vs frontier"
        )
    ax.set_title(f"Outcome-Vector NN Proxy — Cost Across {n} Tasks (k={k_neighbors})\n{title_tail}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    # Figure-level captions BELOW the axes: drawn inside the bars, they collided
    # with the value labels.
    fig.subplots_adjust(bottom=0.34 if phantom_frontier else 0.22)
    fig.text(
        0.5,
        0.02,
        "Note: the middle bar is a similarity PROXY over model-outcome vectors "
        "(pass/cost/tokens/calls),\nnot the routed kNN strategy — its cost is NOT "
        "comparable to strategy_comparison.png",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
        style="italic",
    )
    if phantom_frontier:
        fig.text(
            0.5,
            0.13,
            f"⚠ PHANTOM BASELINE — {frontier_model} evaluated on {frontier_covered}/{n} "
            f"tasks;\nits bar sums only those, so a 'vs frontier' saving is not comparable",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#B00020",
            fontweight="bold",
        )
    fig.savefig(str(output_dir / "knn_cost_comparison.png"), dpi=150)
    print("Saved knn_cost_comparison.png")
    plt.close(fig)

    savings_str = (
        "frontier savings N/A (phantom baseline)"
        if phantom_frontier
        else f"saves {savings_pct:.1f}% vs frontier"
    )
    print(
        f"Cost across {n} tasks: cheapest ${cheap_cost:.4f}, "
        f"outcome-vector NN proxy ${knn_total:.4f}, "
        f"frontier ${frontier_cost:.4f} ({frontier_covered}/{n} tasks); "
        f"proxy pass {knn_pass}/{n}, {savings_str}"
    )


if __name__ == "__main__":
    main()
