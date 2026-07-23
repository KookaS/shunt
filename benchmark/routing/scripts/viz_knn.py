#!/usr/bin/env python3
"""kNN router visualization on REAL jina embeddings for the challenges matrix."""

# Neighbours are found over the REAL shipped jina prompt embeddings (the same
# ``Embedder`` the router runs), then a nearest-neighbour classifier picks a model
# with the SAME cost-aware rule the shipped router uses — cheapest model whose
# neighbourhood pass-rate clears the threshold (see knn_select). Pass-rates are read
# from the real measured outcomes. This mirrors routing.strategies.knn.kNNStrategy's
# decision; the cost bars use each selected model's default-arm cost, so exact totals
# differ from strategy_comparison.png (which runs the live engine).

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from benchmark import config  # noqa: E402
from benchmark.routing import plot_style  # noqa: E402
from benchmark.routing.strategies.knn import _embed_texts  # noqa: E402

# Cheapest-above-threshold cutoff, matching the shipped KnnPolicy.success_rate_threshold
# (0.6) and router.selection.SelectionRule. The proxy applies the product rule so its
# allocation reflects what the router would do, not a best-quality diagnostic.
_DEFAULT_SUCCESS_RATE_THRESHOLD = 0.6

# One-line imputation disclosure, restated verbatim on every figure whose selection
# reads neighbourhood pass-rates (an unmeasured cell counts as a non-pass there).
_IMPUTE_CAVEAT = (
    "Selection reads measured pass-rates; an unmeasured (model, task) cell counts as a "
    "non-pass for that model's neighbourhood rate."
)


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


def build_task_embeddings(matrix, task_ids):
    """Real normalized jina prompt embeddings (unit vectors), aligned to ``task_ids``."""
    tasks = matrix.get("tasks", {})
    descs = [tasks.get(tid, {}).get("description", tid) for tid in task_ids]
    emb = np.asarray(_embed_texts(descs), dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def compute_task_similarity(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = vecs / norms
    return normalized @ normalized.T


def knn_select(vecs, emb, query_idx, models_order, k=10, success_rate_threshold=None):
    """Cost-aware selection over REAL-embedding neighbours: cheapest model whose
    neighbourhood pass-rate clears the threshold, else best-available (mirrors
    SelectionRule). ``emb`` = normalized jina vectors; ``vecs`` = measured pass-rates."""
    threshold = (
        _DEFAULT_SUCCESS_RATE_THRESHOLD
        if success_rate_threshold is None
        else success_rate_threshold
    )
    similarities = emb @ emb[query_idx]
    similarities[query_idx] = -np.inf
    nearest = np.argsort(similarities)[-k:][::-1]

    model_scores = {}
    for model_idx, model in enumerate(models_order):
        model_pass_col = model_idx * 4
        pass_rates = [vecs[n, model_pass_col] for n in nearest]
        model_scores[model] = float(np.mean(pass_rates))

    price_order = sorted(
        models_order, key=lambda m: config.cost_per_1m(m, config.enabled_pricing())
    )

    # Cheapest model that clears the threshold (the product rule).
    for m in price_order:
        if model_scores.get(m, 0.0) >= threshold:
            return m

    # None qualify -> escalate to the best-available model (highest pass, cheapest tie),
    # matching SelectionRule's fall-through past the tested set.
    max_pass = max(model_scores.values()) if model_scores else 0.0
    for m in price_order:
        if model_scores.get(m, 0.0) == max_pass:
            return m
    return price_order[0] if price_order else models_order[0]


def compute_neighborhood_purity(
    similarity_matrix, task_ids, vecs, emb, models_order, k=10, success_rate_threshold=None
):
    n = len(task_ids)
    selections = {}
    for i in range(n):
        selections[task_ids[i]] = knn_select(
            vecs, emb, i, models_order, k=k, success_rate_threshold=success_rate_threshold
        )

    purity = np.zeros(n)
    for i in range(n):
        sims = similarity_matrix[i].copy()
        sims[i] = -np.inf
        nearest = np.argsort(sims)[-k:][::-1]
        selected = selections[task_ids[i]]
        match_count = sum(1 for n_idx in nearest if selections[task_ids[n_idx]] == selected)
        purity[i] = match_count / k

    return purity, selections


def descriptive_model_stats(results, models_order):
    """STRATEGY-AGNOSTIC per-model stats straight from the raw matrix — no routing
    rule in the loop. Returns per-model pass rate + mean cost on (a) the full set of
    tasks THAT model ran and (b) the common-coverage subset every model ran."""
    tasks = sorted(results.keys())
    common = [t for t in tasks if all(m in results[t] for m in models_order)]
    stats = {}
    for m in models_order:
        measured = [t for t in tasks if m in results[t]]
        full_pass = [1.0 if results[t][m].get("pass") else 0.0 for t in measured]
        comm_pass = [1.0 if results[t][m].get("pass") else 0.0 for t in common if m in results[t]]
        comm_cost = [
            float(results[t][m].get("real_cost", results[t][m].get("cost", 0.0)))
            for t in common
            if m in results[t]
        ]
        stats[m] = {
            "n_measured": len(measured),
            "full_pass": float(np.mean(full_pass)) if full_pass else float("nan"),
            "common_pass": float(np.mean(comm_pass)) if comm_pass else float("nan"),
            "common_cost": float(np.mean(comm_cost)) if comm_cost else 0.0,
        }
    return stats, len(common), len(tasks)


def plot_descriptive_model_performance(results, models_order, output_dir):
    """Descriptive all-model figure computed directly from results.csv, irrespective of
    any routing/exploration strategy — complements (does not replace) the strategy-
    conditioned model_allocation / neighborhood_purity plots that show only selected models."""
    stats, n_common, n_total = descriptive_model_stats(results, models_order)
    colors = plot_style.model_color_map(models_order)
    bar_colors = [colors[m] for m in models_order]
    x = np.arange(len(models_order))
    w = 0.38

    fig, (ax_p, ax_c) = plt.subplots(1, 2, figsize=(16, 7))

    full = [stats[m]["full_pass"] for m in models_order]
    common = [stats[m]["common_pass"] for m in models_order]
    gaps = [
        abs(f - c) for f, c in zip(full, common, strict=True) if not (np.isnan(f) or np.isnan(c))
    ]
    max_gap_pp = (max(gaps) * 100) if gaps else 0.0

    ax_p.bar(
        x - w / 2,
        full,
        w,
        color=bar_colors,
        edgecolor="black",
        alpha=0.5,
        label="full measured set",
    )
    ax_p.bar(
        x + w / 2,
        common,
        w,
        color=bar_colors,
        edgecolor="black",
        label=f"common subset (N={n_common}, cross-comparable)",
    )
    for m, xi in zip(models_order, x, strict=True):
        top = max(
            [v for v in (stats[m]["full_pass"], stats[m]["common_pass"]) if not np.isnan(v)] or [0]
        )
        ax_p.text(
            xi, top + 0.02, f"n={stats[m]['n_measured']}", ha="center", va="bottom", fontsize=7
        )
    ax_p.set_xticks(x)
    ax_p.set_xticklabels(models_order, rotation=45, ha="right", fontsize=9)
    ax_p.set_ylabel("Pass rate")
    ax_p.set_ylim(0, 1.12)
    ax_p.set_title("Per-model pass rate — full measured set vs common-coverage subset")
    ax_p.legend(fontsize=8, loc="lower right")

    costs = [max(stats[m]["common_cost"], 1e-9) for m in models_order]
    cbars = ax_c.bar(x, costs, color=bar_colors, edgecolor="black")
    ax_c.set_yscale("log")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(models_order, rotation=45, ha="right", fontsize=9)
    ax_c.set_ylabel("Mean real cost per task, common subset ($, log)")
    ax_c.set_title(f"Per-model mean cost on the {n_common} common-coverage tasks")
    for bar, m in zip(cbars, models_order, strict=True):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"${stats[m]['common_cost']:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.suptitle(
        "Descriptive all-model view — every enabled model, straight from results.csv "
        "(NO routing/exploration strategy in the loop)",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.01,
        "Coverage is ADAPTIVE (frontier models run on fewer tasks): each full-set bar is "
        "computed on the n tasks\nTHAT model ran, so full-set bars are NOT strictly "
        "cross-comparable (a selection / MNAR bias). The darker bars\n"
        f"restrict to the {n_common} of {n_total} tasks EVERY model ran — that subset IS "
        "cross-comparable. Here full-set vs\n"
        f"common-subset pass rates differ by at most {max_gap_pp:.1f}pp, so the pooling bias "
        "is small in this matrix;\ntrust the common-subset bars for ranking.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(str(output_dir / "model_performance_descriptive.png"), dpi=150)
    print("Saved model_performance_descriptive.png")
    plt.close(fig)


# Secondary (non-colour) identity channel for the all-pairs scatter form. The
# canonical model palette (plot_style.OKABE_ITO) lands adjacent hues in the 6-8 CVD
# floor band on the all-pairs pairlist, so every scatter using it MUST carry a
# second channel — here a per-model marker shape — never colour alone (plot_style
# color note; dataviz check 4). Assigned by POSITION in models_order so a model's
# marker is as stable across figures as its colour.
_MODEL_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")


def _model_marker_map(models_in_order: list[str]) -> dict[str, str]:
    """Assign each model a fixed marker shape by position — the scatter's second
    (non-colour) identity channel, mirroring plot_style.model_color_map."""
    return {m: _MODEL_MARKERS[i % len(_MODEL_MARKERS)] for i, m in enumerate(models_in_order)}


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
        help="neighbourhood size for the kNN router (real jina neighbourhoods) "
        "(default: configured strategies.knn.k)",
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
    threshold = float(
        config.knn_params().get("success_rate_threshold", _DEFAULT_SUCCESS_RATE_THRESHOLD)
    )
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
    emb = build_task_embeddings(matrix, task_ids)
    n = len(task_ids)
    print(f"Loaded {n} tasks; real jina embeddings {emb.shape}, pass-rate vectors {vecs.shape}")

    # ONE canonical model palette for the whole file — the same Okabe-Ito map the
    # descriptive figure uses (plot_style.model_color_map), so a model wears the
    # SAME colour in every plot here and in model_performance_descriptive.png.
    model_colors = plot_style.model_color_map(models_order)
    model_markers = _model_marker_map(models_order)

    selections = {}
    for i in range(n):
        selections[task_ids[i]] = knn_select(
            vecs, emb, i, models_order, k=k_neighbors, success_rate_threshold=threshold
        )

    unique, counts = np.unique(list(selections.values()), return_counts=True)
    print("kNN router (real jina neighbourhoods) — model allocation:")
    for m, c in sorted(zip(unique, counts, strict=True), key=lambda x: -x[1]):
        print(f"  {m}: {c} tasks ({c / n * 100:.1f}%)")

    similarity = compute_task_similarity(emb)
    purity, _ = compute_neighborhood_purity(
        similarity,
        task_ids,
        vecs,
        emb,
        models_order,
        k=k_neighbors,
        success_rate_threshold=threshold,
    )
    print(f"Mean neighborhood purity: {np.mean(purity):.3f}")

    # 1. PCA Scatter — one point per task, coloured AND shaped by the proxy's
    # selected model. Marker shape is a second identity channel so the two
    # adjacent-blue Okabe hues (deepseek / qwen) never rely on colour alone in
    # this all-pairs scatter form (plot_style color note; dataviz check 4).
    # random_state pins the randomized SVD solver ('auto' picks it for 768-d input),
    # so the committed scatter is byte-reproducible run-to-run, not just content-stable.
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(emb)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(11, 8))
    selected_models = [m for m in models_order if m in set(unique)]
    for m in selected_models:
        idx = [i for i, tid in enumerate(task_ids) if selections[tid] == m]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            color=model_colors[m],
            marker=model_markers[m],
            s=55,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.4,
            label=f"{m} (n={len(idx)})",
        )

    ax.legend(title="kNN-router-selected model (colour + shape)", fontsize=9, loc="upper right")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% of variance)")
    ax.set_title(
        "kNN router on REAL jina embeddings — PCA of the prompt-embedding space\n"
        "(each task coloured by the model the router selects for it)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    never = [m for m in models_order if m not in set(unique)]
    never_txt = (
        f"Enabled models the router never selected (absent by result, not missing data): "
        f"{', '.join(never)}.\n"
        if never
        else ""
    )
    fig.text(
        0.5,
        0.015,
        "Each dot = one task, placed by the 2 leading principal components of its REAL "
        "768-d jina prompt embedding.\n"
        + never_txt
        + "GOOD = each colour/shape forms a tight cluster (tasks needing the same model "
        "sit together, so a near-neighbour vote is decisive).\n" + _IMPUTE_CAVEAT,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
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
    axes[1].set_title("kNN Router (real jina) — Neighborhood Purity by Selected Model")
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

    # Explain both panels + the 0 bars: a model the proxy (almost) never selects has
    # no neighbourhoods of its own, so its purity is 0 — a real consequence of the
    # allocation, not missing data. "Good" = bars near 1.0 (tight, self-similar clusters).
    fig.suptitle(
        "kNN router (real jina embeddings) — do the router's model choices form clean clusters? "
        "(purity → 1.0 = clean; each x-axis bar is one enabled model, 0 = (almost) never selected)",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.02,
        "GOOD = purity bars near 1.0 (a selected model's neighbours pick the same model). "
        "A model at exactly 0 is a real result — the router never selected it, so it owns no "
        "neighbourhoods — not missing data.\n"
        "Task similarity is cosine over the REAL 768-d jina prompt embeddings (left panel).\n"
        + _IMPUTE_CAVEAT,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
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
    ax.set_title(f"kNN Router (real jina) — Model Allocation Across {n} Tasks (k={k_neighbors})")

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
    # Every enabled model is on the axis. A model at 0 is a REAL result — the proxy
    # never picked it — not missing data. Say so, or a 0 bar reads as a data gap.
    zero_models = [m for m, c in zip(models_order, model_counts, strict=True) if c == 0]
    zero_note = (
        "Each bar is one enabled model; a bar at 0 means the router never selected it "
        "(real result, not missing data).\n"
        "The router applies the product rule (cheapest model whose real-jina-neighbourhood "
        "pass-rate clears the threshold),\nso routine work lands on the cheap models and only "
        "the hard tail escalates to the frontier.\n"
        if zero_models
        else ""
    )
    fig.text(
        0.5,
        0.01,
        zero_note + _IMPUTE_CAVEAT,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
        style="italic",
    )
    fig.subplots_adjust(bottom=0.30 if zero_models else 0.20)
    fig.savefig(str(output_dir / "model_allocation.png"), dpi=150)
    print("Saved model_allocation.png")
    plt.close(fig)

    # 3b. Descriptive all-model view (strategy-agnostic, straight from results.csv).
    # Answers "how does EVERY model do", not "which model does the router pick".
    plot_descriptive_model_performance(results, models_order, output_dir)

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
    # A task whose selected model was never measured is a coverage gap — EXCLUDE it
    # from the kNN bar (cost + pass), never impute the $0/non-pass the phantom guard
    # already screens out of the frontier bar.
    knn_scored = [tid for tid in task_ids if selections[tid] in results[tid]]
    knn_excluded = n - len(knn_scored)
    n_knn = len(knn_scored)
    knn_total = sum(results[tid][selections[tid]].get("cost", 0) for tid in knn_scored)
    knn_pass = sum(1 for tid in knn_scored if results[tid][selections[tid]].get("pass", False))
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
        "kNN router\n(real jina)",
        f"frontier\n({frontier_model})",
    ]
    values = [cheap_cost, knn_total, frontier_cost]
    # The two flanking bars are single models — paint them their CANONICAL model
    # colour (same as every other figure). The middle bar is the proxy's MIXTURE
    # across models, not one model, so it is neutral grey — never a model's hue.
    bar_colors = [
        model_colors.get(cheapest_model, "#9E9E9E"),
        "#9E9E9E",
        model_colors.get(frontier_model, "#9E9E9E"),
    ]
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="black")
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
    knn_pass_pct = (knn_pass / n_knn * 100) if n_knn else 0.0
    if phantom_frontier:
        title_tail = f"pass {knn_pass}/{n_knn} ({knn_pass_pct:.0f}%) — frontier savings N/A"
    else:
        title_tail = (
            f"pass {knn_pass}/{n_knn} ({knn_pass_pct:.0f}%), saves {savings_pct:.0f}% vs frontier"
        )
    ax.set_title(f"kNN Router (real jina) — Cost Across {n} Tasks (k={k_neighbors})\n{title_tail}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    # Figure-level captions BELOW the axes: drawn inside the bars, they collided
    # with the value labels.
    fig.subplots_adjust(bottom=0.34 if phantom_frontier else 0.22)
    knn_excluded_note = (
        f"\nkNN bar EXCLUDES {knn_excluded} coverage-gap task(s) (selected model unmeasured) "
        f"— its cost/pass sum over {n_knn} of {n}, never imputed $0."
        if knn_excluded
        else ""
    )
    fig.text(
        0.5,
        0.02,
        "Note: the middle bar sums each selected model's default-arm cost per task; "
        "strategy_comparison.png\nruns the live kNN engine (per-arm), so exact totals "
        "differ — compare shapes, not cents." + knn_excluded_note,
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
        f"kNN router (real jina) ${knn_total:.4f}, "
        f"frontier ${frontier_cost:.4f} ({frontier_covered}/{n} tasks); "
        f"router pass {knn_pass}/{n_knn} (excluded {knn_excluded} coverage-gap), {savings_str}"
    )


if __name__ == "__main__":
    main()
