#!/usr/bin/env python3
"""Compare general (char n-gram) vs code-specific (word) embedding spaces.

Simulates two TF-IDF pipelines and plots kNN-neighbor Jaccard overlap and
optimal-model agreement at k=5/10 (reports/embedding_compare.png).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

from benchmark import config  # noqa: E402


def _code_keywords() -> list[str]:
    return [
        "implement",
        "write",
        "function",
        "class",
        "method",
        "interface",
        "type",
        "generic",
        "template",
        "async",
        "await",
        "promise",
        "callback",
        "closure",
        "lambda",
        "decorator",
        "context",
        "manager",
        "iterator",
        "generator",
        "thread",
        "lock",
        "mutex",
        "atomic",
        "concurrent",
        "parallel",
        "cache",
        "memoize",
        "buffer",
        "pool",
        "queue",
        "stack",
        "heap",
        "tree",
        "graph",
        "list",
        "map",
        "filter",
        "reduce",
        "sort",
        "search",
        "merge",
        "parse",
        "serialize",
        "deserialize",
        "encode",
        "decode",
        "compress",
        "encrypt",
        "hash",
        "index",
        "query",
        "join",
        "select",
        "insert",
        "update",
        "delete",
        "recursive",
        "iterate",
        "traverse",
        "validate",
        "normalize",
        "middleware",
        "handler",
        "router",
        "pipeline",
        "stream",
        "event",
        "observer",
        "pubsub",
        "singleton",
        "factory",
        "builder",
        "adapter",
        "proxy",
        "decorator",
        "composite",
        "strategy",
        "python",
        "typescript",
        "rust",
        "golang",
        "shell",
        "bash",
        "sql",
        "react",
        "hook",
        "http",
        "rest",
        "grpc",
        "websocket",
        "tcp",
        "udp",
        "array",
        "tuple",
        "record",
        "struct",
        "enum",
        "union",
        "optional",
        "error",
        "exception",
        "result",
        "option",
        "unwrap",
        "macro",
        "derive",
        "trait",
        "lifetime",
        "borrow",
        "ownership",
        "goroutine",
        "channel",
        "defer",
        "panic",
        "recover",
        "cte",
        "window",
        "partition",
        "lateral",
        "pivot",
        "funnel",
        "import",
        "export",
        "module",
        "package",
        "dependency",
        "test",
        "assert",
        "mock",
        "stub",
        "fixture",
        "coverage",
        "benchmark",
        "profile",
        "trace",
        "log",
        "monitor",
        "alert",
        "config",
        "env",
        "secret",
        "token",
        "auth",
        "session",
    ]


def _code_stop_words() -> list[str]:
    return [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "because",
        "but",
        "and",
        "or",
        "if",
        "while",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "which",
        "who",
        "whom",
        "what",
    ]


def load_matrix(path: Path) -> dict:
    return config.load_matrix(path)


def oracle_optimal_model(task_id: str, results_map: dict) -> str | None:
    task_results = results_map.get(task_id, {})
    candidates: list[tuple[str, float]] = []
    for model, outcome in task_results.items():
        if outcome.get("pass"):
            candidates.append((model, outcome.get("cost", 0.0)))
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[1])[0]


def jaccard_overlap(neighbors_a: list[int], neighbors_b: list[int]) -> float:
    set_a = set(neighbors_a)
    set_b = set(neighbors_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def build_feature_pipelines(
    task_descs: list[str],
    code_keywords: list[str] | None = None,
    code_stop_words: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if code_keywords is None:
        code_keywords = _code_keywords()
    if code_stop_words is None:
        code_stop_words = _code_stop_words()

    arctic_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=384,
        sublinear_tf=True,
    )
    arctic_feats = arctic_vec.fit_transform(task_descs).toarray()

    jina_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=1024,
        sublinear_tf=True,
        stop_words=code_stop_words,
        vocabulary=code_keywords,
    )
    try:
        jina_feats = jina_vec.fit_transform(task_descs).toarray()
    except ValueError:
        jina_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=1024,
            sublinear_tf=True,
            stop_words=code_stop_words,
        )
        jina_feats = jina_vec.fit_transform(task_descs).toarray()

    return arctic_feats, jina_feats


def compute_overlap(feats: np.ndarray, k: int) -> np.ndarray:
    n = feats.shape[0]
    # Only n-1 neighbors exist once self is excluded; clamp so a partially
    # populated matrix (n < k, e.g. 6 of 10 challenges run) does not crash on a
    # shape-mismatched row assignment.
    k = max(1, min(k, n - 1)) if n > 1 else min(k, n)
    neighbor_matrix = np.zeros((n, k), dtype=int)
    for i in range(n):
        sims = cosine_similarity(feats[i : i + 1], feats).flatten()
        sims[i] = -1.0
        neighbor_matrix[i] = np.argsort(sims)[::-1][:k]
    return neighbor_matrix


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
    config.load(config_path)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--plot", default="benchmark/routing/reports/embedding_compare.png")
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

    print("Building simulated embedding spaces...")
    arctic_feats, jina_feats = build_feature_pipelines(
        task_descs, _code_keywords(), _code_stop_words()
    )

    print(f"  Arctic (char n-grams, 384d): {arctic_feats.shape}")
    print(f"  Jina-code (word-level, {jina_feats.shape[1]}d): {jina_feats.shape}")

    overlap_stats: dict[int, dict[str, float]] = {}
    for k in (5, 10):
        print(f"\nComputing k={k} nearest neighbors...")
        arctic_neighbors = compute_overlap(arctic_feats, k)
        jina_neighbors = compute_overlap(jina_feats, k)

        overlaps = []
        for i in range(len(task_ids)):
            ov = jaccard_overlap(list(arctic_neighbors[i]), list(jina_neighbors[i]))
            overlaps.append(ov)

        mean_overlap = float(np.mean(overlaps))
        std_overlap = float(np.std(overlaps))
        min_overlap = float(np.min(overlaps))
        max_overlap = float(np.max(overlaps))

        n_high_overlap = sum(1 for o in overlaps if o >= 0.5)
        n_low_overlap = sum(1 for o in overlaps if o < 0.3)
        overlap_stats[k] = {
            "mean": mean_overlap,
            "std": std_overlap,
            "high": n_high_overlap,
            "low": n_low_overlap,
        }

        print(f"  k={k}:")
        print(f"    Mean neighbor overlap (Jaccard): {mean_overlap:.4f} \u00b1 {std_overlap:.4f}")
        print(f"    Range: [{min_overlap:.4f}, {max_overlap:.4f}]")
        hpct = 100 * n_high_overlap // len(task_ids)
        lpct = 100 * n_low_overlap // len(task_ids)
        print(f"    Tasks with >= 50% overlap: {n_high_overlap}/{len(task_ids)} ({hpct}%)")
        print(f"    Tasks with <  30% overlap: {n_low_overlap}/{len(task_ids)} ({lpct}%)")

    optimal_map = {}
    for tid in task_ids:
        opt = oracle_optimal_model(tid, results_map)
        if opt:
            optimal_map[tid] = opt

    print("\n=== OPTIMAL MODEL AGREEMENT AMONG NEIGHBORS ===")

    agreement_stats: dict[int, dict[str, int]] = {}
    for k in (5, 10):
        arctic_neighbors = compute_overlap(arctic_feats, k)
        jina_neighbors = compute_overlap(jina_feats, k)

        arctic_agreement = 0
        jina_agreement = 0
        both_agreement = 0

        for i, tid in enumerate(task_ids):
            opt = optimal_map.get(tid)
            if opt is None:
                continue

            a_models = set()
            j_models = set()
            for ni in arctic_neighbors[i]:
                ntid = task_ids[ni]
                nopt = optimal_map.get(ntid)
                if nopt:
                    a_models.add(nopt)
            for ni in jina_neighbors[i]:
                ntid = task_ids[ni]
                nopt = optimal_map.get(ntid)
                if nopt:
                    j_models.add(nopt)

            if opt in a_models:
                arctic_agreement += 1
            if opt in j_models:
                jina_agreement += 1
            if opt in a_models and opt in j_models:
                both_agreement += 1

        n_tasks = len(task_ids)
        agreement_stats[k] = {
            "arctic": arctic_agreement,
            "jina": jina_agreement,
            "both": both_agreement,
        }
        apct = 100 * arctic_agreement // n_tasks
        jpct = 100 * jina_agreement // n_tasks
        bpct = 100 * both_agreement // n_tasks
        print(f"  k={k}: Arctic optimal-in-neighbors: {arctic_agreement}/{n_tasks} ({apct}%)")
        print(f"  k={k}: Jina-code optimal-in-neighbors: {jina_agreement}/{n_tasks} ({jpct}%)")
        print(f"  k={k}: Both agree on optimal: {both_agreement}/{n_tasks} ({bpct}%)")

    plot_path = _plot_comparison(
        Path(args.plot), matrix_path, len(task_ids), overlap_stats, agreement_stats
    )
    print(f"\nPlot written to {plot_path}")


def _plot_comparison(
    plot_path: Path,
    matrix_path: Path,
    n_tasks: int,
    overlap_stats: dict[int, dict[str, float]],
    agreement_stats: dict[int, dict[str, int]],
) -> Path:
    """Two-panel figure: neighbor-set Jaccard overlap (left) and optimal-model
    agreement among neighbors (right), each split by k. Replaces the text report.
    """
    ks = sorted(overlap_stats.keys())
    x = np.arange(len(ks))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    means = [overlap_stats[k]["mean"] for k in ks]
    stds = [overlap_stats[k]["std"] for k in ks]
    bars = ax1.bar(x, means, yerr=stds, capsize=6, color="#4C72B0", edgecolor="black", width=0.5)
    for bar, m in zip(bars, means, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"k={k}" for k in ks])
    ax1.set_ylabel("Mean neighbor-set Jaccard (± std)")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Arctic vs Jina-code neighbor overlap")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_axisbelow(True)

    width = 0.25
    series = [
        ("Arctic", "arctic", "#4C72B0"),
        ("Jina-code", "jina", "#DD8452"),
        ("Both", "both", "#55A868"),
    ]
    for idx, (label, key, color) in enumerate(series):
        vals = [100 * agreement_stats[k][key] / n_tasks for k in ks]
        offset = (idx - 1) * width
        b = ax2.bar(x + offset, vals, width, label=label, color=color, edgecolor="black")
        for bar, v in zip(b, vals, strict=True):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{v:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"k={k}" for k in ks])
    ax2.set_ylabel("Optimal model in neighbors (% of tasks)")
    ax2.set_title("Optimal-model agreement among neighbors")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    fig.suptitle(
        f"Embedding comparison — {n_tasks} tasks ({matrix_path.name})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    return plot_path


if __name__ == "__main__":
    main()
