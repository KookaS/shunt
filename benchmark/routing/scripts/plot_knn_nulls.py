#!/usr/bin/env python3
"""The two figures a degenerate router cannot fake: transfer-vs-k, and cross-repo transfer."""

# Every other kNN figure in this directory is satisfiable by a constant function. These
# two are not: the first asks whether routing survives holding the task out, the second
# whether it survives moving to another repository. Both carry an outcome-permutation band,
# so "the router learned something" is a claim the picture can refute.

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from benchmark import config, plot_frame  # noqa: E402
from benchmark.plot_frame import Annotations, FigureSpec  # noqa: E402
from benchmark.routing import summary  # noqa: E402
from benchmark.routing.scripts import knn_nulls, viz_knn  # noqa: E402

_INK = "#22252a"
_LOO = "#0072B2"
_CEILING = "#D55E00"
_CONSTANT = "#009E73"
_NULL = "#9aa0a6"

_TRANSFER_SPEC = FigureSpec(
    reading=(
        "x is k, the number of past tasks the router votes over. y is the share of tasks "
        "whose ROUTED model actually passed. The solid blue line holds each task OUT of its "
        "own index (what a deployed router can do); the dashed orange line lets the task see "
        "itself (pure memorisation); the green line is the best single "
        "always-one-model policy, which needs no router at all. The grey band is the "
        "central 95% of the same routing rule run on outcome-shuffled data."
    ),
    goal=(
        "The ONLY claim this figure can support is the blue line sitting ABOVE the grey band "
        "AND above the green constant line. Blue inside the band means the router scores what "
        "it would score if task descriptions carried no information about outcomes."
    ),
    definitions=(
        ("leave-one-task-out", "the task being routed is removed from the neighbour index"),
        ("memorisation reference", "the same rule with the task left IN its own index"),
        (
            "shuffled-outcome null",
            "outcome rows reassigned to tasks at random, preserving each model's own "
            "pass rate and breaking only the description->outcome link",
        ),
    ),
    limitations=(
        "Pass labels come from the coverage-completed matrix, in which every imputed cell is "
        "filled pass=True — so all four series, null included, sit above what measurement "
        "alone supports. The COMPARISON between them is the readable part, not the level.",
        "One workload (SWE-bench-style tasks over 12 repositories); transfer to a different "
        "task distribution is not evidence this figure can give.",
    ),
)

_CROSS_REPO_SPEC = FigureSpec(
    reading=(
        "Both axes are source repositories. A cell is the share of the ROW repo's tasks that "
        "passed when the router could only vote over the COLUMN repo's tasks. The diagonal is "
        "same-repo routing (the router has seen sibling tasks); everything off it is transfer "
        "to a repository the index has never seen. Brighter is a higher pass rate."
    ),
    goal=(
        "A router that learned something generalisable looks FLAT: off-diagonal cells as "
        "bright as the diagonal. A bright diagonal against a dull off-diagonal is "
        "memorisation of repo-local outcomes, not routing skill."
    ),
    definitions=(
        ("index repo", "the only tasks the router is allowed to vote over (the column)"),
        ("query repo", "the tasks being routed (the row)"),
    ),
    limitations=(
        "Repos differ in intrinsic difficulty, so a row's overall brightness mixes 'this repo "
        "is easy' with 'the router did well here' — read ACROSS a row, never down a column.",
        "Pass labels come from the coverage-completed matrix, whose imputed cells are all "
        "pass=True; the diagonal/off-diagonal contrast is the signal, not the absolute level.",
    ),
)


def _verdict(observed: float, band: knn_nulls.Band, what: str) -> str:
    """State the null comparison plainly — including, especially, when it is negative."""
    if band.contains(observed):
        return (
            f"NULL RESULT: {what} is {observed:.4f}, INSIDE the shuffled-outcome null band "
            f"[{band.lo:.4f}, {band.hi:.4f}] (null mean {band.mean:.4f}, "
            f"z={band.z(observed):+.2f}, {band.n} permutations) — the router scores what "
            f"chance scores"
        )
    direction = "above" if observed > band.hi else "BELOW"
    return (
        f"{what} is {observed:.4f}, {direction} the shuffled-outcome null band "
        f"[{band.lo:.4f}, {band.hi:.4f}] (z={band.z(observed):+.2f}, {band.n} permutations)"
    )


def plot_transfer_curve(curve: knn_nulls.TransferCurve, out_path: Path) -> Path:
    """Routing pass-rate vs k against memorisation, the best constant, and the null band."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ks = np.array(curve.ks, dtype=float)

    ax.fill_between(
        ks,
        curve.null_lo,
        curve.null_hi,
        color=_NULL,
        alpha=0.35,
        zorder=1,
        label=f"shuffled-outcome null, central 95% ({curve.n_perm} permutations)",
    )
    ax.plot(ks, curve.null_mean, color=_NULL, linestyle=(0, (1, 2)), linewidth=1.2, zorder=2)
    ax.plot(
        ks,
        curve.memorisation,
        color=_CEILING,
        linestyle="--",
        marker="^",
        markersize=5,
        linewidth=1.6,
        zorder=3,
        label="memorisation reference (task left in its own index)",
    )
    ax.axhline(
        curve.best_constant,
        color=_CONSTANT,
        linestyle="-.",
        linewidth=1.6,
        zorder=3,
        label=f"best constant policy — always {curve.best_constant_model} "
        f"({curve.best_constant:.3f})",
    )
    ax.plot(
        ks,
        curve.loo,
        color=_LOO,
        marker="o",
        markersize=5,
        linewidth=2.2,
        zorder=5,
        label="kNN router, leave-one-task-out (deployable)",
    )

    ax.set_xscale("log")
    ax.set_xticks(list(curve.ks))
    ax.set_xticklabels([str(k) for k in curve.ks])
    ax.minorticks_off()
    ax.set_xlabel("k — number of past tasks the router votes over (log scale)", color=_INK)
    ax.set_ylabel("Share of routed tasks that passed", color=_INK)
    ax.set_title(
        "Does the kNN router transfer, or memorise?  Routing pass-rate vs k\n"
        f"{curve.n_tasks} tasks · real jina embeddings · leave-one-task-out against three "
        "reference lines",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    # Upper-right: the memorisation line falls away to the left, so the top-right quadrant
    # is the only region no series occupies.
    ax.legend(fontsize=8, loc="upper right", framealpha=0.93)
    fig.tight_layout()

    best_i = int(np.argmax(curve.loo))
    best_k = curve.ks[best_i]
    # The blue line's best point is SELECTED over every k, so it must be judged against the
    # null of the SAME max-over-k search — not against one k's own band, which ignores the
    # search and roughly doubles the false-positive rate. `band_at` carries the exact
    # permutation sd rather than reconstructing it from the band width (a normal-shaped
    # assumption that runs ~12% small on this discrete statistic, inflating z upward).
    gap = curve.loo[best_i] - curve.best_constant
    notes = [
        _verdict(
            curve.loo[best_i],
            curve.max_null,
            f"the best-over-k leave-one-out pass rate (k={best_k}, selection-corrected)",
        ),
        _verdict(curve.loo[best_i], curve.band_at(best_i), f"the same value at its own k={best_k}"),
        f"Against the best constant policy (always {curve.best_constant_model}) the router is "
        f"{gap:+.4f} — a router that cannot beat one fixed model is not routing.",
        f"Memorisation reference at k={best_k} is {curve.memorisation[best_i]:.4f}; the "
        f"reference minus leave-one-out gap "
        f"({curve.memorisation[best_i] - curve.loo[best_i]:+.4f}) is how much of the score "
        f"comes from the task seeing itself.",
    ]
    return plot_frame.save(fig, out_path, _TRANSFER_SPEC, extra=Annotations(notes=tuple(notes)))


def plot_cross_repo(cross: knn_nulls.CrossRepo, k: int, out_path: Path) -> Path:
    """Query-repo x index-repo routing pass-rate, with the diagonal ringed."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    n = len(cross.repos)
    im = ax.imshow(cross.grid, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Share of the row repo's tasks that passed under the routed model")

    labels = [f"{r}\n(n={c})" for r, c in zip(cross.repos, cross.counts, strict=True)]
    ax.set_xticks(range(n))
    ax.set_xticklabels(cross.repos, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Index repo — the ONLY tasks the router may vote over")
    ax.set_ylabel("Query repo — the tasks being routed")

    midpoint = float(np.nanmean(cross.grid))
    for qi in range(n):
        for ii in range(n):
            val = cross.grid[qi, ii]
            if np.isnan(val):
                continue
            ax.text(
                ii,
                qi,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if val > midpoint else "white",
                fontweight="bold" if qi == ii else "normal",
            )
    for i in range(n):
        ax.add_patch(
            Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#d03b3b", linewidth=1.8)
        )

    ax.set_title(
        "Cross-repo transfer — does the router work on a repository its index has never seen?\n"
        f"k={k} · red rings mark same-repo (diagonal) routing",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()

    notes = [
        f"Same-repo (diagonal) mean {cross.diagonal_mean:.4f} vs cross-repo (off-diagonal) mean "
        f"{cross.off_diagonal_mean:.4f} — a {cross.advantage:+.4f} diagonal advantage. A "
        "positive gap is repo-local memorisation; a gap near zero means the diagonal carries "
        "no extra information.",
        _verdict(
            cross.advantage, cross.null, "the diagonal advantage (same-repo minus cross-repo)"
        ),
        "Read a uniformly bright or uniformly dull COLUMN as an index-size artefact rather "
        "than as skill: a small index gives every model the same thin neighbourhood, so the "
        "rule returns nearly the same pick for every query and that column reports one "
        "model's own pass rate.",
        f"Repos with fewer than the minimum task count are omitted; {n} repos "
        f"({sum(cross.counts)} tasks) are shown.",
    ]
    return plot_frame.save(fig, out_path, _CROSS_REPO_SPEC, extra=Annotations(notes=tuple(notes)))


def _pass_matrix(results: dict, task_ids: list[str], models_by_price: list[str]) -> np.ndarray:
    """(n_tasks, n_models) 0/1 pass matrix, columns in ascending price order."""
    mat = np.zeros((len(task_ids), len(models_by_price)), dtype=float)
    for i, tid in enumerate(task_ids):
        cells = results.get(tid, {})
        for j, model in enumerate(models_by_price):
            mat[i, j] = 1.0 if cells.get(model, {}).get("pass", False) else 0.0
    return mat


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
    """Render the transfer-curve and cross-repo-transfer figures."""
    config.load(config_path)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--output-dir", default="benchmark/routing/reports")
    ap.add_argument("--permutations", type=int, default=knn_nulls.DEFAULT_PERMUTATIONS)
    args = ap.parse_args()
    if args.config != config_path:
        config.load(args.config)

    matrix_path = Path(args.matrix) if args.matrix else config.challenges_path()
    matrix = summary.load_scored_matrix(matrix_path)
    results = matrix.get("results", {})
    if not results:
        print(
            "No results yet — results.csv holds no rows. "
            "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
        )
        return

    task_ids = sorted(results)
    models = config.enabled_models() or list(matrix.get("models", {}).keys())
    pricing = config.enabled_pricing()
    models_by_price = sorted(models, key=lambda m: config.cost_per_1m(m, pricing))
    threshold = float(config.knn_params().get("success_rate_threshold", 0.6))
    k_cfg = int(config.knn_params().get("k", 10))

    emb = viz_knn.build_task_embeddings(matrix, task_ids)
    sims = emb @ emb.T
    pass_mat = _pass_matrix(results, task_ids, models_by_price)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = [k for k in (2, 5, 10, 20, 40, 80, len(task_ids) - 1) if 1 <= k <= len(task_ids) - 1]
    ks = sorted(set(ks))
    curve = knn_nulls.transfer_curve(
        sims, pass_mat, models_by_price, ks, threshold, n_perm=args.permutations
    )
    plot_transfer_curve(curve, out_dir / "knn_transfer_curve.png")
    print("Saved knn_transfer_curve.png")

    cross = knn_nulls.cross_repo_transfer(
        sims, pass_mat, task_ids, k_cfg, threshold, n_perm=args.permutations
    )
    plot_cross_repo(cross, k_cfg, out_dir / "knn_cross_repo_transfer.png")
    print("Saved knn_cross_repo_transfer.png")

    for k, loo, ceil_, lo, hi in zip(
        curve.ks, curve.loo, curve.memorisation, curve.null_lo, curve.null_hi, strict=True
    ):
        print(f"  k={k:>4}  LOO={loo:.4f}  ceiling={ceil_:.4f}  null95=[{lo:.4f}, {hi:.4f}]")
    print(
        f"  best constant: always {curve.best_constant_model} = {curve.best_constant:.4f}; "
        f"cross-repo diagonal {cross.diagonal_mean:.4f} vs off-diagonal "
        f"{cross.off_diagonal_mean:.4f}"
    )


if __name__ == "__main__":
    main()
