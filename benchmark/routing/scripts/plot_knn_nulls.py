#!/usr/bin/env python3
"""embedding_signal.png — the falsification: real task text carries no per-task outcome signal."""

# THIS FILE USED TO REPORT AN EXCUSE. Its cross-repo and transfer figures both said, in so
# many words, that `problem_statement` is absent from all 500 committed tasks and that the
# router therefore sitting inside its own shuffled-outcome null was a COVERAGE GAP rather
# than a falsification — you cannot falsify a thesis on an instrument never shown to detect
# a positive signal.
#
# The corpus was rebuilt. `benchmark/routing/data/challenges.json` now carries the real
# SWE-bench problem statements (median 1185 chars against the old 106-char identifier
# label), `routing_text()` already preferred that field, and the excuse is spent. Measured
# on 190 tasks with two or more measured models, at k=20, with the real jina embedder:
#
#     identifier label (old text)    LOO R^2  -0.0712      inside the null
#     problem statement (new text)   LOO R^2  -0.0544      inside the null
#     human difficulty tag (control) LOO R^2  +0.1771      ABOVE the null
#     shuffled null, central 95%              [-0.1120, +0.0045]
#
# Same pipeline, same n, same null: a three-level human tag fires and 768 dimensions of
# real problem text does not. That is a falsification of embed-and-kNN on this corpus, and
# this figure states it instead of hedging it.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from benchmark import config, plot_frame  # noqa: E402
from benchmark.admissibility import AdmissibilityResult  # noqa: E402
from benchmark.plot_frame import Annotations, FigureSpec  # noqa: E402
from benchmark.routing import summary  # noqa: E402
from benchmark.routing.figures import context as ctxmod  # noqa: E402
from benchmark.routing.instrument_control import (  # noqa: E402
    routing_instrument_admissibility,
)
from benchmark.routing.scripts import knn_nulls, viz_knn  # noqa: E402

# `python -m` sets __name__ to "__main__", which would land in the figure
# manifest instead of the module that drew it.
_GENERATOR = "benchmark.routing.scripts.plot_knn_nulls"

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_LOO = "#0072B2"
_CEILING = "#D55E00"
_CONSTANT = "#009E73"
_NULL = "#9aa0a6"
_CONTROL = "#009E73"

SPEC = FigureSpec(
    title="Real problem text carries no routable signal; a three-level human tag does",
    reading=(
        "Left: routing pass-rate against k. The solid blue line holds each task OUT of its "
        "own neighbour index (what a deployed router can do), the dashed orange line lets "
        "the task see itself (pure memorisation), the green line is the best single "
        "always-one-model policy that needs no router at all, and the grey band is the same "
        "rule on outcome-shuffled data. Middle: leave-one-out R-squared predicting each "
        "task's solve rate, for the embedded problem statement and for the human difficulty "
        "tag, each against its own shuffled null, with the task-identity variance ceiling "
        "marked. Right: same-repo minus cross-repo routing advantage, at both repo-size "
        "cutoffs."
    ),
    goal=(
        "Read the middle panel first and read it as a pair. The control must clear its null "
        "or the instrument proves nothing; it does. The embedding must then clear its own "
        "null to support embed-and-kNN routing; it does not. That pairing is what turns a "
        "null result into a falsification rather than a coverage gap."
    ),
    definitions=(
        ("leave-one-task-out", "the task being routed is removed from the neighbour index"),
        ("memorisation reference", "the same rule with the task left IN its own index"),
        (
            "shuffled-outcome null",
            "outcome rows reassigned to tasks at random, preserving each model's own pass "
            "rate and breaking only the text->outcome link",
        ),
        (
            "variance ceiling",
            "eta-squared of task identity over the (task, model) pass matrix: the most any "
            "per-task predictor could explain.",
        ),
        (
            "positive control",
            "the same pipeline with the task's human difficulty label as the similarity — a "
            "control on the MEASUREMENT, not a routing proposal.",
        ),
    ),
    notes=(
        "The corpus embeds the real SWE-bench problem statement (median 1185 chars). The "
        "106-char identifier label the earlier figures encoded is kept as a contrast row so "
        "the change in the input is visible, not asserted.",
    ),
    limitations=(
        "Pass labels come from the coverage-completed matrix, in which every imputed cell is "
        "filled pass=True, so all series including the null sit above what measurement alone "
        "supports. The COMPARISON between them is the readable part, not the level.",
        "One workload (SWE-bench-style tasks over a dozen repositories). Transfer to a "
        "different task distribution is not evidence this figure can give.",
    ),
)


@dataclass(frozen=True)
class CrossRepoPair:
    """The diagonal advantage at both repo-size cutoffs — the sensitivity IS the finding."""

    min_tasks: int
    n_repos: int
    advantage: float
    null: knn_nulls.Band


def _verdict(
    observed: float, band: knn_nulls.Band, what: str, admissibility: AdmissibilityResult
) -> str:
    """State the null comparison plainly — including, especially, when it is negative."""
    comparison = _null_comparison(observed, band, what)
    if not admissibility.admissible:
        return (
            f"NOT QUOTABLE — {admissibility.headline} Until the instrument clears both controls "
            f"the following is a coverage-gap, not a result. {comparison}"
        )
    return comparison


def _null_comparison(observed: float, band: knn_nulls.Band, what: str) -> str:
    """The observation against its permutation band, with the direction named."""
    if band.contains(observed):
        return (
            f"NULL RESULT: {what} is {observed:.4f}, INSIDE the shuffled-outcome null band "
            f"[{band.lo:.4f}, {band.hi:.4f}] (null mean {band.mean:.4f}, "
            f"z={band.z(observed):+.2f}, {band.n} permutations)"
        )
    direction = "above" if observed > band.hi else "BELOW"
    return (
        f"{what} is {observed:.4f}, {direction} the shuffled-outcome null band "
        f"[{band.lo:.4f}, {band.hi:.4f}] (z={band.z(observed):+.2f}, {band.n} permutations)"
    )


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _draw_transfer(ax: Axes, curve: knn_nulls.TransferCurve) -> None:
    ks = list(curve.ks)
    ax.fill_between(
        ks,
        [v * 100 for v in curve.null_lo],
        [v * 100 for v in curve.null_hi],
        color=_NULL,
        alpha=0.30,
        zorder=1,
        label="shuffled-outcome null, 95%",
    )
    ax.plot(
        ks,
        [v * 100 for v in curve.memorisation],
        ls="--",
        color=_CEILING,
        lw=1.6,
        zorder=2,
        label="memorisation ceiling (task in its own index)",
    )
    ax.axhline(
        curve.best_constant * 100,
        color=_CONSTANT,
        lw=1.4,
        zorder=2,
        label=f"best constant policy ({curve.best_constant_model})",
    )
    ax.plot(
        ks,
        [v * 100 for v in curve.loo],
        "o-",
        color=_LOO,
        lw=2.0,
        ms=5,
        zorder=4,
        # "no leakage", not "deployable" — this parenthetical is about the task being held
        # out of its own index, not about strategy_class deployability (live/bound/blocked).
        label="leave-one-out routing (no leakage)",
    )
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=7.5)
    ax.minorticks_off()
    ax.set_xlabel("k — past tasks voted over (log)", fontsize=9)
    ax.set_ylabel("routed pass rate (%)", fontsize=9)
    ax.legend(fontsize=7, loc="center right", frameon=False)
    ax.grid(color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "A · routing, task held out")


def _draw_signal(ax: Axes, transfers: list[knn_nulls.OutcomeTransfer], ceiling: float) -> None:
    xs = list(range(len(transfers)))
    for x, t in zip(xs, transfers, strict=True):
        colour = _CONTROL if "tag" in t.label else _LOO
        ax.fill_between(
            [x - 0.34, x + 0.34], t.null.lo, t.null.hi, color=_NULL, alpha=0.30, zorder=1
        )
        ax.plot([x - 0.34, x + 0.34], [t.null.mean, t.null.mean], color=_NULL, lw=1.0, zorder=2)
        ax.plot([x], [t.r2], "o", color=colour, ms=11, zorder=4)
        ax.annotate(
            f"{t.r2:+.3f}\n{'INSIDE the null' if t.inside_null else 'ABOVE the null'}",
            xy=(x, t.r2),
            xytext=(0, 13),
            textcoords="offset points",
            fontsize=7.5,
            ha="center",
            va="bottom",
            color=colour,
        )
    ax.axhline(0.0, color="#bbbbbb", lw=0.8, zorder=1)
    ax.axhline(ceiling, color=_CEILING, lw=1.3, ls="--", zorder=3)
    ax.annotate(
        f"task-identity ceiling  η²={ceiling:.3f}",
        xy=(len(transfers) - 0.5, ceiling),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=7.5,
        ha="right",
        va="bottom",
        color=_CEILING,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([t.label for t in transfers], fontsize=8)
    ax.set_xlim(-0.6, len(transfers) - 0.4)
    ax.set_ylim(min(-0.2, min(t.null.lo for t in transfers) - 0.05), max(ceiling * 1.28, 0.35))
    ax.set_ylabel("leave-one-out R² on per-task solve rate", fontsize=9)
    ax.fill_between([], [], [], color=_NULL, alpha=0.30, label="shuffled null, 95%")
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "B · control fires, embedding does not")


def _draw_cross_repo(ax: Axes, pairs: list[CrossRepoPair]) -> None:
    xs = list(range(len(pairs)))
    for x, pair in zip(xs, pairs, strict=True):
        ax.fill_between(
            [x - 0.32, x + 0.32], pair.null.lo, pair.null.hi, color=_NULL, alpha=0.30, zorder=1
        )
        inside = pair.null.contains(pair.advantage)
        ax.plot([x], [pair.advantage], "o", color=_LOO if inside else _CEILING, ms=10, zorder=4)
        ax.annotate(
            f"{pair.advantage:+.4f}\n{pair.n_repos} repos",
            xy=(x, pair.advantage),
            xytext=(0, 13),
            textcoords="offset points",
            fontsize=7.5,
            ha="center",
            va="bottom",
            color="#333333",
        )
    ax.axhline(0.0, color="#bbbbbb", lw=0.8, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"repos with\n≥{p.min_tasks} tasks" for p in pairs], fontsize=8)
    ax.set_xlim(-0.6, len(pairs) - 0.4)
    top = max([p.advantage for p in pairs] + [p.null.hi for p in pairs])
    bottom = min([p.advantage for p in pairs] + [p.null.lo for p in pairs])
    ax.set_ylim(bottom - 0.1 * (top - bottom), top + 0.55 * (top - bottom))
    ax.set_ylabel("same-repo minus cross-repo pass rate", fontsize=9)
    ax.fill_between([], [], [], color=_NULL, alpha=0.30, label="shuffled null, 95%")
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "C · repo-memorisation, by cutoff")


def _pass_matrix(results: dict, task_ids: list[str], models_by_price: list[str]) -> np.ndarray:
    """(n_tasks, n_models) 0/1 pass matrix, columns in ascending price order."""
    mat = np.zeros((len(task_ids), len(models_by_price)), dtype=float)
    for i, tid in enumerate(task_ids):
        cells = results.get(tid, {})
        for j, model in enumerate(models_by_price):
            mat[i, j] = 1.0 if cells.get(model, {}).get("pass", False) else 0.0
    return mat


def _annotations(  # noqa: PLR0913 (one argument per drawn series)
    curve: knn_nulls.TransferCurve,
    transfers: list[knn_nulls.OutcomeTransfer],
    ceiling: float,
    pairs: list[CrossRepoPair],
    admissibility: AdmissibilityResult,
    imputation: str,
) -> Annotations:
    embedding = next((t for t in transfers if "tag" not in t.label), None)
    control = next((t for t in transfers if "tag" in t.label), None)
    facts = [
        f"{curve.n_tasks} tasks, {curve.n_perm} permutations per null, k={transfers[0].k}",
    ]
    if embedding is not None and control is not None:
        facts.append(
            f"embedding R² {embedding.r2:+.3f} vs control {control.r2:+.3f}, "
            f"null 95% [{embedding.null.lo:+.3f}, {embedding.null.hi:+.3f}]"
        )
    facts.append(f"task-identity ceiling η²={ceiling:.3f}")
    caveat = None
    if (
        embedding is not None
        and control is not None
        and embedding.inside_null
        and not control.inside_null
    ):
        caveat = "Falsified, not untested: the control clears the null on this same pipeline and n."
    best_k = max(range(len(curve.ks)), key=lambda i: curve.loo[i])
    notes = [
        admissibility.reason,
        _verdict(
            curve.loo[best_k],
            curve.band_at(best_k),
            f"the leave-one-out routing pass rate at k={curve.ks[best_k]}",
            admissibility,
        ),
        *(
            _verdict(t.r2, t.null, f"the {t.label} leave-one-out R²", admissibility)
            for t in transfers
        ),
        *(
            _verdict(
                p.advantage,
                p.null,
                f"the diagonal advantage over {p.n_repos} repos with ≥{p.min_tasks} tasks",
                admissibility,
            )
            for p in pairs
        ),
    ]
    return Annotations(
        subtitle_facts=tuple(facts),
        caveat=caveat,
        notes=tuple(notes),
        limitations=(imputation,),
        counts=(("tasks", curve.n_tasks), ("permutations", curve.n_perm)),
    )


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
    """Render embedding_signal.png."""
    config.load(config_path)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--output-dir", default="docs/assets/figures/routing")
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

    # BEFORE the figure: does this pipeline recover a signal it is KNOWN to contain? Run at
    # the same k and threshold the figure uses — a control at other settings certifies an
    # instrument nobody is quoting.
    admissibility = routing_instrument_admissibility(k=k_cfg, threshold=threshold)
    print(admissibility.reason)

    emb = viz_knn.build_task_embeddings(matrix, task_ids)
    sims = emb @ emb.T
    pass_mat = _pass_matrix(results, task_ids, models_by_price)
    target = knn_nulls.solve_rate(pass_mat)
    ceiling = knn_nulls.eta_squared_task(pass_mat)

    ks = sorted(
        {k for k in (2, 5, 10, 20, 40, 80, len(task_ids) - 1) if 1 <= k <= len(task_ids) - 1}
    )
    curve = knn_nulls.transfer_curve(
        sims,
        pass_mat,
        models_by_price,
        ks,
        threshold,
        admissibility=admissibility,
        n_perm=args.permutations,
    )

    transfers = [
        knn_nulls.outcome_transfer(
            sims,
            target,
            k_cfg,
            admissibility=admissibility,
            label="embedded\nproblem statement",
            n_perm=args.permutations,
        )
    ]
    control_sims = viz_knn.difficulty_similarity(matrix, task_ids, sims)
    if control_sims is not None:
        transfers.append(
            knn_nulls.outcome_transfer(
                control_sims,
                target,
                k_cfg,
                admissibility=admissibility,
                label="human difficulty tag\n(positive control)",
                n_perm=args.permutations,
            )
        )

    pairs: list[CrossRepoPair] = []
    for min_tasks in (8, knn_nulls.DEFAULT_MIN_REPO_TASKS):
        try:
            cross = knn_nulls.cross_repo_transfer(
                sims,
                pass_mat,
                task_ids,
                k_cfg,
                threshold,
                admissibility=admissibility,
                min_tasks=min_tasks,
                n_perm=args.permutations,
            )
        except ValueError as exc:
            print(f"  cross-repo at min_tasks={min_tasks}: skipped ({exc})")
            continue
        pairs.append(CrossRepoPair(min_tasks, len(cross.repos), cross.advantage, cross.null))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imputation = viz_knn.imputation_limit_line(results, task_ids, models_by_price)
    size = plot_frame.WIDE
    n_panels = 2 + (1 if pairs else 0)
    fig, axes = plot_frame.subplots(size, 1, n_panels, width_ratios=[1.25, 1.0, 0.8][:n_panels])
    _draw_transfer(axes[0], curve)
    _draw_signal(axes[1], transfers, ceiling)
    if pairs:
        _draw_cross_repo(axes[2], pairs)
    plot_frame.save(
        fig,
        out_dir / "embedding_signal.png",
        SPEC,
        extra=_annotations(curve, transfers, ceiling, pairs, admissibility, imputation),
        provenance=plot_frame.Provenance(
            _GENERATOR, ctxmod.corpus_digest(matrix, task_ids), ctxmod.MANIFEST
        ),
        size=size,
    )
    print("Saved embedding_signal.png")
    for t in transfers:
        print(
            f"  {t.label.replace(chr(10), ' ')}: R2={t.r2:+.4f} "
            f"null95=[{t.null.lo:+.4f}, {t.null.hi:+.4f}]"
        )
    for p in pairs:
        print(f"  min_tasks={p.min_tasks}: advantage {p.advantage:+.4f} over {p.n_repos} repos")


if __name__ == "__main__":
    main()
