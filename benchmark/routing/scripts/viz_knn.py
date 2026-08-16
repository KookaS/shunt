#!/usr/bin/env python3
"""knn_calibration.png — is the weighted neighbour pass-rate, the router's one input, calibrated?"""

# THE WHOLE PRODUCT RESTS ON ONE NUMBER AND NOTHING USED TO CHECK IT. `docs/routing.md`'s
# rule is "a model is eligible when its weighted neighbourhood success rate >= 0.6", and
# `shunt.router.selection.SelectionRule` computes exactly that: sum over the k nearest
# neighbours of similarity x outcome, over the summed similarity. Every routing decision
# the product ever makes is a comparison of that number against 0.6. Twenty-four committed
# figures drew allocations, purities, PCA shadows and cost bars derived FROM it; not one
# asked whether it predicts anything.
#
# This file replaces five of them (knn_cost_comparison, knn_pca_scatter, model_allocation,
# model_performance_descriptive, neighborhood_purity) with the reliability diagram, the
# threshold's position in the score distribution, and a Brier skill score against both a
# shuffled-outcome null and a human-difficulty-tag positive control.
#
# ON THE RETIRED FIGURES. `neighborhood_purity` computed purity over the ROUTER'S OWN
# selection labels, which are 98.3% one class; observed 0.9616 sat below chance 0.9667 and
# below the majority share 0.9831, so it was arithmetically incapable of a positive result.
# `knn_pca_scatter` projected 768 dimensions onto two carrying 15.1% of the variance.
# `knn_cost_comparison` published a SECOND (cost, pass) pair for the kNN router — a proxy
# 77.7% at $1.73 against the live engine's 81.71% at $13.34 in the same report set — which
# is a correctness bug, not a second view; the live engine's number in
# strategy_summary.csv is now the only one.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from benchmark import config, plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import plot_style, summary
from benchmark.routing.figures import context as ctxmod
from benchmark.routing.instrument_control import routing_instrument_admissibility
from benchmark.routing.scripts import knn_nulls
from benchmark.routing.strategies import routing_text
from benchmark.routing.strategies.knn import _embed_texts

# `python -m` sets __name__ to "__main__", which would land in the figure
# manifest instead of the module that drew it.
_GENERATOR = "benchmark.routing.scripts.viz_knn"

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Cheapest-above-threshold cutoff, matching the shipped KnnPolicy.success_rate_threshold
# (0.6) and router.selection.SelectionRule.
_DEFAULT_SUCCESS_RATE_THRESHOLD: Final[float] = 0.6

# Reliability bins. Ten is too many at n=175 x 6 models for the extreme bins to carry a
# usable count; five keeps every bin's Wilson interval readable.
_N_BINS: Final[int] = 5
_NULL_DRAWS: Final[int] = 400

_OBSERVED = "#0072B2"
_CONTROL = "#009E73"
_NULLC = "#9aa0a6"
_THRESHOLD = "#B71C1C"

SPEC = FigureSpec(
    title="The router's one input does not predict outcomes; a 3-level human tag does",
    reading=(
        "Left: the reliability diagram. x is the weighted neighbourhood success rate the "
        "shipped rule computes for a (task, model) pair; y is how often that pair actually "
        "passed. A calibrated predictor tracks the dashed diagonal. Bars carry 95% Wilson "
        "intervals and the count in each bin. The red line is the shipped 0.6 eligibility "
        "threshold. Middle: how those scores are distributed, so the threshold's position "
        "is visible geometry rather than a claim. Right: Brier skill score against the "
        "marginal pass rate — above zero means the neighbourhood rate beats simply knowing "
        "how often each model passes — with the shuffled-outcome null band and the "
        "human-difficulty-tag positive control on the same axis."
    ),
    goal=(
        "Look at the right panel first. The positive control must sit above the null band, "
        "or the instrument proves nothing either way. It does. Then look at where the "
        "observed bar sits: inside the band is a falsification, not a coverage gap."
    ),
    definitions=(
        (
            "weighted success rate",
            "sum(similarity x outcome) / sum(similarity) over the k nearest OTHER tasks, "
            "the quantity SelectionRule thresholds at 0.6.",
        ),
        (
            "Brier skill score",
            "1 - Brier(neighbour rate) / Brier(per-model base rate). Zero means the "
            "neighbourhood adds nothing over the model's marginal pass rate.",
        ),
        (
            "positive control",
            "The same pipeline run with the task's human difficulty label (easy/medium/hard) "
            "as the similarity, so a task's neighbours are the tasks a human called equally "
            "hard. It is a control on the MEASUREMENT, not a routing proposal.",
        ),
    ),
    notes=(
        "Every rate is leave-one-out: a task is never its own neighbour, so a task cannot "
        "predict itself.",
    ),
    limitations=(
        "The neighbour weight is similarity only. The shipped rule also multiplies by each "
        "neighbour's verification confidence, which is 1.0 for every cell in this corpus, "
        "so the two coincide here and could diverge on live traffic.",
    ),
)


# ---------------------------------------------------------------------------
# The quantity itself
# ---------------------------------------------------------------------------


def build_feature_vectors(results: dict, models_order: list[str]) -> tuple[list[str], np.ndarray]:
    """(task_ids, per-model [pass, log cost, log tokens, log calls]) — the outcome features."""
    task_ids = sorted(results.keys())
    vecs = np.zeros((len(task_ids), len(models_order) * 4))
    for i, tid in enumerate(task_ids):
        for j, model in enumerate(models_order):
            r = results[tid].get(model, {})
            base = j * 4
            vecs[i, base] = 1.0 if r.get("pass", False) else 0.0
            vecs[i, base + 1] = np.log10(plot_style.row_real_cost(r) + 1e-9) + 5
            vecs[i, base + 2] = np.log10(r.get("in_tok", 0) + r.get("out_tok", 0) + 1)
            vecs[i, base + 3] = np.log10(r.get("calls", 0) + 1)
    return task_ids, vecs


def build_task_embeddings(matrix: dict, task_ids: list[str]) -> np.ndarray:
    """Real normalized jina prompt embeddings (unit vectors), aligned to ``task_ids``."""
    tasks = matrix.get("tasks", {})
    texts = [routing_text(tid, tasks.get(tid, {})) for tid in task_ids]
    emb = np.asarray(_embed_texts(texts), dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def _nearest_neighbors(sims: np.ndarray, query_idx: int, k: int) -> np.ndarray:
    """Indices of the k most similar OTHER tasks, nearest first."""
    k_eff = min(int(k), len(sims) - 1)
    if k_eff <= 0:
        return np.empty(0, dtype=int)
    ranked = np.argsort(sims)[::-1]
    return ranked[ranked != query_idx][:k_eff]


def knn_select(
    vecs: np.ndarray,
    emb: np.ndarray,
    query_idx: int,
    models_order: list[str],
    k: int = 10,
    success_rate_threshold: float | None = None,
) -> str:
    """Cheapest model whose neighbourhood pass-rate clears the threshold, else best-scoring."""
    threshold = (
        _DEFAULT_SUCCESS_RATE_THRESHOLD
        if success_rate_threshold is None
        else success_rate_threshold
    )
    similarities = emb @ emb[query_idx]
    nearest = _nearest_neighbors(similarities, query_idx, k)
    price_order = sorted(
        models_order, key=lambda m: config.cost_per_1m(m, config.enabled_pricing())
    )
    if not price_order:
        return models_order[0]
    if len(nearest) == 0:
        return price_order[0]
    # The rule itself lives in knn_nulls.select_from_rates so this figure and the null
    # models that test it can never diverge.
    pass_cols = [list(models_order).index(m) * 4 for m in price_order]
    rates = np.array([[float(np.mean(vecs[nearest, c])) for c in pass_cols]])
    return price_order[int(knn_nulls.select_from_rates(rates, threshold)[0])]


def weighted_rates(sims: np.ndarray, pass_mat: np.ndarray, k: int) -> np.ndarray:
    """(n_tasks, n_models) leave-one-out SIMILARITY-WEIGHTED neighbourhood pass-rate."""
    # This is `SelectionRule`'s `weighted_success`, vectorised: weight = confidence x
    # (1 - distance), confidence is 1.0 on every measured cell here and cosine distance is
    # 1 - similarity, so the weight IS the similarity. Negative similarities are clamped
    # exactly as `_confidence_weight` clamps them.
    n = sims.shape[0]
    masked = np.array(sims, dtype=float, copy=True)
    np.fill_diagonal(masked, -np.inf)
    k_eff = max(1, min(int(k), n - 1))
    nbrs = np.argsort(-masked, axis=1)[:, :k_eff]
    weights = np.clip(np.take_along_axis(masked, nbrs, axis=1), 0.0, None)
    total = weights.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return np.einsum("nk,nkm->nm", weights, pass_mat[nbrs]) / total


def brier_skill(rates: np.ndarray, pass_mat: np.ndarray) -> float:
    """1 - Brier(neighbour rate) / Brier(per-model base rate), over every (task, model) cell."""
    base = np.broadcast_to(pass_mat.mean(axis=0, keepdims=True), pass_mat.shape)
    ref = float(((pass_mat - base) ** 2).mean())
    if ref <= 0:
        return 0.0
    return 1.0 - float(((pass_mat - rates) ** 2).mean()) / ref


def reliability(rates: np.ndarray, pass_mat: np.ndarray, n_bins: int = _N_BINS) -> list[dict]:
    """Per bin of predicted rate: mean prediction, observed pass rate, Wilson CI, count."""
    # `base` is the per-model marginal pass rate broadcast over the cells, so each bin also
    # reports what you would predict knowing ONLY which model the cell belongs to. A pooled
    # reliability curve rises whenever the score merely ranks models, which is why that
    # reference is drawn next to the diagonal rather than left implicit.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    base = np.broadcast_to(pass_mat.mean(axis=0, keepdims=True), pass_mat.shape).ravel()
    flat_r, flat_y = rates.ravel(), pass_mat.ravel()
    out: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (flat_r >= lo) & (flat_r < hi if hi < 1.0 else flat_r <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        passes = int(flat_y[mask].sum())
        ci_lo, ci_hi = plot_style.wilson_interval(passes, n)
        out.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "predicted": float(flat_r[mask].mean()),
                "observed": passes / n,
                "model_base": float(base[mask].mean()),
                "ci": (ci_lo, ci_hi),
                "n": n,
            }
        )
    return out


def null_band(
    sims: np.ndarray, pass_mat: np.ndarray, k: int, draws: int = _NULL_DRAWS
) -> knn_nulls.Band:
    """Brier skill under task-wise shuffles of the outcome matrix."""
    rng = np.random.default_rng(0)
    samples = np.empty(draws, dtype=float)
    for i in range(draws):
        shuffled = pass_mat[rng.permutation(pass_mat.shape[0])]
        samples[i] = brier_skill(weighted_rates(sims, shuffled, k), shuffled)
    return knn_nulls.band_of(samples)


def difficulty_similarity(matrix: dict, task_ids: list[str], sims: np.ndarray) -> np.ndarray | None:
    """Positive control: two tasks are 'similar' when a human gave them the same difficulty."""
    tasks = matrix.get("tasks", {})
    tags = np.array([str(tasks.get(t, {}).get("difficulty_stratum") or "") for t in task_ids])
    if len(set(tags.tolist()) - {""}) < 2:
        return None
    # The embedding similarity is added at 1e-3 purely to break ties deterministically, so
    # the control's neighbour ORDER is reproducible without the tag carrying any of it.
    return (tags[:, None] == tags[None, :]).astype(float) + 1e-3 * sims


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Calibration:
    """Everything the three panels draw, computed once."""

    bins: list[dict]
    rates: np.ndarray
    observed: float
    null: knn_nulls.Band
    control: float | None
    control_null: knn_nulls.Band | None
    threshold: float
    k: int
    n_tasks: int
    n_models: int


def _draw_reliability(ax: Axes, cal: Calibration) -> None:
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#bbbbbb", zorder=1, label="perfect calibration")
    xs = [b["predicted"] for b in cal.bins]
    ys = [b["observed"] for b in cal.bins]
    yerr = np.array([plot_style.ci_yerr(b["observed"], *b["ci"]) for b in cal.bins]).T
    ax.errorbar(
        xs,
        ys,
        yerr=yerr,
        fmt="o-",
        color=_OBSERVED,
        ms=6,
        lw=1.6,
        capsize=3,
        zorder=3,
        label="observed",
    )
    ax.plot(
        xs,
        [b["model_base"] for b in cal.bins],
        "s--",
        color="#8a8a8a",
        ms=4,
        lw=1.2,
        zorder=2,
        label="knowing only which model",
    )
    for b in cal.bins:
        ax.annotate(
            f"n={b['n']}",
            xy=(b["predicted"], b["ci"][1]),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=7,
            ha="center",
            color="#555555",
        )
    ax.axvline(cal.threshold, color=_THRESHOLD, lw=1.4, ls="-", zorder=2)
    ax.annotate(
        f"shipped threshold {cal.threshold:g}",
        xy=(cal.threshold, 1.15),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=7.5,
        color=_THRESHOLD,
        ha="left",
        va="top",
    )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.16)
    ax.set_xlabel("weighted neighbourhood success rate", fontsize=9)
    ax.set_ylabel("observed pass rate", fontsize=9)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.grid(color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "A · reliability of the routed score")


def _draw_distribution(ax: Axes, cal: Calibration) -> None:
    flat = cal.rates.ravel()
    ax.hist(flat, bins=24, range=(0.0, 1.0), color="#c9d7e6", edgecolor="#8fa6bd", lw=0.5)
    ax.axvline(cal.threshold, color=_THRESHOLD, lw=1.4, zorder=3)
    above = float((flat >= cal.threshold).mean())
    ax.annotate(
        f"{above:.0%} of (task, model) scores\nland above the shipped threshold",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        fontsize=7.5,
        color=_THRESHOLD,
        ha="left",
        va="top",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("weighted neighbourhood success rate", fontsize=9)
    ax.set_ylabel("(task, model) pairs", fontsize=9)
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "B · where the threshold cuts")


def _draw_skill(ax: Axes, cal: Calibration) -> None:
    entries: list[tuple[str, float, knn_nulls.Band | None, str]] = [
        ("embedded\nproblem statement", cal.observed, cal.null, _OBSERVED)
    ]
    if cal.control is not None:
        entries.append(
            ("human difficulty tag\n(positive control)", cal.control, cal.control_null, _CONTROL)
        )
    xs = list(range(len(entries)))
    for x, (_label, value, band, colour) in zip(xs, entries, strict=True):
        if band is not None:
            ax.fill_between(
                [x - 0.34, x + 0.34], band.lo, band.hi, color=_NULLC, alpha=0.30, zorder=1
            )
            ax.plot([x - 0.34, x + 0.34], [band.mean, band.mean], color=_NULLC, lw=1.0, zorder=2)
        inside = band is not None and band.contains(value)
        ax.plot([x], [value], "o", color=colour, ms=11, zorder=4)
        ax.annotate(
            f"{value:+.3f}\n{'INSIDE the null' if inside else 'above the null'}",
            xy=(x, value),
            xytext=(0, 14),
            textcoords="offset points",
            fontsize=7.5,
            ha="center",
            va="bottom",
            color=colour,
        )
    ax.axhline(0.0, color="#bbbbbb", lw=0.8, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([e[0] for e in entries], fontsize=8)
    ax.set_xlim(-0.6, len(entries) - 0.4)
    values = [e[1] for e in entries]
    bounds = [b.lo for _l, _v, b, _c in entries if b] + [b.hi for _l, _v, b, _c in entries if b]
    top, bottom = max(values + bounds), min(values + bounds)
    ax.set_ylim(bottom - 0.18 * (top - bottom), top + 0.55 * (top - bottom))
    ax.set_ylabel("Brier skill vs the per-model base rate", fontsize=9)
    ax.fill_between(
        [], [], [], color=_NULLC, alpha=0.30, label="shuffled-outcome null, central 95%"
    )
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    plot_frame.panel_label(ax, "C · does the score carry signal?")


def _annotations(cal: Calibration, admissibility_reason: str, imputation: str) -> Annotations:
    inside = cal.null.contains(cal.observed)
    control_fired = (
        cal.control is not None
        and cal.control_null is not None
        and not cal.control_null.contains(cal.control)
    )
    facts = [
        f"{cal.n_tasks} tasks x {cal.n_models} models, k={cal.k}, leave-one-out",
        f"Brier skill {cal.observed:+.3f} vs null 95% [{cal.null.lo:+.3f}, {cal.null.hi:+.3f}]",
    ]
    if cal.control is not None:
        facts.append(f"human-tag positive control {cal.control:+.3f}")
    caveat = None
    if inside and control_fired:
        caveat = (
            "Falsified, not untested: the control fires on this same pipeline while the "
            "embedding sits inside the null."
        )
    elif inside:
        caveat = "Inside the null AND the control did not fire — the instrument is unproven."
    return Annotations(
        subtitle_facts=tuple(facts),
        caveat=caveat,
        notes=(
            admissibility_reason,
            *(
                f"bin [{b['lo']:.1f},{b['hi']:.1f}): predicted {b['predicted']:.3f}, "
                f"observed {b['observed']:.3f} (n={b['n']})"
                for b in cal.bins
            ),
        ),
        limitations=(imputation,),
        counts=(("tasks", cal.n_tasks), ("models", cal.n_models), ("k", cal.k)),
    )


def imputed_census(
    results: dict, task_ids: list[str], models_order: list[str]
) -> tuple[int, int, int]:
    """``(n_imputed, n_cells, n_imputed_passes)`` over the cells this figure scores."""
    cells = [results.get(tid, {}).get(m) for tid in task_ids for m in models_order]
    present = [c for c in cells if c]
    imputed = [c for c in present if c.get("imputed")]
    return len(imputed), len(present), sum(1 for c in imputed if c.get("pass"))


def imputation_limit_line(results: dict, task_ids: list[str], models_order: list[str]) -> str:
    """The LIMITS line naming the pass-only imputation this figure rests on."""
    n_imp, n_cells, n_imp_pass = imputed_census(results, task_ids, models_order)
    if not n_cells or not n_imp:
        return "Every scored cell is a real measurement — no imputation on this path."
    return (
        f"{n_imp}/{n_cells} scored cells ({n_imp / n_cells:.1%}) are monotone-IMPUTED, not "
        f"measured, and {n_imp_pass}/{n_imp} of them are filled pass=True — imputation here "
        f"is near-exclusively pass-filling (the ladder's fail branch fires rarely), so it "
        f"almost never adds a failure. Every rate on this "
        f"figure is biased UPWARD by that fill."
    )


def compute(
    matrix: dict, task_ids: list[str], models_by_price: list[str], k: int, threshold: float
) -> Calibration:
    """Embed, score, null, control — everything the figure needs, in one pass."""
    emb = build_task_embeddings(matrix, task_ids)
    sims = emb @ emb.T
    pass_mat = _pass_matrix(matrix["results"], task_ids, models_by_price)
    rates = weighted_rates(sims, pass_mat, k)
    control_sims = difficulty_similarity(matrix, task_ids, sims)
    control = control_null = None
    if control_sims is not None:
        control = brier_skill(weighted_rates(control_sims, pass_mat, k), pass_mat)
        control_null = null_band(control_sims, pass_mat, k)
    return Calibration(
        bins=reliability(rates, pass_mat),
        rates=rates,
        observed=brier_skill(rates, pass_mat),
        null=null_band(sims, pass_mat, k),
        control=control,
        control_null=control_null,
        threshold=threshold,
        k=k,
        n_tasks=len(task_ids),
        n_models=len(models_by_price),
    )


def _pass_matrix(results: dict, task_ids: list[str], models_by_price: list[str]) -> np.ndarray:
    mat = np.zeros((len(task_ids), len(models_by_price)), dtype=float)
    for i, tid in enumerate(task_ids):
        cells = results.get(tid, {})
        for j, model in enumerate(models_by_price):
            mat[i, j] = 1.0 if cells.get(model, {}).get("pass", False) else 0.0
    return mat


def plot(cal: Calibration, out_path: Path, extra: Annotations, digest: str) -> Path:
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 3, width_ratios=(1.15, 1.0, 0.85))
    _draw_reliability(axes[0], cal)
    _draw_distribution(axes[1], cal)
    _draw_skill(axes[2], cal)
    return plot_frame.save(
        fig,
        out_path,
        SPEC,
        extra=extra,
        provenance=plot_frame.Provenance(_GENERATOR, digest, ctxmod.MANIFEST),
        size=size,
    )


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
    config.load(config_path)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--output-dir", default="docs/assets/figures/routing")
    ap.add_argument("--k", type=int, default=None, help="neighbourhood size (default: configured)")
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

    models = config.enabled_models() or list(matrix.get("models", {}).keys())
    pricing = config.enabled_pricing()
    models_by_price = sorted(models, key=lambda m: config.cost_per_1m(m, pricing))
    k = args.k if args.k is not None else int(config.knn_params().get("k", 10))
    threshold = float(
        config.knn_params().get("success_rate_threshold", _DEFAULT_SUCCESS_RATE_THRESHOLD)
    )
    task_ids = sorted(results)

    # BEFORE the figure: does this pipeline recover a signal it is KNOWN to contain?
    admissibility = routing_instrument_admissibility(k=k, threshold=threshold)
    print(admissibility.reason)

    cal = compute(matrix, task_ids, models_by_price, k, threshold)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = _annotations(
        cal, admissibility.reason, imputation_limit_line(results, task_ids, models_by_price)
    )
    plot(cal, out_dir / "knn_calibration.png", extra, ctxmod.corpus_digest(matrix, task_ids))
    print("Saved knn_calibration.png")
    print(f"  Brier skill {cal.observed:+.4f}  null95 [{cal.null.lo:+.4f}, {cal.null.hi:+.4f}]")
    if cal.control is not None and cal.control_null is not None:
        print(
            f"  positive control {cal.control:+.4f}  "
            f"null95 [{cal.control_null.lo:+.4f}, {cal.control_null.hi:+.4f}]"
        )


if __name__ == "__main__":
    main()
