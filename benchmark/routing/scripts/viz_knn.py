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

from benchmark import config, plot_frame  # noqa: E402
from benchmark.plot_frame import Annotations, FigureSpec  # noqa: E402
from benchmark.routing import plot_style, summary  # noqa: E402
from benchmark.routing.scripts import knn_nulls  # noqa: E402
from benchmark.routing.strategies.knn import _embed_texts  # noqa: E402

# Cheapest-above-threshold cutoff, matching the shipped KnnPolicy.success_rate_threshold
# (0.6) and router.selection.SelectionRule. The proxy applies the product rule so its
# allocation reflects what the router would do, not a best-quality diagnostic.
_DEFAULT_SUCCESS_RATE_THRESHOLD = 0.6

# Above this gap the full-set and common-subset pass rates describe different populations,
# so the faded bars stop being cross-comparable. 5pp is roughly a third of the cheap-to-
# frontier quality gap a router is meant to arbitrage — beyond it, pooling bias is not noise.
_POOLING_BIAS_SMALL_PP = 5.0

_DESCRIPTIVE_SPEC = FigureSpec(
    reading=(
        "Left: pass rate (0-1) for every enabled model, two bars each — the faded bar is "
        "the full set of tasks that model actually ran (its own denominator, printed above "
        "as n=), the solid bar restricts to the common-coverage subset every model ran. "
        "Right: mean real cost per task in dollars on a log10 axis, so one gridline is a "
        "factor of ten. Bar colour is the model's canonical colour, the same in every "
        "figure here."
    ),
    goal=(
        "Look for a small pass-rate gap between the cheap and the frontier bars while the "
        "cost bars span decades — that gap is what a router has to buy back."
    ),
    definitions=(
        ("pass rate", "fraction of tasks where the model's patch passed the graded tests"),
        ("common-coverage subset", "the tasks every enabled model ran, so bars share a base"),
        ("real cost", "the provider's own billed cost, cache discounts included"),
    ),
    notes=(
        "Computed straight from results.csv with no routing or exploration strategy in the loop.",
    ),
    limitations=(
        "Full-set bars each use their own denominator and are NOT cross-comparable: coverage "
        "is adaptive, so frontier models ran on a smaller, differently-selected slice (MNAR "
        "missingness). Rank models on the solid common-subset bars only.",
        "No confidence interval on any bar.",
    ),
)

# Every LIMITS block below used to carry "an unmeasured (model, task) cell counts as a
# non-pass". That branch is DEAD on this path: main() reads the coverage-COMPLETED matrix,
# which has no unmeasured cells, and every cell imputation fills is pass=True — the exact
# inverse of what the caveat claimed. The honest caveat (imputation is pass-only, so every
# neighbourhood rate is biased UP) replaces it, computed per figure from the real flags.

_PCA_SPEC = FigureSpec(
    reading=(
        "Each dot is one task, placed by the two leading principal components of its REAL "
        "768-d jina prompt embedding; the axis labels carry the share of variance each "
        "component explains. COLOUR is how many enabled models solve that task — a measured "
        "difficulty scale, not a router output. MARKER SHAPE is the model the proxy kNN "
        "router picks for it."
    ),
    goal=(
        "Compare the two channels: colour varies across the cloud, so difficulty IS visible "
        "in embedding space. Look for the router's marker shapes tracking that colour "
        "structure. One marker shape covering the whole cloud means the router ignores the "
        "structure the embedding exposes — which is what a constant function looks like, and "
        "what this figure currently shows."
    ),
    definitions=(
        ("PC1 / PC2", "the two directions of largest spread in the embedding space"),
        ("solved-by count", "number of enabled models whose patch passed that task, 0 to 6"),
        ("kNN router", "picks the cheapest model whose neighbourhood pass-rate clears the cut"),
    ),
    limitations=(
        "This is a 2-D shadow of a 768-d space: dots that overlap here may be far apart in "
        "the full space, so visible clustering is necessary but not sufficient.",
        "The marker shapes are the PROXY router's picks over the recorded matrix, not the "
        "live engine's decisions.",
        "Visible geometry belongs to the EMBEDDING, not to the router — a router built on "
        "shuffled outcomes would produce the same point cloud. See knn_transfer_curve.png "
        "for the null comparison that can actually separate the two.",
    ),
)

_PURITY_SPEC = FigureSpec(
    reading=(
        "Left: the task-by-task cosine-similarity matrix over the real 768-d jina prompt "
        "embeddings, rows and columns sorted into blocks by selected model with white dashed "
        "lines at the block edges; brighter is more similar (0-1). Right: the observed mean "
        "neighbourhood purity (blue marker) against the three references that give it "
        "meaning — the permutation null band (grey), the true chance level (sum of squared "
        "label shares), and the majority-class share a constant router scores for free."
    ),
    goal=(
        "On the right, the ONLY reading that supports the router is the blue marker sitting "
        "ABOVE the grey null band. Purity near 1.0 is meaningless on its own: a router that "
        "always picks one model scores ~1.0 by construction, and so does the null."
    ),
    definitions=(
        (
            "purity",
            "share of a task's nearest neighbours — up to k, itself excluded — routed to "
            "the same model as the task",
        ),
        ("chance purity", "sum of squared label shares — what a random neighbour order gives"),
        (
            "permutation null",
            "outcome rows reassigned to tasks at random and the router's picks then "
            "RE-DERIVED — this keeps the selection mechanism (whose picks are correlated "
            "between neighbouring tasks by construction) and breaks only the "
            "description->outcome link",
        ),
        ("cosine similarity", "1.0 = same direction in embedding space, 0 = unrelated"),
    ),
    limitations=(
        "Purity is measured against the router's OWN selection labels, so it scores "
        "self-consistency, not correctness — it can be high while every routing decision "
        "is wrong.",
        "The left matrix is O(n^2) and becomes unreadable as the task count grows.",
    ),
)

_ALLOCATION_SPEC = FigureSpec(
    reading=(
        "One bar pair per enabled model. The solid bar is the number of tasks the proxy kNN "
        "router sends to that model; the hatched outline behind it is what always-cheapest — "
        "the trivial baseline that needs no router — would send. A bar at 0 means the router "
        "never selected that model: an outcome of the rule, not a data gap."
    ),
    goal=(
        "Look for the solid bars departing from the hatched baseline. Where the two profiles "
        "coincide the router IS always-cheapest under a different name, and the cost/pass "
        "delta printed on the figure is what that difference bought."
    ),
    definitions=(
        ("k", "how many nearest past tasks the router votes over — the k= in the title"),
        ("always-cheapest", "send every task to the lowest-priced enabled model, no router"),
        ("enabled model", "a model switched on in the benchmark config, selected or not"),
    ),
    limitations=(
        "This is the proxy router's allocation over the recorded matrix, not the live engine's "
        "per-arm allocation.",
        "Allocation alone says nothing about whether the DIFFERING picks were the right ones — "
        "read the cost/pass delta, and knn_transfer_curve.png for the null comparison.",
    ),
)

_COST_SPEC = FigureSpec(
    reading=(
        "One mark per policy: x is what the whole task suite costs under it (USD, log axis, "
        "one gridline = 10x), y is the share of tasks that passed, with 95% Wilson intervals. "
        "The always-one-model policies are coloured by model; the kNN router is grey because "
        "it is a mixture. The dark step line is the Pareto frontier over the constant "
        "policies, and the dashed cross marks fixed-frontier-with-caching — the project's "
        "kill-gate baseline."
    ),
    goal=(
        "The router earns its existence only by landing OUTSIDE the constant-policy frontier: "
        "left of it at equal height, or above it at equal cost. A router sitting ON or INSIDE "
        "that staircase is Pareto-dominated — some single fixed model matches it for less."
    ),
    definitions=(
        ("Wilson interval", "95% CI for a pass rate, valid near 0 and 1 and at small n"),
        (
            "fixed-frontier-with-caching",
            "always call the strongest enabled model, with its cache discount — the baseline "
            "the project kill gate must be beaten against",
        ),
        ("Pareto-dominated", "another policy is at least as good on BOTH cost and pass rate"),
    ),
    notes=(
        "Costs are each cell's recorded real_cost (the provider's own billed amount, cache "
        "discounts included), never estimated_cost.",
    ),
    limitations=(
        "A backtest over the recorded outcome matrix, not live runs: cache effects are the "
        "per-cell recorded discount, not a replayed cross-task cache.",
        "Cost carries no interval — one point per policy on one suite.",
    ),
)


def build_feature_vectors(results, models_order):
    task_ids = sorted(results.keys())
    n = len(task_ids)
    vecs = np.zeros((n, len(models_order) * 4))

    for i, tid in enumerate(task_ids):
        for j, model in enumerate(models_order):
            r = results[tid].get(model, {})
            passed = 1.0 if r.get("pass", False) else 0.0
            cost = plot_style.row_real_cost(r)
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


def _nearest_neighbors(sims, query_idx, k):
    """Indices of the k most similar OTHER tasks, nearest first."""
    # The query is always excluded and the count clamped to the neighbours that exist,
    # so a k at or above the task count cannot fold the query into its own
    # neighbourhood — which would make every neighbourhood look self-consistent.
    k_eff = min(int(k), len(sims) - 1)
    if k_eff <= 0:
        return np.empty(0, dtype=int)
    ranked = np.argsort(sims)[::-1]
    return ranked[ranked != query_idx][:k_eff]


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
    nearest = _nearest_neighbors(similarities, query_idx, k)
    price_order = sorted(
        models_order, key=lambda m: config.cost_per_1m(m, config.enabled_pricing())
    )
    if not price_order:
        return models_order[0]
    if len(nearest) == 0:
        return price_order[0]

    # The rule itself lives in knn_nulls.select_from_rates so this figure and the
    # null models that test it can never diverge: cheapest model clearing the
    # threshold, else the best-scoring model with ties broken toward the cheaper one.
    pass_cols = [list(models_order).index(m) * 4 for m in price_order]
    rates = np.array([[float(np.mean(vecs[nearest, c])) for c in pass_cols]])
    return price_order[int(knn_nulls.select_from_rates(rates, threshold)[0])]


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
        nearest = _nearest_neighbors(similarity_matrix[i], i, k)
        if len(nearest) == 0:
            continue
        selected = selections[task_ids[i]]
        match_count = sum(1 for n_idx in nearest if selections[task_ids[n_idx]] == selected)
        # Normalise by the neighbours actually available, not the configured k: with
        # fewer than k other tasks the denominator is the neighbourhood, not the request.
        purity[i] = match_count / len(nearest)

    return purity, selections


def imputed_census(results, task_ids, models_order):
    """``(n_imputed, n_cells, n_imputed_passes)`` over the cells these figures score."""
    cells = [results.get(tid, {}).get(m) for tid in task_ids for m in models_order]
    present = [c for c in cells if c]
    imputed = [c for c in present if c.get("imputed")]
    return len(imputed), len(present), sum(1 for c in imputed if c.get("pass"))


def imputation_limit_line(results, task_ids, models_order):
    """The LIMITS line naming the pass-only imputation these figures actually rest on."""
    # Monotone imputation ("a pricier model would also have passed") can only ever add a
    # PASS, so it lifts every neighbourhood rate, every pass bar and every purity number.
    # Stating the share without stating the direction is what made the old caveat useless.
    n_imp, n_cells, n_imp_pass = imputed_census(results, task_ids, models_order)
    if not n_cells or not n_imp:
        return "Every scored cell is a real measurement — no imputation on this path."
    return (
        f"{n_imp}/{n_cells} scored cells ({n_imp / n_cells:.1%}) are monotone-IMPUTED, not "
        f"measured, and {n_imp_pass}/{n_imp} of them are filled pass=True — imputation here "
        f"is pass-only by construction, so it can never add a failure. Every pass rate and "
        f"neighbourhood vote below is biased UPWARD by that fill."
    )


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
        comm_cost = [plot_style.row_real_cost(results[t][m]) for t in common if m in results[t]]
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
    # Above the axes: every in-axes corner sits on a bar (they run 0.5-1.0 tall against
    # a 1.12 ceiling), so an inside legend always covers data.
    ax_p.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)

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
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    n_unmeasured = sum(1 for m in models_order if np.isnan(stats[m]["full_pass"]))
    notes = [
        f"{n_common} of {n_total} tasks were run by every enabled model; the solid bars use "
        f"that subset.",
    ]
    limits = [
        f"{n_unmeasured} enabled model(s) ran no task at all, so their pass bars are absent."
    ] * bool(n_unmeasured)
    # The verdict on the pooling gap is COMPUTED, never asserted: the clause used to read
    # "so the pooling bias is small" unconditionally, and printed it next to 23.4pp.
    pooling = (
        f"Full-set and common-subset pass rates differ by at most {max_gap_pp:.1f}pp — below "
        f"the {_POOLING_BIAS_SMALL_PP:.0f}pp mark, so pooling bias is small here and the "
        f"faded bars can be read alongside the solid ones."
        if max_gap_pp < _POOLING_BIAS_SMALL_PP
        else f"Full-set and common-subset pass rates differ by up to {max_gap_pp:.1f}pp — a "
        f"LARGE pooling bias (over the {_POOLING_BIAS_SMALL_PP:.0f}pp mark). The faded "
        f"full-set bars are NOT comparable across models; rank on the solid bars only."
    )
    (notes if max_gap_pp < _POOLING_BIAS_SMALL_PP else limits).append(pooling)
    plot_frame.save(
        fig,
        output_dir / "model_performance_descriptive.png",
        _DESCRIPTIVE_SPEC,
        extra=Annotations(notes=tuple(notes), limitations=tuple(limits)),
    )
    print("Saved model_performance_descriptive.png")


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


def _policy_totals(results, task_ids, pick):
    """``(total real cost, passes, n_scored)`` for a policy that picks ``pick(tid)``."""
    cost = 0.0
    passes = 0
    scored = 0
    for tid in task_ids:
        cell = results.get(tid, {}).get(pick(tid))
        if not cell:
            continue
        scored += 1
        cost += plot_style.row_real_cost(cell)
        passes += 1 if cell.get("pass") else 0
    return cost, passes, scored


def _allocation_figure(  # noqa: PLR0913
    results,
    task_ids,
    selections,
    models_order,
    model_counts,
    cheapest_model,
    model_colors,
    *,
    n,
    k,
    threshold,
):
    """Router allocation drawn against always-cheapest, with the cost/pass delta it bought."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(models_order))
    baseline_counts = [n if m == cheapest_model else 0 for m in models_order]
    ax.bar(
        x,
        baseline_counts,
        width=0.74,
        facecolor="none",
        edgecolor="#5f6368",
        linewidth=1.3,
        hatch="///",
        label=f"always-cheapest baseline (all {n} tasks -> {cheapest_model})",
    )
    bars = ax.bar(
        x,
        model_counts,
        width=0.46,
        color=[model_colors.get(m, "#9E9E9E") for m in models_order],
        edgecolor="black",
        linewidth=0.8,
        label="kNN router",
    )
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
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Number of tasks routed")
    ax.set_ylim(0, max([*model_counts, *baseline_counts]) * 1.16)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    differing = [tid for tid in task_ids if selections[tid] != cheapest_model]
    r_cost, r_pass, r_n = _policy_totals(results, task_ids, lambda t: selections[t])
    b_cost, b_pass, b_n = _policy_totals(results, task_ids, lambda _t: cheapest_model)
    ax.set_title(
        f"kNN router vs always-cheapest — allocation across {n} tasks (k={k})\n"
        f"router {plot_style.usd(r_cost, 4)} / {r_pass} passes  ·  always-cheapest "
        f"{plot_style.usd(b_cost, 4)} / "
        f"{b_pass} passes  ·  differs on {len(differing)} task(s)",
        fontsize=12,
    )
    fig.tight_layout()

    d_cost = r_cost - b_cost
    d_pass = r_pass - b_pass
    pct = (d_cost / b_cost * 100) if b_cost else 0.0
    if d_pass <= 0 and d_cost >= 0:
        verdict = (
            f"PARETO-DOMINATED: the router costs {d_cost:+.4f} USD ({pct:+.1f}%) for {d_pass:+d} "
            f"passes against always-cheapest — it buys nothing the trivial baseline does not "
            f"already give, and the {len(differing)} differing pick(s) are pure overhead."
        )
    else:
        verdict = (
            f"Against always-cheapest the router is {d_cost:+.4f} ({pct:+.1f}%) on cost for "
            f"{d_pass:+d} pass(es), differing on {len(differing)} of {n} task(s)."
        )
    zero_models = [m for m, c in zip(models_order, model_counts, strict=True) if c == 0]
    notes = [
        f"{n} tasks routed at k={k} with a {threshold:.2f} neighbourhood pass-rate threshold; "
        f"the busiest model, {models_order[int(np.argmax(model_counts))]}, takes "
        f"{max(model_counts) / n * 100:.0f}% of them.",
        verdict,
    ]
    if zero_models:
        notes.append(
            "Enabled models the router never selected, shown at 0 by result rather than by "
            f"missing data: {', '.join(zero_models)}. The 0.6 threshold is non-binding on "
            f"{n - len(differing)}/{n} tasks, so the rule collapses to always-cheapest there."
        )
    limits = []
    if r_n != n or b_n != n:
        limits.append(
            f"Cost/pass totals rest on the tasks each policy had a measured cell for "
            f"(router {r_n}/{n}, always-cheapest {b_n}/{n}); gaps are excluded, never "
            "imputed as $0 or as a failure."
        )
    return fig, notes, limits


def _pareto_figure(  # noqa: PLR0913
    results, task_ids, selections, models_order, model_colors, *, n, k
):
    """Cost vs pass rate for every constant policy plus the router, with Wilson CIs."""
    pricing = config.enabled_pricing()
    by_price = sorted(models_order, key=lambda m: config.cost_per_1m(m, pricing))
    frontier_model = by_price[-1] if by_price else ""

    points = []
    for m in by_price:
        cost, passes, scored = _policy_totals(results, task_ids, lambda _t, mm=m: mm)
        if scored:
            points.append((f"always {m}", cost, passes, scored, model_colors.get(m, "#9E9E9E")))
    r_cost, r_pass, r_n = _policy_totals(results, task_ids, lambda t: selections[t])

    fig, ax = plt.subplots(figsize=(11.5, 7))
    constant_xy = []
    for label, cost, passes, scored, color in points:
        rate = passes / scored
        lo, hi = plot_style.wilson_interval(passes, scored)
        constant_xy.append((cost, rate * 100))
        ax.errorbar(
            cost,
            rate * 100,
            yerr=[[(rate - lo) * 100], [(hi - rate) * 100]],
            fmt="o",
            markersize=9,
            color=color,
            ecolor=color,
            elinewidth=1.3,
            capsize=4,
            zorder=5,
        )
        ax.annotate(
            f"{label}\n{rate * 100:.1f}%, {plot_style.usd(cost)}",
            xy=(cost, rate * 100),
            xytext=(8, -4),
            textcoords="offset points",
            fontsize=7.5,
            va="top",
        )

    r_rate = r_pass / r_n if r_n else 0.0
    r_lo, r_hi = plot_style.wilson_interval(r_pass, r_n)
    ax.errorbar(
        r_cost,
        r_rate * 100,
        yerr=[[(r_rate - r_lo) * 100], [(r_hi - r_rate) * 100]],
        fmt="D",
        markersize=12,
        color="#4a4a4a",
        ecolor="#4a4a4a",
        elinewidth=1.8,
        capsize=5,
        zorder=7,
        label="kNN router (a mixture, so no model hue)",
    )
    ax.annotate(
        f"kNN router\n{r_rate * 100:.1f}%, {plot_style.usd(r_cost)}",
        xy=(r_cost, r_rate * 100),
        xytext=(8, 14),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="#22252a",
    )

    # Frontier over the CONSTANT policies only: the question is whether routing beats
    # every fixed model, so the router must never help define the bar it is measured against.
    hull = sorted(plot_style.pareto_prune(constant_xy))
    if hull:
        hx: list[float] = [hull[0][0]]
        hy: list[float] = [hull[0][1]]
        for cx, cy in hull[1:]:
            hx.extend([cx, cx])
            hy.extend([hy[-1], cy])
        ax.step(
            hx,
            hy,
            where="post",
            color="#22252a",
            linewidth=1.6,
            zorder=3,
            label="Pareto frontier over constant policies",
        )

    f_cost, f_pass, f_n = _policy_totals(results, task_ids, lambda _t: frontier_model)
    if f_n:
        f_rate = f_pass / f_n * 100
        ax.axvline(f_cost, color="#B71C1C", linestyle=(0, (4, 3)), linewidth=1.4, zorder=2)
        ax.axhline(
            f_rate,
            color="#B71C1C",
            linestyle=(0, (4, 3)),
            linewidth=1.4,
            zorder=2,
            label=f"fixed-frontier-with-caching kill-gate baseline ({frontier_model}: "
            f"{f_rate:.1f}% @ {plot_style.usd(f_cost)})",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total real cost over the task suite ($, log scale)")
    ax.set_ylabel("Pass rate — % of tasks passing tests/typecheck")
    ax.set_title(
        f"Does the kNN router beat every fixed model?  Cost vs pass rate over {n} tasks "
        f"(k={k})\n95% Wilson intervals; the router must land OUTSIDE the constant-policy "
        "frontier to be worth having",
        fontsize=12,
    )
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    fig.tight_layout()

    dominators = [
        (label, cost, passes / scored)
        for label, cost, passes, scored, _c in points
        if cost <= r_cost
        and passes / scored >= r_rate
        and (cost < r_cost or passes / scored > r_rate)
    ]
    notes = []
    if dominators:
        names = ", ".join(
            f"{label} ({rate * 100:.1f}% @ {plot_style.usd(cost)})"
            for label, cost, rate in dominators
        )
        notes.append(
            f"PARETO-DOMINATED: the router ({r_rate * 100:.1f}% @ {plot_style.usd(r_cost)}) is "
            f"matched or "
            f"beaten on BOTH cost and pass rate by {len(dominators)} constant policy/policies "
            f"— {names}. On this data the router is not on the frontier."
        )
    else:
        notes.append(
            f"The router ({r_rate * 100:.1f}% @ {plot_style.usd(r_cost)}) is not dominated by any "
            f"single "
            f"constant policy on this data."
        )
    if f_n:
        notes.append(
            f"Against the kill-gate baseline (always {frontier_model}, "
            f"{f_pass}/{f_n} = {f_pass / f_n * 100:.1f}% @ {plot_style.usd(f_cost)}) the router is "
            f"{r_rate * 100 - f_pass / f_n * 100:+.1f}pp on quality for "
            f"{(r_cost / f_cost - 1) * 100:+.1f}% of the cost — cheaper, but NOT at equal "
            f"quality, so this is not a kill-gate pass."
        )
    limits = []
    if r_n != n:
        limits.append(
            f"The router's point rests on {r_n}/{n} tasks whose selected cell was measured; "
            "coverage gaps are excluded, never imputed as $0 or as a failure."
        )
    return fig, notes, limits


def _draw_purity_null(ax, similarity, labels, observed, k, pass_mat, threshold):  # noqa: PLR0913
    """Observed purity against its permutation null, true chance, and the majority share."""
    # Returns the footer notes. The panel this replaces drew a hand-set 0.5 "baseline" while
    # true chance sat at ~0.97 — a 0.47 error that rendered an at-or-below-null result as a
    # towering bar.
    rank = knn_nulls.neighbour_rank(similarity)
    band = knn_nulls.purity_null_band(rank, similarity, pass_mat, k, threshold)
    chance = knn_nulls.chance_purity(labels)
    majority = knn_nulls.majority_share(labels)

    ax.axhspan(
        band.lo,
        band.hi,
        color="#9aa0a6",
        alpha=0.35,
        label=f"outcome-permutation null, central 95% ({band.n} permutations)",
    )
    ax.axhline(band.mean, color="#5f6368", linestyle=(0, (1, 2)), linewidth=1.3)
    ax.axhline(
        chance,
        color="#D55E00",
        linestyle="--",
        linewidth=1.6,
        label=f"chance purity = sum of squared label shares ({chance:.4f})",
    )
    ax.axhline(
        majority,
        color="#009E73",
        linestyle="-.",
        linewidth=1.6,
        label=f"majority-class share — a constant router ({majority:.4f})",
    )
    ax.plot(
        [0],
        [observed],
        marker="o",
        markersize=13,
        color="#0072B2",
        zorder=6,
        label=f"OBSERVED mean purity ({observed:.4f})",
    )
    ax.annotate(
        f"{observed:.4f}\nz = {band.z(observed):+.2f}",
        xy=(0, observed),
        xytext=(14, 0),
        textcoords="offset points",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#0072B2",
    )
    lo = min(band.lo, chance, majority, observed)
    hi = max(band.hi, chance, majority, observed)
    pad = max(0.02, (hi - lo) * 0.6)
    ax.set_ylim(lo - pad, min(1.02, hi + pad))
    ax.set_xlim(-0.5, 1.4)
    ax.set_xticks([])
    ax.set_ylabel(f"Mean neighbourhood purity (k={k})")
    ax.set_title("Observed purity vs its null — is there anything above chance?", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    verdict = (
        f"NULL RESULT: observed purity {observed:.4f} lies INSIDE the permutation null "
        f"[{band.lo:.4f}, {band.hi:.4f}] (null mean {band.mean:.4f}, z={band.z(observed):+.2f}) "
        "— the router's neighbourhoods are no more self-consistent than random labels"
        if band.contains(observed)
        else f"Observed purity {observed:.4f} vs the outcome-permutation null "
        f"[{band.lo:.4f}, {band.hi:.4f}], z={band.z(observed):+.2f}"
    )
    return [
        verdict,
        f"True chance purity is {chance:.4f} and the majority class alone is {majority:.4f}: a "
        f"purity near 1.0 is what a CONSTANT router scores here, so the level carries no "
        f"information — only the distance above the grey band would.",
        f"The left panel's block structure belongs to the EMBEDDING; with {majority:.1%} of "
        f"tasks sharing one label the blocks are near-single-coloured by construction.",
    ]


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
    # Analytical routing views (PCA/purity/allocation/cost) default to the VALID set —
    # complete challenges, censored cells + incomplete challenges excluded.
    matrix = summary.load_scored_matrix(matrix_path)

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
    # One disclosure line, computed from the real `imputed` flags, carried by every figure
    # on this path — the completed matrix is what they all score against.
    imputation_limit = imputation_limit_line(results, task_ids, models_order)
    print(f"  {imputation_limit}")
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

    # Colour carries a quantity that VARIES (how many models solve the task); the router's
    # near-constant pick is demoted to marker shape. Colouring by the argmax label made the
    # figure's own GOAL unfalsifiable — 98% one hue looks like "tight clusters" and would look
    # identical under a null router.
    solved_by = np.array(
        [sum(1 for m in models_order if results[tid].get(m, {}).get("pass")) for tid in task_ids],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    selected_models = [m for m in models_order if m in set(unique)]
    for m in selected_models:
        idx = [i for i, tid in enumerate(task_ids) if selections[tid] == m]
        sc = ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c=solved_by[idx],
            cmap="viridis",
            vmin=0,
            vmax=len(models_order),
            marker=model_markers[m],
            s=58,
            alpha=0.9,
            edgecolors="black",
            linewidth=0.4,
            label=f"router picks {m} (n={len(idx)})",
        )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.82)
    cbar.set_label(f"How many of the {len(models_order)} enabled models solve this task")

    ax.legend(title="marker shape = kNN-router pick", fontsize=9, loc="upper right")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% of variance)")
    ax.set_title(
        "Prompt-embedding space (REAL jina) — measured difficulty vs what the router does\n"
        "colour = how many models solve the task (varies) · shape = the router's pick",
        fontsize=12,
    )
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    never = [m for m in models_order if m not in set(unique)]
    fig.tight_layout()
    majority = knn_nulls.majority_share([selections[t] for t in task_ids])
    pca_notes = [
        f"{n} tasks; the router used k={k_neighbors} neighbours and a "
        f"{threshold:.2f} neighbourhood pass-rate threshold.",
        f"The two components shown carry {(explained[0] + explained[1]) * 100:.1f}% of the "
        f"embedding variance between them.",
        f"DEGENERATE LABELLING: {majority:.1%} of tasks get the same router pick "
        f"({max(zip(counts, unique, strict=True))[1]}), so marker shape is very nearly a "
        f"constant — any apparent 'clustering' by shape is the majority class, not a decision.",
        f"Colour spans {int(solved_by.min())}-{int(solved_by.max())} models solving a task, so "
        f"the difficulty the router is failing to track is genuinely present in this space.",
    ]
    if never:
        pca_notes.append(
            "Enabled models the router never selected are absent from the legend by result, "
            f"not by missing data: {', '.join(never)}."
        )
    plot_frame.save(
        fig,
        output_dir / "knn_pca_scatter.png",
        _PCA_SPEC,
        extra=Annotations(notes=tuple(pca_notes), limitations=(imputation_limit,)),
    )
    print("Saved knn_pca_scatter.png")

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

    labels = [selections[tid] for tid in task_ids]
    # The null re-derives the router's picks from PERMUTED outcomes, so it needs the pass
    # matrix and threshold, not just the labels: shuffling endogenous labels would leave a
    # band too narrow to be a real test (see knn_nulls.purity_null_band).
    by_price = sorted(models_order, key=lambda m: config.cost_per_1m(m, config.enabled_pricing()))
    pass_mat = np.column_stack([vecs[:, list(models_order).index(m) * 4] for m in by_price])
    purity_notes = _draw_purity_null(
        axes[1], similarity, labels, float(np.mean(purity)), k_neighbors, pass_mat, threshold
    )

    fig.suptitle(
        "kNN router (real jina embeddings) — is neighbourhood purity above its own null?",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plot_frame.save(
        fig,
        output_dir / "neighborhood_purity.png",
        _PURITY_SPEC,
        extra=Annotations(notes=tuple(purity_notes), limitations=(imputation_limit,)),
    )
    print("Saved neighborhood_purity.png")

    # 3. Model allocation, against the always-cheapest baseline it has to beat.
    pricing = config.enabled_pricing()
    cheapest_model = min(pricing, key=lambda m: config.cost_per_1m(m, pricing)) if pricing else ""
    model_counts = [counts[list(unique).index(m)] if m in unique else 0 for m in models_order]
    alloc_fig, alloc_notes, alloc_limits = _allocation_figure(
        results,
        task_ids,
        selections,
        models_order,
        model_counts,
        cheapest_model,
        model_colors,
        n=n,
        k=k_neighbors,
        threshold=threshold,
    )
    plot_frame.save(
        alloc_fig,
        output_dir / "model_allocation.png",
        _ALLOCATION_SPEC,
        extra=Annotations(notes=tuple(alloc_notes), limitations=(*alloc_limits, imputation_limit)),
    )
    print("Saved model_allocation.png")

    # 3b. Descriptive all-model view (strategy-agnostic, straight from results.csv).
    # Answers "how does EVERY model do", not "which model does the router pick".
    # DIAGNOSTIC coverage view: it contrasts each model's full measured set against the
    # common-coverage subset (MNAR missingness), so it MUST read the RAW matrix — the
    # valid/imputed matrix has equal coverage by construction, which would collapse that
    # very contrast. Explicit opt-in to raw (config.load_matrix), unlike the plots above.
    raw_results = config.load_matrix(matrix_path)["results"]
    plot_descriptive_model_performance(raw_results, models_order, output_dir)

    # 4. Cost-vs-pass Pareto scatter. This REPLACES a three-bar cost chart that printed a
    # pass rate for one of its three bars and headlined "saves 98% vs frontier" — a ratio
    # against a half-imputed denominator at UNEQUAL quality. Cost alone cannot carry a
    # cost-at-equal-quality claim; the router's position relative to the constant-policy
    # frontier can.
    cost_fig, cost_notes, cost_limits = _pareto_figure(
        results,
        task_ids,
        selections,
        models_order,
        model_colors,
        n=n,
        k=k_neighbors,
    )
    plot_frame.save(
        cost_fig,
        output_dir / "knn_cost_comparison.png",
        _COST_SPEC,
        extra=Annotations(notes=tuple(cost_notes), limitations=(*cost_limits, imputation_limit)),
    )
    print("Saved knn_cost_comparison.png")
    for line in cost_notes:
        print(f"  {line}")


if __name__ == "__main__":
    main()
