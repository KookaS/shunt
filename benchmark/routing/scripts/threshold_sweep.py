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
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from sklearn.metrics.pairwise import cosine_similarity

from benchmark import config, plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import plot_style, summary
from benchmark.routing.strategies import routing_text
from benchmark.routing.strategies.knn import _embed_texts

matplotlib.use("Agg")

SPEC = FigureSpec(
    reading=(
        "Two heatmaps over the SAME grid: x is k, the number of nearest neighbours the "
        "router consults, y is whichever of success_rate_thresh / min_samples explains more "
        "variance (named on the axis), and the third swept parameter is fixed at the value "
        "in the title. LEFT is the held-out pass rate — brighter is better. RIGHT is the "
        "share of tasks that combination routes to the single most expensive model — "
        "brighter means MORE escalation, and 1.00 means the 'router' is Always-Frontier. "
        "The red ring marks the same cell in both panels."
    ),
    goal=(
        "Read the two panels TOGETHER. A cell is only a routing result if it is bright on "
        "the left while NOT saturated on the right: quality that came from choosing between "
        "models, rather than from sending everything to the priciest one. A bright-left, "
        "bright-right cell has bought its pass rate with money, and a fixed policy would "
        "have done the same thing for the same price."
    ),
    definitions=(
        ("held-out pass rate", "scored on tasks whose own fold was excluded from the index"),
        ("frontier share", "fraction of tasks the combination sends to the priciest model"),
        ("k", "how many nearest tasks the router consults before choosing a model"),
        (
            "success_rate_thresh",
            "neighbour pass-rate below which the router escalates off the cheap model",
        ),
        ("min_samples", "min neighbours with a recorded outcome before the router trusts them"),
        (
            "cost at equal quality",
            "cheapest cell whose pass rate clears the best cell's 95% Wilson lower bound — "
            "the selection rule, replacing reward-argmax",
        ),
    ),
    notes=(
        "Neighbourhoods use the real shipped jina embedder — the same Embedder the router "
        "runs, never a TF-IDF proxy.",
    ),
    limitations=(
        "Folds split TASKS, not repositories, so a held-out task can still sit next to a "
        "sibling task from the same repo — this is a lower bound on optimism, not an "
        "estimate of transfer to a new codebase (see knn_cross_repo_transfer.png).",
    ),
)


# η² below this reads as "this knob does not move reward" (Cohen's small/medium
# boundary). Above it the figure must not call the effect negligible.
_NEGLIGIBLE_ETA_SQ = 0.06


def _eta_phrase(eta: float) -> str:
    """How much reward variance one parameter explains, worded to match the number."""
    if eta < _NEGLIGIBLE_ETA_SQ:
        return f"η²={eta:.2f} — negligible effect"
    return (
        f"η²={eta:.2f} — explains {eta * 100:.0f}% of reward variance, "
        "so this slice hides real variation"
    )


def load_matrix(path: Path) -> dict:
    # Analytical (routing reward): default to the VALID set — complete challenges,
    # censored cells and incomplete challenges excluded (summary.load_scored_matrix).
    return summary.load_scored_matrix(path)


def model_cost(model_name: str, matrix: dict) -> float:
    models = matrix["models"]
    if model_name in models:
        m = models[model_name]
        return m.get("input_price", 0) + m.get("output_price", 0)
    return 0.0


def knn_select(  # noqa: PLR0913
    query_idx: int,
    task_ids: list[str],
    task_descs: list[str],
    features: np.ndarray,
    results_map: dict[str, dict],
    matrix: dict,
    k: int,
    success_rate_thresh: float,
    min_samples: int,
    allowed: np.ndarray | None = None,
    sims_row: np.ndarray | None = None,
) -> tuple[str, bool, float, bool]:
    """``(chosen, passed, cost, scored)``; ``scored`` is False on an unmeasured chosen cell."""
    # A coverage gap the caller EXCLUDES from aggregation, never imputes as fail@$0.
    # ``allowed`` restricts which tasks may act as neighbours — that is what turns the
    # sweep from in-sample tuning into held-out evaluation. ``sims_row`` accepts a
    # precomputed similarity row (the whole grid shares one matrix).
    sims = (
        np.array(sims_row, dtype=float, copy=True)
        if sims_row is not None
        else cosine_similarity(features[query_idx : query_idx + 1], features).flatten()
    )
    sims[query_idx] = -1.0
    if allowed is not None:
        blocked = np.ones(len(sims), dtype=bool)
        blocked[allowed] = False
        sims[blocked] = -np.inf

    ranked = np.argsort(sims)[::-1][:k]
    # -inf marks a task outside the allowed index and -1.0 the query itself; both are
    # sentinels, never real neighbours, so they must not reach the vote.
    nearest = [int(i) for i in ranked if np.isfinite(sims[i]) and sims[i] > -1.0]

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
    return chosen, outcome.get("pass", False), plot_style.row_real_cost(outcome), True


def compute_reward(passed: bool, cost: float, gamma: float = 0.1) -> float:
    return 1.0 - gamma * cost if passed else 0.0 - gamma * cost


def cost_at_equal_quality(rows: list[dict]) -> dict | None:
    """The CHEAPEST cell whose pass rate is not significantly below the best one's."""
    # This replaces reward-argmax as the selection rule, and it removes gamma entirely.
    # Reward = passes - gamma*cost with the configured gamma=0.1 makes one extra pass worth
    # $10 against a suite costing $1-$87, so cost is very nearly a no-op and the argmax
    # degenerates to "escalate everything" — the previous optimum routed 86% of tasks to the
    # single priciest model, i.e. it selected Always-Frontier and called it routing. The
    # project's actual criterion is cost AT EQUAL QUALITY, which is what this computes: take
    # the best pass rate's Wilson lower bound as the quality floor, then minimise cost.
    scored = [r for r in rows if int(r.get("n_scored", 0) or 0) > 0]
    if not scored:
        return None
    best = max(scored, key=lambda r: r["AvgPerf%"])
    n_best = int(best["n_scored"])
    floor_lo, _hi = plot_style.wilson_interval(round(best["AvgPerf%"] / 100 * n_best), n_best)
    eligible = [r for r in scored if r["AvgPerf%"] / 100 >= floor_lo]
    return min(eligible, key=lambda r: (r["TotalCost"], -r["AvgPerf%"]))


def imputed_share(results_map: dict, task_ids: list[str]) -> tuple[int, int]:
    """``(imputed cells, total cells)`` in the matrix this sweep is scored on."""
    cells = [c for tid in task_ids for c in results_map.get(tid, {}).values()]
    return sum(1 for c in cells if c.get("imputed")), len(cells)


def fold_assignment(n: int, folds: int, seed: int = 42) -> np.ndarray:
    """Deterministic fold id per task — the held-out split the sweep is selected on."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n) % max(1, folds))


def evaluate_params(  # noqa: PLR0913
    task_ids: list[str],
    task_descs: list[str],
    features: np.ndarray,
    results_map: dict[str, dict],
    matrix: dict,
    k: int,
    success_rate_thresh: float,
    min_samples: int,
    sims: np.ndarray | None = None,
    fold_ids: np.ndarray | None = None,
    frontier_model: str = "",
) -> dict:
    """Score one hyperparameter cell, held out when ``fold_ids`` is given."""
    # With ``fold_ids`` the neighbour index for each task EXCLUDES its own fold, so the
    # combination is evaluated on tasks it never voted over — the held-out reading. Without
    # it, every task votes over every other, which is the in-sample reading the previous
    # sweep reported (and selected its optimum on).
    g = config.gamma()
    passes = 0
    total_cost = 0.0
    total_reward = 0.0
    n = len(task_ids)
    scored = 0
    allocation: dict[str, int] = {}

    for i in range(n):
        allowed = np.flatnonzero(fold_ids != fold_ids[i]) if fold_ids is not None else None
        chosen, passed, cost, ok = knn_select(
            i,
            task_ids,
            task_descs,
            features,
            results_map,
            matrix,
            k,
            success_rate_thresh,
            min_samples,
            allowed=allowed,
            sims_row=sims[i] if sims is not None else None,
        )
        # A coverage-gap escalation (chosen model unmeasured on this task) is
        # UNSCORABLE — excluded from the aggregation, not imputed fail@$0.
        if not ok:
            continue
        allocation[chosen] = allocation.get(chosen, 0) + 1
        scored += 1
        passes += 1 if passed else 0
        total_cost += cost
        total_reward += compute_reward(passed, cost, g)

    # The frontier share is the degeneracy detector: a "routing optimum" that sends
    # almost everything to the priciest model IS Always-Frontier, and no reward number
    # on its own reveals that.
    frontier_share = (allocation.get(frontier_model, 0) / scored) if scored else 0.0
    models = matrix["models"]
    cheapest = min(models, key=lambda m: model_cost(m, matrix)) if models else ""
    return {
        "k": k,
        "success_rate_thresh": success_rate_thresh,
        "min_samples": min_samples,
        "n_scored": scored,
        "n_excluded": n - scored,
        "AvgPerf%": round(passes / scored * 100, 2) if scored else 0.0,
        "TotalCost": round(total_cost, 6),
        "Reward": round(total_reward, 6),
        "n_models_used": len(allocation),
        "frontier_share": round(frontier_share, 4),
        "cheapest_share": round(allocation.get(cheapest, 0) / scored, 4) if scored else 0.0,
    }


def _sweep_grid(  # noqa: PLR0913
    task_ids, task_descs, features, results_map, matrix, sims, grid, frontier_model, fold_ids
):
    """Evaluate every (k, thresh, min_samples) cell on one split; returns the rows."""
    ks, thresholds, mins_samples = grid
    rows: list[dict] = []
    total = len(ks) * len(thresholds) * len(mins_samples)
    for k in ks:
        for thresh in thresholds:
            for ms in mins_samples:
                rows.append(
                    evaluate_params(
                        task_ids,
                        task_descs,
                        features,
                        results_map,
                        matrix,
                        k,
                        thresh,
                        ms,
                        sims=sims,
                        fold_ids=fold_ids,
                        frontier_model=frontier_model,
                    )
                )
                if len(rows) % 200 == 0 or len(rows) == total:
                    print(f"  [{len(rows)}/{total}] k={k}, thresh={thresh}, min_samp={ms}")
    return rows


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
    """Sweep kNN hyperparameters on a held-out split and report cost at equal quality."""
    config.load(config_path)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--output", default="benchmark/routing/reports/threshold_sweep_results.csv")
    ap.add_argument("--plot", default="benchmark/routing/reports/threshold_sweep_heatmap.png")
    ap.add_argument("--ks", nargs="+", type=int, default=None, help="k values to sweep")
    ap.add_argument("--folds", type=int, default=5, help="held-out folds for selection")
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

    task_descs = [routing_text(tid, matrix["tasks"][tid]) for tid in task_ids]
    results_map = matrix["results"]

    print(f"Loaded {len(task_ids)} tasks from {matrix_path}")

    features = np.asarray(_embed_texts(task_descs), dtype=float)
    print(f"Real jina-embedding matrix: {features.shape}")
    # One similarity matrix for the whole grid: recomputing it per (task, cell) was ~9x
    # the cost and is what made a two-split sweep look unaffordable.
    unit = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    sims = unit @ unit.T

    frontier_model = (
        max(matrix["models"], key=lambda m: model_cost(m, matrix)) if matrix["models"] else ""
    )

    k_max = max(2, len(task_ids) - 1)
    ks = args.ks if args.ks is not None else [k for k in range(2, k_max + 1, 2)] or [2]
    thresholds = (
        args.thresholds
        if args.thresholds is not None
        else [round(0.5 + i * 0.1, 1) for i in range(5)]
    )
    mins_samples = args.min_samples_list if args.min_samples_list is not None else [1, 2, 3, 4, 5]
    grid = (ks, thresholds, mins_samples)

    print("Parameters sweeps:")
    print(f"  k = {ks}")
    print(f"  success_rate_thresh = {thresholds}")
    print(f"  min_samples = {mins_samples}")
    print(f"  Total combos: {len(ks) * len(thresholds) * len(mins_samples)} per split")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    fold_ids = fold_assignment(len(task_ids), args.folds)
    print("\nIn-sample split (every task votes over every other):")
    in_sample = _sweep_grid(
        task_ids, task_descs, features, results_map, matrix, sims, grid, frontier_model, None
    )
    print(f"\nHeld-out split ({args.folds} folds; a task never votes over its own fold):")
    held_out = _sweep_grid(
        task_ids, task_descs, features, results_map, matrix, sims, grid, frontier_model, fold_ids
    )

    for row in in_sample:
        row["split"] = "in_sample"
    for row in held_out:
        row["split"] = "held_out"

    all_results = sorted(in_sample + held_out, key=lambda r: r["Reward"], reverse=True)
    fields = [
        "split",
        "k",
        "success_rate_thresh",
        "min_samples",
        "n_scored",
        "n_excluded",
        "AvgPerf%",
        "TotalCost",
        "Reward",
        "n_models_used",
        "frontier_share",
        "cheapest_share",
    ]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in all_results:
            w.writerow({k: row.get(k, "") for k in fields})
    print(f"\nResults written to {args.output}")

    # The two selection rules, side by side — this contrast IS the finding.
    reward_best = max(held_out, key=lambda r: r["Reward"])
    caeq_best = cost_at_equal_quality(held_out)
    print("\n=== SELECTION (held-out) ===")
    for label, row in (
        (f"reward-argmax (gamma={config.gamma():.3g})", reward_best),
        ("cost-at-equal-quality", caeq_best),
    ):
        if row is None:
            continue
        print(
            f"  {label:32} k={row['k']:>3} thresh={row['success_rate_thresh']:.1f} "
            f"min_samp={row['min_samples']} pass={row['AvgPerf%']:.2f}% "
            f"cost=${row['TotalCost']:.4f} frontier_share={row['frontier_share']:.0%} "
            f"models_used={row['n_models_used']}"
        )

    results_arr = np.array(
        [(r["k"], r["success_rate_thresh"], r["min_samples"], r["Reward"]) for r in held_out]
    )
    param_names = ["k", "success_rate_thresh", "min_samples"]
    param_values = [ks, thresholds, mins_samples]
    eta_sq = _variance_explained(results_arr, param_values)
    most_sensitive = param_names[int(np.argmax(eta_sq))]
    print("\n=== SENSITIVITY ANALYSIS (held-out) ===")
    for pn, pv in zip(param_names, eta_sq, strict=True):
        print(f"  {pn}: variance explained (eta^2) = {pv:.4f}")
    print(f"  Most sensitive parameter: {most_sensitive}")

    selected = caeq_best or reward_best
    if most_sensitive in ("success_rate_thresh", "k"):
        fixed_name, fixed_val = "min_samples", selected["min_samples"]
        y_name = "success_rate_thresh"
    else:
        fixed_name, fixed_val = "success_rate_thresh", selected["success_rate_thresh"]
        y_name = "min_samples"

    _plot_sweep_heatmap(
        _slice_at(held_out, fixed_name, fixed_val),
        _slice_at(in_sample, fixed_name, fixed_val),
        Path(args.plot),
        y_name=y_name,
        fixed=(fixed_name, fixed_val),
        swept_ks=ks,
        sensitivity=dict(zip(param_names, eta_sq, strict=True)),
        selected=selected,
        reward_best=reward_best,
        n_folds=args.folds,
        imputed=imputed_share(results_map, task_ids),
        frontier_model=frontier_model,
    )
    print(f"\nHeatmap saved to {args.plot}")


def _slice_at(rows: list[dict], param: str, value: float) -> list[dict]:
    """Rows sitting exactly on one swept grid value of ``param``."""
    # Exact, not within a tolerance: these values come from the same grid the rows were
    # built from, and a 0.1 epsilon admitted the neighbouring 0.1-spaced level
    # (abs(0.6 - 0.5) is 0.0999...), so the heatmap drew a slice its own title disclaimed.
    return [r for r in rows if r[param] == value]


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


def _heat_grid(
    filtered: list[dict], y_name: str, value_key: str = "Reward"
) -> tuple[list, list, np.ndarray]:
    """``(x_vals, y_vals, values)`` over (k x y_name), one row per (y, x) cell."""
    # Two rows on one cell mean the caller's slice still mixes levels of the third
    # parameter: that would render the last writer's rewards under a title promising
    # the other level, so it raises rather than drawing the wrong data.
    x_vals = sorted({r["k"] for r in filtered})
    y_vals = sorted({r[y_name] for r in filtered})
    heat_data = np.full((len(y_vals), len(x_vals)), np.nan)
    for r in filtered:
        yi, xi = y_vals.index(r[y_name]), x_vals.index(r["k"])
        if not np.isnan(heat_data[yi, xi]):
            raise ValueError(
                f"two rows at (k={r['k']}, {y_name}={r[y_name]}): the slice mixes levels "
                "of the fixed parameter, so the heatmap would not match its title"
            )
        heat_data[yi, xi] = r[value_key]
    return x_vals, y_vals, heat_data


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
            # The optimum is always labelled, in a larger bold face; a periodic label in
            # the adjacent cell of the same row would print on top of it.
            if not is_best and yi == best_yi and abs(xi - best_xi) <= 1:
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


def _driver_sentence(a: tuple[str, float], b: tuple[str, float]) -> str:
    """Which of two plotted parameters drives reward — read off their η², not assumed.

    The previous wording hardcoded the axis parameter as the driver and k as the
    passenger, so whenever k dominated the sentence contradicted the η² it printed.
    """
    (driver, driver_eta), (other, other_eta) = sorted((a, b), key=lambda t: t[1], reverse=True)
    if other_eta < _NEGLIGIBLE_ETA_SQ:
        tail = f"{other} barely moves it (η²={other_eta:.2f}) — pick {other} for stability"
    else:
        tail = f"{other} also matters (η²={other_eta:.2f}) — neither can be picked freely"
    return f"Reward is driven by {driver} (η²={driver_eta:.2f}); {tail}"


def _heatmap_annotations(  # noqa: PLR0913
    *,
    selected: dict,
    reward_best: dict,
    y_name: str,
    sensitivity: dict[str, float],
    swept_ks: list[int],
    n_folds: int,
    imputed: tuple[int, int],
    frontier_model: str,
) -> Annotations:
    """Footer content the sweep computes: degeneracy, imputation, and which knob matters."""
    notes: list[str] = []
    limits: list[str] = []

    # The headline: what reward-argmax actually selects. A "routing optimum" that sends
    # almost every task to one model is that model's fixed policy wearing a router's name.
    share = float(reward_best.get("frontier_share", 0.0))
    used = int(reward_best.get("n_models_used", 0))
    degenerate = (
        f"REWARD-ARGMAX IS DEGENERATE: maximising reward (passes - gamma x cost, "
        f"gamma={config.gamma():.3g}) picks k={reward_best['k']}, "
        f"thresh={reward_best['success_rate_thresh']}, which routes "
        f"{share:.0%} of tasks to {frontier_model} using {used} distinct model(s) — that is "
        f"Always-Frontier, not a routing result. At this gamma one extra pass is worth "
        f"{1 / config.gamma():.0f} USD against a suite costing a few dollars, so cost is "
        f"nearly a no-op and the argmax escalates everything."
    )
    (limits if share > 0.5 or used <= 1 else notes).append(degenerate)

    notes.append(
        f"The ringed cell is chosen by COST AT EQUAL QUALITY, not by reward: cheapest "
        f"combination whose held-out pass rate clears the best cell's 95% Wilson lower "
        f"bound. It is k={selected['k']}, thresh={selected['success_rate_thresh']}, "
        f"min_samples={selected['min_samples']} -> {selected['AvgPerf%']:.1f}% at "
        f"{plot_style.usd(float(selected['TotalCost']), 4)}, routing "
        f"{float(selected.get('frontier_share', 0.0)):.0%} to {frontier_model}."
    )
    notes.append(
        f"Selection and scoring are HELD OUT: {n_folds}-fold split over tasks, and a task's "
        f"neighbourhood never contains its own fold. The right panel's in-sample minus "
        f"held-out gap is the optimism the previous in-sample-only sweep reported as fact."
    )

    y_eta, k_eta = sensitivity.get(y_name), sensitivity.get("k")
    if y_eta is not None and k_eta is not None:
        notes.append(_driver_sentence((y_name, y_eta), ("k", k_eta)))

    lo, hi = (min(swept_ks), max(swept_ks)) if swept_ks else (0, 0)
    if swept_ks and selected["k"] in (lo, hi):
        limits.append(
            f"The selected k={selected['k']} sits at the EDGE of the swept range "
            f"k in [{lo}, {hi}] — the optimum is not bracketed and the true peak may lie beyond it"
        )
    elif swept_ks:
        notes.append(
            f"Selected k={selected['k']} lies inside the swept range k in [{lo}, {hi}] — "
            "the optimum is bracketed"
        )

    n_imp, n_cells = imputed
    if n_imp:
        limits.append(
            f"{n_imp}/{n_cells} cells ({n_imp / n_cells:.1%}) in the scored matrix are "
            f"monotone-IMPUTED rather than measured, and the imputation is pass-only, so it "
            f"can never add a failure. The neighbourhood VOTES and the pass rates on this "
            f"grid both read those synthetic passes — every quality number here is biased up"
        )
    limits.append("Cost is model-price dependent — the selected cell moves when model prices move.")
    return Annotations(notes=tuple(notes), limitations=tuple(limits))


def _draw_panel(  # noqa: PLR0913
    ax: Axes,
    rows: list[dict],
    y_name: str,
    value_key: str,
    *,
    title: str,
    cbar_label: str,
    cmap_name: str,
    fig: object,
    ring: tuple[float, float] | None,
    fmt: str = "{:.0f}",
) -> None:
    """One heatmap panel over (k x y_name), with the selected cell ringed."""
    x_vals, y_vals, heat = _heat_grid(rows, y_name, value_key)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")
    im = ax.imshow(heat, aspect="auto", cmap=cmap, interpolation="nearest")
    tick_step = max(1, math.ceil(len(x_vals) / 14))
    ax.set_xticks(range(0, len(x_vals), tick_step))
    ax.set_xticklabels(x_vals[::tick_step], fontsize=8)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(y_vals, fontsize=8)
    ax.set_xlabel("k  (number of nearest neighbours)", fontsize=9)
    ax.set_ylabel(y_name, fontsize=9)
    ax.set_title(title, fontsize=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)  # type: ignore[attr-defined]
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    label_step = max(1, math.ceil(len(x_vals) / 10))
    mid = (np.nanmax(heat) + np.nanmin(heat)) / 2
    for yi in range(heat.shape[0]):
        for xi in range(0, heat.shape[1], label_step):
            val = heat[yi, xi]
            if np.isnan(val):
                continue
            ax.text(
                xi,
                yi,
                fmt.format(val),
                ha="center",
                va="center",
                fontsize=6.5,
                color="black" if val > mid else "white",
            )
    if ring is not None and ring[0] in y_vals and ring[1] in x_vals:
        yi, xi = y_vals.index(ring[0]), x_vals.index(ring[1])
        ax.add_patch(
            Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False, edgecolor="#d03b3b", linewidth=2.4)
        )


def _plot_sweep_heatmap(  # noqa: PLR0913
    held_out: list[dict],
    in_sample: list[dict],
    plot_path: Path,
    *,
    y_name: str,
    fixed: tuple[str, float],
    swept_ks: list[int],
    sensitivity: dict[str, float],
    selected: dict,
    reward_best: dict,
    n_folds: int,
    imputed: tuple[int, int],
    frontier_model: str,
) -> None:
    """Held-out quality beside the allocation that produced it, on one (k x y_name) grid."""
    fig, (ax_q, ax_a) = plt.subplots(1, 2, figsize=(17.5, 6.6))
    ring = (selected[y_name], selected["k"])

    _draw_panel(
        ax_q,
        held_out,
        y_name,
        "AvgPerf%",
        title="Held-out pass rate (%) — what a deployed router would score",
        cbar_label="pass rate (%) on tasks the neighbourhood never saw",
        cmap_name="viridis",
        fig=fig,
        ring=ring,
    )
    # The allocation panel is the point: it makes a degenerate "optimum" impossible to
    # mistake for routing. A cell at 100% is one fixed model, whatever its reward says.
    _draw_panel(
        ax_a,
        held_out,
        y_name,
        "frontier_share",
        title=f"Share of tasks routed to {frontier_model} — 1.00 IS Always-Frontier",
        cbar_label=f"fraction of tasks sent to {frontier_model}",
        cmap_name="magma",
        fig=fig,
        ring=ring,
        fmt="{:.2f}",
    )

    fixed_name, fixed_val = fixed
    fixed_eta = sensitivity.get(fixed_name)
    fixed_note = f", {_eta_phrase(fixed_eta)}" if fixed_eta is not None else ""
    in_best = max(in_sample, key=lambda r: r["AvgPerf%"])["AvgPerf%"] if in_sample else 0.0
    held_best = max(held_out, key=lambda r: r["AvgPerf%"])["AvgPerf%"] if held_out else 0.0
    fig.suptitle(
        f"kNN hyperparameter sweep — held-out quality vs the allocation behind it  "
        f"(real jina embeddings; fixed {fixed_name}={fixed_val}{fixed_note})\n"
        f"best in-sample pass {in_best:.1f}% vs best held-out {held_best:.1f}% "
        f"({in_best - held_best:+.1f}pp optimism) · red ring = cost-at-equal-quality pick",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plot_frame.save(
        fig,
        plot_path,
        SPEC,
        extra=_heatmap_annotations(
            selected=selected,
            reward_best=reward_best,
            y_name=y_name,
            sensitivity=sensitivity,
            swept_ks=swept_ks,
            n_folds=n_folds,
            imputed=imputed,
            frontier_model=frontier_model,
        ),
    )


if __name__ == "__main__":
    main()
