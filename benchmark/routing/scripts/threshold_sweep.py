#!/usr/bin/env python3
"""Sweep kNN hyperparameters (k, success_rate_threshold, min_samples) with a real
outer-loop cross-validation and draw the regimes the grid actually contains.
Output: CSV in benchmark/routing/reports/, sweep_regimes.png in docs/assets/figures/routing/.
"""

# Neighbourhoods use the real shipped jina embedder (the same ``Embedder`` the router
# runs), never a TF-IDF proxy.

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.metrics.pairwise import cosine_similarity

from benchmark import config, plot_frame
from benchmark.plot_frame import Annotations, FigureSpec
from benchmark.routing import plot_style, summary
from benchmark.routing.figures import context as ctxmod
from benchmark.routing.figures.context import corpus_digest
from benchmark.routing.strategies import routing_text
from benchmark.routing.strategies.knn import _embed_texts

_REPO_ROOT = Path(__file__).resolve().parents[3]

# `python -m` sets __name__ to "__main__", which would land in the figure
# manifest instead of the module that drew it.
_GENERATOR = "benchmark.routing.scripts.threshold_sweep"

matplotlib.use("Agg")

# Log-spaced, not uniform. The grid has exactly two regimes — a small-k band where the
# neighbourhood is too thin to clear any threshold (always-cheapest) and a large-k band
# where it is wide enough to always clear one (fixed allocation) — and nothing in
# between. 87 uniformly-spaced k values spent 96% of the compute inside one of them and
# printed the same 77.14 on 2185 of 4350 rows.
K_GRID: Final[tuple[int, ...]] = (2, 3, 5, 8, 12, 20, 32, 50, 80, 128, 174)

CHEAP_REGIME: Final[int] = 0
MIXED_REGIME: Final[int] = 1
FRONTIER_REGIME: Final[int] = 2
_REGIME_LABELS: Final[tuple[str, ...]] = (
    "always-cheapest / degenerate",
    "mixed allocation",
    "always-frontier / degenerate",
)
# Both degenerate labels begin with "a", so the in-cell key is spelled out rather than
# taken from the first letter.
_REGIME_KEYS: Final[tuple[str, ...]] = ("C", "M", "F")
_REGIME_COLORS: Final[tuple[str, ...]] = ("#BFD8E8", "#2E7D32", "#E8A33D")
# A share this high means every task but a rounding-error handful went to one model.
_DEGENERATE_SHARE: Final[float] = 0.99

# A heatmap cell may carry a printed number only when it is this many inches wide.
# Below it the digits are unreadable and the fill has to carry the value on its own —
# which is why this panel is categorical with a legend rather than 55 printed shares.
_MIN_IN_PER_COL: Final[float] = 0.35

_TRACE_COLOR: Final[str] = "#1F4E79"
_MIXTURE_COLOR: Final[str] = "#B71C1C"
# A pass rate is BOUNDED. Panel C's y axis stops here rather than at whatever height its
# annotations wanted, so no part of the panel stands over a value the quantity cannot take.
_PASS_RATE_MAX: Final[float] = 100.0
_SHIPPED_COLOR: Final[str] = "#6A3D9A"

SPEC = FigureSpec(
    title="A narrow mid-k band beats the two-policy mixture; the shipped setting does not",
    reading=(
        "Three panels over ONE sweep. A: each point is one k on the log-spaced grid, "
        "placed at the total cost and pass rate its best threshold achieves out of fold; "
        "the dashed line is the straight mixture of the two fixed policies (send a "
        "fraction of tasks to the cheapest model and the rest to the frontier one). B: "
        "the same grid coloured by WHAT each (k, threshold) combination allocates — "
        "categorical, because the interesting fact is the regime, not the share. C: the "
        "selected configuration scored in sample beside the nested out-of-fold score of "
        "the same selection procedure."
    ),
    goal=(
        "In A, a router is only worth building if its trace sits ABOVE the dashed mixture "
        "line — anything on or below it is reproducible by flipping a weighted coin "
        "between two fixed policies, with no embeddings, no index and no k. In B, look "
        "for how much of the grid is mixed at all: the degenerate bands are the sweep "
        "reporting a fixed policy's number under a routing label. In C, read the gap: it "
        "is how much of the in-sample optimum is selection optimism rather than skill."
    ),
    definitions=(
        (
            "out-of-fold",
            "scored on a fold whose tasks were absent from BOTH the neighbour index and "
            "the selection that picked the configuration",
        ),
        (
            "mixture line",
            "the cost/quality reachable by splitting tasks between two fixed policies",
        ),
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
        "The k grid is log-spaced. A uniform grid spends nearly all of its cells inside one "
        "regime and reports the same number on half of them.",
    ),
    limitations=(
        "Folds split TASKS, not repositories, so an out-of-fold task can still sit next to a "
        "sibling task from the same repo — this is a lower bound on optimism, not an "
        "estimate of transfer to a new codebase (see embedding_signal.png's cross-repo panel).",
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


@dataclass(frozen=True)
class Grid:
    """The swept axes. One object so the grid cannot drift between passes."""

    ks: tuple[int, ...]
    thresholds: tuple[float, ...]
    min_samples: tuple[int, ...]

    @property
    def n_cells(self) -> int:
        return len(self.ks) * len(self.thresholds) * len(self.min_samples)


def k_grid(n_tasks: int) -> tuple[int, ...]:
    """The log-spaced k values a corpus of this size can support."""
    usable = tuple(k for k in K_GRID if k <= max(2, n_tasks - 1))
    return usable or (2,)


def _ranked_neighbours(
    sims_row: np.ndarray, query_idx: int, allowed: np.ndarray | None
) -> list[int]:
    """Every admissible neighbour, most similar first — the query and blocked tasks gone."""
    sims = np.array(sims_row, dtype=float, copy=True)
    sims[query_idx] = -1.0
    if allowed is not None:
        blocked = np.ones(len(sims), dtype=bool)
        blocked[allowed] = False
        sims[blocked] = -np.inf
    ranked = np.argsort(sims)[::-1]
    # -inf marks a task outside the allowed index and -1.0 the query itself; both are
    # sentinels, never real neighbours, so they must not reach the vote.
    return [int(i) for i in ranked if np.isfinite(sims[i]) and sims[i] > -1.0]


def _vote_counts(
    neighbours: list[int], task_ids: list[str], results_map: dict[str, dict]
) -> dict[str, list[int]]:
    """``model -> [observations, passes]`` over a neighbourhood."""
    counts: dict[str, list[int]] = {}
    for nidx in neighbours:
        for model_name, outcome in results_map[task_ids[nidx]].items():
            seen = counts.setdefault(model_name, [0, 0])
            seen[0] += 1
            seen[1] += 1 if outcome.get("pass", False) else 0
    return counts


def _choose_model(
    counts: dict[str, list[int]],
    costs: dict[str, float],
    success_rate_thresh: float,
    min_samples: int,
    cheapest: str,
) -> str:
    """The cheapest model the neighbourhood endorses, else the cheapest model overall."""
    best_key: tuple[float, int] | None = None
    best_model = cheapest
    for model_name, (n_obs, n_pass) in counts.items():
        if n_obs < min_samples or n_pass / n_obs < success_rate_thresh:
            continue
        key = (costs.get(model_name, 0.0), -n_obs)
        if best_key is None or key < best_key:
            best_key, best_model = key, model_name
    return best_model


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
    # sweep from in-sample tuning into out-of-fold evaluation. ``sims_row`` accepts a
    # precomputed similarity row (the whole grid shares one matrix).
    row = (
        sims_row
        if sims_row is not None
        else cosine_similarity(features[query_idx : query_idx + 1], features).flatten()
    )
    nearest = _ranked_neighbours(np.asarray(row, dtype=float), query_idx, allowed)[:k]
    costs = {m: model_cost(m, matrix) for m in matrix["models"]}
    cheapest = min(costs, key=lambda m: costs[m]) if costs else ""
    chosen = _choose_model(
        _vote_counts(nearest, task_ids, results_map),
        costs,
        success_rate_thresh,
        min_samples,
        cheapest,
    )
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
    """Deterministic fold id per task — the partition the outer-loop CV runs over."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n) % max(1, folds))


def fixed_policy(model: str, task_ids: list[str], results_map: dict[str, dict]) -> dict:
    """Score 'send every task to this one model' — the endpoints of the mixture line."""
    passes = 0
    cost = 0.0
    scored = 0
    for tid in task_ids:
        outcome = results_map.get(tid, {}).get(model)
        if outcome is None:
            continue
        scored += 1
        passes += 1 if outcome.get("pass", False) else 0
        cost += plot_style.row_real_cost(outcome)
    return {
        "model": model,
        "n_scored": scored,
        "AvgPerf%": round(passes / scored * 100, 2) if scored else 0.0,
        "TotalCost": round(cost, 6),
    }


@dataclass
class _Accumulator:
    """Running totals for one hyperparameter cell over the tasks it is scored on."""

    passes: int = 0
    cost: float = 0.0
    reward: float = 0.0
    scored: int = 0
    alloc: dict[str, int] = field(default_factory=dict)


def _row_from(  # noqa: PLR0913
    cell: tuple[int, float, int],
    acc: _Accumulator,
    n_tasks: int,
    frontier_model: str,
    cheapest: str,
    fold: str,
) -> dict:
    """One CSV/figure row. ``_alloc`` stays out of the CSV; pooling needs the counts."""
    k, thresh, min_samples = cell
    scored = acc.scored
    return {
        "fold": fold,
        "k": k,
        "success_rate_thresh": thresh,
        "min_samples": min_samples,
        "n_scored": scored,
        "n_passed": acc.passes,
        "n_excluded": n_tasks - scored,
        "AvgPerf%": round(acc.passes / scored * 100, 2) if scored else 0.0,
        "TotalCost": round(acc.cost, 6),
        "Reward": round(acc.reward, 6),
        "n_models_used": len(acc.alloc),
        # The frontier share is the degeneracy detector: a "routing optimum" that sends
        # almost everything to the priciest model IS Always-Frontier, and no reward number
        # on its own reveals that.
        "frontier_share": round(acc.alloc.get(frontier_model, 0) / scored, 4) if scored else 0.0,
        "cheapest_share": round(acc.alloc.get(cheapest, 0) / scored, 4) if scored else 0.0,
        "_alloc": dict(acc.alloc),
    }


@dataclass(frozen=True)
class Split:
    """Which tasks a pass scores, and which tasks its neighbour index may not see."""

    fold_ids: np.ndarray | None = None
    eval_folds: tuple[int, ...] | None = None
    blocked_folds: tuple[int, ...] = ()
    label: str = ""

    def allowed_for(self, fold: int) -> np.ndarray | None:
        """Index membership for a task in ``fold``: never its own fold, never a blocked one."""
        if self.fold_ids is None:
            return None
        excluded = np.isin(self.fold_ids, (fold, *self.blocked_folds))
        return np.flatnonzero(~excluded)

    def scores(self, fold: int) -> bool:
        return self.eval_folds is None or fold in self.eval_folds


def sweep_grid(  # noqa: PLR0913
    task_ids: list[str],
    features: np.ndarray,
    results_map: dict[str, dict],
    matrix: dict,
    grid: Grid,
    split: Split,
    sims: np.ndarray | None = None,
    frontier_model: str = "",
) -> list[dict]:
    """Every cell of the grid on one split, in a single pass over the tasks."""
    # Tasks outer, cells inner: the neighbour ranking and the vote tally depend only on
    # (task, k), so evaluating all thresholds and min_samples against one tally costs a
    # dict lookup instead of re-ranking 175 tasks 25 times.
    gamma = config.gamma()
    costs = {m: model_cost(m, matrix) for m in matrix["models"]}
    cheapest = min(costs, key=lambda m: costs[m]) if costs else ""
    ks = sorted(grid.ks)
    cells: dict[tuple[int, float, int], _Accumulator] = {
        (k, t, ms): _Accumulator() for k in ks for t in grid.thresholds for ms in grid.min_samples
    }
    n = len(task_ids)
    allowed_cache: dict[int, np.ndarray | None] = {}

    for i in range(n):
        fold = int(split.fold_ids[i]) if split.fold_ids is not None else 0
        if not split.scores(fold):
            continue
        if fold not in allowed_cache:
            allowed_cache[fold] = split.allowed_for(fold)
        row = sims[i] if sims is not None else cosine_similarity(features[i : i + 1], features)[0]
        ranked = _ranked_neighbours(np.asarray(row, dtype=float), i, allowed_cache[fold])
        outcomes = results_map[task_ids[i]]
        counts: dict[str, list[int]] = {}
        taken = 0
        for k in ks:
            while taken < min(k, len(ranked)):
                for model_name, outcome in results_map[task_ids[ranked[taken]]].items():
                    seen = counts.setdefault(model_name, [0, 0])
                    seen[0] += 1
                    seen[1] += 1 if outcome.get("pass", False) else 0
                taken += 1
            _score_cell(cells, k, grid, counts, costs, cheapest, outcomes, gamma)
    return [
        _row_from(cell, acc, n, frontier_model, cheapest, split.label)
        for cell, acc in cells.items()
    ]


def _score_cell(  # noqa: PLR0913
    cells: dict[tuple[int, float, int], _Accumulator],
    k: int,
    grid: Grid,
    counts: dict[str, list[int]],
    costs: dict[str, float],
    cheapest: str,
    outcomes: dict[str, dict],
    gamma: float,
) -> None:
    """Fold one task's outcome into every (thresh, min_samples) cell at this k."""
    for thresh in grid.thresholds:
        for ms in grid.min_samples:
            chosen = _choose_model(counts, costs, thresh, ms, cheapest)
            outcome = outcomes.get(chosen)
            # A coverage-gap escalation (chosen model unmeasured on this task) is
            # UNSCORABLE — excluded from the aggregation, not imputed fail@$0.
            if outcome is None:
                continue
            acc = cells[(k, thresh, ms)]
            passed = bool(outcome.get("pass", False))
            cost = plot_style.row_real_cost(outcome)
            acc.alloc[chosen] = acc.alloc.get(chosen, 0) + 1
            acc.scored += 1
            acc.passes += 1 if passed else 0
            acc.cost += cost
            acc.reward += compute_reward(passed, cost, gamma)


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
    """Score one hyperparameter cell; with ``fold_ids`` no task votes over its own fold."""
    grid = Grid((k,), (success_rate_thresh,), (min_samples,))
    split = Split(fold_ids=fold_ids, label="" if fold_ids is None else "all")
    return sweep_grid(
        task_ids,
        features,
        results_map,
        matrix,
        grid,
        split,
        sims=sims,
        frontier_model=frontier_model,
    )[0]


def pool_folds(
    fold_rows: list[dict], n_tasks: int, frontier_model: str, cheapest: str
) -> list[dict]:
    """Sum the per-fold rows back into one out-of-fold row per cell."""
    # Every task is scored in exactly one fold, so the sum IS the cross-validated
    # estimate of that cell — each task graded with an index that never held its own fold.
    merged: dict[tuple[int, float, int], _Accumulator] = {}
    for row in fold_rows:
        cell = (row["k"], row["success_rate_thresh"], row["min_samples"])
        acc = merged.setdefault(cell, _Accumulator())
        acc.passes += int(row["n_passed"])
        acc.scored += int(row["n_scored"])
        acc.cost += float(row["TotalCost"])
        acc.reward += float(row["Reward"])
        for model_name, count in row["_alloc"].items():
            acc.alloc[model_name] = acc.alloc.get(model_name, 0) + count
    return [
        _row_from(cell, acc, n_tasks, frontier_model, cheapest, "pooled")
        for cell, acc in merged.items()
    ]


def _regime(row: dict) -> int:
    """Which of the three allocation regimes a cell sits in."""
    if row["cheapest_share"] >= _DEGENERATE_SHARE:
        return CHEAP_REGIME
    if row["frontier_share"] >= _DEGENERATE_SHARE:
        return FRONTIER_REGIME
    return MIXED_REGIME


@dataclass(frozen=True)
class SweepResult:
    """One sweep run: the grids, the nested CV, and the two fixed policies it is judged against."""

    grid: Grid
    n_tasks: int
    n_folds: int
    in_sample: list[dict]
    fold_rows: list[dict]
    pooled: list[dict]
    nested: list[dict]
    selections: list[dict]
    selected: dict
    frontier_model: str
    cheapest_model: str
    baseline_cheap: dict
    baseline_frontier: dict
    shipped: dict | None
    sensitivity: dict[str, float]
    imputed: tuple[int, int]

    @property
    def in_sample_selected(self) -> dict:
        cell = (
            self.selected["k"],
            self.selected["success_rate_thresh"],
            self.selected["min_samples"],
        )
        match = _find(self.in_sample, cell)
        return match if match is not None else self.selected

    @property
    def oof_passes(self) -> tuple[int, int]:
        """``(passes, scored)`` pooled over the nested out-of-fold rows."""
        return (
            sum(int(r["n_passed"]) for r in self.nested),
            sum(int(r["n_scored"]) for r in self.nested),
        )

    @property
    def oof_rate(self) -> float:
        passes, scored = self.oof_passes
        return passes / scored * 100 if scored else 0.0

    @property
    def optimism(self) -> float:
        return self.in_sample_selected["AvgPerf%"] - self.oof_rate


def _find(rows: list[dict], cell: tuple[int, float, int]) -> dict | None:
    k, thresh, min_samples = cell
    return next(
        (
            r
            for r in rows
            if r["k"] == k
            and r["success_rate_thresh"] == thresh
            and r["min_samples"] == min_samples
        ),
        None,
    )


def run_sweep(  # noqa: PLR0913
    task_ids: list[str],
    features: np.ndarray,
    results_map: dict[str, dict],
    matrix: dict,
    grid: Grid,
    n_folds: int,
) -> SweepResult:
    """In-sample grid, out-of-fold grid, and a nested selection per fold."""
    n = len(task_ids)
    sims = _similarity(features)
    models = matrix["models"]
    frontier_model = max(models, key=lambda m: model_cost(m, matrix)) if models else ""
    cheapest = min(models, key=lambda m: model_cost(m, matrix)) if models else ""
    fold_ids = fold_assignment(n, n_folds)
    folds = tuple(sorted({int(f) for f in fold_ids}))

    def pass_over(split: Split) -> list[dict]:
        return sweep_grid(
            task_ids,
            features,
            results_map,
            matrix,
            grid,
            split,
            sims=sims,
            frontier_model=frontier_model,
        )

    print(f"In-sample pass ({grid.n_cells} cells; every task votes over every other)")
    in_sample = pass_over(Split(label="in_sample"))

    fold_rows: list[dict] = []
    nested: list[dict] = []
    selections: list[dict] = []
    for fold in folds:
        others = tuple(f for f in folds if f != fold)
        print(f"Fold {fold}: selecting on folds {others}, scoring on fold {fold}")
        # The SELECTION never sees the scored fold — not as a scored task and not as a
        # neighbour. That is the whole fix: the previous sweep held out only the index and
        # then picked its optimum on the very rows it reported.
        inner = pass_over(
            Split(fold_ids, eval_folds=others, blocked_folds=(fold,), label=f"inner{fold}")
        )
        picked = cost_at_equal_quality(inner) or max(inner, key=lambda r: r["Reward"])
        scored = pass_over(Split(fold_ids, eval_folds=(fold,), label=str(fold)))
        fold_rows.extend(scored)
        selections.append(picked)
        outer = _find(scored, (picked["k"], picked["success_rate_thresh"], picked["min_samples"]))
        if outer is not None:
            nested.append(outer)

    pooled = pool_folds(fold_rows, n, frontier_model, cheapest)
    for row in pooled:
        row["_regime"] = _regime(row)
    selected = cost_at_equal_quality(pooled) or max(pooled, key=lambda r: r["Reward"])
    shipped_params = config.knn_params()
    shipped = _find(
        pooled,
        (
            int(shipped_params.get("k", 0)),
            float(shipped_params.get("success_rate_threshold", 0.0)),
            int(shipped_params.get("min_samples", 0)),
        ),
    )
    return SweepResult(
        grid=grid,
        n_tasks=n,
        n_folds=len(folds),
        in_sample=in_sample,
        fold_rows=fold_rows,
        pooled=pooled,
        nested=nested,
        selections=selections,
        selected=selected,
        frontier_model=frontier_model,
        cheapest_model=cheapest,
        baseline_cheap=fixed_policy(cheapest, task_ids, results_map),
        baseline_frontier=fixed_policy(frontier_model, task_ids, results_map),
        shipped=shipped,
        sensitivity=_sensitivity(pooled, grid),
        imputed=imputed_share(results_map, task_ids),
    )


def _similarity(features: np.ndarray) -> np.ndarray:
    """One similarity matrix for the whole run — recomputing it per cell was ~9x the cost."""
    unit = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    return np.asarray(unit @ unit.T)


def _sensitivity(rows: list[dict], grid: Grid) -> dict[str, float]:
    names = ("k", "success_rate_thresh", "min_samples")
    arr = np.array(
        [[r["k"], r["success_rate_thresh"], r["min_samples"], r["Reward"]] for r in rows]
    )
    values: list[list] = [list(grid.ks), list(grid.thresholds), list(grid.min_samples)]
    return dict(zip(names, _variance_explained(arr, values), strict=True))


def _slice_at(rows: list[dict], param: str, value: float) -> list[dict]:
    """Rows sitting exactly on one swept grid value of ``param``."""
    # Exact, not within a tolerance: these values come from the same grid the rows were
    # built from, and a 0.1 epsilon admitted the neighbouring 0.1-spaced level
    # (abs(0.6 - 0.5) is 0.0999...), so the panel drew a slice its own title disclaimed.
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
    # parameter: that would render the last writer's values under a title promising
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


def cells_may_carry_numbers(panel_width_in: float, n_cols: int) -> bool:
    """A heatmap cell earns printed text only when it is wide enough to read one."""
    # Structural, not a judgement call: the old two-panel heatmap printed 87 columns of
    # digits into a 7-inch panel (0.08 in per column) and every one of them was noise.
    # Under this floor the fill has to carry the value alone, which is why the regime map
    # is categorical with a legend rather than 55 printed shares.
    return n_cols > 0 and panel_width_in / n_cols >= _MIN_IN_PER_COL


def _mark_regime_cells(ax: Axes, codes: np.ndarray) -> None:
    """One letter per cell — the legend's key, repeated where colour alone would carry it."""
    for yi in range(codes.shape[0]):
        for xi in range(codes.shape[1]):
            code = codes[yi, xi]
            if np.isnan(code):
                continue
            ax.text(
                xi,
                yi,
                _REGIME_KEYS[int(code)],
                ha="center",
                va="center",
                fontsize=6.5,
                color="#222222" if int(code) != MIXED_REGIME else "white",
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


def trace_points(res: SweepResult) -> list[dict]:
    """Per k, the cost-at-equal-quality pick over that k's thresholds — the trace in panel A."""
    out: list[dict] = []
    for k in sorted(res.grid.ks):
        best = cost_at_equal_quality([r for r in res.pooled if r["k"] == k])
        if best is not None:
            out.append(best)
    return out


def mixture_quality(res: SweepResult, cost: float) -> float:
    """The pass rate a weighted coin flip between the two fixed policies buys at ``cost``."""
    cheap, front = res.baseline_cheap, res.baseline_frontier
    span = front["TotalCost"] - cheap["TotalCost"]
    if span <= 0:
        return float(cheap["AvgPerf%"])
    share = (cost - cheap["TotalCost"]) / span
    return float(cheap["AvgPerf%"] + share * (front["AvgPerf%"] - cheap["AvgPerf%"]))


def best_over_mixture(res: SweepResult) -> tuple[dict | None, float]:
    """The trace point that beats the mixture line by the most, and by how much."""
    # This IS the figure's claim: a router that never clears the line is a coin flip
    # between two fixed policies, and the size of the best gain is its whole value.
    best: dict | None = None
    gain = 0.0
    for row in trace_points(res):
        margin = row["AvgPerf%"] - mixture_quality(res, float(row["TotalCost"]))
        if best is None or margin > gain:
            best, gain = row, margin
    return best, gain


def _draw_trace(ax: Axes, res: SweepResult) -> None:
    """Panel A: what the sweep buys per dollar, against what a coin flip buys."""
    cheap, front = res.baseline_cheap, res.baseline_frontier
    mix = np.linspace(0.0, 1.0, 200)
    ax.plot(
        cheap["TotalCost"] + mix * (front["TotalCost"] - cheap["TotalCost"]),
        cheap["AvgPerf%"] + mix * (front["AvgPerf%"] - cheap["AvgPerf%"]),
        color=_MIXTURE_COLOR,
        linestyle="--",
        linewidth=1.6,
        label="mixture of the two fixed policies",
    )
    trace = trace_points(res)
    ax.plot(
        [r["TotalCost"] for r in trace],
        [r["AvgPerf%"] for r in trace],
        "-o",
        color=_TRACE_COLOR,
        markersize=4.5,
        linewidth=1.5,
        label="sweep, best threshold per k",
    )
    ax.scatter(
        [cheap["TotalCost"], front["TotalCost"]],
        [cheap["AvgPerf%"], front["AvgPerf%"]],
        marker="s",
        s=46,
        color=_MIXTURE_COLOR,
        zorder=5,
        label="always-cheapest · always-frontier",
    )
    if res.shipped is not None:
        ax.scatter(
            [res.shipped["TotalCost"]],
            [res.shipped["AvgPerf%"]],
            marker="*",
            s=190,
            color=_SHIPPED_COLOR,
            zorder=6,
            label=(
                f"shipped: k={res.shipped['k']}, thresh={res.shipped['success_rate_thresh']:.1f}"
            ),
        )
    _label_trace(ax, res, trace)
    ax.set_xscale("log")
    # Room at both ends so the two end labels and the frontier marker are not pressed
    # against the spines — a label half off the axes reads as a different number.
    # DERIVED FROM WHAT THE PANEL DRAWS, not from the two fixed policies alone: a trace point
    # or the shipped star outside [cheap, frontier] would otherwise be clipped off an axis
    # that still looks complete. Same rule as the y floor in cost_quality_frontier.
    drawn = [float(cheap["TotalCost"]), float(front["TotalCost"])]
    drawn += [float(r["TotalCost"]) for r in trace]
    if res.shipped is not None:
        drawn.append(float(res.shipped["TotalCost"]))
    drawn = [c for c in drawn if c > 0]
    ax.set_xlim(min(drawn) * 0.72, max(drawn) * 2.1)
    ax.set_xlabel("total cost over the suite (USD, log scale)", fontsize=9)
    ax.set_ylabel("out-of-fold pass rate (%)", fontsize=9)
    ax.grid(True, which="both", alpha=0.18, linewidth=0.6)
    ax.tick_params(labelsize=8)
    # Upper left: the trace climbs to the right, so the legend cannot sit over it or over
    # the mixture line it is there to explain.
    ax.legend(fontsize=6.8, loc="upper left", framealpha=0.92)
    plot_frame.panel_label(ax, "A · cost-quality trace over k")


def _label_trace(ax: Axes, res: SweepResult, trace: list[dict]) -> None:
    """Name the two k values worth naming: the best gain and the top of the range."""
    # Not all 11: the small-k points pile into one corner, and no leader lines and no
    # rotation are allowed to dig them out. A k that lands on its neighbour is a k whose
    # regime the reader can already see from the two it sits between.
    best, _gain = best_over_mixture(res)
    marks = [(best, (-7.0, 7.0), "right", "bottom")] if best is not None else []
    if trace and (best is None or trace[-1]["k"] != best["k"]):
        marks.append((trace[-1], (7.0, -5.0), "left", "top"))
    for row, (dx, dy), ha, va in marks:
        ax.annotate(
            f"k={row['k']}",
            (row["TotalCost"], row["AvgPerf%"]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va=va,
            fontsize=7.5,
            color=_TRACE_COLOR,
        )


def _draw_regimes(ax: Axes, res: SweepResult, panel_width_in: float) -> None:
    """Panel B: which allocation each (k, threshold) produces, as three categories."""
    rows = _slice_at(res.pooled, "min_samples", res.selected["min_samples"])
    ks, thresholds, codes = _heat_grid(rows, "success_rate_thresh", "_regime")
    cmap = ListedColormap(list(_REGIME_COLORS))
    ax.imshow(
        codes,
        aspect="auto",
        cmap=cmap,
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N),
        interpolation="nearest",
    )
    if cells_may_carry_numbers(panel_width_in, len(ks)):
        _mark_regime_cells(ax, codes)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks, fontsize=7)
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f"{t:.1f}" for t in thresholds], fontsize=7.5)
    ax.set_xlabel("k  (log-spaced grid)", fontsize=9)
    ax.set_ylabel("success_rate_thresh", fontsize=9)
    ax.legend(
        handles=[
            Patch(facecolor=c, edgecolor="#444444", linewidth=0.5, label=f"{key} · {lbl}")
            for c, key, lbl in zip(_REGIME_COLORS, _REGIME_KEYS, _REGIME_LABELS, strict=True)
        ],
        fontsize=6.8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
    )
    plot_frame.panel_label(ax, f"B · allocation regime (min_samples={res.selected['min_samples']})")


def _draw_optimism(ax: Axes, res: SweepResult) -> None:
    """Panel C: the selected configuration in sample, beside the nested out-of-fold score."""
    in_row = res.in_sample_selected
    _oof_passes, oof_scored = res.oof_passes
    # A BAR IS ONLY DRAWN FOR A NON-EMPTY SUBJECT SET. With no out-of-fold row scored, the
    # rate is 0.0 over n=0 and the panel drew a zero-height bar with a "0.0% (n=0)" label —
    # an empty set rendered as a measured failure, and the optimism gap beside it as the
    # difference against one. An absent measurement is now stated, not plotted.
    bars = [
        (label, rate, n, color)
        for label, rate, n, color in (
            (
                "in-sample\n(selected config)",
                in_row["AvgPerf%"],
                int(in_row["n_scored"]),
                "#8C8C8C",
            ),
            ("out-of-fold\n(nested CV)", res.oof_rate, oof_scored, _TRACE_COLOR),
        )
        if n > 0
    ]
    for x, (_label, rate, n, color) in enumerate(bars):
        lo, hi = plot_style.wilson_interval(round(rate / 100 * n), n)
        yerr = plot_style.ci_yerr(rate / 100, lo, hi)
        ax.bar(x, rate, width=0.52, color=color, edgecolor="#333333", linewidth=0.6)
        ax.errorbar(
            x,
            rate,
            yerr=[[yerr[0] * 100], [yerr[1] * 100]],
            fmt="none",
            ecolor="#222222",
            capsize=4,
        )
        # Never under the bar: the lower whisker is there, and a label inside it reads as
        # a data point. Above the upper whisker is the only free space.
        ax.text(
            x,
            min(hi * 100 + 1.6, _PASS_RATE_MAX - 1.5),
            f"{rate:.1f}%  (n={n})",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel("pass rate (%)", fontsize=9)
    # A PASS RATE CANNOT EXCEED 100. The axis ran to 124 to park the gap label in empty sky,
    # which drew a fifth of the panel over impossible values and shrank every real difference
    # against them. The label moves into the empty COLUMN between the two bars instead — bars
    # are 0.52 wide at x=0 and x=1, so x=0.5 carries no data at any height.
    ax.set_ylim(0, _PASS_RATE_MAX)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.6)
    # The connector and the gap need BOTH measurements; with one of them missing there is no
    # gap to state, and saying so is the panel's honest output.
    if len(bars) == 2:
        ax.plot([0, 1], [b[1] for b in bars], linestyle=":", color=_MIXTURE_COLOR, linewidth=1.2)
        gap = f"optimism gap {res.optimism:+.1f} pp"
    else:
        gap = "no optimism gap: one of the two scores has no scored task"
    # In the empty COLUMN between the bars (0.52 wide at x=0 and x=1), high in the panel —
    # placed in AXES coordinates so it stays centred and inside the axes whether the panel
    # drew one bar or two.
    ax.text(
        0.5,
        0.965,
        gap,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=_MIXTURE_COLOR,
    )
    plot_frame.panel_label(ax, "C · in-sample vs out-of-fold")


def _subtitle_facts(res: SweepResult) -> tuple[str, ...]:
    sel = res.selected
    facts = [
        f"n={res.n_tasks} tasks · {res.n_folds}-fold outer CV · {res.grid.n_cells} cells "
        f"({len(res.grid.ks)} log-spaced k)",
        f"selected k={sel['k']}, thresh={sel['success_rate_thresh']:.1f}, "
        f"min_samples={sel['min_samples']} -> {sel['AvgPerf%']:.1f}% at "
        f"{plot_style.usd(float(sel['TotalCost']), 2)} out of fold",
    ]
    best, gain = best_over_mixture(res)
    if best is not None:
        facts.append(
            f"best gain over the mixture line {gain:+.1f} pp at k={best['k']}, "
            f"thresh={best['success_rate_thresh']:.1f}"
        )
    if res.shipped is not None:
        facts.append(
            f"shipped k={res.shipped['k']}, thresh={res.shipped['success_rate_thresh']:.1f} -> "
            f"{res.shipped['AvgPerf%']:.1f}% at "
            f"{plot_style.usd(float(res.shipped['TotalCost']), 2)} "
            f"({res.shipped['cheapest_share']:.0%} on {res.cheapest_model})"
        )
    return tuple(facts)


def _annotations(res: SweepResult) -> Annotations:
    """What the canvas no longer says: degeneracy, imputation, and which knob matters."""
    notes: list[str] = []
    limits: list[str] = []
    reward_best = max(res.pooled, key=lambda r: r["Reward"])
    share = float(reward_best.get("frontier_share", 0.0))
    used = int(reward_best.get("n_models_used", 0))
    degenerate = (
        f"REWARD-ARGMAX IS DEGENERATE: maximising reward (passes - gamma x cost, "
        f"gamma={config.gamma():.3g}) picks k={reward_best['k']}, "
        f"thresh={reward_best['success_rate_thresh']}, which routes {share:.0%} of tasks to "
        f"{res.frontier_model} using {used} distinct model(s). At this gamma one extra pass "
        f"is worth {1 / config.gamma():.0f} USD against a suite costing a few dollars, so "
        f"cost is nearly a no-op and the argmax escalates everything."
    )
    (limits if share > 0.5 or used <= 1 else notes).append(degenerate)

    picks = ", ".join(
        f"fold {i}: k={s['k']}/t={s['success_rate_thresh']:.1f}/m={s['min_samples']}"
        for i, s in enumerate(res.selections)
    )
    notes.append(
        f"OUTER-LOOP CV: for each of {res.n_folds} folds the configuration is chosen on the "
        f"other folds — which are also the only tasks its neighbour index may hold — and "
        f"scored on the fold left out. The per-fold picks were {picks}."
    )
    notes.append(
        f"Panel A's trace takes, per k, the cheapest cell whose out-of-fold pass rate clears "
        f"the best cell's 95% Wilson lower bound. The two fixed policies are scored on the "
        f"same corpus: always-cheapest ({res.cheapest_model}) "
        f"{res.baseline_cheap['AvgPerf%']:.1f}% at "
        f"{plot_style.usd(float(res.baseline_cheap['TotalCost']), 2)}, always-frontier "
        f"({res.frontier_model}) {res.baseline_frontier['AvgPerf%']:.1f}% at "
        f"{plot_style.usd(float(res.baseline_frontier['TotalCost']), 2)}."
    )
    # The SWEPT ends, derived. The static note used to name "(2 to 174)", which `k_grid`
    # clips to the corpus — on any corpus under 175 tasks the figure named a k it never swept.
    notes.append(
        f"The k grid actually swept runs {min(res.grid.ks)} to {max(res.grid.ks)} "
        f"({len(res.grid.ks)} values), clipped to a corpus of {res.n_tasks} tasks."
    )
    notes.append(
        _driver_sentence(
            ("success_rate_thresh", res.sensitivity["success_rate_thresh"]),
            ("k", res.sensitivity["k"]),
        )
        + f". min_samples: {_eta_phrase(res.sensitivity['min_samples'])}"
    )
    limits.extend(_data_limits(res))
    _oof_passes, oof_scored = res.oof_passes
    return Annotations(
        subtitle_facts=_subtitle_facts(res),
        caveat=_caveat(res),
        notes=tuple(notes),
        limitations=tuple(limits),
        # The figure shipped with an EMPTY `n` — the one manifest field a reader checks to see
        # what a claim is measured on. Every panel's own denominator is named here: A and B are
        # per-cell over the grid, C is per-task out of fold.
        counts=(
            ("tasks", res.n_tasks),
            ("folds", res.n_folds),
            ("grid_cells", res.grid.n_cells),
            ("in_sample_scored", int(res.in_sample_selected["n_scored"])),
            ("oof_scored", oof_scored),
        ),
    )


def _data_limits(res: SweepResult) -> list[str]:
    limits: list[str] = []
    lo, hi = min(res.grid.ks), max(res.grid.ks)
    if res.selected["k"] in (lo, hi):
        limits.append(
            f"The selected k={res.selected['k']} sits at the EDGE of the swept range "
            f"k in [{lo}, {hi}] — the optimum is not bracketed and the true peak may lie beyond it"
        )
    n_imp, n_cells = res.imputed
    if n_imp:
        # SCOPED, because this count is NOT the corpus-wide one evidence_basis.png publishes:
        # the sweep scores its own task set, so the two disagree legitimately and a reader who
        # meets the bare "in the scored matrix" reads one as a correction of the other. The
        # second clause also used to read "so it almost never can never add a failure" — two
        # half-edits of the same sentence left in place.
        per_task = n_cells // res.n_tasks if res.n_tasks else 0
        limits.append(
            f"{n_imp}/{n_cells} cells ({n_imp / n_cells:.1%}) in THE MATRIX THIS SWEEP SCORES "
            f"({res.n_tasks} tasks x {per_task} ranked models — not the corpus-wide count in "
            f"evidence_basis.png) are monotone-IMPUTED rather than measured, and the "
            f"imputation is near-exclusively pass-filling, so it can almost never add a "
            f"failure. The neighbourhood VOTES and the pass rates on this grid both read "
            f"those synthetic passes — every quality number here is biased up"
        )
    limits.append("Cost is model-price dependent — the selected cell moves when model prices move.")
    return limits


def _caveat(res: SweepResult) -> str | None:
    """One red line, only where a reader would otherwise be misled."""
    _best, gain = best_over_mixture(res)
    if gain <= 0:
        return "No k beats the two-policy mixture line: this grid holds no routing result to ship."
    n_imp, n_cells = res.imputed
    if n_imp:
        # The gain is measured THROUGH a pass-only imputation; the reader has to see both
        # numbers together or the small margin reads as clean signal.
        return (
            f"{n_imp / n_cells:.0%} of scored cells are imputed, near-all pass-filled; the "
            f"trace beats the mixture by at most {gain:.1f} pp."
        )
    return None


def plot_sweep_regimes(res: SweepResult, plot_path: Path, digest: str) -> None:
    """The three panels: what the sweep buys, what it allocates, what it is worth."""
    size = plot_frame.WIDE
    fig, axes = plot_frame.subplots(size, 1, 3)
    _draw_trace(axes[0], res)
    _draw_regimes(axes[1], res, size.width_in / 3.0)
    _draw_optimism(axes[2], res)
    _save(fig, plot_path, res, size, digest)


def _manifest_for(plot_path: Path) -> Path:
    """The half's figures.json — the committed one only when the PNG lands in its real home."""
    # A render into a tmp dir must not rewrite the committed routing manifest, so the
    # manifest is only the committed one when the plot goes where the committed plots go.
    # Keyed on the resolved directory, not on a directory NAME: a scratch dir can be called
    # `routing` too, so only THE committed home may claim the committed manifest.
    if plot_path.resolve().parent == (_REPO_ROOT / "docs/assets/figures/routing").resolve():
        return ctxmod.MANIFEST.resolve()
    return plot_path.parent / "figures.json"


def _save(
    fig: Figure, plot_path: Path, res: SweepResult, size: plot_frame.FigureSize, digest: str
) -> None:
    plot_frame.save(
        fig,
        plot_path,
        SPEC,
        extra=_annotations(res),
        provenance=plot_frame.Provenance(
            generator=_GENERATOR,
            data_digest=digest,
            manifest=_manifest_for(plot_path),
        ),
        size=size,
    )


CSV_FIELDS: Final[tuple[str, ...]] = (
    "split",
    "fold",
    "k",
    "success_rate_thresh",
    "min_samples",
    "n_scored",
    "n_passed",
    "n_excluded",
    "AvgPerf%",
    "TotalCost",
    "Reward",
    "n_models_used",
    "frontier_share",
    "cheapest_share",
)


def write_csv(path: Path, res: SweepResult) -> None:
    """The in-sample grid and every out-of-fold row, one file (gitignored render)."""
    rows: list[dict] = []
    for row in res.in_sample:
        rows.append({**row, "split": "in_sample", "fold": ""})
    for row in res.fold_rows:
        rows.append({**row, "split": "out_of_fold"})
    rows.sort(key=lambda r: (r["split"], -float(r["Reward"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _parse_args(config_path: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--output", default="benchmark/routing/reports/threshold_sweep_results.csv")
    ap.add_argument("--plot", default="docs/assets/figures/routing/sweep_regimes.png")
    ap.add_argument("--ks", nargs="+", type=int, default=None, help="k values to sweep")
    ap.add_argument("--folds", type=int, default=5, help="outer CV folds")
    ap.add_argument(
        "--thresholds", nargs="+", type=float, default=None, help="success rate thresholds to sweep"
    )
    ap.add_argument(
        "--min-samples-list", nargs="+", type=int, default=None, help="min_samples values to sweep"
    )
    return ap.parse_args()


def main(config_path: str = "benchmark/benchmark.yaml") -> None:
    """Sweep kNN hyperparameters under an outer-loop CV and report the regimes it lands in."""
    config.load(config_path)
    args = _parse_args(config_path)
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
    print(f"Loaded {len(task_ids)} tasks from {matrix_path}")
    features = np.asarray(_embed_texts(task_descs), dtype=float)
    print(f"Real jina-embedding matrix: {features.shape}")

    grid = Grid(
        ks=tuple(args.ks) if args.ks is not None else k_grid(len(task_ids)),
        thresholds=tuple(args.thresholds or (round(0.5 + i * 0.1, 1) for i in range(5))),
        min_samples=tuple(args.min_samples_list or (1, 2, 3, 4, 5)),
    )
    print(f"  k = {list(grid.ks)}")
    print(f"  success_rate_thresh = {list(grid.thresholds)}")
    print(f"  min_samples = {list(grid.min_samples)}")

    res = run_sweep(task_ids, features, matrix["results"], matrix, grid, args.folds)
    write_csv(Path(args.output), res)
    print(f"\nResults written to {args.output}")
    _report(res)

    plot_sweep_regimes(res, Path(args.plot), corpus_digest(matrix, task_ids))
    print(f"Figure saved to {args.plot}")


def _report(res: SweepResult) -> None:
    sel = res.selected
    print("\n=== SELECTION (out of fold) ===")
    print(
        f"  cost-at-equal-quality  k={sel['k']:>3} thresh={sel['success_rate_thresh']:.1f} "
        f"min_samp={sel['min_samples']} pass={sel['AvgPerf%']:.2f}% "
        f"cost=${sel['TotalCost']:.4f} frontier_share={sel['frontier_share']:.0%} "
        f"models_used={sel['n_models_used']}"
    )
    print(
        f"  in-sample {res.in_sample_selected['AvgPerf%']:.2f}% vs nested out-of-fold "
        f"{res.oof_rate:.2f}% ({res.optimism:+.2f} pp optimism)"
    )
    counts = {label: 0 for label in _REGIME_LABELS}
    for row in res.pooled:
        counts[_REGIME_LABELS[row["_regime"]]] += 1
    print("\n=== REGIMES (out of fold, pooled) ===")
    for label, count in counts.items():
        print(f"  {label:32} {count:>4}/{len(res.pooled)}")
    print("\n=== SENSITIVITY (out of fold) ===")
    for name, eta in res.sensitivity.items():
        print(f"  {name}: eta^2 = {eta:.4f}")


if __name__ == "__main__":
    main()
