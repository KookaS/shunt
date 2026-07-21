"""Direct-Method matrix replay of the *production* exploration policy — no live spend."""

# ``results.csv`` is a near-dense (task x model) grid of MEASURED (pass, cost) outcomes.
# On a fully-dense sub-slice a candidate policy is evaluated EXACTLY: look up the model
# the policy picks per task, read the recorded outcome, average. No importance weighting
# is needed while every (task, chosen-model) cell is present — that density is what makes
# the Direct Method unbiased here, so ``dense_slice`` finds the slice first and every
# replay reports the cells it had to skip.
#
# The policy under test is imported from ``shunt.router`` (ThompsonSampler,
# ExplorationBudget, ConservativeGate, RouterEngine); the selection algorithm is never
# reimplemented here. One signal IS replay-side: the ``downshift`` flag fed back to the
# conservative gate, which live code derives from the sampler's own greedy pick and this
# replay approximates from the exploit-only arm's choice (see ``replay``).
#
# WHAT THIS DOES NOT MEASURE: the outcome matrix is static, so an exploratory pull can
# never improve a later decision. Exploration's *learning* benefit is structurally zero
# here — this measures its short-run cost and regret only, the pessimistic side of the
# ledger, not a verdict on whether exploration is worth it.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import hnswlib
import numpy as np

from benchmark.routing.strategies.knn import (
    _BUNDLED_MODEL_CONFIG,
    MatrixOutcomeIndex,
    _DummySessionManager,
    _embed_texts,
    _LookupEmbedder,
)
from shunt.models.config import ModelPool
from shunt.router.budget import ConservativeGate, ExplorationBudget
from shunt.router.cold_start import ColdStartStrategy
from shunt.router.engine import RouterEngine
from shunt.router.exploration import ThompsonSampler
from shunt.router.policy import ExplorationPolicy, KnnPolicy
from shunt.router.selection import SelectionRule

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class DenseSlice:
    """The largest (tasks x models) sub-grid of the matrix with no missing cell."""

    tasks: list[str]
    models: list[str]
    total_cells_available: int  # cells present anywhere in the matrix
    total_cells_possible: int  # len(all tasks) * len(all models)

    @property
    def n_cells(self) -> int:
        return len(self.tasks) * len(self.models)

    @property
    def matrix_density(self) -> float:
        """Density of the FULL (task x model) matrix, not of the slice (which is 1.0)."""
        if self.total_cells_possible == 0:
            return 0.0
        return self.total_cells_available / self.total_cells_possible


@dataclass(frozen=True)
class Decision:
    """One replayed routing decision joined to its MEASURED outcome."""

    task: str
    model: str
    reason: str
    passed: bool
    cost: float
    is_exploratory: bool


@dataclass(frozen=True)
class RunOutcome:
    """One replay pass (one seed) over the task stream."""

    decisions: list[Decision]
    missing: list[tuple[str, str]]  # (task, model) cells the policy chose but we never ran
    explore_spend: float  # MEASURED cost of exploratory decisions (the Direct-Method quantity)
    exploit_spend: float  # MEASURED cost of exploit decisions
    budget_explore_ratio: float = 0.0  # the cap's OWN counter (neighborhood costs, not measured)

    @property
    def total_spend(self) -> float:
        return self.explore_spend + self.exploit_spend

    @property
    def explore_ratio(self) -> float:
        """Measured exploratory / exploit spend WITHIN this run.

        Not directly comparable to ``explore_budget_frac``: the cap counts
        confidence-weighted neighborhood costs, this counts realized ones.
        """
        return self.explore_spend / self.exploit_spend if self.exploit_spend > 0 else 0.0


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a percentile bootstrap CI."""

    value: float
    lo: float
    hi: float


@dataclass(frozen=True)
class ReplayReport:
    """Baseline (exploit-only) vs exploit+exploration on the same dense slice."""

    slice_: DenseSlice
    n_seeds: int
    baseline_pass_rate: Estimate
    baseline_cost: Estimate
    exploration_pass_rate: Estimate
    exploration_cost: Estimate
    # Paired per-task differences (exploration - baseline). These, not the two overlapping
    # marginal CIs above, are what the slice has the power to resolve.
    pass_delta: Estimate
    cost_delta: Estimate
    explore_ratio: float  # measured exploratory / exploit spend, mean over seeds
    explore_ratio_worst_seed: float
    # Arm total vs the exploration-OFF total — the honest overhead the ~1.4x claim is about.
    cost_multiple: float
    cost_multiple_worst_seed: float
    budget_explore_ratio: float  # the cap's own internal counter (neighborhood costs)
    baseline_missing: int
    exploration_missing_per_seed: float
    per_task_baseline_pass: dict[str, float]
    per_task_exploration_pass: dict[str, float]
    per_task_baseline_cost: dict[str, float]
    per_task_exploration_cost: dict[str, float]
    explore_share_by_round: list[float]  # fraction of decisions so far that were exploratory


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
def dense_slice(results: dict[str, dict[str, dict]]) -> DenseSlice:
    """Greedily find the largest fully-dense (tasks x models) sub-grid of *results*."""
    # Models are added in descending task coverage; the slice keeps whichever prefix
    # maximizes len(fully-covered tasks) * len(models). Greedy, not optimal — exact
    # maximum-biclique is NP-hard, and the greedy prefix is what the coverage report quotes.
    all_models = sorted({m for per_model in results.values() for m in per_model})
    per_task = {task: set(per_model) for task, per_model in results.items()}
    coverage = {m: sum(1 for s in per_task.values() if m in s) for m in all_models}
    order = sorted(all_models, key=lambda m: (-coverage[m], m))

    best_tasks: list[str] = []
    best_models: list[str] = []
    for k in range(1, len(order) + 1):
        chosen = order[:k]
        covered = sorted(t for t, s in per_task.items() if set(chosen) <= s)
        if len(covered) * k > len(best_tasks) * len(best_models):
            best_tasks, best_models = covered, list(chosen)

    return DenseSlice(
        tasks=best_tasks,
        models=sorted(best_models),
        total_cells_available=sum(len(s) for s in per_task.values()),
        total_cells_possible=len(per_task) * len(all_models),
    )


def restrict_matrix(matrix: dict, tasks: Sequence[str], models: Sequence[str]) -> dict:
    """A copy of *matrix* whose results/tasks are confined to the slice."""
    keep_models = set(models)
    results = {
        t: {m: cell for m, cell in matrix["results"][t].items() if m in keep_models}
        for t in tasks
        if t in matrix["results"]
    }
    restricted = dict(matrix)
    restricted["results"] = results
    restricted["tasks"] = {t: matrix.get("tasks", {}).get(t, {}) for t in results}
    return restricted


# ---------------------------------------------------------------------------
# Engine construction — the SHIPPED classes, never a reimplementation
# ---------------------------------------------------------------------------
def _build_index(matrix: dict) -> tuple[list[str], list[str], np.ndarray, hnswlib.Index]:
    tasks = sorted(matrix["results"])
    descriptions = [matrix.get("tasks", {}).get(t, {}).get("description", t) for t in tasks]
    embeddings = _embed_texts(descriptions)
    index = hnswlib.Index(space="cosine", dim=embeddings.shape[1])
    index.init_index(max_elements=len(tasks), ef_construction=100, M=16)
    # num_threads=1 pins the neighbour graph so a replay is bit-reproducible.
    index.add_items(embeddings, np.arange(len(tasks)), num_threads=1)
    index.set_ef(50)
    return tasks, descriptions, embeddings, index


def build_engine(
    matrix: dict,
    knn: KnnPolicy,
    exploration: ExplorationPolicy | None,
    seed: int,
) -> tuple[RouterEngine, ExplorationBudget | None]:
    """Wire a RouterEngine over *matrix* with the production exploration collaborators.

    Returns the engine and the ExplorationBudget instance (None when exploration is off)
    so the caller can read the realized explore/exploit spend split afterwards.
    """
    tasks, descriptions, embeddings, index = _build_index(matrix)
    outcome_index = MatrixOutcomeIndex(
        task_ids=tasks, embeddings=embeddings, index=index, matrix=matrix
    )
    budget = ExplorationBudget(exploration.explore_budget_frac) if exploration else None
    gate = ConservativeGate(exploration.conservative_alpha) if exploration else None
    sampler = (
        ThompsonSampler(
            np.random.default_rng(seed), exploration.prior_alpha, exploration.prior_beta
        )
        if exploration
        else None
    )
    engine = RouterEngine(
        model_pool=ModelPool(_BUNDLED_MODEL_CONFIG),
        session_manager=_DummySessionManager(),
        outcome_index=outcome_index,
        embedder=_LookupEmbedder(dict(zip(descriptions, list(embeddings), strict=True))),
        selection_rule=SelectionRule(
            min_success_rate=knn.success_rate_threshold, min_samples=knn.min_samples
        ),
        cold_start_strategy=ColdStartStrategy(threshold_tier2=0, threshold_tier1=0),
        cold_start_threshold=0,
        exploration=exploration,
        sampler=sampler,
        budget=budget,
        conservative_gate=gate,
        neighbor_k=knn.k,
    )
    return engine, budget


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
_EXPLORATORY_REASONS = frozenset({"exploration"})


def replay(
    matrix: dict,
    knn: KnnPolicy,
    exploration: ExplorationPolicy | None,
    seed: int,
    baseline_choice: dict[str, str] | None = None,
) -> RunOutcome:
    """Replay one pass of the policy over the matrix, reading MEASURED outcomes."""
    # The budget and the conservative gate are stateful, so task order matters; it is a
    # seeded shuffle. `baseline_choice` (the exploit-only pick per task) supplies the
    # replay's *downshift* approximation: an EXPLORATORY decision whose measured cost came
    # in below the exploit arm's pick for the same task. Live code instead compares the
    # sampler's own greedy pick on neighborhood costs, so the two can disagree; without
    # `baseline_choice` no slack is banked and the gate stays shut.
    engine, budget = build_engine(matrix, knn, exploration, seed)
    tasks = sorted(matrix["results"])
    rng = np.random.default_rng(seed)
    rng.shuffle(tasks)

    decisions: list[Decision] = []
    missing: list[tuple[str, str]] = []
    for task in tasks:
        description = matrix.get("tasks", {}).get(task, {}).get("description", task)
        model, reason, _prov = engine.decide(session_id=task, prompt_text=description)
        cell = matrix["results"].get(task, {}).get(model)
        if cell is None:
            # Never impute: the policy left the dense slice, so this decision is unscorable.
            missing.append((task, model))
            continue
        passed = bool(cell.get("pass", False))
        cost = float(cell.get("cost", 0.0))
        is_exploratory = reason in _EXPLORATORY_REASONS
        decisions.append(
            Decision(
                task=task,
                model=model,
                reason=reason,
                passed=passed,
                cost=cost,
                is_exploratory=is_exploratory,
            )
        )
        if baseline_choice is not None and is_exploratory:
            # Only exploratory downshifts bank slack — banking exploit outcomes too would
            # open the gate far earlier than production does.
            base_cell = matrix["results"].get(task, {}).get(baseline_choice.get(task, ""))
            base_cost = float(base_cell.get("cost", 0.0)) if base_cell else None
            engine.record_outcome(
                downshift=base_cost is not None and cost < base_cost, success=passed
            )

    return RunOutcome(
        decisions=decisions,
        missing=missing,
        explore_spend=sum(d.cost for d in decisions if d.is_exploratory),
        exploit_spend=sum(d.cost for d in decisions if not d.is_exploratory),
        budget_explore_ratio=budget.explore_ratio if budget is not None else 0.0,
    )


def bootstrap_ci(
    values: Sequence[float], seed: int, n_resamples: int = 2000, alpha: float = 0.05
) -> Estimate:
    """Percentile bootstrap CI for the mean of *values* (empty -> all-NaN estimate)."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    return Estimate(
        value=float(arr.mean()),
        lo=float(np.percentile(means, 100 * alpha / 2)),
        hi=float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def _per_task(outcomes: Sequence[RunOutcome]) -> tuple[dict[str, float], dict[str, float]]:
    """Mean pass and mean cost per task across replay passes (seeds)."""
    passes: dict[str, list[float]] = {}
    costs: dict[str, list[float]] = {}
    for outcome in outcomes:
        for d in outcome.decisions:
            passes.setdefault(d.task, []).append(1.0 if d.passed else 0.0)
            costs.setdefault(d.task, []).append(d.cost)
    return (
        {t: float(np.mean(v)) for t, v in passes.items()},
        {t: float(np.mean(v)) for t, v in costs.items()},
    )


def _explore_share_by_round(outcomes: Sequence[RunOutcome]) -> list[float]:
    """Running fraction of decisions that were exploratory, averaged over seeds."""
    if not outcomes:
        return []
    length = min(len(o.decisions) for o in outcomes)
    shares: list[float] = []
    for i in range(length):
        per_seed = [
            sum(1 for d in o.decisions[: i + 1] if d.is_exploratory) / (i + 1) for o in outcomes
        ]
        shares.append(float(np.mean(per_seed)))
    return shares


def evaluate(
    matrix: dict,
    knn: KnnPolicy | None = None,
    exploration: ExplorationPolicy | None = None,
    n_seeds: int = 20,
    seed: int = 1234,
    n_resamples: int = 2000,
) -> ReplayReport:
    """Compare exploit-only against exploit+exploration on the matrix's dense slice."""
    knn = knn or KnnPolicy()
    exploration = exploration or ExplorationPolicy()
    slice_ = dense_slice(matrix["results"])
    sub = restrict_matrix(matrix, slice_.tasks, slice_.models)

    # Baseline is deterministic given the matrix, so one pass is the whole estimate.
    baseline = replay(sub, knn, exploration=None, seed=seed)
    baseline_choice = {d.task: d.model for d in baseline.decisions}

    runs = [
        replay(sub, knn, exploration, seed=seed + i, baseline_choice=baseline_choice)
        for i in range(n_seeds)
    ]

    base_pass, base_cost = _per_task([baseline])
    exp_pass, exp_cost = _per_task(runs)
    tasks = sorted(set(base_pass) & set(exp_pass))
    # Bootstrap seeds are offset far from the replay seeds so the two streams never alias.
    bs = seed + 10_000

    baseline_total = baseline.total_spend
    multiples = [r.total_spend / baseline_total if baseline_total > 0 else 1.0 for r in runs]

    return ReplayReport(
        slice_=slice_,
        n_seeds=n_seeds,
        baseline_pass_rate=bootstrap_ci([base_pass[t] for t in tasks], bs, n_resamples),
        baseline_cost=bootstrap_ci([base_cost[t] for t in tasks], bs + 1, n_resamples),
        exploration_pass_rate=bootstrap_ci([exp_pass[t] for t in tasks], bs + 2, n_resamples),
        exploration_cost=bootstrap_ci([exp_cost[t] for t in tasks], bs + 3, n_resamples),
        pass_delta=bootstrap_ci([exp_pass[t] - base_pass[t] for t in tasks], bs + 4, n_resamples),
        cost_delta=bootstrap_ci([exp_cost[t] - base_cost[t] for t in tasks], bs + 5, n_resamples),
        explore_ratio=float(np.mean([r.explore_ratio for r in runs])),
        explore_ratio_worst_seed=max((r.explore_ratio for r in runs), default=0.0),
        cost_multiple=float(np.mean(multiples)),
        cost_multiple_worst_seed=max(multiples, default=1.0),
        budget_explore_ratio=float(np.mean([r.budget_explore_ratio for r in runs])),
        baseline_missing=len(baseline.missing),
        exploration_missing_per_seed=float(np.mean([len(r.missing) for r in runs])),
        per_task_baseline_pass=base_pass,
        per_task_exploration_pass=exp_pass,
        per_task_baseline_cost=base_cost,
        per_task_exploration_cost=exp_cost,
        explore_share_by_round=_explore_share_by_round(runs),
    )
