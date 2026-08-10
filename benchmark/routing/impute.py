"""Pure monotone-ladder imputation completing the routing outcome matrix in memory."""

# The benchmark collected asymmetric coverage (cheap/mid ran on nearly every task, the
# frontier on a discriminating subset), so strategies were scored on different task sets.
# This module completes the real cache into an equal-coverage matrix under an explicit
# MONOTONICITY AXIOM over the DERIVED per-model capability rank (config.capability_rank,
# weakest -> strongest): for a task there is a single crossover rank `tau` where every
# model at-or-above tau passes and every model below fails. From a task's real cells we
# read s* (lowest observed PASS rank) and f* (highest observed FAIL rank) and impute the rest.
#
# Conservative by design (load-bearing — do not soften): imputation grants the
# always-frontier baseline free quality at frontier cost on every task a cheaper model
# solved (it imputes the strongest model as also passing). So if the axiom is WRONG for a
# task, the baseline's true quality is *lower* than imputed, which makes the router's relative
# value *larger*, not smaller. A broken assumption strengthens the thesis; we impute only
# to keep exploration cost under the budget cap, never to flatter the router.
#
# Observed truth always wins: a real cell is never overwritten by an imputed value, even
# when it contradicts the axiom — the contradicted region is left UNKNOWN and the task is
# emitted as a Violation for the caller to count and stress-test.
#
# PURE FUNCTION — no I/O. It never reads or writes results.csv; the completion is
# recomputed for free on every run. Imputed cost is a point estimate over
# MEASURED real_cost medians (never a fabricated proxy) and every imputed cell is flagged
# imputed=True so a reader can separate measured from imputed spend.

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from benchmark.config import CapabilityRank
from benchmark.routing import censoring

# Only used when no ranked model has a measured cost median to borrow (should not happen
# for an enabled model); keeps an imputed cost strictly positive rather than fabricated.
_COST_FLOOR: float = 1e-6


@dataclass(frozen=True)
class ImputedCell:
    """One completed outcome — real (``imputed=False``) or axiom-implied."""

    passed: bool
    cost: float  # real_cost if measured; per-model/rank-neighbour median real_cost if imputed
    imputed: bool
    source_model: str  # the observed model whose outcome implied this cell (itself if real)

    def to_cell(self) -> dict:
        """Serialize to the plain cell dict that summary/report consumers read."""
        return {
            "pass": self.passed,
            "cost": self.cost,
            "real_cost": self.cost,
            "imputed": self.imputed,
            "source_model": self.source_model,
        }


@dataclass(frozen=True)
class Violation:
    """A task whose real cells contradict monotonicity (a higher-ranked model failed below
    a lower-ranked model that passed)."""

    task_id: str
    pass_model: str  # a lower-ranked model observed to PASS
    fail_model: str  # a higher-ranked model observed to FAIL


@dataclass(frozen=True)
class ImputedMatrix:
    """The completed matrix plus provenance (never persisted)."""

    matrix: dict  # results[tid][model] -> cell dict (same shape as config.load_matrix())
    violations: list[Violation]
    n_real: int
    n_imputed: int
    n_unknown: int  # cells left UNKNOWN (legacy gap / contradiction region) — excluded downstream
    tau: dict[str, str | None]  # per-task crossover MODEL (None = no observed pass / unsolvable)
    n_multi_observed: int = 0  # tasks with >=2 observed ranked models (violation-rate denominator)
    # Tasks with a crossover established (ZERO UNKNOWN cells) — the complete-only analysis set.
    complete: frozenset[str] = field(default_factory=frozenset)

    @property
    def incomplete(self) -> frozenset[str]:
        """Tasks with an UNKNOWN band still open — excluded from analysis entirely."""
        return frozenset(self.matrix) - self.complete


def violation_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Monotonicity violation rate ``k/n`` with a Wilson score CI (point, lo, hi).

    ``n`` is the count of tasks with >=2 observed ranked models (``n_multi_observed``);
    a task with <2 observations can't contradict the axiom, so it isn't in the base.
    """
    if n <= 0:
        return (0.0, 0.0, 0.0)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * ((phat * (1 - phat) / n + z2 / (4 * n * n)) ** 0.5)
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (phat, max(0.0, lo), min(1.0, hi))


def _real_cost(cell: dict) -> float:
    """A measured cell's real spend (falls back to the legacy ``cost`` column)."""
    val = cell.get("real_cost")
    if val is None:
        val = cell.get("cost", 0.0)
    return float(val or 0.0)


def _medians(groups: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(median(vals)) for key, vals in groups.items() if vals}


@dataclass(frozen=True)
class _CostModel:
    """Median measured ``real_cost`` per model, split by pass/fail, with a rank-neighbour
    fallback for a model that was never measured (borrow the nearest ranked model's median)."""

    per_model_pass: dict[str, float]
    per_model_fail: dict[str, float]
    order: tuple[str, ...]  # weakest -> strongest; the rank-neighbour search space

    def cost(self, model: str, passed: bool) -> float:
        """Imputed cost: this model's median, else the nearest ranked model's, else the floor."""
        per_model = self.per_model_pass if passed else self.per_model_fail
        if model in per_model:
            return per_model[model]
        if model in self.order:
            i = self.order.index(model)
            # Expand outward; at equal distance prefer the lower-ranked (cheaper) neighbour.
            for dist in range(1, len(self.order)):
                for j in (i - dist, i + dist):
                    if 0 <= j < len(self.order) and self.order[j] in per_model:
                        return per_model[self.order[j]]
        return _COST_FLOOR


def is_zero_work(cell: dict) -> bool:
    """True iff the cell POSITIVELY records no work: calls and cost both present, both 0."""
    # Distinct from CENSORED. A censored cell ran, burned tokens and cost money, and only its
    # pass/fail is unknown; a zero-work cell never executed — the row is an artifact of an
    # aborted collection, not a measurement. Only the latter is safe to re-collect.
    #
    # Both keys MUST be present. A cell that simply omits `calls`/`real_cost` proves nothing
    # about whether work happened, and defaulting the absent field to 0 would read "I have no
    # record" as "it never ran" — the reasoning error that turns a missing field into a
    # fabricated finding. Unknown is not zero, so an incomplete cell is NOT zero-work.
    if "calls" not in cell or ("real_cost" not in cell and "cost" not in cell):
        return False
    return int(cell.get("calls", 0) or 0) == 0 and _real_cost(cell) == 0.0


def is_non_observation(cell: dict) -> bool:
    """True iff a cell is a NON-observation (censored / zero-work) — not a real $0 measurement."""
    # A CENSORED cell (step/wall/abandon limit — subsumes the old timeout_flag check), or a cell
    # that made zero priced calls AND recorded $0, is a non-event: no model attempted AND completed
    # observed work. Folding its cost into a pass/fail cost median mis-attributes spend whose true
    # outcome is unknown, poisoning the imputed-cost model.
    return censoring.is_censored(cell) or is_zero_work(cell)


def _build_cost_model(matrix: dict, order: tuple[str, ...]) -> _CostModel:
    """Aggregate measured ``real_cost`` medians from the real matrix (never imputed)."""
    mp_pass: dict[str, list[float]] = {}
    mp_fail: dict[str, list[float]] = {}
    for per_model in matrix.values():
        for model, cell in per_model.items():
            if is_non_observation(cell):
                continue  # timeout / zero-work rows are non-observations, not $0 observations
            cost = _real_cost(cell)
            passed = bool(cell.get("pass", False))
            (mp_pass if passed else mp_fail).setdefault(model, []).append(cost)
    return _CostModel(_medians(mp_pass), _medians(mp_fail), order)


def _observe(task_cells: dict, rank_of: dict[str, int]) -> tuple[int | None, int | None]:
    """(s*, f*) as capability RANKS from every real cell of a task (None if none)."""
    pass_ranks: list[int] = []
    fail_ranks: list[int] = []
    for model, r in rank_of.items():
        cell = task_cells.get(model)
        if cell is None:
            continue
        (pass_ranks if bool(cell.get("pass", False)) else fail_ranks).append(r)
    s_star = min(pass_ranks) if pass_ranks else None
    f_star = max(fail_ranks) if fail_ranks else None
    return s_star, f_star


def _n_observed(task_cells: dict, rank_of: dict[str, int]) -> int:
    """How many ranked models were really measured for this task."""
    return sum(1 for m in rank_of if m in task_cells)


@dataclass(frozen=True)
class _Counts:
    real: int = 0
    imputed: int = 0
    unknown: int = 0


def _impute_model(
    model: str,
    rank: int,
    s_rank: int | None,
    f_rank: int | None,
    order: tuple[str, ...],
    costs: _CostModel,
) -> ImputedCell | None:
    """Axiom-implied cell for a model with no real observation (None = UNKNOWN)."""
    # A model both at-or-above s* AND at-or-below f* sits in the contradicted region of a
    # violating task — left UNKNOWN, never imputed (observed truth wins). A model strictly
    # between f* and s* (a legacy coverage gap) is likewise UNKNOWN.
    pass_ok = s_rank is not None and rank >= s_rank
    fail_ok = f_rank is not None and rank <= f_rank
    if pass_ok and fail_ok:
        return None
    if pass_ok:
        return ImputedCell(True, costs.cost(model, True), True, order[s_rank])  # type: ignore[index]
    if fail_ok:
        return ImputedCell(False, costs.cost(model, False), True, order[f_rank])  # type: ignore[index]
    return None


def _complete_task(
    task_cells: dict,
    order: tuple[str, ...],
    rank_of: dict[str, int],
    s_rank: int | None,
    f_rank: int | None,
    costs: _CostModel,
) -> tuple[dict, _Counts]:
    """Every ranked model's completed cell for one task, plus real/imputed/unknown tallies."""
    out: dict[str, dict] = {}
    real = imputed = unknown = 0
    for model in order:
        existing = task_cells.get(model)
        if existing is not None:
            cell = dict(existing)
            cell["imputed"] = False
            cell["source_model"] = model
            out[model] = cell
            real += 1
            continue
        cell_obj = _impute_model(model, rank_of[model], s_rank, f_rank, order, costs)
        if cell_obj is None:
            unknown += 1
        else:
            out[model] = cell_obj.to_cell()
            imputed += 1
    return out, _Counts(real, imputed, unknown)


def complete_matrix(matrix: dict, rank: CapabilityRank) -> ImputedMatrix:
    """Complete a DEFAULT-ARM-FLATTENED matrix (results[tid][model] real cells) to equal
    coverage under the monotonicity axiom over ``rank`` — pure, never persists. The ranked
    model order is both the completion order and the source of measured cost medians.
    """
    # INPUT CONTRACT: `matrix` MUST be the default-arm-flattened view
    # (config.flatten_default_arm) — cells are {model: {pass, cost, ...}}. Feeding the raw
    # arm-nested config.load_results() output ({model: {arm: {pass, cost, ...}}}) reads every
    # `pass` as False, collapsing tau to None for every task. The caller flattens; this stays
    # pure (no config import beyond the CapabilityRank type).
    order = tuple(rm.model for rm in rank.ordered)  # weakest -> strongest, dense 0..n-1 ranks
    rank_of = {m: i for i, m in enumerate(order)}
    costs = _build_cost_model(matrix, order)
    completed: dict[str, dict] = {}
    violations: list[Violation] = []
    tau: dict[str, str | None] = {}
    complete: set[str] = set()
    n_real = n_imputed = n_unknown = n_multi_observed = 0
    for tid, task_cells in matrix.items():
        # A CENSORED cell is NOT an observed pass or fail — its true outcome is unknown. Drop it
        # from the observation set so it never establishes a crossover (a censored top-tier cell
        # leaves the band UNKNOWN → the task is INCOMPLETE, not a clean all-fail complete). It may
        # still be IMPUTED from a genuine observation on the same task under monotonicity.
        observed_cells = {m: c for m, c in task_cells.items() if not censoring.is_censored(c)}
        s_rank, f_rank = _observe(observed_cells, rank_of)
        if _n_observed(observed_cells, rank_of) >= 2:
            n_multi_observed += 1
        if s_rank is not None and f_rank is not None and f_rank > s_rank:
            violations.append(
                Violation(task_id=tid, pass_model=order[s_rank], fail_model=order[f_rank])
            )
        tau[tid] = order[s_rank] if s_rank is not None else None
        out, counts = _complete_task(observed_cells, order, rank_of, s_rank, f_rank, costs)
        completed[tid] = out
        # COMPLETE ⟺ zero UNKNOWN cells: either a crossover was established (an observed PASS,
        # fail/pass determined everywhere) OR every tier was observed and all failed
        # (unsolvable). A task with an open UNKNOWN band has no established crossover.
        if counts.unknown == 0:
            complete.add(tid)
        n_real += counts.real
        n_imputed += counts.imputed
        n_unknown += counts.unknown
    return ImputedMatrix(
        matrix=completed,
        violations=violations,
        n_real=n_real,
        n_imputed=n_imputed,
        n_unknown=n_unknown,
        tau=tau,
        n_multi_observed=n_multi_observed,
        complete=frozenset(complete),
    )


def complete_challenges(matrix: dict, rank: CapabilityRank) -> frozenset[str]:
    """Task ids from ``matrix`` whose crossover is established (ZERO UNKNOWN cells)."""
    # Shared helper: downstream consumers (scoring / report / kill-gate) sample ONLY complete
    # challenges — an open-UNKNOWN-band challenge is excluded from analysis entirely.
    return complete_matrix(matrix, rank).complete
