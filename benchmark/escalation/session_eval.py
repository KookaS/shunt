"""Session-cadence escalation value: what the PRODUCT's per-session mechanism is worth."""

# The per-step sweep (`policy_eval`) measures a policy production does not run: live, a verified
# outcome is produced once per CLOSED SESSION (the off-wire verifier runs at the session boundary),
# so the escalation rule counts failures across sessions, never within one. The committed corpus has
# no multi-session trajectories, which is why the per-step eval exists at all — but it does hold
# several (model, arm) sessions per INSTANCE, i.e. repeated independent attempts at the same task.
# Ordered by price and read as a session sequence, those attempts measure the session-cadence
# question directly:
#
#   given a CHEAP session failed on this task, does escalating the NEXT session to a FRONTIER
#   model resolve it more often than a same-cost retry would?
#
# At session cadence the detector is trivially satisfied — the first session failed with the task's
# target failing-check id, which IS the recurrence key — so the question is pure VALUE, not
# detection. The numbers below are observational, not causal: the arms ran in parallel, and which
# instances got frontier coverage was adaptive. The overlap subset (instances with >=2 cheap AND a
# frontier session) is reported separately so the two rates are read on the SAME instances, which
# removes the worst of the coverage-selection bias. On the current corpus that subset still shows a
# ~2.5x resolution lift for escalating.
#
# Three further arms answer the question a same-cost retry cannot: what if the session had never
# been cheap (always_frontier), never escalated (always_cheap), or escalated at random at the same
# fire rate (random_escalate). They are read on the same overlap instances, through the same paired
# instance bootstrap, and are observational for the same reason the two arms above are.
#
# The per-instance arms are NOT hand-rolled here: each is a registered escalation policy in
# `benchmark/escalation/policies.py` (a name -> builder registry mirroring the router's strategy
# registry), and `session_cadence(policy=...)` headlines whichever registered policy is under test
# — default the shipped escalate decision, bit for bit.
#
# This module reports no skill verdict and gates nothing — it is context for the per-step sweep, and
# its samples are small. Its numbers answer "is the escalation ladder pointed the right way",
# not "is the trigger well-tuned".

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from benchmark.escalation import features, metrics
from benchmark.escalation.policies import (
    ARM_ALWAYS_CHEAP,
    ARM_ALWAYS_FRONTIER,
    ARM_ESCALATE,
    ARM_RANDOM,
    ARM_RETRY,
    REGISTERED_POLICIES,
    ArmSession,
    build_policy,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from benchmark.escalation.schema import Trajectory

# How many of the most-expensive models present in the corpus count as "frontier" for the
# escalation ladder. Two, not one: the ladder steps one rank at a time, so the top two prices are
# both legitimate escalation targets on this corpus.
_FRONTIER_TOP: Final[int] = 2
# The boundary below which a model counts as "cheap" for the session sequence: the cheapest price
# present. On this corpus that is deepseek-v4-flash; a retry at that same price is the
# no-escalation counterfactual.
_CHEAP_BOTTOM: Final[int] = 1

# The three trivial competitors the escalate arm has to beat before "escalation works" means
# anything. A same-cost cheap retry is NOT one of them — it is the incumbent contrast above, kept
# where it is because the committed payload and the figure already name it. The arm-name
# constants themselves live in `benchmark.escalation.policies`, the registry of per-instance
# escalation decisions; this module reads the registry's builders and names only the eval's own
# structural sets — which arm is a baseline, which arms every cost row is computed for.
BASELINE_ARMS: Final[tuple[str, ...]] = (ARM_ALWAYS_FRONTIER, ARM_ALWAYS_CHEAP, ARM_RANDOM)
# DERIVED from the registry, not a second hand-maintained list: every registered per-instance
# policy, plus the eval's own cross-instance random null. A policy added to `REGISTERED_POLICIES`
# joins the interval/cost computation here automatically — the alternative (a literal arm list in
# this module) is the hidden coupling that lets a registered policy pass `--policy` validation
# and then KeyError deep inside `_instance_intervals`.
_ALL_ARMS: Final[tuple[str, ...]] = (*REGISTERED_POLICIES, ARM_RANDOM)

# The random arm fires on a fixed, seeded subset of the overlap instances sized to the escalate
# arm's own fire rate. Seeded and exact-count (not per-instance coin flips) so the number is
# reproducible from the corpus alone and the fire rate it claims is the fire rate it has.
_RANDOM_ARM_SEED: Final[int] = 0

# An instance's expected resolve is a mean over that arm's sessions, so two arms holding the same
# sessions agree to floating-point noise rather than exactly. Below this they are the same outcome.
_RESOLVE_EPS: Final[float] = 1e-9


@dataclass(frozen=True)
class ArmContrast:
    """One alternative arm's rate plus its PAIRED difference against the escalate arm."""

    name: str
    n: int
    resolved: int
    rate: float
    ci: tuple[float, float]
    diff_estimate: float
    diff_ci: tuple[float, float]
    n_instances: int

    @property
    def diff_excludes_zero(self) -> bool:
        """Whether escalate-minus-this-arm's 95% interval is entirely on one side of zero."""
        low, high = self.diff_ci
        if math.isnan(low) or math.isnan(high):
            return False
        return low > 0.0 or high < 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "resolved": self.resolved,
            "rate": round(self.rate, 4),
            "ci95": _rounded_interval(self.ci),
            "paired_difference_vs_escalate": {
                "estimate": _rounded(self.diff_estimate),
                "ci95": _rounded_interval(self.diff_ci),
                "excludes_zero": self.diff_excludes_zero,
                "n_instances": self.n_instances,
            },
        }


@dataclass(frozen=True)
class SessionCosts:
    """What each session was billed, and how to total an attempt sequence in each currency."""

    # trajectory_id -> (model, billed real_cost). Built by `cost_join`, which owns the join to the
    # routing benchmark's measured spend; this module never reads a CSV or a price list itself.
    session: Mapping[str, tuple[str, float]]
    currencies: Mapping[str, Callable[[Sequence[tuple[str, float]]], float]]


@dataclass(frozen=True)
class ArmCost:
    """One arm's spend in one currency, and what a marginal resolve cost it."""

    name: str
    currency: str
    n_sessions: int
    # How many overlap tasks this arm ACTS on. Not every arm acts on every task: the escalate arm
    # exists only where a cheap session failed, so its per-task cost is per task-it-acted-on and
    # says nothing about the tasks it left alone. The always-* arms cover all of them.
    n_instances_covered: int
    total_cost: float
    cost_per_instance: float
    cost_ci: tuple[float, float]
    # Against the always-cheap arm — the do-nothing floor every other arm has to buy resolves
    # above. None when the arm bought no extra resolves at all, never a negative dollar figure.
    marginal_cost_per_resolve: float | None
    marginal_ci: tuple[float, float]
    marginal_undefined_share: float

    def to_dict(self) -> dict[str, object]:
        return {
            "n_sessions": self.n_sessions,
            "n_tasks_acted_on": self.n_instances_covered,
            "total_cost_usd": round(self.total_cost, 6),
            # Per task the arm ACTED ON, which for the escalate arm is the fired subset — read
            # `n_tasks_acted_on` beside it before dividing anything by the overlap subset's size.
            "cost_per_acted_task_usd": round(self.cost_per_instance, 6),
            "cost_per_acted_task_ci95": _rounded_interval(self.cost_ci, digits=6),
            "usd_per_marginal_resolve": (
                None
                if self.marginal_cost_per_resolve is None
                else round(self.marginal_cost_per_resolve, 4)
            ),
            "usd_per_marginal_resolve_ci95": _rounded_interval(self.marginal_ci, digits=4),
            "draws_without_a_marginal_resolve": round(self.marginal_undefined_share, 4),
        }


@dataclass(frozen=True)
class FullPolicyCost:
    """One arm's cost as a POLICY over every overlap instance — one denominator for all arms."""

    # WHY THIS EXISTS BESIDE `ArmCost`. `ArmCost` reads each arm on the instances that arm COVERS,
    # so the escalate arm's figures are computed on the fired subset (where the cheap floor is
    # lowest) while always-frontier's are computed on all of them. Two ratios divided by different
    # denominators are not comparable, and the published pair was never a comparison. Here a
    # conditional arm that does not fire on an instance still RUNS there — it stays cheap — so
    # every arm has a defined (cost, resolve) at every instance and one common denominator.

    name: str
    currency: str
    n_instances: int
    # Instances where the arm ran sessions of its OWN rather than falling back to the always-cheap
    # session. For the escalate arm that is the fired subset; the unconditional arms are at n.
    n_fired: int
    resolve_rate: float
    cost_per_instance: float
    cost_ci: tuple[float, float]
    # Against the always-cheap floor, read on the SAME drawn instances as the arm — never on the
    # arm's own coverage.
    marginal_cost_per_resolve: float | None
    marginal_ci: tuple[float, float]
    marginal_undefined_share: float
    # THE PAIRED COST DIFFERENCE the report never had: this arm minus always-frontier, computed
    # inside each draw. Two overlapping marginal intervals are not a test of a difference.
    cost_diff_vs_always_frontier: float
    cost_diff_ci: tuple[float, float]
    # THE QUALITY AXIS OF THE SAME COMPARISON, and the one number that says whether it carries any
    # information. A cost saving quoted without it reads as "cheaper at equal quality", which is a
    # claim about outcomes; `n_outcome_differs` is how many of the instances actually SEPARATE the
    # two arms. Zero means the comparison's quality side is an identity of construction, not a
    # measured equivalence, and it must be published saying so.
    resolve_diff_vs_always_frontier: float
    resolve_diff_ci: tuple[float, float]
    n_outcome_differs: int

    @property
    def diff_excludes_zero(self) -> bool:
        """Whether the paired cost difference's 95% interval is entirely on one side of zero."""
        low, high = self.cost_diff_ci
        if math.isnan(low) or math.isnan(high):
            return False
        return low > 0.0 or high < 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "n_instances": self.n_instances,
            "n_instances_fired": self.n_fired,
            "resolve_rate": round(self.resolve_rate, 4),
            "cost_per_instance_usd": round(self.cost_per_instance, 6),
            "cost_per_instance_ci95": _rounded_interval(self.cost_ci, digits=6),
            "usd_per_marginal_resolve": (
                None
                if self.marginal_cost_per_resolve is None
                else round(self.marginal_cost_per_resolve, 4)
            ),
            "usd_per_marginal_resolve_ci95": _rounded_interval(self.marginal_ci, digits=4),
            "draws_without_a_marginal_resolve": round(self.marginal_undefined_share, 4),
            "paired_cost_difference_vs_always_frontier": {
                "estimate": round(self.cost_diff_vs_always_frontier, 6),
                "ci95": _rounded_interval(self.cost_diff_ci, digits=6),
                "excludes_zero": self.diff_excludes_zero,
            },
            "paired_resolve_difference_vs_always_frontier": {
                "estimate": round(self.resolve_diff_vs_always_frontier, 4),
                "ci95": _rounded_interval(self.resolve_diff_ci),
                # Instances whose outcome differs between the two arms. 0 means the quality side
                # of this comparison is an identity of construction, never a measured equivalence.
                "n_instances_outcome_differs": self.n_outcome_differs,
            },
        }


@dataclass(frozen=True)
class SessionCadenceReport:
    """The session-cadence escalation contrast, measured on the overlap subset."""

    n_overlap_instances: int
    n_escalated: int
    n_escalated_resolved: int
    escalate_rate: float
    escalate_ci: tuple[float, float]
    n_retried: int
    n_retried_resolved: int
    retry_rate: float
    retry_ci: tuple[float, float]
    # The full-corpus context: frontier resolve rate wherever a cheaper session failed, and the
    # cheap base rate — so a reader can see the subset's selection without trusting it.
    n_frontier_after_fail: int
    n_frontier_after_fail_resolved: int
    frontier_after_fail_rate: float
    cheap_base_rate: float
    # THE PAIRED CONTRAST. Two marginal intervals failing to overlap is a conservative test of a
    # difference; it is not a test OF the difference, and it is not what the figure claims. Both
    # arms are read on the same instances, so the difference is estimable draw-for-draw on one
    # set of instance resamples, and that is the number with a CI a reader can act on.
    diff_estimate: float = float("nan")
    diff_ci: tuple[float, float] = (float("nan"), float("nan"))
    diff_draws: tuple[float, ...] = ()
    n_instances_resampled: int = 0
    # THE ARMS THAT COULD KILL THE CLAIM. `cheap_retry` above is a same-cost competitor; these
    # three are the trivial ones — never being cheap at all, never escalating at all, and firing
    # at random at the escalate arm's own rate. Same overlap instances, same paired bootstrap.
    comparisons: tuple[ArmContrast, ...] = ()
    random_fire_rate: float = float("nan")
    # THE COST AXIS. One row per (arm, currency); empty when the caller passed no billed costs.
    costs: tuple[ArmCost, ...] = ()
    # THE SAME ARMS AS FULL POLICIES, on one common denominator. Read this, not `costs`, for any
    # comparison BETWEEN arms; `costs` answers "what did this arm spend where it acted", which is
    # a different question and not comparable across arms.
    full_policy_costs: tuple[FullPolicyCost, ...] = ()
    # WHICH MODELS EACH ARM IS, and which of them the SHIPPED ladder would actually step to.
    # Carried on the report so the figure can name the arm instead of implying it is production's:
    # the escalate arm is the corpus's most expensive models, and the ladder walks the cheapest
    # ranks before it ever reaches one of them.
    frontier_models: tuple[str, ...] = ()
    cheap_models: tuple[str, ...] = ()
    ladder_visits: tuple[str, ...] = ()
    rank_shortlist: int = 0
    # WHICH POLICY THE REPORT HEADLINES. The `escalate*`/`diff_*`/`lift` fields read the policy
    # under test, not literally the shipped escalate arm; the default is the shipped escalate
    # decision, so the committed payload is unchanged, and a non-default selection carries its
    # own name so the numbers can never be misattributed to escalation.
    policy: str = ARM_ESCALATE

    @property
    def comparison(self) -> dict[str, ArmContrast]:
        """The baseline arms by name, for a caller that wants one of them."""
        return {c.name: c for c in self.comparisons}

    @property
    def diff_excludes_zero(self) -> bool:
        """Whether the paired difference's 95% interval is entirely on one side of zero."""
        low, high = self.diff_ci
        if math.isnan(low) or math.isnan(high):
            return False
        return low > 0.0 or high < 0.0

    @property
    def lift(self) -> float | None:
        """escalate_rate / retry_rate — resolution escalation buys over a same-cost retry."""
        if self.retry_rate <= 0.0:
            return None
        return self.escalate_rate / self.retry_rate

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "n_overlap_instances": self.n_overlap_instances,
            "escalate": {
                "n": self.n_escalated,
                "resolved": self.n_escalated_resolved,
                "rate": round(self.escalate_rate, 4),
                "ci95": _rounded_interval(self.escalate_ci),
            },
            "cheap_retry": {
                "n": self.n_retried,
                "resolved": self.n_retried_resolved,
                "rate": round(self.retry_rate, 4),
                "ci95": _rounded_interval(self.retry_ci),
            },
            "lift": _rounded(self.lift),
            "paired_difference": {
                "estimate": _rounded(self.diff_estimate),
                "ci95": _rounded_interval(self.diff_ci),
                "excludes_zero": self.diff_excludes_zero,
                "n_instances": self.n_instances_resampled,
            },
            # Observational, exactly as the two arms above are: these arms are re-readings of the
            # same parallel runs under adaptive frontier coverage, not assignments anyone made.
            "comparisons": {c.name: c.to_dict() for c in self.comparisons},
            # Cost is keyed by currency FIRST, because a naive total and a cache-aware total are
            # different quantities and a reader must never pick one up thinking it is the other.
            "cost": {
                currency: {c.name: c.to_dict() for c in self.costs if c.currency == currency}
                for currency in dict.fromkeys(c.currency for c in self.costs)
            },
            # The full-policy read: every arm on all overlap instances, one denominator, plus the
            # paired cost difference against always-frontier. Keyed separately from `cost` because
            # the two answer different questions and must never be picked up for each other.
            "cost_full_policy": {
                currency: {
                    c.name: c.to_dict() for c in self.full_policy_costs if c.currency == currency
                }
                for currency in dict.fromkeys(c.currency for c in self.full_policy_costs)
            },
            "context": {
                "frontier_models": list(self.frontier_models),
                "cheap_models": list(self.cheap_models),
                "shipped_ladder_visits": list(self.ladder_visits),
                "shipped_rank_shortlist": self.rank_shortlist,
                "random_escalate_fire_rate": _rounded(self.random_fire_rate),
                "n_frontier_after_fail": self.n_frontier_after_fail,
                "n_frontier_after_fail_resolved": self.n_frontier_after_fail_resolved,
                "frontier_after_fail_rate": round(self.frontier_after_fail_rate, 4),
                "cheap_base_rate": round(self.cheap_base_rate, 4),
            },
        }
        # The headline policy, recorded only when it differs from the shipped escalate decision:
        # the default run's payload is then unchanged bit for bit (the committed metrics.json
        # contract) while a non-default run states which policy its `escalate`-keyed numbers are
        # about, so they can never be misattributed.
        if self.policy != ARM_ESCALATE:
            out["policy"] = self.policy
        return out


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _rounded_interval(interval: tuple[float, float], digits: int = 4) -> list[float] | None:
    """Round a 95% interval; a non-finite one (an arm with no resamples) is null, matching
    ``policy_eval.rounded_interval`` so the JSON null shape is uniform across the payload."""
    if any(not math.isfinite(v) for v in interval):
        return None
    return [round(interval[0], digits), round(interval[1], digits)]


def _price_ranks(trajectories: Sequence[Trajectory]) -> dict[str, int]:
    """Model -> ascending price rank among the models PRESENT in this corpus (0 = cheapest)."""
    # Read off the registry's own price ordering, never a literal list: `ModelPool.ranked_models`
    # sorts by total list price, which is the same order the router's capability rank uses.
    from shunt.models.config import ModelPool  # noqa: PLC0415

    pool = ModelPool()
    prices: dict[str, float] = {}
    for model in pool.ranked_models():
        if model.pricing is not None:
            prices[model.name] = model.pricing.input_cost_per_1m + model.pricing.output_cost_per_1m
    present = {features.model_of(t) for t in trajectories}
    ordered = sorted((m for m in present if m in prices), key=lambda m: prices[m])
    return {m: i for i, m in enumerate(ordered)}


def _shipped_ladder(ranks: dict[str, int]) -> tuple[tuple[str, ...], int]:
    """The models the SHIPPED ladder steps to over the SHIPPED live pool's price order."""
    # Read from the packaged router config and stepped with the product's OWN rank arithmetic, so
    # a shortlist change moves this line instead of silently invalidating it. The walk is over the
    # SHIPPED live pool (the packaged registry restricted to the packaged router.yaml `models:`
    # list), NOT the models present in this corpus: the escalation figure must say what the router
    # would ACTUALLY climb, and a corpus that drops a shipped model must not change that sentence.
    # ``ranks`` stays in the signature because callers pass it, but it no longer builds the walk.
    from shunt.models.config import ModelPool, default_registry_path  # noqa: PLC0415
    from shunt.router.escalation import next_rung_rank  # noqa: PLC0415
    from shunt.router.policy import load_router_policy, packaged_policy_path  # noqa: PLC0415

    policy = load_router_policy(packaged_policy_path())
    # The PACKAGED registry, never SHUNT_CONFIG_DIR / ~/.config: a host override would draw the
    # ladder from a pool the shipped router does not build.
    pool = ModelPool(str(default_registry_path()))
    pool.restrict_to_live(policy.models)
    by_rank = [m.name for m in pool.ranked_models()]
    shortlist = policy.escalation.rank_shortlist
    top = len(by_rank) - 1
    visits: list[str] = []
    current = 0
    while current < top:
        current = min(next_rung_rank(current, top, shortlist), top)
        visits.append(by_rank[current])
    return (tuple(visits), shortlist)


def _bucket_of(rank: int, n_models: int) -> str:
    if rank < _CHEAP_BOTTOM:
        return "cheap"
    if rank >= n_models - _FRONTIER_TOP:
        return "frontier"
    return "mid"


def session_cadence(
    trajectories: Sequence[Trajectory],
    costs: SessionCosts | None = None,
    *,
    policy: str = ARM_ESCALATE,
) -> SessionCadenceReport | None:
    """The escalation-vs-cheap-retry contrast at the session cadence, on the overlap subset.

    ``policy`` names the registered escalation decision to headline (default ``escalate``,
    bit for bit); an unregistered name fails KeyError-style before any instance work.
    """
    build_policy(policy)  # KeyError-style: an unknown policy fails before any instance work
    ranks = _price_ranks(trajectories)
    n_models = len(ranks)
    by_rank = sorted(ranks, key=lambda m: ranks[m])
    ladder_visits, shortlist = _shipped_ladder(ranks)
    by_instance: dict[str, list[Trajectory]] = defaultdict(list)
    for traj in trajectories:
        by_instance[features.group_of(traj)].append(traj)

    def bucket(traj: Trajectory) -> str:
        # A model absent from the registry is neither a cheap base pick nor a frontier target;
        # it is "mid" and takes no part in either arm of the contrast.
        rank = ranks.get(features.model_of(traj))
        if rank is None:
            return "mid"
        return _bucket_of(rank, n_models)

    cheap_base = metrics.prevalence(
        [t.header.terminal_resolved for t in trajectories if bucket(t) == "cheap"]
    )
    # Full-corpus context: frontier resolve wherever a cheaper session failed.
    frontier_after_fail: list[bool] = []
    for inst in by_instance.values():
        sessions = sorted(inst, key=lambda t: ranks.get(features.model_of(t), n_models))
        if any(not s.header.terminal_resolved for s in sessions if bucket(s) == "cheap"):
            frontier_after_fail.extend(
                s.header.terminal_resolved for s in sessions if bucket(s) == "frontier"
            )

    # The controlled subset: instances with BOTH >=2 cheap sessions AND a frontier session, so
    # the escalate and retry arms are read on the same instances (adaptive-coverage selection
    # removed as far as this corpus allows).
    arms: list[_InstanceArms] = []
    for instance, inst in by_instance.items():
        cheap = [s for s in inst if bucket(s) == "cheap"]
        frontier = [s for s in inst if bucket(s) == "frontier"]
        if len(cheap) < 2 or not frontier:
            continue
        arms.append(_instance_arms(instance, cheap, frontier))

    if not arms:
        # No task carries both a retry arm and a frontier arm: neither rate is estimable and the
        # contrast is meaningless, so report "not supported" rather than NaN bars.
        return None

    fire_rate = _assign_random_arm(arms, policy=policy)
    headline_outcomes = _outcomes(arms, policy)
    retry_outcomes = _outcomes(arms, ARM_RETRY)
    intervals, diffs = _instance_intervals(arms, policy)
    headline_ci = intervals[policy]
    retry_ci = intervals[ARM_RETRY]
    diff_draws = diffs[ARM_RETRY]
    # The baseline arms the headline is compared against. The selected policy IS the headline, so
    # it cannot also be a comparator of itself — always_cheap as the policy under test is the
    # never-escalate read, not a comparison AGAINST never-escalate.
    baselines = tuple(name for name in BASELINE_ARMS if name != policy)
    return SessionCadenceReport(
        n_overlap_instances=len(arms),
        n_escalated=len(headline_outcomes),
        n_escalated_resolved=sum(headline_outcomes),
        escalate_rate=metrics.prevalence(headline_outcomes),
        escalate_ci=headline_ci,
        n_retried=len(retry_outcomes),
        n_retried_resolved=sum(retry_outcomes),
        retry_rate=metrics.prevalence(retry_outcomes),
        retry_ci=retry_ci,
        n_frontier_after_fail=len(frontier_after_fail),
        n_frontier_after_fail_resolved=sum(frontier_after_fail),
        frontier_after_fail_rate=metrics.prevalence(frontier_after_fail),
        cheap_base_rate=cheap_base,
        diff_estimate=(
            metrics.prevalence(headline_outcomes) - metrics.prevalence(retry_outcomes)
            if headline_outcomes and retry_outcomes
            else float("nan")
        ),
        diff_ci=metrics.bootstrap_ci(diff_draws),
        diff_draws=tuple(diff_draws),
        n_instances_resampled=len(arms),
        comparisons=tuple(
            _contrast(name, arms, intervals[name], diffs[name], policy) for name in baselines
        ),
        random_fire_rate=fire_rate,
        costs=() if costs is None else _arm_costs(arms, costs),
        full_policy_costs=() if costs is None else _full_policy_costs(arms, costs),
        frontier_models=tuple(m for m in by_rank if _bucket_of(ranks[m], n_models) == "frontier"),
        cheap_models=tuple(m for m in by_rank if _bucket_of(ranks[m], n_models) == "cheap"),
        ladder_visits=ladder_visits,
        rank_shortlist=shortlist,
        policy=policy,
    )


@dataclass(frozen=True)
class _InstanceArms:
    """One overlap instance's arms, kept together so a resample moves all of them at once."""

    instance: str
    arms: dict[str, list[ArmSession]] = field(default_factory=dict)
    # The two candidate next-sessions the random arm draws from, precomputed because they are
    # UNCONDITIONAL where the escalate and retry arms condition on a cheap failure.
    escalate_candidates: tuple[ArmSession, ...] = ()
    retry_candidate: tuple[ArmSession, ...] = ()


def _outcomes(arms: Sequence[_InstanceArms], name: str) -> list[bool]:
    """Every session outcome an arm holds, pooled across instances."""
    return [s.resolved for a in arms for s in a.arms[name]]


def _instance_arms(
    instance: str, cheap: Sequence[Trajectory], frontier: Sequence[Trajectory]
) -> _InstanceArms:
    """One overlap instance's arms; every per-instance arm is built by its registered policy."""
    # Cheap sessions are ordered by length, exactly as the retry arm has always ordered them: the
    # corpus records no wall-clock, so step count is the only within-price ordering it supports.
    cheap_by_length = sorted(cheap, key=lambda s: s.header.n_steps)
    first = cheap_by_length[0]
    # ATTEMPTS ARE WHAT THE ARM PAYS FOR, not what it is scored on. A policy that escalates has
    # already run — and been billed for — the cheap session that failed, so the frontier session's
    # attempt sequence carries that cheap session in front of it. Cost reads this; the rate does
    # not, which is why the outcome stays the last attempt's outcome alone.
    escalate_candidates = tuple(
        ArmSession(f.header.terminal_resolved, (first, f)) for f in frontier
    )
    # The registered policies' `decide` reproduce the arms' outcome and attempt sequences exactly
    # (see `benchmark/escalation/policies.py`); the random arm is filled in afterwards, from these
    # unconditional candidates, because it needs the WHOLE instance set's fire rate.
    arms = {name: list(build_policy(name).decide(cheap, frontier)) for name in REGISTERED_POLICIES}
    arms[ARM_RANDOM] = []
    return _InstanceArms(
        instance=instance,
        arms=arms,
        escalate_candidates=escalate_candidates,
        retry_candidate=(
            ArmSession(cheap_by_length[1].header.terminal_resolved, (first, cheap_by_length[1])),
        ),
    )


def _assign_random_arm(arms: Sequence[_InstanceArms], *, policy: str) -> float:
    """Fire the random arm on a seeded subset sized to the headline arm's fire rate; return it."""
    # The signal-free counterfactual: escalation that fires as OFTEN as the real one but not on
    # the same tasks. Both legs are UNCONDITIONAL, because a policy that ignores the signal cannot
    # condition on it — a fired instance escalates, an unfired one retries cheap, either way.
    # The fire rate is the HEADLINE arm's — the selected policy under test — so the null matches
    # the thing it is a null for.
    fired = [a for a in arms if a.arms[policy]]
    fire_rate = len(fired) / len(arms)
    order = sorted(range(len(arms)), key=lambda i: arms[i].instance)
    random.Random(_RANDOM_ARM_SEED).shuffle(order)
    picked = set(order[: len(fired)])
    for i, arm in enumerate(arms):
        source = arm.escalate_candidates if i in picked else arm.retry_candidate
        arm.arms[ARM_RANDOM] = list(source)
    return fire_rate


def _contrast(
    name: str,
    arms: Sequence[_InstanceArms],
    interval: tuple[float, float],
    diff_draws: Sequence[float],
    policy: str,
) -> ArmContrast:
    """One baseline arm's published row: its rate, and the headline policy minus it, paired."""
    outcomes = _outcomes(arms, name)
    headline = _outcomes(arms, policy)
    return ArmContrast(
        name=name,
        n=len(outcomes),
        resolved=sum(outcomes),
        rate=metrics.prevalence(outcomes),
        ci=interval,
        diff_estimate=(
            metrics.prevalence(headline) - metrics.prevalence(outcomes)
            if headline and outcomes
            else float("nan")
        ),
        diff_ci=metrics.bootstrap_ci(list(diff_draws)),
        n_instances=len(arms),
    )


def _instance_intervals(
    arms: Sequence[_InstanceArms], policy: str
) -> tuple[dict[str, tuple[float, float]], dict[str, list[float]]]:
    """Every arm's marginal interval and its PAIRED difference draws, from ONE resample set."""
    # Wilson over SESSIONS treated several frontier sessions on one instance as independent draws.
    # They are not: the same task, the same repo, the same target test — the instance is the
    # exchangeable unit, exactly as the challenge is in the per-step half. `grouped_resamples`
    # already resamples whole groups, so it is reused here rather than re-derived; the group key
    # is the instance and each group's "rows" are its own sessions.
    #
    # The difference is computed INSIDE the loop, on the same resampled instances that produced
    # the two marginals, which is what makes it paired: an instance that enters a draw contributes
    # to both arms or to neither, so the between-instance variance that both arms share cancels
    # instead of being counted twice.
    #
    # EVERY arm rides the same resample set, so a comparison arm is paired with the headline
    # policy on exactly the instances that entered that draw — the same property that makes the
    # retry contrast estimable, extended rather than duplicated.
    groups = [a.instance for a in arms]
    rate_draws: dict[str, list[float]] = {name: [] for name in _ALL_ARMS}
    # The headline is compared against the retry incumbent and the baselines. When the selected
    # policy IS the retry arm, its retry contrast is an identity — every draw 0.0 — which is the
    # honest read of "retry vs a cheap retry", not a crash.
    diff_arms = {ARM_RETRY} | {name for name in BASELINE_ARMS if name != policy}
    diff_draws: dict[str, list[float]] = {name: [] for name in diff_arms}
    for picked in metrics.grouped_resamples(groups):
        chosen = [arms[i] for i in picked]
        rates: dict[str, float] = {}
        for name in _ALL_ARMS:
            outcomes = _outcomes(chosen, name)
            if outcomes:
                rates[name] = metrics.prevalence(outcomes)
                rate_draws[name].append(rates[name])
        if policy in rates:
            for name in diff_arms:
                if name in rates:
                    diff_draws[name].append(rates[policy] - rates[name])
    return ({name: metrics.bootstrap_ci(d) for name, d in rate_draws.items()}, diff_draws)


def _session_total(session: ArmSession, costs: SessionCosts, currency: str) -> float:
    """What one arm-session cost, including the attempts that arm had to run before it."""
    # A missing trajectory raises KeyError here rather than silently costing zero: `cost_join`
    # guarantees a total join, and this is the second wall behind that guarantee.
    attempts = [costs.session[t.header.trajectory_id] for t in session.attempts]
    return costs.currencies[currency](attempts)


def _instance_point(
    arm: _InstanceArms, name: str, costs: SessionCosts, currency: str
) -> tuple[float, float] | None:
    """One instance's (expected cost, expected resolve) under an arm; None where it has none."""
    sessions = arm.arms[name]
    if not sessions:
        return None
    # MEAN, not sum: several frontier sessions on one task are repeated draws of ONE policy step,
    # so the policy's per-task cost is what one of them costs, not what all of them cost together.
    total = sum(_session_total(s, costs, currency) for s in sessions) / len(sessions)
    return (total, sum(s.resolved for s in sessions) / len(sessions))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _marginal(point: tuple[float, float], reference: tuple[float, float]) -> float | None:
    """Dollars per resolve an arm buys ABOVE the always-cheap floor, or None if it buys none."""
    gain = point[1] - reference[1]
    if gain <= 0.0:
        return None
    return (point[0] - reference[0]) / gain


def _draw_point(
    picked: Sequence[int], points: Sequence[tuple[float, float] | None]
) -> tuple[float, float] | None:
    """One resample's (mean cost, mean resolve) over the instances an arm actually covers."""
    covered = [point for i in picked if (point := points[i]) is not None]
    if not covered:
        return None
    return (_mean([c for c, _r in covered]), _mean([r for _c, r in covered]))


def _arm_costs(arms: Sequence[_InstanceArms], costs: SessionCosts) -> tuple[ArmCost, ...]:
    """Every arm's spend, in every currency, through the SAME instance bootstrap as the rates."""
    # `grouped_resamples` is deterministic in its group list, so this loop draws exactly the
    # instance sets `_instance_intervals` drew: the cost and the rate a reader compares come off
    # the same resamples, and the marginal-resolve ratio is paired against the always-cheap floor
    # inside each draw rather than divided after the fact.
    points = {
        currency: {
            name: [_instance_point(a, name, costs, currency) for a in arms] for name in _ALL_ARMS
        }
        for currency in costs.currencies
    }
    cost_draws: dict[tuple[str, str], list[float]] = {}
    marginal_draws: dict[tuple[str, str], list[float]] = {}
    n_draws = 0
    for picked in metrics.grouped_resamples([a.instance for a in arms]):
        n_draws += 1
        for currency, per_arm in points.items():
            for name in _ALL_ARMS:
                # The floor is read on THIS arm's instances, not on every drawn instance: the
                # escalate arm covers only the tasks where a cheap session failed, and comparing
                # it against an always-cheap rate that includes the tasks cheap already solved
                # compares two different populations. Restricting both to the arm's own coverage
                # is what the point estimate does, so the draws must do it too.
                covered = [i for i in picked if per_arm[name][i] is not None]
                point = _draw_point(covered, per_arm[name])
                reference = _draw_point(covered, per_arm[ARM_ALWAYS_CHEAP])
                if point is None:
                    continue
                cost_draws.setdefault((currency, name), []).append(point[0])
                if reference is None or name == ARM_ALWAYS_CHEAP:
                    continue
                marginal = _marginal(point, reference)
                if marginal is not None:
                    marginal_draws.setdefault((currency, name), []).append(marginal)
    return tuple(
        _arm_cost(
            arms,
            name,
            currency,
            points[currency],
            cost_draws.get((currency, name), []),
            marginal_draws.get((currency, name), []),
            n_draws,
        )
        for currency in costs.currencies
        for name in _ALL_ARMS
    )


def _arm_cost(
    arms: Sequence[_InstanceArms],
    name: str,
    currency: str,
    points: Mapping[str, Sequence[tuple[float, float] | None]],
    cost_draws: Sequence[float],
    marginal_draws: Sequence[float],
    n_draws: int,
) -> ArmCost:
    """One published cost row: the observed point estimates plus their resampled intervals."""
    covered = [i for i, p in enumerate(points[name]) if p is not None]
    observed = _draw_point(covered, points[name])
    reference = _draw_point(covered, points[ARM_ALWAYS_CHEAP])
    n_sessions = sum(len(a.arms[name]) for a in arms)
    return ArmCost(
        name=name,
        currency=currency,
        n_sessions=n_sessions,
        n_instances_covered=len(covered),
        # The billed total over every session this arm holds — what the corpus actually spent,
        # not the per-instance expectation the interval is built on.
        total_cost=sum(
            (points[name][i] or (0.0, 0.0))[0] * len(arms[i].arms[name]) for i in covered
        ),
        cost_per_instance=float("nan") if observed is None else observed[0],
        cost_ci=metrics.bootstrap_ci(list(cost_draws)),
        marginal_cost_per_resolve=(
            None
            if observed is None or reference is None or name == ARM_ALWAYS_CHEAP
            else _marginal(observed, reference)
        ),
        marginal_ci=metrics.bootstrap_ci(list(marginal_draws)),
        marginal_undefined_share=(0.0 if n_draws == 0 else 1.0 - len(marginal_draws) / n_draws),
    )


def _full_policy_sessions(arm: _InstanceArms, name: str) -> Sequence[ArmSession]:
    """What an arm RUNS at one instance — a conditional arm that never fired still stays cheap."""
    # This is the whole of the denominator fix. `arms[name]` is empty exactly where the policy did
    # not act; a policy that does not act is not a policy that is absent, it is one that did the
    # do-nothing thing and paid the do-nothing price.
    return arm.arms[name] or arm.arms[ARM_ALWAYS_CHEAP]


def _full_policy_point(
    arm: _InstanceArms, name: str, costs: SessionCosts, currency: str
) -> tuple[float, float]:
    """One instance's (expected cost, expected resolve) under an arm read as a full policy."""
    sessions = _full_policy_sessions(arm, name)
    total = sum(_session_total(s, costs, currency) for s in sessions) / len(sessions)
    return (total, sum(s.resolved for s in sessions) / len(sessions))


def _mean_point(
    picked: Sequence[int], points: Sequence[tuple[float, float]]
) -> tuple[float, float]:
    """One resample's (mean cost, mean resolve) over EVERY drawn instance — never a subset."""
    return (_mean([points[i][0] for i in picked]), _mean([points[i][1] for i in picked]))


@dataclass(frozen=True)
class _FullPolicyDraws:
    """One (currency, arm)'s resampled draws, kept together so a builder stays under its arity."""

    cost: list[float] = field(default_factory=list)
    marginal: list[float] = field(default_factory=list)
    diff_vs_frontier: list[float] = field(default_factory=list)
    resolve_diff_vs_frontier: list[float] = field(default_factory=list)


def _full_policy_costs(
    arms: Sequence[_InstanceArms], costs: SessionCosts
) -> tuple[FullPolicyCost, ...]:
    """Every arm as a full policy over ALL instances, through the paired instance bootstrap."""
    points = {
        currency: {
            name: [_full_policy_point(a, name, costs, currency) for a in arms] for name in _ALL_ARMS
        }
        for currency in costs.currencies
    }
    draws: dict[tuple[str, str], _FullPolicyDraws] = {
        (currency, name): _FullPolicyDraws() for currency in costs.currencies for name in _ALL_ARMS
    }
    n_draws = 0
    for picked in metrics.grouped_resamples([a.instance for a in arms]):
        n_draws += 1
        for currency, per_arm in points.items():
            # ONE index set for every arm in this draw: the floor and the comparator are read on
            # exactly the instances the arm itself was read on, which is the defect being fixed.
            floor = _mean_point(picked, per_arm[ARM_ALWAYS_CHEAP])
            frontier = _mean_point(picked, per_arm[ARM_ALWAYS_FRONTIER])
            for name in _ALL_ARMS:
                point = _mean_point(picked, per_arm[name])
                row = draws[(currency, name)]
                row.cost.append(point[0])
                row.diff_vs_frontier.append(point[0] - frontier[0])
                row.resolve_diff_vs_frontier.append(point[1] - frontier[1])
                marginal = None if name == ARM_ALWAYS_CHEAP else _marginal(point, floor)
                if marginal is not None:
                    row.marginal.append(marginal)
    return tuple(
        _full_policy_cost(arms, name, currency, points[currency], draws[(currency, name)], n_draws)
        for currency in costs.currencies
        for name in _ALL_ARMS
    )


def _full_policy_cost(
    arms: Sequence[_InstanceArms],
    name: str,
    currency: str,
    points: Mapping[str, Sequence[tuple[float, float]]],
    draws: _FullPolicyDraws,
    n_draws: int,
) -> FullPolicyCost:
    """One published full-policy row: the observed point estimates and their intervals."""
    every = list(range(len(arms)))
    observed = _mean_point(every, points[name])
    floor = _mean_point(every, points[ARM_ALWAYS_CHEAP])
    frontier = _mean_point(every, points[ARM_ALWAYS_FRONTIER])
    return FullPolicyCost(
        name=name,
        currency=currency,
        n_instances=len(arms),
        n_fired=sum(1 for a in arms if a.arms[name]),
        resolve_rate=observed[1],
        cost_per_instance=observed[0],
        cost_ci=metrics.bootstrap_ci(draws.cost),
        marginal_cost_per_resolve=(
            None if name == ARM_ALWAYS_CHEAP else _marginal(observed, floor)
        ),
        marginal_ci=metrics.bootstrap_ci(draws.marginal),
        marginal_undefined_share=(0.0 if n_draws == 0 else 1.0 - len(draws.marginal) / n_draws),
        cost_diff_vs_always_frontier=observed[0] - frontier[0],
        cost_diff_ci=metrics.bootstrap_ci(draws.diff_vs_frontier),
        resolve_diff_vs_always_frontier=observed[1] - frontier[1],
        resolve_diff_ci=metrics.bootstrap_ci(draws.resolve_diff_vs_frontier),
        n_outcome_differs=sum(
            1
            for arm, reference in zip(points[name], points[ARM_ALWAYS_FRONTIER], strict=True)
            if abs(arm[1] - reference[1]) > _RESOLVE_EPS
        ),
    )


__all__ = [
    "BASELINE_ARMS",
    "ArmContrast",
    "ArmCost",
    "FullPolicyCost",
    "SessionCadenceReport",
    "SessionCosts",
    "session_cadence",
]
