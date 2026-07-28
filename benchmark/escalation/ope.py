"""Off-policy value of an escalation policy — or an explicit refusal when it is not identified.

The refusal is the point. Under deterministic logging P(escalate) = 0, the overlap condition
every off-policy estimator needs fails and the estimate is UNDEFINED, not merely noisy.
"""

# This module exists so that "escalation helps" can never again be reported off logs that
# cannot support it. Feed it the propensity records the router writes at flagged checkpoints
# (`ExplorationRecord`, persisted on the session's decision_provenance and read back by
# `OutcomeStore.escalation_exploration_rows`). If those logs contain no randomization, every
# numeric field comes back None and `status` reads NOT_IDENTIFIED.
#
# Estimator: per-decision doubly robust for a one-step contextual bandit — the flagged
# checkpoint is the decision, the arm is escalate/hold, and the reward is the verified outcome
# of the task the decision belonged to.
#
#   V_DR = mean_i [ Sum_a pi(a|x_i) qhat(x_i, a)  +  (pi(a_i|x_i) / p_i) (r_i - qhat(x_i, a_i)) ]
#
# The model component `qhat` is the per-arm mean reward — a direct-method baseline, deliberately
# NOT a fitted model: this module is instrumentation, and a trained outcome model is a
# separate decision with its own validation burden. MAGIC-style blending toward a richer model
# component is the next rung, not this module's job.

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

IDENTIFIED: Final[str] = "identified"
NOT_IDENTIFIED: Final[str] = "not_identified"

# A TWO-SIDED whitelist, matching the verifier vocabulary (`shunt.verifiers.aggregator`).
# Anything outside both sets — `unknown`, `infra_failure`, or any future value — yields NO
# reward and is excluded. A one-sided set would silently score every unrecognized outcome as a
# failure, and the estimator would return a confident 0.0 with a tight interval.
_SUCCESS_OUTCOMES: Final[frozenset[str]] = frozenset({"success", "weak_success"})
_FAILURE_OUTCOMES: Final[frozenset[str]] = frozenset({"failure"})
_ESCALATING_ACTIONS: Final[frozenset[str]] = frozenset({"raise_effort", "raise_rank"})
_DEFAULT_BOOTSTRAP: Final[int] = 2000
_DEFAULT_WEIGHT_CLIP: Final[float] = 20.0
_CI_MASS: Final[float] = 0.90  # the reported interval mass
# Arm existence is necessary, not sufficient. Below these floors the logs cannot support a
# number, so the estimator refuses rather than reporting an interval off a handful of rows or
# off a propensity so small its inverse dominates every other term.
_MIN_PER_ARM: Final[int] = 5
# Correlated decisions are one observation. Too few independent checkpoints and the bootstrap
# returns a near-zero-width interval, which reads as certainty rather than as no evidence.
_MIN_CLUSTERS: Final[int] = 5
_MIN_PROPENSITY: Final[float] = 0.01
_PROPENSITY_TOLERANCE: Final[float] = 1e-9


@dataclass(frozen=True)
class ExplorationLogRow:
    """One logged decision: the arm taken, the propensity that generated it, and the reward."""

    checkpoint_id: str
    escalated: bool
    propensity: float
    reward: float | None
    randomized: bool
    features: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyValueEstimate:
    """The estimated value of a target policy — or a refusal, with the counts either way."""

    status: str
    reason: str
    n_decisions: int
    n_escalated: int
    n_held: int
    n_excluded: int
    dr_estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    max_weight: float = 0.0
    n_clusters: int = 0  # the bootstrap unit — correlated decisions share one
    n_clipped: int = 0  # weights the clip actually bound (max_weight alone cannot say)

    @property
    def identified(self) -> bool:
        return self.status == IDENTIFIED


def always_escalate(row: ExplorationLogRow) -> float:
    """Target policy: escalate at every flagged checkpoint (P(escalate) = 1)."""
    del row
    return 1.0


def rows_from_records(records: Iterable[Mapping[str, object]]) -> list[ExplorationLogRow]:
    """Map the router's persisted exploration records onto estimator rows."""
    # Accepts both the live shape (OutcomeStore.escalation_exploration_rows) and any offline
    # replay that writes the same keys. An outcome outside the verifier vocabulary stays None —
    # the estimator excludes it rather than scoring an unlabelled decision as a failure — and
    # `randomized` is DERIVED from epsilon + propensity, never read from the record's flag.
    return [_row_from_record(record) for record in records]


def _reward_of(record: Mapping[str, object]) -> float | None:
    """1.0 / 0.0 for a recognized verified outcome; None for anything else (excluded)."""
    outcome = record.get("outcome")
    if not isinstance(outcome, str):
        return None
    if outcome in _SUCCESS_OUTCOMES:
        return 1.0
    return 0.0 if outcome in _FAILURE_OUTCOMES else None


def _is_randomized(propensity: float | None, epsilon: float | None) -> bool:
    """Derive randomization from epsilon and the propensity — never from a self-declared flag.

    A caller-supplied ``randomized`` boolean is exactly the field a broken or fabricated log
    would get wrong, so identification is decided by arithmetic the logger cannot fake.
    """
    if propensity is None or epsilon is None or epsilon <= 0.0:
        return False
    return any(
        abs(propensity - expected) < _PROPENSITY_TOLERANCE for expected in (epsilon, 1.0 - epsilon)
    )


def _as_float(value: object) -> float | None:
    """Coerce a logged numeric field; a null or non-numeric value is missing, not an error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _row_from_record(record: Mapping[str, object]) -> ExplorationLogRow:
    raw_features = record.get("features")
    features = (
        {str(k): float(v) for k, v in raw_features.items()}
        if isinstance(raw_features, dict)
        else {}
    )
    propensity = _as_float(record.get("propensity"))
    return ExplorationLogRow(
        checkpoint_id=str(record.get("checkpoint_id", "")),
        escalated=str(record.get("action", "")) in _ESCALATING_ACTIONS,
        propensity=propensity if propensity is not None else 1.0,
        reward=_reward_of(record),
        randomized=_is_randomized(propensity, _as_float(record.get("epsilon"))),
        features=features,
    )


def _usable(rows: Sequence[ExplorationLogRow]) -> list[ExplorationLogRow]:
    """Rows the estimator may weight: randomized, rewarded, and with a real propensity."""
    # A decision logged at propensity 1.0 (the ladder was exhausted, so there was no alternative
    # rung) is EXCLUDED, not weighted: it carries no counterfactual information.
    return [r for r in rows if r.randomized and r.reward is not None and 0.0 < r.propensity < 1.0]


def _overlap_failure(rows: Sequence[ExplorationLogRow], min_per_arm: int) -> str | None:
    """The reason the value is not identified on these rows, or None when it is."""
    if not rows:
        return (
            "no randomized decision with a verified outcome — overlap fails (the logging policy "
            "is deterministic, so P(escalate) is 0 or 1 and every estimator is undefined)"
        )
    escalated = sum(1 for r in rows if r.escalated)
    held = len(rows) - escalated
    if escalated < min_per_arm or held < min_per_arm:
        return (
            f"too little evidence: {escalated} escalate / {held} hold decisions, "
            f"below the {min_per_arm}-per-arm floor — arm existence is not identification"
        )
    n_clusters = len({r.checkpoint_id for r in rows})
    if n_clusters < _MIN_CLUSTERS:
        return (
            f"too few independent checkpoints: {n_clusters} — decisions on one checkpoint are "
            f"correlated, so this is not {len(rows)} observations and no interval is honest"
        )
    weakest = min(r.propensity for r in rows)
    if weakest < _MIN_PROPENSITY:
        return (
            f"overlap is nominal only: the smallest logged propensity is {weakest:.2e}, below "
            f"the {_MIN_PROPENSITY} floor — its inverse would dominate every other term"
        )
    return None


def _direct_method(rows: Sequence[ExplorationLogRow]) -> dict[bool, float]:
    """qhat(x, a): the per-arm mean reward (the model component of the DR estimator)."""
    means: dict[bool, float] = {}
    for arm in (True, False):
        rewards = [r.reward for r in rows if r.escalated is arm and r.reward is not None]
        means[arm] = statistics.fmean(rewards) if rewards else 0.0
    return means


def _dr_terms(
    rows: Sequence[ExplorationLogRow],
    target: Callable[[ExplorationLogRow], float],
    weight_clip: float,
) -> tuple[list[float], float, int]:
    """The per-decision DR contributions, the largest weight used, and how many were clipped."""
    qhat = _direct_method(rows)
    terms: list[float] = []
    max_weight = 0.0
    n_clipped = 0
    for row in rows:
        p_escalate = min(max(target(row), 0.0), 1.0)
        baseline = p_escalate * qhat[True] + (1.0 - p_escalate) * qhat[False]
        target_prob = p_escalate if row.escalated else 1.0 - p_escalate
        raw_weight = target_prob / row.propensity
        weight = min(raw_weight, weight_clip)
        n_clipped += raw_weight > weight_clip
        max_weight = max(max_weight, weight)
        residual = (row.reward or 0.0) - qhat[row.escalated]
        terms.append(baseline + weight * residual)
    return terms, max_weight, n_clipped


def _clusters(rows: Sequence[ExplorationLogRow], terms: Sequence[float]) -> list[list[float]]:
    """Group DR terms by checkpoint — the correlated unit, and so the bootstrap unit."""
    grouped: dict[str, list[float]] = {}
    for row, term in zip(rows, terms, strict=True):
        grouped.setdefault(row.checkpoint_id, []).append(term)
    return list(grouped.values())


def _bootstrap_ci(clusters: Sequence[list[float]], draws: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap resampling whole CLUSTERS at the reported interval mass."""
    rng = random.Random(seed)
    n = len(clusters)
    means = sorted(
        statistics.fmean([t for block in rng.choices(clusters, k=n) for t in block])
        for _ in range(draws)
    )
    tail = (1.0 - _CI_MASS) / 2.0
    low = means[min(int(tail * draws), draws - 1)]
    high = means[min(int((1.0 - tail) * draws), draws - 1)]
    return (low, high)


def estimate_policy_value(
    rows: Sequence[ExplorationLogRow],
    target: Callable[[ExplorationLogRow], float] = always_escalate,
    *,
    weight_clip: float = _DEFAULT_WEIGHT_CLIP,
    bootstrap_draws: int = _DEFAULT_BOOTSTRAP,
    bootstrap_seed: int = 0,
    min_per_arm: int = _MIN_PER_ARM,
) -> PolicyValueEstimate:
    """Doubly-robust value of *target*, or NOT_IDENTIFIED when the logs cannot support one."""
    usable = _usable(rows)
    n_escalated = sum(1 for r in usable if r.escalated)
    counts = {
        "n_decisions": len(usable),
        "n_escalated": n_escalated,
        "n_held": len(usable) - n_escalated,
        "n_excluded": len(rows) - len(usable),
    }
    failure = _overlap_failure(usable, min_per_arm)
    if failure is not None:
        return PolicyValueEstimate(status=NOT_IDENTIFIED, reason=failure, **counts)
    terms, max_weight, n_clipped = _dr_terms(usable, target, weight_clip)
    clusters = _clusters(usable, terms)
    low, high = _bootstrap_ci(clusters, bootstrap_draws, bootstrap_seed)
    return PolicyValueEstimate(
        status=IDENTIFIED,
        reason=f"both arms realized under logged propensities ({int(_CI_MASS * 100)}% interval)",
        dr_estimate=statistics.fmean(terms),
        ci_low=low,
        ci_high=high,
        max_weight=max_weight,
        n_clusters=len(clusters),
        n_clipped=n_clipped,
        **counts,
    )


__all__ = [
    "IDENTIFIED",
    "NOT_IDENTIFIED",
    "ExplorationLogRow",
    "PolicyValueEstimate",
    "always_escalate",
    "estimate_policy_value",
    "rows_from_records",
]
