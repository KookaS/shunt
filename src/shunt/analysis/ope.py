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
#
# THREE THINGS THIS MODULE LEARNED THE HARD WAY, all of which read as certainty when wrong:
#
#   THE CLUSTER IS THE SESSION, NOT THE CHECKPOINT. The reward is a SESSION-level outcome — every
#   decision inside one session shares one verified grade — so a session is one observation. This
#   clustered on `checkpoint_id` (the failing-test id) instead, which both SPLITS one session
#   across clusters and MERGES different sessions that happened to hit the same test. A probe of
#   60 decisions from a SINGLE session touching 5 keys cleared the 5-cluster floor and reported
#   `identified` off what is really one observation. The unit is now `session_id`, carried
#   through from `OutcomeStore.escalation_exploration_rows`.
#
#   A ZERO-WIDTH INTERVAL IS NOT A CERTAIN ONE. Probes returned dr=0.0 ci=(0.0, 0.0) and dr=1.0
#   ci=(1.0, 1.0) with `identified`. That is the exact failure `_MIN_CLUSTERS` exists to prevent,
#   dressed as its opposite: every bootstrap resample landed on the same value because the rows
#   carry no variation, which is NO evidence, not perfect evidence. A degenerate interval now
#   refuses.
#
#   `qhat` IS CROSS-FITTED. Fitting the per-arm means on the same rows the DR correction then
#   scores makes the residual optimistically small and the interval too tight. `qhat` for a row is
#   now fitted on the OTHER cluster folds only.
#
# WHAT IS STILL NOT ANSWERED BY V ALONE. V(always_escalate) is a level, not a comparison: "does
# escalating help" is V(escalate) - V(hold). The contrast against the target's complement is
# therefore reported alongside every estimate (`contrast_*`), paired on the same rows and
# bootstrapped over the same clusters. Read the contrast, not the level.

from __future__ import annotations

import math
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
# Cross-fitting folds for `qhat`. Five, so it can never demand more clusters than `_MIN_CLUSTERS`
# already requires — a fold with no rows of an arm falls back to that arm's global mean.
_N_FOLDS: Final[int] = 5


@dataclass(frozen=True)
class ExplorationLogRow:
    """One logged decision: the arm taken, the propensity that generated it, and the reward."""

    checkpoint_id: str
    escalated: bool
    propensity: float
    reward: float | None
    randomized: bool
    features: dict[str, float] = field(default_factory=dict)
    # The correlated unit. The reward is the SESSION's verified outcome, so every decision in one
    # session is one observation; `checkpoint_id` (a failing-test id) is neither necessary nor
    # sufficient for independence. Empty means the log carried no session — an offline fixture,
    # never `OutcomeStore` output — and `_cluster_key` then degrades to the checkpoint rather than
    # collapsing every such row into one giant cluster.
    session_id: str = ""
    # The arm as logged. Binary escalation carries the rung name and `escalated` is the whole
    # story; a routing decision carries the chosen model, for which `escalated` is always False —
    # the binary estimator then refuses those rows rather than scoring a one-armed log.
    action: str = ""


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
    n_clusters: int = 0  # the bootstrap unit — one SESSION, not one checkpoint
    n_clipped: int = 0  # weights the clip actually bound (max_weight alone cannot say)
    # V(target) - V(complement of target): the decision question. A level cannot answer it, and
    # reporting only the level is what let "escalation helps" be inferred from a number that never
    # compared anything. Paired per decision, bootstrapped over the same clusters.
    contrast_estimate: float | None = None
    contrast_ci_low: float | None = None
    contrast_ci_high: float | None = None
    # Kish effective sample size of the importance weights, and its fraction of n. A value far
    # below n says the estimate rests on a handful of rows however large n looks. Read it against
    # the TARGET: a deterministic target gives the un-taken arm weight 0, so `always_escalate`
    # ceilings the fraction at n_escalated/n_decisions even on a log with perfect overlap.
    ess: float = 0.0
    ess_fraction: float = 0.0

    @property
    def identified(self) -> bool:
        return self.status == IDENTIFIED

    @property
    def contrast_excludes_zero(self) -> bool:
        """True iff the escalate-minus-hold interval sits wholly on one side of zero."""
        if self.contrast_ci_low is None or self.contrast_ci_high is None:
            return False
        return self.contrast_ci_low > 0.0 or self.contrast_ci_high < 0.0


def always_escalate(row: ExplorationLogRow) -> float:
    """Target policy: escalate at every flagged checkpoint (P(escalate) = 1)."""
    del row
    return 1.0


def never_escalate(row: ExplorationLogRow) -> float:
    """The hold-everything comparator — the other half of "does escalating help"."""
    del row
    return 0.0


def rows_from_records(records: Iterable[Mapping[str, object]]) -> list[ExplorationLogRow]:
    """Map the router's persisted exploration records — escalation or routing — onto rows."""
    # Accepts the binary escalation shape (OutcomeStore.escalation_exploration_rows), the
    # multi-arm routing shape (`selection_propensity` + `candidate_model_scores`), and any
    # offline replay that writes the same keys. An outcome outside the verifier vocabulary stays
    # None — the estimator excludes it rather than scoring an unlabelled decision as a failure —
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


def _is_randomized_over_arms(propensity: float | None, epsilon: float | None, n_arms: int) -> bool:
    """The k-arm counterpart: epsilon-greedy over k arms logs eps/k or 1-eps+eps/k, nothing else."""
    if propensity is None or epsilon is None or epsilon <= 0.0 or n_arms <= 0:
        return False
    explore = epsilon / n_arms
    return any(
        abs(propensity - expected) < _PROPENSITY_TOLERANCE
        for expected in (explore, 1.0 - epsilon + explore)
    )


def _as_float(value: object) -> float | None:
    """A logged numeric field as a finite float, or None when it is not one — missing, not error."""
    # "Not finite" is one case of "not a number", not a separate one: a JSON-decoded `nan` or
    # `inf` is as unusable as a string. It matters most for `propensity`, the DIVISOR in the
    # importance weights — `_usable` (0 < p < 1) and `_is_randomized` already reject a non-finite
    # propensity by arithmetic, so this guard states the same rule explicitly, keeping a `nan` off
    # `ExplorationLogRow.propensity` where a future caller could read it without that filter.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    coerced = float(value)
    return coerced if math.isfinite(coerced) else None


def _numeric_map(value: object) -> dict[str, float]:
    """A logged score/feature mapping coerced to floats, dropping only what cannot be one."""
    # A JSON round-trip can hand back a number as a string, and the pre-promotion code parsed
    # those. Dropping them would be a silent data loss, so coercion is tried and only a value
    # that genuinely is not a number is skipped — where the old code raised. Tolerating rather
    # than raising is `_as_float`'s established convention right above: a non-numeric logged
    # value is MISSING, not an error, and the estimator excludes it.
    if not isinstance(value, dict):
        return {}
    coerced = {str(k): _coerce_float(v) for k, v in value.items()}
    return {k: v for k, v in coerced.items() if v is not None}


def _coerce_float(value: object) -> float | None:
    """A logged value as a finite float, or None when it is not one in any representation."""
    # `float()` happily parses "nan", "inf" and "Infinity", and a non-finite reward or feature
    # then traverses IPS, SNIPS and DR unguarded, poisoning every estimate that touches it while
    # still looking like a number. It is MISSING, exactly like any other non-number.
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _row_from_record(record: Mapping[str, object]) -> ExplorationLogRow:
    # Two logged shapes reach the same row. The ESCALATION shape is binary (`propensity`, an
    # `action` naming a rung) and the ROUTING shape is multi-arm (`selection_propensity`, an
    # `action` naming the chosen model, `candidate_model_scores` as the features). The
    # discriminator is `selection_propensity`, the column that exists only on the routing side.
    if "selection_propensity" in record:
        return _routing_row(record)
    return _escalation_row(record)


def _escalation_row(record: Mapping[str, object]) -> ExplorationLogRow:
    propensity = _as_float(record.get("propensity"))
    action = str(record.get("action", ""))
    return ExplorationLogRow(
        checkpoint_id=str(record.get("checkpoint_id", "")),
        escalated=action in _ESCALATING_ACTIONS,
        propensity=propensity if propensity is not None else 1.0,
        reward=_reward_of(record),
        randomized=_is_randomized(propensity, _as_float(record.get("epsilon"))),
        features=_numeric_map(record.get("features")),
        # `OutcomeStore._exploration_row` has always returned this; the estimator dropped it on
        # the floor and then clustered on the wrong thing. It is the reward's own unit.
        session_id=str(record.get("session_id", "")),
        action=action,
    )


def _routing_row(record: Mapping[str, object]) -> ExplorationLogRow:
    # `escalated` is False on every routing row, so the BINARY estimator sees a one-armed log and
    # refuses it at the per-arm floor. That is deliberate: a multi-arm value needs a multi-arm
    # estimator, and returning a number off `escalated=False` would be that estimator's absence
    # dressed as its answer. The row exists so overlap diagnostics (weights, ESS) can be read now.
    #
    # NOT YET REACHABLE WITH EFFECT: the routing provenance the router writes today carries no
    # `epsilon`, so `_is_randomized_over_arms` returns False and every live routing row is
    # excluded. Whoever adds the routing reader must persist the epsilon alongside the
    # propensity, or this branch will refuse the whole log for a reason that reads as "no
    # exploration happened" when what really happened is that nobody logged the epsilon.
    scores = _numeric_map(record.get("candidate_model_scores"))
    propensity = _as_float(record.get("selection_propensity"))
    return ExplorationLogRow(
        checkpoint_id=str(record.get("checkpoint_id", "")),
        escalated=False,
        propensity=propensity if propensity is not None else 1.0,
        reward=_reward_of(record),
        randomized=_is_randomized_over_arms(
            propensity, _as_float(record.get("epsilon")), len(scores)
        ),
        features=scores,
        session_id=str(record.get("session_id", "")),
        # The plan's routing shape calls the arm `action`; the provenance the router writes today
        # calls it `model_chosen`. Both are accepted so the reader is not the thing that breaks.
        action=str(record.get("action") or record.get("model_chosen") or ""),
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
    n_clusters = len({_cluster_key(r) for r in rows})
    if n_clusters < _MIN_CLUSTERS:
        return (
            f"too few independent sessions: {n_clusters} — every decision in one session shares "
            f"that session's verified outcome, so this is not {len(rows)} observations and no "
            f"interval is honest"
        )
    weakest = min(r.propensity for r in rows)
    if weakest < _MIN_PROPENSITY:
        return (
            f"overlap is nominal only: the smallest logged propensity is {weakest:.2e}, below "
            f"the {_MIN_PROPENSITY} floor — its inverse would dominate every other term"
        )
    return None


def _cluster_key(row: ExplorationLogRow) -> str:
    """The row's independent unit: its session, or its checkpoint on a log that carries none."""
    return row.session_id or row.checkpoint_id


def _direct_method(rows: Sequence[ExplorationLogRow]) -> dict[bool, float]:
    """qhat(x, a): the per-arm mean reward (the model component of the DR estimator)."""
    means: dict[bool, float] = {}
    for arm in (True, False):
        rewards = [r.reward for r in rows if r.escalated is arm and r.reward is not None]
        means[arm] = statistics.fmean(rewards) if rewards else 0.0
    return means


def _crossfit_qhat(rows: Sequence[ExplorationLogRow]) -> list[dict[bool, float]]:
    """Each row's qhat, fitted on the OTHER cluster folds — never on the row's own fold."""
    # In-sample fitting makes the DR residual `r_i - qhat(x_i, a_i)` optimistically small: the
    # mean it is measured against was computed FROM r_i. The bias is small here (qhat is two
    # scalars) but it is real and it always points the same way — a tighter interval — so it is
    # removed rather than caveated. Folds are assigned by CLUSTER, not by row, or the row's own
    # session would leak into its training half through its siblings.
    #
    # THE EMPTY-ARM FALLBACK IS THE ARM'S GLOBAL MEAN, NEVER 0.0 — and it is implemented HERE, at
    # the fold boundary, so a fold whose complement holds no rows of an arm does not hand that
    # fold's rows a fabricated zero. A zero made the DR correction `w * (r_i - 0)` manufacture
    # variation out of the propensity weights rather than real outcome variation: measured on a
    # constant-reward log whose escalate rows sat in one cross-fit fold, the shipped code returned
    # `identified` with dr=0.918 and a 90% band [0.633, 1.122] off rows that carry no variation at
    # all — the module's own documented "zero-width interval is no evidence" failure, re-entered
    # through the model component instead of the bootstrap. With the global-mean fallback every DR
    # term lands on the same value and the degenerate-interval guard refuses, as it should.
    keys = sorted({_cluster_key(r) for r in rows})
    fold_of_key = {key: index % _N_FOLDS for index, key in enumerate(keys)}
    folds = [fold_of_key[_cluster_key(r)] for r in rows]
    global_means = _direct_method(rows)
    fitted: dict[int, dict[bool, float]] = {}
    for fold in set(folds):
        complement = [r for r, f in zip(rows, folds, strict=True) if f != fold]
        means = _direct_method(complement)
        # A usable row's reward is never None, so arm presence in the complement is the whole
        # test: an arm with rows has a measured mean; an arm with none falls back to its global
        # mean over ALL usable rows (the contract `_N_FOLDS`' comment promises).
        filled = {
            arm: (means[arm] if any(r.escalated is arm for r in complement) else global_means[arm])
            for arm in (True, False)
        }
        fitted[fold] = filled
    return [fitted[f] for f in folds]


def _weights(
    rows: Sequence[ExplorationLogRow],
    target: Callable[[ExplorationLogRow], float],
    weight_clip: float,
) -> tuple[list[float], int]:
    """The clipped importance weights pi(a_i|x_i)/p_i, and how many the clip actually bound."""
    # One definition, shared by DR, IPS and SNIPS. Three estimators computing their own weights is
    # three chances for them to disagree about clipping, which would make a disagreement between
    # them read as a finding about the logs rather than as a bug here.
    weights: list[float] = []
    n_clipped = 0
    for row in rows:
        p_escalate = min(max(target(row), 0.0), 1.0)
        target_prob = p_escalate if row.escalated else 1.0 - p_escalate
        raw_weight = target_prob / row.propensity
        n_clipped += raw_weight > weight_clip
        weights.append(min(raw_weight, weight_clip))
    return weights, n_clipped


def _dr_terms(
    rows: Sequence[ExplorationLogRow],
    target: Callable[[ExplorationLogRow], float],
    weight_clip: float,
) -> tuple[list[float], list[float], int]:
    """The per-decision DR contributions, the weights they used, and how many were clipped."""
    qhats = _crossfit_qhat(rows)
    weights, n_clipped = _weights(rows, target, weight_clip)
    terms: list[float] = []
    for row, qhat, weight in zip(rows, qhats, weights, strict=True):
        p_escalate = min(max(target(row), 0.0), 1.0)
        baseline = p_escalate * qhat[True] + (1.0 - p_escalate) * qhat[False]
        residual = (row.reward or 0.0) - qhat[row.escalated]
        terms.append(baseline + weight * residual)
    return terms, weights, n_clipped


def effective_sample_size(weights: Sequence[float]) -> float:
    """Kish ESS of importance weights: (sum w)^2 / sum w^2, and 0.0 when there is no mass."""
    total = sum(weights)
    total_sq = sum(w * w for w in weights)
    return (total * total) / total_sq if total_sq > 0.0 else 0.0


def ess_fraction(weights: Sequence[float]) -> float:
    """The effective sample size as a fraction of the nominal one — 0.0 on an empty log."""
    return effective_sample_size(weights) / len(weights) if weights else 0.0


def ips_estimate(
    rows: Sequence[ExplorationLogRow],
    target: Callable[[ExplorationLogRow], float] = always_escalate,
    *,
    weight_clip: float = _DEFAULT_WEIGHT_CLIP,
) -> float | None:
    """Inverse-propensity value of *target*: mean_i w_i r_i, or None when no row is usable."""
    # Unbiased before clipping and unnormalised, so it is the estimator that most visibly blows up
    # under a small propensity. Read it beside SNIPS and the ESS, never alone.
    usable = _usable(rows)
    if not usable:
        return None
    weights, _ = _weights(usable, target, weight_clip)
    return statistics.fmean([w * (r.reward or 0.0) for w, r in zip(weights, usable, strict=True)])


def snips_estimate(
    rows: Sequence[ExplorationLogRow],
    target: Callable[[ExplorationLogRow], float] = always_escalate,
    *,
    weight_clip: float = _DEFAULT_WEIGHT_CLIP,
) -> float | None:
    """Self-normalised IPS: sum(w r) / sum(w). None when no row is usable or the mass is zero."""
    # Biased but bounded by the reward range, where IPS is not. The normaliser is the SAME weight
    # vector IPS divides by n, so a gap between the two is a statement about weight dispersion.
    usable = _usable(rows)
    if not usable:
        return None
    weights, _ = _weights(usable, target, weight_clip)
    total = sum(weights)
    if total <= 0.0:
        return None
    return sum(w * (r.reward or 0.0) for w, r in zip(weights, usable, strict=True)) / total


def _clusters(rows: Sequence[ExplorationLogRow], terms: Sequence[float]) -> list[list[float]]:
    """Group DR terms by SESSION — the correlated unit, and so the bootstrap unit."""
    grouped: dict[str, list[float]] = {}
    for row, term in zip(rows, terms, strict=True):
        grouped.setdefault(_cluster_key(row), []).append(term)
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
    terms, weights, n_clipped = _dr_terms(usable, target, weight_clip)
    clusters = _clusters(usable, terms)
    low, high = _bootstrap_ci(clusters, bootstrap_draws, bootstrap_seed)
    if high <= low:
        # Every resample landed on the same number. That is no evidence, not perfect evidence, and
        # printing (0.0, 0.0) beside `identified` reads as the certainty this module exists to
        # refuse. Measured on probe logs returning dr=0.0 ci=(0.0, 0.0) and dr=1.0 ci=(1.0, 1.0).
        return PolicyValueEstimate(
            status=NOT_IDENTIFIED,
            reason=(
                f"degenerate interval: {len(clusters)} clusters produced a zero-width "
                f"{int(_CI_MASS * 100)}% bootstrap band at {low:.4f} — the rows carry no "
                "variation to resample, so this is no evidence rather than a certain value"
            ),
            n_clusters=len(clusters),
            **counts,
        )
    contrast, contrast_low, contrast_high = _contrast(
        usable, target, weight_clip, bootstrap_draws, bootstrap_seed
    )
    return PolicyValueEstimate(
        status=IDENTIFIED,
        reason=f"both arms realized under logged propensities ({int(_CI_MASS * 100)}% interval)",
        dr_estimate=statistics.fmean(terms),
        ci_low=low,
        ci_high=high,
        max_weight=max(weights, default=0.0),
        n_clusters=len(clusters),
        n_clipped=n_clipped,
        contrast_estimate=contrast,
        contrast_ci_low=contrast_low,
        contrast_ci_high=contrast_high,
        ess=effective_sample_size(weights),
        ess_fraction=ess_fraction(weights),
        **counts,
    )


def _contrast(
    rows: Sequence[ExplorationLogRow],
    target: Callable[[ExplorationLogRow], float],
    weight_clip: float,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> tuple[float, float, float]:
    """V(target) - V(its complement), paired per decision, bootstrapped over the clusters."""
    # The level V(always_escalate) cannot answer "does escalation help" — that needs a comparison,
    # and a reader given only a level will supply the missing comparator themselves (usually zero,
    # which is not the alternative). The difference is taken PER DECISION on the same rows, so the
    # bootstrap sees the variance of the difference rather than the sum of two variances.
    target_terms, _, _ = _dr_terms(rows, target, weight_clip)
    hold_terms, _, _ = _dr_terms(rows, lambda row: 1.0 - target(row), weight_clip)
    diffs = [t - h for t, h in zip(target_terms, hold_terms, strict=True)]
    low, high = _bootstrap_ci(_clusters(rows, diffs), bootstrap_draws, bootstrap_seed)
    return statistics.fmean(diffs), low, high


__all__ = [
    "IDENTIFIED",
    "NOT_IDENTIFIED",
    "ExplorationLogRow",
    "PolicyValueEstimate",
    "always_escalate",
    "effective_sample_size",
    "ess_fraction",
    "estimate_policy_value",
    "ips_estimate",
    "never_escalate",
    "rows_from_records",
    "snips_estimate",
]
