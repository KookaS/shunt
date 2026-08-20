"""The instrument-validity gate for the inference off-policy estimates.

Certifies IPS, SNIPS and DR on both legs against a planted signal and a shuffled-label null,
end to end through a real ``OutcomeStore``; no estimate is quotable without its verdict.
"""

# WHY THE CONTROL ENTERS AT THE FRONT. A control injected at the importance weights certifies
# only the arithmetic. Both legs here WRITE synthetic sessions into a real temporary
# `OutcomeStore` and read them back through the production accessors
# (`routing_ope_rows` / `escalation_exploration_rows`), so a failure implicates the whole
# assembled path: the `bench:`/live discrimination, the `selection_propensity` column
# round-trip, the `decision_provenance` JSON encode/decode, the tier2-over-tier1 outcome
# coalescing, the reward vocabulary, the epsilon-derived randomization test, the clustering
# key, the cross-fitted qhat AND the estimator. Nothing here re-derives an estimator or an
# adjudicator: `shunt.analysis.ope` computes the values and
# `shunt.analysis.admissibility` returns every verdict.
#
# WHY THE MULTI-ARM ROUTING LEG IS SCORED THROUGH THE BINARY ESTIMATOR. The target routing
# policy is DETERMINISTIC — take the top-scored candidate — so pi(a_i|x_i) is 1 or 0 and the
# k-arm problem reduces EXACTLY to the binary one: "did this decision take the target's
# action". `routing_agreement_rows` re-encodes `escalated` as that agreement, which makes
# `qhat[True]` the target arm's mean outcome and gives every disagreeing row weight 0, exactly
# as the k-arm DR estimator would. The re-encoding happens at the estimator boundary, after the
# accessor and `rows_from_records`, so it changes the policy question and nothing upstream.
# ONE ASYMMETRY THE REDUCTION CARRIES, which a figure must not paper over: the LEVEL V(pi*) is
# exact, but `PolicyValueEstimate.contrast_*` on a routing leg is V(top-scored) minus the value
# of the reduction's complement — "took some other arm", an arbitrary mixture of the remaining
# k-1 candidates, not a routing policy anyone could deploy. The escalation leg's contrast IS a
# two-arm decision and means what it says. Never present the two contrasts as comparable.
#
# WHAT THE VERDICT DOES AND DOES NOT COVER. The positive and destroyed-signal SCORES are taken
# through `estimate_policy_value` for DR — the same orchestrator a figure quotes — so its overlap
# floors, its degenerate-interval guard and its SESSION clustering are inside the certified path:
# a regression there refuses the positive control and the instrument is rejected. Only the 200
# band permutations use the cheaper `_dr_point`, which averages the identical `_dr_terms` the
# orchestrator averages on its identified path, so the band and the score remain the same
# statistic. Two things stay OUTSIDE the verdict and are asserted in the test file instead: the
# CI's width and the paired contrast (the null corpus must come back identified, at chance, with
# a contrast that does NOT exclude zero). And nothing here can certify the WRITER: the live
# router does not persist an `epsilon` alongside the routing propensity, so an ADMISSIBLE verdict
# says the read path recovers a signal that is present, never that the rig logs one.
#
# WHAT THIS GATE CANNOT SEE. The measured bands are 8-14 reward-rate points wide and the worst
# positive margin over 20 seeds is 1.62 bands. That is a gate against BREAKAGE — a front end that
# stops discriminating, a filter that stops filtering, an estimator that stops weighting — not a
# gate against a subtle few-percent degradation, which would still clear. Do not read ADMISSIBLE
# as "this estimator is accurate".
#
# WHY THE PLANTED REWARD CARRIES NOISE. A noiseless control makes every DR term identical, the
# bootstrap band zero-width, and `estimate_policy_value`'s own degenerate-interval guard refuse
# the positive control — the instrument would be rejected for being fed a perfect signal. The
# planted signal is therefore strong but not perfect, which is also the only version of this
# control that proves POWER rather than mere detectability.
#
# WHY THE NULL SCORE COMES FROM SHUFFLES THE BAND NEVER SAW. The band is the 97.5th percentile of
# |V-hat - chance| over `_DEFAULT_SHUFFLES` permutations at `seed`. Scoring the null from that
# same set would make `null_at_chance` true by construction and the null leg vacuous, which is
# the precise failure a positive-only gate has. The reported null is the MEDIAN of an independent
# replicate set drawn at a disjoint seed — a location, not a single draw: one draw from a
# distribution whose own 97.5th percentile IS the band lands outside it 2.5% of the time, so a
# single-draw null would make the verdict flip on reseed for no reason but sampling. A pipeline
# that manufactures signal from noise still fails, because leakage moves the whole null
# distribution and its median with it.

from __future__ import annotations

import random
import statistics
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

from shunt.analysis.admissibility import AdmissibilityResult, admissibility_verdict
from shunt.analysis.ope import (
    _DEFAULT_WEIGHT_CLIP,
    ExplorationLogRow,
    PolicyValueEstimate,
    _dr_terms,
    _usable,
    always_escalate,
    estimate_policy_value,
    ips_estimate,
    rows_from_records,
    snips_estimate,
)
from shunt.db.store import OutcomeStore, SessionProvenance

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

# `_dr_terms`, `_usable` and `_DEFAULT_WEIGHT_CLIP` are `shunt.analysis.ope`'s own internals,
# imported rather than reimplemented: the control has to score the DR POINT estimate several
# hundred times per leg, and paying `estimate_policy_value`'s 2000-draw cluster bootstrap for
# each permutation would cost minutes without changing a single band. Reaching for the private
# helpers keeps the control and the shipped number on ONE weight definition and ONE qhat — a
# local copy would let them drift apart, which is the failure this whole module exists to catch.

ESTIMATORS: Final[tuple[str, ...]] = ("ips", "snips", "dr")
ROUTING: Final[str] = "routing"
ESCALATION: Final[str] = "escalation"

# Pinned so a verdict is reproducible; `certify_instrument` takes them as arguments so the test
# can sweep the seed and assert the verdict does not move.
_DEFAULT_SEED: Final[int] = 20260818
_DEFAULT_SESSIONS: Final[int] = 400
_DEFAULT_SHUFFLES: Final[int] = 200
_BAND_MASS: Final[float] = 0.975
# Disjoint from every band permutation: `random.Random(seed)` and `random.Random(seed + offset)`
# are independent streams, and the offset is prime so no small seed sweep can collide.
_NULL_SEED_OFFSET: Final[int] = 10007
_NULL_REPLICATES: Final[int] = 25
# A band of exactly zero would divide by zero in the aggregate and reject every honest
# instrument in the adjudicator; it can only arise if all permutations land on chance exactly.
_MIN_BAND: Final[float] = 1e-9
# Bootstrap draws for the orchestrated control scores. Lower than the shipped default because the
# SCORE being read off is `dr_estimate`, which no number of draws changes — the draws exist here
# only to exercise the degenerate-interval guard and the cluster resampling, which 400 does
# (verified: identical `dr_estimate` at 400 and at the 2000 default, 4.5x cheaper).
_CONTROL_BOOTSTRAP: Final[int] = 400

_ARMS: Final[tuple[str, str, str]] = ("control-a", "control-b", "control-c")
# The logging policy's greedy arm. It is right only when the latent target happens to be
# `_ARMS[0]`, i.e. half the time — deliberately wrong, so the target policy's value is far from
# the on-policy mean and both propensity levels are realised.
_GREEDY_ARM: Final[str] = _ARMS[0]
_TARGET_ARMS: Final[int] = 2
_EPSILON: Final[float] = 0.3
_LABEL_NOISE: Final[float] = 0.1
_ESCALATION_NEED_RATE: Final[float] = 0.9
# Fewer than `ope._MIN_CLUSTERS`, on purpose: if the clustering key ever regresses from the
# session back to the checkpoint, the escalation leg drops to 3 clusters, the estimator refuses,
# and the positive control fails loudly. That regression is the one this module's dependency
# documents as already having shipped once.
_CONTROL_CHECKPOINTS: Final[int] = 3


class InstrumentInadmissibleError(RuntimeError):
    """Raised when the assembled OPE pipeline fails its positive control or its null."""


@dataclass(frozen=True)
class EstimatorCertificate:
    """One (leg, estimator) pair's two-sided instrument verdict and the controls behind it."""

    # REQUIRED and FIRST, with no default, for the same reason `CertifiedEstimates` carries one.
    admissibility: AdmissibilityResult
    leg: str
    estimator: str
    n_rows: int
    n_shuffles: int
    seed: int

    @property
    def key(self) -> tuple[str, str]:
        """The (leg, estimator) pair this certificate speaks for."""
        return (self.leg, self.estimator)

    @property
    def headline(self) -> str:
        """This pair's verdict in its own reward-rate units, quotable verbatim."""
        return f"{self.leg}/{self.estimator}: {self.admissibility.headline}"


@dataclass(frozen=True)
class CertifiedEstimates:
    """Off-policy estimates plus the instrument verdict that earns them the right to be drawn."""

    # REQUIRED and FIRST, with no default, for the same reason `benchmark.routing.summary`'s
    # `StrategyTable` carries one: this is the artifact an off-policy number is read off, and it
    # can only exist once the caller has stated whether the ASSEMBLED path (store write ->
    # production accessor -> `rows_from_records` -> encoding -> estimator) recovers a signal it
    # is known to contain and collapses to chance when that signal is destroyed. A defaulted
    # field would let every call site keep compiling and keep publishing, which is precisely how
    # the gate went unwired the first time.
    admissibility: AdmissibilityResult
    certificates: tuple[EstimatorCertificate, ...]
    routing: PolicyValueEstimate
    escalation: PolicyValueEstimate

    @property
    def headline(self) -> str:
        """One line a manifest note or a figure footer can carry verbatim."""
        # The aggregate adjudication runs in BAND-NORMALISED units, so its scores are multiples
        # of each control's own band and not reward rates. Saying so is the difference between a
        # quotable line and a misleading one; the per-leg numbers live in `certificates`.
        return (
            f"{self.admissibility.headline} Scores are band-normalised worst cases over "
            f"{len(self.certificates)} (leg, estimator) controls."
        )

    def certificate(self, leg: str, estimator: str) -> EstimatorCertificate | None:
        """The certificate for one (leg, estimator) pair, or None if it was never run."""
        return next((c for c in self.certificates if c.key == (leg, estimator)), None)

    def quotable(self, leg: str, estimator: str) -> bool:
        """True iff this estimator cleared BOTH controls on this leg and may be drawn."""
        certificate = self.certificate(leg, estimator)
        return certificate is not None and certificate.admissibility.admissible

    def require_admissible(self) -> None:
        """Raise unless every certificate cleared — the render driver's non-zero exit."""
        if not self.admissibility.admissible:
            raise InstrumentInadmissibleError(self.admissibility.reason)


def routing_agreement_rows(rows: Sequence[ExplorationLogRow]) -> list[ExplorationLogRow]:
    """Re-encode multi-arm routing rows as "took the top-scored candidate", the target policy."""
    return [replace(row, escalated=_took_target_arm(row)) for row in rows]


def _took_target_arm(row: ExplorationLogRow) -> bool:
    """True iff the logged arm is the argmax of the candidate scores (ties broken by name)."""
    if not row.features or not row.action:
        return False
    best = max(row.features.items(), key=lambda item: (item[1], item[0]))[0]
    return row.action == best


def _identity_rows(rows: Sequence[ExplorationLogRow]) -> list[ExplorationLogRow]:
    """The escalation leg is already binary — its arm needs no re-encoding."""
    return list(rows)


def _dr_point(rows: Sequence[ExplorationLogRow]) -> float | None:
    """The DR point estimate under the shipped weights and qhat, without the bootstrap."""
    usable = _usable(rows)
    if not usable:
        return None
    terms, _, _ = _dr_terms(usable, always_escalate, _DEFAULT_WEIGHT_CLIP)
    return statistics.fmean(terms)


def _dr_orchestrated(rows: Sequence[ExplorationLogRow]) -> float | None:
    """The DR value as the SHIPPED orchestrator reports it — None when it refuses to report one."""
    # `estimate_policy_value` is the function a figure quotes, and it carries the parts
    # `_dr_point` skips: the per-arm and cluster floors, the minimum-propensity floor and the
    # degenerate-interval guard. Scoring both controls through it puts all of that inside the
    # verdict — a refusal becomes a -inf positive control and the instrument is rejected. On the
    # identified path its `dr_estimate` IS `fmean(_dr_terms(...))`, so it is the same statistic
    # the 200 band permutations measure through `_dr_point`.
    return estimate_policy_value(rows, bootstrap_draws=_CONTROL_BOOTSTRAP).dr_estimate


def _score(
    rows: Sequence[ExplorationLogRow], estimator: str, *, orchestrated: bool = False
) -> float | None:
    """One estimator's value of the target policy on these rows, or None when none is usable."""
    if estimator == "ips":
        return ips_estimate(rows, always_escalate)
    if estimator == "snips":
        return snips_estimate(rows, always_escalate)
    if estimator == "dr":
        return _dr_orchestrated(rows) if orchestrated else _dr_point(rows)
    # Never silently fall through to one estimator's number under another's name.
    raise ValueError(f"unknown estimator {estimator!r}; expected one of {ESTIMATORS}")


def _permuted(
    records: Sequence[Mapping[str, object]], rng: random.Random
) -> list[dict[str, object]]:
    """The SAME decisions with their outcomes permuted — actions and propensities held fixed."""
    outcomes = [record.get("outcome") for record in records]
    rng.shuffle(outcomes)
    paired = zip(records, outcomes, strict=True)
    return [{**record, "outcome": outcome} for record, outcome in paired]


def _null_deviations(  # noqa: PLR0913 (every knob pinned by the caller, for reproducibility)
    records: Sequence[Mapping[str, object]],
    encode: Callable[[Sequence[ExplorationLogRow]], list[ExplorationLogRow]],
    chance: float,
    seed: int,
    n_shuffles: int,
    *,
    signed: bool = False,
    scored: bool = False,
) -> dict[str, list[float]]:
    """Shuffled-label scores per estimator: |V-hat - chance| by default, raw when *signed*."""
    rng = random.Random(seed)
    out: dict[str, list[float]] = {name: [] for name in ESTIMATORS}
    for _ in range(n_shuffles):
        rows = encode(rows_from_records(_permuted(records, rng)))
        for name in ESTIMATORS:
            score = _score(rows, name, orchestrated=scored)
            # An unscoreable permutation is a broken front end, never a tight band: it is
            # pushed to infinity so it widens nothing and fails the aggregate loudly.
            if score is None:
                out[name].append(float("inf"))
            else:
                out[name].append(score if signed else abs(score - chance))
    return out


def _quantile(values: Sequence[float], mass: float) -> float:
    """The empirical `mass` quantile of `values` (nearest-rank), or 0.0 when there are none."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(round(mass * (len(ordered) - 1))), len(ordered) - 1)]


def _chance_level(rows: Sequence[ExplorationLogRow]) -> float:
    """The COMPUTED marginal reward rate over the rows the estimator may weight."""
    # Never a magic constant: this is what the logs actually paid, so it is where a
    # signal-destroyed estimate has to land.
    usable = _usable(rows)
    return statistics.fmean([row.reward or 0.0 for row in usable]) if usable else 0.0


def _certify_leg(
    records: Sequence[Mapping[str, object]],
    encode: Callable[[Sequence[ExplorationLogRow]], list[ExplorationLogRow]],
    leg: str,
    *,
    seed: int,
    n_shuffles: int,
) -> tuple[EstimatorCertificate, ...]:
    """Score, band and adjudicate all three estimators on one leg's control corpus."""
    rows = encode(rows_from_records(records))
    chance = _chance_level(rows)
    deviations = _null_deviations(records, encode, chance, seed, n_shuffles)
    held_out = _null_deviations(
        records, encode, 0.0, seed + _NULL_SEED_OFFSET, _NULL_REPLICATES, signed=True, scored=True
    )
    out: list[EstimatorCertificate] = []
    for name in ESTIMATORS:
        positive = _score(rows, name, orchestrated=True)
        null = statistics.median(held_out[name])
        verdict = admissibility_verdict(
            positive if positive is not None else float("-inf"),
            null,
            chance_level=chance,
            chance_band=max(_quantile(deviations[name], _BAND_MASS), _MIN_BAND),
        )
        out.append(
            EstimatorCertificate(
                admissibility=verdict,
                leg=leg,
                estimator=name,
                n_rows=len(_usable(rows)),
                n_shuffles=n_shuffles,
                seed=seed,
            )
        )
    return tuple(out)


def overall_admissibility(
    certificates: Sequence[EstimatorCertificate],
) -> AdmissibilityResult:
    """The conjunction of every certificate, adjudicated in band-normalised units."""
    # Each leg has its own chance level and its own band, so the scores cannot be pooled raw.
    # Expressing every control as a MULTIPLE OF ITS OWN BAND makes them commensurable, and
    # adjudicating (min positive, max |null|) against chance 0 +/- 1 band reproduces the
    # conjunction exactly: admissible iff every positive cleared and every null stayed inside.
    positives: list[float] = []
    nulls: list[float] = []
    for certificate in certificates:
        result = certificate.admissibility
        band = max(result.chance_band, _MIN_BAND)
        positives.append((result.positive_score - result.chance_level) / band)
        nulls.append(abs(result.shuffled_score - result.chance_level) / band)
    return admissibility_verdict(
        min(positives, default=float("-inf")),
        max(nulls, default=float("inf")),
        chance_level=0.0,
        chance_band=1.0,
    )


def _timestamp(session_id: str) -> str:
    """A deterministic, ordered stamp derived from the control session's index."""
    index = int(session_id.rsplit(":", 1)[1])
    return f"2020-01-01T{index // 3600:02d}:{index // 60 % 60:02d}:{index % 60:02d}+00:00"


def _store_control_session(
    store: OutcomeStore,
    session_id: str,
    model: str,
    propensity: float | None,
    reward: bool,
    provenance: dict[str, object],
) -> None:
    """Write one synthetic decision + its verified outcome through the production writers."""
    # Tier-1 is deliberately the WRONG label on every row: the accessors coalesce tier2 over
    # tier1, so a regression in that coalescing destroys the planted signal and the positive
    # control fails instead of quietly scoring noise.
    store.store_session(
        session_id=session_id,
        prompt_text=f"instrument control {session_id}",
        embedding=None,
        model_chosen=model,
        cost=0.0,
        cache_stats={},
        duration=0.0,
        timestamp=_timestamp(session_id),
        decision_provenance=provenance,
        provenance=SessionProvenance(cost_known=True, selection_propensity=propensity),
    )
    store.store_outcome(
        session_id,
        tier1_outcome="failure",
        tier1_confidence=1.0,
        tier2_outcome="success" if reward else "failure",
        tier2_confidence=1.0,
    )


def _routing_decision(rng: random.Random) -> tuple[str, str, float, bool]:
    """One epsilon-greedy routing draw: (target arm, arm taken, its propensity, verified reward)."""
    # The SH008 suppressions on every `rng.random()` below: SH008 reads a draw off a bound
    # `random.Random` name as a fabricated feature vector, which is the right default. These
    # draws are neither features nor embeddings — they are the exploration coin and the label
    # noise of the PLANTED POSITIVE CONTROL, which the instrument-validity rule requires and
    # which `control_records` writes only into a TemporaryDirectory store. No real path draws.
    target = _ARMS[rng.randrange(_TARGET_ARMS)]
    arm = _ARMS[rng.randrange(len(_ARMS))] if rng.random() < _EPSILON else _GREEDY_ARM  # noqa: SH008
    # The propensity of the arm ACTUALLY TAKEN under epsilon-greedy over k arms: eps/k for an
    # off-greedy arm, 1 - eps + eps/k for the greedy one. `ope._is_randomized_over_arms` derives
    # randomization from exactly these two values, so a wrong propensity here excludes every row.
    propensity = _EPSILON / len(_ARMS) + (1.0 - _EPSILON if arm == _GREEDY_ARM else 0.0)
    reward = (arm == target) != (rng.random() < _LABEL_NOISE)  # noqa: SH008
    return target, arm, propensity, reward


def _routing_provenance(target: str) -> dict[str, object]:
    """The decision provenance a routing row carries: the candidate scores and the epsilon."""
    return {
        "candidate_model_scores": {arm: 1.0 if arm == target else 0.0 for arm in _ARMS},
        "epsilon": _EPSILON,
    }


def _write_routing_control(store: OutcomeStore, rng: random.Random, n_sessions: int) -> None:
    """Plant a recoverable routing signal: the top-scored candidate is the one that pays."""
    for index in range(n_sessions):
        target, arm, propensity, reward = _routing_decision(rng)
        _store_control_session(
            store, f"live:route:{index}", arm, propensity, reward, _routing_provenance(target)
        )


def _write_seeded_decoys(store: OutcomeStore, rng: random.Random, n_sessions: int) -> None:
    """`bench:`-prefixed rows carrying the INVERTED signal, which the live filter must drop."""
    # If `routing_ope_rows`' live discrimination regresses, these rows enter the corpus and
    # cancel the planted signal: the accessor's `bench:` filter is part of the assembled
    # instrument, not scenery. Written ONE FOR ONE with the live rows because that is what the
    # measurement demands — at a 1:4 decoy ratio a total filter regression still cleared the
    # band (positive 0.787 against chance 0.486 +/- 0.171), so a smaller decoy corpus would have
    # let the comment above claim a protection the numbers did not provide. At 1:1 the same
    # regression lands the positive control at 0.528 against chance 0.500 +/- 0.083 and the gate
    # refuses. A PARTIAL leak still only degrades the margin in proportion.
    for index in range(n_sessions):
        target, arm, propensity, reward = _routing_decision(rng)
        _store_control_session(
            store, f"bench:route:{index}", arm, propensity, not reward, _routing_provenance(target)
        )


def _write_escalation_control(store: OutcomeStore, rng: random.Random, n_sessions: int) -> None:
    """Plant a recoverable escalation signal: a checkpoint feature decides which arm pays."""
    for index in range(n_sessions):
        needs = rng.random() < _ESCALATION_NEED_RATE  # noqa: SH008 (control coin — see above)
        escalate = rng.random() < _EPSILON  # noqa: SH008 (control coin — see above)
        record = {
            "checkpoint_id": f"chk-{index % _CONTROL_CHECKPOINTS}",
            "action": "raise_effort" if escalate else "hold",
            "propensity": _EPSILON if escalate else 1.0 - _EPSILON,
            "epsilon": _EPSILON,
            "features": {"same_failure_count": 3.0 if needs else 0.0},
        }
        _store_control_session(
            store,
            f"live:esc:{index}",
            _GREEDY_ARM,
            None,
            escalate == needs,
            {"escalation_exploration": record},
        )


def control_records(
    *, seed: int = _DEFAULT_SEED, n_sessions: int = _DEFAULT_SESSIONS
) -> dict[str, list[dict[str, object]]]:
    """Both legs' control corpora, written to a real store and read back through the accessors."""
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory(prefix="shunt-instrument-") as tmp:
        store = OutcomeStore(db_path=str(Path(tmp) / "control.db"))
        # Durability off on a database that is deleted at the end of this `with`. It buys ~1000x
        # on the write leg (a per-row fsync dominated everything else), and it changes NOTHING
        # the control certifies: the same SQL, the same columns, the same JSON round-trip, the
        # same accessors. The production store is untouched — this pragma is set on the control
        # connection only.
        store._conn.execute("PRAGMA synchronous=OFF")  # noqa: SLF001
        _write_routing_control(store, rng, n_sessions)
        _write_seeded_decoys(store, rng, n_sessions)
        _write_escalation_control(store, rng, n_sessions)
        return {
            ROUTING: [dict(row) for row in store.routing_ope_rows()],
            ESCALATION: [dict(row) for row in store.escalation_exploration_rows()],
        }


def certify_instrument(
    *,
    seed: int = _DEFAULT_SEED,
    n_sessions: int = _DEFAULT_SESSIONS,
    n_shuffles: int = _DEFAULT_SHUFFLES,
) -> tuple[EstimatorCertificate, ...]:
    """Run the two-sided control on both legs and return one verdict per (leg, estimator)."""
    records = control_records(seed=seed, n_sessions=n_sessions)
    return _certify_leg(
        records[ROUTING], routing_agreement_rows, ROUTING, seed=seed, n_shuffles=n_shuffles
    ) + _certify_leg(
        records[ESCALATION], _identity_rows, ESCALATION, seed=seed, n_shuffles=n_shuffles
    )


def live_estimates(store: OutcomeStore) -> tuple[PolicyValueEstimate, PolicyValueEstimate]:
    """The (routing, escalation) off-policy values on a real store — a refusal is a valid one."""
    # NOT_IDENTIFIED is a first-class outcome, not an exception: on a rig whose logging policy
    # is deterministic there is nothing to estimate, and the figure ships that refusal verbatim.
    routing = estimate_policy_value(
        routing_agreement_rows(rows_from_records(store.routing_ope_rows()))
    )
    escalation = estimate_policy_value(rows_from_records(store.escalation_exploration_rows()))
    return routing, escalation


def certify(
    store: OutcomeStore,
    *,
    seed: int = _DEFAULT_SEED,
    n_sessions: int = _DEFAULT_SESSIONS,
    n_shuffles: int = _DEFAULT_SHUFFLES,
) -> CertifiedEstimates:
    """The store's off-policy estimates, refused unless the instrument clears both controls."""
    certificates = certify_instrument(seed=seed, n_sessions=n_sessions, n_shuffles=n_shuffles)
    routing, escalation = live_estimates(store)
    estimates = CertifiedEstimates(
        admissibility=overall_admissibility(certificates),
        certificates=certificates,
        routing=routing,
        escalation=escalation,
    )
    # Fail closed. A caller that wants to inspect an inadmissible instrument calls
    # `certify_instrument` directly; a caller that wants to DRAW gets nothing without a verdict.
    estimates.require_admissible()
    return estimates


__all__ = [
    "ESCALATION",
    "ESTIMATORS",
    "ROUTING",
    "CertifiedEstimates",
    "EstimatorCertificate",
    "InstrumentInadmissibleError",
    "certify",
    "certify_instrument",
    "control_records",
    "live_estimates",
    "overall_admissibility",
    "routing_agreement_rows",
]
