"""The identification guard: an OPE estimator that refuses logs without randomization.

The last section is the instrument-validity gate (R0), adjudicated by the SHARED gate
(`benchmark.admissibility`, the pinned copy of `admissibility_gate.py`).
"""

from __future__ import annotations

import dataclasses
import itertools
import random
import statistics
from typing import TYPE_CHECKING, Final

import pytest

from benchmark.admissibility import admissibility_verdict
from benchmark.escalation.ope import (
    IDENTIFIED,
    NOT_IDENTIFIED,
    ExplorationLogRow,
    always_escalate,
    estimate_policy_value,
    rows_from_records,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_COUNTER = itertools.count()


def _row(
    *,
    escalated: bool,
    propensity: float,
    reward: float,
    randomized: bool = True,
    depth: float = 3.0,
) -> ExplorationLogRow:
    # A distinct checkpoint per row = independent decisions, which is what these fixtures mean.
    return ExplorationLogRow(
        checkpoint_id=f"test::foo::{next(_COUNTER)}",
        escalated=escalated,
        propensity=propensity,
        reward=reward,
        randomized=randomized,
        features={"countable_failures": depth},
    )


def _randomized_log(n: int = 400, epsilon: float = 0.3, seed: int = 7) -> list[ExplorationLogRow]:
    """Synthetic epsilon-greedy log where escalating genuinely helps (unit-test fixture)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        held = rng.random() < epsilon
        escalated = not held
        propensity = epsilon if held else 1.0 - epsilon
        reward = 1.0 if rng.random() < (0.7 if escalated else 0.3) else 0.0
        rows.append(_row(escalated=escalated, propensity=propensity, reward=reward))
    return rows


# ── the refusal path — the most important behaviour in this module ──


def test_deterministic_log_is_reported_not_identified() -> None:
    rows = [_row(escalated=True, propensity=1.0, reward=1.0, randomized=False) for _ in range(500)]
    result = estimate_policy_value(rows, always_escalate)
    assert result.status == NOT_IDENTIFIED
    assert result.dr_estimate is None
    assert result.ci_low is None and result.ci_high is None
    assert "overlap" in result.reason


def test_all_zero_propensities_are_reported_not_identified() -> None:
    rows = [_row(escalated=False, propensity=0.0, reward=0.0, randomized=False) for _ in range(50)]
    result = estimate_policy_value(rows, always_escalate)
    assert result.status == NOT_IDENTIFIED
    assert result.dr_estimate is None


def test_one_armed_log_is_not_identified_even_when_propensities_look_random() -> None:
    # Every propensity is strictly inside (0,1) but only ONE arm was ever realized.
    rows = [_row(escalated=True, propensity=0.7, reward=1.0) for _ in range(300)]
    result = estimate_policy_value(rows, always_escalate)
    assert result.status == NOT_IDENTIFIED
    assert result.n_escalated == 300
    assert result.n_held == 0
    assert result.dr_estimate is None


def test_empty_log_is_not_identified() -> None:
    result = estimate_policy_value([], always_escalate)
    assert result.status == NOT_IDENTIFIED
    assert result.dr_estimate is None


def test_non_randomized_rows_are_excluded_before_the_overlap_check() -> None:
    # A ladder-exhausted row (propensity 1.0) must not manufacture a held arm.
    rows = [_row(escalated=True, propensity=0.7, reward=1.0) for _ in range(100)]
    excluded = _row(escalated=False, propensity=1.0, reward=0.0, randomized=False)
    rows += [excluded for _ in range(100)]
    result = estimate_policy_value(rows, always_escalate)
    assert result.status == NOT_IDENTIFIED
    assert result.n_decisions == 100
    assert result.n_excluded == 100


def test_a_row_with_a_missing_reward_is_excluded() -> None:
    rows = _randomized_log()
    rows.append(
        ExplorationLogRow(
            checkpoint_id="test::extra",
            escalated=True,
            propensity=0.7,
            reward=None,
            randomized=True,
            features={},
        )
    )
    result = estimate_policy_value(rows, always_escalate)
    assert result.n_excluded == 1


# ── the estimate path ──


def test_randomized_log_yields_a_finite_estimate_with_an_interval() -> None:
    result = estimate_policy_value(_randomized_log(), always_escalate)
    assert result.status == IDENTIFIED
    assert result.dr_estimate is not None
    assert result.ci_low is not None and result.ci_high is not None
    assert result.ci_low <= result.dr_estimate <= result.ci_high
    assert result.n_escalated > 0
    assert result.n_held > 0
    assert result.n_decisions == result.n_escalated + result.n_held


def test_the_estimate_recovers_the_planted_value_of_always_escalating() -> None:
    result = estimate_policy_value(_randomized_log(n=3000), always_escalate)
    assert result.dr_estimate is not None
    assert result.dr_estimate == pytest.approx(0.7, abs=0.06)


def test_the_estimate_recovers_the_planted_value_of_never_escalating() -> None:
    result = estimate_policy_value(_randomized_log(n=3000), lambda row: 0.0)
    assert result.dr_estimate is not None
    assert result.dr_estimate == pytest.approx(0.3, abs=0.06)


def test_the_interval_narrows_with_more_decisions() -> None:
    small = estimate_policy_value(_randomized_log(n=200), always_escalate)
    large = estimate_policy_value(_randomized_log(n=4000), always_escalate)
    assert small.ci_high is not None and small.ci_low is not None
    assert large.ci_high is not None and large.ci_low is not None
    assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)


def test_weight_clipping_bounds_the_importance_ratio() -> None:
    rows = _randomized_log(epsilon=0.02, n=800, seed=3)
    result = estimate_policy_value(rows, always_escalate, weight_clip=5.0)
    assert result.dr_estimate is not None
    assert result.max_weight <= 5.0


def test_the_estimator_is_deterministic_for_a_given_bootstrap_seed() -> None:
    rows = _randomized_log()
    first = estimate_policy_value(rows, always_escalate, bootstrap_seed=42)
    second = estimate_policy_value(rows, always_escalate, bootstrap_seed=42)
    assert first.ci_low == second.ci_low
    assert first.ci_high == second.ci_high


# ── the router -> estimator bridge ──


def test_rows_from_records_maps_the_router_log_shape() -> None:
    records = [
        {
            "checkpoint_id": "test::foo",
            "action": "raise_effort",
            "policy_action": "raise_effort",
            "propensity": 0.8,
            "randomized": True,
            "features": {"countable_failures": 2.0},
            "outcome": "success",
        },
        {
            "checkpoint_id": "test::foo",
            "action": "hold",
            "policy_action": "raise_effort",
            "propensity": 0.2,
            "randomized": True,
            "features": {"countable_failures": 2.0},
            "outcome": "failure",
        },
    ]
    rows = rows_from_records(records)
    assert [r.escalated for r in rows] == [True, False]
    assert [r.reward for r in rows] == [1.0, 0.0]


def test_rows_from_records_leaves_an_unlabelled_decision_rewardless() -> None:
    rows = rows_from_records(
        [{"action": "hold", "propensity": 0.2, "randomized": True, "outcome": None}]
    )
    assert rows[0].reward is None


# ── guards added after adversarial review ──


def test_a_two_row_log_is_refused_for_lack_of_evidence() -> None:
    # Reporting a point estimate and a 90% interval off n=2 is exactly the failure this
    # module exists to prevent; arm existence is necessary, not sufficient.
    rows = [
        _row(escalated=True, propensity=0.7, reward=1.0),
        _row(escalated=False, propensity=0.3, reward=0.0),
    ]
    result = estimate_policy_value(rows, always_escalate)
    assert result.status == NOT_IDENTIFIED
    assert result.dr_estimate is None


def test_a_vanishing_propensity_does_not_count_as_overlap() -> None:
    rows = [_row(escalated=True, propensity=0.7, reward=1.0) for _ in range(50)]
    rows += [_row(escalated=False, propensity=1e-12, reward=0.0) for _ in range(50)]
    result = estimate_policy_value(rows, always_escalate)
    assert result.status == NOT_IDENTIFIED
    assert "propensity" in result.reason


def test_a_self_declared_randomized_flag_cannot_manufacture_identification() -> None:
    # The flag is derived from epsilon and cross-checked against the propensity, not trusted.
    records = [
        {
            "action": "raise_effort" if i % 2 else "hold",
            "propensity": 0.5,
            "epsilon": 0.0,  # deterministic policy — 0.5 could not have been produced
            "randomized": True,
            "outcome": "success",
        }
        for i in range(100)
    ]
    result = estimate_policy_value(rows_from_records(records), always_escalate)
    assert result.status == NOT_IDENTIFIED


def test_a_propensity_inconsistent_with_epsilon_is_excluded() -> None:
    records = [
        {
            "action": "hold",
            "propensity": 0.9,  # neither epsilon nor 1-epsilon
            "epsilon": 0.3,
            "randomized": True,
            "outcome": "success",
        }
    ]
    assert rows_from_records(records)[0].randomized is False


def test_a_null_propensity_is_refused_not_crashed() -> None:
    rows = rows_from_records([{"action": "hold", "propensity": None, "randomized": True}])
    assert rows[0].randomized is False


def test_unknown_and_infra_outcomes_are_excluded_not_scored_as_failures() -> None:
    for outcome in ("unknown", "infra_failure", "something_new"):
        rows = rows_from_records(
            [{"action": "hold", "propensity": 0.3, "epsilon": 0.3, "outcome": outcome}]
        )
        assert rows[0].reward is None, outcome
    for outcome, reward in (("success", 1.0), ("weak_success", 1.0), ("failure", 0.0)):
        rows = rows_from_records(
            [{"action": "hold", "propensity": 0.3, "epsilon": 0.3, "outcome": outcome}]
        )
        assert rows[0].reward == reward, outcome


def test_the_interval_is_bootstrapped_over_clusters_not_correlated_decisions() -> None:
    # Decisions on the same recurring checkpoint are correlated (an explored hold does not
    # retire the window, so the same key re-flags). Resampling them independently understates
    # the interval badly — measured 90% coverage was 0.155 before this was fixed.
    rng = random.Random(5)
    correlated: list[ExplorationLogRow] = []
    for cluster in range(40):
        shared = 1.0 if rng.random() < 0.5 else 0.0  # the task outcome, shared by the cluster
        for _ in range(10):
            held = rng.random() < 0.3
            correlated.append(
                ExplorationLogRow(
                    checkpoint_id=f"c{cluster}",
                    escalated=not held,
                    propensity=0.3 if held else 0.7,
                    reward=shared,
                    randomized=True,
                    features={},
                )
            )
    independent = [
        ExplorationLogRow(f"c{i}", r.escalated, r.propensity, r.reward, True, {})
        for i, r in enumerate(correlated)
    ]
    clustered = estimate_policy_value(correlated, always_escalate)
    unclustered = estimate_policy_value(independent, always_escalate)
    assert clustered.ci_high is not None and clustered.ci_low is not None
    assert unclustered.ci_high is not None and unclustered.ci_low is not None
    # Same 400 rows, same point estimate — but 40 real clusters must give a WIDER interval.
    assert (clustered.ci_high - clustered.ci_low) > 2 * (unclustered.ci_high - unclustered.ci_low)


def test_one_session_across_many_checkpoints_is_not_five_observations() -> None:
    # The clustering probe that exposed the wrong unit: 60 decisions from ONE session touching 5
    # failing-test ids cleared the 5-cluster floor and reported `identified`. The reward is the
    # SESSION's verified outcome, so all 60 share one grade — that is one observation, not five.
    rng = random.Random(11)
    rows = [
        ExplorationLogRow(
            checkpoint_id=f"pkg::test_{i % 5}",
            escalated=rng.random() >= 0.3,
            propensity=0.7,
            reward=1.0,
            randomized=True,
            features={},
            session_id="one-session",
        )
        for i in range(60)
    ]
    result = estimate_policy_value(rows, always_escalate)
    assert result.status == NOT_IDENTIFIED
    assert "sessions" in result.reason
    assert result.dr_estimate is None


def test_the_cluster_unit_is_the_session_not_the_checkpoint() -> None:
    # Two sessions that happen to hit the SAME failing test are two observations, and one session
    # that hits many tests is still one. Clustering on checkpoint_id got both backwards.
    rows = [
        ExplorationLogRow(
            checkpoint_id="pkg::shared",
            escalated=i % 2 == 0,
            propensity=0.7 if i % 2 == 0 else 0.3,
            reward=float(i % 3 == 0),
            randomized=True,
            features={},
            session_id=f"s{i // 4}",
        )
        for i in range(80)
    ]
    result = estimate_policy_value(rows, always_escalate)
    assert result.n_clusters == 20  # 80 decisions / 4 per session, NOT 1 shared checkpoint


def test_a_zero_width_interval_is_refused_rather_than_reported_as_certainty() -> None:
    # dr=1.0 with ci=(1.0, 1.0) used to print beside `identified`. Every resample landing on the
    # same value is NO evidence — the rows carry no variation — not perfect evidence.
    rows = [
        ExplorationLogRow(
            checkpoint_id=f"c{i}",
            escalated=i % 2 == 0,
            propensity=0.5,
            reward=1.0,  # every reward identical -> every bootstrap draw identical
            randomized=True,
            features={},
            session_id=f"s{i}",
        )
        for i in range(40)
    ]
    result = estimate_policy_value(rows, always_escalate)
    assert result.status == NOT_IDENTIFIED
    assert "degenerate interval" in result.reason
    assert result.dr_estimate is None
    assert result.ci_low is None and result.ci_high is None


def test_a_fold_without_an_arm_falls_back_to_the_global_mean_not_zero() -> None:
    from benchmark.escalation.ope import _crossfit_qhat

    # Every reward identical, escalate confined to ONE session: a single constant-reward cluster.
    # Fitted per-fold with a 0.0 empty-arm fallback, the escalate rows in fold 0 got
    # qhat[True] = 0.0 (their complement holds none), so the DR residual `w * (r - 0)` spread the
    # terms apart and the estimator reported `identified` off rows that carry no variation. With
    # the global-mean fallback every term lands on the same value and the estimator must refuse.
    rows = [
        ExplorationLogRow(
            checkpoint_id=f"pkg::t{i}",
            escalated=(i < 10),
            propensity=0.7 if i < 10 else 0.3,
            reward=1.0,
            randomized=True,
            features={},
            session_id=f"s{i // 10}",
        )
        for i in range(50)
    ]
    fitted = _crossfit_qhat(rows)
    assert fitted[0][True] == pytest.approx(1.0)  # global mean of the escalate arm, not 0.0
    result = estimate_policy_value(rows, always_escalate)
    assert result.status == NOT_IDENTIFIED
    assert "degenerate interval" in result.reason
    assert result.dr_estimate is None


def test_qhat_is_cross_fitted_so_a_row_never_scores_against_its_own_mean() -> None:
    from benchmark.escalation.ope import _crossfit_qhat, _direct_method

    # One session carries the only successes. Fitted in-sample, its qhat[True] would be dragged
    # up by its own rewards; cross-fitted, its fold sees only the other sessions' zeros.
    rows = [
        ExplorationLogRow(f"c{i}", True, 0.7, float(i < 10), True, {}, f"s{i // 10}")
        for i in range(50)
    ]
    in_sample = _direct_method(rows)
    fitted = _crossfit_qhat(rows)
    assert in_sample[True] == pytest.approx(0.2)
    assert fitted[0][True] == pytest.approx(0.0)  # row 0's own session is held out


def test_the_contrast_answers_does_escalating_help_not_just_what_is_v() -> None:
    # V(always_escalate) alone is a level. The planted log pays 0.7 for escalating and 0.3 for
    # holding, so the contrast must recover ~+0.4 with an interval that excludes zero.
    result = estimate_policy_value(_randomized_log(n=3000), always_escalate)
    assert result.contrast_estimate is not None
    assert result.contrast_estimate == pytest.approx(0.4, abs=0.08)
    assert result.contrast_excludes_zero


def test_a_refused_estimate_reports_no_contrast_either() -> None:
    # The refusal must not leak a comparison the same logs cannot support.
    rows = [_row(escalated=True, propensity=1.0, reward=1.0, randomized=False) for _ in range(50)]
    result = estimate_policy_value(rows, always_escalate)
    assert result.status == NOT_IDENTIFIED
    assert result.contrast_estimate is None
    assert not result.contrast_excludes_zero


def test_rows_from_records_carries_the_session_id_through() -> None:
    rows = rows_from_records(
        [
            {
                "session_id": "sess-1",
                "checkpoint_id": "pkg::t",
                "action": "hold",
                "propensity": 0.3,
                "epsilon": 0.3,
                "outcome": "failure",
            }
        ]
    )
    assert rows[0].session_id == "sess-1"


def test_clipping_is_reported_when_it_actually_binds() -> None:
    rows = _randomized_log(epsilon=0.02, n=800, seed=3)
    result = estimate_policy_value(rows, lambda row: 0.0, weight_clip=5.0)
    assert result.max_weight == 5.0
    assert result.n_clipped > 0


# ── the instrument-validity gate (R0) — the module's VERDICTS, not its plumbing ──
#
# A refusal guard is only a good instrument if it can also ANSWER. This module exists to
# refuse NOT_IDENTIFIED on deterministic logs; those tests prove the refusal fires, but a
# module that only ever refused would still pass them. The two controls below prove the
# assembled estimator actually RECOVERS a planted policy value (positive control) and
# COLLAPSES to chance when the reward link is destroyed (shuffled-reward null), adjudicated
# by the shared gate — the same two-sided discipline `test_instrument_validity.py` applies
# to the policy sweep. Without these, any `identified` verdict the module ever emits would
# be the report of an instrument that has never been shown to detect anything.

_PLANTED_ESCALATE_VALUE = 0.7  # the ground-truth reward rate of the escalate arm, planted
_PLANTED_HOLD_VALUE = 0.3  # the ground-truth reward rate of the hold arm, planted
_PLANTED_EPSILON = 0.3


def _planted_log(
    n: int = 400, *, epsilon: float = _PLANTED_EPSILON, seed: int = 7
) -> list[ExplorationLogRow]:
    """A planted epsilon-greedy log with one DECISION per SESSION, per the module contract."""
    # The reward is a SESSION-level verified grade — production makes ONE escalation decision
    # per session — so each session's grade is drawn once from its arm's rate. That makes the
    # destroyed-signal null below faithful: grades are permuted across sessions, never within.
    rng = random.Random(seed)
    rows: list[ExplorationLogRow] = []
    for i in range(n):
        held = rng.random() < epsilon
        escalated = not held
        propensity = epsilon if held else 1.0 - epsilon
        reward = (
            1.0
            if rng.random() < (_PLANTED_ESCALATE_VALUE if escalated else _PLANTED_HOLD_VALUE)
            else 0.0
        )
        rows.append(
            ExplorationLogRow(
                checkpoint_id=f"c{i}",
                escalated=escalated,
                propensity=propensity,
                reward=reward,
                randomized=True,
                features={},
                session_id=f"s{i}",
            )
        )
    return rows


def _shuffled_rewards(rows: Sequence[ExplorationLogRow], seed: int = 0) -> list[ExplorationLogRow]:
    """The SAME rows with SESSION GRADES permuted ACROSS sessions — the signal destroyed."""
    # Grades move across sessions (not within), because a session's verified grade is one value
    # shared by its decisions — a within-session shuffle is a no-op on real data.
    rng = random.Random(seed)
    by_session: dict[str, list[ExplorationLogRow]] = {}
    for row in rows:
        by_session.setdefault(row.session_id, []).append(row)
    sessions = list(by_session.values())
    grades = [session[0].reward for session in sessions]
    rng.shuffle(grades)
    out: list[ExplorationLogRow] = []
    for session, grade in zip(sessions, grades, strict=True):
        for row in session:
            out.append(dataclasses.replace(row, reward=grade))
    return out


def test_the_estimator_recovers_a_planted_policy_value_within_its_interval() -> None:
    """Positive control: the DR estimate must CONTAIN the known ground-truth value.

    Recovery to a fixed tolerance is a weaker contract — the planted value must sit INSIDE
    the reported interval, the number the module itself certifies.
    """
    result = estimate_policy_value(_planted_log(), always_escalate)
    assert result.status == IDENTIFIED
    assert result.dr_estimate is not None
    assert result.ci_low is not None and result.ci_high is not None
    assert result.ci_low <= _PLANTED_ESCALATE_VALUE <= result.ci_high


def test_the_estimator_recovers_a_planted_contrast_with_an_interval_excluding_zero() -> None:
    """The contrast — the actual decision quantity — must recover the planted +0.4 with CI>0."""
    result = estimate_policy_value(_planted_log(), always_escalate)
    assert result.contrast_estimate is not None
    assert result.contrast_ci_low is not None and result.contrast_ci_high is not None
    planted_contrast = _PLANTED_ESCALATE_VALUE - _PLANTED_HOLD_VALUE
    # At n=400 sessions the point estimate carries ~0.09 of sampling noise, so the planted
    # value must sit INSIDE the reported interval (the load-bearing contract) rather than a
    # fixed 0.06 tolerance. The interval-excluding-zero clause is what proves the signal is
    # real and the estimator sees it.
    assert result.contrast_ci_low <= planted_contrast <= result.contrast_ci_high
    assert result.contrast_ci_low > 0.0  # the planted signal is real and the interval sees it


# The destroyed-signal null is estimated over this many independent reward shuffles; the
# median is the representative destroyed-signal score and the 97.5th percentile of |null|
# is the empirical chance band. A single draw's own CI must NOT serve as the band — that is
# self-referential (the null would pass by having a wide interval) and the band must come
# from the null DISTRIBUTION, not from the one draw being judged.
_NULL_DRAWS: Final[int] = 100


def _null_contrasts(rows: Sequence[ExplorationLogRow]) -> tuple[list[float], float]:
    """Destroyed-signal contrast across `_NULL_DRAWS` independent reward shuffles.

    Returns (all draws, median). A leakage-free instrument centres the median on chance.
    """
    draws = [
        estimate_policy_value(_shuffled_rewards(rows, seed=seed), always_escalate).contrast_estimate
        for seed in range(_NULL_DRAWS)
    ]
    filtered = [d for d in draws if d is not None]
    if not filtered:
        raise AssertionError(
            f"all {_NULL_DRAWS} destroyed-signal shuffles returned NOT_IDENTIFIED — the null "
            "has no score to judge, which is itself a gate failure, not a pass"
        )
    return filtered, statistics.median(filtered)


def test_the_instrument_clears_the_shared_admissibility_gate() -> None:
    """R0 two-sided control through the SHARED gate, not a local re-derivation."""
    # The band is the null's spread AROUND ITS OWN CENTRE, never around the chance level: a band
    # measured around chance absorbs a uniformly-shifted null (an instrument manufacturing a
    # constant +0.4 sits "at chance" against a band grown from that same shift), while a band
    # measured around the null's median judges the null's CENTRE against the true chance level.
    base = _planted_log()
    positive = estimate_policy_value(base, always_escalate)
    null_draws, null_median = _null_contrasts(base)
    band = sorted(abs(d - null_median) for d in null_draws)[int(0.975 * (len(null_draws) - 1))]
    verdict = admissibility_verdict(
        positive.contrast_estimate,
        null_median,
        chance_level=0.0,
        chance_band=band,
    )
    assert verdict.positive_passed
    assert verdict.null_at_chance
    assert verdict.admissible


def test_the_instrument_refuses_where_it_cannot_answer() -> None:
    """Null: a deterministic log (no randomization) must still refuse with NOT_IDENTIFIED."""
    rows = [_row(escalated=True, propensity=1.0, reward=1.0, randomized=False) for _ in range(500)]
    result = estimate_policy_value(rows, always_escalate)
    assert result.status == NOT_IDENTIFIED
    assert result.dr_estimate is None
    assert result.contrast_estimate is None
    assert not result.contrast_excludes_zero
