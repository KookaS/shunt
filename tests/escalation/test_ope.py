"""The identification guard: an OPE estimator that refuses logs without randomization."""

from __future__ import annotations

import itertools
import random

import pytest

from benchmark.escalation.ope import (
    IDENTIFIED,
    NOT_IDENTIFIED,
    ExplorationLogRow,
    always_escalate,
    estimate_policy_value,
    rows_from_records,
)

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


def test_clipping_is_reported_when_it_actually_binds() -> None:
    rows = _randomized_log(epsilon=0.02, n=800, seed=3)
    result = estimate_policy_value(rows, lambda row: 0.0, weight_clip=5.0)
    assert result.max_weight == 5.0
    assert result.n_clipped > 0
