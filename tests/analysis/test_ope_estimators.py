"""IPS / SNIPS / ESS against hand-computed values, plus the two shapes rows_from_records maps.

Every expected number below is derived in the test from the log's own propensities, never
snapshotted from the implementation — a snapshot would ratify whatever the estimator does.
"""

from __future__ import annotations

from typing import Final

import pytest

from shunt.analysis.ope import (
    IDENTIFIED,
    ExplorationLogRow,
    PolicyValueEstimate,
    always_escalate,
    effective_sample_size,
    ess_fraction,
    estimate_policy_value,
    ips_estimate,
    rows_from_records,
    snips_estimate,
)


def _row(
    *,
    escalated: bool,
    propensity: float,
    reward: float | None,
    randomized: bool = True,
    session: str = "s",
) -> ExplorationLogRow:
    return ExplorationLogRow(
        checkpoint_id=f"chk-{session}-{propensity}-{escalated}",
        escalated=escalated,
        propensity=propensity,
        reward=reward,
        randomized=randomized,
        session_id=session,
    )


def _half(row: ExplorationLogRow) -> float:
    """A target that escalates with probability 0.5, so BOTH arms carry weight."""
    del row
    return 0.5


# The worked log. Under the 0.5 target every row's target probability is 0.5, so
#   w_i = 0.5 / p_i  ->  [2.5, 0.625, 2.5, 0.625]   (no clipping at the default 20.0)
# and the rewards are  [1.0, 0.0, 1.0, 1.0].
_LOG: Final[list[ExplorationLogRow]] = [
    _row(escalated=True, propensity=0.2, reward=1.0, session="a"),
    _row(escalated=True, propensity=0.8, reward=0.0, session="b"),
    _row(escalated=False, propensity=0.2, reward=1.0, session="c"),
    _row(escalated=False, propensity=0.8, reward=1.0, session="d"),
]
_W: Final[list[float]] = [2.5, 0.625, 2.5, 0.625]


def test_ips_is_the_mean_of_weighted_rewards() -> None:
    # (2.5*1 + 0.625*0 + 2.5*1 + 0.625*1) / 4 = 5.625 / 4
    assert ips_estimate(_LOG, _half) == pytest.approx(5.625 / 4.0)
    assert ips_estimate(_LOG, _half) == pytest.approx(1.40625)


def test_snips_is_self_normalised_by_the_same_weights() -> None:
    # sum(w r) / sum(w) = 5.625 / 6.25
    assert snips_estimate(_LOG, _half) == pytest.approx(0.9)
    # Self-normalisation is what bounds SNIPS by the reward range; IPS is not so bounded, and on
    # this log it exceeds 1.0 while every reward is in [0, 1].
    assert ips_estimate(_LOG, _half) > 1.0
    assert 0.0 <= snips_estimate(_LOG, _half) <= 1.0  # type: ignore[operator]


def test_ips_and_snips_share_the_clip() -> None:
    # weight_clip=1.0 binds the two 2.5 weights: w -> [1.0, 0.625, 1.0, 0.625]
    assert ips_estimate(_LOG, _half, weight_clip=1.0) == pytest.approx(2.625 / 4.0)
    assert snips_estimate(_LOG, _half, weight_clip=1.0) == pytest.approx(2.625 / 3.25)


def test_always_escalate_zeroes_the_hold_arm_weights() -> None:
    # target_prob = 1 on escalate rows, 0 on hold rows: w = [5.0, 1.25, 0.0, 0.0]
    assert ips_estimate(_LOG, always_escalate) == pytest.approx(5.0 / 4.0)
    assert snips_estimate(_LOG, always_escalate) == pytest.approx(5.0 / 6.25)


def test_ess_is_the_kish_formula() -> None:
    # (sum w)^2 / sum w^2 = 6.25^2 / 13.28125 = 50/17
    assert effective_sample_size(_W) == pytest.approx(50.0 / 17.0)
    assert ess_fraction(_W) == pytest.approx(50.0 / 68.0)
    assert effective_sample_size([1.0, 1.0, 1.0, 1.0]) == pytest.approx(4.0)
    assert ess_fraction([1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)
    # All the mass on one row: the log is worth ONE observation however long it is.
    assert effective_sample_size([4.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_ess_is_zero_on_an_empty_or_massless_log() -> None:
    assert effective_sample_size([]) == 0.0
    assert ess_fraction([]) == 0.0
    assert effective_sample_size([0.0, 0.0]) == 0.0


def test_ess_fraction_collapses_as_propensity_variance_rises() -> None:
    # Under a constant target the weights are 0.5/p, so propensity spread IS weight spread.
    fractions = [
        ess_fraction([0.5 / p for p in (0.5, 0.5, 0.5, 0.5)]),
        ess_fraction([0.5 / p for p in (0.4, 0.4, 0.6, 0.6)]),
        ess_fraction([0.5 / p for p in (0.2, 0.2, 0.8, 0.8)]),
        ess_fraction([0.5 / p for p in (0.02, 0.02, 0.9, 0.9)]),
    ]
    assert fractions[0] == pytest.approx(1.0)
    assert fractions == sorted(fractions, reverse=True)
    assert fractions[0] > fractions[1] > fractions[2] > fractions[3]
    # Hand-derived closed form for a two-valued weight vector [w1, w1, w2, w2]:
    #   ESS = (2w1 + 2w2)^2 / (2w1^2 + 2w2^2) = 2(w1 + w2)^2 / (w1^2 + w2^2)
    # At p = 0.02 / 0.9 that is w1 = 25, w2 = 5/9 -> ESS = 2(25 + 5/9)^2 / (625 + 25/81),
    # i.e. four logged rows worth barely two independent ones.
    w1, w2 = 0.5 / 0.02, 0.5 / 0.9
    expected_ess = 2.0 * (w1 + w2) ** 2 / (w1**2 + w2**2)
    assert fractions[3] == pytest.approx(expected_ess / 4.0)
    assert expected_ess < 2.1


def test_unusable_rows_are_excluded_not_scored() -> None:
    padded = [
        *_LOG,
        _row(escalated=True, propensity=0.3, reward=None, session="e"),  # no verified outcome
        _row(escalated=True, propensity=0.3, reward=1.0, randomized=False, session="f"),
        _row(escalated=True, propensity=1.0, reward=1.0, session="g"),  # no counterfactual
    ]
    assert ips_estimate(padded, _half) == pytest.approx(ips_estimate(_LOG, _half))
    assert snips_estimate(padded, _half) == pytest.approx(snips_estimate(_LOG, _half))


def test_no_usable_row_returns_none_rather_than_zero() -> None:
    assert ips_estimate([], _half) is None
    assert snips_estimate([], _half) is None
    dead = [_row(escalated=True, propensity=0.5, reward=1.0, randomized=False, session="a")]
    assert ips_estimate(dead, _half) is None
    assert snips_estimate(dead, _half) is None


def test_policy_value_estimate_ess_fields_default_so_old_call_sites_compile() -> None:
    estimate = PolicyValueEstimate(
        status=IDENTIFIED, reason="", n_decisions=0, n_escalated=0, n_held=0, n_excluded=0
    )
    assert estimate.ess == 0.0
    assert estimate.ess_fraction == 0.0


def test_estimate_policy_value_reports_the_ess_of_its_own_weights() -> None:
    rows = [
        _row(
            escalated=i % 2 == 0,
            propensity=0.2 if i % 2 == 0 else 0.8,
            reward=float(i % 3 == 0),
            session=f"s{i}",
        )
        for i in range(20)
    ]
    estimate = estimate_policy_value(rows, _half, bootstrap_draws=200)
    assert estimate.identified, estimate.reason
    # Ten rows at p=0.2 and ten at p=0.8, so under the 0.5 target the weights are ten 2.5s and
    # ten 0.625s:  ESS = (10*2.5 + 10*0.625)^2 / (10*2.5^2 + 10*0.625^2) = 31.25^2 / 66.40625.
    assert estimate.n_decisions == 20
    assert estimate.ess == pytest.approx(31.25**2 / 66.40625)
    assert estimate.ess_fraction == pytest.approx(31.25**2 / 66.40625 / 20.0)


# --- the two logged shapes -------------------------------------------------------------------


def test_escalation_shape_is_unchanged() -> None:
    (row,) = rows_from_records(
        [
            {
                "checkpoint_id": "test_a",
                "action": "raise_effort",
                "propensity": 0.2,
                "epsilon": 0.2,
                "outcome": "success",
                "session_id": "sess-1",
                "features": {"depth": 3.0},
            }
        ]
    )
    assert row.escalated is True
    assert row.action == "raise_effort"
    assert row.randomized is True
    assert row.propensity == pytest.approx(0.2)
    assert row.reward == pytest.approx(1.0)
    assert row.features == {"depth": 3.0}
    assert row.session_id == "sess-1"


def _routing_record(propensity: float) -> dict[str, object]:
    return {
        "session_id": "sess-r",
        "action": "model-c",
        "selection_propensity": propensity,
        "epsilon": 0.3,
        "candidate_model_scores": {"model-a": 0.9, "model-b": 0.4, "model-c": 0.1},
        "outcome": "failure",
    }


def test_routing_shape_maps_selection_propensity_and_candidate_scores() -> None:
    # eps-greedy over k=3 arms logs eps/k = 0.1 on an explore turn.
    (row,) = rows_from_records([_routing_record(0.1)])
    assert row.action == "model-c"
    assert row.propensity == pytest.approx(0.1)
    assert row.randomized is True
    assert row.reward == pytest.approx(0.0)
    assert row.features == {"model-a": 0.9, "model-b": 0.4, "model-c": 0.1}
    assert row.session_id == "sess-r"
    # The binary estimator has no multi-arm value to report, so every routing row reads as the
    # hold arm and the per-arm floor refuses the log rather than returning a one-armed number.
    assert row.escalated is False


def test_routing_greedy_propensity_is_also_recognised_as_randomized() -> None:
    # 1 - eps + eps/k = 1 - 0.3 + 0.1 = 0.8
    (row,) = rows_from_records([_routing_record(0.8)])
    assert row.randomized is True


def test_routing_propensity_off_the_epsilon_grid_is_not_randomized() -> None:
    (row,) = rows_from_records([_routing_record(0.5)])
    assert row.randomized is False


def test_routing_rows_are_refused_by_the_binary_estimator() -> None:
    rows = rows_from_records([{**_routing_record(0.1), "session_id": f"s{i}"} for i in range(20)])
    estimate = estimate_policy_value(rows)
    assert not estimate.identified
    assert "per-arm floor" in estimate.reason


def test_the_live_escalation_record_carries_no_routing_discriminator() -> None:
    """`selection_propensity` is what routes a record to the multi-arm branch — pin its absence."""
    # The discriminator is a naming convention, not an enforced schema. If a future
    # `ExplorationRecord` ever grew this key, every escalation row would silently reroute to
    # `_routing_row` and come back with `escalated=False` — a refusal dressed as a hold arm.
    from shunt.router.escalation import ExplorationRecord

    record = ExplorationRecord(
        checkpoint_id="test_a",
        decision_index=0,
        policy_action="raise_effort",
        action="raise_effort",
        propensity=0.2,
        epsilon=0.2,
        seed=1,
        randomized=True,
    )
    assert "selection_propensity" not in record.persistable()


def test_routing_shape_accepts_the_provenance_key_the_router_actually_writes() -> None:
    """The router's routing provenance names the arm `model_chosen`, not `action`."""
    record = {k: v for k, v in _routing_record(0.1).items() if k != "action"}
    (row,) = rows_from_records({**record, "model_chosen": "model-c"} for _ in range(1))
    assert row.action == "model-c"


def test_a_json_round_tripped_numeric_string_is_parsed_not_dropped() -> None:
    (row,) = rows_from_records(
        [{**_routing_record(0.1), "candidate_model_scores": {"model-a": "0.9", "model-b": 0.4}}]
    )
    assert row.features == {"model-a": 0.9, "model-b": 0.4}
    (escalation,) = rows_from_records(
        [{"action": "raise_effort", "propensity": 0.2, "epsilon": 0.2, "features": {"d": "3"}}]
    )
    assert escalation.features == {"d": 3.0}
