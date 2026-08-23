"""The coercion helpers reject non-finite values, so no `nan` can traverse IPS, SNIPS or DR."""

from __future__ import annotations

import math
from typing import Final

import pytest

from shunt.analysis.ope import (
    _coerce_float,
    _numeric_map,
    _usable,
    estimate_policy_value,
    rows_from_records,
)


@pytest.mark.parametrize(
    "value", ["nan", "NaN", "inf", "-inf", "Infinity", "-Infinity", float("nan"), float("inf")]
)
def test_a_non_finite_value_is_missing_exactly_like_any_other_non_number(value: object) -> None:
    assert _coerce_float(value) is None


@pytest.mark.parametrize("value", ["not-a-number", None, {}, [], object()])
def test_a_genuine_non_number_is_still_missing(value: object) -> None:
    assert _coerce_float(value) is None


@pytest.mark.parametrize(
    ("value", "expected"), [("1.5", 1.5), (2, 2.0), (-0.25, -0.25), ("0", 0.0)]
)
def test_a_finite_number_still_coerces(value: object, expected: float) -> None:
    assert _coerce_float(value) == expected


def test_a_score_map_drops_the_non_finite_arm_and_keeps_the_rest() -> None:
    coerced = _numeric_map({"cheap": 0.4, "frontier": "nan", "mid": "0.6"})
    assert coerced == {"cheap": 0.4, "mid": 0.6}
    assert all(math.isfinite(v) for v in coerced.values())


NON_FINITE: Final = (float("nan"), float("inf"), float("-inf"))


def _log(n_sessions: int = 6) -> list[dict[str, object]]:
    """A minimal randomized escalation log the estimator accepts: both arms, enough clusters."""
    records: list[dict[str, object]] = []
    for i in range(n_sessions):
        for action, propensity in (("raise_effort", 0.1), ("hold", 0.9)):
            records.append(
                {
                    "checkpoint_id": f"c{i}",
                    "action": action,
                    "propensity": propensity,
                    "epsilon": 0.1,
                    "outcome": "success" if (i + (action == "hold")) % 2 else "failure",
                    "session_id": f"s{i}",
                    "features": {"f": float(i)},
                }
            )
    return records


@pytest.mark.parametrize("value", NON_FINITE)
def test_a_non_finite_propensity_reaches_the_row_as_missing(value: float) -> None:
    (row,) = rows_from_records([{**_log(1)[0], "propensity": value}])
    # Missing propensity defaults to 1.0 — the "no alternative rung" marker `_usable` excludes.
    assert row.propensity == 1.0
    assert math.isfinite(row.propensity)
    assert not row.randomized


@pytest.mark.parametrize("value", NON_FINITE)
def test_a_non_finite_selection_propensity_reaches_the_routing_row_as_missing(
    value: float,
) -> None:
    (row,) = rows_from_records(
        [
            {
                "checkpoint_id": "c0",
                "selection_propensity": value,
                "epsilon": 0.1,
                "model_chosen": "cheap",
                "outcome": "success",
                "session_id": "s0",
                "candidate_model_scores": {"cheap": 0.4, "frontier": 0.6},
            }
        ]
    )
    assert row.propensity == 1.0
    assert not row.randomized


@pytest.mark.parametrize("value", NON_FINITE)
def test_a_non_finite_epsilon_cannot_declare_a_row_randomized(value: float) -> None:
    (row,) = rows_from_records([{**_log(1)[0], "epsilon": value}])
    assert not row.randomized


@pytest.mark.parametrize("value", NON_FINITE)
def test_a_non_finite_propensity_never_yields_a_non_finite_estimate(value: float) -> None:
    records = _log()
    records[0] = {**records[0], "propensity": value}
    rows = rows_from_records(records)
    assert all(math.isfinite(row.propensity) for row in rows)
    assert all(math.isfinite(row.propensity) for row in _usable(rows))
    estimate = estimate_policy_value(rows)
    for number in (
        estimate.dr_estimate,
        estimate.ci_low,
        estimate.ci_high,
        estimate.contrast_estimate,
        estimate.max_weight,
        estimate.ess,
    ):
        assert number is None or math.isfinite(number)
