"""F5: the bare `EscalationConfig` validates its own knobs, not just the router.yaml schema."""

# `EscalationPolicy` (pydantic) is only on the router.yaml path; a benchmark grid constructs the
# dataclass DIRECTLY. Unvalidated, an unknown ladder silently degraded to rank_only, an epsilon
# above 1.0 wrote an impossible probability into the OPE log, and n=0 escalated off zero
# countable failures. Both entry points now share `ESCALATION_LADDERS` and the same bounds.

from __future__ import annotations

import pytest

from shunt.router.escalation import (
    ESCALATION_LADDERS,
    EscalationAction,
    EscalationConfig,
    EscalationContext,
    FailureEvent,
    decide_escalation,
)
from shunt.router.policy import EscalationPolicy


def test_defaults_construct() -> None:
    assert EscalationConfig().ladder in ESCALATION_LADDERS


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"ladder": "bogus"}, "unknown ladder"),
        ({"escalate_after_n": 0}, "escalate_after_n"),
        ({"escalate_after_n": -1}, "escalate_after_n"),
        ({"stale_window": 0}, "stale_window"),
        ({"exploration_epsilon": 1.5}, "exploration_epsilon"),
        ({"exploration_epsilon": 1.0}, "exploration_epsilon"),
        ({"exploration_epsilon": -0.1}, "exploration_epsilon"),
    ],
)
def test_invalid_knobs_raise(kwargs: dict[str, object], expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        EscalationConfig(**kwargs)  # type: ignore[arg-type]


def test_valid_edges_are_accepted() -> None:
    assert EscalationConfig(escalate_after_n=1).escalate_after_n == 1
    assert EscalationConfig(stale_window=1).stale_window == 1
    assert EscalationConfig(exploration_epsilon=0.0).exploration_epsilon == 0.0
    assert EscalationConfig(ladder="rank_only").ladder == "rank_only"


def test_ladder_vocabulary_is_shared_with_the_yaml_schema() -> None:
    # One vocabulary, two entry points — the pydantic validator reads the same tuple.
    with pytest.raises(ValueError, match="unknown escalation.ladder"):
        EscalationPolicy(ladder="bogus")
    for ladder in ESCALATION_LADDERS:
        assert EscalationPolicy(ladder=ladder).to_config().ladder == ladder


def test_n_zero_can_no_longer_reach_the_decision_path() -> None:
    # The probed symptom: with n=0, `_recurring_key` fired on a key with ZERO countable failures
    # (a single success), so escalation triggered off nothing. That config is now unbuildable —
    # and the smallest legal n still holds on a success-only window.
    with pytest.raises(ValueError, match="escalate_after_n"):
        EscalationConfig(enabled=True, escalate_after_n=0)
    events = [
        FailureEvent(decision_index=0, dedup_key="t::a", exit_code=0, success=True, confirmed=True)
    ]
    ctx = EscalationContext(
        current_rank_index=0, max_rank_index=1, current_effort_index=0, max_effort_index=1
    )
    config = EscalationConfig(enabled=True, escalate_after_n=1)
    assert decide_escalation(events, 1, ctx, config).action is EscalationAction.HOLD
