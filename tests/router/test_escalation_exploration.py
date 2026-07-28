"""R5 — epsilon-greedy escalation randomization with logged propensities (pure logic)."""

from __future__ import annotations

import pytest

from shunt.router.escalation import (
    EscalationAction,
    EscalationConfig,
    EscalationContext,
    EscalationRunner,
    ExplorationStream,
    FailureEvent,
    decide_escalation,
)


def _fail(idx: int, key: str = "test::foo") -> FailureEvent:
    return FailureEvent(
        decision_index=idx, dedup_key=key, exit_code=1, success=False, confirmed=True, blocking=True
    )


def _ctx(*, effort: int = 0, max_effort: int = 2) -> EscalationContext:
    return EscalationContext(
        current_rank_index=0,
        max_rank_index=3,
        current_effort_index=effort,
        max_effort_index=max_effort,
    )


def _flagged(index: int = 0) -> list[FailureEvent]:
    """Two same-key confirmed failures — the checkpoint the detector flags, in-window at *index*."""
    return [_fail(index), _fail(index)]


def _run(epsilon: float, seed: int, n: int = 200) -> list[EscalationAction]:
    config = EscalationConfig(enabled=True, exploration_epsilon=epsilon, exploration_seed=seed)
    stream = ExplorationStream.from_seed(seed)
    return [
        decide_escalation(_flagged(i), i, _ctx(), config, stream=stream).action for i in range(n)
    ]


def test_seeded_randomization_reproduces_exactly() -> None:
    assert _run(0.3, 20260728) == _run(0.3, 20260728)


def test_distinct_seeds_diverge() -> None:
    # Guards against a "reproducible" stream that is reproducible because it never draws.
    assert _run(0.3, 1) != _run(0.3, 2)


def test_epsilon_zero_is_bit_for_bit_the_deterministic_policy() -> None:
    config = EscalationConfig(enabled=True)
    randomized = EscalationConfig(enabled=True, exploration_epsilon=0.0, exploration_seed=7)
    for events, index in ((_flagged(), 5), ([_fail(0)], 5), ([], 5)):
        base = decide_escalation(events, index, _ctx(), config)
        under_knob = decide_escalation(events, index, _ctx(), randomized)
        assert base.action == under_knob.action
        assert base.reason == under_knob.reason
        assert base.new_label_window == under_knob.new_label_window


def test_default_config_does_not_randomize() -> None:
    assert EscalationConfig().exploration_epsilon == 0.0
    assert EscalationConfig().exploration_seed is None


def test_epsilon_above_zero_without_a_stream_is_a_loud_error() -> None:
    # Silently falling back to deterministic would produce logs that LOOK randomized.
    config = EscalationConfig(enabled=True, exploration_epsilon=0.2)
    with pytest.raises(ValueError, match="ExplorationStream"):
        decide_escalation(_flagged(), 5, _ctx(), config)


def test_every_flagged_decision_records_a_propensity_strictly_inside_the_unit_interval() -> None:
    config = EscalationConfig(enabled=True, exploration_epsilon=0.25, exploration_seed=3)
    stream = ExplorationStream.from_seed(3)
    for i in range(100):
        record = decide_escalation(_flagged(i), i, _ctx(), config, stream=stream).exploration
        assert record is not None
        assert record.randomized is True
        assert 0.0 < record.propensity < 1.0
        assert record.action in (EscalationAction.HOLD, record.policy_action)
        assert record.epsilon == 0.25
        assert record.seed == 3
        assert record.checkpoint_id == "test::foo"


def test_propensity_matches_the_arm_actually_taken() -> None:
    config = EscalationConfig(enabled=True, exploration_epsilon=0.4, exploration_seed=11)
    stream = ExplorationStream.from_seed(11)
    seen = set()
    for i in range(200):
        record = decide_escalation(_flagged(i), i, _ctx(), config, stream=stream).exploration
        assert record is not None
        expected = 0.4 if record.action is EscalationAction.HOLD else 0.6
        assert record.propensity == pytest.approx(expected)
        seen.add(record.action)
    assert len(seen) == 2, "both arms must be realized — one-armed logs are not identified"


def test_unflagged_steps_log_no_propensity() -> None:
    config = EscalationConfig(enabled=True, exploration_epsilon=0.5, exploration_seed=1)
    stream = ExplorationStream.from_seed(1)
    for events in ([], [_fail(0)], [_fail(0, "a"), _fail(1, "b")]):
        directive = decide_escalation(events, 5, _ctx(), config, stream=stream)
        assert directive.action is EscalationAction.HOLD
        assert directive.exploration is None


def test_disabled_and_collapse_suppressed_paths_log_no_propensity() -> None:
    stream = ExplorationStream.from_seed(1)
    disabled = EscalationConfig(exploration_epsilon=0.5)
    assert decide_escalation(_flagged(), 5, _ctx(), disabled, stream=stream).exploration is None
    alarmed = EscalationContext(
        current_rank_index=0,
        max_rank_index=3,
        current_effort_index=0,
        max_effort_index=2,
        loop_health_alarm=True,
    )
    config = EscalationConfig(enabled=True, exploration_epsilon=0.5, exploration_seed=1)
    assert decide_escalation(_flagged(), 5, alarmed, config, stream=stream).exploration is None


def test_exhausted_ladder_logs_hold_at_propensity_one_and_is_excluded() -> None:
    # No viable rung -> record the hold honestly rather than fabricate an arm.
    ceiling = EscalationContext(
        current_rank_index=3, max_rank_index=3, current_effort_index=2, max_effort_index=2
    )
    config = EscalationConfig(enabled=True, exploration_epsilon=0.5, exploration_seed=1)
    stream = ExplorationStream.from_seed(1)
    directive = decide_escalation(_flagged(), 5, ceiling, config, stream=stream)
    assert directive.action is EscalationAction.HOLD
    record = directive.exploration
    assert record is not None
    assert record.propensity == 1.0
    assert record.randomized is False


def test_deterministic_flagged_decisions_record_propensity_one() -> None:
    config = EscalationConfig(enabled=True)
    record = decide_escalation(_flagged(), 5, _ctx(), config).exploration
    assert record is not None
    assert record.propensity == 1.0
    assert record.randomized is False
    assert record.action == record.policy_action


def test_decision_time_features_are_recorded() -> None:
    config = EscalationConfig(enabled=True)
    record = decide_escalation(_flagged(), 7, _ctx(effort=1), config).exploration
    assert record is not None
    assert record.features["decision_index"] == 7.0
    assert record.features["countable_failures"] == 2.0
    assert record.features["distinct_keys"] == 1.0
    assert record.features["effort_headroom"] == 1.0
    assert record.features["rank_headroom"] == 3.0


def test_randomization_never_introduces_a_switch_the_policy_would_not_make() -> None:
    # Cache-safety: the exploration arm is HOLD. Randomizing can only WITHHOLD an escalation,
    # never invent one, so no mid-cached-turn model switch can be created by this knob.
    deterministic = decide_escalation(_flagged(), 5, _ctx(), EscalationConfig(enabled=True))
    config = EscalationConfig(enabled=True, exploration_epsilon=0.5, exploration_seed=99)
    stream = ExplorationStream.from_seed(99)
    for i in range(200):
        directive = decide_escalation(_flagged(i), i, _ctx(), config, stream=stream)
        assert directive.action in (deterministic.action, EscalationAction.HOLD)
        if directive.action is EscalationAction.HOLD:
            assert directive.new_label_window is False


def test_exploration_hold_never_escalates_an_unflagged_boundary() -> None:
    # The alternative arm is only ever offered where the deterministic policy would fire.
    config = EscalationConfig(enabled=True, exploration_epsilon=0.9, exploration_seed=5)
    stream = ExplorationStream.from_seed(5)
    for i in range(100):
        directive = decide_escalation([_fail(i)], i, _ctx(), config, stream=stream)
        assert directive.action is EscalationAction.HOLD
        assert directive.reason == "no_recurring_failure"


def test_runner_threads_the_stream_and_holds_without_advancing_the_ladder() -> None:
    config = EscalationConfig(enabled=True, exploration_epsilon=1.0 - 1e-9, exploration_seed=2)
    stream = ExplorationStream.from_seed(2)
    runner = EscalationRunner(max_effort_index=2, max_rank_index=3)
    for i in range(4):
        directive = runner.step(
            success=False, event=_fail(i), current_index=i, config=config, stream=stream
        )
    assert directive.action is EscalationAction.HOLD
    assert directive.exploration is not None
    assert directive.exploration.randomized is True
    # An explored HOLD must not retire the window — nothing was acted on.
    assert len(runner.log) == 4


def test_stream_rejects_a_reused_generator_identity() -> None:
    first = ExplorationStream.from_seed(4)
    second = ExplorationStream.from_seed(4)
    assert [first.rng.random() for _ in range(5)] == [second.rng.random() for _ in range(5)]
    assert first.seed == 4
