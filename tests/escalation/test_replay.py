"""Replay: directive stream reproduces a direct decide_escalation call (Direct-Method-exact);
sweep emits exactly one point per GridPoint.
"""

from __future__ import annotations

from benchmark.escalation import replay
from benchmark.escalation.replay import GridPoint
from benchmark.escalation.schema import Trajectory
from shunt.router.escalation import (
    EscalationAction,
    EscalationContext,
    FailureEvent,
    decide_escalation,
    failure_event_from_outcome,
)
from tests.escalation.factories import make_step, make_trajectory


def _events(traj: Trajectory) -> list[FailureEvent]:
    """The ordered FailureEvents a trajectory yields — rebuilt via the SHARED constructor, so the
    replay's directive stream can be checked against a direct decide_escalation call.
    """
    return [
        failure_event_from_outcome(
            decision_index=s.decision_index,
            failing_check_id=s.failing_check_id,
            exit_code=s.exit_code,
            success=s.success,
            is_infra_failure=s.is_infra_failure,
            confirmed=s.confirmed,
        )
        for s in traj.steps
        if not s.success and s.failing_check_id is not None
    ]


_CTX = EscalationContext(
    current_tier_index=0,
    max_tier_index=3,
    current_effort_index=0,
    max_effort_index=3,
)


def _two_same_key_failures():
    return make_trajectory(
        [
            make_step(step_index=0, decision_index=0, success=False, failing_check_id="t::a"),
            make_step(step_index=1, decision_index=1, success=False, failing_check_id="t::a"),
        ]
    )


def test_replay_directive_matches_direct_decide_escalation() -> None:
    traj = _two_same_key_failures()
    cfg = GridPoint(escalate_after_n=2, stale_window=10).to_config()
    decision = replay.replay_config(traj, cfg, context=_CTX)
    stream = _events(traj)
    # Boundary 0: only one failure in the window → HOLD (direct call agrees).
    assert decision.directives[0] == decide_escalation(stream[:1], 0, _CTX, cfg).action
    # Boundary 1: two same-key failures → the ladder fires (direct call agrees).
    assert decision.directives[1] == decide_escalation(stream[:2], 1, _CTX, cfg).action
    assert decision.directives[1] == EscalationAction.RAISE_EFFORT
    assert decision.first_escalation_index == 1
    assert decision.escalated


def test_first_failure_alone_holds() -> None:
    traj = make_trajectory(
        [make_step(step_index=0, decision_index=0, success=False, failing_check_id="t::a")]
    )
    cfg = GridPoint(2, 10).to_config()
    decision = replay.replay_config(traj, cfg, context=_CTX)
    assert decision.directives == [EscalationAction.HOLD]
    assert not decision.escalated


def test_success_retires_the_window() -> None:
    # fail, fail, success, fail → the success clears the window so the trailing fail holds.
    traj = make_trajectory(
        [
            make_step(step_index=0, decision_index=0, success=False, failing_check_id="t::a"),
            make_step(step_index=1, decision_index=1, success=False, failing_check_id="t::a"),
            make_step(step_index=2, decision_index=2, success=True),
            make_step(step_index=3, decision_index=3, success=False, failing_check_id="t::a"),
        ]
    )
    cfg = GridPoint(2, 10).to_config()
    decision = replay.replay_config(traj, cfg, context=_CTX)
    assert decision.directives[1] == EscalationAction.RAISE_EFFORT
    assert decision.directives[3] == EscalationAction.HOLD  # window was retired by the success


def test_infra_reds_never_escalate() -> None:
    traj = make_trajectory(
        [
            make_step(
                step_index=0,
                decision_index=0,
                success=False,
                is_infra_failure=True,
                failing_check_id="t::a",
            ),
            make_step(
                step_index=1,
                decision_index=1,
                success=False,
                is_infra_failure=True,
                failing_check_id="t::a",
            ),
        ]
    )
    cfg = GridPoint(2, 10).to_config()
    decision = replay.replay_config(traj, cfg, context=_CTX)
    assert not decision.escalated


def test_sweep_emits_one_point_per_grid_point() -> None:
    trajs = [_two_same_key_failures(), _two_same_key_failures()]
    grid = [GridPoint(2, 10), GridPoint(3, 5), GridPoint(2, 20, ladder="tier_only")]
    result = replay.sweep(trajs, grid)
    assert len(result.points) == len(grid)
    for point in result.points:
        assert len(point.decisions) == len(trajs)  # every trajectory replayed, no drops
    assert [p.grid_point for p in result.points] == grid  # no dups, order preserved
