"""Replay: directive stream reproduces a direct decide_escalation call (Direct-Method-exact);
sweep emits exactly one point per GridPoint.
"""

from __future__ import annotations

from benchmark.escalation import replay
from benchmark.escalation.normalize.mini_swe_agent import stamp_step
from benchmark.escalation.replay import GridPoint
from benchmark.escalation.schema import Trajectory
from shunt.router.escalation import (
    EscalationAction,
    EscalationContext,
    FailureEvent,
    decide_escalation,
    failure_event_from_outcome,
)
from shunt.verifiers.parse import parse_test_outcome
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
    current_rank_index=0,
    max_rank_index=3,
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


def test_infra_reds_do_not_clear_the_window() -> None:
    # The divergence `test_infra_reds_never_escalate` does NOT cover: an `unknown` + infra step is
    # stamped `success=True` (`stamp_step` keys on `outcome == "failure"`), so the pre-fix replay
    # read it as a verified pass and wiped the window between two real same-key failures. Live it
    # is not labellable and never reaches `record_outcome`, so the window must survive it.
    traj = make_trajectory(
        [
            make_step(step_index=0, decision_index=0, success=False, failing_check_id="t::a"),
            make_step(step_index=1, decision_index=1, success=True, is_infra_failure=True),
            make_step(step_index=2, decision_index=2, success=False, failing_check_id="t::a"),
        ]
    )
    decision = replay.replay_config(traj, GridPoint(2, 10).to_config(), context=_CTX)
    assert decision.directives[2] == EscalationAction.RAISE_EFFORT
    assert decision.first_escalation_index == 2


def test_unstamped_steps_do_not_clear_the_window() -> None:
    # The parser default is `success=True, confirmed=False` — "nothing was observed", not "the
    # suite passed". It must be a no-op on the failure log, exactly as the live plane is.
    traj = make_trajectory(
        [
            make_step(step_index=0, decision_index=0, success=False, failing_check_id="t::a"),
            make_step(step_index=1, decision_index=1, success=True, confirmed=False),
            make_step(step_index=2, decision_index=2, success=False, failing_check_id="t::a"),
        ]
    )
    decision = replay.replay_config(traj, GridPoint(2, 10).to_config(), context=_CTX)
    assert decision.directives[2] == EscalationAction.RAISE_EFFORT


def test_verified_outcome_tri_state() -> None:
    # The one predicate the fix rests on, pinned per shape.
    stamped_pass = make_step(step_index=0, success=True)
    assert replay.verified_outcome(stamped_pass) is replay.VerifiedOutcome.PASS
    fail = make_step(step_index=0, success=False, failing_check_id="t::a")
    assert replay.verified_outcome(fail) is replay.VerifiedOutcome.FAIL
    unstamped = make_step(step_index=0, success=True, confirmed=False)
    assert replay.verified_outcome(unstamped) is replay.VerifiedOutcome.NONE
    infra = make_step(step_index=0, success=True, is_infra_failure=True)
    assert replay.verified_outcome(infra) is replay.VerifiedOutcome.NONE


def test_usage_error_step_is_not_a_verified_outcome() -> None:
    # A2, end-to-end through the REAL pipeline (parse → stamp → read), not a hand-built StepView:
    # a pytest USAGE_ERROR (exit 4) means the session never validly ran, so the step carries no
    # verified outcome and contributes no FailureEvent. Before the fix it was stamped a blocking
    # capability red with a `hash:` dedup key, and 394 committed steps carried exactly that.
    outcome = parse_test_outcome("ERROR: not found: /testbed/a.py::test_x", 4)
    step = stamp_step(make_step(step_index=0, decision_index=0), outcome)
    assert step.exit_code == 4
    assert replay.verified_outcome(step) is replay.VerifiedOutcome.NONE
    assert step.failing_check_id is None  # nothing to dedup on ⇒ no FailureEvent is constructible
    assert step.blocking is False


def test_repeated_usage_errors_never_fire_the_recurrence_trigger() -> None:
    # The consequence: two identical exit-4 steps share a CONSTANT key, so at the sharpest
    # threshold the pre-fix pipeline escalated on data no agent action produced.
    outcome = parse_test_outcome("ERROR: not found: /testbed/a.py::test_x", 4)
    traj = make_trajectory(
        [stamp_step(make_step(step_index=i, decision_index=i), outcome) for i in range(2)]
    )
    decision = replay.replay_config(traj, GridPoint(1, 10).to_config(), context=_CTX)
    assert not decision.escalated
    assert decision.directives == [EscalationAction.HOLD, EscalationAction.HOLD]


def test_sweep_emits_one_point_per_grid_point() -> None:
    trajs = [_two_same_key_failures(), _two_same_key_failures()]
    grid = [GridPoint(2, 10), GridPoint(3, 5), GridPoint(2, 20, ladder="rank_only")]
    result = replay.sweep(trajs, grid)
    assert len(result.points) == len(grid)
    for point in result.points:
        assert len(point.decisions) == len(trajs)  # every trajectory replayed, no drops
    assert [p.grid_point for p in result.points] == grid  # no dups, order preserved
