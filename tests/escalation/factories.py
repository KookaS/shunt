"""Small in-test builders for StepView/Trajectory. Test-only — never a live/benchmark path."""

from __future__ import annotations

from benchmark.escalation import schema
from benchmark.escalation.schema import StepView, Trajectory, TrajectoryHeader


def make_step(  # noqa: PLR0913 (test builder — many optional knobs by design)
    *,
    step_index: int = 0,
    decision_index: int = 0,
    failing_check_id: str | None = None,
    success: bool = True,
    is_infra_failure: bool = False,
    confirmed: bool = True,
    test_passed: int | None = None,
    test_total: int | None = None,
    observation: str = "obs",
    action: str = "act",
    args: str | None = "args",
    result: str = "res",
) -> StepView:
    """A StepView with sane defaults; derivable fields set consistently."""
    return StepView(
        step_index=step_index,
        decision_index=decision_index,
        parent_step_index=None,
        metadata={"k": "v"},
        observation=observation,
        action=action,
        tool="bash",
        args=args,
        result=result,
        status="ok" if success else "error",
        test_passed=test_passed,
        test_total=test_total,
        failing_check_id=failing_check_id,
        exit_code=0 if success else 1,
        blocking=not success and not is_infra_failure,
        is_infra_failure=is_infra_failure,
        confirmed=confirmed,
        success=success,
        is_revert=False,
        retry_count=0,
        loop_signal=False,
        subgoal_progress=None,
        dedup_key=failing_check_id,
        model="m",
        reasoning_effort="default",
        tier_index=0,
        effort_index=0,
        real_cost=None,
    )


def make_trajectory(
    steps: list[StepView], *, trajectory_id: str = "t1", terminal_resolved: bool = False
) -> Trajectory:
    """Wrap steps in a header with a recomputed content hash and n_steps."""
    header = TrajectoryHeader(
        schema_version=schema.SCHEMA_VERSION,
        trajectory_id=trajectory_id,
        dataset="test",
        plane="committable",
        framework="swe_agent",
        terminal_resolved=terminal_resolved,
        instance_id=trajectory_id,
        license="MIT",
        dataset_revision="rev0",
        redacted=False,
        content_sha256=schema.content_sha256(steps),
        n_steps=len(steps),
    )
    return Trajectory(header=header, steps=steps)
