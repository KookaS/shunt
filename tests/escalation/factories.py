"""Small in-test builders for StepView/Trajectory. Test-only — never a live/benchmark path."""

from __future__ import annotations

from benchmark.escalation import schema
from benchmark.escalation.metrics import NullResult
from benchmark.escalation.prefix_eval import DepthReport
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
    exit_code: int | None = None,
) -> StepView:
    """A StepView with sane defaults; derivable fields set consistently.

    `exit_code` defaults to the one implied by `success`; pass it to decouple the two.
    """
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
        exit_code=(0 if success else 1) if exit_code is None else exit_code,
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
        rank_index=0,
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


def make_null(observed: float, ci_high: float, *, p_value: float = 0.5) -> NullResult:
    """A permutation null positioned so `beats_null` is exactly `observed > ci_high`."""
    return NullResult(
        observed=observed,
        mean=ci_high / 2.0,
        sd=0.05,
        ci_low=0.0,
        ci_high=ci_high,
        p_value=p_value,
        n_permutations=1000,
        draws=(ci_high,),
    )


def make_depth_report(  # noqa: PLR0913 (test builder — many optional knobs by design)
    *,
    depth: int = 5,
    auroc_prefix: float = 0.5,
    incremental_auroc: float = 0.0,
    null_prefix: NullResult | None = None,
    null_incremental: NullResult | None = None,
    ci_incremental: tuple[float, float] = (0.0, 0.0),
    scores: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
    labels: tuple[bool, ...] = (False, True, False, True),
) -> DepthReport:
    """A DepthReport whose skill-gate inputs are set directly, with the rest left inert."""
    return DepthReport(
        depth=depth,
        n_rows=len(scores),
        n_excluded_short=0,
        n_groups=len(scores),
        base_rate=0.5,
        auroc_prior=0.6,
        auroc_prior_leaked=0.7,
        auroc_prefix=auroc_prefix,
        auroc_prefix_folded=auroc_prefix,
        auroc_combined=auroc_prefix + incremental_auroc,
        incremental_auroc=incremental_auroc,
        auprc_prefix=0.5,
        null_prefix=null_prefix or make_null(auroc_prefix, 0.51),
        null_incremental=null_incremental or make_null(incremental_auroc, 0.05),
        ci_prefix=(0.4, 0.6),
        ci_incremental=ci_incremental,
        scores=scores,
        labels=labels,
    )
