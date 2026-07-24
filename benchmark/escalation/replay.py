"""Direct-Method replay of the production escalation policy over stored trajectories."""

# Mirrors the routing exploration replay: the decision under test (`decide_escalation`) is
# IMPORTED from the shipped router, never re-implemented. Every replayed boundary reads a
# MEASURED outcome, so the estimate is Direct-Method-exact — no importance weighting.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from shunt.router.escalation import (
    EscalationAction,
    EscalationConfig,
    EscalationContext,
    EscalationRunner,
    FailureEvent,
    failure_event_from_outcome,
)

if TYPE_CHECKING:
    from benchmark.escalation.schema import StepView, Trajectory

# A permissive ladder position: neither ceiling reached, no collapse alarm — so a same-key
# recurrence is free to fire. The sweep varies the policy knobs, not the ladder geometry.
#
# SCOPE OF PARITY (read before trusting the tier rung). This replay measures the escalation policy
# IN ISOLATION, holding routing constant at this fixed context. Parity with the live engine is
# guaranteed for (i) the failure-log lifecycle (append / clear-on-success / retire-on-escalation)
# and (ii) the EFFORT rung, which the engine persists per task AND resets to the default on a
# verified success (mirrored here), so effort parity holds even across a success boundary. The TIER
# rung is NOT engine-faithful: here the runner climbs a persistent monotone tier counter that
# saturates at a ceiling,
# whereas the engine re-seeds tier from the base routing pick each decision (no persistent tier
# ladder) and re-escalates indefinitely. Treat the tier stream as an abstract isolation upper
# bound. The detector metric (run_eval) reads only the FIRST-flag prefix, so it stays faithful.
_PERMISSIVE_CONTEXT = EscalationContext(
    current_tier_index=0,
    max_tier_index=3,
    current_effort_index=0,
    max_effort_index=3,
    loop_health_alarm=False,
)


@dataclass(frozen=True)
class GridPoint:
    """One hyperparameter cell of the sweep."""

    escalate_after_n: int
    stale_window: int
    ladder: str = "effort_then_tier"

    def to_config(self) -> EscalationConfig:
        """The EscalationConfig this cell drives `decide_escalation` with (enabled)."""
        return EscalationConfig(
            enabled=True,
            escalate_after_n=self.escalate_after_n,
            stale_window=self.stale_window,
            ladder=self.ladder,
        )


@dataclass(frozen=True)
class ReplayDecision:
    """The directive stream a config produces over one trajectory + its detection point."""

    trajectory_id: str
    grid_point: GridPoint
    directives: list[EscalationAction]
    first_escalation_index: int | None

    @property
    def escalated(self) -> bool:
        return self.first_escalation_index is not None


@dataclass(frozen=True)
class SweepPoint:
    """One GridPoint's replay over every trajectory."""

    grid_point: GridPoint
    decisions: list[ReplayDecision]


@dataclass(frozen=True)
class SweepResult:
    """The full sweep: exactly one SweepPoint per GridPoint."""

    points: list[SweepPoint]


def _step_failure_event(step: StepView) -> FailureEvent | None:
    """The FailureEvent a step contributes, via the SHARED constructor, or None if not a failure.

    Success steps and dedup-less steps contribute nothing — mirroring the live path, which only
    logs confirmed failures that carry a failing-check id.
    """
    if step.success or step.failing_check_id is None:
        return None
    return failure_event_from_outcome(
        decision_index=step.decision_index,
        failing_check_id=step.failing_check_id,
        # None-preserving: the shared constructor owns the missing-exit_code default so this
        # matches the live engine's stored value byte-for-byte (parity by construction).
        exit_code=step.exit_code,
        success=step.success,
        is_infra_failure=step.is_infra_failure,
        confirmed=step.confirmed,
    )


def replay_config(
    traj: Trajectory, cfg: EscalationConfig, *, context: EscalationContext = _PERMISSIVE_CONTEXT
) -> ReplayDecision:
    """Replay the escalation lifecycle boundary-by-boundary over a trajectory (isolation model)."""
    # Drives the SHARED `EscalationRunner` the live engine also uses, so the log lifecycle (append /
    # clear-on-success / retire-on-escalation) and the EFFORT rung match the engine by construction
    # — not a hand-rolled copy. `context` sets the starting ladder position and the ceilings the
    # abstract ladder climbs against. The TIER advance here is a persistent monotone counter with no
    # engine counterpart (the engine re-seeds tier from routing each decision) — an isolation upper
    # bound, not an engine reproduction. See the SCOPE OF PARITY note above.
    runner = EscalationRunner(
        max_effort_index=context.max_effort_index,
        max_tier_index=context.max_tier_index,
        effort_index=context.current_effort_index,
        tier_index=context.current_tier_index,
    )
    directives: list[EscalationAction] = []
    first_escalation: int | None = None
    for step in traj.steps:
        directive = runner.step(
            success=step.success,
            event=_step_failure_event(step),
            current_index=step.decision_index,
            config=cfg,
            loop_health_alarm=context.loop_health_alarm,
        )
        directives.append(directive.action)
        if first_escalation is None and directive.action is not EscalationAction.HOLD:
            first_escalation = step.step_index
    return ReplayDecision(
        trajectory_id=traj.header.trajectory_id,
        grid_point=GridPoint(cfg.escalate_after_n, cfg.stale_window, cfg.ladder),
        directives=directives,
        first_escalation_index=first_escalation,
    )


def sweep(trajs: list[Trajectory], grid: list[GridPoint]) -> SweepResult:
    """One SweepPoint per GridPoint, each replaying every trajectory — no drops, no dups."""
    points = [
        SweepPoint(
            grid_point=gp,
            decisions=[replay_config(traj, gp.to_config()) for traj in trajs],
        )
        for gp in grid
    ]
    return SweepResult(points=points)


__all__ = [
    "GridPoint",
    "ReplayDecision",
    "SweepPoint",
    "SweepResult",
    "replay_config",
    "sweep",
]
