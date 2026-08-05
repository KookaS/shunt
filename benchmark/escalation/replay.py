"""Direct-Method replay of the production escalation policy over stored trajectories."""

# Mirrors the routing exploration replay: the decision under test (`decide_escalation`) is
# IMPORTED from the shipped router, never re-implemented. Every replayed boundary reads a
# MEASURED outcome, so the estimate is Direct-Method-exact — no importance weighting.

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

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

# Write-operations that change the workspace tree, used by `count_from_first_edit` to skip the
# reproduction phase. Reads (`cat`, `grep`, `sed -n`, `less`, `find`, `ls`, `git log`, `git diff`)
# are deliberately absent — the point is to find the first real ATTEMPT to change the code.
# `patch` is matched only as a real patch WRITE (the `patch(1)` utility applying a diff, or
# `applypatch`), never the bare word: `cat patch.txt`, `curl …/patch -o` and `from
# unittest.mock import patch` are reads/mocks, and matching them ended 69 of 794 trajectories'
# reproduction phase on a false edit. `python -i` is absent too: interactive, not in-place.
# This is a heuristic over the logged command string (the committed corpus carries no per-step
# diffs), used ONLY to gate the eval's reproduction phase, never by the live router.
_EDIT_ACTION_RE: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bsed\s+-i"),
    re.compile(r"\bperl\s+-i"),
    re.compile(r"\bpatch\s+-[a-z]*[pi]"),  # patch -p1 / patch -i file
    re.compile(r"\bpatch\s+<"),  # patch < file
    re.compile(r"\bapplypatch\b"),
    re.compile(r"\bgit\s+apply\b"),
    re.compile(r"\btee\b"),
    re.compile(r"\bcat\s+>\s*\S"),
    re.compile(r"\bcat\s+>>\s*\S"),
    re.compile(r"\becho\b.*(?:>>|>)\s*\S"),
    re.compile(r"\bprintf\b.*(?:>>|>)\s*\S"),
    re.compile(r"open\([^)]*['\"]w['\"]"),  # python: open(path, 'w')
    re.compile(r"\.write\("),  # python: file.write(...)
    re.compile(r"\bgit\s+add\b"),
    re.compile(r"\bgit\s+commit\b"),
    re.compile(r"\brm\s+"),
    re.compile(r"\bmv\s+"),
    re.compile(r"\bcp\s+"),
)


def is_edit_action(action: str) -> bool:
    """Whether a command writes to the workspace tree — a real attempt to change the code."""
    if not action:
        return False
    if "/tmp" in action and re.search(r"\b(?:cat|echo|printf)\b", action):
        return False  # scratch writes are not code changes
    return any(pattern.search(action) for pattern in _EDIT_ACTION_RE)


# A permissive ladder position: neither ceiling reached, no collapse alarm — so a same-key
# recurrence is free to fire. The sweep varies the policy knobs, not the ladder geometry.
#
# SCOPE OF PARITY (read before trusting the rank rung). This replay measures the escalation policy
# IN ISOLATION, holding routing constant at this fixed context. Parity with the live engine is
# guaranteed for (i) the failure-log lifecycle — append, clear on a VERIFIED PASS only (an
# unstamped or infra/unknown step is a no-op, as it is live), retire-on-escalation —
# and (ii) the EFFORT rung, which the engine persists per task AND resets to the default on a
# verified success (mirrored here), so effort parity holds even across a success boundary. The RANK
# rung is NOT engine-faithful: here the runner climbs a persistent monotone rank counter that
# saturates at a ceiling,
# whereas the engine re-seeds rank from the base routing pick each decision (no persistent rank
# ladder) and re-escalates indefinitely. Treat the rank stream as an abstract isolation upper
# bound. The detector metric (run_eval) reads only the FIRST-flag prefix, so it stays faithful.
#
# (iii) CADENCE is NOT engine-faithful either, and it is the widest of the gaps. This replay feeds
# the runner one event PER STEP: `replay_config` loops over `traj.steps` and calls
# `EscalationRunner.step` once per StepView — instrumented, an 8-step trajectory produces 8 calls.
# Live, an event is produced once PER CLOSED SESSION: `CaptureCoordinator.capture` runs the
# off-wire verifier once for a closed session and reaches `record_outcome` at most once per
# capture. Every trajectory in `data/live/` is a single session (799 files, 799 distinct
# trajectory ids, median 31 steps, longest 247), so at the live cadence a whole trajectory
# contributes exactly ONE event — and the shipped `escalate_after_n=2` could never fire on this
# corpus at all. What the sweep measures is recurrence across the steps within one session, which
# is not the quantity the shipped rule counts. It is not repairable on this data: a
# session-cadence replay needs trajectories spanning several sessions and none of these do. Any
# `escalate_after_n` result read off this sweep describes a per-step policy production does not run.
#
# (iv) The FLAKE GUARD is not exercised at all. `counts_as_failure` drops any event with
# `confirmed=False` — a failure that did not reproduce on re-run. Offline,
# `normalize.mini_swe_agent.stamp_step` hardcodes `confirmed=True`, ignoring
# `VerifierResult.confirmed`; and `benchmark/runner/offline_replay.replay_step` runs each step's
# test directives exactly once, so there is no second execution a genuine `confirmed` could
# come from. Every replayed event therefore satisfies the guard by construction. Its effect is
# UNMEASURED, not measured-as-zero. `confirmed` cannot be decoupled here in passing: it doubles as
# the stamped-ness marker that `features.is_stamped` and `verified_outcome` below both read.
_PERMISSIVE_CONTEXT = EscalationContext(
    current_rank_index=0,
    max_rank_index=3,
    current_effort_index=0,
    max_effort_index=3,
    loop_health_alarm=False,
)


@dataclass(frozen=True)
class GridPoint:
    """One hyperparameter cell of the sweep."""

    escalate_after_n: int
    stale_window: int
    ladder: str = "effort_then_rank"

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


class VerifiedOutcome(StrEnum):
    """Whether a step carries a verified outcome at all, and which one."""

    PASS = "pass"
    FAIL = "fail"
    NONE = "none"  # the verified-outcome stage never labelled this step


def verified_outcome(step: StepView) -> VerifiedOutcome:
    """The tri-state the live capture path gates on — `success: bool` alone cannot express it."""
    # Live, `CaptureCoordinator` reaches `record_outcome` only for a verifier outcome in
    # {success, weak_success, failure}; an `unknown`/infra red never gets there, so the failure log
    # is untouched. Offline, BOTH the unstamped step (`success=True` is the parser default) and the
    # infra step (`success = outcome != "failure"`) carry `success=True`, and feeding either to
    # `observe` as a pass would clear a window the live engine keeps. Keyed on `confirmed` — the
    # same stamped-ness marker `features.is_stamped` reads — and on `is_infra_failure`, which
    # `parse_test_outcome` sets only for `unknown`, the one outcome that is non-labellable while
    # leaving `success` true.
    if not step.success:
        return VerifiedOutcome.FAIL  # a red is never a parser default: the stage ran
    if step.confirmed and not step.is_infra_failure:
        return VerifiedOutcome.PASS
    return VerifiedOutcome.NONE


def _step_failure_event(step: StepView) -> FailureEvent | None:
    """The FailureEvent a verified failure contributes, via the SHARED constructor, or None.

    A failure with no failing-check id contributes nothing — mirroring the live path, which
    returns early when `dedup_key is None` rather than logging an unkeyed event.
    """
    if step.failing_check_id is None:
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


def max_recurrence(
    traj: Trajectory,
    cfg: EscalationConfig,
    *,
    count_from_first_edit: bool = False,
) -> int:
    """The largest same-key failure count this trajectory's recurrence ever reaches — its score."""
    # The recurrence rule is a threshold over a running count; this returns the count itself (max
    # over keys of the windowed, countable failures), so a continuous ROC/PR over ALL thresholds can
    # be drawn instead of only the grid's operating points. It mirrors `replay_config`'s lifecycle
    # (edit-gated reproduction phase, verified-pass clears the window, `stale_window` retirement)
    # but keeps the running max instead of firing at a threshold, so the score is threshold-free and
    # a single pass serves every `escalate_after_n` a reader might consider.
    from collections import Counter  # noqa: PLC0415

    from shunt.router.escalation import counts_as_failure  # noqa: PLC0415

    window: list[FailureEvent] = []
    max_count = 0
    edit_seen = not count_from_first_edit
    for step in traj.steps:
        if not edit_seen and is_edit_action(step.action):
            edit_seen = True
        if not edit_seen:
            continue  # reproduction phase: failures are not escalation evidence
        outcome = verified_outcome(step)
        if outcome is VerifiedOutcome.PASS:
            window = []  # a verified suite pass retires the window, as the runner's observe does
            continue
        if outcome is not VerifiedOutcome.FAIL:
            continue  # unverified/infra: a no-op, exactly as in replay_config
        event = _step_failure_event(step)
        if event is None:
            continue
        window.append(event)
        window = [e for e in window if (step.decision_index - e.decision_index) < cfg.stale_window]
        counts = Counter(e.dedup_key for e in window if counts_as_failure(e, cfg))
        max_count = max(max_count, max(counts.values(), default=0))
    return max_count


def replay_config(
    traj: Trajectory,
    cfg: EscalationConfig,
    *,
    context: EscalationContext = _PERMISSIVE_CONTEXT,
    count_from_first_edit: bool = False,
) -> ReplayDecision:
    """Replay the escalation lifecycle boundary-by-boundary over a trajectory (isolation model)."""
    # Drives the SHARED `EscalationRunner` the live engine also uses, so the log lifecycle (append /
    # clear-on-success / retire-on-escalation) and the EFFORT rung match the engine by construction
    # — not a hand-rolled copy. `context` sets the starting ladder position and the ceilings the
    # abstract ladder climbs against. The RANK advance here is a persistent monotone counter with no
    # engine counterpart (the engine re-seeds rank from routing each decision) — an isolation upper
    # bound, not an engine reproduction. See the SCOPE OF PARITY note above.
    #
    # `count_from_first_edit`: skip the reproduction phase. On this corpus the first one or two
    # replayed steps of nearly EVERY run — successful or not — are red on the target F2P test,
    # because the agent starts by reproducing the bug, so the shipped `escalate_after_n=2` fires on
    # 727/727 runs and reads the base rate. Those early failures are the target bug at t=0, not
    # "an edit did not work", and a counter that counts them cannot separate a run the agent is
    # stuck on from a run it is about to fix. When set, failures before the first edit-like action
    # feed the runner as a NO-OP (neither appending to the failure log nor clearing it — exactly
    # like an unverified step), so recurrence only accumulates once the agent has actually started
    # changing code. This is an EVAL-ONLY knob: the live router has no per-step action stream, and
    # it is how the sweep measures what a per-step escalation policy could do if the signal were
    # gated on "after the agent tried something".
    runner = EscalationRunner(
        max_effort_index=context.max_effort_index,
        max_rank_index=context.max_rank_index,
        effort_index=context.current_effort_index,
        rank_index=context.current_rank_index,
    )
    directives: list[EscalationAction] = []
    first_escalation: int | None = None
    edit_seen = not count_from_first_edit
    for step in traj.steps:
        if not edit_seen and is_edit_action(step.action):
            edit_seen = True
        outcome = verified_outcome(step)
        if edit_seen:
            event = _step_failure_event(step) if outcome is VerifiedOutcome.FAIL else None
            success = outcome is VerifiedOutcome.PASS
        else:
            # Reproduction phase: the agent has not attempted a change, so its failures are not
            # escalation evidence. Fed as (success=False, event=None), which `observe` treats as a
            # no-op — the live path's exact handling of a non-labellable verifier outcome.
            event = None
            success = False
        # A step with NO verified outcome is fed as (success=False, event=None), which `observe`
        # treats as a no-op — exactly the live path, where a non-labellable verifier outcome never
        # reaches `record_outcome` and leaves the failure log alone. Only a verified pass clears.
        directive = runner.step(
            success=success,
            event=event,
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
    "VerifiedOutcome",
    "replay_config",
    "sweep",
    "verified_outcome",
]
