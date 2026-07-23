"""Auto-escalation: a repeated verified failure raises one rung at the next boundary."""

# Pure decision logic, no I/O. The engine wires it only when router.yaml enables it (off by
# default). Boundary-only by construction: a directive applies to the NEXT decision, never
# mid-cached-turn (cache-safety spine). A "verified same failure seen N times" signal — same
# normalized failing-check id — steps effort first (cache-safe), then a model tier.

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EscalationAction(StrEnum):
    """What the next-boundary decision should do."""

    HOLD = "hold"
    RAISE_EFFORT = "raise_effort"  # same model → cache-safe cheaper rung
    RAISE_TIER = "raise_tier"


@dataclass(frozen=True)
class EscalationConfig:
    """Escalation knobs (mirror ``router.yaml``). ``enabled`` defaults False (kill-gate)."""

    enabled: bool = False
    # 2, not 1: intermediate fail-then-fix is normal, so a single verified failure is not
    # escalation-worthy (escalating on the first verified failure is failure-biased).
    escalate_after_n: int = 2
    # A failure that does not RECUR within this many decisions is retired from the counter —
    # this is the user's "+10 calls elapsed" idea kept as a staleness bound, not a trigger.
    stale_window: int = 10
    # FUTURE hook-stream path only (not the off-wire path). A Claude-Code Stop hook reports a
    # blocking gate with exit 2; a lint-only red is exit 1. The off-wire verifier instead
    # reports the SUBPROCESS code (pytest/jest/go=1, cargo=101) and sets FailureEvent.blocking
    # from result.outcome/is_infra_failure, so counts_as_failure keys on `blocking`, not this.
    blocking_exit_code: int = 2
    # effort_then_tier: raise the current model's reasoning effort first (cache-safe), step a
    # model tier only on recurrence at the higher effort. "tier_only" skips the effort rung.
    ladder: str = "effort_then_tier"


@dataclass(frozen=True)
class FailureEvent:
    """One verified outcome at a decision boundary. ``dedup_key`` = normalized failing-check id."""

    decision_index: int
    dedup_key: str
    # Recorded for the FUTURE hook-stream path only — carried and serialized, never read by a
    # routing/escalation decision (the off-wire path gates on `blocking`). Retained, not dead.
    exit_code: int
    success: bool
    confirmed: bool  # rerun-confirmed (flake guard) — an unconfirmed failure never counts
    # A confirmed, non-infra capability failure. Set by the CaptureCoordinator from the
    # verifier's outcome/is_infra_failure — NOT from the raw subprocess exit code (a real
    # pytest/jest/go failure is exit 1/101, never the hook contract's 2). Defaults False so a
    # non-blocking event (lint, infra) never counts toward escalation.
    blocking: bool = False


@dataclass(frozen=True)
class EscalationContext:
    """The tier/effort ladder position + loop-health guard, read at the boundary."""

    current_tier_index: int
    max_tier_index: int
    current_effort_index: int
    max_effort_index: int
    loop_health_alarm: bool = False  # routing collapse → suppress escalation


@dataclass(frozen=True)
class EscalationDirective:
    """The pre-decided branch for the next decision (never 'ask the human')."""

    action: EscalationAction
    reason: str
    new_label_window: bool = False  # escalation opens a fresh, non-policy label window


def counts_as_failure(event: FailureEvent, config: EscalationConfig) -> bool:
    """A verified failure counts only if blocking (a capability failure) AND rerun-confirmed."""
    # `blocking` — not the raw exit code — is the gate: the off-wire verifier reports the
    # subprocess code (1/101), so an `exit_code == blocking_exit_code` test dropped every real
    # failure. Non-blocking (lint/infra) and unconfirmed (flake) drop out; success never
    # counts. `config` is retained for the future hook-stream path (see blocking_exit_code).
    del config
    if event.success:
        return False
    if not event.confirmed:  # flake: pass→fail→pass on unchanged state is not a failure
        return False
    return event.blocking


def _in_window(
    events: list[FailureEvent], current_index: int, config: EscalationConfig
) -> list[FailureEvent]:
    """The events still inside the staleness window (older ones are retired from the counter).

    A verified suite pass is captured at the engine as a pop-all (the whole suite went green ⇒
    every key resolved), so the log holds only failures — there is no per-key success to retain.
    """
    return [e for e in events if (current_index - e.decision_index) < config.stale_window]


def _recurring_key(windowed: list[FailureEvent], config: EscalationConfig) -> str | None:
    """Return a dedup key with ``escalate_after_n``+ countable failures in the window, else None.

    Distinct keys are counted separately — repeated *different* failures do not aggregate
    into an escalation (that is a hard task — the kNN store's job).
    """
    groups: dict[str, list[FailureEvent]] = {}
    for e in windowed:
        groups.setdefault(e.dedup_key, []).append(e)
    fail_counts = {
        key: sum(1 for e in g if counts_as_failure(e, config)) for key, g in groups.items()
    }
    recurring = [key for key in groups if fail_counts[key] >= config.escalate_after_n]
    if not recurring:
        return None
    # Deterministic tie-break: the most-repeated key, then lexical — stable across runs.
    recurring.sort(key=lambda k: (-fail_counts[k], k))
    return recurring[0]


def _ladder_action(ctx: EscalationContext, config: EscalationConfig) -> EscalationDirective:
    """Pick the next rung: effort first (cache-safe) then tier, honoring the ceiling."""
    at_tier_ceiling = ctx.current_tier_index >= ctx.max_tier_index
    at_effort_ceiling = ctx.current_effort_index >= ctx.max_effort_index
    if config.ladder == "effort_then_tier" and not at_effort_ceiling:
        return EscalationDirective(
            EscalationAction.RAISE_EFFORT, "same_verified_failure_x2", new_label_window=True
        )
    if not at_tier_ceiling:
        return EscalationDirective(
            EscalationAction.RAISE_TIER, "same_verified_failure_x2", new_label_window=True
        )
    # Nothing higher on either axis — hold, don't thrash.
    return EscalationDirective(EscalationAction.HOLD, "escalation_ceiling")


def decide_escalation(
    events: list[FailureEvent], current_index: int, ctx: EscalationContext, config: EscalationConfig
) -> EscalationDirective:
    """The escalation decision for the next boundary. Pure; every branch ends in a directive.

    Collapse suppresses first; then same-key recurrence drives the ladder; otherwise HOLD
    (absent signal). Disabled config always holds.
    """
    if not config.enabled:
        return EscalationDirective(EscalationAction.HOLD, "disabled")
    if ctx.loop_health_alarm:  # escalating into a routing collapse voids the cost gate
        return EscalationDirective(EscalationAction.HOLD, "collapse_suppressed")
    windowed = _in_window(events, current_index, config)
    if _recurring_key(windowed, config) is None:  # no same-key recurrence → nothing to do
        return EscalationDirective(EscalationAction.HOLD, "no_recurring_failure")
    return _ladder_action(ctx, config)


__all__ = [
    "EscalationAction",
    "EscalationConfig",
    "EscalationContext",
    "EscalationDirective",
    "FailureEvent",
    "counts_as_failure",
    "decide_escalation",
]
