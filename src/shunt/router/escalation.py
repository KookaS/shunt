"""Auto-escalation: a repeated verified failure raises one rung at the next boundary."""

# Pure decision logic, no I/O. The engine wires it only when router.yaml enables it (off by
# default). Boundary-only by construction: a directive applies to the NEXT decision, never
# mid-cached-turn (cache-safety spine). A "verified same failure seen N times" signal — same
# normalized failing-check id — steps effort first (cache-safe), then a model rank.

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from enum import StrEnum


class EscalationAction(StrEnum):
    """What the next-boundary decision should do."""

    HOLD = "hold"
    RAISE_EFFORT = "raise_effort"  # same model → cache-safe cheaper rung
    RAISE_RANK = "raise_rank"


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
    # effort_then_rank: raise the current model's reasoning effort first (cache-safe), step a
    # model rank only on recurrence at the higher effort. "rank_only" skips the effort rung.
    ladder: str = "effort_then_rank"
    # SMART-style epsilon-greedy exploration at FLAGGED checkpoints only. 0.0 = fully
    # deterministic (today's behaviour, bit for bit). Above 0 the policy arm is taken with
    # probability 1-epsilon and the HOLD arm with epsilon, and the realized propensity is
    # logged — without which the escalation policy's value stays non-identified (P(escalate)=0
    # breaks the overlap condition every off-policy estimator needs). Additionally opt-in:
    # enabling escalation alone must never randomize anything.
    exploration_epsilon: float = 0.0
    # Seed of the injected decision stream, recorded on every record so a logged propensity
    # can be re-derived after the fact. None ⇒ the caller draws and records one.
    exploration_seed: int | None = None


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
    """The rank/effort ladder position + loop-health guard, read at the boundary."""

    current_rank_index: int
    max_rank_index: int
    current_effort_index: int
    max_effort_index: int
    loop_health_alarm: bool = False  # routing collapse → suppress escalation


@dataclass(frozen=True)
class ExplorationStream:
    """A seeded RNG stream for epsilon-greedy escalation — injected, never a module global."""

    seed: int
    rng: random.Random

    @classmethod
    def from_seed(cls, seed: int) -> ExplorationStream:
        """Build a stream whose draws are exactly reproducible from *seed*."""
        return cls(seed=seed, rng=random.Random(seed))


@dataclass(frozen=True)
class ExplorationRecord:
    """One flagged-checkpoint decision and the propensity that generated it.

    The propensity IS the deliverable: without it, randomized data is still unusable for
    IPS/DR. ``randomized=False`` marks a decision the estimator must exclude, not weight.
    """

    checkpoint_id: str  # the recurring dedup key that flagged this checkpoint
    decision_index: int
    action: EscalationAction  # the arm actually taken
    policy_action: EscalationAction  # what the deterministic policy WOULD have done
    propensity: float  # P(action taken | state) under the logging policy
    epsilon: float
    seed: int | None
    randomized: bool
    features: dict[str, float] = field(default_factory=dict)  # state, as of the decision


@dataclass(frozen=True)
class EscalationDirective:
    """The pre-decided branch for the next decision (never 'ask the human')."""

    action: EscalationAction
    reason: str
    new_label_window: bool = False  # escalation opens a fresh, non-policy label window
    # Present only at a flagged checkpoint; None everywhere else (an unflagged step is not a
    # decision the estimator may weight, so it must carry no propensity).
    exploration: ExplorationRecord | None = None


def derive_blocking(success: bool, is_infra_failure: bool) -> bool:
    """A confirmed, non-infra capability failure — the ONE derivation, shared by every site."""
    # The single predicate for `blocking`. The event constructor, the live trajectory record, and
    # the offline authenticity recompute all call this, so the rule cannot drift across copies.
    return not success and not is_infra_failure


def failure_event_from_outcome(
    *,
    decision_index: int,
    failing_check_id: str,
    exit_code: int | None = None,
    success: bool,
    is_infra_failure: bool,
    confirmed: bool,
) -> FailureEvent:
    """Build the FailureEvent live capture and offline replay must agree on (single builder)."""
    # The `blocking` rule is derived via the shared predicate so the two paths cannot drift and the
    # offline sweep optimizes the policy production actually runs. `exit_code` is decision-inert
    # (carried, never read); its default lives HERE so live and offline inherit the same fallback.
    return FailureEvent(
        decision_index=decision_index,
        dedup_key=failing_check_id,
        exit_code=exit_code if exit_code is not None else EscalationConfig().blocking_exit_code,
        success=success,
        confirmed=confirmed,
        blocking=derive_blocking(success, is_infra_failure),
    )


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
    """Pick the next rung: effort first (cache-safe) then rank, honoring the ceiling."""
    at_rank_ceiling = ctx.current_rank_index >= ctx.max_rank_index
    at_effort_ceiling = ctx.current_effort_index >= ctx.max_effort_index
    if config.ladder == "effort_then_rank" and not at_effort_ceiling:
        return EscalationDirective(
            EscalationAction.RAISE_EFFORT, "same_verified_failure_x2", new_label_window=True
        )
    if not at_rank_ceiling:
        return EscalationDirective(
            EscalationAction.RAISE_RANK, "same_verified_failure_x2", new_label_window=True
        )
    # Nothing higher on either axis — hold, don't thrash.
    return EscalationDirective(EscalationAction.HOLD, "escalation_ceiling")


def _checkpoint_features(
    windowed: list[FailureEvent],
    current_index: int,
    ctx: EscalationContext,
    config: EscalationConfig,
) -> dict[str, float]:
    """The state a doubly-robust model component may condition on, as of THIS decision."""
    # Decision-time only — nothing here reads a later step or the terminal outcome, so a model
    # fit on these features cannot read its own label off the future.
    return {
        "decision_index": float(current_index),
        "window_events": float(len(windowed)),
        "countable_failures": float(sum(1 for e in windowed if counts_as_failure(e, config))),
        "distinct_keys": float(len({e.dedup_key for e in windowed})),
        "effort_headroom": float(ctx.max_effort_index - ctx.current_effort_index),
        "rank_headroom": float(ctx.max_rank_index - ctx.current_rank_index),
    }


def _randomize(
    directive: EscalationDirective,
    record: ExplorationRecord,
    config: EscalationConfig,
    stream: ExplorationStream | None,
) -> EscalationDirective:
    """Epsilon-greedy over {policy arm, HOLD} at a flagged checkpoint, logging the propensity."""
    # The alternative arm is always HOLD, so exploration can only WITHHOLD an escalation, never
    # invent one: no model switch exists here that the deterministic policy would not also make,
    # and the directive still applies to the NEXT boundary (cache-safety is untouched).
    if directive.action is EscalationAction.HOLD:  # no viable rung — do not fabricate an arm
        return replace(directive, exploration=record)
    if config.exploration_epsilon <= 0.0:
        return replace(directive, exploration=record)
    if stream is None:
        raise ValueError(
            "escalation.exploration_epsilon > 0 requires an injected ExplorationStream "
            "(a silent deterministic fallback would log propensities that never happened)"
        )
    if stream.rng.random() < config.exploration_epsilon:
        held = EscalationDirective(EscalationAction.HOLD, "exploration_hold")
        return replace(
            held,
            exploration=replace(
                record,
                action=EscalationAction.HOLD,
                propensity=config.exploration_epsilon,
                randomized=True,
            ),
        )
    return replace(
        directive,
        exploration=replace(record, propensity=1.0 - config.exploration_epsilon, randomized=True),
    )


def decide_escalation(
    events: list[FailureEvent],
    current_index: int,
    ctx: EscalationContext,
    config: EscalationConfig,
    stream: ExplorationStream | None = None,
) -> EscalationDirective:
    """The escalation decision for the next boundary. Pure; every branch ends in a directive.

    Collapse suppresses first; then same-key recurrence flags a checkpoint (the only place
    exploration may randomize); otherwise HOLD. Disabled config always holds.
    """
    if not config.enabled:
        return EscalationDirective(EscalationAction.HOLD, "disabled")
    if ctx.loop_health_alarm:  # escalating into a routing collapse voids the cost gate
        return EscalationDirective(EscalationAction.HOLD, "collapse_suppressed")
    windowed = _in_window(events, current_index, config)
    key = _recurring_key(windowed, config)
    if key is None:  # no same-key recurrence → unflagged, so no decision to weight
        return EscalationDirective(EscalationAction.HOLD, "no_recurring_failure")
    policy_directive = _ladder_action(ctx, config)
    record = ExplorationRecord(
        checkpoint_id=key,
        decision_index=current_index,
        action=policy_directive.action,
        policy_action=policy_directive.action,
        propensity=1.0,
        epsilon=config.exploration_epsilon,
        seed=stream.seed if stream is not None else config.exploration_seed,
        randomized=False,
        features=_checkpoint_features(windowed, current_index, ctx, config),
    )
    return _randomize(policy_directive, record, config, stream)


@dataclass(frozen=True)
class _LadderState:
    """Abstract ladder position: which effort rung, which rank. Advanced by the runner."""

    effort_index: int = 0
    rank_index: int = 0


class EscalationRunner:
    """Pure, engine-state-free driver of the escalation lifecycle over a stream of outcomes."""

    # Owns the failure log AND an abstract ladder position. The log lifecycle and the EFFORT rung
    # evolve exactly as the live router does: append confirmed failures, and on a verified success
    # clear the log AND reset effort to the default rung (mirroring the engine's effort-arm pop);
    # on any non-HOLD directive retire the acted-on window and step effort up to its ceiling. The
    # live engine builds each per-decision directive through `decide` (seeding its own
    # concrete effort/rank position) and applies it concretely; the offline replay drives a whole
    # trajectory through `step`, letting `commit` advance the ladder. The RANK advance in `commit`
    # (effort-ceiling -> raise rank, reset effort) is a persistent monotone counter used ONLY by the
    # trajectory replay's isolation model — the engine has no persistent rank ladder (its rank is
    # re-seeded from the base routing pick each decision), so this is a self-contained upper bound,
    # not a live-engine reproduction. Log-lifecycle and effort parity are the guaranteed part.

    def __init__(
        self,
        *,
        max_effort_index: int,
        max_rank_index: int,
        log: list[FailureEvent] | None = None,
        effort_index: int = 0,
        rank_index: int = 0,
    ) -> None:
        self._max_effort_index = max_effort_index
        self._max_rank_index = max_rank_index
        self._log: list[FailureEvent] = list(log) if log else []
        self._state = _LadderState(effort_index, rank_index)

    @property
    def log(self) -> list[FailureEvent]:
        """The current windowed failure log (retired to empty on each escalation)."""
        return self._log

    def observe(self, *, success: bool, event: FailureEvent | None) -> None:
        """Clear the window AND reset effort on a verified success; else append the failure."""
        if success:
            # Mirror the live engine's success reset precisely: it pops the task's effort arm
            # (effort → the model's default rung) but does NOT persist a tier ladder (it re-seeds
            # tier from the base routing pick each decision), so a verified pass resets effort to 0
            # and leaves the abstract tier counter untouched. Without the effort reset a later
            # same-key failure run would jump straight to a rank while the engine climbs effort.
            self._log = []
            self._state = _LadderState(0, self._state.rank_index)
        elif event is not None:
            self._log.append(event)

    def decide(
        self,
        current_index: int,
        config: EscalationConfig,
        *,
        loop_health_alarm: bool = False,
        stream: ExplorationStream | None = None,
    ) -> EscalationDirective:
        """The directive for the next boundary at the current ladder position (pure)."""
        ctx = EscalationContext(
            current_rank_index=self._state.rank_index,
            max_rank_index=self._max_rank_index,
            current_effort_index=self._state.effort_index,
            max_effort_index=self._max_effort_index,
            loop_health_alarm=loop_health_alarm,
        )
        return decide_escalation(self._log, current_index, ctx, config, stream=stream)

    def commit(self, action: EscalationAction) -> None:
        """Retire the window the escalation acted on and advance one rung (effort then rank)."""
        self._log = []
        if action is EscalationAction.RAISE_EFFORT:
            self._state = _LadderState(self._state.effort_index + 1, self._state.rank_index)
        elif action is EscalationAction.RAISE_RANK:
            self._state = _LadderState(0, self._state.rank_index + 1)

    def step(  # noqa: PLR0913 (lifecycle driver: one arg per stage of observe→decide→commit)
        self,
        *,
        success: bool,
        event: FailureEvent | None,
        current_index: int,
        config: EscalationConfig,
        loop_health_alarm: bool = False,
        stream: ExplorationStream | None = None,
    ) -> EscalationDirective:
        """One outcome → directive via the full lifecycle (observe, decide, retire+advance)."""
        self.observe(success=success, event=event)
        directive = self.decide(
            current_index, config, loop_health_alarm=loop_health_alarm, stream=stream
        )
        if directive.action is not EscalationAction.HOLD:
            self.commit(directive.action)
        return directive


__all__ = [
    "EscalationAction",
    "EscalationConfig",
    "EscalationContext",
    "EscalationDirective",
    "EscalationRunner",
    "ExplorationRecord",
    "ExplorationStream",
    "FailureEvent",
    "counts_as_failure",
    "decide_escalation",
    "derive_blocking",
    "failure_event_from_outcome",
]
