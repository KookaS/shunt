"""Auto-escalation: a repeated verified failure raises one rung at the next boundary."""

# Pure decision logic, no I/O. The engine wires it only when router.yaml enables it (ships
# ON by default; a boot warning covers the enabled-without-a-work_dir state). Boundary-only by
# construction: a directive applies to the NEXT decision, never
# mid-cached-turn (cache-safety spine). A "verified same failure seen N times" signal — same
# normalized failing-check id — steps effort first (cache-safe), then a model rank.
#
# WHY MID-SESSION SWITCHES ARE FORBIDDEN, IN COST TERMS. A model switch inside a cached turn
# invalidates the prompt cache, and the recompute is roughly 4x the cached context — every
# request that would have hit the cache re-bills its full prefix. Routing escalation at the
# boundary is the same decision made at the moment it costs nothing; a maintainer reading only
# this module must see that a switch made at the wrong time is not a free action.

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Final

from shunt.proxy.redaction import redact_secrets

# The ladder vocabulary, defined once. `EscalationPolicy` (the router.yaml schema) validates
# against this same tuple, so the two entry points cannot drift apart.
ESCALATION_LADDERS: Final[tuple[str, ...]] = ("effort_then_rank", "rank_only")

# Fallback for `FailureEvent.exit_code` when a caller reports none. The field is decision-inert
# (carried and serialized, never read by a routing/escalation decision), so this value only has
# to be stable across the live and offline paths. It matches the Claude-Code Stop-hook contract
# (a blocking gate reports exit 2); the off-wire verifier instead reports the SUBPROCESS code
# (pytest/jest/go=1, cargo=101) and the gate keys on `FailureEvent.blocking`, never on this.
DEFAULT_FAILURE_EXIT_CODE: Final[int] = 2


class EscalationAction(StrEnum):
    """What the next-boundary decision should do."""

    HOLD = "hold"
    RAISE_EFFORT = "raise_effort"  # same model → cache-safe cheaper rung
    RAISE_RANK = "raise_rank"


@dataclass(frozen=True)
class EscalationConfig:
    """Escalation knobs (mirror ``router.yaml``). ``enabled`` defaults True (ships ON)."""

    enabled: bool = True
    # Ships ON by owner choice (2026-08-08), with the honest caveats carried in docs/escalation.md
    # and configuration.md: the recurrence trigger is a NULL DETECTOR at the live cadence (the
    # as-shipped counter fires on 723/723 offline runs at the base rate; its only real edge is the
    # eval-only edit-gated family production cannot run), so no measurement yet shows the shipped
    # counter to separate outcomes; the ladder's VALUE is real but OBSERVATIONAL, not causal (the
    # arms ran in parallel under adaptive coverage): at session cadence, escalating resolves
    # +0.416 [+0.239, +0.581] more tasks per instance than a same-cost retry (n=48 paired;
    # a 3.02x ratio, which carries no interval). Read it DIRECTIONALLY, not as a size — the retry
    # arm covers 27 instances to escalate's 30. THAT CONTRAST IS THE WEAKEST OF THE FOUR BASELINES
    # AND MUST NEVER BE QUOTED ALONE: on the same 48 instances the escalate arm LOSES to
    # always-frontier on quality (-0.108 [-0.165, -0.056], interval excludes zero) and is
    # INDISTINGUISHABLE from firing at random at the same rate (-0.039 [-0.152, +0.084]). There is
    # no cost result to fall back on either — the per-arm USD-per-resolve figures are computed on
    # different task sets against different floors, so they are not comparable. The common-task-set
    # (full-policy) read now exists and is sound on money (-0.2168 [-0.3292, -0.1215] per instance
    # against always-frontier over all 48), but its QUALITY axis is empty: the two arms differ in
    # outcome on 0 of 48 instances, so it is context, not a claim. The measured arm is the corpus's
    # two most expensive models, not the ladder `rank_shortlist` walks. Shipping ON is a DESIGN
    # choice (one decision per session, cache-safe, bounded spend), not a measured win. The scoped
    # claim, its falsifiers and its verdicts are
    # in docs/escalation-claim.md. The ε-greedy + logged-propensity path is how value
    # becomes identified — see benchmark/escalation/reports/metrics.json. And
    # escalation is INERT without a capture.work_dir (no verified-failure signal) — a boot warning
    # says so; it never fires silently into an unconfigured state.
    #
    # 2, not 1: intermediate fail-then-fix is normal, so a single verified failure is not
    # escalation-worthy (escalating on the first verified failure is failure-biased).
    #
    # THIS VALUE IS A PRIOR, NOT A TUNED ONE, and the only measurement that may be cited for it
    # is the one THIS counter produces. In the `as_shipped` family of the policy sweep — the
    # family that replays `counts_as_failure` exactly as written below, at the shipped
    # stale_window=10 — every low threshold is at chance: n=2 fires on 727/727 offline runs at
    # AUROC 0.500 and does not clear its permutation null (p=1.0), n=3 fires on 726/727 at
    # AUROC 0.501, n=4 on 720/727 at 0.506. There is therefore no shipped-counter evidence
    # preferring 3 to 2, nor either to any other low threshold. (The 0.724 AUROC quoted around
    # the report belongs to the `edit_gated` family, which skips failures before the agent's
    # first edit — a rule this module does not implement, `EscalationPolicy` is `extra="forbid"`
    # with no counting knob, and `benchmark/escalation/deployability.py` classifies as
    # offline-only. It is not evidence about this default.)
    # See docs/escalation.md and benchmark/escalation/reports/metrics.json (`policy_sweep.png`).
    escalate_after_n: int = 2
    # A failure that does not RECUR within this many decisions is retired from the counter —
    # this is the user's "+10 calls elapsed" idea kept as a staleness bound, not a trigger.
    stale_window: int = 10
    # effort_then_rank: raise the current model's reasoning effort first (cache-safe), step a
    # model rank only on recurrence at the higher effort. "rank_only" skips the effort rung.
    ladder: str = "effort_then_rank"
    # How many of the CHEAPEST ranks the ladder walks one at a time before jumping straight to
    # the top rank. 0 disables the jump (every rank is a rung — the behaviour that predates this
    # knob). See `RouterEngine._next_rung_rank` for the shape and why 3 is the default.
    rank_shortlist: int = 3
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

    def __post_init__(self) -> None:
        """Reject an out-of-range knob at construction (mirrors the ``EscalationPolicy`` schema)."""
        # Only router.yaml goes through the pydantic schema; a benchmark grid constructs this
        # dataclass DIRECTLY, where an unknown ladder silently degraded to rank_only, an epsilon
        # above 1 wrote an invalid probability into the OPE log, and n=0 escalated off zero
        # countable failures. Validating here makes every entry point equally strict.
        if self.escalate_after_n <= 0:
            raise ValueError(f"escalate_after_n must be > 0, got {self.escalate_after_n}")
        if self.stale_window <= 0:
            raise ValueError(f"stale_window must be > 0, got {self.stale_window}")
        if self.rank_shortlist < 0:
            raise ValueError(f"rank_shortlist must be >= 0, got {self.rank_shortlist}")
        if not 0.0 <= self.exploration_epsilon < 1.0:
            raise ValueError(
                f"exploration_epsilon must be in [0.0, 1.0), got {self.exploration_epsilon}"
            )
        if self.ladder not in ESCALATION_LADDERS:
            joined = ", ".join(ESCALATION_LADDERS)
            raise ValueError(f"unknown ladder {self.ladder!r}; allowed: {joined}")


def next_rung_rank(from_rank: int, max_rank_index: int, rank_shortlist: int) -> int:
    """The rank the ladder aims at from *from_rank*: one up inside the shortlist, else the top.

    Shortlist shape: walk the ``rank_shortlist`` cheapest ranks one at a time, then jump.
    """
    # THE RULE LIVES HERE, ONCE, because two callers apply it and they must not drift: the live
    # `RouterEngine._next_rung_rank` and the offline `SessionCascadeStrategy` replay that measures
    # it. The replay was written before the shortlist existed and walked every rank, so it modelled
    # a ladder production had stopped running — the exact drift a second copy of this arithmetic
    # produces. Pure ints, no pool and no config object, so neither side can specialise it.
    #
    # WHY A SHORTLIST AND NOT EVERY RUNG. Rank order IS price order, so a pool of N models made the
    # ladder buy all N-1 intermediate models on the way to a frontier the task was going to need
    # anyway. Each intermediate rung is paid for AND is only left after another `escalate_after_n`
    # verified recurrences, so it costs sessions as well as dollars. The shape is the offline
    # `Price-Cascade`'s (`by_price[:3] + [frontier]`) because that is the shape whose cost was
    # actually measured on this corpus; a live-only shape would be an unmeasured guess.
    if rank_shortlist <= 0 or from_rank + 1 < rank_shortlist:
        return from_rank + 1
    # `max` keeps the target strictly above `from_rank` in a pool smaller than the shortlist.
    return max(from_rank + 1, max_rank_index)


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

    def persistable(self) -> dict[str, object]:
        """The projection that may be stored — `dedup_key` scrubbed like a committable id."""
        # Belt-and-suspenders: `failure_event_from_outcome` already redacts at construction, so
        # the in-memory log and this snapshot hold the SAME (redacted) string and a recurring key
        # still groups after a restart. This pass stays so a HAND-BUILT event (tests, a future
        # caller) can never reach the PLAINTEXT sqlite `router_state` raw.
        out: dict[str, object] = asdict(self)
        out["dedup_key"] = redact_secrets(self.dedup_key)
        return out


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

    def persistable(self) -> dict[str, object]:
        """The projection that may be stored — `checkpoint_id` scrubbed like a committable id."""
        # `checkpoint_id` IS a normalized failing-check id, the one field here carrying arbitrary
        # text (a parametrized test id can embed a secret), so it is redacted on the way out
        # exactly as `StepRecord.committable` does. This seam covers the sqlite
        # `decision_provenance` column and the OPE export read back off it — and ONLY those.
        # A failing-check id reaches persistence and logs through other paths too
        # (`FailureEvent.persistable`, `StepRecord.committable`, `redact_record`, the rerun-flake
        # log line); each redacts at its own seam, because none of them routes through here.
        out: dict[str, object] = asdict(self)
        out["checkpoint_id"] = redact_secrets(self.checkpoint_id)
        return out


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
    # `dedup_key` is redacted HERE — the single construction seam — so the IN-MEMORY failure log
    # and the PLAINTEXT sqlite snapshot hold the SAME string. Without this the snapshot redacted
    # the key while fresh captures kept it raw, so a restart after a secret-bearing failing-check
    # id (a parametrized test id can embed a credential) never grouped the two halves and
    # recurrence silently stopped. Redaction is deterministic, so the redacted key is still a
    # stable group key — it is "the same failure", just without the secret. `persistable()` keeps
    # its own (now idempotent) pass as belt-and-suspenders for a hand-built event.
    return FailureEvent(
        decision_index=decision_index,
        dedup_key=redact_secrets(failing_check_id),
        exit_code=exit_code if exit_code is not None else DEFAULT_FAILURE_EXIT_CODE,
        success=success,
        confirmed=confirmed,
        blocking=derive_blocking(success, is_infra_failure),
    )


def counts_as_failure(event: FailureEvent, config: EscalationConfig) -> bool:
    """A verified failure counts only if blocking (a capability failure) AND rerun-confirmed."""
    # `blocking` — not the raw exit code — is the gate: the off-wire verifier reports the
    # subprocess code (1/101), so keying on a fixed "blocking" exit code dropped every real
    # failure. Non-blocking (lint/infra) and unconfirmed (flake) drop out; success never counts.
    # `config` is unused; it stays in the signature so callers pass the policy uniformly.
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
    # The count is part of the provenance: a hardcoded suffix reports "x2" under any threshold.
    reason = f"same_verified_failure_x{config.escalate_after_n}"
    if config.ladder == "effort_then_rank" and not at_effort_ceiling:
        return EscalationDirective(EscalationAction.RAISE_EFFORT, reason, new_label_window=True)
    if not at_rank_ceiling:
        return EscalationDirective(EscalationAction.RAISE_RANK, reason, new_label_window=True)
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

    # Owns the failure log AND an abstract ladder position, and BOTH rungs now mirror the live
    # router: append confirmed failures; on a verified success clear the log, reset effort to the
    # bottom rung AND drop rank back to the base position (mirroring the engine popping the task's
    # effort arm and its rank floor, which returns the task to base routing); on any non-HOLD
    # directive retire the acted-on window and step effort, then rank, up to their ceilings. The
    # live engine builds each per-decision directive through `decide` (seeding its own concrete
    # effort/rank position) and applies it concretely; the offline replay drives a whole trajectory
    # through `step`, letting `commit` advance the ladder.
    #
    # THE RANK RUNG IS ENGINE-FAITHFUL, under three geometry conditions. The engine persists the
    # climbed rank per task (its capability-rank floor) and re-serves it on the next decision, so
    # this abstract monotone counter reproduces it rather than bounding it. What the counter cannot
    # express is a HETEROGENEOUS pool, because it carries ONE effort ceiling and ONE bottom rung for
    # every rank: parity therefore requires (i) every model to expose the same number of reasoning
    # arms — a higher-rank model with fewer arms hits its effort ceiling sooner than the counter
    # does — and (ii) each model's default arm to be the bottom of its own ladder, since a success
    # or a rank step resets the abstract effort index to 0 while the engine resets to that default.
    # The third is the rank SHORTLIST: this counter advances rank by exactly +1, whereas the engine
    # walks only the `rank_shortlist` cheapest ranks one at a time and then jumps to the top rank.
    # Parity therefore also requires (iii) a pool of at most `rank_shortlist + 1` ranked models (or
    # `rank_shortlist = 0`), where the two shapes coincide; above that the engine reaches its rank
    # ceiling in fewer rungs than the counter and the streams part there. That is a modelling gap in
    # this counter, not a live bug — the engine's jump is the whole point of the shortlist.
    # `benchmark/escalation/deployability.py` is where a replay's context is checked against the
    # production fields, and `tests/escalation/test_parity.py` pins both the parity and the
    # heterogeneous-ladder boundary against the real engine.

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
        # The rank a verified success falls back to: the caller's starting position IS the base
        # routing pick in the isolation model, which is what the engine returns to when it pops
        # the task's rank floor. Resetting to a hardcoded 0 would be that value only by accident.
        self._base_rank_index = rank_index
        self._state = _LadderState(effort_index, rank_index)

    @property
    def log(self) -> list[FailureEvent]:
        """The current windowed failure log (retired to empty on each escalation)."""
        return self._log

    def observe(self, *, success: bool, event: FailureEvent | None) -> None:
        """Clear the window AND reset the whole ladder on a verified success; else append."""
        if success:
            # Mirror the live engine's success reset precisely: a verified pass pops BOTH the
            # task's effort arm (→ the model's default rung) and its rank floor (→ base routing),
            # because the task is no longer stuck and keeping either would pin it to the expensive
            # rung on the strength of an old failure. Resetting only effort left this counter a
            # full rank-cycle ahead of the engine, so it saturated its ceiling early and emitted
            # HOLD where the engine still had rungs to climb.
            self._log = []
            self._state = _LadderState(0, self._base_rank_index)
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
    "ESCALATION_LADDERS",
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
    "next_rung_rank",
]
