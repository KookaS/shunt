"""Read-only assembly of live router state for the inspection CLI (`shunt escalate`)."""

# STRICTLY READ-ONLY. Every function here opens the store, the registry and the policy the
# server reads, and returns a plain dataclass — nothing writes a row, a snapshot or a config.
# That is not a style preference: the running server holds the escalation state IN MEMORY and
# re-serializes its whole snapshot on its own cadence, so a second process writing
# `router_state` would either be clobbered silently or restore a half-consistent ladder (a rank
# floor without its matching effort arm). Inspection therefore reports; it never mutates.
#
# The decision itself is NOT re-implemented here: `escalation_report` seeds the real
# `decide_escalation` with the real persisted state, so what the CLI prints is what the engine
# would do, not a second copy of the rule that can drift from it.

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shunt.db.loop_health import LoopHealthThresholds, compute_loop_health
from shunt.models.config import strict_yaml_load
from shunt.router.cold_start import ColdStartStrategy
from shunt.router.engine import restore_failure_log, task_state_key
from shunt.router.escalation import (
    EscalationConfig,
    EscalationContext,
    EscalationDirective,
    FailureEvent,
    counts_as_failure,
    decide_escalation,
)

if TYPE_CHECKING:
    from shunt.db.loop_health import LoopHealth
    from shunt.db.store import OutcomeStore
    from shunt.models import ModelPool
    from shunt.router.policy import RouterPolicy

# The escalation knobs reported, in the order a reader reasons about them.
_ESCALATION_KEYS: tuple[str, ...] = (
    "enabled",
    "escalate_after_n",
    "stale_window",
    "ladder",
    "rank_shortlist",
    "exploration_epsilon",
)

_BUILTIN_DEFAULT = "built-in default"
_ABSENT_BLOCK = "no `escalation:` block in the file ⇒ off (an old config never opted in)"


def top_capability_cluster(model_pool: ModelPool) -> set[str]:
    """The expensive-tail ('frontier') set: models above the pool's median capability rank."""
    # Moved here from the proxy so the loop-health composition has ONE definition: the CLI's
    # collapse-alarm read and the server's /admin/loop-health endpoint must agree, and a second
    # copy of "the top half of the price-ranked pool" is exactly the drift that would break that.
    ranked = model_pool.ranked_models()
    start = (len(ranked) + 1) // 2
    return {m.name for m in ranked[start:]}


def loop_health_for(outcome_store: OutcomeStore, model_pool: ModelPool) -> LoopHealth:
    """Compute the full loop-health object from the store — the endpoint, the alarm, the CLI."""
    thresholds = LoopHealthThresholds()
    snapshot = outcome_store.loop_health_snapshot(recent_window=thresholds.recent_window)
    return compute_loop_health(
        snapshot,
        frontier_models=top_capability_cluster(model_pool),
        candidate_models=set(model_pool.model_names()),
        thresholds=thresholds,
    )


@dataclass(frozen=True)
class ConfigItem:
    """One effective escalation knob and where its value came from."""

    key: str
    value: object
    source: str


@dataclass(frozen=True)
class WindowEvent:
    """One persisted failure event and whether it counts toward the recurrence threshold."""

    dedup_key: str
    decision_index: int
    in_window: bool
    counts: bool
    verdict: str


@dataclass(frozen=True)
class WindowKey:
    """One dedup key's tally inside the staleness window."""

    dedup_key: str
    events: int
    countable: int
    due: bool


@dataclass(frozen=True)
class LadderPosition:
    """Where the task currently sits on the two-axis ladder (model rank × reasoning arm)."""

    model: str | None
    model_source: str
    rank_index: int
    max_rank_index: int
    rank_floor: int | None
    effort_arm: str | None
    effort_index: int
    max_effort_index: int


@dataclass(frozen=True)
class EscalationReport:
    """Everything `shunt escalate` prints — assembled read-only from the live state."""

    enabled: bool
    policy_path: str
    config: list[ConfigItem]
    work_dir: str | None
    work_dir_source: str
    mapped_work_dirs: dict[str, str]
    task_key: str | None
    state_present: bool
    decision_index: int
    ladder: LadderPosition
    keys: list[WindowKey]
    events: list[WindowEvent]
    collapse_alarm: bool
    cold_start_active: bool
    suppressed: bool
    next_action: str
    next_reason: str
    exploration_epsilon: float


def escalation_sources(policy_path: Path | None) -> dict[str, str]:
    """Per-knob provenance: the file that set it, or the built-in default."""
    # Read off the RAW yaml rather than the parsed policy: a value equal to the default is still
    # file-sourced if the file wrote it, and the operator needs to know which of the two they are
    # editing. An unreadable/absent file degrades to "every knob is a default", never an error —
    # this is an inspection surface, not a loader.
    defaults = dict.fromkeys(_ESCALATION_KEYS, _BUILTIN_DEFAULT)
    if policy_path is None or not policy_path.exists():
        return defaults
    try:
        data = strict_yaml_load(policy_path.read_text())
    except (OSError, ValueError):
        return defaults
    # An empty or comment-only file parses to None, and `false` to a bool — strict_yaml_load's
    # `-> dict` annotation is a promise about the happy path only. Without this, `shunt escalate`
    # (and, since it shares this reader, `shunt doctor`) raised AttributeError on a config the
    # policy loader accepts as "use the built-in defaults".
    if not isinstance(data, dict):
        return defaults
    section = data.get("router", data)
    if not isinstance(section, dict):
        return defaults
    block = section.get("escalation")
    if not isinstance(block, dict):
        # Mirrors `parse_router_policy`: an absent block is an OLD config, not an opt-in, so
        # escalation is OFF regardless of the code default. Reporting it as a default would send
        # the operator hunting for a knob that is not what turned escalation off.
        return {**defaults, "enabled": _ABSENT_BLOCK}
    return {k: (str(policy_path) if k in block else _BUILTIN_DEFAULT) for k in _ESCALATION_KEYS}


def escalation_config_items(policy: RouterPolicy, policy_path: Path | None) -> list[ConfigItem]:
    """The effective escalation knobs with provenance — the ONE list both CLIs render."""
    # Shared so `shunt escalate` and `shunt doctor` cannot disagree. A second copy of the key
    # tuple in the diagnostics module meant a knob added here was silently absent there, which
    # is a drift bug that reports coverage it does not have rather than failing loudly.
    config = policy.escalation.to_config()
    return [
        ConfigItem(key=key, value=getattr(config, key), source=source)
        for key, source in escalation_sources(policy_path).items()
    ]


@dataclass(frozen=True)
class _AnonymousSession:
    """The identity-less stand-in the CLI resolves a work_dir with (no tool is calling)."""

    tool_identity: str = ""


def resolve_inspection_work_dir(
    policy: RouterPolicy, explicit: str | None, launch_dir: str
) -> tuple[str | None, str]:
    """The repo whose escalation state to inspect, and which layer supplied it."""
    # The VALUE comes from the router's own `WorkDirResolver`, so the CLI can never inspect a
    # different repo than the server keys on. Only the human-readable LABEL is derived here.
    # The per-tool `capture.work_dirs` layer is unreachable from a CLI (there is no tool
    # identity to key it by) — the caller reports the mapping instead of guessing an entry.
    from typing import cast

    from shunt.capture import WorkDirResolver
    from shunt.session import Session

    if explicit:
        return explicit, "--work-dir"
    resolver = WorkDirResolver.from_config(
        work_dir=policy.capture.work_dir,
        work_dirs=policy.capture.work_dirs,
        launch_dir=launch_dir,
        trust_launch_dir=policy.capture.trust_launch_dir,
    )
    resolved = resolver.resolve(cast("Session", _AnonymousSession()))
    if resolved is None:
        return None, "unresolved"
    if os.environ.get("SHUNT_WORK_DIR"):
        return resolved, "SHUNT_WORK_DIR"
    if policy.capture.work_dir:
        return resolved, "capture.work_dir"
    return resolved, "launch directory"


def _verdict(event: FailureEvent) -> str:
    """Why the event does (or does not) count — in `counts_as_failure`'s own branch order."""
    # Kept in lockstep with `counts_as_failure`, whose branches this narrates: a verdict that
    # disagreed with the gate would be worse than no verdict at all.
    if event.success:
        return "ignored: verified success"
    if not event.confirmed:
        return "ignored: unconfirmed on rerun (flake guard)"
    if not event.blocking:
        return "ignored: non-blocking (lint / infra failure)"
    return "counts"


def _window_rows(
    events: list[FailureEvent], decision_index: int, config: EscalationConfig
) -> tuple[list[WindowEvent], list[WindowKey]]:
    """Project the persisted log into per-event verdicts and per-key tallies."""
    rows = [
        WindowEvent(
            dedup_key=e.dedup_key,
            decision_index=e.decision_index,
            in_window=(decision_index - e.decision_index) < config.stale_window,
            counts=counts_as_failure(e, config),
            verdict=_verdict(e),
        )
        for e in events
    ]
    keys: dict[str, list[WindowEvent]] = {}
    for row in rows:
        if row.in_window:
            keys.setdefault(row.dedup_key, []).append(row)
    tallies = [
        WindowKey(
            dedup_key=key,
            events=len(group),
            countable=sum(1 for r in group if r.counts),
            due=sum(1 for r in group if r.counts) >= config.escalate_after_n,
        )
        for key, group in keys.items()
    ]
    tallies.sort(key=lambda t: (-t.countable, t.dedup_key))
    return rows, tallies


def _position_model(
    model_pool: ModelPool, floor: int | None, *, cold_start_active: bool
) -> tuple[str | None, str]:
    """The model the next decision starts from, and how that was resolved."""
    if floor is not None:
        for model in model_pool.models_from_rank(floor):
            if model_pool.is_healthy(model.name):
                return model.name, f"escalation rank floor (rank >= {floor})"
    if cold_start_active:
        return ColdStartStrategy().select(model_pool), "cold-start pick (base routing)"
    ranked = model_pool.ranked_models()
    if ranked:
        # Base routing is memoryless — the kNN pick is only known once the next request
        # arrives — so the ladder is read at the cheapest rank and SAID to be an assumption.
        return ranked[0].name, "assumed cheapest rank (the kNN base pick is not known yet)"
    return None, "no ranked models in the registry"


def _effort_position(
    model_pool: ModelPool, model: str | None, effort_arm: str | None
) -> tuple[int, int, str | None]:
    """(index, max index, arm id) on *model*'s reasoning ladder — 0/0 when it has no arms."""
    entry = model_pool.get_model(model) if model else None
    reasoning = entry.reasoning if entry else None
    if reasoning is None:
        return (0, 0, None)
    ids = [arm.id for arm in sorted(reasoning.arms, key=lambda a: a.rank)]
    # A persisted arm foreign to this model resets to the model's own default, exactly as
    # `RouterEngine._effort_ladder` does — otherwise the CLI reports headroom the engine hasn't.
    arm = effort_arm if effort_arm in ids else reasoning.default_arm
    return (ids.index(arm), len(ids) - 1, arm)


def _ladder_position(
    model_pool: ModelPool,
    rank_floor: int | None,
    effort_arm: str | None,
    *,
    cold_start_active: bool,
) -> LadderPosition:
    """Resolve the task's concrete rung: which model, and which reasoning arm on it."""
    max_rank = len(model_pool.ranked_models()) - 1
    floor = min(rank_floor, max_rank) if rank_floor is not None and max_rank >= 0 else rank_floor
    model, source = _position_model(model_pool, floor, cold_start_active=cold_start_active)
    effort_index, max_effort, arm = _effort_position(model_pool, model, effort_arm)
    rank = model_pool.rank_of(model) if model else None
    return LadderPosition(
        model=model,
        model_source=source,
        rank_index=rank if rank is not None else 0,
        max_rank_index=max_rank,
        rank_floor=floor,
        effort_arm=arm,
        effort_index=effort_index,
        max_effort_index=max_effort,
    )


def _task_state(state: dict[str, Any], field: str, task_key: str | None) -> object:
    """One task's entry out of a persisted escalation sub-map, or None."""
    raw = state.get(field)
    if not isinstance(raw, dict) or task_key is None:
        return None
    return raw.get(task_key)


def _task_events(state: dict[str, Any], task_key: str | None) -> list[FailureEvent]:
    """The task's persisted failure log, decoded by the engine's own codec."""
    raw = _task_state(state, "failure_log", task_key)
    if raw is None or task_key is None:
        return []
    return restore_failure_log({task_key: raw}).get(task_key, [])


def _as_str(value: object) -> str | None:
    """A persisted arm id, or None when the snapshot holds something else."""
    return value if isinstance(value, str) else None


def _as_int(value: object, default: int | None = None) -> int | None:
    """Coerce a persisted counter, degrading to *default* rather than raising."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _cold_start_active(outcome_store: OutcomeStore) -> bool:
    """Whether cold-start routing is still active (it suppresses the collapse alarm)."""
    from shunt.db.outcome_index import OutcomeIndexAdapter

    index = OutcomeIndexAdapter(outcome_store)
    return ColdStartStrategy().is_active_effective(
        index.effective_labeled(), index.effective_tier2()
    )


def _next_directive(
    events: list[FailureEvent],
    decision_index: int,
    ladder: LadderPosition,
    config: EscalationConfig,
    *,
    suppressed: bool,
) -> EscalationDirective:
    """Run the REAL `decide_escalation` against the real state — never a second copy of it."""
    # Epsilon is zeroed for the PREDICTION only: an epsilon-greedy draw taken here would consume
    # a stream this process does not own and report an arm the server never drew. The configured
    # epsilon is reported separately, as the probability the flagged arm is instead held.
    ctx = EscalationContext(
        current_rank_index=ladder.rank_index,
        max_rank_index=ladder.max_rank_index,
        current_effort_index=ladder.effort_index,
        max_effort_index=ladder.max_effort_index,
        loop_health_alarm=suppressed,
    )
    return decide_escalation(events, decision_index, ctx, replace(config, exploration_epsilon=0.0))


def escalation_report(
    *,
    policy: RouterPolicy,
    policy_path: Path | None,
    model_pool: ModelPool,
    outcome_store: OutcomeStore,
    work_dir: str | None,
    work_dir_source: str,
) -> EscalationReport:
    """Assemble the escalation inspection report for *work_dir* — read-only."""
    config = policy.escalation.to_config()
    task_key = task_state_key(work_dir) if work_dir else None
    state = outcome_store.load_escalation_state() or {}
    events = _task_events(state, task_key)
    counter = _task_state(state, "decision_index", task_key)
    decision_index = _as_int(counter, 0) or 0
    cold_start = _cold_start_active(outcome_store)
    alarm = loop_health_for(outcome_store, model_pool).routing_collapse.alarm
    suppressed = alarm and not cold_start
    ladder = _ladder_position(
        model_pool,
        _as_int(_task_state(state, "rank_floor", task_key)),
        _as_str(_task_state(state, "effort_arm", task_key)),
        cold_start_active=cold_start,
    )
    rows, tallies = _window_rows(events, decision_index, config)
    directive = _next_directive(events, decision_index, ladder, config, suppressed=suppressed)
    return EscalationReport(
        enabled=config.enabled,
        policy_path=str(policy_path) if policy_path else "built-in defaults (no router.yaml)",
        config=escalation_config_items(policy, policy_path),
        work_dir=work_dir,
        work_dir_source=work_dir_source,
        mapped_work_dirs=dict(policy.capture.work_dirs),
        task_key=task_key,
        state_present=counter is not None,
        decision_index=decision_index,
        ladder=ladder,
        keys=tallies,
        events=rows,
        collapse_alarm=alarm,
        cold_start_active=cold_start,
        suppressed=suppressed,
        next_action=str(directive.action),
        next_reason=directive.reason,
        exploration_epsilon=config.exploration_epsilon,
    )


__all__ = [
    "ConfigItem",
    "EscalationReport",
    "LadderPosition",
    "WindowEvent",
    "WindowKey",
    "escalation_config_items",
    "escalation_report",
    "escalation_sources",
    "resolve_inspection_work_dir",
    "loop_health_for",
    "top_capability_cluster",
]
