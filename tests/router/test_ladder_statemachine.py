"""Property-based proof of the escalation ladder's MULTI-HOP sequencing — zero model calls."""

# The ladder is a pure state machine over (rank, effort, failure window), so its LONG sequences
# are provable for free. Hand-written cases cover the one-hop cases; the bugs live in the long
# ones — restart, arm exhaustion, the ceiling, two tasks interleaved — and Hypothesis shrinks
# any failure it finds to a minimal repro. Fakes only (no I/O, no provider, no embedder).

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from types import MappingProxyType
from typing import Any, Final

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from shunt.router.engine import RouterEngine, task_state_key
from shunt.router.escalation import (
    EscalationAction,
    EscalationConfig,
    EscalationContext,
    decide_escalation,
    failure_event_from_outcome,
)
from shunt.router.strategies.base import RoutingStrategy
from shunt.router.strategies.fixed import AlwaysCheapStrategy
from tests.router.escalation_fakes import (
    EchoSessionManager,
    Embedder,
    Index,
    RankedReasoningPool,
    reasoning_ladder,
)

_AFTER_N = 2  # the shipped threshold; the escalation reason string carries it

# Weakest -> strongest, each model carrying its OWN arm vocabulary. Distinct vocabularies are
# the real registry's shape, and they force every rank step to reset a FOREIGN arm rather than
# coincidentally finding the same id on the next model.
_LADDERS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "qwen": reasoning_ladder("low", "high"),
        "glm": reasoning_ladder("nothink", "think"),
        "opus": reasoning_ladder("medium", "max"),
    }
)
_RANK_ORDER = tuple(_LADDERS)
_TOP_RANK = len(_RANK_ORDER) - 1
_TOP_ARM = "max"  # the strongest model's top effort rung: past it there is nothing on either axis

# The state machine runs a SHORTER shortlist than the shipped 3 on purpose: with three ranked
# models a shortlist of 3 walks every rank, so the skip — the behaviour these invariants now have
# to bound — would never be generated. At 2, rank 0 is the last individually-walked rung and the
# next one jumps to the top, which is the shape a real (much wider) pool takes.
_MACHINE_SHORTLIST = 2

_TASKS = ("repoA", "repoB")
# Two keys, not many: the trigger is RECURRENCE of one key, and a wide key alphabet
# makes a recurrence so rare that the generator never reaches the second rung.
_DEDUP_KEYS = ("check_a", "check_b")
# Effort rung + rank rung per model, plus one decision past the top to land on the ceiling.
_MAX_RUNGS = 2 * len(_RANK_ORDER)


def _engine(  # noqa: PLR0913 (test factory: one arg per fake the ladder is driven over)
    *,
    strategy: RoutingStrategy | None = None,
    trust_neighbors: bool = True,
    cold: bool = False,
    ladders: Mapping[str, Any] | None = None,
    shortlist: int = 0,
    unhealthy: set[str] | None = None,
) -> RouterEngine:
    """A router over the three-rung pool, keyed so ONE engine drives both tasks' ladders."""
    # shortlist=0 (every rank is a rung) is the DEFAULT here, not the shipped 3: these fixtures
    # pin the rung-by-rung sequence over a 3-model pool, which is what a shipped shortlist of 3
    # also produces there. Stating it explicitly keeps them reading the ladder they assert on.
    return RouterEngine(
        model_pool=RankedReasoningPool(
            dict(ladders if ladders is not None else _LADDERS), unhealthy=unhealthy
        ),
        session_manager=EchoSessionManager(),
        outcome_index=_ColdIndex() if cold else Index(),
        embedder=Embedder(),
        strategy=strategy,
        trust_neighbors=trust_neighbors,
        escalation=EscalationConfig(
            enabled=True, escalate_after_n=_AFTER_N, rank_shortlist=shortlist
        ),
        # The session id carries its task ("repoA|7"), so two ladders run inside one engine.
        # Without that, cross-task leakage is not observable at all.
        task_key_resolver=lambda session: str(session.tool_identity).split("|", 1)[0],
    )


class _ColdIndex(Index):
    """Cold-start ACTIVE: too few effective verified outcomes for the router to be choosing."""

    def effective_labeled(self) -> float:
        return 1.0

    def effective_tier2(self) -> float:
        return 0.0


def _record(
    engine: RouterEngine,
    task: str,
    dedup_key: str,
    *,
    is_infra: bool = False,
    confirmed: bool = True,
) -> None:
    """One verified outcome at a boundary. exit_code=1 is a real pytest red, not the hook's 2."""
    engine.record_outcome(
        downshift=False,
        success=False,
        task_key=task,
        dedup_key=dedup_key,
        exit_code=1,
        is_infra_failure=is_infra,
        confirmed=confirmed,
    )


def _escalation_cycle(
    engine: RouterEngine, task: str, session: int
) -> tuple[str, str, dict[str, Any]]:
    """`_AFTER_N` same-key verified failures, then the decision that acts on them."""
    for _ in range(_AFTER_N):
        _record(engine, task, "check_a")
    return engine.decide(f"{task}|{session}", "task")


def _next_rung(from_rank: int) -> int:
    """The rank the machine's ladder may aim at from *from_rank* — the shortlist shape, restated.

    Independent of the engine's own helper on purpose: an invariant that calls the code under
    test proves only self-consistency.
    """
    if from_rank + 1 < _MACHINE_SHORTLIST:
        return from_rank + 1
    return max(from_rank + 1, _TOP_RANK)


class EscalationLadderMachine(RuleBasedStateMachine):
    """Drives one engine's ladder over random interleavings of sessions and verified outcomes."""

    def __init__(self) -> None:
        super().__init__()
        self.engine = _engine(shortlist=_MACHINE_SHORTLIST)
        self.sessions = 0
        # Last rank index SERVED per task — the rung the cascade currently stands on.
        self.rank = dict.fromkeys(_TASKS, 0)
        # Whether the task owns an outcome `counts_as_failure` accepts. Only such an outcome may
        # move a ladder, and only its OWN.
        self.countable = dict.fromkeys(_TASKS, False)

    def _decide(self, task: str) -> tuple[str, str]:
        state_key = task_state_key(task)
        # Read the ceiling BEFORE deciding: at the top rank on the top arm there is no rung left
        # on either axis, so this decision must hold rather than thrash between models.
        at_ceiling = (
            self.rank[task] == _TOP_RANK
            and self.engine.snapshot_escalation_state()["effort_arm"].get(state_key) == _TOP_ARM
        )
        self.sessions += 1
        model, reason, _prov = self.engine.decide(f"{task}|{self.sessions}", "task")
        served = _RANK_ORDER.index(model)
        previous = self.rank[task]
        # Monotone climb, at most ONE RUNG per decision — where a rung is what the shortlist
        # ladder defines it to be, not a single rank. The old form of this invariant was
        # `served <= previous + 1`; it forbade the deliberate jump the shortlist exists to make,
        # so it is replaced by the ladder's own reachability bound rather than dropped. It still
        # forbids an UNBOUNDED jump: the ceiling is the shortlist's next rung from `previous`,
        # so no decision may overshoot the frontier, land on a rank the ladder never aims at, or
        # (the lower bound) fall back below the rung the per-task rank floor already earned.
        # The lower bound holds only because that floor persists ACROSS sessions — base routing
        # is memoryless and re-picks the cheap model every time.
        ceiling = min(_next_rung(previous), _TOP_RANK)
        assert previous <= served <= ceiling, f"{task}: rank {previous} -> {served}"
        if at_ceiling:
            assert served == _TOP_RANK, f"{task}: fell off the ceiling to rank {served}"
            assert reason != "auto_escalation", "escalated past the ceiling"
        self.rank[task] = served
        return (model, reason)

    @rule(task=st.sampled_from(_TASKS))
    def start_session(self, task: str) -> None:
        """Route one session for *task* — the only place the ladder is read."""
        self._decide(task)

    @rule(task=st.sampled_from(_TASKS), dedup_key=st.sampled_from(_DEDUP_KEYS))
    def close_with_verified_failure(self, task: str, dedup_key: str) -> None:
        """A confirmed, non-infra red — the one outcome that counts toward a rung."""
        _record(self.engine, task, dedup_key)
        self.countable[task] = True

    @rule(
        task=st.sampled_from(_TASKS),
        dedup_key=st.sampled_from(_DEDUP_KEYS),
        rungs=st.integers(min_value=1, max_value=_MAX_RUNGS),
    )
    def climb_rungs(self, task: str, dedup_key: str, rungs: int) -> None:
        """Drive *task* up *rungs* rungs — a recurrence plus the decision acting on it, N times."""
        # Whole rungs, not single failures: assembled out of independent random steps a recurrence
        # is rare and five consecutive ones are vanishingly so, which would leave the ceiling — the
        # state this machine exists to reach — unvisited by every generated sequence.
        for _ in range(rungs):
            for _ in range(_AFTER_N):
                _record(self.engine, task, dedup_key)
            self.countable[task] = True
            self._decide(task)

    @rule(task=st.sampled_from(_TASKS), dedup_key=st.sampled_from(_DEDUP_KEYS))
    def close_with_infra_failure(self, task: str, dedup_key: str) -> None:
        """An env/collection red is not a capability failure, so it must never count."""
        _record(self.engine, task, dedup_key, is_infra=True)

    @rule(task=st.sampled_from(_TASKS), dedup_key=st.sampled_from(_DEDUP_KEYS))
    def close_with_unconfirmed_failure(self, task: str, dedup_key: str) -> None:
        """A pass->fail->pass flake on unchanged state is not a failure either."""
        _record(self.engine, task, dedup_key, confirmed=False)

    @rule(task=st.sampled_from(_TASKS))
    def close_with_verified_pass(self, task: str) -> None:
        """A green suite retires the WHOLE ladder: failure window, effort arm and rank floor."""
        self.engine.record_outcome(
            downshift=False, success=True, task_key=task, dedup_key=None, exit_code=0
        )
        self.rank[task] = 0
        self.countable[task] = False
        state = self.engine.snapshot_escalation_state()
        key = task_state_key(task)
        assert key not in state["rank_floor"]
        assert key not in state["effort_arm"]
        assert key not in state["failure_log"]
        # ...and the next decision is the base pick again, not the rung it had been stuck on.
        # Keeping the floor past a pass would pin the task to the expensive model forever.
        model, _reason = self._decide(task)
        assert model == _RANK_ORDER[0]

    @rule()
    def restart_engine(self) -> None:
        """Snapshot -> fresh engine -> restore: the rung must survive a process restart."""
        snapshot = self.engine.snapshot_escalation_state()
        fresh = _engine(shortlist=_MACHINE_SHORTLIST)
        fresh.restore_escalation_state(snapshot)
        # Round-trip equality covers all four maps at once — rank floor, effort arm, failure log
        # and decision index. Dropping any one of them silently un-climbs the ladder on restart.
        assert fresh.snapshot_escalation_state() == snapshot
        self.engine = fresh

    @rule()
    def interleave_second_task(self) -> None:
        """Both tasks decided back to back — the adjacency that would alias a shared counter."""
        for task in _TASKS:
            self._decide(task)

    @invariant()
    def no_cross_task_leakage(self) -> None:
        """A task owns no rung without a countable failure of its own — no other task lends one."""
        state = self.engine.snapshot_escalation_state()
        for task in _TASKS:
            if self.countable[task]:
                continue
            key = task_state_key(task)
            assert key not in state["rank_floor"], f"{task} gained a floor it never earned"
            assert key not in state["effort_arm"], f"{task} gained an arm it never earned"


TestEscalationLadder = EscalationLadderMachine.TestCase
# Bounded on purpose. These steps are pure in-process dict work, so this budget explores a few
# thousand transitions in a few seconds — deep enough that generated sequences do reach the
# ceiling (verified by instrumenting the branch), without adding meaningfully to the suite.
TestEscalationLadder.settings = settings(
    max_examples=60,
    stateful_step_count=30,
    deadline=timedelta(seconds=2),
)


def test_multi_hop_the_ladder_climbs_cheap_to_mid_to_high() -> None:
    """THE headline property: recurring verified failures reach the TOP rank, not just rung one."""
    engine = _engine()
    steps = [_escalation_cycle(engine, "repoA", i) for i in range(6)]
    served = [model for model, _reason, _prov in steps]
    # effort (qwen low->high), rank (->glm), effort (glm), rank (->opus), effort (opus), ceiling.
    assert served == ["qwen", "glm", "glm", "opus", "opus", "opus"]
    reasons = [reason for _model, reason, _prov in steps]
    assert reasons[:5] == ["auto_escalation"] * 5
    # The last cycle has nothing left to climb: the task is HELD at the rung it already owns.
    assert reasons[5] == "escalation_floor"
    floor = engine.snapshot_escalation_state()["rank_floor"][task_state_key("repoA")]
    assert floor == _TOP_RANK


def test_effort_is_spent_before_a_rank_step() -> None:
    """Cache-safety: the first rung is the SAME model at a higher arm, never a model switch."""
    engine = _engine()
    model, _reason, prov = _escalation_cycle(engine, "repoA", 1)
    assert model == "qwen"  # same model ⇒ the prompt-cache namespace is untouched
    assert prov["escalated_reasoning_arm"] == "high"
    model2, _reason2, prov2 = _escalation_cycle(engine, "repoA", 2)
    assert model2 == "glm"  # a rank step only once the arms are exhausted
    assert "escalated_reasoning_arm" not in prov2  # a rank step carries no reasoning arm


def test_the_ceiling_is_terminal_and_never_thrashes() -> None:
    """Past the top rank on the top arm, more failures change nothing — forever."""
    engine = _engine()
    for i in range(6):
        _escalation_cycle(engine, "repoA", i)
    for i in range(6, 20):
        model, reason, prov = _escalation_cycle(engine, "repoA", i)
        assert model == "opus"
        assert reason == "escalation_floor"
        assert "escalated_reasoning_arm" not in prov


def test_at_both_ceilings_the_directive_is_hold_with_the_ceiling_reason() -> None:
    """The pure decision behind the live behaviour above, asserted on its own reason token."""
    events = [
        failure_event_from_outcome(
            decision_index=0,
            failing_check_id="check_a",
            exit_code=1,
            success=False,
            is_infra_failure=False,
            confirmed=True,
        )
        for _ in range(_AFTER_N)
    ]
    ctx = EscalationContext(
        current_rank_index=_TOP_RANK,
        max_rank_index=_TOP_RANK,
        current_effort_index=1,
        max_effort_index=1,
    )
    directive = decide_escalation(events, 0, ctx, EscalationConfig(escalate_after_n=_AFTER_N))
    assert directive.action is EscalationAction.HOLD
    assert directive.reason == "escalation_ceiling"


def test_a_restart_preserves_the_rung_the_task_climbed_to() -> None:
    """Snapshot/restore keeps the floor, the arm, the window and the counter."""
    engine = _engine()
    for i in range(3):  # qwen effort, rank -> glm, glm effort
        _escalation_cycle(engine, "repoA", i)
    snapshot = engine.snapshot_escalation_state()
    restarted = _engine()
    restarted.restore_escalation_state(snapshot)
    assert restarted.snapshot_escalation_state() == snapshot
    model, reason, _prov = restarted.decide("repoA|9", "task")
    assert model == "glm"  # the floor survived — not back to the cheap model
    assert reason == "escalation_floor"


def test_a_verified_pass_retires_the_whole_ladder() -> None:
    """A green suite is the signal the task is no longer stuck: floor and arm both drop."""
    engine = _engine()
    for i in range(3):
        _escalation_cycle(engine, "repoA", i)
    assert engine.decide("repoA|9", "task")[0] == "glm"
    engine.record_outcome(
        downshift=False, success=True, task_key="repoA", dedup_key=None, exit_code=0
    )
    model, reason, _prov = engine.decide("repoA|10", "task")
    assert model == "qwen"  # the base pick again
    assert reason != "escalation_floor"
    state = engine.snapshot_escalation_state()
    assert task_state_key("repoA") not in state["rank_floor"]
    assert task_state_key("repoA") not in state["effort_arm"]


def test_one_tasks_cascade_never_moves_another_tasks_ladder() -> None:
    """Ladders are per task_key: repoA climbing to the frontier leaves repoB on the cheap model."""
    engine = _engine()
    for i in range(6):
        _escalation_cycle(engine, "repoA", i)
    model, reason, _prov = engine.decide("repoB|1", "task")
    assert model == "qwen"
    assert reason != "auto_escalation"


def test_distinct_dedup_keys_never_aggregate_into_a_rung() -> None:
    """Many different failing checks is a HARD TASK (the kNN store's job), not an escalation."""
    engine = _engine()
    for key in ("check_a", "check_b", "check_c", "check_d", "check_e"):
        _record(engine, "repoA", key)
    model, reason, _prov = engine.decide("repoA|1", "task")
    assert model == "qwen"
    assert reason != "auto_escalation"


def test_infra_and_flaky_reds_never_count_toward_a_rung() -> None:
    """Neither an env red nor an unconfirmed one is a capability failure, at any volume."""
    engine = _engine()
    for _ in range(_AFTER_N * 3):
        _record(engine, "repoA", "check_a", is_infra=True)
        _record(engine, "repoA", "check_a", confirmed=False)
    model, reason, _prov = engine.decide("repoA|1", "task")
    assert model == "qwen"
    assert reason != "auto_escalation"


def test_a_stale_embedding_space_still_reaches_the_ladder() -> None:
    """A corpus awaiting `shunt reindex` is kNN DEGRADED, not a pinned control."""
    # Suppressing escalation here left the router stuck on the cheap model for exactly as long
    # as the reindex was outstanding, even though the verified-failure signal was unaffected.
    engine = _engine(trust_neighbors=False)
    model, reason, prov = _escalation_cycle(engine, "repoA", 1)
    assert reason == "auto_escalation"
    assert prov["rank_escalation_reason"] == f"same_verified_failure_x{_AFTER_N}"
    assert model == "qwen"  # the cache-safe effort rung, same model


def test_cold_start_still_reaches_the_ladder() -> None:
    """Cold start is where verified failures FIRST accumulate — it must not be a dead branch."""
    engine = _engine(cold=True)
    _model, reason, prov = _escalation_cycle(engine, "repoA", 1)
    assert reason == "auto_escalation"
    assert prov["rank_escalation_reason"] == f"same_verified_failure_x{_AFTER_N}"


# --- the rank shortlist: the ladder jumps toward the frontier instead of buying every rung ---

# Six ranked models — wide enough that "every rung" and "shortlist then jump" are different
# ladders. Rank order is price order in the live pool, so ranks 3 and 4 are models the ladder
# would otherwise BUY on its way to a frontier the task was going to need anyway.
_WIDE: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "m0": reasoning_ladder("low", "high"),
        "m1": reasoning_ladder("nothink", "think"),
        "m2": reasoning_ladder("small", "big"),
        "m3": reasoning_ladder("draft", "final"),
        "m4": reasoning_ladder("fast", "slow"),
        "m5": reasoning_ladder("medium", "max"),
    }
)
_WIDE_TOP = len(_WIDE) - 1
_SHIPPED_SHORTLIST = 3


def _served_ranks(engine: RouterEngine, cycles: int, task: str = "repoA") -> list[int]:
    """The rank index served by each of *cycles* escalation cycles."""
    order = list(_WIDE)
    return [order.index(_escalation_cycle(engine, task, i)[0]) for i in range(cycles)]


def test_the_shortlist_reaches_the_frontier_in_fewer_rungs_than_the_pool_has_models() -> None:
    """THE point of the shortlist: 3 rank rungs to the frontier of a 6-model pool, not 5."""
    engine = _engine(ladders=_WIDE, shortlist=_SHIPPED_SHORTLIST)
    served = _served_ranks(engine, 10)
    # Each rank spends its own effort arm before the rank steps, so ranks come in pairs.
    assert served == [0, 1, 1, 2, 2, 5, 5, 5, 5, 5]
    rank_rungs = sum(1 for prev, cur in zip(served, served[1:], strict=False) if cur > prev)
    assert rank_rungs == 3
    assert rank_rungs < len(_WIDE) - 1  # the every-rung ladder would have paid 5
    assert 3 not in served and 4 not in served  # the skipped mid-tier is never billed


def test_a_jump_never_overshoots_the_frontier_and_never_falls_below_the_floor() -> None:
    """The skip is BOUNDED: monotone, capped at the top rank, and capped at the ladder's rung."""
    engine = _engine(ladders=_WIDE, shortlist=_SHIPPED_SHORTLIST)
    served = _served_ranks(engine, 20)
    assert max(served) == _WIDE_TOP  # it reaches the frontier...
    for prev, cur in zip(served, served[1:], strict=False):
        assert cur >= prev, f"the ladder walked backwards: {prev} -> {cur}"  # ...never below
        ceiling = prev + 1 if prev + 1 < _SHIPPED_SHORTLIST else _WIDE_TOP
        assert cur <= ceiling, f"unbounded jump: {prev} -> {cur}"  # ...and never past it


def test_rank_shortlist_zero_restores_the_every_rank_walk() -> None:
    """The escape hatch is exact: 0 reproduces the pre-shortlist ladder rung for rung."""
    engine = _engine(ladders=_WIDE, shortlist=0)
    assert _served_ranks(engine, 10) == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]


def test_an_unhealthy_jump_target_degrades_to_the_most_capable_healthy_model() -> None:
    """A dead frontier degrades to the best available rung — it does not re-walk the mid-tier."""
    # The jump from rank 2 aims at the top rank; with it unhealthy the ladder takes the most
    # capable healthy model above the current one (rank 4), NOT the rank-3 model one rung up,
    # which is the silent every-rung fallback the shortlist exists to avoid.
    engine = _engine(ladders=_WIDE, shortlist=_SHIPPED_SHORTLIST, unhealthy={"m5"})
    served = _served_ranks(engine, 6)
    assert served == [0, 1, 1, 2, 2, 4]


def test_a_jump_still_resets_the_effort_arm_to_the_new_models_default() -> None:
    """A skipped rank is still a rank step: the landed model climbs its OWN ladder from the base."""
    engine = _engine(ladders=_WIDE, shortlist=_SHIPPED_SHORTLIST)
    for i in range(5):  # m0 effort, ->m1, m1 effort, ->m2, m2 effort
        _escalation_cycle(engine, "repoA", i)
    model, _reason, prov = _escalation_cycle(engine, "repoA", 5)  # the jump 2 -> 5
    assert model == "m5"
    assert "escalated_reasoning_arm" not in prov  # a rank step carries no arm...
    model2, _reason2, prov2 = _escalation_cycle(engine, "repoA", 6)
    assert model2 == "m5"
    assert prov2["escalated_reasoning_arm"] == "max"  # ...and m5 climbs from its own default


def test_a_fixed_strategy_is_a_pinned_control_and_never_escalates() -> None:
    """always_cheap is a control BY CONTRACT: no verified-failure signal may move it."""
    # `_decide_fixed` deliberately bypasses `_finalize_decision`. A control that escalates is
    # no longer a control, and the routing comparison it anchors becomes unreadable.
    engine = _engine(strategy=AlwaysCheapStrategy())
    for i in range(6):
        model, reason, prov = _escalation_cycle(engine, "repoA", i)
        assert model == "qwen"
        assert reason == "always_cheap"
        # `rank_escalation_reason` is always present in a provenance record; None is the proof
        # that no rung was applied (a delivered rung writes the recurrence token here).
        assert prov["rank_escalation_reason"] is None
        assert prov.get("auto_escalated") is not True
