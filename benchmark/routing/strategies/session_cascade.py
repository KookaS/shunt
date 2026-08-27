"""Session-cadence cascade: the LIVE escalation ladder replayed over the committed matrix.

One decision per session, effort rung then rank rung, the climbed rank persisting to the
next session, every attempt billed — the cache-safe analogue of the offline cascades.
"""

# WHY THIS EXISTS. The two offline cascades (Price-Cascade, kNN-semantic-cascade) retry
# WITHIN a task
# because in SWE-bench a task IS one attempt, so they take a verified outcome per attempt
# mid-session — more than one decision per session, which breaks cache-safety. That is the whole
# of their blocker. Live, a SESSION is the attempt: the cheap model runs, the suite goes red at
# session close, the NEXT session opens one rung higher. The same ladder paced at session cadence
# is cache-safe by construction, and until this module existed no benchmark strategy modelled it,
# so the value of the shipped ladder was unmeasured offline.
#
# WHAT IS SHARED WITH THE LIVE ENGINE, AND WHAT IS NOT. The DECISION is
# `shunt.router.escalation.EscalationRunner` — the same object `RouterEngine._maybe_escalate`
# drives — so the recurrence window, the counting rule and the effort-then-rank ladder cannot
# drift from production. What this module owns is the CONCRETE APPLICATION (which arm id, which
# model, the per-task rank floor), which is exactly the split the engine itself uses: the engine
# builds a runner per decision seeded with its own concrete position, then applies the directive
# against its live pool. A heterogeneous pool is why the runner's own abstract counter is not
# used as the ladder position (its docstring names the two geometry conditions this pool breaks:
# models here carry 1-3 reasoning arms, and three of them default to a rung that is not the
# bottom of their own ladder).
#
# The rank rung's SHAPE is shared too: `escalation.next_rung_rank` decides which rank a rank rung
# aims at, so the shortlist jump the engine takes is the jump replayed here. `rank_shortlist=0`
# restores the every-rank walk this module implemented before the knob existed.
#
# THE ONE MODELLING ASSUMPTION, STATED UP FRONT. The live counter escalates only on RECURRENCE of
# the same normalized failing-check id. results.csv records a per-cell pass/fail and no
# failing-check identity, so this replay assigns one stable key per task: every failure of a task
# recurs identically. That is the assumption MOST favourable to escalation — it makes the ladder
# climb as fast as the policy ever could — so a cost this strategy reports is a LOWER bound on
# what the live ladder spends, never an optimistic one.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from shunt.router.escalation import (
    EscalationAction,
    EscalationConfig,
    EscalationDirective,
    EscalationRunner,
    FailureEvent,
    failure_event_from_outcome,
    next_rung_rank,
)

from . import BilledAttempt, Strategy
from ._cascade_common import (
    billed_attempt,
    cheapest_priced_model,
    measured_models_by_price,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# A hard bound on the replay so a degenerate ladder cannot spin. The natural stop is the
# ceiling directive (see `_run_task`); this only caps pathological configurations.
_MAX_SESSIONS: int = 60

# The reason `_ladder_action` stamps when neither axis has headroom left. Matched rather than
# re-derived: it is the product's own end-of-ladder signal.
_CEILING: str = "escalation_ceiling"

# The ladder a session-cadence row runs when nothing names one. Declared here, beside the
# constructor default it feeds, because `run_eval` has to gate the row on the ladder it will
# ACTUALLY run: a caller that restated a different fallback would ask the positive control to
# certify one ladder while the strategy replayed another.
DEFAULT_LADDER: str = "effort_then_rank"


@dataclass(frozen=True)
class ArmLadder:
    """One model's reasoning ladder: arm ids by ascending rank, plus its default rung."""

    arms: tuple[str, ...]
    default_arm: str

    @property
    def max_index(self) -> int:
        """Index of the top arm (0 when the model exposes a single arm)."""
        return len(self.arms) - 1

    def index_of(self, arm: str) -> int:
        """Position of *arm*, or the default's position when the id is foreign to this model."""
        # Mirrors `RouterEngine._effort_ladder`: a persisted arm foreign to the served model is
        # reset to THAT model's default rather than reported as headroom it does not have.
        return self.arms.index(arm) if arm in self.arms else self.arms.index(self.default_arm)

    def above(self, arm: str) -> str | None:
        """The next arm above *arm*, or None at the ceiling."""
        i = self.index_of(arm)
        return self.arms[i + 1] if i < self.max_index else None


def default_arm_ladders() -> dict[str, ArmLadder]:
    """Each registry model's reasoning ladder, read from the shipped config (no arms ⇒ absent)."""
    from benchmark import config

    ladders: dict[str, ArmLadder] = {}
    for model, reasoning in config.reasoning_configs().items():
        if reasoning is None or not reasoning.arms:
            continue
        arms = tuple(a.id for a in sorted(reasoning.arms, key=lambda a: a.rank))
        ladders[model] = ArmLadder(arms=arms, default_arm=reasoning.default_arm)
    return ladders


@dataclass
class _Trace:
    """What one task's replay did — the instrumentation the hop-depth report is read off."""

    path: list[tuple[str, str]]
    cost: float
    scorable: bool
    passed: bool
    unmeasured: int
    # Per-session (model, cost), in billing order. The session ladder re-serves its rank floor
    # across consecutive sessions, so this sequence — not the collapsed total — is what the
    # cache-aware cost model needs to see.
    attempts: list[BilledAttempt] = field(default_factory=list)

    @property
    def hops(self) -> int:
        """Distinct ladder rungs occupied (1 = never escalated)."""
        return len(dict.fromkeys(self.path))


class SessionCascadeStrategy(Strategy):
    """Replay the shipped escalation ladder at session cadence: one decision per session,
    effort rung before rank rung, rank floor persisting, every attempt billed.
    """

    def __init__(
        self,
        escalate_after_n: int = 2,
        stale_window: int = 10,
        ladder: str = DEFAULT_LADDER,
        max_sessions: int = _MAX_SESSIONS,
        arm_results: Mapping[str, Mapping[str, Mapping[str, dict]]] | None = None,
        arm_ladders: Mapping[str, ArmLadder] | None = None,
        rank_shortlist: int | None = None,
    ) -> None:
        # The policy object IS the product's, so an out-of-range knob raises here exactly as it
        # would at boot, and the ladder vocabulary cannot drift from `ESCALATION_LADDERS`.
        # `rank_shortlist=None` means "whatever the PRODUCT dataclass defaults to" — the shipped
        # value is never restated here, so a change to the default cannot leave this replay
        # measuring the old ladder. Pass an explicit int only to sweep the knob.
        shipped = EscalationConfig(
            enabled=True,
            escalate_after_n=escalate_after_n,
            stale_window=stale_window,
            ladder=ladder,
        )
        self._config = (
            shipped if rank_shortlist is None else replace(shipped, rank_shortlist=rank_shortlist)
        )
        self._max_sessions = max_sessions
        self._arm_results = arm_results
        self._arm_ladders = arm_ladders

        # Cascade metadata (reset per :meth:`select` call), read by summary.evaluate so the
        # reported cost is every billed session, not just the returned model's cell.
        self.cascade_total_cost: float = 0.0
        self.cascade_tried_models: list[str] = []
        self.cascade_attempts: list[BilledAttempt] = []
        self.cascade_scorable: bool = True
        # Instrumentation — proves the escalation path is actually exercised rather than assumed.
        self.session_path: list[tuple[str, str]] = []
        self.session_hops: int = 0
        self.session_unmeasured: int = 0

    @property
    def name(self) -> str:
        """Display name shown in eval output and plots."""
        return "Session-Cascade"

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        """Replay the task's sessions and return the model the last session routed to."""
        del task_meta
        rungs = measured_models_by_price(matrix)
        if not rungs:
            self._publish(_Trace(path=[], cost=0.0, scorable=False, passed=False, unmeasured=0))
            return cheapest_priced_model(matrix)
        floor = self._initial_rank_floor(task_id, matrix, rungs)
        trace = self._run_task(task_id, matrix, rungs, floor)
        self._publish(trace)
        return trace.path[-1][0] if trace.path else rungs[floor]

    def trace(self, task_id: str, matrix: dict) -> _Trace:
        """The full replay record for one task — the reporting entry point."""
        rungs = measured_models_by_price(matrix)
        if not rungs:
            return _Trace(path=[], cost=0.0, scorable=False, passed=False, unmeasured=0)
        return self._run_task(
            task_id, matrix, rungs, self._initial_rank_floor(task_id, matrix, rungs)
        )

    def _initial_rank_floor(self, task_id: str, matrix: dict, rungs: Sequence[str]) -> int:
        """The rung the FIRST session opens on — 0 (cheapest) for the always_cheap base pick."""
        # The seam a kNN-seeded cascade overrides: everything below this line is the ladder, and
        # the ladder must not differ between the two rows, so the only thing a subclass changes
        # is where the climb starts.
        del task_id, matrix, rungs
        return 0

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------
    def _run_task(
        self,
        task_id: str,
        matrix: dict,
        rungs: Sequence[str],
        initial_rank_floor: int = 0,
    ) -> _Trace:
        """One task's whole session sequence, driven by the shared escalation runner."""
        ladders = self._ladders()
        log: list[FailureEvent] = []
        rank_floor = max(0, min(initial_rank_floor, len(rungs) - 1))
        effort_arm: str | None = None
        trace = _Trace(path=[], cost=0.0, scorable=True, passed=False, unmeasured=0)
        pending: FailureEvent | None = None

        for session in range(self._max_sessions):
            # 1. Base routing is memoryless, so the served rung is the task's rank floor —
            #    `RouterEngine._lift_to_rank_floor`, without which the ladder oscillates.
            model = rungs[min(rank_floor, len(rungs) - 1)]
            arm = self._resolve_arm(model, effort_arm, ladders)
            runner = self._runner(model, rungs, ladders, log, arm, rank_floor)
            if pending is not None:
                runner.observe(success=False, event=pending)
            directive = runner.decide(session, self._config)
            if directive.reason == _CEILING:
                break  # ladder exhausted: every rung has had its full attempt quota
            model, arm, rank_floor, effort_arm = self._apply(
                directive, runner, model, arm, rank_floor, rungs, ladders
            )
            log = list(runner.log)

            outcome = self._cell(task_id, model, arm, matrix, ladders)
            self._bill(trace, model, arm, outcome)
            if outcome is not None and outcome.get("pass", False):
                trace.passed = True
                break
            pending = failure_event_from_outcome(
                decision_index=session,
                failing_check_id=f"session-cascade:{task_id}",
                success=False,
                is_infra_failure=False,
                confirmed=True,
            )
        self._mark_terminal_arm(trace, ladders)
        return trace

    def _runner(  # noqa: PLR0913 (one arg per piece of the concrete ladder position)
        self,
        model: str,
        rungs: Sequence[str],
        ladders: Mapping[str, ArmLadder],
        log: list[FailureEvent],
        arm: str,
        rank_floor: int,
    ) -> EscalationRunner:
        """Seed the SHARED runner with this decision's concrete position (as the engine does)."""
        lad = ladders.get(model)
        return EscalationRunner(
            max_effort_index=lad.max_index if lad else 0,
            max_rank_index=len(rungs) - 1,
            log=log,
            effort_index=lad.index_of(arm) if lad else 0,
            rank_index=rank_floor,
        )

    def _apply(  # noqa: PLR0913 (mirrors RouterEngine._apply_effort / _apply_rank)
        self,
        directive: EscalationDirective,
        runner: EscalationRunner,
        model: str,
        arm: str,
        rank_floor: int,
        rungs: Sequence[str],
        ladders: Mapping[str, ArmLadder],
    ) -> tuple[str, str, int, str | None]:
        """Apply the directive concretely; a rung that cannot be delivered leaves it untouched."""
        action = directive.action
        if action is EscalationAction.RAISE_EFFORT:
            lad = ladders.get(model)
            nxt = lad.above(arm) if lad else None
            if nxt is not None:
                runner.commit(action)
                return (model, nxt, rank_floor, nxt)
        elif action is EscalationAction.RAISE_RANK and rank_floor + 1 < len(rungs):
            runner.commit(action)
            # The SHARED shortlist rule, not a +1 walk: the live `_escalate_one_rank` aims at
            # `next_rung_rank` from the rank the task owns, so a replay that stepped one rank at a
            # time here priced intermediate rungs production stopped buying. Clamped to the pool
            # because the rule's ceiling is the pool's top rank, and this pool is the matrix's.
            target = min(
                next_rung_rank(rank_floor, len(rungs) - 1, self._config.rank_shortlist),
                len(rungs) - 1,
            )
            higher = rungs[target]
            # Monotone floor: the climbed rank is what the NEXT session re-serves.
            return (higher, self._resolve_arm(higher, None, ladders), target, None)
        return (model, arm, rank_floor, arm)

    @staticmethod
    def _resolve_arm(model: str, arm: str | None, ladders: Mapping[str, ArmLadder]) -> str:
        """The arm this model is served on: the task's held arm, else the model's default."""
        lad = ladders.get(model)
        if lad is None:
            return ""
        return lad.arms[lad.index_of(arm)] if arm is not None else lad.default_arm

    # ------------------------------------------------------------------
    # Outcome lookup and billing
    # ------------------------------------------------------------------
    def _cell(
        self,
        task_id: str,
        model: str,
        arm: str,
        matrix: dict,
        ladders: Mapping[str, ArmLadder],
    ) -> dict | None:
        """The measured cell for one attempt, or None when that (model, arm) was never run."""
        # The DEFAULT arm reads `matrix["results"]` — the same (possibly coverage-completed) view
        # every other strategy scores on, so the head-to-head is like-for-like. A NON-default arm
        # can only come from the raw arm cache, which imputation never touches.
        lad = ladders.get(model)
        if lad is None or arm == lad.default_arm:
            return matrix.get("results", {}).get(task_id, {}).get(model) or None
        return self._arms().get(task_id, {}).get(model, {}).get(arm) or None

    @staticmethod
    def _bill(trace: _Trace, model: str, arm: str, outcome: dict | None) -> None:
        """Charge one session to the trace — an unmeasured cell makes the whole path unscorable."""
        trace.path.append((model, arm))
        if outcome is None:
            trace.scorable = False
            trace.unmeasured += 1
            return
        session_cost = float(outcome.get("cost", 0.0))
        trace.attempts.append(billed_attempt(model, outcome))
        trace.cost += session_cost

    @staticmethod
    def _mark_terminal_arm(trace: _Trace, ladders: Mapping[str, ArmLadder]) -> None:
        """A terminal non-default arm is unscorable: `summary.evaluate` reads the DEFAULT cell."""
        # The scoring interface reports pass/fail from `matrix["results"][task][model]`, which is
        # the model's default-arm measurement. If the last session ran a higher arm, that cell is a
        # DIFFERENT measurement from the one the replay actually observed, so the row cannot be
        # scored honestly. Marking it unscorable is the conservative call; the count is reported.
        if not trace.path:
            return
        model, arm = trace.path[-1]
        lad = ladders.get(model)
        if lad is not None and arm != lad.default_arm:
            trace.scorable = False

    def _publish(self, trace: _Trace) -> None:
        """Expose one task's trace on the attributes `summary.evaluate` and the report read."""
        self.cascade_total_cost = trace.cost
        self.cascade_tried_models = [m for m, _arm in trace.path]
        self.cascade_attempts = list(trace.attempts)
        self.cascade_scorable = trace.scorable
        self.session_path = list(trace.path)
        self.session_hops = trace.hops
        self.session_unmeasured = trace.unmeasured

    # ------------------------------------------------------------------
    # Lazily-resolved config (injected in tests, read from the repo otherwise)
    # ------------------------------------------------------------------
    def _ladders(self) -> Mapping[str, ArmLadder]:
        if self._arm_ladders is None:
            self._arm_ladders = default_arm_ladders()
        return self._arm_ladders

    def _arms(self) -> Mapping[str, Mapping[str, Mapping[str, dict]]]:
        if self._arm_results is None:
            from benchmark import config

            self._arm_results = config.load_results()
        return self._arm_results
