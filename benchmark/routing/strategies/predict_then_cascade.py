"""Predict-then-cascade: a binary gate routes predicted-cheap tasks cheap-direct, the rest
through the Session-Cascade ladder.
"""

# ONE DECISION PER SESSION BOUNDARY, CACHE-SAFE. The gate fires once per task/session boundary:
# either the task is routed cheap-direct (the cheapest model only, a single attempt, no ladder)
# or it enters the Session-Cascade verify-and-escalate ladder. The ladder re-serves the SAME
# model on consecutive attempts (its recurrence window), so no model switch ever happens
# mid-cached-turn — the gate and the ladder both respect cache-safety by construction.
#
# WHY A SUBCLASS OF THE SESSION CASCADE. The ladder path MUST be byte-for-byte the
# Session-Cascade replay: a chance-level gate at f=0 must degenerate to Session-Cascade with no
# regression, and the ladder's cost/pass accounting (`cascade_total_cost`,
# `cascade_attempts`, `cascade_scorable`, the censoring and terminal-arm conventions) is
# inherited verbatim. The ONE override is `select`: gate says cheap-direct -> one-shot cheap;
# else -> the inherited ladder.
#
# THE SEAM FOR THE REAL PREDICTOR. The gate is a `Gate` object with `decides_cheap(task_id,
# task_meta)`. Today's placeholder, `FractionalGate(f)`, is score-free: a deterministic per-task
# draw sends fraction f cheap-direct (f=0 -> nothing, f=1 -> everything), so a chance-level gate
# interpolates the mixture line between Session-Cascade and Always-Cheap. A real difficulty score
# plugs in as another `Gate` implementation that thresholds a score computed from the task's
# features — the strategy, the billing and the eval do not change.

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING, Protocol

from ._cascade_common import (
    cheapest_priced_model,
    measured_models_by_price,
)
from .session_cascade import (
    _MAX_SESSIONS,
    DEFAULT_LADDER,
    SessionCascadeStrategy,
    _Trace,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .session_cascade import ArmLadder


class Gate(Protocol):
    """The binary cheap-vs-ladder decision, one call per task boundary.

    A future difficulty-score gate implements `decides_cheap` by thresholding a score derived
    from `task_meta` — nothing in the strategy or the eval changes when it does.
    """

    def decides_cheap(self, task_id: str, task_meta: dict) -> bool: ...


class FractionalGate:
    """Score-free placeholder: a deterministic per-task draw sends fraction *f* cheap-direct.

    Deterministic on (seed, task_id), so the same f picks the same task partition on every
    run — the eval's curve is reproducible, not a fresh draw per process.
    """

    def __init__(self, fraction: float, seed: int = 0) -> None:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        self._fraction = fraction
        self._seed = seed

    def decides_cheap(self, task_id: str, task_meta: dict) -> bool:
        """Whether this task draws the cheap-direct branch."""
        del task_meta
        # A pure hash draw, not a `random.Random` (SH008: an RNG draw can read as a proxy
        # vector). `crc32` keys the partition on (seed, task_id) so it is identical per run.
        # The divisor is 2^32, not 2^32-1: `crc32` tops out at 0xFFFFFFFF, so dividing by the
        # maximum value would let the draw equal exactly 1.0 and that one-in-2^32 task id would
        # fail `draw < fraction` for every f<1 — permanently stranded on the ladder path. With
        # the 2^32 divisor the draw is always in [0, 1) and the f=0/f=1 endpoints are exact.
        # A 2^-32 edge not hit on this corpus either way (184 task ids against a 2^32 space),
        # but the endpoint-degeneracy assertions should not depend on that luck.
        draw = zlib.crc32(f"{self._seed}:{task_id}".encode()) / 0x100000000
        return draw < self._fraction


class PerfectCheapGate:
    """Oracle placeholder for the eval and tests: sends exactly the tasks in *cheap_set*
    cheap-direct — the upper bound of the binary gate idea.
    """

    def __init__(self, cheap_set: set[str]) -> None:
        self._cheap_set = cheap_set

    def decides_cheap(self, task_id: str, task_meta: dict) -> bool:
        """Whether the oracle knows the cheap model solves this task."""
        del task_meta
        return task_id in self._cheap_set


class PredictThenCascadeStrategy(SessionCascadeStrategy):
    """Route predicted-cheap tasks to a single cheap attempt and everything else through the
    Session-Cascade ladder — one decision per task boundary, cache-safe by construction.
    """

    def __init__(  # noqa: PLR0913 (mirrors SessionCascadeStrategy's knob set, plus the gate)
        self,
        gate: Gate,
        label: str = "Predict-then-Cascade",
        escalate_after_n: int = 2,
        stale_window: int = 10,
        ladder: str = DEFAULT_LADDER,
        max_sessions: int = _MAX_SESSIONS,
        arm_results: Mapping[str, Mapping[str, Mapping[str, dict]]] | None = None,
        arm_ladders: Mapping[str, ArmLadder] | None = None,
        rank_shortlist: int | None = None,
    ) -> None:
        super().__init__(
            escalate_after_n=escalate_after_n,
            stale_window=stale_window,
            ladder=ladder,
            max_sessions=max_sessions,
            arm_results=arm_results,
            arm_ladders=arm_ladders,
            rank_shortlist=rank_shortlist,
        )
        self._gate = gate
        self._label = label

    @property
    def name(self) -> str:
        """Display name shown in eval output and plots."""
        return self._label

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        """The cheap-direct branch for a gate 'yes', else the inherited ladder."""
        if self._gate.decides_cheap(task_id, task_meta):
            return self._cheap_direct(task_id, matrix)
        return super().select(task_id, task_meta, matrix)

    def trace(self, task_id: str, matrix: dict) -> _Trace:
        """The full replay record for one task — reporting entry point, gate-aware."""
        # Match the base contract on a degenerate matrix BEFORE the gate branch: an empty
        # rungs list would IndexError in `_cheap_direct_trace`, where the base class returns
        # the empty trace. The gate must not change what a degenerate replay reports.
        if not measured_models_by_price(matrix):
            return _Trace(path=[], cost=0.0, scorable=False, passed=False, unmeasured=0)
        if self._gate.decides_cheap(task_id, matrix.get("tasks", {}).get(task_id, {})):
            return self._cheap_direct_trace(task_id, matrix)
        return super().trace(task_id, matrix)

    # ------------------------------------------------------------------
    # Cheap-direct
    # ------------------------------------------------------------------
    def _cheap_direct(self, task_id: str, matrix: dict) -> str:
        """Route one task to the cheapest model only; bill it as a single-attempt path."""
        rungs = measured_models_by_price(matrix)
        if not rungs:
            self._publish(_Trace(path=[], cost=0.0, scorable=False, passed=False, unmeasured=0))
            return cheapest_priced_model(matrix)
        trace = self._cheap_direct_trace(task_id, matrix)
        self._publish(trace)
        return rungs[0]

    def _cheap_direct_trace(self, task_id: str, matrix: dict) -> _Trace:
        """The single-attempt trace for one cheap-direct task, with the ladder's billing rules."""
        rungs = measured_models_by_price(matrix)
        cheapest = rungs[0]
        ladders = self._ladders()
        arm = self._resolve_arm(cheapest, None, ladders)
        outcome = self._cell(task_id, cheapest, arm, matrix, ladders)
        trace = _Trace(path=[], cost=0.0, scorable=True, passed=False, unmeasured=0)
        self._bill(trace, cheapest, arm, outcome)
        trace.passed = bool(outcome is not None and outcome.get("pass", False))
        self._mark_terminal_arm(trace, ladders)
        return trace
