"""Parity kill-gate over both structurally distinct paths (the RouterEngine and
`replay.replay_config`), and the one geometry that still parts them.
"""

# Parity is real and asserted for the failure-log lifecycle (append / clear on a VERIFIED PASS
# only, an unstamped or infra step being a no-op / retire-on-escalation), the EFFORT rung, AND the
# RANK rung. The engine persists both rungs per task — the effort arm and the capability-rank floor
# it re-serves on the next decision — and drops both on a verified pass, which is exactly what the
# runner's abstract monotone counters do. So the whole directive stream matches, ceilings included.
#
# The first condition is a LADDER-HOMOGENEOUS pool. The abstract counter carries a single effort
# ceiling for every rank, so a higher-rank model exposing fewer reasoning arms hits its own ceiling
# before the counter does and the streams part there. `_Pool` below is homogeneous and its ceilings
# equal the offline context's (4 ranked models -> rank ceiling 3; a 3-arm ladder -> effort ceiling
# 2), so full parity is the assertion; `_ArmlessTopPool` pins the heterogeneous boundary instead.
#
# The second is a pool NARROWER THAN THE RANK SHORTLIST + 1. The counter advances rank by exactly
# +1, whereas the engine walks only the `rank_shortlist` cheapest ranks one at a time and then
# jumps to the top rank. At `_Pool`'s 4 models and the shipped shortlist of 3 the two shapes
# coincide exactly (ranks 0,1,2 walked, then the top rank is 3 either way), which is why the parity
# assertions below are unchanged. `_WidePool` pins the boundary where they part: the engine reaches
# its ceiling in fewer rungs and holds while the counter is still climbing. That divergence is the
# shortlist working, not a live bug — the offline replay is the side that models a ladder nothing
# runs, and closing it means teaching the replay's context about the shortlist.

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from benchmark.escalation import replay
from benchmark.escalation.replay import GridPoint
from benchmark.escalation.schema import StepView
from shunt.models.config import ModelConfig, ReasoningArm, ReasoningConfig
from shunt.router.engine import RouterEngine
from shunt.router.escalation import EscalationAction, EscalationConfig, EscalationContext
from tests.escalation.factories import make_step, make_trajectory

_KEY = "tests/x.py::a"


# --- a real engine over fakes, with a 3-arm reasoning ladder so escalations stay on EFFORT ---


def _cfg(name: str, reasoning: ReasoningConfig | None = None) -> ModelConfig:
    return ModelConfig(
        name=name,
        provider="p",
        base_url="http://x",
        api_key_env_var="K",
        reasoning=reasoning,
    )


def _ladder() -> ReasoningConfig:
    return ReasoningConfig(
        default_arm="low",
        arms=[
            ReasoningArm(id="low", rank=0, api={"reasoning_effort": "low"}),
            ReasoningArm(id="mid", rank=1, api={"reasoning_effort": "medium"}),
            ReasoningArm(id="high", rank=2, api={"reasoning_effort": "high"}),
        ],
    )


class _Pool:
    """4 ranked models sharing one 3-arm ladder — ceilings equal to the offline context's."""

    _NAMES = ("qwen", "glm", "kimi", "opus")

    def __init__(self) -> None:
        self._ranked = [_cfg(n, _ladder()) for n in self._NAMES]  # weakest -> strongest
        self._models = {m.name: m for m in self._ranked}

    def get_model(self, name: str) -> ModelConfig | None:
        return self._models.get(name)

    def ranked_models(self) -> list[ModelConfig]:
        return list(self._ranked)

    def rank_of(self, name: str) -> int | None:
        for i, m in enumerate(self._ranked):
            if m.name == name:
                return i
        return None

    def models_from_rank(self, i: int) -> list[ModelConfig]:
        return self._ranked[max(i, 0) :]

    def is_healthy(self, name: str) -> bool:
        return True


@dataclass
class _Session:
    tool_identity: str = "toolA"


class _SessionManager:
    def get_session(self, session_id: str) -> _Session:
        return _Session()


class _Index:
    def count_labeled(self) -> int:
        return 100

    def count_total_labeled(self) -> int:
        return 100

    def effective_labeled(self) -> float:
        return 100.0

    def effective_tier2(self) -> float:
        return 100.0

    def model_priors(self) -> dict[str, tuple[float, float]]:
        return {}

    def query(self, embedding: np.ndarray, k: int = 20) -> list:  # type: ignore[type-arg]
        return []


class _Embedder:
    def embed(self, text: str) -> np.ndarray:  # type: ignore[type-arg]
        return np.zeros(8, dtype=np.float32)


class _WidePool(_Pool):
    """Wider than the shortlist + 1, so the engine's rank rung JUMPS and the counter's does not."""

    _NAMES = ("qwen", "glm", "kimi", "sonnet", "gemini", "opus")


class _ArmlessTopPool(_Pool):
    """The heterogeneous case: the rank-1 model exposes no reasoning arms at all."""

    _NAMES = ("qwen", "glm")

    def __init__(self) -> None:
        super().__init__()
        self._ranked[1] = _cfg("glm")  # strip the top model's reasoning arms
        self._models = {m.name: m for m in self._ranked}


def _engine(pool: _Pool | None = None) -> RouterEngine:
    return RouterEngine(
        model_pool=pool if pool is not None else _Pool(),
        session_manager=_SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2, ladder="effort_then_rank"),
        task_key_resolver=lambda _s: "repoA",
    )


def _engine_action(reason: str, prov: dict[str, Any]) -> EscalationAction:
    """Map one engine decision to the escalation directive it applied."""
    if reason != "auto_escalation":
        return EscalationAction.HOLD
    return (
        EscalationAction.RAISE_EFFORT
        if "escalated_reasoning_arm" in prov
        else EscalationAction.RAISE_RANK
    )


class _Outcome(StrEnum):
    """The four step shapes the live plane actually produces, not just pass/fail."""

    PASS = "pass"
    FAIL = "fail"
    UNVERIFIED = "unverified"  # the per-step stamping stage never ran on this step
    INFRA = "infra"  # verifier said `unknown` + is_infra_failure — not a labellable outcome


def _live_stream_outcomes(
    outcomes: list[_Outcome], *, pool: _Pool | None = None
) -> list[EscalationAction]:
    """Drive the real engine over an arbitrary outcome pattern."""
    # The engine decides, then records outcome i; its decision at turn i sees outcomes 0..i-1, so
    # the directive AFTER the first outcome is what the offline replay (observe-then-decide)
    # reproduces. Failures share one dedup key; a pass records a non-failure (clears the window).
    # UNVERIFIED/INFRA record NOTHING: `CaptureCoordinator` gates on `_LABELLABLE`, so a step with
    # no verified outcome never reaches `record_outcome` and the failure log is untouched.
    eng = _engine(pool)
    actions: list[EscalationAction] = []
    n = len(outcomes)
    for i in range(n + 1):
        _model, reason, prov = eng.decide(f"s{i}", "task")
        actions.append(_engine_action(reason, prov))
        if i < n and outcomes[i] in (_Outcome.PASS, _Outcome.FAIL):
            ok = outcomes[i] is _Outcome.PASS
            eng.record_outcome(
                downshift=False,
                success=ok,
                task_key="repoA",
                dedup_key=None if ok else _KEY,
                exit_code=None if ok else 1,
                is_infra_failure=False,
                confirmed=not ok,
            )
    return actions[1:]  # drop the pre-outcome decide; align with the observe-then-decide replay


def _step_for(index: int, outcome: _Outcome) -> StepView:
    """The StepView the normalizers write for each outcome shape (committed-field shapes only)."""
    if outcome is _Outcome.FAIL:
        return make_step(
            step_index=index, decision_index=index, success=False, failing_check_id=_KEY
        )
    if outcome is _Outcome.UNVERIFIED:  # `base.make_step` defaults: success=True, confirmed=False
        return make_step(step_index=index, decision_index=index, success=True, confirmed=False)
    if outcome is _Outcome.INFRA:  # `stamp_step` on outcome="unknown": success=True, infra=True
        return make_step(
            step_index=index, decision_index=index, success=True, is_infra_failure=True
        )
    return make_step(step_index=index, decision_index=index, success=True)


def _offline_stream_outcomes(
    outcomes: list[_Outcome], *, stale_window: int = 10
) -> list[EscalationAction]:
    """Drive the real replay over the equivalent trajectory, matched ceilings."""
    traj = make_trajectory([_step_for(i, o) for i, o in enumerate(outcomes)])
    ctx = EscalationContext(
        current_rank_index=0,
        max_rank_index=3,  # `_Pool`'s 4 ranked models → rank ceiling index 3
        current_effort_index=0,
        max_effort_index=2,  # the shared 3-arm ladder → effort ceiling index 2
    )
    return replay.replay_config(
        traj, GridPoint(2, stale_window).to_config(), context=ctx
    ).directives


def _live_stream(n_failures: int) -> list[EscalationAction]:
    """Drive the real engine over `n_failures` same-key failures (no interleaved success)."""
    return _live_stream_outcomes([_Outcome.FAIL] * n_failures)


def _offline_stream(n_failures: int) -> list[EscalationAction]:
    """Drive the real replay over `n_failures` same-key failures (no interleaved success)."""
    return _offline_stream_outcomes([_Outcome.FAIL] * n_failures)


def test_live_and_offline_directive_streams_match() -> None:
    # Regression guard: 4 same-key confirmed failures, escalate_after_n=2. The engine
    # retires its window after each escalation, so a HOLD sits between the two effort steps; a
    # replay that did NOT retire would emit RAISE_EFFORT there and diverge.
    live = _live_stream(4)
    offline = _offline_stream(4)
    assert live == offline
    assert live == [
        EscalationAction.HOLD,
        EscalationAction.RAISE_EFFORT,
        EscalationAction.HOLD,
        EscalationAction.RAISE_EFFORT,
    ]


def test_offline_stream_reflects_retire_not_the_naive_over_fire() -> None:
    # Pin the exact bug this test guards: the naive no-retire stream (which the pre-fix replay
    # emitted) re-fires on every step; the real lifecycle must NOT.
    offline = _offline_stream(4)
    naive_over_fire = [
        EscalationAction.HOLD,
        EscalationAction.RAISE_EFFORT,
        EscalationAction.RAISE_EFFORT,
        EscalationAction.RAISE_EFFORT,
    ]
    assert offline != naive_over_fire


# --- the effort<->rank ladder: parity across every crossing, and where a pool's shape breaks it ---


def test_full_directive_stream_parity_across_every_rank_crossing() -> None:
    # 24 same-key failures walk the whole ladder: three effort climbs, three rank crossings, and
    # saturation. The engine persists the climbed rank as a per-task floor and re-serves it, which
    # is precisely the runner's monotone counter, so the streams match rung for rung.
    live = _live_stream(24)
    offline = _offline_stream(24)
    assert live == offline
    assert live.count(EscalationAction.RAISE_RANK) == 3


def test_both_paths_saturate_at_the_same_ceiling() -> None:
    # The end of the ladder is where a memoryless engine used to part from the replay: it re-picked
    # the cheap model each decision and re-climbed forever (RAISE_RANK at index 23) while the replay
    # had already saturated. With the rank floor both sides run out of rungs together — rank AND
    # effort at their ceilings -> HOLD, on the same index.
    live = _live_stream(24)
    offline = _offline_stream(24)
    assert live[23] is EscalationAction.HOLD
    assert offline[23] is EscalationAction.HOLD


def test_rank_parity_breaks_when_a_higher_model_has_a_shorter_effort_ladder() -> None:
    # The one condition the abstract counter cannot express: it carries a SINGLE effort ceiling for
    # every rank. Give the rank-1 model no reasoning arms and the engine hits its effort ceiling the
    # moment it arrives there, while the counter still believes two arms remain — so the engine
    # holds where the replay raises effort. A pool-derived replay context must therefore be built
    # from a ladder-homogeneous pool, or its post-first-rank rungs describe a policy nothing runs.
    live = _live_stream_outcomes([_Outcome.FAIL] * 12, pool=_ArmlessTopPool())
    offline = _offline_stream(12)
    assert live[:7] == offline[:7]  # identical up to and including the rank crossing at index 5
    assert live[7] is EscalationAction.HOLD  # glm has no arm above its default
    assert offline[7] is EscalationAction.RAISE_EFFORT


def test_a_verified_pass_returns_both_paths_to_the_base_rung() -> None:
    # The engine pops BOTH the effort arm and the rank floor on a verified suite pass — the task is
    # unstuck, so it falls back to base routing. A runner that reset only effort stayed a full
    # rank-cycle ahead and saturated early, holding while the engine still had rungs. Climb to rank
    # 1, pass, then climb again: the second ascent must reproduce the first.
    f, p = _Outcome.FAIL, _Outcome.PASS
    outcomes = [f] * 8 + [p] + [f] * 22
    live = _live_stream_outcomes(outcomes)
    offline = _offline_stream_outcomes(outcomes)
    assert live == offline
    assert offline[14] is EscalationAction.RAISE_RANK  # the post-pass ascent re-crosses rank 0->1


def test_effort_parity_holds_across_a_verified_success() -> None:
    # C1 regression: a verified success resets the engine's effort ladder (it pops the task's
    # effort arm), so a later same-key failure run must climb effort from the default again — not
    # jump to a rank because the effort rung was left pinned at its ceiling. Pre-fix the replay
    # cleared only the log on success and left effort at the ceiling, so the streams diverged at
    # the first post-success escalation (offline raise_tier vs the engine's raise_effort). Drive
    # F F F F S F F F F through the REAL engine and the REAL replay and assert they still agree.
    f, p = _Outcome.FAIL, _Outcome.PASS
    outcomes = [f, f, f, f, p, f, f, f, f]
    live = _live_stream_outcomes(outcomes)
    offline = _offline_stream_outcomes(outcomes)
    assert live == offline
    # index 6 = first escalation after the success: effort reset ⇒ raise_effort, NOT raise_tier.
    assert offline[6] is EscalationAction.RAISE_EFFORT


# --- F1 regression: the two step shapes that carry `success=True` for non-pass reasons ---


def test_parity_holds_across_an_unverified_gap() -> None:
    # F1: an unstamped step is `success=True` by PARSER DEFAULT. Live it produces no capture event
    # at all (`CaptureCoordinator` gates on `_LABELLABLE`), so the window survives the gap and the
    # second same-key failure fires. Pre-fix the replay read that default as a verified pass, wiped
    # the window, and held forever — so it could only ever fire on STRICTLY CONSECUTIVE failures.
    outcomes = [_Outcome.FAIL, _Outcome.UNVERIFIED, _Outcome.FAIL]
    live = _live_stream_outcomes(outcomes)
    offline = _offline_stream_outcomes(outcomes)
    assert live == offline
    assert offline[2] is EscalationAction.RAISE_EFFORT


def test_parity_holds_across_an_infra_red() -> None:
    # F1, second shape: `stamp_step` sets `success = outcome != "failure"`, so an `unknown` +
    # is_infra_failure step lands as success=True (1112 such steps sit in the committed corpus).
    # Live that outcome is not labellable and never reaches `record_outcome`, so it must be a no-op
    # on the failure log — NOT the window-clearing verified pass the pre-fix replay treated it as.
    outcomes = [_Outcome.FAIL, _Outcome.INFRA, _Outcome.FAIL]
    live = _live_stream_outcomes(outcomes)
    offline = _offline_stream_outcomes(outcomes)
    assert live == offline
    assert offline[2] is EscalationAction.RAISE_EFFORT


def test_stale_window_is_observable_in_the_replay() -> None:
    # The knob must have teeth: two same-key failures separated by 4 unverified steps recur 5
    # decisions apart, so window=1 retires the first before the second arrives and window=1000
    # keeps it. Pre-fix EVERY window behaved identically (the replay cleared the counter on the
    # gap itself), which is what made the sweep's inertness look like a corpus property.
    gap = [_Outcome.UNVERIFIED] * 4
    outcomes = [_Outcome.FAIL, *gap, _Outcome.FAIL]
    assert _offline_stream_outcomes(outcomes, stale_window=1)[5] is EscalationAction.HOLD
    assert _offline_stream_outcomes(outcomes, stale_window=1000)[5] is EscalationAction.RAISE_EFFORT


def test_rank_parity_breaks_when_the_pool_is_wider_than_the_shortlist() -> None:
    # The second condition, pinned. Six ranked models against the shipped shortlist of 3: the
    # engine walks ranks 0,1,2 and then JUMPS to rank 5, so it saturates after three rank rungs
    # while the abstract +1 counter needs five. Up to the third crossing the streams are identical;
    # past it the engine holds at its ceiling where the counter still raises. A replay context
    # built off a pool this wide therefore describes a ladder production does not walk.
    live = _live_stream_outcomes([_Outcome.FAIL] * 36, pool=_WidePool())
    traj = make_trajectory([_step_for(i, _Outcome.FAIL) for i in range(36)])
    ctx = EscalationContext(
        current_rank_index=0,
        max_rank_index=5,  # the context a replay would DERIVE from `_WidePool`'s 6 models
        current_effort_index=0,
        max_effort_index=2,
    )
    offline = replay.replay_config(traj, GridPoint(2, 10).to_config(), context=ctx).directives
    assert live[:18] == offline[:18]  # identical through the two rungs inside the shortlist
    assert live.count(EscalationAction.RAISE_RANK) == 3  # 0->1, 1->2, 2->5 (the jump)
    assert offline.count(EscalationAction.RAISE_RANK) == 5  # the +1 counter buys every rung
    assert live[23] is EscalationAction.HOLD  # the engine has saturated...
    assert offline[23] is EscalationAction.RAISE_RANK  # ...where the counter is still climbing


def _first_flag(stream: list[EscalationAction]) -> int | None:
    """The index the policy first fired at — the ONLY thing the trajectory-level metric reads."""
    return next((i for i, a in enumerate(stream) if a is not EscalationAction.HOLD), None)


def test_detection_point_is_parity_faithful_under_a_mismatched_ladder() -> None:
    # What the eval actually consumes is the first-escalation index (via replay.ReplayDecision),
    # and it is reached BEFORE any rung is climbed — so it survives even the heterogeneous pool
    # whose later rungs provably diverge above. This is why the metric is robust to the replay's
    # ladder geometry being a modelling choice rather than a measured one.
    live = _live_stream_outcomes([_Outcome.FAIL] * 24, pool=_ArmlessTopPool())
    offline = _offline_stream(24)
    assert live != offline  # the raw streams part once the shorter ladder runs out
    assert _first_flag(live) == _first_flag(offline)
    assert _first_flag(offline) is not None
