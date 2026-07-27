"""Parity kill-gate BOUNDED to what the offline replay actually reproduces, over both structurally
distinct paths (the RouterEngine and `replay.replay_config`).
"""

# Parity is real and asserted for the failure-log lifecycle (append / clear-on-success /
# retire-on-escalation) AND the EFFORT rung (the engine persists effort via its per-task effort
# arm and resets it to the default on a verified success, mirrored by the runner — so effort
# parity holds even across a success boundary). It is NOT real for the RANK rung: the replay
# climbs a persistent monotone abstract rank
# counter that saturates at a ceiling, while the engine re-seeds rank from the base routing pick
# each decision (no persistent rank ladder) and re-escalates indefinitely. The rank stream is an
# isolation-model upper bound, not an engine reproduction; the tests below assert both the real
# parity AND that documented divergence boundary.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmark.escalation import replay, run_eval
from benchmark.escalation.replay import GridPoint
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
    def __init__(self) -> None:
        self._models = {"qwen": _cfg("qwen", _ladder()), "glm": _cfg("glm")}
        self._ranked = [self._models["qwen"], self._models["glm"]]  # weakest -> strongest

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


def _engine() -> RouterEngine:
    return RouterEngine(
        model_pool=_Pool(),
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


def _live_stream_outcomes(successes: list[bool]) -> list[EscalationAction]:
    """Drive the real engine over an arbitrary success/failure pattern (True = verified pass)."""
    # The engine decides, then records outcome i; its decision at turn i sees outcomes 0..i-1, so
    # the directive AFTER the first outcome is what the offline replay (observe-then-decide)
    # reproduces. Failures share one dedup key; a success records a non-failure (clears the window).
    eng = _engine()
    actions: list[EscalationAction] = []
    n = len(successes)
    for i in range(n + 1):
        _model, reason, prov = eng.decide(f"s{i}", "task")
        actions.append(_engine_action(reason, prov))
        if i < n:
            ok = successes[i]
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


def _offline_stream_outcomes(successes: list[bool]) -> list[EscalationAction]:
    """Drive the real replay over the equivalent trajectory, matched ceilings."""
    steps = [
        make_step(
            step_index=i,
            decision_index=i,
            success=successes[i],
            failing_check_id=None if successes[i] else _KEY,
        )
        for i in range(len(successes))
    ]
    traj = make_trajectory(steps)
    ctx = EscalationContext(
        current_rank_index=0,
        max_rank_index=3,  # abstract isolation ceiling (was len(TIER_ORDER) - 1)
        current_effort_index=0,
        max_effort_index=2,  # qwen's 3-arm ladder → effort ceiling index 2
    )
    return replay.replay_config(traj, GridPoint(2, 10).to_config(), context=ctx).directives


def _live_stream(n_failures: int) -> list[EscalationAction]:
    """Drive the real engine over `n_failures` same-key failures (no interleaved success)."""
    return _live_stream_outcomes([False] * n_failures)


def _offline_stream(n_failures: int) -> list[EscalationAction]:
    """Drive the real replay over `n_failures` same-key failures (no interleaved success)."""
    return _offline_stream_outcomes([False] * n_failures)


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


# --- effort->rank boundary: parity holds through effort, then the abstract rank ladder parts ---


def test_effort_rung_and_lifecycle_parity_holds_before_the_tier_ceiling() -> None:
    # Parity is REAL through the first rank crossing and several effort<->effort<->rank cycles: the
    # log lifecycle and the effort rung match by construction. 24 same-key failures line up 0..22.
    live = _live_stream(24)
    offline = _offline_stream(24)
    assert live[:23] == offline[:23]


def test_tier_stream_diverges_at_the_abstract_ceiling_not_engine_faithful() -> None:
    # The documented boundary: the replay's persistent monotone rank counter saturates
    # (rank at ceiling AND effort at ceiling -> HOLD), so it emits HOLD at index 23; the engine has
    # no persistent rank ladder and re-escalates (RAISE_RANK). The replay's rank is an abstract
    # isolation upper bound, NOT an engine reproduction — assert the divergence rather than a false
    # full-stream match.
    live = _live_stream(24)
    offline = _offline_stream(24)
    assert live != offline
    assert live[23] is EscalationAction.RAISE_RANK
    assert offline[23] is EscalationAction.HOLD


def test_effort_parity_holds_across_a_verified_success() -> None:
    # C1 regression: a verified success resets the engine's effort ladder (it pops the task's
    # effort arm), so a later same-key failure run must climb effort from the default again — not
    # jump to a rank because the effort rung was left pinned at its ceiling. Pre-fix the replay
    # cleared only the log on success and left effort at the ceiling, so the streams diverged at
    # the first post-success escalation (offline raise_tier vs the engine's raise_effort). Drive
    # F F F F S F F F F through the REAL engine and the REAL replay and assert they still agree.
    outcomes = [False, False, False, False, True, False, False, False, False]
    live = _live_stream_outcomes(outcomes)
    offline = _offline_stream_outcomes(outcomes)
    assert live == offline
    # index 6 = first escalation after the success: effort reset ⇒ raise_effort, NOT raise_tier.
    assert offline[6] is EscalationAction.RAISE_EFFORT


def test_cumulative_detection_metric_is_parity_faithful_past_the_tier_ceiling() -> None:
    # Even where the raw directive streams diverge (index 23), the detector metric run_eval scores
    # — "has the policy flagged this failing task by prefix t" — is identical for both paths,
    # because both flag at the SAME first-escalation step. This is why the metric is robust to the
    # rank-ladder abstraction: it does not read the post-flag rungs.
    live = _live_stream(24)
    offline = _offline_stream(24)
    assert live != offline  # raw streams part at the ceiling
    live_scores = run_eval._cumulative_detection(live)
    offline_scores = run_eval._cumulative_detection(offline)
    assert live_scores == offline_scores
    assert live_scores == sorted(live_scores)  # monotone
