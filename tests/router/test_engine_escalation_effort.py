"""B5: cache-safe reasoning-effort escalation before a model-tier step.

A model WITH a reasoning ladder raises its effort arm first (same model, cache-safe), and only
steps a tier once the ladder is exhausted; a model WITHOUT arms steps tier directly. Fakes only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shunt.models.config import ModelConfig, ReasoningArm, ReasoningConfig
from shunt.router.engine import RouterEngine
from shunt.router.escalation import EscalationConfig


def _cfg(name: str, tier: str, reasoning: ReasoningConfig | None = None) -> ModelConfig:
    return ModelConfig(
        name=name,
        tier=tier,  # type: ignore[arg-type]
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
            ReasoningArm(id="high", rank=1, api={"reasoning_effort": "high"}),
        ],
    )


class _ReasoningPool:
    """cheap=qwen (2-arm reasoning ladder), mid=glm (no arms). Resolvable via get_model (B5)."""

    def __init__(self, *, base_reasoning: ReasoningConfig | None) -> None:
        self._models = {
            "qwen": _cfg("qwen", "cheap", base_reasoning),
            "glm": _cfg("glm", "mid"),
        }
        self._tiers = {
            "cheap": [self._models["qwen"]],
            "mid": [self._models["glm"]],
            "high": [],
            "frontier": [],
        }

    def get_model(self, name: str) -> ModelConfig | None:
        return self._models.get(name)

    def get_tier_models(self, tier: str) -> list[ModelConfig]:
        return self._tiers.get(tier, [])

    def is_healthy(self, name: str) -> bool:
        return True


@dataclass
class _Session:
    tool_identity: str


class _SessionManager:
    def get_session(self, session_id: str) -> _Session:
        return _Session(tool_identity="toolA")


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


def _engine(*, base_reasoning: ReasoningConfig | None) -> RouterEngine:
    return RouterEngine(
        model_pool=_ReasoningPool(base_reasoning=base_reasoning),
        session_manager=_SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2, ladder="effort_then_tier"),
        task_key_resolver=lambda _s: "repoA",
    )


def _fail(eng: RouterEngine, key: str = "t::a") -> None:
    eng.record_outcome(
        downshift=False,
        success=False,
        task_key="repoA",
        dedup_key=key,
        exit_code=1,
        blocking=True,
        confirmed=True,
    )


def test_effort_first_then_tier_on_continued_failure() -> None:
    eng = _engine(base_reasoning=_ladder())
    m0, _, _ = eng.decide("s0", "task")
    assert m0 == "qwen"

    # First recurrence → raise EFFORT (same model, higher arm) — cache-safe.
    _fail(eng)
    eng.decide("s1", "task")  # holds after one
    _fail(eng)
    m1, r1, prov1 = eng.decide("s2", "task")
    assert m1 == "qwen"  # SAME model on an effort step (cache namespace unchanged)
    assert r1 == "auto_escalation"
    assert prov1["escalated_reasoning_arm"] == "high"
    assert prov1["auto_escalated"] is True

    # Second recurrence, now at the top arm → step a model TIER.
    _fail(eng)
    eng.decide("s3", "task")
    _fail(eng)
    m2, r2, prov2 = eng.decide("s4", "task")
    assert m2 == "glm"  # ladder exhausted → tier step cheap→mid
    assert r2 == "auto_escalation"
    assert "escalated_reasoning_arm" not in prov2  # a tier step carries no reasoning arm


def test_tier_step_resets_the_effort_arm() -> None:
    # Escalate qwen low→high (effort), then exhaust its ladder → step tier to glm. The stale
    # "high" arm from qwen must be cleared so glm starts at its OWN default arm, not a foreign id.
    eng = _engine(base_reasoning=_ladder())
    eng.decide("s0", "task")
    _fail(eng)
    eng.decide("s1", "task")
    _fail(eng)
    _m1, _r1, prov1 = eng.decide("s2", "task")
    assert prov1["escalated_reasoning_arm"] == "high"  # qwen effort at the top arm
    assert eng._task_effort_arm.get("repoA") == "high"

    _fail(eng)
    eng.decide("s3", "task")
    _fail(eng)
    m2, r2, _prov2 = eng.decide("s4", "task")
    assert m2 == "glm"  # ladder exhausted → tier step
    assert r2 == "auto_escalation"
    assert "repoA" not in eng._task_effort_arm  # the stale qwen arm was cleared on the tier step


def test_model_without_arms_steps_tier_directly() -> None:
    eng = _engine(base_reasoning=None)  # qwen has no reasoning ladder
    eng.decide("s0", "task")
    _fail(eng)
    eng.decide("s1", "task")
    _fail(eng)
    m, r, prov = eng.decide("s2", "task")
    assert m == "glm"  # no effort headroom → straight to a tier step
    assert r == "auto_escalation"
    assert "escalated_reasoning_arm" not in prov


def test_success_resets_the_effort_ladder() -> None:
    eng = _engine(base_reasoning=_ladder())
    eng.decide("s0", "task")
    _fail(eng)
    eng.decide("s1", "task")
    _fail(eng)
    _m, _r, prov = eng.decide("s2", "task")
    assert prov["escalated_reasoning_arm"] == "high"  # escalated to the top arm
    # A verified pass retires the effort escalation — the ladder is back at the default arm.
    eng.record_outcome(downshift=False, success=True, task_key="repoA", dedup_key=None, exit_code=0)
    _fail(eng)
    eng.decide("s3", "task")
    _fail(eng)
    m2, r2, prov2 = eng.decide("s4", "task")
    assert m2 == "qwen"  # still an effort step, NOT a tier jump — the ladder reset to default
    assert r2 == "auto_escalation"
    assert prov2["escalated_reasoning_arm"] == "high"  # steps low→high again, not high→(tier)
