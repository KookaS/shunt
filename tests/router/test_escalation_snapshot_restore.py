"""An escalated reasoning-effort arm survives a snapshot -> restore round-trip.

A restart must not silently reset a task to its default arm: the restored engine
resumes the ladder from where it stopped, stepping the NEXT arm on the next recurrence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shunt.models.config import ModelConfig, ReasoningArm, ReasoningConfig
from shunt.router.engine import RouterEngine, task_state_key
from shunt.router.escalation import EscalationConfig


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


class _ReasoningPool:
    """qwen with a 3-arm reasoning ladder < glm with no arms (weakest -> strongest)."""

    def __init__(self) -> None:
        self._models = {
            "qwen": _cfg("qwen", _ladder()),
            "glm": _cfg("glm"),
        }
        self._ranked = [self._models["qwen"], self._models["glm"]]

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


def _engine() -> RouterEngine:
    return RouterEngine(
        model_pool=_ReasoningPool(),
        session_manager=_SessionManager(),
        outcome_index=_Index(),
        embedder=_Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2, ladder="effort_then_rank"),
        task_key_resolver=lambda _s: "repoA",
    )


def _fail(eng: RouterEngine) -> None:
    eng.record_outcome(
        downshift=False,
        success=False,
        task_key="repoA",
        dedup_key="t::a",
        exit_code=1,
        is_infra_failure=False,
        confirmed=True,
    )


def _two_reds_then_decide(eng: RouterEngine, tag: str) -> dict[str, object]:
    eng.decide(f"a{tag}", "task")
    _fail(eng)
    eng.decide(f"b{tag}", "task")
    _fail(eng)
    _model, _reason, prov = eng.decide(f"c{tag}", "task")
    return prov


def test_effort_arm_survives_snapshot_restore_and_resumes_the_ladder() -> None:
    eng = _engine()
    prov = _two_reds_then_decide(eng, "0")
    assert prov["escalated_reasoning_arm"] == "mid"  # low -> mid on the first recurrence

    state = eng.snapshot_escalation_state()
    assert "effort_arm" in state  # the per-task arm mapping is serialized
    assert state["effort_arm"] == {task_state_key("repoA"): "mid"}

    fresh = _engine()  # a restarted process
    fresh.restore_escalation_state(state)
    prov2 = _two_reds_then_decide(fresh, "1")
    # Resumed from "mid" — stepped the NEXT arm, did NOT restart at the default "low".
    assert prov2["escalated_reasoning_arm"] == "high"
