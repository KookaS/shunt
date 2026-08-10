"""B5: cache-safe reasoning-effort escalation before a model-rank step.

A model WITH a reasoning ladder raises its effort arm first (same model, cache-safe), and only
steps a rank once the ladder is exhausted; a model WITHOUT arms steps rank directly. Fakes only.
"""

from __future__ import annotations

from shunt.models.config import ModelConfig, ReasoningArm, ReasoningConfig
from shunt.router.engine import RouterEngine, task_state_key
from shunt.router.escalation import EscalationConfig
from tests.router.escalation_fakes import Embedder, Index, SessionManager
from tests.router.escalation_fakes import model_config as _cfg


def _ladder() -> ReasoningConfig:
    return ReasoningConfig(
        default_arm="low",
        arms=[
            ReasoningArm(id="low", rank=0, api={"reasoning_effort": "low"}),
            ReasoningArm(id="high", rank=1, api={"reasoning_effort": "high"}),
        ],
    )


def _glm_ladder() -> ReasoningConfig:
    # A DIFFERENT arm vocabulary than qwen's {low, high}: an arm persisted from a qwen effort
    # escalation is FOREIGN here, so this models the real registry where each model declares
    # its own arm ids (deepseek {nothink,high,max}, gpt-5-mini {minimal,medium,high}, qwen
    # {nothink,think}).
    return ReasoningConfig(
        default_arm="nothink",
        arms=[
            ReasoningArm(id="nothink", rank=0, api={"enable_thinking": False}),
            ReasoningArm(id="think", rank=1, api={"enable_thinking": True}),
        ],
    )


class _ReasoningPool:
    """qwen (2-arm reasoning ladder) < glm (no arms). Resolvable via get_model (B5)."""

    def __init__(self, *, base_reasoning: ReasoningConfig | None) -> None:
        self._models = {
            "qwen": _cfg("qwen", base_reasoning),
            "glm": _cfg("glm"),
        }
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


class _MixedVocabularyPool:
    """qwen {low, high} < glm {nothink, think}: both have arms, vocabularies differ."""

    def __init__(self) -> None:
        self._models = {"qwen": _cfg("qwen", _ladder()), "glm": _cfg("glm", _glm_ladder())}
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


def _engine(*, base_reasoning: ReasoningConfig | None) -> RouterEngine:
    return RouterEngine(
        model_pool=_ReasoningPool(base_reasoning=base_reasoning),
        session_manager=SessionManager(),
        outcome_index=Index(),
        embedder=Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2, ladder="effort_then_rank"),
        task_key_resolver=lambda _s: "repoA",
    )


def _fail(eng: RouterEngine, key: str = "t::a") -> None:
    eng.record_outcome(
        downshift=False,
        success=False,
        task_key="repoA",
        dedup_key=key,
        exit_code=1,
        is_infra_failure=False,
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

    # Second recurrence, now at the top arm → step a model RANK.
    _fail(eng)
    eng.decide("s3", "task")
    _fail(eng)
    m2, r2, prov2 = eng.decide("s4", "task")
    assert m2 == "glm"  # ladder exhausted → rank step cheap→mid
    assert r2 == "auto_escalation"
    assert "escalated_reasoning_arm" not in prov2  # a rank step carries no reasoning arm


def test_tier_step_resets_the_effort_arm() -> None:
    # Escalate qwen low→high (effort), then exhaust its ladder → step rank to glm. The stale
    # "high" arm from qwen must be cleared so glm starts at its OWN default arm, not a foreign id.
    eng = _engine(base_reasoning=_ladder())
    eng.decide("s0", "task")
    _fail(eng)
    eng.decide("s1", "task")
    _fail(eng)
    _m1, _r1, prov1 = eng.decide("s2", "task")
    assert prov1["escalated_reasoning_arm"] == "high"  # qwen effort at the top arm
    assert eng._task_effort_arm.get(task_state_key("repoA")) == "high"

    _fail(eng)
    eng.decide("s3", "task")
    _fail(eng)
    m2, r2, _prov2 = eng.decide("s4", "task")
    assert m2 == "glm"  # ladder exhausted → rank step
    assert r2 == "auto_escalation"
    # the stale qwen arm was cleared on the rank step
    assert task_state_key("repoA") not in eng._task_effort_arm


def test_model_without_arms_steps_tier_directly() -> None:
    eng = _engine(base_reasoning=None)  # qwen has no reasoning ladder
    eng.decide("s0", "task")
    _fail(eng)
    eng.decide("s1", "task")
    _fail(eng)
    m, r, prov = eng.decide("s2", "task")
    assert m == "glm"  # no effort headroom → straight to a rank step
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
    assert m2 == "qwen"  # still an effort step, NOT a rank jump — the ladder reset to default
    assert r2 == "auto_escalation"
    assert prov2["escalated_reasoning_arm"] == "high"  # steps low→high again, not high→(rank)


def _mixed_vocabulary_engine() -> RouterEngine:
    return RouterEngine(
        model_pool=_MixedVocabularyPool(),
        session_manager=SessionManager(),
        outcome_index=Index(),
        embedder=Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2, ladder="effort_then_rank"),
        task_key_resolver=lambda _s: "repoA",
    )


def test_foreign_effort_arm_resets_to_the_new_models_default() -> None:
    # A task that effort-escalated on qwen (arm "high") later routes to glm, whose arm vocabulary
    # is {nothink, think}. The persisted "high" is FOREIGN to glm: it must reset to glm's OWN
    # default so glm climbs ITS ladder — not report false headroom on an id that has no
    # `next_arm_above` (which used to void the directive, never retire the window, and deadlock
    # escalation before it ever reached the rank rung).
    eng = _mixed_vocabulary_engine()
    task = task_state_key("repoA")
    eng._task_effort_arm[task] = "high"  # qwen's arm, persisted from a prior qwen escalation

    idx, max_idx, cur_arm, _reasoning = eng._effort_ladder(task, "glm")
    assert cur_arm == "nothink"  # reset to glm's default, not the foreign "high"
    assert idx == 0
    assert max_idx == 1
    assert eng._task_effort_arm[task] == "nothink"


def test_escalation_survives_a_model_change_with_different_arm_vocabularies() -> None:
    # End-to-end at the decision point: qwen effort-escalated to "high" in a prior session, then
    # the SAME task routes to glm (a different model with a different arm vocabulary). Two fresh
    # same-key verified failures must escalate glm's OWN effort ladder (nothink → think) and retire
    # the window — never hang forever on the foreign arm with the failure log stuck at full.
    eng = _mixed_vocabulary_engine()
    task = task_state_key("repoA")
    eng._task_effort_arm[task] = "high"  # qwen's arm from the prior session

    _fail(eng)
    _fail(eng)
    m, r, prov = eng._maybe_escalate(task, "glm", "knn", {})
    assert m == "glm"  # same model on the effort rung (cache-safe) — NOT a voided deadlock
    assert r == "auto_escalation"
    assert prov["escalated_reasoning_arm"] == "think"  # glm's own ladder, from its default
    assert task not in eng._failure_log  # the window was retired — a fresh recurrence re-fires
