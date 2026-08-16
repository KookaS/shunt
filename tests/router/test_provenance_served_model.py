"""The persisted provenance must name the model that was SERVED, not the base pick.

`decision_provenance` is read back by `shunt explain` and by the off-policy export, so a row
claiming the pre-escalation model is a mislabelled record, not a cosmetic defect.
"""

from __future__ import annotations

from shunt.router.engine import RouterEngine, task_state_key
from shunt.router.escalation import EscalationAction, EscalationConfig, EscalationDirective
from shunt.router.provenance import build_provenance
from tests.router.escalation_fakes import (
    Embedder,
    Index,
    RankedReasoningPool,
    SessionManager,
    reasoning_ladder,
)

_TASK = "repoA"


def _engine(*, with_arms: bool) -> RouterEngine:
    """qwen < glm < opus by rank; qwen optionally carries a two-arm effort ladder."""
    return RouterEngine(
        model_pool=RankedReasoningPool(
            {
                "qwen": reasoning_ladder("low", "high") if with_arms else None,
                "glm": None,
                "opus": None,
            }
        ),
        session_manager=SessionManager(),
        outcome_index=Index(),
        embedder=Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2, ladder="effort_then_rank"),
        task_key_resolver=lambda _s: _TASK,
    )


def _fail(eng: RouterEngine, dedup_key: str = "t::a") -> None:
    eng.record_outcome(
        downshift=False,
        success=False,
        task_key=_TASK,
        dedup_key=dedup_key,
        exit_code=1,
        is_infra_failure=False,
        confirmed=True,
    )


def _escalate_once(eng: RouterEngine, sids: tuple[str, str]) -> tuple[str, str, dict[str, object]]:
    """Drive two verified same-check failures, returning the decision that escalates."""
    _fail(eng)
    eng.decide(sids[0], "task")  # holds after one failure
    _fail(eng)
    return eng.decide(sids[1], "task")


def test_rank_escalation_provenance_names_the_served_model() -> None:
    eng = _engine(with_arms=False)
    base, _, base_prov = eng.decide("s0", "task")
    assert base == "qwen"
    assert base_prov["model_chosen"] == "qwen"

    served, reason, prov = _escalate_once(eng, ("s1", "s2"))

    assert reason == "auto_escalation"
    assert served != base  # a rank step CHANGES the served model — the broken path
    assert prov["model_chosen"] == served
    assert prov["selection_rule_used"] == "auto_escalation"
    assert prov["auto_escalated"] is True


def test_effort_escalation_provenance_names_the_served_model() -> None:
    eng = _engine(with_arms=True)
    served, reason, prov = _escalate_once(eng, ("s1", "s2"))

    assert reason == "auto_escalation"
    assert served == "qwen"  # an effort step keeps the model (cache-safe)
    assert prov["escalated_reasoning_arm"] == "high"
    assert prov["model_chosen"] == "qwen"
    assert prov["selection_rule_used"] == "auto_escalation"


def test_floor_held_provenance_names_the_served_model() -> None:
    eng = _engine(with_arms=False)
    escalated, _, _ = _escalate_once(eng, ("s0", "s1"))
    assert eng._task_rank_floor[task_state_key(_TASK)] > 0

    # A later session re-picks the cheap model from the corpus; the floor re-serves the rung the
    # task already climbed, WITHOUT climbing a new one.
    served, reason, prov = eng.decide("s2", "task")

    assert reason == "escalation_floor"
    assert served == escalated
    assert prov["model_chosen"] == served
    assert prov["selection_rule_used"] == "escalation_floor"
    assert prov["auto_escalated"] is True


def test_non_policy_provenance_clears_a_stale_downshift() -> None:
    # The base pick may have been an exploratory downshift; the escalated turn never is. Left
    # stale, a verified pass on the escalated model banks ConservativeGate slack as evidence the
    # CHEAPER model works.
    base = build_provenance(
        model_chosen="qwen",
        selection_rule_used="exploration",
        downshift=True,
        router_propensity=0.3,
    )
    directive = EscalationDirective(EscalationAction.RAISE_RANK, "verified_failures")

    prov = RouterEngine._non_policy_provenance(base, directive, "glm", "auto_escalation")

    assert prov["model_chosen"] == "glm"
    assert prov["selection_rule_used"] == "auto_escalation"
    assert prov["downshift"] is False
    assert prov["router_propensity"] is None
