"""Escalation INFO lines must name the session and the model, in a grep-friendly shape.

Every rung now logs `escalation: session=<id> model=<name> ... reason=<reason>`.
"""

from __future__ import annotations

import logging

import pytest

from shunt.router.engine import RouterEngine, task_state_key
from shunt.router.escalation import EscalationAction, EscalationConfig, EscalationDirective
from tests.router.escalation_fakes import (
    Embedder,
    Index,
    RankedReasoningPool,
    SessionManager,
    reasoning_ladder,
)


def _engine() -> RouterEngine:
    return RouterEngine(
        model_pool=RankedReasoningPool({"qwen": reasoning_ladder("low", "high"), "glm": None}),
        session_manager=SessionManager(),
        outcome_index=Index(),
        embedder=Embedder(),
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


def test_effort_rung_logs_session_model_and_arm(caplog: pytest.LogCaptureFixture) -> None:
    eng = _engine()
    task = task_state_key("repoA")
    directive = EscalationDirective(
        EscalationAction.RAISE_EFFORT, "two_consecutive_failures", new_label_window=True
    )
    with caplog.at_level(logging.INFO, logger="shunt.router.engine"):
        result = eng._apply_effort(task, "s-123", "qwen", "knn", {}, directive, "low")
    assert result is not None
    assert (
        "escalation: session=s-123 model=qwen arm=low -> arm=high "
        "reason=two_consecutive_failures" in caplog.text
    )


def test_rank_rung_logs_session_and_both_models(caplog: pytest.LogCaptureFixture) -> None:
    eng = _engine()
    task = task_state_key("repoA")
    directive = EscalationDirective(
        EscalationAction.RAISE_RANK, "two_consecutive_failures", new_label_window=True
    )
    with caplog.at_level(logging.INFO, logger="shunt.router.engine"):
        result = eng._apply_rank(task, "s-456", "qwen", "knn", {}, directive)
    assert result is not None
    assert result[0] == "glm"
    assert (
        "escalation: session=s-456 model=qwen -> model=glm "
        "reason=two_consecutive_failures" in caplog.text
    )


def test_decision_threads_the_session_id_into_the_escalation_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    eng = _engine()
    eng.decide("s1", "task")
    _fail(eng)
    eng.decide("s2", "task")
    _fail(eng)
    with caplog.at_level(logging.INFO, logger="shunt.router.engine"):
        model, reason, _ = eng.decide("s3", "task")
    assert reason == "auto_escalation"
    assert model == "qwen"  # effort rung: same model, higher reasoning arm
    assert "escalation: session=s3 model=qwen arm=low -> arm=high" in caplog.text
