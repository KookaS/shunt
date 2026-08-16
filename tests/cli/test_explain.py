"""`shunt explain` on a planted row: a fallback session prints the SERVED model and the fallback.

The server's persist path corrects a fallback session's provenance so model_chosen
names the served model and fallback_chain_triggered is True; explain prints both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from shunt.cli import _explain
from shunt.db.store import OutcomeStore


def _fallback_provenance(served: str) -> dict[str, object]:
    # The shape `_store_session_with_provenance` persists after its fallback
    # correction: model_chosen == served model, fallback flagged.
    return {
        "model_chosen": served,
        "selection_rule_used": "cold_start",
        "fallback_chain_triggered": True,
        "router_propensity": 1.0,
    }


def test_explain_names_served_model_and_flags_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path))
    store = OutcomeStore()
    store.store_session(
        session_id="sess-fallback",
        prompt_text="hi",
        embedding=None,
        model_chosen="fake-mid",
        cost=1.0,
        cache_stats={},
        duration=1.0,
        decision_provenance=_fallback_provenance("fake-mid"),
    )
    store.close()

    _explain(argparse.Namespace(session_id="sess-fallback"))

    out = capsys.readouterr().out
    # Model chosen names the model that actually served, and the fallback is surfaced.
    assert "Model chosen:   fake-mid" in out
    assert "Fallback:       yes" in out


def _escalated_provenance(ladder: str) -> tuple[str, dict[str, object]]:
    """Drive the REAL engine to an escalation and return (served model, its provenance)."""
    # Recorded live defect: `explain` reported the BASE model for an escalated decision. The
    # provenance is therefore taken from the engine itself rather than hand-built — a fixture
    # that restates the expected shape cannot catch the engine writing the wrong one.
    from shunt.models.config import ReasoningArm, ReasoningConfig
    from shunt.router.engine import RouterEngine
    from shunt.router.escalation import EscalationConfig
    from tests.router.escalation_fakes import (
        EchoSessionManager,
        Embedder,
        Index,
        RankedReasoningPool,
    )

    arms = ReasoningConfig(
        default_arm="low",
        arms=[
            ReasoningArm(id="low", rank=0, api={"reasoning_effort": "low"}),
            ReasoningArm(id="high", rank=1, api={"reasoning_effort": "high"}),
        ],
    )
    engine = RouterEngine(
        model_pool=RankedReasoningPool({"cheap": arms, "pricey": arms}),
        session_manager=EchoSessionManager(),
        outcome_index=Index(),
        embedder=Embedder(),
        escalation=EscalationConfig(enabled=True, escalate_after_n=2, ladder=ladder),
        task_key_resolver=lambda _s: "/repo",
    )
    engine.decide("s0", "task")
    for _ in range(2):
        engine.record_outcome(
            downshift=False,
            success=False,
            task_key="/repo",
            dedup_key="t::a",
            exit_code=1,
            is_infra_failure=False,
            confirmed=True,
        )
    model, reason, provenance = engine.decide("s1", "task")
    assert reason == "auto_escalation"
    return model, provenance


def _explain_stored(
    tmp_path: Path, model: str, provenance: dict[str, object], session_id: str
) -> None:
    store = OutcomeStore()
    store.store_session(
        session_id=session_id,
        prompt_text="hi",
        embedding=None,
        model_chosen=model,
        cost=1.0,
        cache_stats={},
        duration=1.0,
        decision_provenance=provenance,
    )
    store.close()
    _explain(argparse.Namespace(session_id=session_id))


def test_explain_names_the_escalated_model_not_the_base_pick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path))
    model, provenance = _escalated_provenance("rank_only")
    assert model == "pricey"  # the rank rung was actually climbed off the base pick

    _explain_stored(tmp_path, model, provenance, "sess-rank-escalated")

    out = capsys.readouterr().out
    assert "Model chosen:   pricey" in out  # the SERVED model, never the base "cheap"
    assert "Model chosen:   cheap" not in out
    assert "Escalation:     same_verified_failure_x2" in out
    assert "Selection rule: auto_escalation" in out


def test_explain_surfaces_an_effort_escalation_that_kept_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The cache-safe rung: same model, higher reasoning arm. `Model chosen` cannot show it,
    # so without the arm line this session explains as the un-escalated base pick.
    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path))
    model, provenance = _escalated_provenance("effort_then_rank")
    assert model == "cheap"

    _explain_stored(tmp_path, model, provenance, "sess-effort-escalated")

    out = capsys.readouterr().out
    assert "Model chosen:   cheap" in out
    assert "Reasoning arm:  high  (escalated)" in out
    assert "Escalation:     same_verified_failure_x2" in out
