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
