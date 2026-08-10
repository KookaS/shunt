"""The escalation verdict must not be satisfiable by a forged boolean."""

# The defect this guards: the whole pass condition used to be
# `provenance["auto_escalated"] is True`. A hand-forged two-table DB — ONE session, the
# cold-start model unchanged, ZERO rows in outcome_events, and that boolean set — printed
# "OK: … verified failures behind it: 0" and exited 0. The sidecar runs in a container under
# run_scenario.sh, so nothing in the normal suite would have reddened. These tests are that
# missing wall: the same forged store, asserted to FAIL, and a genuine one asserted to PASS.

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tests.integrations import assert_escalation as ae

MARKER = "SHUNT-ESC-curl"
BASE_MODEL = "qwen3.7-plus"  # the router's cold-start pick (src/shunt/proxy/router.py)
HIGHER_MODEL = "fake-frontier"
REAL_REASON = "same_verified_failure_x2"


def _make_db(
    path: Path,
    sessions: list[tuple[str, str, dict[str, Any]]],
    failures: list[tuple[str, int, str]] | None = None,
) -> None:
    """A store with only the two tables the verdict reads — a forger's minimum."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, prompt_text TEXT, "
        "model_chosen TEXT, decision_provenance TEXT)"
    )
    conn.execute(
        "CREATE TABLE outcome_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT, tier INTEGER, outcome TEXT, tombstoned INTEGER DEFAULT 0)"
    )
    for session_id, model, prov in sessions:
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?)",
            (session_id, f"{MARKER} prompt — the test suite is failing", model, json.dumps(prov)),
        )
    for session_id, tier, outcome in failures or []:
        conn.execute(
            "INSERT INTO outcome_events (session_id, tier, outcome) VALUES (?, ?, ?)",
            (session_id, tier, outcome),
        )
    conn.commit()
    conn.close()


def _genuine(tmp_path: Path) -> Path:
    """What Shunt actually writes: 4 sessions, 3 base + 1 escalated, 2 verified failures."""
    db = tmp_path / "outcomes.db"
    sessions: list[tuple[str, str, dict[str, Any]]] = [
        (f"s{i}", BASE_MODEL, {"selection_rule_used": "knn", "auto_escalated": False})
        for i in (1, 2, 3)
    ]
    sessions.append(
        (
            "s4",
            HIGHER_MODEL,
            {
                "selection_rule_used": "auto_escalation",
                "auto_escalated": True,
                "rank_escalation_reason": REAL_REASON,
                "model_chosen": HIGHER_MODEL,
            },
        )
    )
    _make_db(db, sessions, [("s1", 2, "failure"), ("s2", 2, "failure")])
    return db


def _forged(tmp_path: Path) -> Path:
    """The audit's probe, verbatim: one session, model unchanged, no outcome_events."""
    db = tmp_path / "outcomes.db"
    _make_db(
        db,
        [
            (
                "s1",
                BASE_MODEL,
                {"auto_escalated": True, "rank_escalation_reason": "totally_made_up"},
            )
        ],
    )
    return db


def _problems(db: Path) -> list[str]:
    sessions = ae._marker_sessions(db, MARKER)
    escalated = ae._escalated(db, MARKER)
    assert escalated is not None, "fixture must contain a session claiming an escalation"
    return ae._verdict_problems(db, MARKER, escalated, sessions, min_sessions=4)


# ── the teeth ─────────────────────────────────────────────────────────────────


def test_forged_boolean_is_rejected(tmp_path: Path) -> None:
    problems = _problems(_forged(tmp_path))
    joined = " | ".join(problems)
    assert "rank_escalation_reason" in joined  # (d) any-string reason
    assert "Tier-2 failure" in joined  # (a) no verified failures behind it
    assert "marker session" in joined  # (b) one session, not four
    assert "pre-escalation model" in joined  # (c) nothing to compare against
    assert len(problems) == 4


def test_genuine_escalation_passes(tmp_path: Path) -> None:
    assert _problems(_genuine(tmp_path)) == []


# ── one failure at a time: each assertion must be load-bearing on its own ──────


def test_unchanged_served_model_is_rejected(tmp_path: Path) -> None:
    """The escalation claim with the cold-start model still served is not an escalation."""
    db = tmp_path / "outcomes.db"
    prov = {"auto_escalated": True, "rank_escalation_reason": REAL_REASON}
    _make_db(
        db,
        [(f"s{i}", BASE_MODEL, {"auto_escalated": False}) for i in (1, 2, 3)]
        + [("s4", BASE_MODEL, prov)],
        [("s1", 2, "failure"), ("s2", 2, "failure")],
    )
    assert any("unchanged from the pre-escalation model" in p for p in _problems(db))


def test_missing_verified_failures_is_rejected(tmp_path: Path) -> None:
    """The reason names a threshold; the store must hold that many rerun-confirmed reds."""
    db = tmp_path / "outcomes.db"
    _make_db(
        db,
        [(f"s{i}", BASE_MODEL, {"auto_escalated": False}) for i in (1, 2, 3)]
        + [("s4", HIGHER_MODEL, {"auto_escalated": True, "rank_escalation_reason": REAL_REASON})],
        [("s1", 2, "failure")],  # one, but the reason claims two
    )
    assert any("need >= 2" in p for p in _problems(db))


def test_tombstoned_and_tier1_failures_do_not_count(tmp_path: Path) -> None:
    """Only rerun-confirmed, un-retracted Tier-2 reds are evidence."""
    db = tmp_path / "outcomes.db"
    _make_db(
        db,
        [(f"s{i}", BASE_MODEL, {"auto_escalated": False}) for i in (1, 2, 3)]
        + [("s4", HIGHER_MODEL, {"auto_escalated": True, "rank_escalation_reason": REAL_REASON})],
        [("s1", 2, "failure"), ("s2", 1, "failure")],
    )
    conn = sqlite3.connect(db)
    conn.execute("UPDATE outcome_events SET tombstoned = 1 WHERE tier = 2")
    conn.commit()
    conn.close()
    assert ae._verified_failures(db, MARKER) == 0
    assert any("only 0 rerun-confirmed" in p for p in _problems(db))


def test_failures_from_other_traffic_do_not_count(tmp_path: Path) -> None:
    """A red left by a session outside this run cannot stand in for one this run produced."""
    db = tmp_path / "outcomes.db"
    _make_db(
        db,
        [(f"s{i}", BASE_MODEL, {"auto_escalated": False}) for i in (1, 2, 3)]
        + [("s4", HIGHER_MODEL, {"auto_escalated": True, "rank_escalation_reason": REAL_REASON})],
        [("s1", 2, "failure"), ("foreign", 2, "failure")],
    )
    assert ae._verified_failures(db) == 2
    assert ae._verified_failures(db, MARKER) == 1
    assert any("need >= 2" in p for p in _problems(db))


def test_too_few_sessions_is_rejected(tmp_path: Path) -> None:
    """The session-reuse defect: fewer sessions than prompts must not read as a pass."""
    db = tmp_path / "outcomes.db"
    _make_db(
        db,
        [("s1", BASE_MODEL, {"auto_escalated": False})]
        + [("s2", HIGHER_MODEL, {"auto_escalated": True, "rank_escalation_reason": REAL_REASON})],
        [("s1", 2, "failure"), ("s1", 2, "failure")],
    )
    assert any("need >= 4" in p for p in _problems(db))


@pytest.mark.parametrize("reason", ["exploration_untested", "safe_fallback", "", None])
def test_fallback_chain_reasons_are_not_escalation_reasons(
    tmp_path: Path, reason: str | None
) -> None:
    """`rank_escalation_reason` is written by the fallback chain too — those are not this."""
    db = tmp_path / "outcomes.db"
    _make_db(
        db,
        [(f"s{i}", BASE_MODEL, {"auto_escalated": False}) for i in (1, 2, 3)]
        + [("s4", HIGHER_MODEL, {"auto_escalated": True, "rank_escalation_reason": reason})],
        [("s1", 2, "failure"), ("s2", 2, "failure")],
    )
    assert any("not the verified-failure reason" in p for p in _problems(db))


# ── the entrypoint, end to end ────────────────────────────────────────────────


def _run_main(db: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setenv("SHUNT_ESC_TOOL", "curl")
    monkeypatch.setenv("SHUNT_ESC_MARKER", MARKER)
    monkeypatch.setenv("SHUNT_DATA_DIR", str(db.parent))
    monkeypatch.setenv("SHUNT_ESC_TIMEOUT", "1")
    monkeypatch.setenv("SHUNT_ESC_MIN_SESSIONS", "4")
    monkeypatch.delenv("FAKE_UPSTREAM_RECORD_DIR", raising=False)
    monkeypatch.setattr(ae, "_explain", lambda session_id: f"Escalation: yes ({session_id})")
    monkeypatch.setattr(ae.sys, "argv", ["assert_escalation.py"])
    return ae.main()


def test_main_exits_nonzero_on_the_forged_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run_main(_forged(tmp_path), monkeypatch) != 0


def test_main_exits_zero_on_a_genuine_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run_main(_genuine(tmp_path), monkeypatch) == 0
