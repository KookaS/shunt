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
# The rung the CONFIGURED ladder aims at from the cold-start model, not merely "something
# pricier": rank_shortlist=3 over the 3-model fake registry makes the first step rank 0 -> 1.
HIGHER_MODEL = "fake-mid"
TOP_MODEL = "fake-frontier"  # rank 2 — a legal model, but NOT the rung the config declares
REAL_REASON = "same_verified_failure_x2"

# The fake registry's rank order (tests/integrations/fake_registry.yaml, ordered by total list
# price) and the shipped escalation knobs. Constructed rather than loaded so the unit tests do
# not depend on a resolvable SHUNT_CONFIG_DIR; the container path loads the real thing.
LADDER = ae.LadderSpec(
    ranks={BASE_MODEL: 0, HIGHER_MODEL: 1, TOP_MODEL: 2},
    max_rank_index=2,
    escalate_after_n=2,
    rank_shortlist=3,
)

EXPLAIN_OK = f"Session:        s4\nModel chosen:   {HIGHER_MODEL}\nEscalation:     {REAL_REASON}"


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
    return ae._verdict_problems(db, MARKER, escalated, sessions, min_sessions=4, ladder=LADDER)


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


# ── specified behaviour: the threshold, the rung, the cadence, the read-out ───
#
# These are the checks that make the LIVE gate mean something. The escalate arm does not beat
# always-frontier on quality, so a live gate cannot be built around a performance win; what it
# can prove is that the shipped mechanism behaves the way the config says. Each test below is
# one store that satisfies (a)-(d) — a real escalation, backed by real failures — and violates
# exactly one specification, so a gate that only checked "did it escalate" would pass it.


def _store(tmp_path: Path, escalated_model: str, reason: str = REAL_REASON) -> Path:
    """(a)-(d)-clean store: 3 base sessions, 2 verified reds, one escalated 4th."""
    db = tmp_path / "outcomes.db"
    _make_db(
        db,
        [(f"s{i}", BASE_MODEL, {"auto_escalated": False}) for i in (1, 2, 3)]
        + [("s4", escalated_model, {"auto_escalated": True, "rank_escalation_reason": reason})],
        [("s1", 2, "failure"), ("s2", 2, "failure")],
    )
    return db


def test_threshold_other_than_the_configured_one_is_rejected(tmp_path: Path) -> None:
    """(e) firing at x3 when the policy declares escalate_after_n=2 is not the specified trigger."""
    db = _store(tmp_path, HIGHER_MODEL, reason="same_verified_failure_x3")
    assert any("declares escalate_after_n=2" in p for p in _problems(db))


def test_skipping_the_configured_rung_is_rejected(tmp_path: Path) -> None:
    """(f) rank_shortlist=3 aims at rank 1 from rank 0; jumping to the top rank is not that."""
    db = _store(tmp_path, TOP_MODEL)
    assert any("aims at rank 1" in p for p in _problems(db))


def test_a_model_outside_the_live_pool_is_rejected(tmp_path: Path) -> None:
    """(f) a served model the deployment does not rank is not on the ladder at all."""
    db = _store(tmp_path, "some-model-not-in-the-registry")
    assert any("not in the deployment's live model pool" in p for p in _problems(db))


def test_the_aimed_rung_is_clamped_to_the_top_of_the_pool() -> None:
    """A shortlist wider than the pool must not aim at a rank the pool cannot serve."""
    # Measured on the $0 free-tier rig: 2 models, rank_shortlist=3, base at rank 1 — the raw
    # shortlist arithmetic aims at rank 2, which does not exist, and the check reported a ladder
    # violation against a phantom rung.
    narrow = ae.LadderSpec(
        ranks={"a": 0, "b": 1}, max_rank_index=1, escalate_after_n=2, rank_shortlist=3
    )
    assert narrow.expected_rung("a") == 1
    assert narrow.expected_rung("b") == 1


def test_a_pool_with_no_rung_above_the_base_is_named_as_such(tmp_path: Path) -> None:
    """The $0 rig's real state: the base sits at the top rank, so no step is possible."""
    db = _store(tmp_path, TOP_MODEL)
    sessions = ae._marker_sessions(db, MARKER)
    escalated = ae._escalated(db, MARKER)
    assert escalated is not None
    # A pool whose top rank IS the base model's rank.
    exhausted = ae.LadderSpec(
        ranks={BASE_MODEL: 0}, max_rank_index=0, escalate_after_n=2, rank_shortlist=3
    )
    problems = ae._ladder_problems(escalated, sessions, exhausted)
    assert any("already at the pool's TOP rank" in p for p in problems)


def test_a_floor_hold_after_a_step_is_not_a_second_step(tmp_path: Path) -> None:
    """`escalation_floor` is stamped auto_escalated too — it holds a rung, it does not take one.

    A healthy live run is exactly one step followed by floor holds on every later session;
    counting those as steps failed the cadence check on a run that behaved correctly.
    """
    db = tmp_path / "outcomes.db"
    step = {"auto_escalated": True, "rank_escalation_reason": REAL_REASON}
    floor = {"auto_escalated": True, "rank_escalation_reason": "escalation_floor"}
    _make_db(
        db,
        [(f"s{i}", BASE_MODEL, {"auto_escalated": False}) for i in (1, 2, 3)]
        + [("s4", HIGHER_MODEL, step), ("s5", HIGHER_MODEL, floor), ("s6", HIGHER_MODEL, floor)],
        [("s1", 2, "failure"), ("s2", 2, "failure")],
    )
    assert _problems(db) == []


def test_back_to_back_escalations_are_rejected(tmp_path: Path) -> None:
    """(g) the window an escalation consumed is retired, so the next session cannot escalate."""
    db = tmp_path / "outcomes.db"
    escalated = {"auto_escalated": True, "rank_escalation_reason": REAL_REASON}
    _make_db(
        db,
        [(f"s{i}", BASE_MODEL, {"auto_escalated": False}) for i in (1, 2)]
        + [("s3", HIGHER_MODEL, escalated), ("s4", HIGHER_MODEL, escalated)],
        [("s1", 2, "failure"), ("s2", 2, "failure")],
    )
    assert any("consecutive escalated sessions" in p for p in _problems(db))


def test_an_effort_rung_is_exempt_from_rank_conformance(tmp_path: Path) -> None:
    """(f) a cache-safe effort step keeps the model on purpose — rank conformance cannot apply."""
    session = {
        "session_id": "s4",
        "model_chosen": BASE_MODEL,
        "provenance": {
            "auto_escalated": True,
            "rank_escalation_reason": REAL_REASON,
            "escalated_reasoning_arm": "high",
        },
    }
    sessions = [
        {"session_id": f"s{i}", "model_chosen": BASE_MODEL, "provenance": {}} for i in (1, 2, 3)
    ] + [session]
    assert ae._ladder_problems(session, sessions, LADDER) == []


def _effort_session(served: str, before: str | None) -> dict[str, Any]:
    prov: dict[str, Any] = {
        "auto_escalated": True,
        "rank_escalation_reason": REAL_REASON,
        "escalated_reasoning_arm": "high",
        "escalation_exploration": {"action": "raise_effort", "propensity": 1.0},
    }
    if before is not None:
        prov["pre_escalation_model"] = before
    return {"session_id": "s4", "model_chosen": served, "provenance": prov}


def test_an_effort_rung_that_moved_the_model_is_rejected() -> None:
    """(i) the defect shape: the tool exited 0 while the effort rung became a rank jump."""
    problems = ae._effort_rung_problems(_effort_session(HIGHER_MODEL, BASE_MODEL))
    joined = " | ".join(problems)
    assert HIGHER_MODEL in joined and BASE_MODEL in joined  # both models named
    assert "raise_effort" in joined
    assert len(problems) == 1


def test_an_effort_rung_that_kept_the_model_passes() -> None:
    assert ae._effort_rung_problems(_effort_session(BASE_MODEL, BASE_MODEL)) == []


def test_an_effort_rung_with_no_pre_escalation_model_is_rejected() -> None:
    """An unfalsifiable claim is not a passing one: with nothing to compare, the check fails."""
    problems = ae._effort_rung_problems(_effort_session(BASE_MODEL, None))
    assert any("pre_escalation_model" in p for p in problems)


def test_a_rank_rung_is_untouched_by_the_effort_cross_check() -> None:
    session = _effort_session(HIGHER_MODEL, BASE_MODEL)
    session["provenance"]["escalation_exploration"]["action"] = "raise_rank"
    assert ae._effort_rung_problems(session) == []


def test_the_effort_cross_check_runs_inside_the_verdict(tmp_path: Path) -> None:
    """Wired, not merely defined: the defect shape must redden `_verdict_problems` itself."""
    db = tmp_path / "outcomes.db"
    escalated = {
        "selection_rule_used": "auto_escalation",
        "auto_escalated": True,
        "rank_escalation_reason": REAL_REASON,
        "model_chosen": HIGHER_MODEL,
        "pre_escalation_model": BASE_MODEL,
        "escalated_reasoning_arm": "high",
        "escalation_exploration": {"action": "raise_effort", "propensity": 1.0},
    }
    _make_db(
        db,
        [(f"s{i}", BASE_MODEL, {"auto_escalated": False}) for i in (1, 2, 3)]
        + [("s4", HIGHER_MODEL, escalated)],
        [("s1", 2, "failure"), ("s2", 2, "failure")],
    )
    assert any("became a jump onto a different model" in p for p in _problems(db))


@pytest.mark.parametrize(
    "text",
    [
        f"Session: s4\nModel chosen:   {BASE_MODEL}\nEscalation:     {REAL_REASON}",
        "Session: s4\nEscalation:     same_verified_failure_x2",
    ],
)
def test_explain_naming_anything_but_the_escalated_model_is_rejected(text: str) -> None:
    """(h) the regression this exists for: `explain` reporting the BASE model after a step."""
    assert ae.explain_problems(text, HIGHER_MODEL) != []


def test_explain_naming_the_escalated_model_passes() -> None:
    assert ae.explain_problems(EXPLAIN_OK, HIGHER_MODEL) == []


# ── the entrypoint, end to end ────────────────────────────────────────────────


def _run_main(db: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setenv("SHUNT_ESC_TOOL", "curl")
    monkeypatch.setenv("SHUNT_ESC_MARKER", MARKER)
    monkeypatch.setenv("SHUNT_DATA_DIR", str(db.parent))
    monkeypatch.setenv("SHUNT_ESC_TIMEOUT", "1")
    monkeypatch.setenv("SHUNT_ESC_MIN_SESSIONS", "4")
    monkeypatch.delenv("FAKE_UPSTREAM_RECORD_DIR", raising=False)
    monkeypatch.setattr(ae, "_explain", lambda session_id: EXPLAIN_OK)
    # The container resolves this from the deployment's own config dir; here it is pinned to the
    # fake registry's ladder so the test asserts the verdict, not the packaged registry.
    monkeypatch.setattr(ae, "load_ladder_spec", lambda: LADDER)
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


def test_main_fails_when_the_deployment_ladder_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable policy makes conformance unverifiable — that must FAIL, never skip."""

    def _boom() -> ae.LadderSpec:
        raise RuntimeError("no registry")

    db = _genuine(tmp_path)
    monkeypatch.setattr(ae, "load_ladder_spec", _boom)
    monkeypatch.setenv("SHUNT_ESC_TOOL", "curl")
    monkeypatch.setenv("SHUNT_ESC_MARKER", MARKER)
    monkeypatch.setenv("SHUNT_DATA_DIR", str(db.parent))
    monkeypatch.setenv("SHUNT_ESC_TIMEOUT", "1")
    monkeypatch.delenv("FAKE_UPSTREAM_RECORD_DIR", raising=False)
    monkeypatch.setattr(ae.sys, "argv", ["assert_escalation.py"])
    assert ae.main() == 1
