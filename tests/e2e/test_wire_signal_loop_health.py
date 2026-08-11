# The unit suite covers WireSignalCollector, the wire_tier1 quarantine, and loop-health
# each in isolation. This file drives the ASSEMBLED served app over real HTTP (TestClient)
# with the canned upstream + fake embedder (no spend, no ONNX), and asserts the collector's
# signals land on the session, the wire_tier1 outcome is derived and recorded (quarantined),
# and the /admin/loop-health payload reflects the driven sessions without leaking prompt text.
"""Wire-signal collector + loop-health telemetry through the served app's full request path."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from shunt.proxy.wire_signals import (
    WIRE_TERMINAL_STOP,
    WIRE_TOOL_ERROR_COUNT,
    derive_wire_tier1_outcome,
)
from tests.e2e.helpers import close_session, make_repo, poll_until, wait_outcome

MESSAGES_PATH = "/v1/messages"

# A distinctive, NON-secret-shaped token planted in the routed prompt. Every segment is
# under 12 chars so redact_secrets (proxy/redaction.py L30) does not blank it at write
# time — if it ever appears in the loop-health payload the endpoint leaked raw prompt text.
_PROMPT_MARKER = "wire-e2e marker 9c2f"

# The weak Tier-1 confidence a derived wire prior is recorded with (wire_signals.py L21).
_WIRE_CONFIDENCE = 0.3


def _messages_body(*, error: bool = False, content: str = "Fix the build") -> dict[str, object]:
    """An Anthropic /v1/messages body; *error* carries a flagged tool_result in history."""
    if error:
        return {
            "messages": [
                {"role": "user", "content": "run the tests"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "is_error": True,
                            "content": "pytest: 3 failed",
                        }
                    ],
                },
            ]
        }
    return {"messages": [{"role": "user", "content": content}]}


def _post_messages(client: TestClient, body: dict[str, object], *, user_agent: str) -> object:
    """POST an Anthropic turn; each distinct user_agent maps to a distinct session."""
    resp = client.post(MESSAGES_PATH, json=body, headers={"User-Agent": user_agent})
    assert resp.status_code == 200, resp.text
    return resp


def _session_id(resp: object) -> str:
    return resp.headers["X-Shunt-Session-Id"]  # type: ignore[attr-defined]


def _session_metadata(client: TestClient, session_id: str) -> dict[str, object]:
    """A copy of the in-memory session's metadata — where the collector accumulates signals."""
    session = client.app.state.session_manager.get_session(session_id)
    assert session is not None, f"session {session_id} not open"
    return dict(session.metadata)


def _event_rows(client: TestClient, session_id: str) -> list[dict[str, object]]:
    """The append-only outcome_events rows for *session_id*, in event order."""
    store = client.app.state.outcome_store
    with store._lock:  # noqa: SLF001
        rows = store._conn.execute(  # noqa: SLF001
            "SELECT tier, source, outcome FROM outcome_events "
            "WHERE session_id = ? ORDER BY event_id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _wait_for_wire_tier1(client: TestClient, session_id: str, outcome: str) -> None:
    """Poll until the session's capture wrote exactly its weak Tier-1 wire prior."""
    poll_until(
        lambda: (
            _event_rows(client, session_id)
            == [{"tier": 1, "source": "wire_tier1", "outcome": outcome}]
        ),
        what=f"wire_tier1 {outcome!r} event for session {session_id[:8]}",
    )


def test_error_turn_collects_wire_tool_error_signal(
    app_factory: Callable[..., TestClient],
) -> None:
    """A tool_result flagged is_error in the request history lands on session.metadata."""
    with app_factory(repo=None) as client:
        resp = _post_messages(client, _messages_body(error=True), user_agent="wire-e2e/error")
        meta = _session_metadata(client, _session_id(resp))

        # route_messages observes tool errors on the ORIGINAL body (router.py L660); the
        # canned upstream answers with finish_reason "stop", normalized to "end_turn"
        # (router.py L512-517) and observed as the terminal stop (router.py L676).
        assert meta[WIRE_TOOL_ERROR_COUNT] == 1
        assert meta[WIRE_TERMINAL_STOP] == "end_turn"
        # A structured error outranks the clean-close prior (wire_signals.py L78-81).
        assert derive_wire_tier1_outcome(meta) == ("failure", _WIRE_CONFIDENCE)


def test_normal_turn_collects_terminal_stop_only(
    app_factory: Callable[..., TestClient],
) -> None:
    """A plain terminal turn records the stop_reason and never invents an error signal."""
    with app_factory(repo=None) as client:
        resp = _post_messages(
            client,
            _messages_body(content=_PROMPT_MARKER),
            user_agent="wire-e2e/terminal",
        )
        meta = _session_metadata(client, _session_id(resp))

        assert WIRE_TOOL_ERROR_COUNT not in meta
        assert meta[WIRE_TERMINAL_STOP] == "end_turn"
        derived = derive_wire_tier1_outcome(meta)
        assert derived == ("weak_success", _WIRE_CONFIDENCE)


def test_wire_tier1_derived_recorded_and_quarantined_through_served_path(
    app_factory: Callable[..., TestClient],
) -> None:
    """Closing driven sessions writes weak Tier-1 wire priors — quarantined, never trusted."""
    with app_factory(repo=None) as client:
        err_sid = _session_id(
            _post_messages(client, _messages_body(error=True), user_agent="wire-e2e/err")
        )
        term_sid = _session_id(
            _post_messages(
                client, _messages_body(content=_PROMPT_MARKER), user_agent="wire-e2e/term"
            )
        )
        close_session(client, err_sid)
        close_session(client, term_sid)
        _wait_for_wire_tier1(client, err_sid, "failure")
        _wait_for_wire_tier1(client, term_sid, "weak_success")

        # The coordinator derives the weak prior from the accumulated signals and records
        # source="wire_tier1" (coordinator.py L212-232) — observability, not trust.
        assert _event_rows(client, err_sid) == [
            {"tier": 1, "source": "wire_tier1", "outcome": "failure"}
        ]
        assert _event_rows(client, term_sid) == [
            {"tier": 1, "source": "wire_tier1", "outcome": "weak_success"}
        ]
        # Quarantined: no materialized outcome row, no verified count, not a neighbour.
        store = client.app.state.outcome_store
        assert store.get_outcome(err_sid) is None
        assert store.get_outcome(term_sid) is None
        assert store.count_verified_outcomes() == 0


def test_loop_health_payload_reflects_sessions_no_pii_read_only(
    app_factory: Callable[..., TestClient],
) -> None:
    """The payload aggregates the driven sessions, stays aggregate-only, and is GET-only."""
    with app_factory(repo=None) as client:
        err_sid = _session_id(
            _post_messages(client, _messages_body(error=True), user_agent="wire-e2e/err")
        )
        term_sid = _session_id(
            _post_messages(
                client, _messages_body(content=_PROMPT_MARKER), user_agent="wire-e2e/term"
            )
        )
        close_session(client, err_sid)
        close_session(client, term_sid)
        _wait_for_wire_tier1(client, err_sid, "failure")

        store = client.app.state.outcome_store
        stored_prompt = store.get_session(term_sid)["prompt_text"]
        assert _PROMPT_MARKER in stored_prompt  # the marker is really in the store

        resp = client.get("/admin/loop-health")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) >= {
            "label_coverage",
            "propensity_support",
            "routing_collapse",
            "cost",
            "support_deficient_models",
        }
        assert body["label_coverage"]["total_sessions"] == 2
        # The wire-tier1 priors are quarantined by design (store.py L382-388): they live in
        # the append-only log but never materialize an `outcomes` row, so they do NOT inflate
        # label coverage — only a verified Tier-2 does (see the green-repo test below).
        assert body["label_coverage"]["any_labeled"] == 0
        assert body["label_coverage"]["labeled_coverage"] == 0.0

        raw = json.dumps(body)
        assert "prompt_text" not in raw
        assert _PROMPT_MARKER not in raw
        # A mutating verb on the read-only endpoint is rejected, not silently accepted.
        assert client.post("/admin/loop-health").status_code == 405


def test_verified_tier2_drives_label_coverage(
    app_factory: Callable[..., TestClient], tmp_path: Path
) -> None:
    """A session verified against a green repo raises loop-health coverage via the served path."""
    repo = make_repo(tmp_path / "green", kind="green")
    with app_factory(repo=repo) as client:
        resp = _post_messages(
            client,
            _messages_body(content=_PROMPT_MARKER),
            user_agent="wire-e2e/verified",
        )
        sid = _session_id(resp)
        # The wire signal is still collected on the served path even though the off-wire
        # verifier resolves a repo and writes a verified Tier-2 label instead.
        assert _session_metadata(client, sid)[WIRE_TERMINAL_STOP] == "end_turn"

        close_session(client, sid)
        wait_outcome(client, sid, "success")

        store = client.app.state.outcome_store
        row = store.get_outcome(sid)
        assert row is not None
        assert row["outcome_source"] == "auto_tier2"
        assert store.count_verified_outcomes() == 1

        body = client.get("/admin/loop-health").json()
        coverage = body["label_coverage"]
        assert coverage["total_sessions"] >= 1
        assert coverage["any_labeled"] >= 1
        assert coverage["labeled_coverage"] > 0.0
        assert coverage["verified_coverage"] > 0.0


def test_reward_independent_collapse_alarm_fires_when_one_arm_dominates(
    app_factory: Callable[..., TestClient],
) -> None:
    """Repeated sessions to one cheap arm collapse choice entropy — the alarm fires anyway."""
    with app_factory(repo=None) as client:
        for i in range(6):
            _post_messages(
                client,
                _messages_body(content=f"{_PROMPT_MARKER} run {i}"),
                user_agent=f"wire-e2e/collapse/{i}",
            )
        body = client.get("/admin/loop-health").json()
        collapse = body["routing_collapse"]
        assert collapse["window_size"] == 6
        assert collapse["distinct_models"] == 1
        assert collapse["choice_entropy"] == 0.0
        assert collapse["entropy_collapse_alarm"] is True
        assert collapse["alarm"] is True
