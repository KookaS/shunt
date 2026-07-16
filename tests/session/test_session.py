"""Tests for session module — creation, identity, timeout, state transitions."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from shunt.session import SessionManager, SessionState


def test_create_session_has_uuid() -> None:
    mgr = SessionManager()
    session = mgr.create_session("tool-abc")
    assert len(session.session_id) == 36  # UUID4 with hyphens
    assert session.session_id.count("-") == 4
    assert session.state == SessionState.open
    assert session.tool_identity == "tool-abc"
    assert isinstance(session.start_time, datetime)


def test_get_session_returns_none_for_missing() -> None:
    mgr = SessionManager()
    assert mgr.get_session("nonexistent") is None


def test_get_session_by_id() -> None:
    mgr = SessionManager()
    session = mgr.create_session("tool-abc")
    assert mgr.get_session(session.session_id) is session


def test_compute_tool_identity_deterministic() -> None:
    a = SessionManager.compute_tool_identity("1.2.3.4", "tool/v1")
    b = SessionManager.compute_tool_identity("1.2.3.4", "tool/v1")
    assert a == b


def test_compute_tool_identity_changes_on_ip() -> None:
    a = SessionManager.compute_tool_identity("1.2.3.4", "tool/v1")
    b = SessionManager.compute_tool_identity("5.6.7.8", "tool/v1")
    assert a != b


def test_compute_tool_identity_changes_on_ua() -> None:
    a = SessionManager.compute_tool_identity("1.2.3.4", "tool/v1")
    b = SessionManager.compute_tool_identity("1.2.3.4", "tool/v2")
    assert a != b


def test_find_or_create_reuses_open_session() -> None:
    mgr = SessionManager()
    s1 = mgr.find_or_create("tool-abc")
    s2 = mgr.find_or_create("tool-abc")
    assert s1 is s2


def test_find_or_create_creates_new_after_close() -> None:
    mgr = SessionManager()
    s1 = mgr.find_or_create("tool-abc")
    mgr.close_session(s1.session_id)
    s2 = mgr.find_or_create("tool-abc")
    assert s2 is not s1
    assert s2.tool_identity == "tool-abc"
    assert s2.state == SessionState.open


def test_get_session_by_identity() -> None:
    mgr = SessionManager()
    s1 = mgr.create_session("tool-abc")
    s2 = mgr.create_session("tool-def")
    assert mgr.get_session_by_identity("tool-abc") is s1
    assert mgr.get_session_by_identity("tool-def") is s2
    assert mgr.get_session_by_identity("tool-xyz") is None


def test_state_transition_open_to_verifying() -> None:
    callback = MagicMock()
    mgr = SessionManager(verifier_callback=callback)
    session = mgr.create_session("tool-abc")
    assert session.state == SessionState.open

    mgr.close_session(session.session_id)
    assert session.state == SessionState.verifying
    assert session.end_time is not None
    callback.assert_called_once_with(session)


def test_state_transition_open_to_closed_no_callback() -> None:
    mgr = SessionManager()
    session = mgr.create_session("tool-abc")
    mgr.close_session(session.session_id)
    assert session.state == SessionState.closed
    assert session.end_time is not None


def test_close_session_idempotent() -> None:
    mgr = SessionManager()
    session = mgr.create_session("tool-abc")
    mgr.close_session(session.session_id)
    mgr.close_session(session.session_id)  # second call is no-op
    assert session.state in (SessionState.closed, SessionState.verifying)


def test_close_session_unknown() -> None:
    mgr = SessionManager()
    assert mgr.close_session("does-not-exist") is None


def test_cleanup_expired_closes_old_sessions() -> None:
    mgr = SessionManager(inactivity_timeout=0)
    s1 = mgr.create_session("tool-abc")
    s2 = mgr.create_session("tool-def")
    # Both should be expired since timeout is 0
    expired = mgr.cleanup_expired()
    assert len(expired) == 2
    assert s1.session_id in expired
    assert s2.session_id in expired


def test_cleanup_expired_respects_timeout() -> None:
    mgr = SessionManager(inactivity_timeout=3600)
    mgr.create_session("tool-abc")
    mgr.create_session("tool-def")
    expired = mgr.cleanup_expired()
    assert len(expired) == 0


def test_cleanup_only_closes_open_sessions() -> None:
    mgr = SessionManager(inactivity_timeout=0, verifier_callback=lambda s: None)
    s1 = mgr.create_session("tool-abc")
    mgr.close_session(s1.session_id)  # now verifying
    s2 = mgr.create_session("tool-def")  # open
    expired = mgr.cleanup_expired()
    assert len(expired) == 1
    assert s2.session_id in expired
