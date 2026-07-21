"""`inactivity_timeout` must measure inactivity, not total session age."""

# Measuring `now - start_time` force-closed a session that was still being used, and the
# next turn took a fresh routing decision mid-work — the mid-session model switch the
# cache-safety design forbids.

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from shunt.session import SessionManager


def _age(manager: SessionManager, session_id: str, seconds: int) -> None:
    """Backdate a session's clocks so it looks *seconds* old."""
    session = manager.get_session(session_id)
    assert session is not None
    past = datetime.now(UTC) - timedelta(seconds=seconds)
    session.start_time = past
    session.last_activity = past


def test_an_actively_used_session_is_never_expired() -> None:
    manager = SessionManager(inactivity_timeout=900, grace_period=120)
    session = manager.find_or_create("tool-a")
    # Started 2h ago, but used just now — the exact long-coding-session case.
    _age(manager, session.session_id, 7200)
    manager.touch(session.session_id)

    assert manager.cleanup_expired() == []
    same = manager.find_or_create("tool-a")
    assert same.session_id == session.session_id, "an active session must not be re-routed"


def test_a_genuinely_idle_session_still_expires() -> None:
    manager = SessionManager(inactivity_timeout=900, grace_period=120)
    session = manager.find_or_create("tool-b")
    _age(manager, session.session_id, 1200)

    assert manager.cleanup_expired() == [session.session_id]


def test_touch_extends_the_deadline() -> None:
    manager = SessionManager(inactivity_timeout=900, grace_period=120)
    session = manager.find_or_create("tool-c")
    _age(manager, session.session_id, 1200)
    manager.touch(session.session_id)

    assert manager.cleanup_expired() == []


def test_a_session_with_no_recorded_activity_falls_back_to_start_time() -> None:
    # Sessions created before this field existed carry last_activity=None.
    manager = SessionManager(inactivity_timeout=900, grace_period=120)
    session = manager.find_or_create("tool-d")
    stored = manager.get_session(session.session_id)
    assert stored is not None
    stored.start_time = datetime.now(UTC) - timedelta(seconds=1200)
    stored.last_activity = None

    assert manager.cleanup_expired() == [session.session_id]
