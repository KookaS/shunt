"""Session identity is (source_ip, user_agent), by design — one client, one model."""

# This is the cache-safety guarantee, not a bug: a session holds ONE model so the
# provider-side prompt cache is never invalidated mid-conversation. These tests pin the
# consequences so a future change cannot weaken them silently. The visible cost is that
# a client's small/auxiliary requests (e.g. opencode's title generation) inherit the
# session's locked model rather than routing independently.

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from shunt.session import SessionManager


def _identity(ip: str = "127.0.0.1", ua: str = "opencode/1.0") -> str:
    return SessionManager.compute_tool_identity(ip, ua)


def test_one_client_gets_one_session_and_one_locked_model() -> None:
    manager = SessionManager(inactivity_timeout=900, grace_period=120)

    build_task = manager.find_or_create(_identity())
    build_task.model_chosen = "kimi-k3"  # an expensive model, locked by the first task

    # A completely unrelated, trivial request from the same client moments later.
    title_task = manager.find_or_create(_identity())

    assert title_task.session_id == build_task.session_id
    # The deliberate trade: a trivial task inherits the lock rather than
    # re-routing and forcing a full prefill on a second model.
    assert title_task.model_chosen == "kimi-k3"


def test_differing_user_agents_do_separate_sessions() -> None:
    # The only lever a client has — and coding agents do not vary it per conversation.
    manager = SessionManager(inactivity_timeout=900, grace_period=120)
    a = manager.find_or_create(_identity(ua="opencode/1.0"))
    b = manager.find_or_create(_identity(ua="claude-code/1.0"))
    assert a.session_id != b.session_id


def test_continuous_use_holds_the_lock_so_the_cache_survives() -> None:
    # `last_activity` resets the deadline every turn, so an actively-used session never
    # expires and keeps its model — which is the point: expiring it mid-work would force
    # a new decision and a full re-prefill on a different model.
    manager = SessionManager(inactivity_timeout=900, grace_period=120)
    session = manager.find_or_create(_identity())
    first_id = session.session_id

    for _ in range(5):
        stored = manager.get_session(first_id)
        assert stored is not None
        # 14 minutes pass, then the user sends another turn — under the 15m timeout.
        stored.last_activity = datetime.now(UTC) - timedelta(seconds=840)
        assert manager.cleanup_expired() == []
        assert manager.find_or_create(_identity()).session_id == first_id
