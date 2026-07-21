from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from .models import Session, SessionState

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages tool sessions — creation, lookup, expiry, and lifecycle transitions.

    Thread-safe in-memory dict storage. SQLite persistence is planned to replace this.
    """

    def __init__(
        self,
        inactivity_timeout: int = 900,
        grace_period: int = 120,
        verifier_callback: Callable[[Session], None] | None = None,
        retention_seconds: int = 3600,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self._identity_to_session: dict[str, str] = {}
        self._lock = threading.Lock()
        self._inactivity_timeout = inactivity_timeout
        self._grace_period = grace_period
        self._verifier_callback = verifier_callback
        # How long a settled session stays resident after closing. Bounds memory; the
        # durable record is in SQLite, so eviction loses nothing but the cache entry.
        self._retention_seconds = retention_seconds

    @staticmethod
    def compute_tool_identity(source_ip: str, user_agent: str) -> str:
        """Deterministic hash of source IP + User-Agent for session grouping."""
        raw = f"{source_ip}|{user_agent}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def create_session(self, tool_identity: str) -> Session:
        """Create a new session with a UUID4 identifier."""
        session = Session(
            session_id=str(uuid.uuid4()),
            tool_identity=tool_identity,
            start_time=datetime.now(UTC),
        )
        with self._lock:
            self._sessions[session.session_id] = session
            self._identity_to_session[tool_identity] = session.session_id
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Look up a session by its UUID."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_session_by_identity(self, tool_identity: str) -> Session | None:
        """Look up an open session by tool identity hash."""
        with self._lock:
            session_id = self._identity_to_session.get(tool_identity)
            if session_id is None:
                return None
            return self._sessions.get(session_id)

    def find_or_create(self, tool_identity: str) -> Session:
        """Return the active open session for *tool_identity* or create one."""
        with self._lock:
            session_id = self._identity_to_session.get(tool_identity)
            if session_id is not None:
                session = self._sessions.get(session_id)
                if session is not None and session.state == SessionState.open:
                    # Every turn routes through here, so this is the one place that
                    # cannot be forgotten — an explicit touch() call site could be.
                    session.last_activity = datetime.now(UTC)
                    logger.debug(
                        "session: REUSED %s (model_locked=%s) — one decision per session",
                        session.session_id,
                        session.model_chosen,
                    )
                    return session
            now = datetime.now(UTC)
            session = Session(
                session_id=str(uuid.uuid4()),
                tool_identity=tool_identity,
                start_time=now,
                last_activity=now,
            )
            self._sessions[session.session_id] = session
            self._identity_to_session[tool_identity] = session.session_id
            # A NEW id mid-conversation means a fresh routing decision, i.e. a possible
            # cache-safety break — so creation is worth a line of its own.
            logger.debug("session: CREATED %s", session.session_id)
            return session

    def touch(self, session_id: str) -> None:
        """Mark *session_id* as used right now, resetting its inactivity deadline."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None and session.state == SessionState.open:
                session.last_activity = datetime.now(UTC)

    def close_session(self, session_id: str) -> Session | None:
        """Transition *session_id* through open → closing → closed and optionally
        queue the verifier callback (closed → verifying)."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.state in (SessionState.closed, SessionState.verifying):
                return session
            session.state = SessionState.closing
            session.end_time = datetime.now(UTC)
            session.state = SessionState.closed

        if self._verifier_callback is not None:
            self._verifier_callback(session)
            with self._lock:
                session.state = SessionState.verifying

        return session

    def cleanup_expired(self) -> list[str]:
        """Close expired open sessions, then evict long-closed ones from memory."""
        now = datetime.now(UTC)
        expired: list[str] = []
        with self._lock:
            for session_id, session in list(self._sessions.items()):
                if session.state != SessionState.open:
                    continue
                # Fall back to start_time for sessions predating `last_activity`.
                idle_since = session.last_activity or session.start_time
                if (now - idle_since).total_seconds() > self._inactivity_timeout:
                    expired.append(session_id)

        for sid in expired:
            self.close_session(sid)

        self._evict_settled(now)
        return expired

    def _evict_settled(self, now: datetime) -> None:
        # Closing a session only flipped `state`; the entry stayed in `_sessions`
        # forever. Session identity is sha256(source_ip + user_agent), so any local
        # caller varying its User-Agent minted unbounded permanent entries.
        with self._lock:
            for session_id, session in list(self._sessions.items()):
                if session.state != SessionState.closed:
                    continue
                settled_at = session.end_time or session.start_time
                if (now - settled_at).total_seconds() > self._retention_seconds:
                    del self._sessions[session_id]
                    for identity, mapped in list(self._identity_to_session.items()):
                        if mapped == session_id:
                            del self._identity_to_session[identity]

    @property
    def inactivity_timeout(self) -> int:
        return self._inactivity_timeout

    @property
    def grace_period(self) -> int:
        return self._grace_period
