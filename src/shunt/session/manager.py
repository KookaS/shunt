from __future__ import annotations

import hashlib
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from .models import Session, SessionState


class SessionManager:
    """Manages tool sessions — creation, lookup, expiry, and lifecycle transitions.

    Thread-safe in-memory dict storage. SQLite persistence is planned to replace this.
    """

    def __init__(
        self,
        inactivity_timeout: int = 900,
        grace_period: int = 120,
        verifier_callback: Callable[[Session], None] | None = None,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self._identity_to_session: dict[str, str] = {}
        self._lock = threading.Lock()
        self._inactivity_timeout = inactivity_timeout
        self._grace_period = grace_period
        self._verifier_callback = verifier_callback

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
                    return session
            session = Session(
                session_id=str(uuid.uuid4()),
                tool_identity=tool_identity,
                start_time=datetime.now(UTC),
            )
            self._sessions[session.session_id] = session
            self._identity_to_session[tool_identity] = session.session_id
            return session

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
        """Close all open sessions whose start_time exceeds *inactivity_timeout*."""
        now = datetime.now(UTC)
        expired: list[str] = []
        with self._lock:
            for session_id, session in list(self._sessions.items()):
                if session.state != SessionState.open:
                    continue
                elapsed = (now - session.start_time).total_seconds()
                if elapsed > self._inactivity_timeout:
                    expired.append(session_id)

        for sid in expired:
            self.close_session(sid)

        return expired

    @property
    def inactivity_timeout(self) -> int:
        return self._inactivity_timeout

    @property
    def grace_period(self) -> int:
        return self._grace_period
