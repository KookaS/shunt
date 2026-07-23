"""The async boundary for capture: a bounded queue + single daemon consumer that
runs CaptureCoordinator.capture off the request path, plus a periodic sweep so a
session with no follow-up traffic still gets closed and captured."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from typing import Protocol

from shunt.session import Session, SessionManager

logger = logging.getLogger(__name__)

# Bound the backlog so a burst of closes can't grow memory without limit. A full
# queue drops the capture (logged) rather than blocking close_session on the wire —
# a missed auto-label degrades gracefully; a stalled request path does not.
_MAX_QUEUE = 256


class _Capturer(Protocol):
    def capture(self, session: Session) -> None: ...


class CaptureWorker:
    """Single-consumer background capture (``max_workers=1`` semantics) + self-sweep."""

    def __init__(
        self,
        coordinator: _Capturer,
        session_manager: SessionManager,
        sweep_interval: float = 60.0,
        on_sweep: Callable[[], None] | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._session_manager = session_manager
        self._sweep_interval = sweep_interval
        # Runs each sweep tick off the request path — used to flush the router's mutable
        # exploration state to disk so a crash loses at most one interval of drift.
        self._on_sweep = on_sweep
        self._queue: queue.Queue[Session] = queue.Queue(maxsize=_MAX_QUEUE)
        self._stop = threading.Event()
        self._consumer: threading.Thread | None = None
        self._sweeper: threading.Thread | None = None

    def start(self) -> None:
        """Launch the consumer and sweeper daemon threads (idempotent)."""
        if self._consumer is not None:
            return
        self._consumer = threading.Thread(
            target=self._consume, name="shunt-capture-consumer", daemon=True
        )
        self._sweeper = threading.Thread(
            target=self._sweep, name="shunt-capture-sweeper", daemon=True
        )
        self._consumer.start()
        self._sweeper.start()

    def enqueue(self, session: Session) -> None:
        """O(1) push used as the SessionManager verifier_callback — never blocks the wire."""
        try:
            self._queue.put_nowait(session)
        except queue.Full:
            logger.warning(
                "capture queue full — dropping capture for session %s", session.session_id
            )

    def _consume(self) -> None:
        while not self._stop.is_set():
            try:
                session = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(session)
            finally:
                self._queue.task_done()

    def _process(self, session: Session) -> None:
        """Run one capture, downgrading a shutdown-time failure from a traceback to info."""
        try:
            self._coordinator.capture(session)
        except Exception:
            if self._stop.is_set():
                # Clean shutdown closed the store under an in-flight capture (e.g. the off-wire
                # verifier was still running). Expected, not a crash: this one label is dropped
                # and the gate self-corrects on the next verified capture — no scary traceback.
                logger.info(
                    "capture abandoned for session %s during shutdown "
                    "(label dropped; gate updates on the next capture)",
                    session.session_id,
                )
            else:
                logger.exception("capture failed for session %s", session.session_id)

    def _sweep(self) -> None:
        # cleanup_expired only runs when a request arrives; without this a session with
        # no follow-up traffic would never close, so its outcome would never be captured.
        while not self._stop.wait(self._sweep_interval):
            try:
                self._session_manager.cleanup_expired()
            except Exception:
                logger.exception("periodic session sweep failed")
            if self._on_sweep is not None:
                try:
                    self._on_sweep()
                except Exception:
                    logger.exception("periodic sweep hook failed")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal shutdown and join the daemon threads (best-effort drain)."""
        self._stop.set()
        for thread in (self._consumer, self._sweeper):
            if thread is not None:
                thread.join(timeout=timeout)
        self._consumer = None
        self._sweeper = None
