from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

from shunt.capture.worker import _MAX_QUEUE, CaptureWorker
from shunt.session import Session, SessionManager


class _SlowCoordinator:
    """Blocks in capture() so the test can prove enqueue does not run it inline."""

    def __init__(self) -> None:
        self.captured = threading.Event()
        self.release = threading.Event()

    def capture(self, session: Session) -> None:
        self.captured.set()
        self.release.wait(timeout=5)


def _closed_session() -> Session:
    now = datetime.now(UTC)
    s = Session(session_id="s1", tool_identity="t", start_time=now)
    s.end_time = now
    return s


def test_enqueue_returns_immediately_and_runs_on_worker_thread() -> None:
    coord = _SlowCoordinator()
    worker = CaptureWorker(coordinator=coord, session_manager=SessionManager())
    worker.start()
    try:
        start = time.monotonic()
        worker.enqueue(_closed_session())
        elapsed = time.monotonic() - start
        assert elapsed < 0.5  # O(1): did not block on the 5s capture
        assert coord.captured.wait(timeout=2)  # capture ran, but on the worker thread
    finally:
        coord.release.set()
        worker.stop()


def test_capture_runs_off_the_queue() -> None:
    seen: list[str] = []

    class _RecordingCoordinator:
        def capture(self, session: Session) -> None:
            seen.append(session.session_id)

    worker = CaptureWorker(coordinator=_RecordingCoordinator(), session_manager=SessionManager())
    worker.start()
    try:
        worker.enqueue(_closed_session())
        deadline = time.monotonic() + 2
        while not seen and time.monotonic() < deadline:
            time.sleep(0.01)
        assert seen == ["s1"]
    finally:
        worker.stop()


def test_shutdown_abandoned_capture_logs_without_traceback(caplog) -> None:  # type: ignore[no-untyped-def]
    # A capture that raises AFTER stop() was signalled (the store closed under it on clean
    # shutdown) must log a calm INFO — not an alarming exception traceback that reads as a crash.
    import logging

    class _RaisingCoordinator:
        def capture(self, session: Session) -> None:
            raise RuntimeError("Cannot operate on a closed database")

    worker = CaptureWorker(coordinator=_RaisingCoordinator(), session_manager=SessionManager())
    worker._stop.set()  # simulate shutdown already in progress
    with caplog.at_level(logging.INFO, logger="shunt.capture.worker"):
        worker._process(_closed_session())
    assert any(
        r.levelno == logging.INFO and "abandoned" in r.message and "shutdown" in r.message
        for r in caplog.records
    )
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def test_enqueue_drops_when_queue_full_instead_of_raising() -> None:
    # A burst of closes must never block the wire or raise: past the bounded queue the
    # enqueue drops (warns) rather than propagating queue.Full. No consumer is started, so
    # the queue never drains and the (_MAX_QUEUE + 1)-th push exercises the drop path.
    worker = CaptureWorker(coordinator=_SlowCoordinator(), session_manager=SessionManager())
    for _ in range(_MAX_QUEUE):
        worker.enqueue(_closed_session())
    assert worker._queue.full()
    worker.enqueue(_closed_session())  # must not raise
    assert worker._queue.qsize() == _MAX_QUEUE  # the overflow was dropped, not queued


def test_periodic_sweep_fires_on_sweep_hook() -> None:
    hook_calls = {"n": 0}

    def _hook() -> None:
        hook_calls["n"] += 1

    class _NoopCoordinator:
        def capture(self, session: Session) -> None:  # pragma: no cover - unused
            pass

    worker = CaptureWorker(
        coordinator=_NoopCoordinator(),
        session_manager=SessionManager(),
        sweep_interval=0.05,
        on_sweep=_hook,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 2
        while hook_calls["n"] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert hook_calls["n"] >= 2  # the state-flush hook fired each sweep, off any request
    finally:
        worker.stop()


def test_periodic_sweep_calls_cleanup_expired() -> None:
    calls = {"n": 0}

    class _CountingManager(SessionManager):
        def cleanup_expired(self) -> list[str]:
            calls["n"] += 1
            return []

    class _NoopCoordinator:
        def capture(self, session: Session) -> None:  # pragma: no cover - unused
            pass

    worker = CaptureWorker(
        coordinator=_NoopCoordinator(),
        session_manager=_CountingManager(),
        sweep_interval=0.05,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 2
        while calls["n"] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls["n"] >= 2  # the sweeper fired independently of any request
    finally:
        worker.stop()
