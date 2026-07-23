"""Batch offline re-fit of the kNN index from the append-only outcome log."""

# Learning is batch-first, not online: the ``outcome_events`` log is the source of truth and the
# HNSW index a rebuildable projection. A session-count trigger rebuilds every N captured outcomes
# — no second timer thread; it rides the existing capture consumer. The rebuild's ``index.build``
# serializes against the decide-path ``index.query`` on the index's own lock, so a decision never
# reads a half-rebuilt index.

from __future__ import annotations

import logging
import threading
from typing import Protocol

logger = logging.getLogger(__name__)


class _Rebuildable(Protocol):
    def rebuild_index(self) -> None: ...


class RefitScheduler:
    """Count-based re-fit trigger: rebuild the index every ``every_n`` captured outcomes."""

    def __init__(self, store: _Rebuildable, every_n: int) -> None:
        self._store = store
        self._every_n = every_n
        self._since_refit = 0
        self._lock = threading.Lock()

    def note_capture(self) -> bool:
        """Record one fresh capture; rebuild and return True when the cadence is reached.

        ``rebuild_index`` reprojects the whole non-tombstoned log (idempotent); ``every_n=0``
        disables the trigger, leaving the boot-time rebuild as the only re-fit.
        """
        if self._every_n <= 0:
            return False
        with self._lock:
            self._since_refit += 1
            if self._since_refit < self._every_n:
                return False
            self._since_refit = 0
        # Outside the scheduler lock: rebuild_index reads embeddings under the store lock and
        # then rebuilds under the index's own lock; holding the scheduler lock across it would
        # widen the critical section for no gain (the counter is already reset).
        logger.info("refit: rebuilding kNN index from the outcome log (cadence=%d)", self._every_n)
        self._store.rebuild_index()
        return True
