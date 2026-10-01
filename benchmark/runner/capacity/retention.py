"""Reference-counted, targeted image removal (never a global prune)."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable

from benchmark.runner.capacity._exec import Runner, subprocess_runner
from benchmark.runner.capacity.sizes import image_refs_for_ids, local_size
from benchmark.runner.capacity.types import ResourceConfig, RetentionPolicy

RemoveFn = Callable[[str], int | None]
RefsForIdsFn = Callable[[Iterable[str]], list[str]]


def docker_rmi(ref: str, *, runner: Runner = subprocess_runner) -> int | None:
    """Remove one image and return the bytes freed (None when it was not present)."""
    size = local_size(ref, runner=runner)
    code, _ = runner(["docker", "rmi", ref])
    return size if code == 0 else None


class ImageRetention:
    """Track how many cells hold each image and remove it only when unheld.

    Removal is always a targeted ``docker rmi`` of a specific ref; there is no
    ``docker system prune``. Under ``KEEP`` nothing is ever removed.
    """

    def __init__(
        self,
        config: ResourceConfig,
        *,
        remove_fn: RemoveFn | None = None,
        refs_for_ids: RefsForIdsFn = image_refs_for_ids,
    ) -> None:
        self._config = config
        self._remove = remove_fn or docker_rmi
        self._refs_for_ids = refs_for_ids
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._freed = 0

    @property
    def freed_bytes(self) -> int:
        """Total bytes reported freed by successful removals."""
        return self._freed

    def acquire(self, ref: str) -> None:
        """Record that one cell is now using ``ref``."""
        with self._lock:
            self._counts[ref] = self._counts.get(ref, 0) + 1

    def release(self, ref: str) -> None:
        """Record that one cell is done with ``ref`` (never below zero)."""
        with self._lock:
            self._counts[ref] = max(0, self._counts.get(ref, 0) - 1)

    def release_challenge(self, instance_ids: Iterable[str]) -> int:
        """Remove a challenge's unheld images under PER_CHALLENGE; return bytes freed."""
        if self._config.retention is not RetentionPolicy.PER_CHALLENGE:
            return 0
        freed = 0
        for ref in self._refs_for_ids(list(instance_ids)):
            if self._claim_unheld(ref):
                freed += self._remove_one(ref)
        self._freed += freed
        return freed

    def drain(self) -> int:
        """Remove every currently unheld image under PER_CHALLENGE; clear counts."""
        if self._config.retention is not RetentionPolicy.PER_CHALLENGE:
            with self._lock:
                self._counts.clear()
            return 0
        with self._lock:
            candidates = [ref for ref, count in self._counts.items() if count == 0]
            for ref in candidates:
                self._counts.pop(ref, None)
        freed = sum(self._remove_one(ref) for ref in candidates)
        self._freed += freed
        return freed

    def _claim_unheld(self, ref: str) -> bool:
        """True when ``ref`` is tracked and at zero references (then drop it)."""
        with self._lock:
            if self._counts.get(ref, 0) > 0:
                return False
            self._counts.pop(ref, None)
            return True

    def _remove_one(self, ref: str) -> int:
        """Remove one ref and normalise the freed-bytes report to an int."""
        freed = self._remove(ref)
        return int(freed) if freed else 0
