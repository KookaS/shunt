"""One writer at a time over a shared corpus directory, and never a half-written file."""

# WHY THIS EXISTS. The stamping stage fans out over N worker processes, and three files in the
# live data dir are shared by all of them: `admissibility.json` (the gate's verdict cache and the
# ONLY audit record of a clearing), `manifest.json` (which hash-binds every trajectory), and
# `stamp_ledger.json` (the resume point). Every one of them was an unsynchronised
# read-modify-write over a plain `write_text`.
#
# THAT IS NOT MERELY REDUNDANT WORK — IT REACHES WRONG DATA. Measured, by executing it, with 8
# concurrent writers on a temp dir:
#   * `record_verdict` raised an UNCAUGHT JSONDecodeError on 282 of 480 calls (it read a file a
#     second writer had already truncated), 413 of 480 verdicts were lost, and the file left on
#     disk was itself unparseable — so `load_verdicts` returns {} and every clearing in the corpus
#     loses the record that says WHY its stamps were removed;
#   * a worker that rebuilt `manifest.json` from a directory snapshot another worker had already
#     superseded bound a STALE `content_sha256`, and `verify_manifest` then reports
#     `manifest.hash_mismatch` — the corpus failing its own Layer-1 integrity check;
#   * `authenticity.manifest()` reading a trajectory mid-`write_text` crashed on a truncated line.
#
# So the fix is not a smaller window. Two rules, and both are needed:
#   1. every shared write goes through `atomic_write_text` — temp + `os.replace`, so a concurrent
#      reader sees only whole files and a `kill -9` cannot leave a truncated one behind;
#   2. every read-modify-write TRANSACTION runs under `corpus_lock`. Atomicity alone is not
#      enough for a read-modify-write: two workers can each read the same old file and each
#      atomically write a version missing the other's change. The write-trajectory-then-rewrite-
#      manifest pair is one such transaction — the manifest hashes what is on disk, so if the two
#      writes are separable the manifest can bind a hash the file no longer has.
#
# `flock` is the cross-PROCESS half (the replays run as subprocesses) and, because a second
# `open()` creates a separate open file description, it excludes sibling THREADS in the parent
# too — one mechanism covers both. Re-entrancy is tracked per thread and per directory because
# transactions nest: `clear_rejected` records a verdict (a transaction) and then commits the
# cleared trajectory (another), and a non-re-entrant `flock` would deadlock against itself.

from __future__ import annotations

import fcntl
import os
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

LOCK_NAME: Final = ".corpus.lock"

# Per-thread re-entrancy depth, keyed by resolved directory. Thread-local rather than a plain
# dict because two worker threads holding locks on DIFFERENT dirs must not see each other's
# depth, and a thread must never mistake a sibling's held lock for its own.
_HELD: Final = threading.local()


def atomic_write_text(path: Path, text: str) -> None:
    """Write through a sibling temp + os.replace: a reader sees the old file or the new one."""
    # The temp name carries pid+thread so two writers to the same target never share a temp, and
    # it ends in `.tmp` so it can never be picked up by the `*.jsonl` globs that scan this dir.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _depths() -> dict[str, int]:
    """This thread's per-directory re-entrancy counters (created on first use)."""
    depths: dict[str, int] | None = getattr(_HELD, "depths", None)
    if depths is None:
        depths = {}
        _HELD.depths = depths
    return depths


@contextmanager
def corpus_lock(data_dir: Path) -> Iterator[None]:
    """Serialise one read-modify-write transaction over *data_dir*, across threads AND processes."""
    key = str(data_dir.resolve())
    depths = _depths()
    if depths.get(key):
        depths[key] += 1  # already ours: re-entering a held transaction, not taking a new one
        try:
            yield
        finally:
            depths[key] -= 1
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / LOCK_NAME).open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        depths[key] = 1
        try:
            yield
        finally:
            depths[key] = 0
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = ["LOCK_NAME", "atomic_write_text", "corpus_lock"]
