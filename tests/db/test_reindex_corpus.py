"""``reindex_corpus`` re-embeds the corpus atomically, fingerprint written LAST."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shunt.db.store import OutcomeStore


class FakeEmbedder:
    """Deterministic fake mapping every prompt to one fixed vector for a named space."""

    def __init__(
        self, space: str, value: float, dim: int = 8, *, fail_after: int | None = None
    ) -> None:
        self._space = space
        self._value = value
        self._dim = dim
        self._fail_after = fail_after
        self.calls = 0

    def fingerprint(self) -> dict[str, object]:
        return {"repo": self._space, "dim": self._dim, "max_chars": 4000, "revision": None}

    def embed(self, text: str) -> np.ndarray:
        self.calls += 1
        if self._fail_after is not None and self.calls > self._fail_after:
            raise RuntimeError("simulated embed crash mid-reindex")
        return np.full(self._dim, self._value, dtype=np.float32)


def _seed(store: OutcomeStore, n: int, embedder: FakeEmbedder) -> list[str]:
    ids: list[str] = []
    for i in range(n):
        sid = f"s{i}"
        store.store_session(
            session_id=sid,
            prompt_text=f"task {i}",
            embedding=embedder.embed(f"task {i}"),
            model_chosen="model-a",
            cost=1.0,
            cache_stats={},
            duration=1.0,
        )
        store.store_outcome(sid, tier1_outcome="success", tier1_confidence=0.9)
        ids.append(sid)
    return ids


def _blob(store: OutcomeStore, sid: str) -> np.ndarray:
    row = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT embedding_blob FROM sessions WHERE session_id = ?", (sid,)
    ).fetchone()
    return np.frombuffer(row["embedding_blob"], dtype=np.float32)


def test_reindex_rewrites_all_blobs_and_advances_fingerprint(tmp_path: Path) -> None:
    store = OutcomeStore(db_path=str(tmp_path / "r.db"))
    old = FakeEmbedder("space-A", value=0.1)
    ids = _seed(store, 4, old)
    store.save_embedding_fingerprint(old.fingerprint())

    new = FakeEmbedder("space-B", value=0.9)
    summary = store.reindex_corpus(new)

    assert summary["reindexed"] == 4
    assert summary["old_fingerprint"] == old.fingerprint()
    assert summary["new_fingerprint"] == new.fingerprint()
    assert store.load_embedding_fingerprint() == new.fingerprint()
    for sid in ids:
        assert np.allclose(_blob(store, sid), 0.9)
    store.close()


def test_crash_mid_embed_leaves_old_space_intact(tmp_path: Path) -> None:
    # A failure part-way through the re-embed must roll the WHOLE batch back to the old
    # space (no half-A/half-B), and leave the old fingerprint so the next boot refuses.
    store = OutcomeStore(db_path=str(tmp_path / "c.db"))
    old = FakeEmbedder("space-A", value=0.1)
    ids = _seed(store, 4, old)
    store.save_embedding_fingerprint(old.fingerprint())

    broken = FakeEmbedder("space-B", value=0.9, fail_after=2)
    with pytest.raises(RuntimeError):
        store.reindex_corpus(broken)

    # Fingerprint unchanged (old space) and every blob still old — never a partial mix.
    assert store.load_embedding_fingerprint() == old.fingerprint()
    for sid in ids:
        assert np.allclose(_blob(store, sid), 0.1)
    store.close()


def test_fingerprint_written_last_crash_before_it_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate a crash AFTER blobs+index are in the new space but BEFORE the fingerprint
    # write: the stored fingerprint must still be the OLD one, so the next boot mismatches.
    store = OutcomeStore(db_path=str(tmp_path / "l.db"))
    old = FakeEmbedder("space-A", value=0.1)
    ids = _seed(store, 3, old)
    store.save_embedding_fingerprint(old.fingerprint())

    new = FakeEmbedder("space-B", value=0.9)
    original = store.save_embedding_fingerprint

    def _boom(fp: dict[str, object]) -> None:
        raise RuntimeError("crash at the commit marker")

    monkeypatch.setattr(store, "save_embedding_fingerprint", _boom)
    with pytest.raises(RuntimeError):
        store.reindex_corpus(new)
    monkeypatch.setattr(store, "save_embedding_fingerprint", original)

    # Blobs DID move (the crash was after their commit) but the fingerprint is still old.
    for sid in ids:
        assert np.allclose(_blob(store, sid), 0.9)
    assert store.load_embedding_fingerprint() == old.fingerprint()
    store.close()


def test_reindex_empty_corpus_writes_fingerprint(tmp_path: Path) -> None:
    store = OutcomeStore(db_path=str(tmp_path / "e.db"))
    new = FakeEmbedder("space-B", value=0.9)
    summary = store.reindex_corpus(new)
    assert summary["reindexed"] == 0
    assert store.load_embedding_fingerprint() == new.fingerprint()
    store.close()
