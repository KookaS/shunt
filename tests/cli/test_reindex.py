"""`shunt reindex` re-embeds the corpus offline via the store's atomic path."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shunt.cli import _reindex, main
from shunt.db.store import OutcomeStore


class _FakeEmbedder:
    def __init__(self, repo: str, value: float) -> None:
        self._repo = repo
        self._value = value

    @property
    def model_name(self) -> str:
        return self._repo

    def fingerprint(self) -> dict[str, object]:
        return {"repo": self._repo, "dim": 8, "max_chars": 4000, "revision": None}

    def embed(self, text: str) -> np.ndarray:
        return np.full(8, self._value, dtype=np.float32)


def _seed(store: OutcomeStore) -> None:
    store.store_session(
        session_id="s1",
        prompt_text="a task",
        embedding=np.full(8, 0.1, dtype=np.float32),
        model_chosen="m",
        cost=1.0,
        cache_stats={},
        duration=1.0,
    )
    store.store_outcome("s1", tier1_outcome="success", tier1_confidence=0.9)


def test_reindex_rewrites_corpus_and_advances_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path))
    store = OutcomeStore(db_path=str(tmp_path / "outcomes.db"))
    _seed(store)
    store.save_embedding_fingerprint({"repo": "old", "dim": 8, "max_chars": 4000, "revision": None})
    store.close()

    new = _FakeEmbedder("jinaai/x", value=0.9)
    monkeypatch.setattr("shunt.router.embedder.Embedder", lambda *a, **k: new)
    monkeypatch.setattr("shunt.secrets.load_dotenv_file", lambda *a, **k: None)

    _reindex(argparse.Namespace())

    reopened = OutcomeStore(db_path=str(tmp_path / "outcomes.db"))
    try:
        assert reopened.load_embedding_fingerprint() == new.fingerprint()
        row = reopened._conn.execute(  # type: ignore[attr-defined]
            "SELECT embedding_blob FROM sessions WHERE session_id = 's1'"
        ).fetchone()
        assert np.allclose(np.frombuffer(row["embedding_blob"], dtype=np.float32), 0.9)
    finally:
        reopened.close()
    out = capsys.readouterr().out
    assert "reindex OK" in out


def test_reindex_failure_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path))
    store = OutcomeStore(db_path=str(tmp_path / "outcomes.db"))
    _seed(store)
    store.close()

    class _Broken(_FakeEmbedder):
        def embed(self, text: str) -> Any:
            raise RuntimeError("boom")

    monkeypatch.setattr("shunt.router.embedder.Embedder", lambda *a, **k: _Broken("x", 0.5))
    monkeypatch.setattr("shunt.secrets.load_dotenv_file", lambda *a, **k: None)
    with pytest.raises(SystemExit) as e:
        _reindex(argparse.Namespace())
    assert e.value.code == 1


def test_main_dispatches_reindex(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr("shunt.cli._reindex", lambda args: called.append(True))
    monkeypatch.setattr("sys.argv", ["shunt", "reindex"])
    main()
    assert called == [True]
