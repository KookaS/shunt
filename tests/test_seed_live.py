from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from benchmark.routing import seed_live
from benchmark.routing.seed_live import seed_matrix
from shunt.db.store import OutcomeStore


def _fake_embed(text: str) -> np.ndarray:
    """Deterministic hash-derived vector (never a bare rng) — the sanctioned SH008 exception."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    return (np.linspace(0.0, 1.0, 8) + (seed % 7) / 10.0).astype(np.float32)


class FakeEmbedder:
    """Records per-text embed calls so the cache-per-task behaviour is assertable."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def embed(self, text: str) -> np.ndarray:
        self.calls[text] = self.calls.get(text, 0) + 1
        return _fake_embed(text)

    def fingerprint(self) -> dict[str, object]:
        """A deterministic stand-in for a real Embedder's space fingerprint."""
        return {"model": "fake", "dim": 8}


class FlakyEmbedder(FakeEmbedder):
    def __init__(self, fail_on: str) -> None:
        super().__init__()
        self._fail_on = fail_on

    def embed(self, text: str) -> np.ndarray:
        if self._fail_on in text:
            raise RuntimeError("embed failed")
        return super().embed(text)


class FakePool:
    def __init__(self, names: list[str]) -> None:
        self._names = list(names)

    def model_names(self) -> list[str]:
        return list(self._names)


def _matrix() -> dict:
    tasks = {
        "repo-a/task-1": {"problem_statement": "implement the foo parser"},
        "repo-b/task-2": {"problem_statement": "fix the bar overflow"},
        "repo-c/task-3": {"problem_statement": "add the retry loop"},
    }
    results = {
        "repo-a/task-1": {
            "model-cheap": {
                "pass": True,
                "real_cost": 0.004,
                "cost": 0.004,
                "in_tok": 10,
                "out_tok": 5,
            },
            "model-frontier": {
                "pass": True,
                "real_cost": 0.9,
                "cost": 0.9,
                "in_tok": 10,
                "out_tok": 5,
            },
        },
        "repo-b/task-2": {
            "model-cheap": {
                "pass": False,
                "real_cost": 0.002,
                "cost": 0.002,
                "in_tok": 8,
                "out_tok": 3,
            },
            "model-frontier": {
                "pass": True,
                "real_cost": 0.7,
                "cost": 0.7,
                "in_tok": 8,
                "out_tok": 3,
            },
            "model-unmeasured": {
                "pass": True,
                "real_cost": 0.5,
                "cost": 0.5,
                "in_tok": 1,
                "out_tok": 1,
            },
        },
        "repo-c/task-3": {
            "model-cheap": {"pass": True, "imputed": True, "real_cost": 0.001, "cost": 0.001},
            "model-frontier": {"pass": False, "in_tok": 4, "out_tok": 2},
        },
    }
    return {"results": results, "tasks": tasks, "models": {"model-cheap": {}, "model-frontier": {}}}


def _pool() -> FakePool:
    return FakePool(["model-cheap", "model-frontier"])


def _sid(tid: str, model: str) -> str:
    digest = hashlib.sha256(tid.encode("utf-8")).hexdigest()[:12]
    return f"bench:{digest}:{model}"


def _write_matrix(tmp_path: Path, matrix: dict) -> Path:
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps(matrix))
    return p


@pytest.fixture
def store(tmp_path: pytest.TempPathFactory) -> OutcomeStore:
    s = OutcomeStore(db_path=str(tmp_path / "o.db"))
    yield s
    s.close()


def _event_count(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    try:
        return int(con.execute("SELECT COUNT(*) FROM outcome_events").fetchone()[0])
    finally:
        con.close()


def test_seed_matrix_basic(store: OutcomeStore, tmp_path: pytest.TempPathFactory) -> None:
    embedder = FakeEmbedder()
    report = seed_matrix(store, _matrix(), _pool(), embedder=embedder)

    assert report.tasks == 3
    assert report.cells_seeded == 5
    assert report.cells_imputed_skipped == 1
    assert report.cells_model_skipped == 1
    assert report.cells_embed_failed == 0
    assert report.cells_empty_text == 0
    assert report.per_model == {"model-cheap": 2, "model-frontier": 3}
    assert report.total_cost == pytest.approx(0.004 + 0.9 + 0.002 + 0.7 + 0.0)
    assert report.models == ("model-cheap", "model-frontier")

    assert len(embedder.calls) == 3

    cheap = store.get_session(_sid("repo-a/task-1", "model-cheap"))
    assert cheap is not None
    assert cheap["model_chosen"] == "model-cheap"
    assert cheap["session_id"].startswith("bench:")
    assert store.get_outcome(cheap["session_id"])["tier2_outcome"] == "success"

    failed = store.get_session(_sid("repo-b/task-2", "model-cheap"))
    assert store.get_outcome(failed["session_id"])["tier2_outcome"] == "failure"

    assert store.get_session(_sid("repo-c/task-3", "model-cheap")) is None
    assert store.get_session(_sid("repo-b/task-2", "model-unmeasured")) is None

    unknown_cost = store.get_session(_sid("repo-c/task-3", "model-frontier"))
    assert unknown_cost is not None
    assert unknown_cost["cost_known"] == 0
    assert unknown_cost["cost"] == 0.0

    assert store.count_outcomes() == 5
    assert store.get_stats()["index_size"] == 5
    assert (tmp_path / "o.db.hnsw2").exists()


def test_seed_matrix_stamps_embedding_fingerprint(store: OutcomeStore) -> None:
    embedder = FakeEmbedder()
    report = seed_matrix(store, _matrix(), _pool(), embedder=embedder)

    assert report.cells_seeded > 0
    # A real seed must stamp the corpus space so the server's _resolve_embedding_trust
    # trusts the seeded index at boot — otherwise labeled embeddings with no stored
    # fingerprint read as a pre-fingerprint DB and route cold-start (no kNN).
    assert store.load_embedding_fingerprint() == embedder.fingerprint()
    assert store.get_labeled_embeddings()


def test_seed_matrix_idempotent(store: OutcomeStore, tmp_path: pytest.TempPathFactory) -> None:
    first = seed_matrix(store, _matrix(), _pool(), embedder=FakeEmbedder())
    second = seed_matrix(store, _matrix(), _pool(), embedder=FakeEmbedder())

    assert (first.cells_seeded, first.tasks) == (second.cells_seeded, second.tasks)
    assert store.count_outcomes() == 5
    assert _event_count(str(tmp_path / "o.db")) == 5
    assert store.get_stats()["index_size"] == 5


def test_seed_matrix_dry_run_writes_nothing(tmp_path: pytest.TempPathFactory) -> None:
    db_path = tmp_path / "dry.db"
    store = None
    report = seed_matrix(store, _matrix(), _pool(), embedder=FakeEmbedder(), dry_run=True)

    assert report.cells_seeded == 5
    assert report.tasks == 3
    assert not db_path.exists()
    assert not (tmp_path / "dry.db.hnsw2").exists()


def test_seed_matrix_limit(store: OutcomeStore) -> None:
    report = seed_matrix(store, _matrix(), _pool(), embedder=FakeEmbedder(), limit=1)

    assert report.tasks == 1
    assert report.cells_seeded == 2
    assert store.count_outcomes() == 2


def test_seed_matrix_embed_failure_skips_task(store: OutcomeStore) -> None:
    report = seed_matrix(
        store, _matrix(), _pool(), embedder=FlakyEmbedder(fail_on="fix the bar overflow")
    )

    assert report.cells_embed_failed == 2
    assert report.cells_seeded == 3
    assert store.get_session(_sid("repo-b/task-2", "model-cheap")) is None


def test_seed_matrix_empty_matrix(store: OutcomeStore) -> None:
    report = seed_matrix(store, {"results": {}, "tasks": {}}, _pool(), embedder=FakeEmbedder())

    assert report.tasks == 0
    assert report.cells_seeded == 0
    assert report.per_model == {}


def test_seed_matrix_outcome_flip_adds_event(
    store: OutcomeStore, tmp_path: pytest.TempPathFactory
) -> None:
    first = seed_matrix(store, _matrix(), _pool(), embedder=FakeEmbedder())
    assert first.cells_new == 5
    assert _event_count(str(tmp_path / "o.db")) == 5

    flipped = _matrix()
    flipped["results"]["repo-a/task-1"]["model-cheap"]["pass"] = False
    second = seed_matrix(store, flipped, _pool(), embedder=FakeEmbedder())

    assert second.cells_updated == 1
    sid = _sid("repo-a/task-1", "model-cheap")
    # The changed cell's OUTCOME must actually flip — the content-addressed run_signature
    # yields a NEW event instead of being idempotency-dropped.
    assert store.get_outcome(sid)["tier2_outcome"] == "failure"
    con = sqlite3.connect(str(tmp_path / "o.db"))
    try:
        n_session_events = int(
            con.execute(
                "SELECT COUNT(*) FROM outcome_events WHERE session_id = ?", (sid,)
            ).fetchone()[0]
        )
    finally:
        con.close()
    assert n_session_events == 2  # 1 initial + 1 for the content-changed re-import
    assert store.count_outcomes() == 5


def test_limit_never_stamps_or_clobbers_seed_marker(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A --limit run is a TRIAL, not a completion: it must neither write a full-digest
    # marker (or a later no-limit run would skip and leave the store at N tasks) nor
    # clobber an existing full marker.
    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(seed_live, "build_live_pool", lambda: _pool())
    monkeypatch.setattr(seed_live, "Embedder", FakeEmbedder)
    matrix_path = _write_matrix(tmp_path, _matrix())
    digest = hashlib.sha256(matrix_path.read_bytes()).hexdigest()

    def _state() -> tuple[dict | None, int]:
        s = OutcomeStore(db_path=str(tmp_path / "outcomes.db"))
        try:
            return s.load_seed_state(), s.count_outcomes()
        finally:
            s.close()

    assert seed_live.main(["--matrix", str(matrix_path), "--limit", "1"]) == 0
    assert _state() == (None, 2)  # trial: no marker, only task 1 seeded

    assert seed_live.main(["--matrix", str(matrix_path)]) == 0
    marker, full = _state()
    assert marker is not None and marker["results_digest"] == digest
    assert full == 5

    # --force makes the limited run actually import; it must still leave the full marker.
    assert seed_live.main(["--matrix", str(matrix_path), "--limit", "1", "--force"]) == 0
    assert _state() == (marker, 5)
