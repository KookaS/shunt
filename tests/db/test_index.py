from __future__ import annotations

import numpy as np
import pytest

from shunt.db.index import HNSWIndex


def _random_emb(dim: int = 64) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_empty_index() -> None:
    idx = HNSWIndex(dim=64)
    assert idx.count == 0
    assert idx.query(np.zeros(64, dtype=np.float32)) == []


def test_build_empty_list() -> None:
    idx = HNSWIndex(dim=64)
    idx.build([])
    assert idx.count == 0


def test_build_and_query_finds_nearest() -> None:
    idx = HNSWIndex(dim=64)
    target = _normalize(_random_emb())
    others = [_normalize(_random_emb()) for _ in range(50)]
    # target is the first element
    embeddings = [("target", target)] + [(f"e{i}", o) for i, o in enumerate(others)]
    idx.build(embeddings)

    assert idx.count == 51
    results = idx.query(target, k=5)
    assert len(results) == 5
    # the nearest should be index 0 (the target itself) with distance ~0
    nearest_idx, nearest_dist = results[0]
    assert nearest_idx == 0
    assert nearest_dist < 0.01


def test_incremental_add() -> None:
    idx = HNSWIndex(dim=64)
    for i in range(10):
        idx.add(f"s{i}", _normalize(_random_emb()))
    assert idx.count == 10

    query_v = _normalize(_random_emb())
    results = idx.query(query_v, k=3)
    assert len(results) == 3


def test_get_session_id() -> None:
    idx = HNSWIndex(dim=64)
    idx.build([("abc", _normalize(_random_emb())), ("def", _normalize(_random_emb()))])

    assert idx.get_session_id(0) == "abc"
    assert idx.get_session_id(1) == "def"
    assert idx.get_session_id(99) is None
    assert idx.get_session_id(-1) is None


def test_save_and_load(tmp_path: pytest.TempPathFactory) -> None:
    save_dir = tmp_path / "hnsw"
    save_dir.mkdir()
    save_path = str(save_dir / "index.bin")

    idx = HNSWIndex(dim=64)
    vecs = [(_normalize(_random_emb()), _normalize(_random_emb())) for _ in range(20)]
    idx.build([(f"id{i}", v[0]) for i, v in enumerate(vecs)])
    idx.save(save_path)

    idx2 = HNSWIndex(dim=64)
    idx2.load(save_path)

    assert idx2.count == 20
    assert idx2.get_session_id(0) == "id0"
    assert idx2.get_session_id(19) == "id19"

    query_v = vecs[0][0]
    r1 = idx.query(query_v, k=3)
    r2 = idx2.query(query_v, k=3)
    assert r1 == r2


def test_query_more_than_available() -> None:
    idx = HNSWIndex(dim=64)
    for i in range(3):
        idx.add(f"s{i}", _normalize(_random_emb()))
    results = idx.query(_normalize(_random_emb()), k=100)
    assert len(results) == 3


def test_save_empty_raises() -> None:
    idx = HNSWIndex(dim=64)
    with pytest.raises(ValueError, match="Cannot save empty index"):
        idx.save("/tmp/nonexistent/test.hnsw")


def test_load_nonexistent_raises() -> None:
    idx = HNSWIndex(dim=64)
    with pytest.raises(FileNotFoundError):
        idx.load("/tmp/nonexistent/index.bin")


def test_dimension_inferred_on_build() -> None:
    idx = HNSWIndex()
    idx.build([("a", _normalize(_random_emb(128)))])
    assert idx.count == 1
    r = idx.query(_normalize(_random_emb(128)), k=1)
    assert len(r) == 1
