"""Two builds of the seed corpus must write byte-identical session rows.

``seed_bundle`` never calls ``embed``, so both builds run with no network and no weights —
proved by ``SHUNT_DISALLOW_REAL_EMBEDDER=1`` plus a cache dir that is never created.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from benchmark.routing.seed_live import (
    DEFAULT_SEED_DIR,
    SEED_EPOCH,
    _load_bundle,
    seed_bundle,
)
from shunt.db.store import OutcomeStore

_TASKS = 2  # each seeded cell costs several fsynced commits; 2 tasks keeps the suite fast


class _AllModels:
    """A live pool that admits every model the bundle carries."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    def model_names(self) -> list[str]:
        return self._names


def _build(bundle: dict, dest: Path) -> Path:
    """Seed *bundle* into a fresh store under *dest*; returns the db path."""
    dest.mkdir(parents=True, exist_ok=True)
    db = dest / "outcomes.db"
    store = OutcomeStore(db_path=str(db))
    pool = _AllModels(sorted({str(m) for m in bundle["model"]}))
    seed_bundle(store, bundle, pool, limit=_TASKS, fingerprint={"model": "bundle"})
    store.close()
    return db


@pytest.fixture(scope="module")
def builds(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, Path]]:
    """Two independent seed-only builds from the committed bundle."""
    manifest_path = DEFAULT_SEED_DIR / "manifest.json"
    if not manifest_path.exists():
        pytest.skip(f"no seed manifest at {manifest_path}")
    entry = next(iter(json.loads(manifest_path.read_text()).values()))
    bundle_path = DEFAULT_SEED_DIR / str(entry["file"])
    if not bundle_path.exists() or bundle_path.stat().st_size < 1024:
        pytest.skip(f"seed bundle {bundle_path} is absent or an unfetched LFS pointer")

    root = tmp_path_factory.mktemp("seed-determinism")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SHUNT_DISALLOW_REAL_EMBEDDER", "1")
        cache = root / "bogus-embed-cache"
        mp.setenv("SHUNT_EMBED_CACHE_DIR", str(cache))
        bundle = _load_bundle(bundle_path)
        yield _build(bundle, root / "a"), _build(bundle, root / "b")
        assert not cache.exists(), "the seed path touched the embedding cache"


def _dump_sessions(db: Path) -> str:
    """The `sessions` table as SQL — the corpus content, not the whole file.

    `schema_version.applied_at` and `router_state.updated_at` are store-lifecycle columns and
    stay wall-clock by design, so a whole-file dump is not reproducible and is not meant to be.
    """
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return "\n".join(line for line in con.iterdump() if '"sessions"' in line)
    finally:
        con.close()


def test_two_builds_dump_sessions_identically(builds: tuple[Path, Path]) -> None:
    a, b = builds
    assert _dump_sessions(a) == _dump_sessions(b)


def test_seeded_rows_carry_the_epoch_not_the_wall_clock(builds: tuple[Path, Path]) -> None:
    con = sqlite3.connect(f"file:{builds[0]}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT DISTINCT timestamp FROM sessions").fetchall()
        events = con.execute("SELECT DISTINCT created_at FROM outcome_events").fetchall()
        outcomes = con.execute("SELECT DISTINCT created_at FROM outcomes").fetchall()
    finally:
        con.close()
    assert {r[0] for r in rows} == {SEED_EPOCH}
    assert {r[0] for r in events} == {SEED_EPOCH}
    assert {r[0] for r in outcomes} == {SEED_EPOCH}
