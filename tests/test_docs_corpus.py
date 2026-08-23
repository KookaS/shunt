"""The docs seed corpus: input-keyed digest, cache reuse, and build determinism.

Every test runs under ``SHUNT_DISALLOW_REAL_EMBEDDER=1`` with a bogus embed-cache dir — the
positive proof that the build never loads ONNX weights or reaches the network.
"""

# A full build seeds 792 cells, so the determinism test that needs two of them is opt-in via
# SHUNT_SLOW_TESTS=1. The rest stub the seeder and exercise only the cache and digest logic;
# seeding itself is already pinned by tests/test_seed_determinism.py.

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from benchmark.routing import docs_corpus

_SLOW = pytest.mark.skipif(
    os.environ.get("SHUNT_SLOW_TESTS") != "1",
    reason="two full 792-cell builds (~8 min); set SHUNT_SLOW_TESTS=1 to run",
)


@pytest.fixture(autouse=True)
def _no_real_embedder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    cache = tmp_path / "bogus-embed-cache"
    monkeypatch.setenv("SHUNT_DISALLOW_REAL_EMBEDDER", "1")
    monkeypatch.setenv("SHUNT_EMBED_CACHE_DIR", str(cache))
    yield cache
    assert not cache.exists(), "the docs-corpus path touched the embedding cache"


@pytest.fixture(autouse=True)
def _require_bundle() -> None:
    manifest = docs_corpus.DEFAULT_SEED_DIR / "manifest.json"
    if not manifest.exists():
        pytest.skip(f"no seed manifest at {manifest}")


def _dump_sessions(db: Path) -> str:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return "\n".join(line for line in con.iterdump() if '"sessions"' in line)
    finally:
        con.close()


class _StubSeeder:
    """Counts build passes and writes a marker file in place of a real 250 s seed."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, store, bundle, pool, **kwargs):
        self.calls += 1
        store.store_session(
            session_id=f"bench:stub:{self.calls}",
            prompt_text="stub",
            embedding=None,
            model_chosen="stub",
            cost=0.0,
            cache_stats={},
            duration=0.0,
        )


@pytest.fixture
def stub_seeder(monkeypatch: pytest.MonkeyPatch) -> _StubSeeder:
    seeder = _StubSeeder()
    monkeypatch.setattr(docs_corpus, "seed_bundle", seeder)
    monkeypatch.setattr(docs_corpus, "build_live_pool", lambda: None)
    return seeder


class TestDigest:
    """`digest()` keys on the seed INPUTS, never on the database file."""

    def test_is_stable_across_processes(self) -> None:
        # The cache compares a digest recorded by an EARLIER process against this one's, so
        # in-process stability is not the property; a hash-seed-sensitive digest never hits.
        script = "from benchmark.routing import docs_corpus; print(docs_corpus.digest())"
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": "1"},
            cwd=Path(docs_corpus.__file__).resolve().parents[2],
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == docs_corpus.digest()

    def test_changes_when_the_registry_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before = docs_corpus.digest()
        forged = tmp_path / "models.yaml"
        forged.write_text(docs_corpus._registry_path().read_text() + "\n# nudge\n")
        monkeypatch.setenv("SHUNT_MODEL_CONFIG_PATH", str(forged))
        assert docs_corpus.digest() != before

    def test_changes_when_the_router_policy_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before = docs_corpus.digest()
        config_dir = tmp_path / "cfg"
        config_dir.mkdir()
        from shunt.router.policy import packaged_policy_path

        (config_dir / "router.yaml").write_text(packaged_policy_path().read_text() + "\n")
        (config_dir / "models.yaml").write_text(docs_corpus._registry_path().read_text())
        monkeypatch.setenv("SHUNT_CONFIG_DIR", str(config_dir))
        assert docs_corpus.digest() != before

    def test_changes_when_the_bundle_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before = docs_corpus.digest()
        forged = tmp_path / "bundle.npz"
        forged.write_bytes(docs_corpus._bundle_path().read_bytes() + b"\x00")
        monkeypatch.setattr(docs_corpus, "_bundle_path", lambda: forged)
        assert docs_corpus.digest() != before

    def test_ignores_the_database_wall_clock_columns(
        self, tmp_path: Path, stub_seeder: _StubSeeder
    ) -> None:
        # Two stores built seconds apart differ in `schema_version.applied_at`; the digest
        # must not notice, or no cache would ever hit and no figure would be reproducible.
        before = docs_corpus.digest()
        a = docs_corpus.build(tmp_path / "a")
        b = docs_corpus.build(tmp_path / "b")
        assert a.read_bytes() != b.read_bytes()
        assert docs_corpus.digest() == before


class TestCache:
    """A second build with unchanged inputs must not re-seed; `force` must."""

    def test_second_build_reuses_the_cache(self, tmp_path: Path, stub_seeder: _StubSeeder) -> None:
        dest = tmp_path / "corpus"
        first = docs_corpus.build(dest)
        second = docs_corpus.build(dest)
        assert first == second
        assert stub_seeder.calls == 1

    def test_force_reseeds(self, tmp_path: Path, stub_seeder: _StubSeeder) -> None:
        dest = tmp_path / "corpus"
        docs_corpus.build(dest)
        docs_corpus.build(dest, force=True)
        assert stub_seeder.calls == 2

    def test_a_changed_input_invalidates_the_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_seeder: _StubSeeder
    ) -> None:
        dest = tmp_path / "corpus"
        docs_corpus.build(dest)
        forged = tmp_path / "models.yaml"
        forged.write_text(docs_corpus._registry_path().read_text() + "\n# nudge\n")
        monkeypatch.setenv("SHUNT_MODEL_CONFIG_PATH", str(forged))
        docs_corpus.build(dest)
        assert stub_seeder.calls == 2

    def test_the_key_file_records_the_digest(
        self, tmp_path: Path, stub_seeder: _StubSeeder
    ) -> None:
        dest = tmp_path / "corpus"
        docs_corpus.build(dest)
        assert docs_corpus._recorded_key(dest / "corpus-key.json") == docs_corpus.digest()


@_SLOW
def test_two_full_builds_dump_sessions_identically(tmp_path: Path) -> None:
    a = docs_corpus.build(tmp_path / "a")
    b = docs_corpus.build(tmp_path / "b")
    assert _dump_sessions(a) == _dump_sessions(b)
