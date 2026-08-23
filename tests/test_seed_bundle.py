"""Seed-bundle tests: deterministic generation, incremental import, marker, staleness gate.

Fakes are the sanctioned SH008 exception: ``_FakeEmbedder`` derives a stable vector from
the text itself (never an RNG), so bundle bytes are reproducible across runs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from benchmark import config
from benchmark.routing import build_seed_bundle, seed_live
from benchmark.routing.seed_live import (
    DEFAULT_SEED_DIR,
    _resolve_bundle,
    _validate_bundle_fingerprint,
    canonical_fingerprint,
    main,
    seed_bundle,
    seed_matrix,
    should_skip_import,
)
from shunt.db.store import OutcomeStore
from shunt.router.embedder import Embedder

_BUNDLE_KEYS = (
    "cell_id",
    "routing_text",
    "model",
    "pass",
    "cost",
    "cost_known",
    "routing_text_hash",
    "content_hash",
    "embedding",
    "fingerprint",
)


def _fake_vector(text: str, dim: int = 8) -> np.ndarray:
    """A stable per-text vector derived from the text itself (never an RNG)."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    return (np.linspace(0.0, 1.0, dim) + (seed % 7) / 10.0).astype(np.float32)


class _FakeEmbedder:
    """Deterministic stand-in for the shipped Embedder (the sanctioned test exception)."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self.calls: dict[str, int] = {}

    def embed(self, text: str) -> np.ndarray:
        self.calls[text] = self.calls.get(text, 0) + 1
        return _fake_vector(text, self._dim)

    def fingerprint(self) -> dict:
        return {"model": "fake", "dim": self._dim}

    @property
    def model_name(self) -> str:
        return "fake-embedder"

    @property
    def dims(self) -> int:
        return self._dim

    @property
    def max_chars(self) -> int:
        return 1000


class _FakePool:
    def __init__(self, names: list[str]) -> None:
        self._names = list(names)

    def model_names(self) -> list[str]:
        return list(self._names)


def _matrix() -> dict:
    tasks = {
        "repo-a/task-1": {"problem_statement": "implement the foo parser"},
        "repo-b/task-2": {"problem_statement": "fix the bar overflow"},
    }
    results = {
        "repo-a/task-1": {
            "model-cheap": {"pass": True, "real_cost": 0.004, "cost": 0.004},
            "model-frontier": {"pass": True, "real_cost": 0.9, "cost": 0.9},
        },
        "repo-b/task-2": {
            "model-cheap": {"pass": False, "real_cost": 0.002, "cost": 0.002},
            "model-frontier": {"pass": True, "real_cost": 0.7, "cost": 0.7},
            "model-unmeasured": {"pass": True, "real_cost": 0.5, "cost": 0.5},
        },
    }
    return {"results": results, "tasks": tasks, "models": {"model-cheap": {}, "model-frontier": {}}}


def _write_matrix(tmp_path: Path, matrix: dict, name: str = "matrix.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(matrix))
    return p


def _load_bundle(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in _BUNDLE_KEYS}


def _sid(tid: str, model: str) -> str:
    digest = hashlib.sha256(tid.encode("utf-8")).hexdigest()[:12]
    return f"bench:{digest}:{model}"


def _pool() -> _FakePool:
    return _FakePool(["model-cheap", "model-frontier"])


@pytest.fixture
def store(tmp_path: Path) -> OutcomeStore:
    s = OutcomeStore(db_path=str(tmp_path / "o.db"))
    yield s
    s.close()


def _event_count(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    try:
        return int(con.execute("SELECT COUNT(*) FROM outcome_events").fetchone()[0])
    finally:
        con.close()


def test_generate_bundle_deterministic(tmp_path: Path) -> None:
    matrix_path = _write_matrix(tmp_path, _matrix())
    emb = _FakeEmbedder()
    path1 = build_seed_bundle.generate_bundle(matrix_path, tmp_path / "out1", emb)
    path2 = build_seed_bundle.generate_bundle(matrix_path, tmp_path / "out2", emb)

    assert path1.exists()
    assert path2.exists()
    # Two independent generations MUST be byte-identical; comparing the same final file to
    # itself would prove nothing about determinism.
    assert path1.read_bytes() == path2.read_bytes()

    with np.load(path1, allow_pickle=False) as data:
        cells = [str(x) for x in data["cell_id"]]
        assert cells == sorted(cells)
        assert set(data.keys()) == set(_BUNDLE_KEYS)
        assert data["embedding"].shape == (5, 8)
        assert data["embedding"].dtype == np.float32
        assert data["pass"].dtype.kind == "i"
        assert str(data["fingerprint"][0]) == canonical_fingerprint(emb.fingerprint())

    manifest = json.loads((tmp_path / "out1" / "manifest.json").read_text())
    canonical = canonical_fingerprint(emb.fingerprint())
    assert canonical in manifest
    assert manifest[canonical]["n_cells"] == 5
    assert manifest[canonical]["dim"] == 8
    assert manifest[canonical]["embedder"] == "fake-embedder"
    assert manifest[canonical]["file"] == path1.name
    assert (
        manifest[canonical]["results_digest"]
        == hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    )
    assert "applied_at" not in manifest[canonical]


def test_import_from_bundle_idempotent(tmp_path: Path) -> None:
    out = tmp_path / "seed"
    matrix_path = _write_matrix(tmp_path, _matrix())
    emb = _FakeEmbedder()
    bundle_path = build_seed_bundle.generate_bundle(matrix_path, out, emb)
    bundle = _load_bundle(bundle_path)
    store = OutcomeStore(db_path=str(tmp_path / "o.db"))
    try:
        pool = _pool()
        first = seed_bundle(store, bundle, pool, fingerprint=emb.fingerprint())
        second = seed_bundle(store, bundle, pool, fingerprint=emb.fingerprint())

        assert first.bundle_used is True
        assert first.cells_new == 4
        assert first.cells_model_skipped == 1
        assert first.cells_seeded == 4
        assert first.tasks == 2

        assert second.cells_unchanged == 4
        assert second.cells_new == 0
        assert second.cells_seeded == 4
        assert store.count_outcomes() == 4
        assert _event_count(str(tmp_path / "o.db")) == 4
        assert store.load_embedding_fingerprint() == emb.fingerprint()

        row = store.get_session(_sid("repo-a/task-1", "model-cheap"))
        assert row is not None
        assert row["session_id"].startswith("bench:")
        prov = json.loads(row["decision_provenance"])
        assert prov["seed_content_hash"]
        assert prov["model_chosen"] == "model-cheap"
    finally:
        store.close()


def test_import_bundle_updated_and_new_cell(tmp_path: Path) -> None:
    out = tmp_path / "seed"
    matrix_path = _write_matrix(tmp_path, _matrix())
    emb = _FakeEmbedder()
    bundle = _load_bundle(build_seed_bundle.generate_bundle(matrix_path, out, emb))
    store = OutcomeStore(db_path=str(tmp_path / "o.db"))
    try:
        pool = _pool()
        seed_bundle(store, bundle, pool, fingerprint=emb.fingerprint())

        changed = _matrix()
        changed["results"]["repo-a/task-1"]["model-cheap"]["pass"] = False
        changed["results"]["repo-c/task-3"] = {
            "model-cheap": {"pass": True, "real_cost": 0.001, "cost": 0.001},
        }
        changed["tasks"]["repo-c/task-3"] = {"problem_statement": "add the retry loop"}
        changed["models"] = {"model-cheap": {}, "model-frontier": {}}
        bundle2 = _load_bundle(
            build_seed_bundle.generate_bundle(
                _write_matrix(tmp_path, changed, "matrix2.json"), out, emb
            )
        )
        report = seed_bundle(store, bundle2, pool, fingerprint=emb.fingerprint())

        assert report.cells_updated == 1
        assert report.cells_new == 1
        assert report.cells_unchanged == 3
        assert report.cells_seeded == 5
        assert store.count_outcomes() == 5

        # The changed cell's OUTCOME must actually flip: the run_signature is content-addressed,
        # so a new content_hash yields a NEW event instead of being idempotency-dropped.
        flipped = store.get_outcome(_sid("repo-a/task-1", "model-cheap"))
        assert flipped is not None
        assert flipped["tier2_outcome"] == "failure"
        assert _event_count(str(tmp_path / "o.db")) == 6  # 4 first import + changed + new cell
    finally:
        store.close()


def test_seed_matrix_incremental_embed(tmp_path: Path) -> None:
    store = OutcomeStore(db_path=str(tmp_path / "o.db"))
    try:
        emb = _FakeEmbedder()
        first = seed_matrix(store, _matrix(), _pool(), embedder=emb)
        calls_after_first = dict(emb.calls)
        assert first.cells_new == 4

        second = seed_matrix(store, _matrix(), _pool(), embedder=emb)
        assert second.cells_unchanged == 4
        assert second.cells_new == 0
        assert second.bundle_used is False
        assert emb.calls == calls_after_first  # unchanged cells are NOT re-embedded
    finally:
        store.close()


def test_seed_state_marker_round_trip(store: OutcomeStore) -> None:
    marker = {
        "fingerprint": '{"dim": 8, "model": "fake"}',
        "results_digest": "abc",
        "applied_at": "t0",
    }
    store.save_seed_state(marker)
    assert store.load_seed_state() == marker
    store.save_seed_state({"fingerprint": "fp2", "results_digest": "def", "applied_at": "t1"})
    assert store.load_seed_state()["results_digest"] == "def"


def test_seed_state_key_isolation(store: OutcomeStore) -> None:
    # save_seed_state lives under its OWN router_state key (seed_state) and must not
    # disturb the sibling consumers (exploration_state / embedding_fingerprint /
    # escalation_state) sharing the generic KV table.
    store.save_router_state({"budget_cap": 0.5, "slack": 1})
    store.save_embedding_fingerprint({"model": "jina", "dim": 768})
    store.save_escalation_state({"repeated_failures": 3})

    store.save_seed_state({"fingerprint": '{"dim": 8}', "results_digest": "abc", "applied_at": "t"})

    assert store.load_seed_state()["results_digest"] == "abc"
    assert store.load_router_state() == {"budget_cap": 0.5, "slack": 1}
    assert store.load_embedding_fingerprint() == {"model": "jina", "dim": 768}
    assert store.load_escalation_state() == {"repeated_failures": 3}


def test_skip_and_force(store: OutcomeStore) -> None:
    store.save_seed_state({"fingerprint": "f", "results_digest": "abc", "applied_at": "t"})
    assert should_skip_import(store, "abc", None, force=False) is True
    assert should_skip_import(store, "abc", None, force=True) is False
    assert should_skip_import(store, "xyz", None, force=False) is False


def test_skip_checks_challenges_digest(store: OutcomeStore) -> None:
    store.save_seed_state(
        {
            "fingerprint": "f",
            "results_digest": "abc",
            "challenges_digest": "def",
            "applied_at": "t",
        }
    )
    # Both digests match → skip.
    assert should_skip_import(store, "abc", "def", force=False) is True
    # A problem_statement edit moves challenges.json → the marker no longer matches → re-import.
    assert should_skip_import(store, "abc", "CHANGED", force=False) is False
    # JSON-matrix override context (no challenges digest in play): results_digest is the anchor.
    assert should_skip_import(store, "abc", None, force=False) is True
    # --force wins regardless of digest state.
    assert should_skip_import(store, "abc", "CHANGED", force=True) is False
    assert should_skip_import(store, "abc", "def", force=True) is False


def test_discover_bundle_no_match_returns_none(tmp_path: Path) -> None:
    out = tmp_path / "seed"
    matrix_path = _write_matrix(tmp_path, _matrix())
    emb = _FakeEmbedder()
    build_seed_bundle.generate_bundle(matrix_path, out, emb)
    manifest_path = out / "manifest.json"

    # A bare auto-discover for a fingerprint with no manifest entry returns None — the
    # caller falls back to live-embedding instead of hard-failing.
    other = {"model": "other-embedder", "dim": 8}
    assert _resolve_bundle("", manifest_path, other) is None

    found = _resolve_bundle("", manifest_path, emb.fingerprint())
    expected_name = (
        f"seed_{build_seed_bundle.fingerprint_short(canonical_fingerprint(emb.fingerprint()))}.npz"
    )
    assert found == out / expected_name
    assert found.exists()


class _OtherFakeEmbedder(_FakeEmbedder):
    """A bundle-generating embedder whose space differs from ``_FakeEmbedder`` at equal dims."""

    @property
    def model_name(self) -> str:
        return "other-fake-embedder"

    def fingerprint(self) -> dict:
        return {"model": "other-fake", "dim": 8}


def test_bare_from_bundle_no_match_falls_back_to_live_embed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No committed bundle for the active fingerprint — a bare --from-bundle must fall
    # back to live-embedding (warning + bundle_used False) and still seed the store.
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    matrix_path = _write_matrix(tmp_path, _matrix())
    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(seed_live, "DEFAULT_SEED_DIR", seed_dir)
    monkeypatch.setattr(seed_live, "build_live_pool", lambda: _pool())
    monkeypatch.setattr(seed_live, "Embedder", _FakeEmbedder)

    rc = main(["--matrix", str(matrix_path), "--from-bundle"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "no seed bundle for fingerprint" in captured.out
    assert "live-embedding instead" in captured.out
    assert "bundle used                  False" in captured.out
    store = OutcomeStore(db_path=str(tmp_path / "outcomes.db"))
    try:
        assert store.count_outcomes() == 4
        marker = store.load_seed_state()
        assert marker is not None
        assert marker["results_digest"] == hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    finally:
        store.close()


def test_explicit_bundle_wrong_fingerprint_hard_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An EXPLICIT --from-bundle PATH in a foreign embedding space must still hard-error —
    # only the bare/auto-discover path falls back to live-embedding.
    out = tmp_path / "seed"
    matrix_path = _write_matrix(tmp_path, _matrix())
    bundle_path = build_seed_bundle.generate_bundle(matrix_path, out, _OtherFakeEmbedder())
    monkeypatch.setattr(seed_live, "build_live_pool", lambda: _pool())
    monkeypatch.setattr(seed_live, "Embedder", _FakeEmbedder)

    with pytest.raises(SystemExit) as exc:
        main(["--matrix", str(matrix_path), "--from-bundle", str(bundle_path), "--dry-run"])
    msg = str(exc.value)
    assert canonical_fingerprint(_FakeEmbedder().fingerprint()) in msg
    assert canonical_fingerprint(_OtherFakeEmbedder().fingerprint()) in msg
    assert "never import a foreign-space bundle" in msg


def test_explicit_bundle_missing_path_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An EXPLICIT --from-bundle PATH that does not exist must fail with a clean message
    # naming the path and the fix — never a raw np.load FileNotFoundError traceback.
    matrix_path = _write_matrix(tmp_path, _matrix())
    monkeypatch.setattr(seed_live, "build_live_pool", lambda: _pool())
    monkeypatch.setattr(seed_live, "Embedder", _FakeEmbedder)

    missing = tmp_path / "nope.npz"
    with pytest.raises(SystemExit) as exc:
        main(["--matrix", str(matrix_path), "--from-bundle", str(missing), "--dry-run"])
    msg = str(exc.value)
    assert str(missing) in msg
    assert "does not exist" in msg
    assert "make seed-bundle" in msg


def test_import_bundle_wrong_fingerprint_errors(tmp_path: Path) -> None:
    out = tmp_path / "seed"
    matrix_path = _write_matrix(tmp_path, _matrix())
    emb = _FakeEmbedder()
    bundle_path = build_seed_bundle.generate_bundle(matrix_path, out, emb)
    bundle = _load_bundle(bundle_path)

    # The in-bundle fingerprint must match the configured embedder space — a wrong-space
    # bundle at equal dims must be refused loudly, never stamped over.
    other = {"model": "other-embedder", "dim": 8}
    with pytest.raises(SystemExit) as exc:
        _validate_bundle_fingerprint(bundle, other, bundle_path)
    msg = str(exc.value)
    assert canonical_fingerprint(emb.fingerprint()) in msg
    assert canonical_fingerprint(other) in msg
    assert "make seed-bundle" in msg

    _validate_bundle_fingerprint(bundle, emb.fingerprint(), bundle_path)


def test_check_bundle_staleness(tmp_path: Path) -> None:
    out = tmp_path / "seed"
    results = tmp_path / "results.csv"
    results.write_text("challenge_id,model,pass,cost\n")
    matrix_path = _write_matrix(tmp_path, _matrix(), "matrix.json")
    emb = _FakeEmbedder()
    build_seed_bundle.generate_bundle(matrix_path, out, emb)
    manifest_path = out / "manifest.json"

    current, _ = build_seed_bundle.check_bundle(manifest_path, matrix_path, emb.fingerprint())
    assert current is True
    stale, message = build_seed_bundle.check_bundle(manifest_path, results, emb.fingerprint())
    assert stale is False
    assert "run make seed-bundle" in message
    missing, message = build_seed_bundle.check_bundle(
        tmp_path / "nope" / "manifest.json", matrix_path, emb.fingerprint()
    )
    assert missing is False
    assert "run make seed-bundle" in message


def test_check_bundle_catches_challenges_change(tmp_path: Path) -> None:
    real_chal = config.challenges_path()
    if not real_chal.exists():
        pytest.skip("no committed challenges.json to compare against")
    emb = _FakeEmbedder()

    # A results.csv-built entry records challenges_digest, so an edited challenges file
    # must read as stale while the committed one reads as current.
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "challenge_id,model,pass,cost\nastropy__astropy-12907,model-cheap,True,0.004\n"
    )
    out = tmp_path / "seed"
    build_seed_bundle.generate_bundle(csv_path, out, emb)
    manifest_path = out / "manifest.json"
    current, _ = build_seed_bundle.check_bundle(
        manifest_path, csv_path, emb.fingerprint(), challenges_path=real_chal
    )
    assert current is True
    edited = tmp_path / "challenges-edited.json"
    edited.write_text(real_chal.read_text() + "\n// edited\n")
    stale, message = build_seed_bundle.check_bundle(
        manifest_path, csv_path, emb.fingerprint(), challenges_path=edited
    )
    assert stale is False
    assert "challenges.json" in message

    # A JSON matrix carries its own texts and records no challenges_digest → no-op.
    json_matrix = _write_matrix(tmp_path, _matrix(), "matrix.json")
    out2 = tmp_path / "seed2"
    build_seed_bundle.generate_bundle(json_matrix, out2, emb)
    current2, _ = build_seed_bundle.check_bundle(
        out2 / "manifest.json", json_matrix, emb.fingerprint(), challenges_path=edited
    )
    assert current2 is True


def test_from_bundle_stale_matrix_warns_but_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The bundle's recorded results_digest differs from the CURRENT matrix's sha256 — the
    # import must still run (the cells are real) but print a loud staleness warning instead
    # of silently importing old cells and stamping the new digest into the seed marker.
    seed_dir = tmp_path / "seed"
    matrix_path = _write_matrix(tmp_path, _matrix(), "matrix.json")
    emb = _FakeEmbedder()
    bundle_path = build_seed_bundle.generate_bundle(matrix_path, seed_dir, emb)

    changed = _matrix()
    changed["results"]["repo-a/task-1"]["model-cheap"]["real_cost"] = 0.005
    matrix_path.write_text(json.dumps(changed))
    current = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    recorded = json.loads((seed_dir / "manifest.json").read_text())[
        canonical_fingerprint(emb.fingerprint())
    ]["results_digest"]
    assert recorded != current

    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(seed_live, "DEFAULT_SEED_DIR", seed_dir)
    monkeypatch.setattr(seed_live, "build_live_pool", lambda: _pool())
    monkeypatch.setattr(seed_live, "Embedder", _FakeEmbedder)

    rc = main(["--matrix", str(matrix_path), "--from-bundle", str(bundle_path)])

    captured = capsys.readouterr()
    assert rc == 0
    assert "STALE seed bundle" in captured.out
    assert recorded in captured.out
    assert current in captured.out
    assert "make seed-bundle" in captured.out
    store = OutcomeStore(db_path=str(tmp_path / "outcomes.db"))
    try:
        assert store.count_outcomes() == 4
        marker = store.load_seed_state()
        assert marker is not None and marker["results_digest"] == current
    finally:
        store.close()


def test_from_bundle_fresh_matrix_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A bundle whose recorded results_digest matches the current matrix imports cleanly with
    # NO staleness warning.
    seed_dir = tmp_path / "seed"
    matrix_path = _write_matrix(tmp_path, _matrix(), "matrix.json")
    bundle_path = build_seed_bundle.generate_bundle(matrix_path, seed_dir, _FakeEmbedder())

    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(seed_live, "DEFAULT_SEED_DIR", seed_dir)
    monkeypatch.setattr(seed_live, "build_live_pool", lambda: _pool())
    monkeypatch.setattr(seed_live, "Embedder", _FakeEmbedder)

    rc = main(["--matrix", str(matrix_path), "--from-bundle", str(bundle_path)])

    captured = capsys.readouterr()
    assert rc == 0
    assert "STALE seed bundle" not in captured.out
    assert "bundle used                  True" in captured.out


def test_committed_seed_bundle_not_stale() -> None:
    manifest_path = DEFAULT_SEED_DIR / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("no committed seed bundle yet — generate via make seed-bundle")
    try:
        fingerprint = Embedder().fingerprint()
    except Exception:
        pytest.skip("could not compute the active-embedder fingerprint")
    current, message = build_seed_bundle.check_bundle(
        manifest_path, config.results_csv_path(), fingerprint, config.challenges_path()
    )
    assert current, message
