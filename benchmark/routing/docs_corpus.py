"""The seed-only outcome store the committed docs figures are rendered from.

Built from committed data alone — the LFS seed bundle plus the shipped registry and router
policy — so it needs no network, no ONNX weights and no live rig.
"""

# ``seed_bundle`` carries its own embeddings and never calls ``embed``, which is what makes a
# build model-free; a run under SHUNT_DISALLOW_REAL_EMBEDDER=1 with a bogus embed-cache dir is
# the positive proof. The store lands in a cache directory keyed on the three inputs above, so
# a second render reuses the first build instead of paying ~250 s again.

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Final

from benchmark.routing.seed_live import (
    DEFAULT_SEED_DIR,
    _load_bundle,
    _resolve_bundle,
    _validate_bundle_fingerprint,
    build_live_pool,
    file_sha256,
    seed_bundle,
)
from shunt.db.store import OutcomeStore
from shunt.models.config import default_registry_path
from shunt.router.embedder import Embedder
from shunt.router.policy import resolved_policy_path

DEFAULT_BUILD_DIR: Final = Path(__file__).resolve().parent / ".build" / "docs-seed-db"
_DB_NAME: Final = "outcomes.db"
_KEY_NAME: Final = "corpus-key.json"


def _registry_path() -> Path:
    """The models.yaml ``build_live_pool`` reads — same resolution as ``ModelPool``."""
    explicit = os.environ.get("SHUNT_MODEL_CONFIG_PATH")
    if explicit:
        candidate = Path(explicit)
    else:
        config_dir = os.environ.get("SHUNT_CONFIG_DIR")
        base = Path(config_dir) if config_dir else Path.home() / ".config" / "shunt"
        candidate = base / "models.yaml"
    return candidate if candidate.exists() else default_registry_path()


def _bundle_path() -> Path:
    """The seed bundle for the configured embedding space, via the ``--from-bundle`` path."""
    fingerprint = Embedder().fingerprint()
    path = _resolve_bundle("", DEFAULT_SEED_DIR / "manifest.json", fingerprint)
    if path is None:
        raise SystemExit(
            f"no seed bundle for the configured embedding space in "
            f"{DEFAULT_SEED_DIR / 'manifest.json'} — the docs corpus is built from committed "
            "data only and will not live-embed. Run 'make seed-bundle'."
        )
    return path


def digest() -> str:
    """The corpus identity, keyed on the seed inputs — for ``plot_frame.Provenance.data_digest``."""
    # NEVER key this on the .db file. `outcomes.created_at`, `router_state.updated_at` and
    # `schema_version.applied_at` are store-lifecycle columns that stay wall-clock by design
    # (see seed_live.SEED_EPOCH), so the database bytes differ build-to-build even when the
    # corpus content is identical.
    #
    # router.yaml and models.yaml are inputs because build_live_pool (seed_live.py:431)
    # intersects them to decide which model rows survive seeding: renaming or dropping a live
    # model changes the corpus row count and therefore the committed figures. Expect the first
    # such rename to turn the figure-staleness gate red for a reason that reads as unrelated to
    # the rename — this is that reason.
    parts = [file_sha256(_bundle_path()), file_sha256(_registry_path())]
    policy = resolved_policy_path()
    parts.append(file_sha256(policy) if policy is not None else "no-router-yaml")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _recorded_key(key_file: Path) -> str | None:
    """The digest a previous build recorded under *key_file*, or None if unreadable."""
    try:
        return str(json.loads(key_file.read_text())["digest"])
    except (OSError, ValueError, KeyError):
        return None


def build(dest: Path = DEFAULT_BUILD_DIR, *, force: bool = False) -> Path:
    """Seed a fresh store under *dest* from the committed bundle; return the database path."""
    # A build whose key file already records the current digest() is reused as-is unless *force*.
    dest = Path(dest)
    db = dest / _DB_NAME
    key_file = dest / _KEY_NAME
    key = digest()

    if not force and db.exists() and _recorded_key(key_file) == key:
        return db

    dest.mkdir(parents=True, exist_ok=True)
    # Targeted, not rmtree: *dest* is caller-supplied and may hold unrelated files.
    for stale in [*dest.glob(f"{_DB_NAME}*"), key_file]:
        stale.unlink(missing_ok=True)

    bundle_path = _bundle_path()
    fingerprint = Embedder().fingerprint()
    bundle = _load_bundle(bundle_path)
    _validate_bundle_fingerprint(bundle, fingerprint, bundle_path)

    store = OutcomeStore(db_path=str(db))
    try:
        # build_live_pool() = models.yaml x router.yaml (seed_live.py:431). It decides
        # which model rows survive, hence the row count — see digest() on the rename trap.
        seed_bundle(store, bundle, build_live_pool(), fingerprint=fingerprint)
    finally:
        store.close()

    key_file.write_text(
        json.dumps(
            {
                "digest": key,
                "bundle": str(bundle_path),
                "registry": str(_registry_path()),
                "router_policy": str(resolved_policy_path()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return db
