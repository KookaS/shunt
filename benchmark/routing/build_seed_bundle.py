"""Build the deterministic per-fingerprint seed bundle for the LIVE outcome store.

MEASURED outcome cells (routing_text + real embedding + labels + costs) shipped via git
LFS, content-addressed per cell: unchanged results.csv → byte-identical .npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from benchmark import config
from benchmark.routing.seed_live import (
    DEFAULT_SEED_DIR,
    _cell_cost,
    canonical_fingerprint,
    cell_content_hash,
    cell_id,
    file_sha256,
    load_matrix,
)
from benchmark.routing.strategies import routing_text
from shunt.router.embedder import Embedder


class _BundleEmbedder(Protocol):
    """The embedder surface a bundle needs: vectors + the identity stamped in the manifest."""

    def embed(self, text: str) -> np.ndarray: ...

    def fingerprint(self) -> dict: ...

    @property
    def model_name(self) -> str: ...

    @property
    def dims(self) -> int: ...

    @property
    def max_chars(self) -> int: ...


def fingerprint_short(canonical: str) -> str:
    """A 16-hex slug for the canonical fingerprint — the .npz filename tail."""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _build_cells(matrix: dict, embedder: _BundleEmbedder) -> list[dict]:
    """Every measured (non-imputed) cell, sorted by cell_id, hashed + embedded."""
    cells: list[dict] = []
    emb_cache: dict[str, np.ndarray] = {}
    results = matrix.get("results", {})
    tasks_meta = matrix.get("tasks", {})
    for tid in sorted(results):
        text = routing_text(tid, tasks_meta.get(tid, {})).strip()
        if not text:
            continue
        if tid not in emb_cache:
            emb_cache[tid] = embedder.embed(text)
        emb = emb_cache[tid]
        for model, cell in sorted(results[tid].items()):
            if cell.get("imputed"):
                continue
            cost, cost_known = _cell_cost(cell)
            cells.append(
                {
                    "cell_id": cell_id(tid, model),
                    "routing_text": text,
                    "model": model,
                    "pass": 1 if cell.get("pass") else 0,
                    "cost": cost,
                    "cost_known": 1 if cost_known else 0,
                    "routing_text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "content_hash": cell_content_hash(
                        tid, model, cell.get("pass"), cost, text, cost_known
                    ),
                    "embedding": emb,
                }
            )
    cells.sort(key=lambda c: c["cell_id"])
    return cells


def _array_dict(cells: list[dict], canonical: str) -> dict[str, Any]:
    """The deterministic npz payload — key order fixed, cells sorted by cell_id.

    ``fingerprint`` carries the canonical embedder-space string inside the bundle itself so
    the importer can reject a foreign-space bundle (same dims, wrong model) on every import.
    """
    return {
        "cell_id": np.array([c["cell_id"] for c in cells], dtype=str),
        "routing_text": np.array([c["routing_text"] for c in cells], dtype=str),
        "model": np.array([c["model"] for c in cells], dtype=str),
        "pass": np.array([c["pass"] for c in cells], dtype=np.int64),
        "cost": np.array([c["cost"] for c in cells], dtype=np.float64),
        "cost_known": np.array([c["cost_known"] for c in cells], dtype=np.int64),
        "routing_text_hash": np.array([c["routing_text_hash"] for c in cells], dtype=str),
        "content_hash": np.array([c["content_hash"] for c in cells], dtype=str),
        "embedding": np.stack([c["embedding"] for c in cells]).astype(np.float32),
        "fingerprint": np.array([canonical], dtype=str),
    }


def _manifest_path(out_dir: Path) -> Path:
    return out_dir / "manifest.json"


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def generate_bundle(matrix_path: Path, out_dir: Path, embedder: _BundleEmbedder) -> Path:
    """Build the deterministic .npz + manifest entry for *matrix_path*; returns the npz path."""
    matrix = load_matrix(matrix_path)
    if not matrix.get("results"):
        raise SystemExit("No results to bundle — the matrix holds no measured cells.")
    cells = _build_cells(matrix, embedder)
    canonical = canonical_fingerprint(embedder.fingerprint())
    path = out_dir / f"seed_{fingerprint_short(canonical)}.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **_array_dict(cells, canonical))
    # Task texts come from challenges.json only for a results.csv matrix; a self-contained
    # JSON matrix carries its own texts, so its file digest is the whole freshness anchor.
    challenges_digest: str | None = None
    if matrix_path.suffix.lower() == ".csv":
        challenges = config.challenges_path()
        if challenges.exists():
            challenges_digest = file_sha256(challenges)
    manifest = _load_manifest(_manifest_path(out_dir))
    manifest[canonical] = {
        "file": path.name,
        "results_digest": file_sha256(matrix_path),
        "n_cells": len(cells),
        "dim": embedder.dims,
        "embedder": embedder.model_name,
        "max_chars": embedder.max_chars,
    }
    if challenges_digest is not None:
        manifest[canonical]["challenges_digest"] = challenges_digest
    _write_manifest(_manifest_path(out_dir), manifest)
    tasks = len({c["cell_id"].rsplit(":", 1)[0] for c in cells})
    print("── seed bundle ──")
    print(f"  tasks embedded   {tasks}")
    print(f"  cells            {len(cells)}")
    print(f"  dim              {embedder.dims}")
    print(f"  results digest   {file_sha256(matrix_path)}")
    if challenges_digest is not None:
        print(f"  challenges digest {challenges_digest}")
    print(f"  file             {path}")
    return path


def check_bundle(
    manifest_path: Path,
    results_path: Path,
    fingerprint: dict,
    challenges_path: Path | None = None,
) -> tuple[bool, str]:
    """``(current, message)`` — is the active-embedder bundle present AND fresh?"""
    if not manifest_path.exists():
        return False, f"no seed bundle manifest at {manifest_path}; run make seed-bundle"
    manifest = _load_manifest(manifest_path)
    canonical = canonical_fingerprint(fingerprint)
    entry = manifest.get(canonical)
    if entry is None:
        return (
            False,
            "no seed bundle for the active embedder fingerprint "
            f"{canonical!r}; run make seed-bundle",
        )
    if entry.get("results_digest") != file_sha256(results_path):
        return (
            False,
            "seed bundle is stale (results_digest != sha256 of results.csv); run make seed-bundle",
        )
    if challenges_path is not None and challenges_path.exists():
        entry_chal = entry.get("challenges_digest")
        if entry_chal is not None and entry_chal != file_sha256(challenges_path):
            return (
                False,
                "seed bundle is stale (challenges_digest != sha256 of challenges.json); "
                "run make seed-bundle",
            )
    return True, "seed bundle is current"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--matrix",
        default=None,
        help="a results.csv or JSON matrix/challenges file (default: committed results.csv)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output directory for the .npz + manifest.json (default: benchmark/routing/data/seed)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="staleness gate: fail if the bundle is missing or stale",
    )
    args = ap.parse_args(argv)

    matrix_path = Path(args.matrix) if args.matrix else config.results_csv_path()
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_SEED_DIR
    if args.check:
        current, message = check_bundle(
            _manifest_path(out_dir), matrix_path, Embedder().fingerprint(), config.challenges_path()
        )
        print(message)
        return 0 if current else 1
    generate_bundle(matrix_path, out_dir, Embedder())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
