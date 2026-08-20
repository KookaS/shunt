"""Seed the LIVE outcome store from the benchmark's MEASURED matrix.

One measured cell → one ``bench:<digest>:<model>`` session + verified outcome.
Incremental, skippable (``seed_state`` marker), and runnable from a git-LFS bundle.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import numpy as np

from benchmark import config
from benchmark.config import flatten_default_arm
from benchmark.routing.strategies import routing_text
from shunt.db.store import OutcomeEvent, OutcomeStore, SessionProvenance
from shunt.models import ModelPool
from shunt.router.embedder import Embedder
from shunt.router.policy import load_router_policy

_DIGEST_LEN = 12
_SOURCE = "benchmark_seed"
# Every row the seeder writes carries this stamp instead of the wall clock, so two builds
# from the same bundle produce the same corpus content. Two consequences, both intended:
# seeded rows fall out of `loop_health`'s recent_choices window (which is what stops the
# collapse alarm reading the benchmark matrix as recent router behaviour) and they sort last
# in session listings.
#
# Reproducibility here is defined over the SEED INPUTS and the CORPUS CONTENT, never over the
# SQLite file: `schema_version.applied_at` and `router_state.updated_at` are store-lifecycle
# columns and stay wall-clock by design, so the .db bytes still differ build-to-build. A cache
# key or digest over this corpus must be taken on the bundle + models.yaml + router.yaml, not
# on the database.
SEED_EPOCH: Final = "2020-01-01T00:00:00+00:00"
# The committed bundle store — LFS-tracked .npz files + a plain, diffable manifest.json.
DEFAULT_SEED_DIR: Final = Path(__file__).resolve().parent / "data" / "seed"


class _Embedder(Protocol):
    """The embedder surface seeding needs: one float32 vector per text, plus its space."""

    def embed(self, text: str) -> np.ndarray: ...

    def fingerprint(self) -> dict[str, object]: ...


class _LivePool(Protocol):
    """The live-model membership surface (``ModelPool.model_names``)."""

    def model_names(self) -> list[str]: ...


@dataclass(frozen=True)
class SeedReport:
    """What a seeding pass did — counters for the CLI block and the tests."""

    tasks: int
    cells_seeded: int
    cells_imputed_skipped: int
    cells_model_skipped: int
    cells_embed_failed: int
    cells_empty_text: int
    models: tuple[str, ...]
    total_cost: float
    per_model: dict[str, int]
    cells_new: int = 0
    cells_updated: int = 0
    cells_unchanged: int = 0
    bundle_used: bool = False


def _task_digest(task_id: str) -> str:
    """A stable 12-char key for a task id (ids can hold slashes/spaces)."""
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:_DIGEST_LEN]


def cell_id(task_id: str, model: str) -> str:
    """The deterministic session id one measured (task, model) cell maps to."""
    return f"bench:{_task_digest(task_id)}:{model}"


def cell_content_hash(
    task_id: str, model: str, passed: bool, cost: float, text: str, cost_known: bool = False
) -> str:
    """The FULL content identity of one measured cell — what makes a change detectable."""
    payload = json.dumps(
        {
            "task_id": task_id,
            "model": model,
            "pass": bool(passed),
            "cost": float(cost or 0.0),
            "cost_known": bool(cost_known),
            "routing_text": text,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_fingerprint(fingerprint: dict) -> str:
    """The canonical, sort-stable string identity of an embedder's corpus space."""
    return json.dumps(fingerprint, sort_keys=True)


def file_sha256(path: Path) -> str:
    """sha256 of the raw bytes at *path* — the digest that proves a matrix unchanged."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell_cost(cell: dict) -> tuple[float, bool]:
    """The provider's billed cost — ``real_cost``, else ``cost``, else unknown."""
    raw = cell.get("real_cost")
    if raw is None or raw == "":
        raw = cell.get("cost")
    if raw is None or raw == "":
        return 0.0, False
    return float(raw), True


def _decision_provenance(model: str, content_hash: str) -> dict:
    return {
        "model_chosen": model,
        "selection_rule_used": "benchmark_seed",
        "fallback_chain_triggered": False,
        "router_propensity": None,
        "auto_escalated": False,
        "seed_content_hash": content_hash,
    }


def _seeded_candidates(cells: dict, live: set[str]) -> tuple[list[tuple[str, dict]], int, int]:
    """``(model, cell)`` pairs passing imputed + live-model filters, and skip counts."""
    candidates: list[tuple[str, dict]] = []
    n_imputed = 0
    n_model = 0
    for model, cell in sorted(cells.items()):
        if cell.get("imputed"):
            n_imputed += 1
        elif model not in live:
            n_model += 1
        else:
            candidates.append((model, cell))
    return candidates, n_imputed, n_model


def _seeded_hash(session: dict) -> str | None:
    """The ``seed_content_hash`` a session row was written under, or None if unseeded."""
    try:
        prov = json.loads(session.get("decision_provenance") or "{}")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(prov, dict):
        return None
    value = prov.get("seed_content_hash")
    return value if isinstance(value, str) else None


def _cell_status(store: OutcomeStore | None, session_id: str, content_hash: str) -> str:
    """``new`` (no row / different content), ``updated``, or ``unchanged`` vs the store."""
    if store is None:
        return "new"
    session = store.get_session(session_id)
    if session is None:
        return "new"
    return "unchanged" if _seeded_hash(session) == content_hash else "updated"


def _write_cell(
    store: OutcomeStore,
    *,
    session_id: str,
    text: str,
    embedding: np.ndarray,
    model: str,
    cost: float,
    cost_known: bool,
    outcome: str,
    digest: str,
    content_hash: str,
) -> None:
    store.store_session(
        session_id=session_id,
        prompt_text=text,
        embedding=embedding,
        model_chosen=model,
        cost=cost,
        cache_stats={},
        duration=0.0,
        decision_provenance=_decision_provenance(model, content_hash),
        provenance=SessionProvenance(cost_known=cost_known),
        timestamp=SEED_EPOCH,
    )
    store.append_outcome_event(
        OutcomeEvent(
            session_id=session_id,
            tier=2,
            source=_SOURCE,
            outcome=outcome,
            confidence=1.0,
            # Content-addressed: a changed cell gets a NEW event (so the materialized
            # tier2_outcome flips), while an unchanged cell re-imported under the same
            # content stays idempotent and is deduplicated by the event log.
            run_signature=f"bench:{digest}:{model}:{content_hash}",
            created_at=SEED_EPOCH,
        )
    )


def seed_matrix(
    store: OutcomeStore | None,
    matrix: dict,
    pool: _LivePool,
    embedder: _Embedder | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> SeedReport:
    """Seed every measured (task, model) cell in the pool into *store*, incrementally."""
    if not dry_run:
        embedder = embedder or Embedder()
    results = matrix.get("results", {})
    tasks_meta = matrix.get("tasks", {})
    live = set(pool.model_names())
    emb_cache: dict[str, np.ndarray] = {}
    per_model: dict[str, int] = {}
    tasks_seeded = cells_seeded = n_imputed = n_model = n_embed = n_empty = 0
    cells_new = cells_updated = cells_unchanged = 0
    total_cost = 0.0

    for tid in sorted(results):
        if limit is not None and tasks_seeded >= limit:
            break
        candidates, imp, mod = _seeded_candidates(results[tid], live)
        n_imputed += imp
        n_model += mod
        text = routing_text(tid, tasks_meta.get(tid, {})).strip()
        if not text:
            n_empty += len(candidates)
            continue
        digest = _task_digest(tid)
        # Per-cell incremental diff: an unchanged cell is never re-embedded, so a
        # re-run after the matrix barely moved embeds only the changed/new tasks.
        deltas: list[tuple[str, dict, str, str]] = []
        for model, cell in candidates:
            cost, cost_known = _cell_cost(cell)
            content_hash = cell_content_hash(
                tid, model, bool(cell.get("pass")), cost, text, cost_known
            )
            deltas.append(
                (model, cell, content_hash, _cell_status(store, cell_id(tid, model), content_hash))
            )
        if any(status != "unchanged" for _, _, _, status in deltas) and not dry_run:
            assert embedder is not None
            if tid not in emb_cache:
                try:
                    emb_cache[tid] = embedder.embed(text)
                except Exception:
                    n_embed += len(candidates)
                    continue
            emb = emb_cache[tid]
        else:
            emb = np.zeros(0, dtype=np.float32)
        seeded_this_task = 0
        for model, cell, content_hash, status in deltas:
            session_id = cell_id(tid, model)
            cost, cost_known = _cell_cost(cell)
            outcome = "success" if cell.get("pass") else "failure"
            if dry_run:
                print(
                    f"would seed {session_id}  task={tid}  model={model}  "
                    f"outcome={outcome}  cost={cost:.6f}"
                )
                cells_new += 1
            elif status == "unchanged":
                cells_unchanged += 1
            else:
                assert store is not None
                _write_cell(
                    store,
                    session_id=session_id,
                    text=text,
                    embedding=emb,
                    model=model,
                    cost=cost,
                    cost_known=cost_known,
                    outcome=outcome,
                    digest=digest,
                    content_hash=content_hash,
                )
                if status == "new":
                    cells_new += 1
                else:
                    cells_updated += 1
            cells_seeded += 1
            seeded_this_task += 1
            total_cost += cost
            per_model[model] = per_model.get(model, 0) + 1
        if seeded_this_task:
            tasks_seeded += 1

    if not dry_run and store is not None and cells_new + cells_updated > 0:
        assert embedder is not None
        store.rebuild_index()
        store.persist_index()
        # Stamp the corpus space so the server's _resolve_embedding_trust adopts the
        # seeded index at boot; a seeded corpus without a fingerprint reads as a
        # pre-fingerprint DB and stays cold-start. Test fakes may lack fingerprint().
        fingerprint = getattr(embedder, "fingerprint", None)
        if fingerprint is not None:
            store.save_embedding_fingerprint(fingerprint())
    return SeedReport(
        tasks=tasks_seeded,
        cells_seeded=cells_seeded,
        cells_imputed_skipped=n_imputed,
        cells_model_skipped=n_model,
        cells_embed_failed=n_embed,
        cells_empty_text=n_empty,
        models=tuple(sorted(per_model)),
        total_cost=total_cost,
        per_model=per_model,
        cells_new=cells_new,
        cells_updated=cells_updated,
        cells_unchanged=cells_unchanged,
        bundle_used=False,
    )


def _digest_from_cell_id(session_id: str) -> str:
    """The ``bench:<digest>`` task digest embedded in a seeded session id."""
    return session_id.split(":", 2)[1]


def seed_bundle(
    store: OutcomeStore | None,
    bundle: dict,
    pool: _LivePool,
    limit: int | None = None,
    dry_run: bool = False,
    fingerprint: dict | None = None,
) -> SeedReport:
    """Seed the store from a precomputed bundle (no embedding) — the LFS warm-start path."""
    live = set(pool.model_names())
    order = sorted(range(len(bundle["cell_id"])), key=lambda i: str(bundle["cell_id"][i]))
    per_model: dict[str, int] = {}
    tasks_seeded = cells_seeded = n_imputed = n_model = n_empty = 0
    cells_new = cells_updated = cells_unchanged = 0
    total_cost = 0.0
    groups: list[tuple[str, list[int]]] = []
    for i in order:
        key = str(bundle["cell_id"][i]).rsplit(":", 1)[0]
        if groups and groups[-1][0] == key:
            groups[-1][1].append(i)
        else:
            groups.append((key, [i]))
    for _, cell_idx in groups:
        if limit is not None and tasks_seeded >= limit:
            break
        seeded_this_task = 0
        for i in cell_idx:
            session_id = str(bundle["cell_id"][i])
            model = str(bundle["model"][i])
            if model not in live:
                n_model += 1
                continue
            text = str(bundle["routing_text"][i]).strip()
            if not text:
                n_empty += 1
                continue
            content_hash = str(bundle["content_hash"][i])
            cost = float(bundle["cost"][i])
            cost_known = bool(bundle["cost_known"][i])
            outcome = "success" if bool(bundle["pass"][i]) else "failure"
            status = "new" if dry_run else _cell_status(store, session_id, content_hash)
            if dry_run:
                print(f"would seed {session_id}  model={model}  outcome={outcome}  cost={cost:.6f}")
                cells_new += 1
            elif status == "unchanged":
                cells_unchanged += 1
            else:
                assert store is not None
                _write_cell(
                    store,
                    session_id=session_id,
                    text=text,
                    embedding=np.asarray(bundle["embedding"][i], dtype=np.float32),
                    model=model,
                    cost=cost,
                    cost_known=cost_known,
                    outcome=outcome,
                    digest=_digest_from_cell_id(session_id),
                    content_hash=content_hash,
                )
                if status == "new":
                    cells_new += 1
                else:
                    cells_updated += 1
            cells_seeded += 1
            seeded_this_task += 1
            total_cost += cost
            per_model[model] = per_model.get(model, 0) + 1
        if seeded_this_task:
            tasks_seeded += 1

    if not dry_run and store is not None and cells_new + cells_updated > 0:
        store.rebuild_index()
        store.persist_index()
        # Stamp the corpus space exactly as seed_matrix does — a seeded corpus without a
        # fingerprint reads as a pre-fingerprint DB and stays cold-start at boot.
        if fingerprint is not None:
            store.save_embedding_fingerprint(fingerprint)
    return SeedReport(
        tasks=tasks_seeded,
        cells_seeded=cells_seeded,
        cells_imputed_skipped=n_imputed,
        cells_model_skipped=n_model,
        cells_embed_failed=0,
        cells_empty_text=n_empty,
        models=tuple(sorted(per_model)),
        total_cost=total_cost,
        per_model=per_model,
        cells_new=cells_new,
        cells_updated=cells_updated,
        cells_unchanged=cells_unchanged,
        bundle_used=True,
    )


def load_matrix(path: str | Path | None) -> dict:
    """The scored matrix + task texts, from a results.csv or a JSON matrix file."""
    p = Path(path) if path else config.challenges_path()
    if p.suffix.lower() == ".csv":
        results = flatten_default_arm(config.load_results(p))
        tasks = dict(config.load_challenges().get("tasks", {}))
        return {"results": results, "tasks": tasks, "models": config.models_matrix(results)}
    return config.load_matrix(p)


def build_live_pool() -> ModelPool:
    """The live-routable model set, restricted exactly as the server restricts it."""
    pool = ModelPool(config_path=os.environ.get("SHUNT_MODEL_CONFIG_PATH"))
    pool.restrict_to_live(load_router_policy().models)
    return pool


def _load_bundle(path: Path) -> dict:
    """Read a seed bundle's arrays (materialized — never keep an open NpzFile handle)."""
    if not path.exists():
        raise SystemExit(
            f"seed bundle {path} does not exist — there is nothing to import. "
            "Regenerate it with 'make seed-bundle', or pass a PATH that exists."
        )
    keys = (
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
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in keys}


def _validate_bundle_fingerprint(bundle: dict, fingerprint: dict, path: Path) -> None:
    """Hard-fail on a foreign-space bundle — never import, never stamp over one.

    The bundle records the canonical fingerprint string of the embedder it was built with.
    A mismatch (even at equal dims) names both fingerprints and the fix: 'make seed-bundle'.
    """
    canonical = canonical_fingerprint(fingerprint)
    stored = str(bundle["fingerprint"][0])
    if stored != canonical:
        raise SystemExit(
            f"seed bundle {path} was built for embedding fingerprint {stored!r}, "
            f"not the configured {canonical!r} — never import a foreign-space bundle. "
            "Regenerate with 'make seed-bundle'."
        )


def _stale_bundle_warning(
    manifest_path: Path,
    fingerprint: dict,
    results_digest: str,
    challenges_digest: str | None,
) -> str | None:
    """A loud non-fatal warning when the bundle predates the current matrix, else None.

    Stale = the manifest's recorded digests differ from the current files'; importing writes
    OLD cells yet stamps CURRENT digests, invisible to everything but 'make check-seed-bundle'.
    """
    if not manifest_path.exists():
        return None
    entry = json.loads(manifest_path.read_text()).get(canonical_fingerprint(fingerprint))
    if entry is None:
        return None
    mismatches: list[str] = []
    recorded = entry.get("results_digest")
    if recorded is not None and recorded != results_digest:
        mismatches.append(f"results_digest {recorded} != current {results_digest}")
    recorded_chal = entry.get("challenges_digest")
    if (
        recorded_chal is not None
        and challenges_digest is not None
        and recorded_chal != challenges_digest
    ):
        mismatches.append(f"challenges_digest {recorded_chal} != current {challenges_digest}")
    if not mismatches:
        return None
    return (
        "WARNING: STALE seed bundle — "
        + "; ".join(mismatches)
        + ". The import writes the bundle's OLD cells and stamps the CURRENT digests into the "
        "seed marker. Regenerate with 'make seed-bundle' — a fresh bundle re-imports "
        "idempotently via per-cell content hashes."
    )


def _discover_bundle(manifest_path: Path, fingerprint: str) -> Path | None:
    """The bundle file for *fingerprint* per manifest.json, or None if absent."""
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    entry = manifest.get(fingerprint)
    if entry is None:
        return None
    return manifest_path.parent / str(entry["file"])


def _resolve_bundle(flag_value: str, manifest_path: Path, fingerprint: dict) -> Path | None:
    """Resolve ``--from-bundle`` to a bundle path; None on a bare call with no match.

    An explicit PATH is returned verbatim (a missing/foreign bundle hard-errors later),
    while a bare call returns the manifest entry for this fingerprint, or None.
    """
    if flag_value:
        return Path(flag_value)
    canonical = canonical_fingerprint(fingerprint)
    return _discover_bundle(manifest_path, canonical)


def _already_seeded(
    store: OutcomeStore, results_digest: str, challenges_digest: str | None
) -> bool:
    """True when the store's seed marker names this results digest (and challenges digest).

    When ``challenges_digest`` is None the digest is the whole anchor; otherwise the marker
    must also match it, so a problem_statement edit without a re-run never reads as seeded.
    """
    state = store.load_seed_state()
    if not state or state.get("results_digest") != results_digest:
        return False
    if challenges_digest is None:
        return True
    return state.get("challenges_digest") == challenges_digest


def should_skip_import(
    store: OutcomeStore, results_digest: str, challenges_digest: str | None, force: bool
) -> bool:
    """True when the seed marker matches and the caller did not force a re-import."""
    return not force and _already_seeded(store, results_digest, challenges_digest)


def _seed_marker(fingerprint: dict, results_digest: str, challenges_digest: str | None) -> dict:
    """The "seed applied" marker: fingerprint + digests + when it was applied."""
    marker: dict = {
        "fingerprint": canonical_fingerprint(fingerprint),
        "results_digest": results_digest,
        "applied_at": _datetime.datetime.now(_datetime.UTC).isoformat(),
    }
    if challenges_digest is not None:
        marker["challenges_digest"] = challenges_digest
    return marker


def _print_summary(report: SeedReport) -> None:
    print("── seed report ──")
    print(f"  tasks seeded                 {report.tasks}")
    print(f"  cells seeded                 {report.cells_seeded}")
    print(f"  cells skipped (imputed)      {report.cells_imputed_skipped}")
    print(f"  cells skipped (not in pool)  {report.cells_model_skipped}")
    print(f"  cells skipped (empty text)   {report.cells_empty_text}")
    print(f"  cells skipped (embed fail)   {report.cells_embed_failed}")
    print(f"  cells new                    {report.cells_new}")
    print(f"  cells updated                {report.cells_updated}")
    print(f"  cells unchanged              {report.cells_unchanged}")
    print(f"  bundle used                  {report.bundle_used}")
    print(f"  total real cost              ${report.total_cost:.4f}")
    for model in report.models:
        print(f"    {model:<20} {report.per_model[model]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--matrix",
        default=None,
        help="a results.csv or JSON matrix/challenges file (default: committed results.csv)",
    )
    ap.add_argument("--limit", type=int, default=None, help="cap on tasks seeded (quick trial)")
    ap.add_argument(
        "--dry-run", action="store_true", help="print what would be seeded, write nothing"
    )
    ap.add_argument(
        "--from-bundle",
        metavar="PATH",
        nargs="?",
        const="",
        default=None,
        help="seed from a precomputed .npz bundle (no PATH: auto-discover via manifest.json)",
    )
    ap.add_argument(
        "--force", action="store_true", help="ignore the seed-applied marker; always re-import"
    )
    args = ap.parse_args(argv)

    matrix_path = Path(args.matrix) if args.matrix else config.results_csv_path()
    digest = file_sha256(matrix_path)
    # Task TEXTS come from challenges.json only when the matrix is a results.csv; a JSON
    # matrix override is self-contained, so its file digest is the whole anchor there.
    challenges_digest: str | None = None
    if matrix_path.suffix.lower() == ".csv":
        challenges = config.challenges_path()
        if challenges.exists():
            challenges_digest = file_sha256(challenges)
    store = None
    if not args.dry_run:
        store = OutcomeStore()
        if should_skip_import(store, digest, challenges_digest, args.force):
            print(
                "already seeded (results.csv + challenges.json unchanged) — "
                "skipping; use --force to re-import"
            )
            store.close()
            return 0

    pool = build_live_pool()
    fingerprint: dict | None = None
    embedder: _Embedder | None = None
    bundle_path: Path | None = None
    if args.from_bundle is not None:
        embedder = Embedder()
        fingerprint = embedder.fingerprint()
        bundle_path = _resolve_bundle(
            args.from_bundle, DEFAULT_SEED_DIR / "manifest.json", fingerprint
        )
        if bundle_path is None:
            print(
                f"no seed bundle for fingerprint {canonical_fingerprint(fingerprint)} in "
                f"{DEFAULT_SEED_DIR / 'manifest.json'} — live-embedding instead "
                "(run 'make seed-bundle')"
            )

    if bundle_path is not None:
        assert fingerprint is not None
        bundle = _load_bundle(bundle_path)
        _validate_bundle_fingerprint(bundle, fingerprint, bundle_path)
        warning = _stale_bundle_warning(
            DEFAULT_SEED_DIR / "manifest.json", fingerprint, digest, challenges_digest
        )
        if warning is not None:
            print(warning)
        report = seed_bundle(
            store, bundle, pool, limit=args.limit, dry_run=args.dry_run, fingerprint=fingerprint
        )
    else:
        matrix = load_matrix(matrix_path)
        if not matrix.get("results"):
            if store is not None:
                store.close()
            print(
                "No results to seed — the matrix holds no measured cells. "
                "Run the live matrix first: python -m benchmark.runner.run_matrix --live"
            )
            return 0
        embedder = embedder or (None if args.dry_run else Embedder())
        report = seed_matrix(
            store, matrix, pool, embedder=embedder, limit=args.limit, dry_run=args.dry_run
        )
        if embedder is not None:
            fingerprint = embedder.fingerprint()

    _print_summary(report)
    # A --limit run is a trial, not a completion: never stamp the full matrix digest
    # (a later no-limit run must not skip) and never clobber an existing full marker.
    if not args.dry_run and store is not None and args.limit is None:
        assert fingerprint is not None
        store.save_seed_state(_seed_marker(fingerprint, digest, challenges_digest))
        print(
            f"seeded {report.cells_seeded} measured benchmark cells — the embedding signal "
            "measured at chance on this corpus; the router re-fits from live verified "
            "outcomes as they accumulate."
        )
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
