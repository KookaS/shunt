#!/usr/bin/env python3
"""Benchmark caching loop: find missing/stale (challenge×model) cells and
refresh artifacts. Simulated by default; --live + keys delegate to the real
SWE-bench harness executor for instances with a materialised spec.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Final

from benchmark import config
from benchmark.routing import coverage, integrity
from benchmark.routing.strategies.fixed import AlwaysCheap, AlwaysFrontier
from benchmark.routing.strategies.oracle import Oracle
from benchmark.runner import image_version, swebench_specs
from shunt.secrets import load_dotenv_file

# History log columns: the full cache row plus when the row was superseded.
HISTORY_FIELDS: Final[tuple[str, ...]] = (*integrity.RESULTS_FIELDS, "superseded_at")


def _now_iso() -> str:
    """Current UTC timestamp in ISO-8601 (audit only — never a staleness key)."""
    return datetime.now(UTC).isoformat()


_KEY_ENV: Final[tuple[str, ...]] = (
    "DEEPSEEK_API_KEY",
    "REQUESTY_API_KEY",
    "XAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)


@dataclass
class CellStatus:
    """Classification of every (challenge, model, reasoning-arm) cell."""

    missing: list[tuple[str, str, str]] = field(default_factory=list)
    stale: list[tuple[str, str, str]] = field(default_factory=list)
    present: int = 0

    @property
    def to_run(self) -> list[tuple[str, str, str]]:
        """Cells needing (re)computation: missing plus stale."""
        return self.missing + self.stale


def _has_keys() -> bool:
    return any(os.environ.get(k) for k in _KEY_ENV)


def _default_selected_arms(tasks: list[str], models: list[str]) -> dict[tuple[str, str], list[str]]:
    """Back-compat fallback: check every cell only at its declared default arm."""
    defaults = config.default_arm_ids(models)
    fallback = {model: [defaults.get(model, integrity.DEFAULT_REASONING)] for model in models}
    return {(cid, model): fallback[model] for cid in tasks for model in models}


def classify_cells(
    tasks: list[str],
    models: list[str],
    cache: dict,
    hashes: dict[str, str],
    versions: dict[str, str],
    digests: dict[str, str | None] | None = None,
    selected_arms: dict[tuple[str, str], list[str]] | None = None,
    arm_hash_map: dict[str, dict[str, str]] | None = None,
) -> CellStatus:
    """Split (challenge, model, reasoning-arm) cells into missing / stale / present."""
    # `cache` is 3-level: challenge_id -> model -> arm_id -> row.
    # `selected_arms` restricts each (challenge, model) to its sampled arms;
    # absent, every cell is checked only at its declared default arm
    # (config.default_arm_ids), reproducing the single-cell-per-model
    # behaviour from before arm sampling existed. Stale iff spec hash, image digest,
    # model version, or the arm's own api-param hash drifted; missing iff no row for that arm.
    arms_by_cell = selected_arms or _default_selected_arms(tasks, models)
    status = CellStatus()
    for cid in tasks:
        for model in models:
            for arm in arms_by_cell.get((cid, model), [integrity.DEFAULT_REASONING]):
                cell = cache.get(cid, {}).get(model, {}).get(arm)
                if cell is None:
                    status.missing.append((cid, model, arm))
                elif _is_stale(cell, cid, model, arm, hashes, versions, digests, arm_hash_map):
                    status.stale.append((cid, model, arm))
                else:
                    status.present += 1
    return status


def _is_stale(
    cell: dict,
    cid: str,
    model: str,
    arm: str,
    hashes: dict[str, str],
    versions: dict[str, str],
    digests: dict[str, str | None] | None,
    arm_hash_map: dict[str, dict[str, str]] | None = None,
) -> bool:
    """Stale iff stored spec hash, model version, arm-param hash, or image digest drifted."""
    if cell.get("version_hash") != hashes.get(cid):
        return True
    if cell.get("model_version") != versions.get(model):
        return True
    if _arm_stale(cell, model, arm, arm_hash_map):
        return True
    return _image_stale(cell, cid, digests)


def _arm_stale(
    cell: dict, model: str, arm: str, arm_hash_map: dict[str, dict[str, str]] | None
) -> bool:
    """Stale iff a NON-EMPTY stored arm_hash resolves AND differs — else never stales."""
    # Mirrors the digest-axis guard (_image_stale): no resolved anchor (arm_hash_map
    # absent, or the model/arm missing from it — e.g. a model with no reasoning block)
    # never marks a cell stale.
    if arm_hash_map is None:
        return False
    expected = arm_hash_map.get(model, {}).get(arm)
    if expected is None:
        return False
    # Guard on non-empty stored (like _image_stale): a legacy row (pre-arm-hash) has no
    # arm_hash column (""), so it degrades to a no-op instead of recomputing the
    # PAID cell. A genuinely new arm is MISSING (no row), not present-with-empty-hash.
    stored = str(cell.get("arm_hash", ""))
    return bool(stored) and stored != expected


def _image_stale(cell: dict, cid: str, digests: dict[str, str | None] | None) -> bool:
    """Stale iff a NON-EMPTY stored digest resolves AND differs — else never stales."""
    if digests is None:
        return False
    resolved = digests.get(cid)
    if resolved is None:
        return False
    # Single-arch assumption: the stored RepoDigest (docker inspect) and the resolved
    # manifest digest (imagetools inspect) coincide for a single-platform image. Guard
    # on non-empty stored (mirrors check_integrity.check_image_digests) so a first-live
    # cell — or a multi-arch manifest-list divergence that leaves the stored digest
    # unset — degrades to a no-op rather than recomputing the PAID cell forever.
    stored = cell.get("image_digest", "")
    return bool(stored) and resolved != stored


def _build_row(
    cid: str,
    model: str,
    outcome: dict,
    hashes: dict[str, str],
    versions: dict[str, str],
    pricing: dict[str, dict[str, float]],
    digests: dict[str, str | None] | None = None,
    arm: str = integrity.DEFAULT_REASONING,
    arm_hash_map: dict[str, dict[str, str]] | None = None,
) -> dict:
    in_tok = int(outcome.get("in_tok", 0))
    out_tok = int(outcome.get("out_tok", 0))
    real_cost = float(outcome.get("real_cost", 0.0))
    estimated_cost = integrity.estimated_cost(model, in_tok, out_tok, pricing)
    # `cost` is what every metric/strategy/kill-gate reads. Prefer the litellm-measured
    # (cache-aware) real_cost, but fall back to the registry's listing estimate when
    # litellm can't price the route (e.g. requesty `openai/<provider>/<model>` strings)
    # — otherwise those cells would cache cost=0 and score as free. NOTE: this makes the
    # basis mixed (measured for litellm-priceable models, listing-estimate otherwise) —
    # a known limitation to reconcile before running the live kill-gate.
    cost = real_cost if real_cost > 0 else estimated_cost
    return {
        "challenge_id": cid,
        "model": model,
        # The row's arm identity comes from the caller (the (cid, model, arm) cell
        # classify_cells decided was missing/stale), not the harness outcome —
        # we label the row with the input arm, not anything inferred from the outcome.
        "reasoning": arm,
        "pass": outcome.get("pass", False),
        "cost": cost,
        "in_tok": in_tok,
        "out_tok": out_tok,
        "calls": int(outcome.get("calls", 0)),
        "version_hash": hashes.get(cid, ""),
        "model_version": versions.get(model, integrity.UNKNOWN_VERSION),
        "arm_hash": (arm_hash_map or {}).get(model, {}).get(arm, ""),
        "real_cost": real_cost,
        "estimated_cost": estimated_cost,
        "timeout_flag": bool(outcome.get("timeout_flag", False)),
        "image_digest": _row_digest(cid, outcome, digests),
        "computed_at": str(outcome.get("computed_at") or _now_iso()),
    }


def _row_digest(cid: str, outcome: dict, digests: dict[str, str | None] | None) -> str:
    """Prefer the digest the harness actually used (outcome); else the resolved map."""
    used = outcome.get("image_digest")
    if used:
        return str(used)
    if digests and digests.get(cid):
        return str(digests[cid])
    return ""


@dataclass(frozen=True)
class _LiveContext:
    """Read-only shared inputs for one batch of live cells (thread-safe: never mutated)."""

    hashes: dict[str, str]
    versions: dict[str, str]
    pricing: dict[str, dict[str, float]]
    digests: dict[str, str | None] | None
    arm_hash_map: dict[str, dict[str, str]] | None
    work_dir: Path
    timeout: int


def _run_one_cell(cell: tuple[str, str, str], ctx: _LiveContext) -> dict | None:
    """Run one (challenge, model, arm) cell → results row, or None on ANY failure.

    Per-cell isolation: any failure skips just this cell (never fabricates); pure
    w.r.t. shared state, so it is thread-safe.
    """
    cid, model, arm = cell
    # The ENTIRE body is guarded, not just the harness call: in a worker thread an
    # unhandled raise (import, spec-hash, row-build) would propagate out of the pool's
    # result loop and discard every already-collected (paid) row. Returning None keeps
    # isolation total — one bad cell never costs the batch.
    try:
        from benchmark.routing import integrity
        from benchmark.runner import infer

        outcome = infer.run_live_cell(
            cid,
            model,
            work_dir=ctx.work_dir,
            run_id=f"live-{cid}-{model}-{arm}",
            timeout=ctx.timeout,
            arm=arm,
        )
        cell_hashes = {
            **ctx.hashes,
            cid: integrity.swebench_spec_hash(cid) or ctx.hashes.get(cid, ""),
        }
        return _build_row(
            cid,
            model,
            outcome,
            cell_hashes,
            ctx.versions,
            ctx.pricing,
            ctx.digests,
            arm,
            ctx.arm_hash_map,
        )
    except Exception as exc:  # noqa: BLE001 (per-cell isolation: skip one, never fabricate)
        print(f"  skip {cid}:{model}:{arm} — {exc}", file=sys.stderr)
        return None


def _over_budget(spent: float, max_cost: float | None) -> bool:
    """True once cumulative real_cost has reached the ceiling (None = unbounded)."""
    return max_cost is not None and spent >= max_cost


def _group_by_challenge(
    cells: list[tuple[str, str, str]],
) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Group cells by challenge, preserving each challenge's first-appearance order."""
    # Challenge-major batching: a challenge split across the (missing + stale)
    # concatenation is reunited at its first appearance, so its whole model×arm set runs
    # together and a ceiling cut leaves a prefix of FULLY-covered challenges.
    groups: dict[str, list[tuple[str, str, str]]] = {}
    order: list[str] = []
    for cell in cells:
        cid = cell[0]
        if cid not in groups:
            groups[cid] = []
            order.append(cid)
        groups[cid].append(cell)
    return [(cid, groups[cid]) for cid in order]


def _run_cells_serial(
    cells: list[tuple[str, str, str]],
    ctx: _LiveContext,
    max_cost: float | None,
    checkpoint: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Serial (workers=1), challenge-major: a ceiling cuts cleanly at a challenge boundary."""
    # The ceiling is checked BEFORE each challenge, never mid-challenge: once a challenge
    # starts its full model×arm set runs, so no started challenge is left partial. The
    # kept challenges are a prefix of the input challenge order.
    rows: list[dict] = []
    spent = 0.0
    for _cid, batch in _group_by_challenge(cells):
        if _over_budget(spent, max_cost):
            print(
                f"  cost ceiling ${max_cost:g} reached — stopping ({len(rows)} cells done)",
                file=sys.stderr,
            )
            break
        for cell in batch:
            row = _run_one_cell(cell, ctx)
            if row is not None:
                rows.append(row)
                spent += float(row["real_cost"])
                if checkpoint is not None:
                    checkpoint(row)
    return rows


def _run_challenge_batch(
    batch: list[tuple[str, str, str]], ctx: _LiveContext, pool: ThreadPoolExecutor
) -> list[dict]:
    """Run one challenge's cells concurrently; return completed rows in input order."""
    # Barrier: as_completed drains only THIS challenge's futures before the caller
    # advances, so a challenge fully completes before the next starts. Rows are
    # re-sorted by input index → byte-identical to serial for the same cell set.
    results: dict[int, dict] = {}
    futures = {pool.submit(_run_one_cell, cell, ctx): i for i, cell in enumerate(batch)}
    for fut in as_completed(futures):
        row = fut.result()
        if row is not None:
            results[futures[fut]] = row
    return [results[i] for i in sorted(results)]


def _run_cells_parallel(
    cells: list[tuple[str, str, str]],
    ctx: _LiveContext,
    workers: int,
    max_cost: float | None,
    checkpoint: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Challenge-major thread pool: each challenge's cells complete (up to ``workers`` at
    once) before the next starts, so a ceiling cut leaves a clean prefix of FULLY-covered
    challenges.
    """
    # One pool, reused across challenges; concurrency is INTRA-challenge (its model×arm
    # cells). The ceiling is checked at challenge boundaries only — a started challenge
    # always finishes (never left partial). ``checkpoint`` and ``spent`` run HERE, on the
    # main thread in input order — worker threads (_run_one_cell) never write, so the
    # read-modify-write in merge_rows stays race-free without a lock, and the persisted
    # bytes stay order-independent (identical to serial for a completed cell set).
    rows: list[dict] = []
    spent = 0.0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _cid, batch in _group_by_challenge(cells):
            if _over_budget(spent, max_cost):
                print(
                    f"  cost ceiling ${max_cost:g} reached — stopping ({len(rows)} cells done)",
                    file=sys.stderr,
                )
                break
            for row in _run_challenge_batch(batch, ctx, pool):
                rows.append(row)
                spent += float(row["real_cost"])
                if checkpoint is not None:
                    checkpoint(row)
    return rows


def run_live_cells(
    cells: list[tuple[str, str, str]],
    matrix: dict,  # noqa: ARG001 (kept for signature stability; spec is loaded per-cell)
    hashes: dict[str, str],
    versions: dict[str, str],
    timeout: int,
    verbose: bool,  # noqa: ARG001 (kept for signature stability; per-cell logging is unconditional)
    digests: dict[str, str | None] | None = None,
    arm_hash_map: dict[str, dict[str, str]] | None = None,
    workers: int = 1,
    max_cost: float | None = None,
    results_path: Path | None = None,
) -> list[dict]:
    """Delegate each (challenge, model, arm) cell to the real SWE-bench harness executor.

    ``workers`` runs cells concurrently; ``max_cost`` caps spend; ``results_path``
    persists each cell as it completes so a kill loses only in-flight cells.
    """
    # Structural wall (defense-in-depth): never spend uncached budget. An uncached
    # model resends the full context every agent turn at full input price — a cell that
    # churns then fails still bills for all of it (qwen3-max billed 50x its recorded
    # cost). main() checks this too; this backstop protects any other run_live_cells caller.
    uncached = config.models_missing_cache(sorted({m for _, m, _ in cells}))
    if uncached:
        raise ValueError(
            f"benchmark refuses to run uncached models (no cache-read discount): {uncached}. "
            "Remove them or give them cache_read_cost_per_1m in the model registry."
        )
    ctx = _LiveContext(
        hashes=hashes,
        versions=versions,
        pricing=config._pricing_dict(),
        digests=digests,
        arm_hash_map=arm_hash_map,
        work_dir=Path.cwd() / "benchmark" / "runner" / "artifacts",
        timeout=timeout,
    )
    # Persist each completed cell through merge_rows (history-log supersession +
    # key-upsert + atomic write preserved); None ⇒ old in-memory-only behaviour.
    checkpoint = partial(_checkpoint_row, path=results_path) if results_path is not None else None
    if workers <= 1:
        return _run_cells_serial(cells, ctx, max_cost, checkpoint)
    return _run_cells_parallel(cells, ctx, workers, max_cost, checkpoint)


def _checkpoint_row(row: dict, path: Path) -> None:
    """Persist one completed row through merge_rows (main-thread callers only)."""
    merge_rows([row], path)


def _run_and_merge(
    cells: list[tuple[str, str, str]],
    hashes: dict[str, str],
    versions: dict[str, str],
    digests: dict[str, str | None] | None,
    arm_hash_map: dict[str, dict[str, str]] | None,
    *,
    timeout: int,
    workers: int,
    max_cost: float | None,
    results_path: Path,
) -> int:
    """Run a cell list through the challenge-major executor and upsert results.csv.

    Shared by full-matrix ``main`` and the adaptive ``collect_phase``; cells are
    checkpointed as they complete and the ``models_missing_cache`` wall still applies.
    """
    new_rows = run_live_cells(
        cells,
        {},
        hashes,
        versions,
        timeout,
        False,
        digests,
        arm_hash_map,
        workers=workers,
        max_cost=max_cost,
        results_path=results_path,
    )
    return merge_rows(new_rows, results_path)


def collect_phase(
    tasks: list[str],
    models: list[str],
    cache: dict,
    hashes: dict[str, str],
    versions: dict[str, str],
    digests: dict[str, str | None] | None = None,
    *,
    live: bool = False,
    timeout: int = 600,
    workers: int = 1,
    max_cost: float | None = None,
    results_path: Path | None = None,
) -> CellStatus:
    """Classify one (tasks x models) block against the cache and, when live, run+merge it.

    The adaptive ``collect`` mode's phase primitive — reuses _arm_context/classify_cells/
    run_live_cells verbatim; simulated (live=False) only classifies (state in results.csv).
    """
    selected_arms, arm_hash_map = _arm_context(tasks, models)
    status = classify_cells(
        tasks, models, cache, hashes, versions, digests, selected_arms, arm_hash_map
    )
    if live and status.to_run and results_path is not None:
        _run_and_merge(
            status.to_run,
            hashes,
            versions,
            digests,
            arm_hash_map,
            timeout=timeout,
            workers=workers,
            max_cost=max_cost,
            results_path=results_path,
        )
    return status


def _row_key(row: dict) -> tuple[str, str, str]:
    """The results.csv cache key: (challenge_id, model, reasoning)."""
    # Literal string-equality on whatever `reasoning` the row carries (no alias
    # resolution here — that is a read-time concern, config.load_results). Two
    # distinct arm values for the same (challenge, model) are DISTINCT keys, so
    # they never collide or archive one another as history.
    reasoning = str(row.get("reasoning") or integrity.DEFAULT_REASONING)
    return (row["challenge_id"], row["model"], reasoning)


def _read_raw_rows(path: Path) -> dict[tuple[str, str, str], dict]:
    rows: dict[tuple[str, str, str], dict] = {}
    if not path.exists():
        return rows
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[_row_key(row)] = row
    return rows


def _write_raw_rows(rows: dict[tuple[str, str, str], dict], path: Path) -> None:
    """Atomically rewrite results.csv: write a sibling temp, then os.replace onto target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: _row_key(r))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(integrity.RESULTS_FIELDS))
        w.writeheader()
        for row in ordered:
            w.writerow({k: row.get(k, "") for k in integrity.RESULTS_FIELDS})
    os.replace(tmp, path)


def _history_path(results_path: Path) -> Path:
    """Append-only supersession log, gitignored under routing/artifacts/ (beside
    results.csv — NOT benchmark/artifacts/, which is not gitignored).
    """
    return results_path.parent / "artifacts" / "results_history.csv"


def _row_differs(old: dict, new: dict) -> bool:
    """True iff any cache column differs between a stored row and its replacement."""
    return any(str(old.get(k, "")) != str(new.get(k, "")) for k in integrity.RESULTS_FIELDS)


def _append_history(rows: list[dict], path: Path) -> None:
    """Append superseded rows (append-only) with a supersession timestamp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = _now_iso()
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(HISTORY_FIELDS))
        if new_file:
            w.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in integrity.RESULTS_FIELDS}
            out["superseded_at"] = ts
            w.writerow(out)


def merge_rows(new_rows: list[dict], path: Path, history_path: Path | None = None) -> int:
    """Upsert cells into results.csv (keyed by challenge×model×reasoning-arm)."""
    # A replaced row (same key, changed content) is moved to the append-only
    # history log — nothing is discarded. results.csv keeps only current rows.
    # Two arms of the same (challenge, model) are DISTINCT keys: a
    # new arm never collides with, or archives, a sibling arm's row.
    existing = _read_raw_rows(path)
    superseded: list[dict] = []
    for row in new_rows:
        norm = {k: row.get(k, "") for k in integrity.RESULTS_FIELDS}
        key = _row_key(norm)
        if key in existing and _row_differs(existing[key], norm):
            superseded.append(existing[key])
        existing[key] = norm
    if superseded:
        _append_history(superseded, history_path or _history_path(path))
    _write_raw_rows(existing, path)
    return len(new_rows)


def _report_coverage(matrix: dict, tasks: list[str]) -> None:
    print("\nStrategy cache coverage (embedding-free strategies):")
    for strategy in (Oracle(), AlwaysCheap(), AlwaysFrontier()):
        cov = coverage.cell_coverage(strategy, matrix, tasks)
        flag = "OK" if cov.complete else f"MISSING {len(cov.missing)} cell(s)"
        print(f"  {cov.strategy:16} needs {len(cov.needed):>3} cells  [{flag}]")


def refresh_summary(matrix: dict, tasks: list[str]) -> None:
    """Write the per-strategy summary derived from the results.csv cache to the
    gitignored reports/ dir. It is a regenerable artifact, never committed — the
    sole committed source of truth is results.csv.
    """
    from benchmark.routing import run_eval, summary

    bm = config.benchmark_params()
    strategies = run_eval.get_strategies()
    rows = summary.compute_strategy_rows(
        matrix,
        tasks,
        strategies,
        gamma=config.gamma(),
        bootstrap=bm.get("bootstrap_iterations", 1000),
        seed=bm.get("seed", 42),
    )
    out = Path(__file__).resolve().parent.parent / "routing" / "reports" / "strategy_summary.csv"
    summary.write_summary_csv(rows, out)
    print(f"Refreshed strategy summary -> {out}")


def regenerate_plots() -> None:
    """Regenerate the reports/ plots (gitignored, regenerable from results.csv) via
    report.py. Path is module-relative so it works from any cwd.
    """
    report_py = Path(__file__).resolve().parent.parent / "routing" / "report.py"
    cmd = [sys.executable, str(report_py), "--matrix", str(config.challenges_path())]
    print(f"Regenerating plots: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def _print_status(status: CellStatus, n_tasks: int, n_models: int) -> None:
    total = n_tasks * n_models
    print(
        f"Cells: {total}+ base ({n_tasks} challenges x {n_models} models; "
        f"reasoning-arm sampling adds non-default-arm cells)  "
        f"present={status.present}  missing={len(status.missing)}  stale={len(status.stale)}"
    )
    for label, cells in (("missing", status.missing), ("stale", status.stale)):
        if cells:
            sample = ", ".join(f"{c}:{m}:{a}" for c, m, a in cells[:3])
            print(f"  {label}: {len(cells)} — e.g. {sample}{' ...' if len(cells) > 3 else ''}")


def _arm_context(
    tasks: list[str], models: list[str]
) -> tuple[dict[tuple[str, str], list[str]], dict[str, dict[str, str]]]:
    """Selected arms per (challenge, model) + each model's arm-hash anchors."""
    resolved = config.resolved_models()
    arm_hash_map = integrity.arm_hashes(resolved)
    if not config.arm_sampling_enabled():
        # Gate OFF (disabled): stay on the single-default-arm-per-cell path
        # to reproduce default-arm-only behavior. The live executor now sends
        # distinct requests per arm (infer._scaffold_model_kwargs), so enabling
        # the gate generates and bills separate cells per arm.
        # arm_hash_map is still built (harmless, no extra cells result from it).
        return _default_selected_arms(tasks, models), arm_hash_map
    return _sampled_selected_arms(tasks, models, resolved), arm_hash_map


def _sampled_selected_arms(
    tasks: list[str], models: list[str], resolved: dict
) -> dict[tuple[str, str], list[str]]:
    """Multi-arm p(arm|model) sweep — gated behind ``arm_sampling.enabled``.

    Models in ``default_only_models`` are pinned to their default arm (cost control on
    the expensive tiers) even while the sweep runs for everyone else.
    """
    from benchmark.runner import sampling

    weights = config.arm_sampling_weights()
    default_only = config.arm_sampling_default_only_models()
    selected: dict[tuple[str, str], list[str]] = {}
    for model in models:
        bracket = resolved[model].reasoning if model in resolved else None
        for cid in tasks:
            if bracket is None:
                selected[(cid, model)] = [integrity.DEFAULT_REASONING]
            elif model in default_only:
                selected[(cid, model)] = [bracket.default_arm]
            else:
                selected[(cid, model)] = sampling.select_arms(cid, model, bracket, weights)
    return selected


def _add_args(ap: argparse.ArgumentParser, config_path: str) -> None:
    ap.add_argument(
        "--strategy",
        choices=("cost_optimal", "full"),
        default="cost_optimal",
        help="cost_optimal (default) = adaptive cheap-first collection (Phase A/B/C); "
        "full = exhaustive every-model x every-challenge matrix.",
    )
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument(
        "--live", action="store_true", help="Run uncached cells for real (needs Docker + keys)"
    )
    ap.add_argument(
        "--check-images",
        action="store_true",
        help="Resolve image digests to detect env drift (registry queries; implied by --live)",
    )
    ap.add_argument("--timeout", type=int, default=600, help="Per-cell timeout (live mode)")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent live cells (I/O-bound: Docker + LLM). Each worker runs a "
        "SWE-bench container — raise it with an eye on host memory. 1 = serial.",
    )
    ap.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Abort remaining cells once cumulative real_cost (USD) crosses this. "
        "Completed cells are kept; recommended for any paid run.",
    )
    ap.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip writing the regenerable reports/strategy_summary.csv",
    )
    ap.add_argument("--no-plots", action="store_true", help="Skip regenerating plots")
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose output")


def _prompt_confirm(prompt: str) -> str | None:
    """Read one confirmation line from stdin; None if non-interactive (not a TTY / EOF)."""
    if not sys.stdin.isatty():
        return None
    try:
        return input(prompt)
    except EOFError:
        return None


def _confirm_uncapped_live() -> bool:
    """Interactive gate for `full --live` with no --max-cost; abort unless the user types Y/y."""
    answer = _prompt_confirm(
        "FULL matrix --live with NO --max-cost cap can spend unbounded real money. Proceed? [y/N] "
    )
    if answer is None:
        print("  aborted: non-interactive stdin, refusing uncapped live spend.", file=sys.stderr)
        return False
    choice = answer.strip().lower()
    if choice == "y":
        return True
    if choice == "n":
        print("  aborted by user.", file=sys.stderr)
        return False
    print(f"  aborted: invalid input {answer!r} (expected y or n).", file=sys.stderr)
    return False


def _dispatch(args: argparse.Namespace) -> int:
    """Route parsed args to the cost_optimal (adaptive) or full (exhaustive) flow."""
    if args.strategy == "cost_optimal":
        from benchmark.runner.collect import run_collect

        return run_collect(
            args.config,
            live=args.live,
            timeout=args.timeout,
            workers=args.workers,
            max_cost=args.max_cost,
        )
    if args.live and args.max_cost is None and not _confirm_uncapped_live():
        return 3
    return _run_full(args)


def main(config_path: str = "benchmark/config.yaml") -> int:
    # Self-sufficient credentials: load a local .env (gitignored) if present so
    # `--live` works from a bare shunt checkout; real env vars still take precedence.
    load_dotenv_file()
    config.load(config_path)
    ap = argparse.ArgumentParser(
        description="Shunt benchmark runner (adaptive cost_optimal default; simulated by default)."
    )
    _add_args(ap, config_path)
    args = ap.parse_args()
    if args.config != config_path:
        config.load(args.config)
    return _dispatch(args)


def _run_full(args: argparse.Namespace) -> int:
    """Exhaustive matrix: classify every enabled model x sampled challenge, run+merge when live."""
    matrix = config.load_matrix(config.challenges_path())
    hashes = integrity.all_hashes()
    versions = integrity.model_versions()
    models = config.enabled_models()
    cache = config.load_results()
    tasks = config.sample_tasks(
        sorted(hashes.keys()), seed=config.benchmark_params().get("seed", 42)
    )
    # Resolve digests only when asked (or for a live run) — each is a registry query,
    # so the default simulated pass stays offline/fast. None ⇒ image axis skipped, never stale.
    digests = (
        image_version.resolve_spec_digests(swebench_specs.spec_image_refs(tasks))
        if (args.check_images or args.live)
        else None
    )

    selected_arms, arm_hash_map = _arm_context(tasks, models)
    status = classify_cells(
        tasks, models, cache, hashes, versions, digests, selected_arms, arm_hash_map
    )
    _print_status(status, len(tasks), len(models))

    live = args.live and _has_keys()
    if args.live and not _has_keys():
        print("  --live requested but no API keys in env — staying simulated (no fabrication).")

    # Caching gate: refuse a live run if any enabled model lacks a cache-read discount.
    # Uncached models are a budget bomb in agentic loops (see config.model_has_cache).
    if live:
        uncached = config.models_missing_cache(models)
        if uncached:
            print(
                f"  REFUSING --live: enabled models without caching {uncached} would burn "
                "uncached budget (full-context resend every turn). Disable them in "
                "config.yaml or add cache_read_cost_per_1m to the model registry.",
                file=sys.stderr,
            )
            return 2

    if status.to_run and live:
        # Cells are checkpointed as they complete; _run_and_merge returns the final
        # idempotent merge count for the summary line below.
        n = _run_and_merge(
            status.to_run,
            hashes,
            versions,
            digests,
            arm_hash_map,
            timeout=args.timeout,
            workers=args.workers,
            max_cost=args.max_cost,
            results_path=config.results_csv_path(),
        )
        print(f"  live: wrote {n} cell(s) to {config.results_csv_path()}")
        matrix = config.load_matrix(config.challenges_path())
    elif status.to_run:
        print(f"  simulated: would run {len(status.to_run)} cell(s); leaving them uncached.")

    _report_coverage(matrix, tasks)

    if not args.no_summary:
        refresh_summary(matrix, tasks)
    if not args.no_plots:
        regenerate_plots()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
