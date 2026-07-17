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
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    """Classification of every (challenge, model) cell against the cache."""

    missing: list[tuple[str, str]] = field(default_factory=list)
    stale: list[tuple[str, str]] = field(default_factory=list)
    present: int = 0

    @property
    def to_run(self) -> list[tuple[str, str]]:
        """Cells needing (re)computation: missing plus stale."""
        return self.missing + self.stale


def _has_keys() -> bool:
    return any(os.environ.get(k) for k in _KEY_ENV)


def classify_cells(
    tasks: list[str],
    models: list[str],
    cache: dict,
    hashes: dict[str, str],
    versions: dict[str, str],
    digests: dict[str, str | None] | None = None,
) -> CellStatus:
    """Split cells into missing / stale / present by anchor string-equality.

    Stale iff spec hash, image digest, or model version drifted; missing iff no
    row. ``digests`` None-valued (unresolved) never marks a cell stale.
    """
    status = CellStatus()
    for cid in tasks:
        for model in models:
            cell = cache.get(cid, {}).get(model)
            if cell is None:
                status.missing.append((cid, model))
            elif _is_stale(cell, cid, model, hashes, versions, digests):
                status.stale.append((cid, model))
            else:
                status.present += 1
    return status


def _is_stale(
    cell: dict,
    cid: str,
    model: str,
    hashes: dict[str, str],
    versions: dict[str, str],
    digests: dict[str, str | None] | None,
) -> bool:
    """A cell is stale iff its stored spec hash, image digest, or model version drifted."""
    if cell.get("version_hash") != hashes.get(cid):
        return True
    if cell.get("model_version") != versions.get(model):
        return True
    return _image_stale(cell, cid, digests)


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
        "reasoning": outcome.get("reasoning", integrity.DEFAULT_REASONING),
        "pass": outcome.get("pass", False),
        "cost": cost,
        "in_tok": in_tok,
        "out_tok": out_tok,
        "calls": int(outcome.get("calls", 0)),
        "version_hash": hashes.get(cid, ""),
        "model_version": versions.get(model, integrity.UNKNOWN_VERSION),
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


def run_live_cells(
    cells: list[tuple[str, str]],
    matrix: dict,  # noqa: ARG001 (kept for signature stability; spec is loaded per-cell)
    hashes: dict[str, str],
    versions: dict[str, str],
    timeout: int,
    verbose: bool,
    digests: dict[str, str | None] | None = None,
) -> list[dict]:
    """Delegate each cell to the real SWE-bench harness executor (one
    (instance, model) run scored by the harness; version_hash from the spec
    content hash). Cells with no materialised spec or no keys are skipped.
    """
    from benchmark.routing import integrity
    from benchmark.runner import infer

    # Structural wall (defense-in-depth): never spend uncached budget. An uncached
    # model resends the full context every agent turn at full input price — a cell that
    # churns then fails still bills for all of it (qwen3-max billed 50x its recorded
    # cost). main() checks this too; this backstop protects any other run_live_cells caller.
    uncached = config.models_missing_cache(sorted({m for _, m in cells}))
    if uncached:
        raise ValueError(
            f"benchmark refuses to run uncached models (no cache-read discount): {uncached}. "
            "Remove them or give them cache_read_cost_per_1m in the model registry."
        )

    work_dir = Path.cwd() / "benchmark" / "runner" / "artifacts"
    pricing = config._pricing_dict()
    rows: list[dict] = []
    for cid, model in cells:
        try:
            outcome = infer.run_live_cell(
                cid, model, work_dir=work_dir, run_id=f"live-{cid}-{model}", timeout=timeout
            )
        except Exception as exc:  # noqa: BLE001 (per-cell isolation: skip one, never fabricate)
            # ANY failure (missing spec/keys, harness infra crash, import, network)
            # skips just this cell and leaves it MISSING to recompute — the batch
            # continues and nothing is fabricated into the paid cache.
            print(f"  skip {cid}:{model} — {exc}", file=sys.stderr)
            continue
        cell_hashes = {**hashes, cid: integrity.swebench_spec_hash(cid) or hashes.get(cid, "")}
        rows.append(_build_row(cid, model, outcome, cell_hashes, versions, pricing, digests))
    return rows


def _read_raw_rows(path: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return rows
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[(row["challenge_id"], row["model"])] = row
    return rows


def _write_raw_rows(rows: dict[tuple[str, str], dict], path: Path) -> None:
    """Atomically rewrite results.csv: write a sibling temp, then os.replace onto target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: (r["challenge_id"], r["model"]))
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
    """Upsert cells into results.csv (keyed by challenge×model); archive superseded rows.

    A replaced row (same key, changed content) is moved to the append-only
    history log — nothing is discarded. results.csv keeps only current rows.
    """
    existing = _read_raw_rows(path)
    superseded: list[dict] = []
    for row in new_rows:
        key = (row["challenge_id"], row["model"])
        norm = {k: row.get(k, "") for k in integrity.RESULTS_FIELDS}
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
        f"Cells: {total} total ({n_tasks} challenges x {n_models} models)  "
        f"present={status.present}  missing={len(status.missing)}  stale={len(status.stale)}"
    )
    for label, cells in (("missing", status.missing), ("stale", status.stale)):
        if cells:
            sample = ", ".join(f"{c}:{m}" for c, m in cells[:3])
            print(f"  {label}: {len(cells)} — e.g. {sample}{' ...' if len(cells) > 3 else ''}")


def _add_args(ap: argparse.ArgumentParser, config_path: str) -> None:
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
        "--no-summary",
        action="store_true",
        help="Skip writing the regenerable reports/strategy_summary.csv",
    )
    ap.add_argument("--no-plots", action="store_true", help="Skip regenerating plots")
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose output")


def main(config_path: str = "benchmark/config.yaml") -> int:
    # Self-sufficient credentials: load a local .env (gitignored) if present so
    # `--live` works from a bare shunt checkout; real env vars still take precedence.
    load_dotenv_file()
    config.load(config_path)
    ap = argparse.ArgumentParser(description="Shunt benchmark caching loop (simulated by default).")
    _add_args(ap, config_path)
    args = ap.parse_args()
    if args.config != config_path:
        config.load(args.config)

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

    status = classify_cells(tasks, models, cache, hashes, versions, digests)
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
        new_rows = run_live_cells(
            status.to_run, matrix, hashes, versions, args.timeout, args.verbose, digests
        )
        n = merge_rows(new_rows, config.results_csv_path())
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
