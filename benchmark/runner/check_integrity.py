#!/usr/bin/env python3
"""Benchmark integrity gate: detect changed/removed challenges and stale model versions."""

from __future__ import annotations

import argparse

from benchmark import config
from benchmark.routing import integrity, summary
from benchmark.routing.strategies.fixed import AlwaysCheap, AlwaysFrontier
from benchmark.routing.strategies.oracle import Oracle
from benchmark.runner import image_version, swebench_specs


def _stored_hashes(cache: dict) -> dict[str, str]:
    """Map each cached challenge to its stored version_hash (first model row)."""
    out: dict[str, str] = {}
    for cid, models in cache.items():
        for cell in models.values():
            out[cid] = str(cell.get("version_hash", ""))
            break
    return out


def check_hashes(cache: dict, hashes: dict[str, str]) -> tuple[list[str], list[str]]:
    """Return (removed, changed) challenge ids by comparing stored vs current hashes."""
    stored = _stored_hashes(cache)
    removed = [cid for cid in stored if cid not in hashes]
    changed = [
        cid for cid in stored if cid in hashes and stored[cid] and stored[cid] != hashes[cid]
    ]
    return sorted(removed), sorted(changed)


def check_model_versions(cache: dict, versions: dict[str, str]) -> list[tuple[str, str, str]]:
    """Return (model, stored_version, current_version) for every stale model row."""
    stale: set[tuple[str, str, str]] = set()
    for models in cache.values():
        for model, cell in models.items():
            stored = str(cell.get("model_version", ""))
            current = versions.get(model, integrity.UNKNOWN_VERSION)
            if stored and current and stored != current:
                stale.add((model, stored, current))
    return sorted(stale)


def check_image_digests(
    cache: dict, digests: dict[str, str | None]
) -> list[tuple[str, str, str, str]]:
    """Return (cid, model, stored, current) for every cell whose image digest drifted.

    Only reports when a digest RESOLVES and the stored anchor is non-empty —
    an unresolved digest (offline) never counts as drift.
    """
    drift: list[tuple[str, str, str, str]] = []
    for cid, models in cache.items():
        current = digests.get(cid)
        if current is None:
            continue
        for model, cell in models.items():
            stored = str(cell.get("image_digest", ""))
            if stored and stored != current:
                drift.append((cid, model, stored, current))
    return sorted(drift)


def check_derived(matrix: dict, tasks: list[str]) -> list[str]:
    """Recompute strategy metrics twice and flag any difference in deterministic columns."""
    bm = config.benchmark_params()

    def _compute() -> dict[str, dict]:
        rows = summary.compute_strategy_rows(
            matrix,
            tasks,
            [Oracle(), AlwaysCheap(), AlwaysFrontier()],
            gamma=config.gamma(),
            bootstrap=bm.get("bootstrap_iterations", 1000),
            seed=bm.get("seed", 42),
        )
        return {r["strategy"]: r for r in rows}

    first, second = _compute(), _compute()
    problems: list[str] = []
    for name, r1 in first.items():
        r2 = second.get(name)
        if r2 is None:
            problems.append(f"strategy '{name}' vanished on recompute")
            continue
        problems.extend(_diff_row(r1, r2))
    return problems


def _diff_row(first: dict, second: dict) -> list[str]:
    out: list[str] = []
    for col in summary.DETERMINISTIC_FIELDS:
        a = str(first.get(col, ""))
        b = str(second.get(col, ""))
        if a != b:
            out.append(f"{first['strategy']}.{col}: nondeterministic ({a!r} vs {b!r})")
    return out


def _report(
    removed: list[str],
    changed: list[str],
    stale: list,
    derived: list[str],
    image_drift: list[tuple[str, str, str, str]] | None = None,
) -> bool:
    ok = True
    if changed:
        ok = False
        print(f"CHANGED challenges ({len(changed)}): content hash differs from stored version_hash")
        for cid in changed[:10]:
            print(f"  - {cid}")
    if removed:
        ok = False
        print(f"REMOVED challenges ({len(removed)}): rows exist but challenge file is gone")
        for cid in removed[:10]:
            print(f"  - {cid}")
    if stale:
        ok = False
        print(f"STALE model versions ({len(stale)}):")
        for model, stored, current in stale:
            print(f"  - {model}: stored={stored} current={current}")
    if image_drift:
        ok = False
        print(f"IMAGE-DIGEST drift ({len(image_drift)}): resolved manifest differs from stored")
        for cid, model, stored, current in image_drift[:10]:
            print(f"  - {cid}:{model} stored={stored} current={current}")
    if derived:
        ok = False
        print(f"DERIVED nondeterminism ({len(derived)}): strategy summary differs across recompute")
        for msg in derived[:10]:
            print(f"  - {msg}")
    if ok:
        print("Integrity OK: hashes match, no removed challenges, model versions current.")
    return ok


def main(config_path: str = "benchmark/config.yaml") -> int:
    ap = argparse.ArgumentParser(description="Benchmark integrity + derived-artifact drift gate.")
    ap.add_argument("--config", default=config_path, help="Path to config YAML")
    ap.add_argument(
        "--check-derived",
        action="store_true",
        help="Also verify the per-strategy summary derives deterministically from results.csv",
    )
    ap.add_argument(
        "--check-images",
        action="store_true",
        help="Also resolve manifest digests and report image drift (needs docker + network)",
    )
    args = ap.parse_args()
    config.load(args.config)

    cache = config.load_results()
    # SWE-bench Verified is the sole challenge source: a cell's version_hash is the
    # content hash of its instance spec (base_commit + F2P/P2P + provenance).
    hashes = integrity.swebench_spec_hashes()
    versions = integrity.model_versions()

    removed, changed = check_hashes(cache, hashes)
    stale = check_model_versions(cache, versions)

    image_drift: list[tuple[str, str, str, str]] = []
    if args.check_images:
        refs = swebench_specs.spec_image_refs(sorted(cache.keys()))
        image_drift = check_image_digests(cache, image_version.resolve_spec_digests(refs))

    derived: list[str] = []
    if args.check_derived:
        matrix = config.load_matrix(config.challenges_path())
        # Derive tasks identically to run_matrix.refresh_summary (the writer). Tasks
        # come from the SWE-bench spec store — the sole challenge source.
        seed = config.benchmark_params().get("seed", 42)
        tasks = config.sample_tasks(sorted(hashes.keys()), seed=seed)
        derived = check_derived(matrix, tasks)

    return 0 if _report(removed, changed, stale, derived, image_drift) else 1


if __name__ == "__main__":
    raise SystemExit(main())
