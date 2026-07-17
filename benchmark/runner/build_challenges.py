"""Reproducible producer for the challenges manifest (routing/data/challenges.json)."""

from __future__ import annotations

# Materialises ALL SWE-bench Verified instance specs and rebuilds the challenges
# manifest from them, joined with the external difficulty prior:
#     python -m benchmark.runner.build_challenges            # all 500
#     python -m benchmark.runner.build_challenges --limit 5  # cheap dry run
# The HF dataset is pinned to ``swebench_specs.DATASET_REVISION`` for provenance;
# specs + manifest are sorted by instance id, so the content is reproducible apart
# from the ``updated`` date stamp (pin ``--updated`` for a byte-identical rebuild).
import argparse
import datetime
import json
import os
from pathlib import Path

from benchmark import config
from benchmark.runner import swebench_specs
from benchmark.runner.select_swebench import load_external_rates, routing_headroom
from benchmark.runner.swebench_specs import SwebenchSpec

SOURCE = swebench_specs.SOURCE
Rates = dict[str, tuple[float, float]]


def _dataset_rows() -> list[dict[str, object]]:
    """Every Verified row at the pinned revision (falls back to latest with a warning)."""
    from datasets import load_dataset

    try:
        ds = load_dataset(
            swebench_specs.DATASET_NAME,
            split=swebench_specs.DATASET_SPLIT,
            revision=swebench_specs.DATASET_REVISION,
        )
    except (ValueError, OSError, FileNotFoundError) as exc:
        print(
            f"warning: could not load revision {swebench_specs.DATASET_REVISION} "
            f"({exc}); falling back to latest"
        )
        ds = load_dataset(swebench_specs.DATASET_NAME, split=swebench_specs.DATASET_SPLIT)
    return [dict(row) for row in ds]


def _description(spec: SwebenchSpec) -> str:
    """``<repo>@<base_commit[:12]> — resolve <fail_to_pass[0]>`` (em-dash, U+2014)."""
    target = spec.fail_to_pass[0] if spec.fail_to_pass else ""
    return f"{spec.repo}@{spec.base_commit[:12]} — resolve {target}"


def _challenge_entry(spec: SwebenchSpec) -> dict[str, str]:
    """The lightweight index row for the ``challenges`` list."""
    return {
        "id": spec.instance_id,
        "source": SOURCE,
        "language": "python",
        "difficulty": spec.difficulty_stratum,
    }


def _task_entry(spec: SwebenchSpec, rates: Rates) -> dict[str, object]:
    """The rich ``tasks`` entry; p_solve/p_frontier/headroom only when rated."""
    stratum = spec.difficulty_stratum
    entry: dict[str, object] = {
        "description": _description(spec),
        "language": "python",
        "tags": ["swebench-verified", stratum],
        "repo": spec.repo,
        "base_commit": spec.base_commit,
        "difficulty_stratum": stratum,
        "source_dataset": "SWE-bench_Verified",
        "spec": f"challenges/{SOURCE}/{spec.instance_id}.json",
    }
    if spec.instance_id in rates:
        p_solve, p_frontier = rates[spec.instance_id]
        entry["p_solve"] = round(p_solve, 4)
        entry["p_frontier"] = round(p_frontier, 4)
        entry["routing_headroom"] = round(routing_headroom(p_solve, p_frontier), 4)
    return entry


def build_manifest(specs: list[SwebenchSpec], rates: Rates, updated: str) -> dict[str, object]:
    """Assemble the manifest dict; challenges list and tasks keys sorted by id."""
    ordered = sorted(specs, key=lambda s: s.instance_id)
    return {
        "version": "3.0",
        "updated": updated,
        "source": SOURCE,
        "source_dataset": swebench_specs.DATASET_NAME,
        "dataset_revision": swebench_specs.DATASET_REVISION,
        "count": len(ordered),
        "challenges": [_challenge_entry(s) for s in ordered],
        "tasks": {s.instance_id: _task_entry(s, rates) for s in ordered},
    }


def _specs_from_rows(rows: list[dict[str, object]], limit: int | None) -> list[SwebenchSpec]:
    """Build specs for every row (or the first ``limit`` ids, sorted by id)."""
    specs = [swebench_specs.spec_from_dataset_row(row) for row in rows]
    specs.sort(key=lambda s: s.instance_id)
    if limit is not None:
        specs = specs[:limit]
    return specs


def _write_manifest(manifest: dict[str, object], out: Path) -> None:
    """Atomically serialise ascii-escaped + trailing newline (temp then os.replace)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    os.replace(tmp, out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild challenges.json from SWE-bench Verified.")
    ap.add_argument("--updated", default=datetime.date.today().isoformat(), help="updated date")
    ap.add_argument("--limit", type=int, default=None, help="materialise only first N ids (test)")
    args = ap.parse_args()

    specs = _specs_from_rows(_dataset_rows(), args.limit)
    for spec in specs:
        swebench_specs.write_spec(spec)
    manifest = build_manifest(specs, load_external_rates(), args.updated)
    out = config.challenges_path()
    _write_manifest(manifest, out)
    print(f"Wrote {len(specs)} specs -> {swebench_specs.spec_dir()}")
    print(f"Wrote manifest ({manifest['count']} challenges) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
