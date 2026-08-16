"""Reproducible producer for the challenges manifest (routing/data/challenges.json)."""

from __future__ import annotations

# Materialises ALL SWE-bench Verified instance specs and rebuilds the challenges
# manifest from them:
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
from benchmark.runner.swebench_specs import SwebenchSpec

SOURCE = swebench_specs.SOURCE


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
    # A LABEL, not the task: 62% test node-id, 14% repo name, 12% a random commit prefix.
    # Routing PREFERS `problem_statement` and falls back here when it is absent — which is
    # every task in the committed manifest, so today this label is what the router actually
    # embeds (see strategies/routing_text).
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


def _task_entry(spec: SwebenchSpec) -> dict[str, object]:
    """The rich ``tasks`` entry for one instance."""
    stratum = spec.difficulty_stratum
    entry: dict[str, object] = {
        "description": _description(spec),
        # The text the AGENT is given (infer.py: ``instance["problem_statement"]``), carried so
        # that once this manifest is rebuilt the router embeds the same task it is routing. The
        # committed manifest predates the field and omits it entirely, so that alignment is the
        # intent of this line, not yet a property of the shipped data. Carried in the manifest
        # because routing/ must not import runner/ to read the spec files.
        "problem_statement": spec.problem_statement,
        "language": "python",
        "tags": ["swebench-verified", stratum],
        "repo": spec.repo,
        "base_commit": spec.base_commit,
        "difficulty_stratum": stratum,
        "source_dataset": "SWE-bench_Verified",
        "spec": f"challenges/{SOURCE}/{spec.instance_id}.json",
    }
    return entry


def build_manifest(specs: list[SwebenchSpec], updated: str) -> dict[str, object]:
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
        "tasks": {s.instance_id: _task_entry(s) for s in ordered},
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
    manifest = build_manifest(specs, args.updated)
    out = config.challenges_path()
    _write_manifest(manifest, out)
    print(f"Wrote {len(specs)} specs -> {swebench_specs.spec_dir()}")
    print(f"Wrote manifest ({manifest['count']} challenges) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
