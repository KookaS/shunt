"""SWE-bench Multimodal instance specs — the multimodal sibling of ``swebench_specs``.

Dispatched when the configured manifest declares ``source: swebench_multimodal``; the
``dev`` split is the only runnable one (``test`` labels are leaderboard-hidden).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from benchmark import config
from benchmark.runner import swebench_specs
from benchmark.runner.swebench_specs import SwebenchSpec, _as_list

SOURCE: Final[str] = "swebench_multimodal"
DATASET_NAME: Final[str] = "princeton-nlp/SWE-bench_Multimodal"
DATASET_SPLIT: Final[str] = "dev"
# HF dataset commit the specs were materialised from — pins the provenance of every
# field (repo/base_commit/version/F2P/P2P), mirroring swebench_specs.DATASET_REVISION.
DATASET_REVISION: Final[str] = "aa2db68940196b6b59ae3f577faa0c25157bdd50"
DEFAULT_NAMESPACE: Final[str] = swebench_specs.DEFAULT_NAMESPACE
DEFAULT_ARCH: Final[str] = swebench_specs.DEFAULT_ARCH

# Multimodal rows carry no difficulty label; specs keep this deterministic placeholder.
_DIFFICULTY_STRATUM: Final[str] = "unknown"

# This source's OWN manifest, relative to the benchmark package dir (the same anchor
# ``config.challenges_path()`` uses). Never ``config.challenges_path()``: that resolves to
# whichever manifest is CONFIGURED — the 500-row Verified index under the shipped
# benchmark.yaml — so defaulting to it would overwrite another source's manifest.
MANIFEST_REL: Final[str] = "routing/data/challenges_multimodal.json"


def manifest_path() -> Path:
    """Default output path for this source's manifest."""
    return Path(config.__file__).resolve().parent / MANIFEST_REL


def spec_dir() -> Path:
    """Directory holding the per-instance spec JSON files."""
    return config.challenge_dir(SOURCE)


def spec_path(instance_id: str) -> Path:
    """Path to one instance's spec JSON."""
    return spec_dir() / f"{instance_id}.json"


def spec_from_dict(row: dict[str, object]) -> SwebenchSpec:
    """Build a spec from an on-disk spec JSON (already-normalised fields)."""
    return SwebenchSpec(
        instance_id=str(row["instance_id"]),
        repo=str(row["repo"]),
        base_commit=str(row["base_commit"]),
        version=str(row["version"]),
        difficulty_stratum=str(row["difficulty_stratum"]),
        fail_to_pass=_as_list(row["FAIL_TO_PASS"]),
        pass_to_pass=_as_list(row["PASS_TO_PASS"]),
        image_ref=str(row["image_ref"]),
        dataset_revision=str(row.get("dataset_revision", "")),
        problem_statement=str(row.get("problem_statement", "")),
    )


def spec_from_dataset_row(row: dict[str, object]) -> SwebenchSpec:
    """Build a spec from a raw HF Multimodal row (F2P/P2P are JSON strings)."""
    instance_id = str(row["instance_id"])
    return SwebenchSpec(
        instance_id=instance_id,
        repo=str(row["repo"]),
        base_commit=str(row["base_commit"]),
        version=str(row["version"]),
        difficulty_stratum=_DIFFICULTY_STRATUM,
        fail_to_pass=_as_list(row["FAIL_TO_PASS"]),
        pass_to_pass=_as_list(row["PASS_TO_PASS"]),
        image_ref=swebench_specs.image_ref(
            instance_id, namespace=DEFAULT_NAMESPACE, arch=DEFAULT_ARCH
        ),
        dataset_revision=DATASET_REVISION,
        problem_statement=str(row.get("problem_statement", "")),
    )


def load_spec(instance_id: str) -> SwebenchSpec | None:
    """Load a single instance spec by id; None if the file is absent."""
    path = spec_path(instance_id)
    if not path.exists():
        return None
    return spec_from_dict(json.loads(path.read_text()))


def spec_image_refs(instance_ids: list[str]) -> dict[str, str]:
    """Map each instance id to its image ref (from the stored spec, else derived)."""
    refs: dict[str, str] = {}
    for iid in instance_ids:
        spec = load_spec(iid)
        refs[iid] = spec.image_ref if spec else swebench_specs.image_ref(iid)
    return refs


def all_specs() -> list[SwebenchSpec]:
    """Every materialised spec, ordered by instance id."""
    directory = spec_dir()
    if not directory.exists():
        return []
    return [spec_from_dict(json.loads(p.read_text())) for p in sorted(directory.glob("*.json"))]


def write_spec(spec: SwebenchSpec) -> Path:
    """Persist one spec to ``challenges/swebench_multimodal/<instance_id>.json``."""
    directory = spec_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = spec_path(spec.instance_id)
    path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=False) + "\n")
    return path


def materialize(instance_ids: list[str] | None = None) -> list[Path]:
    """Pull every (or the named) dev-split row at the pinned revision; write their specs."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT, revision=DATASET_REVISION)
    if instance_ids is not None:
        wanted = set(instance_ids)
        rows = {str(r["instance_id"]): r for r in ds if str(r["instance_id"]) in wanted}
        missing = wanted - rows.keys()
        if missing:
            raise KeyError(f"instance ids not found in {DATASET_NAME}: {sorted(missing)}")
        ordered = [rows[i] for i in instance_ids]
    else:
        ordered = list(ds)
    return [write_spec(spec_from_dataset_row(dict(row))) for row in ordered]


def _description(spec: SwebenchSpec) -> str:
    """``<repo>@<base_commit[:12]> — resolve <fail_to_pass[0]>`` (em-dash, U+2014)."""
    target = spec.fail_to_pass[0] if spec.fail_to_pass else ""
    return f"{spec.repo}@{spec.base_commit[:12]} — resolve {target}"


def build_manifest(specs: list[SwebenchSpec], updated: str) -> dict[str, object]:
    """Assemble the multimodal manifest; challenges list and tasks keys sorted by id.

    Same shape as ``build_challenges.build_manifest`` (the runner reads both through
    ``config.load_challenges`` / ``config.load_matrix`` interchangeably).
    """
    ordered = sorted(specs, key=lambda s: s.instance_id)
    return {
        "version": "3.0",
        "updated": updated,
        "source": SOURCE,
        "source_dataset": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "count": len(ordered),
        "challenges": [
            {
                "id": s.instance_id,
                "source": SOURCE,
                "language": "javascript",
                "difficulty": s.difficulty_stratum,
            }
            for s in ordered
        ],
        "tasks": {
            s.instance_id: {
                "description": _description(s),
                "problem_statement": s.problem_statement,
                "language": "javascript",
                "tags": ["swebench-multimodal", s.difficulty_stratum],
                "repo": s.repo,
                "base_commit": s.base_commit,
                "difficulty_stratum": s.difficulty_stratum,
                "source_dataset": "SWE-bench_Multimodal",
                "spec": f"challenges/{SOURCE}/{s.instance_id}.json",
            }
            for s in ordered
        },
    }


def write_manifest(manifest: dict[str, object], out: Path | None = None) -> Path:
    """Atomically serialise the manifest to ``manifest_path()`` (or an explicit ``out``)."""
    import os

    out = manifest_path() if out is None else out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    os.replace(tmp, out)
    return out


def _main() -> int:
    import argparse
    import datetime

    ap = argparse.ArgumentParser(
        description="Materialise SWE-bench Multimodal (dev) specs + rebuild the manifest."
    )
    ap.add_argument("--updated", default=datetime.date.today().isoformat(), help="updated date")
    ap.add_argument("--manifest-out", default=None, help="override the default manifest path")
    args = ap.parse_args()

    paths = materialize()
    manifest = build_manifest(all_specs(), args.updated)
    out = write_manifest(manifest, Path(args.manifest_out) if args.manifest_out else None)
    print(f"Wrote {len(paths)} specs -> {spec_dir()}")
    print(f"Wrote manifest ({manifest['count']} challenges) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
