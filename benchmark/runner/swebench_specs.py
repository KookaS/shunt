"""SWE-bench Verified instance specs — the minimal per-challenge setup needed to
run + identify an instance. Repo snapshots and patches are pulled on demand by
the harness (no vendored repos). See ``benchmark/README.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from benchmark import config

SOURCE = "swebench_verified"
DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DATASET_SPLIT = "test"
# HF dataset commit the specs were materialised from — pins the provenance of
# every field (repo/base_commit/version/F2P/P2P) so a spec is reproducible.
DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
DEFAULT_NAMESPACE = "swebench"
DEFAULT_ARCH = "x86_64"
DEFAULT_IMAGE_TAG = "latest"

# Verified human time-estimate label -> coarse difficulty stratum used for the
# stratified 30/50/20 selection. See runnable-code-eval-datasets research.
_DIFFICULTY_STRATUM: Final = {
    "<15 min fix": "easy",
    "15 min - 1 hour": "medium",
    "1-4 hours": "hard",
    ">4 hours": "hard",
}
# Canonical spec key order — the JSON on disk and the hashed content both use it.
_SPEC_KEYS = (
    "instance_id",
    "repo",
    "base_commit",
    "version",
    "difficulty_stratum",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "image_ref",
    "dataset_revision",
)


def difficulty_stratum(difficulty: str) -> str:
    """Map a Verified ``difficulty`` label to easy/medium/hard (default medium)."""
    return _DIFFICULTY_STRATUM.get(difficulty, "medium")


def image_ref(
    instance_id: str,
    namespace: str = DEFAULT_NAMESPACE,
    arch: str = DEFAULT_ARCH,
    tag: str = DEFAULT_IMAGE_TAG,
) -> str:
    """Prebuilt instance-image reference the harness pulls (mirrors swebench's key)."""
    key = f"sweb.eval.{arch}.{instance_id.lower()}:{tag}"
    return f"{namespace}/{key}".replace("__", "_1776_")


@dataclass(frozen=True)
class SwebenchSpec:
    """Minimal per-instance setup; content (repo/patch) is pulled by the harness."""

    instance_id: str
    repo: str
    base_commit: str
    version: str
    difficulty_stratum: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    image_ref: str
    dataset_revision: str

    def to_dict(self) -> dict[str, object]:
        """Serialise in canonical key order (F2P/P2P upper-cased to match SWE-bench)."""
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "version": self.version,
            "difficulty_stratum": self.difficulty_stratum,
            "FAIL_TO_PASS": list(self.fail_to_pass),
            "PASS_TO_PASS": list(self.pass_to_pass),
            "image_ref": self.image_ref,
            "dataset_revision": self.dataset_revision,
        }


def _as_list(value: object) -> list[str]:
    """Coerce a SWE-bench F2P/P2P field (JSON string or list) to a list of str."""
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise TypeError(f"expected list, got {type(value)!r}")
    return [str(v) for v in value]


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
    )


def spec_from_dataset_row(
    row: dict[str, object],
    namespace: str = DEFAULT_NAMESPACE,
    arch: str = DEFAULT_ARCH,
) -> SwebenchSpec:
    """Build a spec from a raw HF Verified dataset row (F2P/P2P are JSON strings)."""
    instance_id = str(row["instance_id"])
    return SwebenchSpec(
        instance_id=instance_id,
        repo=str(row["repo"]),
        base_commit=str(row["base_commit"]),
        version=str(row["version"]),
        difficulty_stratum=difficulty_stratum(str(row.get("difficulty", ""))),
        fail_to_pass=_as_list(row["FAIL_TO_PASS"]),
        pass_to_pass=_as_list(row["PASS_TO_PASS"]),
        image_ref=image_ref(instance_id, namespace=namespace, arch=arch),
        dataset_revision=DATASET_REVISION,
    )


def spec_dir() -> Path:
    """Directory holding the per-instance spec JSON files."""
    return config.challenge_dir(SOURCE)


def spec_path(instance_id: str) -> Path:
    """Path to one instance's spec JSON."""
    return spec_dir() / f"{instance_id}.json"


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
        refs[iid] = spec.image_ref if spec else image_ref(iid)
    return refs


def all_specs() -> list[SwebenchSpec]:
    """Every materialised spec, ordered by instance id."""
    directory = spec_dir()
    if not directory.exists():
        return []
    specs = [spec_from_dict(json.loads(p.read_text())) for p in sorted(directory.glob("*.json"))]
    return specs


def write_spec(spec: SwebenchSpec) -> Path:
    """Persist one spec to ``challenges/swebench_verified/<instance_id>.json``."""
    directory = spec_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = spec_path(spec.instance_id)
    path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=False) + "\n")
    return path


def materialize(
    instance_ids: list[str],
    namespace: str = DEFAULT_NAMESPACE,
    arch: str = DEFAULT_ARCH,
) -> list[Path]:
    """Pull the named Verified rows from HF and write their specs to disk."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    wanted = set(instance_ids)
    rows = {str(r["instance_id"]): r for r in ds if str(r["instance_id"]) in wanted}
    missing = wanted - rows.keys()
    if missing:
        raise KeyError(f"instance ids not found in {DATASET_NAME}: {sorted(missing)}")
    return [write_spec(spec_from_dataset_row(rows[iid], namespace, arch)) for iid in instance_ids]


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Materialise SWE-bench Verified instance specs.")
    ap.add_argument("instance_ids", nargs="+", help="Verified instance ids to materialise")
    ap.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="Prebuilt-image namespace")
    ap.add_argument("--arch", default=DEFAULT_ARCH, help="Image arch (x86_64 / arm64)")
    args = ap.parse_args()
    paths = materialize(args.instance_ids, namespace=args.namespace, arch=args.arch)
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
