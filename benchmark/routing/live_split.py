"""Held-out task split for the live evaluation — a pure function of (task_id, salt, fraction).

Membership is decided by hash-threshold over the challenges.json universe, so no benchmark
hyperparameter (sample_size, seed, arm weights, run order, model pool, coverage) can move it.
"""

# The split is drawn over the UNIVERSE (every id in challenges.json), never over the tasks that
# happen to be collected in results.csv: a split conditioned on coverage would silently re-draw
# itself every time a cell is added, and a "held-out" task would stop being held out.
#
# `LIVE_SPLIT_SALT` is the FIFTH salt namespace in this benchmark, and it must alias none of the
# four already in use (calibration.DEFAULT_SALT, sampling.ORDER_SALT, sampling.ARM_SALT_PREFIX,
# sampling.AUDIT_SALT) — see the comment at benchmark/runner/sampling.py:26-29. A shared salt
# would correlate live-eval membership with the calibration holdout, the run order or the audit
# stratum, so the live result would measure the sampler rather than the router.
#
# The draw reuses `calibration.holdout_score` rather than a private RNG: that primitive is
# already nested in `fraction` (raising it only ADDS members) and independent per id.

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Final

from benchmark.routing.scripts.knn_nulls import repo_of
from benchmark.runner.calibration import in_calibration_holdout

LIVE_SPLIT_SALT: Final[str] = "live-split-v1"
DEFAULT_FRACTION: Final[float] = 0.2

CHALLENGES_PATH: Final[Path] = Path(__file__).resolve().parent / "data" / "challenges.json"
MANIFEST_PATH: Final[Path] = Path(__file__).resolve().parent / "data" / "live_split_manifest.json"


def _challenges() -> dict[str, object]:
    """Parsed challenges.json — the universe and the revision it is pinned to."""
    with CHALLENGES_PATH.open(encoding="utf-8") as handle:
        payload: dict[str, object] = json.load(handle)
    return payload


def universe() -> list[str]:
    """Every task id in challenges.json, in the manifest's own order."""
    challenges = _challenges().get("challenges")
    if not isinstance(challenges, list) or not challenges:
        raise ValueError(f"no challenges found in {CHALLENGES_PATH}")
    return [str(c["id"]) for c in challenges]


def dataset_revision() -> str:
    """The upstream revision the ids are immutable relative to."""
    revision = _challenges().get("dataset_revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"no dataset_revision in {CHALLENGES_PATH}")
    return revision


def is_holdout(
    task_id: str, fraction: float = DEFAULT_FRACTION, salt: str = LIVE_SPLIT_SALT
) -> bool:
    """True iff this task is withheld from the live evaluation's training/tuning side."""
    return in_calibration_holdout(task_id, fraction, salt)


def holdout_tasks(fraction: float = DEFAULT_FRACTION, salt: str = LIVE_SPLIT_SALT) -> list[str]:
    """Sorted holdout ids drawn over the whole challenges.json universe."""
    return sorted(t for t in universe() if is_holdout(t, fraction, salt))


def tasks_digest(task_ids: list[str]) -> str:
    """sha256 over the newline-joined sorted ids — the manifest's staleness anchor."""
    return hashlib.sha256("\n".join(sorted(task_ids)).encode()).hexdigest()


def split_manifest(
    fraction: float = DEFAULT_FRACTION, salt: str = LIVE_SPLIT_SALT
) -> dict[str, object]:
    """The auditable record of one resolved split: inputs, counts, ids and digest."""
    ids = universe()
    holdout = sorted(t for t in ids if is_holdout(t, fraction, salt))
    return {
        "salt": salt,
        "fraction": fraction,
        "dataset_revision": dataset_revision(),
        "universe_size": len(ids),
        "holdout_count": len(holdout),
        "per_repo_counts": dict(sorted(Counter(repo_of(t) for t in holdout).items())),
        "tasks_digest": tasks_digest(holdout),
        "tasks": holdout,
    }


def write_manifest(
    fraction: float = DEFAULT_FRACTION, salt: str = LIVE_SPLIT_SALT
) -> dict[str, object]:
    """Regenerate the committed manifest; the file is the reviewable artefact, not the source."""
    manifest = split_manifest(fraction, salt)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    """Rewrite live_split_manifest.json and report the resolved counts."""
    ap = argparse.ArgumentParser(description="Regenerate the live-evaluation split manifest.")
    ap.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    ap.add_argument("--salt", default=LIVE_SPLIT_SALT)
    args = ap.parse_args()
    manifest = write_manifest(args.fraction, args.salt)
    print(
        f"{MANIFEST_PATH}: {manifest['holdout_count']}/{manifest['universe_size']} tasks "
        f"@ fraction={manifest['fraction']} salt={manifest['salt']} "
        f"digest={manifest['tasks_digest']}"
    )


if __name__ == "__main__":
    main()
