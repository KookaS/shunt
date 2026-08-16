"""Live census of the committed escalation corpus — the ONE place its size is derived.

The corpus grows without touching code, so any committed count of it rots. Consumers read
`census()`, never a literal; docs are pinned against it by `tests/escalation/test_corpus_census.py`.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

# The committed escalation trajectories live under the data plane. The directory is
# the census root; `manifest.json` beside it is a derived ledger, not a source.
LIVE_DIR = Path(__file__).resolve().parent / "data" / "live"


@dataclass(frozen=True)
class CorpusCensus:
    """The live trajectory and step counts of a corpus directory."""

    trajectories: int
    steps: int


def census(live_dir: str | Path = LIVE_DIR) -> CorpusCensus:
    """Count trajectories and total steps under ``live_dir`` (default: the committed corpus).

    One trajectory per ``*.jsonl`` file; steps are the StepView records after the header.
    ``live_dir`` is injectable so a test can plant extra trajectories.
    """
    root = Path(live_dir)
    trajectories = 0
    steps = 0
    for path in root.glob("*.jsonl"):
        if not path.is_file():
            continue
        records = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            # A zero-byte / whitespace-only file is not a trajectory: it contributes 0.
            continue
        # The header is the first record; the rest are StepViews. A corrupt first line must
        # not crash the census — a defective committed file is a defect to fix, not a count
        # to blow up on — so it is skipped with a warning and contributes 0 trajectories.
        try:
            header = json.loads(records[0])
        except json.JSONDecodeError:
            warnings.warn(
                f"census: skipping {path}: first line is not valid JSON — contributes 0",
                stacklevel=2,
            )
            continue
        if not isinstance(header, dict):
            warnings.warn(
                f"census: skipping {path}: header is not a JSON object — contributes 0",
                stacklevel=2,
            )
            continue
        trajectories += 1
        # `n_steps` in the header is authoritative where present, but counting the records
        # is the ground truth and also tolerates a stale header.
        n_steps = header.get("n_steps")
        steps += n_steps if isinstance(n_steps, int) else len(records) - 1
    return CorpusCensus(trajectories=trajectories, steps=steps)


__all__ = ["CorpusCensus", "LIVE_DIR", "census"]
