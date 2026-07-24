"""Repo hygiene: the escalation data plane is LFS-tracked and un-ignored, plots ship, the live
plane is ignored, and manifest.json stays diffable (out of LFS). Asserted via git itself.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True, check=False)


def _check_ignore(path: str) -> bool:
    """True iff git ignores `path` (check-ignore exits 0 when the path is ignored)."""
    return _git("check-ignore", "-q", path).returncode == 0


def _lfs_filter(path: str) -> str:
    out = _git("check-attr", "filter", path).stdout.strip()
    return out.rsplit(": ", 1)[-1]


@pytest.mark.skipif(not (_ROOT / ".git").exists(), reason="not a git checkout")
def test_escalation_data_is_tracked_not_ignored() -> None:
    assert not _check_ignore("benchmark/escalation/data/foo.jsonl")


@pytest.mark.skipif(not (_ROOT / ".git").exists(), reason="not a git checkout")
def test_escalation_reports_png_shipped_intermediates_ignored() -> None:
    assert not _check_ignore("benchmark/escalation/reports/pr_curve.png")
    assert _check_ignore("benchmark/escalation/reports/scratch.csv")


@pytest.mark.skipif(not (_ROOT / ".git").exists(), reason="not a git checkout")
def test_live_plane_is_ignored() -> None:
    assert _check_ignore("trajectories/session.traj.enc")
    assert _check_ignore("benchmark/escalation/data/live.traj.enc")


@pytest.mark.skipif(not (_ROOT / ".git").exists(), reason="not a git checkout")
def test_trajectory_data_is_lfs_but_manifest_is_not() -> None:
    assert _lfs_filter("benchmark/escalation/data/train.jsonl") == "lfs"
    assert _lfs_filter("benchmark/escalation/data/train.parquet") == "lfs"
    manifest = _lfs_filter("benchmark/escalation/data/manifest.json")
    assert manifest != "lfs"  # manifest.json stays diffable, out of LFS
