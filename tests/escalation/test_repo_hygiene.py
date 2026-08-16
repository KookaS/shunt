"""Repo hygiene: the escalation data plane is LFS-tracked and un-ignored, plots ship, the live
plane is ignored, manifest.json stays diffable, and the committed corpus carries no
agent-independent all-one-outcome instance. Asserted via git and the data itself.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import dataclass, field
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
def test_escalation_figures_shipped_intermediates_ignored() -> None:
    # The PNGs moved into the published docs tree; the reports dir now holds only
    # intermediates plus the tracked metrics.json. Both halves of that must hold, or a
    # figure silently stops shipping.
    assert not _check_ignore("docs/assets/figures/escalation/session_value.png")
    assert not _check_ignore("docs/assets/figures/routing/kill_gate.png")
    assert not _check_ignore("benchmark/escalation/reports/metrics.json")
    assert _check_ignore("benchmark/escalation/reports/scratch.csv")
    assert _check_ignore("benchmark/escalation/reports/stray.png")


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


# ---------------------------------------------------------------------------
# CORPUS-LEVEL INSTRUMENT GUARD — reads the committed artefact, needs no container.
#
# This is the check that would have caught all three replay defects from the data alone. A
# replayed step outcome is supposed to be a measurement OF THE AGENT'S EDIT. So an instance whose
# every replayed step lands on the same side, with no variation an agent could have caused, is
# evidence the selector — not the agent — determined the outcome:
#
#   * 100% exit-0        → the selector matched no test; `0 passed` at exit 0 stamped green.
#   * 100% nonzero exit, → the selector could not even be loaded; every step is the SAME
#     ≤ 1 dedup key        constant error, so the "recurrence" is a property of the instance.
#
# It is a heuristic on purpose: a real instance the agent never fixed can legitimately be all-red,
# but never all-red on a SINGLE constant key across every trajectory and model. Distinct non-null
# keys are what is counted — a null key means the step was classified infra and carries no key.
# ---------------------------------------------------------------------------

_LIVE_DIR = _ROOT / "benchmark" / "escalation" / "data" / "live"

# The corpus is LFS-tracked; an un-hydrated checkout holds pointer files, not JSON.
_CORPUS_PRESENT = _LIVE_DIR.is_dir() and any(
    p.read_bytes()[:1] == b"{" for p in list(_LIVE_DIR.glob("*.jsonl"))[:1]
)
_needs_corpus = pytest.mark.skipif(
    not _CORPUS_PRESENT, reason="live corpus absent or LFS pointers not hydrated"
)

# The committed corpus once predated the replay fixes (raw FAIL_TO_PASS ids as selectors, exit-0
# stamped green, exit 3/4/5 stamped as capability reds), so both scans below failed on it: 13
# all-green instances (sympy) and 25 all-red-single-key instances (django + friends). `strict=True`
# is the ratchet — the moment a regenerated corpus makes one of these pass, the XPASS turns the
# suite red and forces the marker's removal. It fired for one of the two: the all-green scan now
# passes on the rebuilt + state-capture-marked corpus, so its marker is gone.
#
# THE OTHER ONE MOVED THE WRONG WAY, AND THAT IS THE NUMBER TO CARRY. On the rebuilt corpus the
# all-red-single-key scan does not merely still fail — it fails LARGER: 29 offending instances
# against the pre-fix 25 (measured 2026-08-02 over the 152 instances that carry any replayed step,
# of 166 in the corpus). The 29 are MEASURED reds where the 25 were the pre-fix selector's, which
# is why the rise is worth reading rather than dismissing — but whether the extra four are
# instances the honest replay newly exposes as unmeasurable, or an artifact of the new one, is not
# settled here. So the marker stays, and a reader must not carry the historical 25 forward as the
# current size of the problem. The count is corpus-bound and will move again; the assertion below
# prints the live figure and the offending ids, which is the number to trust over any written here.
_PENDING_REGEN = pytest.mark.xfail(
    strict=True,
    reason="committed corpus was replayed with the pre-fix selector; pending corpus regeneration",
)


@dataclass
class _InstanceScan:
    """Per-SWE-bench-instance tallies over every replayed step, across all its trajectories."""

    zero_exit: int = 0
    nonzero_exit: int = 0
    dedup_keys: Counter[str] = field(default_factory=Counter)

    @property
    def replayed(self) -> int:
        return self.zero_exit + self.nonzero_exit


def _scan_live_corpus() -> dict[str, _InstanceScan]:
    """Tally replayed (exit_code-bearing) non-terminal steps per instance id."""
    # The terminal step is excluded: it carries the SWE-bench grader's `resolved` verdict, not a
    # replayed test run, so including it would mix two different measurements.
    scans: dict[str, _InstanceScan] = {}
    for path in sorted(_LIVE_DIR.glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        instance_id = records[0].get("instance_id")
        if instance_id is None:
            continue
        scan = scans.setdefault(instance_id, _InstanceScan())
        for step in records[1:-1]:
            # The replayed step's process return code lives in `replay_rc`; committed corpora
            # predating the two-referent split carried it in `exit_code`. Read both spellings so
            # the scan stays green over legacy AND newly-restamped data.
            exit_code = (
                step.get("replay_rc")
                if step.get("replay_rc") is not None
                else step.get("exit_code")
            )
            if exit_code is None:
                continue  # never replayed (or left unstamped by the admissibility gate)
            if exit_code == 0:
                scan.zero_exit += 1
                continue
            scan.nonzero_exit += 1
            key = step.get("dedup_key")
            if key is not None:
                scan.dedup_keys[key] += 1
    return scans


@_needs_corpus
def test_no_instance_is_100_percent_exit_zero() -> None:
    # A1 (sympy): `bin/test` given a bare F2P name selects nothing, exits 0, and every step of
    # every trajectory for that instance is stamped a pass — including tasks the grader failed.
    scans = _scan_live_corpus()
    offenders = sorted(
        iid for iid, s in scans.items() if s.replayed > 0 and s.zero_exit == s.replayed
    )
    assert not offenders, (
        f"{len(offenders)} instance(s) have a 100% exit-0 replayed step history — no agent edit "
        f"could have moved the outcome: {offenders}"
    )


@_needs_corpus
@_PENDING_REGEN
def test_no_instance_is_100_percent_nonzero_on_a_single_key() -> None:
    # A3 (django) + A2 (astropy/pylint): the selector cannot be loaded at all, so every step is
    # the same constant error under one dedup key — which fires the recurrence trigger by
    # construction, independent of anything the agent did.
    scans = _scan_live_corpus()
    offenders = sorted(
        iid
        for iid, s in scans.items()
        if s.replayed > 0 and s.nonzero_exit == s.replayed and len(s.dedup_keys) <= 1
    )
    assert not offenders, (
        f"{len(offenders)} instance(s) are 100% nonzero-exit on a single dedup key — the failure "
        f"is a property of the instance, not the agent: {offenders}"
    )
