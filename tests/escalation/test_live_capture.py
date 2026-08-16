"""The live-benchmark capture gate: a real run with real agent messages persists one
committable, redaction-scrubbed, manifest-listed trajectory; a simulated run (no messages,
or no assistant turns) writes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.escalation import authenticity, schema
from benchmark.escalation.live_capture import capture_live_trajectory, record_snapshot_provenance
from benchmark.runner.step_snapshots import write_snapshots

_FIX = Path(__file__).parent / "fixtures" / "mini_swe_agent.json"


def _messages() -> list[dict[str, object]]:
    return list(json.loads(_FIX.read_text()))


def _capture(out_dir: Path, **kw: object) -> Path | None:
    defaults: dict[str, object] = {
        "instance_id": "psf__requests-1142",
        "model": "deepseek-v4-flash",
        "arm": "default",
        "resolved": False,
    }
    defaults.update(kw)
    return capture_live_trajectory(_messages(), out_dir=out_dir, **defaults)  # type: ignore[arg-type]


def test_real_run_writes_a_committable_trajectory(tmp_path: Path) -> None:
    path = _capture(tmp_path)
    assert path is not None and path.exists()
    traj = schema.load_jsonl(path)
    assert traj.header.plane == "committable"
    assert traj.header.dataset == "swebench_verified"
    assert traj.header.framework == "mini_swe_agent"
    assert traj.header.instance_id == "psf__requests-1142"
    assert traj.header.terminal_resolved is False
    assert traj.steps[-1].blocking is True  # the failed terminal is a verified capability failure


def test_manifest_validates_the_written_trajectory(tmp_path: Path) -> None:
    _capture(tmp_path)
    assert (tmp_path / authenticity.MANIFEST_NAME).exists()
    assert authenticity.errors(authenticity.verify_manifest(tmp_path)) == []


def test_dump_path_is_redaction_scrubbed(tmp_path: Path) -> None:
    messages = _messages()
    # a secret leaking into a bash observation must never reach the committed file
    messages[3]["content"] = "<output>\nsk-secret AKIAIOSFODNN7EXAMPLE token\n</output>"
    path = capture_live_trajectory(
        messages,
        instance_id="psf__requests-1142",
        model="m",
        arm="default",
        resolved=True,
        out_dir=tmp_path,
    )
    assert path is not None
    assert "AKIAIOSFODNN7EXAMPLE" not in path.read_text(encoding="utf-8")


def test_no_messages_writes_nothing(tmp_path: Path) -> None:
    # simulated / classify-only path: no real agent → nothing persisted, ever
    assert (
        capture_live_trajectory(
            [], instance_id="i", model="m", arm="default", resolved=True, out_dir=tmp_path
        )
        is None
    )
    assert list(tmp_path.iterdir()) == []


def test_the_captured_snapshot_count_rides_the_committed_header(tmp_path: Path) -> None:
    # The per-step diffs are gitignored scratch, so their COUNT is the only committed evidence
    # that a trajectory was ever replayable. 0 means "unreplayable for ever", not "not here".
    path = _capture(tmp_path, snapshot_steps=0)
    assert path is not None
    assert schema.load_jsonl(path).header.snapshot_steps == 0
    manifest = json.loads((tmp_path / authenticity.MANIFEST_NAME).read_text())
    assert (
        manifest["trajectories"]["psf__requests-1142__deepseek-v4-flash__default"]["snapshot_steps"]
        == 0
    )


def test_snapshot_provenance_backfill_reads_the_real_scratch(tmp_path: Path) -> None:
    # The migration for trajectories captured before the field existed: it counts what is on the
    # collection host's scratch and must not disturb the step payload (the hash is unchanged).
    out_dir, root = tmp_path / "live", tmp_path / "scratch"
    path = _capture(out_dir)
    assert path is not None
    before = schema.load_jsonl(path)
    assert before.header.snapshot_steps is None
    write_snapshots(before.header.trajectory_id, {0: "d0", 1: "d1"}, root=root)

    counts = record_snapshot_provenance(out_dir, root=root)

    after = schema.load_jsonl(path)
    assert counts[before.header.trajectory_id] == 2
    assert after.header.snapshot_steps == 2
    assert after.header.content_sha256 == before.header.content_sha256
    assert after.steps == before.steps
    assert authenticity.errors(authenticity.verify_manifest(out_dir)) == []


def test_no_assistant_turns_writes_nothing(tmp_path: Path) -> None:
    # messages present but only system/user (no model decision) → no fabricated trajectory
    stub = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
    assert (
        capture_live_trajectory(
            stub, instance_id="i", model="m", arm="default", resolved=True, out_dir=tmp_path
        )
        is None
    )
    assert list(tmp_path.iterdir()) == []


def test_the_backfill_refuses_to_zero_a_recorded_count(tmp_path: Path) -> None:
    # "Run on the collection host only" has to be a wall: run it anywhere else and every count
    # comes back 0 — the one value that authorises the offline replay to clear a trajectory.
    out_dir, root = tmp_path / "live", tmp_path / "scratch"
    path = _capture(out_dir, snapshot_steps=12)
    assert path is not None
    before = path.read_bytes()
    # A scratch that exists and holds diffs for a DIFFERENT trajectory: the wrong-host signature
    # does not fire, so this is the downgrade guard on its own.
    write_snapshots("some-other-trajectory", {0: "d"}, root=root)

    with pytest.raises(RuntimeError, match="already records snapshot_steps"):
        record_snapshot_provenance(out_dir, root=root)
    assert path.read_bytes() == before  # aborts before writing anything


def test_the_backfill_refuses_to_run_without_a_snapshot_scratch(tmp_path: Path) -> None:
    # THE corpus-wiping landmine: on a fresh clone every count comes back 0, and 0 is the one
    # value that authorises the offline replay to clear a trajectory. The downgrade guard cannot
    # see this — a not-yet-migrated header reads None, which is exactly the corpus being migrated
    # — so the WRONG-HOST signature is what is checked.
    out_dir = tmp_path / "live"
    path = _capture(out_dir)  # header snapshot_steps is None, as all 799 committed ones are
    assert path is not None
    before = path.read_bytes()

    with pytest.raises(RuntimeError, match="not the collection host"):
        record_snapshot_provenance(out_dir, root=tmp_path / "absent")
    assert path.read_bytes() == before


def test_the_backfill_refuses_when_the_scratch_holds_nothing_for_this_corpus(
    tmp_path: Path,
) -> None:
    out_dir, root = tmp_path / "live", tmp_path / "scratch"
    write_snapshots("some-other-trajectory", {0: "d"}, root=root)  # root exists, wrong contents
    path = _capture(out_dir)
    assert path is not None
    before = path.read_bytes()

    with pytest.raises(RuntimeError, match="wrong-host signature"):
        record_snapshot_provenance(out_dir, root=root)
    assert path.read_bytes() == before
