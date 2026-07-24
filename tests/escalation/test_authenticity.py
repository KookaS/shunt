"""Layer-1 authenticity: clean data passes; a hand-edited label FAILS (tamper kill-gate);
manifest cross-check flags corruption.
"""

from __future__ import annotations

import json

from benchmark.escalation import authenticity, schema
from benchmark.escalation.authenticity import errors
from tests.escalation.factories import make_step, make_trajectory


def _write(data_dir, traj) -> None:
    schema.dump_jsonl(traj, data_dir / f"{traj.header.trajectory_id}.jsonl")


def _clean_trajectory(tid: str = "t1"):
    return make_trajectory(
        [
            make_step(step_index=0, decision_index=0, success=True),
            make_step(step_index=1, decision_index=1, success=False, failing_check_id="t::a"),
        ],
        trajectory_id=tid,
    )


def test_clean_trajectory_passes() -> None:
    assert errors(authenticity.verify_trajectory(_clean_trajectory())) == []


def test_hand_edited_label_fails_layer1(tmp_path) -> None:
    # Kill-gate: dump a real trajectory, then hand-edit a step's success label on disk without
    # updating the header hash — reload must produce an ERROR Finding.
    traj = _clean_trajectory()
    path = tmp_path / "t1.jsonl"
    schema.dump_jsonl(traj, path)
    lines = path.read_text().splitlines()
    step = json.loads(lines[2])
    step["success"] = True  # flip the failing step to a pass — a fabricated label
    lines[2] = json.dumps(step, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")
    tampered = schema.load_jsonl(path)
    findings = errors(authenticity.verify_trajectory(tampered))
    assert any(f.rule == "content_sha256.mismatch" for f in findings)


def test_forged_blocking_field_is_caught_by_recompute() -> None:
    # Even if a forger rebuilds the content hash, an inconsistent derivable field is caught.
    forged = make_step(step_index=0, decision_index=0, success=True, failing_check_id="t::a")
    object.__setattr__(forged, "blocking", True)  # success but blocking=True is impossible
    traj = make_trajectory([forged], trajectory_id="forge")
    findings = errors(authenticity.verify_trajectory(traj))
    assert any(f.rule == "blocking.mismatch" for f in findings)


def test_manifest_cross_check_clean_and_orphan(tmp_path) -> None:
    _write(tmp_path, _clean_trajectory("a"))
    _write(tmp_path, _clean_trajectory("b"))
    man = authenticity.manifest(tmp_path)
    (tmp_path / authenticity.MANIFEST_NAME).write_text(json.dumps(man))
    assert errors(authenticity.verify_manifest(tmp_path)) == []
    # drop file b but leave it in the manifest → an orphan WARN, and unlisted has no ERROR
    (tmp_path / "b.jsonl").unlink()
    findings = authenticity.verify_manifest(tmp_path)
    assert any(f.rule == "manifest.orphan" for f in findings)


def test_manifest_hash_mismatch_is_error(tmp_path) -> None:
    _write(tmp_path, _clean_trajectory("a"))
    man = authenticity.manifest(tmp_path)
    man["trajectories"]["a"]["content_sha256"] = "0" * 64  # wrong recorded hash
    (tmp_path / authenticity.MANIFEST_NAME).write_text(json.dumps(man))
    findings = errors(authenticity.verify_manifest(tmp_path))
    assert any(f.rule == "manifest.hash_mismatch" for f in findings)
