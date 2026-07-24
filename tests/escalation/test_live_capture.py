"""The live-benchmark capture gate: a real run with real agent messages persists one
committable, redaction-scrubbed, manifest-listed trajectory; a simulated run (no messages,
or no assistant turns) writes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.escalation import authenticity, schema
from benchmark.escalation.live_capture import capture_live_trajectory

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
