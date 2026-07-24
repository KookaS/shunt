"""Offline snapshot replay (Mechanism B): derive real per-step outcomes from captured diffs."""

# For each captured step diff we rebuild the instance's base state in a container, apply the diff,
# apply the gold ``test_patch`` (which adds the FAIL_TO_PASS tests), run that subset, and classify
# the output through the SAME `parse_test_outcome` the live tier2 verifier uses — so the per-step
# dedup key matches production. A container/apply/collection error is marked ``is_infra_failure``,
# never fabricated into a pass or a fail (real-only).

from __future__ import annotations

import logging
import shlex
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from benchmark.escalation import authenticity, schema
from benchmark.escalation.normalize.mini_swe_agent import restamp_trajectory
from benchmark.runner import step_snapshots, swebench_specs
from benchmark.runner.step_snapshots import TESTBED
from shunt.verifiers.base import VerifierResult
from shunt.verifiers.parse import parse_test_outcome

if TYPE_CHECKING:
    from collections.abc import Iterator

_LOG = logging.getLogger(__name__)


class ContainerExec(Protocol):
    """Run a shell command inside the replay container, returning (combined output, returncode)."""

    def __call__(self, command: str, stdin: str | None = None) -> tuple[str, int]: ...


def _infra(detail: str) -> VerifierResult:
    """A step whose state could not be reconstructed — real-only: never a pass or a fail."""
    return VerifierResult(outcome="unknown", confidence=0.0, detail=detail, is_infra_failure=True)


def _reset_to_base(exec_fn: ContainerExec) -> bool:
    """Discard any prior step's changes so each snapshot replays from the instance base."""
    _out, rc = exec_fn(f"git -C {TESTBED} checkout -- . && git -C {TESTBED} clean -fdq")
    return rc == 0


def replay_step(
    diff: str, test_patch: str, test_cmd: str, fail_to_pass: list[str], exec_fn: ContainerExec
) -> VerifierResult:
    """Reconstruct one step's state in the container, run the F2P subset, classify the outcome."""
    if not _reset_to_base(exec_fn):
        return _infra("container reset to base failed")
    if diff.strip():
        _out, rc = exec_fn(f"git -C {TESTBED} apply -", stdin=diff)
        if rc != 0:
            return _infra(f"step diff did not apply (rc={rc})")
    if test_patch.strip():
        _out, rc = exec_fn(f"git -C {TESTBED} apply -", stdin=test_patch)
        if rc != 0:
            return _infra(f"test_patch did not apply (rc={rc})")
    ids = " ".join(shlex.quote(t) for t in fail_to_pass)
    combined, rc = exec_fn(f"cd {TESTBED} && {test_cmd} {ids}")
    return parse_test_outcome(combined, rc)


def replay_snapshots(
    *,
    snapshots: dict[int, str],
    test_patch: str,
    test_cmd: str,
    fail_to_pass: list[str],
    exec_fn: ContainerExec,
) -> dict[int, VerifierResult]:
    """Replay every captured per-step diff, returning the per-step VerifierResult keyed by step."""
    return {
        index: replay_step(snapshots[index], test_patch, test_cmd, fail_to_pass, exec_fn)
        for index in sorted(snapshots)
    }


# ---------------------------------------------------------------------------
# Real-container plumbing — needs Docker + swebench + the datasets row. Exercised only in the
# owner's offline replay step (flagged); unit tests drive `replay_snapshots` with a fake exec_fn.
# ---------------------------------------------------------------------------


def swebench_test_command(repo: str, version: str) -> str:
    """The per-repo/version test command from SWE-bench's own spec map (django ≠ pytest)."""
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS  # noqa: PLC0415

    return str(MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"])


def docker_exec(container: str) -> ContainerExec:
    """A ContainerExec backed by ``docker exec`` on a container (stdin piped for patches)."""

    def _exec(command: str, stdin: str | None = None) -> tuple[str, int]:
        argv = ["docker", "exec", "-i", container, "bash", "-lc", command]
        proc = subprocess.run(argv, input=stdin, capture_output=True, text=True, check=False)
        return f"{proc.stdout}\n{proc.stderr}", proc.returncode

    return _exec


@contextmanager
def instance_container(image_ref: str, name: str) -> Iterator[str]:
    """Start a detached container from the instance image; always reap it (context manager)."""
    subprocess.run(
        ["docker", "run", "-d", "--name", name, image_ref, "sleep", "infinity"],
        check=True,
        capture_output=True,
    )
    try:
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)


def _dataset_test_patch(instance_id: str) -> str:
    """The gold ``test_patch`` (adds the F2P tests) for an instance, from the HF Verified row."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(swebench_specs.DATASET_NAME, split=swebench_specs.DATASET_SPLIT)
    for row in ds:
        if str(row["instance_id"]) == instance_id:
            return str(row["test_patch"])
    raise KeyError(f"instance {instance_id!r} not in {swebench_specs.DATASET_NAME}")


def run_offline_replay(trajectory_id: str, instance_id: str, jsonl_path: Path) -> Path | None:
    """Full offline pass: snapshots → container replay → restamp the committed trajectory jsonl."""
    snapshots = step_snapshots.read_snapshots(trajectory_id)
    if not snapshots:
        _LOG.warning("no snapshots for %s; nothing to replay", trajectory_id)
        return None
    spec = swebench_specs.load_spec(instance_id)
    if spec is None:
        raise KeyError(f"no SWE-bench spec for {instance_id!r}; materialise it first")
    test_cmd = swebench_test_command(spec.repo, spec.version)
    test_patch = _dataset_test_patch(instance_id)
    container = f"shunt-replay-{trajectory_id}"
    with instance_container(spec.image_ref, container) as name:
        outcomes = replay_snapshots(
            snapshots=snapshots,
            test_patch=test_patch,
            test_cmd=test_cmd,
            fail_to_pass=spec.fail_to_pass,
            exec_fn=docker_exec(name),
        )
    traj = restamp_trajectory(schema.load_jsonl(jsonl_path), outcomes)
    schema.dump_jsonl(traj, jsonl_path)
    _rewrite_manifest(jsonl_path.parent)
    _LOG.info("restamped %s with %d per-step outcomes", jsonl_path, len(outcomes))
    return jsonl_path


def _rewrite_manifest(out_dir: Path) -> None:
    """Rebuild the Layer-1 manifest after restamping (idempotent)."""
    import json  # noqa: PLC0415

    payload = json.dumps(authenticity.manifest(out_dir), indent=2, sort_keys=True)
    (out_dir / authenticity.MANIFEST_NAME).write_text(payload + "\n", encoding="utf-8")


def _main() -> int:
    import argparse  # noqa: PLC0415

    from benchmark.escalation.live_capture import LIVE_DIR  # noqa: PLC0415

    ap = argparse.ArgumentParser(description="Offline snapshot replay → per-step outcomes.")
    ap.add_argument("trajectory_id", help="captured trajectory id (also the snapshot dir name)")
    ap.add_argument("--instance-id", required=True, help="SWE-bench Verified instance id")
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="trajectory jsonl to restamp (default: the live plane's <trajectory_id>.jsonl)",
    )
    args = ap.parse_args()
    jsonl = args.jsonl or (LIVE_DIR / f"{args.trajectory_id}.jsonl")
    result = run_offline_replay(args.trajectory_id, args.instance_id, jsonl)
    print(f"restamped {result}" if result else "no snapshots; nothing restamped")
    return 0


__all__ = [
    "ContainerExec",
    "docker_exec",
    "instance_container",
    "replay_snapshots",
    "replay_step",
    "run_offline_replay",
    "swebench_test_command",
]


if __name__ == "__main__":
    raise SystemExit(_main())
