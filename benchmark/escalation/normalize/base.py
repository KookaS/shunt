"""The TrajectoryParser seam: one protocol, one concrete impl per framework dialect."""

# The parsers canonicalize a framework's raw trace STRUCTURE into StepView. Per-step verified
# outcomes (pass/fail, failing-check id) are NOT invented here — a generic `.traj` does not carry
# them; they arrive with the outcome join at collection time. So a parsed step defaults to a
# non-failure, and the terminal label comes from the caller's meta.

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from benchmark.escalation import schema
from benchmark.escalation.schema import StepView, Trajectory, TrajectoryHeader

if TYPE_CHECKING:
    from collections.abc import Mapping


class TrajectoryParser(Protocol):
    """Normalize one framework's raw trace into the canonical Trajectory."""

    framework: str

    def parse(self, raw: object, meta: Mapping[str, object]) -> Trajectory: ...


def make_step(
    *,
    step_index: int,
    observation: str,
    action: str,
    tool: str,
    args: str | None,
    result: str,
    metadata: dict[str, str],
    status: str = "ok",
) -> StepView:
    """A structural StepView with conservative (non-failure) verified-outcome defaults."""
    return StepView(
        step_index=step_index,
        decision_index=step_index,
        parent_step_index=step_index - 1 if step_index > 0 else None,
        metadata=metadata,
        observation=observation,
        action=action,
        tool=tool,
        args=args,
        result=result,
        status=status,
        test_passed=None,
        test_total=None,
        failing_check_id=None,
        exit_code=None,
        blocking=False,  # = ¬success ∧ ¬infra with success defaulting True → recomputes equal
        is_infra_failure=False,
        confirmed=False,
        success=True,
        is_revert=False,
        retry_count=0,
        loop_signal=False,
        subgoal_progress=None,
        dedup_key=None,
        model=None,
        reasoning_effort=None,
        rank_index=None,
        effort_index=None,
        real_cost=None,
    )


def build_trajectory(
    steps: list[StepView], meta: Mapping[str, object], framework: str
) -> Trajectory:
    """Wrap normalized steps in a header from `meta` with a recomputed content hash."""
    header = TrajectoryHeader(
        schema_version=schema.SCHEMA_VERSION,
        trajectory_id=str(meta.get("trajectory_id", "unknown")),
        dataset=str(meta.get("dataset", "unknown")),
        plane=str(meta.get("plane", "committable")),
        framework=framework,
        terminal_resolved=bool(meta.get("terminal_resolved", False)),
        instance_id=_opt_str(meta.get("instance_id")),
        license=_opt_str(meta.get("license")),
        dataset_revision=_opt_str(meta.get("dataset_revision")),
        redacted=bool(meta.get("redacted", False)),
        content_sha256=schema.content_sha256(steps),
        n_steps=len(steps),
        snapshot_steps=_opt_int(meta.get("snapshot_steps")),
    )
    return Trajectory(header=header, steps=steps)


def _opt_str(value: object) -> str | None:
    return None if value is None else str(value)


def _opt_int(value: object) -> int | None:
    return None if value is None else int(str(value))


def first_token(text: str) -> tuple[str, str | None]:
    """Split a command line into (tool, args-or-None) on the first whitespace run."""
    stripped = text.strip()
    if not stripped:
        return "", None
    head, _, tail = stripped.partition(" ")
    return head, tail.strip() or None


__all__ = ["TrajectoryParser", "build_trajectory", "first_token", "make_step"]
