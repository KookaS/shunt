"""mini-swe-agent in-memory normalizer: the ``agent.messages`` list a live benchmark run
holds after ``agent.run()`` — assistant tool-call turns paired with their bash observations,
one StepView per model decision. A distinct dialect from the three ``.traj`` file formats.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from benchmark.escalation import schema
from benchmark.escalation.normalize import base

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from benchmark.escalation.schema import StepView, Trajectory
    from shunt.verifiers.base import VerifierResult


class MiniSweAgentParser:
    """Normalize a mini-swe-agent message list into canonical StepViews (one per model call)."""

    framework = "mini_swe_agent"

    def parse(
        self,
        raw: object,
        meta: Mapping[str, object],
        step_outcomes: Mapping[int, VerifierResult] | None = None,
    ) -> Trajectory:
        outcomes = step_outcomes or {}
        messages = raw if isinstance(raw, list) else []
        steps: list[StepView] = []
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                step = self._step(len(steps), msg, _next_observation(messages, i))
                if step.step_index in outcomes:  # ADR-PSV-4 side-channel map
                    step = stamp_step(step, outcomes[step.step_index])
                steps.append(step)
        if steps and "terminal_resolved" in meta:
            steps[-1] = _stamp_terminal(steps[-1], bool(meta["terminal_resolved"]))
        return base.build_trajectory(steps, meta, self.framework)

    def _step(self, index: int, msg: dict[str, object], obs: dict[str, object] | None) -> StepView:
        content = str(msg.get("content") or "")
        command = _command(msg)
        tool, args = ("bash", command) if command is not None else base.first_token(content)
        result = str(obs.get("content", "")) if obs else ""
        status = "error" if obs is not None and _returncode(obs) not in (0, None) else "ok"
        return base.make_step(
            step_index=index,
            observation=content,
            action=command if command is not None else content,
            tool=tool,
            args=args,
            result=result,
            metadata=_metadata(msg),
            status=status,
        )


def _command(msg: dict[str, object]) -> str | None:
    """The bash command mini-swe-agent parsed from this assistant turn's tool call, if any."""
    extra = msg.get("extra")
    actions = extra.get("actions") if isinstance(extra, dict) else None
    if isinstance(actions, list) and actions and isinstance(actions[0], dict):
        cmd = actions[0].get("command")
        return None if cmd is None else str(cmd)
    return None


def _next_observation(messages: Sequence[object], start: int) -> dict[str, object] | None:
    """The tool-result message for the assistant turn at `start` (before the next decision)."""
    for ev in messages[start + 1 :]:
        if not isinstance(ev, dict):
            continue
        if ev.get("role") == "tool":
            return ev
        if ev.get("role") == "assistant":
            break  # next decision reached with no observation for this turn
    return None


def _returncode(obs: dict[str, object]) -> object:
    extra = obs.get("extra")
    return extra.get("returncode") if isinstance(extra, dict) else None


def _metadata(msg: dict[str, object]) -> dict[str, str]:
    extra = msg.get("extra")
    actions = extra.get("actions") if isinstance(extra, dict) else None
    if isinstance(actions, list) and actions and isinstance(actions[0], dict):
        tcid = actions[0].get("tool_call_id")
        if tcid is not None:
            return {"tool_call_id": str(tcid)}
    return {}


def _stamp_terminal(step: StepView, resolved: bool) -> StepView:
    """Stamp the terminal step with the harness verified outcome (blocking via the ONE rule)."""
    # Only the terminal step carries a verified label: the SWE-bench harness `resolved` bool is
    # authoritative (not a flake), so confirmed=True and it is not an infra failure. It is a grade,
    # not a parsed node id, so any per-step failing-check id is cleared — the terminal step never
    # feeds the recurrence trigger. `blocking` is recomputed through the shared predicate so live
    # and offline agree, and Layer-1 stays clean.
    stamped = replace(
        step,
        success=resolved,
        confirmed=True,
        is_infra_failure=False,
        failing_check_id=None,
        dedup_key=None,
    )
    return replace(stamped, blocking=schema.recompute_blocking(stamped))


def stamp_step(step: StepView, outcome: VerifierResult) -> StepView:
    """Stamp a per-step verified outcome onto a StepView (shared by the live map + offline replay).

    A capability failure carries its failing-check id (→ dedup key); a success or an infra/env
    error keeps the non-failure default (no id, non-blocking). Deterministic harness ⇒ confirmed.
    """
    is_failure = outcome.outcome == "failure"
    stamped = replace(
        step,
        success=not is_failure,
        failing_check_id=outcome.failing_check_id if is_failure else None,
        exit_code=outcome.exit_code,
        is_infra_failure=outcome.is_infra_failure,
        confirmed=True,
    )
    stamped = replace(stamped, dedup_key=schema.normalize_dedup_key(stamped.failing_check_id))
    return replace(stamped, blocking=schema.recompute_blocking(stamped))


def restamp_trajectory(traj: Trajectory, step_outcomes: Mapping[int, VerifierResult]) -> Trajectory:
    """Apply offline per-step outcomes to a parsed trajectory (terminal stays authoritative)."""
    # The committed plane holds StepViews (raw messages are gone), so an offline replay restamps
    # the persisted trajectory directly. The terminal step's harness `resolved` label is re-asserted
    # last so an intermediate test outcome can never override the authoritative grade.
    steps = [
        stamp_step(s, step_outcomes[s.step_index]) if s.step_index in step_outcomes else s
        for s in traj.steps
    ]
    if steps:
        steps[-1] = _stamp_terminal(steps[-1], traj.header.terminal_resolved)
    header = replace(traj.header, content_sha256=schema.content_sha256(steps), n_steps=len(steps))
    return schema.Trajectory(header=header, steps=steps)


__all__ = ["MiniSweAgentParser", "restamp_trajectory", "stamp_step"]
