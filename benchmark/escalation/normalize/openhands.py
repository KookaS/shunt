"""OpenHands normalizer: an event list (`action`/`observation` dicts with `args`/`extras`/
`source`), structurally different from the SWE-agent `.traj` — this is the dialect the
per-framework protocol is earned by.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from benchmark.escalation.normalize import base

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from benchmark.escalation.schema import StepView, Trajectory


def _events(raw: object) -> list[object]:
    """The event stream, whether the file is a bare list or a {'history': [...]} record."""
    if isinstance(raw, dict):
        history = raw.get("history")
        return list(history) if isinstance(history, list) else []
    return list(raw) if isinstance(raw, list) else []


def _next_observation(events: Sequence[object], start: int) -> dict[str, object] | None:
    """The first observation event after `start` (an action's result), or None."""
    for ev in events[start + 1 :]:
        if isinstance(ev, dict) and "observation" in ev:
            return ev
    return None


class OpenHandsParser:
    """Normalize an OpenHands event stream into canonical StepViews (one per action event)."""

    framework = "openhands"

    def parse(self, raw: object, meta: Mapping[str, object]) -> Trajectory:
        events = _events(raw)
        steps: list[StepView] = []
        for i, ev in enumerate(events):
            if not (isinstance(ev, dict) and "action" in ev):
                continue
            steps.append(self._step(len(steps), ev, _next_observation(events, i)))
        return base.build_trajectory(steps, meta, self.framework)

    def _step(self, index: int, ev: dict[str, object], obs: dict[str, object] | None) -> StepView:
        action = str(ev.get("action", ""))
        args = ev.get("args")
        result = str(obs.get("content", "")) if obs else ""
        return base.make_step(
            step_index=index,
            observation=str(ev.get("message", "")),
            action=action,
            tool=action,
            args=json.dumps(args, sort_keys=True) if isinstance(args, dict) and args else None,
            result=result,
            metadata={"source": str(ev.get("source", ""))},
        )


__all__ = ["OpenHandsParser"]
