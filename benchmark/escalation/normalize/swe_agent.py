"""SWE-agent `.traj` normalizer: a {trajectory: [{action, observation, response, state,
thought}]} record with a JSON-string `state` per step.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from benchmark.escalation.normalize import base

if TYPE_CHECKING:
    from collections.abc import Mapping

    from benchmark.escalation.schema import StepView, Trajectory


def _state_metadata(state: object) -> dict[str, str]:
    """Parse the SWE-agent `state` JSON string into a flat str->str metadata map."""
    if not isinstance(state, str) or not state.strip():
        return {}
    try:
        parsed = json.loads(state)
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in parsed.items()} if isinstance(parsed, dict) else {}


class SweAgentParser:
    """Normalize a SWE-agent trajectory into canonical StepViews."""

    framework = "swe_agent"

    def parse(self, raw: object, meta: Mapping[str, object]) -> Trajectory:
        steps_raw = raw.get("trajectory", []) if isinstance(raw, dict) else []
        steps = [self._step(i, entry) for i, entry in enumerate(steps_raw)]
        return base.build_trajectory(steps, meta, self.framework)

    def _step(self, index: int, entry: object) -> StepView:
        data = entry if isinstance(entry, dict) else {}
        action = str(data.get("action", ""))
        tool, args = base.first_token(action)
        observation = str(data.get("observation", ""))
        return base.make_step(
            step_index=index,
            observation=observation,
            action=action,
            tool=tool,
            args=args,
            result=observation,
            metadata=_state_metadata(data.get("state")),
        )


__all__ = ["SweAgentParser"]
