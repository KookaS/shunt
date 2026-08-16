"""SWE-smith `.traj` normalizer: the SWE-agent trajectory shape with extra per-step keys
(`execution_time`, `extra_info`, a nested `messages` log). A distinct parser so a dialect
drift in either framework changes one file, not a shared god-parser.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmark.escalation.normalize import base
from benchmark.escalation.normalize.swe_agent import _state_metadata

if TYPE_CHECKING:
    from collections.abc import Mapping

    from benchmark.escalation.schema import StepView, Trajectory


class SweSmithParser:
    """Normalize a SWE-smith trajectory into canonical StepViews."""

    framework = "swe_smith"

    def parse(self, raw: object, meta: Mapping[str, object]) -> Trajectory:
        steps_raw = raw.get("trajectory", []) if isinstance(raw, dict) else []
        steps = [self._step(i, entry) for i, entry in enumerate(steps_raw)]
        return base.build_trajectory(steps, meta, self.framework)

    def _step(self, index: int, entry: object) -> StepView:
        data = entry if isinstance(entry, dict) else {}
        action = str(data.get("action", ""))
        tool, args = base.first_token(action)
        observation = str(data.get("observation", ""))
        metadata = _state_metadata(data.get("state"))
        extra = data.get("extra_info")
        if isinstance(extra, dict):
            metadata |= {f"extra.{k}": str(v) for k, v in extra.items()}
        return base.make_step(
            step_index=index,
            observation=observation,
            action=action,
            tool=tool,
            args=args,
            result=observation,
            metadata=metadata,
        )


__all__ = ["SweSmithParser"]
