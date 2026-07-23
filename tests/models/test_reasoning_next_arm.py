"""D: ReasoningConfig.next_arm_above — the effort-escalation resolution primitive."""

from __future__ import annotations

from shunt.models.config import ReasoningArm, ReasoningConfig


def _cfg() -> ReasoningConfig:
    return ReasoningConfig(
        default_arm="low",
        arms=[
            ReasoningArm(id="low", rank=0, api={"reasoning_effort": "low"}),
            ReasoningArm(id="high", rank=1, api={"reasoning_effort": "high"}),
            ReasoningArm(id="max", rank=2, api={"reasoning_effort": "max"}),
        ],
    )


def test_next_arm_steps_up_by_rank() -> None:
    assert _cfg().next_arm_above("low") == "high"
    assert _cfg().next_arm_above("high") == "max"


def test_next_arm_above_top_is_none() -> None:
    assert _cfg().next_arm_above("max") is None


def test_next_arm_unknown_id_is_none() -> None:
    assert _cfg().next_arm_above("nonexistent") is None
