"""The mini-swe-agent parser maps a real-shaped ``agent.messages`` list into a schema-valid
Trajectory: one step per assistant tool-call turn, terminal outcome stamped from the harness
``resolved`` label, and Layer-1 clean.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.escalation import authenticity, schema
from benchmark.escalation.normalize.mini_swe_agent import MiniSweAgentParser

_FIX = Path(__file__).parent / "fixtures" / "mini_swe_agent.json"


def _messages() -> list[dict[str, object]]:
    return list(json.loads(_FIX.read_text()))


def test_one_step_per_assistant_turn() -> None:
    traj = MiniSweAgentParser().parse(_messages(), {"trajectory_id": "t"})
    # system + user + exit are not decisions; the two assistant tool-call turns are.
    assert traj.header.framework == "mini_swe_agent"
    assert traj.header.n_steps == len(traj.steps) == 2


def test_step_carries_command_and_observation() -> None:
    traj = MiniSweAgentParser().parse(_messages(), {"trajectory_id": "t"})
    first, second = traj.steps
    assert first.tool == "bash"
    assert first.args == "grep -n resolve_redirects requests/sessions.py"
    assert "resolve_redirects" in first.result
    assert first.status == "ok"
    # the pytest turn returned rc 1 → the observation marks the step as an error
    assert second.status == "error"


def test_terminal_resolved_stamps_the_last_step_only() -> None:
    passed = MiniSweAgentParser().parse(_messages(), {"terminal_resolved": True})
    assert passed.steps[-1].success is True
    assert passed.steps[-1].confirmed is True
    assert passed.steps[-1].blocking is False
    # a non-terminal step keeps the conservative default (not a verified failure)
    assert passed.steps[0].confirmed is False

    failed = MiniSweAgentParser().parse(_messages(), {"terminal_resolved": False})
    assert failed.steps[-1].success is False
    assert failed.steps[-1].blocking is True  # ¬success ∧ ¬infra
    assert failed.steps[0].blocking is False


def test_parsed_trajectory_is_layer1_clean_and_round_trips(tmp_path: Path) -> None:
    traj = MiniSweAgentParser().parse(
        _messages(), {"trajectory_id": "t", "terminal_resolved": False}
    )
    assert authenticity.errors(authenticity.verify_trajectory(traj)) == []
    out = tmp_path / "t.jsonl"
    schema.dump_jsonl(traj, out)
    assert schema.load_jsonl(out) == traj


def test_empty_messages_yield_no_steps() -> None:
    assert MiniSweAgentParser().parse([], {"trajectory_id": "t"}).steps == []
