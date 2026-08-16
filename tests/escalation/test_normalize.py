"""Per-framework parsers each normalize their REAL checked-in fixture into a schema-valid
Trajectory. Fixtures are trimmed real captures (see fixtures/PROVENANCE.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest

from benchmark.escalation import authenticity, schema
from benchmark.escalation.normalize.base import TrajectoryParser
from benchmark.escalation.normalize.openhands import OpenHandsParser
from benchmark.escalation.normalize.swe_agent import SweAgentParser
from benchmark.escalation.normalize.swe_smith import SweSmithParser

_FIX = Path(__file__).parent / "fixtures"

_CASES: Final = [
    (SweAgentParser(), "swe_agent.traj", "swe_agent"),
    (SweSmithParser(), "swe_smith.traj", "swe_smith"),
    (OpenHandsParser(), "openhands.json", "openhands"),
]


@pytest.mark.parametrize(("parser", "fixture", "framework"), _CASES)
def test_parser_normalizes_real_fixture(
    parser: TrajectoryParser, fixture: str, framework: str
) -> None:
    raw = json.loads((_FIX / fixture).read_text())
    meta = {"trajectory_id": framework, "dataset": "fixture", "terminal_resolved": True}
    traj = parser.parse(raw, meta)
    assert traj.header.framework == framework
    assert traj.header.n_steps == len(traj.steps)
    assert traj.steps, "the real fixture produced at least one step"
    # schema-valid: derivable fields are self-consistent (Layer-1 clean)
    assert authenticity.errors(authenticity.verify_trajectory(traj)) == []


@pytest.mark.parametrize(("parser", "fixture", "framework"), _CASES)
def test_parsed_trajectory_round_trips(
    parser: TrajectoryParser, fixture: str, framework: str, tmp_path: Path
) -> None:
    raw = json.loads((_FIX / fixture).read_text())
    traj = parser.parse(raw, {"trajectory_id": framework})
    out = tmp_path / "t.jsonl"
    schema.dump_jsonl(traj, out)
    assert schema.load_jsonl(out) == traj


def test_all_three_parsers_satisfy_the_protocol() -> None:
    parsers: list[TrajectoryParser] = [SweAgentParser(), SweSmithParser(), OpenHandsParser()]
    assert [p.framework for p in parsers] == ["swe_agent", "swe_smith", "openhands"]
