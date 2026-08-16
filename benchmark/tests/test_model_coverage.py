"""The per-model coverage flag: a configured model with no data must fail loudly."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark import model_coverage
from benchmark.escalation import features, schema
from benchmark.escalation.normalize.base import build_trajectory, make_step

_DEPTH = 2
_ADMISSIBLE = _DEPTH + features.MIN_WITHHELD + 1  # +1: the terminal step is never scorable


def _write(tmp_path: Path, trajectory_id: str, n_steps: int, *, stamped: bool) -> None:
    steps = [
        make_step(
            step_index=i,
            observation="o",
            action="a",
            tool="bash",
            args=None,
            result="r",
            metadata={},
        )
        for i in range(n_steps)
    ]
    if stamped:
        steps = [schema.StepView(**{**vars(s), "confirmed": True}) for s in steps]
    meta = {"trajectory_id": trajectory_id, "instance_id": trajectory_id.split("__")[1]}
    traj = build_trajectory(steps, meta, "mini-swe-agent")
    schema.dump_jsonl(traj, tmp_path / f"{trajectory_id}.jsonl")


def test_escalation_counts_buckets_by_model_and_gates_on_stamping(tmp_path: Path) -> None:
    _write(tmp_path, "repo__inst-1__cheap__high", _ADMISSIBLE, stamped=True)
    _write(tmp_path, "repo__inst-2__cheap__high", _ADMISSIBLE, stamped=False)  # never stamped
    _write(tmp_path, "repo__inst-3__cheap__high", _DEPTH, stamped=True)  # too short
    _write(tmp_path, "repo__inst-4__dear__max", _ADMISSIBLE, stamped=True)

    counts = model_coverage.escalation_counts(tmp_path, _DEPTH)

    assert counts["cheap"] == (3, 2 * (_ADMISSIBLE - 1) + (_DEPTH - 1), 1)
    assert counts["dear"] == (1, _ADMISSIBLE - 1, 1)


def test_a_configured_model_with_no_data_is_absent_not_omitted() -> None:
    # The whole point: `escalation.features.model_coverage` enumerates the corpus, so a newly
    # enabled model is invisible there. Here it gets a row, at zero, and grades ABSENT.
    rows = model_coverage.build_rows(["cheap", "brand-new"], {"cheap": 9}, {"cheap": (4, 100, 4)})

    new = next(r for r in rows if r.model == "brand-new")
    assert (new.routing_cells, new.trajectories, new.steps, new.admissible) == (0, 0, 0, 0)
    assert new.escalation_status(1) == "ABSENT"
    assert new.routing_status(1) == "ABSENT"


@pytest.mark.parametrize(
    ("count", "floor", "expected"), [(0, 5, "ABSENT"), (4, 5, "THIN"), (5, 5, "OK")]
)
def test_grade_boundaries(count: int, floor: int, expected: str) -> None:
    assert model_coverage._grade(count, floor) == expected


def test_unconfigured_models_are_surfaced() -> None:
    stale = model_coverage.unconfigured_models(
        ["cheap"], {"cheap": 9, "dropped": 3}, {"retired": (1, 10, 1)}
    )

    assert stale == ["dropped", "retired"]
