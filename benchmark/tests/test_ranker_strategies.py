"""Unit tests for the ranker-predicted-difficulty / ranker-defer strategies."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Final

from benchmark import config
from benchmark.routing.strategies import ranker_defer
from benchmark.routing.strategies import ranker_difficulty as rd
from benchmark.routing.strategies._cascade_common import measured_models_by_price
from benchmark.routing.summary import complete_scored_matrix

_DATA: Final[Path] = Path(__file__).resolve().parents[1] / "routing" / "data"

# Predicted-difficulty space: every neighbour sits ~0.2 from the query, so with k=20 the
# whole tiny corpus votes (the k cap only binds on a larger matrix).
_LABELS: Final[dict[str, float]] = {"q": 2.0, "n1": 2.2, "n2": 2.1, "n3": 2.0}

_MODELS: Final[dict[str, dict[str, float]]] = {
    "c0": {"input_price": 1.0, "output_price": 0.0},
    "m0": {"input_price": 2.0, "output_price": 0.0},
    "f0": {"input_price": 4.0, "output_price": 0.0},
}


def _matrix(results: dict) -> dict:
    return {
        "tasks": {tid: {"description": tid} for tid in results},
        "results": results,
        "models": _MODELS,
    }


def _pass_cells(c0: bool, m0: bool) -> dict:
    return {
        "c0": {"pass": c0, "cost": 0.01},
        "m0": {"pass": m0, "cost": 0.03},
        "f0": {"pass": False, "cost": 0.09},
    }


def test_committed_ranker_difficulty_covers_challenge_tasks() -> None:
    challenges = json.loads((_DATA / "challenges.json").read_text())["tasks"]
    with (_DATA / "ranker_predicted_difficulty.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 500
    assert list(rows[0]) == ["challenge_id", "pred_difficulty"]
    assert {r["challenge_id"] for r in rows} == set(challenges)
    assert all(r["pred_difficulty"] != "" for r in rows)


def test_committed_ranker_defer_matches_defer_labels() -> None:
    with (_DATA / "defer_labels.csv").open(newline="") as f:
        defer_ids = {r["challenge_id"] for r in csv.DictReader(f)}
    with (_DATA / "ranker_predicted_defer.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["challenge_id"] for r in rows} == defer_ids
    assert all(0.0 <= float(r["pred_defer"]) <= 1.0 for r in rows)


def test_pick_cheapest_when_predicted_neighbours_pass(monkeypatch) -> None:
    monkeypatch.setattr(rd, "predicted_difficulty", lambda tid: _LABELS.get(tid))
    results = {f"n{i}": _pass_cells(True, False) for i in range(1, 4)}
    got = rd.pick("q", _matrix(results), k=20, success_rate_threshold=0.6, min_samples=3)
    assert got == "c0"


def test_pick_next_rung_when_cheapest_fails(monkeypatch) -> None:
    monkeypatch.setattr(rd, "predicted_difficulty", lambda tid: _LABELS.get(tid))
    results = {f"n{i}": _pass_cells(False, True) for i in range(1, 4)}
    got = rd.pick("q", _matrix(results), k=20, success_rate_threshold=0.6, min_samples=3)
    assert got == "m0"


def test_pick_strongest_when_none_clears_bar(monkeypatch) -> None:
    monkeypatch.setattr(rd, "predicted_difficulty", lambda tid: _LABELS.get(tid))
    results = {f"n{i}": _pass_cells(False, False) for i in range(1, 4)}
    got = rd.pick("q", _matrix(results), k=20, success_rate_threshold=0.6, min_samples=3)
    assert got == "f0"


def test_unlabelled_task_opens_cheap(monkeypatch) -> None:
    monkeypatch.setattr(rd, "predicted_difficulty", lambda tid: None)
    results = {f"n{i}": _pass_cells(False, True) for i in range(1, 4)}
    got = rd.pick("q", _matrix(results), k=20, success_rate_threshold=0.6, min_samples=3)
    assert got == "c0"


def test_neighbor_ids_band_restricts_radius(monkeypatch) -> None:
    monkeypatch.setattr(rd, "predicted_difficulty", lambda tid: _LABELS.get(tid))
    matrix = _matrix({f"n{i}": _pass_cells(True, False) for i in range(1, 4)})
    banded = rd.neighbor_ids("q", matrix, k=20, band=True)
    assert set(banded) == {"n1", "n2", "n3"}
    assert "q" not in banded


def test_cascade_floor_follows_pick(monkeypatch) -> None:
    monkeypatch.setattr(rd, "predicted_difficulty", lambda tid: _LABELS.get(tid))
    strat = rd.RankerDifficultyCascadeStrategy()
    results = {f"n{i}": _pass_cells(False, True) for i in range(1, 4)}
    rungs = ["c0", "m0", "f0"]
    assert strat._initial_rank_floor("q", _matrix(results), rungs) == 1


def test_defer_high_opens_one_rung_up(monkeypatch) -> None:
    monkeypatch.setattr(ranker_defer, "predicted_defer", lambda tid: 0.9 if tid == "q" else 0.1)
    strat = ranker_defer.RankerDeferCascadeStrategy()
    rungs = ["c0", "m0", "f0"]
    assert strat._initial_rank_floor("q", {}, rungs) == 1


def test_defer_low_and_missing_open_cheap(monkeypatch) -> None:
    monkeypatch.setattr(ranker_defer, "predicted_defer", lambda tid: None if tid == "q2" else 0.1)
    strat = ranker_defer.RankerDeferCascadeStrategy()
    rungs = ["c0", "m0", "f0"]
    assert strat._initial_rank_floor("q1", {}, rungs) == 0
    assert strat._initial_rank_floor("q2", {}, rungs) == 0


def test_defer_single_rung_ladder_clamps() -> None:
    strat = ranker_defer.RankerDeferCascadeStrategy()
    assert strat._initial_rank_floor("q", {}, ["c0"]) == 0


# ---------------------------------------- committed tables, real matrix, no monkeypatch
# Every behavioural test above stubs `predicted_difficulty` / `predicted_defer`, so the
# whole suite stays green even if the committed table became unreadable and every task fell
# through the unlabelled path (no neighbours -> cheapest model, rank floor 0). That failure
# is silent and looks exactly like the real, measured result: rows byte-identical to
# Always-Cheap and Session-Cascade. These tests separate the two by pinning BOTH views of
# the committed matrix.


def _real_matrix() -> dict:
    config.load("benchmark/benchmark.yaml")
    return config.load_matrix()


def test_committed_difficulty_pick_varies_on_the_measured_matrix() -> None:
    """The table is live: on the MEASURED matrix the pick is not a constant.

    This is the guard the monkeypatched tests cannot give. A missing or unreadable
    prediction table collapses this to a single model, which is what a degenerate row
    would look like from the outside.
    """
    matrix = _real_matrix()
    rungs = measured_models_by_price(matrix)
    tasks = sorted(matrix["results"])
    picks = Counter(rd.RankerDifficultyStrategy().select(tid, {}, matrix) for tid in tasks)
    assert len(picks) > 1, "constant pick — the predicted-difficulty table is not reaching pick()"
    assert picks[rungs[0]] < len(tasks)
    assert picks == Counter({rungs[0]: 171, rungs[1]: 30})
    floors = Counter(
        rd.RankerDifficultyCascadeStrategy()._initial_rank_floor(tid, matrix, rungs)
        for tid in tasks
    )
    assert set(floors) != {0}
    assert floors == Counter({0: 171, 1: 30})


def test_committed_difficulty_pick_collapses_on_the_completed_matrix() -> None:
    """And the SCORED view is degenerate for a measured reason, not a missing table.

    `compute_strategy_rows` scores the monotone-ladder-COMPLETED matrix. Completion lifts
    the cheap rung's neighbourhood pass rate from 0.689 to 0.751, clearing the 0.6 bar in
    every neighbourhood, so the pick opens cheap on all 181 scored tasks and the row is
    byte-identical to Always-Cheap / Session-Cascade. Pinning it here means a future change
    that makes the rows vary — or that breaks the table — is a visible test failure rather
    than a silently rewritten published number.
    """
    completed, _imputed = complete_scored_matrix(_real_matrix())
    rungs = measured_models_by_price(completed)
    tasks = sorted(completed["results"])
    assert len(tasks) == 181
    assert all(rd.predicted_difficulty(tid) is not None for tid in tasks)
    picks = Counter(rd.RankerDifficultyStrategy().select(tid, {}, completed) for tid in tasks)
    assert picks == Counter({rungs[0]: 181})


def test_committed_defer_cascade_opens_high_on_the_scored_tasks() -> None:
    """The defer row is NOT degenerate: it skips the cheap rung on 11 of the 181 scored."""
    completed, _imputed = complete_scored_matrix(_real_matrix())
    rungs = measured_models_by_price(completed)
    floors = Counter(
        ranker_defer.RankerDeferCascadeStrategy()._initial_rank_floor(tid, completed, rungs)
        for tid in sorted(completed["results"])
    )
    assert set(floors) != {0}
    assert floors == Counter({0: 170, 1: 11})
