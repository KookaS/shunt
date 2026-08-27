"""Tests for the judge-difficulty routing strategies (knn_difficulty.py)."""

from __future__ import annotations

import pytest

import benchmark.routing.strategies.knn_difficulty as kd
from benchmark.routing.strategies.knn_difficulty import (
    DifficultyBandCascadeStrategy,
    knnDifficultyCascadeStrategy,
    knnDifficultyStrategy,
    neighbor_ids,
    pick,
)


def test_the_committed_difficulty_table_resolves_absolutely() -> None:
    # Resolved from the module (like config.py's path convention), never from the cwd: a
    # caller in any directory must load the same committed table.
    assert kd._DIFFICULTY_TABLE.is_absolute()
    assert kd._DIFFICULTY_TABLE.exists()
    assert kd._DIFFICULTY_TABLE.name == "judge_difficulty.json"


@pytest.fixture
def diff(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    """A planted difficulty table: t1..t5 at levels 1..5 plus a level-3 band pair (t3, t3b)."""
    table = {"t1": 1.0, "t2": 2.0, "t3": 3.0, "t3b": 3.2, "t4": 4.0, "t5": 5.0}
    monkeypatch.setattr(
        "benchmark.routing.strategies.knn_difficulty.difficulty", lambda t: table.get(t)
    )
    monkeypatch.setattr(
        "benchmark.routing.strategies.knn_difficulty.judge_cost",
        lambda t: 0.001 if t in table else 0.0,
    )
    return table


def _matrix() -> dict:
    return {
        "models": {
            "cheap": {"input_price": 0.2, "output_price": 0.2},
            "mid": {"input_price": 2.0, "output_price": 2.0},
            "frontier": {"input_price": 10.0, "output_price": 10.0},
        },
        "results": {
            "t1": {"cheap": {"pass": True, "cost": 1.0}},
            "t2": {"cheap": {"pass": True, "cost": 1.0}, "mid": {"pass": True, "cost": 2.5}},
            "t3": {"cheap": {"pass": False, "cost": 0.5}, "mid": {"pass": True, "cost": 2.5}},
            "t3b": {"cheap": {"pass": False, "cost": 0.5}, "mid": {"pass": True, "cost": 2.5}},
            "t4": {
                "cheap": {"pass": False, "cost": 0.5},
                "mid": {"pass": False, "cost": 2.5},
                "frontier": {"pass": True, "cost": 10.0},
            },
            "t5": {
                "cheap": {"pass": False, "cost": 0.5},
                "mid": {"pass": False, "cost": 2.5},
                "frontier": {"pass": True, "cost": 10.0},
            },
        },
    }


class TestNeighbourhood:
    def test_neighbors_order_by_difficulty_distance(self, diff: dict[str, float]) -> None:
        # t3 (3.0): t3b is 0.2 away, then t2 and t4 tie at 1.0 (tid breaks the tie).
        assert neighbor_ids("t3", _matrix(), k=3) == ["t3b", "t2", "t4"]

    def test_self_excluded(self, diff: dict[str, float]) -> None:
        assert "t3" not in neighbor_ids("t3", _matrix(), k=10)

    def test_band_restricts_to_same_level(self, diff: dict[str, float]) -> None:
        assert neighbor_ids("t3", _matrix(), k=10, band=True) == ["t3b"]

    def test_unlabelled_task_has_no_neighbors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "benchmark.routing.strategies.knn_difficulty.difficulty", lambda _t: None
        )
        assert neighbor_ids("t3", _matrix(), k=3) == []


class TestPick:
    def test_cheapest_eligible_over_difficulty_neighbors(self, diff: dict[str, float]) -> None:
        # t5's two nearest (t4, t3b): cheap fails both, mid passes only t3b (0.5 < 0.6),
        # frontier passes t4 — so the pick is frontier.
        assert pick("t5", _matrix(), k=2, success_rate_threshold=0.6, min_samples=1) == "frontier"

    def test_easy_task_picks_cheapest(self, diff: dict[str, float]) -> None:
        assert pick("t2", _matrix(), k=1, success_rate_threshold=0.6, min_samples=1) == "cheap"

    def test_empty_matrix_degrades_to_cheapest(self, diff: dict[str, float]) -> None:
        assert (
            pick(
                "t1",
                {"models": _matrix()["models"], "results": {}},
                k=3,
                success_rate_threshold=0.6,
                min_samples=1,
            )
            == "cheap"
        )

    def test_unlabelled_task_degrades_to_cheapest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "benchmark.routing.strategies.knn_difficulty.difficulty", lambda _t: None
        )
        assert pick("t1", _matrix(), k=3, success_rate_threshold=0.6, min_samples=1) == "cheap"


class TestStrategies:
    def test_knn_difficulty_single_shot_picks_and_reports_judge_cost(
        self, diff: dict[str, float]
    ) -> None:
        s = knnDifficultyStrategy(k=2)
        assert s.select("t5", {}, _matrix()) == "frontier"
        assert s.judge_cost_total == pytest.approx(0.001)

    def test_unlabelled_task_reports_zero_judge_cost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "benchmark.routing.strategies.knn_difficulty.difficulty", lambda _t: None
        )
        monkeypatch.setattr(
            "benchmark.routing.strategies.knn_difficulty.judge_cost", lambda _t: 0.0
        )
        s = knnDifficultyStrategy()
        s.select("t1", {}, _matrix())
        assert s.judge_cost_total == 0.0

    def test_knn_difficulty_cascade_opens_on_the_pick(self, diff: dict[str, float]) -> None:
        s = knnDifficultyCascadeStrategy(k=2)
        rungs = ["cheap", "mid", "frontier"]
        assert s._initial_rank_floor("t5", _matrix(), rungs) == 2
        assert s.judge_cost_total == pytest.approx(0.001)

    def test_band_cascade_opens_on_the_in_band_pick(self, diff: dict[str, float]) -> None:
        s = DifficultyBandCascadeStrategy(min_samples=1)
        rungs = ["cheap", "mid", "frontier"]
        # t3's band is only t3b (level ~3), where mid passes -> floor at mid.
        assert s._initial_rank_floor("t3", _matrix(), rungs) == 1
        assert s.judge_cost_total == pytest.approx(0.001)

    def test_band_cascade_thin_band_escalates_to_strongest(self, diff: dict[str, float]) -> None:
        # The shipped min_samples=3 gate: a band with under 3 observations gives no model
        # enough evidence, so the rule falls through to the strongest model rather than
        # claiming cheap is safe on one sample.
        s = DifficultyBandCascadeStrategy()
        rungs = ["cheap", "mid", "frontier"]
        assert s._initial_rank_floor("t3", _matrix(), rungs) == 2

    def test_cascade_degrade_on_empty_matrix(self, diff: dict[str, float]) -> None:
        s = knnDifficultyCascadeStrategy()
        assert s.select("t1", {}, {"models": _matrix()["models"], "results": {}}) == "cheap"

    def test_empty_rungs_does_not_leak_the_previous_tasks_judge_bill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # SessionCascadeStrategy.select returns early (before _initial_rank_floor) when the
        # matrix prices no rungs; the difficulty override must still charge a FRESH bill for
        # the new task, not the previous one's.
        table = {"t5": 5.0}
        monkeypatch.setattr(
            "benchmark.routing.strategies.knn_difficulty.difficulty", lambda t: table.get(t)
        )
        monkeypatch.setattr(
            "benchmark.routing.strategies.knn_difficulty.judge_cost",
            lambda t: {"t1": 0.002, "t5": 0.001}.get(t, 0.0),
        )
        s = knnDifficultyCascadeStrategy()
        s.select("t5", {}, _matrix())
        assert s.judge_cost_total == pytest.approx(0.001)
        s.select("t1", {}, {"results": {}})
        assert s.judge_cost_total == pytest.approx(0.002)

    def test_judge_bill_is_folded_into_the_summary_billing(self, diff: dict[str, float]) -> None:
        # The row's reported cost must be model cost + the measured judge bill, per task, and
        # the per-task judge bill must be surfaced (the judge_label_cost column is its sum).
        from benchmark.routing import summary

        matrix = {**_matrix(), "tasks": {}}
        tasks = ["t1", "t2", "t3", "t3b", "t4", "t5"]
        decisions, _uns, _att, judge_by_task = summary.evaluate_billed(
            knnDifficultyStrategy(k=2), matrix, tasks
        )
        assert judge_by_task == {t: pytest.approx(0.001) for t in tasks}
        for tid, model, _passed, cost in decisions:
            cell = matrix["results"][tid].get(model)
            cell_cost = float(cell["cost"]) if cell else 0.0
            assert cost == pytest.approx(cell_cost + 0.001)

    def test_names_are_distinct(self, diff: dict[str, float]) -> None:
        names = {
            knnDifficultyStrategy().name,
            knnDifficultyCascadeStrategy().name,
            DifficultyBandCascadeStrategy().name,
        }
        assert len(names) == 3
