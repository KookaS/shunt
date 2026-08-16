"""FIX A: the paired router-vs-frontier contrast aligns on the shared scorable set."""

from __future__ import annotations

import pytest

from benchmark.routing import run_eval


class _Fake:
    """Minimal Strategy stub that always routes to one fixed model."""

    def __init__(self, name: str, model: str) -> None:
        self._name = name
        self._model = model

    @property
    def name(self) -> str:
        return self._name

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        return self._model


def _matrix() -> dict:
    # Router 'R' unmeasured on t4; frontier 'F' unmeasured on t1 -> scorable sets diverge.
    return {
        "tasks": {},
        "results": {
            "t1": {"R": {"pass": True, "cost": 1.0}},
            "t2": {"R": {"pass": True, "cost": 1.0}, "F": {"pass": True, "cost": 5.0}},
            "t3": {"R": {"pass": True, "cost": 1.0}, "F": {"pass": False, "cost": 5.0}},
            "t4": {"F": {"pass": True, "cost": 5.0}},
        },
    }


def test_paired_delta_uses_intersection_not_raw_subtraction() -> None:
    router, frontier = _Fake("kNN", "R"), _Fake("Always-Frontier", "F")
    res = run_eval._paired_delta(_matrix(), ["t1", "t2", "t3", "t4"], router, frontier, seed=1)
    assert res is not None
    # Shared scorable set is {t2, t3} only (t1 lacks F, t4 lacks R).
    assert res["n"] == 2
    # On the shared set: router passes both, frontier passes t2 fails t3 -> +50pp.
    assert res["delta"] == pytest.approx(50.0)
    # A raw AvgPerf% subtraction (router 100% over 3 vs frontier 66.7% over 3) would give
    # +33.3pp — asserting 50.0 proves alignment on the intersection, not raw denominators.


def test_paired_delta_none_when_no_shared_task() -> None:
    router, frontier = _Fake("kNN", "R"), _Fake("Always-Frontier", "F")
    disjoint = {
        "tasks": {},
        "results": {
            "t1": {"R": {"pass": True, "cost": 1.0}},
            "t2": {"F": {"pass": True, "cost": 5.0}},
        },
    }
    assert run_eval._paired_delta(disjoint, ["t1", "t2"], router, frontier, seed=1) is None


def test_paired_bootstrap_ci_brackets_point_estimate() -> None:
    lo, hi = run_eval._paired_bootstrap_ci([1, 1, 0, 1, 0, 1, 1, 0], seed=7)
    assert lo <= 62.5 <= hi  # mean of the diffs (5/8) in pp


def test_pick_router_picks_the_best_live_router_not_the_best_row() -> None:
    # kNN-cascade outranks every live candidate on Reward and is still not the answer:
    # the router named in the headline has to be one `router.strategy` can be set to.
    rows = [
        {"strategy": "Oracle", "Reward": 9.0, "n_tasks": 5},
        {"strategy": "Always-Frontier", "Reward": 8.0, "n_tasks": 5},
        {"strategy": "kNN-cascade", "Reward": 7.0, "n_tasks": 5},
        {"strategy": "kNN", "Reward": 6.5, "n_tasks": 5},
        {"strategy": "Always-Cheap", "Reward": 6.0, "n_tasks": 5},
    ]
    assert run_eval._pick_router(rows) == "kNN"


def test_pick_router_excludes_the_frontier_baseline_even_though_it_is_live() -> None:
    rows = [
        {"strategy": "Always-Frontier", "Reward": 8.0, "n_tasks": 5},
        {"strategy": "Always-Cheap", "Reward": 6.0, "n_tasks": 5},
    ]
    assert run_eval._pick_router(rows) == "Always-Cheap"
