"""compute_strategy_rows routes through impute when enabled (equal coverage), and
only COMPLETE challenges (an established crossover) survive to be scored."""

from __future__ import annotations

from typing import Final

import pytest

from benchmark import config
from benchmark.routing.strategies.fixed import AlwaysCheap, AlwaysFrontier
from benchmark.routing.summary import complete_scored_matrix, compute_strategy_rows


def _rank_models() -> list[str]:
    """The real enabled capability ladder (weakest -> strongest), so fixtures build
    challenges whose crossover is COMPLETE against the SAME rank impute uses."""
    config.load("benchmark/benchmark.yaml")
    return [r.model for r in config.capability_rank().ordered]


ORD: Final[list[str]] = _rank_models()


def _cell(passed: bool, cost: float) -> dict:
    return {"pass": passed, "cost": cost, "real_cost": cost}


def _models_meta() -> dict:
    # Price ascending by rank index so AlwaysCheap resolves to ORD[0] and AlwaysFrontier
    # to ORD[-1] — both always present after equal-coverage completion.
    return {
        m: {"input_price": float(i + 1), "output_price": float(i + 1)} for i, m in enumerate(ORD)
    }


def _matrix() -> dict:
    # Three COMPLETE challenges (no UNKNOWN band): t1 crosses at the weakest tier, t2 at the
    # second, t3 is fully observed (frontier measured) with only the top tier passing. All
    # tiers ABOVE a task's crossover impute pass; every tier below is a real observed fail.
    return {
        "models": _models_meta(),
        "tasks": {"t1": {}, "t2": {}, "t3": {}},
        "results": {
            "t1": {ORD[0]: _cell(True, 0.01)},  # weakest passes -> tau = ORD[0]
            "t2": {ORD[0]: _cell(False, 0.01), ORD[1]: _cell(True, 0.20)},  # tau = ORD[1]
            # Fully observed, only the frontier passes -> complete, tau = ORD[-1].
            "t3": {
                **{m: _cell(False, 0.02 * (i + 1)) for i, m in enumerate(ORD[:-1])},
                ORD[-1]: _cell(True, 0.60),
            },
        },
    }


@pytest.fixture()
def _cfg(monkeypatch: pytest.MonkeyPatch):
    config.load("benchmark/benchmark.yaml")

    def _set(enabled: bool) -> None:
        monkeypatch.setattr(config, "impute_config", lambda: {"enabled": enabled})

    return _set


def test_impute_enabled_gives_equal_coverage(_cfg) -> None:
    _cfg(True)
    matrix = _matrix()
    tasks = ["t1", "t2", "t3"]
    rows = compute_strategy_rows(
        matrix, tasks, [AlwaysCheap(), AlwaysFrontier()], bootstrap=50, seed=1
    )
    n_tasks = {r["strategy"]: r["n_tasks"] for r in rows}
    # Every strategy scores the SAME task count — the whole point of imputation.
    assert n_tasks["Always-Cheap"] == n_tasks["Always-Frontier"] == 3
    assert all(r["n_unscorable"] == 0 for r in rows)


def test_impute_disabled_reproduces_raw_asymmetry(_cfg) -> None:
    _cfg(False)
    matrix = _matrix()
    tasks = ["t1", "t2", "t3"]
    rows = compute_strategy_rows(
        matrix, tasks, [AlwaysCheap(), AlwaysFrontier()], bootstrap=50, seed=1
    )
    row = {r["strategy"]: r for r in rows}
    # Raw coverage: the frontier is only measured on t3, so 2 tasks are unscorable.
    assert row["Always-Frontier"]["n_tasks"] == 1
    assert row["Always-Frontier"]["n_unscorable"] == 2
    assert row["Always-Cheap"]["n_tasks"] == 3


def test_incomplete_challenge_excluded_from_scoring(_cfg) -> None:
    # A challenge with an UNKNOWN band still open (a tier below the observed pass never run)
    # has no established crossover, so it is excluded from analysis ENTIRELY, not merely
    # left with UNKNOWN cells. ORD[1] passes but ORD[0] (weaker) was never observed.
    _cfg(True)
    matrix = _matrix()
    matrix["tasks"]["gap"] = {}
    matrix["results"]["gap"] = {ORD[1]: _cell(True, 0.20)}  # crossover not pinned from below
    completed, im = complete_scored_matrix(matrix)
    assert im is not None
    assert "gap" in im.incomplete and "gap" not in im.complete
    assert "gap" not in completed["results"]  # excluded entirely
    assert {"t1", "t2", "t3"} <= set(completed["results"])  # the complete ones survive


def test_censored_top_tier_challenge_excluded_from_scoring(_cfg) -> None:
    # Every tier below the frontier genuinely fails, but the frontier merely ran out of steps
    # (censored). Old semantics scored this as a complete all-fail; now the censored top rung is
    # a non-observation, the crossover is unestablished, and the challenge is excluded entirely.
    _cfg(True)
    matrix = _matrix()
    matrix["tasks"]["cens"] = {}
    below = {m: _cell(False, 0.02 * (i + 1)) for i, m in enumerate(ORD[:-1])}
    matrix["results"]["cens"] = {
        **below,
        ORD[-1]: {"pass": False, "cost": 0.6, "real_cost": 0.6, "stop_reason": "step_limit"},
    }
    completed, im = complete_scored_matrix(matrix)
    assert im is not None
    assert "cens" in im.incomplete and "cens" not in im.complete
    assert "cens" not in completed["results"]  # excluded entirely (unknown, not complete-all-fail)


def test_completed_matrix_flags_and_never_mutates_input(_cfg) -> None:
    _cfg(True)
    matrix = _matrix()
    before = {tid: dict(cells) for tid, cells in matrix["results"].items()}
    completed, im = complete_scored_matrix(matrix)
    assert im is not None
    assert matrix["results"] == before  # the source matrix is not mutated in place
    # Imputed cells are flagged; real cells are not.
    assert completed["results"]["t1"][ORD[-1]]["imputed"] is True  # frontier imputed on t1
    assert completed["results"]["t1"][ORD[0]]["imputed"] is False  # weakest is the real pass


def test_drop_unsolvable_removes_dead_tasks(_cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    config.load("benchmark/benchmark.yaml")
    monkeypatch.setattr(config, "impute_config", lambda: {"enabled": True, "drop_unsolvable": True})
    matrix = _matrix()
    # Fully observed, every tier fails -> COMPLETE but unsolvable (tau None): kept by the
    # complete-only filter, then removed by drop_unsolvable.
    matrix["tasks"]["dead"] = {}
    matrix["results"]["dead"] = {m: _cell(False, 0.02 * (i + 1)) for i, m in enumerate(ORD)}
    completed, im = complete_scored_matrix(matrix)
    assert im is not None
    assert "dead" in im.complete and im.tau["dead"] is None
    assert "dead" not in completed["results"]  # dropped under drop_unsolvable
