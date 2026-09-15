"""Strategy containment: disabled/foreign models never reach a strategy's select().

A registry matrix's ``results`` carries a row for every measured model — including
models measured historically but absent from ``benchmark.yaml``'s enabled list.
Imputation-ON drops them by rebuilding over the enabled ladder; the imputation-OFF
branch must contain them too, or a strategy that reads ``matrix["results"]`` (the
oracle above all) selects a model the run never enabled. Pre-fix, the impute-off
oracle selected ``glm-5.3-flash`` on 40 of the corpus's challenges.
"""

from __future__ import annotations

from typing import Final

import pytest

from benchmark import config
from benchmark.routing.strategies.fixed import AlwaysFrontier
from benchmark.routing.strategies.oracle import Oracle
from benchmark.routing.summary import complete_scored_matrix, evaluate

# A priced registry model deliberately NOT in benchmark.yaml's enabled list.
FOREIGN: Final[str] = "glm-5.3-flash"


def _rank_models() -> list[str]:
    """The real enabled capability ladder, so fixtures build against the same rank."""
    config.load("benchmark/benchmark.yaml")
    return [r.model for r in config.capability_rank().ordered]


ORD: Final[list[str]] = _rank_models()


def _cell(passed: bool, cost: float) -> dict:
    return {"pass": passed, "cost": cost, "real_cost": cost}


def _matrix() -> dict:
    # One COMPLETE challenge (weakest enabled model passes, pinning the crossover, so
    # every higher rung imputes pass) where a FOREIGN model is the cheapest passer —
    # exactly the configuration that makes an unfiltered oracle pick the foreign model.
    return {
        "models": {m: {"input_price": 1.0, "output_price": 1.0} for m in ORD},
        "tasks": {"t1": {}},
        "results": {
            "t1": {ORD[0]: _cell(True, 0.02), FOREIGN: _cell(True, 0.005)},
        },
    }


@pytest.fixture()
def _cfg(monkeypatch: pytest.MonkeyPatch):
    config.load("benchmark/benchmark.yaml")
    enabled = set(config.enabled_models())
    assert FOREIGN not in enabled, "fixture assumption: the foreign model is not enabled"

    def _set(impute_enabled: bool) -> None:
        monkeypatch.setattr(config, "impute_config", lambda: {"enabled": impute_enabled})

    return _set


@pytest.mark.parametrize("impute_enabled", [True, False])
def test_foreign_model_dropped_from_scored_matrix(_cfg, impute_enabled: bool) -> None:
    _cfg(impute_enabled)
    completed, _ = complete_scored_matrix(_matrix())
    for cells in completed["results"].values():
        assert FOREIGN not in cells


@pytest.mark.parametrize("impute_enabled", [True, False])
def test_oracle_never_returns_a_non_enabled_model(_cfg, impute_enabled: bool) -> None:
    _cfg(impute_enabled)
    completed, _ = complete_scored_matrix(_matrix())
    decisions, _unscorable = evaluate(Oracle(), completed, list(completed["results"].keys()))
    enabled = set(config.enabled_models())
    for _tid, model, _passed, _cost in decisions:
        assert model == "" or model in enabled


@pytest.mark.parametrize("impute_enabled", [True, False])
def test_real_corpus_strategy_selections_are_enabled_models(_cfg, impute_enabled: bool) -> None:
    """End-to-end: on the committed corpus, no strategy selects a non-enabled model."""
    _cfg(impute_enabled)
    matrix = config.load_matrix()
    completed, _ = complete_scored_matrix(matrix)
    tasks = list(completed["results"].keys())
    enabled = set(config.enabled_models())
    for strategy in (Oracle(), AlwaysFrontier()):
        decisions, _unscorable = evaluate(strategy, completed, tasks)
        for _tid, model, _passed, _cost in decisions:
            assert model == "" or model in enabled


@pytest.mark.parametrize("impute_enabled", [True, False])
def test_synthetic_non_registry_matrix_is_untouched(_cfg, impute_enabled: bool) -> None:
    """A test slice whose models are not registry models is never filtered as foreign."""
    _cfg(impute_enabled)
    matrix = {
        "models": {"mock-a": {"input_price": 1.0, "output_price": 1.0}},
        "tasks": {"t1": {}},
        "results": {"t1": {"mock-a": _cell(True, 0.5), "mock-b": _cell(False, 0.1)}},
    }
    out, _ = complete_scored_matrix(matrix)
    if impute_enabled:
        # No registry model present -> the enabled branch also returns the slice raw.
        assert out == matrix
    else:
        # No enabled model present -> the off-branch containment guard does not fire.
        assert out == matrix
