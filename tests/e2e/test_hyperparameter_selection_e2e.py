"""Hyperparameter e2e harness — selection + exploration knobs (prove each is LIVE)."""


# Knob → case matrix:
#   k (router.policy.k)                 k=1 vs k=10
#   success_rate_threshold              0.6 vs 0.9
#   min_samples                         1 vs 99
#   SHUNT_ROUTER_STRATEGY               always_cheap / always_frontier / knn
#   SHUNT_EXPLORATION_ENABLED           0 vs 1
#   SHUNT_EXPLORE_BUDGET_FRAC           0.05 vs 0.4
#   exploration.prior_alpha             1.0 vs 1e9

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.e2e.helpers import (
    chat_body,
    close_session,
    parse_decision,
    post_completion,
    wait_capture_idle,
)
from tests.e2e.hyperparameter_helpers import (
    TASK_A,
    TASK_B,
    cheapest_untested,
    pin_exploration_rng,
    ranked_model,
    reset_cold_start_env,
    seed_exploration_corpus,
    seed_outcome,
    write_policy,
)

_MID = "zai-glm-5.2"
_CHEAP = "deepseek-v4-flash"
_FRONTIER = "claude-fable-5"


def _seed_single_cluster_corpus(client: TestClient, *, n_a: int, n_b: int) -> None:
    """Seed *n_a* zai-glm-5.2 successes on TASK_A and *n_b* on TASK_B (all one model)."""
    store = client.app.state.outcome_store
    for i in range(n_a):
        seed_outcome(
            store,
            session_id=f"seed-a-{i}",
            text=TASK_A,
            model=_MID,
            cost=0.01,
            outcome="success",
        )
    for i in range(n_b):
        seed_outcome(
            store,
            session_id=f"seed-b-{i}",
            text=TASK_B,
            model=_MID,
            cost=0.01,
            outcome="success",
        )


# ── 1. k: the number of neighbours returned changes which models can qualify ──


@pytest.mark.parametrize(
    ("k", "expected_reason"),
    [
        # k=1 returns a single neighbour (zai-glm-5.2) — fewer samples than min_samples=3,
        # so nothing qualifies and the router escalates to the cheapest untested model.
        pytest.param(1, "exploration_untested", id="k=1"),
        # k=10 returns a qualifying neighbourhood of zai-glm-5.2 successes → cheapest wins.
        pytest.param(10, "cheapest_above_threshold", id="k=10"),
    ],
)
def test_k_changes_the_neighbourhood_decision(
    app_factory: Callable[..., TestClient],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    k: int,
    expected_reason: str,
) -> None:
    reset_cold_start_env(monkeypatch)
    write_policy(
        tmp_path / "config",
        mutate=lambda r: (
            r["policy"].update(k=k),
            r["exploration"].update(enabled=False),
        ),
    )
    with app_factory() as client:
        _seed_single_cluster_corpus(client, n_a=16, n_b=8)
        model, reason = parse_decision(
            post_completion(client, chat_body(content=TASK_A)).headers["X-Shunt-Decision"]
        )
        if expected_reason == "exploration_untested":
            assert model == cheapest_untested(client, _MID)
        else:
            assert model == _MID
        assert reason == expected_reason


# ── 2. success_rate_threshold: which models clear the bar changes ────────────


@pytest.mark.parametrize(
    ("threshold", "expected_reason"),
    [
        # 0.75 weighted success clears 0.6 → the cheap model qualifies.
        pytest.param(0.6, "cheapest_above_threshold", id="threshold=0.6"),
        # 0.75 does not clear 0.9 → nothing qualifies, the router escalates.
        pytest.param(0.9, "exploration_untested", id="threshold=0.9"),
    ],
)
def test_success_rate_threshold_changes_eligibility(
    app_factory: Callable[..., TestClient],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    threshold: float,
    expected_reason: str,
) -> None:
    reset_cold_start_env(monkeypatch)
    write_policy(
        tmp_path / "config",
        mutate=lambda r: (
            r["policy"].update(success_rate_threshold=threshold),
            r["exploration"].update(enabled=False),
        ),
    )
    with app_factory() as client:
        store = client.app.state.outcome_store
        # TASK_A: 9 zai-glm-5.2 successes + 3 zai-glm-5.2 failures → weighted success 0.75.
        for i in range(9):
            seed_outcome(
                store,
                session_id=f"sa-{i}",
                text=TASK_A,
                model=_MID,
                cost=0.01,
                outcome="success",
            )
        for i in range(3):
            seed_outcome(
                store,
                session_id=f"fa-{i}",
                text=TASK_A,
                model=_MID,
                cost=0.01,
                outcome="failure",
            )
        # TASK_B: 12 more successes so the corpus ends cold start (>=20 Tier-2).
        for i in range(12):
            seed_outcome(
                store,
                session_id=f"sb-{i}",
                text=TASK_B,
                model=_MID,
                cost=0.01,
                outcome="success",
            )
        model, reason = parse_decision(
            post_completion(client, chat_body(content=TASK_A)).headers["X-Shunt-Decision"]
        )
        if expected_reason == "exploration_untested":
            assert model == cheapest_untested(client, _MID)
        else:
            assert model == _MID
        assert reason == expected_reason


# ── 3. min_samples: the sample-size gate on a model's neighbourhood ──────────


@pytest.mark.parametrize(
    ("min_samples", "expected_reason"),
    [
        pytest.param(1, "cheapest_above_threshold", id="min_samples=1"),
        pytest.param(99, "exploration_untested", id="min_samples=99"),
    ],
)
def test_min_samples_gates_qualification(
    app_factory: Callable[..., TestClient],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    min_samples: int,
    expected_reason: str,
) -> None:
    reset_cold_start_env(monkeypatch)
    write_policy(
        tmp_path / "config",
        mutate=lambda r: (
            r["policy"].update(min_samples=min_samples),
            r["exploration"].update(enabled=False),
        ),
    )
    with app_factory() as client:
        # Exactly 20 zai-glm-5.2 outcomes: just enough to end cold start, and the whole
        # neighbourhood is one zai-glm-5.2 group whose size is *min_samples*'s gate.
        _seed_single_cluster_corpus(client, n_a=6, n_b=14)
        model, reason = parse_decision(
            post_completion(client, chat_body(content=TASK_A)).headers["X-Shunt-Decision"]
        )
        if expected_reason == "exploration_untested":
            assert model == cheapest_untested(client, _MID)
        else:
            assert model == _MID
        assert reason == expected_reason


# ── 4. SHUNT_ROUTER_STRATEGY: the env overlay picks the active strategy ──────


@pytest.mark.parametrize(
    ("strategy", "expected_model", "expected_reason"),
    [
        pytest.param(
            "always_cheap",
            _CHEAP,
            "always_cheap",
            id="strategy=always_cheap",
        ),
        pytest.param(
            "always_frontier",
            _FRONTIER,
            "always_frontier",
            id="strategy=always_frontier",
        ),
        pytest.param(
            "knn_semantic_cascade", _CHEAP, "cold_start", id="strategy=knn_semantic_cascade"
        ),
    ],
)
def test_strategy_env_override_changes_the_served_model(
    app_factory: Callable[..., TestClient],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    expected_model: str,
    expected_reason: str,
) -> None:
    reset_cold_start_env(monkeypatch)
    with app_factory(strategy=strategy) as client:
        model, reason = parse_decision(
            post_completion(client, chat_body(content=TASK_A)).headers["X-Shunt-Decision"]
        )
        # Fixed strategies pick by rank directly; knn cold-starts to the cheap default.
        if strategy == "always_cheap":
            assert model == ranked_model(client, 0)
        elif strategy == "always_frontier":
            assert model == ranked_model(client, -1)
        else:
            assert model == expected_model
        assert reason == expected_reason


# ── 5. SHUNT_EXPLORATION_ENABLED: the exploration reason only appears when on ─


@pytest.mark.parametrize(
    ("enabled", "expected_reason"),
    [
        pytest.param("1", "exploration", id="exploration=enabled"),
        pytest.param("0", "cheapest_above_threshold", id="exploration=disabled"),
    ],
)
def test_exploration_enabled_switch_is_observable_on_the_wire(
    app_factory: Callable[..., TestClient],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enabled: str,
    expected_reason: str,
) -> None:
    reset_cold_start_env(monkeypatch)
    pin_exploration_rng(monkeypatch)
    client = app_factory(repo=None)
    monkeypatch.setenv("SHUNT_EXPLORATION_ENABLED", enabled)
    with client:
        seed_exploration_corpus(client.app.state.outcome_store)
        model, reason = parse_decision(
            post_completion(client, chat_body()).headers["X-Shunt-Decision"]
        )
        if enabled == "1":
            # Seed 15 makes the first decision diverge from greedy (an upshift).
            assert reason == "exploration"
            assert model == _MID
        else:
            assert reason == "cheapest_above_threshold"
            assert model == _CHEAP
        assert reason == expected_reason


# ── 6. SHUNT_EXPLORE_BUDGET_FRAC: the cumulative cap binds the exploration rate ─


@pytest.mark.parametrize(
    ("budget_frac", "expected_explorations"),
    [
        # Tiny: only the bootstrap exploration fits (each upshift books 4.0 of explore
        # spend against a 1.0 baseline, so the 0.05 cap binds at once and never re-opens
        # within 30 decisions).
        pytest.param("0.05", 1, id="budget_frac=tiny"),
        # Shipped 0.4: the cap re-opens once the baseline reaches 10x the booked explore
        # spend, and the pinned sampler diverges again → 3 exploratory decisions in 30.
        pytest.param("0.4", 3, id="budget_frac=default"),
    ],
)
def test_explore_budget_frac_binds_the_number_of_explorations(
    app_factory: Callable[..., TestClient],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget_frac: str,
    expected_explorations: int,
) -> None:
    reset_cold_start_env(monkeypatch)
    # Seed 10 makes the FIRST decision diverge AND diverge again after the default budget
    # re-opens (30 decisions: exactly 1 exploration at frac=0.05, 3 at frac=0.4) — the
    # config that discriminates.
    pin_exploration_rng(monkeypatch, seed=10)
    client = app_factory(repo=None)
    monkeypatch.setenv("SHUNT_EXPLORATION_ENABLED", "1")
    monkeypatch.setenv("SHUNT_EXPLORE_BUDGET_FRAC", budget_frac)
    with client:
        seed_exploration_corpus(client.app.state.outcome_store)
        engine = client.app.state.router._engine
        for _ in range(30):
            resp = post_completion(client, chat_body())
            close_session(client, resp.headers["X-Shunt-Session-Id"])
        wait_capture_idle(client)
        # The cumulative budget counter records exactly the exploratory decisions the cap
        # admitted — the observable that the fraction binds (not just the first sample).
        assert engine._budget.snapshot()["explorations"] == expected_explorations


# ── 7. exploration.prior_alpha: a huge prior makes the sampler near-deterministic ─


@pytest.mark.parametrize(
    ("prior_alpha", "expects_exploration"),
    [
        pytest.param(1.0, True, id="prior_alpha=1.0"),
        pytest.param(1e9, False, id="prior_alpha=1e9"),
    ],
)
def test_prior_alpha_dominates_the_sampler(
    app_factory: Callable[..., TestClient],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior_alpha: float,
    expects_exploration: bool,
) -> None:
    reset_cold_start_env(monkeypatch)
    pin_exploration_rng(monkeypatch)
    write_policy(
        tmp_path / "config",
        mutate=lambda r: r["exploration"].update(prior_alpha=prior_alpha),
    )
    with app_factory(repo=None) as client:
        seed_exploration_corpus(client.app.state.outcome_store)
        reasons: list[str] = []
        for _ in range(6):
            resp = post_completion(client, chat_body())
            reasons.append(parse_decision(resp.headers["X-Shunt-Decision"])[1])
            close_session(client, resp.headers["X-Shunt-Session-Id"])
        wait_capture_idle(client)
        if expects_exploration:
            assert reasons[0] == "exploration"
        else:
            # Beta(1e9,1e9) degenerates every draw to the prior mean 0.5, so the
            # sampler always reproduces the greedy pick (exploration_exploit) and the
            # divergent `exploration` reason never appears.
            assert "exploration" not in reasons
            assert all(reason == "exploration_exploit" for reason in reasons)
