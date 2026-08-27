"""Warm kNN end-to-end through the assembled router (hermetic).

The Docker/e2e harness only exercised cold-start; this test seeds a store through the
served app and asserts a request routes to a kNN (non-cold-start) decision.
"""

# Seeding note (the spec's committed-vectors contingency): the shipped routing corpus
# lives in the real jina-code embedding space; the fake embedder here produces
# deterministic 768-d standard-normal vectors, so the two spaces cannot share an index
# even though the dims match. Per the contingency the index is seeded fresh through the
# store API in the fake embedder's space — the same space the router's own queries are
# computed in — so the assembled path (OutcomeStore -> HNSW -> OutcomeIndexAdapter ->
# engine kNN query -> SelectionRule -> provenance) is exercised for real; only the
# vectors themselves are synthetic.

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from shunt.db.store import OutcomeEvent, OutcomeStore
from shunt.proxy import server as server_module
from shunt.proxy.server import app
from tests.e2e.helpers import CHAT_PATH, chat_body, parse_decision
from tests.fake_embedder import FakeEmbedder

# Cheap model (cold-start default) and the mid model the hard task escalates to.
# Both must be in the live pool (router.yaml models:) — the pool now spans
# deepseek-v4-flash -> zai-glm-5.2 -> kimi-k3 after the dominated models were removed.
_CHEAP = "deepseek-v4-flash"
_MID = "zai-glm-5.2"

# Two task texts, each a cluster of its own vector (FakeEmbedder is deterministic per
# text, so every seed of one text lands on the same point and a query of the SAME text
# lands exactly on the cluster).
TASK_A = "fix the flaky parser test in the checkout module"
TASK_B = "port the legacy cron scheduler onto the new event bus"

# Router env vars the conftest suite controls; mirrored here because this test also
# pins SHUNT_EXPLORATION_ENABLED=0, which app_factory's knobs cannot express.
_CONTROLLED_ENV: tuple[str, ...] = (
    "SHUNT_ROUTER_STRATEGY",
    "SHUNT_WORK_DIR",
    "SHUNT_EXPLORATION_ENABLED",
    "SHUNT_EXPLORE_BUDGET_FRAC",
    "SHUNT_MODEL_CONFIG_PATH",
)

# One deterministic embedder shared by the seeds; identical (deterministic) to the
# instance the router computes its query embeddings with, so seeded vectors and query
# vectors live in the same space.
_EMBEDDER = FakeEmbedder()


@pytest.fixture
def warm_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The served app booted hermetically with cost-aware exploration OFF.

    The packaged router.yaml ships exploration on at budget 0.4, which would pre-empt the
    kNN SelectionRule with a random reason and make the token assertion flaky.
    """
    for name in _CONTROLLED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("SHUNT_EXPLORATION_ENABLED", "0")
    # The shipped default is the CHEAP-START cascade, which never queries the neighbourhood.
    # This test is about the kNN path, so it names the strategy that runs one.
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "knn_semantic_cascade")
    # SHUNT_DISALLOW_REAL_EMBEDDER keeps the test hermetic by construction: dropping the
    # FakeEmbedder injection below would fail loudly instead of downloading ~600MB.
    monkeypatch.setenv("SHUNT_DISALLOW_REAL_EMBEDDER", "1")
    monkeypatch.setattr(server_module, "_MODEL_CONFIG_PATH", None)
    # Boot from a non-repo dir so capture/escalation stay inert: no work_dir means no
    # verifier subprocess, no auto-escalation — the routing path alone is under test.
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir(exist_ok=True)
    monkeypatch.chdir(not_a_repo)
    monkeypatch.setattr(server_module, "Embedder", FakeEmbedder)
    with TestClient(app) as client:
        yield client


def _post(client: TestClient, text: str, user_agent: str) -> tuple[dict, str, str]:
    """POST a single-turn chat request under its own tool identity.

    Returns ``(json_body, decision_header, session_id)``.
    """
    resp = client.post(
        CHAT_PATH,
        json=chat_body(content=text),
        headers={"User-Agent": user_agent},
    )
    assert resp.status_code == 200, resp.text
    return resp.json(), resp.headers["X-Shunt-Decision"], resp.headers["X-Shunt-Session-Id"]


def _seed_outcome(
    store: OutcomeStore,
    *,
    session_id: str,
    text: str,
    model: str,
    cost: float,
    outcome: str,
) -> None:
    """One verified (Tier-2) session through the store's own API — the same seam the
    served app's capture path writes through (store_session + append_outcome_event)."""
    store.store_session(
        session_id=session_id,
        prompt_text=text,
        embedding=_EMBEDDER.embed(text),
        model_chosen=model,
        cost=cost,
        cache_stats={},
        duration=1.0,
        decision_provenance={
            "model_chosen": model,
            "selection_rule_used": "cold_start",
            "top_k_neighbor_ids": [],
            "neighbor_confidence_scores": [],
            "fallback_chain_triggered": False,
            "rank_escalation_reason": None,
            "router_propensity": 1.0,
            "candidate_model_scores": {},
            "downshift": False,
        },
    )
    # A wire Tier-1 prior corroborated by a verified Tier-2 label — how the real capture
    # path records outcomes, so the materialized view + the kNN index both see Tier-2.
    sig = f"seed-{session_id}"
    store.append_outcome_event(
        OutcomeEvent(
            session_id=session_id,
            tier=1,
            source="wire_tier1",
            outcome=outcome,
            confidence=0.9,
            run_signature=f"{sig}-t1",
        )
    )
    store.append_outcome_event(
        OutcomeEvent(
            session_id=session_id,
            tier=2,
            source="auto_tier2",
            outcome=outcome,
            confidence=0.9,
            run_signature=f"{sig}-t2",
        )
    )


def test_warm_knn_routes_through_the_assembled_router(warm_app: TestClient) -> None:
    """The warm kNN loop end to end: a fresh cold app, then a seeded index, routes a
    request to a LEARNED decision (reason=cheapest_above_threshold) with provenance —
    and the cheap-vs-hard tasks land on DIFFERENT models."""
    client = warm_app
    store: OutcomeStore = client.app.state.outcome_store
    assert store.count_verified_outcomes() == 0  # fresh, hermetic store per test

    # Cold path is not vacuous: before ANY seed, the same request cold-starts.
    _, cold_header, _ = _post(client, TASK_A, "warm-knn-cold/0.1")
    assert parse_decision(cold_header) == (_CHEAP, "cold_start")

    # Seed 28 verified outcomes across two task clusters. Task A is cheap-succeeds (16
    # deepseek successes); task B is the HARD task (8 deepseek failures + 4 zai-glm-5.2
    # successes).
    # All seeds of one text share a vector, so a query of that exact text is distance-0 to
    # its own cluster and ~1 (cosine) to the other — the neighborhood split is exact.
    seed_ids: list[str] = []
    specs = [
        *[(f"seed-a-{i:02d}", TASK_A, _CHEAP, 0.01, "success") for i in range(16)],
        *[(f"seed-b-{i:02d}", TASK_B, _CHEAP, 0.01, "failure") for i in range(8)],
        *[(f"seed-bm-{i:02d}", TASK_B, _MID, 0.05, "success") for i in range(4)],
    ]
    for sid, text, model, cost, outcome in specs:
        _seed_outcome(store, session_id=sid, text=text, model=model, cost=cost, outcome=outcome)
        seed_ids.append(sid)

    # The seeded corpus ends cold start (>=20 verified outcomes, uniform confidence => n_e
    # equals the raw count) — the engine must now query kNN, not default.
    assert store.count_verified_outcomes() == len(seed_ids) == 28
    engine = client.app.state.router._engine
    index = engine._outcome_index
    assert index.effective_tier2() >= 20
    assert not engine._cold_start_strategy.is_active_effective(
        index.effective_labeled(), index.effective_tier2()
    )

    # Near the cheap-succeeds cluster: deepseek-v4-flash is eligible and cheapest -> kNN pick.
    _, warm_header_a, sid_a = _post(client, TASK_A, "warm-knn-a/0.1")
    assert parse_decision(warm_header_a) == (_CHEAP, "cheapest_above_threshold")
    # Near the HARD cluster the cheap model is ineligible (0% success) and zai-glm-5.2
    # wins -> the router learned a DIFFERENT model for a different task.
    _, warm_header_b, sid_b = _post(client, TASK_B, "warm-knn-b/0.1")
    assert parse_decision(warm_header_b) == (_MID, "cheapest_above_threshold")

    # Provenance: the persisted session rows record the warm decision, the model, and a
    # real (non-empty) neighborhood drawn from the seeded corpus.
    row_a = client.app.state.outcome_store.get_session(sid_a)
    prov_a = json.loads(row_a["decision_provenance"])
    assert prov_a["selection_rule_used"] == "cheapest_above_threshold"
    assert prov_a["model_chosen"] == _CHEAP
    assert prov_a["top_k_neighbor_ids"]
    assert set(prov_a["top_k_neighbor_ids"]) <= set(seed_ids)

    row_b = client.app.state.outcome_store.get_session(sid_b)
    prov_b = json.loads(row_b["decision_provenance"])
    assert prov_b["selection_rule_used"] == "cheapest_above_threshold"
    assert prov_b["model_chosen"] == _MID
    assert set(prov_b["top_k_neighbor_ids"]) <= set(seed_ids)
