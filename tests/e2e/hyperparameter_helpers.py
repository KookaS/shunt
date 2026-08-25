"""Shared machinery for the hyperparameter e2e harness (boot + seed + RNG-pin)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from shunt.db.store import OutcomeEvent, OutcomeStore
from shunt.models.config import strict_yaml_load
from shunt.router.policy import packaged_policy_path
from tests.fake_embedder import FakeEmbedder

# Two task texts, each a deterministic cluster in the fake embedding space (see
# tests/e2e/test_warm_knn.py): a query of one text lands at distance ~0 on its own
# cluster and ~1 on the other, so the kNN neighbourhood split is exact.
TASK_A = "fix the flaky parser test in the checkout module"
TASK_B = "port the legacy cron scheduler onto the new event bus"

# One deterministic embedder shared by every seed; identical to the instance the
# router computes its query embeddings with, so seeded vectors and query vectors
# live in the same space.
EMBEDDER = FakeEmbedder()

# The exploration RNG pin (mirrors tests/e2e/test_exploration.py): the engine builds
# its Thompson sampler with an unseeded ``np.random.default_rng()``, so the tests pin
# unseeded draws to this seed while preserving explicit seeds (the FakeEmbedder relies
# on those for per-text vectors). Seed 15 makes the first decision on the exploration
# corpus below explore; the whole path then runs bit-for-bit the same every time.
_RNG_SEED = 15
_REAL_DEFAULT_RNG = np.random.default_rng


def write_policy(config_dir: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    """Write a copy of the packaged router.yaml, mutated, into *config_dir*.

    ``$SHUNT_CONFIG_DIR/router.yaml`` REPLACES the packaged file wholesale (no merge),
    so *mutate* is applied to a full copy of the packaged policy, never a fragment.
    """
    data: dict[str, Any] = strict_yaml_load(packaged_policy_path().read_text())
    mutate(data["router"])
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "router.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def pin_exploration_rng(monkeypatch: pytest.MonkeyPatch, seed: int = _RNG_SEED) -> None:
    """Pin unseeded ``np.random.default_rng`` draws to *seed* (delegating explicit seeds)."""
    # The engine's sampler is the only unseeded numpy-draw consumer in the served path;
    # the FakeEmbedder always passes an explicit seed, so it is untouched by the pin.

    def _pinned(*args: int) -> np.random.Generator:
        return _REAL_DEFAULT_RNG(args[0] if args else seed)

    monkeypatch.setattr(np.random, "default_rng", _pinned)


def seed_outcome(
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
        embedding=EMBEDDER.embed(text),
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
    sig = f"seed-{session_id}"
    # A wire Tier-1 prior corroborated by a verified Tier-2 label — how the real capture
    # path records outcomes, so the materialized view + the kNN index both see Tier-2.
    store.append_outcome_event(
        OutcomeEvent(
            session_id=session_id,
            tier=1,
            source="wire_tier1",
            outcome=outcome,
            confidence=1.0,
            run_signature=f"{sig}-t1",
        )
    )
    store.append_outcome_event(
        OutcomeEvent(
            session_id=session_id,
            tier=2,
            source="auto_tier2",
            outcome=outcome,
            confidence=1.0,
            run_signature=f"{sig}-t2",
        )
    )


def seed_exploration_corpus(store: OutcomeStore) -> None:
    """Write the deterministic verified corpus the exploration knobs read back."""
    # Mirrors tests/e2e/test_exploration.py: 24 Tier-2 sessions split between the two
    # live-pool models, both near the 0.6 success threshold (deepseek 9/12 passes,
    # zai-glm-5.2 5/12), the pricier at cost 5.0 vs 1.0 — the precondition that lets the
    # Thompson layer sometimes diverge from the greedy pick and end cold start (>=20
    # effective Tier-2).
    counter = 0

    def add(session_id: str, model: str, cost: float, outcome: str) -> None:
        nonlocal counter
        prompt = f"seed task {counter} {model} {session_id.rsplit('-', 1)[-1]}"
        store.store_session(
            session_id,
            prompt,
            EMBEDDER.embed(prompt),
            model,
            cost,
            {},
            1.0,
        )
        store.append_outcome_event(
            OutcomeEvent(
                session_id=session_id,
                tier=2,
                source="auto_tier2",
                outcome=outcome,
                confidence=1.0,
                run_signature=f"sig-{session_id}",
            )
        )
        counter += 1

    for i in range(12):
        add(f"seed-ds-{i}", "deepseek-v4-flash", 1.0, "success" if i < 9 else "failure")
    for i in range(12):
        add(f"seed-glm-{i}", "zai-glm-5.2", 5.0, "success" if i < 5 else "failure")


def reset_cold_start_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the cold-start threshold env vars so a boot uses the packaged defaults."""
    monkeypatch.delenv("SHUNT_COLD_START_THRESHOLD_TIER2", raising=False)
    monkeypatch.delenv("SHUNT_COLD_START_THRESHOLD_TIER1", raising=False)


def refit_cadence(client: TestClient) -> int:
    """The ``router.refit.every_n_outcomes`` cadence the served app wired its scheduler with."""
    worker = client.app.state.session_manager._verifier_callback.__self__
    scheduler = worker._coordinator._refit_scheduler
    return scheduler._every_n


def ranked_model(client: TestClient, index: int) -> str:
    """The *index*-th model in the live pool's capability rank order (0 = cheapest)."""
    return client.app.state.model_pool.ranked_models()[index].name


def cheapest_untested(client: TestClient, tested: str) -> str:
    """The cheapest live model other than *tested* — the escalation rule's pick."""
    return next(m.name for m in client.app.state.model_pool.ranked_models() if m.name != tested)
