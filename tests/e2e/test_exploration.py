"""End-to-end exploration routing through the REAL served app (fake embedder, mocked
upstream, zero spend)."""

# The loop under test: HTTP request -> engine decide -> (warm corpus) cost-aware
# Thompson layer -> canned upstream response -> session row persisted with provenance.
#
# The engine's exploration preconditions (src/shunt/router/engine.py):
#   * strategy consults neighbours (knn)                     decide() L352
#   * embedding space trusted                                 decide() L355
#   * cold-start INACTIVE (>= 20 effective Tier-2 outcomes)   decide() L363-382
#   * exploration policy enabled + budget.can_explore()       _maybe_explore() L538-545
#   * the neighbourhood yields >= 1 candidate model           _maybe_explore() L548-550
# When those hold, the sampler picks a model; if it diverges from the greedy pick the
# reason token is `exploration` (upshift is always allowed; a downshift becomes
# `conservative_fallback`), and the recorded provenance carries the TS propensity.
#
# Determinism: the engine builds its Thompson sampler with an UNSEEDED
# np.random.default_rng() (engine.py L311-318), so the draw cannot be pinned by config.
# The test pins it by replacing numpy.random.default_rng BEFORE boot with a stand-in
# that seeds unseeded draws to _RNG_SEED while preserving explicit seeds (the
# FakeEmbedder relies on those for per-text vectors). Seed 12 makes the FIRST decision
# on the fixed corpus below explore (found by sweeping seeds against the deterministic
# neighbourhood); the whole path then runs bit-for-bit the same every time.
#
# The second test mirrors the unit budget cap (tests/router/test_engine_exploration.py
# L121-138) at the served level: a tiny explore_budget_frac lets only the bootstrap
# exploration through, then the selection rule takes over.

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Final

import numpy as np
import pytest
from fastapi.testclient import TestClient

from shunt.db.store import OutcomeEvent, OutcomeStore, SessionProvenance
from tests.e2e.helpers import (
    chat_body,
    close_session,
    parse_decision,
    post_completion,
    wait_capture_idle,
)
from tests.fake_embedder import FakeEmbedder

_QWEN: Final[str] = "qwen3.7-plus"
_DS: Final[str] = "deepseek-v4-flash"
_PROMPT: Final[str] = "Fix the build"
_RNG_SEED: Final[int] = 12
# The REAL numpy factory, captured before the test pins the module attribute — the pin
# must delegate to it or it would recurse into itself.
_REAL_DEFAULT_RNG: Final = np.random.default_rng


def _pinned_default_rng(*args: int) -> np.random.Generator:
    """``np.random.default_rng`` stand-in: seed unseeded draws to ``_RNG_SEED``."""
    # The engine's sampler is the only unseeded numpy-draw consumer in the served path
    # (grep: src/shunt has exactly one bare default_rng() call). The FakeEmbedder always
    # passes an explicit seed, so it is untouched by the pin.
    return _REAL_DEFAULT_RNG(args[0] if args else _RNG_SEED)


def _seed_corpus(store: OutcomeStore, embedder: FakeEmbedder) -> None:
    """Write the deterministic verified corpus the kNN neighbourhood reads back."""
    # 24 Tier-2 sessions split between the two models, both near the 0.6 success
    # threshold (qwen 8/12 passes, deepseek 6/12), the pricier at cost 5.0 vs 1.0.
    # This is what lets the Thompson layer sometimes diverge from the greedy pick —
    # and it is the precondition exploration needs: cold-start ends at >= 20 effective
    # Tier-2 outcomes, so the engine reaches the exploration branch at all.
    counter = 0

    def add(session_id: str, model: str, cost: float, outcome: str) -> None:
        nonlocal counter
        prompt = f"seed task {counter} {model} {session_id.rsplit('-', 1)[-1]}"
        store.store_session(
            session_id,
            prompt,
            embedder.embed(prompt),
            model,
            cost,
            {},
            1.0,
            provenance=SessionProvenance(cost_known=True),
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
        add(f"seed-{_QWEN}-{i}", _QWEN, 1.0, "success" if i < 8 else "failure")
    for i in range(12):
        add(f"seed-{_DS}-{i}", _DS, 5.0, "success" if i < 6 else "failure")


def _boot(
    app_factory: Callable[..., TestClient],
    monkeypatch: pytest.MonkeyPatch,
    *,
    budget_frac: str,
) -> TestClient:
    """Boot the served app with exploration enabled + the RNG pinned (not yet entered)."""
    client = app_factory(repo=None)
    monkeypatch.setenv("SHUNT_EXPLORATION_ENABLED", "1")
    monkeypatch.setenv("SHUNT_EXPLORE_BUDGET_FRAC", budget_frac)
    monkeypatch.setattr(np.random, "default_rng", _pinned_default_rng)
    return client


def test_exploratory_decision_is_observable_and_recorded(
    app_factory: Callable[..., TestClient],
    monkeypatch: pytest.MonkeyPatch,
    mock_acompletion: Any,
) -> None:
    """An exploratory decision surfaces on the wire AND is persisted with provenance."""
    del mock_acompletion  # the autouse fixture already mocks the upstream; declared for intent
    client = _boot(app_factory, monkeypatch, budget_frac="0.4")
    with client:
        _seed_corpus(client.app.state.outcome_store, FakeEmbedder())

        resp = post_completion(client, chat_body(content=_PROMPT))
        sid = resp.headers["X-Shunt-Session-Id"]
        model, reason = parse_decision(resp.headers["X-Shunt-Decision"])

        # Observable on the wire: the Thompson layer diverged from greedy and took an
        # upshift to the pricier model — the `exploration` reason token (engine.py
        # L577-588, docs/routing.md reason table).
        assert reason == "exploration"
        assert model == _DS

        # Recorded with provenance: the session row carries the rule, the realised TS
        # propensity (logged for off-policy evaluation) and the downshift flag.
        row = client.app.state.outcome_store.get_session(sid)
        assert row is not None
        prov = json.loads(row["decision_provenance"] or "{}")
        assert prov["selection_rule_used"] == "exploration"
        assert prov["model_chosen"] == _DS
        assert 0.0 < prov["router_propensity"] < 1.0
        assert row["selection_propensity"] == prov["router_propensity"]
        assert prov["downshift"] is False  # an upshift is not gate evidence
        assert len(prov["top_k_neighbor_ids"]) > 0

        close_session(client, sid)
        wait_capture_idle(client)


def test_exploration_budget_cap_binds_at_the_served_level(
    app_factory: Callable[..., TestClient],
    monkeypatch: pytest.MonkeyPatch,
    mock_acompletion: Any,
) -> None:
    """A tiny explore_budget_frac admits the bootstrap exploration, then the selection
    rule takes over and the sampler is never consulted again."""
    del mock_acompletion
    client = _boot(app_factory, monkeypatch, budget_frac="0.05")
    with client:
        _seed_corpus(client.app.state.outcome_store, FakeEmbedder())
        engine = client.app.state.router._engine

        reasons: list[str] = []
        sids: list[str] = []
        for _ in range(8):
            resp = post_completion(client, chat_body(content=_PROMPT))
            sids.append(resp.headers["X-Shunt-Session-Id"])
            reasons.append(parse_decision(resp.headers["X-Shunt-Decision"])[1])
            close_session(client, sids[-1])
        wait_capture_idle(client)

        # The first decision may explore (bootstrap allowance: nothing recorded yet —
        # budget.py can_explore() L76-79), and with the pinned seed it does.
        assert reasons[0] == "exploration"
        # That one upshift (cost 5.0 vs a 1.0 baseline) books 4.0 of explore spend,
        # which already exceeds 0.05 * exploit spend — so the cap trips and every later
        # decision runs the selection rule (cheapest_above_threshold) instead.
        assert all(r == "cheapest_above_threshold" for r in reasons[1:])
        assert engine._budget.snapshot()["explorations"] == 1
        assert engine._budget.can_explore() is False
