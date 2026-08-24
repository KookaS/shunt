# The fingerprint gate lives in server.py's boot-time `_resolve_embedding_trust`
# (server.py:234-267): the configured embedder's fingerprint is compared against the one
# recorded in the outcome store, and the resulting trust verdict is frozen into the engine
# (engine.py:255, consulted at engine.py:355). So the served lifecycle under test is the
# operator flow: trusted → the embedder's space changes → RESTART the app so the gate
# re-runs → every request serves cold-start `stale_embedding_space` (no embed, no kNN) →
# `shunt reindex` re-embeds the corpus and stamps the new fingerprint → RESTART again →
# warm kNN is trusted once more.
# The fingerprint change is triggered by swapping the injected `server_module.Embedder`
# for an embedder with a different `repo` (hence a different fingerprint) and re-booting
# the app against the SAME data dir — the real comparison path at server.py:234-267, never
# a hand-set mismatch. `shunt reindex` runs through the real CLI entrypoint
# (shunt.cli._reindex → OutcomeStore.reindex_corpus) with only the embedder class
# monkeypatched to the fake, exactly as this hermetic suite fakes the LLM upstream.
"""Stale-embedding-space detection + reindex recovery through the SERVED app (hermetic)."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from shunt import cli
from shunt.db.store import OutcomeEvent, OutcomeStore
from shunt.proxy import server as server_module
from shunt.proxy.server import app
from tests.e2e.helpers import CHAT_PATH, chat_body, parse_decision
from tests.fake_embedder import FakeEmbedder

# Cheap model (cold-start default) and the mid model the hard task escalates to.
_CHEAP = "qwen3.7-plus"
_MID = "gpt-5-mini"

# Two task texts, each a cluster of its own vector (FakeEmbedder is deterministic per
# text, so every seed of one text lands on the same point and a query of the SAME text
# lands exactly on the cluster).
TASK_A = "fix the flaky parser test in the checkout module"
TASK_B = "port the legacy cron scheduler onto the new event bus"

# Space A is the suite's stock fake embedder; space B is a different repo, hence a
# different fingerprint (embedding_config.py:35-37) AND a genuinely different vector
# space (every vector shifted by a constant), so a reindex into B is observable at the
# blob level, not just in the fingerprint.
_SPACE_B_REPO = "fake-embedder-768-b"
_SPACE_B_OFFSET = 3.5


class _ShiftedEmbedder(FakeEmbedder):
    """FakeEmbedder in space B: the stock per-text determinism plus a fixed offset, so
    vectors differ from space A while keeping the exact-cluster geometry that makes the
    seeded warm routing deterministic."""

    def __init__(self, *, repo: str = _SPACE_B_REPO) -> None:
        super().__init__(repo=repo)

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(super().embed(text) + _SPACE_B_OFFSET, dtype=np.float32)


# Router env vars the suite controls; mirrored from conftest because this test also pins
# SHUNT_EXPLORATION_ENABLED=0 and boots multiple times with a swapped embedder class.
_CONTROLLED_ENV: tuple[str, ...] = (
    "SHUNT_ROUTER_STRATEGY",
    "SHUNT_WORK_DIR",
    "SHUNT_EXPLORATION_ENABLED",
    "SHUNT_EXPLORE_BUDGET_FRAC",
    "SHUNT_MODEL_CONFIG_PATH",
)


def _boot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    embedder_factory: Callable[[], FakeEmbedder],
) -> TestClient:
    """Boot the served app hermetically with *embedder_factory* injected at server.Embedder."""
    for name in _CONTROLLED_ENV:
        monkeypatch.delenv(name, raising=False)
    # SHUNT_CONFIG_DIR empty + SHUNT_DATA_DIR pinned ⇒ packaged router/embedding config
    # and one shared store across boots (so the corpus + fingerprint persist per test).
    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("SHUNT_EXPLORATION_ENABLED", "0")
    # The stale-space refusal is only observable on a strategy that queries the index; the
    # shipped default (`session_cascade`) never embeds, so it names the kNN one.
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "knn_cascade")
    # Hermetic by construction: dropping the fake injection fails loudly instead of
    # downloading ~600MB (embedder.py:16-19).
    monkeypatch.setenv("SHUNT_DISALLOW_REAL_EMBEDDER", "1")
    monkeypatch.setattr(server_module, "_MODEL_CONFIG_PATH", None)
    # Boot from a non-repo dir so capture/escalation stay inert (no verifier subprocess).
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir(exist_ok=True)
    monkeypatch.chdir(not_a_repo)
    monkeypatch.setattr(server_module, "Embedder", embedder_factory)
    return TestClient(app)


def _post(client: TestClient, text: str, user_agent: str) -> tuple[dict, str, str]:
    """POST a single-turn chat request under its own tool identity.

    A unique User-Agent means a unique session (the session key is tool identity), so the
    model decision is re-made per request rather than locked from a previous one.
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
        embedding=FakeEmbedder().embed(text),
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


def _seed_warm_corpus(store: OutcomeStore) -> None:
    """Seed 28 verified outcomes across two task clusters (test_warm_knn's corpus): task A
    cheap-succeeds, task B is the hard task. Ends cold start; both clusters query warm."""
    specs = [
        *[(f"seed-a-{i:02d}", TASK_A, _CHEAP, 0.01, "success") for i in range(16)],
        *[(f"seed-b-{i:02d}", TASK_B, _CHEAP, 0.01, "failure") for i in range(8)],
        *[(f"seed-bm-{i:02d}", TASK_B, _MID, 0.05, "success") for i in range(4)],
    ]
    for sid, text, model, cost, outcome in specs:
        _seed_outcome(store, session_id=sid, text=text, model=model, cost=cost, outcome=outcome)


class _CountingEmbedder:
    """Delegating embedder that counts ``embed`` calls — proof of the no-embed stale path."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0

    def embed(self, text: str) -> Any:
        self.calls += 1
        return self._inner.embed(text)

    def fingerprint(self) -> dict[str, object]:
        return self._inner.fingerprint()

    def warm(self) -> None:
        self._inner.warm()


def _blob(store: OutcomeStore, sid: str) -> np.ndarray:
    row = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT embedding_blob FROM sessions WHERE session_id = ?", (sid,)
    ).fetchone()
    return np.frombuffer(row["embedding_blob"], dtype=np.float32)


def test_stale_embedding_space_reindex_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mock_acompletion: Any,
) -> None:
    """The served-app lifecycle: trusted → space change (restart re-runs the boot gate) →
    stale_embedding_space with no embed/kNN → `shunt reindex` → trusted again."""
    del mock_acompletion  # autouse; the fake upstream is already wired by the fixture

    # ── Phase 1: trusted space A ────────────────────────────────────────────────
    with _boot(monkeypatch, tmp_path, embedder_factory=FakeEmbedder) as client:
        store: OutcomeStore = client.app.state.outcome_store
        assert store.count_verified_outcomes() == 0

        _seed_warm_corpus(store)
        assert store.count_verified_outcomes() == 28

        # The fresh empty corpus was ADOPTED at boot and stamped with space A's fingerprint,
        # so the boot gate (server.py:234-267) trusts the seeded space.
        assert store.load_embedding_fingerprint() == FakeEmbedder().fingerprint()
        engine = client.app.state.router._engine
        assert engine._trust_neighbors is True  # noqa: SLF001

        _, decision, _ = _post(client, TASK_A, "stale-phase1/0.1")
        # kNN/warm, NOT stale: the seeded cluster routes to the learned cheap pick.
        assert parse_decision(decision) == (_CHEAP, "cheapest_above_threshold")

    # ── Phase 2: the embedder's space changes; a restart re-runs the gate → stale ──
    with _boot(monkeypatch, tmp_path, embedder_factory=_ShiftedEmbedder) as client:
        engine = client.app.state.router._engine
        # server.py:379 re-ran _resolve_embedding_trust: configured B != stored A → refuse.
        assert engine._trust_neighbors is False  # noqa: SLF001

        # Prove the stale path never embeds: instrument the live engine embedder.
        counter = _CountingEmbedder(engine._embedder)
        engine._embedder = counter  # noqa: SLF001

        _, decision, sid = _post(client, TASK_A, "stale-phase2/0.1")
        assert parse_decision(decision) == (_CHEAP, "stale_embedding_space")
        assert counter.calls == 0  # no embed ⇒ no kNN query on the stale path

        prov = json.loads(client.app.state.outcome_store.get_session(sid)["decision_provenance"])
        assert prov["selection_rule_used"] == "stale_embedding_space"
        assert prov["top_k_neighbor_ids"] == []

    # ── Phase 3: `shunt reindex` re-embeds the corpus into space B, fingerprint last ──
    # The CLI builds its own OutcomeStore (same SHUNT_DATA_DIR) and an Embedder — the
    # embedder class is the only thing faked, exactly like the rest of this suite.
    monkeypatch.setattr("shunt.router.embedder.Embedder", _ShiftedEmbedder)
    monkeypatch.setattr(sys, "argv", ["shunt", "reindex"])
    capsys.readouterr()
    cli.main()
    assert "reindex OK" in capsys.readouterr().out

    store = OutcomeStore()
    try:
        # The reindex ordering contract (store.py:690-706): the corpus blobs moved into
        # space B AND the fingerprint advanced to B, so a crash could never leave a
        # trusted-mismatch state. The stored fingerprint is B, not A.
        assert store.load_embedding_fingerprint() == _ShiftedEmbedder().fingerprint()
        assert store.load_embedding_fingerprint() != FakeEmbedder().fingerprint()
        assert np.allclose(_blob(store, "seed-a-00"), _ShiftedEmbedder().embed(TASK_A))
        assert not np.allclose(_blob(store, "seed-a-00"), FakeEmbedder().embed(TASK_A))
    finally:
        store.close()

    # ── Phase 4: a restart after reindex trusts space B again ────────────────────
    with _boot(monkeypatch, tmp_path, embedder_factory=_ShiftedEmbedder) as client:
        engine = client.app.state.router._engine
        assert engine._trust_neighbors is True  # noqa: SLF001

        _, decision, sid = _post(client, TASK_A, "stale-phase4/0.1")
        # Recovered: the re-embedded corpus queries warm in space B.
        assert parse_decision(decision) == (_CHEAP, "cheapest_above_threshold")
        prov = json.loads(client.app.state.outcome_store.get_session(sid)["decision_provenance"])
        assert prov["selection_rule_used"] == "cheapest_above_threshold"
        assert prov["top_k_neighbor_ids"]  # a real kNN query ran over the re-embedded corpus


def test_fingerprint_change_is_detected_at_boot_not_mid_flight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_acompletion: Any,
) -> None:
    """Pin the ACTUAL served behavior: the fingerprint gate is boot-time only.

    The configured vs stored fingerprint is compared ONCE in lifespan (server.py:379) and
    the verdict frozen into the engine (engine.py:255, read at engine.py:355).
    """
    # A fingerprint change with the app still running is therefore NOT detected until the
    # next boot — this test documents that boundary rather than assuming per-request
    # detection (which would require the gate to re-run per request).
    del mock_acompletion  # autouse; the fake upstream is already wired by the fixture

    with _boot(monkeypatch, tmp_path, embedder_factory=FakeEmbedder) as client:
        store: OutcomeStore = client.app.state.outcome_store
        _seed_warm_corpus(store)

        _, decision, _ = _post(client, TASK_A, "stale-mf-1/0.1")
        assert parse_decision(decision) == (_CHEAP, "cheapest_above_threshold")

        # The embedder's space changes WHILE the app is serving (same client, no reboot).
        monkeypatch.setattr(server_module, "Embedder", _ShiftedEmbedder)

        # The served decision is unchanged: trust was frozen at boot and the engine's
        # embedder instance is still the space-A one, so the request still routes warm.
        _, decision, _ = _post(client, TASK_A, "stale-mf-2/0.1")
        assert parse_decision(decision) == (_CHEAP, "cheapest_above_threshold")
