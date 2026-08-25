# Sequential multi-turn inside one session is already e2e-tested
# (tests/e2e/test_routing_escalation.py); TRUE concurrency was unit-only
# (tests/router/test_engine_concurrency.py). This file drives the REAL served app
# (FakeEmbedder + canned upstream via the shared conftest) with concurrent multi-turn
# sessions and asserts the cache-safety spine under load:
#   * per-session decision lock honored — one model per session across every turn;
#   * no cross-session model leak — concurrent sessions never swap decisions;
#   * the store records each session's provenance correctly under concurrency.
# Driver: httpx.AsyncClient over ASGITransport with the FastAPI lifespan run manually.
# Production runs ONE event loop whose async handlers interleave and offload the blocking
# decision to a threadpool; ASGITransport reproduces exactly that model, so the
# concurrency exercised here is the concurrency the product actually experiences.
# (FastAPI's sync TestClient instead spins up a fresh portal + event loop PER REQUEST —
# more parallelism than production, not the shape under test.) A slow upstream sleep
# makes the overlap real and measurable.
# Determinism: the outcome corpus is pre-seeded (hermetic) with two prompt clusters
# labelled to two distinct models, so concurrent sessions with different prompts route to
# different models — the leak-detection signal. The clusters use identical prompt text,
# so live embeddings land at distance 0 on their cluster and the selection rule picks
# that cluster's model deterministically.
"""Concurrent multi-turn sessions hammering the SERVED app (hermetic, no spend)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from shunt.db.store import OutcomeStore
from tests.e2e.helpers import CHAT_PATH, chat_body, make_fake_acompletion, parse_decision
from tests.fake_embedder import FakeEmbedder

_N_SESSIONS: Final[int] = 8
_TURNS: Final[int] = 3
_SEEDS_PER_CLUSTER: Final[int] = 3  # >= SelectionRule.min_samples (3) so the cluster is eligible
_UPSTREAM_DELAY_S: Final[float] = 0.05
_TURN_COST_USD: Final[float] = 0.001  # the canned upstream's reported usage.cost per turn
_BUDGET_USD: Final[float] = 0.01

_CLUSTER_A_PROMPT: Final[str] = "deploy the api gateway to production"
_CLUSTER_B_PROMPT: Final[str] = "fix the flaky database migration test"
_CLUSTER_A_MODEL: Final[str] = "deepseek-v4-flash"
_CLUSTER_B_MODEL: Final[str] = "zai-glm-5.2"

_CLUSTERS: Final[tuple[tuple[str, str], ...]] = (
    (_CLUSTER_A_PROMPT, _CLUSTER_A_MODEL),
    (_CLUSTER_B_PROMPT, _CLUSTER_B_MODEL),
)

_SEED_SID_PREFIX: Final[str] = "seed-"


class _InFlight:
    """Max number of upstream calls concurrently in flight (overlap proof).

    The upstream await runs on the app's single event loop, so these counters
    are touched by one thread only — no lock needed.
    """

    def __init__(self) -> None:
        self._current = 0
        self.max = 0

    def enter(self) -> None:
        self._current += 1
        self.max = max(self.max, self._current)

    def exit(self) -> None:
        self._current -= 1


def _arm_slow_upstream(
    mock_acompletion: Any, inflight: _InFlight, *, delay: float = _UPSTREAM_DELAY_S
) -> None:
    """Give the mocked upstream a bounded latency so requests genuinely overlap."""
    canned = make_fake_acompletion()

    async def _slow(config: Any, **kwargs: Any) -> Any:
        inflight.enter()
        try:
            await asyncio.sleep(delay)
            return canned(config, **kwargs)
        finally:
            inflight.exit()

    mock_acompletion.side_effect = _slow


def _seed_corpus(db_path: Path) -> None:
    """Label two prompt clusters to two distinct models in a fresh outcome store."""
    embedder = FakeEmbedder()
    store = OutcomeStore(db_path=str(db_path))
    for cluster, (prompt, model) in enumerate(_CLUSTERS):
        for k in range(_SEEDS_PER_CLUSTER):
            sid = f"{_SEED_SID_PREFIX}c{cluster}-{k}"
            store.store_session(
                session_id=sid,
                prompt_text=prompt,
                embedding=embedder.embed(prompt),
                model_chosen=model,
                cost=1.0,
                cache_stats={},
                duration=0.0,
            )
            store.store_outcome(
                session_id=sid,
                tier1_outcome="success",
                tier1_confidence=1.0,
                tier2_outcome="success",
                tier2_confidence=1.0,
                aggregated_confidence=1.0,
            )
    # The stored fingerprint must match the configured FakeEmbedder's, or the app
    # refuses the seeded neighbours as a foreign embedding space (cold-start).
    store.save_embedding_fingerprint(embedder.fingerprint())
    store.close()


@asynccontextmanager
async def _served_app(
    app_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    budget_usd: float | None = None,
) -> AsyncIterator[Any]:
    """Boot the real served app (env + FakeEmbedder via app_factory) for concurrent driving."""
    client = app_factory(repo=None)  # manual-only: escalation/capture stay inert, no subprocess
    app = client.app
    # Determinism: no Thompson exploration, and kNN active from the first turn (seeded corpus).
    monkeypatch.setenv("SHUNT_EXPLORATION_ENABLED", "0")
    monkeypatch.setenv("SHUNT_COLD_START_THRESHOLD_TIER2", "0")
    if budget_usd is not None:
        config_dir = tmp_path / "config"
        config_dir.mkdir(exist_ok=True)
        (config_dir / "router.yaml").write_text(
            f"router:\n  budget:\n    max_spend_usd: {budget_usd}\n"
        )
    _seed_corpus(tmp_path / "data" / "outcomes.db")
    async with app.router.lifespan_context(app):
        yield app


def _session_plan(n: int) -> tuple[list[str], list[str], list[str]]:
    """Deterministic per-session plan: prompt, UA (the session identity), expected model."""
    prompts: list[str] = []
    expected: list[str] = []
    for i in range(n):
        prompt, model = _CLUSTERS[i % len(_CLUSTERS)]
        prompts.append(prompt)
        expected.append(model)
    user_agents = [f"shunt-e2e/concurrent-{i:02d}" for i in range(n)]
    return prompts, user_agents, expected


async def _concurrent_wave(
    ac: httpx.AsyncClient, prompts: list[str], user_agents: list[str]
) -> list[tuple[str, str, str]]:
    """Fire one turn per session concurrently; return ``(session_id, model, reason)`` in order."""
    responses = await asyncio.gather(
        *(
            ac.post(
                CHAT_PATH,
                json=chat_body(content=prompt),
                headers={"User-Agent": ua},
            )
            for prompt, ua in zip(prompts, user_agents, strict=True)
        )
    )
    wave: list[tuple[str, str, str]] = []
    for resp in responses:
        assert resp.status_code == 200, resp.text
        model, reason = parse_decision(resp.headers["X-Shunt-Decision"])
        wave.append((resp.headers["X-Shunt-Session-Id"], model, reason))
    return wave


async def _drive(app: Any, ac: httpx.AsyncClient, *, turns: int = _TURNS) -> dict[str, list[str]]:
    """Drive ``_N_SESSIONS`` concurrent sessions for ``turns`` rounds; return sid -> models."""
    prompts, user_agents, _expected = _session_plan(_N_SESSIONS)
    per_session: dict[str, list[str]] = {}
    for _ in range(turns):
        for sid, model, _reason in await _concurrent_wave(ac, prompts, user_agents):
            per_session.setdefault(sid, []).append(model)
    return per_session


# ── 1. Per-session decision lock under concurrency ──────────────────────────


@pytest.mark.asyncio
async def test_per_session_model_lock_holds_under_concurrency(
    app_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_acompletion: Any,
) -> None:
    """Every turn of every session carries the SAME model (one decision per session)."""
    async with _served_app(app_factory, tmp_path, monkeypatch) as app:
        inflight = _InFlight()
        _arm_slow_upstream(mock_acompletion, inflight)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            per_session = await _drive(app, ac)

            assert len(per_session) == _N_SESSIONS, "expected N distinct concurrent sessions"
            for sid, models in per_session.items():
                assert len(models) == _TURNS, f"session {sid} dropped a turn: {models}"
                assert len(set(models)) == 1, f"session {sid} switched models mid-session: {models}"
            assert inflight.max >= 2, (
                "requests never overlapped — the test did not run concurrently"
            )


# ── 2. Cross-session isolation ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_cross_session_model_leak_under_concurrency(
    app_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_acompletion: Any,
) -> None:
    """Concurrent sessions keep THEIR model on every turn; nobody borrows another's."""
    async with _served_app(app_factory, tmp_path, monkeypatch) as app:
        _arm_slow_upstream(mock_acompletion, _InFlight())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            prompts, user_agents, expected = _session_plan(_N_SESSIONS)
            for _ in range(_TURNS):
                wave = await _concurrent_wave(ac, prompts, user_agents)
                for (sid, model, _reason), want in zip(wave, expected, strict=True):
                    assert model == want, f"session {sid} served {model}, not its own {want}"


# ── 3. Deterministic differentiation ────────────────────────────────────────


@pytest.mark.asyncio
async def test_distinct_prompts_route_to_distinct_stable_models(
    app_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_acompletion: Any,
) -> None:
    """The fake embedder's distinct prompts DO route to distinct models, and each
    session keeps its own cluster's model across turns (the leak-detection signal)."""
    async with _served_app(app_factory, tmp_path, monkeypatch) as app:
        _arm_slow_upstream(mock_acompletion, _InFlight())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            per_session = await _drive(app, ac)

            first_models = {models[0] for models in per_session.values()}
            assert first_models == {_CLUSTER_A_MODEL, _CLUSTER_B_MODEL}, (
                f"differentiation absent (got {sorted(first_models)}) — every session routed "
                "identically, so a cross-session leak would be invisible"
            )
            expected = {_CLUSTER_A_MODEL, _CLUSTER_B_MODEL}
            for sid, models in per_session.items():
                assert set(models) <= expected, f"session {sid} routed to a foreign model: {models}"


# ── 4. Store integrity under concurrency ────────────────────────────────────


@pytest.mark.asyncio
async def test_store_records_each_session_under_concurrency(
    app_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_acompletion: Any,
) -> None:
    """Exactly N live rows, each with the session's own model + provenance — no lost or
    collided rows, no wrong-model attribution."""
    async with _served_app(app_factory, tmp_path, monkeypatch) as app:
        _arm_slow_upstream(mock_acompletion, _InFlight())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            per_session = await _drive(app, ac)

            store = app.state.outcome_store
            all_rows = store.get_sessions(limit=1000)
            live_rows = [
                row for row in all_rows if not str(row["session_id"]).startswith(_SEED_SID_PREFIX)
            ]
            assert len(live_rows) == len(per_session) == _N_SESSIONS, (
                f"expected exactly {_N_SESSIONS} live rows, got {len(live_rows)}"
            )
            assert len(all_rows) == _N_SESSIONS + len(_CLUSTERS) * _SEEDS_PER_CLUSTER
            by_sid = {str(row["session_id"]): row for row in live_rows}
            assert set(by_sid) == set(per_session), "stored session ids differ from the wire"

            for sid, models in per_session.items():
                row = by_sid[sid]
                expected_model = models[0]
                assert row["model_chosen"] == expected_model, (
                    f"store attributed {row['model_chosen']} to {sid}, the wire said "
                    f"{expected_model}"
                )
                provenance = json.loads(row["decision_provenance"] or "{}")
                assert provenance.get("model_chosen") == expected_model, (
                    f"{sid} provenance names {provenance.get('model_chosen')}, not {expected_model}"
                )


# ── 5. Budget: concurrent spend counted exactly once ────────────────────────


@pytest.mark.asyncio
async def test_concurrent_spend_accumulates_exactly_once(
    app_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_acompletion: Any,
) -> None:
    """Each session's reported spend lands in the store exactly once per turn — no lost
    update, no double count — even with all N sessions routing at once."""
    async with _served_app(app_factory, tmp_path, monkeypatch, budget_usd=_BUDGET_USD) as app:
        _arm_slow_upstream(mock_acompletion, _InFlight())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            await _drive(app, ac)  # every turn asserts 200 — no budget mis-fire, no double-stop

            store = app.state.outcome_store
            live_rows = [
                row
                for row in store.get_sessions(limit=1000)
                if not str(row["session_id"]).startswith(_SEED_SID_PREFIX)
            ]
            assert len(live_rows) == _N_SESSIONS
            for row in live_rows:
                assert row["cost_known"] == 1, f"session {row['session_id']} cost is UNKNOWN"
                expected = _TURNS * _TURN_COST_USD
                assert abs(float(row["cost"]) - expected) < 1e-9, (
                    f"session {row['session_id']} accumulated {row['cost']}, expected {expected}"
                )
            total = sum(float(row["cost"]) for row in live_rows)
            assert abs(total - _N_SESSIONS * _TURNS * _TURN_COST_USD) < 1e-6, f"total {total}"
