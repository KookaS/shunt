"""Cache hit ratio must divide like-for-like across a multi-turn session."""

# Reproduces a real logged session: `cache_tax` accumulated across turns while
# `prompt_length_tokens` was overwritten each turn, so the ratio grew past 1.0 and the
# `min(1.0, ...)` clamp reported a perfect 1.0000 from the second cached turn onward.

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Final

import pytest

from shunt.models.config import ModelPool
from shunt.proxy.router import ProxyRouter
from shunt.session import Session, SessionManager

# The observed turns: (cached_tokens, prompt_tokens). Cached is a flat 19072 per turn
# against a ~20.9k prompt, i.e. a true per-turn hit ratio just under 0.92 — never 1.0.
_OBSERVED_TURNS: Final[tuple[tuple[int, int], ...]] = (
    (0, 20760),
    (0, 550),
    (19072, 20789),
    (19072, 20937),
    (19072, 20983),
)


class _Usage:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


def _response(cached: int, prompt: int) -> Any:
    return _Usage(
        usage=_Usage(
            prompt_tokens=prompt,
            prompt_tokens_details=_Usage(cached_tokens=cached),
        )
    )


@pytest.fixture
def router() -> ProxyRouter:
    return ProxyRouter(
        model_pool=ModelPool(),
        session_manager=SessionManager(inactivity_timeout=900, grace_period=120),
        retry_count=1,
    )


@pytest.fixture
def session() -> Session:
    return Session(session_id="s1", tool_identity="t", start_time=datetime.now(UTC))


def test_hit_ratio_never_exceeds_one_across_turns(router: ProxyRouter, session: Session) -> None:
    for cached, prompt in _OBSERVED_TURNS:
        router._update_cache_tax(session, _response(cached, prompt))
        ratio = router._session_hit_ratio(session)
        assert 0.0 <= ratio <= 1.0

    # 3 cached turns of 19072 against the cumulative prompt total — not a clamped 1.0.
    assert session.cache_tax == pytest.approx(3 * 19072)
    assert session.prompt_tokens_total == sum(p for _, p in _OBSERVED_TURNS)
    assert router._session_hit_ratio(session) == pytest.approx(57216 / 84019, abs=1e-4)


def test_a_fully_cached_turn_is_not_reported_as_a_growing_ratio(
    router: ProxyRouter, session: Session
) -> None:
    # Identical turns must hold a steady ratio; the old code drifted upward every turn.
    ratios = []
    for _ in range(4):
        router._update_cache_tax(session, _response(900, 1000))
        ratios.append(router._session_hit_ratio(session))
    assert ratios == pytest.approx([0.9, 0.9, 0.9, 0.9])


def test_log_reports_both_turn_and_session_ratio(
    router: ProxyRouter, session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    router._update_cache_tax(session, _response(0, 1000))
    with caplog.at_level(logging.INFO, logger="shunt.proxy.router"):
        router._update_cache_tax(session, _response(500, 1000))
        router._log_cache_metrics(session)
    message = caplog.text
    # The turn ratio is what tells an operator whether THIS call hit cache.
    assert "turn_hit_ratio=0.5000" in message
    assert "session_hit_ratio=0.2500" in message
