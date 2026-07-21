"""Provider-reported cost ingestion: ``_reported_cost`` parsing and the unreported signal.

The router never derives a price locally — an unreported charge must stay unknown rather
than be guessed, and a *reported* zero must stay a real zero (free / fully-cached call).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shunt.models.config import ModelPool
from shunt.proxy.router import ProxyRouter
from shunt.session import Session, SessionManager

_ACOMPLETION_PATCH = "shunt.proxy.router._acompletion"


class _Usage:
    """Minimal usage object; ``cost`` omitted entirely when not passed."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


@pytest.fixture
def session_manager() -> SessionManager:
    return SessionManager(inactivity_timeout=900, grace_period=120)


@pytest.fixture
def router(session_manager: SessionManager) -> ProxyRouter:
    return ProxyRouter(
        model_pool=ModelPool(),
        session_manager=session_manager,
        retry_count=1,
    )


@pytest.fixture
def session(session_manager: SessionManager) -> Session:
    return session_manager.create_session("test-tool")


# ── _reported_cost: the five defensive branches ─────────────────────────────


def test_reported_cost_none_usage() -> None:
    assert ProxyRouter._reported_cost(None) is None


def test_reported_cost_absent_attribute_is_none() -> None:
    # No `cost` at all → unknown, never 0.0 (0.0 would understate the real bill).
    assert ProxyRouter._reported_cost(_Usage(prompt_tokens=100)) is None


def test_reported_cost_absent_dict_key_is_none() -> None:
    assert ProxyRouter._reported_cost({"prompt_tokens": 100}) is None


def test_reported_cost_zero_is_zero_not_none() -> None:
    # THE regression branch: `if not raw` would reclassify a genuinely free or
    # fully-cached call as "cost unreported" and silently stop accumulating.
    assert ProxyRouter._reported_cost(_Usage(cost=0.0)) == 0.0
    assert ProxyRouter._reported_cost({"cost": 0.0}) == 0.0
    assert ProxyRouter._reported_cost(_Usage(cost=0)) == 0.0


def test_reported_cost_positive_value() -> None:
    assert ProxyRouter._reported_cost(_Usage(cost=0.00042)) == pytest.approx(0.00042)


def test_reported_cost_dict_usage_read_from_key() -> None:
    assert ProxyRouter._reported_cost({"cost": 1.25}) == pytest.approx(1.25)


def test_reported_cost_negative_is_none() -> None:
    # A negative charge is nonsense; treat as unreported rather than crediting the session.
    assert ProxyRouter._reported_cost(_Usage(cost=-1.0)) is None


@pytest.mark.parametrize("raw", [float("nan"), float("inf"), float("-inf")])
def test_reported_cost_non_finite_is_none(raw: float) -> None:
    assert ProxyRouter._reported_cost(_Usage(cost=raw)) is None


@pytest.mark.parametrize("raw", [True, False])
def test_reported_cost_bool_is_none(raw: bool) -> None:
    # bool is a subclass of int: True would otherwise silently bill 1.0.
    assert ProxyRouter._reported_cost(_Usage(cost=raw)) is None


def test_reported_cost_numeric_string_is_parsed() -> None:
    # Some gateways serialize the charge as a JSON string.
    assert ProxyRouter._reported_cost(_Usage(cost="0.5")) == pytest.approx(0.5)


def test_reported_cost_non_numeric_string_is_none() -> None:
    assert ProxyRouter._reported_cost(_Usage(cost="free")) is None


# ── _accumulate_cost: reported vs unreported ────────────────────────────────


def test_accumulate_reported_zero_does_not_flag_unreported(
    router: ProxyRouter, session: Session
) -> None:
    router._accumulate_cost(session, _Usage(cost=0.0))
    assert session.total_cost == 0.0
    assert "cost_unreported" not in session.metadata


def test_accumulate_sums_reported_costs(router: ProxyRouter, session: Session) -> None:
    router._accumulate_cost(session, _Usage(cost=0.25))
    router._accumulate_cost(session, _Usage(cost=0.75))
    assert session.total_cost == pytest.approx(1.0)
    assert "cost_unreported" not in session.metadata


def test_accumulate_unreported_flags_once(router: ProxyRouter, session: Session) -> None:
    router._accumulate_cost(session, _Usage(prompt_tokens=10))
    router._accumulate_cost(session, _Usage(prompt_tokens=10))
    assert session.total_cost == 0.0
    assert session.metadata["cost_unreported"] is True


# ── Streaming: usage present but no cost, and no usage at all ───────────────


def _content_chunk() -> MagicMock:
    chunk = MagicMock()
    chunk.usage = None
    chunk.model = "qwen3.7-plus"
    choice = MagicMock()
    choice.finish_reason = None
    delta = MagicMock()
    delta.role = "assistant"
    delta.content = "Hello"
    delta.model_dump.return_value = {"role": "assistant", "content": "Hello"}
    choice.delta = delta
    chunk.choices = [choice]
    return chunk


async def _drain(router: ProxyRouter, session: Session, chunks: list[Any]) -> None:
    async def mock_stream() -> AsyncGenerator[Any, None]:
        for chunk in chunks:
            yield chunk

    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": True}
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_stream()
        result, _model, _reason = await router.route_chat_completion(body, session)
        _ = [c async for c in result]


@pytest.mark.asyncio
async def test_streaming_usage_without_cost_flags_unreported(
    router: ProxyRouter, session: Session
) -> None:
    usage_chunk = MagicMock()
    usage_chunk.choices = []
    usage_chunk.usage = _Usage(prompt_tokens=100, prompt_tokens_details=None)
    await _drain(router, session, [_content_chunk(), usage_chunk])

    assert session.total_cost == 0.0
    assert session.metadata["cost_unreported"] is True


@pytest.mark.asyncio
async def test_streaming_usage_with_zero_cost_is_reported(
    router: ProxyRouter, session: Session
) -> None:
    usage_chunk = MagicMock()
    usage_chunk.choices = []
    usage_chunk.usage = _Usage(cost=0.0, prompt_tokens=100, prompt_tokens_details=None)
    await _drain(router, session, [_content_chunk(), usage_chunk])

    assert session.total_cost == 0.0
    assert "cost_unreported" not in session.metadata


@pytest.mark.asyncio
async def test_streaming_with_no_usage_chunk_never_signals_anything(
    router: ProxyRouter,
    session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Pins the CURRENT behaviour: when no chunk carries usage at all, _accumulate_cost is
    # never invoked, so the "cost unreported" signal is neither flagged nor logged — the
    # session simply looks free. A future fix that surfaces this must update this test.
    with caplog.at_level(logging.INFO, logger="shunt.proxy.router"):
        await _drain(router, session, [_content_chunk()])

    assert session.total_cost == 0.0
    assert "cost_unreported" not in session.metadata
    assert not [r for r in caplog.records if "usage.cost" in r.getMessage()]
    assert session.prompt_length_tokens == 0
