"""A provider key quoted in a fallback error must not reach the WARNING log.

The scrubbed body/headers left the fallback line live, so a 402 insufficient-balance
body (DeepSeek/Requesty quote the key back) hit stdout and docker logs per candidate.
"""

from __future__ import annotations

import logging
from typing import Any, Final
from unittest.mock import AsyncMock, patch

import pytest

from shunt.models.config import ModelPool
from shunt.proxy.router import ProxyRouter, UpstreamError
from shunt.session import Session, SessionManager

_ACOMPLETION_PATCH = "shunt.proxy.router._acompletion"
# Assembled at runtime so a secret scanner cannot mistake the fixture for a live key.
_SECRET: Final[str] = "sk-" + "A" * 8 + "0123456789bcdef"
_LEAKY_BODY: Final[str] = f"Insufficient Balance for api key: {_SECRET}. Top up your account."


class _Leaky402Error(Exception):
    """A non-retryable upstream failure whose body quotes the submitted key."""

    status_code = 402


@pytest.fixture
def router() -> ProxyRouter:
    return ProxyRouter(
        model_pool=ModelPool(),
        session_manager=SessionManager(inactivity_timeout=900, grace_period=120),
        retry_count=1,
    )


@pytest.fixture
def session() -> Session:
    return SessionManager(inactivity_timeout=900, grace_period=120).create_session("test-tool")


@pytest.mark.asyncio
async def test_fallback_warning_never_logs_the_provider_key(
    router: ProxyRouter, session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    kwargs: dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    with (
        caplog.at_level(logging.WARNING, logger="shunt.proxy.router"),
        patch(_ACOMPLETION_PATCH, new=AsyncMock(side_effect=_Leaky402Error(_LEAKY_BODY))),
        pytest.raises(UpstreamError),
    ):
        await router._route_with_fallback(kwargs, session)

    warnings = [r for r in caplog.records if "trying next" in r.getMessage()]
    assert warnings, "the fallback path must have logged at least one candidate failure"
    for record in warnings:
        # The FORMATTED message, not just the args: %s-lazy formatting hides the leak
        # from an args-only assertion while the handler still writes it out.
        assert _SECRET not in record.getMessage()
        assert "<redacted>" in record.getMessage()


@pytest.mark.asyncio
async def test_non_upstream_fallback_warning_never_logs_the_provider_key(
    router: ProxyRouter, session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    # The second warning site: anything that is not an UpstreamError. Same leak, same fix.
    kwargs: dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    with (
        caplog.at_level(logging.WARNING, logger="shunt.proxy.router"),
        patch.object(router, "_try_model", new=AsyncMock(side_effect=RuntimeError(_LEAKY_BODY))),
        pytest.raises(UpstreamError),
    ):
        await router._route_with_fallback(kwargs, session)

    warnings = [r for r in caplog.records if "trying next" in r.getMessage()]
    assert warnings
    for record in warnings:
        assert _SECRET not in record.getMessage()
        assert "<redacted>" in record.getMessage()
