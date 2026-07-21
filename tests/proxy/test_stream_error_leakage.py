"""An error raised MID-STREAM must not carry a provider key out of the process."""

# The endpoint's try/except has already returned by the time the SSE generator is
# consumed, so `_persist_after_stream` was the last line of defence — and it had only
# `try/finally`, no `except`. The raw upstream text escaped to uvicorn's traceback logger.

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Final

import pytest

from shunt.proxy.router import UpstreamError
from shunt.proxy.server import _persist_after_stream

_SECRET: Final[str] = "gsk_" + "A" * 8 + "0123456789bcdef"
_LEAKY_ERROR: Final[str] = f"Invalid API Key: {_SECRET}"


async def _stream_that_dies() -> AsyncGenerator[bytes, None]:
    yield b"data: {}\n\n"
    raise RuntimeError(_LEAKY_ERROR)


@pytest.mark.asyncio
async def test_mid_stream_error_is_redacted_before_it_escapes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    persisted: list[bool] = []

    stream = _persist_after_stream(_stream_that_dies(), lambda: persisted.append(True))
    with (
        caplog.at_level(logging.ERROR, logger="shunt.proxy.server"),
        pytest.raises(UpstreamError) as raised,
    ):
        async for _ in stream:
            pass

    assert _SECRET not in str(raised.value)
    assert _SECRET not in caplog.text
    # The raw cause must not survive on the exception chain either.
    assert raised.value.__cause__ is None
    assert persisted == [True], "the session must still be persisted when a stream dies"
