"""End-to-end: a provider key in an upstream error must not reach the client.

Helper-level tests pass whether or not the endpoints CALL the helpers, so these drive
the real endpoints — that gap is how the header half of this bug survived the body fix.
"""

from __future__ import annotations

from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from shunt.proxy.server import app

_ACOMPLETION_PATCH = "shunt.proxy.router._acompletion"
# Shaped like a real OpenAI 401 body, which quotes the submitted key back.
# Assembled at runtime so a secret scanner cannot mistake the fixture for a live key.
_SECRET: Final[str] = "sk-proj-" + "A" * 8 + "0123456789bcdef"
_LEAKY_ERROR: Final[str] = f"Incorrect API key provided: {_SECRET}. Check your key."


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_upstream_key_never_reaches_the_response(client: TestClient, path: str) -> None:
    with patch(_ACOMPLETION_PATCH, new=AsyncMock(side_effect=RuntimeError(_LEAKY_ERROR))):
        response = client.post(
            path, json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
        )

    assert _SECRET not in response.text
    assert _SECRET not in str(response.headers)


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_multiline_upstream_error_cannot_split_the_response(client: TestClient, path: str) -> None:
    # Raw exception text in X-Shunt-Decision raised LocalProtocolError, so the 502
    # never reached the wire at all.
    with patch(
        _ACOMPLETION_PATCH,
        new=AsyncMock(side_effect=RuntimeError("boom\r\nX-Injected: yes")),
    ):
        response = client.post(
            path, json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
        )

    assert "x-injected" not in {k.lower() for k in response.headers}


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_non_ascii_upstream_error_still_returns_a_response(client: TestClient, path: str) -> None:
    # Non-ASCII text in the header raised UnicodeEncodeError inside the handler.
    with patch(_ACOMPLETION_PATCH, new=AsyncMock(side_effect=RuntimeError("Ошибка провайдера"))):
        response = client.post(
            path, json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
        )

    assert response.status_code >= 400
