"""A malformed client body must answer 400, never 500."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from shunt.proxy.server import app


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_malformed_json_is_a_client_error(client: TestClient, path: str) -> None:
    # An unguarded request.json() raised JSONDecodeError, which FastAPI rendered as a 500
    # with a traceback — a client mistake reported as a server fault, and noise that hides
    # real server failures in the same log.
    response = client.post(
        path, content=b'{"model":"auto","messages":[', headers={"content-type": "application/json"}
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "bad_request"


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_json_that_is_not_an_object_is_a_client_error(client: TestClient, path: str) -> None:
    # `[1,2,3]` parses fine, then .get() blows up further in — same 500, different cause.
    response = client.post(path, json=[1, 2, 3])

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "bad_request"


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_empty_body_is_a_client_error(client: TestClient, path: str) -> None:
    response = client.post(path, content=b"", headers={"content-type": "application/json"})

    assert response.status_code == 400
