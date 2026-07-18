"""Always-on hermetic handshake: a client → Shunt → fake-upstream roundtrip.

Layer 1 — no Docker, no real model. Drives Shunt in-process via TestClient (real
routing, `_acompletion` not mocked) against a live FakeUpstream over both wires.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import shunt.proxy.server as server
from tests.integrations.fake_upstream import FakeUpstream

_REGISTRY = Path(__file__).parent / "fake_registry.yaml"
_DOCKER_BASE_URL = "http://fake-upstream:9099/v1"


@pytest.fixture
def handshake_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, FakeUpstream]]:
    """Shunt (TestClient) wired to a live FakeUpstream via a port-rewritten registry."""
    with FakeUpstream() as upstream:
        registry = _REGISTRY.read_text().replace(_DOCKER_BASE_URL, f"{upstream.base_url}/v1")
        registry_path = tmp_path / "fake_registry.yaml"
        registry_path.write_text(registry)
        monkeypatch.setattr(server, "_MODEL_CONFIG_PATH", str(registry_path))
        with TestClient(server.app) as client:
            yield client, upstream


def test_models_endpoint_lists_fake_registry(
    handshake_client: tuple[TestClient, FakeUpstream],
) -> None:
    client, _ = handshake_client
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    ids = [row["id"] for row in resp.json()["data"]]
    assert "qwen3.7-plus" in ids  # the cold-start model must be discoverable


def test_openai_wire_roundtrip(handshake_client: tuple[TestClient, FakeUpstream]) -> None:
    """OpenAI-wire client → Shunt → fake upstream, with the decision header back."""
    client, upstream = handshake_client
    body = {"model": "auto", "messages": [{"role": "user", "content": "hi"}], "stream": False}
    resp = client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer dummy"})
    assert resp.status_code == 200
    assert "X-Shunt-Decision" in resp.headers
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    assert any("/chat/completions" in hit for hit in upstream.received)


def test_anthropic_wire_roundtrip(handshake_client: tuple[TestClient, FakeUpstream]) -> None:
    """Anthropic-wire client → Shunt → fake upstream, translated both ways."""
    client, upstream = handshake_client
    body = {
        "model": "auto",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    resp = client.post("/v1/messages", json=body, headers={"x-api-key": "dummy"})
    assert resp.status_code == 200
    assert "X-Shunt-Decision" in resp.headers
    assert any("/chat/completions" in hit for hit in upstream.received)
