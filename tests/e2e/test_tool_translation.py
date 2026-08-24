# The unit suite (tests/proxy/test_anthropic_tool_translation.py) proves the format
# conversion in isolation; this file proves the whole assembled path: an Anthropic-wire
# client POSTs /v1/messages with a tool conversation, the served app translates to the
# OpenAI wire and forwards over real HTTP to a live FakeUpstream, and translates the
# response back (SSE when streaming) so the client sees the tool_use block, the tool_use
# stop_reason, and the usage. `_acompletion` is NOT mocked here — the sibling suite's
# autouse canned-upstream mock is suppressed and the router's real OpenAI client drives
# the wire.
"""Anthropic-wire tool calls end to end through the SERVED app to a live fake upstream."""

from __future__ import annotations

import functools
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from shunt.proxy import router as router_module
from shunt.proxy import server as server_module
from shunt.proxy.server import app
from tests.e2e.helpers import MESSAGES_PATH, parse_decision, user_agent_headers
from tests.fake_embedder import FakeEmbedder
from tests.integrations.fake_upstream import FakeUpstream

_REGISTRY = Path(__file__).parent.parent / "integrations" / "fake_registry.yaml"
_DOCKER_BASE_URL = "http://fake-upstream:9099/v1"
_CHEAP = "qwen3.7-plus"
_ANTHROPIC_KEY: Final[dict[str, str]] = {"x-api-key": "dummy"}

# Router env vars the e2e suite controls; a dev-machine leftover would break hermeticity.
_CONTROLLED_ENV: tuple[str, ...] = (
    "SHUNT_ROUTER_STRATEGY",
    "SHUNT_WORK_DIR",
    "SHUNT_EXPLORATION_ENABLED",
    "SHUNT_EXPLORE_BUDGET_FRAC",
    "SHUNT_MODEL_CONFIG_PATH",
    "SHUNT_COLD_START_THRESHOLD_TIER1",
    "SHUNT_COLD_START_THRESHOLD_TIER2",
)


def _read_file_tool() -> dict[str, object]:
    """The Anthropic shape of one read_file tool declaration."""
    return {
        "name": "read_file",
        "description": "Read a file",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
    }


def _parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    """Parse the Anthropic SSE body into (event, data) pairs, in wire order."""
    events: list[tuple[str, dict[str, object]]] = []
    for block in body.split("\n\n"):
        event_name: str | None = None
        payload: list[str] = []
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                payload.append(line[len("data: ") :].strip())
        if event_name is not None and payload:
            events.append((event_name, json.loads("".join(payload))))
    return events


def _types(events: list[tuple[str, dict[str, object]]], event: str) -> list[dict[str, object]]:
    """The data payloads of every *event* in *events*, in order."""
    return [data for name, data in events if name == event]


@functools.lru_cache(maxsize=32)
def _no_retry_client(base_url: str, api_key: str) -> AsyncOpenAI:
    """The router's client factory with the SDK's hidden retry layer disabled."""
    return AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)


@pytest.fixture(autouse=True)
def mock_acompletion() -> None:
    """Shadow tests/e2e/conftest.py's autouse canned-upstream mock: this suite needs the
    REAL ``_acompletion`` over HTTP, so a closer-scope fixture of the same name must
    suppress it (pytest fixture override by name; the autouse flag does not carry over).
    """


@pytest.fixture
def tool_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, FakeUpstream]]:
    """The served app wired to a live FakeUpstream via the port-rewritten fake registry."""
    for name in _CONTROLLED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(server_module, "Embedder", FakeEmbedder)
    monkeypatch.setattr(router_module, "_client_for", _no_retry_client)
    # Isolated store + config per test; empty `models:` routes over the whole registry.
    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path / "config"))
    with FakeUpstream() as upstream:
        registry = _REGISTRY.read_text().replace(_DOCKER_BASE_URL, f"{upstream.base_url}/v1")
        registry_path = tmp_path / "fake_registry.yaml"
        registry_path.write_text(registry)
        monkeypatch.setattr(server_module, "_MODEL_CONFIG_PATH", str(registry_path))
        config_dir = tmp_path / "config"
        config_dir.mkdir(exist_ok=True)
        # `knn_cascade` explicitly: this test asserts the kNN reason vocabulary, which the
        # shipped `session_cascade` default does not produce. `escalation` must be spelled out
        # because a user file replaces the packaged one wholesale and an absent block reads
        # as OFF — which a cascade id refuses at load.
        (config_dir / "router.yaml").write_text(
            "router:\n  strategy: knn_cascade\n  escalation:\n    enabled: true\n  models: []\n"
        )
        with TestClient(app) as client:
            yield client, upstream


def _messages_headers() -> dict[str, str]:
    """One fixed tool identity per test so every POST joins the same session."""
    return {**_ANTHROPIC_KEY, **user_agent_headers()}


def _post_streamed(client: TestClient, body: dict[str, object]) -> tuple[httpx.Headers, str]:
    """POST an Anthropic body with stream:true; return (headers, decoded SSE body)."""
    with client.stream("POST", MESSAGES_PATH, json=body, headers=_messages_headers()) as resp:
        assert resp.status_code == 200
        return resp.headers, resp.read().decode("utf-8")


# ── RESPONSE direction ─────────────────────────────────────────────────────


def test_tool_use_block_survives_served_stream(tool_app: tuple[TestClient, FakeUpstream]) -> None:
    """The fake upstream's tool_calls reply comes back as an Anthropic tool_use block."""
    client, upstream = tool_app
    body = {
        "model": "auto",
        "max_tokens": 64,
        "stream": True,
        "tools": [_read_file_tool()],
        "messages": [{"role": "user", "content": "what is in main.py?"}],
    }

    headers, raw = _post_streamed(client, body)
    events = _parse_sse(raw)

    assert "X-Shunt-Decision" in headers
    assert any("/chat/completions" in hit for hit in upstream.received)
    # The request declared tools, so the upstream answered with a tool call.
    assert upstream.bodies[0]["tools"]

    tool_starts = [
        d
        for d in _types(events, "content_block_start")
        if d["content_block"]["type"] == "tool_use"  # type: ignore[index]
    ]
    assert len(tool_starts) == 1
    block = tool_starts[0]["content_block"]  # type: ignore[index]
    assert block["name"] == "read_file"
    assert block["id"] == "call_abc123"

    deltas = [
        d
        for d in _types(events, "content_block_delta")
        if d["delta"]["type"] == "input_json_delta"  # type: ignore[index]
    ]
    assert len(deltas) == 1
    assert json.loads(deltas[0]["delta"]["partial_json"]) == {"path": "main.py"}  # type: ignore[index]

    message_delta = _types(events, "message_delta")
    assert len(message_delta) == 1
    assert message_delta[0]["delta"]["stop_reason"] == "tool_use"  # type: ignore[index]
    assert len(_types(events, "message_stop")) == 1


# ── REQUEST direction ──────────────────────────────────────────────────────


def test_tool_result_conversation_reaches_upstream_as_openai_wire(
    tool_app: tuple[TestClient, FakeUpstream],
) -> None:
    """An assistant tool_use turn + tool_result become OpenAI tool_calls + a tool message."""
    client, upstream = tool_app
    body = {
        "model": "auto",
        "max_tokens": 64,
        "stream": False,
        "tools": [_read_file_tool()],
        "messages": [
            {"role": "user", "content": "what is in main.py?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "read_file",
                        "input": {"path": "main.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "def main(): pass"},
                    {"type": "text", "text": "now fix it"},
                ],
            },
        ],
    }

    resp = client.post(MESSAGES_PATH, json=body, headers=_messages_headers())
    assert resp.status_code == 200
    # The upstream answered with a tool call, which must translate back too.
    payload = resp.json()
    assert payload["stop_reason"] == "tool_use"
    assert payload["content"][0]["type"] == "tool_use"
    assert payload["content"][0]["name"] == "read_file"

    assert len(upstream.bodies) == 1
    sent = upstream.bodies[0]
    messages = sent["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "user"]

    assistant = messages[1]
    call = assistant["tool_calls"][0]
    assert call["id"] == "tu_1"
    assert call["function"]["name"] == "read_file"
    # OpenAI wants arguments as a JSON string; it must decode to the Anthropic input.
    assert json.loads(call["function"]["arguments"]) == {"path": "main.py"}

    tool_msg = messages[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "tu_1"
    assert tool_msg["content"] == "def main(): pass"

    # The Anthropic tool declaration reached the upstream in the OpenAI function shape.
    assert sent["tools"][0]["type"] == "function"
    assert sent["tools"][0]["function"]["name"] == "read_file"


# ── STREAMING ──────────────────────────────────────────────────────────────


def test_streamed_tool_call_event_order_and_usage(
    tool_app: tuple[TestClient, FakeUpstream],
) -> None:
    """The full SSE translation: tool deltas stream, then message_delta (tool_use), then
    message_stop, with the trailing usage-only chunk folded into message_delta."""
    client, _ = tool_app
    body = {
        "model": "auto",
        "max_tokens": 64,
        "stream": True,
        "tools": [_read_file_tool()],
        "messages": [{"role": "user", "content": "what is in main.py?"}],
    }

    _headers, raw = _post_streamed(client, body)
    events = _parse_sse(raw)

    # message_start, then: text block 0, tool block 1, one args delta, both blocks closed,
    # then message_delta (stop_reason=tool_use) and finally message_stop.
    tail = [name for name, _ in events if name != "message_start"]
    assert tail == [
        "content_block_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]

    # The two tool_use fragments stream as raw JSON fragments the client reassembles.
    deltas = [
        d
        for d in _types(events, "content_block_delta")
        if d["delta"]["type"] == "input_json_delta"  # type: ignore[index]
    ]
    assert deltas[0]["delta"]["partial_json"] == '{"path": "main.py"}'  # type: ignore[index]

    # The trailing usage-only chunk (choices==[]) is folded into message_delta.
    message_delta = _types(events, "message_delta")
    assert len(message_delta) == 1
    assert message_delta[0]["delta"]["stop_reason"] == "tool_use"  # type: ignore[index]
    assert message_delta[0]["usage"] == {"input_tokens": 5, "output_tokens": 7}  # type: ignore[index]

    # Both content blocks are closed — the text block (0) and the tool block (1).
    stops = [d["index"] for d in _types(events, "content_block_stop")]
    assert sorted(stops) == [0, 1]


# ── Cache-control passthrough ──────────────────────────────────────────────


def test_cache_control_block_survives_to_upstream(
    tool_app: tuple[TestClient, FakeUpstream],
) -> None:
    """Cache-safety: an ephemeral marker on a text block reaches the upstream untouched."""
    client, upstream = tool_app
    block = {"type": "text", "text": "big prefix", "cache_control": {"type": "ephemeral"}}
    body = {
        "model": "auto",
        "max_tokens": 8,
        "stream": False,
        "messages": [{"role": "user", "content": [block]}],
    }

    resp = client.post(MESSAGES_PATH, json=body, headers=_messages_headers())
    assert resp.status_code == 200
    assert upstream.bodies[0]["messages"][0]["content"] == [block]


# ── Routing decision + session provenance on the tool-call path ────────────


def test_decision_and_session_recorded_on_tool_path(
    tool_app: tuple[TestClient, FakeUpstream],
) -> None:
    """Routing still happened and cost was still recorded on a tool-calling turn."""
    client, _ = tool_app
    body = {
        "model": "auto",
        "max_tokens": 64,
        "stream": True,
        "tools": [_read_file_tool()],
        "messages": [{"role": "user", "content": "what is in main.py?"}],
    }

    headers, _raw = _post_streamed(client, body)
    sid = headers["X-Shunt-Session-Id"]
    model, reason = parse_decision(headers["X-Shunt-Decision"])

    assert model == _CHEAP
    assert reason == "cold_start"

    # The usage-only trailing chunk (prompt_tokens=5, cost=0.0001) was folded into the
    # session row exactly as on the plain-text path — the tool wire did not skip it.
    row = client.app.state.outcome_store.get_session(sid)
    assert row is not None
    assert row["model_chosen"] == _CHEAP
    assert json.loads(row["cache_stats"])["prompt_tokens"] == 5
    assert row["cost"] == 0.0001
    assert row["cost_known"] == 1
    prov = json.loads(row["decision_provenance"])
    assert prov["model_chosen"] == _CHEAP
