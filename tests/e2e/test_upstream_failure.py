"""Upstream failure/retry/fallback through the SERVED app — hermetic end-to-end.

The unit suite mocks `_acompletion`; here the assembled router+server runs the real
OpenAI client over real HTTP to a live, scripted fake upstream.
"""

# How this harness works (SH002 caps the docstring; the detail lives here):
#
# The real served app (FastAPI TestClient) runs the real OpenAI client over real HTTP
# against a live, scripted fake upstream — `_acompletion` is NOT mocked — so the
# router's own retry/fallback/exhaustion loop is what the assertions observe.
#
# Two seams are injected, and both are configuration, not upstream mocks:
#   1. `server.Embedder` -> `tests.fake_embedder.FakeEmbedder` (no ONNX download).
#   2. `shunt.proxy.router._client_for` -> an `AsyncOpenAI(max_retries=0)`.
#
# The second seam exists because the openai SDK ALSO retries 408/409/429/5xx and
# connection errors internally (DEFAULT_MAX_RETRIES=2, openai/_base_client.py). Left
# on, a persistent 500 is retried by the SDK, never by the router, so the router's own
# retry loop would never fire and the request-count assertions would count the SDK's
# hidden retries instead of the router's policy. Disabling the SDK layer makes the
# router the only retry layer — exactly how the unit tests model it.
#
# Non-streaming responses persist the session row synchronously inside the request, so
# no capture polling is needed. The fake embedder routes the empty store via cold-start
# to `deepseek-v4-flash` (src/shunt/proxy/router.py `_DEFAULT_MODEL`), so the fallback
# chain is deepseek-v4-flash -> fake-mid -> fake-frontier (ModelPool.fallback_chain,
# config.py L416-434: self, then rank neighbours outward).

from __future__ import annotations

import contextlib
import functools
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from shunt.proxy import router as router_module
from shunt.proxy import server as server_module
from shunt.proxy.server import app
from tests.e2e.helpers import CHAT_PATH, parse_decision
from tests.fake_embedder import FakeEmbedder

_REGISTRY = Path(__file__).parent.parent / "integrations" / "fake_registry.yaml"
_DOCKER_BASE_URL = "http://fake-upstream:9099/v1"

# The registry's provider-side ids, as sent in the upstream ``model`` field.
_CHEAP_ID = "fake/cheap"
_MID_ID = "fake/mid"
_FRONTIER_ID = "fake/frontier"

# Registry model names, as the router knows them (cold-start locks the cheapest).
_CHEAP = "deepseek-v4-flash"
_MID = "fake-mid"
_FRONTIER = "fake-frontier"


@pytest.fixture(autouse=True)
def mock_acompletion() -> None:
    """Shadow tests/e2e/conftest.py's autouse upstream mock — this suite needs the REAL
    ``_acompletion`` over HTTP, so a closer-scope fixture of the same name must suppress
    it (pytest fixture override by name; the autouse flag does not carry over).
    """


# Retries the router performs per model before falling back (mirrors the unit-test
# fixture, tests/proxy/test_router.py L82). Kept small so the 0.5s backoff bounds the
# suite; the backoff itself is 2.0**attempt * 0.5 (router.py L811).
RETRY_COUNT = 2

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


def _completion_body(model: str) -> str:
    """A minimal but valid non-streaming ChatCompletion the OpenAI SDK can parse."""
    return json.dumps(
        {
            "id": "fake-cmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


def _error_body(status: int) -> str:
    return json.dumps({"error": {"message": f"scripted {status} failure", "type": "server_error"}})


class FailingUpstream:
    """A scripted OpenAI-compatible stub: per-model failure + connection-drop modes."""

    # `fail(model_id, count=, status=)` answers *status* for the next *count* requests;
    # `drop(model_id, count=)` closes the TCP connection without a response for the next
    # *count* requests — what a real provider outage looks like to the openai SDK, which
    # raises `APIConnectionError`. Every chat-completions POST payload is recorded on
    # `requests` so a test can count exactly how many attempts the router made.

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.requests: list[dict[str, object]] = []
        # model_id -> [remaining_failures, status]; mutated by the handler thread.
        self.failures: dict[str, list[int]] = {}
        # model_id -> [remaining_drops]; mutated by the handler thread.
        self.drops: dict[str, list[int]] = {}
        self._server = ThreadingHTTPServer((host, port), _handler_for(self))
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}"

    def fail(self, model_id: str, *, count: int, status: int = 500) -> None:
        """Serve *status* for the next *count* requests to *model_id*, then succeed."""
        self.failures[model_id] = [count, status]

    def drop(self, model_id: str, *, count: int) -> None:
        """Close the connection without responding for the next *count* requests."""
        self.drops[model_id] = [count]

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> FailingUpstream:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def _handler_for(upstream: FailingUpstream) -> type[BaseHTTPRequestHandler]:
    """Build a handler bound to one FailingUpstream's script + request log."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            if not self.path.endswith("/chat/completions"):
                self._send(404, json.dumps({"error": {"message": f"Path not found: {self.path}"}}))
                return
            payload: dict[str, object] = {}
            with contextlib.suppress(json.JSONDecodeError):
                payload = json.loads(raw)
            upstream.requests.append(payload)
            model = str(payload.get("model", ""))
            if (spec := upstream.drops.get(model)) and spec[0] > 0:
                spec[0] -= 1
                self.connection.close()
                return
            if (spec := upstream.failures.get(model)) and spec[0] > 0:
                spec[0] -= 1
                self._send(spec[1], _error_body(spec[1]))
                return
            self._send(200, _completion_body(model))

        def _send(self, status: int, body: str) -> None:
            encoded = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib name
            """Silence per-request stderr logging."""

    return Handler


@functools.lru_cache(maxsize=32)
def _no_retry_client(base_url: str, api_key: str) -> AsyncOpenAI:
    """The router's client factory, with the SDK's hidden retry layer disabled.

    See the module docstring — the router's own retry loop is what is under test, so
    the SDK's ``max_retries=2`` is turned off rather than left to muddy the counts.
    """
    return AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)


@pytest.fixture
def failing_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, FailingUpstream]]:
    """The served app wired to a live scripted fake upstream (hermetic env per test)."""
    for name in _CONTROLLED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(server_module, "Embedder", FakeEmbedder)
    monkeypatch.setattr(server_module, "_RETRY_COUNT", RETRY_COUNT)
    monkeypatch.setattr(router_module, "_client_for", _no_retry_client)
    # Isolated store + config per test; empty `models:` routes over the whole registry.
    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path / "config"))
    with FailingUpstream() as upstream:
        registry = _REGISTRY.read_text().replace(_DOCKER_BASE_URL, f"{upstream.base_url}/v1")
        registry_path = tmp_path / "fake_registry.yaml"
        registry_path.write_text(registry)
        monkeypatch.setattr(server_module, "_MODEL_CONFIG_PATH", str(registry_path))
        config_dir = tmp_path / "config"
        config_dir.mkdir(exist_ok=True)
        (config_dir / "router.yaml").write_text("router:\n  models: []\n")
        with TestClient(app) as client:
            yield client, upstream


def _decision(resp: object) -> tuple[str, str]:
    """Parse the decision header of a *resp* with an ``X-Shunt-Decision`` header."""
    return parse_decision(resp.headers["X-Shunt-Decision"])  # type: ignore[attr-defined]


def _session_row(client: TestClient, resp: object) -> dict[str, object] | None:
    """The persisted session row for *resp*, or None when the request wrote none."""
    session_id = resp.headers["X-Shunt-Session-Id"]  # type: ignore[attr-defined]
    row = client.app.state.outcome_store.get_session(session_id)
    assert row is not None
    return row


def _post(client: TestClient) -> object:
    body = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    return client.post(CHAT_PATH, json=body, headers={"Authorization": "Bearer dummy"})


def test_retry_transient_failure_then_success(
    failing_app: tuple[TestClient, FailingUpstream],
) -> None:
    """A single transient 500 is retried by the router and the client still gets 200."""
    client, upstream = failing_app
    upstream.fail(_CHEAP_ID, count=1, status=500)

    resp = _post(client)

    # One 500 then one 200: the router's retry loop made exactly 2 upstream attempts
    # (router.py L801-821 — attempt, wait 2.0**attempt*0.5, re-attempt).
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    served_model, _ = _decision(resp)
    assert served_model == _CHEAP
    assert len(upstream.requests) == 2, upstream.requests
    assert [r.get("model") for r in upstream.requests] == [_CHEAP_ID, _CHEAP_ID]

    # The persisted row names the model that actually served — the locked cheap model
    # (retried, never fell back), not a fabricated success.
    row = _session_row(client, resp)
    assert row["model_chosen"] == _CHEAP
    # Retry is NOT a fallback: same model served, so provenance must NOT flag it.
    prov = json.loads(row["decision_provenance"] or "{}")
    assert prov.get("fallback_chain_triggered") is not True


def test_persistent_failure_falls_back_to_next_pool_model(
    failing_app: tuple[TestClient, FailingUpstream],
) -> None:
    """The cheap model failing every retry falls back to the NEXT pool model."""
    client, upstream = failing_app
    upstream.fail(_CHEAP_ID, count=10**6, status=500)

    resp = _post(client)

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    served_model, _ = _decision(resp)
    # Fallback order: deepseek-v4-flash retries RETRY_COUNT times, then fake-mid serves.
    assert served_model == _MID
    models_seen = [r.get("model") for r in upstream.requests]
    assert models_seen == [_CHEAP_ID] * RETRY_COUNT + [_MID_ID], models_seen

    # Provenance honesty: the failed cheap model is NOT recorded as the served model.
    row = _session_row(client, resp)
    assert row["model_chosen"] == _MID
    assert row["model_chosen"] != _CHEAP
    # Fallback-honesty regression: decision_provenance must name the SERVED model too
    # (what `shunt explain` prints), and the fallback must be flagged — a fallback
    # session must never explain as "chose <locked>, fallback: no".
    prov = json.loads(row["decision_provenance"] or "{}")
    assert prov.get("model_chosen") == _MID
    assert prov.get("fallback_chain_triggered") is True


def test_rate_limit_falls_back(failing_app: tuple[TestClient, FailingUpstream]) -> None:
    """429 (RateLimitError) is retryable and falls back exactly like a 500."""
    client, upstream = failing_app
    upstream.fail(_CHEAP_ID, count=10**6, status=429)

    resp = _post(client)

    assert resp.status_code == 200
    served_model, _ = _decision(resp)
    assert served_model == _MID
    models_seen = [r.get("model") for r in upstream.requests]
    assert models_seen == [_CHEAP_ID] * RETRY_COUNT + [_MID_ID], models_seen
    row = _session_row(client, resp)
    assert row["model_chosen"] == _MID


def test_connection_error_falls_back(failing_app: tuple[TestClient, FailingUpstream]) -> None:
    """A connection that dies without responding (APIConnectionError) falls back too."""
    client, upstream = failing_app
    # Drop the connection on every request to the cheap model; the openai SDK surfaces
    # this as APIConnectionError, which router._is_retryable treats as retryable.
    upstream.drop(_CHEAP_ID, count=10**6)

    resp = _post(client)

    assert resp.status_code == 200
    served_model, _ = _decision(resp)
    assert served_model == _MID
    # Every cheap-model attempt was a dropped connection; only the fallback served.
    models_seen = [r.get("model") for r in upstream.requests]
    assert models_seen == [_CHEAP_ID] * RETRY_COUNT + [_MID_ID], models_seen
    row = _session_row(client, resp)
    assert row["model_chosen"] == _MID


def test_all_models_exhausted_returns_error_no_fabricated_success(
    failing_app: tuple[TestClient, FailingUpstream],
) -> None:
    """Every model exhausted -> a real upstream error response, never a fake success."""
    client, upstream = failing_app
    for model_id in (_CHEAP_ID, _MID_ID, _FRONTIER_ID):
        upstream.fail(model_id, count=10**6, status=500)

    resp = _post(client)

    assert resp.status_code == 502
    body = resp.json()
    assert body["error"]["type"] == "proxy_error"
    assert "All models exhausted" in body["error"]["message"]
    served_model, reason = _decision(resp)
    assert served_model == "error"
    assert "All models exhausted" in reason

    # No session row is persisted for a failed request (the endpoint returns without
    # writing one), so nothing can be read back as a success on the wrong model.
    session_id = resp.headers["X-Shunt-Session-Id"]
    assert client.app.state.outcome_store.get_session(session_id) is None
