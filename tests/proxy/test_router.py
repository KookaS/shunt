"""Tests for shunt proxy — ProxyRouter integration and server endpoints."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from shunt.models.config import ModelPool
from shunt.proxy.router import (
    _DEFAULT_MODEL,
    ProxyRouter,
    UpstreamError,
    _anthropic_request_to_openai,
    _is_retryable,
    _openai_chunk_to_anthropic_sse,
    _openai_response_to_anthropic,
    final_sse_events,
)
from shunt.proxy.server import app
from shunt.session import Session, SessionManager

_ACOMPLETION_PATCH = "shunt.proxy.router._acompletion"

# ── Fixtures ────────────────────────────────────────────────────────────────


class _FakeEmbedder:
    """A fixed-vector embedder so the full-app lifespan never loads real ONNX in tests."""

    def embed(self, text: str) -> Any:
        import numpy as np

        return np.full(768, 0.1, dtype=np.float32)

    def fingerprint(self) -> dict[str, Any]:
        return {"repo": "fake", "dim": 768, "max_chars": 4000, "revision": None}

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def max_chars(self) -> int:
        return 4000

    def warm(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _fake_lifespan_embedder() -> Any:
    """Tests that boot the whole app (TestClient(app)) get a fake embedder in the lifespan.

    Without this the real ONNX load is (correctly) blocked by SHUNT_DISALLOW_REAL_EMBEDDER,
    so the routing path would 502. Injecting a fake keeps these endpoint tests hermetic.
    """
    with patch("shunt.proxy.server.Embedder", _FakeEmbedder):
        yield


@pytest.fixture
def model_pool() -> ModelPool:
    pool = ModelPool()
    return pool


@pytest.fixture
def session_manager() -> SessionManager:
    return SessionManager(inactivity_timeout=900, grace_period=120)


@pytest.fixture
def router(model_pool: ModelPool, session_manager: SessionManager) -> ProxyRouter:
    return ProxyRouter(
        model_pool=model_pool,
        session_manager=session_manager,
        retry_count=2,
    )


@pytest.fixture
def session(session_manager: SessionManager) -> Session:
    return session_manager.create_session("test-tool")


# ── Test helper: _is_retryable ──────────────────────────────────────────────


def test_is_retryable_rate_limit() -> None:
    exc = Exception("rate limit exceeded")
    exc.status_code = 429  # type: ignore[attr-defined]
    assert _is_retryable(exc) is True


def test_is_retryable_server_error() -> None:
    exc = Exception("internal server error")
    exc.status_code = 500  # type: ignore[attr-defined]
    assert _is_retryable(exc) is True


def test_is_retryable_bad_request() -> None:
    exc = Exception("bad request")
    exc.status_code = 400  # type: ignore[attr-defined]
    assert _is_retryable(exc) is False


def test_is_retryable_auth_error() -> None:
    exc = Exception("unauthorized")
    exc.status_code = 401  # type: ignore[attr-defined]
    assert _is_retryable(exc) is False


def test_is_retryable_forbidden() -> None:
    exc = Exception("forbidden")
    exc.status_code = 403  # type: ignore[attr-defined]
    assert _is_retryable(exc) is False


def test_is_retryable_timeout_string() -> None:
    exc = Exception("request timeout")
    assert _is_retryable(exc) is True


def test_is_retryable_service_unavailable_string() -> None:
    exc = Exception("service unavailable")
    assert _is_retryable(exc) is True


# ── Test helper: Anthropic request conversion ───────────────────────────────


class TestAnthropicRequestToOpenAI:
    def test_basic_conversion(self) -> None:
        body = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stream": False,
        }
        result = _anthropic_request_to_openai(body)
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["max_tokens"] == 100
        assert result["stream"] is False

    def test_with_system(self) -> None:
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "system": "You are helpful.",
            "stream": False,
        }
        result = _anthropic_request_to_openai(body)
        assert result["messages"] == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

    def test_content_blocks(self) -> None:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"},
                    ],
                }
            ],
            "stream": False,
        }
        result = _anthropic_request_to_openai(body)
        assert result["messages"] == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}],
            }
        ]

    def test_image_blocks_skipped(self) -> None:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "abc",
                            },
                        },
                    ],
                }
            ],
            "stream": False,
        }
        result = _anthropic_request_to_openai(body)
        assert result["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "Describe this:"}]}
        ]

    def test_stop_sequences_mapped(self) -> None:
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "stop_sequences": ["\n\n"],
            "stream": False,
        }
        result = _anthropic_request_to_openai(body)
        assert result["stop"] == ["\n\n"]


# ── Test helper: Anthropic response conversion ──────────────────────────────


class TestOpenAIResponseToAnthropic:
    def test_basic_conversion(self) -> None:
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-123"
        mock_response.model = "qwen3.7-plus"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = "stop"
        choice.message.content = "Hello there"
        choice.message.role = "assistant"
        mock_response.choices = [choice]

        result = _openai_response_to_anthropic(mock_response)
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"] == [{"type": "text", "text": "Hello there"}]
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 20

    def test_length_finish_reason(self) -> None:
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-123"
        mock_response.model = "qwen3.7-plus"
        mock_response.usage = None

        choice = MagicMock()
        choice.finish_reason = "length"
        choice.message.content = "Partial"
        choice.message.role = "assistant"
        mock_response.choices = [choice]

        result = _openai_response_to_anthropic(mock_response)
        assert result["stop_reason"] == "max_tokens"

    def test_empty_content(self) -> None:
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-123"
        mock_response.model = "qwen3.7-plus"
        mock_response.usage = None

        choice = MagicMock()
        choice.finish_reason = "stop"
        choice.message.content = None
        choice.message.role = "assistant"
        mock_response.choices = [choice]

        result = _openai_response_to_anthropic(mock_response)
        assert result["content"] == [{"type": "text", "text": ""}]


# ── Test helper: Anthropic SSE conversion ───────────────────────────────────


class TestOpenAIChunkToAnthropicSSE:
    def test_role_chunk_emits_start_events(self) -> None:
        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-1"
        mock_chunk.model = "qwen3.7-plus"
        mock_chunk.usage = None

        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = None
        delta = MagicMock()
        delta.role = "assistant"
        delta.content = None
        delta.model_dump.return_value = {"role": "assistant"}
        choice.delta = delta
        mock_chunk.choices = [choice]

        events = _openai_chunk_to_anthropic_sse(mock_chunk)
        event_text = "\n".join(events)

        assert "message_start" in event_text
        assert "content_block_start" in event_text

    def test_content_chunk_emits_delta(self) -> None:
        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-2"
        mock_chunk.model = "qwen3.7-plus"
        mock_chunk.usage = None

        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = None
        delta = MagicMock()
        delta.role = None
        delta.content = "Hello"
        delta.model_dump.return_value = {"content": "Hello"}
        choice.delta = delta
        mock_chunk.choices = [choice]

        events = _openai_chunk_to_anthropic_sse(mock_chunk)
        event_text = "\n".join(events)

        assert "content_block_delta" in event_text
        assert "Hello" in event_text

    def test_finish_chunk_emits_stop_events(self) -> None:
        mock_chunk = MagicMock()
        mock_chunk.id = "chunk-3"
        mock_chunk.model = "qwen3.7-plus"
        mock_chunk.usage = MagicMock()
        mock_chunk.usage.prompt_tokens = 10
        mock_chunk.usage.completion_tokens = 20

        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = "stop"
        delta = MagicMock()
        delta.role = None
        delta.content = None
        delta.model_dump.return_value = {}
        choice.delta = delta
        mock_chunk.choices = [choice]

        # message_delta/message_stop are deferred to final_sse_events so the
        # trailing usage-only chunk (choices: []) can be folded in first.
        state: dict[str, Any] = {}
        events = _openai_chunk_to_anthropic_sse(mock_chunk, state=state)
        assert "content_block_stop" in "\n".join(events)

        event_text = "\n".join(events + final_sse_events(state))
        assert "message_delta" in event_text
        assert "message_stop" in event_text
        assert "end_turn" in event_text
        # The usage on the finish chunk is still carried through.
        assert '"input_tokens": 10' in event_text


# ── Test: Session model locking ─────────────────────────────────────────────


def test_get_or_lock_model_first_call(router: ProxyRouter, session: Session) -> None:
    model = router._get_or_lock_model(session)
    assert model == "qwen3.7-plus"
    assert session.model_chosen == "qwen3.7-plus"
    assert session.metadata["model"] == "qwen3.7-plus"
    assert session.metadata["model_source"] == "cold-start-always-cheap"


def test_get_or_lock_model_reuses(router: ProxyRouter, session: Session) -> None:
    session.model_chosen = "custom-model"
    model = router._get_or_lock_model(session)
    assert model == "custom-model"


def test_get_or_lock_model_logs_decision_once(
    router: ProxyRouter, session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        model = router._get_or_lock_model(session)

    routed_records = [r for r in caplog.records if "routed to" in r.getMessage()]
    assert len(routed_records) == 1
    message = routed_records[0].getMessage()
    assert session.session_id in message
    assert model in message
    assert "cold-start-always-cheap" in message


def test_get_or_lock_model_no_log_on_reuse(
    router: ProxyRouter, session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        router._get_or_lock_model(session)
        caplog.clear()
        router._get_or_lock_model(session)

    routed_records = [r for r in caplog.records if "routed to" in r.getMessage()]
    assert len(routed_records) == 0


# ── Test: Routing with mocked upstream ──────────────────────────────────────


@pytest.mark.asyncio
async def test_route_chat_non_streaming(router: ProxyRouter, session: Session) -> None:
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": False}

    mock_response = _make_mock_chat_response()

    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response
        result, model_name, reason = await router.route_chat_completion(body, session)

    assert model_name == "qwen3.7-plus"
    assert result["choices"][0]["message"]["content"] == "Hello back"


@pytest.mark.asyncio
async def test_route_chat_streaming(router: ProxyRouter, session: Session) -> None:
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": True}

    async def mock_stream() -> AsyncGenerator[MagicMock, None]:
        for i in range(2):
            chunk = MagicMock()
            chunk.id = f"chunk-{i}"
            chunk.model = "qwen3.7-plus"
            chunk.usage = None
            choice = MagicMock()
            choice.index = 0
            choice.finish_reason = None
            delta = MagicMock()
            delta.role = "assistant" if i == 0 else None
            delta.content = "" if i == 0 else f"chunk-{i}"
            delta.model_dump.return_value = (
                {"role": "assistant"} if i == 0 else {"content": f"chunk-{i}"}
            )
            choice.delta = delta
            chunk.choices = [choice]
            yield chunk
        # Final chunk with finish
        final = MagicMock()
        final.id = "chunk-final"
        final.model = "qwen3.7-plus"
        final.usage = MagicMock()
        final.usage.prompt_tokens = 5
        final.usage.completion_tokens = 10
        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = "stop"
        delta = MagicMock()
        delta.role = None
        delta.content = None
        delta.model_dump.return_value = {}
        choice.delta = delta
        final.choices = [choice]
        yield final

    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_stream()
        result, model_name, reason = await router.route_chat_completion(body, session)

    assert model_name == "qwen3.7-plus"
    assert reason == "stream:qwen3.7-plus"
    # result should be an async generator
    chunks = [c async for c in result]
    assert len(chunks) > 0
    assert all(isinstance(c, bytes) for c in chunks)
    decoded = b"".join(chunks).decode("utf-8")
    assert "data: [DONE]" in decoded


@pytest.mark.asyncio
async def test_route_messages_non_streaming(router: ProxyRouter, session: Session) -> None:
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": False}

    mock_response = MagicMock()
    mock_response.id = "cmpl-1"
    mock_response.model = "qwen3.7-plus"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 5
    mock_response.usage.completion_tokens = 10
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = "Hello back"
    choice.message.role = "assistant"
    mock_response.choices = [choice]

    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response
        result, model_name, reason = await router.route_messages(body, session)

    assert model_name == "qwen3.7-plus"
    assert isinstance(result, dict)
    assert result["type"] == "message"
    assert result["content"][0]["text"] == "Hello back"


@pytest.mark.asyncio
async def test_route_messages_streaming(router: ProxyRouter, session: Session) -> None:
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": True}

    async def mock_stream() -> AsyncGenerator[MagicMock, None]:
        chunk = MagicMock()
        chunk.id = "chunk-1"
        chunk.model = "qwen3.7-plus"
        chunk.usage = None
        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = None
        delta = MagicMock()
        delta.role = "assistant"
        delta.content = "Hello"
        delta.model_dump.return_value = {"role": "assistant", "content": "Hello"}
        choice.delta = delta
        chunk.choices = [choice]
        yield chunk

    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_stream()
        result, model_name, reason = await router.route_messages(body, session)

    assert reason == "stream:qwen3.7-plus"
    chunks = [c async for c in result]
    assert len(chunks) > 0
    decoded = b"".join(chunks).decode("utf-8")
    assert "message_start" in decoded


@pytest.mark.asyncio
async def test_streaming_request_asks_for_usage(router: ProxyRouter, session: Session) -> None:
    # The OpenAI SDK omits streaming usage unless stream_options.include_usage is set.
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": True}

    async def mock_stream() -> AsyncGenerator[MagicMock, None]:
        chunk = MagicMock()
        chunk.usage = None
        chunk.choices = []
        yield chunk

    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_stream()
        result, _model, _reason = await router.route_chat_completion(body, session)
        _ = [c async for c in result]

    assert mock_acompletion.call_args.kwargs.get("stream_options") == {"include_usage": True}


@pytest.mark.asyncio
async def test_streaming_cache_tax_from_trailing_usage_chunk(
    router: ProxyRouter, session: Session
) -> None:
    # Regression for the litellm→OpenAI migration: content chunks carry NO usage;
    # usage arrives ONLY in a trailing choices==[] chunk (the include_usage shape).
    # The session cache-tax/prompt-length metric must still be captured.
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": True}

    async def mock_stream() -> AsyncGenerator[MagicMock, None]:
        content = MagicMock()
        content.usage = None
        content.model = "qwen3.7-plus"
        choice = MagicMock()
        choice.finish_reason = None
        delta = MagicMock()
        delta.role = "assistant"
        delta.content = "Hello"
        delta.model_dump.return_value = {"role": "assistant", "content": "Hello"}
        choice.delta = delta
        content.choices = [choice]
        yield content
        # Trailing usage-only chunk: no choices, real usage with cached tokens.
        usage_chunk = MagicMock()
        usage_chunk.choices = []
        usage = MagicMock()
        usage.prompt_tokens = 100
        details = MagicMock()
        details.cached_tokens = 40
        usage.prompt_tokens_details = details
        usage_chunk.usage = usage
        yield usage_chunk

    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_stream()
        result, _model, _reason = await router.route_chat_completion(body, session)
        _ = [c async for c in result]

    assert session.prompt_length_tokens == 100
    assert session.cache_tax == 40.0


@pytest.mark.asyncio
async def test_cache_control_block_passes_through(router: ProxyRouter, session: Session) -> None:
    # Cache-safety is the product spine: a cache_control marker inside a message
    # content block must survive untouched into the upstream request.
    body = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "ctx", "cache_control": {"type": "ephemeral"}}
                ],
            },
            {"role": "user", "content": "Hi"},
        ],
        "stream": False,
    }
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _make_mock_chat_response()
        await router.route_chat_completion(body, session)

    sent = mock_acompletion.call_args.kwargs["messages"]
    assert sent[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


# ── Test: Error handling ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fail_fast_on_auth_error(router: ProxyRouter, session: Session) -> None:
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": False}

    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        auth_err = Exception("Authentication failed")
        auth_err.status_code = 401  # type: ignore[attr-defined]
        mock_acompletion.side_effect = auth_err

        with pytest.raises(UpstreamError) as excinfo:
            await router.route_chat_completion(body, session)

        assert excinfo.value.status_code == 401  # fail fast, original status preserved


@pytest.mark.asyncio
async def test_retry_then_fallback(router: ProxyRouter, session: Session) -> None:
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": False}

    mock_response = MagicMock()
    mock_response.id = "cmpl-fallback"
    mock_response.model = "deepseek-v4-flash"
    mock_response.usage.prompt_tokens = 5
    mock_response.usage.completion_tokens = 10
    choice = MagicMock()
    choice.index = 0
    choice.finish_reason = "stop"
    choice.message.content = "from fallback"
    choice.message.role = "assistant"
    mock_response.choices = [choice]
    mock_response.model_dump.return_value = {
        "id": "cmpl-fallback",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "from fallback"},
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        "model": "deepseek-v4-flash",
    }

    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        rate_err = Exception("Rate limit")
        rate_err.status_code = 429  # type: ignore[attr-defined]
        # First call fails (qwen3.7-plus), second succeeds (deepseek-v4-flash)
        mock_acompletion.side_effect = [rate_err, rate_err, rate_err, mock_response]

        # qwen3.7-plus will be retried 2x, then fallback to deepseek-v4-flash
        result, model_name, reason = await router.route_chat_completion(body, session)

    assert model_name == "deepseek-v4-flash"
    assert result["choices"][0]["message"]["content"] == "from fallback"


@pytest.mark.asyncio
async def test_all_models_exhausted(router: ProxyRouter, session: Session) -> None:
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": False}

    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        err = Exception("Server error")
        err.status_code = 500  # type: ignore[attr-defined]
        mock_acompletion.side_effect = err

        with pytest.raises(UpstreamError) as excinfo:
            await router.route_chat_completion(body, session)

        assert "All models exhausted" in str(excinfo.value)


# ── Test: Server endpoints ──────────────────────────────────────────────────


def test_health_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_models_endpoint_lists_registry() -> None:
    """GET /v1/models returns an OpenAI-shaped list over the local registry."""
    with TestClient(app) as client:
        resp = client.get("/v1/models")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["object"] == "list"
    ids = [row["id"] for row in payload["data"]]
    # The shipped registry is non-empty; every row carries the OpenAI shape.
    assert ids, "registry should list at least one model"
    assert all(row["object"] == "model" and row["owned_by"] == "shunt" for row in payload["data"])


def test_models_endpoint_needs_no_auth() -> None:
    """The stub is unauthenticated — clients discover models before they hold a key."""
    with TestClient(app) as client:
        resp = client.get("/v1/models")  # no Authorization / x-api-key header
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer dummy"}, {"x-api-key": ""}],
    ids=["absent", "dummy-bearer", "empty-x-api-key"],
)
def test_requests_route_without_client_api_key(headers: dict[str, str]) -> None:
    """Shunt holds the real provider keys, so an absent/dummy/empty client key still routes.

    Locks the contract on the ROUTING path (not just /v1/models): a client may send a
    dummy key or none at all, and neither may be rejected.
    """
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": False}
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _make_mock_chat_response()
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions", json=body, headers=headers)
    assert resp.status_code == 200
    assert "X-Shunt-Decision" in resp.headers


def test_cold_start_default_model_is_in_the_shipped_registry() -> None:
    """The hardcoded cold-start default must exist in the registry, or a fresh session
    locks to an unknown model and every first request fails at routing time."""
    assert _DEFAULT_MODEL in ModelPool().model_names()


def _make_mock_chat_response() -> MagicMock:
    """Build a mock OpenAI ChatCompletion for non-streaming chat."""
    mock_response = MagicMock()
    mock_response.id = "cmpl-1"
    mock_response.model = "qwen3.7-plus"
    mock_response.usage.prompt_tokens = 5
    mock_response.usage.completion_tokens = 10
    choice = MagicMock()
    choice.index = 0
    choice.finish_reason = "stop"
    choice.message.content = "Hello back"
    choice.message.role = "assistant"
    mock_response.choices = [choice]
    # ModelResponse.model_dump returns a JSON-serialisable dict
    mock_response.model_dump.return_value = {
        "id": "cmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hello back"},
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        "model": "qwen3.7-plus",
    }
    return mock_response


def test_chat_completions_returns_headers() -> None:
    """Verify X-Shunt-Decision and X-Shunt-Session-Id appear in stub response."""
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": False}
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _make_mock_chat_response()
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert "X-Shunt-Decision" in resp.headers
    assert "X-Shunt-Session-Id" in resp.headers


def test_messages_returns_headers() -> None:
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": False}
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _make_mock_chat_response()
        with TestClient(app) as client:
            resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 200
    assert "X-Shunt-Decision" in resp.headers
    assert "X-Shunt-Session-Id" in resp.headers


def test_session_id_persists_across_requests() -> None:
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": False}
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _make_mock_chat_response()
        with TestClient(app) as client:
            resp1 = client.post("/v1/chat/completions", json=body)
            assert resp1.status_code == 200
            sid1 = resp1.headers.get("X-Shunt-Session-Id")

            resp2 = client.post("/v1/chat/completions", json=body)
            assert resp2.status_code == 200
            sid2 = resp2.headers.get("X-Shunt-Session-Id")

    assert sid1 == sid2


def test_streaming_returns_event_stream() -> None:
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": True}

    async def _mock_stream() -> AsyncGenerator[MagicMock, None]:
        chunk = MagicMock()
        chunk.id = "chunk-1"
        chunk.model = "qwen3.7-plus"
        chunk.usage = None
        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = "stop"
        delta = MagicMock()
        delta.role = "assistant"
        delta.content = "Hello"
        delta.model_dump.return_value = {"role": "assistant", "content": "Hello"}
        choice.delta = delta
        chunk.choices = [choice]
        yield chunk

    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _mock_stream()
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("text/event-stream")


def test_messages_streaming_returns_event_stream() -> None:
    body = {"messages": [{"role": "user", "content": "Hi"}], "stream": True}

    async def _mock_stream() -> AsyncGenerator[MagicMock, None]:
        chunk = MagicMock()
        chunk.id = "chunk-1"
        chunk.model = "qwen3.7-plus"
        chunk.usage = None
        choice = MagicMock()
        choice.index = 0
        choice.finish_reason = "stop"
        delta = MagicMock()
        delta.role = "assistant"
        delta.content = "Hello"
        delta.model_dump.return_value = {"role": "assistant", "content": "Hello"}
        choice.delta = delta
        chunk.choices = [choice]
        yield chunk

    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _mock_stream()
        with TestClient(app) as client:
            resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("text/event-stream")
