"""ProxyRouter — proxies LLM requests to upstream providers via the OpenAI SDK,
with retry, fallback, and session tracking (OpenAI-compatible endpoints only).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from typing import Any

import openai
from openai import AsyncOpenAI

from shunt.models import ModelConfig, ModelPool
from shunt.session import Session, SessionManager

logger = logging.getLogger(__name__)

# Default cheap model — cold-start placeholder (kNN replaces this later)
_DEFAULT_MODEL = "qwen3.7-plus"


class UpstreamError(Exception):
    """Raised when all retries to an upstream provider are exhausted."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        self.status_code = status_code
        super().__init__(message)


@functools.lru_cache(maxsize=32)
def _client_for(base_url: str, api_key: str) -> AsyncOpenAI:
    """Cached OpenAI-compatible async client keyed by (base_url, api_key)."""
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


async def _acompletion(config: ModelConfig, **openai_kwargs: Any) -> Any:
    """One upstream chat-completion via the OpenAI SDK (the single mockable seam).

    The provider is chosen by ``base_url``; the model string sent is ``model_id``
    (the provider-side id, e.g. ``alibaba/qwen3.7-plus``), falling back to ``name``.
    """
    api_key = os.environ.get(config.api_key_env_var) or "sk-missing"
    client = _client_for(config.base_url, api_key)
    model_id = config.model_id or config.name
    return await client.chat.completions.create(model=model_id, **openai_kwargs)


_RETRYABLE_OPENAI: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


def _is_retryable(exc: Exception) -> bool:
    """Return True if *exc* represents a transient upstream error worth retrying."""
    err_str = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if status is not None:
        if status in (400, 401, 403):
            return False
        if status in (408, 429, 500, 502, 503, 504):
            return True
    if isinstance(exc, _RETRYABLE_OPENAI):
        return True
    if isinstance(exc, openai.APIStatusError):
        api_status = getattr(exc, "status_code", None)
        if api_status is not None and api_status in (408, 429, 500, 502, 503, 504):
            return True
    patterns = (
        "rate_limit",
        "rate limit",
        "timeout",
        "timed out",
        "server error",
        "internal server error",
        "service unavailable",
    )
    return any(pattern in err_str for pattern in patterns)


# ── Format conversion helpers ──────────────────────────────────────────────


def _anthropic_request_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic /v1/messages request body to OpenAI-compatible kwargs.

    Preserves cache_control markers on content blocks (passthrough).
    """
    messages: list[dict[str, Any]] = []
    system = body.get("system")

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = [
                block
                for block in content
                if isinstance(block, dict) and block.get("type") != "image"
            ]
        messages.append({"role": role, "content": content})

    if system:
        if isinstance(system, list):
            messages.insert(0, {"role": "system", "content": system})
        else:
            messages.insert(0, {"role": "system", "content": system})

    kwargs: dict[str, Any] = {
        "messages": messages,
        "stream": body.get("stream", False),
    }
    if kwargs["stream"]:
        # Ask the OpenAI SDK to emit a trailing usage chunk (else streaming
        # cache-tax/usage is silently 0 — see _track_cache_tax).
        kwargs["stream_options"] = {"include_usage": True}
    if "max_tokens" in body:
        kwargs["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        kwargs["temperature"] = body["temperature"]
    if "top_p" in body:
        kwargs["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        kwargs["stop"] = body["stop_sequences"]

    return kwargs


def _openai_chunk_to_anthropic_sse(
    chunk: Any,
    *,
    message_id: str | None = None,
    model_name: str | None = None,
) -> list[str]:
    """Convert an OpenAI-format streaming chunk to Anthropic SSE event text(s).

    Returns a list of *event lines* — each entry is one ``event: ...\\ndata: ...\\n``
    block, ready to join with ``\\n`` and encode.
    """
    events: list[str] = []

    if not hasattr(chunk, "choices") or not chunk.choices:
        return events

    choice = chunk.choices[0]
    delta = choice.delta if hasattr(choice, "delta") else {}
    finish = choice.finish_reason if hasattr(choice, "finish_reason") else None

    delta_dict = delta.model_dump(exclude_none=True) if hasattr(delta, "model_dump") else {}
    content = delta_dict.get("content", "")
    role = delta_dict.get("role")

    mid = message_id or f"msg_{int(time.time() * 1000)}"
    mn = model_name or (chunk.model if hasattr(chunk, "model") and chunk.model else "")

    if role:
        msg_start = {
            "type": "message_start",
            "message": {
                "id": mid,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": mn,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {},
            },
        }
        events.append(f"event: message_start\ndata: {json.dumps(msg_start)}\n")
        cb_start = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        events.append(f"event: content_block_start\ndata: {json.dumps(cb_start)}\n")

    if content:
        cb_delta = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": content},
        }
        events.append(f"event: content_block_delta\ndata: {json.dumps(cb_delta)}\n")

    if finish:
        finish_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "content_filter": "content_filter",
        }
        anthropic_finish = finish_map.get(finish, finish)
        events.append('event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n')
        usage = {}
        if hasattr(chunk, "usage") and chunk.usage:
            usage = {
                "input_tokens": chunk.usage.prompt_tokens or 0,
                "output_tokens": chunk.usage.completion_tokens or 0,
            }
        msg_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": anthropic_finish, "stop_sequence": None},
            "usage": usage,
        }
        events.append(f"event: message_delta\ndata: {json.dumps(msg_delta)}\n")
        events.append('event: message_stop\ndata: {"type":"message_stop"}\n')

    return events


def _openai_response_to_anthropic(response: Any) -> dict[str, Any]:
    """Convert an OpenAI-format *ModelResponse* to an Anthropic /v1/messages response dict."""
    choice = response.choices[0] if response.choices and len(response.choices) > 0 else None
    message = choice.message if choice else None

    content_text = (message.content or "") if message else ""
    usage_in = 0
    usage_out = 0
    if hasattr(response, "usage") and response.usage:
        usage_in = response.usage.prompt_tokens or 0
        usage_out = response.usage.completion_tokens or 0

    finish_reason = None
    if choice and hasattr(choice, "finish_reason") and choice.finish_reason:
        fr_map = {"stop": "end_turn", "length": "max_tokens", "content_filter": "content_filter"}
        finish_reason = fr_map.get(choice.finish_reason, choice.finish_reason)

    return {
        "id": getattr(response, "id", None) or f"msg_{int(time.time() * 1000)}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content_text or ""}],
        "model": getattr(response, "model", None) or "",
        "stop_reason": finish_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
    }


class ProxyRouter:
    """Routes requests to upstream LLM providers via the OpenAI SDK.

    Provides retry logic, model fallback (via *ModelPool*), session-aware
    model locking, and format conversion for the Anthropic surface.
    """

    def __init__(
        self,
        model_pool: ModelPool,
        session_manager: SessionManager,
        retry_count: int = 3,
    ) -> None:
        self._pool = model_pool
        self._sessions = session_manager
        self.retry_count = retry_count

    # ── Public API ──────────────────────────────────────────────────────────

    async def route_chat_completion(
        self,
        body: dict[str, Any],
        session: Session,
    ) -> tuple[dict[str, Any] | AsyncGenerator[bytes, None], str, str]:
        """Route an OpenAI-format request, returning *(response_or_stream,
        model_name, decision_reason)* — a byte-generator when streaming, else a
        JSON-serialisable dict.
        """
        return await self._route(body, session, output_format="openai")

    async def route_messages(
        self,
        body: dict[str, Any],
        session: Session,
    ) -> tuple[dict[str, Any] | AsyncGenerator[bytes, None], str, str]:
        """Route an Anthropic-format /v1/messages request.

        Returns *(response_data_or_stream, model_name, decision_reason)*,
        with the response converted back to Anthropic format.
        """
        # Convert request format
        openai_kwargs = _anthropic_request_to_openai(body)
        result = await self._route_with_fallback(openai_kwargs, session)
        response, model_name = result

        stream = body.get("stream", False)
        if stream:
            tracked = self._track_cache_tax(response, session)
            gen: AsyncGenerator[bytes, None] = self._anthropic_stream(tracked)
            return gen, model_name, f"stream:{model_name}"

        self._update_cache_tax(session, response)
        self._log_cache_metrics(session)
        anthropic_body = _openai_response_to_anthropic(response)
        return anthropic_body, model_name, model_name

    # ── Internal routing ────────────────────────────────────────────────────

    async def _route(
        self,
        body: dict[str, Any],
        session: Session,
        output_format: str = "openai",
    ) -> tuple[dict[str, Any] | AsyncGenerator[bytes, None], str, str]:
        """Common routing path shared by both endpoint formats."""
        stream = body.get("stream", False)
        msg = body.get("messages", [])

        openai_kwargs: dict[str, Any] = {
            "messages": msg,
            "stream": stream,
        }
        if stream:
            # The OpenAI SDK omits usage from streamed chunks unless asked; without
            # this, cache-tax/prompt-length read 0 for all streaming traffic.
            openai_kwargs["stream_options"] = {"include_usage": True}
        option_keys = (
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "frequency_penalty",
            "presence_penalty",
        )
        for key in option_keys:
            if key in body:
                openai_kwargs[key] = body[key]

        response, model_name = await self._route_with_fallback(openai_kwargs, session)

        if stream:
            tracked = self._track_cache_tax(response, session)
            gen: AsyncGenerator[bytes, None] = self._openai_stream(tracked)
            return gen, model_name, f"stream:{model_name}"

        self._update_cache_tax(session, response)
        self._log_cache_metrics(session)
        if hasattr(response, "model_dump"):
            return response.model_dump(), model_name, model_name
        return response, model_name, model_name

    async def _route_with_fallback(
        self,
        openai_kwargs: dict[str, Any],
        session: Session,
    ) -> tuple[Any, str]:
        """Route *openai_kwargs* through the model chain with retry+fallback.

        Returns *(openai_response | async_generator, selected_model_name)*.
        """
        model_name = self._get_or_lock_model(session)
        chain = self._pool.fallback_chain(model_name)
        last_error: Exception | None = None

        for candidate in chain:
            if not self._pool.is_healthy(candidate):
                continue

            try:
                return await self._try_model(candidate, openai_kwargs), candidate
            except UpstreamError as exc:
                if exc.status_code in (400, 401, 403):
                    raise  # fail fast — auth / bad request
                last_error = exc
                self._pool.mark_unhealthy(candidate)
                logger.warning("Model %s failed with %s; trying next", candidate, exc)
                continue
            except Exception as exc:
                if isinstance(exc, UpstreamError) and exc.status_code in (400, 401, 403):
                    raise
                last_error = exc
                self._pool.mark_unhealthy(candidate)
                logger.warning("Model %s failed with %s; trying next", candidate, exc)
                continue

        raise UpstreamError(
            f"All models exhausted. Last error: {last_error}",
            status_code=getattr(last_error, "status_code", 502) if last_error else 502,
        )

    async def _try_model(self, model_name: str, openai_kwargs: dict[str, Any]) -> Any:
        """Attempt routing through *model_name* with retries."""
        config = self._pool.get_model(model_name)
        if config is None:
            raise UpstreamError(f"Unknown model: {model_name}", status_code=400)

        for attempt in range(self.retry_count):
            try:
                return await _acompletion(config, **openai_kwargs)
            except Exception as exc:
                if not _is_retryable(exc):
                    # A real HTTP status (e.g. 400/401/403) fails fast; an unknown
                    # error with no status defaults to 502 so fallback can still try.
                    status = getattr(exc, "status_code", 502)
                    raise UpstreamError(str(exc), status_code=status) from exc
                if attempt < self.retry_count - 1:
                    wait = 2.0**attempt * 0.5
                    logger.info(
                        "Retry %d/%d for %s in %.1fs",
                        attempt + 1,
                        self.retry_count,
                        model_name,
                        wait,
                    )
                    await asyncio.sleep(wait)

        raise UpstreamError(f"All {self.retry_count} retries exhausted for {model_name}")

    # ── Session model locking ───────────────────────────────────────────────

    def _get_or_lock_model(self, session: Session) -> str:
        """Return the model chosen for *session*, locking it on first access."""
        if session.model_chosen:
            return session.model_chosen
        model_name = _DEFAULT_MODEL
        session.model_chosen = model_name
        session.metadata["model"] = model_name
        session.metadata["model_source"] = "cold-start-always-cheap"
        return model_name

    # ── Cache-tax measurement ──────────────────────────────────────────────

    @staticmethod
    def _extract_cache_tax(response: Any) -> float:
        """Extract cached prompt tokens from an OpenAI response usage."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0.0
        details = getattr(usage, "prompt_tokens_details", None)
        return float(getattr(details, "cached_tokens", 0) or 0)

    async def _track_cache_tax(
        self,
        gen: Any,
        session: Session,
    ) -> AsyncGenerator[Any, None]:
        """Wrap a streaming generator, extracting cache tax from the final chunk."""
        last_usage = None
        async for chunk in gen:
            if hasattr(chunk, "usage") and chunk.usage:
                last_usage = chunk.usage
            yield chunk
        if last_usage is not None:
            details = getattr(last_usage, "prompt_tokens_details", None)
            cache_tax = float(getattr(details, "cached_tokens", 0) or 0)
            if cache_tax > 0:
                session.cache_tax = getattr(session, "cache_tax", 0.0) + cache_tax
            prompt_tokens = getattr(last_usage, "prompt_tokens", 0) or 0
            if prompt_tokens > 0:
                session.prompt_length_tokens = prompt_tokens
            self._log_cache_metrics(session)

    @staticmethod
    def _extract_prompt_tokens(response: Any) -> int:
        """Extract prompt_tokens from an OpenAI response usage."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0
        return getattr(usage, "prompt_tokens", 0) or 0

    def _update_cache_tax(self, session: Session, response: Any) -> None:
        """Extract cache metrics from a completed response and update session."""
        cache_tax = self._extract_cache_tax(response)
        prompt_tokens = self._extract_prompt_tokens(response)
        if cache_tax > 0:
            session.cache_tax = getattr(session, "cache_tax", 0.0) + cache_tax
        if prompt_tokens > 0:
            session.prompt_length_tokens = prompt_tokens

    def _log_cache_metrics(self, session: Session) -> None:
        """Log cache hit ratio and cache tax for a session request."""
        cache_tax = getattr(session, "cache_tax", 0.0)
        prompt_tokens = getattr(session, "prompt_length_tokens", 0)
        hit_ratio = min(1.0, cache_tax / prompt_tokens) if prompt_tokens > 0 else 0.0
        if cache_tax > 0 or prompt_tokens > 0:
            logger.info(
                "Session %s cache_tax=%.0f prompt_tokens=%d cache_hit_ratio=%.4f",
                session.session_id,
                cache_tax,
                prompt_tokens,
                hit_ratio,
            )

    # ── Streaming helpers ───────────────────────────────────────────────────

    async def _openai_stream(self, gen: Any) -> AsyncGenerator[bytes, None]:
        """Wrap an OpenAI streaming async generator into OpenAI-format SSE bytes."""
        async for chunk in gen:
            if hasattr(chunk, "model_dump"):
                chunk_dict = chunk.model_dump(exclude_none=True)
            else:
                chunk_dict = chunk
            yield f"data: {json.dumps(chunk_dict, default=str)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def _anthropic_stream(self, gen: Any) -> AsyncGenerator[bytes, None]:
        """Wrap an OpenAI streaming async generator into Anthropic-format SSE bytes."""
        message_id = f"msg_{int(time.time() * 1000)}"
        model_name = ""
        role_seen = False

        async for chunk in gen:
            if not role_seen and hasattr(chunk, "model") and chunk.model:
                model_name = chunk.model
            events = _openai_chunk_to_anthropic_sse(
                chunk,
                message_id=message_id,
                model_name=model_name,
            )
            for event in events:
                yield event.encode("utf-8")
                yield b"\n"
            if events:
                yield b"\n"

            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta if hasattr(choice, "delta") else None
                if delta and hasattr(delta, "role") and delta.role:
                    role_seen = True
