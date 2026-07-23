"""ProxyRouter — proxies LLM requests to upstream providers via the OpenAI SDK,
with retry, fallback, and session tracking (OpenAI-compatible endpoints only).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import os
import threading
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Final

import openai
from openai import AsyncOpenAI
from starlette.concurrency import run_in_threadpool

from shunt.models import ModelConfig, ModelPool
from shunt.models.config import model_fingerprint as _resolve_fingerprint
from shunt.proxy.wire_signals import WireSignalCollector
from shunt.session import Session, SessionManager

if TYPE_CHECKING:
    from shunt.router.engine import RouterEngine

logger = logging.getLogger(__name__)

# Default cheap model — cold-start placeholder (kNN replaces this later)
_DEFAULT_MODEL = "qwen3.7-plus"

# Request fields forwarded upstream. `tools`/`tool_choice` are the load-bearing entries:
# Shunt sits in front of CODING AGENTS, which are tool-calling clients, and omitting them
# meant every agent request reached the model with its tools stripped. The model then
# improvised tool calls as plain text (observed live: opencode emitted a raw
# "<tool_calls>{...}" string instead of calling Read), so the agent could not act at all.
#
# An allow-list keeps client-specific junk and shunt-internal keys off the upstream call,
# but a SILENT allow-list is what hid this for so long — hence _log_dropped_keys.
_FORWARDED_OPENAI_KEYS: tuple[str, ...] = (
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "seed",
    "n",
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "user",
    "reasoning_effort",
)

# Keys the proxy consumes itself, so they are dropped on purpose and never warned about.
_CONSUMED_KEYS: frozenset[str] = frozenset({"model", "messages", "stream", "stream_options"})


def _log_dropped_keys(body: dict[str, Any], forwarded: dict[str, Any]) -> None:
    """Warn once per request about client fields the proxy did not pass upstream."""
    dropped = sorted(set(body) - set(forwarded) - _CONSUMED_KEYS)
    if dropped:
        logger.warning(
            "Dropping unforwarded request fields: %s. If a client needs one, add it to "
            "_FORWARDED_OPENAI_KEYS — a silently dropped field looks like a model defect.",
            ", ".join(dropped),
        )


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


def _block_text(block: Any) -> str:
    """Extract text from one content block (str passthrough, dict → its ``text``)."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        return str(block.get("text", ""))
    return ""


def _prompt_text_from_messages(messages: list[dict[str, Any]]) -> str:
    """Flatten OpenAI-format messages into a single string, in wire order."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, list):
            parts.extend(_block_text(block) for block in content)
        else:
            parts.append(_block_text(content))
    return "\n".join(p for p in parts if p)


# Roles that carry the user's actual task. `system` is excluded on purpose: a coding
# agent's system prompt alone runs ~29k chars, which is 7x the embedder's clip window.
_TASK_ROLES: Final[frozenset[str]] = frozenset({"user", "tool"})


def _routing_text_from_messages(messages: list[dict[str, Any]]) -> str:
    """Task-bearing text for the routing embedding, most-recent turn first."""
    # The embedder clips to its HEAD, so wire order is the wrong order here: with a
    # system prompt in slot 0 the clip window closed before the task began, and every
    # session embedded to the same vector. Recency-first keeps the task inside it.
    parts: list[str] = []
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") not in _TASK_ROLES:
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            parts.extend(_block_text(block) for block in content)
        else:
            parts.append(_block_text(content))
    text = "\n".join(p for p in parts if p)
    # A body with no task-bearing role at all is better routed on something than on
    # nothing, so fall back to the flat wire-order text rather than embedding "".
    return text or _prompt_text_from_messages(messages)


# ── Format conversion helpers ──────────────────────────────────────────────


def _tool_result_to_text(content: Any) -> str:
    """Flatten an Anthropic tool_result payload into OpenAI's plain string content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts) or json.dumps(content)
    return "" if content is None else json.dumps(content)


def _anthropic_message_to_openai(role: str, content: Any) -> list[dict[str, Any]]:
    """One Anthropic message -> the OpenAI message(s) it corresponds to.

    A tool turn expands: Anthropic carries tool results as blocks inside a *user*
    message, while OpenAI needs a separate `tool` message per result.
    """
    if not isinstance(content, list):
        return [{"role": role, "content": content}]

    passthrough: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []

    for block in content:
        if not isinstance(block, dict) or block.get("type") == "image":
            continue
        if block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        # OpenAI wants arguments as a JSON *string*.
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
        elif block.get("type") == "tool_result":
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": _tool_result_to_text(block.get("content")),
                }
            )
        else:
            # Text and everything else passes through untouched, so cache_control
            # markers on those blocks survive.
            passthrough.append(block)

    messages: list[dict[str, Any]] = []
    # Results first: OpenAI requires each `tool` message to follow the assistant turn
    # that requested it, before any new user text.
    messages.extend(tool_messages)
    if tool_calls:
        messages.append(
            {
                "role": role,
                "content": passthrough or None,
                "tool_calls": tool_calls,
            }
        )
    elif passthrough:
        messages.append({"role": role, "content": passthrough})
    return messages


def _anthropic_request_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic /v1/messages request body to OpenAI-compatible kwargs.

    Preserves cache_control markers on content blocks (passthrough).
    """
    messages: list[dict[str, Any]] = []
    system = body.get("system")

    for msg in body.get("messages", []):
        messages.extend(
            _anthropic_message_to_openai(msg.get("role", "user"), msg.get("content", ""))
        )

    if system:
        # Anthropic allows `system` as a string or a block list; the OpenAI shape
        # accepts either as message content, so both forms pass through identically.
        messages.insert(0, {"role": "system", "content": system})

    kwargs: dict[str, Any] = {
        "messages": messages,
        "stream": body.get("stream", False),
    }
    if kwargs["stream"]:
        # Ask the OpenAI SDK to emit a trailing usage chunk (else streaming
        # cache-tax/usage is silently 0 — see _track_cache_tax).
        kwargs["stream_options"] = {"include_usage": True}
    for source, target in (
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("stop_sequences", "stop"),
    ):
        if source in body:
            kwargs[target] = body[source]
    if tools := _anthropic_tools_to_openai(body.get("tools")):
        kwargs["tools"] = tools
    if (choice := _anthropic_tool_choice_to_openai(body.get("tool_choice"))) is not None:
        kwargs["tool_choice"] = choice

    return kwargs


def _anthropic_tools_to_openai(tools: Any) -> list[dict[str, Any]]:
    """Map Anthropic tool declarations onto the OpenAI function-tool shape."""
    # Anthropic: {name, description, input_schema}. OpenAI nests the same information
    # under {type: function, function: {name, description, parameters}}. Without this a
    # Claude Code request reached the upstream model with no tools at all.
    if not isinstance(tools, list):
        return []
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or "name" not in tool:
            continue
        function: dict[str, Any] = {"name": tool["name"]}
        if "description" in tool:
            function["description"] = tool["description"]
        function["parameters"] = tool.get("input_schema", {"type": "object", "properties": {}})
        converted.append({"type": "function", "function": function})
    return converted


def _anthropic_tool_choice_to_openai(choice: Any) -> Any:
    """Map Anthropic's tool_choice object onto OpenAI's string/object form."""
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "tool" and "name" in choice:
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


def _tool_call_delta_events(
    call: dict[str, Any],
    tool_blocks: dict[int, int],
    state: dict[str, Any] | None,
) -> list[str]:
    """SSE events for one streamed tool-call fragment (opening its block if new)."""
    events: list[str] = []
    call_index = call.get("index", 0)
    function = call.get("function") or {}

    if call_index not in tool_blocks:
        # Index 0 is the text block, opened eagerly with message_start, so tool blocks
        # start at 1. `next_index` lives in state so several calls do not collide.
        block_index = state.setdefault("next_index", 1) if state is not None else 1
        if state is not None:
            state["next_index"] = block_index + 1
        tool_blocks[call_index] = block_index
        start: dict[str, Any] = {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{block_index}",
                "name": function.get("name") or "",
                "input": {},
            },
        }
        events.append(f"event: content_block_start\ndata: {json.dumps(start)}\n")

    if arguments := function.get("arguments"):
        # Anthropic streams tool arguments as raw JSON text fragments, not decoded.
        delta = {
            "type": "content_block_delta",
            "index": tool_blocks[call_index],
            "delta": {"type": "input_json_delta", "partial_json": arguments},
        }
        events.append(f"event: content_block_delta\ndata: {json.dumps(delta)}\n")
    return events


def _openai_chunk_to_anthropic_sse(
    chunk: Any,
    *,
    message_id: str | None = None,
    model_name: str | None = None,
    state: dict[str, Any] | None = None,
) -> list[str]:
    """Convert an OpenAI-format streaming chunk to Anthropic SSE event text(s).

    Returns a list of *event lines* — each entry is one ``event: ...\\ndata: ...\\n``
    block, ready to join with ``\\n`` and encode.
    """
    events: list[str] = []

    if state is None:
        state = {}

    # With stream_options.include_usage the token counts arrive in a TRAILING
    # chunk that has an empty `choices` list, AFTER the finish_reason chunk
    # (whose own `usage` is null). Returning early here dropped it, so every
    # streamed Anthropic response reported `usage: {}` to the client. Stash it;
    # `final_sse_events` emits message_delta once the stream is exhausted.
    if not hasattr(chunk, "choices") or not chunk.choices:
        _capture_usage(chunk, state)
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

    # Tool calls arrive spread across chunks: the first carries id+name, later ones carry
    # argument fragments. Anthropic models that as one content block per call, so the
    # open-block bookkeeping has to outlive a single chunk — hence `state`.
    tool_blocks: dict[int, int] = state.setdefault("tool_blocks", {}) if state is not None else {}
    for call in delta_dict.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        events.extend(_tool_call_delta_events(call, tool_blocks, state))

    if finish:
        anthropic_finish = _ANTHROPIC_STOP_REASON.get(finish, finish)
        events.append('event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n')
        for block_index in sorted(tool_blocks.values()):
            events.append(
                f'event: content_block_stop\ndata: {{"type":"content_block_stop",'
                f'"index":{block_index}}}\n'
            )
        # message_delta / message_stop are deferred to final_sse_events so the
        # trailing usage-only chunk can be folded in first.
        _capture_usage(chunk, state)
        state["stop_reason"] = anthropic_finish

    return events


def _capture_usage(chunk: Any, state: dict[str, Any]) -> None:
    """Record token counts from whichever chunk actually carries them."""
    usage = getattr(chunk, "usage", None)
    if not usage:
        return
    state["usage"] = {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }


def final_sse_events(state: dict[str, Any]) -> list[str]:
    """Emit the terminal message_delta/message_stop once the stream is exhausted."""
    if "stop_reason" not in state:
        return []
    msg_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": state["stop_reason"], "stop_sequence": None},
        "usage": state.get("usage", {}),
    }
    return [
        f"event: message_delta\ndata: {json.dumps(msg_delta)}\n",
        'event: message_stop\ndata: {"type":"message_stop"}\n',
    ]


# OpenAI finish_reason -> Anthropic stop_reason. `tool_calls -> tool_use` is the one
# clients branch on to decide whether to run a tool; passing it through untranslated
# left Claude Code with a stop_reason it does not recognise.
_ANTHROPIC_STOP_REASON: Final[dict[str, str]] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "content_filter",
    "tool_calls": "tool_use",
}


def _tool_arguments_to_input(raw: Any) -> dict[str, Any]:
    """Anthropic wants a decoded object; OpenAI sends a JSON *string*."""
    if isinstance(raw, dict):
        return raw
    try:
        decoded = json.loads(raw or "{}")
    except (TypeError, ValueError):
        # A truncated/invalid argument string must not 500 the whole response — the
        # client sees an empty input and can retry, which beats losing the turn.
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _openai_message_to_anthropic_content(message: Any) -> list[dict[str, Any]]:
    """Anthropic content blocks for one OpenAI assistant message: text plus tool calls."""
    blocks: list[dict[str, Any]] = []
    text = (getattr(message, "content", None) or "") if message is not None else ""
    if text:
        blocks.append({"type": "text", "text": text})

    for index, call in enumerate(getattr(message, "tool_calls", None) or []):
        function = getattr(call, "function", None)
        blocks.append(
            {
                "type": "tool_use",
                "id": getattr(call, "id", None) or f"toolu_{index}",
                "name": getattr(function, "name", None) or "",
                "input": _tool_arguments_to_input(getattr(function, "arguments", None)),
            }
        )

    # Anthropic requires a non-empty content list.
    return blocks or [{"type": "text", "text": ""}]


def _response_finish_reason(response: Any) -> str | None:
    """Raw OpenAI ``finish_reason`` of the first choice, or None when absent."""
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    choice = choices[0]
    return getattr(choice, "finish_reason", None) or None


def _openai_response_to_anthropic(response: Any) -> dict[str, Any]:
    """Convert an OpenAI-format *ModelResponse* to an Anthropic /v1/messages response dict."""
    choice = response.choices[0] if response.choices and len(response.choices) > 0 else None
    message = choice.message if choice else None

    usage_in = 0
    usage_out = 0
    if hasattr(response, "usage") and response.usage:
        usage_in = response.usage.prompt_tokens or 0
        usage_out = response.usage.completion_tokens or 0

    finish_reason = None
    if choice and hasattr(choice, "finish_reason") and choice.finish_reason:
        finish_reason = _ANTHROPIC_STOP_REASON.get(choice.finish_reason, choice.finish_reason)

    return {
        "id": getattr(response, "id", None) or f"msg_{int(time.time() * 1000)}",
        "type": "message",
        "role": "assistant",
        "content": _openai_message_to_anthropic_content(message),
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
        engine: RouterEngine | None = None,
    ) -> None:
        self._pool = model_pool
        self._sessions = session_manager
        self.retry_count = retry_count
        self._engine = engine
        # Accumulates structured, non-model-authored wire signals (tool_result.is_error,
        # terminal stop_reason) onto session.metadata — the weak, quarantined Tier-1 prior.
        self._wire_collector = WireSignalCollector()
        # Serializes the first-turn decision so concurrent first turns of one session
        # cannot both route (which would break the one-decision-per-session guarantee).
        self._decision_lock = threading.Lock()
        # Opt-in label on the `model` field of responses. Clients echo that field in
        # their UI, so without it a routed session is indistinguishable from talking to
        # the provider directly — you cannot tell the router is in the path at all.
        # Off by default: it changes a wire-visible field, so a deployment opts in.
        self._model_label = os.environ.get("SHUNT_RESPONSE_MODEL_LABEL", "")

    def _label(self, model_name: str) -> str:
        """Prefix a response's model id with the configured label, if any."""
        if not self._model_label or not model_name:
            return model_name
        return f"{self._model_label}{model_name}"

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
        return await self._route(body, session)

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

        # Structured tool-error signals live in the resent request history, so they are
        # collected the same way whether or not the response streams.
        self._wire_collector.observe_tool_errors(body, session)

        stream = body.get("stream", False)
        if stream:
            tracked = self._track_cache_tax(response, session)
            gen: AsyncGenerator[bytes, None] = self._anthropic_stream(tracked)
            return gen, model_name, f"stream:{model_name}"

        self._update_cache_tax(session, response)
        self._log_cache_metrics(session)
        anthropic_body = _openai_response_to_anthropic(response)
        # The label was applied on the OpenAI path only, so Claude Code — the one client
        # that can actually surface it — never saw which model served the turn.
        anthropic_body["model"] = self._label(str(anthropic_body.get("model") or ""))
        # stop_reason is already Anthropic-normalized; the collector drops loop-reopening
        # stops (tool_use/pause_turn) so only a terminal close contributes a prior.
        self._wire_collector.observe_terminal_stop(anthropic_body.get("stop_reason"), session)
        return anthropic_body, model_name, model_name

    # ── Internal routing ────────────────────────────────────────────────────

    async def _route(
        self,
        body: dict[str, Any],
        session: Session,
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
        for key in _FORWARDED_OPENAI_KEYS:
            if key in body:
                openai_kwargs[key] = body[key]
        _log_dropped_keys(body, openai_kwargs)

        response, model_name = await self._route_with_fallback(openai_kwargs, session)

        if stream:
            tracked = self._track_cache_tax(response, session)
            gen: AsyncGenerator[bytes, None] = self._openai_stream(tracked)
            return gen, model_name, f"stream:{model_name}"

        self._update_cache_tax(session, response)
        self._log_cache_metrics(session)
        payload = response.model_dump() if hasattr(response, "model_dump") else response
        if isinstance(payload, dict) and payload.get("model"):
            payload["model"] = self._label(str(payload["model"]))
        # OpenAI has no structured per-tool is_error on the wire, so only the terminal
        # finish_reason contributes a prior (normalized to the Anthropic vocab so a
        # loop-reopening tool_calls is dropped by the collector).
        raw_finish = _response_finish_reason(response)
        normalized = _ANTHROPIC_STOP_REASON.get(raw_finish, raw_finish) if raw_finish else None
        self._wire_collector.observe_terminal_stop(normalized, session)
        return payload, model_name, model_name

    async def _route_with_fallback(
        self,
        openai_kwargs: dict[str, Any],
        session: Session,
    ) -> tuple[Any, str]:
        """Route *openai_kwargs* through the model chain with retry+fallback.

        Returns *(openai_response | async_generator, selected_model_name)*.
        """
        prompt_text = _routing_text_from_messages(openai_kwargs.get("messages", []))
        # Off the event loop: the decision runs ONNX inference and takes a blocking
        # lock, so doing it inline serialized concurrent first turns and stalled every
        # in-flight SSE stream for the duration (measured 1.50s for 3 concurrent turns
        # of a 0.5s embed).
        model_name = await run_in_threadpool(self._get_or_lock_model, session, prompt_text)
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

        logger.debug(
            "upstream: model=%s provider_route=%s stream=%s messages=%d forwarded_keys=%s",
            model_name,
            # The route, never the key — this identifies WHICH provider actually served
            # the call, which the response body alone does not tell you.
            getattr(config, "route", "?"),
            openai_kwargs.get("stream", False),
            len(openai_kwargs.get("messages", []) or []),
            sorted(k for k in openai_kwargs if k not in ("messages", "model")),
        )
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

    def _get_or_lock_model(self, session: Session, prompt_text: str = "") -> str:
        """Return the model chosen for *session*, locking it on first access.

        Cache-safe: the decision is made exactly once (when ``model_chosen`` is unset)
        and reused for every later turn — never re-routed mid-session.
        """
        if session.model_chosen:
            return session.model_chosen
        with self._decision_lock:
            if session.model_chosen:
                return session.model_chosen
            model = self._decide_once(session, prompt_text)
            logger.info(
                "Session %s routed to model=%s reason=%s",
                session.session_id,
                model,
                session.metadata.get("model_source", "unknown"),
            )
            return model

    def _decide_once(self, session: Session, prompt_text: str) -> str:
        """Make the session's single routing decision — caller holds the decision lock."""
        if self._engine is not None:
            return self._decide_via_engine(session, prompt_text)
        model_name = _DEFAULT_MODEL
        session.model_chosen = model_name
        session.metadata["model"] = model_name
        session.metadata["model_source"] = "cold-start-always-cheap"
        session.metadata["last_prompt"] = prompt_text
        return model_name

    def cached_embedding(self, session_id: str) -> Any:
        """Embedding the engine computed for *session_id* (None when no engine embedded it)."""
        if self._engine is None:
            return None
        return self._engine.cached_embedding(session_id)

    def model_fingerprint(self, model_name: str) -> str | None:
        """Resolved version fingerprint of *model_name* at route time.

        None only when the name is not in the registry — never a fabricated tag.
        """
        config = self._pool.get_model(model_name)
        return _resolve_fingerprint(config) if config is not None else None

    def _decide_via_engine(self, session: Session, prompt_text: str) -> str:
        """Route the first turn through the injected engine and lock the result."""
        assert self._engine is not None
        model_name, reason, provenance = self._engine.decide(session.session_id, prompt_text)
        session.model_chosen = model_name
        session.metadata["model"] = model_name
        session.metadata["model_source"] = reason
        session.metadata["last_prompt"] = prompt_text
        session.decision_provenance = provenance
        return model_name

    # ── Cache-tax measurement ──────────────────────────────────────────────

    @staticmethod
    def _reported_cost(usage: Any) -> float | None:
        """The provider-reported billed amount for one call, or None when absent."""
        # OpenAI-compatible gateways return the real, cache-aware charge on ``usage.cost``.
        # Nothing here derives a price locally: an unreported charge stays unknown rather
        # than being guessed from a list price.
        if usage is None:
            return None
        raw = getattr(usage, "cost", None)
        if raw is None and isinstance(usage, dict):
            raw = usage.get("cost")
        if raw is None or isinstance(raw, bool):
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value >= 0.0 else None

    def _accumulate_cost(self, session: Session, usage: Any) -> None:
        """Add the provider-reported charge to the session total; log once if unreported."""
        reported = self._reported_cost(usage)
        if reported is not None:
            session.total_cost += reported
            return
        if not session.metadata.get("cost_unreported"):
            session.metadata["cost_unreported"] = True
            logger.info(
                "Upstream reported no usage.cost for session %s (model %s); "
                "session cost stays unaccumulated rather than estimated",
                session.session_id,
                session.model_chosen,
            )

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
        # try/finally so cost is settled even on an early disconnect: aclose() throws
        # GeneratorExit at the yield, and a stream that ended having seen NO usage is
        # cost-UNKNOWN, not a free 0.0 — flag it (the None path) so persist writes
        # cost_known=0 rather than a fabricated zero that sorts cheapest in the read-back.
        last_usage = None
        try:
            async for chunk in gen:
                if hasattr(chunk, "usage") and chunk.usage:
                    last_usage = chunk.usage
                yield chunk
        finally:
            if last_usage is not None:
                self._accumulate_cost(session, last_usage)
                details = getattr(last_usage, "prompt_tokens_details", None)
                self._record_cache_turn(
                    session,
                    float(getattr(details, "cached_tokens", 0) or 0),
                    getattr(last_usage, "prompt_tokens", 0) or 0,
                )
                self._log_cache_metrics(session)
            else:
                self._accumulate_cost(session, None)

    @staticmethod
    def _extract_prompt_tokens(response: Any) -> int:
        """Extract prompt_tokens from an OpenAI response usage."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0
        return getattr(usage, "prompt_tokens", 0) or 0

    def _update_cache_tax(self, session: Session, response: Any) -> None:
        """Extract cache metrics and the billed cost from a response; update the session."""
        self._accumulate_cost(session, getattr(response, "usage", None))
        self._record_cache_turn(
            session, self._extract_cache_tax(response), self._extract_prompt_tokens(response)
        )

    @staticmethod
    def _record_cache_turn(session: Session, cached_tokens: float, prompt_tokens: int) -> None:
        """Fold one turn's cache numbers into the session's running totals."""
        # Both sides accumulate. Only `cache_tax` used to, so the ratio divided a
        # session total by one turn's prompt and saturated at the clamp.
        if cached_tokens > 0:
            session.cache_tax = getattr(session, "cache_tax", 0.0) + cached_tokens
        if prompt_tokens > 0:
            session.prompt_length_tokens = prompt_tokens
            session.prompt_tokens_total = getattr(session, "prompt_tokens_total", 0) + prompt_tokens
        session.metadata["last_turn_cached_tokens"] = cached_tokens
        session.metadata["last_turn_prompt_tokens"] = prompt_tokens

    @staticmethod
    def _session_hit_ratio(session: Session) -> float:
        """Cached share of every prompt token this session has sent."""
        total = getattr(session, "prompt_tokens_total", 0)
        return getattr(session, "cache_tax", 0.0) / total if total > 0 else 0.0

    def _log_cache_metrics(self, session: Session) -> None:
        """Log this turn's cache hit rate and the session's running rate."""
        turn_cached = float(session.metadata.get("last_turn_cached_tokens", 0.0))
        turn_prompt = int(session.metadata.get("last_turn_prompt_tokens", 0))
        if session.cache_tax <= 0 and session.prompt_tokens_total <= 0:
            return
        turn_ratio = turn_cached / turn_prompt if turn_prompt > 0 else 0.0
        logger.info(
            "Session %s cached_tokens=%.0f prompt_tokens=%d turn_hit_ratio=%.4f "
            "session_cached=%.0f session_prompt=%d session_hit_ratio=%.4f",
            session.session_id,
            turn_cached,
            turn_prompt,
            turn_ratio,
            session.cache_tax,
            session.prompt_tokens_total,
            self._session_hit_ratio(session),
        )

    # ── Streaming helpers ───────────────────────────────────────────────────

    async def _openai_stream(self, gen: Any) -> AsyncGenerator[bytes, None]:
        """Wrap an OpenAI streaming async generator into OpenAI-format SSE bytes."""
        async for chunk in gen:
            if hasattr(chunk, "model_dump"):
                chunk_dict = chunk.model_dump(exclude_none=True)
            else:
                chunk_dict = chunk
            if isinstance(chunk_dict, dict) and chunk_dict.get("model"):
                chunk_dict["model"] = self._label(str(chunk_dict["model"]))
            yield f"data: {json.dumps(chunk_dict, default=str)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def _anthropic_stream(self, gen: Any) -> AsyncGenerator[bytes, None]:
        """Wrap an OpenAI streaming async generator into Anthropic-format SSE bytes."""
        message_id = f"msg_{int(time.time() * 1000)}"
        model_name = ""
        role_seen = False
        # Owned here, not per chunk: a tool call's block spans many chunks.
        state: dict[str, Any] = {}

        async for chunk in gen:
            if not role_seen and hasattr(chunk, "model") and chunk.model:
                model_name = self._label(chunk.model)
            events = _openai_chunk_to_anthropic_sse(
                chunk,
                message_id=message_id,
                model_name=model_name,
                state=state,
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

        trailing = final_sse_events(state)
        for event in trailing:
            yield event.encode("utf-8")
            yield b"\n"
        if trailing:
            yield b"\n"
