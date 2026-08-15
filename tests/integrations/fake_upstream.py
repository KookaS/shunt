"""A happy-path OpenAI-compatible upstream stub for integration handshakes.

Always answers 200 with a well-formed ChatCompletion (no key, no cost, no real
network) and records requests. Unhappy-path replay: ``tests/mock_openai_server.py``.
"""

# It can also answer a request that declares tools with a function CALL, which the
# tool-translation suites need — but that reply is bounded twice over, because a
# stateless stub answering every turn with the same call holds an agentic client in an
# infinite loop. See ``_TOOL_CALLS_ENV`` and ``_MAX_TOOL_CALLS_ENV``.

from __future__ import annotations

import contextlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Name of the env var holding the directory to append one JSON line per POSTed body to.
# When it is unset, which is always the case in-process, nothing is written at all.
# Recording exists for the Docker scenarios, which read the log back to answer which
# tools announce their working directory to the model.
_RECORD_DIR_ENV = "FAKE_UPSTREAM_RECORD_DIR"


def _record(path: str, payload: dict[str, object]) -> None:
    """Append ``{path, body}`` to ``$FAKE_UPSTREAM_RECORD_DIR/requests.jsonl``, or do nothing."""
    record_dir = os.environ.get(_RECORD_DIR_ENV)
    if not record_dir:
        return
    line = json.dumps({"path": path, "body": payload}, default=str)
    with contextlib.suppress(OSError):
        target = Path(record_dir)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "requests.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


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


# The canned function call every fake tool-call response makes. Name comes from the
# request's first declared tool; id + arguments are fixed so tests can assert them.
_TOOL_CALL_ID = "call_abc123"
_TOOL_CALL_ARGUMENTS = '{"path": "main.py"}'
# Placeholder value for a required argument the canned arguments do not cover.
_TOOL_CALL_FILLER = "shunt-fake"

# Name of the env var that turns tool-call replies ON for the containerised stub.
# They are OFF by default there: an agentic CLI (opencode, claude-code, continue)
# loops forever against a STATELESS stub that answers every turn with the same call
# and never a terminal text turn. That is not hypothetical — it burned 2h16m of CI
# on three handshake legs before the run was cancelled by hand. The in-process
# FakeUpstream keeps them on, because the tool-translation e2e suites need them.
_TOOL_CALLS_ENV = "FAKE_UPSTREAM_TOOL_CALLS"
# How many tool-call replies one server process will make before it answers plain
# text regardless. The second wall, and the one that does not depend on anybody
# setting an env var right: even with tool calls ON, an agent loop terminates.
_MAX_TOOL_CALLS_ENV = "FAKE_UPSTREAM_MAX_TOOL_CALLS"
_DEFAULT_MAX_TOOL_CALLS = 4


def _tool_call_body(model: str, tool_name: str, arguments: str) -> str:
    """A non-streaming ChatCompletion whose only assistant turn is a function call."""
    usage = {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12, "cost": 0.0001}
    return json.dumps(
        {
            "id": "fake-cmpl-tool",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": _TOOL_CALL_ID,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": arguments,
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": usage,
        }
    )


def _tool_usage() -> dict[str, object]:
    """The usage object every fake tool-call response reports (incl. a real cost)."""
    return {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12, "cost": 0.0001}


def _tool_call_chunks(model: str, tool_name: str, arguments: str) -> list[str]:
    """SSE frames for a streamed function call: role, id+name, one argument fragment,
    finish_reason=tool_calls, a usage-only trailing chunk (include_usage shape), DONE."""
    base = {"id": "fake-cmpl-tool", "object": "chat.completion.chunk", "created": 0, "model": model}

    def frame(delta: dict[str, object], finish: str | None) -> str:
        chunk = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(chunk)}\n\n"

    name_delta = {
        "tool_calls": [
            {
                "index": 0,
                "id": _TOOL_CALL_ID,
                "type": "function",
                "function": {"name": tool_name, "arguments": ""},
            }
        ]
    }
    args_delta = {"tool_calls": [{"index": 0, "function": {"arguments": arguments}}]}
    usage_chunk = {**base, "choices": [], "usage": _tool_usage()}
    return [
        frame({"role": "assistant", "content": None}, None),
        frame(name_delta, None),
        frame(args_delta, None),
        frame({}, "tool_calls"),
        f"data: {json.dumps(usage_chunk)}\n\n",
        "data: [DONE]\n\n",
    ]


def _completion_chunks(model: str) -> list[str]:
    """SSE frames for a streamed ChatCompletion: leading role delta (required — the
    ai-sdk `openai-compatible` provider drops all text without it), content, stop, DONE."""
    base = {"id": "fake-cmpl-1", "object": "chat.completion.chunk", "created": 0, "model": model}

    def frame(delta: dict[str, object], finish: str | None) -> str:
        chunk = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(chunk)}\n\n"

    return [
        frame({"role": "assistant"}, None),
        frame({"content": "ok"}, None),
        frame({}, "stop"),
        "data: [DONE]\n\n",
    ]


def _models_body() -> str:
    return json.dumps({"object": "list", "data": [{"id": "fake/cheap", "object": "model"}]})


def _requested_tool(payload: dict[str, object]) -> dict[str, object] | None:
    """The first function tool the request declared, or None when it declared none."""
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function
    return None


def _tool_call_arguments(function: dict[str, object]) -> str:
    """Arguments the declared tool's schema will actually accept."""
    # The canned `{"path": "main.py"}` is kept whenever it satisfies the schema, so the
    # translation suites that assert on it are unaffected. It is replaced only when the
    # tool declares `required` keys it does not cover — the case that made opencode reject
    # every call ('bash ... SchemaError(Missing key at ["command"])') and retry forever.
    canned: dict[str, object] = json.loads(_TOOL_CALL_ARGUMENTS)
    parameters = function.get("parameters")
    required = parameters.get("required") if isinstance(parameters, dict) else None
    if not isinstance(required, list):
        return _TOOL_CALL_ARGUMENTS
    missing = [key for key in required if isinstance(key, str) and key not in canned]
    if not missing:
        return _TOOL_CALL_ARGUMENTS
    return json.dumps({key: _TOOL_CALL_FILLER for key in missing})


def _chat_completion_response(
    payload: dict[str, object], *, tool_calls: bool
) -> tuple[bool, list[str]]:
    """What one /chat/completions request should receive: (is_sse, frames|body)."""
    # A function call only when tool-call replies are enabled AND the request declared
    # tools; otherwise plain text. The stub is stateless, so an unconditional tool call
    # never terminates an agent loop — see `_TOOL_CALLS_ENV`.
    model = str(payload.get("model", "fake/cheap"))
    function = _requested_tool(payload) if tool_calls else None
    if function is None:
        if payload.get("stream"):
            return True, _completion_chunks(model)
        return False, [_completion_body(model)]
    name = str(function["name"])
    arguments = _tool_call_arguments(function)
    if payload.get("stream"):
        return True, _tool_call_chunks(model, name, arguments)
    return False, [_tool_call_body(model, name, arguments)]


class _BaseHandler(BaseHTTPRequestHandler):
    """Response helpers shared by every fake-upstream handler."""

    def _send(self, status: int, body: str) -> None:
        payload = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_sse(self, frames: list[str]) -> None:
        """Stream SSE frames; connection-close (HTTP/1.0) delimits the stream."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for frame in frames:
            self.wfile.write(frame.encode())
            self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib's name
        """Silence per-request stderr logging."""


def _parse_body(raw: bytes) -> dict[str, object]:
    """The POSTed JSON body, or {} when it is not valid JSON."""
    payload: dict[str, object] = {}
    with contextlib.suppress(json.JSONDecodeError):
        payload = json.loads(raw)
    return payload


def _handler_for(
    received: list[str],
    bodies: list[dict[str, object]] | None = None,
    *,
    tool_calls: bool = True,
    max_tool_calls: int | None = None,
) -> type[BaseHTTPRequestHandler]:
    """Build a handler bound to one request log (and an optional in-memory body log)."""
    # A one-element list, not a module global (SH001): the budget belongs to THIS
    # server, and a per-process counter is what bounds an agent loop no matter what
    # the client does with the tool result.
    remaining = [_DEFAULT_MAX_TOOL_CALLS if max_tool_calls is None else max_tool_calls]

    def grant_tool_call(payload: dict[str, object]) -> bool:
        """Whether to answer THIS request with a tool call, spending one from the budget.

        Spent only when a call is actually emitted, so a conversation that declares no
        tools never eats into the budget of one that does.
        """
        if not tool_calls or remaining[0] <= 0 or _requested_tool(payload) is None:
            return False
        remaining[0] -= 1
        return True

    class Handler(_BaseHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
            received.append(f"GET {self.path}")
            if self.path.endswith("/models"):
                self._send(200, _models_body())
                return
            self._send(404, json.dumps({"error": {"message": f"Path not found: {self.path}"}}))

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
            received.append(f"POST {self.path}")
            if not self.path.endswith("/chat/completions"):
                self._send(404, json.dumps({"error": {"message": f"Path not found: {self.path}"}}))
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            payload = _parse_body(raw)
            _record(self.path, payload)
            if bodies is not None:
                bodies.append(payload)
            is_sse, frames = _chat_completion_response(payload, tool_calls=grant_tool_call(payload))
            if is_sse:
                self._send_sse(frames)
                return
            self._send(200, frames[0])

    return Handler


class FakeUpstream:
    """A happy-path OpenAI-compatible stub on 127.0.0.1, for in-process tests."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        tool_calls: bool = True,
        max_tool_calls: int | None = None,
    ) -> None:
        self.received: list[str] = []
        # Parsed POST bodies, in order — so tests can assert what the upstream RECEIVED.
        self.bodies: list[dict[str, object]] = []
        handler = _handler_for(
            self.received, self.bodies, tool_calls=tool_calls, max_tool_calls=max_tool_calls
        )
        self._server = ThreadingHTTPServer((host, port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    @property
    def base_url(self) -> str:
        """The stub's root URL — append ``/v1`` for an OpenAI provider base_url."""
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> FakeUpstream:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def main() -> None:
    """Serve on 0.0.0.0:${FAKE_UPSTREAM_PORT:-9099} for the Docker harness."""
    port = int(os.environ.get("FAKE_UPSTREAM_PORT", "9099"))
    # OFF by default here, ON for the in-process FakeUpstream — see _TOOL_CALLS_ENV.
    tool_calls = os.environ.get(_TOOL_CALLS_ENV, "0").strip().lower() in {"1", "true", "yes"}
    max_tool_calls = int(os.environ.get(_MAX_TOOL_CALLS_ENV, str(_DEFAULT_MAX_TOOL_CALLS)))
    received: list[str] = []
    handler = _handler_for(received, tool_calls=tool_calls, max_tool_calls=max_tool_calls)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)  # noqa: S104
    mode = f"tool-calls {'on' if tool_calls else 'off'} (max {max_tool_calls})"
    print(f"fake-upstream listening on 0.0.0.0:{port} — {mode}")  # noqa: T201 - entrypoint
    server.serve_forever()


if __name__ == "__main__":
    main()
