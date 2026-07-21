"""A happy-path OpenAI-compatible upstream stub for integration handshakes.

Always answers 200 with a well-formed ChatCompletion (no key, no cost, no real
network) and records requests. Unhappy-path replay: ``tests/mock_openai_server.py``.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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


def _handler_for(received: list[str]) -> type[BaseHTTPRequestHandler]:
    """Build a handler bound to one request log."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
            received.append(f"GET {self.path}")
            if self.path.endswith("/models"):
                self._send(200, _models_body())
                return
            self._send(404, json.dumps({"error": {"message": f"Path not found: {self.path}"}}))

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            received.append(f"POST {self.path}")
            if self.path.endswith("/chat/completions"):
                payload: dict[str, object] = {}
                with contextlib.suppress(json.JSONDecodeError):
                    payload = json.loads(raw)
                model = str(payload.get("model", "fake/cheap"))
                if payload.get("stream"):
                    self._send_sse(_completion_chunks(model))
                    return
                self._send(200, _completion_body(model))
                return
            self._send(404, json.dumps({"error": {"message": f"Path not found: {self.path}"}}))

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

    return Handler


class FakeUpstream:
    """A happy-path OpenAI-compatible stub on 127.0.0.1, for in-process tests."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.received: list[str] = []
        self._server = ThreadingHTTPServer((host, port), _handler_for(self.received))
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
    received: list[str] = []
    server = ThreadingHTTPServer(("0.0.0.0", port), _handler_for(received))  # noqa: S104
    print(f"fake-upstream listening on 0.0.0.0:{port}")  # noqa: T201 - container entrypoint
    server.serve_forever()


if __name__ == "__main__":
    main()
