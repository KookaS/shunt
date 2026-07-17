"""A local OpenAI-compatible stub that replays a provider's declared auth-rejection signature."""

# WHY THIS EXISTS. The live probe's interesting branches are the ones a naive
# 401-only stub would never reach: Requesty's 403, xAI's 400-carrying-an-auth-
# message, Fireworks answering on /v1/models instead of chat/completions, and
# the 404 that a wrong base_url and Fireworks' "model inaccessible" share. This
# server replays each signature straight from the measured signature, so those
# branches are exercised hermetically on every push instead of a nightly run.
#
# It grows no provider knowledge of its own — `signature_for()` derives every
# response from the same signature the probe reads. That is what keeps
# the offline check and the live check from drifting apart: they cannot disagree
# about what a provider does, because neither one holds an opinion.

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType

from shunt.models.config import AuthProbe


@dataclass(frozen=True)
class MockSignature:
    """What the stub answers, and on which path."""

    endpoint: str
    status: int
    body: str


@dataclass(frozen=True)
class RecordedRequest:
    """One request the stub received — lets a test assert what the probe sent."""

    method: str
    path: str
    authorization: str | None
    user_agent: str | None


def signature_for(probe: AuthProbe) -> MockSignature:
    """Derive the faithful replay of a provider's declared auth-rejection signature."""
    # The pattern is echoed VERBATIM into the body. Every measured pattern is a
    # literal substring, so this matches by construction; a fancy regex in the
    # registry would fail here loudly rather than pass a hollow test.
    message = probe.expect_body_pattern or "Invalid authentication credentials"
    return MockSignature(
        endpoint=probe.endpoint,
        status=probe.expect_status[0],
        body=json.dumps({"error": {"message": message, "type": "invalid_request_error"}}),
    )


def _handler_for(
    signature: MockSignature, received: list[RecordedRequest]
) -> type[BaseHTTPRequestHandler]:
    """Build a handler class bound to one signature and one request log."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
            self._record()
            self._respond()

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            self._record()
            self._respond()

        def _record(self) -> None:
            received.append(
                RecordedRequest(
                    self.command,
                    self.path,
                    self.headers.get("Authorization"),
                    self.headers.get("User-Agent"),
                )
            )

        def _respond(self) -> None:
            if self.path == signature.endpoint:
                self._send(signature.status, signature.body)
                return
            # Mirrors the measured wrong-path control case. It is deliberately a
            # 404 with a *path* message: that is what makes the probe's job of
            # telling "wrong base_url" apart from Fireworks' "model inaccessible"
            # a real, testable distinction rather than a comment in a doc.
            self._send(
                404,
                json.dumps(
                    {"error": {"message": f"Path not found: {self.path}", "type": "not_found"}}
                ),
            )

        def _send(self, status: int, body: str) -> None:
            payload = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib's name
            """Silence the per-request stderr logging."""

    return Handler


class MockOpenAIServer:
    """An OpenAI-compatible stub on 127.0.0.1 replaying one signature."""

    def __init__(self, signature: MockSignature) -> None:
        self.received: list[RecordedRequest] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(signature, self.received))
        # A tight poll interval: shutdown() blocks until serve_forever notices,
        # and the 0.5s default turns a per-test stub into ~0.5s of dead wall time.
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    @property
    def base_url(self) -> str:
        """The stub's root URL — drop-in for a provider's base_url."""
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}"

    def start(self) -> None:
        """Begin serving in a background thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop serving and release the port."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> MockOpenAIServer:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
