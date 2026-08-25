"""Shared helpers for the e2e suite: canned upstream responses, repo fixtures, and
the drive→close→poll machinery that makes the off-wire capture deterministic."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

CHAT_PATH = "/v1/chat/completions"
MESSAGES_PATH = "/v1/messages"

_PYPROJECT = "[tool.pytest.ini_options]\naddopts = ''\n"


def user_agent_headers() -> dict[str, str]:
    """A fixed User-Agent so every request maps to ONE tool_identity per test."""
    return {"User-Agent": "shunt-e2e/0.1"}


def chat_body(*, stream: bool = False, content: str = "Fix the build") -> dict[str, Any]:
    """An OpenAI chat-completion body routed to the default cheap model."""
    return {"messages": [{"role": "user", "content": content}], "stream": stream}


# ── Dummy repos: the verified-outcome target the off-wire verifier re-runs ─────


def make_repo(repo: Path, *, kind: str) -> Path:
    """Write a deterministic repo of the requested *kind* under *repo*."""
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "pyproject.toml").write_text(_PYPROJECT)
    if kind == "green":
        (repo / "test_x.py").write_text("def test_x():\n    assert True\n")
    elif kind == "red":
        (repo / "test_x.py").write_text("def test_x():\n    assert False\n")
    # red_a and red_b fail DIFFERENT tests, so their failures carry distinct dedup keys.
    elif kind == "red_a":
        (repo / "test_a.py").write_text("def test_a():\n    assert False\n")
        (repo / "test_b.py").write_text("def test_b():\n    assert True\n")
    elif kind == "red_b":
        (repo / "test_a.py").write_text("def test_a():\n    assert True\n")
        (repo / "test_b.py").write_text("def test_b():\n    assert False\n")
    # An import error, so the verifier attributes the failure to the environment.
    elif kind == "infra":
        (repo / "test_x.py").write_text(
            "import definitely_missing_pkg_xyz\n\ndef test_x():\n    pass\n"
        )
    # Fails once then passes on rerun — the write-once marker is what makes it flaky.
    elif kind == "flaky":
        (repo / "test_x.py").write_text(
            "import os\n"
            "import pathlib\n"
            "\n"
            "MARKER = pathlib.Path(os.path.join(os.path.dirname(__file__), 'flake.marker'))\n"
            "def test_x():\n"
            "    if MARKER.exists():\n"
            "        assert True\n"
            "    else:\n"
            "        MARKER.write_text('1')\n"
            "        assert False\n"
        )
    else:
        raise ValueError(f"unknown repo kind: {kind!r}")
    return repo


# ── Canned upstream: the "fake model" that returns predefined text, $0 ────────


def _chat_payload(content: str, model: str) -> dict[str, Any]:
    return {
        "id": "cmpl-e2e",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    }


def canned_chat_response(
    content: str = "Hello back", model: str = "deepseek-v4-flash"
) -> MagicMock:
    """A canned non-streaming ChatCompletion the mocked upstream returns verbatim."""
    resp = MagicMock()
    resp.id = "cmpl-e2e"
    resp.model = model
    usage = MagicMock()
    usage.prompt_tokens = 5
    usage.completion_tokens = 10
    usage.cost = 0.001
    usage.prompt_tokens_details = MagicMock()
    usage.prompt_tokens_details.cached_tokens = 0
    resp.usage = usage
    choice = MagicMock()
    choice.index = 0
    choice.finish_reason = "stop"
    choice.message.content = content
    choice.message.role = "assistant"
    resp.choices = [choice]
    resp.model_dump.return_value = _chat_payload(content, model)
    return resp


async def _canned_stream(content: str) -> AsyncGenerator[MagicMock, None]:
    """A canned SSE chunk stream whose content reaches the client unchanged."""
    first = MagicMock()
    first.usage = None
    first.model_dump.return_value = {
        "id": "c1",
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": None,
                "delta": {"role": "assistant", "content": content},
            }
        ],
    }
    yield first
    last = MagicMock()
    last.usage = None
    last.model_dump.return_value = {
        "id": "c2",
        "model": "deepseek-v4-flash",
        "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
    }
    yield last


def make_fake_acompletion(
    *, content: str = "Hello back", model: str = "deepseek-v4-flash"
) -> Callable[..., Any]:
    """A ``shunt.proxy.router._acompletion`` stand-in: predefined text, no API call."""

    def _fake(config: Any, **kwargs: Any) -> Any:
        del config
        if kwargs.get("stream"):
            return _canned_stream(content)
        return canned_chat_response(content, model)

    return _fake


# ── Decision/response parsing ────────────────────────────────────────────────


def parse_decision(header: str) -> tuple[str, str]:
    """Split an ``X-Shunt-Decision`` header into ``(model, reason)``."""
    model, _, reason = header.partition("; reason=")
    return model, reason


def post_completion(client: TestClient, body: dict[str, Any], *, path: str = CHAT_PATH) -> Any:
    """POST *body* with the fixed User-Agent, failing loudly on a non-200."""
    resp = client.post(path, json=body, headers=user_agent_headers())
    assert resp.status_code == 200, resp.text
    return resp


# ── Deterministic capture waits ───────────────────────────────────────────────


def poll_until(pred: Callable[[], bool], *, timeout: float = 30.0, what: str) -> None:
    """Poll *pred* to True, bounded; the capture runs on a daemon thread + subprocess."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {what}")


def close_session(client: TestClient, session_id: str) -> None:
    """Close via the lifespan's session manager — the same seam the inactivity sweep uses."""
    client.app.state.session_manager.close_session(session_id)


def failure_log_len(client: TestClient, work_dir: str) -> int:
    """The engine's in-memory failure-log size for *work_dir* — the escalate decision input."""
    from shunt.router.engine import task_state_key

    engine = client.app.state.router._engine
    return len(engine._failure_log.get(task_state_key(work_dir), []))


def wait_failure_count(client: TestClient, work_dir: str, n: int) -> None:
    """Wait until the failure log holds *n* verified same-repo failures (0 = cleared)."""
    # The log is the escalate decision input, and it is updated AFTER the DB commit
    # (record_outcome runs after append_outcome_event), so poll it — never the DB —
    # or a green capture would appear done while the pop is still in flight.
    if n == 0:
        pred = lambda: failure_log_len(client, work_dir) == 0  # noqa: E731
        what = "failure log cleared"
    else:
        pred = lambda: failure_log_len(client, work_dir) >= n  # noqa: E731
        what = f"{n} verified failure(s) for the repo"
    poll_until(pred, what=what)


def wait_outcome(client: TestClient, session_id: str, outcome: str) -> None:
    """Wait until the session's materialized Tier-2 outcome is *outcome*."""
    store = client.app.state.outcome_store

    def _seen() -> bool:
        row = store.get_outcome(session_id)
        return row is not None and row.get("tier2_outcome") == outcome

    poll_until(_seen, what=f"tier2_outcome={outcome!r} for session {session_id[:8]}")


def wait_capture_idle(client: TestClient, *, timeout: float = 30.0) -> None:
    """Wait until the capture worker's queue is drained (nothing left in flight)."""
    worker = client.app.state.session_manager._verifier_callback.__self__
    poll_until(
        lambda: worker._queue.unfinished_tasks == 0,
        timeout=timeout,
        what="capture worker queue to drain",
    )
