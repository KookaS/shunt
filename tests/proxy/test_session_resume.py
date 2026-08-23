"""Session keying + resume on external session ids (X-Session-Id / x-session-affinity).

Pins the cache-safety invariant: an OPEN in-memory session is never re-decided on resume.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from shunt.proxy.server import _get_external_session_id, app

_ACOMPLETION_PATCH = "shunt.proxy.router._acompletion"
# A real registry model distinct from the cold-start default, so a resume that reused it
# proves the model was NOT re-decided (the engine-less default is qwen3.7-plus).
_SESSION_RESUME_MODEL = "deepseek-v4-flash"

_CHAT_BODY: Final[dict[str, object]] = {
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": False,
}


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
def _fake_lifespan_embedder() -> Iterator[None]:
    with patch("shunt.proxy.server.Embedder", _FakeEmbedder):
        yield


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    # Isolate the outcome store on a tmp dir so resume lookups read only test data.
    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path))
    with TestClient(app) as c:
        yield c


def _canned_chat_response() -> MagicMock:
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


def _seed_external_session(
    client: TestClient, *, external_id: str, model: str, session_id: str = "seeded-ses"
) -> None:
    """Persist a session row carrying *external_id* — what a PRIOR run would have written."""
    client.app.state.outcome_store.store_session(
        session_id=session_id,
        prompt_text="old conversation",
        embedding=None,
        model_chosen=model,
        cost=0.0,
        cache_stats={},
        duration=1.0,
        external_session_id=external_id,
    )


def _post_chat(
    client: TestClient,
    *,
    session_id: str | None = None,
    parent_id: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> Any:
    headers = {"User-Agent": "shunt-e2e/0.1"}
    if extra_headers:
        headers.update(extra_headers)
    if session_id is not None:
        headers["X-Session-Id"] = session_id
    if parent_id is not None:
        headers["x-parent-session-id"] = parent_id
    return client.post("/v1/chat/completions", json=_CHAT_BODY, headers=headers)


# ── header parsing / sanitization ─────────────────────────────────────────────


def _no_decide(client: TestClient) -> Any:
    """A mock whose invocation means the engine was consulted — a resume must never do that."""
    return patch.object(
        client.app.state.router,
        "_decide_via_engine",
        side_effect=AssertionError("engine consulted"),
    )


def _scope_request(headers: dict[str, str]) -> Request:
    scope: dict[str, Any] = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "method": "POST",
        "path": "/v1/chat/completions",
        "query_string": b"",
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
        "scheme": "http",
        "http_version": "1.1",
    }
    return Request(scope)


def test_x_session_affinity_is_a_fallback_alias() -> None:
    sid, parent = _get_external_session_id(_scope_request({"x-session-affinity": "ses-9"}))
    assert sid == "ses-9"
    assert parent is None


def test_x_session_id_takes_precedence_over_affinity() -> None:
    sid, _ = _get_external_session_id(
        _scope_request({"x-session-id": "ses-1", "x-session-affinity": "ses-2"})
    )
    assert sid == "ses-1"


def test_parent_session_id_is_read() -> None:
    sid, parent = _get_external_session_id(
        _scope_request({"X-Session-Id": "ses-child", "x-parent-session-id": "ses-parent"})
    )
    assert sid == "ses-child"
    assert parent == "ses-parent"


def test_blank_or_absent_headers_return_none() -> None:
    assert _get_external_session_id(_scope_request({})) == (None, None)
    assert _get_external_session_id(_scope_request({"X-Session-Id": ""})) == (None, None)
    assert _get_external_session_id(_scope_request({"X-Session-Id": "  \t "})) == (None, None)


def test_external_id_is_sanitized() -> None:
    dirty = "ses\u0001-\u00e9\u007f\x02X" + "y" * 200
    sid, _ = _get_external_session_id(_scope_request({"X-Session-Id": dirty}))
    assert sid is not None
    # Control + non-ASCII chars stripped; length capped at 128.
    assert "\u0001" not in sid and "\u00e9" not in sid and "\x7f" not in sid
    assert len(sid) <= 128
    assert "ses-X" in sid and "yyyy" in sid


# ── header-keyed session identity (end-to-end) ────────────────────────────────


def test_same_external_id_reuses_the_same_session(client: TestClient) -> None:
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _canned_chat_response()
        r1 = _post_chat(client, session_id="ses-A")
        r2 = _post_chat(client, session_id="ses-A")
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.headers["X-Shunt-Session-Id"] == r2.headers["X-Shunt-Session-Id"]


def test_new_external_id_gets_a_new_session(client: TestClient) -> None:
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _canned_chat_response()
        r1 = _post_chat(client, session_id="ses-A")
        r2 = _post_chat(client, session_id="ses-B")
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.headers["X-Shunt-Session-Id"] != r2.headers["X-Shunt-Session-Id"]


def test_no_header_falls_back_to_ip_ua_grouping(client: TestClient) -> None:
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _canned_chat_response()
        r1 = _post_chat(client)
        r2 = _post_chat(client)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.headers["X-Shunt-Session-Id"] == r2.headers["X-Shunt-Session-Id"]


def test_session_carries_external_id(client: TestClient) -> None:
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _canned_chat_response()
        _post_chat(client, session_id="ses-child", parent_id="ses-parent")
    session = client.app.state.session_manager.get_session_by_identity("ext:ses-child")
    assert session is not None
    assert session.external_session_id == "ses-child"


# ── resume / fork: reuse the persisted model without re-deciding ──────────────


def test_resume_reuses_persisted_model_without_engine(client: TestClient) -> None:
    _seed_external_session(client, external_id="ses-A", model=_SESSION_RESUME_MODEL)
    with (
        patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion,
        _no_decide(client) as decide,
    ):
        mock_acompletion.return_value = _canned_chat_response()
        resp = _post_chat(client, session_id="ses-A")

    assert resp.status_code == 200
    decision = resp.headers["X-Shunt-Decision"]
    assert _SESSION_RESUME_MODEL in decision
    assert "session_resume" in decision
    # The resumed session is a NEW session, not the seeded row's id.
    assert resp.headers["X-Shunt-Session-Id"] != "seeded-ses"
    # The engine was never consulted: the model came from the persisted row.
    decide.assert_not_called()

    session = client.app.state.session_manager.get_session_by_identity("ext:ses-A")
    assert session is not None
    assert session.model_chosen == _SESSION_RESUME_MODEL
    assert session.metadata["model_source"] == "session_resume"
    assert session.external_session_id == "ses-A"
    assert session.decision_provenance is not None
    assert session.decision_provenance["router_propensity"] is None
    # Not re-embedded: no last_prompt recorded on the resumed session.
    assert "last_prompt" not in session.metadata


def test_open_session_is_never_re_decided(client: TestClient) -> None:
    # The cache-safety invariant: once a session is open and has a model, a later request
    # with the same external id reuses it in-memory — even if the persisted row differs.
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _canned_chat_response()
        r1 = _post_chat(client, session_id="ses-A")
    first_sid = r1.headers["X-Shunt-Session-Id"]
    _seed_external_session(
        client, external_id="ses-A", model=_SESSION_RESUME_MODEL, session_id="other-ses"
    )
    with (
        patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion,
        _no_decide(client) as decide,
    ):
        mock_acompletion.return_value = _canned_chat_response()
        r2 = _post_chat(client, session_id="ses-A")
    assert r2.status_code == 200
    assert r2.headers["X-Shunt-Session-Id"] == first_sid
    # The in-memory open session was reused verbatim — resume never fired.
    decide.assert_not_called()


def test_fork_reuses_parent_model_without_engine(client: TestClient) -> None:
    _seed_external_session(client, external_id="ses-A", model=_SESSION_RESUME_MODEL)
    with (
        patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion,
        _no_decide(client) as decide,
    ):
        mock_acompletion.return_value = _canned_chat_response()
        resp = _post_chat(client, session_id="ses-C", parent_id="ses-A")

    assert resp.status_code == 200
    decision = resp.headers["X-Shunt-Decision"]
    assert _SESSION_RESUME_MODEL in decision
    assert "fork_resume" in decision
    decide.assert_not_called()

    session = client.app.state.session_manager.get_session_by_identity("ext:ses-C")
    assert session is not None
    assert session.model_chosen == _SESSION_RESUME_MODEL
    assert session.metadata["model_source"] == "fork_resume"


def test_resume_with_unknown_model_falls_through_to_normal_routing(client: TestClient) -> None:
    # The persisted model is not routable (not in the registry) → resume must NOT lock it;
    # normal routing decides instead.
    _seed_external_session(client, external_id="ses-A", model="ghost-model")
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _canned_chat_response()
        resp = _post_chat(client, session_id="ses-A")
    assert resp.status_code == 200
    assert "ghost-model" not in resp.headers["X-Shunt-Decision"]
