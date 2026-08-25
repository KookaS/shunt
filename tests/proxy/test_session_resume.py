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

from shunt.proxy.prefix import compute_prefix_digest, is_replayed_conversation
from shunt.proxy.server import _cache_key_session_id, _get_external_session_id, app
from shunt.session import SessionManager

_ACOMPLETION_PATCH = "shunt.proxy.router._acompletion"
# A real registry model with a "max" reasoning arm (the resume tests exercise escalated-arm
# restore). It coincides with the cold-start default; the "not re-decided" proof rests on
# `_no_decide`/`decide.assert_not_called()`, not on the model name differing from the default.
_SESSION_RESUME_MODEL = "deepseek-v4-flash"

_TOOL_IDENTITY: Final[str] = SessionManager.compute_tool_identity("testclient", "shunt-e2e/0.1")

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


def _seed_external_session(  # noqa: PLR0913 (one arg per persisted column under test)
    client: TestClient,
    *,
    external_id: str | None = None,
    model: str,
    session_id: str = "seeded-ses",
    prefix_digest: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    """Persist a session row — what a PRIOR run of this conversation would have written."""
    client.app.state.outcome_store.store_session(
        session_id=session_id,
        prompt_text="old conversation",
        embedding=None,
        model_chosen=model,
        cost=0.0,
        cache_stats={},
        duration=1.0,
        decision_provenance=provenance,
        external_session_id=external_id,
        prefix_digest=prefix_digest,
    )


def _digest_for(client: TestClient, body: dict[str, object], ua: str = "shunt-e2e/0.1") -> str:
    """The prefix digest the server will compute for *body* sent with *ua* from this client."""
    identity = SessionManager.compute_tool_identity("testclient", ua)
    work_dir = client.app.state.work_dir_resolver.resolve_identity(identity)
    digest = compute_prefix_digest(body, identity, work_dir)
    assert digest is not None
    return digest


def _replay_body(opening: str = "Hi", turns: int = 1) -> dict[str, Any]:
    """A body REPLAYING a conversation: the opening turn plus *turns* completed exchanges."""
    # Byte-identical opening block to `_CHAT_BODY`, so it hashes to the same digest — the only
    # difference is that the history is present, which is what makes it a resume rather than a
    # fresh conversation that happens to repeat a question.
    messages: list[dict[str, Any]] = [{"role": "user", "content": opening}]
    for i in range(turns):
        messages.append({"role": "assistant", "content": f"reply {i}"})
        messages.append({"role": "user", "content": f"next {i}"})
    return {"messages": messages, "stream": False}


def _post_chat(
    client: TestClient,
    *,
    session_id: str | None = None,
    parent_id: str | None = None,
    extra_headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> Any:
    headers = {"User-Agent": "shunt-e2e/0.1"}
    if extra_headers:
        headers.update(extra_headers)
    if session_id is not None:
        headers["X-Session-Id"] = session_id
    if parent_id is not None:
        headers["x-parent-session-id"] = parent_id
    return client.post("/v1/chat/completions", json=body or _CHAT_BODY, headers=headers)


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


def test_x_shunt_session_id_is_checked_before_the_other_aliases() -> None:
    sid, _ = _get_external_session_id(
        _scope_request(
            {
                "x-shunt-session-id": "ses-0",
                "x-session-id": "ses-1",
                "x-session-affinity": "ses-2",
            }
        )
    )
    assert sid == "ses-0"


def test_the_cache_key_tier_is_declared_and_empty() -> None:
    # Tier 2 of the cascade: no provider we serve mints a readable cache key, and the tier
    # says so rather than being silently absent. Pinned so wiring one is a deliberate edit.
    assert _cache_key_session_id(_scope_request({}), dict(_CHAT_BODY)) is None


# ── the escalated reasoning arm survives the resume ───────────────────────────


def test_resume_restores_the_escalated_reasoning_arm(client: TestClient) -> None:
    # The regression: restoring the model but not the arm silently re-serves the BASE effort
    # of a conversation that had already escalated, and `explain` shows a resume with no arm.
    _seed_external_session(
        client,
        external_id="ses-A",
        model=_SESSION_RESUME_MODEL,
        provenance={
            "model_chosen": _SESSION_RESUME_MODEL,
            "selection_rule_used": "auto_escalation",
            "escalated_reasoning_arm": "max",
        },
    )
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _canned_chat_response()
        resp = _post_chat(client, session_id="ses-A")

    assert resp.status_code == 200
    session = client.app.state.session_manager.get_session_by_identity("ext:ses-A")
    assert session is not None
    assert session.metadata["reasoning_arm"] == "max"
    # Carried into the new provenance, so `shunt explain` still names the arm being served.
    assert session.decision_provenance is not None
    assert session.decision_provenance["escalated_reasoning_arm"] == "max"
    assert session.decision_provenance["selection_rule_used"] == "session_resume"
    # And it actually reaches upstream on the resumed turn.
    assert mock_acompletion.call_args.kwargs["reasoning_effort"] == "max"


def test_an_arm_foreign_to_the_resumed_model_is_not_restored(client: TestClient) -> None:
    # The effort ladder's own rule: an arm this model does not declare resolves to the
    # model's default, i.e. there is no escalation to carry. No second rule, no invention.
    _seed_external_session(
        client,
        external_id="ses-A",
        model=_SESSION_RESUME_MODEL,
        provenance={"escalated_reasoning_arm": "think"},  # a qwen arm, not a deepseek one
    )
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _canned_chat_response()
        _post_chat(client, session_id="ses-A")
    session = client.app.state.session_manager.get_session_by_identity("ext:ses-A")
    assert session is not None
    assert "reasoning_arm" not in session.metadata
    assert session.decision_provenance is not None
    assert "escalated_reasoning_arm" not in session.decision_provenance


def test_resume_restores_the_frozen_summary_prefix(client: TestClient) -> None:
    # The regression: a resumed summarised session restored the model and the arm but dropped
    # `context_transfer`, so the client's ORIGINAL messages were resent to a model that had been
    # handed a compaction — a cache MISS every turn, which is the cost `summary` exists to avoid.
    # (Reproduced live under a 1s idle timeout; invisible at the 900s default.)
    _seed_external_session(
        client,
        external_id="ses-A",
        model=_SESSION_RESUME_MODEL,
        provenance={
            "model_chosen": _SESSION_RESUME_MODEL,
            "selection_rule_used": "auto_escalation",
            "context_transfer": {"mode": "summary", "summariser": "qwen3.7-plus"},
            "context_transfer_prefix": {
                "mode": "summary",
                "prefix": [{"role": "user", "content": "[shunt context transfer] frozen"}],
                "consumed": 2,
                "summariser": "qwen3.7-plus",
            },
        },
    )
    body = {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "next"},
        ],
        "stream": False,
    }
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _canned_chat_response()
        resp = _post_chat(client, session_id="ses-A", body=body)

    assert resp.status_code == 200
    # The frozen prefix replaced the two consumed messages; the turn since it is kept verbatim.
    assert mock_acompletion.call_args.kwargs["messages"] == [
        {"role": "user", "content": "[shunt context transfer] frozen"},
        {"role": "user", "content": "next"},
    ]
    session = client.app.state.session_manager.get_session_by_identity("ext:ses-A")
    assert session is not None
    assert session.metadata["context_transfer"].consumed == 2
    # Re-persisted, so the NEXT eviction-and-resume of this conversation restores it too.
    assert session.decision_provenance is not None
    assert session.decision_provenance["context_transfer_prefix"]["consumed"] == 2
    assert session.decision_provenance["context_transfer"]["mode"] == "summary"


def test_resume_without_a_stored_prefix_forwards_the_client_messages(client: TestClient) -> None:
    # `full` sessions (the default) carry no prefix; nothing must be invented for them.
    _seed_external_session(
        client,
        external_id="ses-A",
        model=_SESSION_RESUME_MODEL,
        provenance={"model_chosen": _SESSION_RESUME_MODEL},
    )
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _canned_chat_response()
        _post_chat(client, session_id="ses-A")
    session = client.app.state.session_manager.get_session_by_identity("ext:ses-A")
    assert session is not None
    assert "context_transfer" not in session.metadata
    assert mock_acompletion.call_args.kwargs["messages"] == _CHAT_BODY["messages"]


# ── tier 3: the prompt-prefix digest (no session id anywhere) ─────────────────


def test_a_replayed_conversation_resolves_to_its_own_prior_turns(client: TestClient) -> None:
    # (b) The tier's reason for existing: a request carrying the conversation's history is a
    # resume, and it must find the row its own opening turn wrote.
    digest = _digest_for(client, dict(_CHAT_BODY))
    _seed_external_session(client, model=_SESSION_RESUME_MODEL, prefix_digest=digest, provenance={})
    with (
        patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion,
        _no_decide(client) as decide,
    ):
        mock_acompletion.return_value = _canned_chat_response()
        resp = _post_chat(client, body=_replay_body())

    assert resp.status_code == 200
    decision = resp.headers["X-Shunt-Decision"]
    assert _SESSION_RESUME_MODEL in decision
    assert "prefix_resume" in decision
    decide.assert_not_called()
    session = client.app.state.session_manager.get_session_by_identity(_TOOL_IDENTITY)
    assert session is not None
    assert session.prefix_digest == digest


def test_an_ambiguous_prefix_opens_a_fresh_session(client: TestClient) -> None:
    # Two stored conversations share this digest, so resolving would risk attaching this
    # conversation's verified outcome to the other one. Route cold instead.
    digest = _digest_for(client, dict(_CHAT_BODY))
    _seed_external_session(
        client, model=_SESSION_RESUME_MODEL, session_id="seed-1", prefix_digest=digest
    )
    _seed_external_session(client, model="glm-5.2", session_id="seed-2", prefix_digest=digest)
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _canned_chat_response()
        resp = _post_chat(client, body=_replay_body())

    assert resp.status_code == 200
    assert "prefix_resume" not in resp.headers["X-Shunt-Decision"]
    session = client.app.state.session_manager.get_session_by_identity(_TOOL_IDENTITY)
    assert session is not None
    assert session.metadata.get("model_source") != "prefix_resume"


def test_live_grouping_still_puts_one_client_on_one_session(client: TestClient) -> None:
    # The digest deliberately does NOT split live traffic: consecutive requests from one
    # header-less client remain ONE session, which is what per-session cumulative spend and
    # the one-decision-per-session rule are counted on. The digest identifies a conversation
    # across a RESTART; it is not a live demultiplexer.
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _canned_chat_response()
        first = _post_chat(client)
        second = _post_chat(client, body=_replay_body(opening="Something else entirely"))
    assert first.headers["X-Shunt-Session-Id"] == second.headers["X-Shunt-Session-Id"]


def test_a_declared_session_id_is_never_re_keyed_onto_a_prefix(client: TestClient) -> None:
    digest = _digest_for(client, dict(_CHAT_BODY))
    _seed_external_session(
        client, model=_SESSION_RESUME_MODEL, prefix_digest=digest, session_id="seed-prefix"
    )
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _canned_chat_response()
        # A REPLAY, so the prefix tier would resolve if the header had not won outright.
        resp = _post_chat(client, session_id="ses-declared", body=_replay_body())
    # The header wins tier 1 outright: no prefix lookup, no prefix key on the session.
    assert "prefix_resume" not in resp.headers["X-Shunt-Decision"]
    session = client.app.state.session_manager.get_session_by_identity("ext:ses-declared")
    assert session is not None
    assert session.prefix_digest is None


# ── the opening/replay distinction (the uniqueness gap the digest alone leaves) ──


def test_two_independent_openings_with_the_same_prompt_are_two_sessions(
    client: TestClient,
) -> None:
    # (a) The defect this gate closes: a byte-identical opening question, same repo, same tool
    # identity, is a DIFFERENT conversation — not a resume. Both must route on their own.
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _canned_chat_response()
        first = _post_chat(client)
        sid1 = first.headers["X-Shunt-Session-Id"]
        # Close the first, exactly as a finished conversation does, then ask the same thing.
        client.app.state.session_manager.close_session(sid1)
        second = _post_chat(client)

    sid2 = second.headers["X-Shunt-Session-Id"]
    assert sid1 != sid2
    for resp in (first, second):
        assert "prefix_resume" not in resp.headers["X-Shunt-Decision"]


def test_an_opening_request_never_consults_the_prefix_tier(client: TestClient) -> None:
    # Even with a matching stored row sitting there, an OPENING request routes cold: the row
    # belongs to whatever conversation asked this question first, and this is not that one.
    digest = _digest_for(client, dict(_CHAT_BODY))
    _seed_external_session(client, model=_SESSION_RESUME_MODEL, prefix_digest=digest)
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _canned_chat_response()
        resp = _post_chat(client)
    assert "prefix_resume" not in resp.headers["X-Shunt-Decision"]
    session = client.app.state.session_manager.get_session_by_identity(_TOOL_IDENTITY)
    assert session is not None
    assert session.metadata.get("model_source") != "prefix_resume"


def test_an_opening_turn_still_writes_the_key_a_later_resume_reads(client: TestClient) -> None:
    # The write side is deliberately NOT gated: the opening turn persists the digest, and the
    # resume that arrives later — replaying this conversation — resolves against that row.
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = _canned_chat_response()
        first = _post_chat(client)
    sid1 = first.headers["X-Shunt-Session-Id"]
    model1 = first.headers["X-Shunt-Decision"].partition("; reason=")[0]
    digest = _digest_for(client, dict(_CHAT_BODY))
    stored = client.app.state.outcome_store.get_session(sid1)
    assert stored is not None
    assert stored["prefix_digest"] == digest

    # The conversation ends, then comes back replaying its history.
    client.app.state.session_manager.close_session(sid1)
    with (
        patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_acompletion,
        _no_decide(client) as decide,
    ):
        mock_acompletion.return_value = _canned_chat_response()
        resumed = _post_chat(client, body=_replay_body())
    assert "prefix_resume" in resumed.headers["X-Shunt-Decision"]
    assert model1 in resumed.headers["X-Shunt-Decision"]
    decide.assert_not_called()


def test_the_replay_gate_counts_turns_not_scaffolding() -> None:
    # An OpenAI opening request carries `[system, user]`; counting the system block would
    # class it as a replay and reopen the whole defect.
    assert not is_replayed_conversation(dict(_CHAT_BODY))
    assert not is_replayed_conversation(
        {"messages": [{"role": "system", "content": "you are"}, {"role": "user", "content": "hi"}]}
    )
    assert not is_replayed_conversation({"system": "you are", "messages": []})
    assert is_replayed_conversation(_replay_body())
    assert is_replayed_conversation(
        {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}
    )
