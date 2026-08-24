"""`escalation.context_transfer: summary` — the one place shunt authors the conversation.

Covers the compaction itself, the frozen-prefix cache guarantee, every degrade path, and the
four conditions that gate the hook. Upstream mocked; nothing here spends.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shunt.models import ModelPool
from shunt.proxy.context_transfer import (
    ContextTransfer,
    ContextTransferPolicy,
    apply,
    compact,
    estimate_tokens,
    split_messages,
    summarisation_request,
)
from shunt.proxy.router import ProxyRouter
from shunt.router.policy import RouterPolicy
from shunt.session import Session, SessionManager

_ACOMPLETION_PATCH = "shunt.proxy.router._acompletion"
_OUTGOING = "deepseek-v4-flash"
_INCOMING = "kimi-k3"


# ── the compaction, pure ────────────────────────────────────────────────────


def test_split_keeps_leading_systems_and_the_trailing_turn() -> None:
    messages = [
        {"role": "system", "content": "tools"},
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    systems, history, trailing = split_messages(messages)
    assert [m["content"] for m in systems] == ["tools", "rules"]
    assert [m["content"] for m in history] == ["one", "two"]
    assert trailing["content"] == "three"


def test_compact_replaces_history_with_one_authored_user_turn() -> None:
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "fix the parser"},
        {"role": "assistant", "content": "edited parser.py"},
        {"role": "user", "content": "still failing"},
    ]
    prefix, consumed = compact(messages, "note")
    assert consumed == 3  # everything but the trailing turn
    assert prefix[0] == {"role": "system", "content": "rules"}  # system blocks verbatim
    assert prefix[1]["role"] == "user"
    assert prefix[1]["content"].endswith("note")
    # The model is told it is reading a summary, not a transcript.
    assert "summary" in prefix[1]["content"].lower()


def test_compact_refuses_a_request_with_no_history() -> None:
    with pytest.raises(ValueError, match="nothing to compact"):
        compact([{"role": "user", "content": "hi"}], "note")
    with pytest.raises(ValueError, match="nothing to compact"):
        compact([{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}], "note")


def test_summarisation_request_appends_one_instruction_turn() -> None:
    history = [{"role": "user", "content": "one"}]
    built = summarisation_request(history, 2000)
    assert built[:-1] == history
    assert "2000" in built[-1]["content"]


def test_estimate_tokens_is_monotone_and_nonzero() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 4000) > estimate_tokens("a" * 400)


def test_apply_realigns_later_turns_onto_the_frozen_prefix() -> None:
    prefix = ({"role": "system", "content": "rules"}, {"role": "user", "content": "note"})
    transfer = ContextTransfer(mode="summary", prefix=prefix, consumed=3)
    later = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
        {"role": "user", "content": "five"},
    ]
    out = apply(transfer, later)
    assert out[:2] == list(prefix)
    assert [m["content"] for m in out[2:]] == ["three", "four", "five"]


def test_apply_passes_through_when_the_client_rewound_below_the_boundary() -> None:
    transfer = ContextTransfer(
        mode="summary", prefix=({"role": "user", "content": "n"},), consumed=5
    )
    short = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    assert apply(transfer, short) == short  # never a silent drop


def test_apply_is_identity_under_full() -> None:
    messages = [{"role": "user", "content": "a"}]
    assert apply(ContextTransfer(mode="full", degraded_reason="x"), messages) == messages


# ── the hook, through the proxy ─────────────────────────────────────────────


def _response(content: str = "ok") -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(finish_reason="stop", message=MagicMock(content=content))]
    resp.choices[0].message.tool_calls = []
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, cost=0.0)
    resp.usage.prompt_tokens_details = MagicMock(cached_tokens=0)
    resp.model = _INCOMING
    resp.id = "resp-1"
    return resp


def _router(mode: str = "summary", max_tokens: int | None = 2000) -> ProxyRouter:
    return ProxyRouter(
        model_pool=ModelPool(),
        session_manager=SessionManager(),
        context_transfer=ContextTransferPolicy(mode=mode, max_tokens=max_tokens),
    )


def _escalated(router: ProxyRouter, source: str = "auto_escalation") -> Session:
    """A session already locked to an escalated rung, as the engine would leave it."""
    session = router._sessions.create_session("test-tool")
    session.model_chosen = _INCOMING
    session.metadata["model"] = _INCOMING
    session.metadata["model_source"] = source
    session.decision_provenance = {
        "model_chosen": _INCOMING,
        "selection_rule_used": source,
        "pre_escalation_model": _OUTGOING,
    }
    return session


def _turn(n: int) -> list[dict[str, Any]]:
    """The conversation a CLI resends on turn *n* (it always resends everything)."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "you are a coding agent"},
        {"role": "user", "content": "fix the parser"},
        {"role": "assistant", "content": "edited parser.py"},
    ]
    for i in range(n):
        messages.append({"role": "user", "content": f"turn {i}"})
        if i < n - 1:
            messages.append({"role": "assistant", "content": f"reply {i}"})
    return messages


@pytest.mark.asyncio
async def test_summary_replaces_the_prefix_and_the_outgoing_model_writes_it() -> None:
    router = _router()
    session = _escalated(router)
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = _response("handover note")
        await router.route_chat_completion({"messages": _turn(1)}, session)

    summariser_call, serving_call = mock_ac.call_args_list
    # The OUTGOING, already-warm model wrote the note — not the incoming one.
    assert summariser_call.args[0].name == _OUTGOING
    assert serving_call.args[0].name == _INCOMING
    sent = serving_call.kwargs["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "user"]
    assert "handover note" in sent[1]["content"]
    prov = session.decision_provenance or {}
    assert prov["context_transfer"] == {
        "mode": "summary",
        "summariser": _OUTGOING,
        "replaced_messages": 3,
        "prefix_messages": 2,
    }
    assert session.metadata["context_transfer_header"] == "summary"


@pytest.mark.asyncio
async def test_turn_two_sends_a_byte_identical_prefix_and_never_re_summarises() -> None:
    router = _router()
    session = _escalated(router)
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = _response("handover note")
        await router.route_chat_completion({"messages": _turn(1)}, session)
        first = mock_ac.call_args_list[-1].kwargs["messages"]
        await router.route_chat_completion({"messages": _turn(2)}, session)
        second = mock_ac.call_args_list[-1].kwargs["messages"]

    # THE cache-safety guarantee: the frozen prefix is resent byte for byte, so the provider
    # sees the same leading bytes and the prefix cache hits. A per-turn summary would be a
    # cache miss per turn and cost more than `full`.
    assert json.dumps(second[:2]) == json.dumps(first[:2])
    assert [m["content"] for m in second[2:]] == ["turn 0", "reply 0", "turn 1"]
    # Exactly one summariser call across both turns (2 serving calls + 1 summary).
    summarisers = [c for c in mock_ac.call_args_list if c.args[0].name == _OUTGOING]
    assert len(summarisers) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (RuntimeError("boom"), "summariser failed: RuntimeError"),
        ("", "summariser returned no text"),
        ("   ", "summariser returned no text"),
    ],
)
async def test_a_broken_summariser_degrades_to_full_with_the_list_intact(
    failure: Any, expected: str
) -> None:
    router = _router()
    session = _escalated(router)
    original = _turn(1)

    async def _side_effect(config: Any, **kwargs: Any) -> Any:
        if config.name == _OUTGOING:
            if isinstance(failure, Exception):
                raise failure
            return _response(failure)
        return _response("ok")

    with patch(_ACOMPLETION_PATCH, new=AsyncMock(side_effect=_side_effect)) as mock_ac:
        await router.route_chat_completion({"messages": list(original)}, session)

    sent = mock_ac.call_args_list[-1].kwargs["messages"]
    assert sent == original  # the FULL conversation, nothing dropped
    prov = session.decision_provenance or {}
    assert prov["context_transfer"]["mode"] == "full"
    assert prov["context_transfer"]["requested"] == "summary"
    assert prov["context_transfer"]["degraded_reason"] == expected


@pytest.mark.asyncio
async def test_an_over_budget_summary_degrades_to_full() -> None:
    router = _router(max_tokens=10)
    session = _escalated(router)
    original = _turn(1)
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = _response("word " * 500)
        await router.route_chat_completion({"messages": list(original)}, session)
    assert mock_ac.call_args_list[-1].kwargs["messages"] == original
    reason = (session.decision_provenance or {})["context_transfer"]["degraded_reason"]
    assert "over budget" in reason


@pytest.mark.asyncio
async def test_a_hanging_summariser_times_out_and_degrades_to_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shunt.proxy.context_transfer.SUMMARY_TIMEOUT_SECONDS", 0.01)
    router = _router()
    session = _escalated(router)
    original = _turn(1)

    async def _side_effect(config: Any, **kwargs: Any) -> Any:
        if config.name == _OUTGOING:
            await asyncio.sleep(5)
        return _response("ok")

    with patch(_ACOMPLETION_PATCH, new=AsyncMock(side_effect=_side_effect)) as mock_ac:
        await router.route_chat_completion({"messages": list(original)}, session)

    assert mock_ac.call_args_list[-1].kwargs["messages"] == original
    reason = (session.decision_provenance or {})["context_transfer"]["degraded_reason"]
    assert "timed out" in reason


@pytest.mark.asyncio
async def test_a_degraded_session_never_retries_on_a_later_turn() -> None:
    router = _router()
    session = _escalated(router)

    calls: list[str] = []

    async def _side_effect(config: Any, **kwargs: Any) -> Any:
        calls.append(config.name)
        if config.name == _OUTGOING:
            raise RuntimeError("boom")
        return _response("ok")

    with patch(_ACOMPLETION_PATCH, new=AsyncMock(side_effect=_side_effect)):
        await router.route_chat_completion({"messages": _turn(1)}, session)
        await router.route_chat_completion({"messages": _turn(2)}, session)

    assert calls.count(_OUTGOING) == 1  # one attempt, no retry loop


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["cheapest_above_threshold", "exploration_untested"])
async def test_the_hook_does_not_fire_on_a_non_escalated_session(source: str) -> None:
    router = _router()
    session = _escalated(router, source=source)
    original = _turn(1)
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = _response()
        await router.route_chat_completion({"messages": list(original)}, session)
    assert len(mock_ac.call_args_list) == 1  # no summariser call at all
    assert mock_ac.call_args.kwargs["messages"] == original
    assert "context_transfer" not in (session.decision_provenance or {})


@pytest.mark.asyncio
async def test_the_hook_does_not_fire_when_the_mode_is_full() -> None:
    router = _router(mode="full")
    session = _escalated(router)
    original = _turn(1)
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = _response()
        await router.route_chat_completion({"messages": list(original)}, session)
    assert len(mock_ac.call_args_list) == 1
    assert mock_ac.call_args.kwargs["messages"] == original


@pytest.mark.asyncio
async def test_the_hook_does_not_fire_mid_session_when_the_mode_is_switched_on_late() -> None:
    # A session that already served a turn is past its first request, so the decision is
    # never taken mid-conversation.
    router = _router()
    session = _escalated(router)
    session.metadata["last_turn_served_model"] = _INCOMING
    original = _turn(2)
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = _response()
        await router.route_chat_completion({"messages": list(original)}, session)
    assert len(mock_ac.call_args_list) == 1
    assert mock_ac.call_args.kwargs["messages"] == original


@pytest.mark.asyncio
async def test_escalation_floor_also_transfers() -> None:
    router = _router()
    session = _escalated(router, source="escalation_floor")
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = _response("note")
        await router.route_chat_completion({"messages": _turn(1)}, session)
    assert (session.decision_provenance or {})["context_transfer"]["mode"] == "summary"


@pytest.mark.asyncio
async def test_an_effort_only_escalation_does_not_transfer() -> None:
    # The ladder's first rung raises the reasoning arm and KEEPS the model, deliberately
    # leaving the cache namespace — and the warm prefix — untouched. Compacting there would
    # turn a cache hit into a miss AND pay a summariser call for it: strictly worse than
    # doing nothing, on the axis the feature exists to optimise.
    router = _router()
    session = _escalated(router)
    session.model_chosen = _OUTGOING  # effort rung: served model == outgoing model
    session.metadata["model"] = _OUTGOING
    session.decision_provenance = {
        "model_chosen": _OUTGOING,
        "selection_rule_used": "auto_escalation",
        "pre_escalation_model": _OUTGOING,
        "escalated_reasoning_arm": "think",
    }
    original = _turn(1)
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = _response()
        await router.route_chat_completion({"messages": list(original)}, session)

    assert len(mock_ac.call_args_list) == 1  # no summariser call
    assert mock_ac.call_args.kwargs["messages"] == original
    assert "context_transfer" not in (session.decision_provenance or {})


@pytest.mark.asyncio
async def test_a_rank_escalation_does_transfer() -> None:
    # The contrast to the effort rung above: a rank step hands the conversation to a model
    # that has never seen it, so the prefix is cold either way and there is a real transfer.
    router = _router()
    session = _escalated(router)  # pre_escalation_model != model_chosen
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = _response("note")
        await router.route_chat_completion({"messages": _turn(1)}, session)

    assert [c.args[0].name for c in mock_ac.call_args_list] == [_OUTGOING, _INCOMING]
    assert (session.decision_provenance or {})["context_transfer"]["mode"] == "summary"


@pytest.mark.asyncio
async def test_an_absent_pre_escalation_model_fails_closed() -> None:
    # A missing key must mean "do not fire", never "fire": without it the two rungs are
    # indistinguishable, and firing on a maybe breaks cache safety on the rung designed to
    # preserve it.
    router = _router()
    session = _escalated(router)
    session.decision_provenance = {
        "model_chosen": _INCOMING,
        "selection_rule_used": "auto_escalation",
    }
    original = _turn(1)
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = _response()
        await router.route_chat_completion({"messages": list(original)}, session)
    assert len(mock_ac.call_args_list) == 1
    assert mock_ac.call_args.kwargs["messages"] == original


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["session_resume", "fork_resume", "prefix_resume"])
async def test_a_resumed_session_never_triggers_a_context_transfer(source: str) -> None:
    # A resume continues an EXISTING conversation on an already-locked model: there is no
    # escalation boundary to compact at, and nothing was handed to a new model. Pinned rather
    # than left to the fact that a resume token happens not to be in ESCALATED_SOURCES.
    router = _router()
    session = router._sessions.create_session("test-tool")
    session.model_chosen = _INCOMING
    session.metadata["model"] = _INCOMING
    session.metadata["model_source"] = source
    session.decision_provenance = {
        "model_chosen": _INCOMING,
        "selection_rule_used": source,
        "escalated_reasoning_arm": "think",
    }
    original = _turn(1)
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = _response()
        await router.route_chat_completion({"messages": list(original)}, session)

    assert len(mock_ac.call_args_list) == 1  # no summariser call
    assert mock_ac.call_args.kwargs["messages"] == original
    assert "context_transfer" not in (session.decision_provenance or {})


# ── config ──────────────────────────────────────────────────────────────────


def test_summary_mode_round_trips_through_the_policy() -> None:
    policy = RouterPolicy.model_validate(
        {
            "escalation": {
                "enabled": True,
                "context_transfer": "summary",
                "context_transfer_max_tokens": 500,
                "context_transfer_model": "gpt-5-mini",
            }
        }
    )
    assert policy.escalation.context_transfer == "summary"
    assert policy.escalation.context_transfer_max_tokens == 500
    assert policy.escalation.context_transfer_model == "gpt-5-mini"
    # The application knob stays OUT of the pure-logic escalation config.
    assert not hasattr(policy.escalation.to_config(), "context_transfer")


def test_the_default_is_full_pass_through() -> None:
    policy = RouterPolicy()
    assert policy.escalation.context_transfer == "full"
    assert policy.escalation.context_transfer_max_tokens == 2000
    assert policy.escalation.context_transfer_model is None


def test_none_is_refused_with_the_reason() -> None:
    with pytest.raises(ValueError, match="cannot make the client forget"):
        RouterPolicy.model_validate({"escalation": {"context_transfer": "none"}})


# ── disclosure surfaces ─────────────────────────────────────────────────────


def _summary_policy() -> RouterPolicy:
    return RouterPolicy.model_validate(
        {"escalation": {"enabled": True, "context_transfer": "summary"}}
    )


def test_boot_warns_that_the_model_will_not_see_what_the_user_sees(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from shunt.proxy.server import _build_work_dir_resolver, _log_capture_disclosure

    policy = _summary_policy()
    with caplog.at_level("WARNING"):
        _log_capture_disclosure(policy, _build_work_dir_resolver(policy, ""))
    assert any("DOES NOT SEE WHAT YOU SEE" in r.getMessage() for r in caplog.records)


def test_boot_is_silent_about_context_under_the_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from shunt.proxy.server import _build_work_dir_resolver, _log_capture_disclosure

    policy = RouterPolicy()
    with caplog.at_level("WARNING"):
        _log_capture_disclosure(policy, _build_work_dir_resolver(policy, ""))
    assert not any("context_transfer" in r.getMessage() for r in caplog.records)


def test_doctor_reports_the_transfer_under_the_escalation_check() -> None:
    from shunt.router.diagnostics import _escalation_check

    check = _escalation_check(_summary_policy(), None)
    assert "context_transfer: SUMMARY" in check.detail
    assert check.warn
    assert _escalation_check(RouterPolicy(), None).detail.count("\n") == 0


@pytest.mark.asyncio
async def test_the_response_carries_x_shunt_context_once() -> None:
    from shunt.proxy.server import _build_decision_headers

    router = _router()
    session = _escalated(router)
    with patch(_ACOMPLETION_PATCH, new_callable=AsyncMock) as mock_ac:
        mock_ac.return_value = _response("note")
        await router.route_chat_completion({"messages": _turn(1)}, session)
    first = await _build_decision_headers(session, _INCOMING, "auto_escalation")
    assert first["X-Shunt-Context"] == "summary"
    # One-shot: the substitution happened on that turn, not on every turn after it.
    second = await _build_decision_headers(session, _INCOMING, "auto_escalation")
    assert "X-Shunt-Context" not in second


def test_explain_prints_the_context_line(capsys: pytest.CaptureFixture[str]) -> None:
    from shunt.cli import _print_context_transfer

    _print_context_transfer(
        {"context_transfer": {"mode": "summary", "summariser": _OUTGOING, "replaced_messages": 7}}
    )
    out = capsys.readouterr().out
    assert "Context:" in out
    assert _OUTGOING in out
    assert "did not see the full conversation" in out

    _print_context_transfer({"context_transfer": {"mode": "full", "degraded_reason": "boom"}})
    assert "degraded: boom" in capsys.readouterr().out

    _print_context_transfer({})
    assert capsys.readouterr().out == ""
