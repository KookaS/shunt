"""The live smoke's spend gates and pass criteria — pure, hermetic, no spend.

The guard functions are the unit under test: `--live`, an interactive TTY, an
explicit y and an injected key must all hold, in order, or the smoke refuses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmark.runner import live_smoke
from shunt.models.config import ModelConfig, default_registry_path
from shunt.router.policy import packaged_policy_path

# ── spend gates ───────────────────────────────────────────────────────────────


def test_live_flag_refusal() -> None:
    assert live_smoke.live_flag_refusal(live=False) is not None
    assert live_smoke.live_flag_refusal(live=True) is None


def test_tty_refusal() -> None:
    assert live_smoke.tty_refusal(lambda: False) is not None
    assert live_smoke.tty_refusal(lambda: True) is None


def test_confirm_refusal_requires_y() -> None:
    assert live_smoke.confirm_refusal(lambda _prompt: "y") is None
    assert live_smoke.confirm_refusal(lambda _prompt: "Y") is None
    assert live_smoke.confirm_refusal(lambda _prompt: " yes ") is None
    assert live_smoke.confirm_refusal(lambda _prompt: "n") is not None
    assert live_smoke.confirm_refusal(lambda _prompt: "") is not None


def test_confirm_refusal_on_eof() -> None:
    # EOF (None) is the non-interactive signal even when the TTY gate is bypassed.
    assert live_smoke.confirm_refusal(lambda _prompt: None) is not None


def test_key_refusal() -> None:
    assert live_smoke.key_refusal("DEEPSEEK_API_KEY", None) is not None
    assert live_smoke.key_refusal("DEEPSEEK_API_KEY", "") is not None
    assert live_smoke.key_refusal("DEEPSEEK_API_KEY", "sk-live") is None


def test_refusal_reason_all_gates_pass() -> None:
    assert (
        live_smoke.refusal_reason(
            live=True,
            isatty=lambda: True,
            confirm=lambda _prompt: "y",
            key_env="DEEPSEEK_API_KEY",
            key_value="sk-live",
        )
        is None
    )


def test_refusal_reason_checks_gates_in_order() -> None:
    # No --live refuses even with every other gate held open (cheapest gate first).
    reason = live_smoke.refusal_reason(
        live=False,
        isatty=lambda: True,
        confirm=lambda _prompt: "y",
        key_env="DEEPSEEK_API_KEY",
        key_value="sk-live",
    )
    assert reason is not None and "--live" in reason

    # A non-interactive TTY refuses even with the spend flag accepted.
    reason = live_smoke.refusal_reason(
        live=True,
        isatty=lambda: False,
        confirm=lambda _prompt: "y",
        key_env="DEEPSEEK_API_KEY",
        key_value="sk-live",
    )
    assert reason is not None and "TTY" in reason

    # A declined confirmation refuses even with the key present.
    reason = live_smoke.refusal_reason(
        live=True,
        isatty=lambda: True,
        confirm=lambda _prompt: "n",
        key_env="DEEPSEEK_API_KEY",
        key_value="sk-live",
    )
    assert reason is not None and "confirmation" in reason

    # A missing key refuses last — after the confirmation, before any server boots.
    reason = live_smoke.refusal_reason(
        live=True,
        isatty=lambda: True,
        confirm=lambda _prompt: "y",
        key_env="DEEPSEEK_API_KEY",
        key_value=None,
    )
    assert reason is not None and "DEEPSEEK_API_KEY" in reason


def test_main_refuses_without_live_flag(monkeypatch, capsys) -> None:
    # End-to-end gate: a bare run (no --live) exits 2 before any server starts.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert live_smoke.main([]) == 2
    assert "--live" in capsys.readouterr().err


def test_main_refuses_non_interactive_tty(monkeypatch, capsys) -> None:
    # --live accepted but stdin is not a TTY: refused (this is what keeps CI out).
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert live_smoke.main(["--live"]) == 2
    assert "interactive TTY" in capsys.readouterr().err


# ── expected model (the always_cheap target) ─────────────────────────────────


def test_cheapest_live_model_is_shipped_cheapest() -> None:
    # The packaged registry + router.yaml parse, restrict to the live set, and the
    # smoke's expected model is the cheapest of them (deepseek-v4-flash today).
    pool = live_smoke.build_pool(default_registry_path(), packaged_policy_path())
    expected = live_smoke.cheapest_live_model(pool)
    assert isinstance(expected, ModelConfig)
    assert expected.name == "deepseek-v4-flash"
    assert expected.api_key_env_var == "DEEPSEEK_API_KEY"


def test_resolve_config_defaults_to_packaged() -> None:
    registry, policy = live_smoke.resolve_config(None)
    assert registry.name == "models.yaml"
    assert policy.name == "router.yaml"
    assert registry.parent.name == "config"  # packaged registry lives in shunt.config


def test_resolve_config_config_dir() -> None:
    registry, policy = live_smoke.resolve_config("/tmp/cfg")
    assert registry == Path("/tmp/cfg/models.yaml")
    assert policy == Path("/tmp/cfg/router.yaml")


# ── pass criteria ─────────────────────────────────────────────────────────────


def test_decision_header_parse() -> None:
    assert live_smoke.decision_from_headers(
        {"X-Shunt-Decision": "deepseek-v4-flash; reason=always_cheap"}
    ) == ("deepseek-v4-flash", "always_cheap")
    assert live_smoke.decision_from_headers({}) is None


def test_verify_headers_pass() -> None:
    headers = {
        "X-Shunt-Decision": "deepseek-v4-flash; reason=always_cheap",
        "X-Shunt-Session-Id": "sess-1",
    }
    assert live_smoke.verify_headers(200, headers, "deepseek-v4-flash", "always_cheap") == []


def test_verify_headers_wrong_model_fails() -> None:
    headers = {
        "X-Shunt-Decision": "zai-glm-5.2; reason=always_cheap",
        "X-Shunt-Session-Id": "sess-1",
    }
    problems = live_smoke.verify_headers(200, headers, "deepseek-v4-flash", "always_cheap")
    assert any("zai-glm-5.2" in p for p in problems)


def test_verify_headers_missing_session_fails() -> None:
    problems = live_smoke.verify_headers(
        200,
        {"X-Shunt-Decision": "deepseek-v4-flash; reason=always_cheap"},
        "deepseek-v4-flash",
        "always_cheap",
    )
    assert any("X-Shunt-Session-Id" in p for p in problems)


def test_verify_headers_non_200_fails() -> None:
    problems = live_smoke.verify_headers(401, {}, "deepseek-v4-flash", "always_cheap")
    assert problems and "HTTP 401" in problems[0]


def test_content_problems() -> None:
    assert live_smoke.content_problems('{"choices":[{"message":{"content":"OK"}}]}') == []
    assert live_smoke.content_problems('{"choices":[]}') != []
    assert live_smoke.content_problems("not json") != []


def test_verify_capture_pass() -> None:
    row = {
        "model_chosen": "deepseek-v4-flash",
        "cost": 0.00012,
        "cost_known": 1,
        "prompt_text": "shunt-live-smoke: reply with the single word OK.",
        "decision_provenance": json.dumps(
            {
                "model_chosen": "deepseek-v4-flash",
                "selection_rule_used": "always_cheap",
                "router_propensity": 1.0,
            }
        ),
    }
    problems, warnings = live_smoke.verify_capture(
        row, "deepseek-v4-flash", 0.10, "shunt-live-smoke"
    )
    assert problems == []
    assert warnings == []


def test_verify_capture_missing_row_fails() -> None:
    problems, _warnings = live_smoke.verify_capture(None, "deepseek-v4-flash", 0.10, "marker")
    assert any("no session row" in p for p in problems)


def test_verify_capture_wrong_model_fails() -> None:
    row = {
        "model_chosen": "kimi-k3",
        "cost": 0.001,
        "cost_known": 1,
        "prompt_text": "shunt-live-smoke: x",
        "decision_provenance": json.dumps(
            {"model_chosen": "kimi-k3", "selection_rule_used": "always_cheap"}
        ),
    }
    problems, _warnings = live_smoke.verify_capture(
        row, "deepseek-v4-flash", 0.10, "shunt-live-smoke"
    )
    assert any("model_chosen" in p for p in problems)


def test_verify_capture_over_cap_fails() -> None:
    row = {
        "model_chosen": "deepseek-v4-flash",
        "cost": 5.0,
        "cost_known": 1,
        "prompt_text": "shunt-live-smoke: x",
        "decision_provenance": json.dumps(
            {"model_chosen": "deepseek-v4-flash", "selection_rule_used": "always_cheap"}
        ),
    }
    problems, _warnings = live_smoke.verify_capture(
        row, "deepseek-v4-flash", 0.10, "shunt-live-smoke"
    )
    assert any("max-cost" in p for p in problems)


def test_verify_capture_unreported_cost_warns() -> None:
    row = {
        "model_chosen": "deepseek-v4-flash",
        "cost": 0.0,
        "cost_known": 0,
        "prompt_text": "shunt-live-smoke: x",
        "decision_provenance": json.dumps(
            {
                "model_chosen": "deepseek-v4-flash",
                "selection_rule_used": "always_cheap",
                "router_propensity": 1.0,
            }
        ),
    }
    problems, warnings = live_smoke.verify_capture(
        row, "deepseek-v4-flash", 0.10, "shunt-live-smoke"
    )
    assert problems == []
    assert any("cost_known" in w for w in warnings)


def test_verify_capture_foreign_row_fails() -> None:
    row = {
        "model_chosen": "deepseek-v4-flash",
        "cost": 0.001,
        "cost_known": 1,
        "prompt_text": "someone else's traffic",
        "decision_provenance": json.dumps(
            {"model_chosen": "deepseek-v4-flash", "selection_rule_used": "always_cheap"}
        ),
    }
    problems, _warnings = live_smoke.verify_capture(
        row, "deepseek-v4-flash", 0.10, "shunt-live-smoke"
    )
    assert any("marker" in p for p in problems)
