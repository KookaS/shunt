"""Local-serving pre-flight: refuse an endpoint that can corrupt a run while returning 200 OK.

All stubbed (no server, no `ps`): covers endpoint classification, command-line parsing, each
refusal arm, the hosted no-op, and the wiring into ``infer.preflight_api_check``.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from benchmark import config
from benchmark.runner import infer, serving_guard

# The literal command line ollama 0.33.0 spawned on the run that motivated this guard.
OLLAMA_CMD = (
    "/home/node/.local/lib/ollama/llama-server --model /blobs/sha256-fb66ad57 --port 44147 "
    "-c 16384 -np 1 --cache-type-k q8_0 --flash-attn on -b 512 -ub 512 --context-shift --keep 4"
)
SAFE_CMD = (
    "/home/node/.local/lib/ollama/llama-server --model /blobs/sha256-fb66ad57 --port 18080 "
    "-c 32768 -np 1 -ngl 999 --jinja --no-context-shift --cache-type-k q8_0 --flash-attn on"
)
LOCAL_URL = "http://127.0.0.1:44147/v1"


def _stub_ps(monkeypatch: pytest.MonkeyPatch, *commands: str) -> None:
    monkeypatch.setattr(serving_guard, "_server_command_lines", lambda: list(commands))


def _stub_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence the artifact write; the log line is exercised, the scratch path is not."""
    monkeypatch.setattr(serving_guard, "_record", lambda *_a, **_kw: None)


# --- endpoint classification ----------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434/v1",
        "http://127.0.0.1:8080/v1",
        "http://host.docker.internal/v1",
        "http://[::1]:8080/v1",
        # Everything below classified HOSTED under the old five-name allowlist, silently
        # disabling the guard — including the compose rig's own service-name form.
        "http://ollama:11434/v1",  # docker-compose service name (single label)
        "http://192.168.1.50:8080/v1",  # RFC1918 LAN box
        "http://10.0.0.5:8080/v1",
        "http://127.0.0.2:8080/v1",  # loopback is a /8, not one address
        "http://gpu-box.lan:8080/v1",
        "http://llama.local:8080/v1",
        "http://[fd00::1]:8080/v1",
        "unix:///var/run/llama.sock",
    ],
)
def test_local_endpoints_recognised(url: str) -> None:
    assert serving_guard.is_local_endpoint(url) is True


@pytest.mark.parametrize(
    "url", [None, "", "https://api.deepseek.com/v1", "https://x.ai/v1", "http://8.8.8.8:8080/v1"]
)
def test_hosted_endpoints_are_not_local(url: str | None) -> None:
    assert serving_guard.is_local_endpoint(url) is False


@pytest.mark.parametrize(
    ("url", "match"),
    [
        ("http://100.64.0.1:8080/v1", "neither private nor globally routable"),  # CGNAT / overlay
        ("127.0.0.1:8080/v1", "no hostname"),  # scheme-less: urlsplit spells the host as a path
    ],
)
def test_an_undecidable_endpoint_refuses_instead_of_defaulting_to_hosted(
    url: str, match: str
) -> None:
    # Failing closed matters more than the classification: "hosted" is a silent no-op, and the
    # subclass makes every existing UnsafeServingError handler catch it.
    with pytest.raises(serving_guard.UndecidableEndpointError, match=match):
        serving_guard.is_local_endpoint(url)
    assert issubclass(serving_guard.UndecidableEndpointError, serving_guard.UnsafeServingError)


# --- command-line parsing -------------------------------------------------------------


def test_parses_ollama_destructive_defaults() -> None:
    # The whole point: ollama overrides llama.cpp's disabled default with --keep 4.
    assert serving_guard.parse_server_command(OLLAMA_CMD) == (16384, True, 4)


def test_parses_explicit_safe_serving() -> None:
    assert serving_guard.parse_server_command(SAFE_CMD) == (32768, False, None)


def test_flags_are_read_in_the_equals_spelling_too() -> None:
    # `--ctx-size=32768` and `--port=18080` are the same flags; a value-follows-token reader
    # silently returned None for the first and mis-matched the second.
    assert serving_guard.parse_server_command("llama-server --ctx-size=32768 --keep=4") == (
        32768,
        None,
        4,
    )
    assert serving_guard.server_port("llama-server --port=18080") == 18080


def test_absent_shift_flag_is_unknown_not_safe() -> None:
    # An absent flag means the supervisor's compiled default governs — unknown, never assumed off.
    _n_ctx, shift, _keep = serving_guard.parse_server_command("llama-server -c 8192")
    assert shift is None


# --- process IDENTIFICATION (which process IS a server) ---------------------------------

# Every one of these carries the text "llama-server" and none of them IS one. Under substring
# identification each was adopted as the serving configuration and ITS flags were asserted.
IMPOSTORS = (
    "/opt/start_llama-server.sh --port 18080 -c 262144 --no-context-shift",  # wrapper script
    "grep llama-server --port 18080 -c 262144 --no-context-shift",  # someone looking for it
    "vim /home/node/llama-server.service",  # an editor with the unit file open
    "/usr/bin/python3 /opt/llama-server/serve.py --port 18080 -c 262144 --no-context-shift",
    "tail -f /var/log/llama-server.log",
)


def _stub_ps_output(monkeypatch: pytest.MonkeyPatch, *lines: str) -> None:
    """Stub `ps` itself, so the identification filter under test actually runs."""

    class _Completed:
        stdout = "\n".join(lines) + "\n"

    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: _Completed())


@pytest.mark.parametrize("command", IMPOSTORS)
def test_a_process_merely_naming_the_server_is_not_the_server(command: str) -> None:
    assert serving_guard.is_server_command(command) is False


@pytest.mark.parametrize(
    "command",
    [
        OLLAMA_CMD,
        SAFE_CMD,
        "llama-server --port 18080",  # bare, resolved from PATH
        "/usr/local/bin/llama-cpp-server --port 18080",
        "'/opt/my dir/llama-server' --port 18080",  # a quoted path with a space
    ],
)
def test_the_real_server_is_identified_by_its_executable(command: str) -> None:
    assert serving_guard.is_server_command(command) is True


def test_impostors_are_filtered_out_of_the_ps_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ps_output(monkeypatch, *IMPOSTORS, SAFE_CMD)
    assert serving_guard._server_command_lines() == [SAFE_CMD]


def test_a_wrapper_script_is_never_adopted_as_the_serving_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The residual: a shell script whose FILENAME contains "llama-server", bound to the very
    # port under measurement, was read as the serving configuration and returned PASS. What a
    # wrapper hands the real server is not what the wrapper's own command line says.
    _stub_ps_output(monkeypatch, *IMPOSTORS)
    _stub_record(monkeypatch)
    with pytest.raises(serving_guard.UnsafeServingError, match="no inspectable"):
        serving_guard.assert_serving_safe(
            "http://127.0.0.1:18080/v1", max_tokens=6000, prompt_budget=26000
        )


def test_the_real_server_still_passes_through_the_real_ps_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The filter must not be so tight it rejects the thing it exists to find: the same listing
    # with the real server present resolves it, impostors and all.
    _stub_ps_output(monkeypatch, *IMPOSTORS, SAFE_CMD)
    _stub_record(monkeypatch)
    resolved = serving_guard.assert_serving_safe(
        "http://127.0.0.1:18080/v1", max_tokens=6000, prompt_budget=26000
    )
    assert resolved is not None and resolved.command == SAFE_CMD


# --- the assertion ---------------------------------------------------------------------


def test_hosted_endpoint_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ps(monkeypatch)  # no server anywhere; a hosted URL must still pass
    assert (
        serving_guard.assert_serving_safe(
            "https://api.deepseek.com/v1", max_tokens=None, prompt_budget=None
        )
        is None
    )


def test_context_shift_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ps(monkeypatch, OLLAMA_CMD)
    _stub_record(monkeypatch)
    with pytest.raises(serving_guard.UnsafeServingError, match="context shift enabled"):
        serving_guard.assert_serving_safe(LOCAL_URL, max_tokens=6000, prompt_budget=10000)


def test_undeclared_shift_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ps(monkeypatch, "llama-server --port 44147 -c 32768")
    _stub_record(monkeypatch)
    with pytest.raises(serving_guard.UnsafeServingError, match="not declared either way"):
        serving_guard.assert_serving_safe(LOCAL_URL, max_tokens=6000, prompt_budget=10000)


def test_insufficient_n_ctx_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ps(monkeypatch, SAFE_CMD)
    _stub_record(monkeypatch)
    with pytest.raises(serving_guard.UnsafeServingError, match="below the"):
        serving_guard.assert_serving_safe(
            "http://127.0.0.1:18080/v1", max_tokens=6000, prompt_budget=30000
        )


def test_undeclared_budget_refuses_with_the_key_to_add(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ps(monkeypatch, SAFE_CMD)
    _stub_record(monkeypatch)
    monkeypatch.setattr(serving_guard, "serving_prompt_budget", lambda: None)
    with pytest.raises(serving_guard.UnsafeServingError, match=r"live\.serving\.prompt_budget"):
        serving_guard.assert_serving_safe("http://127.0.0.1:18080/v1", max_tokens=6000)


def test_undeclared_max_tokens_refuses_with_the_key_to_add(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ps(monkeypatch, SAFE_CMD)
    _stub_record(monkeypatch)
    with pytest.raises(serving_guard.UnsafeServingError, match=r"live\.serving\.max_tokens"):
        serving_guard.assert_serving_safe(
            "http://127.0.0.1:18080/v1", max_tokens=None, prompt_budget=26000
        )


def test_invisible_server_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    # "Could not tell" is not neutral: ollama's default IS destructive head-eviction.
    _stub_ps(monkeypatch)
    with pytest.raises(serving_guard.UnsafeServingError, match="no inspectable"):
        serving_guard.assert_serving_safe(LOCAL_URL, max_tokens=6000, prompt_budget=10000)


def test_correct_serving_passes_and_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ps(monkeypatch, OLLAMA_CMD, SAFE_CMD)  # two servers; the port disambiguates
    _stub_record(monkeypatch)
    resolved = serving_guard.assert_serving_safe(
        "http://127.0.0.1:18080/v1", max_tokens=6000, prompt_budget=26000
    )
    assert resolved is not None
    assert (resolved.n_ctx, resolved.context_shift) == (32768, False)


def test_two_servers_on_one_port_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ps(monkeypatch, SAFE_CMD, SAFE_CMD.replace("-c 32768", "-c 8192"))
    with pytest.raises(serving_guard.UnsafeServingError, match="matches 2"):
        serving_guard.assert_serving_safe(
            "http://127.0.0.1:18080/v1", max_tokens=6000, prompt_budget=10000
        )


def test_a_process_on_another_port_is_never_attributed(monkeypatch: pytest.MonkeyPatch) -> None:
    # The unmatched-fallback bug: one safe server on 9999 was attributed to an endpoint on
    # 18080 and returned PASS — a fabricated verdict about a process serving nobody.
    _stub_ps(monkeypatch, SAFE_CMD.replace("--port 18080", "--port 9999"))
    _stub_record(monkeypatch)
    with pytest.raises(serving_guard.UnsafeServingError, match="no inference server process"):
        serving_guard.assert_serving_safe(
            "http://127.0.0.1:18080/v1", max_tokens=6000, prompt_budget=26000
        )


def test_the_ollama_proxy_topology_refuses_rather_than_reading_the_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ollama listens on 11434 and spawns llama-server on a random high port, so NO process binds
    # the endpoint's port. The child here is safely served, which is what made the old fallback
    # return PASS for an endpoint whose own policy was never read — on the exact topology this
    # module exists for.
    _stub_ps(monkeypatch, SAFE_CMD.replace("--port 18080", "--port 44147"))
    _stub_record(monkeypatch)
    with pytest.raises(serving_guard.UnsafeServingError, match="EXACTLY ollama's topology"):
        serving_guard.assert_serving_safe(
            "http://127.0.0.1:11434/v1", max_tokens=6000, prompt_budget=26000
        )


def test_a_port_bearing_string_elsewhere_on_the_line_does_not_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Substring matching read `--port 18080` out of a model PATH; attribution is by argv.
    _stub_ps(monkeypatch, "llama-server --model /blobs/--port 18080/ggml.bin --port 44147 -c 4096")
    with pytest.raises(serving_guard.UnsafeServingError, match="no inference server process"):
        serving_guard.assert_serving_safe(
            "http://127.0.0.1:18080/v1", max_tokens=6000, prompt_budget=26000
        )


# --- wiring into the existing pre-flight ------------------------------------------------


def test_preflight_refuses_unsafe_local_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    monkeypatch.setattr(infer, "_cheapest_enabled_model", lambda: "local8b")
    monkeypatch.setattr(
        infer, "litellm_model_target", lambda _m: ("openai/local8b", {"api_base": LOCAL_URL})
    )
    monkeypatch.setattr(infer, "_effective_max_tokens", lambda _m: 6000)
    monkeypatch.setattr(serving_guard, "serving_prompt_budget", lambda: 10000)
    _stub_ps(monkeypatch, OLLAMA_CMD)
    _stub_record(monkeypatch)

    def _never(**_kw: Any) -> object:
        raise AssertionError("the health probe must not run once serving is refused")

    monkeypatch.setattr(litellm, "completion", _never)
    with pytest.raises(infer.ApiUnusableError, match="unsafe local serving config"):
        infer.preflight_api_check()


# --- satisfiability: the guard must be passable, not merely a wall ----------------------


def test_the_shipped_config_declares_both_arithmetic_inputs() -> None:
    # A guard that can only ever refuse is a wall. benchmark.yaml must supply both operands so a
    # correctly served local endpoint PASSES without editing the model registry.
    assert serving_guard.serving_prompt_budget() is not None
    assert serving_guard.serving_max_tokens() is not None


def test_effective_max_tokens_falls_back_to_the_declared_cap() -> None:
    # Non-vacuous by construction: `== serving_max_tokens()` alone also holds in the state this
    # test exists to exclude (both sides None, the guard unsatisfiable for every model). The
    # positive assertions below are what fail there — a POSITIVE INT, for EVERY enabled model.
    declared = serving_guard.serving_max_tokens()
    assert isinstance(declared, int) and declared > 0, "the declared fallback itself is missing"
    enabled = config.enabled_models()
    assert enabled, "no enabled models to check the cap against"
    caps = {model: infer._effective_max_tokens(model) for model in enabled}
    assert all(cap is not None for cap in caps.values()), f"an unbounded model: {caps}"
    assert caps == dict.fromkeys(enabled, declared)


def test_the_real_corrected_serving_command_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    # The VERBATIM command line of the corrected re-measurement server, so the guard is proven
    # against real bytes rather than a hand-written lookalike.
    real = (
        "/home/node/.local/lib/ollama/llama-server --model /blobs/sha256-fb66ad5750680c77 "
        "--port 18080 --host 127.0.0.1 --no-webui -c 32768 -np 1 -ngl 999 --jinja "
        "--no-context-shift --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on -b 512 "
        "-ub 512"
    )
    _stub_ps(monkeypatch, real)
    _stub_record(monkeypatch)
    resolved = serving_guard.assert_serving_safe(
        "http://127.0.0.1:18080/v1", max_tokens=6000, prompt_budget=26000
    )
    assert resolved is not None
    assert (resolved.n_ctx, resolved.context_shift, resolved.n_keep) == (32768, False, None)


def test_preflight_passes_a_correctly_served_local_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # End to end through the real seam: a good local server must reach the health probe.
    import litellm

    monkeypatch.setattr(infer, "_cheapest_enabled_model", lambda: "local8b")
    monkeypatch.setattr(
        infer,
        "litellm_model_target",
        lambda _m: ("openai/local8b", {"api_base": "http://127.0.0.1:18080/v1"}),
    )
    monkeypatch.setattr(infer, "_effective_max_tokens", lambda _m: 6000)
    monkeypatch.setattr(serving_guard, "serving_prompt_budget", lambda: 26000)
    _stub_ps(monkeypatch, SAFE_CMD)
    _stub_record(monkeypatch)
    probed: list[str] = []
    monkeypatch.setattr(litellm, "completion", lambda **_kw: probed.append("ran"))
    assert infer.preflight_api_check() is True
    assert probed == ["ran"]  # the guard let it through rather than refusing


def test_hosted_preflight_never_touches_the_serving_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    # The no-op must be structural: a hosted target must not even resolve a serving config.
    import litellm

    monkeypatch.setattr(infer, "_cheapest_enabled_model", lambda: "hosted")
    monkeypatch.setattr(infer, "litellm_model_target", lambda _m: ("deepseek/hosted", {}))

    def _boom(*_a: object, **_kw: object) -> object:
        raise AssertionError("hosted providers must never resolve a serving config")

    monkeypatch.setattr(serving_guard, "resolve_serving_config", _boom)
    monkeypatch.setattr(litellm, "completion", lambda **_kw: object())
    assert infer.preflight_api_check() is True


def test_preflight_fails_closed_on_an_undecidable_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    # An endpoint the guard cannot classify must abort the run, not escape as an uncaught
    # exception and not slip through the hosted no-op.
    import litellm

    monkeypatch.setattr(infer, "_cheapest_enabled_model", lambda: "odd")
    monkeypatch.setattr(
        infer,
        "litellm_model_target",
        lambda _m: ("openai/odd", {"api_base": "http://100.64.0.1:8080/v1"}),
    )
    monkeypatch.setattr(litellm, "completion", lambda **_kw: object())
    with pytest.raises(infer.ApiUnusableError, match="unsafe local serving config"):
        infer.preflight_api_check()
