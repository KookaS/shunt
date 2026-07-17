"""Hermetic tests for the keyless auth-probe, driven against the mock stub."""

# Every test here is offline. The live probe is the opt-in provider-probe.yml
# workflow — a network call in this suite would make CI depend on eleven third
# parties' uptime, and the measured status codes are perishable by design.
#
# The parametrized tests read the signatures table (tools/provider_auth_
# signatures.yaml) and the examples catalog, the same two leaves the probe reads,
# so every measured provider is covered the moment its signature exists — no list
# to update. Targets are built through the probe module's own load_signatures().

from __future__ import annotations

import socket
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Final
from urllib.parse import urlsplit

import pytest
import yaml

from shunt.models.config import AuthProbe
from tests.mock_openai_server import MockOpenAIServer, MockSignature, signature_for

_ROOT: Final = Path(__file__).resolve().parents[1]
_EXAMPLES_DIR: Final = _ROOT / "examples" / "providers"
_SIGNATURES_PATH: Final = _ROOT / "tools" / "provider_auth_signatures.yaml"

# Read directly at collection time (parametrize can't use a fixture). Same single-
# owner file the probe's load_signatures() reads — not a second source of truth.
_SIGNATURES: Final = {
    name: AuthProbe.model_validate(block)
    for name, block in yaml.safe_load(_SIGNATURES_PATH.read_text()).items()
}
_PROBED: Final = sorted(_SIGNATURES)
_WITH_BODY_PATTERN: Final = sorted(
    n for n, p in _SIGNATURES.items() if p.expect_body_pattern is not None
)

_TIMEOUT: Final = 5.0


def _example_base_url(name: str) -> str:
    """The provider's base_url from its examples-catalog fragment, the single owner."""
    providers = yaml.safe_load((_EXAMPLES_DIR / f"{name}.yaml").read_text())["providers"]
    return str(providers[name]["base_url"])


def _example_provider_row(name: str) -> dict:
    """The provider's full row from its catalog fragment (base_url + api_key_env_var)."""
    return dict(yaml.safe_load((_EXAMPLES_DIR / f"{name}.yaml").read_text())["providers"][name])


def _target(name: str, pp: ModuleType) -> object:
    """A ProbeTarget for a measured provider: catalog base_url + signature, via the probe module."""
    row = _example_provider_row(name)
    return pp.ProbeTarget(
        name=name,
        base_url=row["base_url"],
        api_key_env_var=row["api_key_env_var"],
        probe=pp.load_signatures()[name],
    )


def _at(server: MockOpenAIServer, target: object) -> object:
    """The target pointed at the local stub, KEEPING its base_url path prefix."""
    # Preserving the path is load-bearing. Groq (/openai/v1), OpenRouter
    # (/api/v1) and Fireworks (/inference/v1) serve the OpenAI API under a
    # prefix; a stub at a bare host would let a probe that silently drops that
    # prefix pass every offline test, which is exactly what it did until a live
    # run caught it.
    return replace(target, base_url=server.base_url + urlsplit(target.base_url).path)  # type: ignore[type-var]


def _served_path(target: object) -> str:
    """Where an OpenAI client aimed at this base_url would send the probe request."""
    # Derived from the client contract (base_url + the API call's suffix), NOT
    # from probe_url — so the assertion is an independent oracle, not an echo.
    probe: AuthProbe | None = target.probe  # type: ignore[attr-defined]
    assert probe is not None
    suffix = probe.endpoint.rsplit("/v1/", 1)[-1]
    return urlsplit(target.base_url).path.rstrip("/") + "/" + suffix  # type: ignore[attr-defined]


def _replay(target: object) -> MockSignature:
    """The target's declared signature, served where its client would ask for it."""
    return replace(signature_for(target.probe), endpoint=_served_path(target))  # type: ignore[attr-defined]


def _closed_port() -> int:
    """Bind and immediately release a port, so connecting to it is refused."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = int(s.getsockname()[1])
    return port


# --- the catalog's own rows, replayed faithfully ------------------------------


@pytest.mark.parametrize("name", _PROBED)
def test_declared_signature_is_accepted(name, provider_probe, mock_openai_server) -> None:
    target = _target(name, provider_probe)
    server = mock_openai_server(_replay(target))

    result = provider_probe.probe_provider(_at(server, target), timeout=_TIMEOUT)

    assert result.outcome is provider_probe.Outcome.PASS
    assert not result.failed


@pytest.mark.parametrize("name", _PROBED)
def test_probe_sends_the_bogus_key_to_the_declared_endpoint(
    name, provider_probe, mock_openai_server
) -> None:
    target = _target(name, provider_probe)
    server = mock_openai_server(_replay(target))

    provider_probe.probe_provider(_at(server, target), timeout=_TIMEOUT)

    assert len(server.received) == 1
    sent = server.received[0]
    assert sent.path == _served_path(target)
    # The whole design rests on never holding a real credential. If this ever
    # reads a live key, the probe has stopped being free and safe to run in CI.
    assert sent.authorization == f"Bearer {provider_probe.BOGUS_API_KEY}"
    # Measured: Groq/Together/Cerebras sit behind a Cloudflare front that answers
    # 403 to `Python-urllib/*` BEFORE checking auth — indistinguishable from
    # Requesty's real 403 signature. Letting urllib's default agent back in
    # would fail three correctly configured providers every nightly run.
    assert sent.user_agent == provider_probe.USER_AGENT
    assert "urllib" not in (sent.user_agent or "")


@pytest.mark.parametrize("name", _PROBED)
def test_probe_urls_target_the_routed_path(name, provider_probe) -> None:
    target = _target(name, provider_probe)

    url = provider_probe.probe_url(target)

    # Two ways to get this wrong, both silent: concatenating yields the doubled
    # /v1/v1/... (404), and urljoin'ing the leading slash drops a base_url path
    # prefix (probes the wrong host root, and 403s). The probe must hit exactly
    # what AsyncOpenAI(base_url=...) hits — nothing else is worth asserting.
    assert url == f"https://{urlsplit(target.base_url).netloc}{_served_path(target)}"
    assert "/v1/v1/" not in url


# --- the failure modes --------------------------------------------------------


@pytest.mark.parametrize("name", _PROBED)
def test_wrong_base_url_is_reported_not_passed(name, provider_probe, mock_openai_server) -> None:
    target = _target(name, provider_probe)
    if 404 in _SIGNATURES[name].expect_status:
        pytest.skip(
            "404 is this provider's declared auth signature; covered by the 404 hazard test"
        )
    # A host that answers, but not at the path we think it does — exactly what a
    # stale or mistyped base_url looks like from the outside.
    server = mock_openai_server(replace(_replay(target), endpoint="/somewhere/else"))

    result = provider_probe.probe_provider(_at(server, target), timeout=_TIMEOUT)

    assert result.outcome is provider_probe.Outcome.FAIL_STATUS
    assert result.status == 404
    assert result.failed


@pytest.mark.parametrize("name", _PROBED)
def test_unexpected_status_fails(name, provider_probe, mock_openai_server) -> None:
    target = _target(name, provider_probe)
    server = mock_openai_server(replace(_replay(target), status=500))

    result = provider_probe.probe_provider(_at(server, target), timeout=_TIMEOUT)

    assert result.outcome is provider_probe.Outcome.FAIL_STATUS
    assert result.status == 500
    assert "expected" in result.detail


@pytest.mark.parametrize("name", _WITH_BODY_PATTERN)
def test_body_pattern_mismatch_fails(name, provider_probe, mock_openai_server) -> None:
    target = _target(name, provider_probe)
    # Right status, wrong message: the provider changed its wording, or we are
    # talking to something that is not the provider at all.
    server = mock_openai_server(
        replace(_replay(target), body='{"error": {"message": "unrelated failure"}}')
    )

    result = provider_probe.probe_provider(_at(server, target), timeout=_TIMEOUT)

    assert result.outcome is provider_probe.Outcome.FAIL_BODY
    assert result.failed


def test_unreachable_host_is_distinct_from_a_wrong_status(provider_probe) -> None:
    target = provider_probe.ProbeTarget(
        name="closed",
        base_url=f"http://127.0.0.1:{_closed_port()}/v1",
        api_key_env_var="UNUSED_IN_KEYLESS",
        probe=AuthProbe(measured_as_of="2026-07-17"),
    )

    result = provider_probe.probe_provider(target, timeout=_TIMEOUT)

    assert result.outcome is provider_probe.Outcome.FAIL_UNREACHABLE
    assert result.status is None
    assert "unreachable" in result.detail
    assert result.failed


# --- the Fireworks hazard: a 404 that means two different things --------------


def _hazard_target(pp: ModuleType, base_url: str) -> object:
    """A target whose auth rejection IS a 404 — the measured Fireworks shape."""
    return pp.ProbeTarget(
        name="hazard",
        base_url=base_url,
        api_key_env_var="UNUSED_IN_KEYLESS",
        probe=AuthProbe(
            endpoint="/v1/chat/completions",
            expect_status=[404],
            expect_body_pattern="model inaccessible",
            measured_as_of="2026-07-17",
        ),
    )


def test_404_auth_signature_passes_only_on_the_right_body(
    provider_probe, mock_openai_server
) -> None:
    server = mock_openai_server(
        MockSignature(
            endpoint="/v1/chat/completions",
            status=404,
            body='{"error": {"message": "Model not found, model inaccessible"}}',
        )
    )

    result = provider_probe.probe_provider(_hazard_target(provider_probe, server.base_url))

    assert result.outcome is provider_probe.Outcome.PASS


def test_404_from_a_wrong_base_url_is_not_mistaken_for_the_404_auth_signature(
    provider_probe, mock_openai_server
) -> None:
    # THE test this whole design exists for. The status matches the declared
    # signature (404 == 404), so status checking alone would report PASS on a
    # completely wrong base_url. Only the body pattern tells the two apart.
    server = mock_openai_server(MockSignature(endpoint="/elsewhere", status=404, body="unused"))

    result = provider_probe.probe_provider(_hazard_target(provider_probe, server.base_url))

    assert result.outcome is provider_probe.Outcome.FAIL_BODY
    assert result.status == 404


# --- contract details ---------------------------------------------------------


def test_probe_marked_provider_without_a_signature_fails(provider_probe) -> None:
    # Only `# shunt-ci: probe` files become targets (local is `skip`, filtered
    # upstream), so a target with no signature is a real misconfiguration — it
    # must fail loudly, not skip, or a dropped signature would ship green.
    target = provider_probe.ProbeTarget(
        name="unmeasured",
        base_url="https://api.example.com/v1",
        api_key_env_var="UNUSED_IN_KEYLESS",
        probe=None,
    )

    result = provider_probe.probe_provider(target)

    assert result.outcome is provider_probe.Outcome.FAIL_NO_SIGNATURE
    assert result.failed


def test_relative_endpoint_is_rejected(provider_probe) -> None:
    target = provider_probe.ProbeTarget(
        name="x",
        base_url="https://api.example.com/v1",
        api_key_env_var="UNUSED_IN_KEYLESS",
        probe=AuthProbe(endpoint="chat/completions"),
    )

    with pytest.raises(ValueError, match="host-absolute"):
        provider_probe.probe_url(target)


@pytest.mark.parametrize(
    ("base_url", "endpoint", "expected"),
    [
        # A plain /v1 root: the endpoint's version segment is redundant, not a
        # second path element.
        (
            "https://api.deepseek.com/v1",
            "/v1/chat/completions",
            "https://api.deepseek.com/v1/chat/completions",
        ),
        ("https://api.x.ai/v1", "/v1/models", "https://api.x.ai/v1/models"),
        # The prefix providers a host-absolute join silently truncates. These
        # three literals are what the live run proved correct.
        (
            "https://api.groq.com/openai/v1",
            "/v1/chat/completions",
            "https://api.groq.com/openai/v1/chat/completions",
        ),
        (
            "https://openrouter.ai/api/v1",
            "/v1/chat/completions",
            "https://openrouter.ai/api/v1/chat/completions",
        ),
        (
            "https://api.fireworks.ai/inference/v1",
            "/v1/models",
            "https://api.fireworks.ai/inference/v1/models",
        ),
        # A trailing slash must not double up.
        ("https://api.example.com/v1/", "/v1/models", "https://api.example.com/v1/models"),
    ],
)
def test_probe_url_joins_base_url_and_endpoint(
    base_url: str, endpoint: str, expected: str, provider_probe: ModuleType
) -> None:
    target = provider_probe.ProbeTarget(
        name="x",
        base_url=base_url,
        api_key_env_var="UNUSED_IN_KEYLESS",
        probe=AuthProbe(endpoint=endpoint),
    )

    assert provider_probe.probe_url(target) == expected


def test_probe_targets_reports_every_row(provider_probe, mock_openai_server) -> None:
    good = _target(_PROBED[0], provider_probe)
    server = mock_openai_server(_replay(good))
    targets = [
        replace(_at(server, good), name="good"),
        provider_probe.ProbeTarget(
            name="unmeasured",
            base_url="https://api.example.com/v1",
            api_key_env_var="UNUSED_IN_KEYLESS",
            probe=None,
        ),
    ]

    results = provider_probe.probe_targets(targets, timeout=_TIMEOUT)

    assert [r.provider for r in results] == ["good", "unmeasured"]
    assert [r.outcome for r in results] == [
        provider_probe.Outcome.PASS,
        provider_probe.Outcome.FAIL_NO_SIGNATURE,
    ]


def test_unknown_provider_selection_is_an_error(provider_probe) -> None:
    targets = provider_probe.load_probe_targets()
    with pytest.raises(SystemExit, match="unknown provider"):
        provider_probe._select(targets, "not-a-provider")


# --- the live probe must stay opt-in, structurally ----------------------------


def test_live_probe_workflow_has_no_automatic_trigger() -> None:
    # The measured status codes are perishable. If this workflow ever gains a
    # push/pull_request trigger, the day a provider changes its rejection code
    # every unrelated PR goes red. A comment cannot enforce that; this can.
    workflow = yaml.safe_load((_ROOT / ".github/workflows/provider-probe.yml").read_text())
    # PyYAML 1.1 parses a bare `on:` key as the boolean True, not the string.
    assert set(workflow[True]) == {"workflow_dispatch", "schedule"}


def test_ci_does_not_reference_the_live_probe() -> None:
    ci = (_ROOT / ".github/workflows/ci.yml").read_text()

    assert "provider-probe" not in ci
    assert "provider_probe" not in ci
    assert "provider-auth-check" not in ci


# --- authenticated (200) probe: proves a REAL key is ACCEPTED ------------------


def _auth_target(
    provider_probe: ModuleType,
    server: MockOpenAIServer,
    *,
    env_var: str,
    positive_endpoint: str | None = "/v1/models",
) -> object:
    """A target pointed at the stub, for the authenticated real-key path."""
    probe = AuthProbe(positive_endpoint=positive_endpoint)
    return provider_probe.ProbeTarget(
        name="x", base_url=server.base_url, api_key_env_var=env_var, probe=probe
    )


def test_authenticated_passes_when_the_real_key_is_accepted(
    provider_probe: ModuleType, mock_openai_server, monkeypatch
) -> None:
    server = mock_openai_server(MockSignature(endpoint="/v1/models", status=200, body="{}"))
    monkeypatch.setenv("PROBE_TEST_KEY", "a-real-looking-key")
    target = _auth_target(provider_probe, server, env_var="PROBE_TEST_KEY")

    result = provider_probe.probe_provider_authenticated(target, timeout=_TIMEOUT)

    assert result.outcome is provider_probe.Outcome.PASS
    assert result.status == 200
    # Always a free GET — never a billed completion — carrying the real key.
    assert server.received[-1].method == "GET"
    assert server.received[-1].authorization == "Bearer a-real-looking-key"


def test_authenticated_never_sends_a_post_body(
    provider_probe: ModuleType, mock_openai_server, monkeypatch
) -> None:
    # The whole no-charge guarantee: the authenticated check must be a GET, so a
    # provider can never bill it. This pins that it never POSTs a completion.
    server = mock_openai_server(MockSignature(endpoint="/v1/models", status=200, body="{}"))
    monkeypatch.setenv("PROBE_TEST_KEY", "a-real-looking-key")
    target = _auth_target(provider_probe, server, env_var="PROBE_TEST_KEY")

    provider_probe.probe_provider_authenticated(target, timeout=_TIMEOUT)

    assert all(r.method == "GET" for r in server.received)


def test_authenticated_honours_a_custom_positive_endpoint(
    provider_probe: ModuleType, mock_openai_server, monkeypatch
) -> None:
    # OpenRouter's /v1/models is public, so it checks /v1/auth/key instead — still
    # a free GET, just a different path.
    server = mock_openai_server(MockSignature(endpoint="/v1/auth/key", status=200, body="{}"))
    monkeypatch.setenv("PROBE_TEST_KEY", "a-real-looking-key")
    target = _auth_target(
        provider_probe, server, env_var="PROBE_TEST_KEY", positive_endpoint="/v1/auth/key"
    )

    result = provider_probe.probe_provider_authenticated(target, timeout=_TIMEOUT)

    assert result.outcome is provider_probe.Outcome.PASS
    assert server.received[-1].path == "/v1/auth/key"


def test_authenticated_skips_when_no_key_is_set(provider_probe: ModuleType, monkeypatch) -> None:
    monkeypatch.delenv("PROBE_ABSENT_KEY", raising=False)
    target = provider_probe.ProbeTarget(
        name="x",
        base_url="https://api.example.com/v1",
        api_key_env_var="PROBE_ABSENT_KEY",
        probe=AuthProbe(),
    )

    result = provider_probe.probe_provider_authenticated(target)

    # Secret-optional: absent key is a skip, not a failure — coverage grows as
    # keys are added, and a missing secret never reddens the run.
    assert result.outcome is provider_probe.Outcome.SKIP_NO_KEY
    assert result.skipped
    assert not result.failed


def test_authenticated_skips_a_provider_with_no_free_endpoint(
    provider_probe: ModuleType, monkeypatch
) -> None:
    # Requesty: positive_endpoint is None (its /v1/models is public, no free auth
    # endpoint). With a key SET, it must still SKIP — never fall through to a
    # billed completion — so the no-charge guarantee holds even when armed.
    monkeypatch.setenv("PROBE_TEST_KEY", "a-real-looking-key")
    target = provider_probe.ProbeTarget(
        name="requesty-like",
        base_url="https://api.example.com/v1",
        api_key_env_var="PROBE_TEST_KEY",
        probe=AuthProbe(positive_endpoint=None),
    )

    result = provider_probe.probe_provider_authenticated(target, timeout=_TIMEOUT)

    assert result.outcome is provider_probe.Outcome.SKIP_NO_POSITIVE_CHECK
    assert result.skipped
    assert not result.failed


def test_authenticated_fails_when_the_real_key_is_rejected(
    provider_probe: ModuleType, mock_openai_server, monkeypatch
) -> None:
    server = mock_openai_server(MockSignature(endpoint="/v1/models", status=401, body="{}"))
    monkeypatch.setenv("PROBE_TEST_KEY", "a-real-looking-key")
    target = _auth_target(provider_probe, server, env_var="PROBE_TEST_KEY")

    result = provider_probe.probe_provider_authenticated(target, timeout=_TIMEOUT)

    assert result.outcome is provider_probe.Outcome.FAIL_STATUS
    assert result.status == 401
    assert result.failed


@pytest.mark.parametrize("status", [429, 500, 503])
def test_authenticated_treats_a_transient_status_as_a_warning(
    provider_probe: ModuleType, mock_openai_server, monkeypatch, status: int
) -> None:
    # A rate-limit (xAI's 1/min) or a provider 5xx is not a key verdict — it must
    # WARN, not redden the weekly cron, or the job goes red for a blip nobody can fix.
    server = mock_openai_server(MockSignature(endpoint="/v1/models", status=status, body="{}"))
    monkeypatch.setenv("PROBE_TEST_KEY", "a-real-looking-key")
    target = _auth_target(provider_probe, server, env_var="PROBE_TEST_KEY")

    result = provider_probe.probe_provider_authenticated(target, timeout=_TIMEOUT)

    assert result.outcome is provider_probe.Outcome.SKIP_TRANSIENT
    assert result.skipped
    assert not result.failed


def test_authenticated_fails_loudly_on_a_probe_marked_provider_without_signature(
    provider_probe: ModuleType, monkeypatch
) -> None:
    # probe=None with a key set is a real misconfiguration (probe-marked, unsigned).
    # It must FAIL with the true cause, not be confused with the requesty no-free-
    # endpoint skip.
    monkeypatch.setenv("PROBE_TEST_KEY", "a-real-looking-key")
    target = provider_probe.ProbeTarget(
        name="x",
        base_url="https://api.example.com/v1",
        api_key_env_var="PROBE_TEST_KEY",
        probe=None,
    )

    result = provider_probe.probe_provider_authenticated(target)

    assert result.outcome is provider_probe.Outcome.FAIL_NO_SIGNATURE
    assert result.failed


def test_authenticated_workflow_has_no_automatic_trigger() -> None:
    workflow = yaml.safe_load((_ROOT / ".github/workflows/provider-auth-check.yml").read_text())
    # PyYAML 1.1 parses a bare `on:` key as boolean True. Secrets-gated, so it
    # must never run on fork PRs — workflow_dispatch/schedule only.
    assert set(workflow[True]) == {"workflow_dispatch", "schedule"}
