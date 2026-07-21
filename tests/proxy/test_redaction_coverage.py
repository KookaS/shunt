"""Redaction must cover the shapes the SHIPPED providers actually emit."""

# The original regex demanded an `sk|rk|pk|api[_-]?key|token|bearer` prefix, so a Groq
# `gsk_`, an xAI `xai-`, a Google `AIza` or the plain `api key: ` spelling (space, which
# the character class omitted) all reached the client body, the header and the logs.
# Fixtures are assembled at runtime so a secret scanner cannot mistake them for live keys.

from __future__ import annotations

from typing import Final

import pytest

from shunt.proxy.redaction import redact_secrets

_BODY = "A" * 8 + "0123456789bcdef"

# (label, text containing a credential) — every one must be scrubbed.
_LEAKY: Final[tuple[tuple[str, str], ...]] = (
    ("openai", f"Incorrect API key provided: sk-proj-{_BODY}."),
    ("groq", f"Invalid API Key: gsk_{_BODY}"),
    ("xai", f"auth failed for xai-{_BODY}"),
    ("anthropic", f"bad key sk-ant-api03-{_BODY}"),
    ("google", f"key AIza{_BODY} invalid"),
    ("space form", f"api key: {_BODY}"),
    ("underscore form", f"api_key={_BODY}"),
    ("bearer", f"Authorization: Bearer {_BODY}"),
    ("requesty", f"rejected credential rq-{_BODY}"),
)


@pytest.mark.parametrize(("label", "text"), _LEAKY, ids=[label for label, _ in _LEAKY])
def test_provider_credential_shapes_are_redacted(label: str, text: str) -> None:
    scrubbed = redact_secrets(text)
    assert _BODY not in scrubbed, f"{label}: credential body survived redaction"


def test_the_live_key_value_is_scrubbed_whatever_its_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The decisive guard: the process knows its own keys, so redaction should not have
    # to guess a morphology. A 12th provider with an unheard-of prefix stays covered.
    exotic = "zz" + "Q7w" * 6
    monkeypatch.setenv("WEIRDPROVIDER_API_KEY", exotic)
    assert exotic not in redact_secrets(f"upstream rejected {exotic} at 401")


def test_short_env_values_are_not_treated_as_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 2-char key would otherwise blank out unrelated substrings of every message.
    monkeypatch.setenv("TINY_API_KEY", "ab")
    assert redact_secrets("cannot grab the tab") == "cannot grab the tab"


def test_ordinary_prose_is_left_alone() -> None:
    message = "Upstream returned 503 for model qwen3.7-plus after 2 retries"
    assert redact_secrets(message) == message


# Assembled at runtime (prefix + _BODY), same as _LEAKY, so a secret scanner sees no
# complete key-shaped literal. Each still exercises a QUOTE between label and value —
# the case that used to break the match and leak the key verbatim.
_QUOTED_SEPARATORS: Final[tuple[str, ...]] = (
    # A JSON error body: the quote between label and value broke the match.
    '{"apiKey": "fw_' + _BODY + '"}',
    # A Python repr of a client config, which is what str(exc) often yields.
    "Config(api_key='sk-" + _BODY + "', base_url='x')",
    '{"api_key":"rq-' + _BODY + '"}',
    "authorization: Bearer eyJ" + _BODY,
)


@pytest.mark.parametrize("text", _QUOTED_SEPARATORS)
def test_quoted_separators_do_not_defeat_the_shape_net(text: str) -> None:
    # These are keys the process did NOT load from env (a rotated key quoted back
    # by an upstream, or another tenant's), so pass 1 cannot catch them — the
    # shape net is the only defence and it must survive quoting.
    assert "<redacted>" in redact_secrets(text)


@pytest.mark.parametrize(
    "text",
    [
        "the router picked a cheaper model today",
        "see docs/configuration.md for the api key setup",
    ],
)
def test_benign_prose_is_not_redacted(text: str) -> None:
    assert redact_secrets(text) == text
