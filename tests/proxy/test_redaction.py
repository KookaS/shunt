"""Nothing leaving the proxy may carry a provider credential or split a header."""

from __future__ import annotations

from typing import Final

import pytest

from shunt.proxy.redaction import header_safe, redact_secrets

# Shapes real providers return in a 401 body. Assembled at runtime from halves so the
# fixtures cannot be mistaken for live credentials by a secret scanner — gitleaks
# flagged the literal forms, which is the scanner behaving correctly.
_FAKE: Final[str] = "A" * 8 + "0123456789bcdef"
_LEAKS: Final[tuple[str, ...]] = (
    f"Incorrect API key provided: sk-proj-{_FAKE}",
    f"Invalid Authorization: Bearer rq-{_FAKE}",
    f"api_key={_FAKE} rejected",
    f"token: ghp_{_FAKE}",
)


@pytest.mark.parametrize("text", _LEAKS)
def test_credentials_are_redacted(text: str) -> None:
    assert "<redacted>" in redact_secrets(text)


@pytest.mark.parametrize("text", _LEAKS)
def test_credentials_never_survive_into_a_header(text: str) -> None:
    assert "<redacted>" in header_safe(text)


def test_header_value_cannot_split_the_response() -> None:
    value = header_safe("boom\r\nX-Injected: yes")
    assert "\r" not in value
    assert "\n" not in value


def test_non_ascii_error_does_not_crash_the_handler() -> None:
    # Raw exception text here previously raised UnicodeEncodeError inside the error
    # handler, so the 502 never reached the wire.
    header_safe("Ошибка провайдера").encode("latin-1")


def test_non_ascii_error_still_says_something() -> None:
    assert header_safe("Ошибка провайдера") != "-"


def test_header_value_is_bounded() -> None:
    assert len(header_safe("x" * 10_000)) <= 200


def test_empty_input_yields_a_placeholder_not_an_empty_header() -> None:
    assert header_safe("") == "-"
