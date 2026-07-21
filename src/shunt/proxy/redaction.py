"""Scrub provider secrets and unsafe bytes out of anything leaving the proxy."""

from __future__ import annotations

import os
import re
from typing import Final

# Provider 401 bodies routinely quote the key back ("Incorrect API key provided:
# sk-..."), and the proxy has no client auth — so upstream text reaching a client,
# a header, or a log is a key-disclosure path.
#
# Two passes, because a shape blacklist alone was never going to hold: it covered 3 of
# the 11 shipped providers, and every new provider silently added a leak. Pass 1 matches
# the actual live key VALUES the process loaded, which is exact and morphology-blind;
# pass 2 is the shape net for anything the environment did not name.
_SECRET_RE: Final[re.Pattern[str]] = re.compile(
    # Labelled form: `api key: X`, `token=X`, `Bearer X`. The separator class includes a
    # literal space — omitting it missed the most common human spelling.
    r"(?i)"
    # `api[_ -]?key` — the separator inside the PHRASE matters as much as the one after
    # it: "api key" (space) is the spelling provider errors use most.
    # The separator also tolerates quotes: a JSON body (`{"apiKey": "..."}`) and a
    # Python repr (`Config(api_key='...')`) both put a quote between the label and
    # the value, which broke the match and leaked the key verbatim.
    r"\b(?:api[_ -]?key|secret|token|bearer|authorization)"
    r"[\"']?[-_ :=]+[\"']?[A-Za-z0-9_\-]{12,}"
    # Self-identifying provider prefixes, which carry no label at all.
    r"|\b(?:sk|rk|pk|rq|gsk|csk|xai|nvapi|hf|fw|key|api|or|ds|mk|tgp)[-_]"
    r"[A-Za-z0-9_\-]{12,}"
    r"|\bAIza[A-Za-z0-9_\-]{10,}"
)

# Env vars whose VALUE is a credential. Suffix-matched so a new provider is covered the
# moment its key is configured, with no code change.
_SECRET_ENV_SUFFIXES: Final[tuple[str, ...]] = ("_API_KEY", "_TOKEN", "_SECRET", "_KEY")
# Below this length a "secret" is more likely to be a placeholder, and blanking every
# occurrence of a 3-char string would shred unrelated prose.
_MIN_SECRET_LEN: Final[int] = 8

# HTTP header values are latin-1 and may not contain CR/LF. Raw exception text put
# there crashed the handler on non-ASCII and allowed header splitting on newlines.
_HEADER_UNSAFE: Final[re.Pattern[str]] = re.compile(r"[^\x20-\x7e]")

HEADER_VALUE_LIMIT: Final[int] = 200


def _live_secrets() -> list[str]:
    """The credential values this process actually holds, longest first."""
    # Longest first so a key that contains another as a prefix is scrubbed whole.
    values = {
        value
        for name, value in os.environ.items()
        if name.upper().endswith(_SECRET_ENV_SUFFIXES) and len(value) >= _MIN_SECRET_LEN
    }
    return sorted(values, key=len, reverse=True)


def redact_secrets(text: str) -> str:
    """Replace any live credential, or anything shaped like one, with a marker."""
    for secret in _live_secrets():
        if secret in text:
            text = text.replace(secret, "<redacted>")
    return _SECRET_RE.sub("<redacted>", text)


def header_safe(value: str, *, limit: int = HEADER_VALUE_LIMIT) -> str:
    """Make *value* safe to send as an HTTP header: redacted, ASCII, single-line."""
    # backslashreplace rather than a blanket strip: a non-ASCII provider error would
    # otherwise reduce to nothing and tell an operator less than the escaped text does.
    ascii_only = redact_secrets(value).encode("ascii", "backslashreplace").decode("ascii")
    return _HEADER_UNSAFE.sub(" ", ascii_only).strip()[:limit] or "-"
