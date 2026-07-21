#!/usr/bin/env python3
"""Provider auth-probe: a bogus key is rejected (keyless), a real key accepted (--authenticated)."""

# WHAT THIS PROVES — and what it does not.
#
# KEYLESS (default): it sends a DELIBERATELY INVALID bearer token to a provider's
# real endpoint and asserts the rejection matches that provider's measured
# signature. A pass means: the host resolves, the base_url + endpoint path is
# right, and the auth layer answers the way we measured it. It costs nothing —
# auth fails before billing — and needs no secret, so CI can run it on a fork PR.
# It does NOT prove a real key works; it is a wiring check, not a smoke test.
#
# AUTHENTICATED (--authenticated): the complement. It reads the REAL key from each
# provider's api_key_env_var and asserts the provider ACCEPTS it (200) — proving
# the credential works. It is ALWAYS FREE: a GET of the provider's positive_endpoint
# (a model listing, or a key-info endpoint like OpenRouter's /v1/auth/key), NEVER a
# billed completion. Secret-optional: a provider with no key set is SKIPped with a
# warning, not failed. A provider with no free authenticated endpoint (Requesty,
# whose /v1/models is public) is also skipped rather than billed. It needs secrets,
# so it runs only in the secrets-gated provider-auth-check.yml, never on fork PRs.
#
# WHERE ITS DATA COMES FROM (two single-owner leaves, merged by provider name):
#   - CONNECTION facts (base_url) come from the examples catalog,
#     examples/providers/<name>.yaml — the files a user copies. Only files whose
#     line-1 marker is `# shunt-ci: probe` are probed (`local` is `skip`).
#   - SIGNATURES (endpoint/expect_status/expect_body_pattern) come from
#     tools/provider_auth_signatures.yaml, measured 2026-07-17 for 11 providers.
# The runtime registry (src/shunt/config/models.yaml) is NEVER read here.
#
# WHY THE SIGNATURE IS PER-PROVIDER DATA: there is no universal shape. Most answer
# 401; Requesty answers 403; xAI answers 400 carrying "Incorrect API key provided"
# (it validates the model id before auth); Fireworks answers 404 "model
# inaccessible" on chat/completions even with NO auth header, and only its
# /v1/models gives a clean 401. That 404 is indistinguishable BY STATUS from a
# wrong base_url — only the body differs. Hence endpoint + expect_status +
# expect_body_pattern all live on the signature, and this file stays a dumb
# executor with no provider knowledge of its own.

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from shunt.models.config import AuthProbe, parse_registry, strict_yaml_load

# The two data leaves, resolved relative to this script (tools/).
_HERE: Final = Path(__file__).resolve().parent
DEFAULT_SIGNATURES_PATH: Final = _HERE / "provider_auth_signatures.yaml"
DEFAULT_EXAMPLES_DIR: Final = _HERE.parent / "examples" / "providers"

# Line-1 marker opting a catalog file into (or out of) the live probe.
_MARKER_PREFIX: Final = "# shunt-ci:"

# Deliberately invalid, and self-describing: if it ever shows up in a provider's
# logs, the operator can tell at a glance it was a wiring check, not an attack.
BOGUS_API_KEY = "shunt-probe-deliberately-invalid-key"

# MEASURED, load-bearing: urllib's default `Python-urllib/3.12` agent is bot-
# blocked by the Cloudflare front on Groq, Together and Cerebras, which answer
# 403 BEFORE auth. That looks exactly like Requesty's genuine 403 auth rejection,
# so the default agent would make the probe report a false failure on correctly
# configured providers — and a self-identifying agent is the honest thing to send
# to someone else's API anyway.
USER_AGENT = "shunt-provider-probe/1.0 (+https://github.com/KookaS/shunt)"

DEFAULT_TIMEOUT_S = 15.0

# A base_url's trailing API version (/v1, /openai/v1, /inference/v1) and the same
# segment at the head of a host-absolute endpoint (/v1/models).
_VERSION_SUFFIX_RE: Final = re.compile(r"/v\d+$")
_VERSION_PREFIX_RE: Final = re.compile(r"^/v\d+/")


class Outcome(StrEnum):
    """The four distinguishable results of probing one provider."""

    PASS = "PASS"
    FAIL_STATUS = "FAIL_STATUS"
    FAIL_BODY = "FAIL_BODY"
    FAIL_UNREACHABLE = "FAIL_UNREACHABLE"
    FAIL_NO_SIGNATURE = "FAIL_NO_SIGNATURE"
    SKIP_NO_KEY = "SKIP_NO_KEY"
    SKIP_NO_POSITIVE_CHECK = "SKIP_NO_POSITIVE_CHECK"
    SKIP_TRANSIENT = "SKIP_TRANSIENT"


@dataclass(frozen=True)
class ProbeTarget:
    """One provider to probe: connection facts (catalog) merged with its signature (validation)."""

    name: str
    base_url: str
    api_key_env_var: str
    probe: AuthProbe | None


@dataclass(frozen=True)
class ProbeResult:
    """One provider's probe outcome, with the evidence that produced it."""

    provider: str
    outcome: Outcome
    status: int | None
    detail: str

    @property
    def failed(self) -> bool:
        """True for outcomes that must make the run exit non-zero."""
        return self.outcome.startswith("FAIL")

    @property
    def skipped(self) -> bool:
        """True for outcomes that are neither pass nor fail (nothing to check)."""
        return self.outcome.startswith("SKIP")


def load_signatures(path: str | Path | None = None) -> dict[str, AuthProbe]:
    """Load + validate the measured auth-rejection signatures, keyed by provider name."""
    path_obj = Path(path) if path is not None else DEFAULT_SIGNATURES_PATH
    data = strict_yaml_load(path_obj.read_text())
    return {name: AuthProbe.model_validate(block) for name, block in data.items()}


def _marker(text: str) -> str | None:
    """The `# shunt-ci:` value on line 1 (`probe`/`skip`), or None if no marker."""
    first = text.splitlines()[0].strip() if text else ""
    if first.startswith(_MARKER_PREFIX):
        return first[len(_MARKER_PREFIX) :].strip()
    return None


def load_probe_targets(
    examples_dir: str | Path | None = None,
    signatures: dict[str, AuthProbe] | None = None,
) -> list[ProbeTarget]:
    """Merge the `# shunt-ci: probe` catalog files with their signatures into probe targets."""
    dir_obj = Path(examples_dir) if examples_dir is not None else DEFAULT_EXAMPLES_DIR
    sigs = signatures if signatures is not None else load_signatures()

    targets: list[ProbeTarget] = []
    for path in sorted(dir_obj.glob("*.yaml")):
        text = path.read_text()
        if _marker(text) != "probe":
            continue
        # parse_registry validates the fragment (schema + provider FK), so a
        # leaked auth_probe or a malformed row fails loudly here, not silently.
        registry = parse_registry(strict_yaml_load(text))
        for name, provider in registry.providers.items():
            targets.append(
                ProbeTarget(
                    name=name,
                    base_url=provider.base_url,
                    api_key_env_var=provider.api_key_env_var,
                    probe=sigs.get(name),
                )
            )
    return targets


def _join_endpoint(base_url: str, endpoint: str) -> str:
    """Join base_url with a host-absolute endpoint, dropping the duplicated version segment."""
    if not endpoint.startswith("/"):
        raise ValueError(f"endpoint must be host-absolute (start with '/'), got {endpoint!r}")
    root = base_url.rstrip("/")
    path = endpoint
    if _VERSION_SUFFIX_RE.search(root):
        path = _VERSION_PREFIX_RE.sub("/", path, count=1)
    return root + path


def _send(url: str, *, api_key: str, body: bytes | None, timeout: float) -> tuple[int, str]:
    """Send one request with the given key; return (status, body) even for error statuses.

    Raises URLError/OSError/TimeoutError to the caller for the host-never-answered case.
    """
    request = urllib.request.Request(  # noqa: S310 - catalog base_urls are https config, not user input
        url,
        data=body,
        method="POST" if body is not None else "GET",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return int(response.status), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", "replace")


def probe_url(target: ProbeTarget) -> str:
    """Resolve the URL an OpenAI client aimed at this provider's base_url would request."""
    probe = target.probe
    if probe is None:
        raise ValueError("target has no signature")
    # base_url IS the API root, and it already ends in the version segment — but
    # NOT always as bare `/v1`: Groq serves at /openai/v1, OpenRouter at /api/v1,
    # Fireworks at /inference/v1. The endpoint is written host-absolutely
    # (/v1/chat/completions) because that is how provider docs read, so its
    # version segment is redundant with base_url's and must be dropped, not
    # duplicated (/v1/v1/...) and not urljoin'd (which would honour the leading
    # slash and silently discard /openai, /api and /inference).
    #
    # The target is defined by the router's real behaviour: AsyncOpenAI(base_url)
    # requests base_url + "/chat/completions". Probing anything else would test a
    # URL production never calls.
    return _join_endpoint(target.base_url, probe.endpoint)


def _request_body(url: str) -> bytes | None:
    """The POST payload for a chat/completions probe; None makes it a GET."""
    if not url.endswith("/chat/completions"):
        return None
    # A minimal, well-formed request: we want the auth layer to answer, not a
    # schema validator. The model id is bogus on purpose — a provider that
    # checks the model first (xAI, Fireworks) is handled by its body pattern.
    return json.dumps(
        {
            "model": "shunt-probe-nonexistent-model",
            "messages": [{"role": "user", "content": "probe"}],
            "max_tokens": 1,
        }
    ).encode()


def _fetch(url: str, timeout: float) -> tuple[int, str]:
    """Send the bogus-key request; return (status, body) even for error statuses."""
    return _send(url, api_key=BOGUS_API_KEY, body=_request_body(url), timeout=timeout)


def probe_provider(target: ProbeTarget, *, timeout: float = DEFAULT_TIMEOUT_S) -> ProbeResult:
    """Probe one provider and classify the response against its declared signature."""
    probe = target.probe
    if probe is None:
        # A target only reaches here if its catalog file is `# shunt-ci: probe`
        # (local, the only unmeasured provider, is `skip` and filtered upstream).
        # So a probe-marked provider with no signature is a misconfiguration —
        # fail loudly rather than skip, or a dropped signature ships green.
        return ProbeResult(
            target.name,
            Outcome.FAIL_NO_SIGNATURE,
            None,
            f"{target.name} is # shunt-ci: probe but has no signature — "
            f"add it to provider_auth_signatures.yaml or mark the file skip",
        )

    url = probe_url(target)
    try:
        status, body = _fetch(url, timeout)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        # Distinct from a wrong status: the host never answered at all. Bad DNS
        # and a closed port both land here, and neither says anything about auth.
        return ProbeResult(target.name, Outcome.FAIL_UNREACHABLE, None, f"{url} unreachable: {exc}")

    if status not in probe.expect_status:
        expected = ", ".join(str(s) for s in probe.expect_status)
        return ProbeResult(
            target.name,
            Outcome.FAIL_STATUS,
            status,
            f"{url} answered {status}, expected {expected} — "
            f"wrong base_url, a moved endpoint, or a changed rejection code",
        )

    if probe.expect_body_pattern is not None and not re.search(probe.expect_body_pattern, body):
        # The Fireworks/xAI case: the status alone is ambiguous, so a pass here
        # without the body check would greenlight a wrong base_url.
        return ProbeResult(
            target.name,
            Outcome.FAIL_BODY,
            status,
            f"{url} answered {status} but body did not match "
            f"{probe.expect_body_pattern!r}: {body[:200]!r}",
        )

    return ProbeResult(
        target.name, Outcome.PASS, status, f"{url} rejected the bogus key with {status}"
    )


def probe_provider_authenticated(
    target: ProbeTarget, *, timeout: float = DEFAULT_TIMEOUT_S
) -> ProbeResult:
    """With the REAL key from the provider's env var, assert the provider ACCEPTS it (200)."""
    # Zero cost: GETs the provider's positive_endpoint (a model listing or key-info
    # endpoint), never a billed completion. Secret-optional (unset key -> SKIP_NO_KEY),
    # and a provider with no free authenticated endpoint -> SKIP_NO_POSITIVE_CHECK.
    api_key = os.environ.get(target.api_key_env_var)
    if not api_key:
        return ProbeResult(
            target.name, Outcome.SKIP_NO_KEY, None, f"{target.api_key_env_var} not set"
        )
    if target.probe is None:
        # Probe-marked but unsigned: a real misconfiguration (the bijection test
        # blocks it upstream). Fail loudly with the true cause, not a positive-check
        # message — distinct from the requesty "no free endpoint" skip below.
        return ProbeResult(
            target.name,
            Outcome.FAIL_NO_SIGNATURE,
            None,
            f"{target.name} is # shunt-ci: probe but has no signature — "
            f"add it to provider_auth_signatures.yaml or mark the file skip",
        )
    endpoint = target.probe.positive_endpoint
    if endpoint is None:
        # Requesty: its /v1/models is a public catalog, so only a billed completion
        # would exercise the key. Skipping keeps the check strictly free — the
        # keyless rejection probe still covers this provider's wiring.
        return ProbeResult(
            target.name,
            Outcome.SKIP_NO_POSITIVE_CHECK,
            None,
            f"{target.name} has no free authenticated endpoint — a real completion "
            f"would be the only positive check, and it is not run to avoid billing",
        )
    url = _join_endpoint(target.base_url, endpoint)
    try:
        status, body = _send(url, api_key=api_key, body=None, timeout=timeout)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        # The keyless probe already proved this base_url reaches the provider, so
        # unreachable HERE is transient (provider down / network) — a warning, not
        # a key verdict. Don't redden a weekly cron for a blip the owner can't fix.
        return ProbeResult(
            target.name, Outcome.SKIP_TRANSIENT, None, f"{url} unreachable (transient): {exc}"
        )

    if status == 200:
        # 200 proves the key AUTHENTICATES. It does not prove the account can pay —
        # a model listing answers 200 even for an out-of-credits key.
        return ProbeResult(
            target.name, Outcome.PASS, status, f"{url} accepted the key (authenticates) with 200"
        )
    if status == 429 or status >= 500:
        # Rate-limited (e.g. xAI's 1/min) or a provider-side 5xx: the key itself may
        # be fine, so this is a warning, not a failure — the cron stays green.
        return ProbeResult(
            target.name,
            Outcome.SKIP_TRANSIENT,
            status,
            f"{url} answered {status} (rate-limit/provider error) — not a key verdict",
        )
    return ProbeResult(
        target.name,
        Outcome.FAIL_STATUS,
        status,
        f"{url} answered {status} with a real key — key rejected/expired, wrong base_url, "
        f"or a moved endpoint: {body[:200]!r}",
    )


def probe_targets(
    targets: list[ProbeTarget],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    authenticated: bool = False,
) -> list[ProbeResult]:
    """Probe every target in catalog order, keyless (rejection) or authenticated (acceptance)."""
    probe_one = probe_provider_authenticated if authenticated else probe_provider
    return [probe_one(t, timeout=timeout) for t in targets]


def _select(targets: list[ProbeTarget], only: str | None) -> list[ProbeTarget]:
    """Narrow the target list to a single named provider, or pass it through."""
    if only is None:
        return targets
    matches = [t for t in targets if t.name == only]
    if not matches:
        known = ", ".join(sorted(t.name for t in targets))
        raise SystemExit(f"unknown provider {only!r} (known: {known})")
    return matches


def main(argv: list[str] | None = None) -> int:
    """Probe the selected providers; return 1 if any failed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", help="probe only this provider (default: all)")
    parser.add_argument("--examples", help="catalog dir (default: the packaged examples/providers)")
    parser.add_argument("--signatures", help="signatures YAML (default: the packaged one)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--authenticated",
        action="store_true",
        help="use the REAL key from each provider's env var and assert 200 (needs secrets; "
        "providers with no key set are skipped). Default: keyless, assert rejection.",
    )
    args = parser.parse_args(argv)

    targets = load_probe_targets(args.examples, load_signatures(args.signatures))
    results = probe_targets(
        _select(targets, args.provider), timeout=args.timeout, authenticated=args.authenticated
    )

    in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    for r in results:
        print(f"{r.outcome:<22} {r.provider:<14} {r.detail}")
        # A skip is not a failure (job stays green), but it must be VISIBLE — a
        # provider silently never checked is the thing to avoid. Surface it as a
        # GitHub annotation in CI; a plain WARNING line otherwise.
        if r.skipped:
            if in_ci:
                print(f"::warning title=provider {r.provider} not checked::{r.detail}")
            else:
                print(f"  WARNING: {r.provider} not checked — {r.detail}")

    failed = [r for r in results if r.failed]
    skipped = [r for r in results if r.skipped]
    passed = len(results) - len(failed) - len(skipped)
    print(f"\n{passed} passed, {len(failed)} failed, {len(skipped)} skipped")
    if args.authenticated:
        print("Each PASS proves a REAL key AUTHENTICATES (200), for free — not that the account")
        print("can pay. A skip is a missing key, no free endpoint, or a transient/rate-limit")
        print("blip (not a key verdict); this check never bills.")
    else:
        print("This proves base_url + auth wiring only — NOT that a real key or completion works.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
