"""Pure test-output → VerifierResult parser, shared by the live tier2 verifier and offline
replay so both derive a byte-identical failing-check id / dedup key from the same output.
"""

# The classification body used to live welded inside ``AutoDetectVerifier.verify`` after its
# subprocess returned. It is extracted here as a pure function of (combined output, returncode)
# so an offline test execution in a container yields the SAME dedup key production would — the
# recurrence signal cannot drift because there is only one definition of it.

from __future__ import annotations

import hashlib
import re
from typing import Final

from .base import VerifierResult

# A pytest/jest node id in the combined output: "path::Test::case" or "path::case". The first
# such id is the failing check identity used as the escalation dedup key.
_NODE_ID_RE: Final = re.compile(r"^(?:FAILED\s+)?([\w./\\-]+::[\w:.\[\]\-]+)", re.MULTILINE)

# Volatile fragments that make the SAME recurring failure hash differently run-to-run. Stripped
# before the hash fallback so a go/rust red with only a different timing/address/temp-path
# hashes stably. Order matters: paths and timestamps before the bare-number/duration passes.
_TS_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
_NORMALIZERS: Final[list[tuple[re.Pattern[str], str]]] = [
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),  # hex addresses / pointers
    (_TS_RE, "TS"),  # ISO-8601 timestamps
    (re.compile(r"(?:/tmp|/var/folders|/private/var/folders)/\S+"), "TMPPATH"),  # temp dirs
    (re.compile(r"\bpytest-of-\S+"), "TMPPATH"),
    # Durations: "0.53s", "in 4.5s", "123ms", "1.2 seconds", "(0.00s)" (go/rust timing).
    (re.compile(r"\b\d+(?:\.\d+)?\s*(?:ns|µs|us|ms|s|secs?|seconds?)\b"), "DUR"),
]


def _normalize_detail(detail: str) -> str:
    """Strip run-to-run volatility (timings, hex addresses, temp paths, timestamps) from *detail*.

    Only the hash fallback uses this — a recurring go/rust failure that differs solely in a
    timing or address must hash to the SAME key, or recurrence never accumulates.
    """
    normalized = detail.strip()
    for pattern, repl in _NORMALIZERS:
        normalized = pattern.sub(repl, normalized)
    return normalized


# Collection/build/link-phase phrases that appear ONLY when the runner failed to
# collect/build a target, never inside a rendered assertion value — so they classify as
# environmental at any exit code. No bigger model fixes these; they must not escalate.
_COLLECTION_MARKER_RE: Final = re.compile(
    r"^ERROR collecting "
    r"|error(?:s)? during collection"
    r"|ImportError while importing"
    r"|conftest\.py.*(?:ImportError|ModuleNotFoundError)"
    r"|cannot find module "  # jest / go build error
    r"|no required module provides package"  # go
    r"|can't find crate for"  # rust
    r"|unresolved import",  # rust compile error
    re.IGNORECASE | re.MULTILINE,
)

# Python import phrases that a real collection error emits — but that can ALSO be quoted
# verbatim inside a failing assertion (`assert "No module named x" in ...`). These are gated
# on the runner's exit code: pytest returns 2 for a collection/usage error, 1 for a test
# failure, so an assert (exit 1) that merely mentions them stays a genuine red.
_AMBIGUOUS_IMPORT_RE: Final = re.compile(r"No module named|ModuleNotFoundError")

# Exit code a pytest run returns for a collection/usage error (vs 1 for a real test failure).
_PYTEST_COLLECTION_EXIT: Final = 2


def _is_environment_failure(combined: str, returncode: int) -> bool:
    """True when the output is an environment/collection error, not a real capability red."""
    # Unambiguous collection markers classify at any exit code; the ambiguous Python import
    # phrases only when the runner signalled a collection/usage error (pytest exit 2) — an
    # assertion (exit 1) whose message quotes an import string is a genuine failure.
    if _COLLECTION_MARKER_RE.search(combined):
        return True
    if returncode != _PYTEST_COLLECTION_EXIT:
        return False
    return _AMBIGUOUS_IMPORT_RE.search(combined) is not None


def _failing_check_id(detail: str) -> str:
    """Stable dedup key for a failure: the first test node id, else a hash of the detail."""
    # A node id (`path::case`) is ideal — a recurrence of the SAME failing test is the signal;
    # opaque output with no node id falls back to a hash of the NORMALIZED detail (timings, hex
    # addresses, temp paths, timestamps stripped) so a recurrence still hashes stably.
    match = _NODE_ID_RE.search(detail)
    if match:
        return match.group(1).strip()
    return "hash:" + hashlib.sha256(_normalize_detail(detail).encode()).hexdigest()[:16]


def parse_test_outcome(combined: str, returncode: int) -> VerifierResult:
    """Classify a test run's combined stdout+stderr + exit code into a VerifierResult.

    Pure (no subprocess): the ONE derivation of outcome / infra / failing_check_id, shared by
    the live tier2 verifier and the offline container replay so their dedup keys match exactly.
    """
    if returncode == 0:
        return VerifierResult(outcome="success", confidence=0.8, detail="tests passed", exit_code=0)
    if _is_environment_failure(combined, returncode):
        # An environmental red (missing module, broken collection) is not a capability outcome —
        # treat it like infra: unknown + is_infra_failure, so no verified label is written and it
        # never counts toward escalation.
        return VerifierResult(
            outcome="unknown",
            confidence=0.0,
            detail=f"environment/collection error (rc={returncode})",
            exit_code=returncode,
            is_infra_failure=True,
        )
    return VerifierResult(
        outcome="failure",
        confidence=0.7,
        detail=f"tests failed (rc={returncode})",
        exit_code=returncode,
        failing_check_id=_failing_check_id(combined),
    )


__all__ = ["parse_test_outcome"]
