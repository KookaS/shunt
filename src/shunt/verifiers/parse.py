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

# Randomized per-run seeds a runner prints into its own output. sympy's `bin/test` header emits
#     random seed:        63896174
#     hash randomization: on (PYTHONHASHSEED=3309299595)
# on EVERY invocation, and `_failing_check_id` hashes the output wholesale — so replaying one
# identical sympy state four times produced four different keys (hash:14dee27f…, 956e761d…,
# cdc44945…, ac61a97d…) for the same failure. The dedup key is the escalation trigger's only
# recurrence signal, so a per-run-random key does not merely add noise: it disables escalation
# for that slice while still looking like a populated field. `randomly-seed` covers
# pytest-randomly's banner, the same class of line from a different runner.
_SEED_RE = re.compile(r"(?i)\b(random[ _-]?seed|randomly-seed|PYTHONHASHSEED)\s*[:=]\s*\d+")

# tox prints its own wall-clock accounting as `OK (12.34=setup[0.01]+cmd[12.33] seconds)`; the
# bracket separates the number from the unit, so the duration pass below cannot see those three
# floats. sphinx's `tox --current-env` test_cmd puts that line in every run's output, next to a
# `pid=54` that changes on every invocation.
_TOX_TIMING_RE = re.compile(r"\(\d+(?:\.\d+)?=setup\[[\d.]+\](?:\+\w+\[[\d.]+\])* seconds\)")

# pytest's `--durations` block is the worst of the class, because normalizing the NUMBERS is not
# enough: the block is sorted BY time, so the run-to-run jitter permutes the LINES. sphinx's
# test_cmd runs `pytest --durations 25`, and three replays of one identical state produced three
# different keys purely from that reordering. The entries are pure timing telemetry — they list
# passing tests too and carry no failure identity — so the whole line goes, before the duration
# pass below can rewrite the number and hide the shape.
_DURATION_LINE_RE = re.compile(
    r"^\s*\d+(?:\.\d+)?m?s\s+(?:call|setup|teardown)\s+\S+[^\n]*\n?", re.MULTILINE
)

_NORMALIZERS: Final[list[tuple[re.Pattern[str], str]]] = [
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),  # hex addresses / pointers
    (_TS_RE, "TS"),  # ISO-8601 timestamps
    (re.compile(r"(?:/tmp|/var/folders|/private/var/folders)/\S+"), "TMPPATH"),  # temp dirs
    (re.compile(r"\bpytest-of-\S+"), "TMPPATH"),
    (_SEED_RE, r"\1=SEED"),  # randomized run seeds (sympy bin/test, pytest-randomly)
    (_DURATION_LINE_RE, ""),  # pytest --durations entries (values AND their order vary)
    (_TOX_TIMING_RE, "(DUR)"),  # tox's bracketed setup/cmd accounting
    (re.compile(r"\bpid=\d+"), "pid=PID"),  # tox's per-invocation subprocess pid
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


# Collection/build/link-phase phrases a runner emits when it failed to collect/build a target.
# They are NOT safe to trust unconditionally: an assertion can render any of them verbatim
# (`assert "cannot find module" in out`), which used to reclassify a genuine red as infra.
# They apply only when the run did not also signal that tests ran and failed (see below).
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

# Exit codes that mean the test SESSION never validly ran, so no capability verdict exists to
# read: pytest's ExitCode 2 INTERRUPTED, 3 INTERNAL_ERROR, 4 USAGE_ERROR, 5 NO_TESTS_COLLECTED.
# Only 0 (passed) and 1 (tests ran, some failed) carry an outcome. Gating on the whole set — not
# on 2 alone — is what stops a bad selector (`ERROR: not found: /testbed/a.py::test_x`, exit 4)
# being stamped a capability red: with only the exit-2 guard, 3/4/5 fell through to the failure
# branch and were classified on output WORDING, so two runners failing identically diverged
# (one printed "ImportError while importing", the other an underscore-wrapped "____ ERROR
# collecting ____" the `^ERROR collecting` anchor misses) — a coin flip, not a classification.
_NON_SESSION_EXITS: Final = frozenset({2, 3, 4, 5})

# A shell reports a signal death as 128+N (SIGKILL/OOM 137, SIGSEGV 139, SIGTERM 143); a raw
# `subprocess` returncode reports it as -N. Either way the process was killed mid-run, so its
# output is a truncated fragment and never a verdict about the code under test.
_SIGNAL_EXIT_FLOOR: Final = 128

# Lines a runner prints ONLY after tests actually executed and at least one failed: pytest's
# short-summary `FAILED <nodeid>` and its `N failed` tail, go's `--- FAIL:`, rust's `failures:`
# block, jest's `Tests: … N failed`. A build/collection error never reaches any of these — it
# aborts before a test runs — so their presence is positive proof the run produced a real red.
_RAN_AND_FAILED_RE: Final = re.compile(
    r"^FAILED\s+\S+::"  # pytest short summary
    r"|^\s*\d+ failed\b"  # pytest -q tail
    r"|^--- FAIL:"  # go
    r"|^failures:$"  # rust
    r"|^Tests:.*\b\d+ failed",  # jest
    re.MULTILINE,
)


def _is_non_session_exit(returncode: int) -> bool:
    """True when the exit code itself proves no valid test session produced a verdict."""
    return returncode in _NON_SESSION_EXITS or returncode < 0 or returncode >= _SIGNAL_EXIT_FLOOR


def _ran_and_failed(combined: str, returncode: int) -> bool:
    """True when the output proves tests executed and at least one genuinely failed."""
    # Gated on the exit code: a usage/collection/interrupt/signal exit is never a genuine red no
    # matter what its output quotes, because the session that would have produced the red never
    # validly ran. A truncated SIGKILL fragment can still contain a `FAILED …` line from a test
    # that ran before the kill; the run as a whole is still not a verdict.
    if _is_non_session_exit(returncode):
        return False
    return _RAN_AND_FAILED_RE.search(combined) is not None


def _is_environment_failure(combined: str, returncode: int) -> bool:
    """True when the output is an environment/collection error, not a real capability red."""
    # The exit code decides first and needs no wording: 2/3/4/5 and every signal death are
    # environmental by definition. This is what makes the classification wording-independent —
    # previously only exit 2 was covered, so a usage error (4) was read off prose it happened to
    # print, and identical failures landed on opposite verdicts.
    if _is_non_session_exit(returncode):
        return True
    # Below here the exit code is 0 or 1. Proof that tests ran and failed wins over every marker:
    # the collection/build phrases are unanchored and an assertion can render any of them verbatim,
    # which silently reclassified genuine reds (`assert "cannot find module" in out`, exit 1).
    if _ran_and_failed(combined, returncode):
        return False
    return _COLLECTION_MARKER_RE.search(combined) is not None


# Lines a runner prints when it exited CLEANLY having selected/executed nothing: sympy's
# `tests finished: 0 passed`, pytest's `collected 0 items` / `no tests ran`, unittest's
# `Ran 0 tests`, go's `no test files`. This is deliberately EVIDENCE OF ABSENCE, not evidence of
# presence: it is not inverted into "prove a test ran", because the green tail of a runner family
# these patterns do not know would then be demoted from a real pass — regressing the live tier2
# verifier on every stack we have not enumerated. Absence-only can only ever demote a run that
# SAID it ran nothing.
#
# `\b0 passed\b` cannot match "10 passed"/"460 passed": there is no word boundary mid-number.
_NO_TESTS_RE: Final = re.compile(
    r"\b0 passed\b"
    r"|collected 0 items"
    r"|\bno tests ran\b"
    r"|^Ran 0 tests\b"
    r"|\bno test files\b",
    re.IGNORECASE | re.MULTILINE,
)

# The markers above are unanchored, and a test suite that runs a runner INSIDE a test prints that
# inner session's output verbatim inside its own. pytest's own suite does exactly this via
# `pytester`: `testing/logging/test_fixture.py` at the gold patch reports `16 passed in 0.16s` at
# exit 0, while a captured inner session in the same text says `collected 0 items` /
# `no tests ran in 0.00s`. The absence marker then demoted a real, complete pass to "no
# measurement" — and with the admissibility gate now CLEARING what it rejects, that
# misclassification stops being inert and starts destroying measured data.
#
# So the demotion is vetoed by an explicit non-zero pass count. This is not the inversion the
# comment above rejects: absence-only still demotes every unenumerated runner family, and nothing
# has to prove a test ran. Only a runner that counted at least one passing test can override its
# own absence marker. sympy's `tests finished: 0 passed` is untouched (zero is not `[1-9]…`).
_RAN_AND_PASSED_RE: Final = re.compile(
    r"\b[1-9]\d* passed\b|^Ran [1-9]\d* tests\b", re.IGNORECASE | re.MULTILINE
)


def _selected_nothing(combined: str) -> bool:
    """True when the runner said it executed nothing AND never counted a passing test."""
    return _NO_TESTS_RE.search(combined) is not None and _RAN_AND_PASSED_RE.search(combined) is None


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
        if _selected_nothing(combined):
            # Exit 0 with nothing selected is not a verified pass — it is the ABSENCE of a
            # measurement. A selector that matches no test (sympy's `bin/test` given bare
            # unittest names) exits 0 printing `0 passed`, and stamping that green fabricates a
            # green step history for a run in which no test ever executed. Infra, like any other
            # state we could not measure: not labellable, never counts toward escalation.
            return VerifierResult(
                outcome="unknown",
                confidence=0.0,
                detail="no tests were selected or executed (rc=0)",
                exit_code=0,
                is_infra_failure=True,
            )
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
