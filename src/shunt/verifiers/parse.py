"""Pure test-output → VerifierResult parser, shared by the live tier2 verifier and offline
replay so both derive a byte-identical failing-check id / dedup key from the same output.
"""

# The classification body used to live welded inside ``AutoDetectVerifier.verify`` after its
# subprocess returned. It is extracted here as a pure function of (combined output, returncode)
# so an offline test execution in a container yields the SAME dedup key production would — the
# recurrence signal cannot drift because there is only one definition of it.
#
# The classifier is ONE multi-stage rule (docs/escalation.md "Which runners are supported"): the
# exit-code taxonomy decides session-never-ran vs genuine-red first, then positive proof that
# tests ran and failed, then the collection / nothing-selected fallbacks. Every runner family in
# the Framework matrix is served by the same vocabularies below — never a per-framework parser.
# Two machine-readable channels (JUnit XML and TAP) are read when present and precede the regex
# fallback: pytest --junitxml, Surefire TEST-*.xml, Gradle test-results, PHPUnit --log-junit and
# gtest --gtest_output=xml emit one, and so do node:test / bats / prove / tasty TAP.

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
    # JUnit XML timings (pytest --junitxml / Surefire / PHPUnit --log-junit emit `time="0.123"`).
    (re.compile(r'\btime="[0-9.eE+-]+"'), 'time="DUR"'),
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
# They apply only when the run did not also signal that tests ran and failed (see below), and
# they are gated on the exit code exactly like every other marker here.
_COLLECTION_MARKER_RE: Final = re.compile(
    r"^ERROR collecting "  # pytest
    r"|error(?:s)? during collection"  # pytest
    r"|ImportError while importing"  # pytest
    r"|conftest\.py.*(?:ImportError|ModuleNotFoundError)"  # pytest
    r"|Failed to import test module"  # unittest loader
    r"|cannot find module "  # jest / go build error
    r"|no required module provides package"  # go
    r"|can't find crate for"  # rust
    r"|unresolved import"  # rust compile error
    r"|error\[E\d+\]"  # rust / cargo compile failure (error[E0308])
    r"|COMPILATION ERROR"  # maven build failure
    r"|cannot find symbol"  # javac
    r"|\.java:\d+:\s*error:"  # javac file:line: error
    r"|> Task :compile\w+ FAILED"  # gradle compile step
    r"|error (?:CS|MSB)\d+"  # dotnet csc / msbuild
    r"|^Build FAILED\.$"  # msbuild
    r"|cannot load such file"  # ruby require
    r"|^PHP (?:Fatal|Parse) error:"  # php pre-run fatal
    r"|fatal error:"  # gcc / clang
    r"|undefined reference to"  # linker
    r"|could not find module"  # swift build
    r"|cannot find '[^']*' in scope"  # swift build
    r"|Can't locate \S+ in @INC"  # perl load error
    r"|there is no package called"  # R package load error
    r"|^\*\* \(CompileError\)"  # mix test compile failure
    r"|No tests were executed"  # surefire found nothing to run
    r"|No tests to run"  # surefire
    r"|No tests were found!!!"  # ctest
    r"|No tests found"  # jest / gradle
    r"|Bail out!",  # TAP abort (further tests useless)
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

# Exit 2 is shared by two families with opposite meanings: pytest's INTERRUPTED (a non-session
# abort) and PHPUnit's errored-tests (a genuine red — its ShellExitCodeCalculator maps a test
# ERROR to EXCEPTION_EXIT=2). PHPUnit's 2 is a red only when the output proves a test errored
# (its summary markers); a pytest interruption never prints them. Verified against PHPUnit
# 11.x's ShellExitCodeCalculator: 0 success, 1 failure, 2 exception.
_PHPUNIT_ERRORED_RE: Final = re.compile(
    r"^ERRORED!$|^ERRORS!$|Errors:\s*[1-9]\d*|There (?:was|were) \d+ error", re.MULTILINE
)

# Lines a runner prints ONLY after tests actually executed and at least one failed. A
# build/collection error never reaches any of these — it aborts before a test runs — so their
# presence is positive proof the run produced a real red. Count-summary forms carry the failing
# count in the pattern ([1-9]\d*) so a "0 failures" green tail can never match. Every family in
# the Framework matrix is covered; a red whose dialect this set does not know still lands on the
# exit-code branch, so the shape stays safe until a vocabulary catches up.
_RAN_AND_FAILED_RE: Final = re.compile(
    r"^FAILED\s+\S+::"  # pytest short summary
    r"|^\s*\d+ failed\b"  # pytest -q tail
    r"|^FAILED\s+\(failures="  # unittest / nose2
    r"|^Tests:.*\b\d+ failed"  # jest
    r"|^Test Suites:.*\b\d+ failed"  # jest
    r"|^FAIL\s+\S+"  # jest / vitest failing-suite line
    r"|^\s*Test Files\s+\d+ failed"  # vitest
    r"|^\s*Tests\s+\d+ failed"  # vitest
    r"|^\s*\d+ failing\b"  # mocha
    r"|^\s*\d+ specs?, \s*[1-9]\d* failures?"  # jasmine
    r"|\b\d+ tests? failed\b"  # ava
    r"|^--- FAIL:"  # go
    r"|^FAIL$"  # go bare tail
    r"|^FAIL\s"  # go package line
    r"|panic: test timed out"  # go
    r"|^test result: FAILED"  # rust / libtest
    r"|^failures:$"  # rust
    r"|^\s*test \S+ \.\.\. FAILED"  # rust
    r"|^Tests run:.*Failures: [1-9]\d*"  # surefire / failsafe
    r"|^Tests run:.*Errors: [1-9]\d*"  # surefire
    r"|^\[ERROR\]\s+Failures?:\s*[1-9]\d*"  # surefire
    r"|^\[ERROR\]\s+Errors?:\s*[1-9]\d*"  # surefire
    r"|<<< FAILURE!"  # surefire per-test marker
    r"|<<< ERROR!"  # surefire per-test error marker
    r"|\b\d+ tests? completed, [1-9]\d* failed\b"  # gradle
    r"|^Failed!"  # dotnet vstest summary
    r"|^Test Run Failed\.$"  # dotnet
    r"|Total tests: \d+.*\bFailed: [1-9]\d*"  # dotnet legacy console logger
    r"|\bFailed: [1-9]\d*, Passed: \d+"  # dotnet default console logger
    r"|\b\d+ runs, \d+ assertions, [1-9]\d* failures"  # minitest / test-unit
    r"|\b\d+ runs, \d+ assertions, \d+ failures, [1-9]\d* errors"  # minitest
    r"|\b\d+ tests, \d+ assertions, [1-9]\d* failures"  # test-unit
    r"|^\s*\d+ examples?, [1-9]\d* failures?"  # rspec / hspec
    r"|^Failed examples:$"  # rspec
    r"|\b\d+ scenarios? \(\s*[1-9]\d* failed"  # cucumber / behat
    r"|^Failed scenarios?:$"  # cucumber / behat
    r"|^FAILED!$"  # phpunit v11
    r"|^FAILURES!$"  # phpunit pre-v11
    r"|^ERRORED!$"  # phpunit v11
    r"|^ERRORS!$"  # phpunit pre-v11
    r"|^Tests: \d+, Assertions: \d+, Failures: [1-9]\d*"  # phpunit summary
    r"|^Tests: \d+, Assertions: \d+, Errors: [1-9]\d*"  # phpunit summary
    r"|There (?:was|were) \d+ failure"  # phpunit
    r"|^\s*⨯\s"  # pest failing-test glyph
    r"|^\[\s+FAILED\s+\]"  # google test
    r"|^\s*\d+ FAILED TEST"  # google test tail
    r"|^\s*\d+% tests passed, [1-9]\d* tests? failed"  # ctest
    r"|^The following tests FAILED:$"  # ctest
    r"|test cases: \d+ \| \d+ passed \| [1-9]\d* failed"  # catch2
    r"|assertions: \d+ \| \d+ passed \| [1-9]\d* failed"  # catch2
    r"|\d+/\d+ test cases FAILED!"  # doctest
    r"|^TEST CASE FAILED!"  # doctest
    r"|^\*\*\* [1-9]\d* failures?"  # boost.test
    r"|^Test Case '[^']*' failed \(\d"  # swift xctest
    r"|^Executed \d+ tests?, with [1-9]\d* failure"  # swift xctest
    r"|^Test Suite '[^']*' failed"  # swift xctest
    r"|FAILED\s*\(failures=[1-9]\d*"  # shunit2
    r"|^Result: FAIL"  # perl prove
    r"|Failed \d+/\d+ subtests"  # perl prove
    r"|^Failed:\s*[1-9]\d*\s*$"  # testthat
    r"|^── Failure \("  # testthat
    r"|^── Error \("  # testthat
    r"|^Status:\s*[1-9]\d*\s+ERROR"  # R CMD check
    r"|^\s*\d+ tests?, [1-9]\d* failures?"  # elixir exunit
    r"|^\s*\*\* \("  # elixir in-test exception
    r"|^Failed tests:$"  # haskell tasty
    r"|\bFailures:\s*[1-9]\d*",  # haskell HUnit
    re.MULTILINE,
)

# Failure markers a runner emits at EXIT 0, because its shell code is unreliable (karma exits 0
# even when tests fail — the matrix class is "marker-gated, never exit-gated"). Consulted only
# in the exit-0 branch, where a passing run would otherwise swallow the red.
_RAN_AND_FAILED_AT_ZERO_RE: Final = re.compile(r"Executed \d+ of \d+ \([1-9]\d* FAILED\)")


def _is_non_session_exit(returncode: int, combined: str = "") -> bool:
    """True when the exit code itself proves no valid test session produced a verdict."""
    if returncode in _NON_SESSION_EXITS:
        # Exit 2 is ambiguous across families: pytest INTERRUPTED vs PHPUnit errored-tests.
        # PHPUnit's 2 is a genuine red only when its errored-test markers are present; a pytest
        # interruption never prints them.
        return not (returncode == 2 and _PHPUNIT_ERRORED_RE.search(combined))
    return returncode < 0 or returncode >= _SIGNAL_EXIT_FLOOR


def _ran_and_failed(combined: str, returncode: int, channel: tuple[int, int] | None = None) -> bool:
    """True when the output proves tests executed and at least one genuinely failed."""
    # Gated on the exit code: a usage/collection/interrupt/signal exit is never a genuine red no
    # matter what its output quotes, because the session that would have produced the red never
    # validly ran. A truncated SIGKILL fragment can still contain a `FAILED …` line from a test
    # that ran before the kill; the run as a whole is still not a verdict.
    if _is_non_session_exit(returncode, combined):
        return False
    if channel is not None and channel[1] > 0:
        return True
    return _RAN_AND_FAILED_RE.search(combined) is not None


def _is_environment_failure(
    combined: str, returncode: int, channel: tuple[int, int] | None = None
) -> bool:
    """True when the output is an environment/collection error, not a real capability red."""
    # The exit code decides first and needs no wording: 2/3/4/5 and every signal death are
    # environmental by definition. This is what makes the classification wording-independent —
    # previously only exit 2 was covered, so a usage error (4) was read off prose it happened to
    # print, and identical failures landed on opposite verdicts.
    if _is_non_session_exit(returncode, combined):
        return True
    # Below here the exit code is 0 or 1. Proof that tests ran and failed wins over every marker:
    # the collection/build phrases are unanchored and an assertion can render any of them verbatim,
    # which silently reclassified genuine reds (`assert "cannot find module" in out`, exit 1).
    if _ran_and_failed(combined, returncode, channel):
        return False
    return _COLLECTION_MARKER_RE.search(combined) is not None


# Lines a runner prints when it exited CLEANLY having selected/executed nothing: sympy's
# `tests finished: 0 passed`, pytest's `collected 0 items` / `no tests ran`, unittest's
# `Ran 0 tests`, go's `no test files`, a TAP `1..0` plan, a JUnit `<testsuite tests="0">`.
# This is deliberately EVIDENCE OF ABSENCE, not evidence of presence: it is not inverted into
# "prove a test ran", because the green tail of a runner family these patterns do not know would
# then be demoted from a real pass — regressing the live tier2 verifier on every stack we have
# not enumerated. Absence-only can only ever demote a run that SAID it ran nothing.
#
# `\b0 passed\b` cannot match "10 passed"/"460 passed": there is no word boundary mid-number.
_NO_TESTS_RE: Final = re.compile(
    r"\b0 passed\b"
    r"|collected 0 items"
    r"|\bno tests ran\b"
    r"|^Ran 0 tests\b"
    r"|\bno test files\b"
    r"|running 0 tests"  # rust / libtest
    r"|\b0 tests, 0 failures\b"  # elixir / exunit
    r"|\b0 runs, 0 assertions\b"  # ruby minitest / test-unit
    r"|OK \(\s*0 tests"  # phpunit
    r"|No tests found"  # jest / gradle
    r"|No tests were found!!!"
    r"|No tests to run"  # surefire
    r"|No tests were executed",  # surefire
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
    r"\b[1-9]\d* passed\b"  # pytest / sympy
    r"|^Ran [1-9]\d* tests\b"  # unittest
    r"|^OK \(\s*[1-9]\d* tests?"  # phpunit
    r"|^\[\s+PASSED\s+\]\s*[1-9]\d* tests?\.?"  # google test
    r"|All tests passed"  # catch2
    r"|All tests successful\."  # perl prove
    r"|^Test Run Successful\.$"  # dotnet vstest
    r"|^Passed!"  # dotnet vstest
    r"|^Status: OK$"  # R CMD check
    r"|^[1-9]\d* tests?, 0 failures?",  # elixir / exunit
    re.IGNORECASE | re.MULTILINE,
)


def _selected_nothing(combined: str, channel: tuple[int, int] | None = None) -> bool:
    """True when the runner said it executed nothing AND never counted a passing test."""
    # The channel branch applies the SAME veto as the regex branch below: a structured
    # channel reading (0,0) is still overruled by an explicit non-zero pass count in the
    # text (e.g. pytest's `4 passed` tail next to a `1..0` TAP plan or a `<testsuite
    # tests="0">` wrapper). Without it, a genuinely passing run that also prints an empty
    # plan is demoted to "no measurement" and the escalation signal stops accruing.
    if channel is not None:
        return channel[0] == 0 and _RAN_AND_PASSED_RE.search(combined) is None
    return _NO_TESTS_RE.search(combined) is not None and _RAN_AND_PASSED_RE.search(combined) is None


# --- Machine-readable channels: JUnit XML and TAP. --------------------------
# When a runner's output carries one of these, the counts are read from the structured channel
# directly and the regex fallback is only reached when neither channel is present. Both are
# cross-cutting: pytest --junitxml, Surefire TEST-*.xml, Gradle test-results, PHPUnit
# --log-junit and gtest --gtest_output=xml emit JUnit XML; node:test, bats, prove and tasty
# emit TAP.

_JUNIT_TESTSUITE_TAG_RE: Final = re.compile(r"<\s*testsuites?\b[^>]*>", re.IGNORECASE)
_JUNIT_ATTR_RE: Final = re.compile(r'\b(\w+)\s*=\s*"(\d+)"')
_JUNIT_BAD_CHILD_RE: Final = re.compile(r"<\s*(?:failure|error)\b", re.IGNORECASE)


def _int_attr(tag: str, name: str) -> int | None:
    """The value of a decimal XML attribute in *tag*, or None when absent."""
    for attr, value in _JUNIT_ATTR_RE.findall(tag):
        if attr == name:
            return int(value)
    return None


def _junit_counts(combined: str) -> tuple[int, int] | None:
    """(total, failed) summed across EVERY JUnit <testsuite> tag with a `tests` attribute.

    Reading only the first suite hid a real failing run behind a leading zero-test
    wrapper; a suite lacking failure/error attributes counts its children within its span.
    """
    matches = list(_JUNIT_TESTSUITE_TAG_RE.finditer(combined))
    if not matches:
        return None
    # A Surefire-style <testsuites> ROOT carries the aggregate over its child suites
    # (`tests`/`failures`/`errors` on the wrapper itself). Summing the children as well
    # would double-count the wrapper, so a plural root that carries its own `tests` is
    # authoritative; children are summed only when the root has no counts of its own.
    root = matches[0].group(0)
    if re.match(r"<\s*testsuites\b", root, re.IGNORECASE):
        root_total = _int_attr(root, "tests")
        if root_total is not None:
            root_failed = _int_attr(root, "failures")
            root_errors = _int_attr(root, "errors")
            return (root_total, (root_failed or 0) + (root_errors or 0))
    total = 0
    failed = 0
    seen_tests = False
    for i, match in enumerate(matches):
        tag = match.group(0)
        suite_total = _int_attr(tag, "tests")
        if suite_total is None:
            continue
        seen_tests = True
        total += suite_total
        suite_failed = _int_attr(tag, "failures")
        suite_errors = _int_attr(tag, "errors")
        if suite_failed is None and suite_errors is None:
            span_end = matches[i + 1].start() if i + 1 < len(matches) else len(combined)
            failed += len(_JUNIT_BAD_CHILD_RE.findall(combined[match.end() : span_end]))
        else:
            failed += (suite_failed or 0) + (suite_errors or 0)
    if not seen_tests:
        return None
    return (total, failed)


_TAP_PLAN_RE: Final = re.compile(r"^\s*1\.\.(\d+)\b", re.MULTILINE)
_TAP_VERSION_RE: Final = re.compile(r"^\s*TAP version \d+", re.MULTILINE)
_TAP_OK_RE: Final = re.compile(r"^\s*ok\b", re.MULTILINE)
_TAP_NOT_OK_RE: Final = re.compile(r"^\s*not ok\b", re.MULTILINE)
_TAP_NOT_OK_REAL_RE: Final = re.compile(r"^\s*not ok\b(?!.*#\s*TODO)", re.MULTILINE)


def _tap_counts(combined: str) -> tuple[int, int] | None:
    """(planned, failed) from a TAP stream, or None when the output is not TAP.

    The `1..N` plan (or `TAP version` header) is the presence marker; a planless stream's total
    is the observed ok+not-ok lines, and a `not ok … # TODO` point is expected, not a failure.
    """
    plan = _TAP_PLAN_RE.search(combined)
    if plan is None and _TAP_VERSION_RE.search(combined) is None:
        return None
    if plan is not None:
        total = int(plan.group(1))
    else:
        total = len(_TAP_OK_RE.findall(combined)) + len(_TAP_NOT_OK_RE.findall(combined))
    return (total, len(_TAP_NOT_OK_REAL_RE.findall(combined)))


def _structured_channel(combined: str) -> tuple[int, int] | None:
    """(total, failed) from a JUnit-XML or TAP channel when one is present, else None."""
    junit = _junit_counts(combined)
    if junit is not None:
        return junit
    return _tap_counts(combined)


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
    channel = _structured_channel(combined)
    if _is_non_session_exit(returncode, combined):
        # An environmental red (missing module, broken collection, interrupted session) is not a
        # capability outcome — unknown + is_infra_failure, so no verified label is written and it
        # never counts toward escalation.
        return VerifierResult(
            outcome="unknown",
            confidence=0.0,
            detail=f"environment/collection error (rc={returncode})",
            exit_code=returncode,
            is_infra_failure=True,
        )
    if channel is not None and channel[1] > 0:
        # A machine-readable channel counting failing tests is positive proof a test ran and
        # failed — also at exit 0, for the runners (karma, some TAP harnesses) whose shell code
        # does not say so. A genuine red is never demoted to infra.
        return VerifierResult(
            outcome="failure",
            confidence=0.7,
            detail=f"tests failed (rc={returncode})",
            exit_code=returncode,
            failing_check_id=_failing_check_id(combined),
        )
    if returncode == 0:
        if _selected_nothing(combined, channel):
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
        if _RAN_AND_FAILED_AT_ZERO_RE.search(combined):
            # karma exits 0 even when tests fail; its summary line is the only proof.
            return VerifierResult(
                outcome="failure",
                confidence=0.7,
                detail="tests failed (rc=0)",
                exit_code=0,
                failing_check_id=_failing_check_id(combined),
            )
        return VerifierResult(outcome="success", confidence=0.8, detail="tests passed", exit_code=0)
    if _is_environment_failure(combined, returncode, channel):
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
