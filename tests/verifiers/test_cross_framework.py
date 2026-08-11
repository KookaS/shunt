# Cross-framework runner coverage for the test-output classifier. Every newly supported
# runner family (docs/escalation.md "Which runners are supported") is a CAPTURED fixture
# here — pass, genuine failure, and (where a family has a distinct build/collection phase)
# an environment error — all classified by the ONE shared multi-stage rule
# `parse_test_outcome`, never a per-framework parser. JUnit-XML and TAP channels are parsed
# when present; the regex fallback only sees output with neither.
"""Cross-framework runner coverage for the test-output classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from shunt.verifiers.parse import _structured_channel, parse_test_outcome

_FIXTURES = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (_FIXTURES / name).read_text()


# (family, scenario, fixture file, exit code the runner really reports, expected outcome).
# Every "unknown" row must also be an is_infra_failure (never counts toward escalation).
_OUTCOME_MATRIX: Final[tuple[tuple[str, str, str, int, str], ...]] = (
    ("python-unittest", "pass", "unittest_pass.txt", 0, "success"),
    ("python-unittest", "failure", "unittest_fail.txt", 1, "failure"),
    ("python-unittest", "collection", "unittest_collection.txt", 1, "unknown"),
    ("python-tox", "pass", "tox_pass.txt", 0, "success"),
    ("python-tox", "failure", "tox_fail.txt", 1, "failure"),
    ("js-node:test", "pass", "node_test_pass.txt", 0, "success"),
    ("js-node:test", "failure", "node_test_fail.txt", 0, "failure"),  # TAP at exit 0
    ("js-vitest", "pass", "vitest_pass.txt", 0, "success"),
    ("js-vitest", "failure", "vitest_fail.txt", 1, "failure"),
    ("js-mocha", "pass", "mocha_pass.txt", 0, "success"),
    ("js-mocha", "failure", "mocha_fail.txt", 1, "failure"),
    ("js-jasmine", "pass", "jasmine_pass.txt", 0, "success"),
    ("js-jasmine", "failure", "jasmine_fail.txt", 1, "failure"),
    ("js-karma", "pass", "karma_pass.txt", 0, "success"),
    ("js-karma", "failure", "karma_fail.txt", 0, "failure"),  # exit 0 even on failure
    ("js-ava", "pass", "ava_pass.txt", 0, "success"),
    ("js-ava", "failure", "ava_fail.txt", 1, "failure"),
    ("rust-cargo", "pass", "cargo_pass.txt", 0, "success"),
    ("rust-cargo", "failure", "cargo_fail.txt", 101, "failure"),
    ("rust-cargo", "collection", "cargo_compile.txt", 101, "unknown"),
    ("java-surefire", "pass", "surefire_pass.txt", 0, "success"),
    ("java-surefire", "failure", "surefire_fail.txt", 1, "failure"),
    ("java-surefire", "collection", "surefire_compile.txt", 1, "unknown"),
    ("java-gradle", "pass", "gradle_pass.txt", 0, "success"),
    ("java-gradle", "failure", "gradle_fail.txt", 1, "failure"),
    ("java-gradle", "collection", "gradle_compile.txt", 1, "unknown"),
    ("dotnet", "pass", "dotnet_pass.txt", 0, "success"),
    ("dotnet", "failure", "dotnet_fail.txt", 1, "failure"),
    ("dotnet", "collection", "dotnet_build.txt", 1, "unknown"),
    ("ruby-minitest", "pass", "minitest_pass.txt", 0, "success"),
    ("ruby-minitest", "failure", "minitest_fail.txt", 1, "failure"),
    ("ruby-minitest", "collection", "minitest_collection.txt", 1, "unknown"),
    ("ruby-rspec", "pass", "rspec_pass.txt", 0, "success"),
    ("ruby-rspec", "failure", "rspec_fail.txt", 1, "failure"),
    ("ruby-rspec", "collection", "rspec_collection.txt", 1, "unknown"),
    ("php-phpunit", "pass", "phpunit_pass.txt", 0, "success"),
    ("php-phpunit", "failure", "phpunit_fail.txt", 1, "failure"),
    ("php-phpunit", "error", "phpunit_error.txt", 2, "failure"),
    ("php-phpunit", "collection", "phpunit_collection.txt", 255, "unknown"),
    ("cpp-gtest", "pass", "gtest_pass.txt", 0, "success"),
    ("cpp-gtest", "failure", "gtest_fail.txt", 1, "failure"),
    ("cpp-gtest", "collection", "gtest_collection.txt", 1, "unknown"),
    ("cpp-ctest", "pass", "ctest_pass.txt", 0, "success"),
    ("cpp-ctest", "failure", "ctest_fail.txt", 1, "failure"),
    ("cpp-ctest", "no-tests", "ctest_notests.txt", 0, "unknown"),
    ("swift-xctest", "pass", "xctest_pass.txt", 0, "success"),
    ("swift-xctest", "failure", "xctest_fail.txt", 1, "failure"),
    ("swift-xctest", "collection", "xctest_collection.txt", 1, "unknown"),
    ("shell-bats", "pass", "bats_pass.txt", 0, "success"),
    ("shell-bats", "failure", "bats_fail.txt", 1, "failure"),
    ("shell-shunit2", "pass", "shunit2_pass.txt", 0, "success"),
    ("shell-shunit2", "failure", "shunit2_fail.txt", 1, "failure"),
    ("perl-prove", "pass", "prove_pass.txt", 0, "success"),
    ("perl-prove", "failure", "prove_fail.txt", 1, "failure"),
    ("r-testthat", "pass", "testthat_pass.txt", 0, "success"),
    ("r-testthat", "failure", "testthat_fail.txt", 1, "failure"),
    ("r-r-cmd-check", "failure", "rcheck_fail.txt", 1, "failure"),
    ("elixir-exunit", "pass", "exunit_pass.txt", 0, "success"),
    ("elixir-exunit", "failure", "exunit_fail.txt", 1, "failure"),
    ("haskell-hspec", "pass", "hspec_pass.txt", 0, "success"),
    ("haskell-hspec", "failure", "hspec_fail.txt", 1, "failure"),
    ("haskell-tasty", "failure", "tasty_fail.txt", 1, "failure"),
    ("haskell-hunit", "failure", "hunit_fail.txt", 1, "failure"),
    ("junit-xml", "pytest-failure", "junit_pytest_fail.txt", 1, "failure"),
    ("junit-xml", "surefire-pass", "junit_surefire_pass.txt", 0, "success"),
    ("junit-xml", "empty", "junit_empty.txt", 0, "unknown"),
    ("junit-xml", "fail-at-exit-0", "junit_fail_at_zero.txt", 0, "failure"),
    ("junit-xml", "zero-test-wrapper-then-fail", "junit_wrapper_empty_then_fail.txt", 0, "failure"),
    ("junit-xml", "counted-wrapper", "junit_counted_wrapper_with_children.txt", 0, "failure"),
    ("junit-xml", "zero-wrapper-with-passed", "junit_zero_wrapper_with_passed.txt", 0, "success"),
    ("tap", "failure", "tap_general_fail.txt", 0, "failure"),
    ("tap", "empty-plan-with-passed", "tap_empty_plan_with_passed.txt", 0, "success"),
    ("precedence", "unittest-quotes-env", "unittest_quoted_env.txt", 1, "failure"),
    ("precedence", "cargo-quotes-env", "cargo_quoted_env.txt", 101, "failure"),
)


@pytest.mark.parametrize(
    ("family", "scenario", "fixture", "exit_code", "expected"),
    [(r[0], r[1], r[2], r[3], r[4]) for r in _OUTCOME_MATRIX],
    ids=[f"{r[0]}-{r[1]}" for r in _OUTCOME_MATRIX],
)
def test_captured_fixture_classifies(
    family: str, scenario: str, fixture: str, exit_code: int, expected: str
) -> None:
    result = parse_test_outcome(_read(fixture), exit_code)
    assert result.outcome == expected, f"{family}/{scenario}: {result}"
    if expected == "unknown":
        # An environment/collection error is not labellable and never counts toward escalation.
        assert result.is_infra_failure is True, f"{family}/{scenario}: expected infra"
        assert result.confidence == 0.0
    if expected == "failure":
        # A genuine red keeps its escalation dedup key (node id, or a stable hash).
        assert result.is_infra_failure is False, f"{family}/{scenario}: expected a red"
        assert result.failing_check_id is not None, f"{family}/{scenario}: red lost its dedup key"
        assert result.failing_check_id != ""


def test_positive_proof_wins_over_a_quoted_environment_phrase() -> None:
    # A failure whose assertion quotes "cannot find module" / "error[E0308]" (unittest
    # exit 1, cargo exit 101) is a genuine red — the ran-and-failed proof beats every marker.
    for fixture, exit_code in (("unittest_quoted_env.txt", 1), ("cargo_quoted_env.txt", 101)):
        result = parse_test_outcome(_read(fixture), exit_code)
        assert result.outcome == "failure", fixture
        assert result.is_infra_failure is False, fixture


@pytest.mark.parametrize(
    ("fixture", "exit_code", "expected_total", "expected_failed"),
    [
        ("junit_pytest_fail.txt", 1, 4, 1),
        ("junit_surefire_pass.txt", 0, 4, 0),
        ("junit_empty.txt", 0, 0, 0),
        ("junit_fail_at_zero.txt", 0, 5, 2),
        ("junit_wrapper_empty_then_fail.txt", 0, 5, 2),
        ("junit_counted_wrapper_with_children.txt", 0, 12, 2),
    ],
    ids=lambda v: str(v)[:24],
)
def test_junit_channel_counts_are_read_from_the_xml(
    fixture: str, exit_code: int, expected_total: int, expected_failed: int
) -> None:
    combined = _read(fixture)
    channel = _structured_channel(combined)
    assert channel is not None and channel[0] == expected_total
    assert channel[1] == expected_failed
    result = parse_test_outcome(combined, exit_code)
    if expected_failed > 0:
        # A structured failure is positive proof even when the shell exits 0.
        assert result.outcome == "failure"
    elif expected_total > 0:
        assert result.outcome == "success"
    else:
        assert result.outcome == "unknown" and result.is_infra_failure


def test_junit_leading_zero_test_wrapper_cannot_zero_out_a_real_failure() -> None:
    # A first <testsuite tests="0"> wrapper followed by a failing suite used to be read
    # as "no tests selected" (exit 0, infra) — suppressing a genuine red. Counts are now
    # aggregated across ALL suite tags, so the wrapper cannot zero out the real failure.
    combined = _read("junit_wrapper_empty_then_fail.txt")
    channel = _structured_channel(combined)
    assert channel == (5, 2)
    result = parse_test_outcome(combined, 0)
    assert result.outcome == "failure"
    assert result.is_infra_failure is False


def test_a_passing_run_with_an_empty_tap_plan_is_not_demoted_to_nothing_selected() -> None:
    # A real pass (`4 passed`) that ALSO prints a `1..0` TAP plan used to be read as "no
    # tests selected": the channel branch lacked the ran-and-passed veto the regex branch
    # applies, so the empty plan demoted a genuinely passing run to infra. The explicit
    # non-zero pass count must override the empty plan.
    combined = _read("tap_empty_plan_with_passed.txt")
    assert _structured_channel(combined) == (0, 0)
    result = parse_test_outcome(combined, 0)
    assert result.outcome == "success"
    assert result.is_infra_failure is False


def test_a_passing_run_with_a_zero_test_junit_wrapper_is_not_demoted() -> None:
    # The same veto on the JUnit channel: a `<testsuite tests="0">` wrapper next to a
    # `4 passed` tail is a pass, not "nothing selected".
    combined = _read("junit_zero_wrapper_with_passed.txt")
    assert _structured_channel(combined) == (0, 0)
    result = parse_test_outcome(combined, 0)
    assert result.outcome == "success"
    assert result.is_infra_failure is False


def test_junit_counted_wrapper_is_not_double_counted_with_its_children() -> None:
    # A Surefire-style <testsuites> root carries the aggregate (`tests="12" failures="2"`);
    # summing the child suites as well used to double the counts to (24, 4). The root's own
    # counts are authoritative and the children are not added on top.
    combined = _read("junit_counted_wrapper_with_children.txt")
    channel = _structured_channel(combined)
    assert channel == (12, 2)
    result = parse_test_outcome(combined, 0)
    assert result.outcome == "failure"
    assert result.is_infra_failure is False


def test_tap_channel_counts_are_read_from_the_stream() -> None:
    combined = _read("tap_general_fail.txt")
    channel = _structured_channel(combined)
    assert channel is not None and channel == (4, 2)
    assert parse_test_outcome(combined, 0).outcome == "failure"  # not-ok at exit 0


def test_junit_channel_incomplete_tag_falls_back_to_regex() -> None:
    # A <testsuite> tag without a tests attribute is not a channel; the classifier must not
    # swallow the run just because some XML-ish text is present.
    combined = '<testsuite name="widget" time="0.1">\n1 failed in 0.05s\n</testsuite>'
    assert _structured_channel(combined) is None
    assert parse_test_outcome(combined, 1).outcome == "failure"


def test_tap_todo_and_skip_directives_are_not_failures() -> None:
    # TAP-13: `not ok … # TODO` is expected (a known-bug marker), `# SKIP` is a skip. A listing
    # whose only not-ok points are TODO is a pass; a `1..0` skip-all is nothing selected.
    todo = "TAP version 13\n1..3\nok 1 - a\nnot ok 2 - bend space # TODO unsolved\nok 3 - b\n"
    assert parse_test_outcome(todo, 0).outcome == "success"
    skip_all = "TAP version 13\n1..0 # Skipped: translator not installed\n"
    result = parse_test_outcome(skip_all, 0)
    assert result.outcome == "unknown" and result.is_infra_failure


@pytest.mark.parametrize(
    ("returncode", "fixture"),
    [
        (101, "cargo_fail.txt"),  # cargo red: 101 with test-result proof
        (1, "surefire_fail.txt"),
        (1, "dotnet_fail.txt"),
        (1, "minitest_fail.txt"),
        (2, "phpunit_error.txt"),  # PHPUnit exit 2 = errored test, a RED not infra
        (0, "karma_fail.txt"),  # karma exit 0 with a FAILED summary
        (0, "tap_general_fail.txt"),  # TAP not-ok at exit 0
    ],
    ids=lambda v: str(v)[:30],
)
def test_red_exit_code_taxonomy_is_never_an_environment_failure(
    returncode: int, fixture: str
) -> None:
    result = parse_test_outcome(_read(fixture), returncode)
    assert result.outcome == "failure"
    assert result.is_infra_failure is False


@pytest.mark.parametrize(
    ("returncode", "fixture"),
    [
        (101, "cargo_compile.txt"),  # cargo 101 that never reached a test (compile error)
        (1, "surefire_compile.txt"),
        (1, "dotnet_build.txt"),
        (1, "phpunit_collection.txt"),  # PHP fatal before the runner starts
        (2, "unittest_collection.txt"),  # pytest-style exit 2 stays infra for python
    ],
    ids=lambda v: str(v)[:30],
)
def test_collection_and_build_errors_stay_infra_at_red_exit_codes(
    returncode: int, fixture: str
) -> None:
    result = parse_test_outcome(_read(fixture), returncode)
    assert result.outcome == "unknown"
    assert result.is_infra_failure is True


def test_signal_death_stays_infra_for_new_families() -> None:
    # 128+N (SIGKILL/OOM 137, SIGSEGV 139, SIGTERM 143) is a killed run's fragment, never a
    # verdict — even when the fragment quotes a new-family failure marker.
    fragment = "Executed 3 of 3 (1 FAILED)\n"  # karma marker, but the run was killed
    result = parse_test_outcome(fragment, 137)
    assert result.outcome == "unknown" and result.is_infra_failure is True


def test_karma_zero_failed_summary_is_not_a_red() -> None:
    # The karma summary counts failures in the parens; a "(0 FAILED)" tail is a pass even at
    # exit 0, so the exit-0 proof must require a nonzero count.
    assert parse_test_outcome("Chrome 91: Executed 2 of 2 (0 FAILED)", 0).outcome == "success"
