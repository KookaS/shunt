"""B9: an environment/collection failure is classified non-capability (never escalates)."""

# The classifier is gated on the runner's exit code so a genuine assertion red whose message
# merely quotes an import string is NOT misclassified as infra. The `..._subprocess` cases
# prove both directions against a REAL pytest run in a temp repo (local subprocess only).

from __future__ import annotations

from pathlib import Path

import pytest

from shunt.verifiers.parse import parse_test_outcome
from shunt.verifiers.tier2 import AutoDetectVerifier, _is_environment_failure


@pytest.mark.parametrize(
    "text",
    [
        "ImportError while importing test module '/x/test_a.py'",
        "!!!!!! Interrupted: 1 error during collection !!!!!!",
        "ERROR collecting tests/test_a.py",
        "go: cannot find module providing package example/x",
        "no required module provides package; to add it: go get ...",
        "error[E0463]: can't find crate for `serde`",
        "Cannot find module 'lodash' from 'index.js'",
    ],
)
def test_unambiguous_collection_markers_detected_any_exit_code(text: str) -> None:
    # These phrases appear only at collection/build time → environmental at exit 1 OR 2.
    assert _is_environment_failure(text, returncode=1) is True
    assert _is_environment_failure(text, returncode=2) is True


def test_ambiguous_import_string_gated_on_exit_code() -> None:
    text = "E   ModuleNotFoundError: No module named 'widgets'"
    assert _is_environment_failure(text, returncode=2) is True  # pytest collection error
    # The SAME string quoted inside a failing assertion (pytest exit 1) is a real red (B9).
    assert _is_environment_failure(text, returncode=1) is False


@pytest.mark.parametrize(
    "text",
    [
        "assert 1 == 2\nE   assert 1 == 2",
        "AssertionError: expected 3 got 4",
        "FAILED tests/test_a.py::test_x - assert False",
        "panic: runtime error: index out of range [5]",
        # The confirmed B9 bug: an assert whose message quotes an import string (exit 1).
        'FAILED test_a.py::test_x - AssertionError: assert "No module named foo" in "bar"',
    ],
)
def test_genuine_capability_failures_not_flagged(text: str) -> None:
    assert _is_environment_failure(text, returncode=1) is False


def _pytest_repo(tmp_path: Path, test_body: str) -> str:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (tmp_path / "test_case.py").write_text(test_body)
    return str(tmp_path)


def test_assert_quoting_import_string_stays_failure_subprocess(tmp_path: Path) -> None:
    # (a) A genuine assertion red (pytest exit 1) whose message contains "No module named"
    # must stay outcome=failure and NOT be treated as infra.
    work_dir = _pytest_repo(tmp_path, 'def test_x():\n    assert "No module named foo" in "bar"\n')
    result = AutoDetectVerifier().verify(work_dir=work_dir)
    assert result.outcome == "failure"
    assert result.is_infra_failure is False
    assert result.failing_check_id is not None  # a real red keeps its escalation dedup key


_QUOTABLE_MARKERS = (
    "cannot find module foo",  # the regex requires the trailing separator
    "unresolved import",
    "no required module provides package",
    "can't find crate for",
)


@pytest.mark.parametrize("marker", _QUOTABLE_MARKERS)
def test_assert_quoting_collection_marker_stays_failure_subprocess(
    tmp_path: Path, marker: str
) -> None:
    # F2: these markers used to classify as infra at ANY exit code, so a genuine pytest
    # assertion red that merely QUOTED one was silently reclassified — outcome "unknown" is not
    # labellable, so the verified red never reached the outcome store nor the escalation counter.
    work_dir = _pytest_repo(tmp_path, f'def test_x():\n    assert {marker!r} in "bar"\n')
    result = AutoDetectVerifier().verify(work_dir=work_dir)
    assert result.outcome == "failure"
    assert result.is_infra_failure is False
    assert result.failing_check_id == "test_case.py::test_x"


@pytest.mark.parametrize("marker", _QUOTABLE_MARKERS)
def test_quoted_marker_without_a_ran_and_failed_signal_stays_infra(marker: str) -> None:
    # The other direction: a genuine build/collection error (no test ever ran, so no
    # `FAILED <nodeid>` / `N failed` line) still classifies environmental at its own exit code.
    assert _is_environment_failure(f"error: {marker} 'x'", returncode=1) is True
    assert _is_environment_failure(f"error: {marker} 'x'", returncode=101) is True


@pytest.mark.parametrize(
    "signal",
    [
        "FAILED tests/test_a.py::test_x - AssertionError",
        "1 failed in 0.02s",
        "--- FAIL: TestWidget (0.00s)",
        "failures:",
        "Tests:       1 failed, 2 passed, 3 total",
    ],
)
def test_ran_and_failed_signal_overrides_a_quoted_marker(signal: str) -> None:
    combined = f"{signal}\nE   AssertionError: assert 'cannot find module foo' in 'bar'"
    assert _is_environment_failure(combined, returncode=1) is False
    # ...but never at the pytest collection exit code: exit 2 means no test outcome exists.
    assert _is_environment_failure(combined, returncode=2) is True


def test_real_collection_error_is_infra_subprocess(tmp_path: Path) -> None:
    # (b) An un-importable module → pytest collection error (exit 2) → unknown + infra, so it
    # never counts toward escalation.
    work_dir = _pytest_repo(tmp_path, "import nonexistent_module_xyz\n\ndef test_x():\n    pass\n")
    result = AutoDetectVerifier().verify(work_dir=work_dir)
    assert result.outcome == "unknown"
    assert result.is_infra_failure is True


# ---------------------------------------------------------------------------
# A3/A2: an exit code that means the SESSION never validly ran is never a capability red.
# Only exit 0 (passed) and 1 (ran, some failed) carry a verdict. Before this was enforced, the
# guard covered exit 2 alone, so 3/4/5 fell through to the failure branch and were classified on
# output WORDING — 746 replayed benchmark steps at exit 4 (pytest USAGE_ERROR, i.e. a selector
# that named nothing) split 394 fabricated capability reds / 352 infra purely on whether the
# runner happened to print "ImportError while importing" or an underscore-wrapped
# "____ ERROR collecting ____" that the `^ERROR collecting` anchor misses.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("returncode", "output"),
    [
        # (exit code, the output that runner really emits at it)
        (2, "!!!!!! Interrupted: 1 error during collection !!!!!!"),
        (3, "INTERNALERROR> Traceback (most recent call last):\nINTERNALERROR>   File ..."),
        (4, "ERROR: not found: /testbed/a.py::test_x\n(no name '/testbed/a.py::test_x' in ...)"),
        (5, "collected 0 items\n\n=========== no tests ran in 0.01s ==========="),
        (137, "running tests ...\n"),  # SIGKILL / OOM — killed, so the output is a fragment
    ],
    ids=lambda v: str(v)[:12],
)
def test_non_session_exit_is_infra_never_a_capability_red(returncode: int, output: str) -> None:
    result = parse_test_outcome(output, returncode)
    assert result.is_infra_failure is True
    assert result.outcome == "unknown"
    # The load-bearing one: no dedup key means it can never accumulate toward a recurrence
    # trigger. The 394 fabricated reds each carried a per-instance CONSTANT `hash:` key, which
    # fires the escalation trigger by construction on every trajectory of that instance.
    assert result.failing_check_id is None


def test_signal_death_is_infra_even_when_output_quotes_a_failure() -> None:
    # A killed run's output is a truncated fragment: a `FAILED …` line from a test that ran
    # before the kill does not make the RUN a verdict. 137 must not be read as a red.
    combined = "FAILED tests/test_a.py::test_x - AssertionError\nKilled"
    assert parse_test_outcome(combined, 137).is_infra_failure is True


# ---------------------------------------------------------------------------
# REGRESSION GUARD for the exit-0 no-tests-ran gate. The gate is EVIDENCE OF ABSENCE: a clean
# exit only stops being a pass when the output SAYS nothing was selected. It must never be
# inverted into "prove a test ran", which would demote every green whose runner dialect the
# patterns do not know — a live-path regression. These two cases are what pins that.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output",
    [
        "5 passed in 1.2s",
        "===== 460 passed, 19 skipped in 12.03s =====",  # `\b0 passed\b` must not match "460 pa…"
        "ok\ncoverage: 100% of statements",  # a dialect the patterns do not know stays a pass
        "tests finished: 10 passed, in 0.42 seconds",  # sympy's real green tail
    ],
)
def test_clean_exit_with_a_real_green_tail_stays_success(output: str) -> None:
    result = parse_test_outcome(output, 0)
    assert result.outcome == "success"
    assert result.is_infra_failure is False


@pytest.mark.parametrize(
    "output",
    [
        "tests finished: 0 passed, in 0.00 seconds",  # sympy, given a selector matching nothing
        "collected 0 items\n\n===== no tests ran in 0.01s =====",
        "Ran 0 tests in 0.000s\n\nOK",  # unittest
        "?   \tgithub.com/x/y\t[no test files]",  # go
    ],
)
def test_clean_exit_that_selected_nothing_is_not_a_pass(output: str) -> None:
    # A2/A1: `bin/test -C --verbose test_coth` selects zero tests, prints `0 passed`, exits 0.
    # Stamping that green produced 4064 benchmark steps marked PASS on runs where nothing ran.
    result = parse_test_outcome(output, 0)
    assert result.outcome == "unknown"
    assert result.is_infra_failure is True
    assert result.confidence == 0.0
