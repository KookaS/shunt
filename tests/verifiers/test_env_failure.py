"""B9: an environment/collection failure is classified non-capability (never escalates)."""

# The classifier is gated on the runner's exit code so a genuine assertion red whose message
# merely quotes an import string is NOT misclassified as infra. The `..._subprocess` cases
# prove both directions against a REAL pytest run in a temp repo (local subprocess only).

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_real_collection_error_is_infra_subprocess(tmp_path: Path) -> None:
    # (b) An un-importable module → pytest collection error (exit 2) → unknown + infra, so it
    # never counts toward escalation.
    work_dir = _pytest_repo(tmp_path, "import nonexistent_module_xyz\n\ndef test_x():\n    pass\n")
    result = AutoDetectVerifier().verify(work_dir=work_dir)
    assert result.outcome == "unknown"
    assert result.is_infra_failure is True
