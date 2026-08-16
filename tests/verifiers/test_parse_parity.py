"""Kill-gate: shared `parse_test_outcome` matches the live `AutoDetectVerifier` path."""

# Byte-identical dedup key, outcome, infra flag, exit code — so an offline container replay
# optimizes production, not a fiction. Mirrors the live/offline failure_event parity discipline.

from __future__ import annotations

import tempfile
from pathlib import Path

from shunt.verifiers.parse import parse_test_outcome
from shunt.verifiers.tier2 import AutoDetectVerifier

_PYPROJECT = "[tool.pytest.ini_options]\naddopts = ''\n"


def _run_live(test_body: str) -> tuple[object, str, int]:
    """Run the REAL verifier subprocess, returning its result + the combined output + rc."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / "pyproject.toml").write_text(_PYPROJECT)
        (repo / "test_x.py").write_text(test_body)
        # Reproduce the exact combined string tier2.verify feeds the parser.
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "--tb=short", "-q"],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )
        combined = f"{proc.stdout}\n{proc.stderr}"
        live = AutoDetectVerifier().verify(work_dir=str(repo))
        return live, combined, proc.returncode


def _assert_parity(test_body: str) -> None:
    live, combined, rc = _run_live(test_body)
    offline = parse_test_outcome(combined, rc)
    # The parity-critical field: the dedup key must be byte-identical.
    assert offline.failing_check_id == live.failing_check_id
    # And the fields the escalation trigger and authenticity read.
    assert offline.outcome == live.outcome
    assert offline.is_infra_failure == live.is_infra_failure
    assert offline.exit_code == live.exit_code
    assert offline.confidence == live.confidence


def test_parity_on_a_genuine_assertion_failure() -> None:
    _assert_parity("def test_x():\n    assert 1 == 2\n")


def test_parity_on_a_passing_suite() -> None:
    _assert_parity("def test_x():\n    assert True\n")


def test_parity_on_a_collection_import_error() -> None:
    # A module-level import of a missing package → collection error (env-cause) in BOTH paths.
    _assert_parity("import definitely_missing_pkg_xyz\n\n\ndef test_x():\n    pass\n")


def test_offline_dedup_key_is_the_pytest_node_id() -> None:
    # A real failing pytest names the node id; the offline parser must recover it verbatim so a
    # recurrence of the SAME test accumulates.
    live, combined, rc = _run_live("def test_widget():\n    assert 1 == 2\n")
    offline = parse_test_outcome(combined, rc)
    assert offline.failing_check_id is not None
    assert offline.failing_check_id.endswith("::test_widget")
    assert offline.failing_check_id == live.failing_check_id
