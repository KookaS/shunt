from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

from .base import Verifier, VerifierResult

# The output-classification core (failing_check_id, environment-vs-capability, node-id normalizer)
# lives in `parse` as a pure function so the offline benchmark replay derives byte-identical dedup
# keys from the same output. Re-exported here for callers that import them from this module.
from .parse import (  # noqa: F401 (re-exported: existing importers use these names from tier2)
    _failing_check_id,
    _is_environment_failure,
    _normalize_detail,
    parse_test_outcome,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120
# Fraction of the timeout budget a run may consume before the log says so. Past this the
# next slightly-slower run silently becomes `unknown` + infra, which reads as "no signal".
_BUDGET_WARN_FRACTION = 0.7
# Files whose mere existence is a pytest suite: nothing else creates them.
_PYTEST_MARKER_FILES: Final = ("pytest.ini", "conftest.py")
# Files that name pytest when the project uses it. Matched by substring on purpose — the
# declaration lives in a different table for every packaging tool (`[project]`,
# `[tool.poetry.group.dev.dependencies]` where the package is the KEY, `[tool.pdm…]`,
# `[tool.uv]`, `[tool.hatch.envs…]`, `[options.extras_require]`), and a structured parse
# that misses one turns capture silently off — the exact failure this layer exists to end.
# This predicate is NOT a security control: whether a directory may be verified at all is
# decided by `capture.trust_launch_dir` and the operator's own choice of work_dir, not by
# how convincingly a tree declares a test runner.
_PYTEST_MENTION_FILES: Final = ("pyproject.toml", "setup.cfg", "requirements-dev.txt", "tox.ini")


def _mentions_pytest(path: Path) -> bool:
    """True when *path* exists and names pytest anywhere in its text."""
    try:
        return path.is_file() and "pytest" in path.read_text()
    except (OSError, UnicodeDecodeError):
        return False


def _has_pytest(work_dir: str) -> bool:
    root = Path(work_dir)
    if any((root / name).is_file() for name in _PYTEST_MARKER_FILES):
        return True
    return any(_mentions_pytest(root / name) for name in _PYTEST_MENTION_FILES)


def _has_typescript(work_dir: str) -> bool:
    root = Path(work_dir)
    pkg = root / "package.json"
    if not pkg.is_file():
        return False
    content = pkg.read_text()
    return "jest" in content or "vitest" in content


def _has_go(work_dir: str) -> bool:
    return (Path(work_dir) / "go.mod").is_file()


def _has_rust(work_dir: str) -> bool:
    return (Path(work_dir) / "Cargo.toml").is_file()


# `-p no:cacheprovider`: the verification run is a read-only observation of someone else's
# repo, so it must not write a `.pytest_cache/` into their tree.
_PYTEST_CMD: Final = [
    sys.executable,
    "-m",
    "pytest",
    "-x",
    "--tb=short",
    "-q",
    "-p",
    "no:cacheprovider",
]
_LANGUAGE_DETECTORS: Final[list[tuple[str, str, list[str]]]] = [
    ("python", "pytest", _PYTEST_CMD),
    ("typescript", "jest", ["npx", "jest", "--passWithNoTests"]),
    ("go", "go-test", ["go", "test", "./..."]),
    ("rust", "cargo-test", ["cargo", "test"]),
]


def _detect(work_dir: str) -> tuple[str, list[str]] | None:
    for lang_name, _framework, cmd in _LANGUAGE_DETECTORS:
        if lang_name == "python" and _has_pytest(work_dir):
            return (lang_name, cmd)
        if lang_name == "typescript" and _has_typescript(work_dir):
            return (lang_name, cmd)
        if lang_name == "go" and _has_go(work_dir):
            return (lang_name, cmd)
        if lang_name == "rust" and _has_rust(work_dir):
            return (lang_name, cmd)
    return None


def detect_framework(work_dir: str) -> str | None:
    """The test-framework language detected at *work_dir*, or None when there is none."""
    detected = _detect(work_dir)
    return detected[0] if detected is not None else None


class AutoDetectVerifier(Verifier):
    def __init__(self, timeout: float | None = None) -> None:
        self._timeout = _DEFAULT_TIMEOUT if timeout is None else timeout

    def detect(self, work_dir: str) -> str | None:
        return detect_framework(work_dir)

    def _log_duration(self, lang_name: str, elapsed: float) -> None:
        """Report what the run cost against its budget — the knob is otherwise unguessable."""
        # A run per session close, so the healthy case is debug: at INFO it would be permanent
        # noise. The near-budget case is a WARNING because the next slightly slower run records
        # nothing at all, and a silent stop looks exactly like "there was no signal".
        if elapsed > self._timeout * _BUDGET_WARN_FRACTION:
            logger.warning(
                "verify: %s run took %.1fs of a %.0fs budget (>%d%%) — a slightly slower run "
                "will time out and record NOTHING; raise capture.verify_timeout_seconds",
                lang_name,
                elapsed,
                self._timeout,
                int(_BUDGET_WARN_FRACTION * 100),
            )
        else:
            logger.debug(
                "verify: %s run took %.1fs of a %.0fs budget", lang_name, elapsed, self._timeout
            )

    def verify(self, text: str = "", work_dir: str | None = None) -> VerifierResult:
        if work_dir is None or not os.path.isdir(work_dir):
            return VerifierResult(
                outcome="unknown",
                confidence=0.0,
                detail="no work_dir provided or directory does not exist",
            )

        detected = _detect(work_dir)
        if detected is None:
            return VerifierResult(
                outcome="unknown",
                confidence=0.0,
                detail="no test framework detected",
            )

        lang_name, cmd = detected
        started = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except FileNotFoundError:
            return VerifierResult(
                outcome="unknown",
                confidence=0.0,
                detail=f"{lang_name} runner ({cmd[0]}) not found",
                is_infra_failure=True,
            )
        except subprocess.TimeoutExpired:
            return VerifierResult(
                outcome="unknown",
                confidence=0.0,
                detail=f"{lang_name} tests timed out after {self._timeout}s",
                is_infra_failure=True,
            )

        self._log_duration(lang_name, time.monotonic() - started)
        # The failing check identity is parsed from stdout+stderr (pytest prints node ids to
        # stdout); it becomes the escalation dedup key so a recurrence of the SAME test is what
        # triggers a step, not two unrelated reds. Classification is the shared pure parser, so
        # an offline container replay of the same output derives the identical key.
        combined = f"{result.stdout}\n{result.stderr}"
        return parse_test_outcome(combined, result.returncode)
