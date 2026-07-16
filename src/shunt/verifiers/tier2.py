from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final

from .base import Verifier, VerifierResult

_DEFAULT_TIMEOUT = 120


def _has_pytest(work_dir: str) -> bool:
    root = Path(work_dir)
    cfg = root / "pyproject.toml"
    if cfg.is_file():
        content = cfg.read_text()
        if "pytest" in content or "[tool.pytest" in content:
            return True
    if (root / "setup.cfg").is_file():
        content = (root / "setup.cfg").read_text()
        if "pytest" in content:
            return True
    if (root / "requirements-dev.txt").is_file():
        content = (root / "requirements-dev.txt").read_text()
        if "pytest" in content:
            return True
    return False


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


_PYTEST_CMD: Final = [sys.executable, "-m", "pytest", "-x", "--tb=short", "-q"]
_LANGUAGE_DETECTORS: Final[list[tuple[str, str, list[str], int]]] = [
    ("python", "pytest", _PYTEST_CMD, _DEFAULT_TIMEOUT),
    ("typescript", "jest", ["npx", "jest", "--passWithNoTests"], _DEFAULT_TIMEOUT),
    ("go", "go-test", ["go", "test", "./..."], _DEFAULT_TIMEOUT),
    ("rust", "cargo-test", ["cargo", "test"], _DEFAULT_TIMEOUT),
]


def _detect(work_dir: str) -> tuple[str, list[str], int] | None:
    for lang_name, _framework, cmd, timeout in _LANGUAGE_DETECTORS:
        if lang_name == "python" and _has_pytest(work_dir):
            return (lang_name, cmd, timeout)
        if lang_name == "typescript" and _has_typescript(work_dir):
            return (lang_name, cmd, timeout)
        if lang_name == "go" and _has_go(work_dir):
            return (lang_name, cmd, timeout)
        if lang_name == "rust" and _has_rust(work_dir):
            return (lang_name, cmd, timeout)
    return None


class AutoDetectVerifier(Verifier):
    def __init__(self, timeout: int = _DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    def detect(self, work_dir: str) -> str | None:
        detected = _detect(work_dir)
        if detected is None:
            return None
        return detected[0]

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

        lang_name, cmd, timeout = detected
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
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
                detail=f"{lang_name} tests timed out after {timeout}s",
                is_infra_failure=True,
            )

        if result.returncode == 0:
            return VerifierResult(
                outcome="success",
                confidence=0.8,
                detail=f"{lang_name} tests passed:\n{result.stdout}",
            )
        else:
            stderr = result.stderr[:500] if result.stderr else ""
            return VerifierResult(
                outcome="failure",
                confidence=0.7,
                detail=f"{lang_name} tests failed (rc={result.returncode}):\n{stderr}",
            )
