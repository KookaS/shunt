from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

from .base import Verifier, VerifierResult

_DEFAULT_TIMEOUT = 120

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


class AutoDetectVerifier(Verifier):
    def __init__(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
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

        lang_name, cmd = detected
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

        if result.returncode == 0:
            return VerifierResult(
                outcome="success",
                confidence=0.8,
                detail=f"{lang_name} tests passed:\n{result.stdout}",
                exit_code=0,
            )
        # The failing check identity is parsed from stdout+stderr (pytest prints node ids to
        # stdout); it becomes the escalation dedup key so a recurrence of the SAME test is what
        # triggers a step, not two unrelated reds.
        combined = f"{result.stdout}\n{result.stderr}"
        stderr = result.stderr[:500] if result.stderr else ""
        if _is_environment_failure(combined, result.returncode):
            # An environmental red (missing module, broken collection) is not a capability
            # outcome — treat it like infra: unknown + is_infra_failure, so the capture path
            # writes no Tier-2 label and it never counts toward escalation.
            return VerifierResult(
                outcome="unknown",
                confidence=0.0,
                detail=f"{lang_name} environment/collection error (rc={result.returncode}):\n"
                f"{stderr}",
                exit_code=result.returncode,
                is_infra_failure=True,
            )
        return VerifierResult(
            outcome="failure",
            confidence=0.7,
            detail=f"{lang_name} tests failed (rc={result.returncode}):\n{stderr}",
            exit_code=result.returncode,
            failing_check_id=_failing_check_id(combined),
        )
