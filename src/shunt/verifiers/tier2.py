from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
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

# Sane default for a REAL repo suite (30 min), not a toy. The 120s that preceded it made the
# whole capture loop a silent no-op on any repo bigger than a toy: a suite slower than the
# budget times out, records NOTHING, and the escalation signal never accrues — the exact
# "looks configured, does nothing" failure the disclosure exists to end. 1800s is the value
# this project's own suite needs (~22m35s at 2026-08). A per-repo tighter budget is still
# configurable via capture.verify_timeout_seconds; this is the floor for an unconfigured boot.
_DEFAULT_TIMEOUT = 1800
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
# JavaScript/TypeScript runners, matched as quoted package.json keys so a stray prose word
# ("canvas") cannot trigger a false detection. `node --test` matches the test-script form.
_TS_RUNNER_MENTIONS: Final = (
    '"jest"',
    '"vitest"',
    '"mocha"',
    '"jasmine"',
    '"karma"',
    '"ava"',
    "node --test",
)


def _mentions(path: Path, needle: str) -> bool:
    """True when *path* exists and contains *needle* anywhere in its text."""
    try:
        return path.is_file() and needle in path.read_text()
    except (OSError, UnicodeDecodeError):
        return False


def _mentions_pytest(path: Path) -> bool:
    """True when *path* exists and names pytest anywhere in its text."""
    return _mentions(path, "pytest")


def _has_pytest(root: Path) -> bool:
    if any((root / name).is_file() for name in _PYTEST_MARKER_FILES):
        return True
    return any(_mentions_pytest(root / name) for name in _PYTEST_MENTION_FILES)


def _has_typescript(root: Path) -> bool:
    pkg = root / "package.json"
    if not pkg.is_file():
        return False
    try:
        content = pkg.read_text()
    except (OSError, UnicodeDecodeError):
        return False
    return any(name in content for name in _TS_RUNNER_MENTIONS)


def _has_go(root: Path) -> bool:
    return (root / "go.mod").is_file()


def _has_rust(root: Path) -> bool:
    return (root / "Cargo.toml").is_file()


def _has_java(root: Path) -> bool:
    return any((root / name).is_file() for name in ("pom.xml", "build.gradle", "build.gradle.kts"))


def _has_dotnet(root: Path) -> bool:
    return any(any(root.glob(f"*{suffix}")) for suffix in (".csproj", ".fsproj", ".vbproj", ".sln"))


def _has_ruby(root: Path) -> bool:
    if (root / ".rspec").is_file():
        return True
    if _mentions(root / "Gemfile", "rspec"):
        return True
    if _mentions(root / "Gemfile", "minitest"):
        return True
    if _mentions(root / "Gemfile", "test-unit"):
        return True
    if _mentions(root / "Gemfile", "cucumber"):
        return True
    rake = root / "Rakefile"
    return _mentions(rake, "RSpec") or _mentions(rake, "Rake::TestTask")


def _has_php(root: Path) -> bool:
    configs = ("phpunit.xml", "phpunit.dist.xml", "phpunit.xml.dist")
    if any((root / name).is_file() for name in configs):
        return True
    composer = root / "composer.json"
    if not composer.is_file():
        return False
    try:
        content = composer.read_text()
    except (OSError, UnicodeDecodeError):
        return False
    return any(name in content for name in ("phpunit", "pest", "behat", "codeception"))


def _has_cpp(root: Path) -> bool:
    if (root / "CTestTestfile.cmake").is_file():
        return True
    cmake = root / "CMakeLists.txt"
    if not cmake.is_file():
        return False
    try:
        text = cmake.read_text()
    except (OSError, UnicodeDecodeError):
        return False
    return "enable_testing" in text or "add_test" in text


def _has_swift(root: Path) -> bool:
    return (root / "Package.swift").is_file()


def _has_shell(root: Path) -> bool:
    if any(root.rglob("*.bats")):
        return True
    return any(_mentions(root / name, "shunit2") for name in os.listdir(root))


def _has_perl(root: Path) -> bool:
    if any(root.glob("t/*.t")):
        return True
    return any((root / name).is_file() for name in ("Makefile.PL", "Build.PL", "cpanfile"))


def _has_r(root: Path) -> bool:
    return _mentions(root / "DESCRIPTION", "testthat")


def _has_elixir(root: Path) -> bool:
    return (root / "mix.exs").is_file()


def _has_haskell(root: Path) -> bool:
    return (
        (root / "stack.yaml").is_file()
        or (root / "package.yaml").is_file()
        or any(root.glob("*.cabal"))
    )


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
# Default command per language for the families whose invocation does not depend on which
# runner a project declares. Languages with variable commands are resolved in *_command.
_DEFAULT_COMMANDS: Final[dict[str, list[str]]] = {
    "python": _PYTEST_CMD,
    "go": ["go", "test", "./..."],
    "rust": ["cargo", "test"],
    "dotnet": ["dotnet", "test"],
    "swift": ["swift", "test"],
    "shell": ["bats", "--tap", "."],  # --tap forces the machine-readable TAP channel
    "perl": ["prove", "-l"],
    "r": ["Rscript", "-e", "testthat::test_local()"],
    "elixir": ["mix", "test"],
}
# CMake/CTest needs configure + build before the tests exist; the chain is ONE entry so the
# classification still sees a single output stream (the build step's output included, so a
# compile error classifies as infra via its markers rather than as a fabricated red).
_CPP_CMD: Final = [
    "sh",
    "-c",
    "cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && "
    "cmake --build build -j 2 && ctest --test-dir build --output-on-failure",
]
# Per-language detection: (lang name, canonical framework, predicate over the repo root).
# ORDER IS PRECEDENCE — python first because a polyglot tree that also has a pom.xml or go.mod
# is most likely a Python repo the operator wants verified with pytest. New families append.
_LANGUAGE_DETECTORS: Final[list[tuple[str, str, Callable[[Path], bool]]]] = [
    ("python", "pytest", _has_pytest),
    ("typescript", "jest", _has_typescript),
    ("go", "go-test", _has_go),
    ("rust", "cargo-test", _has_rust),
    ("java", "maven-surefire", _has_java),
    ("dotnet", "vstest", _has_dotnet),
    ("ruby", "rspec", _has_ruby),
    ("php", "phpunit", _has_php),
    ("cpp", "ctest", _has_cpp),
    ("swift", "xctest", _has_swift),
    ("shell", "bats", _has_shell),
    ("perl", "prove", _has_perl),
    ("r", "testthat", _has_r),
    ("elixir", "exunit", _has_elixir),
    ("haskell", "hspec", _has_haskell),
]


def _typescript_command(root: Path) -> list[str]:
    """The command for the TypeScript runner a package.json actually declares."""
    pkg = root / "package.json"
    try:
        content = pkg.read_text() if pkg.is_file() else ""
    except (OSError, UnicodeDecodeError):
        content = ""
    if '"vitest"' in content:
        return ["npx", "vitest", "run"]
    if '"mocha"' in content:
        return ["npx", "mocha"]
    if '"jasmine"' in content:
        return ["npx", "jasmine"]
    if '"karma"' in content:
        return ["npx", "karma", "start", "--single-run"]
    if '"ava"' in content:
        return ["npx", "ava"]
    if "node --test" in content:
        return ["node", "--test"]
    return ["npx", "jest", "--passWithNoTests"]


def _java_command(root: Path) -> list[str]:
    """Maven when the tree declares a pom.xml, else the Gradle wrapper / gradle."""
    if (root / "pom.xml").is_file():
        return ["mvn", "test"]
    if (root / "gradlew").is_file():
        return ["./gradlew", "test"]
    return ["gradle", "test"]


def _ruby_command(root: Path) -> list[str]:
    """RSpec when declared, else the rake test task (minitest / test-unit default)."""
    if (root / ".rspec").is_file() or _mentions(root / "Gemfile", "rspec"):
        return ["bundle", "exec", "rspec"]
    return ["bundle", "exec", "rake", "test"]


def _php_command(root: Path) -> list[str]:
    """Pest wraps PHPUnit; run the binary the project's vendor dir actually holds."""
    if _mentions(root / "composer.json", "pest"):
        return ["vendor/bin/pest"]
    return ["vendor/bin/phpunit"]


def _haskell_command(root: Path) -> list[str]:
    """stack when a stack.yaml pins the resolver, else cabal."""
    if (root / "stack.yaml").is_file():
        return ["stack", "test"]
    return ["cabal", "test"]


# Languages whose command depends on which runner the project declares.
_COMMAND_RESOLVERS: Final[dict[str, Callable[[Path], list[str]]]] = {
    "typescript": _typescript_command,
    "java": _java_command,
    "ruby": _ruby_command,
    "php": _php_command,
    "cpp": lambda _root: _CPP_CMD,
    "haskell": _haskell_command,
}


def _command_for(lang_name: str, root: Path) -> list[str]:
    resolver = _COMMAND_RESOLVERS.get(lang_name)
    if resolver is not None:
        return resolver(root)
    return _DEFAULT_COMMANDS[lang_name]


def _detect(work_dir: str) -> tuple[str, list[str]] | None:
    root = Path(work_dir)
    for lang_name, _framework, predicate in _LANGUAGE_DETECTORS:
        if predicate(root):
            return (lang_name, _command_for(lang_name, root))
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
