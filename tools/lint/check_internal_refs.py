#!/usr/bin/env python3
"""SH004: scan shipped text for internal planning vocab so it never leaks to the public repo."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from _shared import Finding, line_has_noqa

_CODE = "SH004"
# check_internal_refs.py lives at <repo>/tools/lint/, so the repo root is two
# levels up (parents[2]); the default targets are resolved against it.
_ROOT = Path(__file__).resolve().parents[2]
# Scan the WHOLE shipped surface, not an allow-list of subtrees. The previous
# default named five targets, which silently exempted every root-level file
# except README.md — so internal vocab in CONTRIBUTING.md, SECURITY.md, AGENTS.md
# or pyproject.toml shipped unscanned. The hook passes no filenames
# (pass_filenames: false), so this default IS the coverage. Verified: planted
# internal vocab in CONTRIBUTING.md returned exit 0 before this change.
_DEFAULT_TARGETS = (".",)
# Directories that are not shipped text: VCS, virtualenvs, caches, build output.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
        "site",
        ".eggs",
    }
)
_SCAN_SUFFIXES = frozenset({".py", ".md", ".yaml", ".yml", ".toml"})

# Internal planning vocab that must not appear in shipped text.
_PATTERNS = (
    re.compile(r"\bSTORY-\d"),
    re.compile(r"\bEPIC-\d"),
    re.compile(r"\bKR-\d"),
    re.compile(r"\bKI-\d"),
    re.compile(r"\bMonth-?\d"),
    re.compile(r"see backlog", re.IGNORECASE),  # noqa: SHUNT-ISO (this gate's own vocab)
    re.compile(r"see journal", re.IGNORECASE),  # noqa: SHUNT-ISO (this gate's own vocab)
)

# Public vocab is safe by construction — model/provider names, ``kill_gate`` and
# ``dogfood`` do not match the patterns above. The one heading that could read as
# planning is the legitimate README ``## Roadmap`` section, allow-listed here.
_ALLOW_LINE = re.compile(r"^\s*#+\s*roadmap\b", re.IGNORECASE)


def _iter_files(targets: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        p = Path(target)
        if not p.is_absolute():
            p = _ROOT / target
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(_walk(p))
    return files


def _walk(root: Path) -> list[Path]:
    """Collect scannable files, PRUNING skip-dirs during the walk, not after it."""
    # Walk the filesystem rather than `git ls-files`: an untracked file is still
    # a file a contributor is about to add, and scanning it is the stricter,
    # more useful behaviour. Pruning happens during the walk so .venv and the
    # caches are never descended into.
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        found.extend(
            Path(dirpath) / name
            for name in sorted(filenames)
            if Path(name).suffix in _SCAN_SUFFIXES
        )
    return found


def _suppressed(source: str, lineno: int) -> bool:
    """True if the line opts out via this gate's token or the wrapper's SHUNT-ISO."""
    # Both gates police the same boundary, so one tag serves both — otherwise
    # every deliberately-public line needs two that can drift apart.
    return line_has_noqa(source, lineno, _CODE) or line_has_noqa(source, lineno, "SHUNT-ISO")


def check_file(path: Path) -> list[Finding]:
    """Return one Finding per internal-ref match in a single file."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    findings: list[Finding] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        if _ALLOW_LINE.match(line):
            continue
        for pattern in _PATTERNS:
            match = pattern.search(line)
            # Evaluate suppression ONLY on a match: line_has_noqa re-splits the
            # whole file, so hoisting it above this guard cost 527k calls (~13s)
            # instead of one per hit.
            if match and not _suppressed(source, lineno):
                msg = f"internal-ref leak '{match.group(0)}'"
                findings.append(Finding(str(path), lineno, match.start(), msg))
    return findings


def main(argv: list[str]) -> int:
    """Scan the target files; exit non-zero if any internal ref is found."""
    targets = tuple(a for a in argv if not a.startswith("-")) or _DEFAULT_TARGETS
    findings: list[Finding] = []
    for path in _iter_files(targets):
        findings.extend(check_file(path))
    for f in findings:
        print(f"{f.path}:{f.line}:{f.col}: [SH004 ERROR] {f.message}", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
