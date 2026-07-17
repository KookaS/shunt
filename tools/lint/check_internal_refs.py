#!/usr/bin/env python3
"""SH004: scan shipped text for internal planning vocab so it never leaks to the public repo."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from _shared import Finding, line_has_noqa

_CODE = "SH004"
# check_internal_refs.py lives at <repo>/tools/lint/, so the repo root is two
# levels up (parents[2]); the default targets are resolved against it.
_ROOT = Path(__file__).resolve().parents[2]
# `examples/` ships to users and is config, not prose — so it was scanned by
# NOTHING before: this checker skipped the tree and the suffix, and the docs
# integrity script only walks docs/*.md. gitleaks would catch a pasted key
# there, but not an internal-vocab leak, which is this gate's job.
_DEFAULT_TARGETS = ("benchmark", "examples", "src", "docs", "README.md")
_SCAN_SUFFIXES = frozenset({".py", ".md", ".yaml"})

# Internal planning vocab that must not appear in shipped text.
_PATTERNS = (
    re.compile(r"\bSTORY-\d"),
    re.compile(r"\bEPIC-\d"),
    re.compile(r"\bKR-\d"),
    re.compile(r"\bKI-\d"),
    re.compile(r"\bMonth-?\d"),
    re.compile(r"see backlog", re.IGNORECASE),
    re.compile(r"see journal", re.IGNORECASE),
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
            files.extend(f for f in sorted(p.rglob("*")) if f.suffix in _SCAN_SUFFIXES)
    return files


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
            if match and not line_has_noqa(source, lineno, _CODE):
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
