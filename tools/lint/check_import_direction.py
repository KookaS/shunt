#!/usr/bin/env python3
"""SH006: the shipped package must never import the benchmark harness."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# `benchmark/` may import `shunt`; the reverse would drag the eval harness (and its
# sklearn/matplotlib/swebench extras) into the wheel, where it is deliberately absent.
# The plan named this a hard gate; until now it was upheld only by review.
_FORBIDDEN_ROOT = "benchmark"
_SRC = "src/shunt/"


def _offending_imports(tree: ast.AST) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == _FORBIDDEN_ROOT:
                    hits.append((node.lineno, f"import {alias.name}"))
        # level > 0 is a relative import, which can never reach `benchmark`.
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and (node.module or "").split(".")[0] == _FORBIDDEN_ROOT
        ):
            hits.append((node.lineno, f"from {node.module} import ..."))
    return hits


def check_file(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, statement) for every forbidden import in one file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    return _offending_imports(tree)


def _iter_sources(argv: list[str]) -> list[Path]:
    # Scan the whole package rather than only the staged files: an unstaged import
    # would otherwise slip through, the same rationale as SH004/SH005.
    if argv:
        return [Path(a) for a in argv if _SRC in Path(a).as_posix()]
    return sorted(Path(_SRC).rglob("*.py"))


def main(argv: list[str]) -> int:
    """Fail if any file under src/shunt/ imports the benchmark package."""
    failed = False
    for path in _iter_sources(argv):
        for lineno, statement in check_file(path):
            print(
                f"{path}:{lineno}: [SH006] '{statement}' — src/shunt must not import "
                f"'{_FORBIDDEN_ROOT}' (the harness is not shipped in the wheel)",
                file=sys.stderr,
            )
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
