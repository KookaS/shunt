#!/usr/bin/env python3
"""SH006: enforce import direction as a (scanned-root, forbidden-import-prefix) rule table."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Each rule: files under `scanned_root` must not import anything under `forbidden_prefix`.
# - src/shunt must not import `benchmark` — the reverse would drag the eval harness (and its
#   sklearn/matplotlib/swebench extras) into the wheel, where it is deliberately absent.
# - the routing/calibration spine must not import `benchmark.escalation` — escalation is a
#   LEAF consumer of the spine; the reverse would make the spine depend on its consumer (a
#   cycle). Baseline 0: escalation is new, so there is nothing to grandfather.
_RULES: tuple[tuple[str, str], ...] = (
    ("src/shunt/", "benchmark"),
    ("benchmark/routing/", "benchmark.escalation"),
    ("benchmark/calibration/", "benchmark.escalation"),
)


def _matches(module: str, prefix: str) -> bool:
    """True iff `module` is `prefix` or a dotted child of it (top-level or package prefix)."""
    return module == prefix or module.startswith(prefix + ".")


def _imports(tree: ast.AST) -> list[tuple[int, str, str]]:
    """(lineno, statement, imported module) for every absolute import in a tree."""
    hits: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits.extend((node.lineno, f"import {a.name}", a.name) for a in node.names)
        # level > 0 is a relative import, which can never reach a top-level package.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            hits.append((node.lineno, f"from {node.module} import ...", node.module))
    return hits


def _rules_for(path_posix: str) -> list[tuple[str, str]]:
    return [(root, prefix) for root, prefix in _RULES if root in path_posix]


def check_file(path: Path) -> list[tuple[int, str, str]]:
    """Return (lineno, statement, forbidden_prefix) for every rule-violating import in one file."""
    rules = _rules_for(path.as_posix())
    if not rules:
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    out: list[tuple[int, str, str]] = []
    for lineno, statement, module in _imports(tree):
        for _root, prefix in rules:
            if _matches(module, prefix):
                out.append((lineno, statement, prefix))
    return out


def _iter_sources(argv: list[str]) -> list[Path]:
    # Scan the whole tree rather than only staged files: an unstaged import would otherwise
    # slip through, the same rationale as SH004/SH005.
    if argv:
        return [Path(a) for a in argv if _rules_for(Path(a).as_posix())]
    seen: set[Path] = set()
    for root, _prefix in _RULES:
        seen.update(Path(root).rglob("*.py"))
    return sorted(seen)


def main(argv: list[str]) -> int:
    """Fail if any file imports a package the rule table forbids from its location."""
    failed = False
    for path in _iter_sources(argv):
        for lineno, statement, prefix in check_file(path):
            print(
                f"{path}:{lineno}: [SH006] '{statement}' — {path.as_posix()} must not import "
                f"'{prefix}' (forbidden import direction)",
                file=sys.stderr,
            )
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
