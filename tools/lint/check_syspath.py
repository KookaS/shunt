#!/usr/bin/env python3
"""SH003: ban sys.path mutation (backstop to ruff TID251)."""

from __future__ import annotations

import ast
import sys

from _shared import Finding, run

_CODE = "SH003"
_MUTATORS = frozenset({"insert", "append", "extend"})


def _is_sys_path(node: ast.expr) -> bool:
    """True if the expression is the ``sys.path`` attribute access."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "path"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def check(path: str, tree: ast.Module) -> list[Finding]:
    """Flag sys.path.insert/append/extend calls and assignments to sys.path."""
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _MUTATORS
            and _is_sys_path(node.func.value)
        ):
            findings.append(
                Finding(
                    path,
                    node.lineno,
                    node.col_offset,
                    "no sys.path mutation — use absolute imports "
                    "(`benchmark` is reached via pytest pythonpath, not installed)",
                )
            )
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if _is_sys_path(tgt):
                    findings.append(
                        Finding(path, node.lineno, node.col_offset, "no sys.path assignment")
                    )
    return findings


if __name__ == "__main__":
    sys.exit(run("SH003", _CODE, check, sys.argv[1:]))
