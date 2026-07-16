#!/usr/bin/env python3
"""SH002: cap module/class/function docstrings at 3 non-blank lines."""

from __future__ import annotations

import ast
import sys

from _shared import Finding, run

_CODE = "SH002"
_MAX_LINES = 3
_Documentable = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def check(path: str, tree: ast.Module) -> list[Finding]:
    """Flag any docstring whose non-blank line count exceeds the cap."""
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, _Documentable):
            continue
        doc = ast.get_docstring(node, clean=True)
        if doc is None:
            continue
        n = sum(1 for line in doc.splitlines() if line.strip())
        if n > _MAX_LINES:
            line = _docstring_line(node)
            findings.append(
                Finding(
                    path,
                    line,
                    0,
                    f"docstring {n} non-blank lines > {_MAX_LINES}; keep it to one intent",
                )
            )
    return findings


def _docstring_line(node: ast.AST) -> int:
    """Return the line of the docstring expression, or 1 for a module."""
    body = getattr(node, "body", None)
    if body and isinstance(body[0], ast.Expr):
        return body[0].lineno
    return 1


if __name__ == "__main__":
    sys.exit(run("SH002", _CODE, check, sys.argv[1:]))
