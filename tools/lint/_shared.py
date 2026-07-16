"""Shared helpers for the SH0xx custom AST lint checks."""

from __future__ import annotations

import ast
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    """One rule violation at a file location."""

    path: str
    line: int
    col: int
    message: str


def line_has_noqa(source: str, lineno: int, code: str) -> bool:
    """Return True if the 1-based source line carries ``# noqa: <code>``."""
    lines = source.splitlines()
    if not 1 <= lineno <= len(lines):
        return False
    comment = lines[lineno - 1].partition("#")[2]
    return "noqa" in comment and code in comment


CheckFn = Callable[[str, ast.Module], list[Finding]]


def run(name: str, code: str, check: CheckFn, argv: list[str]) -> int:
    """Run ``check`` over the file args; exit non-zero on findings unless advisory."""
    advisory = "--advisory" in argv
    paths = [a for a in argv if not a.startswith("-")]
    findings = _collect(paths, code, check)
    _report(name, findings, advisory=advisory)
    if findings and not advisory:
        return 1
    return 0


def _collect(paths: Iterable[str], code: str, check: CheckFn) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        if not path.endswith(".py"):
            continue
        try:
            source = open(path, encoding="utf-8").read()  # noqa: SIM115
        except OSError:
            continue
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            continue
        findings.extend(f for f in check(path, tree) if not line_has_noqa(source, f.line, code))
    return findings


def _report(name: str, findings: list[Finding], *, advisory: bool) -> None:
    if not findings:
        return
    tag = "ADVISORY" if advisory else "ERROR"
    for f in findings:
        print(f"{f.path}:{f.line}:{f.col}: [{name} {tag}] {f.message}", file=sys.stderr)
