#!/usr/bin/env python3
"""SH005: keep benchmark pricing out of the router's request path."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from _shared import Finding, run

_CODE = "SH005"

# The registry loader is the ONE module allowed to name pricing: it parses the
# block and hands it to the benchmark. Everything else under src/shunt/ routes
# requests, and a router that reads a list price is a router that depends on
# benchmark data — the import arrow this gate makes structural.
_LOADER_SUFFIX = "src/shunt/models/config.py"
_DEFAULT_TARGET = "src/shunt"

_BANNED = frozenset({"pricing", "input_cost_per_1m", "output_cost_per_1m"})
_BANNED_SUFFIX = "_cost_per_1m"


def _is_router_module(path: str) -> bool:
    """True for modules under src/shunt/ other than the registry loader itself."""
    norm = path.replace("\\", "/")
    if "src/shunt/" not in norm:
        return False
    return not norm.endswith(_LOADER_SUFFIX)


def _is_banned(name: str) -> bool:
    return name in _BANNED or name.endswith(_BANNED_SUFFIX)


def check(path: str, tree: ast.Module) -> list[Finding]:
    """Flag any reference to a pricing field from a non-loader src/shunt module."""
    if not _is_router_module(path):
        return []
    findings: list[Finding] = []
    for node in ast.walk(tree):
        # Narrow before reading lineno/col_offset: ast.AST does not carry them,
        # only the expression nodes _referenced_name can actually match do.
        if not isinstance(node, ast.Attribute | ast.Name | ast.Constant):
            continue
        name = _referenced_name(node)
        if name is not None and _is_banned(name):
            findings.append(
                Finding(
                    path,
                    node.lineno,
                    node.col_offset,
                    f"'{name}' is benchmark pricing; the router must not read it "
                    f"(only {_LOADER_SUFFIX} may)",
                )
            )
    return findings


def _referenced_name(node: ast.Attribute | ast.Name | ast.Constant) -> str | None:
    """The identifier a node names, for the node kinds that can reach a field."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        # Catches the string-key back door: cfg["input_cost_per_1m"].
        return node.value
    return None


def _default_paths() -> list[str]:
    """Every router module — the scan target when pre-commit passes no filenames."""
    return sorted(str(p) for p in Path(_DEFAULT_TARGET).rglob("*.py"))


if __name__ == "__main__":
    # Mirror SH004: scan the whole surface when given no paths, so an unstaged
    # file cannot smuggle a pricing read past the gate (pre-commit runs this
    # with pass_filenames:false + always_run).
    _argv = sys.argv[1:]
    if not [a for a in _argv if not a.startswith("-")]:
        _argv = _argv + _default_paths()
    sys.exit(run("SH005", _CODE, check, _argv))
