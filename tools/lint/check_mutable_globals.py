#!/usr/bin/env python3
"""SH001: flag module-level mutable global state (backstop to ruff PLW0603)."""

from __future__ import annotations

import ast
import sys

from _shared import Finding, run

_CODE = "SH001"
_MUTABLE_CTORS = frozenset({"list", "dict", "set", "defaultdict", "OrderedDict"})


def check(path: str, tree: ast.Module) -> list[Finding]:
    """Flag `global` rebindings and module-level mutable-literal bindings."""
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            names = ", ".join(node.names)
            findings.append(
                Finding(
                    path,
                    node.lineno,
                    node.col_offset,
                    f"module-level mutable global '{names}'; inject state or use a class",
                )
            )
    for node in tree.body:
        _check_module_binding(path, node, findings)
    return findings


def _check_module_binding(path: str, node: ast.stmt, findings: list[Finding]) -> None:
    if isinstance(node, ast.AnnAssign):
        targets: list[ast.expr] = [node.target]
        if _is_final(node.annotation):
            return
    elif isinstance(node, ast.Assign):
        targets = node.targets
    else:
        return
    value = node.value
    if value is None or not _is_mutable(value):
        return
    for tgt in targets:
        # UPPER names are exempt only when truly immutable (numbers, strings,
        # tuples, frozenset) — those never reach here because `_is_mutable` is
        # False for them. An uppercase mutable container (`CACHE = {}`) is still
        # shared mutable state and ruff PLW0603 misses it without a `global`, so
        # flag it too.
        if isinstance(tgt, ast.Name) and not _is_dunder(tgt.id):
            findings.append(
                Finding(
                    path,
                    node.lineno,
                    node.col_offset,
                    f"module-level mutable binding '{tgt.id}'; make it Final/const or local",
                )
            )


def _is_dunder(name: str) -> bool:
    """True for dunder names like `__all__` (module API, not mutable state)."""
    return name.startswith("__") and name.endswith("__")


def _is_mutable(value: ast.expr) -> bool:
    """True if the RHS is a mutable literal or a mutable-container constructor."""
    if isinstance(value, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
        return True
    return isinstance(value, ast.Call) and _ctor_name(value.func) in _MUTABLE_CTORS


def _ctor_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _is_final(annotation: ast.expr) -> bool:
    """True if the annotation is `Final` or `Final[...]`."""
    if isinstance(annotation, ast.Name):
        return annotation.id == "Final"
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        return annotation.value.id == "Final"
    return False


if __name__ == "__main__":
    sys.exit(run("SH001", _CODE, check, sys.argv[1:]))
