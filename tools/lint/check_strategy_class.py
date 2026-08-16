#!/usr/bin/env python3
"""SH011: every strategy is classified, and the `live` set matches the product's."""

# The benchmark used to carry its own idea of "deployable" — three literal sets in three
# files, none derived from `LIVE_STRATEGIES`. The result was a strategy the router
# rejects at boot being drawn as a Pareto-optimal deployable point, a public headline
# built on it, and the kill gate adjudicated on it. This gate is the wall: a new strategy
# cannot be plotted until it is classified, and the classification cannot drift from the
# product's own allowlist.
#
# Pure AST, no imports: the other SH0xx gates work the same way, it runs with no deps
# installed, and a check that imported the code it audits could be broken by the very
# error it is meant to catch.

from __future__ import annotations

import ast
import sys
from pathlib import Path

_CODE = "SH011"
_POLICY = Path("src/shunt/router/policy.py")
_CLASSES = Path("benchmark/routing/strategy_class.py")
_RUN_EVAL = Path("benchmark/routing/run_eval.py")


def _strings(node: ast.expr | None) -> list[str]:
    """Every string literal in a list/tuple/set display, in order."""
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return []
    return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]


def _assigned(tree: ast.Module, name: str) -> ast.expr | None:
    """The value bound to a module-level `name`, annotated or not."""
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return node.value
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return node.value
    return None


def _unwrap_call(node: ast.expr | None) -> ast.expr | None:
    """Strip a single wrapping call such as MappingProxyType({...})."""
    if isinstance(node, ast.Call) and node.args:
        return node.args[0]
    return node


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


def _dict_str_keys(node: ast.expr | None) -> list[str]:
    """The string keys of a dict display."""
    if not isinstance(node, ast.Dict):
        return []
    return [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]


def _dict_str_values(node: ast.expr | None) -> list[str]:
    """The string values of a dict display."""
    if not isinstance(node, ast.Dict):
        return []
    return [
        v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)
    ]


def _classifications(node: ast.expr | None) -> dict[str, tuple[str, bool]]:
    """name -> (StrategyClass member, has a non-empty path_to_live)."""
    out: dict[str, tuple[str, bool]] = {}
    if not isinstance(node, ast.Dict):
        return out
    for key, value in zip(node.keys, node.values, strict=True):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        if not isinstance(value, ast.Call) or not value.args:
            continue
        first = value.args[0]
        member = first.attr if isinstance(first, ast.Attribute) else "?"
        has_path = len(value.args) >= 3 or any(k.arg == "path_to_live" for k in value.keywords)
        out[key.value] = (member, has_path)
    return out


def _report(problems: list[str]) -> int:
    for line in problems:
        print(line)
    if problems:
        print(
            f"\n{_CODE}: {len(problems)} problem(s). Every strategy needs a class in "
            f"{_CLASSES}, and its `live` set must equal LIVE_STRATEGIES."
        )
        return 1
    return 0


def check(root: Path) -> list[str]:
    """Every SH011 problem found under `root`, as printable messages."""
    problems: list[str] = []
    policy, classes, run_eval = (_parse(root / p) for p in (_POLICY, _CLASSES, _RUN_EVAL))
    if policy is None or classes is None or run_eval is None:
        return [f"{_CLASSES}:1:0: [{_CODE} ERROR] cannot parse one of the three source files"]

    live_ids = set(_strings(_assigned(policy, "LIVE_STRATEGIES")))
    display_to_id = _unwrap_call(_assigned(classes, "DISPLAY_TO_ID"))
    non_live = _classifications(_unwrap_call(_assigned(classes, "_NON_LIVE")))
    mapped_ids = set(_dict_str_values(display_to_id))
    mapped_names = set(_dict_str_keys(display_to_id))
    registry_ids = set(_dict_str_keys(_assigned_in_func(run_eval, "registry")))

    if not live_ids:
        problems.append(f"{_POLICY}:1:0: [{_CODE} ERROR] LIVE_STRATEGIES is empty or unreadable")
    for missing in sorted(live_ids - mapped_ids):
        problems.append(
            f"{_CLASSES}:1:0: [{_CODE} ERROR] live strategy {missing!r} has no display-name "
            f"entry in DISPLAY_TO_ID — the benchmark cannot recognise it as live"
        )
    for phantom in sorted(mapped_ids - registry_ids):
        problems.append(
            f"{_CLASSES}:1:0: [{_CODE} ERROR] DISPLAY_TO_ID maps to {phantom!r}, which no "
            f"benchmark strategy registers in {_RUN_EVAL}"
        )
    for unmapped in sorted(registry_ids - mapped_ids):
        problems.append(
            f"{_CLASSES}:1:0: [{_CODE} ERROR] strategy {unmapped!r} is registered but has no "
            f"display name in DISPLAY_TO_ID, so it cannot be classified"
        )
    problems.extend(_class_problems(non_live, mapped_names, live_ids, display_to_id))
    return problems


def _class_problems(
    non_live: dict[str, tuple[str, bool]],
    mapped_names: set[str],
    live_ids: set[str],
    display_to_id: ast.expr | None,
) -> list[str]:
    """Contradictions and missing path-to-live in the non-live table."""
    problems: list[str] = []
    pairs = dict(zip(_dict_str_keys(display_to_id), _dict_str_values(display_to_id), strict=True))
    for name, (member, has_path) in sorted(non_live.items()):
        if pairs.get(name) in live_ids:
            problems.append(
                f"{_CLASSES}:1:0: [{_CODE} ERROR] {name!r} is in LIVE_STRATEGIES and ALSO "
                f"listed as {member} — one of the two is wrong"
            )
        if member == "BLOCKED" and not has_path:
            problems.append(
                f"{_CLASSES}:1:0: [{_CODE} ERROR] {name!r} is BLOCKED with no path_to_live; a "
                f"blocked strategy is kept only with a named route to live, or retired"
            )
    unclassified = sorted(
        n for n in mapped_names if n not in non_live and pairs.get(n) not in live_ids
    )
    for name in unclassified:
        problems.append(
            f"{_CLASSES}:1:0: [{_CODE} ERROR] {name!r} is neither live nor in _NON_LIVE — "
            f"classify it (live / bound / control / blocked)"
        )
    return problems


def _assigned_in_func(tree: ast.Module, name: str) -> ast.expr | None:
    """The value bound to `name` anywhere in the module, including inside a function."""
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return node.value
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return node.value
    return None


def main(argv: list[str]) -> int:
    """Print every SH011 problem under --root (default cwd); exit non-zero on any."""
    root = Path(argv[argv.index("--root") + 1]) if "--root" in argv else Path.cwd()
    return _report(check(root))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
