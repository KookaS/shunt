"""Closure gate: every module that can write a stamp/verdict must move the instrument digest."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from benchmark.escalation import authenticity, schema  # noqa: F401 (instrument surface)
from benchmark.escalation.normalize import mini_swe_agent  # noqa: F401 (instrument surface)
from benchmark.runner import (  # noqa: F401
    offline_replay,
    replay_admissibility,
    state_capture_audit,
    step_snapshots,
    swebench_grading,
)
from shunt.verifiers import aggregator, parse, tier2  # noqa: F401 (instrument surface)

_STAMP_VERDICT_SYMBOLS = frozenset(
    {
        "stamp_step",
        "restamp_trajectory",
        "unstamp_step",
        "_stamp_terminal",
        "parse_test_outcome",
        "_outcome_rank",
        "_OUTCOME_ORDER",
    }
)


def _in_repo_modules() -> set[str]:
    """Every benchmark.* / shunt.verifiers.* module importable in this process."""
    return {
        name
        for name, module in sys.modules.items()
        if (name.startswith("benchmark.") or name.startswith("shunt.verifiers."))
        and getattr(getattr(module, "__spec__", None), "origin", None) is not None
    }


def _symbols_in(name: str) -> set[str]:
    """The stamp/verdict-writing symbols that appear as code in module *name*."""
    spec = sys.modules[name].__spec__
    assert spec is not None and spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _STAMP_VERDICT_SYMBOLS:
            found.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in _STAMP_VERDICT_SYMBOLS:
            found.add(node.attr)
        elif isinstance(node, ast.alias):
            leaf = node.name.rsplit(".", 1)[-1]
            if leaf in _STAMP_VERDICT_SYMBOLS:
                found.add(leaf)
    return found


def test_every_stamp_verdict_module_is_fingerprinted() -> None:
    hits = sorted(name for name in _in_repo_modules() if _symbols_in(name))
    assert hits, "no stamp/verdict-writing module was found — the closure walk is broken"
    covered = set(replay_admissibility._INSTRUMENT_MODULES)
    missing = [name for name in hits if name not in covered]
    assert not missing, (
        "stamp/verdict-writing module(s) can change verdicts without moving the instrument "
        f"digest; add to replay_admissibility._INSTRUMENT_MODULES: {', '.join(missing)}"
    )
