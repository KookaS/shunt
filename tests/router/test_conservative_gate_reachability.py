"""The conservative downshift gate cannot fire in production — documented, not assumed."""

# `ConservativeGate.allows_downshift` needs slack >= 1 - alpha (0.9 at the shipped
# alpha=0.1). Slack is banked only by `RouterEngine.record_outcome`, which NOTHING in
# src/ calls: the sole outcome-write path is `shunt flag`, a separate CLI process
# writing SQLite, while the gate's slack lives in the server process's memory. So the
# gate blocks every downshift forever. These tests pin that reality so it is a
# deliberate state rather than a silent one — if the read-back loop is ever wired,
# they fail and force the startup disclosure to be revisited with it.

from __future__ import annotations

import ast
from pathlib import Path

from shunt.router.budget import ConservativeGate

_SRC = Path(__file__).resolve().parents[2] / "src" / "shunt"


def _calls_named(name: str) -> list[str]:
    """Every src/ module containing a CALL to `name` (definitions don't count)."""
    hits: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            attr = func.attr if isinstance(func, ast.Attribute) else None
            ident = func.id if isinstance(func, ast.Name) else attr
            if ident == name:
                hits.append(str(path.relative_to(_SRC)))
    return sorted(set(hits))


def test_router_engine_record_outcome_has_no_production_caller() -> None:
    # engine.py calls the GATE's record_outcome from inside its own method; that is
    # the definition site, not a caller of RouterEngine.record_outcome itself.
    assert _calls_named("record_outcome") == ["router/engine.py"]


def test_a_fresh_gate_blocks_downshift_at_the_shipped_alpha() -> None:
    # 0.1 is the shipped default in src/shunt/config/router.yaml.
    gate = ConservativeGate(alpha=0.1)
    assert not gate.allows_downshift()


def test_only_a_verified_downshift_success_can_unblock_it() -> None:
    gate = ConservativeGate(alpha=0.1)
    for _ in range(10):
        gate.record_outcome(downshift=True, success=True)
    assert gate.allows_downshift()
