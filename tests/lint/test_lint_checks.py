"""Contract tests for the custom SH0xx lint checks (subprocess = real CLI)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_LINT_DIR = _ROOT / "tools" / "lint"


def _run(script: str, *args: str) -> int:
    return subprocess.run(
        [sys.executable, str(_LINT_DIR / script), *args],
        capture_output=True,
        text=True,
    ).returncode


def test_sh001_catches_uppercase_mutable_container(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    f.write_text("CACHE = {}\n")
    assert _run("check_mutable_globals.py", str(f)) == 1


def test_sh001_ignores_immutable_uppercase_constants(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    f.write_text('MAX = 5\nNAMES = ("a",)\nPATH = "x"\nFROZEN = frozenset({1})\n')
    assert _run("check_mutable_globals.py", str(f)) == 0


def test_sh001_ignores_final_annotated_container(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    f.write_text("from typing import Final\n\nTABLE: Final = {}\n")
    assert _run("check_mutable_globals.py", str(f)) == 0


def test_sh004_catches_planted_story_ref(tmp_path: Path) -> None:
    f = tmp_path / "leak.py"
    f.write_text("# tracked in STORY-9.9\nx = 1\n")
    assert _run("check_internal_refs.py", str(f)) == 1


def test_sh004_passes_clean_public_vocab(tmp_path: Path) -> None:
    f = tmp_path / "clean.md"
    f.write_text("# Roadmap\n\nUses kill_gate and dogfood on claude-opus-4-6.\n")
    assert _run("check_internal_refs.py", str(f)) == 0


def test_sh004_default_tree_is_clean() -> None:
    assert _run("check_internal_refs.py") == 0


def test_sh004_scans_yaml_when_walking_a_directory(tmp_path: Path) -> None:
    f = tmp_path / "provider.yaml"
    f.write_text("# see backlog for the rollout\nbase_url: https://example.com/v1\n")
    assert _run("check_internal_refs.py", str(tmp_path)) == 1


def test_sh004_default_scan_covers_examples() -> None:
    # examples/ ships to users but was scanned by NOTHING: this checker skipped
    # both the tree and the .yaml suffix, and check-docs-integrity.sh only walks
    # docs/*.md. Planting a real leak is the only way to prove the tree is wired
    # into the DEFAULT target list, which is the thing that silently regresses.
    examples = _ROOT / "examples"
    created = not examples.exists()
    examples.mkdir(parents=True, exist_ok=True)
    planted = examples / "_sh004_contract_probe.yaml"
    planted.write_text("# see backlog\n")
    try:
        assert _run("check_internal_refs.py") == 1
    finally:
        planted.unlink()
        if created:
            examples.rmdir()


def _src_shunt_file(tmp_path: Path, rel: str, body: str) -> Path:
    """Write a file at a real-looking src/shunt/<rel> path (SH005 keys on the path)."""
    f = tmp_path / "src" / "shunt" / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(body)
    return f


def test_sh005_catches_pricing_attribute_in_router(tmp_path: Path) -> None:
    f = _src_shunt_file(tmp_path, "router/pick.py", "def f(cfg):\n    return cfg.pricing\n")
    assert _run("check_pricing_isolation.py", str(f)) == 1


def test_sh005_catches_cost_field_string_key(tmp_path: Path) -> None:
    # The string-subscript back door must not bypass the gate.
    f = _src_shunt_file(
        tmp_path, "router/cost.py", 'def f(d):\n    return d["input_cost_per_1m"]\n'
    )
    assert _run("check_pricing_isolation.py", str(f)) == 1


def test_sh005_exempts_the_registry_loader(tmp_path: Path) -> None:
    f = _src_shunt_file(tmp_path, "models/config.py", "def f(cfg):\n    return cfg.pricing\n")
    assert _run("check_pricing_isolation.py", str(f)) == 0


def test_sh005_ignores_modules_outside_src_shunt(tmp_path: Path) -> None:
    # The benchmark is the legitimate pricing consumer.
    f = tmp_path / "benchmark" / "config.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("def f(cfg):\n    return cfg.pricing\n")
    assert _run("check_pricing_isolation.py", str(f)) == 0


def test_sh005_passes_a_clean_router_module(tmp_path: Path) -> None:
    f = _src_shunt_file(tmp_path, "router/clean.py", "def f(cfg):\n    return cfg.tier\n")
    assert _run("check_pricing_isolation.py", str(f)) == 0
