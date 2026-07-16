"""Contract tests for the custom SH0xx lint checks (subprocess = real CLI)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_LINT_DIR = Path(__file__).resolve().parents[2] / "tools" / "lint"


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
