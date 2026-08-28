"""Regressions for the swebench_multimodal store and the source-aware call sites."""

# Three defects, all offline: the multimodal producer defaulting its manifest to whichever
# manifest is CONFIGURED (which clobbers the 500-row Verified index), collectors that read
# the Verified spec store while the manifest declares another source, and the JS ``test_cmd``
# shape (a list for some repos) being stringified with ``str()``.

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

import pytest

from benchmark import config
from benchmark.routing import integrity
from benchmark.runner import ladder_collect, offline_replay
from benchmark.runner import swebench_multimodal_specs as mm

_MULTIMODAL: Final[str] = "swebench_multimodal"


class TestManifestTarget:
    """Defect 1: the producer must own its output path, not inherit the configured one."""

    def setup_method(self):
        config.load("benchmark/benchmark.yaml")

    def test_default_target_is_the_multimodal_index_not_the_configured_one(self):
        assert config.challenges_path().name == "challenges.json"  # shipped config = Verified
        assert mm.manifest_path().name == "challenges_multimodal.json"
        assert mm.manifest_path() != config.challenges_path()

    def test_write_manifest_never_resolves_the_configured_manifest(self, tmp_path, monkeypatch):
        def _configured() -> Path:
            raise AssertionError("write_manifest must not target the CONFIGURED manifest")

        monkeypatch.setattr(config, "challenges_path", _configured)
        monkeypatch.setattr(mm, "manifest_path", lambda: tmp_path / "challenges_multimodal.json")
        out = mm.write_manifest({"source": _MULTIMODAL, "count": 0})
        assert out == tmp_path / "challenges_multimodal.json"
        assert out.read_text().startswith("{")

    def test_write_manifest_honours_an_explicit_out(self, tmp_path):
        out = mm.write_manifest({"source": _MULTIMODAL}, tmp_path / "elsewhere.json")
        assert out == tmp_path / "elsewhere.json"


class TestSourceAwareCallSites:
    """Defect 2: every collector reads the store the configured manifest declares."""

    def setup_method(self):
        config.load("benchmark/benchmark.yaml")

    def _record(self, monkeypatch) -> list[str]:
        seen: list[str] = []

        def _all_hashes(source: str = integrity.SWEBENCH_SOURCE) -> dict[str, str]:
            seen.append(source)
            return {"chartjs__Chart.js-9027": "h"}

        monkeypatch.setattr(ladder_collect.integrity, "all_hashes", _all_hashes)
        monkeypatch.setattr(ladder_collect.swebench_specs, "manifest_source", lambda: _MULTIMODAL)
        return seen

    def test_sampled_tasks_reads_the_declared_source(self, monkeypatch):
        seen = self._record(monkeypatch)
        ladder_collect._sampled_tasks()
        assert seen == [_MULTIMODAL]

    def test_prepare_reads_the_declared_source(self, monkeypatch):
        seen = self._record(monkeypatch)
        monkeypatch.setattr(ladder_collect.integrity, "model_versions", dict)
        ladder_collect._prepare(["chartjs__Chart.js-9027"])
        assert seen == [_MULTIMODAL]

    def test_no_source_blind_hash_call_sites_remain(self):
        """No caller may take the Verified default of the source-parameterised readers."""
        blind: list[str] = []
        for path in sorted(Path("benchmark").rglob("*.py")):
            if path == Path("benchmark/routing/integrity.py"):
                continue  # the definitions themselves
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr == "swebench_spec_hashes" or (
                    node.func.attr == "all_hashes" and not node.args and not node.keywords
                ):
                    blind.append(f"{path}:{node.lineno} {node.func.attr}()")
        assert blind == []


class TestJsTestCommand:
    """Defect 3: swebench's ``test_cmd`` is a list for some repos; ``str()`` leaks a repr."""

    def test_list_shaped_command_is_a_runnable_shell_sequence(self):
        cmd = offline_replay.swebench_test_command("chartjs/Chart.js", "4.0")
        assert not cmd.startswith("[")
        assert "pnpm install && pnpm run build && " in cmd
        assert "'" not in cmd.split(" && ")[0]

    def test_string_shaped_command_is_unchanged(self):
        assert offline_replay.swebench_test_command("psf/requests", "2.9") == "pytest -rA"

    def test_unknown_shape_is_refused(self):
        with pytest.raises(TypeError):
            offline_replay._as_shell_sequence(42)
