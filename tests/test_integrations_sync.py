"""Integration-example dirs stay well-formed — the marker convention can't rot.

A tool is CI-eligible iff it ships a handshake.yaml; these guards keep each such dir
consistent (verdict service exists, tier is real, docs-only dirs document).
"""

from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_INTEGRATIONS = _ROOT / "examples" / "integrations"
_FAKE_REGISTRY = _ROOT / "tests" / "integrations" / "fake_registry.yaml"

_WIRES = frozenset({"openai", "anthropic", "both"})
_REQUIRED_KEYS = ("tool", "wire", "service", "expected_tier", "best_effort")


def _handshake_dirs() -> list[Path]:
    return sorted(p.parent for p in _INTEGRATIONS.glob("*/handshake.yaml"))


def _docs_only_dirs() -> list[Path]:
    return sorted(
        d for d in _INTEGRATIONS.iterdir() if d.is_dir() and not (d / "handshake.yaml").exists()
    )


def _fake_registry_tiers() -> set[str]:
    models = yaml.safe_load(_FAKE_REGISTRY.read_text())["models"]
    return {row["tier"] for row in models.values()}


def test_ci_tools_exist() -> None:
    # Guard the parametrized tests against silently passing on an empty glob.
    assert len(_handshake_dirs()) >= 10


@pytest.mark.parametrize("tool_dir", _handshake_dirs(), ids=lambda p: p.name)
def test_ci_tool_dir_is_well_formed(tool_dir: Path) -> None:
    for required in ("README.md", "compose.yaml", "handshake.yaml"):
        assert (tool_dir / required).exists(), f"{tool_dir.name}/ is missing {required}"

    spec = yaml.safe_load((tool_dir / "handshake.yaml").read_text())
    for key in _REQUIRED_KEYS:
        assert key in spec, f"{tool_dir.name}/handshake.yaml is missing '{key}'"

    assert spec["tool"] == tool_dir.name, "handshake.yaml 'tool' must match the directory name"
    assert spec["wire"] in _WIRES, f"'wire' must be one of {_WIRES}, got {spec['wire']!r}"
    assert isinstance(spec["best_effort"], bool), "'best_effort' must be a boolean"
    assert spec["expected_tier"] in _fake_registry_tiers(), (
        f"{tool_dir.name}: expected_tier {spec['expected_tier']!r} is not a tier in "
        f"tests/integrations/fake_registry.yaml"
    )

    compose = yaml.safe_load((tool_dir / "compose.yaml").read_text())
    assert spec["service"] in compose.get("services", {}), (
        f"{tool_dir.name}: handshake.yaml names verdict service {spec['service']!r}, "
        f"but compose.yaml defines no such service"
    )


@pytest.mark.parametrize("tool_dir", _docs_only_dirs(), ids=lambda p: p.name)
def test_docs_only_dir_documents(tool_dir: Path) -> None:
    # A dir with no handshake.yaml is docs-only — it must carry a README and must NOT
    # ship a compose.yaml. A compose without a handshake would be silently skipped by
    # the CI matrix (it globs */handshake.yaml) yet look like a real, tested leg.
    assert (tool_dir / "README.md").exists(), f"docs-only {tool_dir.name}/ needs a README.md"
    assert not (tool_dir / "compose.yaml").exists(), (
        f"{tool_dir.name}/ has a compose.yaml but no handshake.yaml — it would never run "
        f"in CI. Add a handshake.yaml to make it CI-eligible, or remove the compose."
    )
