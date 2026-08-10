"""Integration-example dirs stay well-formed — the marker convention can't rot.

A tool is CI-eligible iff it ships a handshake.yaml; these guards keep each such dir
consistent (verdict service exists, expected_model is real, docs-only dirs document).
"""

from pathlib import Path
from typing import Final

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_INTEGRATIONS = _ROOT / "examples" / "integrations"
_FAKE_REGISTRY = _ROOT / "tests" / "integrations" / "fake_registry.yaml"

_WIRES = frozenset({"openai", "anthropic", "both"})
_REQUIRED_KEYS = ("tool", "wire", "service", "expected_model", "best_effort", "scenarios")

# Scenario -> the compose file it runs from. `wiring` reads its verdict service from the
# spec's top level (it is the original leg); the others read theirs from a same-named block.
_SCENARIO_COMPOSE: Final[dict[str, str]] = {
    "wiring": "compose.yaml",
    "escalation": "compose.escalation.yaml",
    "live": "compose.live.yaml",
}
# The stable subset. These clients have no onboarding to be brittle about, so their
# escalation leg is REQUIRED: a regression in the ladder must turn the build red rather
# than be absorbed by continue-on-error. Encoded as a test, not as a comment, because the
# whole value of the leg is that someone cannot quietly flip it to best_effort.
_REQUIRED_ESCALATION_TOOLS = frozenset({"curl", "openai-python", "anthropic-python"})


def _handshake_dirs() -> list[Path]:
    return sorted(p.parent for p in _INTEGRATIONS.glob("*/handshake.yaml"))


def _docs_only_dirs() -> list[Path]:
    return sorted(
        d for d in _INTEGRATIONS.iterdir() if d.is_dir() and not (d / "handshake.yaml").exists()
    )


def _fake_registry_models() -> set[str]:
    models = yaml.safe_load(_FAKE_REGISTRY.read_text())["models"]
    return set(models)


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
    assert spec["expected_model"] in _fake_registry_models(), (
        f"{tool_dir.name}: expected_model {spec['expected_model']!r} is not a model in "
        f"tests/integrations/fake_registry.yaml"
    )

    compose = yaml.safe_load((tool_dir / "compose.yaml").read_text())
    assert spec["service"] in compose.get("services", {}), (
        f"{tool_dir.name}: handshake.yaml names verdict service {spec['service']!r}, "
        f"but compose.yaml defines no such service"
    )


@pytest.mark.parametrize("tool_dir", _handshake_dirs(), ids=lambda p: p.name)
def test_declared_scenarios_are_runnable(tool_dir: Path) -> None:
    # A scenario the matrix would dispatch but nothing can run is worse than a missing
    # leg: CI reports a name, not a check. Every declared scenario must therefore own a
    # compose file AND a service in it.
    spec = yaml.safe_load((tool_dir / "handshake.yaml").read_text())
    scenarios = spec["scenarios"]

    assert isinstance(scenarios, list) and scenarios, "'scenarios' must be a non-empty list"
    assert len(set(scenarios)) == len(scenarios), f"{tool_dir.name}: duplicate scenario"
    unknown = set(scenarios) - set(_SCENARIO_COMPOSE)
    assert not unknown, f"{tool_dir.name}: unknown scenario(s) {sorted(unknown)}"
    assert "wiring" in scenarios, f"{tool_dir.name}: every CI-eligible tool runs 'wiring'"

    for scenario in scenarios:
        compose_path = tool_dir / _SCENARIO_COMPOSE[scenario]
        assert compose_path.exists(), (
            f"{tool_dir.name} declares scenario {scenario!r} but ships no {compose_path.name}"
        )
        block = spec if scenario == "wiring" else spec.get(scenario)
        assert isinstance(block, dict), (
            f"{tool_dir.name} declares scenario {scenario!r} but has no '{scenario}:' block "
            f"naming its verdict service and best_effort flag"
        )
        assert isinstance(block.get("best_effort"), bool), (
            f"{tool_dir.name}/{scenario}: 'best_effort' must be a boolean — the CI matrix "
            f"turns it into continue-on-error, and a missing flag silently means 'required'"
        )
        service = block.get("service")
        compose = yaml.safe_load(compose_path.read_text())
        assert service in compose.get("services", {}), (
            f"{tool_dir.name}/{scenario}: names verdict service {service!r}, but "
            f"{compose_path.name} defines no such service"
        )


def test_escalation_is_required_for_the_stable_subset() -> None:
    # The point of the escalation scenario is that it CAN fail the build. If every leg
    # were best_effort, a broken ladder would ship green.
    for tool in sorted(_REQUIRED_ESCALATION_TOOLS):
        spec = yaml.safe_load((_INTEGRATIONS / tool / "handshake.yaml").read_text())
        assert "escalation" in spec["scenarios"], f"{tool} must declare the escalation scenario"
        assert spec["escalation"]["best_effort"] is False, (
            f"{tool}: the escalation leg must be REQUIRED (best_effort: false) — a "
            f"harness that cannot turn the build red is theatre"
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
