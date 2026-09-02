"""The strategy templates stay in step with the product — a copy-paste file can't rot.

`examples/strategies/*.yaml` are complete `router.yaml` files users copy. These guards
keep each one loadable, on an offered strategy, with the ladder its name promises, and
carrying the same live-model pool the packaged default ships.
"""

from pathlib import Path
from typing import Final

import pytest
import yaml

from shunt.models.config import load_registry
from shunt.router.policy import (
    KNN_CASCADE_STRATEGY,
    LIVE_STRATEGIES,
    SESSION_CASCADE_STRATEGY,
    parse_router_policy,
)

_ROOT = Path(__file__).resolve().parents[1]
_TEMPLATES = _ROOT / "examples" / "strategies"
_PACKAGED_ROUTER = _ROOT / "src" / "shunt" / "config" / "router.yaml"

# The two strategies offered as a user-facing choice, keyed by template filename. The
# other two LIVE_STRATEGIES values (always_cheap / always_frontier) are pinned baselines,
# deliberately without a template — encoded here so adding one is a visible decision.
_OFFERED: Final[dict[str, str]] = {
    "session-cascade": SESSION_CASCADE_STRATEGY,
    "knn-semantic-cascade": KNN_CASCADE_STRATEGY,
}


def _template_files() -> list[Path]:
    return sorted(_TEMPLATES.glob("*.yaml"))


def _packaged_models() -> list[str]:
    return list(yaml.safe_load(_PACKAGED_ROUTER.read_text())["router"]["models"])


def test_exactly_the_offered_strategies_have_templates() -> None:
    assert {p.stem for p in _template_files()} == set(_OFFERED)


@pytest.mark.parametrize("path", _template_files(), ids=lambda p: p.stem)
def test_template_loads_and_names_its_strategy(path: Path) -> None:
    policy = parse_router_policy(yaml.safe_load(path.read_text()))
    assert policy.strategy == _OFFERED[path.stem]
    assert policy.strategy in LIVE_STRATEGIES


@pytest.mark.parametrize("path", _template_files(), ids=lambda p: p.stem)
def test_template_spells_out_the_ladder(path: Path) -> None:
    # A user file replaces the packaged one wholesale and an ABSENT `escalation:` block
    # reads as OFF, which makes an explicitly-named cascade a load error. Every template
    # must therefore carry the block — and carry it ENABLED, since the ladder is the
    # whole content of both names.
    raw = yaml.safe_load(path.read_text())["router"]
    assert "escalation" in raw, f"{path.name} must spell out `escalation:` — absent reads as OFF"
    assert parse_router_policy({"router": raw}).escalation.enabled is True


@pytest.mark.parametrize("path", _template_files(), ids=lambda p: p.stem)
def test_template_live_pool_matches_the_shipped_default(path: Path) -> None:
    # The templates duplicate the packaged `models:` list so a copy-paste install keeps
    # the measured-evidence pool rather than silently widening to the whole registry.
    # That duplication is only safe while this holds.
    assert parse_router_policy(yaml.safe_load(path.read_text())).models == _packaged_models()


@pytest.mark.parametrize("path", _template_files(), ids=lambda p: p.stem)
def test_template_names_only_registered_models(path: Path) -> None:
    registry = set(load_registry().models)
    unknown = set(parse_router_policy(yaml.safe_load(path.read_text())).models) - registry
    assert not unknown, f"{path.name} names models absent from the registry: {sorted(unknown)}"


@pytest.mark.parametrize("path", _template_files(), ids=lambda p: p.stem)
def test_template_carries_no_credential(path: Path) -> None:
    # Templates are copy-paste config; a key belongs in the environment, never in a file.
    text = path.read_text().lower()
    assert "api_key:" not in text
    assert "sk-" not in text


def test_readme_links_every_template() -> None:
    readme = (_TEMPLATES / "README.md").read_text()
    for path in _template_files():
        assert f"({path.name})" in readme, f"README.md does not link {path.name}"
