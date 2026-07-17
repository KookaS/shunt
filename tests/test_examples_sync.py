"""Catalog ↔ registry ↔ validation coherence — no source of truth drifts from another."""

# The config lives in three files with one owner per fact: the runtime registry
# (2 shipped providers), the examples catalog (the wider provider set, as
# copy-paste fragments), and the validation signatures (auth-rejection metadata).
# These three guards keep them from silently recombining or diverging — replacing
# the old whole-row equality that could not survive the split.

from pathlib import Path

import pytest
import yaml

from shunt.models.config import load_registry, parse_registry

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_DIR = _ROOT / "examples" / "providers"

# The connection facts a shipped provider duplicates between runtime and example.
_CONNECTION_KEYS = ("base_url", "api_key_env_var", "litellm_prefix")


def _example_files() -> list[Path]:
    return sorted(_EXAMPLES_DIR.glob("*.yaml"))


def _marker(path: Path) -> str | None:
    first = path.read_text().splitlines()[0].strip()
    prefix = "# shunt-ci:"
    return first[len(prefix) :].strip() if first.startswith(prefix) else None


def _sole_provider(path: Path) -> tuple[str, dict]:
    providers = yaml.safe_load(path.read_text())["providers"]
    assert len(providers) == 1, f"{path.name} should declare exactly one provider"
    name, row = next(iter(providers.items()))
    return name, row


def test_examples_exist() -> None:
    # Guards the parametrized tests below against silently passing on an empty glob.
    assert len(_example_files()) >= 10


# Guard 1 — shipped-provider fidelity. The one residual (pre-existing) duplication:
# deepseek/requesty connection facts live in both the runtime registry (to route)
# and their example fragments (to stand alone). Keep the two copies in agreement.
@pytest.mark.parametrize("path", _example_files(), ids=lambda p: p.stem)
def test_shipped_provider_connection_matches_runtime(path: Path) -> None:
    name, row = _sole_provider(path)
    registry = load_registry().providers
    if name not in registry:
        pytest.skip(f"{name} is catalog-only, not a shipped provider")

    reg = registry[name]
    for key in _CONNECTION_KEYS:
        assert row[key] == getattr(reg, key), (
            f"{path.name} '{key}' has drifted from the runtime registry row for {name!r}."
        )


# Guard 2 — catalog ↔ validation bijection (the ratchet against silent
# recombination). Every `# shunt-ci: probe` example has exactly one signature and
# every signature has exactly one example; a mismatch would let the offline and
# live checks disagree about what a provider does.
def test_catalog_and_validation_are_a_bijection(provider_probe) -> None:
    marked = {_sole_provider(p)[0] for p in _example_files() if _marker(p) == "probe"}
    signatures = set(provider_probe.load_signatures())

    assert marked == signatures, (
        f"probe-marked examples {sorted(marked)} must match measured signatures "
        f"{sorted(signatures)} exactly — add the missing example or the missing signature."
    )


# Guard 3 — schema conformance / no leaked auth_probe. `Provider` forbids extras,
# so an example that still carries an `auth_probe` block (or any stray key) fails
# to parse here — the free structural wall, applied to the catalog too.
@pytest.mark.parametrize("path", _example_files(), ids=lambda p: p.stem)
def test_example_parses_under_the_registry_schema(path: Path) -> None:
    parse_registry(yaml.safe_load(path.read_text()))


# Guard 4 — runtime→catalog coverage. Guards 1-2 iterate the CATALOG, so a
# provider that is in the runtime registry but has no catalog fragment + signature
# would be silently unprobed by every check. This closes that direction: every
# shipped provider must be probeable.
def test_every_runtime_provider_has_a_catalog_fragment_and_signature(provider_probe) -> None:
    runtime = set(load_registry().providers)
    catalog = {_sole_provider(p)[0] for p in _example_files()}
    signatures = set(provider_probe.load_signatures())

    assert runtime <= catalog, (
        f"runtime providers with no examples/ fragment: {sorted(runtime - catalog)} — "
        f"they would be silently unprobed"
    )
    assert runtime <= signatures, (
        f"runtime providers with no measured signature: {sorted(runtime - signatures)} — "
        f"the keyless probe cannot check them"
    )
