"""A malformed router.yaml must fail loudly at load — never silently fall back."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from shunt.router.policy import RouterPolicy, load_router_policy


def _write_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> Path:
    monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path))
    path = tmp_path / "router.yaml"
    path.write_text(text)
    return path


def test_invalid_yaml_syntax_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_user_config(tmp_path, monkeypatch, "router:\n  strategy: [unclosed\n")
    with pytest.raises(yaml.YAMLError):
        load_router_policy()


def test_duplicate_key_in_user_config_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # strict_yaml_load rejects last-wins shadowing — a copy-paste bug must not silently
    # pick the second value.
    _write_user_config(
        tmp_path,
        monkeypatch,
        "router:\n  strategy: knn\n  strategy: always_cheap\n",
    )
    with pytest.raises(ValueError, match="duplicate key 'strategy'"):
        load_router_policy()


def test_unknown_strategy_in_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Semantically valid YAML, invalid value — and read FROM THE FILE, not from env/kwargs.
    _write_user_config(tmp_path, monkeypatch, "router:\n  strategy: oracle\n")
    with pytest.raises(ValidationError, match="unknown router.strategy"):
        load_router_policy()


def test_unknown_key_in_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # extra="forbid": a typo'd knob must not be silently ignored.
    _write_user_config(tmp_path, monkeypatch, "router:\n  stratergy: knn\n")
    with pytest.raises(ValidationError):
        load_router_policy()


def test_out_of_range_value_in_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_user_config(
        tmp_path,
        monkeypatch,
        "router:\n  exploration:\n    explore_budget_frac: -1.0\n",
    )
    with pytest.raises(ValidationError):
        load_router_policy()


def test_explicit_path_to_malformed_file_also_raises(tmp_path: Path) -> None:
    # The explicit-path rung must fail the same way as the discovered-path rung.
    path = tmp_path / "router.yaml"
    path.write_text("router:\n  strategy: oracle\n")
    with pytest.raises(ValidationError, match="unknown router.strategy"):
        load_router_policy(path)


def test_malformed_config_does_not_silently_yield_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The failure mode this whole file guards: a broken user config resolving to the
    # shipped defaults, so the operator's intended policy is never applied and nothing says so.
    _write_user_config(tmp_path, monkeypatch, "router:\n  strategy: nonsense\n")
    try:
        loaded = load_router_policy()
    except (ValidationError, yaml.YAMLError):
        return
    pytest.fail(f"malformed config silently loaded as {loaded!r} (defaults: {RouterPolicy()!r})")
