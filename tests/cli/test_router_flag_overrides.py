"""CLI flags > env vars > file > packaged defaults in router config precedence."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from shunt.cli import _add_start_flags, _apply_router_flag_overrides
from shunt.router.policy import apply_env_overrides, load_router_policy

_ENV_VARS = ("SHUNT_ROUTER_STRATEGY", "SHUNT_EXPLORATION_ENABLED", "SHUNT_EXPLORE_BUDGET_FRAC")


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _add_start_flags(parser)
    return parser.parse_args(argv)


@pytest.fixture(autouse=True)
def _clean_env(tmp_path: Path) -> Iterator[None]:
    # `_apply_router_flag_overrides` writes os.environ directly, so save/restore by hand
    # rather than relying on monkeypatch (which only undoes what it set itself).
    saved = {name: os.environ.get(name) for name in (*_ENV_VARS, "SHUNT_CONFIG_DIR")}
    for name in _ENV_VARS:
        os.environ.pop(name, None)
    # Point at an empty dir so the packaged router.yaml is the file rung, not a
    # developer's ~/.config/shunt/router.yaml.
    os.environ["SHUNT_CONFIG_DIR"] = str(tmp_path / "empty-config")
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


# ── Flag beats env ──────────────────────────────────────────────────────────


def test_strategy_flag_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "knn")
    _apply_router_flag_overrides(_parse(["--strategy", "always_cheap"]))
    assert apply_env_overrides(load_router_policy()).strategy == "always_cheap"


def test_no_explore_flag_beats_enabling_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_EXPLORATION_ENABLED", "1")
    _apply_router_flag_overrides(_parse(["--no-explore"]))
    assert apply_env_overrides(load_router_policy()).exploration.enabled is False


def test_explore_flag_beats_disabling_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_EXPLORATION_ENABLED", "0")
    _apply_router_flag_overrides(_parse(["--explore"]))
    assert apply_env_overrides(load_router_policy()).exploration.enabled is True


def test_budget_frac_flag_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_EXPLORE_BUDGET_FRAC", "0.9")
    _apply_router_flag_overrides(_parse(["--explore-budget-frac", "0.25"]))
    policy = apply_env_overrides(load_router_policy())
    assert policy.exploration.explore_budget_frac == pytest.approx(0.25)


# ── Absent flag leaves the env var intact ───────────────────────────────────


def test_absent_flags_leave_all_env_vars_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "always_frontier")
    monkeypatch.setenv("SHUNT_EXPLORATION_ENABLED", "0")
    monkeypatch.setenv("SHUNT_EXPLORE_BUDGET_FRAC", "0.75")
    _apply_router_flag_overrides(_parse([]))
    policy = apply_env_overrides(load_router_policy())
    assert policy.strategy == "always_frontier"
    assert policy.exploration.enabled is False
    assert policy.exploration.explore_budget_frac == pytest.approx(0.75)


def test_absent_explore_flag_is_tri_state_none_not_false(monkeypatch: pytest.MonkeyPatch) -> None:
    # BooleanOptionalAction default=None: neither --explore nor --no-explore means
    # "don't touch". Defaulting to False here would silently disable an env-enabled
    # exploration on every `shunt start`.
    monkeypatch.setenv("SHUNT_EXPLORATION_ENABLED", "1")
    args = _parse([])
    assert args.explore is None
    _apply_router_flag_overrides(args)
    assert apply_env_overrides(load_router_policy()).exploration.enabled is True


def test_one_flag_does_not_clobber_the_other_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_EXPLORATION_ENABLED", "0")
    monkeypatch.setenv("SHUNT_EXPLORE_BUDGET_FRAC", "0.75")
    _apply_router_flag_overrides(_parse(["--strategy", "always_cheap"]))
    policy = apply_env_overrides(load_router_policy())
    assert policy.strategy == "always_cheap"
    assert policy.exploration.enabled is False
    assert policy.exploration.explore_budget_frac == pytest.approx(0.75)


def test_no_flags_and_no_env_sets_nothing() -> None:
    _apply_router_flag_overrides(_parse([]))
    assert [name for name in _ENV_VARS if name in os.environ] == []


# ── Full chain: flag > env > file ───────────────────────────────────────────


def test_flag_beats_env_which_beats_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "router.yaml"
    config.write_text(
        "router:\n"
        "  strategy: knn\n"
        "  exploration:\n"
        "    enabled: true\n"
        "    explore_budget_frac: 0.1\n"
    )
    monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("SHUNT_EXPLORE_BUDGET_FRAC", "0.5")
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "always_frontier")

    # File alone: 0.1 / knn.
    assert load_router_policy().exploration.explore_budget_frac == pytest.approx(0.1)
    assert load_router_policy().strategy == "knn"

    _apply_router_flag_overrides(_parse(["--strategy", "always_cheap"]))
    policy = apply_env_overrides(load_router_policy())
    assert policy.strategy == "always_cheap"  # flag beat the env var
    assert policy.exploration.explore_budget_frac == pytest.approx(0.5)  # env beat the file
    assert policy.exploration.enabled is True  # untouched: from the file
