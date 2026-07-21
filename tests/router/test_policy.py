from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from shunt.router.policy import (
    LIVE_STRATEGIES,
    ExplorationPolicy,
    KnnPolicy,
    RouterPolicy,
    apply_env_overrides,
    load_router_policy,
    packaged_policy_path,
    parse_router_policy,
)


def test_defaults_are_shipped_values() -> None:
    p = RouterPolicy()
    assert p.strategy == "knn"
    assert p.policy == KnnPolicy(k=20, success_rate_threshold=0.6, min_samples=3)
    assert p.exploration.enabled is True
    assert p.exploration.explore_budget_frac == pytest.approx(0.4)


class TestPackagedPolicyShipsWithThePackage:
    """Packaged router.yaml matches shipped defaults, except `models:` (see below)."""

    def test_packaged_policy_path_exists(self) -> None:
        assert packaged_policy_path().is_file()

    def test_packaged_file_equals_shipped_defaults(self) -> None:
        # A drift here means the YAML a user copies no longer describes what an
        # unconfigured install actually does (`models:` excluded — see class docstring).
        packaged = load_router_policy(packaged_policy_path())
        assert packaged.model_copy(update={"models": []}) == RouterPolicy()
        assert packaged.models  # the packaged file always curates a non-empty list

    def test_packaged_default_used_when_no_user_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path))
        assert load_router_policy() == load_router_policy(packaged_policy_path())

    def test_user_config_wins_over_packaged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "router.yaml").write_text(
            "router:\n  strategy: always_cheap\n  exploration:\n    enabled: false\n"
        )
        monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path))
        policy = load_router_policy()
        assert policy.strategy == "always_cheap"
        assert policy.exploration.enabled is False


def test_unknown_strategy_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown router.strategy"):
        RouterPolicy(strategy="oracle")


def test_knn_cascade_is_not_live_eligible() -> None:
    # knn_cascade is a benchmark-only quality cascade (mid-session verify-escalate is not
    # one cache-safe per-session decision), so it must fail live-policy validation.
    assert "knn_cascade" not in LIVE_STRATEGIES
    with pytest.raises(ValidationError, match="unknown router.strategy"):
        RouterPolicy(strategy="knn_cascade")


@pytest.mark.parametrize("name", LIVE_STRATEGIES)
def test_all_live_strategies_accepted(name: str) -> None:
    assert RouterPolicy(strategy=name).strategy == name


def test_extra_key_forbidden() -> None:
    with pytest.raises(ValidationError):
        RouterPolicy.model_validate({"strategy": "knn", "bogus": 1})


def test_parse_router_policy_unwraps_router_key() -> None:
    data = {"router": {"strategy": "always_cheap", "exploration": {"enabled": False}}}
    p = parse_router_policy(data)
    assert p.strategy == "always_cheap"
    assert p.exploration.enabled is False


def test_parse_router_policy_empty_is_defaults() -> None:
    assert parse_router_policy(None) == RouterPolicy()
    assert parse_router_policy({}) == RouterPolicy()


def test_load_router_policy_missing_file_is_defaults(tmp_path: Path) -> None:
    # A missing explicit path falls back to the packaged file (which curates a
    # non-empty `models:` list), not the bare code default — see
    # TestPackagedPolicyShipsWithThePackage's docstring for why the two differ.
    assert load_router_policy(tmp_path / "nope.yaml") == load_router_policy(packaged_policy_path())


def test_load_router_policy_reads_file(tmp_path: Path) -> None:
    f = tmp_path / "router.yaml"
    f.write_text("router:\n  strategy: always_cheap\n  policy:\n    k: 10\n")
    p = load_router_policy(f)
    assert p.strategy == "always_cheap"
    assert p.policy.k == 10


def test_exploration_policy_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        ExplorationPolicy.model_validate({"enabled": True, "typo": 1})


@pytest.mark.parametrize(
    "field,value",
    [
        ("prior_alpha", 0.0),  # Beta(0,.) would crash np.random.beta on the live path
        ("prior_beta", -1.0),
        ("explore_budget_frac", -0.1),
        ("conservative_alpha", 1.5),
        ("propensity_mc_samples", -1),
    ],
)
def test_exploration_policy_bounds_reject_bad_values(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        ExplorationPolicy.model_validate({field: value})


@pytest.mark.parametrize(
    "field,value",
    [("k", 0), ("success_rate_threshold", 1.5), ("min_samples", -1)],
)
def test_knn_policy_bounds_reject_bad_values(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        KnnPolicy.model_validate({field: value})


def test_null_router_key_is_defaults() -> None:
    assert parse_router_policy({"router": None}) == RouterPolicy()


# ── Env overrides (env > file > packaged default) ────────────────────────────


def test_env_overrides_noop_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNT_ROUTER_STRATEGY", raising=False)
    monkeypatch.delenv("SHUNT_EXPLORATION_ENABLED", raising=False)
    monkeypatch.delenv("SHUNT_EXPLORE_BUDGET_FRAC", raising=False)
    base = RouterPolicy(strategy="always_frontier")
    assert apply_env_overrides(base) is base


def test_env_overrides_strategy_and_exploration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "always_cheap")
    monkeypatch.setenv("SHUNT_EXPLORATION_ENABLED", "0")
    monkeypatch.setenv("SHUNT_EXPLORE_BUDGET_FRAC", "0.25")
    out = apply_env_overrides(RouterPolicy())
    assert out.strategy == "always_cheap"
    assert out.exploration.enabled is False
    assert out.exploration.explore_budget_frac == pytest.approx(0.25)


def test_env_override_bad_strategy_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "oracle")
    with pytest.raises(ValidationError, match="unknown router.strategy"):
        apply_env_overrides(RouterPolicy())


def test_env_override_negative_budget_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_EXPLORE_BUDGET_FRAC", "-1")
    with pytest.raises(ValidationError):
        apply_env_overrides(RouterPolicy())


# ── Live-routable model selection (router.yaml `models:`) ────────────────────


def test_models_default_is_empty() -> None:
    assert RouterPolicy().models == []


def test_models_field_parses_a_list() -> None:
    p = RouterPolicy.model_validate({"strategy": "knn", "models": ["qwen3.7-plus", "kimi-k3"]})
    assert p.models == ["qwen3.7-plus", "kimi-k3"]


def test_models_extra_key_still_forbidden() -> None:
    with pytest.raises(ValidationError):
        RouterPolicy.model_validate({"strategy": "knn", "models": [], "bogus": 1})


def test_packaged_router_yaml_models_are_all_in_the_packaged_registry() -> None:
    # Guards against a typo shipping: a live-routable name that the registry
    # doesn't know about would only surface at ModelPool wiring time otherwise.
    from shunt.models.config import load_registry

    policy = load_router_policy(packaged_policy_path())
    registry = load_registry()
    assert policy.models, "packaged router.yaml must declare a non-empty models: list"
    unknown = [m for m in policy.models if m not in registry.models]
    assert not unknown, f"router.yaml models not in the registry: {unknown}"
