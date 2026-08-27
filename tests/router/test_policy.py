from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from shunt.router.policy import (
    LIVE_STRATEGIES,
    BudgetPolicy,
    CapturePolicy,
    EscalationPolicy,
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
    assert p.strategy == "session_cascade"
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


@pytest.mark.parametrize("name", ["knn_cascade_withintask", "price_cascade"])
def test_within_task_cascades_are_not_live_eligible(name: str) -> None:
    # Both are WITHIN-TASK cascades: they take a verified outcome per attempt mid-session,
    # which is not one cache-safe per-session decision. The exclusion is permanent and by
    # design, so this is a wall, not a not-yet. The cache-safe analogues are the two
    # session-cadence presets, `knn_cascade` and `session_cascade`.
    assert name not in LIVE_STRATEGIES
    with pytest.raises(ValidationError, match="unknown router.strategy"):
        RouterPolicy(strategy=name)


def test_session_cascade_is_live_eligible() -> None:
    # The cache-safe operating point the two blocked rows approximate: same ladder, paced one
    # decision per SESSION. It is nameable precisely because it never switches mid-turn.
    assert "session_cascade" in LIVE_STRATEGIES
    policy = RouterPolicy(strategy="session_cascade")
    assert policy.strategy == "session_cascade"
    assert policy.escalation.enabled is True


def test_session_cascade_without_escalation_is_a_load_error() -> None:
    # The whole content of the name is always_cheap + the ladder. With the ladder off the
    # config resolves to a fixed cheap router wearing a cascade's name — the deployability-
    # honesty defect this id exists to remove, so it is refused rather than warned about.
    # Contrast test_escalation_enabled_default_without_a_repo_is_not_a_load_error: that state
    # is the common default and must not brick a boot; this one is only ever explicit.
    with pytest.raises(ValidationError, match="requires router.escalation.enabled"):
        RouterPolicy(strategy="session_cascade", escalation=EscalationPolicy(enabled=False))


def test_session_cascade_in_a_minimal_user_file_is_refused_with_the_reason() -> None:
    # The REAL boot path, not the constructor: `parse_router_policy` reads an absent
    # `escalation:` block as OFF (a user file replaces the packaged one wholesale), so the
    # most natural hand-written config for this preset is exactly the one that fails. That is
    # deliberate — silently turning the ladder on would make the shim mean two things — but
    # the error has to say WHY, or it reads as a contradiction of the documented example.
    with pytest.raises(ValidationError, match="no `escalation:` block"):
        parse_router_policy({"router": {"strategy": "session_cascade"}})


def test_session_cascade_with_an_explicit_escalation_block_parses() -> None:
    policy = parse_router_policy(
        {"router": {"strategy": "session_cascade", "escalation": {"enabled": True}}}
    )
    assert policy.strategy == "session_cascade"
    assert policy.escalation.enabled is True


def test_always_cheap_without_escalation_is_still_fine() -> None:
    # The coherence rule is scoped to the preset. Turning the ladder off under `always_cheap`
    # is a legitimate configuration and must stay one.
    policy = RouterPolicy(strategy="always_cheap", escalation=EscalationPolicy(enabled=False))
    assert policy.escalation.enabled is False


def test_escalation_enabled_default_without_a_repo_is_not_a_load_error() -> None:
    # Escalation ships ON; "enabled without a work_dir" is the common default state (a plain
    # install has no repo to test), so it must NOT be a load error — the router would fail to
    # boot for every install. The never-silently-inert guarantee moved to a boot WARNING
    # (server._log_capture_disclosure). A hard load error returns only if escalation gains a
    # second verified-failure signal that needs no repo.
    assert RouterPolicy().escalation.enabled is True  # ships ON
    RouterPolicy()  # no ValidationError — a plain default config loads


def test_escalation_enabled_with_a_work_dir_or_map_is_accepted() -> None:
    assert (
        RouterPolicy(
            escalation=EscalationPolicy(enabled=True),
            capture=CapturePolicy(work_dir="/srv/repo"),
        ).escalation.enabled
        is True
    )
    assert (
        RouterPolicy(
            escalation=EscalationPolicy(enabled=True),
            capture=CapturePolicy(work_dirs={"toolA": "/srv/repo"}),
        ).escalation.enabled
        is True
    )


def test_escalation_can_still_be_explicitly_disabled() -> None:
    # Turning it off remains a supported, documented config — on a strategy that is not a
    # cascade preset, where the coherence rule refuses it (see the cascade tests above).
    policy = RouterPolicy(strategy="always_cheap", escalation=EscalationPolicy(enabled=False))
    assert policy.escalation.enabled is False


@pytest.mark.parametrize("name", LIVE_STRATEGIES)
def test_all_live_strategies_accepted(name: str) -> None:
    assert RouterPolicy(strategy=name).strategy == name


def test_extra_key_forbidden() -> None:
    with pytest.raises(ValidationError):
        RouterPolicy.model_validate({"strategy": "knn_semantic_cascade", "bogus": 1})


def test_parse_router_policy_unwraps_router_key() -> None:
    data = {"router": {"strategy": "always_cheap", "exploration": {"enabled": False}}}
    p = parse_router_policy(data)
    assert p.strategy == "always_cheap"
    assert p.exploration.enabled is False


def test_parse_router_policy_empty_is_defaults() -> None:
    assert parse_router_policy(None) == RouterPolicy()
    assert parse_router_policy({}) == RouterPolicy()


def test_parse_router_policy_absent_escalation_block_is_off() -> None:
    # ESCAPE HATCH: a user config that predates the `escalation:` block (no key at all)
    # must stay OFF — it never had a chance to opt in, and the wholesale-replacement
    # policy means it never saw the shipped `enabled: true`. An absent key is an old
    # config, not an opt-in.
    old_config = {"router": {"strategy": "always_cheap", "exploration": {"enabled": False}}}
    assert parse_router_policy(old_config).escalation.enabled is False


def test_parse_router_policy_explicit_escalation_values_win() -> None:
    assert (
        parse_router_policy({"router": {"escalation": {"enabled": True}}}).escalation.enabled
        is True
    )
    assert (
        parse_router_policy({"router": {"escalation": {"enabled": False}}}).escalation.enabled
        is False
    )


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
    p = RouterPolicy.model_validate(
        {"strategy": "knn_semantic_cascade", "models": ["qwen3.7-plus", "kimi-k3"]}
    )
    assert p.models == ["qwen3.7-plus", "kimi-k3"]


def test_models_extra_key_still_forbidden() -> None:
    with pytest.raises(ValidationError):
        RouterPolicy.model_validate({"strategy": "knn_semantic_cascade", "models": [], "bogus": 1})


def test_packaged_router_yaml_models_are_all_in_the_packaged_registry() -> None:
    # Guards against a typo shipping: a live-routable name that the registry
    # doesn't know about would only surface at ModelPool wiring time otherwise.
    from shunt.models.config import load_registry

    policy = load_router_policy(packaged_policy_path())
    registry = load_registry()
    assert policy.models, "packaged router.yaml must declare a non-empty models: list"
    unknown = [m for m in policy.models if m not in registry.models]
    assert not unknown, f"router.yaml models not in the registry: {unknown}"


# ── Per-session spend cap (router.budget.max_spend_usd) ─────────────────────


def test_budget_default_is_unlimited() -> None:
    # null = unlimited is the shipped default; an absent `budget:` block is NOT an
    # opt-in to a cap (unlike escalation), so an old config stays behaviour-identical.
    assert RouterPolicy().budget.max_spend_usd is None
    parsed = parse_router_policy(
        {"router": {"strategy": "knn_semantic_cascade", "escalation": {"enabled": True}}}
    )
    assert parsed.budget.max_spend_usd is None


def test_budget_key_is_accepted() -> None:
    p = RouterPolicy.model_validate(
        {"strategy": "knn_semantic_cascade", "budget": {"max_spend_usd": 0.5}}
    )
    assert p.budget.max_spend_usd == pytest.approx(0.5)


def test_budget_live_tier_block_parses() -> None:
    # The exact shape compose.live.yaml writes — this was REJECTED by extra="forbid"
    # before the field existed, making the documented live-tier cap unrunnable.
    p = parse_router_policy({"router": {"models": [], "budget": {"max_spend_usd": 1.0}}})
    assert p.budget.max_spend_usd == pytest.approx(1.0)


@pytest.mark.parametrize("value", [-0.01, -1.0])
def test_budget_rejects_negative_cap(value: float) -> None:
    with pytest.raises(ValidationError):
        BudgetPolicy.model_validate({"max_spend_usd": value})


def test_budget_unknown_key_still_forbidden() -> None:
    with pytest.raises(ValidationError):
        RouterPolicy.model_validate({"strategy": "knn_semantic_cascade", "budget": {"bogus": 1}})


class TestTheKnnCascadeRenameDoesNotBrickExistingConfigs:
    """`strategy: knn` (then `knn_cascade`) were the pre-rename spellings of
    `knn_semantic_cascade`; they must keep booting."""

    def test_the_default_is_a_cascade_but_not_the_knn_one(self) -> None:
        # The kNN pick has participated in escalation since escalation shipped ON, so `knn`
        # named something the router never did. The default says what it does — and it is the
        # CHEAP-START cascade, because that is the measured frontier row.
        assert RouterPolicy().strategy == "session_cascade"
        assert "knn" not in LIVE_STRATEGIES

    def test_a_pre_rename_strategy_is_aliased(self) -> None:
        assert parse_router_policy({"router": {"strategy": "knn"}}).strategy == (
            "knn_semantic_cascade"
        )
        assert parse_router_policy({"router": {"strategy": "knn_cascade"}}).strategy == (
            "knn_semantic_cascade"
        )

    def test_a_pre_rename_config_with_no_escalation_block_still_boots(self) -> None:
        # THE BRICK CASE. The absent-block escape hatch resolves to OFF, and a cascade id with
        # the ladder off is a load error — so without the carve-out below, every pre-escalation
        # config in the world would stop booting on upgrade.
        policy = parse_router_policy({"router": {"strategy": "knn"}})
        assert policy.escalation.enabled is True

    def test_the_alias_never_resolves_to_the_new_default(self) -> None:
        # `strategy: knn` means kNN ROUTING. The shipped default moved to `session_cascade`,
        # which does not consult the neighbourhood at all — resolving the alias there would
        # silently migrate every pre-rename install off the routing model, which is a
        # data-losing migration wearing a rename's clothes.
        for raw in (
            {"router": {"strategy": "knn"}},
            {"router": {"strategy": "knn", "escalation": {"enabled": True}}},
            {"router": {"strategy": "knn_cascade", "escalation": {"enabled": True}}},
        ):
            assert parse_router_policy(raw).strategy == "knn_semantic_cascade"

    def test_a_config_with_no_strategy_and_no_escalation_block_still_boots(self) -> None:
        policy = parse_router_policy({"router": {"policy": {"k": 5}}})
        assert policy.strategy == "session_cascade"
        assert policy.escalation.enabled is True

    def test_the_defaulted_cascade_with_the_ladder_off_warns_and_boots(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The default was ALREADY a cascade id before it moved, so this branch is not newly
        # reachable — but it is the commonest way to turn escalation off (name nothing, write
        # `escalation: {enabled: false}`), and it must stay a warning rather than a load error
        # whichever cascade the default names.
        with caplog.at_level("WARNING"):
            policy = parse_router_policy({"router": {"escalation": {"enabled": False}}})
        assert policy.strategy == "session_cascade"
        assert policy.escalation.enabled is False
        assert any("defaults to" in r.getMessage() for r in caplog.records)

    def test_an_explicit_default_cascade_with_the_ladder_off_is_a_load_error(self) -> None:
        with pytest.raises(ValidationError, match="requires router.escalation.enabled"):
            parse_router_policy(
                {"router": {"strategy": "session_cascade", "escalation": {"enabled": False}}}
            )

    def test_the_escape_hatch_still_holds_for_an_explicit_strategy(self) -> None:
        # The carve-out is narrow: only a DEFAULTED or ALIASED strategy resolves the absent
        # block to ON. A file that names its own strategy keeps the old reading.
        policy = parse_router_policy({"router": {"strategy": "always_cheap"}})
        assert policy.escalation.enabled is False

    def test_an_explicit_cascade_with_the_ladder_off_is_still_a_load_error(self) -> None:
        with pytest.raises(ValidationError, match="requires router.escalation.enabled"):
            parse_router_policy(
                {"router": {"strategy": "knn_semantic_cascade", "escalation": {"enabled": False}}}
            )

    def test_the_env_override_is_aliased_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An operator exporting the pre-rename SHUNT_ROUTER_STRATEGY must not be refused at
        # boot by a rename they never saw.
        monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "knn")
        assert apply_env_overrides(RouterPolicy()).strategy == "knn_semantic_cascade"
