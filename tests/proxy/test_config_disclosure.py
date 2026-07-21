"""Startup must state the configuration in force — and must never print a credential."""

from __future__ import annotations

import logging

import pytest

from shunt.models.config import ModelPool
from shunt.proxy.server import _log_config_disclosure
from shunt.router.policy import RouterPolicy


@pytest.fixture
def disclosure(caplog: pytest.LogCaptureFixture) -> str:
    with caplog.at_level(logging.INFO, logger="shunt.proxy.server"):
        _log_config_disclosure(RouterPolicy(), ModelPool())
    return caplog.text


def test_it_states_the_selected_algorithm_and_its_knobs(disclosure: str) -> None:
    assert "strategy=knn" in disclosure
    assert "k=20" in disclosure
    assert "success_rate_threshold=0.60" in disclosure
    assert "min_samples=3" in disclosure


def test_it_states_the_exploration_settings(disclosure: str) -> None:
    assert "exploration: enabled=True" in disclosure
    assert "budget_frac=0.40" in disclosure
    assert "conservative_alpha=0.10" in disclosure


def test_it_lists_the_routable_models_with_their_tiers(disclosure: str) -> None:
    pool = ModelPool()
    assert pool.model_names(), "fixture pool must expose models for this to mean anything"
    for name in pool.model_names():
        tier = pool.get_tier(name)
        assert f"{tier}:{name}" in disclosure


def test_it_never_prints_a_credential(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "zz" + "Q7w" * 6
    # Set every env var the pool might name, so a naive dump of the environment fails.
    pool = ModelPool()
    env_vars = {
        model.api_key_env_var
        for name in pool.model_names()
        if (model := pool.get_model(name)) is not None
    }
    for name in env_vars | {"OPENAI_API_KEY"}:
        monkeypatch.setenv(name, secret)
    with caplog.at_level(logging.INFO, logger="shunt.proxy.server"):
        _log_config_disclosure(RouterPolicy(), ModelPool())
    assert secret not in caplog.text
