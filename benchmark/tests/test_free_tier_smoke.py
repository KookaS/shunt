"""The free-tier smoke's gates — pure, hermetic, no spend, no network.

CI runs must refuse any billable model, require an explicit CI override for
non-TTY operation, and skip — never fail — on a missing key.
"""

from __future__ import annotations

import json
from pathlib import Path

import benchmark.runner.live_smoke as live_smoke
from benchmark.runner import free_tier_smoke
from shunt.models.config import ModelConfig, Pricing

FREE_TIER_DIR = Path(__file__).resolve().parents[2] / "configs" / "free-tier"


def _model(
    model_id: str = "openai/gpt-oss-20b:free",
    base_url: str = "https://openrouter.ai/api/v1",
    price: float = 0.0,
    provider: str = "openrouter",
) -> ModelConfig:
    """A ModelConfig shaped like the free-tier registry row."""
    return ModelConfig(
        name="gpt-oss-20b-free",
        model_id=model_id,
        provider=provider,
        base_url=base_url,
        api_key_env_var="OPENROUTER_API_KEY",
        litellm_prefix="openrouter",
        pricing=Pricing(
            input_cost_per_1m=price,
            output_cost_per_1m=price,
            price_provider="openrouter",
            price_source="https://openrouter.ai/models/x",
            price_as_of="2026-08-11",
        ),
    )


# ── the :free assertion (refuses before any network call) ─────────────────────


def test_free_tier_refusal_passes_for_free_model() -> None:
    assert free_tier_smoke.free_tier_refusal(_model()) is None


def test_free_tier_refusal_rejects_non_free_model_id() -> None:
    reason = free_tier_smoke.free_tier_refusal(_model(model_id="openai/gpt-4o"))
    assert reason is not None and ":free" in reason


def test_free_tier_refusal_rejects_non_openrouter_base_url() -> None:
    reason = free_tier_smoke.free_tier_refusal(_model(base_url="https://router.requesty.ai/v1"))
    assert reason is not None and "OpenRouter" in reason


def test_free_tier_refusal_rejects_positive_price() -> None:
    reason = free_tier_smoke.free_tier_refusal(_model(price=2.5))
    assert reason is not None and "price" in reason


def test_free_tier_refusal_rejects_missing_pricing() -> None:
    model = _model()
    model.pricing = None
    reason = free_tier_smoke.free_tier_refusal(model)
    assert reason is not None and "price" in reason


# ── the CI TTY override ───────────────────────────────────────────────────────


def test_ci_tty_refusal_allows_tty() -> None:
    assert free_tier_smoke.ci_tty_refusal(lambda: True, None) is None


def test_ci_tty_refusal_requires_override_for_non_tty() -> None:
    reason = free_tier_smoke.ci_tty_refusal(lambda: False, None)
    assert reason is not None and "SHUNT_FREE_TIER_CI" in reason
    assert free_tier_smoke.ci_tty_refusal(lambda: False, "0") is not None


def test_ci_tty_refusal_override_opens_non_tty() -> None:
    assert free_tier_smoke.ci_tty_refusal(lambda: False, "1") is None


# ── missing key → skip, never fail ────────────────────────────────────────────


def test_missing_key_skip() -> None:
    assert free_tier_smoke.missing_key_skip("OPENROUTER_API_KEY", None) is True
    assert free_tier_smoke.missing_key_skip("OPENROUTER_API_KEY", "") is True
    assert free_tier_smoke.missing_key_skip("OPENROUTER_API_KEY", "sk-live") is False


# ── run-dir stamping + verdict write ─────────────────────────────────────────


def test_two_run_dir_calls_in_the_same_second_produce_distinct_dirs(tmp_path: Path) -> None:
    # A 1-second stamp made two invocations within the same second collide on one run
    # dir, so a concurrent PASS and FAIL fought over a single directory. The stamp now
    # carries microseconds, so consecutive calls must land in distinct dirs.
    first, first_data, _first_stamp = free_tier_smoke._make_run_dir(tmp_path)
    second, second_data, _second_stamp = free_tier_smoke._make_run_dir(tmp_path)
    assert first != second
    assert first_data != second_data
    assert first.exists() and second.exists()


def test_verdict_write_is_atomic_and_leaves_no_temp_sibling(tmp_path: Path) -> None:
    # The verdict is written through a temp + os.replace, so a reader (or a concurrent
    # sibling run) sees the whole file or none of it — never a truncated half. Assert the
    # write is complete and no `.tmp` staging file survives.
    run_dir, _data, stamp = free_tier_smoke._make_run_dir(tmp_path)
    path = free_tier_smoke._write_verdict(run_dir, stamp, {"status": "PASS", "problems": []})
    assert json.loads(path.read_text())["status"] == "PASS"
    assert json.loads(path.read_text())["run"] == stamp
    assert [p.name for p in run_dir.iterdir() if p.suffix == ".tmp"] == []


# ── the committed free-tier config ────────────────────────────────────────────


def test_free_tier_config_loads_and_ranks_the_free_model_first() -> None:
    pool = live_smoke.build_pool(FREE_TIER_DIR / "models.yaml", FREE_TIER_DIR / "router.yaml")
    expected = live_smoke.cheapest_live_model(pool)
    assert pool.model_names() == ["gpt-oss-20b-free"]
    assert expected.model_id == "openai/gpt-oss-20b:free"
    assert expected.api_key_env_var == "OPENROUTER_API_KEY"
    assert free_tier_smoke.free_tier_refusal(expected) is None


def test_free_tier_config_contains_only_one_free_model() -> None:
    pool = live_smoke.build_pool(FREE_TIER_DIR / "models.yaml", FREE_TIER_DIR / "router.yaml")
    names = pool.model_names()
    assert len(names) == 1  # a paid or extra model here would change routing
    assert free_tier_smoke.free_tier_refusal(live_smoke.cheapest_live_model(pool)) is None


# ── main(): gate ordering, with the server boot provably never reached ────────


def test_main_skips_on_missing_key(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(live_smoke, "start_server", _fail_if_booted)

    exit_code = free_tier_smoke.main(
        ["--config-dir", str(FREE_TIER_DIR), "--run-dir", str(tmp_path)]
    )
    assert exit_code == 0  # skip, not fail
    assert "SKIP" in capsys.readouterr().err
    verdict_path = next(tmp_path.glob("*/free-tier-smoke.json"))
    verdict = json.loads(verdict_path.read_text())
    assert verdict["status"] == "SKIP"
    assert "OPENROUTER_API_KEY" in verdict["reason"]


def test_main_refuses_non_free_model_before_boot(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(live_smoke, "start_server", _fail_if_booted)
    cfg = _write_paid_config(tmp_path)
    assert free_tier_smoke.main(["--config-dir", str(cfg)]) == 2
    assert "does not end in ':free'" in capsys.readouterr().err


def _write_paid_config(tmp_path: Path) -> Path:
    """A temp config dir whose model is NOT an OpenRouter :free model."""
    cfg = tmp_path / "paid-config"
    cfg.mkdir()
    (cfg / "models.yaml").write_text(
        """
providers:
  openrouter:
    base_url: https://openrouter.ai/api/v1
    api_key_env_var: OPENROUTER_API_KEY
    litellm_prefix: openrouter
models:
  paid-model:
    model_id: openai/gpt-4o
    provider: openrouter
    version: paid-model
    pricing:
      input_cost_per_1m: 2.5
      output_cost_per_1m: 10.0
      price_provider: openrouter
      price_source: https://openrouter.ai/models/x
      price_as_of: "2026-08-11"
""".lstrip(),
        encoding="utf-8",
    )
    (cfg / "router.yaml").write_text(
        "router:\n  strategy: knn\n  models:\n    - paid-model\n", encoding="utf-8"
    )
    return cfg


def _fail_if_booted(*_args, **_kwargs):
    raise AssertionError("the smoke must refuse before any server boot / network call")


def test_free_tier_models_load_via_shunt_config_dir(monkeypatch, tmp_path) -> None:
    # SHUNT_CONFIG_DIR pointing at the free-tier dir resolves models.yaml for the
    # registry layer exactly as the smoke's server env drives it.
    monkeypatch.setenv("SHUNT_CONFIG_DIR", str(FREE_TIER_DIR))
    from shunt.models.config import ModelPool

    pool = ModelPool.load()
    assert pool.model_names() == ["gpt-oss-20b-free"]
    assert pool.ranked_models()[0].model_id == "openai/gpt-oss-20b:free"
