"""Tests for the arm-sampling gate — multi-arm sweep stays OFF by default."""

# The live executor does not yet send a distinct request per arm (see
# run_matrix's run_live_cells docstring) — sweeping arms before that lands
# would bill duplicate identical requests under fake arm labels.

from __future__ import annotations

from typing import Final

from benchmark import config
from benchmark.routing import integrity
from benchmark.runner import run_matrix

_TASKS: Final[tuple[str, ...]] = ("repo__task-1", "repo__task-2", "repo__task-3")


class TestArmSamplingGateDefault:
    def test_config_default_is_false(self):
        config.load("benchmark/config.yaml")
        assert config.arm_sampling_enabled() is False


class TestArmContextGate:
    def _models(self) -> list[str]:
        # A registry model with a declared multi-arm reasoning bracket exercises
        # the gate — a model with no bracket collapses to one arm either way.
        return [m for m, cfg in config.reasoning_configs().items() if cfg and len(cfg.arms) > 1]

    def test_gate_false_yields_only_default_arm_cells(self, monkeypatch):
        config.load("benchmark/config.yaml")
        models = self._models()
        assert models, "expected at least one multi-arm registry model to exercise the gate"
        monkeypatch.setitem(config.get(), "arm_sampling", {"enabled": False})

        selected, arm_hash_map = run_matrix._arm_context(_TASKS, models)
        status = run_matrix.classify_cells(_TASKS, models, {}, {}, {}, None, selected, arm_hash_map)

        # Exactly one arm per (cid, model): default-arm-only, byte-identical to
        # the pre-arm-sampling cell count.
        assert len(status.missing) == len(_TASKS) * len(models)
        defaults = config.default_arm_ids(models)
        for _cid, model, arm in status.missing:
            assert arm == defaults.get(model, integrity.DEFAULT_REASONING)

    def test_gate_true_yields_more_than_default_arm_cells(self, monkeypatch):
        config.load("benchmark/config.yaml")
        models = self._models()
        assert models
        monkeypatch.setitem(
            config.get(), "arm_sampling", {"enabled": True, "weights": [0.5, 0.35, 0.25]}
        )

        selected, arm_hash_map = run_matrix._arm_context(_TASKS, models)
        status = run_matrix.classify_cells(_TASKS, models, {}, {}, {}, None, selected, arm_hash_map)

        # At least one (cid, model) cell selected more than just its default arm.
        assert len(status.missing) > len(_TASKS) * len(models)

    def test_gate_false_matches_default_selected_arms_helper(self, monkeypatch):
        config.load("benchmark/config.yaml")
        models = self._models()
        monkeypatch.setitem(config.get(), "arm_sampling", {"enabled": False})
        selected, _ = run_matrix._arm_context(_TASKS, models)
        assert selected == run_matrix._default_selected_arms(_TASKS, models)
