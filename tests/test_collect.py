"""Adaptive collect-mode tests: strata derivation, determinism, and a simulated run.

No live/paid path is exercised; the frontier selection is a pure function
of the cache + pinned constants, so it is checked deterministically.
"""

from __future__ import annotations

from typing import Final

from benchmark import config
from benchmark.runner import collect
from benchmark.runner.sampling import in_frontier_audit

_MODELS_A: Final = ("deepseek-v4-flash", "gpt-5-mini")


def _cache(defaults: dict[str, str]) -> dict:
    """Synthetic cheap+mid cache: t_disc discriminates, two uncontested, one uncovered."""
    da, gp = defaults["deepseek-v4-flash"], defaults["gpt-5-mini"]

    def task(cheap: bool, mid: bool) -> dict:
        return {"deepseek-v4-flash": {da: {"pass": cheap}}, "gpt-5-mini": {gp: {"pass": mid}}}

    return {
        "t_disc": task(True, False),
        "t_allpass": task(True, True),
        "t_allfail": task(False, False),
        "t_uncov": {"deepseek-v4-flash": {da: {"pass": True}}},  # mid missing → uncovered
    }


class TestPhaseModels:
    def setup_method(self):
        config.load("benchmark/config.yaml")

    def test_single_picks_one_representative_per_tier(self):
        models = collect.phase_a_models("single")
        assert models == ["deepseek-v4-flash", "gpt-5-mini"]

    def test_full_includes_every_cheap_and_mid_model(self):
        models = collect.phase_a_models("full")
        assert set(models) == {"deepseek-v4-flash", "qwen3.7-plus", "gpt-5-mini", "kimi-k2.5"}

    def test_frontier_defaults_to_control_model(self):
        assert collect.frontier_models(include_high=False) == ["kimi-k3"]

    def test_include_high_adds_the_high_tier(self):
        assert collect.frontier_models(include_high=True) == ["kimi-k3", "zai-glm-5.2"]


class TestDeriveStrata:
    def setup_method(self):
        config.load("benchmark/config.yaml")
        self.defaults = config.default_arm_ids(_MODELS_A)
        self.cache = _cache(self.defaults)
        self.tasks = ["t_disc", "t_allpass", "t_allfail", "t_uncov"]

    def test_full_audit_selects_discriminating_plus_all_uncontested(self):
        d, a = collect.derive_strata(self.cache, self.tasks, _MODELS_A, 1.0, "frontier-audit-v1")
        assert d == ["t_disc"]
        assert set(a) == {"t_allpass", "t_allfail"}  # t_uncov excluded (never fully covered)

    def test_zero_audit_selects_only_the_discriminating_set(self):
        d, a = collect.derive_strata(self.cache, self.tasks, _MODELS_A, 0.0, "frontier-audit-v1")
        assert d == ["t_disc"]
        assert a == []

    def test_audit_membership_matches_the_salted_hash_draw(self):
        _d, a = collect.derive_strata(self.cache, self.tasks, _MODELS_A, 0.5, "frontier-audit-v1")
        expected = [
            t for t in ("t_allpass", "t_allfail") if in_frontier_audit(t, 0.5, "frontier-audit-v1")
        ]
        assert a == expected

    def test_deterministic_across_calls(self):
        first = collect.derive_strata(self.cache, self.tasks, _MODELS_A, 0.3, "frontier-audit-v1")
        second = collect.derive_strata(self.cache, self.tasks, _MODELS_A, 0.3, "frontier-audit-v1")
        assert first == second

    def test_raising_audit_fraction_only_adds_frontier_tasks(self):
        # Nested (churn-free) escalation over a larger synthetic uncontested pool.
        defaults = self.defaults
        da, gp = defaults["deepseek-v4-flash"], defaults["gpt-5-mini"]
        cache = {
            f"u{i}": {"deepseek-v4-flash": {da: {"pass": True}}, "gpt-5-mini": {gp: {"pass": True}}}
            for i in range(300)
        }
        tasks = list(cache)
        _d1, a_small = collect.derive_strata(cache, tasks, _MODELS_A, 0.2, "frontier-audit-v1")
        _d2, a_large = collect.derive_strata(cache, tasks, _MODELS_A, 0.4, "frontier-audit-v1")
        assert set(a_small) <= set(a_large)

    def test_canonical_task_order_is_preserved(self):
        d, a = collect.derive_strata(self.cache, self.tasks, _MODELS_A, 1.0, "frontier-audit-v1")
        frontier = [t for t in self.tasks if t in set(d) | set(a)]
        # Both strata keep the input task ordering (challenge-major prefix safety).
        assert frontier == [t for t in self.tasks if t in ("t_disc", "t_allpass", "t_allfail")]


class TestSimulatedRun:
    def test_end_to_end_simulated_writes_manifest_and_no_spend(self, tmp_path, monkeypatch):
        import json

        manifest = tmp_path / "collect_manifest.json"
        monkeypatch.setattr(collect, "_MANIFEST_PATH", manifest)
        # Simulated run against the real registry + challenge store: no keys ⇒ no spend.
        rc = collect.run_collect("benchmark/config.yaml", live=False)
        assert rc == 0
        assert manifest.exists()
        data = json.loads(manifest.read_text())
        assert data["audit_salt"] == "frontier-audit-v1"
        assert data["frontier_models"] == ["kimi-k3"]
        assert data["n_tasks"] > 0

    def test_refuses_when_audit_salt_aliases_calibration(self, monkeypatch):
        monkeypatch.setattr(
            config, "collect_config", lambda: {"audit_salt": "calib-v1", "audit_fraction": 0.2}
        )
        assert collect.run_collect("benchmark/config.yaml", live=False) == 2

    def test_refuses_live_when_constants_unpinned(self, monkeypatch):
        # --live must not spend real money with placeholder audit_fraction / margin.
        monkeypatch.setattr(collect, "_has_keys", lambda: True)
        monkeypatch.setattr(
            config,
            "collect_config",
            lambda: {
                "audit_salt": "frontier-audit-v1",
                "audit_fraction": 0.2,
                "constants_pinned": False,
            },
        )
        assert collect.run_collect("benchmark/config.yaml", live=True) == 2


class TestFrontierGate:
    def test_gate_estimates_from_a_synthetic_matrix(self):
        from benchmark.routing.run_eval import compute_frontier_gate

        # 3 discriminating (frontier observed, π=1) + 2 audit (π=0.5); covariate = cheap proxy.
        covariate = {"d0": 0.5, "d1": 0.5, "d2": 0.5, "a0": 1.0, "a1": 1.0, "u0": 1.0}
        frontier_pass = {"d0": 1, "d1": 0, "d2": 1, "a0": 1, "a1": 0}
        discriminating = {"d0", "d1", "d2"}
        audit_ids = {"a0", "a1"}
        router_pass = {"d0": 1, "d1": 1, "d2": 0}
        baseline_pass = {"d0": 1, "d1": 0, "d2": 1}
        gate = compute_frontier_gate(
            covariate,
            frontier_pass,
            discriminating,
            audit_ids,
            0.5,
            router_pass=router_pass,
            baseline_pass=baseline_pass,
            margin=0.05,
        )
        assert gate["n_labeled"] == 5
        assert 0.0 <= gate["q_f"].point <= 1.0
        assert gate["mcnemar"].b == 1 and gate["mcnemar"].c == 1  # d1 router-win, d2 router-loss
        assert gate["sequence"] is not None

    def test_gate_without_paired_outcomes_skips_mcnemar(self):
        from benchmark.routing.run_eval import compute_frontier_gate

        gate = compute_frontier_gate(
            {"d0": 0.5, "d1": 0.5}, {"d0": 1, "d1": 0}, {"d0", "d1"}, set(), 0.2
        )
        assert "mcnemar" not in gate
        assert "q_f" in gate
