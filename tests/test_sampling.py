"""Tests for the deterministic, diversity-first run ordering (``runner.sampling``).

Pins what partial cost-safe runs depend on: repeat-determinism, nested prefixes,
breadth-first repo/stratum spread, growth-stability, and a real-manifest sanity check.
"""

from __future__ import annotations

import json

from benchmark import config
from benchmark.runner import sampling
from shunt.models.config import ReasoningArm, ReasoningConfig

_STRATA = ("easy", "medium", "hard")


def _bracket(default_arm: str, *ranked_ids: str) -> ReasoningConfig:
    """A synthetic bracket: arms ranked 0..n in the given id order."""
    arms = [ReasoningArm(id=i, rank=r, api={}) for r, i in enumerate(ranked_ids)]
    return ReasoningConfig(default_arm=default_arm, arms=arms)


def _triples(repos: int, per_cell: int) -> list[tuple[str, str, str]]:
    """Synthetic ``(id, repo, stratum)`` set: every repo populated in every stratum."""
    out: list[tuple[str, str, str]] = []
    for r in range(repos):
        repo = f"org/repo{r:02d}"
        for stratum in _STRATA:
            for k in range(per_cell):
                out.append((f"repo{r:02d}__{stratum}-{k}", repo, stratum))
    return out


class TestDeterminism:
    def test_repeated_calls_are_identical(self):
        items = _triples(repos=6, per_cell=3)
        assert sampling.stratified_order(items) == sampling.stratified_order(items)


class TestNesting:
    def test_growing_sample_size_only_appends(self, monkeypatch):
        # The operational guarantee (cost-safety): raising sample_size ADDS tasks,
        # never reshuffles, so cached results.csv cells are reused. Exercised through
        # the real config.sample_tasks path over the live 500-task manifest.
        manifest = json.loads(config.challenges_path().read_text())
        ids = sorted(manifest["tasks"])
        config.load()

        def run(n: int) -> list[str]:
            monkeypatch.setitem(config.get()["benchmark"], "sample_size", n)
            return config.sample_tasks(ids)

        s10, s20, s200 = run(10), run(20), run(200)
        assert (len(s10), len(s20), len(s200)) == (10, 20, 200)
        assert s20[:10] == s10  # nesting: 10 ⊂ 20
        assert s200[:20] == s20  # nesting: 20 ⊂ 200
        # The stratified path is active (not the fallback shuffle): the first pass
        # spreads across many repos, which a seeded shuffle would not guarantee.
        assert len({manifest["tasks"][i]["repo"] for i in s10}) >= 8


class TestDiversity:
    def test_first_pass_hits_every_repo(self):
        order = sampling.stratified_order(_triples(repos=12, per_cell=2))
        first12_repos = {iid.split("__")[0] for iid in order[:12]}
        assert len(first12_repos) == 12

    def test_early_order_spreads_across_strata(self):
        order = sampling.stratified_order(_triples(repos=12, per_cell=2))
        early_strata = {iid.split("__")[1].split("-")[0] for iid in order[:24]}
        assert len(early_strata) >= 2
        all_strata = {iid.split("__")[1].split("-")[0] for iid in order}
        assert set(_STRATA) <= all_strata


class TestGrowthStability:
    def test_untouched_repos_relative_order_unchanged(self):
        base = _triples(repos=5, per_cell=3)
        before = sampling.stratified_order(base)
        grown = base + _triples(repos=3, per_cell=4)  # adds repo00..repo02 rows
        after = sampling.stratified_order(grown)
        # repo04 is untouched by the growth; its ids must keep their relative order.
        sub_before = [i for i in before if i.startswith("repo04__")]
        sub_after = [i for i in after if i.startswith("repo04__")]
        assert sub_before == sub_after


class TestManifestFallback:
    def test_returns_none_when_a_task_lacks_repo(self):
        manifest = {"tasks": {"a__b-1": {"difficulty_stratum": "easy"}}}
        assert sampling.order_from_manifest(["a__b-1"], manifest) is None

    def test_orders_when_repo_present(self):
        manifest = {"tasks": {"a__b-1": {"repo": "a/b", "difficulty_stratum": "easy"}}}
        assert sampling.order_from_manifest(["a__b-1"], manifest) == ["a__b-1"]


class TestRealManifest:
    def test_orders_all_ids_as_a_diverse_permutation(self):
        manifest = json.loads(config.challenges_path().read_text())
        ids = sorted(manifest["tasks"])
        order = sampling.order_from_manifest(ids, manifest)
        assert order is not None
        assert sorted(order) == ids  # a permutation — no drops, no dupes
        assert len(order) == len(ids)
        early_repos = {manifest["tasks"][iid]["repo"] for iid in order[:24]}
        assert len(early_repos) >= 8


# ---------------------------------------------------------------------------
# select_arms: p(arm|model) exploration sampling.
# ---------------------------------------------------------------------------


class TestSelectArms:
    def test_default_arm_always_included(self):
        bracket = _bracket("high", "none", "high", "max")
        weights = [0.0, 0.0, 0.0]  # every non-default arm's weight is zero
        for cid in ("c1", "c2", "c3"):
            assert "high" in sampling.select_arms(cid, "m", bracket, weights)

    def test_determinism_same_inputs_same_arms(self):
        bracket = _bracket("high", "none", "high", "max")
        weights = [0.5, 0.35, 0.25]
        a = sampling.select_arms("astropy__astropy-1", "deepseek-v4-flash", bracket, weights)
        b = sampling.select_arms("astropy__astropy-1", "deepseek-v4-flash", bracket, weights)
        assert a == b

    def test_rerun_selects_identical_arms_cache_stable(self):
        # A re-run over many challenges must select byte-identical arm sets.
        bracket = _bracket("high", "none", "high", "max")
        weights = [0.5, 0.35, 0.25]
        ids = [f"repo/x-{i}" for i in range(100)]
        model = "deepseek-v4-flash"
        first = {cid: sampling.select_arms(cid, model, bracket, weights) for cid in ids}
        second = {cid: sampling.select_arms(cid, model, bracket, weights) for cid in ids}
        assert first == second

    def test_weight_one_always_selects_arm(self):
        bracket = _bracket("high", "none", "high", "max")
        weights = [1.0, 0.0, 0.0]
        for cid in ("c1", "c2", "c3", "c4"):
            assert "none" in sampling.select_arms(cid, "m", bracket, weights)

    def test_weight_zero_never_selects_arm(self):
        bracket = _bracket("high", "none", "high", "max")
        weights = [0.0, 0.0, 0.0]
        for cid in ("c1", "c2", "c3", "c4"):
            arms = sampling.select_arms(cid, "m", bracket, weights)
            assert "none" not in arms
            assert "max" not in arms
            assert arms == ["high"]

    def test_single_arm_bracket_collapses_to_default_only(self):
        # kimi-k3 shape: one arm, which is also the default.
        bracket = _bracket("max", "max")
        assert sampling.select_arms("c1", "kimi-k3", bracket, [0.5]) == ["max"]

    def test_cheaper_rank_selected_on_more_challenges_than_pricier_rank(self):
        # Cost-skew: over a sample of ids, the cheap (rank 0, w=0.5) arm should be
        # selected on roughly twice as many challenges as the max (rank 2, w=0.25) arm.
        bracket = _bracket("high", "none", "high", "max")
        weights = [0.5, 0.35, 0.25]
        ids = [f"repo/task-{i}" for i in range(2000)]
        model = "deepseek-v4-flash"
        drawn = [sampling.select_arms(cid, model, bracket, weights) for cid in ids]
        none_count = sum(1 for arms in drawn if "none" in arms)
        max_count = sum(1 for arms in drawn if "max" in arms)
        assert none_count > max_count
        # Loose bounds around the declared 0.5 / 0.25 fractions (large-sample tolerance).
        assert 0.4 < none_count / len(ids) < 0.6
        assert 0.15 < max_count / len(ids) < 0.35

    def test_different_models_get_independent_draws(self):
        # Same challenge, same bracket, different model name -> the salt differs,
        # so the two models need not agree on whether a non-default arm is in.
        bracket = _bracket("high", "none", "high", "max")
        weights = [0.5, 0.35, 0.25]
        a = sampling.select_arms("c1", "model-a", bracket, weights)
        b = sampling.select_arms("c1", "model-b", bracket, weights)
        # Not asserting inequality (could coincide) — just that both are valid,
        # well-formed arm subsets that always include the default.
        assert "high" in a and "high" in b
        assert set(a) <= {"none", "high", "max"}
        assert set(b) <= {"none", "high", "max"}

    def test_real_registry_brackets_resolve(self):
        # Sanity check against the real registry: every enabled model's declared
        # bracket selects at least its default arm on a fixed challenge id.
        config.load("benchmark/config.yaml")
        cfgs = config.reasoning_configs()
        weights = config.arm_sampling_weights()
        for model in config.enabled_models():
            bracket = cfgs[model]
            arms = sampling.select_arms("astropy__astropy-13453", model, bracket, weights)
            assert bracket.default_arm in arms
            assert set(arms) <= {a.id for a in bracket.arms}
