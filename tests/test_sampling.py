"""Tests for the deterministic, diversity-first run ordering (``runner.sampling``).

Pins what partial cost-safe runs depend on: repeat-determinism, nested prefixes,
breadth-first repo/stratum spread, growth-stability, and a real-manifest sanity check.
"""

from __future__ import annotations

import json

from benchmark import config
from benchmark.runner import sampling

_STRATA = ("easy", "medium", "hard")


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
