# `compute_cost_decomposition` gated its per-task loop on pass/fail alone and never read
# `scorable` (index 7), so a censored `pass=True` cell entered the Oaxaca-Blinder
# decomposition while being excluded from the bootstrap, the scorable subset, and the
# cache-aware cost basis. It now routes through the SAME `_scorable_pair` predicate every
# other pairing uses, and `n_eq_pass` is the scorable equal-pass count.
"""The cost decomposition admits the SAME pairs the kill gate's other pairings do."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from benchmark import config
from benchmark.routing.metrics import _scorable_pair, compute_cost_decomposition
from benchmark.routing.strategies import knn
from benchmark.runner import kill_gate


def _d(
    tid: str, model: str, passed: bool, cost: float, *, scorable: bool = True
) -> kill_gate.Decision:
    """Build an 8-field Decision tuple (last field = scorable)."""
    return (tid, model, passed, cost, 100, 50, 1, scorable)


class TestDecompositionExcludesUnscorable:
    def test_censored_pass_true_pair_contributes_nothing(self):
        # A censored (unscorable) TEST arm recorded pass=True: it must not enter the
        # decomposition — the cell is not a real observation, so it cannot explain a saving.
        control = [_d("t1", "frontier", True, 10.0)]
        test = [_d("t1", "router", True, 2.0, scorable=False)]
        res = compute_cost_decomposition(control, test)
        assert res["n_eq_pass"] == 0
        assert res["total_direct_saving"] == 0.0
        assert res["price_savings"] == 0.0
        assert res["volume_savings"] == 0.0
        assert res["interaction"] == 0.0

    def test_censored_pass_true_in_control_arm_excluded(self):
        control = [_d("t1", "frontier", True, 10.0, scorable=False)]
        test = [_d("t1", "router", True, 2.0)]
        res = compute_cost_decomposition(control, test)
        assert res["n_eq_pass"] == 0
        assert res["total_direct_saving"] == 0.0

    def test_scorable_pass_pair_still_enters(self):
        control = [_d("t1", "frontier", True, 10.0)]
        test = [_d("t1", "router", True, 2.0)]
        res = compute_cost_decomposition(control, test)
        assert res["n_eq_pass"] == 1
        assert res["total_direct_saving"] == pytest.approx(8.0)

    def test_components_sum_to_direct_saving(self):
        # Oaxaca identity on a scorable pair with differing token counts: price + volume
        # + interaction must equal the direct saving exactly (identity pin).
        control = [("t1", "frontier", True, 30.0, 200, 100, 1, True)]  # 300 tok
        test = [("t1", "router", True, 5.0, 100, 50, 1, True)]  # 150 tok
        res = compute_cost_decomposition(control, test)
        assert res["n_eq_pass"] == 1
        assert res["total_direct_saving"] == pytest.approx(25.0)
        assert (res["price_savings"] + res["volume_savings"] + res["interaction"]) == pytest.approx(
            res["total_direct_saving"]
        )


class TestSharedScorablePredicate:
    def test_scorable_pair_resolves_to_one_function(self):
        # The decomposition and kill_gate's `_scorable_pair` are the SAME function — not
        # two hand-written restatements of the index that can drift apart.
        assert kill_gate._scorable_pair is _scorable_pair

    def test_decomposition_and_predicate_admit_same_pairs(self):
        # On one fixture: the pairs the decomposition counts are exactly the pairs the
        # shared predicate admits AND that pass in both arms.
        control = [
            _d("t1", "f", True, 1.0),
            _d("t2", "f", True, 2.0, scorable=False),
            _d("t3", "f", False, 3.0),
            _d("t4", "f", True, 4.0),
        ]
        test = [
            _d("t1", "r", True, 0.5),
            _d("t2", "r", True, 1.0, scorable=False),
            _d("t3", "r", False, 0.2),
            _d("t4", "r", False, 0.1),  # scorable but FAILS in the test arm
        ]
        admitted = [
            cd[0]
            for cd, td in zip(control, test, strict=True)
            if kill_gate._scorable_pair(cd, td) and cd[2] and td[2]
        ]
        assert admitted == ["t1"]
        res = compute_cost_decomposition(control, test)
        assert res["n_eq_pass"] == len(admitted)


def _fake_embed_texts(texts: list[str]) -> np.ndarray:
    """Deterministic pseudo-embedding (sha256 -> 32-dim), mirroring test_kill_gate.py."""
    dim = 32
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = digest * 4
        out[i] = np.frombuffer(raw, dtype=np.float32)
    return out


class TestDecompositionMatchesBootstrapOnCorpus:
    def test_n_eq_pass_equals_bootstrap_n_eq_on_committed_corpus(self, monkeypatch):
        # `n_eq_pass` reported by the decomposition must equal the count of scorable
        # equal-pass pairs the bootstrap uses, on the committed corpus fixture.
        if not config.results_csv_path().exists():
            pytest.skip("no committed results.csv in this checkout")
        monkeypatch.setattr(knn, "_embed_texts", _fake_embed_texts)
        config.load("benchmark/benchmark.yaml")
        matrix = config.load_matrix()
        frontier = config.frontier_model()
        all_tasks = sorted(matrix.get("results", {}).keys())
        sample = config.sample_tasks(all_tasks, seed=42)
        task_ids = kill_gate.select_tasks({"results": {t: {} for t in sample}}, 20, seed=42)
        control = kill_gate.evaluate_control(matrix, task_ids, frontier)
        router = kill_gate.evaluate_router(matrix, task_ids)

        dec = compute_cost_decomposition(control, router)
        cost_delta = kill_gate.bootstrap_cost_delta(control, router, n_iterations=200)

        assert dec["n_eq_pass"] == cost_delta["n_eq"]
