# The resamples were drawn from the module-level unseeded RNG, so the published
# intervals were irreproducible run to run; and `oracle_groups` was a
# `defaultdict(list)`, so a strategy task absent from the oracle silently scored as an
# empty group instead of raising. Seeding matches kill_gate.py's bootstraps (42).
"""bootstrap_ci is seeded and loud about oracle coverage gaps."""

from __future__ import annotations

import inspect
import random

import pytest

from benchmark.routing.metrics import bootstrap_ci


def _decisions(n: int = 30) -> list[tuple[str, str, bool, float]]:
    """Rich per-task variance (cost == task index, passes every third task)."""
    return [(f"t{i}", "m", i % 3 == 0, float(i)) for i in range(n)]


class TestBootstrapSeed:
    def test_bootstrap_same_seed_is_bit_identical(self):
        a = bootstrap_ci(_decisions(), _decisions(), n_bootstrap=200, seed=42)
        b = bootstrap_ci(_decisions(), _decisions(), n_bootstrap=200, seed=42)
        assert a == b
        assert a.avgperf == b.avgperf
        assert a.cumreg == b.cumreg
        assert a.total_cost == b.total_cost
        assert a.avg_cost == b.avg_cost

    def test_bootstrap_different_seeds_differ(self):
        a = bootstrap_ci(_decisions(), _decisions(), n_bootstrap=200, seed=42)
        b = bootstrap_ci(_decisions(), _decisions(), n_bootstrap=200, seed=7)
        assert a != b

    def test_bootstrap_ignores_module_level_rng(self, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise AssertionError("module-level random.choices was used")

        monkeypatch.setattr(random, "choices", _boom)
        cis = bootstrap_ci(_decisions(), _decisions(), n_bootstrap=50, seed=42)
        assert cis.avgperf[0] <= cis.avgperf[1]
        assert cis.total_cost[0] <= cis.total_cost[1]

    def test_bootstrap_default_seed_is_42(self):
        assert inspect.signature(bootstrap_ci).parameters["seed"].default == 42
        decisions = _decisions()
        default = bootstrap_ci(decisions, decisions, n_bootstrap=100)
        explicit = bootstrap_ci(decisions, decisions, n_bootstrap=100, seed=42)
        assert default == explicit


class TestOracleCoverage:
    def test_oracle_missing_task_raises_instead_of_empty_group(self):
        strategy = [("t1", "m", True, 10.0), ("t2", "m", True, 20.0)]
        oracle = [("t1", "m", True, 10.0)]
        with pytest.raises(ValueError, match="oracle"):
            bootstrap_ci(strategy, oracle, n_bootstrap=10)

    def test_oracle_missing_task_names_the_gap(self):
        strategy = [("t1", "m", True, 10.0), ("t3", "m", False, 20.0), ("t2", "m", True, 15.0)]
        oracle = [("t1", "m", True, 10.0)]
        with pytest.raises(ValueError, match="t2.*t3"):
            bootstrap_ci(strategy, oracle, n_bootstrap=10)

    def test_oracle_full_coverage_still_succeeds(self):
        decisions = _decisions(8)
        cis = bootstrap_ci(decisions, decisions, n_bootstrap=50, seed=42)
        assert cis.total_cost[0] <= cis.total_cost[1]
