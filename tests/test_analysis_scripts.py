"""Guard tests for the routing analysis/reporting periphery — clean exit on an
empty results.csv, and report.py's regret plot including the shipped
kNN / kNN-cascade routers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pytest

from benchmark import config
from benchmark.routing.scripts import (
    compute_costs,
    embedding_compare,
    plot_strategies,
    threshold_sweep,
    viz_knn,
)

# Absolute config path so the test is independent of the process CWD.
CONFIG_PATH = str(Path(config.__file__).resolve().parent / "config.yaml")

# The analysis scripts that must exit cleanly on a header-only results.csv.
_GUARDED_SCRIPTS: Final = [
    compute_costs.main,
    embedding_compare.main,
    plot_strategies.main,
    threshold_sweep.main,
    viz_knn.main,
]


@pytest.mark.parametrize("script_main", _GUARDED_SCRIPTS, ids=lambda m: m.__module__)
def test_script_exits_cleanly_on_empty_matrix(script_main, monkeypatch, capsys, tmp_path):
    """Each analysis script must early-return with a message, not raise."""
    monkeypatch.setattr("sys.argv", ["prog"])
    monkeypatch.setattr(config, "results_csv_path", lambda: tmp_path / "empty.csv")
    # Must not raise (ZeroDivisionError / KeyError / ValueError before the fix).
    assert script_main(CONFIG_PATH) is None
    out = capsys.readouterr().out
    assert "No results yet" in out


class TestEmbeddingCompareNeighbors:
    """compute_overlap must not crash when fewer challenges than k are populated."""

    def test_fewer_tasks_than_k_does_not_crash(self):
        # 6 populated challenges, k=10 (the shipped k) — the partial-population state.
        feats = np.eye(6, dtype=float)
        neighbors = embedding_compare.compute_overlap(feats, k=10)
        # One row per task; width clamped to the available non-self neighbors.
        assert neighbors.shape[0] == 6
        assert 1 <= neighbors.shape[1] <= 5

    def test_single_task_does_not_crash(self):
        feats = np.ones((1, 4), dtype=float)
        neighbors = embedding_compare.compute_overlap(feats, k=10)
        assert neighbors.shape[0] == 1


class TestReportRegretFactories:
    """report.py must not silently drop the headline kNN routers."""

    # The old ImportError-degradation path this class used to also cover was
    # dropped: _build_strategy_factories now single-sources the enabled
    # strategy set from run_eval.get_strategies, and fastembed/hnswlib are
    # unconditional base dependencies (pyproject.toml [project.dependencies],
    # not a benchmark-only extra) — "embedding deps unavailable" is not a
    # reachable state to degrade for. See
    # test_report_plots.py::TestStrategyFactoriesMatchEnabledSet for the
    # config-enabled-set coverage the refactor introduced.

    def test_knn_strategies_present_in_factory_map(self):
        from benchmark.routing.report import _build_strategy_factories

        factories = _build_strategy_factories(gamma=0.1)
        # Keys must equal the strategies' .name so they match results.csv rows.
        assert "kNN" in factories
        assert "kNN-cascade" in factories
        assert factories["kNN"]().name == "kNN"
        assert factories["kNN-cascade"]().name == "kNN-cascade"
