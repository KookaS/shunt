"""Tests for the report plot helpers — the phantom-frontier guard in particular."""

from __future__ import annotations

from benchmark import config
from benchmark.routing import report


class TestDisabledModelExcluded:
    def test_disabled_model_cannot_leak_via_stray_row(self):
        # A disabled model (opus) with a stray results row must never re-enter the
        # matrix — otherwise it silently re-promotes to frontier (opus $30 > k3 $18).
        config.load("benchmark/config.yaml")
        stray = {
            "t1": {
                "deepseek-v4-flash": {"pass": True, "cost": 0.01},
                "claude-opus-4-6": {"pass": True, "cost": 0.2},  # disabled in config
            }
        }
        assert "claude-opus-4-6" not in config.models_matrix(stray)


def _matrix(results: dict) -> dict:
    # Two models; opus is far more expensive so it is the "frontier" pick.
    return {
        "models": {
            "cheap": {"input_price": 0.1, "output_price": 0.2},
            "opus": {"input_price": 5.0, "output_price": 25.0},
        },
        "results": results,
    }


class TestFrontierCoverage:
    def test_flags_partial_frontier(self):
        # opus present on 1 of 3 tasks -> phantom baseline.
        m = _matrix(
            {
                "t1": {"cheap": {"pass": True, "cost": 0.01}, "opus": {"pass": True, "cost": 0.2}},
                "t2": {"cheap": {"pass": True, "cost": 0.01}},
                "t3": {"cheap": {"pass": False, "cost": 0.01}},
            }
        )
        assert report._frontier_coverage(m) == ("opus", 1, 3)

    def test_full_coverage_is_not_phantom(self):
        m = _matrix(
            {
                "t1": {"cheap": {"pass": True, "cost": 0.01}, "opus": {"pass": True, "cost": 0.2}},
                "t2": {"cheap": {"pass": True, "cost": 0.01}, "opus": {"pass": True, "cost": 0.2}},
            }
        )
        frontier, covered, total = report._frontier_coverage(m)
        assert (frontier, covered, total) == ("opus", 2, 2)

    def test_none_matrix_returns_none(self):
        assert report._frontier_coverage(None) is None
        assert report._frontier_coverage({"models": {}, "results": {}}) is None


class TestPlotsRender:
    def test_cost_savings_renders_with_phantom(self, tmp_path):
        rows = [
            {"strategy": "Always-Cheap", "TotalCost": "0.03", "AvgPerf%": "80"},
            {"strategy": "Always-Frontier", "TotalCost": "0.20", "AvgPerf%": "10"},
        ]
        m = _matrix(
            {
                "t1": {"cheap": {"pass": True, "cost": 0.01}, "opus": {"pass": True, "cost": 0.2}},
                "t2": {"cheap": {"pass": True, "cost": 0.01}},
            }
        )
        out = report.plot_cost_savings(rows, tmp_path, m)
        assert out.exists() and out.stat().st_size > 0

    def test_heatmap_renders_task_by_model(self, tmp_path):
        m = _matrix(
            {
                "proj__t1": {
                    "cheap": {"pass": True, "cost": 0.01},
                    "opus": {"pass": True, "cost": 0.2},
                },
                "proj__t2": {"cheap": {"pass": False, "cost": 0.01}},
            }
        )
        challenges = tmp_path / "challenges.json"
        challenges.write_text("{}")
        # plot_heatmap reloads via config.load_matrix; drive it through the public
        # path by monkeypatching load_matrix to return our in-memory matrix.
        orig = report.load_matrix
        report.load_matrix = lambda _p: m  # type: ignore[assignment]
        try:
            out = report.plot_heatmap(challenges, tmp_path)
        finally:
            report.load_matrix = orig  # type: ignore[assignment]
        assert out.exists() and out.stat().st_size > 0
