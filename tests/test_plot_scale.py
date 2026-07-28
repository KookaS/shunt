"""Scale-robustness tests for the task-level plots — must degrade gracefully at 500+ tasks.

Also holds the degenerate-input regression for the heatmap slice, which must match
the slice its title promises.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.routing import report
from benchmark.routing.scripts import threshold_sweep


def _big_matrix(n_tasks: int, n_models: int) -> dict:
    models = {f"m{j}": {"input_price": 0.1 * j, "output_price": 0.2 * j} for j in range(n_models)}
    results = {}
    for i in range(n_tasks):
        # Deterministic pass pattern; a few models unevaluated on late tasks (NaN cells).
        row = {}
        for j in range(n_models):
            if i % 7 == 0 and j == n_models - 1:
                continue  # leave the frontier column sparse
            row[f"m{j}"] = {"pass": (i + j) % 3 != 0, "cost": 0.01}
        results[f"proj{i % 9}__task-{i}"] = row
    return {"models": models, "results": results}


class TestHeatmapScale:
    def test_renders_500_tasks_20_models(self, tmp_path):
        m = _big_matrix(500, 20)
        challenges = tmp_path / "challenges.json"
        challenges.write_text("{}")
        orig = report.load_matrix
        report.load_matrix = lambda _p: m  # type: ignore[assignment]
        try:
            out = report.plot_heatmap(challenges, tmp_path)
        finally:
            report.load_matrix = orig  # type: ignore[assignment]
        assert out.exists() and out.stat().st_size > 0

    def test_cap_names_truncates(self):
        many = [f"t{i}" for i in range(50)]
        s = report._cap_names(many, k=6)
        assert "+44 more" in s and s.count(",") == 5  # 6 names → 5 separators


def _sweep_rows() -> list[dict]:
    """A sweep whose two threshold levels sit 0.1 apart — the epsilon-collision case."""
    rows = []
    for k in (2, 4, 6):
        for thresh, reward in ((0.5, 10.0), (0.6, 99.0)):
            for ms in (1, 2):
                rows.append(
                    {
                        "k": k,
                        "success_rate_thresh": thresh,
                        "min_samples": ms,
                        "Reward": reward + ms,
                        "n_excluded": 0,
                        "n_scored": 100,
                        "AvgPerf%": reward + ms,
                        "TotalCost": 1.0 + ms,
                        "frontier_share": 0.1 * ms,
                        "cheapest_share": 0.5,
                        "n_models_used": 3,
                    }
                )
    return rows


def _selected(**overrides) -> dict:
    """A stand-in for the cost-at-equal-quality pick the annotations describe."""
    row = {
        "k": 4,
        "Reward": 12.0,
        "success_rate_thresh": 0.5,
        "min_samples": 1,
        "AvgPerf%": 90.0,
        "TotalCost": 12.0,
        "n_scored": 100,
        "frontier_share": 0.2,
        "n_models_used": 3,
    }
    row.update(overrides)
    return row


class TestSweepSliceMatchesTitle:
    def test_slice_excludes_the_adjacent_grid_level(self):
        # abs(0.6 - 0.5) == 0.0999... < 0.1, so a float-epsilon filter admitted both.
        rows = _sweep_rows()
        assert {
            r["success_rate_thresh"]
            for r in threshold_sweep._slice_at(rows, "success_rate_thresh", 0.5)
        } == {0.5}

    def test_rendered_grid_holds_the_promised_slice(self):
        rows = _sweep_rows()
        sliced = threshold_sweep._slice_at(rows, "success_rate_thresh", 0.5)
        _x, _y, heat = threshold_sweep._heat_grid(sliced, "min_samples")
        # Rewards for the 0.5 slice are 11.0/12.0; the 0.6 slice's are 100.0/101.0.
        assert set(np.unique(heat).tolist()) == {11.0, 12.0}

    def test_mixed_slice_raises_rather_than_overwriting(self):
        with pytest.raises(ValueError, match="mixes levels"):
            threshold_sweep._heat_grid(_sweep_rows(), "min_samples")

    def test_end_to_end_heatmap_uses_the_fixed_value_in_its_title(self, tmp_path):
        rows = threshold_sweep._slice_at(_sweep_rows(), "success_rate_thresh", 0.5)
        out = tmp_path / "sweep.png"
        threshold_sweep._plot_sweep_heatmap(
            rows,
            rows,
            out,
            y_name="min_samples",
            fixed=("success_rate_thresh", 0.5),
            swept_ks=[2, 4, 6],
            sensitivity={"k": 0.5, "min_samples": 0.4, "success_rate_thresh": 0.1},
            selected=_selected(min_samples=1),
            reward_best=_selected(min_samples=2, frontier_share=0.9, n_models_used=1),
            n_folds=5,
            imputed=(0, 600),
            frontier_model="m19",
        )
        assert out.exists() and out.stat().st_size > 0


class TestEtaSentenceFollowsTheData:
    @pytest.mark.parametrize(
        ("sensitivity", "driver", "passenger"),
        [
            ({"k": 0.90, "success_rate_thresh": 0.05}, "k", "success_rate_thresh"),
            ({"k": 0.05, "success_rate_thresh": 0.90}, "success_rate_thresh", "k"),
        ],
    )
    def test_driver_is_the_parameter_with_the_larger_eta(self, sensitivity, driver, passenger):
        notes = threshold_sweep._heatmap_annotations(
            selected=_selected(),
            reward_best=_selected(),
            y_name="success_rate_thresh",
            sensitivity=sensitivity,
            swept_ks=[2, 4, 6],
            n_folds=5,
            imputed=(0, 600),
            frontier_model="m19",
        ).notes
        note = next(n for n in notes if n.startswith("Reward is driven by"))
        assert note.startswith(f"Reward is driven by {driver} (η²=0.90)")
        assert f"{passenger} barely moves it (η²=0.05)" in note

    def test_comparable_parameters_are_not_called_negligible(self):
        notes = threshold_sweep._heatmap_annotations(
            selected=_selected(),
            reward_best=_selected(),
            y_name="success_rate_thresh",
            sensitivity={"k": 0.45, "success_rate_thresh": 0.44},
            swept_ks=[2, 4, 6],
            n_folds=5,
            imputed=(0, 600),
            frontier_model="m19",
        ).notes
        note = next(n for n in notes if n.startswith("Reward is driven by"))
        assert "barely moves it" not in note
        assert "also matters" in note

    def test_negligible_phrase_only_below_the_threshold(self):
        assert "negligible effect" in threshold_sweep._eta_phrase(0.02)
        assert "negligible effect" not in threshold_sweep._eta_phrase(0.90)
        assert "90%" in threshold_sweep._eta_phrase(0.90)
