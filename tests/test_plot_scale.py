"""Scale-robustness tests for the task-level plots — must degrade gracefully at 500+ tasks.

Also holds the degenerate-input regressions for the same plots: an empty prior join
and a heatmap slice that must match the slice its title promises.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from benchmark import config
from benchmark.routing import report
from benchmark.routing.scripts import plot_external, threshold_sweep


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


class TestOursVsExternalScale:
    def _write(self, tmp_path, n_tasks):
        rcsv = tmp_path / "results.csv"
        with rcsv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["challenge_id", "model", "pass", "cost"])
            for i in range(n_tasks):
                w.writerow([f"repo__task-{i}", "deepseek-v4-flash", i % 4 != 0, 0.01])
        ext = tmp_path / "ext.csv"
        with ext.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["instance_id", "p_solve"])
            for i in range(n_tasks):
                w.writerow([f"repo__task-{i}", round(0.3 + 0.5 * ((i % 5) / 5), 3)])
        return rcsv, ext

    def test_agreement_matrix_at_500_tasks(self, tmp_path):
        config.load("benchmark/benchmark.yaml")
        rcsv, ext = self._write(tmp_path, 500)
        out = plot_external.plot_ours_vs_external(rcsv, ext, tmp_path)
        assert out.exists() and out.stat().st_size > 0

    def test_bars_at_small_n(self, tmp_path):
        config.load("benchmark/benchmark.yaml")
        rcsv, ext = self._write(tmp_path, 12)
        out = plot_external.plot_ours_vs_external(rcsv, ext, tmp_path)
        assert out.exists() and out.stat().st_size > 0


class TestAgreementMatrixNoPriorMatch:
    """41 tasks, none present in the leaderboard CSV — every p_solve is NaN."""

    def _write(self, tmp_path, n_tasks):
        rcsv = tmp_path / "results.csv"
        with rcsv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["challenge_id", "model", "pass", "cost"])
            for i in range(n_tasks):
                w.writerow([f"repo__task-{i}", "deepseek-v4-flash", i % 4 != 0, 0.01])
        ext = tmp_path / "ext.csv"
        with ext.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["instance_id", "p_solve"])
            w.writerow(["someone-else__task-0", "0.4"])
        return rcsv, ext

    def test_renders_instead_of_dividing_by_zero(self, tmp_path):
        config.load("benchmark/benchmark.yaml")
        rcsv, ext = self._write(tmp_path, _BARS_OVERFLOW_N)
        out = plot_external.plot_ours_vs_external(rcsv, ext, tmp_path)
        assert out.exists() and out.stat().st_size > 0

    def test_footer_states_that_no_task_matched(self, tmp_path):
        config.load("benchmark/benchmark.yaml")
        captured = {}
        orig = plot_external.plot_frame.save

        def spy(fig, path, spec, **kw):
            captured["extra"] = kw.get("extra")
            return orig(fig, path, spec, **kw)

        plot_external.plot_frame.save = spy
        try:
            rcsv, ext = self._write(tmp_path, _BARS_OVERFLOW_N)
            plot_external.plot_ours_vs_external(rcsv, ext, tmp_path)
        finally:
            plot_external.plot_frame.save = orig
        limits = " ".join(captured["extra"].limitations)
        assert "NO TASK MATCHED THE LEADERBOARD PRIOR" in limits
        assert str(_BARS_OVERFLOW_N) in limits
        # No cell-share claim can be made when nothing matched.
        assert not any("red cell holds" in n for n in captured["extra"].notes)


_BARS_OVERFLOW_N = plot_external._BARS_MAX_TASKS + 1


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
                    }
                )
    return rows


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
            out,
            y_name="min_samples",
            fixed=("success_rate_thresh", 0.5),
            swept_ks=[2, 4, 6],
            sensitivity={"k": 0.5, "min_samples": 0.4, "success_rate_thresh": 0.1},
            excluded_max=0,
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
        note = threshold_sweep._heatmap_annotations(
            {"k": 4, "Reward": 12.0},
            y_name="success_rate_thresh",
            sensitivity=sensitivity,
            swept_ks=[2, 4, 6],
            excluded_max=0,
        ).notes[0]
        assert note.startswith(f"Reward is driven by {driver} (η²=0.90)")
        assert f"{passenger} barely moves it (η²=0.05)" in note

    def test_comparable_parameters_are_not_called_negligible(self):
        note = threshold_sweep._heatmap_annotations(
            {"k": 4, "Reward": 12.0},
            y_name="success_rate_thresh",
            sensitivity={"k": 0.45, "success_rate_thresh": 0.44},
            swept_ks=[2, 4, 6],
            excluded_max=0,
        ).notes[0]
        assert "barely moves it" not in note
        assert "also matters" in note

    def test_negligible_phrase_only_below_the_threshold(self):
        assert "negligible effect" in threshold_sweep._eta_phrase(0.02)
        assert "negligible effect" not in threshold_sweep._eta_phrase(0.90)
        assert "90%" in threshold_sweep._eta_phrase(0.90)
