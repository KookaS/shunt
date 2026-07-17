"""Scale-robustness tests for the task-level plots — must degrade gracefully at 500+ tasks."""

from __future__ import annotations

import csv

from benchmark import config
from benchmark.routing import report
from benchmark.routing.scripts import plot_external


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
        config.load("benchmark/config.yaml")
        rcsv, ext = self._write(tmp_path, 500)
        out = plot_external.plot_ours_vs_external(rcsv, ext, tmp_path)
        assert out.exists() and out.stat().st_size > 0

    def test_bars_at_small_n(self, tmp_path):
        config.load("benchmark/config.yaml")
        rcsv, ext = self._write(tmp_path, 12)
        out = plot_external.plot_ours_vs_external(rcsv, ext, tmp_path)
        assert out.exists() and out.stat().st_size > 0
