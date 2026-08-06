"""Scale-robustness tests for the task-level plots — must degrade gracefully at 500+ tasks.

Also holds the degenerate-input regression for the heatmap slice, which must match
the slice its title promises.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark import config, plot_frame
from benchmark.routing.figures import complementarity
from benchmark.routing.scripts import threshold_sweep

CONFIG_PATH = "benchmark/benchmark.yaml"


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


def _tiny_corpus(n: int = 12):
    """A corpus small enough to sweep in a test, with a real cheap/dear split."""
    matrix = {
        "models": {
            "cheap": {"input_price": 0.1, "output_price": 0.1},
            "dear": {"input_price": 5.0, "output_price": 5.0},
        }
    }
    results = {
        f"t{i:02d}": {
            "cheap": {"pass": i % 2 == 0, "real_cost": 0.01},
            "dear": {"pass": True, "real_cost": 0.50},
        }
        for i in range(n)
    }
    rng = np.random.default_rng(0)
    return sorted(results), rng.normal(size=(n, 8)), results, matrix


class TestOuterLoopCrossValidation:
    def test_out_of_fold_rows_exclude_the_folds_they_did_not_score(self):
        config.load(CONFIG_PATH)
        task_ids, feats, results, matrix = _tiny_corpus()
        grid = threshold_sweep.Grid((2, 4), (0.5, 0.9), (1,))
        res = threshold_sweep.run_sweep(task_ids, feats, results, matrix, grid, 4)
        # The defect this replaced: every row reported n_excluded == 0, so the sweep
        # selected its optimum on the very rows it scored.
        assert res.fold_rows and all(r["n_excluded"] > 0 for r in res.fold_rows)
        assert {r["fold"] for r in res.fold_rows} == {"0", "1", "2", "3"}
        assert sum(r["n_scored"] for r in res.nested) == len(task_ids)

    def test_k_grid_is_log_spaced_and_clipped_to_the_corpus(self):
        assert threshold_sweep.k_grid(200) == (2, 3, 5, 8, 12, 20, 32, 50, 80, 128, 174)
        assert threshold_sweep.k_grid(13) == (2, 3, 5, 8, 12)
        assert threshold_sweep.k_grid(1) == (2,)

    def test_end_to_end_regimes_figure_renders(self, tmp_path):
        config.load(CONFIG_PATH)
        task_ids, feats, results, matrix = _tiny_corpus()
        grid = threshold_sweep.Grid((2, 4), (0.5, 0.9), (1,))
        res = threshold_sweep.run_sweep(task_ids, feats, results, matrix, grid, 4)
        out = tmp_path / "sweep_regimes.png"
        threshold_sweep.plot_sweep_regimes(res, out, "deadbeef")
        assert out.exists() and out.stat().st_size > 0


class TestEtaSentenceFollowsTheData:
    def test_driver_is_the_parameter_with_the_larger_eta(self):
        note = threshold_sweep._driver_sentence(("k", 0.90), ("success_rate_thresh", 0.05))
        assert note.startswith("Reward is driven by k (η²=0.90)")
        assert "success_rate_thresh barely moves it (η²=0.05)" in note

    def test_comparable_parameters_are_not_called_negligible(self):
        note = threshold_sweep._driver_sentence(("k", 0.45), ("success_rate_thresh", 0.44))
        assert "barely moves it" not in note
        assert "also matters" in note


def _big_raw(n_tasks: int, n_cols: int) -> tuple[dict, list[tuple[str, str]]]:
    """A tri-state cache far larger than the committed one, to prove the grid degrades."""
    columns = [(f"model-{c:02d}", "default") for c in range(n_cols)]
    raw = {
        f"repo__task-{i:04d}": {
            model: {arm: {"pass": (i + hash(model)) % 3 == 0, "real_cost": 0.01}}
            for model, arm in columns
            if (i + len(model)) % 2 == 0
        }
        for i in range(n_tasks)
    }
    return raw, columns


class TestComplementarityScale:
    """The one figure that draws a cell per (task, column) must degrade, not clip.

    It passed at scale only because `bbox_inches="tight"` grew the canvas; the PNG is
    now exactly `size x dpi`, so an over-full axis is CLIPPED rather than accommodated.
    """

    def test_row_labels_thin_rather_than_overprint(self):
        step, size = complementarity.row_label_step(500, 11.0)
        assert step > 1
        assert 500 / step <= 11.0 * 72.0 / complementarity._MIN_ROW_PITCH_PT + 1
        assert complementarity._MIN_ROW_LABEL_PT <= size <= complementarity._MAX_ROW_LABEL_PT

    def test_a_small_grid_labels_every_row(self):
        step, _size = complementarity.row_label_step(20, 11.0)
        assert step == 1

    def test_renders_500_tasks_20_models(self, tmp_path):
        raw, columns = _big_raw(500, 20)
        census = complementarity.build_census(raw, columns)
        assert census.n_tasks == 500
        assert census.n_cols == 20
        fig, axes = plot_frame.subplots(plot_frame.WIDE_TALL, 1, 3)
        complementarity._draw_grid(axes[0], census, plot_frame.WIDE_TALL.height_in - 2.0)
        complementarity._draw_coverage(axes[1], census)
        complementarity._draw_census(axes[2], census)
        out = plot_frame.save(
            fig,
            tmp_path / "complementarity.png",
            complementarity.SPEC,
            extra=complementarity._annotations(census),
            size=plot_frame.WIDE_TALL,
        )
        # bbox_inches is gone, so the canvas is exactly the declared size in pixels.
        from PIL import Image

        assert Image.open(out).size == (
            int(plot_frame.WIDE_TALL.width_in * plot_frame.DPI),
            int(plot_frame.WIDE_TALL.height_in * plot_frame.DPI),
        )
