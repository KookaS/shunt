"""A figure may only claim what the data behind THAT claim supports."""

# Four defects of one family, each caught after the figure had already shipped: a caveat
# written for a case that could not reach it, a literal count frozen beside a derived one, an
# axis limit computed from point estimates while intervals are drawn, and an axis running past
# the range its quantity can take. The probes below are on the producers, not on the rendered
# PNG, so they fail on the arithmetic rather than on a pixel.

from __future__ import annotations

import dataclasses

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark import config, plot_frame  # noqa: E402
from benchmark.routing import plot_style  # noqa: E402
from benchmark.routing.figures import cost_quality_frontier as frontier  # noqa: E402
from benchmark.routing.figures import evidence_basis, kill_gate  # noqa: E402
from benchmark.routing.scripts import threshold_sweep  # noqa: E402

CONFIG_PATH = "benchmark/benchmark.yaml"


def _basis(label: str, n: int, b: int, c: int, decision: str) -> kill_gate.Basis:
    return kill_gate.Basis(
        label=label,
        n=n,
        diff_pp=100.0 * (b - c) / n,
        lo_pp=-1.0,
        hi_pp=1.0,
        decision=decision,
        b=b,
        c=c,
        router_cost=1.0,
        baseline_cost=2.0,
    )


class TestTheThinEvidenceCaveatFiresOnTheRowItGuards:
    """kill_gate: the caveat must be computed from the rows that CLEARED the bar."""

    # The shipped state of this figure: three inferior rows on wide bases, and one green row
    # — the headlined one — carried by three discordant pairs. The caveat was computed from
    # the WIDEST basis (38 discordant) inside an `elif` after the inferior branch, so in this
    # state, the only one the figure has ever rendered in, it could not fire at all.
    SHIPPED = [
        _basis("completed (imputed)", 184, 3, 35, "inferior"),
        _basis("measured only", 94, 3, 21, "inferior"),
        _basis("gate sample (N=20)", 20, 0, 4, "inferior"),
        _basis("shipped default", 184, 3, 0, "non_inferior"),
    ]

    def test_a_green_row_on_three_discordant_pairs_is_caveated(self):
        ann = kill_gate._annotations(self.SHIPPED, 0.05, None)
        assert ann.caveat is not None
        assert "clearing the bar" in ann.caveat
        # The number quoted is the GREEN row's discordance, never the widest basis's 38.
        assert " 3 discordant" in ann.caveat
        assert "38" not in ann.caveat

    def test_the_inferior_verdict_is_still_carried(self):
        ann = kill_gate._annotations(self.SHIPPED, 0.05, None)
        assert ann.caveat is not None
        assert "3 of 4 rows" in ann.caveat

    def test_a_green_row_on_ample_evidence_raises_no_thin_caveat(self):
        bases = [
            _basis("completed (imputed)", 184, 30, 10, "non_inferior"),
            _basis("measured only", 94, 20, 8, "non_inferior"),
        ]
        ann = kill_gate._annotations(bases, 0.05, None)
        assert ann.caveat is None

    def test_the_caveat_never_exceeds_what_the_canvas_renders(self):
        many = self.SHIPPED + [_basis(f"extra {i}", 40, 1, 0, "non_inferior") for i in range(6)]
        ann = kill_gate._annotations(many, 0.05, None)
        assert ann.caveat is None or len(ann.caveat) <= plot_frame.MAX_CAVEAT_CHARS


def _cell(passed: bool, imputed: bool) -> dict:
    return {"pass": passed, "cost": 1.0, "real_cost": 1.0, "imputed": imputed}


class TestTheFilledCellSplitIsDerived:
    """evidence_basis: no count on the subtitle line may be a literal."""

    COMPLETED = {
        "t1": {"a": _cell(True, False), "b": _cell(True, True), "c": _cell(False, True)},
        "t2": {"a": _cell(False, False), "b": _cell(True, True), "c": _cell(True, True)},
    }

    def test_filled_outcomes_counts_the_matrix_it_is_given(self):
        assert evidence_basis.filled_outcomes(self.COMPLETED) == (3, 4)

    def test_the_subtitle_quotes_that_split_rather_than_a_frozen_pair(self):
        bands = [{"band": 1, "real": 2, "imputed": 4, "unknown": 0}]
        ann = evidence_basis._annotations({}, bands, self.COMPLETED)
        joined = " · ".join(ann.subtitle_facts)
        assert "3 of 4 filled cells" in joined
        # The literal this replaced, beside a derived imputed-cell count that had moved on.
        assert "397" not in joined and "398" not in joined

    def test_a_matrix_with_no_filled_cell_makes_no_claim_about_one(self):
        real_only = {"t1": {"a": _cell(True, False)}}
        bands = [{"band": 1, "real": 1, "imputed": 0, "unknown": 0}]
        ann = evidence_basis._annotations({}, bands, real_only)
        assert not any("filled cells" in f for f in ann.subtitle_facts)


def _row(name: str, cost: float, n: int, passes: int) -> dict:
    return {
        "strategy": name,
        "n_tasks": str(n),
        "n_pass": str(passes),
        "AvgPerf%": f"{passes / n * 100:.2f}",
        "TotalCost": str(cost),
        "TotalCost_cacheaware": str(cost),
        "TotalCost_ci_lower": str(cost * 0.9),
        "TotalCost_ci_upper": str(cost * 1.1),
        "context_cost_alpha_01": "0",
        "context_cost_alpha_03": "0",
        "context_cost_alpha_10": "0",
        "context_cost_n": "0",
    }


class TestTheYFloorClearsTheIntervalsDrawnOnIt:
    """cost_quality_frontier: the least precise point must not read as the most precise."""

    # The shipped case: kNN-semantic-tier at 65.8% on 121 scored tasks, whose Wilson interval
    # reaches ~58.7 — nearly nine points below a `min(perfs) - 3.0` floor.
    ROWS = [_row("wide", 12.0, 121, 80), _row("tight", 90.0, 184, 175)]

    def test_the_floor_sits_below_the_lowest_wilson_cap(self):
        fig, ax = plt.subplots()
        try:
            floor, ceiling = frontier._panel_limits(ax, self.ROWS, None)
            lo, _hi = plot_style.wilson_interval(80, 121)
            assert floor <= lo * 100.0
            assert ax.get_ylim()[0] <= lo * 100.0
            # A pass rate stops at 100: the ceiling is the bound, not a window.
            assert ceiling == 100.0
        finally:
            plt.close(fig)

    def test_the_floor_never_goes_below_zero(self):
        fig, ax = plt.subplots()
        try:
            floor, _ceiling = frontier._panel_limits(ax, [_row("all-fail", 1.0, 4, 0)], None)
            assert floor >= 0.0
        finally:
            plt.close(fig)


def _tiny_corpus(n: int = 24) -> tuple[list[str], np.ndarray, dict, dict]:
    rng = np.random.default_rng(0)
    models = ["deepseek-v4-flash", "kimi-k3"]
    results: dict[str, dict] = {}
    for i in range(n):
        tid = f"t{i:03d}"
        results[tid] = {
            m: {
                "pass": bool((i + j) % 3),
                "cost": 0.01 * (j + 1),
                "real_cost": 0.01 * (j + 1),
                "imputed": False,
            }
            for j, m in enumerate(models)
        }
    matrix = {"results": results, "models": {m: {} for m in models}}
    return sorted(results), rng.normal(size=(n, 8)), results, matrix


class TestPanelCStaysInsideThePassRateRange:
    """sweep_regimes: a bounded quantity gets a bounded axis, and the figure states its n."""

    @staticmethod
    def _result() -> threshold_sweep.SweepResult:
        config.load(CONFIG_PATH)
        task_ids, feats, results, matrix = _tiny_corpus()
        grid = threshold_sweep.Grid((2, 4), (0.5, 0.9), (1,))
        return threshold_sweep.run_sweep(task_ids, feats, results, matrix, grid, 4)

    def test_the_optimism_panel_never_draws_above_100_percent(self):
        res = self._result()
        fig, ax = plt.subplots()
        try:
            threshold_sweep._draw_optimism(ax, res)
            assert ax.get_ylim()[1] <= 100.0
            for text in ax.texts:
                if text.get_transform() is ax.transData:
                    assert text.get_position()[1] <= 100.0
        finally:
            plt.close(fig)

    def test_the_figure_publishes_the_n_every_panel_rests_on(self):
        ann = threshold_sweep._annotations(self._result())
        counts = dict(ann.counts)
        assert counts, "sweep_regimes shipped with an empty `n` footprint"
        assert counts["tasks"] > 0
        assert counts["oof_scored"] > 0
        assert counts["grid_cells"] > 0

    def test_an_unscored_arm_is_stated_rather_than_drawn_as_a_zero(self):
        # With no nested row the out-of-fold rate is 0.0 over n=0 — an empty subject set that
        # the panel used to draw as a measured 0% beside a full-height in-sample bar.
        res = dataclasses.replace(self._result(), nested=[])
        fig, ax = plt.subplots()
        try:
            threshold_sweep._draw_optimism(ax, res)
            labels = [t.get_text() for t in ax.texts]
            assert not any("(n=0)" in t for t in labels)
            assert any("no optimism gap" in t for t in labels)
        finally:
            plt.close(fig)

    def test_the_imputation_limit_names_the_matrix_it_counted(self):
        # The tiny corpus is fully measured, so the imputation clause is planted rather than
        # left to chance: an absent line would pass this probe vacuously.
        res = dataclasses.replace(self._result(), imputed=(9, 48))
        limits = threshold_sweep._data_limits(res)
        imputation = [line for line in limits if "IMPUTED" in line]
        assert imputation, "the imputation limit did not fire on a matrix with imputed cells"
        for line in imputation:
            assert "THE MATRIX THIS SWEEP SCORES" in line
            # Two half-edits of one sentence had left this contradiction in the shipped text.
            assert "almost never can never" not in line
