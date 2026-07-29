"""Guards on the claims the routing report's figures are allowed to print."""

# Each test pins a defect a forensic audit found in a shipped PNG: a headline crediting
# a hindsight oracle, an unpaired marginal presented as a dose-response, a footer
# mangled by matplotlib mathtext, a saving that lives entirely in imputed cells. These
# are CLAIM tests, not render tests — a figure that renders while saying something
# false is the failure mode, and it renders green.

from __future__ import annotations

import numpy as np
import pytest

from benchmark.routing import report


class TestHindsightIsNeverDeployable:
    def test_every_oracle_is_hindsight(self):
        for name in ("Oracle", "Oracle-reward", "Arm-oracle"):
            assert report._is_hindsight(name)

    def test_a_router_is_not(self):
        for name in ("kNN-cascade", "Always-Frontier", "Always-Cheap", "Tier-Classifier"):
            assert not report._is_hindsight(name)

    def test_dominance_is_judged_against_deployable_strategies_only(self):
        # The oracle is cheaper AND better than both routers, but it cannot deploy,
        # so neither router may be painted "dominated" because of it.
        names = ["Oracle", "kNN-cascade", "Always-Frontier"]
        costs = [13.6, 76.9, 87.0]
        perfs = [96.6, 96.6, 96.0]
        ns = [177, 177, 177]
        assert report._deployable_pareto(names, costs, perfs, ns) == {"kNN-cascade"}

    def test_a_zero_evidence_row_is_not_pareto_optimal(self):
        # ($0, 0%) is un-dominated by construction — absence of evidence, not efficiency.
        assert report._deployable_pareto(["Broken", "kNN"], [0.0, 5.0], [0.0, 78.0], [0, 177]) == {
            "kNN"
        }


class TestDollarsSurviveMathtext:
    """Two bare `$` in one string open a mathtext span: matplotlib italicises the text
    between them and deletes its spaces. The shipped footer read
    `costs 1.3590againstthe87.0438`."""

    def test_no_string_literal_carries_an_unescaped_dollar(self):
        # AST-level, so comments and docstrings are out of scope by construction and
        # only literals that can actually reach a canvas are checked.
        import ast
        import inspect
        import re

        tree = ast.parse(inspect.getsource(report))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.FunctionDef | ast.ClassDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        offenders = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and re.search(r"(?<!\\)\$", node.value)
        ]
        assert not offenders, offenders


class TestPairedArmContrast:
    """The shipped facets compared each arm's MARGINAL pass rate over whatever tasks
    that arm happened to run. Restricted to co-measured tasks, arm pairs flip sign."""

    # `high` ran only on the two tasks the model finds easy, so its marginal rate is
    # 100% against `low`'s 50% — but on the tasks BOTH ran, `low` is the one at 100%.
    _RAW = {
        "t1": {"m": {"low": {"pass": True, "cost": 1.0}, "high": {"pass": True, "cost": 2.0}}},
        "t2": {"m": {"low": {"pass": True, "cost": 1.0}, "high": {"pass": False, "cost": 2.0}}},
        "t3": {"m": {"low": {"pass": False, "cost": 1.0}}},
        "t4": {"m": {"low": {"pass": False, "cost": 1.0}}},
    }
    _RANKS = {("m", "low"): 0, ("m", "high"): 1}

    def test_contrast_uses_only_co_measured_tasks(self):
        (pair,) = report.arm_pair_contrasts(self._RAW, self._RANKS)
        assert pair["n"] == 2  # not 4: t3/t4 never ran `high`
        assert pair["low_rate"] == 100.0
        assert pair["high_rate"] == 50.0
        assert pair["delta_pp"] == -50.0  # the marginal comparison would say +50

    def test_a_higher_arm_failing_below_a_lower_arm_is_a_violation(self):
        (pair,) = report.arm_pair_contrasts(self._RAW, self._RANKS)
        assert pair["violations"] == 1
        assert pair["gains"] == 0

    def test_cost_delta_reads_real_cost(self):
        raw = {
            "t1": {
                "m": {
                    "low": {"pass": True, "cost": 99.0, "real_cost": 1.0},
                    "high": {"pass": True, "cost": 99.0, "real_cost": 3.0},
                }
            }
        }
        (pair,) = report.arm_pair_contrasts(raw, self._RANKS)
        assert pair["cost_delta"] == 2.0  # not 0.0 from the stale `cost` column

    def test_totals_report_the_pooled_paired_effect(self):
        totals = report.arm_pair_totals(report.arm_pair_contrasts(self._RAW, self._RANKS))
        assert totals["n"] == 2.0
        assert totals["violation_rate"] == 0.5
        assert totals["p"] == 1.0  # 1 vs 0 discordant resolves nothing

    def test_footer_states_the_null_plainly(self):
        pairs = report.arm_pair_contrasts(self._RAW, self._RANKS)
        footer = " ".join(report._monotonicity_annotations(pairs).notes)
        assert "DATASET-WIDE PAIRED RESULT" in footer
        assert "indistinguishable from zero" in footer
        assert "VIOLATED" in footer


class TestMcNemar:
    def test_no_discordant_pairs_resolves_nothing(self):
        assert report._mcnemar_exact_p(0, 0) == 1.0

    def test_zero_versus_one_is_not_significant(self):
        assert report._mcnemar_exact_p(0, 1) == 1.0

    def test_a_lopsided_split_is(self):
        assert report._mcnemar_exact_p(0, 12) < 0.01


class TestMeasuredVersusImputed:
    """409 of 1062 analytical cells are imputed and EVERY imputed cell is pass=True,
    so a strategy's headline number must be separable into evidence and inference."""

    _CELLS = {
        "t1": (True, 1.0, False),
        "t2": (True, 4.0, True),
        "t3": (False, 2.0, False),
        "t4": (True, 3.0, True),
    }

    def test_split_separates_billed_dollars_from_projected_ones(self):
        split = report._split_measured(self._CELLS, set())
        assert split["measured_cost"] == 3.0
        assert split["imputed_cost"] == 7.0
        assert split["imputed_pass"] == 2.0
        assert split["measured_pass"] == 1.0

    def test_unscorable_tasks_are_not_counted_as_either(self):
        split = report._split_measured(self._CELLS, {"t2"})
        assert split["imputed_cost"] == 3.0
        assert split["imputed_cells"] == 1.0

    def test_footer_names_the_projected_share_for_every_strategy(self):
        ann = report._measured_split_annotations(
            {"Always-Frontier": report._split_measured(self._CELLS, set())}
        )
        footer = " ".join(ann.notes)
        assert "70.0%) projected" in footer
        assert "EVERY projected cell is filled pass=True" in footer


class TestStrategyCellsPathAwareImputation:
    """A cascade bills every model it probed, so the imputed flag must cover the PATH."""

    # Filtering on the FINAL cell alone put $0.1567 of projected spend inside the panel
    # titled "MEASURED ONLY — no imputed cell on either side".

    class _Cascade:
        """Minimal stand-in for the shipped cascades' evaluate contract."""

        name = "Cascade"

        def __init__(self) -> None:
            self.cascade_total_cost = 0.0
            self.cascade_tried_models = []

        def select(self, task_id, task_meta, matrix):  # noqa: ANN001, ANN201, ARG002
            self.cascade_tried_models = []
            self.cascade_total_cost = 0.0
            for model in ("cheap", "dear"):
                cell = matrix["results"].get(task_id, {}).get(model, {})
                self.cascade_tried_models.append(model)
                self.cascade_total_cost += cell.get("cost", 0.0)
                if cell.get("pass"):
                    return model
            return "dear"

    _MATRIX = {
        "tasks": {"t1": {}, "t2": {}},
        "results": {
            # t1: the probe is imputed, the returned cell is measured — the leak.
            "t1": {
                "cheap": {"pass": False, "cost": 0.1, "real_cost": 0.1, "imputed": True},
                "dear": {"pass": True, "cost": 1.0, "real_cost": 1.0},
            },
            # t2: nothing on the path is imputed.
            "t2": {
                "cheap": {"pass": True, "cost": 0.2, "real_cost": 0.2},
                "dear": {"pass": True, "cost": 1.0, "real_cost": 1.0},
            },
        },
    }

    def _cells(self):  # noqa: ANN202
        cells, _unscorable = report.strategy_cells(self._MATRIX, ["t1", "t2"], [self._Cascade()])[
            "Cascade"
        ]
        return cells

    def test_an_imputed_probe_marks_the_whole_decision(self):
        assert self._cells()["t1"][2] is True

    def test_a_fully_measured_path_stays_measured(self):
        assert self._cells()["t2"][2] is False

    def test_the_cascade_total_cost_still_reconciles(self):
        # The recorder must not disturb what evaluate reads back off the strategy.
        assert self._cells()["t1"][1] == pytest.approx(1.1)

    def test_the_measured_only_panel_excludes_the_tainted_task(self):
        cells = self._cells()
        by_strategy = {
            "Cascade": (cells, set()),
            "Always-Frontier": ({"t1": (True, 1.0, False), "t2": (True, 1.0, False)}, set()),
        }
        paired = report.paired_measured("Cascade", "Always-Frontier", by_strategy)
        assert paired is not None
        assert paired["n"] == 1.0  # t1 is out: its cascade billed a projected cell


class TestPairedMeasuredKillGate:
    def test_only_tasks_measured_on_both_sides_enter(self):
        by_strategy = {
            "Always-Frontier": (
                {"t1": (True, 5.0, False), "t2": (True, 5.0, True), "t3": (True, 5.0, False)},
                set(),
            ),
            "kNN-cascade": (
                {"t1": (True, 4.0, False), "t2": (True, 4.0, False), "t3": (False, 6.0, True)},
                set(),
            ),
        }
        paired = report.paired_measured("kNN-cascade", "Always-Frontier", by_strategy)
        assert paired is not None
        assert paired["n"] == 1.0  # only t1 is measured on both sides
        assert paired["router_cost"] == 4.0
        assert paired["baseline_cost"] == 5.0

    def test_a_zero_saving_is_reported_as_a_null_result(self):
        by_strategy = {
            "Always-Frontier": ({"t1": (True, 42.74, False)}, set()),
            "kNN-cascade": ({"t1": (True, 42.87, False)}, set()),
        }
        paired = report.paired_measured("kNN-cascade", "Always-Frontier", by_strategy)
        ann = report._paired_annotations(paired)
        footer = " ".join(ann.notes)
        assert "NULL RESULT" in footer
        assert "MORE than the baseline" in footer

    def test_no_overlap_is_none_rather_than_an_empty_claim(self):
        by_strategy = {
            "Always-Frontier": ({"t1": (True, 1.0, True)}, set()),
            "kNN-cascade": ({"t1": (True, 1.0, False)}, set()),
        }
        assert report.paired_measured("kNN-cascade", "Always-Frontier", by_strategy) is None
        assert report._paired_annotations(None).notes == ()


class TestRegretHeadline:
    def test_overlapping_intervals_are_not_reported_as_a_ranking(self):
        title = report._regret_title(
            {"kNN-cascade": 6.33, "Always-Frontier": 8.35},
            {"kNN-cascade": (5.62, 7.02), "Always-Frontier": (6.82, 10.88)},
            177,
        )
        assert "OVERLAPS" in title
        assert "tracks the oracle closest" not in title

    def test_separated_intervals_still_get_a_ranking(self):
        title = report._regret_title(
            {"kNN-cascade": 6.33, "Always-Frontier": 30.0},
            {"kNN-cascade": (5.6, 7.0), "Always-Frontier": (25.0, 35.0)},
            177,
        )
        assert "tracks the oracle closest" in title

    def test_an_oracle_never_takes_the_headline(self):
        title = report._regret_title(
            {"Oracle": 0.0, "Arm-oracle": -0.05, "kNN-cascade": 6.33},
            {},
            177,
        )
        assert "kNN-cascade" in title
        assert "Oracle" not in title.replace("oracle closest", "")


class TestDroppedTasksAreDisclosedByComposition:
    """All 44 tasks missing from the chosen-arm cloud were frontier picks — the
    missingness is perfectly confounded with the y axis it hides."""

    def test_composition_and_true_share_are_stated(self):
        lines = report._routing_share_lines(
            plotted={"kimi-k3": 94, "deepseek-v4-flash": 52}, dropped={"kimi-k3": 44}
        )
        joined = " ".join(lines)
        assert "kimi-k3 44" in joined
        assert "64% of the plotted points" in joined
        assert "73% of the true routing" in joined

    def test_nothing_is_claimed_when_nothing_was_dropped(self):
        assert report._routing_share_lines({"kimi-k3": 94}, {}) == []


class TestHeatmapRowLabels:
    """Every task row the canvas can fit must carry its name, and the footer must
    only claim rows are unlabelled when they actually are."""

    # The figure's own LIMITS used to state "labels are thinned to ~40 evenly spaced
    # rows, so most rows are unlabelled" while the 32-inch canvas had 10.8 pt of pitch
    # per row — room for all 200.

    def test_the_committed_shape_labels_every_row(self):
        step, fontsize = report._row_label_step(200, 32.0)
        assert step == 1
        assert 4.0 <= fontsize <= 7.0

    def test_an_impossible_density_thins_and_says_so(self):
        step, fontsize = report._row_label_step(2000, 32.0)
        assert step > 1
        assert fontsize == 4.0
        limits = report._heatmap_annotations(
            np.zeros((2000, 3)), np.array([1, 1, 1]), False, {}, False, step
        ).limitations
        assert any("unlabelled" in text for text in limits)

    def test_no_unlabelled_claim_when_every_row_is_labelled(self):
        limits = report._heatmap_annotations(
            np.zeros((200, 3)), np.array([1, 1, 1]), False, {}, False, 1
        ).limitations
        assert not any("unlabelled" in text for text in limits)
