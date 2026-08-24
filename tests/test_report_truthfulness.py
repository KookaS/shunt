"""Guards on the claims the routing report's figures are allowed to print."""

# Each test pins a defect a forensic audit found in a shipped PNG: a headline crediting
# a hindsight oracle, an unpaired marginal presented as a dose-response, a footer
# mangled by matplotlib mathtext, a saving that lives entirely in imputed cells. These
# are CLAIM tests, not render tests — a figure that renders while saying something
# false is the failure mode, and it renders green.

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark import plot_frame
from benchmark.routing import metrics, report, strategy_class
from benchmark.routing.figures import arm_manipulation, context


class TestHindsightIsNeverDeployable:
    def test_every_oracle_is_a_bound(self):
        for name in ("Oracle", "Oracle-reward", "Arm-oracle"):
            assert strategy_class.classify(name).cls is strategy_class.StrategyClass.BOUND
            assert not strategy_class.is_live(name)

    def test_only_the_products_own_allowlist_is_live(self):
        # The whole point of the module: the benchmark cannot mint its own "deployable".
        for name in ("kNN-cascade", "Always-Frontier", "Always-Cheap"):
            assert strategy_class.is_live(name)
        for name in ("kNN-cascade (within-task)", "Price-Cascade", "Tier-Classifier"):
            assert not strategy_class.is_live(name)
            assert strategy_class.classify(name).path_to_live

    def test_the_bare_selection_rule_is_a_control_with_no_route(self):
        # `knn` left LIVE_STRATEGIES when the kNN strategy was renamed to `knn_cascade`: the kNN
        # pick has always run WITH the escalation ladder, so the pick alone is a contrast,
        # not a configuration. A CONTROL takes no path_to_live — there is no route and there
        # should not be one.
        found = strategy_class.classify("kNN")
        assert found.cls is strategy_class.StrategyClass.CONTROL
        assert found.path_to_live is None

    def test_the_figures_default_router_is_live(self):
        # Every caption that says "the shipped default" resolves through this constant.
        assert strategy_class.is_live(context.DEFAULT_STRATEGY)
        assert strategy_class.is_live(context.BASELINE_STRATEGY)

    def test_the_kill_gates_pre_registered_arm_is_deliberately_not_live(self):
        # A pre-registered verdict arm may not be repointed after seeing the data, so the
        # gate keeps adjudicating the kNN row even though no router.strategy names it. That
        # gap is published on the canvas beside the default's own row, not hidden.
        assert not strategy_class.is_live(context.ROUTER_STRATEGY)
        assert context.ROUTER_STRATEGY != context.DEFAULT_STRATEGY


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
        assert metrics.mcnemar_exact_p(0, 0) == 1.0

    def test_zero_versus_one_is_not_significant(self):
        assert metrics.mcnemar_exact_p(0, 1) == 1.0

    def test_a_lopsided_split_is(self):
        assert metrics.mcnemar_exact_p(0, 12) < 0.01


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


class TestAManipulationThatNeverFiredIsNotANull:
    """A treatment that was never applied has no null effect — it has nothing to measure."""

    @staticmethod
    def _raw(low_tok: int, high_tok: int, n: int = 20) -> dict:
        return {
            f"t{i}": {
                "m": {
                    "lo": {"pass": True, "out_tok": low_tok, "real_cost": 0.01},
                    "hi": {"pass": True, "out_tok": high_tok, "real_cost": 0.02},
                }
            }
            for i in range(n)
        }

    def test_a_flat_output_token_ratio_is_reported_as_not_fired(self):
        rows = metrics.arm_manipulation(self._raw(1000, 1010), [("m", "lo", "hi")])
        assert rows[0]["fired"] is False
        assert rows[0]["out_tok_ratio"] == pytest.approx(1.01, abs=0.01)

    def test_a_real_manipulation_fires(self):
        rows = metrics.arm_manipulation(self._raw(1000, 2900), [("m", "lo", "hi")])
        assert rows[0]["fired"] is True

    def test_too_few_pairs_never_fires_however_large_the_ratio(self):
        rows = metrics.arm_manipulation(
            self._raw(1000, 5000, n=metrics.MIN_ARM_PAIRS - 1), [("m", "lo", "hi")]
        )
        assert rows[0]["fired"] is False

    def test_a_row_that_did_not_fire_is_labelled_and_never_drawn_as_a_null(self):
        rows = metrics.arm_manipulation(self._raw(1000, 1010), [("m", "lo", "hi")])
        pairs = {
            ("m", "lo", "hi"): {
                "model": "m",
                "low": "lo",
                "high": "hi",
                "n": 20,
                "delta_pp": 0.0,
                "ci": (-5.0, 5.0),
                "cost_delta": 0.0,
            }
        }
        totals = {"n": 20.0, "net_pp": 0.0, "p": 1.0}
        notes = arm_manipulation._annotations(rows, pairs, totals).notes
        assert any(arm_manipulation.NOT_A_NULL in note for note in notes)
        # The word "null" may appear only inside that exact phrase — an unfired row
        # described as a null result is the defect this whole figure exists to stop.
        for note in notes:
            if "null" in note.lower():
                assert arm_manipulation.NOT_A_NULL in note


class TestCensoredCellsNeverEnterAnArmContrast:
    """A resource-limit stop has an unknown outcome and must not score as a capability fail."""

    @staticmethod
    def _raw_with_censored() -> dict:
        rows = {}
        for i in range(10):
            rows[f"t{i}"] = {
                "m": {
                    "lo": {"pass": True, "real_cost": 0.01},
                    "hi": {"pass": True, "real_cost": 0.02},
                }
            }
        rows["censored"] = {
            "m": {
                "lo": {"pass": True, "real_cost": 0.01},
                "hi": {"pass": False, "stop_reason": "step_limit", "real_cost": 0.02},
            }
        }
        return rows

    def test_a_censored_high_arm_is_dropped_rather_than_counted_as_a_violation(self):
        pair = report._arm_pair(self._raw_with_censored(), "m", "lo", "hi")
        assert pair["n"] == 10
        assert pair["violations"] == 0


class TestOneStrategyHasOneCommittedNumber:
    """Two figures reporting different (cost, pass) for one strategy is a correctness bug."""

    def test_no_committed_figure_publishes_a_rival_router_number(self):
        figures = Path("docs/assets/figures/routing")
        if not figures.exists():
            pytest.skip("no rendered figure set")
        proxies = [
            p.name
            for p in figures.glob("*.png")
            if p.name in {"knn_cost_comparison.png", "strategy_comparison.png"}
        ]
        # Those two figures published a proxy kNN router at 77.7% / $1.73 alongside the
        # live engine's 81.71% / $13.34 in the same committed set.
        assert proxies == []


class TestNoReadingOrGoalStringReachesTheCanvas:
    """`reading` and `goal` are mandatory in code and must never be drawn."""

    @staticmethod
    def _specs() -> list:
        from benchmark.routing.figures import (
            cache_economics,
            complementarity,
            cost_quality_frontier,
            decision_audit,
            evidence_basis,
            kill_gate,
            ladder_rungs,
            oracle_gap,
            task_difficulty,
        )

        return [
            m.SPEC
            for m in (
                arm_manipulation,
                cache_economics,
                complementarity,
                cost_quality_frontier,
                decision_audit,
                evidence_basis,
                kill_gate,
                ladder_rungs,
                oracle_gap,
                task_difficulty,
            )
        ]

    def test_every_caveat_is_within_the_canvas_limit(self):
        for spec in self._specs():
            if spec.caveat is not None:
                assert len(spec.caveat) <= plot_frame.MAX_CAVEAT_CHARS

    def test_every_title_is_a_claim_within_the_limit(self):
        for spec in self._specs():
            assert spec.title.strip()
            assert len(spec.title) <= plot_frame.MAX_TITLE_CHARS

    def test_no_rendered_block_contains_the_reading_or_goal_text(self):
        for spec in self._specs():
            rendered = " ".join(
                part for part in (spec.title, spec.subtitle, spec.caveat or "") if part
            )
            assert spec.reading not in rendered
            assert spec.goal not in rendered


class TestPairedQualityContrastIsRegenerable:
    # The published number must equal what the committed generator emits from committed
    # data, byte for byte — regenerating the report reproduces it, and when the corpus
    # grows the number goes stale loudly instead of silently.
    """benchmark.md's Price-Cascade-vs-frontier headline (paired quality contrast)."""

    _DOC = Path("docs/benchmark.md")

    def test_regeneration_reproduces_the_published_fragment(self) -> None:
        generated = report.paired_quality_contrast()
        assert generated  # non-empty
        # The doc quotes the generator's numbers (bold markers split the fragment).
        doc_text = self._DOC.read_text(encoding="utf-8")
        plain = doc_text.replace("**", "")
        assert generated in plain, (
            "benchmark.md no longer contains the generated paired-quality fragment. "
            f"generated: {generated!r}"
        )

    def test_published_number_matches_generator_numbers(self) -> None:
        # Byte-for-byte on the numeric claims: what the generator prints is what the
        # the doc publishes the corrected numbers, not the stale n=180 ones.
        generated = report.paired_quality_contrast()
        doc_text = self._DOC.read_text(encoding="utf-8")
        plain = doc_text.replace("**", "")
        for token in generated.replace("→", "").split():
            if token and token[0] in "+-0123456789":
                assert token in plain, f"generated number {token!r} missing from benchmark.md"
