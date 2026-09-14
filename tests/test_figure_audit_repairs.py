"""The 2026-09-14 figure-audit repairs: denominators, canonical identity, derived claims.

Each class pins one defect the consolidated audit found in a shipped figure so a re-render
cannot reintroduce it: an undrawn share that left stacks short of 1.0, unscored tasks counted
as unwinnable, silently dropped decisions, a hardcoded multiple the data had outgrown, a
lane-keyed lookup that lost channel mirrors, a four-column matrix over a five-conjunct
predicate, and an empty-match fallback that re-admitted the models it was meant to filter.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

from benchmark import plot_frame  # noqa: E402
from benchmark.routing import model_validity  # noqa: E402
from benchmark.routing.figures import (  # noqa: E402
    cache_economics,
    decision_audit,
    kill_gate,
    task_difficulty,
)
from benchmark.routing.figures import context as ctxmod  # noqa: E402
from benchmark.routing.figures import (
    model_validity as fig_validity,
)
from benchmark.routing.model_validity import FREE, PAID, ModelValidity  # noqa: E402


def _row(model: str, *, valid: bool = True, channel: str = PAID, cells: int = 25) -> ModelValidity:
    return ModelValidity(
        model=model,
        channel=channel,
        providers=("p",),
        listings=(model,),
        collection_only=False,
        live=valid,
        enabled=valid,
        triage="KEEP",
        capability="measured",
        cells=cells,
        covered=cells,
        corpus=100,
        valid=valid,
        reason="inference-valid" if valid else "not in the live pool",
    )


# ── F1 / F2: task_difficulty denominators ─────────────────────────────────────────────


class TestAllocationStacksReachOne:
    """Every pick enters the denominator; picks outside the pool are drawn as one grey bar."""

    _ALLOC = {1: Counter({"a": 3, "gpt-5-mini": 1}), 0: Counter({"b": 1})}
    _MODELS = ["a", "b"]

    def test_outside_picks_are_counted(self):
        assert task_difficulty.outside_pool_picks(self._ALLOC, self._MODELS) == 1

    def test_every_stack_sums_to_one_including_the_grey_segment(self):
        fig, ax = plt.subplots()
        try:
            task_difficulty._draw_allocation(
                ax, self._ALLOC, self._MODELS, {"a": "#111", "b": "#222"}
            )
            by_x: dict[float, float] = {}
            for patch in ax.patches:
                by_x[round(patch.get_x(), 6)] = (
                    by_x.get(round(patch.get_x(), 6), 0.0) + patch.get_height()
                )
            assert len(by_x) == 2
            for total in by_x.values():
                assert total == pytest.approx(1.0, abs=1e-9)
        finally:
            plt.close(fig)

    def test_the_outside_segment_is_grey_and_not_named_as_a_model(self):
        fig, ax = plt.subplots()
        try:
            task_difficulty._draw_allocation(
                ax, self._ALLOC, self._MODELS, {"a": "#111", "b": "#222"}
            )
            grey = [
                p for p in ax.patches if p.get_facecolor()[:3] == to_rgb(task_difficulty._OUTSIDE)
            ]
            assert grey, "no aggregated outside-pool segment drawn"
            labels = [t.get_text() for t in ax.get_legend().get_texts()]
            assert task_difficulty._OUTSIDE_LABEL in labels
            assert not any("gpt-5-mini" in label for label in labels)
        finally:
            plt.close(fig)


class TestUnscoredTasksAreNotUnwinnable:
    """A task the completion dropped (no cells) is excluded, not pushed into the n=0 bucket."""

    def test_no_cell_task_is_excluded_from_both_buckets(self):
        chosen = {"scored": "a", "dropped": "a"}
        results = {"scored": {"a": {"pass": False}}, "dropped": {}}
        alloc = task_difficulty.allocation_by_difficulty(chosen, results, ["a"])
        assert alloc == {0: Counter({"a": 1})}
        counts, unsolved, unscored = task_difficulty.band_histogram(
            results, list(chosen), {"a": 1}, ["a"]
        )
        assert (counts, unsolved, unscored) == ({}, 1, 1)
        # Panel A's tasks (0 solvers, 1 unsolved) match panel B's denominator (1).
        assert sum(counts.values()) + unsolved == sum(sum(c.values()) for c in alloc.values())

    def test_annotations_state_both_denominators_and_the_outside_count(self):
        alloc = {1: Counter({"a": 2, "outside": 1})}
        ann = task_difficulty._annotations({1: 1}, 0, 3, alloc, ["a"])
        joined = " ".join(ann.subtitle_facts)
        assert "outside the inference-valid pool" in joined
        assert dict(ann.counts)["outside_pool"] == 1
        assert any("share one denominator" in note for note in ann.notes)

    def test_the_modal_claim_aggregates_an_outside_pool_mode(self):
        # The spread fact used the raw pick counter, so an outside-pool modal was NAMED even
        # though the stack and the limitation say outside-pool picks are not named.
        alloc = {1: Counter({"outside-model": 5, "a": 1})}
        ann = task_difficulty._annotations({1: 4}, 0, 2, alloc, ["a"])
        joined = " ".join(ann.subtitle_facts)
        assert "outside-model" not in joined
        assert task_difficulty._OUTSIDE_LABEL in joined

    def test_the_modal_claim_names_an_in_pool_mode(self):
        alloc = {1: Counter({"a": 5, "outside-model": 1})}
        ann = task_difficulty._annotations({1: 4}, 0, 2, alloc, ["a"])
        assert "mostly a" in " ".join(ann.subtitle_facts)


# ── F3: decision_audit states the dropped picks ───────────────────────────────────────


class TestDecisionAuditDenominator:
    def _audit(self) -> decision_audit.Audit:
        chosen = {"t1": "a", "t2": "outside", "t3": "b"}
        results = {
            "t1": {"a": {"pass": True}, "b": {"pass": True}},
            "t2": {"a": {"pass": True}, "b": {"pass": True}},
            "t3": {"a": {"pass": False}},
        }
        return decision_audit.build_audit(chosen, results, ["a", "b"])

    def test_outside_pool_pick_is_counted_not_dropped(self):
        audit = self._audit()
        assert audit.outside == 1
        assert audit.exact + audit.over + audit.under + audit.unwinnable + audit.outside == 3

    def test_annotations_name_the_outside_count_and_denominator(self):
        ann = decision_audit._annotations(self._audit())
        joined = " ".join(ann.subtitle_facts)
        assert "outside the inference-valid pool" in joined
        assert "1 picks outside" in joined
        assert dict(ann.counts)["outside_pool"] == 1
        assert any("denominator" in note for note in ann.notes)

    def test_budget_axis_states_its_base_and_the_excluded_picks(self):
        # The x-label used to say "share of scored decisions" while the bar's base excluded
        # both the outside-pool picks AND was smaller than the decision denominator.
        audit = self._audit()
        fig = plot_frame.new_figure(plot_frame.WIDE)
        try:
            ax = fig.subplots()
            decision_audit._draw_budget(ax, audit)
            label = ax.get_xlabel()
            assert "decidable" in label and "unwinnable" in label
            assert "outside-pool picks excluded" in label
        finally:
            plt.close(fig)

    def test_notes_reconcile_the_two_exact_rates(self):
        ann = decision_audit._annotations(self._audit())
        joined = " ".join(ann.notes)
        assert "budget bar's base" in joined
        assert "decidable set alone" in joined


# ── F5: kill_gate derives the default-to-preregistered multiple ───────────────────────


def _basis(label: str, router_cost: float) -> kill_gate.Basis:
    return kill_gate.Basis(
        label=label,
        n=100,
        diff_pp=0.0,
        lo_pp=-1.0,
        hi_pp=1.0,
        decision="non_inferior",
        b=4,
        c=0,
        router_cost=router_cost,
        baseline_cost=router_cost * 3.0,
    )


class TestKillGateMultipleIsDerived:
    def test_ratio_comes_from_the_drawn_rows(self):
        bases = [
            _basis("pre-registered kNN-semantic · completed (imputed)", 4.63),
            _basis("Session-Cascade — shipped default, NOT pre-registered", 28.21),
        ]
        ratio = kill_gate._default_vs_preregistered(bases)
        assert ratio == pytest.approx(28.21 / 4.63, rel=1e-6)
        assert "about 6.1 times" in kill_gate._goal_text(ratio)

    def test_no_hardcoded_four_in_the_module_goal(self):
        assert "four times" not in kill_gate.SPEC.goal
        assert "four times" not in kill_gate._goal_text(None)

    def test_missing_rows_fall_back_to_non_numeric_wording(self):
        assert kill_gate._default_vs_preregistered([]) is None
        assert "several times" in kill_gate._goal_text(None)

    def test_zero_preregistered_bill_cannot_divide(self):
        bases = [
            _basis("pre-registered kNN-semantic · completed (imputed)", 0.0),
            _basis("Session-Cascade — shipped default, NOT pre-registered", 5.0),
        ]
        assert kill_gate._default_vs_preregistered(bases) is None


# ── F6: cache_economics merges channel mirrors by canonical identity ──────────────────


class TestBilledShareMergesIdentity:
    def test_mirror_lanes_fold_into_one_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            cache_economics,
            "resolve_identity",
            lambda label: "base" if label.startswith("base") else label,
        )
        csv = tmp_path / "r.csv"
        csv.write_text(
            "lane,model,real_cost,estimated_cost,replicate\n"
            "base,base,1.0,2.0,0\n"
            "base-explabs,base,1.0,2.0,0\n"
            "other,other,4.0,8.0,0\n",
            encoding="utf-8",
        )
        shares = cache_economics.billed_shares(csv)
        assert shares["base"] == pytest.approx((0.5, 2))
        assert shares["other"] == pytest.approx((0.5, 1))

    def test_committed_mirror_rows_are_not_lost(self):
        from benchmark import config

        path = config.results_csv_path()
        if not path.exists():
            pytest.skip("no committed results.csv in this checkout")
        from benchmark.routing import integrity
        from benchmark.routing.model_universe import resolve_identity

        expected = Counter()
        for row in integrity.rep_zero_rows(path):
            try:
                estimated = float(row.get("estimated_cost") or 0.0)
            except ValueError:
                continue
            if estimated > 0:
                expected[resolve_identity(str(row.get("lane") or row["model"]))] += 1
        shares = cache_economics.billed_shares(path)
        assert {k: n for k, (_s, n) in shares.items()} == dict(expected)


# ── F9 / F13 / F14: model_validity criteria and the one evidenced definition ──────────


class TestModelValidityCriteria:
    def test_five_criteria_and_channel_is_the_fifth(self):
        assert len(model_validity.CRITERIA) == 5
        assert "triage pass" in model_validity.CRITERIA[1]
        assert "paid" in model_validity.CRITERIA[4]

    def test_channel_criterion_is_a_pass_fail_conjunct(self):
        assert len(_row("paid-model", channel=PAID).criteria) == 5
        assert _row("paid-model", channel=PAID).criteria[-1] is True
        assert _row("free-model", channel=FREE).criteria[-1] is False

    def test_valid_reason_names_the_triage_pass(self):
        # `classify` composes the reason; the census is the committed instance of it. A valid
        # row's reason must name the triage PASS (KEEP or EXCEPTION), not a bare "KEEP".
        reasons = [r.reason for r in model_validity.validity_census() if r.valid]
        assert reasons
        assert all("triage pass" in reason for reason in reasons)


class TestOneEvidencedDefinition:
    def test_is_evidenced_is_cells_positive(self):
        assert model_validity.is_evidenced(_row("m", cells=1)) is True
        assert model_validity.is_evidenced(_row("m", cells=0)) is False

    def test_census_reports_roster_and_evidenced_separately(self):
        from types import SimpleNamespace

        from benchmark.routing.figures import model_validity as fig

        census = fig.build(
            SimpleNamespace(validity=[_row("a"), _row("b", cells=0), _row("c", cells=0)])
        )
        assert census is not None
        assert census.evidenced == 1
        ann = fig._annotations(census)
        subtitle = " ".join(ann.subtitle_facts)
        assert "canonical identities named" in subtitle
        assert "evidenced" in subtitle
        assert dict(ann.counts)["evidenced"] == 1

    def test_committed_evidenced_set_is_not_narrowed(self):
        # The strict-arm alternative would drop deepseek-v4.1-flash (declared arm absent) and
        # move the evidenced count off the universe's 27. The labelled fallback keeps it.
        rows = {r.model: r for r in model_validity.validity_census()}
        assert rows["deepseek-v4.1-flash"].cells > 0
        assert sum(1 for r in rows.values() if r.cells > 0) == 27
        assert rows["glm-5.2"].cells == 84 and rows["glm-5.2"].valid is True

    def test_cell_count_definition_is_labelled(self):
        from types import SimpleNamespace

        from benchmark.routing.figures import model_validity as fig

        census = fig.build(SimpleNamespace(validity=[_row("a")]))
        assert census is not None
        limits = " ".join(fig._annotations(census).limitations)
        assert "sole cached arm" in limits
        assert "model_grid.png" in limits


# ── F15: context empty-match fallback ─────────────────────────────────────────────────


def _ctx(validity: list[ModelValidity] | None) -> ctxmod.RoutingContext:
    return ctxmod.RoutingContext(
        out_dir=Path("/tmp"),
        manifest=Path("/tmp/figures.json"),
        matrix={},
        completed={},
        imputed=None,
        tasks=[],
        rows=[],
        raw=None,
        models_by_price=["a", "b"],
        banner=None,
        by_strategy={},
        digest="d",
        validity=validity,
    )


class TestContextEmptyValidityFallback:
    def test_no_census_falls_back_to_the_enabled_order(self):
        assert _ctx(None).inference_valid_models == ["a", "b"]

    def test_present_census_with_no_valid_match_returns_empty(self):
        assert _ctx([_row("a", valid=False), _row("b", valid=False)]).inference_valid_models == []

    def test_present_census_returns_only_the_valid_models(self):
        got = _ctx([_row("a", valid=False), _row("b", valid=True)]).inference_valid_models
        assert got == ["b"]


class TestEmptyMatchFallbacksReturnEmptyNotRaw:
    """F15's pattern at the other three sites: present-but-empty must not re-admit invalid rows."""

    def test_cache_economics_present_census_with_no_valid_match_draws_nothing(self):
        ctx = _ctx([_row("a", valid=False), _row("b", valid=False)])
        assert cache_economics._drawn_models(ctx) == []

    def test_cache_economics_absent_census_keeps_the_enabled_order(self):
        assert cache_economics._drawn_models(_ctx(None)) == ["a", "b"]

    def test_model_validity_present_evidence_with_no_valid_match_filters_to_empty(self):
        from benchmark.routing.model_validity import Evidence

        evidence = Evidence(
            cells={},
            coverage={},
            corpus=0,
            triage={},
            capability={},
            identities={},
            providers={},
            listings={},
            paid_identities=frozenset(),
            live=frozenset(),
            enabled=frozenset(),
            floor=25,
        )
        raw = {"t": {"anything": {"default": {}}}}
        assert model_validity.filter_valid(raw, evidence) == {"t": {}}

    def test_model_validity_absent_evidence_returns_raw_unchanged(self):
        raw = {"t": {"anything": {"default": {}}}}
        assert model_validity.filter_valid(raw) is raw


# ── The 2026-09-14 vision-audit repairs: budget hues, never-chosen rows, column clarity ──


class TestDecisionBudgetIsCVDSafe:
    def test_the_four_budget_hues_are_not_red_green_adjacent(self) -> None:
        assert decision_audit._EXACT == "#0072B2"
        assert decision_audit._OVER == "#E69F00"
        assert decision_audit._UNDER == "#CC0066"
        assert decision_audit._FREE == "#9E9E9E"

    def test_a_row_the_router_never_chose_is_shaded_and_named(self) -> None:
        audit = decision_audit.Audit(
            models=["a", "b"],
            grid=np.array([[3.0, 0.0], [0.0, 0.0]]),
            exact=3,
            over=0,
            under=0,
            unwinnable=0,
            outside=0,
        )
        fig = plot_frame.new_figure(plot_frame.WIDE)
        try:
            ax = fig.subplots()
            decision_audit._draw_grid(ax, audit, 11.0)
            shaded = [p for p in ax.patches if p.get_facecolor()[:3] == to_rgb("#E0E0E0")]
            assert shaded, "an all-zero row was left reading as missing data"
            assert any("never chosen" in t.get_text() for t in ax.texts)
        finally:
            plt.close(fig)


class TestModelValidityCanvas:
    def test_the_two_count_columns_state_their_distinction(self) -> None:
        census = fig_validity._Census(rows=(_row("a"), _row("b", cells=0)), floor=25, corpus=500)
        fig = plot_frame.new_figure(plot_frame.SINGLE_TALL)
        try:
            ax = fig.subplots()
            fig_validity._draw_matrix(ax, census)
            joined = " ".join(t.get_text() for t in ax.get_xticklabels())
            assert "default-" in joined
            assert "distinct" in joined
        finally:
            plt.close(fig)

    def test_the_inference_valid_bar_is_not_green(self) -> None:
        census = fig_validity._Census(
            rows=(_row("valid"), _row("bad", valid=False)), floor=25, corpus=500
        )
        fig = plot_frame.new_figure(plot_frame.SINGLE)
        try:
            ax = fig.subplots()
            fig_validity._draw_reasons(ax, census)
            colours = [p.get_facecolor()[:3] for p in ax.patches]
            assert to_rgb("#0072B2") in colours
            assert to_rgb("#2E7D32") not in colours
        finally:
            plt.close(fig)
