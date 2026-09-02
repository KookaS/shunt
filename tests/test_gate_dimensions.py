"""The pre-registered multi-dimensional kill gate: its criterion, and its instrument validity.

The load-bearing tests are (a) the quality bar cannot be bought with an operational win, and
(b) the two mutants the new criterion exists to reject score strictly worse on a planted corpus.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmark.routing import gate_dimensions as gd
from benchmark.routing import gate_dimensions_control as control
from benchmark.routing.metrics import Axis

_AXES = tuple(g.axis.column for g in gd.OPERATIONAL_AXES)


@pytest.fixture(scope="module")
def rows():
    """The recorded strategy table — the gate's only input, read never repriced."""
    if not gd.SUMMARY_PATH.exists():
        pytest.skip(f"no recorded strategy table at {gd.SUMMARY_PATH}")
    return gd.load_arm_rows(gd.SUMMARY_PATH)


@pytest.fixture(scope="module")
def result():
    """The R0 two-sided validity verdict on the assembled gate."""
    return control.run_control()


def _arm(quality: float, **axes: float) -> dict[str, float]:
    row = {gd.QUALITY_COLUMN: quality}
    row.update({c: axes.get(c, 1.0) for c in _AXES})
    return row


def _table(router: dict[str, float], frontier: dict[str, float], constant: dict[str, float]):
    return {"R": router, "F": frontier, "C": constant}


def _adjudicate(rows, **kw):
    return gd.adjudicate(rows, router="R", baselines=("F", "C"), **kw)


def _load_mutant(source: str):
    """Import a mutated copy of the gate under its own module name — never touches the real one."""
    name = "_mutant_gate_dimensions"
    spec = importlib.util.spec_from_loader(name, loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(source, "<mutant>", "exec"), module.__dict__)  # noqa: S102
    finally:
        sys.modules.pop(name, None)
    return module


# ---------------------------------------------------------------------------
# Axis arithmetic
# ---------------------------------------------------------------------------


class TestAxisComparison:
    @pytest.mark.parametrize(
        ("router", "expected"),
        [(0.90, gd.BETTER), (0.98, gd.TIE), (1.00, gd.TIE), (1.02, gd.TIE), (1.10, gd.WORSE)],
    )
    def test_relative_tolerance_calls_small_differences_a_tie_in_both_directions(
        self, router, expected
    ):
        got = gd.compare_axis(Axis("c", "min"), router, 1.0, gd.OPERATIONAL_TOLERANCE)
        assert got.result == expected

    def test_a_zero_baseline_reports_a_relative_delta_that_agrees_with_its_verdict(self):
        # The unsigned +inf that used to be reported here printed a genuine improvement as
        # "(+inf%)" in the human report while the machine verdict said "better".
        got = gd.compare_axis(Axis("c", "min"), -1.0, 0.0, 0.05)
        assert got.result == gd.BETTER
        assert got.relative_delta < 0

    def test_a_zero_baseline_is_not_smoothed_into_a_tie(self):
        # A baseline that costs nothing and a router that costs something is a real loss;
        # reporting it as a tie is how a divide-by-zero becomes a pass.
        assert gd.compare_axis(Axis("c", "min"), 1.0, 0.0, 0.05).result == gd.WORSE
        assert gd.compare_axis(Axis("c", "min"), 0.0, 0.0, 0.05).result == gd.TIE


# ---------------------------------------------------------------------------
# The criterion
# ---------------------------------------------------------------------------


class TestCriterion:
    def test_passes_only_when_it_dominates_both_baselines(self):
        rows = _table(_arm(90.0, **dict.fromkeys(_AXES, 0.5)), _arm(90.0), _arm(90.0))
        assert _adjudicate(rows).verdict == gd.PASS

    def test_a_win_against_the_frontier_alone_is_not_a_pass(self):
        # The Scrouting shape: beats an expensive frontier model, loses to a policy that
        # does no routing. The old single-baseline gate could not express this at all.
        rows = _table(
            _arm(90.0, **dict.fromkeys(_AXES, 2.0)),
            _arm(90.0, **dict.fromkeys(_AXES, 10.0)),
            _arm(90.0, **dict.fromkeys(_AXES, 1.0)),
        )
        verdict = _adjudicate(rows)
        assert verdict.verdict == gd.FAIL
        assert "vs C" in verdict.reason

    def test_tying_a_policy_that_does_no_routing_earns_nothing(self):
        rows = _table(_arm(90.0), _arm(90.0), _arm(90.0))
        verdict = _adjudicate(rows)
        assert verdict.verdict == gd.FAIL
        assert "no axis strictly better" in verdict.reason

    def test_the_quality_bar_cannot_be_bought_with_an_operational_win(self):
        # THE anti-laundering test. The router is better on EVERY operational axis by a mile
        # and inferior on quality beyond the pre-registered margin. It must still FAIL.
        rows = _table(
            _arm(90.0 - gd.QUALITY_MARGIN_PP - 0.01, **dict.fromkeys(_AXES, 0.1)),
            _arm(90.0),
            _arm(90.0),
        )
        verdict = _adjudicate(rows)
        assert verdict.verdict == gd.FAIL
        assert "quality inferior" in verdict.reason

    def test_a_quality_win_does_not_substitute_for_an_operational_win(self):
        # Shunt's thesis is "the cheapest model that can do the job" — a cost/effort claim.
        # Crediting a quality win here would let a router pass by escalating more.
        rows = _table(_arm(99.0), _arm(90.0), _arm(90.0))
        assert _adjudicate(rows).verdict == gd.FAIL

    def test_worse_on_one_axis_defeats_better_on_three(self):
        router = _arm(90.0, **dict.fromkeys(_AXES, 0.5))
        router["sessions_p95"] = 4.0
        rows = _table(router, _arm(90.0), _arm(90.0))
        verdict = _adjudicate(rows)
        assert verdict.verdict == gd.FAIL
        assert "sessions_p95" in verdict.reason


# ---------------------------------------------------------------------------
# Adjudicability — a gate that cannot be run has no verdict, it has a gap
# ---------------------------------------------------------------------------


class TestAdjudicability:
    def test_a_missing_axis_value_is_untested_not_a_free_zero(self):
        router = _arm(90.0, **dict.fromkeys(_AXES, 0.5))
        del router["cost_cv"]
        verdict = _adjudicate(_table(router, _arm(90.0), _arm(90.0)))
        assert verdict.verdict == gd.UNTESTED
        assert "cost_cv" in verdict.reason
        assert verdict.comparisons == ()

    @pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
    def test_a_non_finite_cell_is_missing_not_a_tie(self, bad, tmp_path):
        # Every comparison against a NaN is False, so a NaN would sail through as a TIE and let
        # a router pass on an axis nobody measured — the exact silent-accept a blank cell is
        # refused for. Checked at BOTH doors: the CSV loader and the adjudicator's own guard.
        csv_path = tmp_path / "s.csv"
        header = ["strategy", gd.QUALITY_COLUMN, *_AXES]
        csv_path.write_text(
            ",".join(header)
            + "\n"
            + f"R,90.0,{bad},0.5,0.5,0.5\n"
            + "F,90.0,1,1,1,1\n"
            + "C,90.0,1,1,1,1\n",
            encoding="utf-8",
        )
        loaded = gd.load_arm_rows(csv_path)
        assert "TotalCost_cacheaware" not in loaded["R"]
        assert _adjudicate(loaded).verdict == gd.UNTESTED

        router = _arm(90.0, **dict.fromkeys(_AXES, 0.5))
        router["cost_cv"] = float(bad)
        verdict = _adjudicate(_table(router, _arm(90.0), _arm(90.0)))
        assert verdict.verdict == gd.UNTESTED
        assert "cost_cv" in verdict.reason

    def test_a_missing_arm_is_untested(self):
        rows = {"R": _arm(90.0), "F": _arm(90.0)}
        assert gd.adjudicate(rows, router="R", baselines=("F", "C")).verdict == gd.UNTESTED

    def test_a_blocker_forces_untested_but_still_publishes_which_way_it_leaned(self):
        rows = _table(_arm(90.0, **dict.fromkeys(_AXES, 0.5)), _arm(90.0), _arm(90.0))
        verdict = _adjudicate(rows, blockers=("coverage floor tripped",))
        assert verdict.verdict == gd.UNTESTED
        assert verdict.provisional == gd.PASS
        # The provisional reading is never the verdict, and the artifact keeps them apart.
        payload = gd.verdict_payload(verdict)
        assert payload["verdict"] == gd.UNTESTED
        assert payload["provisional_verdict"] == gd.PASS


# ---------------------------------------------------------------------------
# The recorded corpus — what the OLD gate concluded vs what the NEW one concludes
# ---------------------------------------------------------------------------


class TestRecordedCorpus:
    def test_the_old_cost_only_criterion_would_have_passed_the_router(self, rows):
        # The OLD gate, restated in three lines: quality non-inferior at the 5pp margin
        # against fixed-frontier, and a cache-aware cost ratio below 1. Pinned so the change
        # can never be described as "no practical difference" — it flips this arm's verdict.
        router, frontier = rows[gd.DEFAULT_ROUTER_ARM], rows["Always-Frontier"]
        quality_ok = (
            router[gd.QUALITY_COLUMN] - frontier[gd.QUALITY_COLUMN] >= -gd.QUALITY_MARGIN_PP
        )
        cheaper = router["TotalCost_cacheaware"] / frontier["TotalCost_cacheaware"] < 1.0
        assert quality_ok and cheaper

    @pytest.mark.parametrize("arm", ["kNN-semantic-cascade", "Session-Cascade", "kNN-semantic"])
    def test_the_new_criterion_fails_every_router_configuration_on_this_corpus(self, rows, arm):
        if arm not in rows:
            pytest.skip(f"{arm} absent from the recorded table")
        verdict = gd.adjudicate(rows, router=arm)
        # No blockers passed in, so this is the criterion's own reading of the recorded data.
        assert verdict.verdict == gd.FAIL, verdict.reason

    def test_the_shipped_verdict_artifact_is_untested_and_says_why(self):
        path = gd.VERDICT_PATH
        if not path.exists():
            pytest.skip(f"no verdict artifact at {path}")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["verdict"] == gd.UNTESTED
        assert payload["blockers"]

    def test_the_verdict_artifact_is_byte_deterministic(self, rows, tmp_path):
        verdict = gd.adjudicate(rows, router=gd.DEFAULT_ROUTER_ARM)
        first, second = tmp_path / "a.json", tmp_path / "b.json"
        gd.write_verdict(gd.verdict_payload(verdict), first)
        gd.write_verdict(gd.verdict_payload(verdict), second)
        assert first.read_bytes() == second.read_bytes()


# ---------------------------------------------------------------------------
# Instrument validity — the gate emits a verdict about whether a signal exists
# ---------------------------------------------------------------------------


class TestInstrumentValidity:
    def test_the_gate_clears_both_r0_controls(self, result):
        assert result.admissible, result.reason
        assert result.positive_passed
        # `null_at_chance` used to be asserted here on its own and could not fail: the old null
        # permuted labels around a FROZEN prediction vector, which is at chance by arithmetic
        # whatever produced the vector. The claim that carries weight now is that the same null
        # REJECTS a degenerate gate — TestTheNullCanFail — and that the real gate's null is
        # responsive and signal-dependent rather than merely near 0.5.
        assert result.null_at_chance, result.reason
        assert result.numbers["null_responsive"]
        assert result.numbers["null_signal_dependent"]
        assert abs(result.shuffled_score - control.CHANCE_LEVEL) <= result.chance_band

    def test_the_planted_corpus_separates_the_two_mutants_this_change_rejects(self, result):
        # A cost-only gate is the OLD criterion; a frontier-only gate is the OLD baseline set.
        # Both must score materially worse, or the positive leg certified nothing new.
        assert result.numbers["cost_only_mutant"] < result.positive_score
        assert result.numbers["frontier_only_mutant"] < result.positive_score

    @pytest.mark.parametrize(
        ("anchor", "mutation"),
        [
            # Each pair deletes one clause of the criterion. All three used to score 0.8375-0.90
            # and still read ADMISSIBLE, because "clearly above chance" is the wrong question for
            # a corpus with no ambiguous scenarios. The perfect-recovery assertion is what makes
            # them loud, so these are the regression tests for the control's POWER.
            (
                "return not self.worse_axes and bool(self.better_axes)",
                "return not self.worse_axes",
            ),
            (
                "        better, worse = raw < -tolerance, raw > tolerance",
                "        better, worse = raw < tolerance, raw > tolerance",
            ),
            (
                "        quality_non_inferior=delta_pp >= -quality_margin_pp,",
                "        quality_non_inferior=True,",
            ),
        ],
    )
    def test_the_control_catches_a_deleted_clause_of_the_criterion(self, anchor, mutation):
        source = Path(gd.__file__).read_text(encoding="utf-8")
        assert anchor in source, "mutation anchor drifted; the sweep is no longer testing it"
        module = _load_mutant(source.replace(anchor, mutation, 1))
        corpus, labels, modes = control.build_corpus(240, seed=0)
        predicted = np.array(
            [
                module.adjudicate(
                    rows,
                    router="router",
                    baselines=("frontier", "constant"),
                    axes=module.OPERATIONAL_AXES,
                ).verdict
                == module.PASS
                for rows in corpus
            ]
        )
        with pytest.raises(RuntimeError, match="UNAMBIGUOUSLY planted"):
            control._assert_every_planted_scenario_is_recovered(predicted, labels, modes)

    def test_the_unmutated_gate_recovers_every_planted_scenario(self):
        corpus, labels, modes = control.build_corpus(240, seed=0)
        control._assert_every_planted_scenario_is_recovered(
            control.gate_predictions(corpus), labels, modes
        )

    def test_the_control_refuses_a_corpus_that_does_not_exercise_the_axes(self):
        # A corpus of nothing but quality-planted losses cannot separate the mutants, and the
        # control must SAY so rather than certify the gate on it.
        corpus, labels, _ = control.build_corpus(8, seed=3)
        flat = [
            {
                arm: dict.fromkeys(_AXES, 1.0) | {gd.QUALITY_COLUMN: 90.0}
                for arm in ("router", "frontier", "constant")
            }
            for _ in corpus
        ]
        full = control.balanced_accuracy(control.gate_predictions(flat), labels)
        with pytest.raises(RuntimeError, match="did not exercise"):
            control._assert_axes_are_load_bearing(flat, labels, full)


def _degenerate(name: str):
    """A gate that is broken by construction, for the null to reject."""
    if name in ("frozen_pass", "frozen_fail"):
        value = name == "frozen_pass"
        return lambda corpus: np.full(len(corpus), value, dtype=bool)
    if name == "inverted":
        return lambda corpus: ~control.gate_predictions(corpus)
    rng = np.random.default_rng(11)
    return lambda corpus: rng.random(len(corpus)) < 0.5


class TestTheNullCanFail:
    """A null that cannot reject a degenerate instrument is not a null. This is that test."""

    @pytest.mark.parametrize("name", ["frozen_pass", "frozen_fail", "inverted", "coinflip"])
    def test_a_degenerate_gate_is_rejected_by_the_null_leg(self, name):
        # THE ACCEPTANCE TEST. Measured against the previous control, all four of these returned
        # `null_at_chance=True` — the null could not fail, so its clearance said nothing. Each
        # one is now caught by a different clause: the two frozen gates are not RESPONSIVE (they
        # return the same verdicts after the signal is destroyed), and the inverted and coinflip
        # gates are not SIGNAL-DEPENDENT (their score does not survive the comparison with the
        # destroyed-signal re-run).
        result = control.run_control(n_scenarios=120, n_perm=100, predict=_degenerate(name))
        assert not result.null_at_chance, result.reason
        assert not result.admissible

    def test_the_real_gate_is_not_rejected_by_the_same_null(self):
        result = control.run_control(n_scenarios=120, n_perm=100)
        assert result.null_at_chance, result.reason
        assert result.admissible, result.reason


class TestTheBlockedBranchIsScored:
    """UNTESTED — the word the shipped artifact carries — comes from a branch the control enters."""

    def test_a_planted_win_is_forced_to_untested_by_each_precondition(self, tmp_path):
        rows, _ = control._scenario(np.random.default_rng(3), True)
        assert control.chain_verdict(rows, tmp_path).verdict == gd.PASS
        for kwargs in ({"coverage_tripped": True}, {"admissible": False}):
            blocked = control.chain_verdict(rows, tmp_path, **kwargs)
            assert blocked.verdict == gd.UNTESTED
            assert blocked.blockers
            assert blocked.provisional == gd.PASS

    def test_the_assertion_fires_when_the_coverage_precondition_stops_blocking(
        self, tmp_path, monkeypatch
    ):
        # Non-vacuity for the blocked-branch leg: neuter the precondition and the control must
        # say so. Without this the leg could pass by never being able to fail.
        rows, _ = control._scenario(np.random.default_rng(3), True)
        monkeypatch.setattr(gd, "coverage_blockers", lambda path: ())
        with pytest.raises(RuntimeError, match="coverage floor"):
            control._assert_the_blocked_branch_is_untested(rows, tmp_path)

    def test_the_cli_leg_fires_when_the_exit_code_contradicts_the_criterion(self, tmp_path):
        win, _ = control._scenario(np.random.default_rng(3), True)
        loss, _ = control._scenario(np.random.default_rng(4), False)
        control._assert_the_cli_agrees(win, loss, tmp_path)
        with pytest.raises(RuntimeError, match="CLI disagrees"):
            control._assert_the_cli_agrees(loss, win, tmp_path)


class TestCoveragePreconditionIsDerived:
    """The precondition follows from the census, not from a hand-editable boolean."""

    def test_the_committed_offline_census_is_self_consistent(self):
        if not gd.OFFLINE_VERDICT_PATH.exists():
            pytest.skip("no offline verdict artifact")
        census = gd.coverage_census(json.loads(gd.OFFLINE_VERDICT_PATH.read_text(encoding="utf-8")))
        assert census.problems == ()
        assert census.tripped

    def test_flipping_the_stored_flag_does_not_clear_the_floor(self, tmp_path):
        # THE F5 REGRESSION. `tripped` used to be the whole precondition: one character moved
        # this gate UNTESTED -> FAIL, in a JSON no job regenerates.
        payload = json.loads(gd.OFFLINE_VERDICT_PATH.read_text(encoding="utf-8"))
        payload["coverage"]["tripped"] = False
        forged = tmp_path / "kill_gate_verdict.json"
        forged.write_text(json.dumps(payload), encoding="utf-8")
        blockers = gd.coverage_blockers(forged)
        assert any("coverage floor tripped" in b for b in blockers)
        assert any("edited away from the data" in b for b in blockers)

    def test_a_hand_edited_coverage_summary_is_caught(self, tmp_path):
        payload = json.loads(gd.OFFLINE_VERDICT_PATH.read_text(encoding="utf-8"))
        payload["coverage"]["control_coverage"] = 0.99
        payload["coverage"]["router_coverage"] = 0.99
        payload["coverage"]["tripped"] = False
        forged = tmp_path / "kill_gate_verdict.json"
        forged.write_text(json.dumps(payload), encoding="utf-8")
        problems = gd.coverage_blockers(forged)
        assert any("control_coverage" in p for p in problems)
        assert any("router_coverage" in p for p in problems)

    def test_a_census_with_no_counts_is_a_blocker_not_a_pass(self, tmp_path):
        forged = tmp_path / "kill_gate_verdict.json"
        forged.write_text(json.dumps({"n": 100, "coverage": {"tripped": False}}), encoding="utf-8")
        assert gd.coverage_blockers(forged)


class TestVerdictArtifactIntegrity:
    """A tracked verdict is checkable against the corpus it summarises, or it is just a claim."""

    def test_the_committed_verdict_re_derives_from_the_committed_corpus(self):
        if not gd.SUMMARY_PATH.exists() or not gd.VERDICT_PATH.exists():
            pytest.skip("no committed corpus / verdict artifact")
        assert gd.verdict_integrity_problems() == ()

    @pytest.mark.parametrize(
        ("field", "value"),
        [("provisional_verdict", "PASS"), ("verdict", "PASS"), ("blockers", [])],
    )
    def test_a_falsified_headline_field_is_caught(self, field, value, tmp_path):
        # THE F4 REGRESSION. Falsifying the artifact to a PASS with a fabricated cost win left
        # the whole suite green at exit 0, and a verdict artifact is what a reader quotes.
        payload = json.loads(gd.VERDICT_PATH.read_text(encoding="utf-8"))
        payload[field] = value
        forged = tmp_path / "v.json"
        forged.write_text(json.dumps(payload), encoding="utf-8")
        problems = gd.verdict_integrity_problems(forged)
        assert any(field in p for p in problems), problems

    def test_a_fabricated_axis_number_is_caught(self, tmp_path):
        payload = json.loads(gd.VERDICT_PATH.read_text(encoding="utf-8"))
        payload["comparisons"][0]["axes"][0]["router"] = 0.01
        payload["comparisons"][0]["axes"][0]["result"] = gd.BETTER
        forged = tmp_path / "v.json"
        forged.write_text(json.dumps(payload), encoding="utf-8")
        assert any("comparisons" in p for p in gd.verdict_integrity_problems(forged))

    def test_an_invented_field_is_caught(self, tmp_path):
        payload = json.loads(gd.VERDICT_PATH.read_text(encoding="utf-8"))
        payload["headline"] = "the router wins"
        forged = tmp_path / "v.json"
        forged.write_text(json.dumps(payload), encoding="utf-8")
        assert any("headline" in p for p in gd.verdict_integrity_problems(forged))
