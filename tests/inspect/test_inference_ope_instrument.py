"""The blocking instrument-validity gate for the inference off-policy estimates.

Both legs, both controls, per estimator: the assembled pipeline must recover a planted signal
AND collapse to chance when that signal is destroyed, or no number it produces is quotable.
"""

from __future__ import annotations

import dataclasses
import random
import statistics

import pytest

from shunt.analysis.admissibility import admissibility_verdict
from shunt.analysis.ope import (
    IDENTIFIED,
    NOT_IDENTIFIED,
    estimate_policy_value,
    rows_from_records,
)
from shunt.db.store import OutcomeStore, SessionProvenance
from shunt.inspect.inference import estimators as inst

_SEEDS = range(20)


@pytest.fixture(scope="module")
def certificates() -> tuple[inst.EstimatorCertificate, ...]:
    """The pinned-seed verdicts, computed once — the whole gate runs inside this call."""
    return inst.certify_instrument()


@pytest.fixture(scope="module")
def records() -> dict[str, list[dict[str, object]]]:
    return inst.control_records()


class TestTheAssembledPipelineIsCertified:
    def test_every_leg_and_estimator_clears_both_controls(self, certificates) -> None:
        # Per-estimator verdicts, never one pooled number: an estimator that passes DR and
        # fails SNIPS is a diagnosis about weight dispersion, not a blind spot.
        assert len(certificates) == len(inst.ESTIMATORS) * 2
        failed = [c.key for c in certificates if not c.admissibility.admissible]
        assert failed == [], "\n".join(
            c.admissibility.reason for c in certificates if not c.admissibility.admissible
        )
        assert inst.overall_admissibility(certificates).admissible is True

    def test_the_positive_control_clears_by_a_real_margin(self, certificates) -> None:
        for certificate in certificates:
            result = certificate.admissibility
            margin = (result.positive_score - result.chance_level) / result.chance_band
            assert margin > 1.0, f"{certificate.key}: positive control at {margin:.2f} bands"
            assert result.positive_passed

    def test_the_null_leg_genuinely_collapses(self, certificates) -> None:
        # THE point of this test file. A positive-only gate is the precise failure the
        # instrument-validity rule exists to catch: an instrument that manufactures signal from
        # noise passes a positive control and is worthless.
        for certificate in certificates:
            result = certificate.admissibility
            drift = abs(result.shuffled_score - result.chance_level)
            assert result.null_at_chance
            assert drift <= result.chance_band
            # The destroyed-signal score is not merely "inside the band": it sits far below the
            # planted one, so the band is not doing the work a real collapse should do.
            assert result.shuffled_score < result.positive_score - result.chance_band

    def test_the_null_leg_is_not_vacuous(self, certificates) -> None:
        # Feed the adjudicator the POSITIVE score as if the shuffle had been a no-op. If the
        # permutation ever stopped destroying the signal, this is the shape the gate would see,
        # and it must be rejected — otherwise `null_at_chance` is true by construction.
        for certificate in certificates:
            result = certificate.admissibility
            leaked = admissibility_verdict(
                result.positive_score,
                result.positive_score,
                chance_level=result.chance_level,
                chance_band=result.chance_band,
            )
            assert not leaked.admissible
            assert not leaked.null_at_chance

    def test_a_pipeline_that_cannot_recover_the_planted_signal_is_refused(self, records) -> None:
        # The gate has teeth: drop the routing agreement encoding and the multi-arm rows reach a
        # BINARY estimator that sees one arm. Every score collapses and the leg is inadmissible.
        broken = inst._certify_leg(
            records[inst.ROUTING],
            lambda rows: list(rows),
            inst.ROUTING,
            seed=0,
            n_shuffles=20,
        )
        assert all(not c.admissibility.admissible for c in broken)
        assert not inst.overall_admissibility(broken).admissible


class TestTheControlEntersAtTheFront:
    def test_a_regressed_live_filter_would_fail_the_gate(self, records) -> None:
        # The `bench:` decoys are written one-for-one with the live rows and carry the inverted
        # signal. This is what the leg would score if `routing_ope_rows` stopped excluding them:
        # the planted signal cancels and every estimator is refused.
        flip = {"success": "failure", "failure": "success"}
        live = records[inst.ROUTING]
        leaked = live + [
            {**row, "session_id": f"bench:{i}", "outcome": flip[str(row["outcome"])]}
            for i, row in enumerate(live)
        ]
        certificates = inst._certify_leg(
            leaked, inst.routing_agreement_rows, inst.ROUTING, seed=1, n_shuffles=40
        )
        assert all(not c.admissibility.admissible for c in certificates)

    def test_the_corpus_round_trips_through_the_production_accessors(self, records) -> None:
        routing = records[inst.ROUTING]
        escalation = records[inst.ESCALATION]
        assert len(routing) == 400
        assert len(escalation) == 400
        # The `bench:` decoys carry the INVERTED signal and a live-looking propensity. Their
        # absence here is the accessor's live/seed discrimination, under test.
        assert not [r for r in routing if str(r["session_id"]).startswith("bench:")]
        # The propensity column round-trip, the provenance JSON round-trip, and the epsilon the
        # randomization test is derived from.
        assert {round(float(r["selection_propensity"]), 4) for r in routing} == {0.1, 0.8}
        assert {len(r["candidate_model_scores"]) for r in routing} == {3}
        assert {r["epsilon"] for r in routing} == {0.3}
        assert {round(float(r["propensity"]), 4) for r in escalation} == {0.3, 0.7}
        # tier2 wins over tier1: every control row's tier1 is deliberately "failure".
        assert {r["outcome"] for r in routing} == {"success", "failure"}

    @pytest.mark.parametrize(
        ("regression", "mutate"),
        [
            # tier1 wins over tier2: every control row's tier1 is "failure", so the planted
            # reward vanishes and the positive control lands exactly on chance.
            ("tier2-coalescing", lambda r: {**r, "outcome": "failure"}),
            # the epsilon disappears from the provenance: `_is_randomized_over_arms` can no
            # longer derive randomization, every row is excluded, and nothing is scoreable.
            ("epsilon-provenance", lambda r: {k: v for k, v in r.items() if k != "epsilon"}),
        ],
    )
    def test_a_broken_front_end_is_refused(self, records, regression, mutate) -> None:
        certificates = inst._certify_leg(
            [mutate(row) for row in records[inst.ROUTING]],
            inst.routing_agreement_rows,
            inst.ROUTING,
            seed=1,
            n_shuffles=20,
        )
        assert all(not c.admissibility.admissible for c in certificates), regression

    def test_the_two_accessors_do_not_leak_into_each_other(self, records) -> None:
        assert not [r for r in records[inst.ROUTING] if "escalation_exploration" in r]
        assert not [r for r in records[inst.ESCALATION] if "selection_propensity" in r]

    def test_the_control_rows_are_identified_end_to_end(self, records) -> None:
        # The bootstrap, the SESSION clustering and the contrast — the parts `_dr_point` skips —
        # exercised on the same corpus, so the certified score is not the only thing proven.
        rows = inst.routing_agreement_rows(rows_from_records(records[inst.ROUTING]))
        estimate = estimate_policy_value(rows)
        assert estimate.status == IDENTIFIED, estimate.reason
        assert estimate.n_clusters == 400  # sessions, not checkpoints
        assert estimate.dr_estimate is not None
        assert estimate.ci_low < estimate.dr_estimate < estimate.ci_high
        # The contrast is V(top-scored arm) - V(the reduction's complement), and that complement
        # is "some other arm", not a deployable routing policy. Asserted here as a signal that
        # the paired bootstrap ran and separated, never quoted as a policy comparison.
        assert estimate.contrast_excludes_zero

    def test_the_escalation_cluster_is_the_session(self, records) -> None:
        # The corpus deliberately shares 3 checkpoint ids across 400 sessions — below the
        # 5-cluster floor. If the clustering key ever regresses to the checkpoint, this refuses.
        estimate = estimate_policy_value(rows_from_records(records[inst.ESCALATION]))
        assert estimate.status == IDENTIFIED, estimate.reason
        assert estimate.n_clusters == 400
        assert len({r["checkpoint_id"] for r in records[inst.ESCALATION]}) == 3


class TestTheOrchestratorIsInsideTheGate:
    """`estimate_policy_value` is what a figure quotes, so both controls run through it."""

    @pytest.mark.parametrize("leg", [inst.ROUTING, inst.ESCALATION])
    def test_the_destroyed_signal_corpus_is_identified_but_at_chance(self, records, leg) -> None:
        # The half of the orchestrator the certified score cannot reach: the interval and the
        # paired contrast. On shuffled labels the estimator must still IDENTIFY (the rows carry
        # both arms and 400 clusters) while landing at chance with a contrast that does NOT
        # exclude zero. An estimator whose contrast separates on destroyed labels manufactures
        # the decision it exists to inform.
        encode = inst.routing_agreement_rows if leg == inst.ROUTING else list
        live = encode(rows_from_records(records[leg]))
        chance = statistics.fmean([row.reward or 0.0 for row in live])
        null_records = inst._permuted(records[leg], random.Random(7))
        null = estimate_policy_value(encode(rows_from_records(null_records)))
        assert null.status == IDENTIFIED, null.reason
        assert null.n_clusters == 400
        assert null.dr_estimate is not None
        assert null.ci_low < null.dr_estimate < null.ci_high
        # Deliberately NOT "the interval covers chance": that is a 90%-coverage statement and
        # would flake by design. The destroyed-signal estimate must land within one interval
        # width of chance, which a real leak (a shift of many widths) could not do.
        assert abs(null.dr_estimate - chance) <= null.ci_high - null.ci_low
        assert not null.contrast_excludes_zero

    def test_an_intermittent_leak_would_show_in_the_replicate_spread(self, certificates, records):
        # The reported null is a MEDIAN, which a leak firing on a minority of permutations could
        # hide. The whole held-out replicate spread is checked here instead: no single one may
        # drift beyond twice the band.
        for leg, encode in ((inst.ROUTING, inst.routing_agreement_rows), (inst.ESCALATION, list)):
            rows = encode(rows_from_records(records[leg]))
            chance = statistics.fmean([row.reward or 0.0 for row in rows])
            deviations = inst._null_deviations(records[leg], encode, chance, 99, 25)
            for certificate in (c for c in certificates if c.leg == leg):
                worst = max(deviations[certificate.estimator])
                assert worst <= 2.0 * certificate.admissibility.chance_band, certificate.key


class TestTheBandIsEmpirical:
    def test_chance_is_the_computed_marginal_reward_rate(self, certificates, records) -> None:
        rewards = {
            inst.ROUTING: [r["outcome"] == "success" for r in records[inst.ROUTING]],
            inst.ESCALATION: [r["outcome"] == "success" for r in records[inst.ESCALATION]],
        }
        for certificate in certificates:
            measured = statistics.fmean(float(x) for x in rewards[certificate.leg])
            assert certificate.admissibility.chance_level == pytest.approx(measured, abs=1e-9)
        # Computed per leg, never a shared constant: the two legs pay different marginal rates.
        levels = {c.leg: c.admissibility.chance_level for c in certificates}
        assert levels[inst.ROUTING] != pytest.approx(levels[inst.ESCALATION], abs=0.05)

    def test_the_band_is_a_positive_finite_width(self, certificates) -> None:
        for certificate in certificates:
            assert 0.0 < certificate.admissibility.chance_band < 1.0
            assert certificate.n_shuffles == 200
            assert certificate.seed == 20260818

    @pytest.mark.parametrize("seed", _SEEDS)
    def test_the_verdict_does_not_flip_on_reseed(self, seed) -> None:
        # A gate that flips on reseed is not a gate. Every seed rebuilds the whole corpus, the
        # band and both controls.
        certificates = inst.certify_instrument(seed=seed)
        assert inst.overall_admissibility(certificates).admissible is True
        assert all(c.admissibility.admissible for c in certificates)


class TestTheReceiptIsStructural:
    def test_admissibility_is_the_required_first_field(self) -> None:
        # A defaulted field is exactly how the gate went unwired the first time.
        for klass in (inst.CertifiedEstimates, inst.EstimatorCertificate):
            first = dataclasses.fields(klass)[0]
            assert first.name == "admissibility"
            assert first.default is dataclasses.MISSING
            assert first.default_factory is dataclasses.MISSING

    def test_an_inadmissible_instrument_raises_before_anything_is_drawn(self) -> None:
        verdict = admissibility_verdict(0.05, 0.4, chance_level=0.0, chance_band=0.1)
        estimates = inst.CertifiedEstimates(
            admissibility=verdict,
            certificates=(),
            routing=estimate_policy_value([]),
            escalation=estimate_policy_value([]),
        )
        with pytest.raises(inst.InstrumentInadmissibleError, match="INADMISSIBLE"):
            estimates.require_admissible()

    def test_the_headline_is_one_quotable_line(self, certificates) -> None:
        headline = inst.overall_admissibility(certificates).headline
        assert headline.startswith("INSTRUMENT ADMISSIBLE:")
        assert "\n" not in headline


class TestTheAccessorsAreOutcomeValueBlind:
    """The band permutes records in memory; that is only sound if a read never reads the value."""

    @pytest.mark.parametrize("outcomes", [("success", "failure"), ("failure", "success")])
    def test_a_read_returns_every_row_whatever_the_outcome_says(self, tmp_path, outcomes) -> None:
        # If `routing_ope_rows` or `escalation_exploration_rows` ever branched on the outcome
        # STRING — a WHERE clause, a coalescing rule — permuting outcomes in memory would stop
        # being equivalent to permuting them in the store, and the band would certify a
        # population the positive control never had. Both accessors must be blind to the value.
        store = OutcomeStore(db_path=str(tmp_path / f"{outcomes[0]}.db"))
        for index in range(10):
            outcome = outcomes[index % 2]
            provenance: dict[str, object] = {
                "candidate_model_scores": {"control-a": 1.0, "control-b": 0.0},
                "epsilon": 0.3,
                "escalation_exploration": {
                    "checkpoint_id": "chk-0",
                    "action": "hold",
                    "propensity": 0.7,
                    "epsilon": 0.3,
                },
            }
            store.store_session(
                session_id=f"live:{index}",
                prompt_text="value blindness",
                embedding=None,
                model_chosen="control-a",
                cost=0.0,
                cache_stats={},
                duration=0.0,
                timestamp=f"2020-01-01T00:00:{index:02d}+00:00",
                decision_provenance=provenance,
                provenance=SessionProvenance(selection_propensity=0.8),
            )
            store.store_outcome(f"live:{index}", "failure", 1.0, outcome, 1.0)
        expected = [outcomes[i % 2] for i in range(10)]
        assert [r["outcome"] for r in store.routing_ope_rows()] == expected
        assert [r["outcome"] for r in store.escalation_exploration_rows()] == expected


class TestTheRefusalPathIsFirstClass:
    def test_a_deterministically_logged_store_is_refused_not_crashed(self, tmp_path) -> None:
        # What the real rig measures today: every propensity 1.0, no epsilon, one checkpoint.
        # NOT_IDENTIFIED with a reason is the designed outcome — the figure ships the refusal.
        store = OutcomeStore(db_path=str(tmp_path / "rig.db"))
        for index in range(12):
            store.store_session(
                session_id=f"live:{index}",
                prompt_text="deterministic",
                embedding=None,
                model_chosen="control-a",
                cost=0.0,
                cache_stats={},
                duration=0.0,
                timestamp=f"2020-01-01T00:00:{index:02d}+00:00",
                decision_provenance={"candidate_model_scores": {"control-a": 1.0}},
                provenance=SessionProvenance(selection_propensity=1.0),
            )
            store.store_outcome(f"live:{index}", "success", 1.0, "success", 1.0)
        routing, escalation = inst.live_estimates(store)
        for estimate in (routing, escalation):
            assert estimate.status == NOT_IDENTIFIED
            assert estimate.reason
            assert estimate.dr_estimate is None
        assert routing.n_excluded == 12

    def test_certify_returns_the_refusal_once_the_instrument_clears(self, tmp_path) -> None:
        store = OutcomeStore(db_path=str(tmp_path / "empty.db"))
        certified = inst.certify(store, n_sessions=200, n_shuffles=40)
        assert certified.admissibility.admissible
        assert certified.routing.status == NOT_IDENTIFIED
        assert certified.escalation.status == NOT_IDENTIFIED
        pairs = [(leg, e) for leg in (inst.ROUTING, inst.ESCALATION) for e in inst.ESTIMATORS]
        assert all(certified.quotable(leg, estimator) for leg, estimator in pairs)
        # The quotable line is the adjudicator's verdict plus the units it is stated in — the
        # aggregate runs in band-normalised units and a note that omitted that would mislead.
        assert certified.headline.startswith(certified.admissibility.headline)
        assert "band-normalised" in certified.headline
        assert "\n" not in certified.headline
