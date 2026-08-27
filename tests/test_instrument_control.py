"""The routing instrument's positive control and destroyed-signal null, both directions.

A gate that passes everything is the failure mode this file exists to prevent, so every
deliberately-broken front end below MUST be rejected, and the intact one MUST pass.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from benchmark.routing import instrument_control as ic
from benchmark.routing.scripts import knn_nulls

# The real embedder is a ~600MB ONNX load the unit suite refuses (conftest sets
# SHUNT_DISALLOW_REAL_EMBEDDER). Every test here therefore INJECTS a front end and asserts what
# the control does with it; the real embedder is exercised by running the control's own CLI,
# `python -m benchmark.routing.instrument_control`, which is where the published verdict comes
# from. That split is the point: these tests pin the control's DISCRIMINATION, not one corpus's
# score.
_K = 20
_THRESHOLD = 0.6
_PERM = 80
# The strategy leg rebuilds an HNSW index and a RouterEngine per draw, so it runs a smaller
# permutation count here. It is the DISCRIMINATION being pinned, not a published band's width.
_MIN_SAMPLES = 3
_STRATEGY_PERM = 24

_TOKEN = re.compile(r"[^A-Za-z0-9]+")


def _hashing_embedder(dim: int = 512) -> Callable[[list[str]], np.ndarray]:
    """A front end that genuinely reads the text: token-hashing bag of words."""

    # Not a stand-in for the shipped embedder's QUALITY — it is a front end that propagates
    # lexical content, which is the property the positive control is written to detect.
    def embed(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), dim))
        for i, text in enumerate(texts):
            for token in _TOKEN.split(text.lower()):
                if token:
                    out[i, zlib.crc32(token.encode()) % dim] += 1.0
        return out

    return embed


def _blind_embedder(dim: int = 64, seed: int = 3) -> Callable[[list[str]], np.ndarray]:
    """A front end that ignores the text entirely — the 'embedder is mis-wired' failure."""

    def embed(texts: list[str]) -> np.ndarray:
        return np.random.default_rng(seed).normal(size=(len(texts), dim))

    return embed


def _repo_only_embedder(dim: int = 64) -> Callable[[list[str]], np.ndarray]:
    """A front end that propagates ONLY the repository name out of the label."""

    # The concrete shape of "the control passed and certified nothing": on a corpus whose text is
    # a label dominated by the repo name, a pipeline that recovers the repo and nothing else would
    # satisfy any control whose planted signal happened to line up with the repo.
    def embed(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), dim))
        for i, text in enumerate(texts):
            repo = text.split("@", 1)[0]
            out[i, zlib.crc32(repo.encode()) % dim] = 1.0
        return out

    return embed


def _memorising_score(sims: np.ndarray, pass_mat: np.ndarray, k: int, threshold: float) -> float:
    """A scorer that lets each task vote on itself — the leakage the null leg must catch."""
    return knn_nulls._memorisation_rate(sims, pass_mat, k, threshold)


class TestThePlantedCorpus:
    """The control's claims about its own corpus, asserted rather than assumed."""

    def test_repository_carries_zero_information_about_the_label(self) -> None:
        # THE property that makes the positive leg mean something. If repo predicted the label at
        # all, a repo-only front end could clear the control and certify nothing.
        _matrix, task_ids, family_pass, _repo_pass = ic.build_control_matrix()
        repo_of = [tid.split("-", 1)[0] for tid in task_ids]
        label = family_pass[:, 1]  # 1 iff only the dear arm solves it
        for repo in set(repo_of):
            rows = [i for i, r in enumerate(repo_of) if r == repo]
            assert label[rows].mean() == pytest.approx(0.5), repo

    def test_neither_arm_can_win_by_being_constant(self) -> None:
        # Each arm passes exactly half the corpus, so 0.5 is the analytic chance level the gate
        # centres on, whatever the selection threshold does.
        _matrix, _ids, family_pass, repo_pass = ic.build_control_matrix()
        assert family_pass.mean(axis=0) == pytest.approx([0.5, 0.5])
        assert repo_pass.mean(axis=0) == pytest.approx([0.5, 0.5])

    def test_the_two_arms_are_crossed_so_every_task_is_solvable_by_exactly_one(self) -> None:
        _matrix, _ids, family_pass, _repo = ic.build_control_matrix()
        assert (family_pass.sum(axis=1) == 1).all()

    def test_every_task_text_is_distinct(self) -> None:
        matrix, task_ids, _f, _r = ic.build_control_matrix()
        texts = {matrix["tasks"][tid]["problem_statement"] for tid in task_ids}
        assert len(texts) == len(task_ids)

    def test_the_corpus_mirrors_the_field_shape_routing_text_resolves_through(self) -> None:
        # The control plants into `problem_statement` because that is the field the committed
        # corpus resolves through (routing_text prefers it over `description`). A task meta
        # carrying a field the real one does not have would be certifying a code path no real
        # task takes.
        matrix, task_ids, _f, _r = ic.build_control_matrix()
        meta = matrix["tasks"][task_ids[0]]
        assert "description" not in meta
        assert "problem_statement" in meta
        assert set(meta) == {"problem_statement", "repo"}

    def test_the_planted_text_length_matches_the_corpus_median_within_tolerance(self) -> None:
        # The control used to plant ~106-char labels while the real corpus's
        # routing channel is ~1180 chars — a positive control on ~9x shorter inputs exercises a
        # different embedding regime. The planted texts must sit at the corpus median.
        matrix, task_ids, _f, _r = ic.build_control_matrix()
        control_lens = [len(matrix["tasks"][tid]["problem_statement"]) for tid in task_ids]
        control_median = float(np.median(control_lens))
        corpus_median = float(ic.corpus_median_chars())
        assert abs(control_median - corpus_median) / corpus_median <= 0.2, (
            f"control median {control_median} vs corpus median {corpus_median}"
        )

    def test_the_planted_text_is_longer_than_the_old_label_regime(self) -> None:
        # The defect was a ~106-char label; the fix must have moved the control well past it.
        matrix, task_ids, _f, _r = ic.build_control_matrix()
        control_lens = [len(matrix["tasks"][tid]["problem_statement"]) for tid in task_ids]
        assert float(np.median(control_lens)) >= 1000

    def test_an_empty_corpus_median_raises_a_clear_error(self, monkeypatch) -> None:
        # An empty corpus turned `np.median([])` NaN and then `int(NaN)` into a raw
        # ValueError naming nothing. It must raise a named error that identifies the
        # missing corpus instead.
        monkeypatch.setattr(ic, "_corpus_problem_statements", lambda: ())
        with pytest.raises(ValueError, match="no problem statements"):
            ic.corpus_median_chars()

    def test_results_block_agrees_with_the_gate_outcome_matrix(self) -> None:
        # Asserted at BOTH arm counts. `_results_for` and the returned outcome matrix are two
        # SEPARATE `_pass_matrix` calls, so their agreement holds only as long as both are
        # given the same arm count — and the strategy leg's middle arms exist only on the
        # multi-arm path. Iterating the module constant against a default-argument matrix
        # proved the 2-arm analysis leg and left the 6-arm leg the shipped rule runs on
        # unchecked; iterating `models` also stops the loop silently mis-aligning if the
        # default ever stops being `CONTROL_MODELS`.
        for models in (ic.CONTROL_MODELS, ic.strategy_arms()):
            matrix, task_ids, family_pass, _repo = ic.build_control_matrix(models=models)
            assert family_pass.shape == (len(task_ids), len(models))
            for i, tid in enumerate(task_ids):
                assert set(matrix["results"][tid]) == set(models)
                for j, model in enumerate(models):
                    assert matrix["results"][tid][model]["pass"] is bool(family_pass[i, j])


class TestTheGateDiscriminates:
    """Positive leg passes on an intact front end; every broken one is rejected."""

    def _run(self, embed: Callable[[list[str]], np.ndarray], **kw: object) -> object:
        return ic.run_control(
            k=kw.pop("k", _K),  # type: ignore[arg-type]
            threshold=_THRESHOLD,
            n_perm=_PERM,
            embed_texts=embed,
            **kw,  # type: ignore[arg-type]
        )

    def test_an_intact_front_end_clears_both_legs(self) -> None:
        result = self._run(_hashing_embedder())
        assert result.admissible, result.reason
        assert result.positive_passed and result.null_at_chance

    def test_a_front_end_that_ignores_the_text_fails_the_positive_control(self) -> None:
        # The whole reason the control enters at `matrix["tasks"]`: a control injected downstream
        # of the embedder would score identically here and report the instrument healthy.
        result = self._run(_blind_embedder())
        assert not result.admissible
        assert not result.positive_passed
        assert "coverage-gap, not a falsification" in result.reason

    def test_a_front_end_that_only_propagates_repository_identity_fails(self) -> None:
        result = self._run(_repo_only_embedder())
        assert not result.admissible
        assert not result.positive_passed

    def test_the_repo_only_front_end_is_still_visible_in_the_diagnostic(self) -> None:
        # The pair of numbers is the diagnosis: dead on the planted signal, near-perfect on
        # repository identity means "the front end propagates the repo name and nothing finer".
        result = self._run(_repo_only_embedder())
        assert result.numbers["repo_identity_positive_score"] > 0.9
        assert result.positive_score < result.numbers["repo_identity_positive_score"]

    def test_a_leaking_scorer_fails_the_destroyed_signal_leg(self) -> None:
        # Signal destroyed, score still perfect: the shape the null leg exists for.
        result = self._run(_hashing_embedder(), k=1, score=_memorising_score)
        assert not result.admissible
        assert not result.null_at_chance
        assert "leakage" in result.reason

    def test_the_empirical_null_corroborates_the_analytic_chance_level(self) -> None:
        # If the permutation null did not sit on 0.5, the corpus construction would be wrong and
        # every verdict read off this control would be centred on the wrong point.
        result = self._run(_hashing_embedder())
        assert result.numbers["null_mean"] == pytest.approx(0.5, abs=result.chance_band)


class TestTheControlIsPinnedToTheFrontEnd:
    def test_it_refuses_to_run_when_routing_text_stops_reading_the_planted_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A silent fallback here would embed the task id, score at chance, and be indistinguishable
        # from a genuinely dead front end. It must name its own cause instead.
        monkeypatch.setattr(ic, "routing_text", lambda tid, _meta: tid)
        with pytest.raises(RuntimeError, match="no longer planting into the field"):
            ic.run_control(k=_K, threshold=_THRESHOLD, n_perm=4, embed_texts=_hashing_embedder())


class TestTheShippedRuleGetsItsOwnLeg:
    """The strategy table runs `SelectionRule`, not `select_from_rates` — so it needs its own leg.

    Every deliberately-broken front end below must be rejected here too; a leg that only ever
    passes certifies nothing, and the analysis leg passing says nothing about this one.
    """

    def _run(self, embed: Callable[[list[str]], np.ndarray], **kw: object) -> object:
        return ic.run_strategy_control(
            k=_K,
            threshold=_THRESHOLD,
            min_samples=_MIN_SAMPLES,
            n_perm=_STRATEGY_PERM,
            embed_texts=embed,
            **kw,  # type: ignore[arg-type]
        )

    def test_an_intact_front_end_clears_both_legs(self) -> None:
        result = self._run(_hashing_embedder())
        assert result.admissible, result.reason
        assert result.positive_passed and result.null_at_chance

    def test_a_front_end_that_ignores_the_text_fails_the_positive_control(self) -> None:
        result = self._run(_blind_embedder())
        assert not result.admissible
        assert not result.positive_passed
        assert "coverage-gap, not a falsification" in result.reason

    def test_a_front_end_that_only_propagates_repository_identity_fails(self) -> None:
        # The planted label is orthogonal to repository identity on BOTH legs, so a pipeline that
        # recovers only the repo name cannot satisfy either one.
        result = self._run(_repo_only_embedder())
        assert not result.admissible
        assert not result.positive_passed

    def test_the_corpus_covers_every_arm_the_shipped_rule_can_escalate_to(self) -> None:
        # `SelectionRule._escalate` walks the whole ranked pool. An arm the corpus has no cell for
        # would send tasks to `unscorable`, and the surviving score would no longer sit on 0.5.
        arms = ic.strategy_arms()
        matrix, task_ids, _f, _r = ic.build_control_matrix(models=arms)
        assert set(matrix["results"][task_ids[0]]) == set(arms)

    def test_it_refuses_rather_than_scoring_a_corpus_the_rule_can_escape(self) -> None:
        # Restricting the arms to the two extremes while the engine still ranks six is exactly the
        # silent-drop failure; it must name its own cause instead of scoring the remainder.
        arms = ic.strategy_arms()
        with pytest.raises(RuntimeError, match="outside the planted corpus"):
            ic.run_strategy_control(
                k=_K,
                threshold=_THRESHOLD,
                min_samples=_MIN_SAMPLES,
                n_perm=2,
                embed_texts=_blind_embedder(),
                arms=(arms[0], arms[-1]),
            )

    def test_both_legs_score_the_same_planted_corpus(self) -> None:
        # One control construction, two rules. A second corpus is how the first goes stale.
        _m, ids_analysis, _f, _r = ic.build_control_matrix()
        _m2, ids_strategy, _f2, _r2 = ic.build_control_matrix(models=ic.strategy_arms())
        assert ids_analysis == ids_strategy

    def test_middle_arms_never_pass_so_chance_stays_one_half(self) -> None:
        arms = ic.strategy_arms()
        _m, _ids, family_pass, _r = ic.build_control_matrix(models=arms)
        assert (family_pass.sum(axis=1) == 1).all()
        assert family_pass[:, 1:-1].sum() == 0
        assert family_pass.mean(axis=0)[[0, -1]] == pytest.approx([0.5, 0.5])


class TestTheStrategyTableCarriesItsVerdict:
    """A row of `strategy_summary.csv` cannot be published without its instrument status."""

    def test_the_table_cannot_be_built_without_an_admissibility_verdict(self) -> None:
        from benchmark.routing import summary

        with pytest.raises(TypeError):
            summary.StrategyTable(rows=({"strategy": "kNN-semantic"},))  # type: ignore[call-arg]

    def test_every_written_row_carries_the_verdict(self, tmp_path: Path) -> None:
        import csv

        from benchmark.admissibility import admissibility_verdict
        from benchmark.routing import summary

        rejected = admissibility_verdict(0.51, 0.50, chance_level=0.5, chance_band=0.1)
        table = summary.StrategyTable(
            admissibility=rejected,
            rows=(
                {"strategy": "kNN-semantic", "AvgPerf%": 78.53},
                {"strategy": "Oracle", "AvgPerf%": 96.61},
            ),
        )
        out = tmp_path / "strategy_summary.csv"
        summary.write_summary_csv(table, out)
        rows = list(csv.DictReader(out.open(newline="")))
        assert [r["strategy"] for r in rows] == ["kNN-semantic", "Oracle"]
        assert {r["instrument_admissible"] for r in rows} == {"False"}
        assert all(r["instrument_verdict"].startswith("INSTRUMENT INADMISSIBLE") for r in rows)

    def test_the_table_is_certified_at_the_parameters_the_rows_were_scored_at(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A control run at other settings certifies an instrument nobody is quoting, so the
        # caller's overrides must reach the control rather than config's defaults.
        from benchmark.routing import summary

        seen: dict[str, object] = {}

        def fake(**kw: object) -> object:
            seen.update(kw)
            from benchmark.admissibility import admissibility_verdict

            return admissibility_verdict(1.0, 0.5, chance_level=0.5, chance_band=0.1)

        monkeypatch.setattr(ic, "strategy_instrument_admissibility", fake)
        table = summary.certified_table(
            [{"strategy": "kNN-semantic"}], k=7, threshold=0.42, min_samples=5
        )
        assert seen == {"k": 7, "threshold": 0.42, "min_samples": 5}
        assert table.admissibility.admissible
        assert table.rows[0]["strategy"] == "kNN-semantic"


class TestFrontEndParity:
    """The control's embedding step must be the figures' embedding step, not a lookalike."""

    def test_it_matches_viz_knn_build_task_embeddings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from benchmark.routing.scripts import viz_knn

        embed = _hashing_embedder()
        monkeypatch.setattr(viz_knn, "_embed_texts", embed)
        matrix, task_ids, _f, _r = ic.build_control_matrix()
        assert ic._embeddings(matrix, task_ids, embed) == pytest.approx(
            viz_knn.build_task_embeddings(matrix, task_ids)
        )


class TestTheVerdictIsAttachedToWhatItCertifies:
    def test_a_transfer_curve_cannot_be_built_without_an_admissibility_verdict(self) -> None:
        # The structural half of the fix: not a reminder to run the gate, but a result object that
        # will not exist until someone states whether the instrument cleared it.
        with pytest.raises(TypeError):
            knn_nulls.TransferCurve(  # type: ignore[call-arg]
                ks=(10,),
                loo=(0.7,),
                memorisation=(0.7,),
                null_lo=(0.6,),
                null_hi=(0.8,),
                null_mean=(0.7,),
                null_sd=(0.05,),
                max_null=knn_nulls.Band(mean=0.7, sd=0.05, lo=0.6, hi=0.8, n=10),
                best_constant=0.7,
                best_constant_model="m",
                n_tasks=10,
                n_perm=10,
            )

    def test_an_inadmissible_instrument_makes_the_figure_note_refuse_the_claim(self) -> None:
        from benchmark.admissibility import admissibility_verdict
        from benchmark.routing.scripts import plot_knn_nulls

        band = knn_nulls.Band(mean=0.50, sd=0.02, lo=0.46, hi=0.54, n=200)
        rejected = admissibility_verdict(0.51, 0.50, chance_level=0.5, chance_band=0.1)
        note = plot_knn_nulls._verdict(0.90, band, "the pass rate", rejected)
        assert note.startswith("NOT QUOTABLE")
        assert "coverage-gap, not a result" in note

    def test_an_admissible_instrument_leaves_the_note_unchanged(self) -> None:
        from benchmark.admissibility import admissibility_verdict
        from benchmark.routing.scripts import plot_knn_nulls

        band = knn_nulls.Band(mean=0.50, sd=0.02, lo=0.46, hi=0.54, n=200)
        cleared = admissibility_verdict(0.95, 0.50, chance_level=0.5, chance_band=0.1)
        note = plot_knn_nulls._verdict(0.50, band, "the pass rate", cleared)
        assert note.startswith("NULL RESULT")
