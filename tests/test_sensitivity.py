"""The MDE estimator, both directions."""

# A floor is only a floor if the thing measuring it fails when it should: a pipeline that cannot
# see a MAXIMAL planted signal must come back inadmissible, and a deliberately blurred pipeline
# must come back with a LARGER floor than the intact one. A sweep that always returns a small
# number is the failure mode this file exists to prevent.

from __future__ import annotations

import numpy as np
import pytest

from benchmark.routing import sensitivity as sv
from benchmark.routing.scripts import knn_nulls

# A hermetic stand-in for the real corpus: repository clusters (the nuisance structure the real
# embedding space has) plus a family coordinate ORTHOGONAL to them (the plantable structure). No
# ONNX, no committed data — these tests pin the estimator's behaviour, not one corpus's floor.
_REPOS = 6
_PER_REPO = 12
_N = _REPOS * _PER_REPO
_ESCALATION_SHARE = 0.25
_KS = (2, 5, 10, 20)
_K = 10
_THRESHOLD = 0.6


def _synthetic_corpus(*, blur: float = 0.0, seed: int = 0) -> sv.Corpus:
    """Repo clusters + an orthogonal family axis; ``blur`` mixes the geometry toward noise."""
    rng = np.random.default_rng(seed)
    repos: list[str] = []
    emb = np.zeros((_N, _REPOS + 2))
    for r in range(_REPOS):
        for j in range(_PER_REPO):
            i = r * _PER_REPO + j
            repos.append(f"org/repo{r}")
            emb[i, r] = 1.0
            # Balanced within the repository, so the family axis carries no repo information.
            emb[i, _REPOS] = 1.0 if j < _PER_REPO // 2 else -1.0
            emb[i, _REPOS + 1] = 0.05 * rng.normal()
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    sims = emb @ emb.T
    if blur:
        noise = rng.normal(size=(_N, _N))
        sims = (1.0 - blur) * sims + blur * (noise + noise.T) / 2.0
    n_e = int(round(_N * _ESCALATION_SHARE))
    pass_mat = np.tile(np.array([1.0, 0.0]), (_N, 1))
    pass_mat[:n_e] = np.array([0.0, 1.0])
    return sv.Corpus(sims=sims, emb=emb, repos=tuple(repos), pass_mat=pass_mat)


def _spec(**kw: object) -> sv.SweepSpec:
    base: dict = {
        "k": _K,
        "threshold": _THRESHOLD,
        "ks": _KS,
        "rho_grid": (0.0, 0.25, 0.5, 0.75, 1.0),
        "n_trials": 40,
        "n_perm": 60,
    }
    base.update(kw)
    return sv.SweepSpec(**base)


class TestThePlantedSignalIsOrthogonalToRepository:
    """The property that makes the floor mean something — reused from the instrument control."""

    def test_every_repository_carries_the_same_class_share(self) -> None:
        corpus = _synthetic_corpus()
        classes = sv.planted_classes(corpus, "linear")
        share = corpus.n_escalation_rows / corpus.n
        for idx in corpus.repo_index.values():
            assert classes[idx].mean() == pytest.approx(share, abs=1.0 / _PER_REPO)

    def test_the_class_count_matches_the_real_row_count_exactly(self) -> None:
        # Off by one task and the row multiset changes, which moves the null band away from the
        # published one and the whole comparison stops being a comparison.
        corpus = _synthetic_corpus()
        for geometry in sv.GEOMETRIES:
            assert sv.planted_classes(corpus, geometry).sum() == corpus.n_escalation_rows

    def test_repository_identity_does_not_predict_the_planted_class(self) -> None:
        corpus = _synthetic_corpus()
        classes = sv.planted_classes(corpus, "linear")
        repo_rank = np.array([int(r.removeprefix("org/repo")) for r in corpus.repos], dtype=float)
        assert sv.auroc(repo_rank, classes) == pytest.approx(0.5, abs=0.05)


class TestPlantingReAssignsAndNeverFabricates:
    def test_the_row_multiset_survives_every_strength(self) -> None:
        corpus = _synthetic_corpus()
        classes = sv.planted_classes(corpus, "linear")
        original = np.sort(corpus.pass_mat.view("f8,f8").ravel(), axis=0)  # type: ignore[call-overload]
        for rho in (0.0, 0.3, 1.0):
            planted = sv.plant(corpus, classes, rho, np.random.default_rng(1))
            assert planted.shape == corpus.pass_mat.shape
            assert np.array_equal(
                np.sort(planted.view("f8,f8").ravel(), axis=0),  # type: ignore[call-overload]
                original,
            )

    def test_strength_maps_onto_auroc_as_advertised(self) -> None:
        # The unit the floor is reported in. If this drifts, every quoted AUROC is wrong.
        corpus = _synthetic_corpus()
        classes = sv.planted_classes(corpus, "linear")
        for rho in (0.0, 0.5, 1.0):
            got = np.mean(
                [
                    sv.auroc(
                        classes.astype(float),
                        (
                            sv.plant(corpus, classes, rho, np.random.default_rng(s))[:, 0] == 0
                        ).astype(int),
                    )
                    for s in range(40)
                ]
            )
            assert got == pytest.approx(0.5 + rho / 2.0, abs=0.05)


class TestTheNullIsTheOneThePublishedAnalysisUses:
    def test_the_detection_bar_does_not_move_with_the_planted_strength(self) -> None:
        # THE claim that makes this sweep comparable to the committed NULL RESULT: the permutation
        # band is a function of the row multiset alone, so the bar a planted signal must clear is
        # the same bar the real analysis quoted. Asserted, not assumed.
        corpus = _synthetic_corpus()
        classes = sv.planted_classes(corpus, "linear")
        spec = _spec()
        bar = sv.detection_threshold(corpus, spec)
        for rho in (0.4, 1.0):
            planted = sv.plant(corpus, classes, rho, np.random.default_rng(7))
            moved = sv.Corpus(
                sims=corpus.sims, emb=corpus.emb, repos=corpus.repos, pass_mat=planted
            )
            assert sv.detection_threshold(moved, spec) == bar

    def test_the_canonicalised_bar_agrees_with_the_published_permutation_null(self) -> None:
        # Canonicalising rows buys exact invariance across rho; this is what keeps that from
        # quietly becoming a DIFFERENT null than the one `knn_nulls` draws for the figures.
        corpus = _synthetic_corpus()
        spec = _spec(n_perm=400)
        rng = np.random.default_rng(11)
        idx = np.arange(corpus.n)
        published = knn_nulls.band_of(
            np.array(
                [
                    knn_nulls.routed_pass_rate(
                        corpus.sims,
                        corpus.pass_mat[rng.permutation(corpus.n)],
                        idx,
                        idx,
                        _K,
                        _THRESHOLD,
                    )
                    for _ in range(400)
                ]
            )
        )
        assert sv.detection_threshold(corpus, spec) == pytest.approx(published.hi, abs=0.02)

    def test_the_statistic_is_the_published_one(self) -> None:
        corpus = _synthetic_corpus()
        idx = np.arange(corpus.n)
        expected = knn_nulls.routed_pass_rate(
            corpus.sims, corpus.pass_mat, idx, idx, _K, _THRESHOLD
        )
        scores = sv.routed_scores(corpus, corpus.pass_mat, _KS, _THRESHOLD, "ungrouped")
        assert scores[_KS.index(_K)] == pytest.approx(expected)

    def test_the_grouped_split_never_lets_a_task_see_its_own_repository(self) -> None:
        # A one-repo corpus has no out-of-repo index at all, so the grouped statistic must be the
        # empty-index value rather than quietly falling back to the leaky one.
        corpus = _synthetic_corpus()
        single = sv.Corpus(
            sims=corpus.sims,
            emb=corpus.emb,
            repos=("org/only",) * corpus.n,
            pass_mat=corpus.pass_mat,
        )
        scores = sv.routed_scores(single, single.pass_mat, (_K,), _THRESHOLD, "repo-grouped")
        assert np.isnan(scores)


class TestTheEstimatorFailsInTheRightDirections:
    def test_an_intact_pipeline_detects_a_maximal_planted_signal(self) -> None:
        mde = sv.minimum_detectable_effect(_synthetic_corpus(), _spec())
        assert mde.admissibility.admissible, mde.admissibility.reason
        assert mde.curve[-1].power > mde.curve[0].power
        assert mde.rho is not None

    def test_a_blinded_pipeline_cannot_and_is_marked_inadmissible(self) -> None:
        # The hard direction. A sweep that returned a floor here would be reporting a number about
        # an instrument that has never been shown to detect anything.
        mde = sv.minimum_detectable_effect(_synthetic_corpus(blur=1.0), _spec())
        assert not mde.admissibility.admissible
        assert not mde.admissibility.positive_passed
        assert mde.rho is None
        assert "MDE UNATTAINABLE" in mde.headline

    def test_a_blurred_pipeline_returns_a_strictly_larger_floor(self) -> None:
        # The graded direction: weakening the front end must RAISE the floor, not leave it alone.
        sharp = sv.minimum_detectable_effect(_synthetic_corpus(), _spec())
        blurred = sv.minimum_detectable_effect(_synthetic_corpus(blur=0.5), _spec())
        assert sharp.rho is not None
        assert blurred.rho is None or blurred.rho > sharp.rho

    def test_the_false_positive_rate_stays_at_the_nominal_level(self) -> None:
        mde = sv.minimum_detectable_effect(_synthetic_corpus(), _spec())
        assert mde.curve[0].ci_lo <= sv.NOMINAL_FALSE_POSITIVE_RATE * 3
        assert mde.admissibility.null_at_chance

    def test_the_interval_brackets_the_point_estimate(self) -> None:
        mde = sv.minimum_detectable_effect(_synthetic_corpus(), _spec())
        assert mde.rho_lo is not None and mde.rho is not None
        assert mde.rho_lo <= mde.rho
        assert mde.rho_hi is None or mde.rho_hi >= mde.rho

    def test_a_floor_is_reported_as_an_auroc_too(self) -> None:
        mde = sv.minimum_detectable_effect(_synthetic_corpus(), _spec())
        assert mde.auroc_of(mde.rho) == pytest.approx(0.5 + (mde.rho or 0.0) / 2.0)
        assert "AUROC" in mde.headline


class TestTheSweepRefusesIncoherentQuestions:
    @pytest.mark.parametrize(
        "kw",
        [
            {"split": "leave-one-out"},
            {"rule": "argmax"},
            {"geometry": "lexical"},
            {"k": 7},
        ],
    )
    def test_a_spec_that_names_nothing_real_is_rejected(self, kw: dict) -> None:
        with pytest.raises(ValueError):
            _spec(**kw)


class TestTheFloorCannotBeQuotedWithoutItsVerdict:
    def test_the_result_object_requires_an_admissibility_verdict(self) -> None:
        with pytest.raises(TypeError):
            sv.MinimumDetectableEffect(  # type: ignore[call-arg]
                spec=_spec(),
                n_tasks=_N,
                curve=(),
                rho=0.5,
                rho_lo=0.4,
                rho_hi=0.6,
            )
