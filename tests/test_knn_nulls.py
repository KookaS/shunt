"""Unit tests for the kNN null models (benchmark/routing/scripts/knn_nulls.py)."""

# These are the statistics that decide whether the router learned anything, so the
# properties that matter are: a constant router scores chance, a signal-free input lands
# inside the null band, and a genuinely separable input lands above it.

from __future__ import annotations

import numpy as np
import pytest

from benchmark.admissibility import admissibility_verdict
from benchmark.routing.scripts import knn_nulls as kn

# Fewer permutations than the 200 the figures use — these assert shape and direction,
# not a published band.
_PERM = 60

# Both result objects REQUIRE the instrument-validity verdict, so every construction here has
# to state one. These tests exercise the null models on synthetic similarity matrices — the
# front end is not in scope — so they pass a stub. The real control runs against the real
# embedder in tests/test_instrument_control.py.
_STUB_GATE = admissibility_verdict(1.0, 0.5, chance_level=0.5, chance_band=0.1)


class TestBaseRates:
    def test_chance_purity_of_a_constant_label_is_one(self):
        assert kn.chance_purity(["a"] * 10) == pytest.approx(1.0)

    def test_chance_purity_is_sum_of_squared_shares(self):
        # 174/177 + 3/177 — the real allocation. 0.5 (the line the old figure drew) is
        # not even close, which is the whole defect.
        labels = ["a"] * 174 + ["b"] * 3
        expected = (174 / 177) ** 2 + (3 / 177) ** 2
        assert kn.chance_purity(labels) == pytest.approx(expected)
        assert kn.chance_purity(labels) > 0.96

    def test_chance_purity_of_a_balanced_binary_split_is_half(self):
        assert kn.chance_purity(["a"] * 50 + ["b"] * 50) == pytest.approx(0.5)

    def test_majority_share(self):
        assert kn.majority_share(["a"] * 174 + ["b"] * 3) == pytest.approx(174 / 177)

    def test_empty_labels_do_not_divide_by_zero(self):
        assert kn.chance_purity([]) == 0.0
        assert kn.majority_share([]) == 0.0


class TestBand:
    def test_z_and_containment(self):
        band = kn.band_of(np.array([0.5, 0.6, 0.4, 0.5, 0.55]))
        assert band.contains(band.mean)
        assert band.z(band.mean) == pytest.approx(0.0)
        assert band.z(band.mean + band.sd) == pytest.approx(1.0)

    def test_zero_variance_band_reports_zero_z(self):
        band = kn.band_of(np.full(10, 0.77))
        assert band.sd == 0.0
        assert band.z(0.9) == 0.0


class TestPurityNull:
    """A degenerate allocation must NOT clear its own null — the audit's core finding."""

    def _sims(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        emb = rng.normal(size=(n, 8))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return emb @ emb.T

    def test_query_is_never_its_own_neighbour(self):
        rank = kn.neighbour_rank(self._sims(12))
        assert rank.shape == (12, 11)
        for i, row in enumerate(rank):
            assert i not in row.tolist()

    def _pure_null(self, n: int, seed: int):
        """Embeddings and outcomes drawn INDEPENDENTLY — there is nothing to find."""
        rng = np.random.default_rng(seed)
        emb = rng.normal(size=(n, 8))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        sims = emb @ emb.T
        pass_mat = rng.integers(0, 2, size=(n, 3)).astype(float)
        return sims, pass_mat

    def _observed_purity(self, sims, pass_mat, k, threshold=0.6):
        rank = kn.neighbour_rank(sims)
        idx = np.arange(pass_mat.shape[0])
        rates = kn.neighbourhood_rates(sims, pass_mat, idx, idx, k)
        labels = kn.select_from_rates(rates, threshold)
        return rank, kn.mean_purity(rank, labels, k)

    def test_null_does_not_manufacture_signal_on_pure_null_data(self):
        """THE regression: outcomes independent of embeddings must not clear the band."""
        # The earlier label-permutation null failed this in 24/30 trials at mean z=+3.6,
        # because the router's own picks are autocorrelated by construction (neighbouring
        # tasks share neighbourhoods) and shuffling them destroyed that too.
        above = 0
        trials = 12
        for t in range(trials):
            sims, pass_mat = self._pure_null(90, seed=t)
            rank, observed = self._observed_purity(sims, pass_mat, k=10)
            band = kn.purity_null_band(rank, sims, pass_mat, k=10, threshold=0.6, n_perm=_PERM)
            above += observed > band.hi
        # A correct 95% band false-positives ~2.5% of the time one-sided; allow slack for
        # 12 trials but fail loudly on the systematic breakage the old null showed.
        assert above <= 2, f"null cleared in {above}/{trials} pure-null trials"

    def test_constant_labels_give_purity_one(self):
        # A router that picks one model everywhere scores 1.0 — and so does its null.
        sims, _ = self._pure_null(30, seed=0)
        rank = kn.neighbour_rank(sims)
        assert kn.mean_purity(rank, np.zeros(30, dtype=int), k=10) == pytest.approx(1.0)

    def test_real_structure_still_clears_the_band(self):
        # Outcomes made a deterministic function of position in embedding space: a valid
        # null must still detect this, or the test above would pass for a dead statistic.
        rng = np.random.default_rng(5)
        a = rng.normal(loc=+4.0, scale=0.2, size=(45, 4))
        b = rng.normal(loc=-4.0, scale=0.2, size=(45, 4))
        emb = np.vstack([a, b])
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        sims = emb @ emb.T
        # Cluster A is solved only by the pricey model, cluster B only by the cheap one, so
        # the router's pick genuinely tracks geometry.
        pass_mat = np.zeros((90, 2))
        pass_mat[:45, 1] = 1.0
        pass_mat[45:, 0] = 1.0
        rank, observed = self._observed_purity(sims, pass_mat, k=10)
        band = kn.purity_null_band(rank, sims, pass_mat, k=10, threshold=0.6, n_perm=_PERM)
        assert observed > band.hi


class TestSelectionRule:
    """select_from_rates is the ONE definition of the routing rule; viz_knn delegates."""

    def test_picks_the_cheapest_clearing_the_threshold(self):
        # Columns are price-ascending, so index 0 is cheapest.
        rates = np.array([[0.7, 0.9, 0.95]])
        assert kn.select_from_rates(rates, 0.6).tolist() == [0]

    def test_escalates_when_nothing_clears(self):
        rates = np.array([[0.1, 0.2, 0.5]])
        assert kn.select_from_rates(rates, 0.6).tolist() == [2]

    def test_ties_break_toward_the_cheaper_model(self):
        rates = np.array([[0.5, 0.5, 0.5]])
        assert kn.select_from_rates(rates, 0.9).tolist() == [0]

    def test_matches_viz_knn_knn_select(self):
        """The shared rule must agree with the figure's own entry point."""
        from benchmark import config
        from benchmark.routing.scripts import viz_knn

        config.load("benchmark/benchmark.yaml")
        models = config.enabled_models()
        by_price = sorted(models, key=lambda m: config.cost_per_1m(m, config.enabled_pricing()))
        rng = np.random.default_rng(11)
        n = 40
        emb = rng.normal(size=(n, 12))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        vecs = np.zeros((n, len(models) * 4))
        for j in range(len(models)):
            vecs[:, j * 4] = rng.integers(0, 2, size=n)

        pass_mat = np.column_stack([vecs[:, models.index(m) * 4] for m in by_price])
        sims = emb @ emb.T
        all_idx = np.arange(n)
        rates = kn.neighbourhood_rates(sims, pass_mat, all_idx, all_idx, k=7)
        shared = [by_price[i] for i in kn.select_from_rates(rates, 0.6)]
        direct = [
            viz_knn.knn_select(vecs, emb, i, models, k=7, success_rate_threshold=0.6)
            for i in range(n)
        ]
        assert shared == direct


class TestTransferAndCrossRepo:
    def _fixture(self, n: int = 60, seed: int = 7):
        rng = np.random.default_rng(seed)
        emb = rng.normal(size=(n, 8))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        pass_mat = rng.integers(0, 2, size=(n, 3)).astype(float)
        return emb @ emb.T, pass_mat

    def test_memorisation_ceiling_is_at_least_leave_one_out(self):
        sims, pass_mat = self._fixture()
        curve = kn.transfer_curve(
            sims, pass_mat, ["a", "b", "c"], [5, 10], 0.6, admissibility=_STUB_GATE, n_perm=_PERM
        )
        for ceiling, loo in zip(curve.memorisation, curve.loo, strict=True):
            assert ceiling >= loo - 1e-9

    def test_random_outcomes_land_inside_the_null(self):
        sims, pass_mat = self._fixture()
        curve = kn.transfer_curve(
            sims, pass_mat, ["a", "b", "c"], [10], 0.6, admissibility=_STUB_GATE, n_perm=_PERM
        )
        band = kn.Band(
            mean=curve.null_mean[0],
            sd=1.0,
            lo=curve.null_lo[0],
            hi=curve.null_hi[0],
            n=_PERM,
        )
        assert band.contains(curve.loo[0])

    def test_best_constant_is_the_best_column_mean(self):
        sims, pass_mat = self._fixture()
        curve = kn.transfer_curve(
            sims, pass_mat, ["a", "b", "c"], [10], 0.6, admissibility=_STUB_GATE, n_perm=_PERM
        )
        assert curve.best_constant == pytest.approx(pass_mat.mean(axis=0).max())
        assert curve.best_constant_model == ["a", "b", "c"][int(np.argmax(pass_mat.mean(axis=0)))]

    def test_repo_of_parses_swebench_ids(self):
        assert kn.repo_of("astropy__astropy-12907") == "astropy/astropy"
        assert kn.repo_of("scikit-learn__scikit-learn-10297") == "scikit-learn/scikit-learn"

    def test_cross_repo_grid_is_square_over_kept_repos(self):
        sims, pass_mat = self._fixture(n=60)
        task_ids = [f"org__repo{i % 3}-{i}" for i in range(60)]
        cross = kn.cross_repo_transfer(
            sims,
            pass_mat,
            task_ids,
            k=5,
            threshold=0.6,
            admissibility=_STUB_GATE,
            min_tasks=8,
            n_perm=_PERM,
        )
        assert len(cross.repos) == 3
        assert cross.grid.shape == (3, 3)
        assert not np.isnan(cross.grid).any()

    def test_repos_below_the_minimum_are_dropped(self):
        sims, pass_mat = self._fixture(n=60)
        # repo2 gets only 4 tasks; the rest split between repo0/repo1.
        task_ids = [f"org__repo{2 if i < 4 else i % 2}-{i}" for i in range(60)]
        cross = kn.cross_repo_transfer(
            sims,
            pass_mat,
            task_ids,
            k=5,
            threshold=0.6,
            admissibility=_STUB_GATE,
            min_tasks=8,
            n_perm=_PERM,
        )
        assert "org/repo2" not in cross.repos


class TestNeighbourhoodIndexRestriction:
    """Restricting the index is what makes held-out and cross-repo routing possible."""

    def test_index_restriction_excludes_the_query_itself(self):
        rng = np.random.default_rng(2)
        emb = rng.normal(size=(20, 6))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        sims = emb @ emb.T
        # Model 0 passes ONLY on task 0. If task 0 could see itself, its neighbourhood
        # rate for model 0 would be non-zero; held out, it must be exactly 0.
        pass_mat = np.zeros((20, 2))
        pass_mat[0, 0] = 1.0
        rates = kn.neighbourhood_rates(sims, pass_mat, np.array([0]), np.arange(20), k=5)
        assert rates[0, 0] == 0.0

    def test_disjoint_index_uses_only_the_given_tasks(self):
        rng = np.random.default_rng(4)
        emb = rng.normal(size=(20, 6))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        sims = emb @ emb.T
        pass_mat = np.zeros((20, 2))
        pass_mat[10:, 0] = 1.0  # only the second half passes on model 0
        rates = kn.neighbourhood_rates(sims, pass_mat, np.arange(10), np.arange(10), k=4)
        assert (rates[:, 0] == 0.0).all()
        rates_other = kn.neighbourhood_rates(sims, pass_mat, np.arange(10), np.arange(10, 20), k=4)
        assert (rates_other[:, 0] == 1.0).all()


class TestDegenerateNullReportsZeroZ:
    """A null with zero spread must never produce a z from an epsilon divisor."""

    def test_zero_width_band_reports_zero_z(self):
        band = kn.Band(mean=0.7740, sd=0.0, lo=0.7740, hi=0.7740, n=200)
        assert band.z(0.9) == 0.0
        assert band.contains(0.7740)

    def test_transfer_curve_footer_survives_a_degenerate_null(self):
        """At k >= corpus size every permutation gives the same answer (real, at k=40+)."""
        from benchmark.routing.scripts import plot_knn_nulls

        curve = kn.TransferCurve(
            admissibility=_STUB_GATE,
            ks=(40,),
            loo=(0.7740,),
            memorisation=(0.7740,),
            null_lo=(0.7740,),
            null_hi=(0.7740,),
            null_mean=(0.7740,),
            null_sd=(0.0,),
            max_null=kn.Band(mean=0.7740, sd=0.0, lo=0.7740, hi=0.7740, n=200),
            best_constant=0.9605,
            best_constant_model="kimi-k3",
            n_tasks=177,
            n_perm=200,
        )
        note = plot_knn_nulls._verdict(curve.loo[0], curve.band_at(0), "the pass rate", _STUB_GATE)
        assert "NULL RESULT" in note
        assert "z=+0.00" in note


class TestSelectionCorrectedNull:
    """The best-over-k point must be judged against the null of the SAME search."""

    def _fixture(self, n: int = 70, seed: int = 9):
        rng = np.random.default_rng(seed)
        emb = rng.normal(size=(n, 8))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return emb @ emb.T, rng.integers(0, 2, size=(n, 3)).astype(float)

    def test_max_null_is_at_least_every_per_k_band(self):
        sims, pass_mat = self._fixture()
        ks = [2, 5, 10, 20]
        curve = kn.transfer_curve(
            sims, pass_mat, ["a", "b", "c"], ks, 0.6, admissibility=_STUB_GATE, n_perm=_PERM
        )
        # A max over k can only be >= any single k's upper bound, so the corrected band is
        # never more permissive than the uncorrected one it replaces.
        assert curve.max_null.hi >= max(curve.null_hi) - 1e-12

    def test_null_sd_is_exact_not_reconstructed(self):
        sims, pass_mat = self._fixture()
        curve = kn.transfer_curve(
            sims, pass_mat, ["a", "b", "c"], [5, 10], 0.6, admissibility=_STUB_GATE, n_perm=_PERM
        )
        for i in range(len(curve.ks)):
            band = curve.band_at(i)
            assert band.sd == curve.null_sd[i]
            # The width/3.92 shortcut assumes a normal null; on this discrete statistic it
            # differs from the real sd, which is exactly why the sd is carried through.
            assert band.sd >= 0.0

    def test_band_at_round_trips_the_stored_percentiles(self):
        sims, pass_mat = self._fixture()
        curve = kn.transfer_curve(
            sims, pass_mat, ["a", "b", "c"], [10], 0.6, admissibility=_STUB_GATE, n_perm=_PERM
        )
        band = curve.band_at(0)
        assert (band.lo, band.hi, band.mean) == (
            curve.null_lo[0],
            curve.null_hi[0],
            curve.null_mean[0],
        )


class TestCrossRepoGuards:
    def test_fewer_than_two_repos_raises_instead_of_returning_nan(self):
        rng = np.random.default_rng(2)
        n = 40
        emb = rng.normal(size=(n, 6))
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        pass_mat = rng.integers(0, 2, size=(n, 2)).astype(float)
        task_ids = [f"org__only-{i}" for i in range(n)]
        # One repo leaves no off-diagonal: the advantage would be nan and _verdict would
        # render it as "BELOW the null band", which is worse than failing.
        with pytest.raises(ValueError, match="at least 2 repos"):
            kn.cross_repo_transfer(
                emb @ emb.T,
                pass_mat,
                task_ids,
                k=5,
                threshold=0.6,
                admissibility=_STUB_GATE,
                min_tasks=8,
                n_perm=_PERM,
            )


class TestRelationshipToTheShippedSelectionRule:
    """`select_from_rates` is an analysis approximation, NOT `router.SelectionRule`."""

    # Its docstring once claimed to BE the shipped product rule. It is not, and an audit probe
    # found the two returning different models on identical evidence. These tests pin what the
    # two genuinely share and each documented way they diverge, so the claim cannot silently
    # drift back.

    # price-ascending, matching the rates-column contract
    _MODELS = ("cheap", "mid", "dear")

    def _pool(self):
        models = self._MODELS

        class _Pool:
            def ranked_models(self):
                return [type("M", (), {"name": n})() for n in models]

        return _Pool()

    def _neighbors(self, per_model, distance=0.0, confidence=1.0):
        """Neighbours with EQUAL distance, so confidence weighting is uniform."""
        from shunt.router.selection import NeighborResult

        out = []
        for model, outcomes in per_model.items():
            for i, passed in enumerate(outcomes):
                out.append(
                    NeighborResult(
                        model=model,
                        outcome=passed,
                        cost=1.0,
                        verification_confidence=confidence,
                        distance=distance,
                        session_id=f"{model}-{i}",
                    )
                )
        return out

    def _shipped(self, per_model, threshold, min_samples):
        from shunt.router.selection import SelectionRule

        rule = SelectionRule(min_success_rate=threshold, min_samples=min_samples)
        model, _reason = rule.select(self._neighbors(per_model), self._pool(), False)
        return model

    def _analysis(self, per_model, threshold):
        rates = np.array([[float(np.mean(per_model[m])) for m in self._MODELS]], dtype=float)
        return self._MODELS[int(kn.select_from_rates(rates, threshold)[0])]

    def test_the_shared_core_agrees_cheapest_above_threshold(self):
        # Uniform distances, every group above min_samples, at least one model clearing the
        # bar: the three documented differences are all neutralized, so the two rules MUST
        # agree. This is the part that is genuinely "the shipped rule".
        per_model = {
            "cheap": [True, True, True, True],  # 1.00 — clears
            "mid": [True, True, True, False],  # 0.75 — clears
            "dear": [True, True, True, True],  # 1.00 — clears
        }
        assert self._analysis(per_model, 0.7) == "cheap"
        assert self._shipped(per_model, 0.7, min_samples=3) == "cheap"

    def test_difference_2_the_shipped_rule_applies_a_min_samples_floor(self):
        # `cheap` has the best rate but only ONE neighbour. The analysis rule has no floor and
        # takes it; the shipped rule rejects it and takes the next model that clears the bar.
        per_model = {
            "cheap": [True],
            "mid": [True, True, True, True],
            "dear": [True, True, True, True],
        }
        assert self._analysis(per_model, 0.7) == "cheap"
        assert self._shipped(per_model, 0.7, min_samples=3) == "mid"

    def test_difference_3_the_fallback_diverges_when_nothing_clears_the_bar(self):
        # Nothing clears the threshold. The analysis rule returns argmax — the best-scoring
        # TESTED model. The shipped rule escalates to the cheapest UNTESTED model instead, so
        # the two return different models on identical evidence. This is the audit's probe.
        per_model = {
            "cheap": [False, False, True, False],  # 0.25
            "dear": [False, True, True, False],  # 0.50 — the argmax
        }
        rates = np.array([[0.25, 0.0, 0.50]], dtype=float)
        assert self._MODELS[int(kn.select_from_rates(rates, 0.7)[0])] == "dear"

        from shunt.router.selection import SelectionRule

        rule = SelectionRule(min_success_rate=0.7, min_samples=3)
        model, reason = rule.select(self._neighbors(per_model), self._pool(), False)
        assert model == "mid"  # the cheapest UNTESTED model, not the best-scoring tested one
        assert reason == "exploration_untested"

    def test_difference_1_the_shipped_rule_weights_neighbours_by_distance(self):
        # `cheap`'s passes are all FAR (distance 0.9, weight 0.1) and its failures are NEAR
        # (distance 0.0, weight 1.0). Unweighted that is 0.50; distance-weighted it is 0.09.
        # So the analysis rule takes `cheap` and the shipped rule rejects it as too weak.
        from shunt.router.selection import NeighborResult, SelectionRule

        neighbors = [
            NeighborResult("cheap", True, 1.0, 1.0, 0.9, "c1"),
            NeighborResult("cheap", True, 1.0, 1.0, 0.9, "c2"),
            NeighborResult("cheap", False, 1.0, 1.0, 0.0, "c3"),
            NeighborResult("cheap", False, 1.0, 1.0, 0.0, "c4"),
            NeighborResult("mid", True, 1.0, 1.0, 0.0, "m1"),
            NeighborResult("mid", True, 1.0, 1.0, 0.0, "m2"),
            NeighborResult("mid", True, 1.0, 1.0, 0.0, "m3"),
        ]
        rates = np.array([[0.5, 1.0, 0.0]], dtype=float)  # the UNWEIGHTED view
        assert self._MODELS[int(kn.select_from_rates(rates, 0.4)[0])] == "cheap"

        rule = SelectionRule(min_success_rate=0.4, min_samples=3)
        model, _reason = rule.select(neighbors, self._pool(), False)
        assert model == "mid"
