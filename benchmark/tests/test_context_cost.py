"""The alpha context-cost model: the identity at alpha=0, and the refusal on missing tokens.

Offline replay over committed data and registry prices only — this suite spends nothing.
"""

from __future__ import annotations

from typing import Final

import pytest

from benchmark.routing import context_cost, summary, validate
from benchmark.routing.strategies import BilledAttempt
from benchmark.routing.strategies.fixed import AlwaysCheap


def _att(model: str, cost: float, in_tok: int = 0, out_tok: int = 0, calls: int = 0):
    return BilledAttempt(model=model, cost=cost, in_tok=in_tok, out_tok=out_tok, calls=calls)


# Per-token input prices, not per-1M: the surcharge is priced per carried token.
_RATES: Final[dict[str, float]] = {"cheap": 1e-6, "strong": 1e-5}


class TestEstimatedFinalContext:
    """t = 2 x in_tok / calls, and the absence of a measurement is None, never 0."""

    def test_linear_growth_recovers_the_final_prefix(self):
        # 10 calls whose prefixes grow 200, 400, ... 2000 sum to 11_000; the estimator returns
        # 2200 — the right order, biased high by the (calls+1)/calls term the docstring names.
        assert context_cost.estimated_final_context(11_000, 10) == pytest.approx(2200.0)

    def test_no_calls_is_an_absent_measurement_not_a_zero(self):
        assert context_cost.estimated_final_context(0, 0) is None
        assert context_cost.estimated_final_context(5_000, 0) is None
        assert context_cost.estimated_final_context(5_000, -1) is None


class TestAlphaZeroIdentity:
    """alpha=0 must reproduce the published TotalCost EXACTLY — the whole model's anchor."""

    def test_a_single_task_reproduces_the_billed_sum(self):
        attempts = [_att("cheap", 1.5, 1000, 100, 4), _att("strong", 9.0, 8000, 400, 8)]
        assert context_cost.context_cost_total(attempts, 0.0, _RATES) == pytest.approx(10.5)

    def test_the_bracket_baseline_equals_the_naive_total_over_the_subset(self):
        attempts = {
            "t1": [_att("cheap", 1.5, 1000, 100, 4), _att("strong", 9.0, 8000, 400, 8)],
            "t2": [_att("cheap", 2.0, 2000, 100, 5)],
        }
        bracket = context_cost.context_bracket(attempts, (0.0, 0.1), _RATES)
        assert bracket.n_tasks == 2
        assert bracket.baseline == pytest.approx(12.5)
        assert bracket.totals[0] == pytest.approx(12.5)
        assert bracket.ratio(0.0) == pytest.approx(1.0)

    def test_the_identity_holds_on_the_real_corpus(self):
        # The same claim against committed data rather than a fixture: the model is only
        # auditable if C(0) is the number already published, on the corpus already published.
        matrix = summary.load_scored_matrix()
        if not matrix.get("results"):
            pytest.skip("results.csv holds no rows")
        tasks = sorted(matrix["results"])
        decisions, _unscorable, attempts, _judge, _sessions = summary.evaluate_billed(
            AlwaysCheap(), matrix, tasks
        )
        assert decisions
        keep = context_cost.token_complete_tasks(attempts)
        billed = sum(a.cost for tid in keep for a in attempts[tid])
        bracket = context_cost.context_bracket(attempts, (0.0,), context_cost.input_rates())
        assert bracket.baseline == pytest.approx(billed)
        assert bracket.totals[0] == pytest.approx(billed)


class TestTheSurchargeIsPricedAsACacheMiss:
    """The carried prefix is billed at the RECEIVING model's full input rate."""

    def test_alpha_scales_the_carried_prefix_linearly(self):
        attempts = [_att("cheap", 1.0, 10_000, 100, 10), _att("strong", 5.0, 4_000, 200, 4)]
        # t after attempt 1 = 2 * 10_000 / 10 = 2000 tokens, received by `strong` at 1e-5/token.
        assert context_cost.context_cost_total(attempts, 1.0, _RATES) == pytest.approx(6.02)
        assert context_cost.context_cost_total(attempts, 0.1, _RATES) == pytest.approx(6.002)

    def test_the_first_attempt_of_a_task_carries_nothing(self):
        one = [_att("strong", 5.0, 4_000, 200, 4)]
        assert context_cost.context_cost_total(one, 1.0, _RATES) == pytest.approx(5.0)


class TestTokenCompletenessRaisesRatherThanDropping:
    """A token-less attempt is the ABSENCE of a measurement, and the model refuses it."""

    def test_an_attempt_without_tokens_raises(self):
        attempts = [_att("cheap", 1.0, 10_000, 100, 10), _att("strong", 5.0)]
        with pytest.raises(context_cost.TokenIncompleteError, match="no measured tokens"):
            context_cost.context_cost_total(attempts, 1.0, _RATES)

    def test_the_subset_excludes_the_task_rather_than_charging_it_zero(self):
        attempts = {
            "good": [_att("cheap", 1.0, 10_000, 100, 10)],
            "imputed": [_att("cheap", 1.0)],
        }
        assert context_cost.token_complete_tasks(attempts) == {"good"}
        assert context_cost.context_bracket(attempts, (1.0,), _RATES).n_tasks == 1

    def test_an_empty_attempt_list_is_not_token_complete(self):
        assert context_cost.token_complete_tasks({"t": []}) == set()


class TestBracketRestsOnMeasuredCellsOnly:
    """The token filter and the `imputed` flag are two independent signals; they must agree.

    The obvious ceiling — the tasks measured on EVERY model — is not the right one: a bracket
    needs tokens only on the models its path billed, so the real invariant is per-path.
    """

    _MATRIX = {
        "models": {"cheap": {}, "strong": {}},
        "results": {
            "real": {"cheap": {"in_tok": 9, "calls": 3}},
            "projected": {"cheap": {"imputed": True}},
        },
    }

    def test_a_path_through_real_cells_is_accepted(self):
        validate.enforce_bracket_coverage(["real"], {"real": ["cheap"]}, self._MATRIX)

    def test_a_path_through_an_imputed_cell_is_refused(self):
        with pytest.raises(validate.DataIntegrityError, match="IMPUTED"):
            validate.enforce_bracket_coverage(
                ["real", "projected"], {"projected": ["cheap"]}, self._MATRIX
            )

    def test_more_bracket_tasks_than_scored_tasks_is_refused(self):
        with pytest.raises(validate.DataIntegrityError, match="scored matrix holds"):
            validate.enforce_bracket_coverage(["real", "projected", "ghost"], {}, self._MATRIX)

    def test_the_two_signals_agree_on_the_real_corpus(self):
        # The published check, run against committed data: every task the token filter keeps for
        # the shipped default's own path is a task whose billed cells are all real.
        matrix = summary.load_scored_matrix()
        if not matrix.get("results"):
            pytest.skip("results.csv holds no rows")
        tasks = sorted(matrix["results"])
        _dec, _uns, attempts, _judge, _sessions = summary.evaluate_billed(
            AlwaysCheap(), matrix, tasks
        )
        bracket = context_cost.context_bracket(attempts, (1.0,), context_cost.input_rates())
        assert 0 < bracket.n_tasks <= len(tasks)
        validate.enforce_bracket_coverage(
            bracket.tasks,
            {tid: [a.model for a in attempts[tid]] for tid in bracket.tasks},
            matrix,
        )


class TestAnEmptyBracketPublishesNothing:
    """A bracket resting on zero token-complete tasks may not emit a surcharge number."""

    # The latent failure this pins: `ratio()` used to return 1.0 on a zero baseline, so a corpus
    # with no token columns published `context_cost_alpha_* == cache_total` — the affirmative
    # claim "carrying context costs nothing" — beside `context_cost_n: 0`, every gate green.

    def test_a_corpus_with_no_token_columns_yields_an_empty_bracket(self):
        bracket = context_cost.context_bracket({"t1": [_att("cheap", 1.0)]}, (0.1,), _RATES)
        assert bracket.n_tasks == 0
        assert bracket.publishable is False

    def test_ratio_refuses_an_empty_bracket_instead_of_defaulting_to_one(self):
        bracket = context_cost.ContextBracket(alphas=(0.1,), totals=(0.0,), baseline=0.0, tasks=())
        with pytest.raises(context_cost.EmptyBracketError, match="no publishable ratio"):
            bracket.ratio(0.1)

    def test_ratio_refuses_a_zero_baseline_even_with_tasks(self):
        bracket = context_cost.ContextBracket(
            alphas=(0.1,), totals=(0.0,), baseline=0.0, tasks=("t1",)
        )
        with pytest.raises(context_cost.EmptyBracketError):
            bracket.ratio(0.1)

    def test_the_summary_omits_the_alpha_columns_rather_than_defaulting_them(self):
        # The exact shape the defect published: a strategy row whose attempts carry no tokens.
        decisions = [("t1", "cheap", True, 1.0)]
        attempts = {"t1": [_att("cheap", 1.0)]}
        matrix = {"models": {"cheap": {}}, "results": {"t1": {"cheap": {}}}}
        cols = summary._context_columns(decisions, attempts, 12.3456, matrix)
        assert cols == {"context_cost_n": 0}
        assert "context_cost_alpha_10" not in cols

    def test_a_bracket_task_that_billed_no_model_fails_closed(self):
        # `billed_models.get(tid, ())` made `any()` False, so an empty mapping passed vacuously.
        matrix = {"models": {"cheap": {}}, "results": {"t1": {"cheap": {"imputed": True}}}}
        with pytest.raises(validate.DataIntegrityError, match="billed NO model"):
            validate.enforce_bracket_coverage(["t1"], {}, matrix)
