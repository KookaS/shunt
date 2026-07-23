from __future__ import annotations

import logging
import math
import threading
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

from shunt.router.budget import ConservativeGate, ExplorationBudget
from shunt.router.cold_start import ColdStartStrategy
from shunt.router.embedder import Embedder
from shunt.router.exploration import CandidateStats, ExplorationDecision, ThompsonSampler
from shunt.router.policy import ExplorationPolicy
from shunt.router.provenance import build_provenance
from shunt.router.selection import (
    ModelPoolProtocol,
    NeighborResult,
    SelectionRule,
    _confidence_weight,
)
from shunt.router.strategies import RoutingStrategy
from shunt.router.strategies.knn import KnnStrategy


class SessionManagerProtocol(Protocol):
    """Minimal session-manager interface the engine holds a reference to."""

    def get_session(self, session_id: str) -> object: ...


class EmbedderProtocol(Protocol):
    """Minimal embedder interface: text in, vector out."""

    def embed(self, text: str) -> npt.NDArray[np.float32]: ...


logger = logging.getLogger(__name__)


class OutcomeIndex(Protocol):
    """Interface the RouterEngine expects from the outcome + HNSW index.

    Implemented by the DB module (separate build) — the engine never
    hardcodes DB paths or storage internals.
    """

    def count_labeled(self) -> int:
        """Return the number of sessions with Tier-2 (verified) outcomes."""

    def count_total_labeled(self) -> int:
        """Return the total number of sessions with any labeled outcome (Tier-1 or Tier-2)."""

    def effective_labeled(self) -> float:
        """Return the effective sample size ``nₑ`` over all labeled outcomes."""

    def effective_tier2(self) -> float:
        """Return the effective sample size ``nₑ`` over Tier-2 (verified) outcomes."""

    def model_priors(self) -> dict[str, tuple[float, float]]:
        """Return per-model ``(estimate, strength)`` offline aggregates for prior seeding."""

    def query(
        self,
        embedding: npt.NDArray[np.float32],
        k: int = 20,
    ) -> list[NeighborResult]:
        """Return the *k* nearest labeled sessions to *embedding*."""


class RouterEngine:
    """Top-level router decision engine, tying together the embedder, outcome
    index (kNN), and selection rule. Embeddings are cached per session.
    """

    def __init__(  # noqa: PLR0913 (config-heavy constructor wiring router collaborators)
        self,
        model_pool: ModelPoolProtocol,
        session_manager: SessionManagerProtocol,
        outcome_index: OutcomeIndex,
        embedder: EmbedderProtocol | None = None,
        cold_start_threshold: int = 20,
        selection_rule: SelectionRule | None = None,
        cold_start_strategy: ColdStartStrategy | None = None,
        exploration: ExplorationPolicy | None = None,
        sampler: ThompsonSampler | None = None,
        budget: ExplorationBudget | None = None,
        conservative_gate: ConservativeGate | None = None,
        strategy: RoutingStrategy | None = None,
        neighbor_k: int = 20,
        trust_neighbors: bool = True,
    ) -> None:
        self._model_pool = model_pool
        self._session_manager = session_manager
        self._outcome_index = outcome_index
        self._embedder = embedder or Embedder()
        self._cold_start_strategy = cold_start_strategy or ColdStartStrategy()
        self._selection_rule = selection_rule or SelectionRule(
            cold_start_threshold=cold_start_threshold,
        )
        # Default strategy = knn (wraps the selection rule) so engine-only callers keep
        # today's behavior; the server injects the strategy named by router.strategy.
        self._strategy: RoutingStrategy = strategy or KnnStrategy(self._selection_rule)
        self._neighbor_k = neighbor_k
        # False when the configured embedding fingerprint mismatches the stored corpus:
        # the stored vectors are in a foreign space, so kNN over them is garbage. Decided
        # in server.py (where both fingerprints are visible) and injected here.
        self._trust_neighbors = trust_neighbors
        self._cold_start_threshold = cold_start_threshold
        self._cache: dict[str, npt.NDArray[np.float32]] = {}
        # Guards the embedding cache and the budget check→select→record section so the
        # cap stays enforceable under the async/threaded ProxyRouter (no check-then-act race).
        self._lock = threading.Lock()
        self._exploration = exploration
        self._sampler, self._budget, self._conservative_gate = self._build_exploration(
            exploration, sampler, budget, conservative_gate
        )

    @staticmethod
    def _build_exploration(
        exploration: ExplorationPolicy | None,
        sampler: ThompsonSampler | None,
        budget: ExplorationBudget | None,
        gate: ConservativeGate | None,
    ) -> tuple[ThompsonSampler | None, ExplorationBudget | None, ConservativeGate | None]:
        """Wire the exploration collaborators only when a policy enables exploration."""
        if exploration is None or not exploration.enabled:
            return (None, None, None)
        import numpy as np_rt  # runtime RNG (injected sampler overrides in tests)

        built_sampler = sampler or ThompsonSampler(
            np_rt.random.default_rng(),
            exploration.prior_alpha,
            exploration.prior_beta,
            exploration.prior_strength_cap,
        )
        built_budget = budget or ExplorationBudget(exploration.explore_budget_frac)
        built_gate = gate or ConservativeGate(exploration.conservative_alpha)
        return (built_sampler, built_budget, built_gate)

    def _compute_candidate_model_scores(
        self,
        neighbors: list[NeighborResult],
    ) -> dict[str, float]:
        """Compute per-model weighted success rates from *neighbors*."""
        groups: dict[str, list[NeighborResult]] = {}
        for n in neighbors:
            groups.setdefault(n.model, []).append(n)

        scores: dict[str, float] = {}
        for model, group in groups.items():
            weights = [_confidence_weight(n) for n in group]
            total_weight = sum(weights)
            if total_weight <= 0:
                scores[model] = 0.0
                continue
            weighted_success = (
                sum(w * (1.0 if n.outcome else 0.0) for w, n in zip(weights, group, strict=True))
                / total_weight
            )
            scores[model] = weighted_success
        return scores

    def decide(self, session_id: str, prompt_text: str) -> tuple[str, str, dict[str, Any]]:
        """Return ``(model_name, reason, provenance_dict)`` for *prompt_text*.

        Checks cold-start; embeds and queries the outcome index; applies the
        selection rule.
        """
        if not self._strategy.consults_neighbors:
            return self._decide_fixed()

        if not self._trust_neighbors:
            # Foreign embedding space: the stored corpus was built by a different embedder,
            # so its neighbours are meaningless. Route as if cold (no embed, no query) until
            # `shunt reindex` re-embeds the corpus and the fingerprint matches again.
            return self._decide_stale_space()

        # Effective-sample-size gate (nₑ), not raw count: a low-confidence or decayed outcome
        # counts for less, so cold-start ends on *trustworthy* evidence, not row volume.
        ne_total = self._outcome_index.effective_labeled()
        ne_tier2 = self._outcome_index.effective_tier2()
        cold_start_active = self._cold_start_strategy.is_active_effective(ne_total, ne_tier2)
        logger.debug(
            "decide[%s]: strategy=%s ne_labeled=%.2f ne_tier2=%.2f "
            "cold_start_threshold=%d cold_start_active=%s",
            session_id,
            type(self._strategy).__name__,
            ne_total,
            ne_tier2,
            self._cold_start_threshold,
            cold_start_active,
        )

        # Embed every neighbor-consulting session, cold-start included: cold-start exists
        # only to gather the kNN bootstrap corpus, so its sessions must be indexable —
        # returning before this left the HNSW index permanently empty.
        embedding = self._get_embedding(session_id, prompt_text)

        if cold_start_active:
            model_name = self._cold_start_strategy.select(self._model_pool)
            provenance = build_provenance(
                model_chosen=model_name,
                selection_rule_used="cold_start",
                fallback_chain_triggered=False,
                router_propensity=1.0,
                candidate_model_scores={},
            )
            return (model_name, "cold_start", provenance)

        neighbors = self._outcome_index.query(embedding, k=self._neighbor_k)
        if logger.isEnabledFor(logging.DEBUG):
            # The routing evidence itself: too few neighbours here is the difference
            # between a learned choice and a silent fall-through to the cheapest model.
            logger.debug(
                "decide[%s]: kNN k=%d returned=%d neighbors=[%s]",
                session_id,
                self._neighbor_k,
                len(neighbors),
                ", ".join(
                    f"{n.model}/{'ok' if n.outcome else 'fail'}"
                    f"/d={n.distance:.3f}/conf={n.verification_confidence:.2f}"
                    for n in neighbors[:10]
                )
                or "none",
            )

        with self._lock:  # budget check→select→record is one critical section
            model_name, reason, provenance = self._route_locked(neighbors)
        logger.debug("decide[%s]: chose model=%s reason=%s", session_id, model_name, reason)
        return (model_name, reason, provenance)

    def _decide_stale_space(self) -> tuple[str, str, dict[str, Any]]:
        """Route via the cold-start default when the corpus is in a stale embedding space.

        No embed, no index query — the reason token surfaces in ``shunt explain``.
        """
        model_name = self._cold_start_strategy.select(self._model_pool)
        provenance = build_provenance(
            model_chosen=model_name,
            selection_rule_used="stale_embedding_space",
            fallback_chain_triggered=False,
            router_propensity=1.0,
            candidate_model_scores={},
        )
        return (model_name, "stale_embedding_space", provenance)

    def _decide_fixed(self) -> tuple[str, str, dict[str, Any]]:
        """Route a neighbor-independent (fixed) strategy: no cold-start, embedding, or query.

        A fixed strategy (always_cheap/always_frontier) is a pure function of the pool, so it
        must route deterministically from the first turn — never masked by cold-start warmup.
        """
        model_name, reason = self._strategy.select([], self._model_pool)
        provenance = build_provenance(
            model_chosen=model_name,
            selection_rule_used=reason,
            fallback_chain_triggered=False,
            router_propensity=1.0,
            candidate_model_scores={},
        )
        return (model_name, reason, provenance)

    def _route_locked(self, neighbors: list[NeighborResult]) -> tuple[str, str, dict[str, Any]]:
        """Choose a model (exploration if budgeted, else selection rule) — caller holds the lock."""
        explored = self._maybe_explore(neighbors)
        if explored is not None:
            return explored

        model_name, reason = self._strategy.select(neighbors, self._model_pool)
        if self._budget is not None:
            # Every decision feeds the baseline, exploit ones included: the cap is a ratio
            # against cumulative baseline spend, so a frozen denominator disables it.
            cost = self._budget_cost(self._representative_cost(neighbors, model_name))
            self._budget.record(baseline_cost=cost, actual_cost=cost, is_exploratory=False)

        candidate_scores = self._compute_candidate_model_scores(neighbors)
        fallback = reason in ("exploration_untested", "safe_fallback")
        provenance = build_provenance(
            model_chosen=model_name,
            selection_rule_used=reason,
            neighbors=neighbors,
            fallback_chain_triggered=fallback,
            tier_escalation_reason=reason if fallback else None,
            router_propensity=candidate_scores.get(model_name, 1.0),
            candidate_model_scores=candidate_scores,
        )
        return (model_name, reason, provenance)

    def _candidate_stats(
        self,
        neighbors: list[NeighborResult],
        priors: dict[str, tuple[float, float]] | None = None,
    ) -> list[CandidateStats]:
        """Aggregate the neighborhood's weighted verified counts per model.

        ``priors`` seeds each model's informative Thompson prior from its offline aggregate;
        omitted (e.g. the cost-only ``_representative_cost`` path) leaves the flat prior.
        """
        priors = priors or {}
        groups: dict[str, list[NeighborResult]] = {}
        for n in neighbors:
            groups.setdefault(n.model, []).append(n)
        stats: list[CandidateStats] = []
        for model, group in groups.items():
            weights = [_confidence_weight(n) for n in group]
            total = sum(weights)
            if total <= 0:
                continue
            successes = sum(w for w, n in zip(weights, group, strict=True) if n.outcome)
            cost = sum(w * n.cost for w, n in zip(weights, group, strict=True)) / total
            if not math.isfinite(cost):  # a non-finite upstream cost must never sort cheapest
                cost = math.inf
            prior = priors.get(model)
            stats.append(
                CandidateStats(
                    model=model,
                    successes=successes,
                    failures=total - successes,
                    cost=cost,
                    prior_estimate=prior[0] if prior is not None else None,
                    prior_strength=prior[1] if prior is not None else 0.0,
                )
            )
        return stats

    def _representative_cost(self, neighbors: list[NeighborResult], model: str) -> float:
        """Confidence-weighted cost of *model*; the priciest known model when it's untested."""
        total_weight = 0.0
        total_cost = 0.0
        for n in neighbors:
            if n.model != model:
                continue
            w = _confidence_weight(n)
            total_weight += w
            total_cost += w * n.cost
        if total_weight > 0:
            return total_cost / total_weight
        # No history for this model — which is exactly the escalation case
        # (exploration_untested / safe_fallback), i.e. the most expensive routes. Returning
        # 0.0 booked those as free and let them inflate the budget's baseline for nothing;
        # the priciest observed model is a conservative stand-in. Reading a price table here
        # is not an option: the router must not depend on benchmark pricing (SH005).
        known = [s.cost for s in self._candidate_stats(neighbors) if math.isfinite(s.cost)]
        return max(known) if known else 0.0

    @staticmethod
    def _budget_cost(cost: float) -> float:
        """Non-finite cost → 0.0 so it can never poison the cumulative budget counters."""
        return cost if math.isfinite(cost) else 0.0

    def _maybe_explore(
        self, neighbors: list[NeighborResult]
    ) -> tuple[str, str, dict[str, Any]] | None:
        """Run the cost-aware Thompson layer if enabled and within budget; else None."""
        if self._sampler is None or self._budget is None or self._exploration is None:
            return None
        if not self._budget.can_explore():
            return None
        # Seed informative priors from the offline per-model aggregate (empirical Bayes);
        # a model with accumulated verified history no longer restarts at Beta(1,1).
        stats = self._candidate_stats(neighbors, self._outcome_index.model_priors())
        if not stats:
            return None
        decision = self._sampler.select(
            stats,
            self._selection_rule.min_success_rate,
            self._exploration.propensity_mc_samples,
        )
        model_name, reason, propensity = self._resolve_exploration(decision, stats)
        chosen_cost = next((s.cost for s in stats if s.model == model_name), 0.0)
        greedy_cost = next((s.cost for s in stats if s.model == decision.greedy_model), chosen_cost)
        self._budget.record(
            baseline_cost=self._budget_cost(greedy_cost),
            actual_cost=self._budget_cost(chosen_cost),
            is_exploratory=(reason == "exploration"),
        )
        provenance = build_provenance(
            model_chosen=model_name,
            selection_rule_used=reason,
            neighbors=neighbors,
            fallback_chain_triggered=False,
            router_propensity=propensity,
            candidate_model_scores=self._compute_candidate_model_scores(neighbors),
            # A taken exploration cheaper than the greedy pick is the downshift the gate
            # learns from; an upshift or a conservative_fallback to greedy is not.
            downshift=(reason == "exploration" and chosen_cost < greedy_cost),
        )
        return (model_name, reason, provenance)

    def _resolve_exploration(
        self, decision: ExplorationDecision, stats: list[CandidateStats]
    ) -> tuple[str, str, float]:
        """Apply the conservative gate: block an exploratory *downshift* without slack."""
        if not decision.is_exploratory:
            return (decision.model, "exploration_exploit", decision.propensity)
        costs = {s.model: s.cost for s in stats}
        is_downshift = costs[decision.model] < costs[decision.greedy_model]
        gate = self._conservative_gate
        if is_downshift and gate is not None and not gate.allows_downshift():
            return (decision.greedy_model, "conservative_fallback", 1.0)
        return (decision.model, "exploration", decision.propensity)

    def snapshot_exploration_state(self) -> dict[str, Any]:
        """Serialize the mutable exploration state (budget cap + gate slack) for persistence.

        Empty dict when exploration is disabled — nothing mutable to persist.
        """
        # Under the same lock as decide()/record_outcome(): the budget and gate are not
        # thread-safe on their own, and a snapshot read concurrent with a decision's record()
        # could serialize a torn counter pair. Reading is cheap; it never runs mid-decision.
        state: dict[str, Any] = {}
        with self._lock:
            if self._budget is not None:
                state["budget"] = self._budget.snapshot()
            if self._conservative_gate is not None:
                state["gate"] = self._conservative_gate.snapshot()
        return state

    def restore_exploration_state(self, state: dict[str, Any] | None) -> None:
        """Load a persisted snapshot into the budget + gate; a missing/partial one no-ops."""
        if not state:
            return
        with self._lock:
            if self._budget is not None:
                self._budget.restore(state.get("budget"))
            if self._conservative_gate is not None:
                self._conservative_gate.restore(state.get("gate"))

    def record_outcome(self, *, downshift: bool, success: bool) -> None:
        """Feed a verified downshift-exploration outcome to the conservative safety gate."""
        # `downshift` = the routed decision picked a cheaper/weaker model than the greedy
        # exploit pick (persisted per-decision by the pending queue when the read-back
        # loop is wired). Only these outcomes move the gate — see ConservativeGate.
        # Under the same lock as routing: the gate's slack is a read-modify-write, and
        # ExplorationBudget/ConservativeGate document that RouterEngine serializes their
        # callers. Outcomes arrive from a different path than decisions, so without this a
        # lost update would under-count verified downshift evidence and hold the gate shut.
        if self._conservative_gate is not None:
            with self._lock:
                self._conservative_gate.record_outcome(downshift=downshift, success=success)

    def get_neighbors(
        self,
        session_id: str,
        prompt_text: str,
        k: int | None = None,
    ) -> list[NeighborResult]:
        """Return raw kNN neighbors for *prompt_text* (for ``SHUNT EXPLAIN``).

        Defaults to the configured ``neighbor_k`` so EXPLAIN shows the same neighborhood
        that drove routing.
        """
        embedding = self._get_embedding(session_id, prompt_text)
        return self._outcome_index.query(embedding, k=k if k is not None else self._neighbor_k)

    def cached_embedding(self, session_id: str) -> npt.NDArray[np.float32] | None:
        """Return the embedding computed for *session_id* (None for fixed strategies)."""
        with self._lock:
            return self._cache.get(session_id)

    def warm(self) -> None:
        """Pre-load the embedding model so the first request does not pay for it."""
        warm = getattr(self._embedder, "warm", None)
        if callable(warm):
            warm()

    def _get_embedding(self, session_id: str, prompt_text: str) -> npt.NDArray[np.float32]:
        with self._lock:
            cached = self._cache.get(session_id)
        if cached is not None:
            return cached
        embedding = self._embedder.embed(prompt_text)  # slow — computed outside the lock
        with self._lock:
            return self._cache.setdefault(session_id, embedding)
