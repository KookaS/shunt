"""A seeded synthetic corpus that populates the inference figures — illustration, never evidence."""

# WHAT THIS IS. The seven inference figures are drawn from the owner's live outcome store, which
# on 2026-08-18 held 40 live sessions — too few for a reader to see what any panel is SHAPED
# like. This module mints a few hundred sessions that a reader can look at instead. It answers
# "what does F3 look like when it has data", and nothing else.
#
# WHAT IT IS NOT. It is not measurement, and no number it produces may be quoted, cited, or
# compared against a baseline. Three fences make a stray row greppable rather than plausible:
# every session id carries the `demo:` prefix (never `bench:`, which would classify the corpus
# as SEEDED), every decision provenance carries `synthetic_demo: true`, and the family it draws
# lives in its own figure half on a watermarked canvas that no analysis reads.
#
# HOW THE SHAPE IS PRESERVED. The 40 measured live rows below are the empirical distribution,
# resampled JOINTLY with replacement: one draw takes a whole row — its model, its selection
# rule, its cost, its `cost_known` flag, its propensity, its fingerprint, its provenance and its
# label — so every marginal AND every correlation between them survives by construction. That is
# why three of the four `cost_known=0` rows still carry a NONZERO cost (unknown is not zero), why
# ~70% of rows still have a NULL propensity, and why only ~35% carry any outcome at all. A smooth
# parametric fit was rejected outright: the measured cost and duration distributions are sparse
# and heavy-tailed (durations run 3.1 s to 659.7 s over 40 rows), and an i.i.d. generator drawn
# from a fitted family would teach the reader a shape the router has never produced.
#
# THE ONE DELIBERATE DEPARTURE is the arrival RATE: the measured 28-day span is kept while the
# count is scaled 7.5x, so the windowed panels have enough rows to be legible. Nothing else is
# scaled. Timestamps are anchored to a FIXED end date, so the `7d`/`30d` windows — which are
# measured against wall clock — thin out as the anchor recedes; the `all` window does not.
#
# WHY THE ROWS ARE WRITTEN THROUGH `store_session` + `append_outcome_event` AND NOT RAW SQL.
# `outcomes` is a projection of the append-only `outcome_events` log (`_materialize_outcome`).
# Writing the projection directly desynchronises the log from the view, and only the v5 backfill
# repairs that — so the demo takes the production path the live router takes.

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np

from benchmark.routing.docs_corpus import _bundle_path
from benchmark.routing.seed_live import _load_bundle
from shunt.db.store import OutcomeEvent, OutcomeStore, SessionProvenance

DEMO_PREFIX: Final[str] = "demo:"
DEMO_SEED: Final[int] = 20260820
DEMO_SESSIONS: Final[int] = 300

# The measured span of the 40 live rows (2026-07-20T11:26:50Z .. 2026-08-17T13:30:12Z), and a
# fixed anchor so two builds at the same seed are byte-identical.
_SPAN_SECONDS: Final[float] = 2426601.942414
_ANCHOR: Final[datetime] = datetime(2026, 8, 18, tzinfo=timezone.utc)  # noqa: UP017

# Smoothed-bootstrap kernel width. 40 atoms drawn 300 times puts ~8 rows on each exact measured
# value, and a reader reads that quantisation as structure ("why are there eight sessions at
# exactly $0.0083?"). +/-10% multiplicative is narrower than the gap between adjacent measured
# atoms, so it cannot move mass between modes, blur the heavy tail, or change any marginal it is
# applied to. It is a kernel, not a fitted family.
_JITTER: Final[float] = 0.10


@dataclass(frozen=True)
class _LiveAtom:
    """One measured live session, carried whole so a draw preserves its internal correlations."""

    model: str
    cost: float
    cost_known: bool
    duration: float
    propensity: float | None
    fingerprint: str | None
    prompt_tokens: int
    provenance: dict[str, Any]
    # (tier1 outcome, tier1 confidence, tier2 outcome, tier2 confidence, tier1 source,
    # tier2 source), or None for the ~65% of live sessions that were never labelled.
    outcome: tuple[str, float, str, float, str, str] | None


@dataclass(frozen=True)
class _DemoRow:
    """One drawn session: an atom, its own id and stamp, and its jittered continuous fields."""

    session_id: str
    timestamp: str
    atom: _LiveAtom
    cost: float
    duration: float


def build_demo_store(db_path: Path | str, *, seed: int = DEMO_SEED) -> Path:
    """Write the illustrative corpus to *db_path* through the production writers."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in path.parent.glob(path.name + "*"):
        stale.unlink()
    rng = random.Random(seed)
    rows = _draw_rows(rng)
    vectors = _borrowed_vectors(rng, len(rows))
    store = OutcomeStore(db_path=str(path))
    try:
        for row, vector in zip(rows, vectors, strict=True):
            _write_row(store, row, vector)
    finally:
        store.close()
    return path


def _draw_rows(rng: random.Random) -> list[_DemoRow]:
    """Resample whole atoms and empirical inter-arrival gaps into the demo's session list."""
    atoms = rng.choices(_ATOMS, k=DEMO_SESSIONS)
    gaps = rng.choices(_GAPS, k=DEMO_SESSIONS)
    scale = _SPAN_SECONDS / sum(gaps)
    start = _ANCHOR - timedelta(seconds=_SPAN_SECONDS)
    rows: list[_DemoRow] = []
    elapsed = 0.0
    for index, (atom, gap) in enumerate(zip(atoms, gaps, strict=True)):
        elapsed += gap * scale
        rows.append(
            _DemoRow(
                session_id=f"{DEMO_PREFIX}{index:04d}",
                timestamp=(start + timedelta(seconds=elapsed)).isoformat(),
                atom=atom,
                cost=_jitter(rng, atom.cost),
                duration=_jitter(rng, atom.duration),
            )
        )
    return rows


def _jitter(rng: random.Random, value: float) -> float:
    """Spread one measured value inside the bootstrap kernel; an exact zero stays exact."""
    # SH008 reads a draw off a bound `random.Random` as a fabricated feature vector, which is
    # the right default under the real-only rule. This draw is neither a feature nor an
    # embedding: it is the kernel width of a smoothed bootstrap over `cost` and
    # `session_duration_seconds`, two scalar COLUMNS that no embedder, index or estimator ever
    # reads. The output of this module is fenced from measurement by construction — `demo:` ids
    # that classify LIVE only inside a throwaway demo store, `synthetic_demo: true` on every
    # row, its own watermarked figure half, and no analysis, report or gate that reads it — and
    # the generator plus its seed are recorded in the figure job's digest, so the provenance of
    # every pixel it produces resolves back to this file rather than to a measurement. The
    # embeddings this corpus carries are REAL vectors borrowed from the committed seed bundle
    # (see `_borrowed_vectors`); none is fabricated here.
    if value <= 0.0:
        return value
    return value * (1.0 + rng.uniform(-_JITTER, _JITTER))  # noqa: SH008


def _borrowed_vectors(rng: random.Random, count: int) -> list[np.ndarray]:
    """Real embeddings from the committed seed bundle, one vector per demo session."""
    # THE EMBEDDING DECISION, made explicitly rather than by default. There is no honest
    # synthetic source for a 768-d code embedding: a random or hashed vector is exactly the
    # proxy-featurizer fake SH008 exists to stop, and it would give F4 a neighbourhood geometry
    # that no embedder could produce — the one panel where a reader would most easily mistake
    # invention for measurement. The alternative, emitting no embeddings at all, leaves F4's
    # panels empty and defeats the only purpose this module has. So the vectors are REAL, taken
    # from the committed LFS seed bundle: the GEOMETRY (distances, cluster structure, the
    # distance scale F4 draws) is genuine, and the ASSOCIATION between a vector and the demo
    # session that carries it is invented. F4 therefore shows a truthful-looking neighbourhood
    # whose reliability curve is meaningless, which is true of every panel in this family.
    bundle = _load_bundle(_bundle_path())
    embeddings = bundle["embedding"]
    order = list(range(len(embeddings)))
    rng.shuffle(order)
    return [np.asarray(embeddings[order[i % len(order)]], dtype=np.float32) for i in range(count)]


def _write_row(store: OutcomeStore, row: _DemoRow, vector: np.ndarray) -> None:
    """Persist one demo session and, when the atom carried one, its outcome events."""
    atom = row.atom
    provenance = dict(atom.provenance)
    provenance["synthetic_demo"] = True
    store.store_session(
        session_id=row.session_id,
        prompt_text=f"synthetic demo session {row.session_id}",
        embedding=vector,
        model_chosen=atom.model,
        cost=row.cost,
        cache_stats={"cache_tax": 0.0, "prompt_tokens": atom.prompt_tokens},
        duration=row.duration,
        timestamp=row.timestamp,
        decision_provenance=provenance,
        # `selection_propensity` is 1.0 or NULL and never anything else, because that is what
        # the shipped router writes on the policy path. Depicting an epsilon-greedy propensity
        # would advertise a randomization the router does not ship, and would flip F7 from its
        # honest NOT_IDENTIFIED refusal to a fabricated identified estimate.
        provenance=SessionProvenance(
            cost_known=atom.cost_known,
            selection_propensity=atom.propensity,
            model_fingerprint=atom.fingerprint,
        ),
    )
    if atom.outcome is not None:
        _append_outcome(store, row, atom.outcome)


def _append_outcome(
    store: OutcomeStore, row: _DemoRow, outcome: tuple[str, float, str, float, str, str]
) -> None:
    """Append the atom's tier-1 and tier-2 events; the store projects the `outcomes` view."""
    tier1, confidence1, tier2, confidence2, source1, source2 = outcome
    for tier, verdict, confidence, source in (
        (1, tier1, confidence1, source1),
        (2, tier2, confidence2, source2),
    ):
        store.append_outcome_event(
            OutcomeEvent(
                session_id=row.session_id,
                tier=tier,
                source=source,
                outcome=verdict,
                confidence=confidence,
                run_signature=f"{row.session_id}|demo|{tier}",
                model_fingerprint=row.atom.fingerprint,
                created_at=row.timestamp,
            )
        )


# The 39 measured inter-arrival gaps, in seconds, between the 40 live sessions. Resampled rather
# than modelled for the same reason the rows are: real traffic is bursty (7 s to 19 days here)
# and a Poisson process would draw a calm that never happened.
_GAPS: Final[tuple[float, ...]] = (
    257.979,
    155.615,
    1301.084,
    415.463,
    257.861,
    235.531,
    9969.95,
    790.867,
    3112.506,
    58.919,
    1115.085,
    44.342,
    816.064,
    208.561,
    546.445,
    2735.889,
    47994.854,
    470.08,
    2051.728,
    84.902,
    177857.316,
    826.889,
    1044.01,
    199.208,
    1621587.54,
    405.233,
    7.379,
    7.717,
    7.79,
    189.524,
    7.226,
    7.32,
    30.115,
    674.94,
    14.097,
    20.977,
    13.7,
    10824.103,
    540253.131,
)


# The empirical distribution: the owner's 40 live rows as of 2026-08-18, read from a read-only
# copy of the outcome store, with prompt text dropped and the one recorded checkpoint id
# replaced by a demo token. Nothing here is invented; nothing here is evidence either, because a
# resampling of 40 rows measures nothing the 40 rows did not already measure.
_ATOMS: Final[tuple[_LiveAtom, ...]] = (
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.00053312,
        cost_known=True,
        duration=28.228092193603516,
        propensity=None,
        fingerprint=None,
        prompt_tokens=17,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=("success", 1.0, "success", 1.0, "human", "human"),
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.0003376,
        cost_known=True,
        duration=15.91381573677063,
        propensity=None,
        fingerprint=None,
        prompt_tokens=11,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.00025248,
        cost_known=True,
        duration=15.662911891937256,
        propensity=None,
        fingerprint=None,
        prompt_tokens=17,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.0018143999999999999,
        cost_known=True,
        duration=211.3648681640625,
        propensity=None,
        fingerprint=None,
        prompt_tokens=2420,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.006898368,
        cost_known=True,
        duration=43.46024751663208,
        propensity=None,
        fingerprint=None,
        prompt_tokens=14650,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.007763904,
        cost_known=True,
        duration=18.701792240142822,
        propensity=None,
        fingerprint=None,
        prompt_tokens=20865,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.00040992000000000003,
        cost_known=True,
        duration=18.053621768951416,
        propensity=None,
        fingerprint=None,
        prompt_tokens=17,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.00017792,
        cost_known=True,
        duration=12.883657932281494,
        propensity=None,
        fingerprint=None,
        prompt_tokens=12,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.021807488,
        cost_known=True,
        duration=659.6676714420319,
        propensity=None,
        fingerprint=None,
        prompt_tokens=20374,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.012436672000000001,
        cost_known=True,
        duration=66.03037023544312,
        propensity=None,
        fingerprint=None,
        prompt_tokens=20983,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.00652864,
        cost_known=True,
        duration=4.64693808555603,
        propensity=None,
        fingerprint=None,
        prompt_tokens=20374,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.00399968,
        cost_known=True,
        duration=29.034309148788452,
        propensity=None,
        fingerprint=None,
        prompt_tokens=3321,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.010820288,
        cost_known=True,
        duration=25.756759881973267,
        propensity=None,
        fingerprint=None,
        prompt_tokens=25621,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.008473984,
        cost_known=True,
        duration=25.99604368209839,
        propensity=None,
        fingerprint=None,
        prompt_tokens=20821,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.0033831679999999998,
        cost_known=True,
        duration=31.567083597183228,
        propensity=None,
        fingerprint=None,
        prompt_tokens=20821,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.012246272,
        cost_known=True,
        duration=82.93779873847961,
        propensity=None,
        fingerprint=None,
        prompt_tokens=20950,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.00016352,
        cost_known=True,
        duration=3.138690948486328,
        propensity=None,
        fingerprint=None,
        prompt_tokens=279,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.0002864,
        cost_known=True,
        duration=21.724759578704834,
        propensity=None,
        fingerprint=None,
        prompt_tokens=15,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.049872672,
        cost_known=True,
        duration=207.88142371177673,
        propensity=None,
        fingerprint=None,
        prompt_tokens=40344,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.019449664,
        cost_known=True,
        duration=37.50681471824646,
        propensity=None,
        fingerprint=None,
        prompt_tokens=40807,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.013159232,
        cost_known=True,
        duration=26.654756784439087,
        propensity=None,
        fingerprint=None,
        prompt_tokens=23915,
        provenance={
            "candidate_model_scores": {},
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.014279488,
        cost_known=True,
        duration=91.84007906913757,
        propensity=1.0,
        fingerprint="alibaba/qwen3.7-plus@qwen3.7-plus",
        prompt_tokens=24280,
        provenance={
            "candidate_model_scores": {},
            "downshift": False,
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "stale_embedding_space",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=("success", 1.0, "success", 1.0, "human", "human"),
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.010862656000000002,
        cost_known=True,
        duration=28.591259717941284,
        propensity=1.0,
        fingerprint="alibaba/qwen3.7-plus@qwen3.7-plus",
        prompt_tokens=22686,
        provenance={
            "candidate_model_scores": {},
            "downshift": False,
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=("success", 1.0, "success", 1.0, "human", "human"),
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.008034368,
        cost_known=False,
        duration=186.7180371284485,
        propensity=1.0,
        fingerprint="alibaba/qwen3.7-plus@qwen3.7-plus",
        prompt_tokens=23415,
        provenance={
            "candidate_model_scores": {},
            "downshift": False,
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=("failure", 1.0, "failure", 1.0, "human", "human"),
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.017598528000000002,
        cost_known=True,
        duration=100.70734786987305,
        propensity=1.0,
        fingerprint="alibaba/qwen3.7-plus@qwen3.7-plus",
        prompt_tokens=42015,
        provenance={
            "candidate_model_scores": {},
            "downshift": False,
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "tier_escalation_reason": None,
            "top_k_neighbor_ids": [],
        },
        outcome=("weak_success", 0.3, "failure", 1.0, "human", "human"),
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.0083168,
        cost_known=False,
        duration=4.421890497207642,
        propensity=1.0,
        fingerprint="alibaba/qwen3.7-plus@qwen3.7-plus",
        prompt_tokens=25846,
        provenance={
            "candidate_model_scores": {},
            "downshift": False,
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "rank_escalation_reason": None,
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.00077056,
        cost_known=True,
        duration=49.61015248298645,
        propensity=1.0,
        fingerprint="alibaba/qwen3.7-plus@qwen3.7-plus",
        prompt_tokens=28,
        provenance={
            "candidate_model_scores": {},
            "decision_index": 0,
            "downshift": False,
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "rank_escalation_reason": None,
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.00091008,
        cost_known=True,
        duration=47.13972210884094,
        propensity=1.0,
        fingerprint="alibaba/qwen3.7-plus@qwen3.7-plus",
        prompt_tokens=28,
        provenance={
            "candidate_model_scores": {},
            "decision_index": 1,
            "downshift": False,
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "rank_escalation_reason": None,
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.00066816,
        cost_known=True,
        duration=43.864588022232056,
        propensity=1.0,
        fingerprint="alibaba/qwen3.7-plus@qwen3.7-plus",
        prompt_tokens=28,
        provenance={
            "candidate_model_scores": {},
            "decision_index": 2,
            "downshift": False,
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "rank_escalation_reason": None,
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.00104448,
        cost_known=True,
        duration=44.31322407722473,
        propensity=1.0,
        fingerprint="alibaba/qwen3.7-plus@qwen3.7-plus",
        prompt_tokens=28,
        provenance={
            "candidate_model_scores": {},
            "decision_index": 3,
            "downshift": False,
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "rank_escalation_reason": None,
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.00043392,
        cost_known=True,
        duration=7.2048962116241455,
        propensity=1.0,
        fingerprint="alibaba/qwen3.7-plus@qwen3.7-plus",
        prompt_tokens=28,
        provenance={
            "candidate_model_scores": {},
            "decision_index": 4,
            "downshift": False,
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "rank_escalation_reason": None,
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "top_k_neighbor_ids": [],
        },
        outcome=("failure", 0.7, "failure", 0.7, "auto_tier2", "auto_tier2"),
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.00034048,
        cost_known=True,
        duration=5.073335409164429,
        propensity=1.0,
        fingerprint="alibaba/qwen3.7-plus@qwen3.7-plus",
        prompt_tokens=28,
        provenance={
            "candidate_model_scores": {},
            "decision_index": 5,
            "downshift": False,
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "rank_escalation_reason": None,
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "top_k_neighbor_ids": [],
        },
        outcome=("failure", 0.7, "failure", 0.7, "auto_tier2", "auto_tier2"),
    ),
    _LiveAtom(
        model="qwen3.7-plus",
        cost=0.00032896,
        cost_known=True,
        duration=5.227494716644287,
        propensity=1.0,
        fingerprint="alibaba/qwen3.7-plus@qwen3.7-plus",
        prompt_tokens=28,
        provenance={
            "candidate_model_scores": {},
            "decision_index": 6,
            "downshift": False,
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "rank_escalation_reason": None,
            "router_propensity": 1.0,
            "selection_rule_used": "cold_start",
            "top_k_neighbor_ids": [],
        },
        outcome=("failure", 0.7, "failure", 0.7, "auto_tier2", "auto_tier2"),
    ),
    _LiveAtom(
        model="deepseek-v4-flash",
        cost=0.0,
        cost_known=False,
        duration=28.035216331481934,
        propensity=None,
        fingerprint="deepseek-v4-flash@deepseek-v4-flash",
        prompt_tokens=101,
        provenance={
            "auto_escalated": True,
            "candidate_model_scores": {},
            "decision_index": 7,
            "downshift": False,
            "escalated_reasoning_arm": "think",
            "escalation_exploration": {
                "action": "raise_effort",
                "checkpoint_id": "demo_checkpoint::demo_task",
                "decision_index": 7,
                "epsilon": 0.0,
                "features": {
                    "countable_failures": 2.0,
                    "decision_index": 7.0,
                    "distinct_keys": 1.0,
                    "effort_headroom": 1.0,
                    "rank_headroom": 4.0,
                    "window_events": 2.0,
                },
                "policy_action": "raise_effort",
                "propensity": 1.0,
                "randomized": False,
                "seed": None,
            },
            "fallback_chain_triggered": False,
            "model_chosen": "qwen3.7-plus",
            "neighbor_confidence_scores": [],
            "new_label_window": True,
            "rank_escalation_reason": "same_verified_failure_x2",
            "router_propensity": None,
            "selection_rule_used": "auto_escalation",
            "top_k_neighbor_ids": [],
        },
        outcome=("failure", 0.7, "failure", 0.7, "auto_tier2", "auto_tier2"),
    ),
    _LiveAtom(
        model="gpt-5-mini",
        cost=0.00367,
        cost_known=True,
        duration=14.505309820175171,
        propensity=None,
        fingerprint="openai/gpt-5-mini@gpt-5-mini",
        prompt_tokens=24,
        provenance={
            "auto_escalated": True,
            "candidate_model_scores": {},
            "decision_index": 8,
            "downshift": False,
            "escalation_exploration": {
                "action": "raise_rank",
                "checkpoint_id": "demo_checkpoint::demo_task",
                "decision_index": 8,
                "epsilon": 0.0,
                "features": {
                    "countable_failures": 2.0,
                    "decision_index": 8.0,
                    "distinct_keys": 1.0,
                    "effort_headroom": 0.0,
                    "rank_headroom": 4.0,
                    "window_events": 2.0,
                },
                "policy_action": "raise_rank",
                "propensity": 1.0,
                "randomized": False,
                "seed": None,
            },
            "fallback_chain_triggered": False,
            "model_chosen": "gpt-5-mini",
            "neighbor_confidence_scores": [],
            "new_label_window": True,
            "rank_escalation_reason": "same_verified_failure_x2",
            "router_propensity": None,
            "selection_rule_used": "auto_escalation",
            "top_k_neighbor_ids": [],
        },
        outcome=("failure", 0.7, "failure", 0.7, "auto_tier2", "auto_tier2"),
    ),
    _LiveAtom(
        model="gpt-5-mini",
        cost=0.003196,
        cost_known=True,
        duration=11.958397150039673,
        propensity=None,
        fingerprint="openai/gpt-5-mini@gpt-5-mini",
        prompt_tokens=24,
        provenance={
            "auto_escalated": True,
            "candidate_model_scores": {},
            "decision_index": 9,
            "downshift": False,
            "fallback_chain_triggered": False,
            "model_chosen": "gpt-5-mini",
            "neighbor_confidence_scores": [],
            "new_label_window": False,
            "rank_escalation_reason": "escalation_floor",
            "router_propensity": None,
            "selection_rule_used": "escalation_floor",
            "top_k_neighbor_ids": [],
        },
        outcome=("failure", 0.7, "failure", 0.7, "auto_tier2", "auto_tier2"),
    ),
    _LiveAtom(
        model="gpt-5-mini",
        cost=0.003336,
        cost_known=True,
        duration=18.871729850769043,
        propensity=None,
        fingerprint="openai/gpt-5-mini@gpt-5-mini",
        prompt_tokens=24,
        provenance={
            "auto_escalated": True,
            "candidate_model_scores": {},
            "decision_index": 10,
            "downshift": False,
            "fallback_chain_triggered": False,
            "model_chosen": "gpt-5-mini",
            "neighbor_confidence_scores": [],
            "new_label_window": False,
            "rank_escalation_reason": "escalation_floor",
            "router_propensity": None,
            "selection_rule_used": "escalation_floor",
            "top_k_neighbor_ids": [],
        },
        outcome=("failure", 0.7, "failure", 0.7, "auto_tier2", "auto_tier2"),
    ),
    _LiveAtom(
        model="gpt-5-mini",
        cost=0.002458,
        cost_known=True,
        duration=12.591482400894165,
        propensity=None,
        fingerprint="openai/gpt-5-mini@gpt-5-mini",
        prompt_tokens=24,
        provenance={
            "auto_escalated": True,
            "candidate_model_scores": {},
            "decision_index": 11,
            "downshift": False,
            "fallback_chain_triggered": False,
            "model_chosen": "gpt-5-mini",
            "neighbor_confidence_scores": [],
            "new_label_window": False,
            "rank_escalation_reason": "escalation_floor",
            "router_propensity": None,
            "selection_rule_used": "escalation_floor",
            "top_k_neighbor_ids": [],
        },
        outcome=("failure", 0.7, "failure", 0.7, "auto_tier2", "auto_tier2"),
    ),
    _LiveAtom(
        model="gpt-5-mini",
        cost=0.00831625,
        cost_known=False,
        duration=32.731884479522705,
        propensity=None,
        fingerprint="openai/gpt-5-mini@gpt-5-mini",
        prompt_tokens=24233,
        provenance={
            "auto_escalated": True,
            "candidate_model_scores": {},
            "decision_index": 12,
            "downshift": False,
            "escalated_reasoning_arm": "high",
            "escalation_exploration": {
                "action": "raise_effort",
                "checkpoint_id": "demo_checkpoint::demo_task",
                "decision_index": 12,
                "epsilon": 0.0,
                "features": {
                    "countable_failures": 4.0,
                    "decision_index": 12.0,
                    "distinct_keys": 1.0,
                    "effort_headroom": 1.0,
                    "rank_headroom": 3.0,
                    "window_events": 4.0,
                },
                "policy_action": "raise_effort",
                "propensity": 1.0,
                "randomized": False,
                "seed": None,
            },
            "fallback_chain_triggered": False,
            "model_chosen": "gpt-5-mini",
            "neighbor_confidence_scores": [],
            "new_label_window": True,
            "rank_escalation_reason": "same_verified_failure_x2",
            "router_propensity": None,
            "selection_rule_used": "auto_escalation",
            "top_k_neighbor_ids": [],
        },
        outcome=None,
    ),
    _LiveAtom(
        model="gpt-5-mini",
        cost=0.0084543,
        cost_known=True,
        duration=17.28285241127014,
        propensity=None,
        fingerprint="openai/gpt-5-mini@gpt-5-mini",
        prompt_tokens=525,
        provenance={
            "auto_escalated": True,
            "candidate_model_scores": {},
            "decision_index": 13,
            "downshift": False,
            "fallback_chain_triggered": False,
            "model_chosen": "gpt-5-mini",
            "neighbor_confidence_scores": [],
            "new_label_window": False,
            "rank_escalation_reason": "escalation_floor",
            "router_propensity": None,
            "selection_rule_used": "escalation_floor",
            "top_k_neighbor_ids": [],
        },
        outcome=("failure", 0.7, "failure", 0.7, "auto_tier2", "auto_tier2"),
    ),
)
