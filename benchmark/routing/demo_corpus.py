"""A seeded synthetic corpus that populates the inference figures — illustration, never evidence."""

# WHAT THIS IS. The seven inference figures are drawn from the owner's live outcome store, which
# on 2026-08-18 held 40 live sessions — too few for a reader to see what any panel is SHAPED
# like. This module mints a few hundred sessions that a reader can look at instead. It answers
# "what does F3 look like when it has data", and nothing else.
#
# WHAT IT IS NOT. It is not measurement, and no number it produces may be quoted, cited, or
# compared against a baseline. Three fences make a stray row greppable rather than plausible:
# every session id contains the substring `demo`, every decision provenance carries
# `synthetic_demo: true`, and the family it draws lives in its own figure half on a watermarked
# canvas that no analysis reads.
#
# THE ID FENCE IS SPLIT IN TWO, because the corpus carries both strata. A LIVE row is `demo:`-
# prefixed (`demo:0000` drawn, `demo:invented:0000` invented); a SEEDED row is `bench:demo:`-
# prefixed, and it MUST start with `bench:` or `classify_stratum` cannot adjudicate it SEEDED at
# all (`data.py:114`). The earlier fence "never `bench:`" is therefore split rather than dropped:
# BOTH halves contain `demo`, so one grep for `demo` still finds every row this module ever
# wrote, wherever it leaked to, and neither half can pass as a measured row of either stratum.
#
# WHAT IS MEASURED-SHAPED AND WHAT IS INVENTED. The 40 measured rows in `_ATOMS` shape the 300
# drawn LIVE sessions and nothing else. The regimes those 40 rows never contained — a seeded
# stratum, escalation holds, the undeliverable hold, a frontier arm, escalated successes — are
# INVENTED outright, and live in their own tuples (`_INVENTED_ATOMS`, `_SEED_ATOMS`) with their
# own comments. A panel drawn to CONTRAST two regimes teaches nothing when one of them is
# structurally empty, which is what a live-only corpus made of six of the seven figures.
#
# HOW THE MEASURED SHAPE IS PRESERVED. The 40 measured live rows below are the empirical
# distribution for the DRAWN half, resampled JOINTLY with replacement: one draw takes a whole
# row — its model, its selection rule, its cost, its `cost_known` flag, its fingerprint, its
# provenance and its label — so every marginal AND every correlation between them survives by
# construction. That is why three of the four `cost_known=0` rows still carry a NONZERO cost
# (unknown is not zero) and why only ~35% of the drawn rows carry any outcome at all. The one
# measured marginal deliberately NOT preserved is the propensity COLUMN: the shipped router
# writes 1.0 on every deterministic decision, the measured store predates that, and F5 panel C
# needs the coverage — see `_write_row`. A smooth parametric fit was rejected outright: the
# measured cost and duration distributions are sparse and heavy-tailed (durations run 3.1 s to
# 659.7 s over 40 rows), and an i.i.d. generator drawn from a fitted family would teach the
# reader a shape the router has never produced.
#
# THE ONE DELIBERATE DEPARTURE from the measured marginals is the arrival RATE: the measured
# 28-day span is kept while the drawn count is scaled 7.5x, so the windowed panels have enough
# rows to be legible. Nothing else is scaled. Timestamps are anchored to a FIXED end date AND
# the windows are evaluated against a FIXED `DEMO_NOW` rather than the wall clock, so the
# `7d`/`30d` panels are a pure function of the seed instead of thinning toward empty (and toward
# a DRIFTED figure job) as real time passes.
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
# The invented LIVE rows sit under the live prefix (they are live rows) with their own segment,
# so `SELECT ... WHERE session_id LIKE 'demo:invented:%'` separates invented from drawn without
# a second table.
DEMO_INVENTED_PREFIX: Final[str] = "demo:invented:"
# Seeded rows. `bench:` is what `classify_stratum` reads; `demo:` is what keeps them greppable.
DEMO_SEED_PREFIX: Final[str] = "bench:demo:"
DEMO_SEED: Final[int] = 20260820
DEMO_SESSIONS: Final[int] = 300
DEMO_SEEDED_SESSIONS: Final[int] = 250

# The measured span of the 40 live rows (2026-07-20T11:26:50Z .. 2026-08-17T13:30:12Z), and a
# fixed anchor so two builds at the same seed are byte-identical.
_SPAN_SECONDS: Final[float] = 2426601.942414
_ANCHOR: Final[datetime] = datetime(2026, 8, 18, tzinfo=timezone.utc)  # noqa: UP017

# The demo's frozen clock. `data._in_window` defaults to `datetime.now()`, which made every
# windowed panel a function of the wall clock: the committed PNGs would have thinned toward
# empty 7d bars and gone DRIFTED on their own. Two days past the anchor puts the newest drawn
# row inside the 7d window and all but the corpus's first 2 h 03 min inside the 30d one, so both
# windows carry a legible count. NOT the whole span: it measures 28.086 days and the 30d cutoff
# lands at `_ANCHOR - 28 d`, so the oldest ~5 live rows sit outside `30d` and `all` exceeds it.
# That is the honest shape of a trailing window and the panels report both. `render_demo_figures`
# passes this; nothing else may.
DEMO_NOW: Final[datetime] = _ANCHOR + timedelta(days=2)

# One deterministic stamp for all 250 seeded rows: a seeded stratum arrives as a single import
# burst, not as traffic, and F1 panel B exists to show exactly that vertical column. Placed
# three weeks before the anchor so the burst precedes most of the live traffic it is meant to
# have informed.
_SEED_BURST: Final[datetime] = _ANCHOR - timedelta(days=21)

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
    # The measured `sessions.selection_propensity`, kept as part of the audited transcription
    # but NO LONGER the value written: `_column_propensity` writes 1.0 on every live row so F5
    # panel C has a dot per model. Retained rather than dropped because `_ATOMS` is a field-by-
    # field record of the measured store and deleting a column from it would break that claim.
    propensity: float | None
    fingerprint: str | None
    prompt_tokens: int
    provenance: dict[str, Any]
    # (tier1 outcome, tier1 confidence, tier2 outcome, tier2 confidence, tier1 source,
    # tier2 source), or None for the ~65% of live sessions that were never labelled. Both source
    # slots carry the projection's one `outcomes.outcome_source` — see departure 3 by `_ATOMS`.
    outcome: tuple[str, float, str, float, str, str] | None


@dataclass(frozen=True)
class _DemoRow:
    """One drawn session: an atom, its own id and stamp, and its jittered continuous fields."""

    session_id: str
    timestamp: str
    atom: _LiveAtom
    cost: float
    duration: float
    # A seeded row is written the way `seed_live._write_row` writes one — a TIER-2 event only,
    # no tier-1 — because that is what the real seeder emits and `classify_stratum`'s third
    # witness reads the resulting `outcome_source`.
    seeded: bool = False


def build_demo_store(db_path: Path | str, *, seed: int = DEMO_SEED) -> Path:
    """Write the illustrative corpus to *db_path* through the production writers."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The three EXACT sqlite artefacts, never a prefix glob. `glob(path.name + "*")` was an
    # unbounded wipe of every sibling sharing the prefix — it unlinked `outcomes.db.notes.md`
    # and `outcomes.dbackup` beside an `outcomes.db`, and raised IsADirectoryError on a
    # matching directory, part-way through deleting. Today's callers pass a throwaway temp
    # dir, so it was latent; a caller that points this at a real directory is the whole risk.
    for stale in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
        if stale.is_file():
            stale.unlink()
    rng = random.Random(seed)
    # Order is fixed, and here is what that does and does NOT buy. The drawn half's SCALAR
    # columns are a function of the seed alone: `_draw_rows` consumes the rng first, so adding
    # or removing an invented regime leaves all 300 drawn rows' model, cost, duration, stamp
    # and provenance byte-identical. Their EMBEDDINGS are not — `_borrowed_vectors` runs last
    # and is sized by the TOTAL row count, so changing the invented set reshuffles the vector
    # assignment for essentially every drawn row (measured: 299 of 300) and redraws F4's
    # neighbourhood geometry. Do not read this ordering as "the drawn half is insulated".
    rows = _draw_rows(rng) + _invented_rows(rng) + _seeded_rows(rng)
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
    # reads. The SAME kernel, for the same reason, spreads the invented and seeded rows
    # (`_invented_rows`, `_seeded_rows`), where a single literal cost repeated 60 times would
    # read as structure just as an unjittered measured value would; it is applied to the same
    # two columns and to nothing else. The output of this module is fenced from measurement by
    # construction — ids that all contain `demo` and exist only inside a throwaway demo store,
    # `synthetic_demo: true` on every row, its own watermarked figure half, and no analysis,
    # report or gate that reads it — and
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


# The two verified labels an invented row can carry. Confidences match the measured atoms so a
# reader cannot tell an invented outcome from a drawn one by its confidence column — which is the
# point: the WATERMARK and the id prefix are the fences, never a tell hidden in the data.
_PASS: Final[tuple[str, float, str, float, str, str]] = (
    "success",
    1.0,
    "success",
    1.0,
    "auto_tier2",
    "auto_tier2",
)
_FAIL: Final[tuple[str, float, str, float, str, str]] = (
    "failure",
    0.7,
    "failure",
    0.7,
    "auto_tier2",
    "auto_tier2",
)


@dataclass(frozen=True)
class _Regime:
    """One invented live regime: the atom to replay, and how many rows of it to write."""

    count: int
    atom: _LiveAtom


@dataclass(frozen=True)
class _SeedArm:
    """One benchmark model's slice of the seeded stratum: how many rows, how many passed."""

    model: str
    count: int
    successes: int
    cost: float


def _live_provenance(model: str, rule: str, extra: dict[str, Any]) -> dict[str, Any]:
    """The decision provenance every invented LIVE row shares, plus its regime's own keys."""
    base: dict[str, Any] = {
        "candidate_model_scores": {},
        "fallback_chain_triggered": False,
        "model_chosen": model,
        "neighbor_confidence_scores": [],
        "router_propensity": 1.0,
        "selection_rule_used": rule,
        "top_k_neighbor_ids": [],
    }
    base.update(extra)
    return base


def _plain(
    model: str,
    cost: float,
    fingerprint: str,
    outcome: tuple[str, float, str, float, str, str] | None,
) -> _LiveAtom:
    """An ordinary served session: neighbours cleared the threshold, nothing escalated."""
    return _LiveAtom(
        model=model,
        cost=cost,
        cost_known=True,
        duration=21.4,
        propensity=None,
        fingerprint=fingerprint,
        prompt_tokens=318,
        provenance=_live_provenance(model, "cheapest_above_threshold", {"auto_escalated": False}),
        outcome=outcome,
    )


def _hold(model: str, cost: float, fingerprint: str, reason: str) -> _LiveAtom:
    """A session where escalation was CONSIDERED and withheld, carrying its hold token."""
    # No `escalation_exploration` record is attached, not even on `exploration_hold`. A genuinely
    # explored hold serialises `randomized: true` with a non-zero epsilon, and writing one here
    # would advertise a randomization the shipped router does not perform and would put a usable
    # row in front of F7's estimator. The hold TOKEN is what panel C draws; the record is not.
    return _LiveAtom(
        model=model,
        cost=cost,
        cost_known=True,
        duration=11.8,
        propensity=None,
        fingerprint=fingerprint,
        prompt_tokens=262,
        provenance=_live_provenance(
            model,
            "cheapest_above_threshold",
            {"auto_escalated": False, "escalation_hold_reason": reason},
        ),
        outcome=None,
    )


def _escalated(
    model: str, cost: float, arm: str | None, outcome: tuple[str, float, str, float, str, str]
) -> _LiveAtom:
    """An auto-escalated session on the frontier arm; *arm* set means the effort rung."""
    extra: dict[str, Any] = {
        "auto_escalated": True,
        "decision_index": 6,
        "downshift": False,
        "new_label_window": True,
        "rank_escalation_reason": "same_verified_failure_x2",
        "router_propensity": None,
    }
    if arm is not None:
        extra["escalated_reasoning_arm"] = arm
    return _LiveAtom(
        model=model,
        cost=cost,
        cost_known=True,
        duration=73.6,
        propensity=None,
        fingerprint=f"{model}@{model}",
        prompt_tokens=1462,
        provenance=_live_provenance(model, "auto_escalation", extra),
        outcome=outcome,
    )


def _undeliverable() -> _LiveAtom:
    """The one voided-exploration atom: a raise directive the engine could not deliver."""
    # Matches `data._undeliverable`'s four conditions exactly (`data.py:143-152`): action
    # `hold`, `randomized: false`, `propensity: 1.0`, and NO `escalation_hold_reason` key —
    # `engine._void_exploration` returns before the branch that writes the token, so its absence
    # IS the signature. `epsilon` stays 0.0 and `randomized` stays False, which is also what
    # keeps this row out of `ope._usable`.
    record: dict[str, Any] = {
        "action": "hold",
        "checkpoint_id": "demo_checkpoint::demo_task",
        "decision_index": 6,
        "epsilon": 0.0,
        "features": {
            "countable_failures": 2.0,
            "decision_index": 6.0,
            "distinct_keys": 1.0,
            "effort_headroom": 0.0,
            "rank_headroom": 0.0,
            "window_events": 2.0,
        },
        "policy_action": "raise_rank",
        "propensity": 1.0,
        "randomized": False,
        "seed": None,
    }
    return _LiveAtom(
        model="deepseek-v4-flash",
        cost=0.0091,
        cost_known=True,
        duration=26.9,
        propensity=None,
        fingerprint="deepseek-v4-flash@deepseek-v4-flash",
        prompt_tokens=402,
        provenance=_live_provenance(
            "deepseek-v4-flash",
            "cheapest_above_threshold",
            {"auto_escalated": False, "escalation_exploration": record},
        ),
        outcome=_FAIL,
    )


# THESE ROWS ARE INVENTED. Nothing below was measured — not the counts, not the costs, not the
# pass/fail split. They exist because six of the seven figures draw a CONTRAST, and the 40
# measured rows in `_ATOMS` contain only one side of each: no hold was ever recorded, no session
# ever escalated to a frontier model, and every escalated session failed. A panel whose second
# series is structurally zero teaches a reader the degenerate case, which is the defect this
# tuple removes. Each regime is here for one named panel:
#
#   * `_plain(...)` on `deepseek-v4-flash` and `gpt-5-mini` — F2 panel A drew three bars and
#     `deepseek-v4-flash` was invisible at one session; these also give F6 panel D a
#     NOT-escalated cohort with a real success rate to contrast against.
#   * `_hold(...)` x4 tokens at four different counts — F6 panel C was fully empty ("no live
#     hold recorded"). `disabled` is deliberately ABSENT and stays at zero: `specs.py:22-25`
#     documents it as unreachable from the serving path, so inventing it would depict something
#     the router cannot emit.
#   * `_undeliverable()` x1, exactly one — F6 panel C's derived hatched bar has never been
#     rendered. It is a DERIVED category, not a token, so one row is enough to light it.
#   * `_escalated(...)` on `kimi-k3` — F5 panel B legends a frontier share and drew no line
#     (all three demo models sit in the cheap half of `top_capability_cluster`), and F6 panel D
#     read `escalated 0/35`. The success split is deliberately WORSE than the non-escalated
#     cohort's: escalation fires on the hard tail, so a parity would be the implausible shape.
_INVENTED_ATOMS: Final[tuple[_Regime, ...]] = (
    _Regime(18, _plain("deepseek-v4-flash", 0.00942, "deepseek-v4-flash@deepseek-v4-flash", _PASS)),
    _Regime(6, _plain("deepseek-v4-flash", 0.00942, "deepseek-v4-flash@deepseek-v4-flash", _FAIL)),
    _Regime(12, _plain("deepseek-v4-flash", 0.00811, "deepseek-v4-flash@deepseek-v4-flash", None)),
    _Regime(11, _plain("gpt-5-mini", 0.03307, "openai/gpt-5-mini@gpt-5-mini", _PASS)),
    _Regime(9, _plain("gpt-5-mini", 0.03307, "openai/gpt-5-mini@gpt-5-mini", _FAIL)),
    _Regime(
        16,
        _hold(
            "deepseek-v4-flash",
            0.00889,
            "deepseek-v4-flash@deepseek-v4-flash",
            "collapse_suppressed",
        ),
    ),
    _Regime(
        11, _hold("qwen3.7-plus", 0.00061, "qwen3.7-plus@qwen3.7-plus", "no_recurring_failure")
    ),
    _Regime(6, _hold("gpt-5-mini", 0.00318, "openai/gpt-5-mini@gpt-5-mini", "escalation_ceiling")),
    _Regime(3, _hold("kimi-k2.5", 0.01204, "kimi-k2.5@kimi-k2.5", "exploration_hold")),
    _Regime(1, _undeliverable()),
    _Regime(11, _escalated("kimi-k3", 0.5487, "high", _PASS)),
    _Regime(26, _escalated("kimi-k3", 0.5487, "high", _FAIL)),
    _Regime(7, _escalated("kimi-k3", 0.4913, None, _PASS)),
    _Regime(16, _escalated("kimi-k3", 0.4913, None, _FAIL)),
)


# THE SEEDED STRATUM IS INVENTED TOO — as an ASSOCIATION, not as an arithmetic. `classify_stratum`
# only returns SEEDED for a `bench:`-prefixed row whose rule is `benchmark_seed` (`data.py:117`),
# and the demo store had none, so F1 A/B/C drew `seeded n=0`, F3 panel A could not draw the grey
# reference band that is its whole point, and F4 panel C collapsed to a single spike at 0.0
# because every live decision's seeded neighbour share was exactly zero.
#
# The per-model counts here are invented. The per-model PASS RATES and COSTS are the measured
# marginals of `benchmark/routing/results.csv` (its `model` column resolves to exactly six
# models; rates 0.414-0.845, mean `real_cost` $0.0094-$0.549), transcribed here rather than read
# at build time so the corpus stays a pure function of `DEMO_SEED`. That gives F3's reference
# band a per-model SHAPE instead of a flat line — but the rows carrying it are minted, they are
# not the benchmark, and no number they produce may be quoted.
#
# Every seeded row is labelled, which is the point of seeding: a Tier-2 verdict is what puts a
# row in the materialized `outcomes` view and therefore in the kNN index (`store.py:484-497`),
# and an unindexed seed is invisible to the live decisions F4 panel C measures.
_SEED_ATOMS: Final[tuple[_SeedArm, ...]] = (
    _SeedArm("deepseek-v4-flash", 60, 40, 0.009419),
    _SeedArm("gpt-5-mini", 55, 30, 0.033072),
    _SeedArm("kimi-k2.5", 40, 21, 0.121673),
    _SeedArm("qwen3.7-plus", 40, 17, 0.080656),
    _SeedArm("kimi-k3", 30, 25, 0.548696),
    _SeedArm("zai-glm-5.2", 25, 15, 0.347076),
)


def _invented_rows(rng: random.Random) -> list[_DemoRow]:
    """Expand every invented regime into rows, spread evenly across the measured span."""
    # SHUFFLED, then stamped by position: laid out in regime order the frontier rows would be
    # one contiguous block in time, and F5 panel B's ROLLING frontier share would read zero for
    # most of the series and then spike — a shape produced by the writer's loop order rather
    # than by anything a router did.
    atoms = [regime.atom for regime in _INVENTED_ATOMS for _ in range(regime.count)]
    rng.shuffle(atoms)
    start = _ANCHOR - timedelta(seconds=_SPAN_SECONDS)
    step = _SPAN_SECONDS / (len(atoms) + 1)
    return [
        _DemoRow(
            session_id=f"{DEMO_INVENTED_PREFIX}{index:04d}",
            timestamp=(start + timedelta(seconds=step * (index + 1))).isoformat(),
            atom=atom,
            cost=_jitter(rng, atom.cost),
            duration=_jitter(rng, atom.duration),
        )
        for index, atom in enumerate(atoms)
    ]


def _seed_provenance(model: str) -> dict[str, Any]:
    """The seeder's own provenance shape (`seed_live._decision_provenance`), replayed."""
    return {
        "model_chosen": model,
        "selection_rule_used": "benchmark_seed",
        "fallback_chain_triggered": False,
        "router_propensity": None,
        "auto_escalated": False,
        "seed_content_hash": "synthetic-demo",
    }


def _seeded_rows(rng: random.Random) -> list[_DemoRow]:
    """The seeded stratum: one import burst at one stamp, split across the benchmark models."""
    # ONE timestamp for all of them, deliberately. A seeded stratum is imported, not served, so
    # its arrival plot is a single vertical column — the shape F1 panel B exists to expose, and
    # the one `docs/inference-demo.md` already describes. Successes are assigned by POSITION
    # rather than sampled, so each arm's rate is exactly the transcribed one and no draw is
    # spent on a label.
    stamp = _SEED_BURST.isoformat()
    rows: list[_DemoRow] = []
    for arm in _SEED_ATOMS:
        for member in range(arm.count):
            index = len(rows)
            rows.append(
                _DemoRow(
                    session_id=f"{DEMO_SEED_PREFIX}{index:04d}",
                    timestamp=stamp,
                    atom=_seed_atom(arm, member),
                    cost=_jitter(rng, arm.cost),
                    duration=0.0,
                    seeded=True,
                )
            )
    return rows


def _seed_atom(arm: _SeedArm, member: int) -> _LiveAtom:
    """One seeded session: a verified benchmark cell replayed as a store row."""
    outcome = ("success", 1.0, "success", 1.0, "benchmark_seed", "benchmark_seed")
    failed = ("failure", 1.0, "failure", 1.0, "benchmark_seed", "benchmark_seed")
    return _LiveAtom(
        model=arm.model,
        cost=arm.cost,
        cost_known=True,
        # 0.0 like `seed_live._write_row`: a replayed benchmark cell has no serving duration.
        duration=0.0,
        propensity=None,
        fingerprint=None,
        prompt_tokens=0,
        provenance=_seed_provenance(arm.model),
        outcome=outcome if member < arm.successes else failed,
    )


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
        # honest NOT_IDENTIFIED refusal to a fabricated identified estimate. See
        # `_column_propensity` for which half gets which.
        provenance=SessionProvenance(
            cost_known=atom.cost_known,
            selection_propensity=_column_propensity(row),
            model_fingerprint=atom.fingerprint,
        ),
    )
    if atom.outcome is not None:
        _append_outcome(store, row, atom.outcome)


def _column_propensity(row: _DemoRow) -> float | None:
    """The `sessions.selection_propensity` column: 1.0 on every live row, NULL on every seed."""
    # 1.0 EVERYWHERE ON THE LIVE HALF, and this is the one measured marginal the demo overrides.
    # The shipped router is deterministic, so the propensity it logs is 1.0 (`engine.py:445`);
    # the ~70% NULLs in the 40 measured rows are sessions written before the column was filled,
    # not a randomization the router has. Filling it turns F5 panel C from one bar into one dot
    # per model, and it CANNOT weaken F7: `ope._usable` keeps only rows with `0 < p < 1`
    # (`ope.py:304`), so a propensity of exactly 1.0 is excluded from every estimator either way,
    # and `_is_randomized_over_arms` still fails on the absent epsilon. Writing `0 < p < 1` here
    # is what would flip F7 from its honest refusal to a fabricated identified estimate, so it is
    # never written — not by this function and not by any atom.
    #
    # NULL on the seeded half, because `seed_live` writes `SessionProvenance(cost_known=...)` and
    # nothing else (`seed_live.py:198`); a seeded row that carried a propensity would be counted
    # as a logged live routing decision by `store.routing_ope_rows`.
    return None if row.seeded else 1.0


def _append_outcome(
    store: OutcomeStore, row: _DemoRow, outcome: tuple[str, float, str, float, str, str]
) -> None:
    """Append the atom's outcome events; the store projects the `outcomes` view from them."""
    tier1, confidence1, tier2, confidence2, source1, source2 = outcome
    events = ((1, tier1, confidence1, source1), (2, tier2, confidence2, source2))
    # A seeded row gets the TIER-2 event alone, exactly as `seed_live._write_row` does: the
    # benchmark matrix is a verified pass/fail, never a wire-level tier-1 prior.
    for tier, verdict, confidence, source in events[1:] if row.seeded else events:
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
# copy of the outcome store. Nothing here is invented; nothing here is evidence either, because a
# resampling of 40 rows measures nothing the 40 rows did not already measure.
#
# AUDITED against that store (read-only, `mode=ro&immutable=1`) field by field — model, cost,
# cost_known, duration, propensity, fingerprint, prompt_tokens, the whole decision_provenance
# dict, and the outcome projection — plus the 39 inter-arrival gaps above, which reproduce the
# measured timestamps exactly. THREE departures exist, all deliberate:
#   1. prompt text is dropped (the atoms carry none);
#   2. the one recorded `escalation_exploration.checkpoint_id` — a local test path — is replaced
#      by `demo_checkpoint::demo_task` in the three atoms that carry it;
#   3. the last two slots of `outcome` are the projection's SINGLE `outcomes.outcome_source`,
#      duplicated across both tiers, not a per-tier transcription of the `outcome_events` log.
#      The measured log holds a tier-2 event for all 14 labelled sessions and a tier-1 event for
#      exactly one of them (source `wire_tier1`); this module replays both tiers from the
#      projection instead. The field that is actually consumed is faithful: replaying these
#      atoms through `_write_row` reproduces the measured `outcomes` rows — tier-1 and tier-2
#      outcome, both confidences, and `outcome_source` — as an identical multiset, and
#      `outcome_source` is the only label-source field `shunt.inspect.inference` reads.
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
