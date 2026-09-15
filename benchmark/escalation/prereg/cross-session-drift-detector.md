# Pre-registration — cross-session drift detector

> **Status:** written and frozen BEFORE any scoring arithmetic ran.
> **Version:** 1 · **Timestamp (UTC):** 2026-09-12T22:01:26Z
> **Scope:** offline only, zero API spend, local CPU. The detector it defines is the
> one implemented at `src/shunt/proxy/session_drift.py`; the offline scorer at
> `benchmark/escalation/cross_session_drift.py`.

## 1. Question and disposition

The shipped escalation rule decides **once per session**, at the boundary, from a
verified outcome. A **cross-session drift** signal is the natural next input: if the last
few sessions on the **same repo** show rising carry-cost counters (reverts, retries,
loops, structured tool errors), the *next* session on that repo may be more likely to
fail — and a session-boundary decision is exactly where cache-safe routing is allowed to
act.

This pre-registration tests one narrow question: **do session-close summaries of four
whitelisted behavioural counters, aggregated over the previous K sessions on the same
repo, separate the next session's terminal failure from its resolution — above a
session-count-stratified shuffled-label null?** It is behavioural only: no model text,
no free-text field, no per-step outcome, enters the counter or the window.

**Reach target (pre-registered).** The target is MET iff there exists a K in the grid of
§4 such that, on the scored population of §3, the continuous drift score of §2:

- has an `AUROC ≥ 0.60` against next-session terminal failure, **and**
- clears the session-count-stratified shuffled-label null (`observed > null 97.5th pct`,
  §5.2), **and**
- at the break-even flag rate (the population base failure rate) fires on `≥ 5%` of scored
  sessions with `precision ≥ 0.50`.

A miss is a complete, reportable outcome: report the closest K and the gap on each axis.
**No post-hoc threshold tuning.** The K grid, the 0.60 bar, the 0.50 precision floor, the
0.05 volume floor, the null construction and the seeds below are fixed here. Any different
value tried after seeing results is labelled a post-hoc deviation and does not count
toward the target.

**Scope accounting, declared up front.** This detector's scope names four whitelisted counters:
`is_revert`, `retry_count`, `loop_signal` and the live `WIRE_TOOL_ERROR_COUNT`. Three are
schema fields with **documented zero variance** on the committed corpus and one is a
live-only `session.metadata` key with **no committed-corpus source** (`§3b`). The primary
measurement is therefore expected to be a **coverage gap**, not a falsification. To make
that expectation checkable rather than asserted, the same assembled pipeline is also run
on an explicitly-labelled **coverage-companion** counter (§3c) built only from a real
corpus field, and the instrument is proven on a planted positive control (§6) before any
verdict is written.

## 2. Detector definition (frozen)

### 2a. Session-close summary

For one closed session, aggregate the four whitelisted counters over its ordered steps
(behavioural fields only — no `action`, `args`, `observation`, `result`, `metadata` text):

| counter | source field (offline / live) | session aggregate |
|---|---|---|
| `is_revert` | `StepView.is_revert` / `StepRecord.is_revert` | count of steps that are `True` |
| `retry_count` | `StepView.retry_count` / `StepRecord.retry_count` | sum over steps |
| `loop_signal` | `StepView.loop_signal` / `StepRecord.loop_signal` | count of steps that are `True` |
| `wire_tool_error_count` | live `Session.metadata[WIRE_TOOL_ERROR_COUNT]` | the session's peak value |

The first three are integers bounded by `n_steps`; the fourth is the proxy's monotone
structured `tool_result.is_error` peak (`src/shunt/proxy/wire_signals.py`). The summary is
produced **at the session boundary**, from a finalized session, never mid-turn.

### 2b. Window and drift features

Fix a K in the grid of §4. For a repo, order its sessions deterministically and, for each
session that has at least K prior same-repo sessions, take the **K immediately preceding**
same-repo summaries `w_1 … w_K` (the rolling window; stride 1). Define the per-counter
rate of a session as `count / max(n_steps, 1)`, then:

```
mean_rate(c)  = (1/K) · Σ_j rate_j(c)                 for c in the four counters
drift_level   = Σ_c mean_rate(c)                      (the PRIMARY continuous score)
drift_slope   = Σ_c ( rate_K(c) − rate_1(c) )         (SECONDARY continuous score)
```

Higher = more drift. `drift_level` is monotone in every counter and is zero when the
window's counters are all zero. A window with fewer than K prior same-repo sessions is not
scored.

### 2c. Predicted label and firing

The scored unit is the **next** session. Its label is terminal failure
(`not header.terminal_resolved`; offline). The detector predicts failure when the score is
high. At the break-even flag rate the operating cut is
`score >= sorted(scores, reverse=True)[budget-1]` with
`budget = max(1, round(base_failure_rate · n))`, the same convention the escalation
harness uses; ties are flagged whole and the realised volume is reported.

## 3. Population

### 3a. Primary

Every **stamped** trajectory under `benchmark/escalation/data/live/` (`features.is_stamped`),
scored on the `drift_level` of §2b. Repo key = the `instance_id` prefix before the first
`__` (e.g. `astropy__astropy-12907 → astropy`). Sessions are ordered within a repo by
ascending `trajectory_id` — the schema carries **no wall-clock** and the committed manifest
carries none either, so this is an ordering **proxy**, reported as a coverage gap (§3b). The
all-model and flash-only populations are reported separately and labelled; neither is
substituted for the other.

### 3b. Declared coverage gaps (must be reported with the verdict)

1. `is_revert`, `retry_count` and `loop_signal` are **constant** on every committed step
   (all `False` / `0` / `False`; data card §"Fields that are empty or constant"). Zero
   variance ⇒ the primary score is expected to carry no separation.
2. `WIRE_TOOL_ERROR_COUNT` is written only into live `Session.metadata`; no committed
   trajectory's step metadata carries it (the only key present is `tool_call_id`).
3. The schema has **no capture timestamp**; "the last K sessions" is chronological only
   under the `trajectory_id` ordering proxy, not real time. Sessions on the same repo were
   also run in parallel across models, so they are not one operator's chronology.
4. The committed corpus has **no multi-session trajectories**; a "session" here is one
   independent attempt, and windows span different instances of the same repo.

### 3c. Coverage-companion (labelled, not part of the primary target)

Because §3b predicts a dead primary, one explicitly-labelled companion is pre-registered
so the *mechanism* can still be measured on a real varying field: substitute
`wire_tool_error_count` for the companion with the count of steps whose real
`StepView.status == "error"` (the mini-swe-agent normalizer sets `status="error"` from the
tool observation's nonzero return code — a real behavioural field, not text and not the
live wire key). The companion keeps the `is_revert` / `retry_count` / `loop_signal`
counters as-is. It is reported under its own heading and **cannot** satisfy the primary
reach target.

## 4. Pre-registered K grid

`K ∈ {3, 4, 5, 6, 7, 8, 9, 10}`. All are reported; the target is met if ANY does. The grid
spans the pre-registered 3–10 range; no K is chosen because it scored well.

## 5. Metrics

Computed with the shipped escalation metrics (no local re-derivation) where one exists:

1. **AUROC** of `drift_level` (primary) and `drift_slope` (secondary) against next-session
   failure (`benchmark.escalation.metrics.auroc`), per K and pooled across K.
2. **Session-count-stratified shuffled-label null**: failure labels permuted within each
   `(repo, min(n_prior, 10))` cell, so every cell's failure rate is fixed while the
   window-count and repo composition are preserved; 2000 permutations, seed 0; the observed
   AUROC is placed against the 2.5–97.5% band. `drift_level` beats the null iff observed >
   97.5th percentile.
3. **Fire volume / precision** at the break-even flag rate, per K and pooled, plus the
   base failure rate.
4. **Companion table** of the same statistics over the §3c counter substitution.

## 6. Instrument-validity controls (mandatory before the verdict)

Both controls run the **same assembled pipeline** (`session summary → K-window →
drift_level`) and are adjudicated by the shared admissibility gate (imported
read-only), with `chance_level = 0.5`:

- **Positive control:** a planted corpus in which failed next-sessions follow a window of
  rising counters and resolved next-sessions follow flat-zero windows. The assembled
  pipeline must recover it: AUROC must clear the empirical chance band (the shuffled-null
  97.5th percentile minus 0.5).
- **Shuffled-label null:** the same planted sessions with terminal labels permuted within
  repo; the pipeline's AUROC must collapse inside the band.

An evaluation that fails the positive control is a **coverage gap**, not a negative: no
verdict may be written on it.

## 7. What is reported

The per-K operating table, the primary and companion AUROC with their stratified null
bands, the fire volume and precision, the two control outcomes and the gate verdict, and a
single explicit **target MET / NOT MET** with the gap and the §3b coverage gaps. All
numbers are produced by the probe over the committed corpus; no number is typed by hand.

## 8. Result and coverage status (appended after scoring; method above unedited)

**Coverage gap: 3/4 counters dead by construction; the one varying companion shows no
usable signal (AUROC ~0.52).** The committed-corpus census is `is_revert`, `retry_count`
and `loop_signal` non-zero on **0 of 34,237** steps and `WIRE_TOOL_ERROR_COUNT` on **0**
steps (live-only), so the primary `drift_level` is identically zero and its AUROC 0.500 is
arithmetic, not evidence. The pre-registered §3c companion (real per-step
`status == "error"`, 2,221 non-zero steps) has genuine variance and scores AUROC
0.5032–0.5282 across K ∈ {3…10}, every value inside its session-count-stratified null band
(e.g. K=8: 0.5282 vs [0.4675, 0.5375]) and below the 0.60 bar — disclosed, not suppressed.
This is a coverage gap, not a falsification: do not wire the detector and do not fund a
live follow-on until a corpus populates the whitelisted counters. Instrument validity
clears the shared gate: positive control AUROC 0.985, shuffled-label null 0.4996 →
ADMISSIBLE.
