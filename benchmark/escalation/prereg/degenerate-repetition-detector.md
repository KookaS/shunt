# Pre-registration — degenerate-repetition-loop detector

> **Status:** written and frozen BEFORE any scoring arithmetic ran.
> **Version:** 1 · **Timestamp (UTC):** 2026-09-12T21:54:11Z
> **Scope:** offline only, zero API spend, local CPU. The detector it defines is the
> one implemented at `src/shunt/proxy/repetition.py`.

## 1. Question and disposition

A mid-session abandon policy is profitable at shallow abandon depth **only if** the
detector that fires it reaches that depth with high precision. The incumbent
recurrence detector keys on a repeated *verified failure* (same failing-check id) and
therefore cannot fire until a run has failed the same check several times — its
shallowest fire sits near depth 0.11 and few of its fires land at depth ≤ 0.20, where
the remaining cheap cost is still worth saving.

This pre-registration tests one alternative trigger: a **degenerate-repetition loop**,
defined as N consecutive identical model actions. It is computable online from data
the proxy already receives on every turn (the assistant message content and the tool
call arguments), needs no verified outcome, and can in principle fire very early.

**Reach target (pre-registered).** The target is MET iff there exists an N in the
grid of §4 such that, on the population of §3:

- `precision(d ≤ 0.20)` = P(run failed | fired, fire depth ≤ 0.20) **≥ 0.63**, and
- `volume(d ≤ 0.20)` = fraction of the population firing with depth ≤ 0.20 **≥ 0.05**.

A miss is a complete, reportable outcome: report the closest N and the gap on each
axis. **No post-hoc threshold tuning:** the N grid, the 0.20 depth cutoff, the 0.63
precision bar and the 0.05 volume floor are fixed here. Any different N or cutoff
tried after seeing results is labelled a post-hoc deviation and does not count toward
the target.

## 2. Detector definition (frozen)

For a trajectory whose ordered steps are `s_0 … s_{n-1}`, define the action key

```
k_i = sha256( content_i || 0x1f || arguments_i )
content_i    = the model-authored action content  (offline field: step.action)
arguments_i  = the tool call arguments            (offline field: step.args, "" if null)
```

On the wire these are exactly the assistant message `content` and the tool call
`arguments` that the proxy already receives; the committed mini-swe-agent corpus
records them as `action` and `args` (for its bash dialect the two are the same string).

- **Continuous score** `R(traj)` = the length of the longest unbroken run of equal
  keys in `k_0 … k_{n-1}` (≥ 1; a run of one distinct key per step scores 1).
- **Thresholded fire at N (N ≥ 2)** = the FIRST index `i` at which the trailing run of
  equal keys reaches length N, i.e. `k_{i-N+1} = … = k_i` and `i = N-1` or
  `k_{i-N} ≠ k_i`. A trajectory that never reaches N does not fire.
- **Fire depth** `d = (i + 1) / n_steps`, matching the offline evaluation convention
  already used by the abandon-depth analysis (1-based position over run length).
- **Predicted label:** fired ⇒ terminal failure (`not header.terminal_resolved`).

The rule is *consecutive*: a break (a different key) resets the run, so N−1 identical
actions followed by a different one does not fire at N.

## 3. Population

Primary: every **stamped `deepseek-v4-flash`** trajectory under
`benchmark/escalation/data/live/` (the cheap-rung population the abandon economics are
priced on; `features.is_stamped`). The all-model stamped population is reported
secondarily and labelled, never substituted.

## 4. Pre-registered N grid

`N ∈ {2, 3, 4, 5, 6, 8, 10, 15, 20}`. All are reported; the target is met if ANY does.
No value is chosen because it scored well; the grid spans short local loops through
longer stuck runs.

## 5. Metrics

Computed with the shipped escalation metrics (no local re-derivation):

1. **AUROC** of the continuous score `R` against terminal failure
   (`benchmark.escalation.metrics.auroc`).
2. **Length-stratified shuffled-label null band**: failure labels permuted within
   equal-count run-length bins of `n_steps` (10 bins), 2000 permutations; observed
   AUROC placed against the 2.5–97.5% band. This is the same defence the incumbent
   detector is judged on, because loop length correlates with failure on its own.
3. **P(fail | fired)** and the population base failure rate.
4. **Fire-depth distribution**: min / median / max and the share of fires at depth
   ≤ 0.20.
5. **Precision at depth bands** (0.0–0.1, 0.1–0.2, 0.2–0.3, …, 0.8–1.0), and
   specifically `precision(d ≤ 0.20)`.
6. **Volume at depth ≤ 0.20** = fires at depth ≤ 0.20 / population size.

## 6. Instrument-validity controls (mandatory before the verdict)

Both controls run the **same assembled detector** and are adjudicated by the shared
admissibility gate (imported read-only), with `chance_level = 0.5`:

- **Positive control:** a synthetic corpus in which failed runs carry a planted
  unbroken run of identical actions of length ≥ 6 and resolved runs carry no run
  longer than 2. The assembled detector must recover it: its AUROC must clear the
  empirical chance band (the shuffled-null 97.5th percentile minus 0.5).
- **Shuffled-label null:** the same planted trajectories with terminal labels permuted
  within challenge; the detector's AUROC must collapse inside the band.

An evaluation that fails the positive control is a **coverage gap**, not a negative:
no verdict may be written on it.

## 7. What is reported

The per-N operating table, the continuous-score AUROC with its length-stratified null
band, the two control outcomes and the gate verdict, and a single explicit
**target MET / NOT MET** with the gap. All numbers are produced by the probe over the
committed corpus; no number is typed by hand.

## 8. Result and coverage status (appended after scoring; method above unedited)

**Coverage-limited negative: the corpus cannot exercise the trigger; do not fund C3
pending a corpus that contains the phenomenon.** On the stamped `deepseek-v4-flash`
population (n=252) the persisted operating-point census (`repetition_census`) is a
longest-run histogram of `{1: 248, 2: 4}` with `max_longest_run = 2`: **no trajectory
reaches N ≥ 3**, so every N in the pre-registered grid from 3 to 20 has zero fires by
construction and the grid is untestable on this corpus. The one testable cell, N=2,
reaches 0.4% shallow volume (floor 0.05) at shallow precision 0.00; its continuous AUROC
is 0.4886 against a length-stratified null band of [0.4886, 0.5166]. The N≥3 cells are a
coverage gap, not a falsification of the trigger. Instrument validity clears the shared
gate: positive control AUROC 1.000, shuffled-label null 0.551 → ADMISSIBLE. Reopen only
with a corpus in which runs of ≥ 3 identical `(action, args)` pairs actually occur.
