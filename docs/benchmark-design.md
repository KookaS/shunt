---
title: Benchmark design
description: Two-part benchmark structure — live model execution and offline routing strategy evaluation.
---

# Benchmark design

The benchmark has two parts. `runner/` executes models against SWE-bench Verified instances and records the outcomes. `routing/` evaluates routing strategies offline against those recorded outcomes.

| Tree | Question | Output |
|---|---|---|
| `runner/` | Which models solve which tasks? | Verified per-cell pass/fail, cost, and tokens, written to `routing/results.csv` |
| `routing/` | Which routing strategy maximizes reward? | Per-strategy metrics across a task × model matrix |

`runner/` is the empirical source; `routing/results.csv` is the committed record it produces. `routing/` consumes that file — no dependency on runner infrastructure at eval time.

## Why split them

`runner/` answers a model-selection question: given N models, which is the cheapest that solves each task? This is the discrimination test. If every model passes everything, routing is pointless.

`routing/` answers a strategy-selection question: given a known task × model matrix, which algorithm (kNN, cascade, bandit, fixed) maximizes pass rate minus cost?

They share a `benchmark/` root because both evaluate model-decision capability. They stay separate because they have different runners, metrics, and output formats.

## Structure

```
benchmark/
  README.md                               Model-capability benchmark overview
  benchmark.yaml                          Enabled models, strategies, and run settings

  challenges/
    swebench_verified/                    Instance specs (500 instances, live benchmark source)
    swebench_multimodal/                  Instance specs (102 instances, committed store, not wired to live runs)

  validate_results.py                     Pre-analysis data-integrity gate (fail closed, exits nonzero on ERROR)
  cost_reconcile.py                       Reconcile tracked real_cost vs owner-billed bill; cross-check + accounting-hole scan (exits nonzero on alarm)

  runner/                                 Live execution against the SWE-bench harness
    run_matrix.py                         Runs the (challenge x model x arm) matrix, upserts rows
                                          (validates every row at write-time — DataIntegrityError aborts)
    collect.py                            Adaptive collection (phase A + frontier tail)
    check_integrity.py                    Anchor/authenticity audit of the committed rows

  routing/                                Routing strategy evaluation
    results.csv                           THE committed source of truth (per-cell outcomes)
    validate.py                           Row-invariant validator (accounting-hole/ran-ness/schema/well-formed)
    data/                                 Curated read-only inputs
      challenges.json                     Challenge index + task metadata
      seed/                               LFS-tracked warm-start bundles (one .npz per embedder fingerprint + plain manifest.json)
    reports/                              Gitignored — derived strategy_summary.csv + plots
    strategies/
      __init__.py                         Strategy protocol
      oracle.py                           Best per-task (upper bound)
      fixed.py                            Always-cheap, always-frontier, random
      knn.py                              Embed task → retrieve neighbours → cheapest capable
      knn_cascade.py                      kNN-informed try-verify-escalate (within one task)
      knn_difficulty.py                   Judge-difficulty pick (single-shot + session cascades)
      knn_session_cascade.py              The opt-in `knn_semantic_cascade`: kNN pick + the session ladder
      session_cascade.py                  The shipped default: always-cheap pick + the session ladder
      predict_then_cascade.py             Binary gate: cheap-direct vs session-cascade ladder
      price_cascade.py                    Try-verify-escalate in ascending price order (zero-ML)
      tier_classifier.py                  Single-shot: predict crossover tier, route there directly
      _cascade_common.py                  Shared cascade utilities (internal)
      _template.py                        Skeleton for a new strategy
    run_eval.py                           Evaluate all strategies × tasks
    metrics.py                            Reward, regret, efficiency
```

## Strategy interface

```python
class Strategy(ABC):
    @property
    def name(self) -> str: ...

    def select(self, task_id: str, task_meta: dict, matrix: dict) -> str:
        """Return the model name to route this task to."""
```

The evaluator iterates tasks, calls `select()` per strategy, looks up the outcome, and accumulates metrics.

## Metrics

| Metric | Formula | Meaning |
|---|---|---|
| AvgPerf% | `pass_count / total_tasks × 100` | % of tasks solved |
| TotalCost | `sum(cost of every billed attempt)` | Raw dollar cost, cache-blind |
| TotalCost_cacheaware | `TotalCost − Σ(cost × input_share × hit_rate × discount)` over consecutive same-model attempts | Dollar cost once repeat-model caching is priced. `input_share` is the cost-weighted input share of the model's MEASURED token mix; `discount` is `1 − cache_read_price/input_price` from the registry; only `hit_rate` is assumed |
| context_cost_alpha_α | `TotalCost_cacheaware × C(α)/C(0)` where `C(α) = Σ[α·t_{i-1}·input_price_i + billed_i]` and `t = 2·in_tok/calls` | Cost model for carrying context across an escalation. The carried prefix is a cache MISS (a new model receiving a prefix it has never seen), so it is priced at full input rate. `C(0)` is `TotalCost` exactly, which is what makes the surcharge the only thing α moves. Computed on the token-complete subset (`context_cost_n`); asserts no pass rate. Published at α = 0.1, 0.3 and 1.0: the 0.1-0.3 band is `context_transfer: summary`, α = 1.0 is `context_transfer: full` (the shipped default), and α = 0 is the marker itself — a fresh context per rung, which is what the offline replay does and is deliberately not a config value |
| Reward | `1.0 × pass_rate − γ × total_cost` | Cost-aware utility |
| CumReg | `sum(oracle_reward − strategy_reward)` | Regret vs oracle — reward lost by not routing optimally |
| rAcc | fraction of tasks routed to the oracle's model | Routing accuracy |

γ defaults to 0.1, matching the `agent-as-a-router` cost-weight baseline.

**Regret**, borrowed from bandit/RL theory, is the decision-quality metric here: how much
worse off you are for not having made the best possible choice, always measured against an
optimal baseline (the Oracle, which routes with perfect hindsight). Per task it is
`oracle_reward − strategy_reward`; match the oracle's pick and it is 0. A strategy accumulates
regret two ways — **quality regret** (it failed a task the oracle solved) and **cost regret**
(it passed, but paid for a bigger model than the task needed). `CumReg` is that gap summed over
the task sequence, and `oracle_gap.png` reports the regret totals; the retired per-task line figure's slope was
average regret per task, so a lower, flatter curve means routing closer to optimal.

Cost is recorded from actual model API responses: the provider-returned cache-aware `usage.cost` when present (e.g. Requesty-routed models, including cache-aware rates), falling back to litellm's computed cost otherwise (e.g. direct routes litellm can price, such as deepseek). For offline eval, costs come from the cached `results.csv` (recorded during live benchmark matrix runs). Recording per-request API cost on the live proxy path is roadmap, not a current feature.

## Baselines

The two fixed baselines bracket the problem. Always-Frontier arrives but pays top
price on every task; Always-Cheap is cheap but grinds through the tail it cannot
solve. A router earns its place only by being cost-effective against both — which
is what the strategies below are scored on.

![The same workload under three policies. Always-frontier arrives and is the most costly, paying frontier price on every trip. Always-cheap arrives but not reliably — its hard tail stalls in traffic before reaching the goal. Shunt is cheap by default and pays frontier price only for the tail.](assets/route/versus.svg)

The figure is a conceptual framing of the three policies, not a result: cost is
shown as a price word rather than a number. The measured comparison is in
[Results](results.md).

| Strategy | Behavior |
|---|---|
| **Oracle** | Cheapest model that passes each task. Upper bound. |
| **Always-Cheap** | Always cheapest model (derived from pricing matrix). Lower bound — if a router can't beat this, it is pointless. |
| **Always-Frontier** | Always most expensive model (derived from pricing matrix). Maximum cost baseline. |
| **Random** | Random model per task (mean over N seeds). Null baseline. |

Additional strategies in `strategies/`: kNN-semantic (a control — the selection rule without
the ladder), Session-Cascade (**the shipped default**: the session-cadence escalation ladder
over an always-cheap pick), kNN-semantic-cascade (the opt-in `knn_semantic_cascade`: that same ladder over
the kNN pick), kNN-semantic-cascade (within-task), Price-Cascade (the zero-ML price-ascending
cascade — the floor a learned router has to beat), and kNN-semantic-tier. The judge-difficulty
family routes on an LLM-judge difficulty label instead of an embedding:
`knn_difficulty.py` holds kNN-difficulty (single-shot control), kNN-difficulty-cascade and
Difficulty-Band-cascade (session-cadence), reading the committed `data/judge_difficulty.json` —
all three measured, none cleared the inference bar, all benchmark-only.

**Every rung of a replayed cascade starts from a fresh tree and a fresh context.** That is
a property of this harness, not of the product: the matrix records one outcome per
(task, model) pair, run independently, so a cascade's second attempt is scored as if the
stronger model had been handed the untouched original task. A live escalation is not like
that, and the difference is stated once in
[what the escalated model is told](escalation.md#offline-vs-live-cascade).

## Equal-coverage scoring — monotone-rank imputation

The runner collects an **asymmetric** matrix on purpose: cheap and mid models run on
nearly every task, the frontier only on a subset, because running the most expensive model
everywhere is the cost the project exists to avoid. But routing strategies must be compared
on the *same* tasks, or a cheaper strategy just looks better by dodging the hard ones the
frontier column happens to cover. The default report resolves this by completing the matrix
in memory before scoring. This section states the method precisely.

### The monotonicity axiom

Order the models by a measured **capability rank** (below), weakest to strongest. The axiom:
for any task there is a single **crossover model τ** — every model at or above τ passes,
every model below τ fails. Plainly: *stronger than a success passes; weaker than a failure
fails.* This is the one assumption the completion rests on, and it is **measured, not
assumed** — see the violation rate below.

### The derived capability rank

Rather than sort models into a handful of hardcoded tiers, Shunt derives a **strict
per-model order from the verified outcomes**. The order is the **Copeland score over
co-measured tasks**: for each pair of models, over the tasks where *both* really ran, who
wins more (passes where the other fails); a model's score is how many models it dominates
minus how many dominate it. This is deliberately *not* each model's raw pass-rate — the
frontier ran only on the hard subset, so its raw rate is computed over harder tasks than a
cheap model's, and ranking on raw rates would sort the frontier *below* cheap models scored
on easy work. Each model also carries its **marginal pass-rate and a Wilson confidence
interval**, used as the reported stat and a confidence gate: a model with too few cells, too
wide an interval, or too few co-measured peers falls back to a **price-implied slot**
(cheaper assumed weaker) until it earns a measured rank. Ties break deterministically by
(pass-rate, price, name), so the order is reproducible across runs on the same data. The top
of the rank is asserted to equal the kill-gate control model; a disagreement is surfaced as
a loud finding rather than silently reordered. The imputation, the classifier strategy, and
the ladder collector all read this one derived order, and a per-model evidence artifact
(`capability_evidence.json`, regenerable) records the pass-rate, CI, rank, source, and price
behind each model's position, plus an **ordinal-band** grouping (band 1 = weakest … band N =
strongest) for the narrative. Bands have no semantic names — each is described only by its
metadata (member models, price range, pass-rate range with CI, `n_models`, and the share of
tasks it is the weakest to solve) — and the band count is data-driven: adjacent models whose
capability CIs overlap merge into one band.

### The completion rule (s\*/f\*/UNKNOWN)

Per task, from its **real** default-arm cells:

1. Read each observed model's outcome.
2. `s*` = the weakest model observed to **pass**; `f*` = the strongest model observed to
   **fail** (by capability rank).
3. Impute: model at rank `≥ s*` → pass; model at rank `≤ f*` → fail; a model strictly
   between `f*` and `s*` is left **UNKNOWN**. `τ = s*`.
4. Extend to every ranked model, so always-cheap, kNN, the cascades and always-frontier are
   all scorable on the same task set.
5. **Observed truth always wins** — a real cell is never overwritten, even when it
   contradicts the axiom.

The rule is collection-order-agnostic: it completes gap-free data (from ladder collection)
and random-order historical data alike, leaving an UNKNOWN band only where no bracketing
observation exists. Every imputed cell is tagged with the observed model that implied it;
real cells are tagged as themselves.

**Nothing imputed is ever persisted.** The completion returns an in-memory matrix;
`results.csv` is read, never written, and a test pins it byte-identical across a report run.
Imputation is recomputed fresh every time, so a stored cell can never drift from its own
definition.

### Cost imputation and its caveats

An imputed cell — pass or fail, since a failed attempt still bills — is priced from the
**per-model median measured `real_cost`** over that model's real cells, falling back to the
nearest ranked model's median (rank-neighbour) when a model was never measured. This is a point estimate **from measured
data**, never a fabricated proxy, and every such cell stays flagged so a reader can separate
measured from imputed spend. The median is taken over real **observations only**: a
**censored** cell (any resource-limit stop — `stop_reason` in `{step_limit, wall_limit,
abandoned}`; see below) or a zero-work cell (`calls == 0` and `real_cost == 0`) is a
non-observation, not a `$0` measurement, and is excluded so it cannot drag the per-model
median toward zero. Cost is the near-deterministic half of a cell (the task fixes
it), so the median is tight; the current report does not attach a confidence interval to
imputed cost, and because the gate is a **paired-quality** test, cost-point noise moves only
the cost axis of the plot, not the decision.

### The violation rate (measured ~90%)

Stronger models sometimes fail where weaker ones pass, so the axiom is not exact. When a
task has a real higher-ranked fail sitting below a real lower-ranked pass (`f* > s*`), the
region between them is left un-imputed and the task is recorded as a violation. The
**violation rate** — violating tasks over tasks with two or more observed models, with a
Wilson interval — is reported as a first-class number: on the current data monotonicity
holds about **90% of the time** (19 violations over 192 multi-observed pairs — holds
90.1%, violation rate 0.099, 95% CI [0.064, 0.149]). The report also recomputes every
conclusion with the violating tasks **excluded** (the sensitivity check) and flags any
result whose sign or CI side flips.

### Why the axiom is conservative

Completing the matrix credits the always-frontier baseline with a pass on every task a
weaker model already solved — free quality at frontier cost. If the axiom is wrong for some
task, the baseline's *true* quality is only lower than imputed, so the router's measured
advantage is only larger. A broken axiom **strengthens** the case for routing; it can never
inflate it. Imputation buys cheap exploration, not a flattering headline — and this is the
reason a sub-perfect axiom is safe to ship.

### Censored data (why a cell stopped)

A cell that stops because it hit a **resource limit** — the agent ran out of steps, hit its
graceful wall-clock ceiling, or was reaped mid-run by the hard watchdog — is **not** a
capability failure: its true pass/fail is *unknown*. Treating such a cell as a clean fail
biases both completeness (a challenge looks "all-tiers-fail / unsolvable" when its top tier
merely ran out of steps) and the kill-gate quality comparison (a censored frontier cell
understates the baseline's real quality).

Every produced row therefore records **why** it stopped in a `stop_reason` column, drawn from
a fixed vocabulary:

| `stop_reason` | Meaning | Censored? |
|---|---|---|
| `solved` | harness resolved the instance (`pass=True`) | no |
| `unsolved` | the agent genuinely finished (submitted / ran to completion) and the harness did not resolve — a real capability fail | no |
| `step_limit` | the agent hit its step (or cost) limit | **yes** |
| `wall_limit` | the agent hit its graceful `wall_time_limit_seconds` | **yes** |
| `abandoned` | the external hard watchdog reaped it mid-run | **yes** |

The signal is the scaffold's own exit status (mini-swe-agent's `LimitsExceeded` →
`step_limit`, `TimeExceeded` → `wall_limit`) plus the harness verdict; the watchdog path maps
to `abandoned`. `timeout_flag` is retained for back-compat and is now exactly
`stop_reason ∈ {wall_limit, abandoned}`. **Legacy rows** written before the column derive a
`stop_reason` on read (`solved` if pass, else `wall_limit` if `timeout_flag`, else `unsolved`).

A **censored** cell (`stop_reason ∈ {step_limit, wall_limit, abandoned}`) is treated as a
**non-observation** everywhere it would otherwise be mistaken for an observed fail:

- **Completeness / imputation.** A censored cell never establishes a crossover. A challenge
  whose crossover would depend on a censored top-tier cell is **incomplete** (unknown), not
  complete-all-fail — so it is excluded from analysis rather than counted as unsolvable. (A
  censored cell may still be *imputed* from a genuine observation on the same task under
  monotonicity — e.g. a weaker model passed, so the stronger censored cell is imputed pass.)
  This correctly **reduces** the complete-challenge count; on the current data 15 challenges
  (192 → 177 complete) revert to incomplete once censoring is respected.
- **Kill-gate / summary quality.** A censored cell is excluded from the pass-rate denominator
  (marked unscorable), so it never counts as a clean `pass=False` that understates a model's
  quality. Cost for censored cells is already excluded from the cost model (above).

### Kill-gate coverage floor (pre-registered)

**The first-milestone kill-gate verdict may only be called (PASS / FAIL / INCONCLUSIVE) when ≥90% of the
cells in BOTH arms of the comparison are MEASURED** — a real cell on the committed corpus, not an
imputed one. This floor is pre-registered here, on the record, so a verdict can never again be
quoted off a matrix whose decisive cells are mostly imputed (a prior analytical matrix was 38.5%
imputed with every imputed cell filled pass, so a published ~12% saving lived entirely in those
cells).

- **Both arms** means the router arm AND the control (fixed-frontier) arm on the same scored task
  set.
- **Measured** means the cell's `pass` and `cost` come from a real provider call recorded in
  `results.csv` with a non-`ERROR` integrity status — never a monotone-imputed fill, never a
  synthetic row.
- **Below the floor, the verdict is UNTESTED** — not PASS, not FAIL. Reports may publish measured
  numbers, but no verdict line.
- The gate currently sits below this floor: on the scored 184-task set the control (`kimi-k3`) has <!-- frozen-value: n=184, date=2026-08-11, run=6bbd898 -->
  51.6% of cells measured (95 of 184), and only 74 tasks have all six models measured. Closing the control
  gap is the bounded recollection of the step-limit-censored cells (the recollection work keyed
  on the `step_limit` staleness anchor below); the floor is what makes that spend meaningful
  rather than cosmetic.

**The verdict is a tracked, deterministic artifact.** Each run writes
`benchmark/runner/kill_gate_verdict.json` — the verdict, both cost ratios (cache-aware and
naive), the cache-aware ratio's paired-task 90% bootstrap CI (cache cost is scoped per
task, so a whole-task resample preserves within-task adjacency), `n`, the scorable subset,
and the coverage guard. It is a pure function of the
committed inputs (no timestamps, no paths), so two regenerations over the same corpus are
byte-identical and a verdict move shows up as a one-line diff on a tracked file. The
human-readable `kill_gate.log` stays gitignored (free-form, run-local); the JSON is what
version control audits.

### Collection-param anchors and the mixed-budget history

Every row in `results.csv` records the **regime it was collected under** (added when the
recollection work landed): `step_limit` and `cost_limit` (the agent-scaffold caps in force),
`scaffold_version` (the installed `mini-swe-agent`), `sampling_hash` (SHA256 of the merged
request kwargs — base scaffold kwargs + routing target + reasoning-arm params, auth secrets
excluded), and `prompt_hash` (SHA256 of the scaffold's system+instance templates). A row that
ships with an empty hash is a **legacy** row written before these columns existed; empty anchors
are never a staleness event (grandfathered — the paid cell is not recollected on an unknown).

Three of the five are **staleness anchors**: raising `live.step_limit`, upgrading the scaffold's
prompt templates, or changing the merged request kwargs marks affected cells stale so they
recompute rather than serve an outcome from a different regime. The `step_limit` anchor fires
**only for step-limit-censored cells** — a cell that hit the old cap gets the new budget; a cell
that solved or finished naturally is regime-independent and stays valid.

The corpus has a mixed budget history that the anchor makes visible. `step_limit` was 250 for
the July cells (upstream's default) until the `_DEFAULT_STEP_LIMIT = 70` changeover on
2026-07-27, then 70 for the August cells. The rows backfill this from `computed_at`: 1185 rows
record `step_limit=250`, 39 record `70`. The 92 cells with `calls > 70`, all dated
2026-07-17..2026-07-27, are the lower bound of the 250-regime collection. In 2026-08 the caps
were raised together to `step_limit=150` / `cost_limit=4.0` (from a measured hazard, not
passers' percentiles — see `benchmark.yaml`), so the 20 cells censored at 70 are now stale and
candidates for bounded recollection.

### Unsolvable tasks

A task no model solves (`s*` undefined) counts as a fail — at its own cost — for **every**
strategy, including always-frontier. That keeps the task count `N` identical across
strategies (the whole point of equalizing coverage) and leaves always-frontier quality
well-defined. A toggle can exclude such tasks instead, but universal-fail is the default
because it keeps the denominator honest.

### Ladder collection (cheap-first)

To get gap-free equal coverage without running the frontier everywhere, the `ladder`
collection mode runs each task weakest-first and escalates only until the first model passes,
then stops. Under the axiom this observes τ exactly with **zero UNKNOWN gaps** at minimum
spend — everything above the first pass is imputed-pass, everything below is real-fail — and
it never runs the frontier on a task a weaker model already solved. It reuses the same
budget-capped executor and `collect_phase` primitive as the other modes, so cache-safety and
the `--max-cost` wall are unchanged.

Escalation runs up to `--workers` **different challenges concurrently** — the same
`--workers` fan-out (and default) `cost_optimal`/`full` use, not a new knob — while each
challenge escalates cheap→strong serially, stopping at its first pass (tier N+1 runs only if
tier N failed). Challenge-atomic budgeting survives concurrency through a thread-safe
worst-case **reservation**: before dispatching a challenge the coordinator, under a shared
lock, checks committed spend + the sum of in-flight reservations + this challenge's worst-case
against `--max-cost`, and only then reserves and dispatches (releasing the reservation when it
finishes). Because a reservation is the pessimistic worst-case, N challenges in flight can
never jointly overspend, and a challenge that doesn't fit waits for an in-flight drain to free
headroom rather than being skipped — so the **set of cells collected is identical to the
serial (`--workers 1`) run**; only order and wall-clock differ. Run-level aborts (API-unusable,
the shared consecutive-/start-failure catch-all) are cross-thread: the first worker to hit one
raises the existing `RunAbortError`, which latches a shared abort flag that halts new dispatch
and is re-raised after the in-flight challenges drain. Concurrent writes to `results.csv` are
serialized by a shared write lock.

### How strategies are scored on the completed matrix

The evaluator's shape is unchanged — it still iterates tasks, calls each strategy's
`select()`, looks up the outcome, and accumulates the [metrics](#metrics) above. Because the
completed matrix fills the cells, the set of tasks a strategy can't score collapses to the
genuine UNKNOWN band (empty after ladder collection), so every strategy is scored on the
same tasks automatically. The headline stops being an over-read "X% cheaper" over mismatched
sets and becomes the **paired** router-vs-frontier quality delta on the completed matrix
with an honest CI; oracle-relative regret stays a diagnostic, reported only where full
coverage exists.

### Reporting outputs

The completion adds a few report artifacts, all regenerable from `results.csv` plus
imputation:

| Output | What it shows |
|---|---|
| Capability-distribution histogram | τ per task bucketed over {cheap, mid, high, frontier, unsolvable} — what fraction of the suite models at each rank can solve |
| Violation-rate metric | `v̂` with its Wilson CI, printed to stdout and carried in the capability-distribution figure's footer |
| Per-stratum win-rates | grouping tasks by τ, which strategy wins on reward in each stratum (where routing helps, and where it can't) |
| Coverage table | real vs imputed vs UNKNOWN cell counts per ranked model — the audit of how much rests on imputation |

The cost/quality plots carry this disclosure in their footer NOTE section — the imputed
fraction, the measured violation rate, and the conservative-assumption caveat — so no
reader mistakes a completed matrix for a fully measured one.

### Full message-list capture (per-run transcript dumps)

Every live cell also writes its **full agent message list** — the system prompt, the reasoning
text, the tool calls, the per-call `usage` envelope — to a gitignored scratch. It is produced by
setting mini-swe-agent's `output_path` to
`benchmark/runner/artifacts/message_lists/<trajectory_id>.json`; `DefaultAgent.run()` persists
it (via `self.save(self.config.output_path)` in its `finally` block) after every step, so even a
wedged or killed run leaves the trajectory it reached. The dump pairs one-to-one with the two
existing per-cell artifacts, all keyed by the same trajectory id:

| Artifact | Path | Committed? |
|---|---|---|
| Escalation trajectory (normalized action/observation trace) | `benchmark/escalation/data/live/<trajectory_id>.jsonl` | yes (LFS, scrubbed) |
| Per-step `git diff` snapshots | `benchmark/runner/artifacts/step_snapshots/<trajectory_id>/step_NNNN.diff` | no (scratch) |
| **Full message list** | `benchmark/runner/artifacts/message_lists/<trajectory_id>.json` | **no (scratch)** |

The message list is the *runnable* agent: the normalized trace is enough to replay for analysis,
not enough to reconstitute an agent. The dump is what makes a future run **resumable and
replayable**; the 822 existing trajectories predate it and cannot be retrofitted.

**Retention policy.** Dumps live in the gitignored scratch alongside the per-step snapshot
scratch, one file per cell, last-write-wins per trajectory id. Nothing deletes them
automatically; they are regenerable from the next live run of the same cell, and **deleting them
is always safe** — unlike a missing snapshot scratch, a missing dump never blocks offline replay
or stamping (the committed header's `snapshot_steps` plus the per-step scratch already cover
that); it only forfeits the raw transcript for that cell. Capture is observe-only: a dump write
failure is swallowed, so a paid run's outcome, cost and exit status never depend on it (the same
contract `_attach_snapshot_recorder` and `_capture_escalation_trajectory` hold).

**Security.** The dump is a **raw, unredacted transcript**: the message list passes through
`redact_secrets` nowhere. An agent may echo anything inside its container, so treat every dump as
untrusted output — it is kept out of git (the `benchmark/runner/artifacts/` ignore rule), never
published, and stays on the collection host. gitleaks gates commits, and the committed escalation
corpus is independently scrubbed on its own write path (`schema.dump_jsonl`); neither protects a
dump that never enters git, which is why it must not.

## Relationship to src/shunt/

The strategies in `benchmark/routing/strategies/` are evaluation copies — they consume a known matrix and compute metrics offline. They are separate from `src/shunt/router/`, the decision module that is now called on the first turn by the live proxy and learns from verified outcomes recorded at session close. The offline kNN strategy is designed to mirror that module's algorithm, so that live behavior matches what the benchmark scored.
