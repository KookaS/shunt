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
    reports/                              Derived intermediates + plots — gitignored, except the tracked strategy_summary.csv
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

`results.csv` is the single committed source of truth, and everything under `reports/` is a
rebuildable intermediate that regenerates from it — with one deliberate exception.
`strategy_summary.csv` is tracked: it is the derived table the cost/quality figures are drawn
from and scored against, so a committed PNG can be checked against its numbers after the fact,
and the figure freshness digest names it as an input, which a fresh clone cannot verify on a
file it does not have.

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

### The kill-gate criterion is multi-dimensional, against two baselines (pre-registered)

The gate used to judge on two things: paired quality non-inferiority at a 5pp margin, and an
aggregate cache-aware cost ratio below 1, against one baseline — the strongest enabled frontier
model with caching. That shape has two blind spots, and neither is fixable by moving a threshold.

**It could not see what else a user pays.** The strategy summary publishes `sessions_mean`,
`sessions_p95`, `cost_cv`, `TotalCalls` and `TotalOutTok`, and no verdict read any of them. A
cheap-first ladder buys its saving partly with extra round trips, and a gate that prices only
money records that as a clean win — while recording a router that saves round trips at equal
money as a failure.

**It had only one baseline, and not the hardest one.** Published work on agentic coding routers
has run the decisive ablation: always sending every task to one cheap strong model, with the
*same* handoff, matched a learned router at the same cost per solved task. A gate that only ever
compares against an expensive frontier model cannot express that falsifier at all.

So the criterion is now, in order:

1. **Quality is a gate, not a tradeable axis.** Non-inferiority at the same 5pp margin. A router
   inferior beyond the margin fails regardless of every other axis, and a quality *win* does not
   substitute for an operational win — the claim being tested is "the cheapest model that can do
   the job", so crediting quality would let any router pass by escalating more.
2. **Then tolerance-aware Pareto dominance over four operational axes** — cache-aware cost, mean
   sessions, p95 sessions, and the coefficient of variation of per-task cost. The router must be
   no worse on any of them and strictly better on at least one.
3. **Against both baselines independently** — the fixed-frontier-with-caching arm, and a zero-ML
   constant policy (`Price-Cascade`: cheapest-first, escalate on a verified failure, no routing
   decision anywhere in it). Clearing one and losing the other is a failure, not an average.

`TotalCalls` and `TotalOutTok` are deliberately not axes: provider-side work is already priced
into cache-aware cost, and admitting them would count the same failure twice.

**One tolerance, 5% relative, for all four axes** — not a per-axis table. A per-axis tolerance is
one free parameter per axis, and each one is a place to move a result after seeing it. The single
tolerance cuts both ways: widening it turns wins into ties as readily as losses.

**The gate stays on recorded cost.** It reads the tracked strategy summary and imports no pricing
module, so a price-sheet refresh can never flip a pre-registered verdict. Repricing belongs to
plots, never to the gate.

**Instrument validity is a precondition.** The gate emits a verdict about whether a signal
exists, so it clears a positive control and a destroyed-signal null before any of its verdicts
may be quoted: routers whose win or loss is true by construction are planted, and the assembled
gate must recover them. The control enters at the gate's *input files* — it writes a planted
strategy table and a planted coverage census per scenario and runs the same assembly the CLI runs
— so the CSV parse, both preconditions, the axis arithmetic, the serialised record and the exit
code are all inside what is scored, including the branch that produces an UNTESTED verdict.

The null destroys the signal in the *data* and re-runs the whole chain on it, rather than
permuting labels around a frozen prediction vector, which is at chance by arithmetic whatever
produced it. Three conditions must hold: the re-run scores at chance, destroying the signal
actually changes some verdicts (a frozen gate does not), and the positive score exceeds the
destroyed one by more than the band (an inverted gate and coinflip noise do not). Two blinded
mutants — a cost-only gate and a single-baseline gate — are scored on the same planted corpus and
must both do measurably worse, which is what makes the added axes and the added baseline
load-bearing rather than decorative. Run it with
`python -m benchmark.routing.gate_dimensions_control`; the gate itself is
`python -m benchmark.routing.gate_dimensions`, which writes a tracked, deterministic verdict
artifact beside the older one.

**A tracked verdict is re-derivable, not merely tracked.** Both verdict artifacts are committed
files that no job regenerates, so a hand edit would otherwise be invisible. The multi-dimensional
record is a pure function of the committed inputs, and the `SH016` gate recomputes it in full and
fails on any disagreement. The coverage precondition is likewise *derived*: the multi-dimensional
gate recomputes whether the floor is tripped from the offline verdict's own census — the task
count, the floor and the measured-cell counts — and refuses when the stored summary or the stored
flag disagrees with the numbers it claims to summarise. The offline verdict's cost ratios and
bootstrap intervals need a live matrix and are out of that gate's scope.

**What this does not fix.** The coverage and imputation limits below are upstream of the
criterion and unchanged by it; the multi-dimensional gate inherits that verdict, so a corpus
below the coverage floor is UNTESTED here too. The live outcome store measures no sessions and no
wall clock, so the online path can carry only the quality and cost criteria today — the
multi-dimensional verdict is an offline statement about a backtest until that instrumentation
exists.

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

### Optional columns and replicates

`RESULTS_FIELDS` declares nine append-only columns beyond the collection-param block, and the
schema widened backward-compatibly: the file on disk still carries only the 22 columns written
before them, because no run has yet emitted one. They fall in three classes, deliberately kept
apart because they answer to different rules — and outside the replicate key, a column absent
from a row means MISSING permanently, never zero.

| Class | Columns | Blank means |
|---|---|---|
| Replicate key | `rep` | `0` — the first observation of the cell |
| Measurement-optional | `wall_clock_s`, `ttft_s`, `latency_per_call_s`, `cached_in_tok`, `retry_count` | MISSING, forever — **never** zero |
| Provenance-optional | `provider`, `serving_mode` (`hosted`/`local`), `provider_latency_source` | MISSING; audit-only, never a staleness key |

`rep` is the one column with a defined legacy value, and that is a tautology rather than an
imputation: the row that exists *is* the first observation of its cell. Everything else is
blank on the whole legacy corpus, so a consumer that read a blank as `0` would publish "this
cell took no time" as a measured claim. Aggregation therefore goes through
`validate.require_measured`, which raises rather than averaging over a missing value; a
consumer with nothing to aggregate **omits the column and publishes its `n`**, the same shape
the context-cost bracket already uses. Run `uv run --extra benchmark python -m
benchmark.column_coverage` for the per-column and per-`(model, arm, rep)` fill report.

**What a live run collects, and what it cannot.** A completed live cell records `wall_clock_s`
(the agent loop for that invocation — inference plus in-container tool execution, and *not* the
SWE-bench grading harness, which is not model work) and `latency_per_call_s` (the mean of the
individually measured provider round trips, never `wall_clock_s / calls`, which would charge
tool execution to the model). Both are the client's own clock, so the row says so in
`provider_latency_source: client_wall_clock` — a provider-*reported* latency is a different
quantity and gets its own value rather than being pooled with this one. `serving_mode` is
written alongside, because a batch-1 request to a local `llama-server` and a request to a
batched hosted API are different physical experiments; `validate` refuses a row that carries a
timing without both labels, and the runner drops a timing it cannot label rather than writing
an unattributable number.

`ttft_s` is **never** written. The scaffold calls `litellm.completion` without `stream=True`,
so the first and last token of a response arrive in one event and time-to-first-token is not
observable at that seam at all; recording time-to-full-response under the TTFT name would
publish a different quantity. Obtaining it needs a streaming scaffold. Two further cases stay
blank on purpose: an errored or abandoned cell (its wall clock is the watchdog ceiling, a
property of the limit rather than of the model), and a *resumed* cell's `wall_clock_s` (the
earlier process's seconds were never recorded, and its per-call latencies are unaffected).
The 1265 legacy rows stay blank permanently, which is the correct state — nothing backfills a
measurement that was never taken.

A **replicate** is a second independent observation of an unchanged cell. `rep 0` is
canonical and the scoring path never sees a replicate: `config.load_results` reduces each
cell to its rep-0 row, exposing `n_reps` and `rep_pass_rate` as audit-only keys no metric
reads. Averaging reps inside a cell is refused for two structural reasons — `pass` is a
validated boolean invariant (`pass ⇔ stop_reason == solved`), and the bootstrap's resampling
unit is the *task*, so averaging would silently turn every published CI into a task-rep
bootstrap. Consumers that read the raw CSV state a policy explicitly, under one rule: **spend
questions sum every rep; measurement questions take rep 0** (`integrity.all_rows` versus
`integrity.rep_zero_rows`).

Depth is configured per model in `benchmark.yaml` under `replicates:` and is **disabled by
default as a spend safety** — `R > 1` multiplies live collection cost linearly. It is read at
exactly two sites (cell classification and the coverage report); the scoring path never reads
it. A re-run of an unchanged cell must be written with `merge_rows(..., mode="replicate")`:
the default `supersede` mode refuses (`REPLICATE_MISKEYED`) when a row's staleness anchors are
unchanged, because intent cannot be inferred from content — a re-run always differs on
`computed_at` and `real_cost`, so only the caller knows whether it is a correction or a second
observation.

### Temporal drift — what a July cell and an August cell do not share

Nothing in this benchmark is static. Provider prices change, inference stacks get optimised,
and a hosted model can be continuously trained under an unchanged name. A cell measured in
July and one measured in August sit in the same file, on the same axis, as if one world
produced both. Four kinds of drift follow from that, and only the first is fixed.

| Drift | Status | Policy |
|---|---|---|
| Provider price | **Handled** — see below | Naive cost axes are repriced from a dated sheet; cache-aware axes stay historical |
| Latency / throughput | Unfixable without recompute | Never pool latency across time; keep the LATEST value per `(model, provider, serving_mode)`; every latency figure states its window |
| Model identity under a fixed name | Detectable, not detected | `model_version` is a staleness anchor but only moves if the provider moves it; `model_fingerprint` is the hook for a future probe |
| Quantisation / serving stack | Handled by naming | Every local quantised rung has its own registry id, so a Q8 local model never joins its hosted namesake |
| Repricing rewrites past economics | Accepted | Confined to plots, stamped with a sheet version, excluded from the gate |

#### The price sheet, and the two costs that are not the same question

Every row already carries two costs, and they answer different questions.

| Column | What it is | Mutable? |
|---|---|---|
| `real_cost` | what the provider billed, cache included | **No** — a historical fact, the audit record |
| `estimated_cost` | the registry's list price times the row's tokens | No — the registry's `pricing` block means *the price in force when the run happened*, which is what its `price_as_of` records |

Neither answers the question a router's cost axis is asked: *what would this cost a user who
shops around today.* `benchmark/routing/data/price_sheet.json` answers it. It is a dated,
committed artifact carrying, per model and per channel, the input and output price per 1M
tokens, the `source` URL it was fetched from, its `as_of` date, and the resolved
cheapest-available choice **with the winning provider named**. It is kept strictly separate
from the registry's `pricing` block, whose meaning does not change.

Refresh it with:

```bash
uv run --extra benchmark python -m benchmark.routing.scripts.refresh_price_sheet
```

The script fetches OpenRouter, Requesty and HuggingFace Inference Providers — the three
channels that publish machine-readable prices without a key. Which listings count as the same
product is *not* the script's judgement: it is the hand-authored map in
`benchmark/routing/data/price_channels.yaml`, because every catalogue carries near-namesakes
that are different models (`...-vision-exp`, `...-pro`, `...-fast`, a dated `-0731` snapshot),
and a substring match would silently join a model to something it is not.

**Canonical price = cheapest available today**, ranked by `input + output` per 1M with ties
broken on the lower input price then lexicographically — the same total-price order the
shipped ladder already ranks by. That answers "what would this cost a user who shops around",
which is the decision-relevant number for a router. It has one accepted property: **the
winning provider can change between refreshes, so a model's plotted cost can move without the
model changing.** The sheet records every quote behind the winner, so the move is always
attributable to a named listing rather than to nothing.

Two bounds on "cheapest", both deliberate. Free tiers, batch endpoints, SLA tiers (`:flex`,
`:priority`) and region pins are excluded as different products — a `$0` tier would dominate
every cost axis it touched. And **no native provider publishes a machine-readable price
endpoint**, so a native list price that undercuts every aggregator is invisible here; the
sheet admits only prices a script fetched.

A model no channel prices has **no repriced cost**. That is MISSING, not `$0` and not
interpolated — the same read-side rule the optional columns live under, enforced by the same
`validate.require_measured`.

#### What repricing may and may not touch

- **Naive only.** A repriced cost is `in_tok * input + out_tok * output`. The **cache-aware
  axis keeps historical `real_cost`**, labelled as priced at run time. Legacy rows carry no
  `cached_in_tok`, so a cache-aware repricing would not merely be unavailable — it would be
  invented. This maps onto the existing `TotalCost` / `TotalCost_cacheaware` split, so no
  single figure mixes the two.
- **Plots only. The kill gate stays on recorded cost.** A verdict must remain a statement
  about an experiment that really happened; a price refresh must never silently flip a
  pre-registered result. Nothing in `benchmark/routing/repricing.py` is reachable from
  `strategy_summary.csv`, from `benchmark/runner/kill_gate.py`, or from
  `benchmark/routing/online_kill_gate.py`.
- **All or nothing, per figure.** `live_gap.png` is the half's naive-cost axis and the figure
  that gets repriced. Every plotted row is repriced or none is: one row at August prices
  beside one at July prices is a comparison of two price sheets wearing the label of a
  comparison of two strategies. A row is unrepriceable when the sheet does not price a model
  on its billed path, or when any billed cell carries no tokens — an **imputed** cell carries
  none, and repricing that as `$0.00` would publish "this projected rung is free". When any
  row falls out, the whole panel falls back to recorded cost and its subtitle says so.

**On today's corpus the fallback is what fires, and that is the honest state.** 406 of the
1104 completed cells carry no `in_tok`/`out_tok` at all — a pre-existing gap in the recorded
corpus, concentrated in four of the six models (`qwen3.7-plus` and `zai-glm-5.2` 113/184 each,
`kimi-k3` 89, `kimi-k2.5` 81) and the same gap that leaves `context_cost_n` at 170 of 184
scored tasks. Every strategy's billed path touches at least one of them, so no strategy has a
repriced total over the same task set as its recorded one, and `live_gap.png` renders at
recorded cost with `cost as billed when each run happened` in its subtitle and no
`price_sheet` stamp in its manifest row. The mechanism is live and tested; it starts drawing
the moment the token gap closes. The alternative — repricing only the token-complete subset
and plotting it beside totals over the full set — was rejected: that compares two different
denominators under one axis label.

#### Provenance, or this is untraceable

The sheet's `as_of` and content digest are stamped into each repriced figure's
`figures.json` row under `price_sheet`, so a figure states which prices drew it instead of
leaving the reader to assume they are current. The sheet is also a declared **figure input**
(`benchmark/pipeline.py`), so a refresh re-digests it and correctly marks every routing figure
STALE — a figure drawn at last month's prices loses its certificate rather than keeping it.
Any repriced number quoted in prose carries an SH012 provenance marker like any other
generated number.

#### The three drifts we are not fixing

**Latency and throughput.** A provider's serving stack is optimised continuously, so a
latency measured in July is not comparable to one measured in August, and there is no fix
short of re-running the cell. The policy is therefore not to pool: keep the **latest** value
per `(model, provider, serving_mode)` and have every latency figure state its measurement
window. `computed_at` already dates each row, and the declared `provider` / `serving_mode` columns are
where the stack gets recorded once a run writes them.

**Model identity under a fixed name.** Open weights are immutable; the serving stack in front
of them is not, and a hosted provider may continuously train under an unchanged id.
`model_version` is a staleness anchor, but it only helps if the provider bumps it — which is
exactly the case where the drift was never silent. The live DB carries a `model_fingerprint`
field; that is the hook a future behavioural drift probe would hang on. **No such probe
exists, and none is planned here.**

**Repricing rewrites past economics.** A figure's cost axis may move with no new measurement.
That is accepted, and it is only acceptable because the move is confined to plots, stamped
with a dated sheet version, marked STALE rather than silently redrawn, and excluded from every
gate.

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
