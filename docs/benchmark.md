---
title: Benchmark
description: How Shunt benchmarks model capability and evaluates routing strategies offline against verified SWE-bench outcomes.
---

# Benchmark

Shunt's benchmark answers one question: *which routing strategy maximizes reward
(performance − λ·cost)?* It runs in two stages. A live harness executes coding
challenges against each model and records verified pass/fail outcomes. A routing
evaluator then scores strategies offline against that outcome cache — no extra API
spend.

## Configuration — which knob tunes what

The two stages have separate controls. The word "strategy" appears in **both** with
different meanings, so keep them apart (see the note below the table).

| Knob | Stage | Tunes |
|------|-------|-------|
| `models:` | collect | **Which models** are enabled for live runs. |
| `--strategy full` \| `cost_optimal` (CLI flag) | collect | **How** the live matrix is sampled — exhaustive vs adaptive frontier collection. A *collection mode*, unrelated to the `strategies:` block. |
| `arm_sampling.weights` | collect | **Reasoning-effort exploration within each model** (`nothink`/`high`/`max` …) — *not* model selection. Per-arm inclusion probabilities by cost rank; the default arm always runs. |
| `arm_sampling.default_only_models` | collect | Models pinned to their default reasoning arm (no effort sweep). |
| `collect.*` (`audit_fraction`, `noninferiority_margin`, `phase_a_mode` …) | collect | Knobs for the `cost_optimal` sampler only. |
| `sample_size`, `seed`, `n_default` | collect | **Which tasks** run and how many (nested order). |
| `strategies.enabled` | evaluate | **Which routing policies are scored offline** over the cache — `oracle`, `always_cheap`, `always_frontier`, `knn_semantic`, `knn_semantic_cascade`, `knn_semantic_cascade_withintask`, `knn_difficulty`, `knn_difficulty_cascade`, `difficulty_band_cascade`, `price_cascade`, `session_cascade`, `knn_semantic_tier`. |
| `strategies.knn_semantic.*`, `knn_semantic_cascade_withintask.*`, `knn_difficulty.*`, `difficulty_band.*` … | evaluate | Per-policy hyperparameters (`k`, `success_rate_threshold`, `max_tries`). `knn_semantic_cascade` has no block of its own — it takes the `knn_semantic` selection knobs and the `session_cascade` ladder knobs, so the four session-cadence rows (`session_cascade`, `knn_semantic_cascade`, `knn_difficulty_cascade`, `difficulty_band_cascade`) can never be scored at two different ladders. The difficulty family's judge cost is not a knob — it is the measured per-task bill from `judge_difficulty.json`. |
| `routing.control_model` | evaluate | The fixed-frontier baseline the kill-gate is measured against. |

**The two "strategy" words.** `--strategy` (a CLI flag) chooses *how live data is
collected*; `strategies:` (a config block) lists *the routing policies scored on that
data*. A cheap-first cascade is a **policy you evaluate** (`knn_semantic_cascade_withintask`), never the
way data is collected — a cascade collector would never observe the frontier on easy
tasks and would bias the baseline (see [the kill-gate and partial-coverage limits](#honest-limits)).

## Challenge source

The live benchmark uses **SWE-bench Verified** — real GitHub bug-fix tasks
with human-verified test sets (500 instances across 12 Python repositories). A second
store, **SWE-bench Multimodal** (102 instances in multiple languages), is committed
but not wired to live runs. Each task is a minimal spec under
`benchmark/challenges/<source>/{instance_id}.json` carrying the upstream
`repo`, `base_commit`, `version`, `FAIL_TO_PASS` / `PASS_TO_PASS` test sets, a
`difficulty_stratum`, an `image_ref`, and a pinned `dataset_revision`. Repo and
patch content are pulled on demand by the official harness — nothing is vendored.
The `problem_statement` handed to the agent is fetched from the dataset at run time;
it is **also stored** in each spec and mirrored in every `tasks` entry of
`routing/data/challenges.json` (swebench_verified, backfilled 2026-08-05 from the pinned
dataset revision) or `routing/data/challenges_multimodal.json` (multimodal), so
`routing_text()` embeds the issue text rather than the `description`
label ([Results](results.md#routing-results)).

The challenge suite is the full **500-instance** SWE-bench Verified set across 12
repos, spanning a spread of difficulty strata, each with a verified prebuilt
SWE-bench image. Live results cover a **nested partial subset** (set by
`sample_size`): the run order is diversity-first and nested, so raising the sample
`10 → 20 → 200 → 500` only adds tasks and reuses already-computed cells. Provenance:
[`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified),
dataset revision `c104f840`.

## Model pool

Prices below are the **Requesty router listing** rates (as of mid-July 2026), in
USD per 1M tokens; each entry carries its own `price_as_of`, `price_note`, and
cache-read/write rate in the model registry (`src/shunt/config/models.yaml`). Listed cheapest-first by total price.

| Model | Input $/1M | Output $/1M |
|-------|-----------:|------------:|
| deepseek-v4-flash | 0.14 | 0.28 |
| deepseek-v4-pro | 0.435 | 0.87 |
| qwen3.7-plus | 0.32 | 1.28 |
| gpt-5-mini | 0.25 | 2.00 |
| kimi-k2.5 | 0.60 | 3.00 |
| zai-glm-5.2 | 1.40 | 4.40 |
| kimi-k3 | 3.00 | 15.00 |

Spread: ~21x input, ~54x output between the cheapest and the frontier model.
`deepseek-v4-pro` (native api.deepseek.com pricing) is **measured and served**: 200/200
cells on the committed corpus, pinned to its default reasoning arm, and — since
2026-09-04 — a live router model as well as a benchmark one. It is the only pool member
that clears the model-triage frontier on both strata at once (routine, and net-helpful as
an escalation rung above `deepseek-v4-flash` at +15.3pp), which is why it sits directly
above the cheap base in the live ladder.
The model registry (`src/shunt/config/models.yaml`) is the single source of truth — the table above is a
snapshot of it. (claude-opus-4-6 is priced in the registry for provenance but is left out of
`benchmark/benchmark.yaml`'s `models` list — excluded from runs; the strongest enabled frontier model is the baseline.)

## Benchmark execution

### One command: the pipeline

A single run collects **both** routing cells (`results.csv`) and escalation
trajectories, and each needs different downstream processing. `make benchmark`
(`python -m benchmark.pipeline`) is the **primary, one-command** way to run the whole
lifecycle end to end:

```bash
make benchmark ARGS="--live --max-cost 2"   # collect + process everything
make benchmark ARGS="--from report"         # recompute artifacts from existing data (no spend)
```

It composes six existing stages in order and prints one consolidated summary:

1. **collect** — `run_matrix` runs the outcome matrix (honours `--strategy`, `--live`,
   `--max-cost`, `--max-cost-overshoot`, `--workers`, `--timeout`, `--step-limit`,
   `--max-start-failures`, `--max-consecutive-failures`, `--check-images`, and — for the
   `full` strategy — `--cells`, a comma-separated `cid:model:arm` list that recollects exactly
   those named cells, bypassing cache classification, e.g. a censored-cell pilot at a new cap).
   `--step-limit` (default from `benchmark.yaml` `live.step_limit`, 150) is the **primary**,
   model-speed-agnostic per-cell bound: every model gets the same number of agent steps
   regardless of inference speed (wall-clock timing unfairly penalises slow models). `--timeout`
   (default 1800s) is a **generous graceful** wall-clock backstop — it is the agent's own
   `wall_time_limit_seconds`, so hitting it (or the step limit) terminates the run *gracefully*
   with the real `usage.cost` captured. An external hard watchdog fires strictly later
   (`--timeout` + 300s) only as a last resort for a genuine single-call hang; a cell abandoned
   that way still records the partial spend it already incurred, never a fabricated `$0`.
   `--max-consecutive-failures` (default 5) aborts the whole run after that many consecutive
   cell failures of any cause; an unusable API (invalid/empty key, no balance) aborts
   immediately and is never recorded as a fake failure. `--max-start-failures` (default 5)
   aborts the run after that many consecutive container-start failures (image missing /
   registry throttled) instead of repeatedly hammering the registry; reset by any successful
   cell. A `--live` launch first runs a **preflight health check** — one minimal real
   completion against the cheapest enabled model — and refuses to start (exit 2) if the key
   is invalid, out of balance, or the provider is down, before any Docker container is
   created; a transient blip does not refuse. That probe covers exactly **one** model — the
   cheapest enabled one — and so does the serving check below; the other enabled models are not
   probed. When *that* model's `base_url` is served locally, the preflight additionally asserts
   its **resolved serving configuration**: the inference server's own command line is read and
   recorded, matched to the endpoint by the port the process itself binds and identified by its
   executable name (argv[0]'s basename — a wrapper script, a `grep`, or a path that merely
   contains `llama-server` is not the server and is never read as one), and the run is
   refused unless context shift is provably disabled and
   `n_ctx >= live.serving.prompt_budget + max_tokens`. Local means loopback, a private-range or
   link-local address, a single-label or `.local`/`.internal`/`.lan` host name, or a unix
   socket; an endpoint whose locality cannot be decided (carrier-grade NAT, a URL with no
   scheme) refuses rather than being silently skipped. A local endpoint that **no** visible
   server process binds is refused as well — notably ollama's topology, where it listens on
   11434 and spawns `llama-server` on a random high port, so its child's flags cannot be
   attributed to the endpoint; point the benchmark at the `llama-server` port instead.
   Both operands come from `benchmark.yaml` `live.serving`; the generation cap prefers whatever
   the request will actually carry (a reasoning arm's `max_tokens`, then the scaffold overlay's)
   and falls back to the declared `live.serving.max_tokens`, so the check is satisfiable without
   editing the model registry, and a missing operand refuses by naming the key to add.
   Head-eviction during generation deletes the system prompt and tool schema and still returns
   HTTP 200, so no status-based retry can see it — this check exists because that failure
   already destroyed one measurement arm here, mutilating 47% of a run's generated tokens
   behind clean `200`s ([the corruption class, and what it
   cost](benchmark-data.md#known-defect-a-local-model-can-be-silently-mutilated-behind-a-200-ok)).
   Hosted providers are unaffected — the check
   is a deliberate no-op there. The whole check lives in `benchmark/runner/serving_guard.py`,
   and it has three limits worth knowing: it probes one model per run, it runs only at
   pre-flight and never re-reads a server that restarts mid-run, and it reads the process table
   of the machine it runs on — so from inside a container a *host*-side server is invisible and
   the guard refuses loudly rather than passing. `--check-images` (off by default, NOT implied
   by `--live`) resolves image digests to detect environment drift; it makes a registry query
   per image (~sample_size tasks to Docker Hub), which rate-limits (429) on the swebench
   namespace and can defeat GHCR pre-staging.
2. **columns** — `column_coverage` rewrites the optional-column coverage report from
   `results.csv`. Read-only, seconds, no spend; it never fails on coverage (a blank optional
   column is a fact about the corpus, not a defect) — only an unreadable corpus is an error.
3. **stamp** — `offline_replay` derives real per-step outcomes for the *newly collected*
   trajectories, timeout-bounded per trajectory by `--replay-timeout` (default 3600s) and, inside
   that, per container command (30 min): a step diff can introduce an infinite loop, so an
   expired command is killed, its container reaped, and the step recorded as infra — never a red.
   Trajectories replay in parallel, `--stamp-workers` at a time (default: host cores minus two,
   capped at 6 — each replay is a container that pins roughly one core, and the cap is memory: a
   whole test file per worker OOMs before it saturates the CPU). The unit of parallelism is the
   **instance**, never the trajectory: an instance's
   trajectories always replay in one worker, so its admissibility gate is measured once and the
   rest hit the cached verdict. Every write to the shared corpus files (`manifest.json`,
   `admissibility.json`, `stamp_ledger.json`, the trajectories) is atomic and serialised, and the
   trajectory write plus the manifest rebuild are one transaction — so a parallel run leaves the
   corpus byte-identical to what a serial one produces. The stage prints a per-trajectory
   completion line, tags every replay's own output with its trajectory id, and emits a heartbeat
   (done / left / failed / rejected, plus the in-flight trajectories and their ages) so a stall is
   distinguishable from slow progress.
   A step's outcome is the **SWE-bench grade**, not the test file's exit code: the run's log
   goes through SWE-bench's own per-test log parser and only `FAIL_TO_PASS ∪ PASS_TO_PASS`
   counts, so a test the file happens to fail that the grader excluded from both lists cannot
   turn a step red. Each instance first clears a two-sided admissibility gate adjudicated the
   same way (the gold patch must classify SUCCESS and the fixless base must classify FAILURE);
   an instance that fails it has every per-step stamp **cleared** from its trajectories and the
   rejection recorded in `admissibility.json`. Where a rejection has been diagnosed as a defect of
   the replay itself rather than a property of the instance, that record also carries a
   `known_artifact` note explaining the defect and why it was left unfixed — so an exclusion is
   readable from the data, not only from the code. `admissibility.json` is therefore a record of
   the instances the replay **reached**, not a roster of the corpus: the gate runs inside the
   replay, after the per-step diffs are loaded, so an instance whose every trajectory failed that
   load has no entry at all. Count its entries against the corpus's instances before reading it
   as complete — a missing id means "never gated", which is not the same as "gated and passed".
   That verdict is cached per instance, keyed on the dataset revision, image, test command,
   selectors, the F2P/P2P lists and the replay source
   itself, so editing any of them re-measures rather than reusing. A replay that times out or
    exits non-zero **fails the stage** — it is never silently skipped, and a trajectory whose
    per-step diffs are missing from the local scratch fails loudly rather than being reported
    done (only a trajectory whose header records `snapshot_steps: 0` — it captured none, so it
    can never be replayed anywhere — is cleared instead). A trajectory whose header records
    **no** `snapshot_steps` at all (a pre-backfill corpus) also refuses to restamp: a partial
    scratch cannot be told from a complete one, so replaying what is present would mix two
    adjudicators into a file that still passes its own hash check. Backfill the counts first
    with `record_snapshot_provenance` on the collection host. Skipped entirely on a simulated
    (non-`--live`) run — there are no new live trajectories.
4. **evaluate** — `escalation.run_eval` scores the escalation detector (metrics + plots).
5. **report** — `routing.report` regenerates the routing plots plus
   `capability_evidence.json`, `coverage_table.csv`, and `strategy_summary.csv`.
6. **figures** — the standalone plots under `benchmark/routing/scripts/` that
   `report` does not draw. They are heavy (several load the real fastembed
   embedder), so they run last and only when their inputs changed.
   `--check-figures` proves the committed PNGs are not stale without regenerating
   anything.

The final **summary** block prints the routing paired kill-gate line, the escalation
status (`OK` / `OK_OFFLINE_ONLY` / `NO_SKILL`), the capability rank order with the strongest-vs-control
check, the band count, and the total real cost, followed by a per-stage ran/failed
ledger. A stage that fails never corrupts collected data or aborts the rest — the
ledger records which stages ran.

Flags: `--no-report` runs only collect; `--from {collect,columns,stamp,evaluate,report,figures}`
starts at a later stage (so a failed report never forces re-collection); `--replay-timeout`
bounds each stamp and `--stamp-workers` sets how many replay in parallel; `--restamp`
re-replays already-stamped trajectories (the full-corpus rebuild — without it the stamp stage
only picks up unstamped ones). Both modes resume from
`stamp_ledger.json`, which records the replay-source digest each trajectory was last
completed under, so an interrupted rebuild does not start over and a source change
re-queues everything. Resume is per trajectory and unaffected by the worker count: a killed
parallel run restarts only what had not finished. Every stage prints a `=== [pipeline] stage: <name> ===` banner so a
supervising monitor can tell collection from reporting.

The four modules below stay independently runnable as **advanced / debug** entrypoints
(`make benchmark-live`, `make offline-replay`, `make escalation-eval`,
`make routing-report`) — reach for them to run or inspect a single stage.

**Always run the harness through the `benchmark` extra.** Use the Make targets
(`make benchmark`, `make benchmark-live ARGS="--live --max-cost 2"`, `make
offline-replay`, `make escalation-eval`, `make routing-report`) or `uv run --extra
benchmark python -m benchmark.pipeline …`. A bare `uv run` auto-syncs the venv to the
default locked deps and **strips** the extra (`mini-swe-agent`, `swebench`,
`matplotlib`), and the live scaffold imports `minisweagent` lazily per cell — so a
stripped venv fails every cell with `No module named 'minisweagent'`, burning time and
budget for zero data.

The live harness runs each `(challenge, model, reasoning-arm)` cell as an isolated,
reproducible Docker job:

1. Resolve the challenge spec at its pinned `base_commit` and dataset revision.
2. Pull the challenge's prebuilt SWE-bench image (per-challenge, by manifest
   digest) — source mounted read-only, with a writable sandbox.
3. Run the coding agent with the target model against the task. The cell's
   reasoning arm is overlaid on the request (e.g. `reasoning_effort`,
   `thinking` on/off), so each arm bills a distinct call.
4. Run the deterministic judge (the spec's `FAIL_TO_PASS` / `PASS_TO_PASS` tests).
5. Record the verified pass/fail, real cost (from the API response), estimated
   cost (from the registry's prices × token counts), and token usage.

Which arms run is `p(arm|model)` exploration sampling — this tunes **reasoning
effort within a model**, not which model runs. A model's default arm always runs,
and each extra arm runs on a deterministic, cost-skewed fraction of challenges
(hash-thresholded on the challenge id, so a re-run selects the identical arms).
`weights` are per-arm inclusion probabilities by cost rank (`[0.6, 0.4, 0.25]` =
cheapest extra arm on 60% of challenges, next on 40%, priciest on 25%); the flatter
tail keeps the higher-effort arms — the ones the escalation effort-rung targets —
sampled often enough to evaluate. Set `arm_sampling.enabled: false` in `benchmark/benchmark.yaml` to run
default-arm-only, or list models under `arm_sampling.default_only_models` to pin
just those (e.g. the most expensive, highest-rank models) to their default arm while the
rest keep exploring.

Per-challenge images give reproducibility, isolation, and parallelization. Cells
run concurrently with `--workers N` (each worker runs one SWE-bench container, so
raise it with an eye on host memory). Cells complete challenge-at-a-time — every
model (and sampled reasoning arm) for one challenge finishes before the next
challenge starts. `--max-cost USD` stops the run once cumulative real cost crosses
a ceiling. Serial runs (`--workers 1`, the default) enforce it as a **hard per-cell
stop** — no further cell starts once the cap is crossed, so overrun is bounded to at
most one in-progress cell (its final challenge may be left partial). `--max-cost-overshoot
USD` (default `0`) relaxes that hard stop for the serial path only: it allows up to that
many extra dollars to **finish the challenge already in progress**, so a partially-collected
challenge is not discarded — no *new* challenge starts once spend crosses `--max-cost`.
Parallel runs check at challenge boundaries — a challenge's cells are already in flight
together — so they keep a prefix of **fully-covered** (comparable) challenges.
Only model API costs enter routing metrics; judging costs are excluded.

Outcomes are appended to `benchmark/routing/results.csv`. **This file
is populated by live runs** (`make benchmark-live ARGS="--live"`, i.e. `uv run
--extra benchmark python -m benchmark.runner.run_matrix --live`), which need
Docker and API keys.
Each cell is written to `results.csv` the moment it completes (an atomic
temp-file-then-`os.replace`), so a kill or crash only loses the handful of cells
still in flight — never the whole batch. A `--max-cost` stop is per-cell in serial
runs (the final challenge may be partial) and per-challenge in parallel runs.
The evaluator can backtest strategies against cached outcomes; if the cache is empty, it reports
coverage gaps rather than fabricating numbers.

Each row also records **why** the cell stopped in a `stop_reason` column
(`solved` · `unsolved` · `step_limit` · `wall_limit` · `abandoned`). The last three are
**censored** — the cell hit a resource limit, so its true pass/fail is unknown and it is
excluded from completeness crossovers and quality denominators rather than counted as a clean
fail (see the "Censored data" section in the design notes). `timeout_flag` is kept for
back-compat and now means exactly `stop_reason ∈ {wall_limit, abandoned}`; rows written before
the column derive a `stop_reason` on read.

Rows also carry nine optional columns — the replicate key `rep`, five measurement columns
(`wall_clock_s`, `ttft_s`, `latency_per_call_s`, `cached_in_tok`, `retry_count`) and three
provenance strings (`provider`, `serving_mode`, `provider_latency_source`). A blank `rep`
means 0; every other blank means **MISSING, never zero**, and aggregating one raises rather
than defaulting.

**None of the optional measurement or provenance columns carries a value today.** The
schema was widened backward-compatibly so `wall_clock_s`, `latency_per_call_s`, `provider`,
`serving_mode` and `provider_latency_source` are *collectable* by a live run — but no run
has yet written one, so on the committed corpus they are blank on every row, and every
consumer of them raises rather than defaulting. `ttft_s` is a step further out: the
scaffold does not stream, so nothing can fill it as things stand. Treat the five as a
plumbed-in capability, not as data. `uv run --extra benchmark python -m benchmark.column_coverage` reports which
columns are populated and on which cells — run it before assuming any of them is there. Replicate collection is off by default (see
`replicates:` in `benchmark/benchmark.yaml`); when on, `rep 0` stays canonical and no metric
sees a replicate. Details: the design notes' "Optional columns and replicates".

### Result integrity

CI validates every `results.csv` row (`benchmark/runner/check_integrity.py`): identity anchors (spec
content hash, model version, reasoning-arm hash, image digest) must match the current
source, and
every derivable field is recomputed and cross-checked — the `cost` column against its
derivation rule, `real_cost` against a token-based plausibility floor (a real cost far
below the estimate — an expensive run billed as ~free — fails the build; unusually high
ratios warn), the challenge/model/arm against the registry, and basic plausibility (a
resolved cell must have emitted output and not be a timeout). A
corrupted or hand-edited row fails the build. This is an *internal-consistency* check:
it catches corruption and casual fabrication, not a determined forger reproducing every
invariant — stronger provenance (signed runs) and sampled re-execution are planned.

### Data-integrity validator (write-time + pre-analysis)

`benchmark/runner/check_integrity.py` runs in CI, after data already exists. A separate validator
(`benchmark/routing/validate.py`) encodes what a *valid row* means as invariants that
fail loud at two earlier points, so invalid data is never silently recorded or analysed:

- **Accounting-hole (ERROR)** — a **paid** model (registry input/output price > 0) with
  `calls > 0` but `real_cost == 0`. This is the fingerprint of the miss that motivated
  the validator: a live run recorded `$0` frontier cells while real spend was ~$35. A
  genuinely free model (price 0) with zero cost is fine, and **censored** rows (a
  resource-limit stop whose cost may be legitimately unharvested) are exempt.
- **Ran-ness (ERROR)** — a genuine `unsolved` capability fail must have `calls > 0` (the
  agent must have actually run to fail).
- **Schema (ERROR)** — `stop_reason` in the vocabulary; `pass ⇔ stop_reason == solved`;
  `timeout_flag ⇔ stop_reason ∈ {wall_limit, abandoned}`.
- **Well-formedness (ERROR)** — `real_cost`/`in_tok`/`out_tok`/`calls` are finite and
  `≥ 0`; `computed_at` is an ISO-8601 UTC timestamp.
- **Suspicious (WARN)** — a `real_cost == 0, calls == 0` row that was neither run nor
  censored (a legacy/odd row). WARN never aborts.

**Write-time:** the runner validates every row as it is built (`run_matrix._build_row`);
any ERROR raises `DataIntegrityError` and aborts the run rather than persisting a poison
row — the $35 run would have stopped on its first frontier cell.

**Pre-analysis gate:** `uv run --extra benchmark python -m benchmark.validate_results`
scans `results.csv`, prints a report, and exits nonzero on any ERROR. The kill gate runs
it first and **refuses to run** on data with ERROR-severity violations (fail closed).

### Cost reconciliation

The write-time and pre-analysis gates above catch *poison rows*, but they can't tell
you whether the total the harness tracked matches what the provider actually billed.
Requesty exposes no balance API, so tracked cost can silently drift from the real bill
— a live run once spent ~$35 while the harness tracked $1.25. `reconcile-cost` closes
that loop: it sums the tracked `real_cost` over a window and compares it to a
ground-truth billed amount, alarms on drift, independently cross-checks each model's
tokens × posted price against its summed `usage.cost`, and flags `real_cost == 0` rows
that still made calls (accounting holes).

The billed amount is **owner-supplied** — read the total for the window off the
Requesty dashboard and pass it as `--billed`:

```bash
make reconcile-cost ARGS="--billed 34.80 --timestamp 2026-07-27T00:00:00 \
  --start 2026-07-25T00:00:00 --end 2026-07-27T00:00:00"
```

Drift beyond `--tolerance` (default 0.15), a per-model gap beyond `--gap-threshold`
(default 3×), or any accounting hole prints an alarm and **exits nonzero** — so it
gates a supervised run before you scale spend. Each reconciliation appends one row to
an append-only ledger. Omit `--billed` to skip the tracked-vs-billed leg and run only
the cross-check and hole scan.

## Re-running the benchmark without API spend

Most of what you'd want to change — a threshold, a policy, a feature — costs seconds to
re-score, because the verified outcomes are already on disk. Re-deriving those outcomes costs
the better part of a day of container time. The two get confused, so keep them apart.

### "Backtest" means three different things

| What you're doing | Needs | Cost |
|---|---|---|
| **Re-score a policy** over the stamped corpus | nothing but the corpus | seconds |
| **Re-replay trajectories** to re-derive verified outcomes (`--restamp`) | `make state-import`, Docker + the SWE-bench images, HF dataset access | hours |
| **Evaluate a new model** | live inference | real API spend — *cannot be done offline* |

Re-scoring is the loop you iterate in. `make escalation-eval` over the committed
escalation corpus (trajectory/step counts via `benchmark.escalation.corpus.census()`),
with its default 200-permutation nulls, takes **~90 s** — no containers, no
requests. [Routing evaluation](#routing-evaluation) below is the same idea for routing
strategies over `results.csv`.

**Both halves re-score for $0.** A code change — a routing strategy, the escalation regex, a
config default — never invalidates `results.csv` or the escalation corpus. Routing cell
staleness is decided by four immutable anchors (`version_hash`, `model_version`, `arm_hash`,
`image_digest`), and none of them move when code changes; the escalation stamps key on the
replay source, not on the detector scoring them. So changing the router, the escalation
policy, or `benchmark.yaml` defaults re-runs both evals free: `make routing-report` and `make
escalation-eval` are **$0 of API money**. Only collecting **new** live model outcomes spends.

What *does* go stale after a code change is the derived artifacts committed beside the
results — `escalation/reports/metrics.json`, the PNGs under `docs/assets/figures/<half>/`, and
the per-half `figures.json`. The figure targets:

| Target | Does |
|---|---|
| `make escalation-eval` | redraws the **escalation** half |
| `make routing-report` | redraws the **routing** half |
| `make benchmark-figures` | runs the pipeline's `figures` stage (`--from figures`) — redraws the standalone routing figures **and the inference half**, and is the only target that **re-records** the freshness manifest |
| `make inference-figures` | redraws the eight **inference** PNGs only (`OUT=/tmp/x` for a scratch copy). It renders but does **not** re-record the manifest — certify with `make benchmark-figures` |
| `make demo-figures` | redraws the eight **demo** PNGs (synthetic, watermarked, evidence of nothing) only (`OUT=/tmp/x` for a scratch copy). It renders but does **not** re-record the manifest — certify with `make benchmark-figures` |
| `make check-figures` | `benchmark.pipeline --check-figures` — proves every committed figure is current without regenerating anything |
| `make check-inference-figures` | the same gate narrowed to one half (`--check-figures --half inference`) |
| `make check-demo-figures` | the same gate narrowed to one half (`--check-figures --half demo`) |

`--check-figures` accepts `--half {demo,escalation,inference,routing}`, so one half's staleness
never decides another half's exit code; `--half` is a check-only flag and is a hard `parser.error`
anywhere else. `make check-figures` is the gate to run after a code change before trusting an
old PNG. The same gate also runs at **commit time** as the `SH017` pre-commit hook (whole-tree,
`always_run`), so a code or data edit that re-stales a committed figure — without the manifest
being re-recorded for it — is refused on the commit that would ship it, not surfaced later as a
red `benchmark-integrity` job. Fix a red `SH017` by re-running the certifying pipeline stage(s),
e.g. `uv run --extra benchmark python -m benchmark.pipeline --from evaluate`; never by hand-editing
`benchmark/routing/figure_inputs.json`.

Re-replaying is what `--restamp` does, and you only need it when the *instrument* changes —
the classifier, the grader, the admissibility adjudicator. Old stamps came from a different
instrument, so they aren't comparable to new ones.

Re-replaying reads the per-step code captures that the live run recorded. Those live in a
gitignored scratch. `make state-export` packs them into `benchmark/escalation/data/live/state/`
as ~1.7 MB of deterministic per-trajectory archives, and once that directory is in git a fresh
checkout restores them with `make state-import` — but **no such export is committed today**, so
`state-import` currently fails with `no committed state plane` and the captures must come from
the collecting host. Run **`make replay-inputs`**
before you start: it lists every input this checkout still lacks — the captures, the ~100 GB of
instance images, the gold patch rows fetched from the HF dataset — and exits non-zero rather than
letting a partial run produce numbers that quietly differ.

Evaluating a new model has no offline path. A model with no trajectories has no steps to
replay and nothing to re-score; its outcomes have to be collected live first. The offline
corpus lets you re-score **policies** over **existing** model runs — that is its whole scope.

### What a re-replay costs

Measured on the committed escalation corpus (trajectory/step counts via
`benchmark.escalation.corpus.census()`), from rebuild logs covering 76% of the steps. Six workers
on a 16-core, 15.9 GB host with every image already pulled.

| Unit | Median | Aggregate mean |
|---|---:|---:|
| Per step | 3.5 s | 12.7 s |
| Per trajectory (~37 steps) | 118 s | 495 s |
| Per challenge (~4.8 trajectories) | ~13 min | ~38 min |
| Whole corpus | — | **~104 worker-hours ⇒ ~18 h wall at 6 workers** |

The mean runs roughly 3–4× the median at every level, and that is the shape of the data rather
than noise. Rates vary ~30× by repository — 2.1 s/step on `astropy`, 57.7 s/step on `psf` — and
a handful of instances dominate: **under 2% of trajectories consume 18% of the total time**, all
of them from four challenges whose test suites run for over an hour. Plan with the aggregate mean;
debug with the median. (The per-challenge median comes from the challenges a partial pass
covers, and the scheduler runs the largest first, so it reads high for a typical challenge —
per step is the number that transfers.)

Two knobs move the total. `--replay-timeout` (default 3600s) decides how much of that tail
gets counted rather than abandoned — raising it to 7200s rescued most of the timeouts but
lengthened the run. `--stamp-workers` sets the parallelism; the six workers above were busy 95%
of the wall clock, so **wall time ≈ worker-hours ÷ (workers × 0.95)** on a host that is not
memory-starved. That estimate has a floor: an instance's trajectories run serially in one worker,
and the longest instance here totals ~8 h on its own, so past roughly a dozen workers the extra
parallelism buys nothing.

Read the total as a **lower bound**. Ten trajectories are recorded at their timeout cap rather
than their true duration, and the measured passes re-used admissibility verdicts from an earlier
pass — a genuinely cold run, or any run after the replay source changes, pays every instance's
two gate legs again.

Per model, if you only want to re-replay one model's runs:

| Model | Trajectories | Steps | Worker-hours |
|---|---:|---:|---:|
| deepseek-v4-flash | 268 | 11,602 | 39 |
| gpt-5-mini | 284 | 6,775 | 26 |
| kimi-k2.5 | 105 | 4,739 | 18 |
| qwen3.7-plus | 60 | 3,023 | 9 |
| kimi-k3 | 56 | 2,036 | 7 |
| zai-glm-5.2 | 26 | 1,247 | 4 |

Trajectory count is a bad proxy for cost: `gpt-5-mini` has the most trajectories and 23% of
the steps, `deepseek-v4-flash` fewer trajectories and 39%, because its runs are longer. A
single-model pass also loses the admissibility-gate amortisation — it touches nearly as many
instances but puts only one trajectory in each, so most instances pay their two gate legs
(median ~76 s, occasionally far worse on network-dependent instances) for a single trajectory
instead of spreading them over 4.8.

**These are host numbers, not portable ones.** Replay time is dominated by container test
execution, so it tracks your disk and memory pressure as much as your clock speed. Measure
your own host before planning around them.

## Routing evaluation

The routing evaluator is a backtest over the outcome cache. Install the harness
once, then run it:

```bash
pip install -e '.[dev,benchmark]'
python3 -m benchmark.routing.run_eval
```

It scores each strategy by looking up cached `(challenge × model)` cells (the
evaluator uses each model's default reasoning arm). A
strategy whose decision needs an uncached cell is flagged (it can't be
backtested) rather than silently skipped. With an empty cache the evaluator
prints *"no results yet — run the live matrix"* and exits cleanly.

Metrics per strategy:

| Metric | Meaning |
|--------|---------|
| AvgPerf% | Tasks solved correctly |
| AvgPerf_ci_lower / AvgPerf_ci_upper | 95% bootstrap CI on AvgPerf% (resample tasks, B=1000) |
| TotalCost | Total backend model cost (USD), summed raw over every billed attempt |
| TotalCost_ci_lower / TotalCost_ci_upper | 95% bootstrap CI on TotalCost (same task resample as AvgPerf%). Cost here is heavy-tailed, so the point total does not travel alone |
| TotalCost_cacheaware | TotalCost once a repeat of the same model on consecutive attempts banks its cache-read discount. Cache cost is scoped PER TASK (one task = one session, so a discount fires only on a within-task repeat), which is exactly why resampling whole tasks is safe here — each task's attempt adjacency survives — and the paired CI is emitted as TotalCost_cacheaware_ci_lower / TotalCost_cacheaware_ci_upper |
| AvgCost_ci_lower / AvgCost_ci_upper / AvgCost_cacheaware / AvgCost_cacheaware_ci_lower / AvgCost_cacheaware_ci_upper | The per-task forms of the above, including the cache-aware CI |
| Reward | `Σ(1.0 × passed − γ × cost)` per task (γ=0.1 default) |
| CumReg | `total(oracle_reward) − total(strategy_reward)` |
| CumReg_ci_lower / CumReg_ci_upper | 95% bootstrap CI on CumReg |
| rAcc | Fraction of tasks where strategy picked the same model as the oracle |
| Pareto | True if no other strategy has higher AvgPerf% AND lower **TotalCost_cacheaware** — the cost a deployment actually pays |
| Pareto_naive | The same frontier computed on raw TotalCost, published beside it so the cache assumption is auditable rather than silent |
| context_cost_alpha_01 / context_cost_alpha_03 / context_cost_alpha_10 / context_cost_n | A **cost model**, not a measurement: TotalCost_cacheaware re-priced when 10%, 30% and 100% of the context an attempt ends holding is resent to the model an escalation moves to. The 10-30% pair is the router's `context_transfer: summary`, published as a band because a summariser's compression ratio is not a constant; 100% is `context_transfer: full`, the shipped default. That prefix is a cache **miss** by construction — new model, new prefix — so it is charged at the full input rate, never at the cache-read rate. Context size is estimated as `t = 2 × in_tok / calls` (linear prefix growth). Computed on the token-complete subset — the tasks whose every billed cell carries measured tokens, since an imputed cell carries none — whose size is `context_cost_n`; what transfers to the published dollars is the dimensionless surcharge factor. It asserts no pass rate |
| subset_selected / subset_note | Whether the row was scored on a coverage-selected slice of the sample rather than the whole of it, and the measured difficulty gap between what it scored on and what it dropped. Collection is adaptive, so full coverage tracks difficulty — a subset row is not comparable to a full-sample row |
| instrument_admissible / instrument_verdict | The two-sided instrument verdict for the SHIPPED selection path, stamped on every row (see [instrument validity](#is-the-router-measuring-anything-instrument-validity)) |

### Is the router measuring anything? (instrument validity)

A permutation null answers *"could chance have produced this number?"*. It cannot answer the
question that comes first: *"is this pipeline computing anything about the task text at all?"*
A router whose front end embeds the wrong field — or embeds nothing — produces an observation
and a null that agree perfectly, and reports "no signal" forever.

```bash
python3 -m benchmark.routing.instrument_control   # exit 0 = admissible, 1 = not
```

This plants a known-learnable signal in the task text and hands it to the pipeline at the
**front**, upstream of the text selection and the embedder, so a broken front end fails it. Two
legs must both hold: the assembled pipeline recovers the planted signal well above chance, and
the same pipeline collapses back to chance once the outcomes are shuffled. The planted signal is
deliberately independent of which repository a task comes from, so a pipeline that recovers only
the repository name scores at chance and is rejected.

The transfer-curve and cross-repo figures carry the verdict in their footer and cannot be built
without it. **A "no signal" result from a pipeline that has not cleared both legs is a gap in
coverage, not a finding** — it says nothing about whether routing signal exists.

Clearing the control is a floor, not a ceiling: it shows the pipeline can carry a strong,
explicit signal end to end. It does not show the pipeline is sensitive enough to resolve a weak
one.

### How weak a signal could this suite resolve? (minimum detectable effect)

```bash
python3 -m benchmark.routing.sensitivity   # prints; writes nothing
```

This answers the question the control cannot. It re-assigns the real outcome rows to the real
tasks so that a controlled fraction of them line up with a direction in the real embedding
space, sweeps that fraction downward, and reports the smallest effect the null test still flags
at 80% power — as an interval, not a point. Re-assignment leaves every model's marginal pass
rate untouched, so the permutation null does not move and the floor is directly comparable to
the null it interprets.

The floor is reported as the AUROC a perfect reader of the planted signal would achieve at
separating "the cheapest model suffices" from "escalation is needed" — the same unit the
escalation results use. It is reported under **both** splits (the ungrouped one the figures use,
in which a held-out task's own repository siblings sit in its index, and a repo-grouped one) and
under **both** k-rules (the configured `k` and the transfer figure's selection-corrected
best-over-*k*), because those configurations do not have the same sensitivity.

**A null from a configuration whose floor sits above any plausible effect bounds your
resolution, not the idea.** See [Results](results.md#how-weak-a-routing-signal-could-this-suite-have-seen)
for what this suite's floor turned out to be.

### Every figure explains itself

A plot is a display, not a document. Every PNG under `docs/assets/figures/<half>/` carries three
things and nothing else:

| On the canvas | What it tells you |
|---------------|-------------------|
| **Title** | the figure's claim, in a sentence — not a label |
| **Subtitle** | the sample size, the units, and the operating point the numbers are measured at |
| **Caveat** | in red, and only when the figure would otherwise be actively misread: the one thing that would change your conclusion |

Everything else a reader needs — how to read the axes, what to look for, what the jargon
means, the method, and every limitation — lives beside the figure in
[`routing.md`](routing.md#figures) and [`escalation.md`](escalation.md#figures), one
section per figure. That is a deliberate split: the figures used to carry a five-section
footer that was, on some plots, taller than the plot, and on the sweep table it collided
with the data.

The full record is not prose someone has to remember to keep. `src/shunt/inspect/plot_frame.py`
is the one legal figure writer in the repo — a lint gate (SH007) blocks the spellings that
try to skip it and a runtime guard (`benchmark/plot_guard.py`, active under the test suite and
the figure targets) refuses the write itself from any other caller, whatever it is spelled
like. `benchmark/plot_frame.py` is a re-export shim over the same implementation,
so the benchmark figures and the ephemeral `shunt inspect` diagnostics share one contract
rather than a copy that drifts. The frame records each figure's reading, goal, terms, notes,
limitations, sample counts and input digest into a committed `figures.json` beside the code
that writes it — `benchmark/<half>/figures.json` for the three benchmark halves, and
`src/shunt/inspect/inference/figures.json` for the inference half, whose producer ships inside
the package (the diagnostics pass no `Provenance` and so write no row). A second
gate (SH009) then holds that manifest in a bijection with the docs: every figure has a
section, every section has a figure, and every rendered string the manifest carries — title,
subtitle, caveat, notes and limitations — must match byte-for-byte in both places. So a retired figure cannot leave a stale explanation behind it, which is the way
this kind of documentation normally rots.

Anything that depends on the data (how many tasks were dropped as coverage gaps, whether
the frontier ran on a subset, whether a detector has no usable signal) is computed at
render time rather than written into a caption, so it cannot go stale as the data grows.
Layout is checked the same structural way: `src/shunt/inspect/plot_contract.py` measures every
rendered artist and refuses to write a figure with an overlapping title, a table spilling
past its axes, or a clipped tick label.

### Regret, and how to read the regret plot

Regret is the bandit/RL measure of decision quality: **how much worse off you are for not
having made the best possible choice.** It is always relative to an optimal baseline — here
the **Oracle**, which routes every task to the ideal model with perfect hindsight.

Per task, `regret = oracle_reward − strategy_reward`, in the benchmark's reward units
(`passed − γ × cost`, γ=0.1). Route where the oracle routed and the regret for that task is 0.
You incur it two ways:

- **Quality regret** — you routed to a model that *failed* a task the oracle solved.
- **Cost regret** — you solved it, but paid for a bigger model when a cheaper one would also
  have passed.

`CumReg` is that per-task gap summed over the task sequence, and
`docs/assets/figures/routing/oracle_gap.png` draws it as one climbing line per strategy.
Reading it:

- The oracle line is **flat at 0** by definition — it is the baseline, not a competitor.
- **Lower and flatter is better.** The **slope** is average regret per task; a steep line means
  the strategy makes costly-or-wrong choices consistently, not once.
- Coverage is uneven (the frontier only ran on a subset), so curves for different strategies
  can span different numbers of tasks. Compare **slope and shape** before the endpoint; the
  plot states how many tasks were dropped as coverage gaps.

The figure carries a one-paragraph version of this definition on the canvas, so it stands on
its own when read outside these docs.

## Strategies

| Strategy | Description |
|----------|-------------|
| Oracle | Upper bound: cheapest model that passes each task |
| Always-Cheap | Route all to the cheapest model (derived from the pricing matrix) |
| Always-Frontier | Route all to the most expensive model |
| Random | Uniform random per task (mean over seeds) |
| kNN-semantic | Embed task → retrieve similar → cheapest capable model. A CONTROL: the selection rule with the escalation ladder removed, which no `router.strategy` value produces |
| kNN-semantic-cascade | The opt-in routing strategy: the kNN pick, then the escalation ladder at session cadence |
| Session-Cascade | The shipped default: the cheapest model, then that same ladder — no embedding, no neighbourhood query |
| kNN-semantic-cascade (within-task) | kNN-informed try-verify-escalate INSIDE one task — blocked, not deployable |
| kNN-difficulty | Judge-difficulty selection rule with the ladder removed. A CONTROL — no `router.strategy` value produces it. Judge labels from `gpt-5.6-terra` (committed `judge_difficulty.json`) |
| kNN-difficulty-cascade | Judge-difficulty pick, then the session ladder — blocked, not deployable (needs a per-task judge call at inference) |
| Difficulty-Band-cascade | "Just the judge label + escalation": same-difficulty-band members vote, the cheapest in-band model whose pass rate clears the bar opens the ladder — blocked, not deployable |
| Price-Cascade | Try-verify-escalate in ascending price order — no embeddings, no kNN |
| kNN-semantic-tier | Single-shot: predict the crossover tier, route there directly |

`Price-Cascade` is the zero-ML floor for cascade routing: it tries the `max_tries`
cheapest measured models cheapest-first, stops at the first patch that passes, and
falls back to the frontier model. It has one knob (`max_tries`) and no learned
component, so it is the baseline any learned router has to beat before its
embeddings can be said to earn their keep.

Both cascades escalate to the **same** frontier — the most expensive model the
benchmark actually measured, which is also what `Always-Frontier` routes to.
Escalating to a model that was never run would make the task unscorable rather
than answer it, and a different escalation target on each cascade would make the
zero-ML baseline and the learned router incomparable.

The embedding-based strategies other than the shipped single-shot kNN router are
**offline evaluation strategies**, not live product behavior — the proxy wires in the
router engine (it decides the first turn; see below), but the multi-attempt cascade
(try-verify-escalate) exists only here in the benchmark; it is not implemented on
the live request path, where escalation happens once per session boundary, not per
attempt.

### What the offline eval found about routing

Scored offline, the embedding-based routing strategies split by workload:

- On **QA and reasoning-style** tasks, the task embedding separates
  cheap-solvable from frontier-only work, so kNN has signal to route on.
- On the **agentic-coding** tasks this benchmark targets, the embedding-based
  difficulty signal did not clear the viability bar for cost-at-equal-quality
  relative to fixed-frontier-with-caching. Ranking hard tasks from easy ones off the
  prompt embedding came out near chance — and that result was measured while the
  strategies embedded the short `description` label rather than the task's
  `problem_statement`, so it was pending re-measurement. The manifest has since been
  rebuilt with the real statements (2026-08-05), and the re-measured numbers fell:
  kNN-semantic 77.72% is inside noise of Always-Cheap — a settled null, not a coverage gap
  ([Results](results.md#routing-results)). The router is wired into the live proxy
  (it decides the first turn), outcomes are recorded automatically at session close
  (via off-wire test re-execution when configured), and the learning loop is live.
  On this particular workload, the embedding signal is not presently strong enough
  to justify routing below frontier, though outcomes continue to accumulate.

### Evaluating the exploration policy without spending money

Exploration ships on ([configuration](configuration.md#tune-the-router)), so the
obvious question is what it costs. You can answer it from the committed data alone.
`results.csv` is a partly-dense grid of *measured* (task, model) outcomes. The replay
runs on the largest fully dense sub-grid inside it, found greedily — currently
**165 tasks × 2 models = 330 measured cells** against a full matrix that is 62.1%
dense. On a fully dense sub-grid, replaying a routing policy is exact rather than
estimated: look up the model the policy picks, read the outcome that was actually
recorded for that cell, average. Nothing is simulated and no request is sent.

```bash
python -m benchmark.routing.scripts.plot_exploration
```

This replays the shipped router — the same Thompson sampler, budget cap, and
conservative gate that run in the proxy — over the matrix, once with exploration
off and once with it on, and writes `docs/assets/figures/routing/exploration_cost.png` plus a
summary to stdout. Cells the policy routes to but the benchmark never ran are
skipped and counted, never filled in with a guess.

On the 165-task dense slice, averaged over 20 seeds: exploration costs **1.28× the
exploration-off bill** on average and **1.48× on the worst seed**. That ratio is
paired over the 138 tasks both arms scored — the exploit-only arm drops 27 cells as
unscorable and all 27 are `qwen3.7-plus`, a model outside the dense slice, so
comparing the arms' raw totals would compare different task sets. We previously
published **1.65×/1.77×** from that unpaired ratio; it was ~30% too high. The paired
per-task difference is **−3.2 pp pass rate (95% CI −4.8 to −1.8)** and **+$0.0023
per task (95% CI +$0.0018 to +$0.0029)** — the paired numbers are the ones to read,
since the two arms' marginal pass-rate intervals ([71%, 85%] vs [69%, 82%]) overlap
heavily.

Four caveats keep this honest. The replay's outcome matrix is **static**, so an
exploratory pull can never improve a later decision — this measures exploration's
cost with its learning benefit set to zero, which is the pessimistic half of the
ledger, not a verdict on whether exploration pays. The budget cap counts the
router's own confidence-weighted neighbourhood costs, not realized ones, so the
realized explore/exploit spend ratio can exceed `explore_budget_frac` on an unlucky
seed (0.66 against a 0.4 cap on the worst of 20 seeds here) even though the cap is doing its job. The dense
slice maximises *cells*, which currently favours many tasks over many models: it
holds only the two cheapest models and **no frontier arm**, so the measured overhead
is the cost of exploring between cheap models and is a **lower bound** on the shipped
policy's, where an exploratory pull can land on a model ~40× the price. And it is
one workload.

## Scoring every strategy on one task set — monotone-rank imputation

Running the most expensive ("frontier") model on every task is costly, so Shunt collects
frontier outcomes only where they are most informative. That leaves an **asymmetric
outcome matrix**: cheap and mid models ran on nearly every task, the frontier on only a
subset. Scored naively, each routing strategy would be graded on a *different* set of
tasks — apples to oranges. The default report fixes this by **completing the matrix in
memory** so every strategy is scored on one comparable task set. Full method:
[benchmark-design.md](benchmark-design.md).

**Capability is measured per model, not bucketed into fixed tiers.** Shunt derives a
**per-model capability rank** — a strict weakest-to-strongest order — directly from the
verified outcomes, so a cheaper model that measures stronger than a pricier one ranks above
it from the data, with no hand-tuned tier. The order comes from **pairwise dominance on
co-measured tasks** (which model wins more where both actually ran), never from each model's
raw pass-rate — the frontier ran only on the hard subset, so a raw-rate ranking would wrongly
sort it below cheap models scored on easy tasks. A model with too little data falls back to a
price-implied position (cheaper is assumed weaker) until it earns a measured rank. For the
narrative only, the report groups the ranked models into **ordinal bands** — band 1 (weakest)
to band N (strongest). The bands carry no semantic names: each is described purely by
metadata (its member models, price range, marginal-pass-rate range with CI, and the share of
tasks it is the weakest to solve), and the band count is data-driven — adjacent models whose
capability CIs overlap merge into one band. Bands are a grouping of the measured rank, never
the routing unit.

**The monotonicity assumption.** The rank is a capability ladder. The assumption: if a model
solves a task, every higher-ranked model solves it too. From each task's measured cells Shunt
reads the weakest model observed to pass and the strongest observed to fail, then fills the
gaps under that assumption — above a pass is imputed pass, below a fail is imputed fail. This
completion is **recomputed on every run and never written to `results.csv`**: the committed
data stays real-only, and imputation is a pure in-memory analysis layer that flags every cell
as measured or imputed so the two never blur.

**Why the assumption is conservative (this is load-bearing).** Completing the matrix
credits the always-frontier baseline with a *pass* on every task a weaker model already
solved — free quality at frontier cost. So if the assumption is ever wrong for a task, the
baseline's true quality is only *lower* than imputed, and the router's measured advantage
only *larger*. Imputation can understate routing's lead; it cannot flatter it. We impute to
keep exploration cheap, never to make the router look better than it is.

**We measure how often it breaks — we don't assume it away.** Stronger models sometimes
fail where weaker ones pass. Shunt reports that **monotonicity violation rate** as a
first-class, measured number: on the current data it holds about **90% of the time**
(19 violations over 192 multi-observed pairs — holds 90.1%, violation rate 0.099, 95% CI
[0.064, 0.149]). Where a real higher-ranked fail sits below a
real lower-ranked pass, the contradicted cells are kept as measured — never overwritten by an
imputed value — and the task is flagged. The report also ships a **sensitivity check**:
every conclusion is recomputed with the violating tasks excluded, and any result whose sign
or confidence-interval side flips is surfaced, not hidden.

**Honest about coverage.** When imputation is enabled (default), the completed matrix
excludes every **incomplete challenge** — one whose crossover is still bracketed by an
UNKNOWN band (an unclosed gap between the weakest observed fail and the strongest observed
pass). Only tasks with an **established crossover** (complete) feed the analysis, so every
strategy is scored on the same set with no guessing required. This makes `cost_optimal` and
`full` modes comparable to `ladder`, which collects gap-free data by design. On a partially-
collected `cost_optimal` run, excluded challenges are reported at evaluation time; fully
equal coverage across the whole suite is guaranteed by **ladder collection mode** below.

**What the report shows.** The headline is a **paired** cost/quality contrast — the router
versus fixed-frontier on the *same* completed task set — with its confidence interval, and
it stays honest when that interval crosses zero (equal quality is reported as equal, not
spun as a win). On the current coverage-incomplete data, the cheapest strategy that
matches fixed-frontier quality is `Price-Cascade` (**+2.2 pp, CI crosses zero → not statistically equal**, at roughly **76% lower cost** on the shared measurable set) — but <!-- generated-by: benchmark.routing.report:paired_quality_contrast -->
it is **blocked, not deployable**: the router rejects `price_cascade` at boot, because
stopping at the first passing patch needs a verified outcome mid-session and that is not
one cache-safe decision per session. So this is a bound on what the mechanism is worth,
not an offer. The headline gate itself is adjudicated on the shipped single-shot kNN
router; the full-distribution figure waits on ladder-mode collection.

**A population estimate, as a cross-check.** Alongside imputation, Shunt can estimate the
fixed-frontier baseline's pass-rate and cost directly from a *uniformly random audit* of
frontier outcomes, using a doubly-robust (PPI++/AIPW) estimator that treats cheap+mid
outcomes as covariates. Its validity rests on the random audit, not on cheap outcomes
predicting frontier ones — a poor predictor only widens the interval. It answers a
different question — the *population* pass-rate with an honest interval — and measures the
same violation rate on its audit stratum, so the two methods cross-check rather than
compete. At this task count the interval on the *absolute* frontier pass-rate is wide,
which is exactly why the gate rests on the paired contrast (a McNemar non-inferiority test
with an anytime-valid stopping rule), not on an absolute score. A near-zero paired edge is
itself the signal to stop.

**Running it.** The runner collects live data in one of three modes, selected with
`--strategy`:

- `cost_optimal` (**default**) — a plain `python -m benchmark.runner.run_matrix`: cheap+mid
  on every task, frontier only on tasks where cheaper models *disagree* plus a uniformly
  random audit. The cheapest way to a defensible baseline estimate; the measured
  cheap↔frontier correlation is low (ρ²≈0.04), so the gate rests on the paired contrast plus
  the audit, not the covariate.
- `ladder` — cheap-first, escalating per task weakest-to-strongest model only until the
  first one passes. Observes each task's crossover model exactly, giving gap-free equal
  coverage at minimum spend — the mode that makes the imputed matrix fully equal across the
  whole suite. It escalates up to `--workers` **different challenges concurrently** (the same
  fan-out mechanism and default as `cost_optimal`/`full`); *within* a challenge escalation
  stays serial cheap→strong, stopping at its first pass. Collection is **challenge-atomic**
  even under concurrency: before starting a challenge it predicts the worst-case cost to fully
  complete it (the median measured cost of its still-untested tiers), and a thread-safe
  worst-case **reservation** means concurrent challenges can never jointly cross `--max-cost`
  — a challenge that doesn't fit isn't started, so none is ever left half-collected. The set of
  cells collected is identical regardless of `--workers`; only order and wall-clock differ.
  Only challenges whose crossover is established (complete) feed the analysis.
  Optionally pass `--tasks-file <path.json>` (a JSON list of challenge ids) to run only a
  targeted subset (e.g. only the challenges whose crossover is still unknown); cached rungs
  are reused on resume.
- `full` — the exhaustive every-enabled-model × every-sampled-challenge matrix
  (`--strategy full`). `full --live` with **no** `--max-cost` prompts for interactive
  confirmation before spending (uncapped live spend is dangerous); a non-interactive stdin
  aborts. `cost_optimal` keeps its own `constants_pinned` safety guard and needs no such
  prompt.

`python -m benchmark.runner.collect` is a **deprecated alias** for `--strategy
cost_optimal`. Key `cost_optimal` knobs live under `collect:` in `benchmark/benchmark.yaml`:
`phase_a_mode` (`single` = one representative model from the lower-ranked models, or `full` = every lower-ranked
model), and the two sizing
constants `audit_fraction` (audit sampling probability π) and `noninferiority_margin` (δ).
Pin those two from the live `results.csv` and set `constants_pinned: true` before any paid
run, or the interval is mis-sized.

## Honest limits

- **Task selection bias**: SWE-bench Verified is mostly Python bug fixes, so the
  benchmark doesn't reflect the full distribution of real coding work. Documented
  limitation; addressed by adding diverse task sources later.
- **Timeout handling**: a timeout counts as a fail for that model on that task and
  is recorded in the result row for separate auditing.
- **Cost**: both real (from the API response) and estimated (pricing × tokens) are
  stored; the evaluator can use either.
- **Deterministic judges only**: every task is judged by its test set — no
  LLM-judged tasks. This rules out judge noise but limits task types.
- **Pricing** is taken from the Requesty router listing (2026-07-15); each model
  records its rate, cache-read/write rate, and source in a `price_note` in
  the model registry. That registry rate means *the price in force when the run
  happened*, and it never changes retroactively.
- **Historical cost vs repriced cost.** Provider prices move, so a cell measured in
  July and one measured in August are not automatically comparable. Two costs are
  therefore recorded on every row, forever, and they are used for different things.
  `real_cost` — what the provider actually billed, cache included — is the immutable
  audit record, and it is what the **cache-aware cost axis and the kill gate use**. A
  price refresh can never flip a recorded verdict. The **naive cost axis** on
  `live_gap.png` is instead **repriced** from a dated sheet of today's cheapest listed
  price across OpenRouter, Requesty and HuggingFace Inference Providers, so it answers
  what the same work would cost a user shopping around now. That figure's subtitle
  always says which of the two drew it, and its manifest row records the sheet's date
  and digest. A model no channel prices has no repriced cost — it is reported missing,
  never estimated. On the corpus published here the figure **falls back to recorded
  cost** and says so, because 406 of 1104 cells carry no token counts and repricing
  them at $0 would be a fabrication. Full treatment, including the drift we are *not* fixing (latency,
  and model identity under a fixed name): [Benchmark design](benchmark-design.md).
- **Benchmark ≠ production**: the benchmark can reject bad routing strategies but
  can't prove a good one works in production. The kill gate — non-inferior quality,
  then no worse on cost, sessions, session tail or bill variance and strictly better on
  one of them, against **both** a fixed-frontier-with-caching baseline (the most
  expensive enabled model, currently kimi-k3) **and** a zero-ML cheapest-first policy
  running the same handoff — must be measured on a real workflow, not in the
  benchmark. See [the criterion](benchmark-design.md#the-kill-gate-criterion-is-multi-dimensional-against-two-baselines-pre-registered).
- **Small measured sample, single run**: the suite is 500 tasks but live results
  cover only a nested partial subset so far (all Python), with one stochastic run
  per cell (pass@1), and only ~15–20% of tasks carry routing headroom. See the
  benchmark harness README for the full limitations.

## Citation

```
@inproceedings{jimenez2024swebench,
  title     = {{SWE-bench}: Can Language Models Resolve Real-World GitHub Issues?},
  author    = {Jimenez, Carlos E. and Yang, John and others},
  booktitle = {ICLR},
  year      = {2024}
}
```
