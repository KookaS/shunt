# Routing Benchmark

Evaluates routing strategies against a task×model performance matrix to answer: *which routing algorithm maximizes reward (performance − λ·cost)?*

## Layout

```
routing/
  results.csv                 # THE committed source of truth — per-cell outcomes from live runs
  seed_live.py                # Seed the LIVE outcome store from measured cells (incremental, skippable)
  build_seed_bundle.py        # Regenerate the committed LFS seed bundle from results.csv
  docs_corpus.py              # Cached seed-only store the committed docs figures render from
  live_split.py               # Held-out task split for the live evaluation (pure hash-threshold)
  data/                       # Curated read-only inputs
    challenges.json           # Index of swebench_verified specs (500 instances, challenges, tasks)
    challenges_multimodal.json # Index of swebench_multimodal specs (102 instances, challenges, tasks)
    live_split_manifest.json  # The resolved live-eval holdout: salt, fraction, revision, ids, digest
    seed/                     # LFS-tracked warm-start bundles (one .npz per embedder fingerprint + plain manifest.json)
  strategies/
    __init__.py               # Strategy protocol
    oracle.py                 # Upper bound: perfect per-task selection
    fixed.py                  # Always-cheap, always-frontier, random
    knn.py                    # kNN retrieval (shunt's approach)
    knn_cascade.py             # kNN-informed verify-and-escalate, WITHIN one task (blocked)
    knn_session_cascade.py    # THE OPT-IN `knn_semantic_cascade`: kNN pick + the session-cadence ladder
  exploration_replay.py       # Direct-Method replay of the SHIPPED exploration policy on the dense slice
  run_eval.py                 # Evaluate all strategies
  instrument_control.py       # Positive control + destroyed-signal null (both selection rules)
  sensitivity.py              # Minimum detectable effect — how weak a signal the corpus can see
  metrics.py                  # Metric definitions
  report.py                   # Drives the nine report figures (derived from results.csv)
  figures/                    # One module per report figure; `context.py` loads the corpus once
    kill_gate.py              # Pre-registered delta=5pp non-inferiority forest + paired cost
    cost_quality_frontier.py  # The one cost-quality plane (replaced four)
    evidence_basis.py         # Measured vs imputed, on dollars AND passes, plus per band
    oracle_gap.py             # Oaxaca cost split, regret ladder, gamma-invariance
    cache_economics.py        # List price vs the invoice against the gate's cache model
    decision_audit.py         # Chosen x cheapest-sufficient: the over/under-provisioning budget
    complementarity.py        # Tri-state grid, coverage range, and the routing ceiling
    task_difficulty.py        # Capability bands + what the cascade picked, by difficulty
    arm_manipulation.py       # Manipulation check FIRST, then the reasoning-arm contrast
  scripts/                    # Analysis + figure producers (read results.csv, write docs/assets/figures/routing/)
    compute_costs.py          # Per-model cost/pass rollup from the outcome cache
    consistency_probe.py      # Benchmark-internal consistency: replay same task/model, measure variance
    consistency_probe_metrics.py # Compute consistency metrics (variance, determinism tests)
    cost_quality_headline.py   # Four-point simplification of cost_quality_frontier.png for the front page
    derive_judge_difficulty.py # Generate judge_difficulty.json from judge probe outputs
    judge_probe.py            # Live LLM-as-a-judge difficulty labels on the measured task set (gitignored JSONL)
    judge_probe_metrics.py    # Judge labels vs measured outcomes: AUC/R², agreement, stability, LLM-as-router analysis, human-tag control
    learnability_probe.py     # Embedding learnability control: how much signal survives random shuffling
    knn_nulls.py              # Permutation nulls + the shared kNN selection rule (no plotting)
    ladder_evidence.py        # Per-rung escalation evidence: price multiple, helps/hurts, null
    plot_exploration.py       # Exploration cost/quality and where the budget went
    plot_knn_nulls.py         # embedding_signal: transfer curve, positive control, cross-repo
    predict_then_cascade_eval.py # Evaluate predict-then-cascade gates (accuracy, threshold sweep)
    threshold_sweep.py        # kNN sweep with real outer-loop CV -> the regime map
    viz_knn.py                # knn_calibration: reliability of the weighted neighbour rate
  artifacts/                  # gitignored — parameterized run_eval outputs + embedding cache
  reports/                    # derived CSV/JSON only (gitignored); the PNGs live in docs/assets/figures/routing/
benchmark/
  challenges/
    swebench_verified/        # The 500 swebench_verified instance specs (Python-only)
    swebench_multimodal/      # The 102 swebench_multimodal instance specs (multi-language)
```

There is a **single committed data source of truth**:
`routing/results.csv` — the **per-cell** outcome cache in long/tidy form
(the raw benchmark data, see schema below). Everything else is **derived and
regenerable**: the **per-strategy** summary
(`strategy_summary.csv`) is computed in-memory by `summary.py` (used by
`report.py`, `run_matrix.py`, `run_eval.py`) and written to the gitignored
`reports/` dir; plots (into `docs/assets/figures/routing/`) and parameter sweeps likewise
regenerate from `results.csv`. The one committed exception is the **warm-start seed
bundle** (`data/seed/`, LFS-tracked): precomputed embeddings derived from `results.csv`
so a fresh deployment can warm the live kNN index without re-embedding — regenerate it
with `make seed-bundle`, prove it current with `make check-seed-bundle`.

**One strategy, one committed number.** `cost_quality_frontier.png` is drawn from the
same in-memory rows `strategy_summary.csv` is written from, so the scatter and the table
cannot disagree. The retired `knn_cost_comparison.png` published a *second* (cost, pass)
pair for the kNN router — a proxy 77.7% at \$1.73 against the live engine's number in the
same report set — which is a correctness bug, not a second view. The proxy publisher was
deleted; `strategy_summary.csv` is now the single producer of every strategy's (cost, pass).

## Live-evaluation holdout (`live_split.py`)

Membership is `holdout_score(task_id, "live-split-v1") < fraction` — the same hash-threshold
primitive the calibration holdout uses (`../runner/calibration.py`), under a fifth salt namespace
that aliases none of the four already in draw (`calib-v1`, `order-v1`, `arm-v1`,
`frontier-audit-v1`). Three properties follow, and the tests in `tests/test_live_split.py` pin
each one:

- **Pure.** The split is a function of `(task_id, salt, fraction)` alone. `sample_size`, the
  benchmark seed, the arm weights, the run order, the enabled model pool and how many cells
  `results.csv` happens to hold cannot move it.
- **Over the universe.** The draw runs over every id in `data/challenges.json`, never over the
  collected subset — a split conditioned on coverage would silently re-draw itself as cells land.
- **Nested and pinned, and the ratchet is gated in the generator.** Raising `fraction` only adds
  members. The holdout may therefore be GROWN but never SHRUNK — live cells already collected
  against the committed ids would be invalidated. Two walls hold that, and only the first is
  load-bearing:
  - `write_manifest()` **refuses to write** a manifest that drops any id the on-disk manifest
    already declares, naming the departed tasks and exiting non-zero. Lowering `DEFAULT_FRACTION`,
    changing the salt, renaming an id, or bumping `dataset_revision` in a way that drops a member
    all fail here — at regeneration time, in the same command the maintenance path runs.
  - `TestHoldoutRatchet` compares the committed manifest against a fresh draw. It catches a
    *forgotten* regeneration only: both of its sides move together when the constant is changed
    and the manifest regenerated in one edit, so on its own it is not a ratchet (measured: at
    `fraction=0.1` with the manifest regenerated, 52 of 97 tasks departed and it still passed). <!-- frozen-value: n=97, date=2026-08-20, run=468883b -->

  A genuine reset is therefore deliberate and greppable: delete
  `data/live_split_manifest.json` first, then regenerate. Ids are immutable only relative to
  `challenges.json`'s `dataset_revision`, so the manifest records it and a test fails when the
  two diverge.

`data/live_split_manifest.json` is the committed, reviewable record (plain JSON, deliberately not
LFS, like the seed bundle's manifest). Regenerate it with
`python -m benchmark.routing.live_split`; its `tasks_digest` is the staleness anchor.

## Instrument validity (`instrument_control.py`) — run this before quoting anything

The permutation nulls in `scripts/knn_nulls.py` answer *"could chance have produced this
number?"*. They cannot answer the prior question, *"is this pipeline computing anything about
the task text at all?"* — a front end that embeds the wrong field, or embeds nothing, produces
an observation and a null that agree perfectly and report `NULL RESULT` forever.

`instrument_control.py` answers that one. It builds a small corpus whose task text carries a
known-learnable signal, hands it to the pipeline at the **front** (`matrix["tasks"]`, upstream of
`routing_text` and the embedder), and requires two things:

- **positive control** — the assembled pipeline recovers the planted signal, scoring clearly
  above chance;
- **destroyed-signal null** — the same pipeline, with the outcomes permuted, collapses back to
  chance.

The planted signal is deliberately **orthogonal to repository identity**: every repository
contributes equally to both outcome classes, so a pipeline that recovers only the repo name out
of the label scores at chance and is rejected. `repo_identity_positive_score` re-scores the same
corpus with labels re-aligned onto the repo and is reported as a diagnostic contrast, never as
part of the verdict — dead on the planted signal but high there means the front end propagates
repository identity and nothing finer.

Two rules decide, so the control has **two legs** over that one planted corpus:

| Leg | Rule under test | What it certifies |
|---|---|---|
| `run_control` | `knn_nulls.select_from_rates` | the transfer-curve and cross-repo **figures** |
| `run_strategy_control` | `kNNStrategy` → `RouterEngine.decide` → `SelectionRule` | `reports/strategy_summary.csv`, the table the kill-gate comparison reads |

Clearing one certifies nothing about the other: `select_from_rates` documents three named
divergences from the shipped rule (weighting, `min_samples`, the fallback branch). The strategy
leg hands the engine its whole ranked pool, so `SelectionRule._escalate` can never land on an arm
the corpus has no cell for — it refuses loudly rather than scoring the remainder.

```sh
python3 -m benchmark.routing.instrument_control              # both legs; exit 0 = admissible
python3 -m benchmark.routing.instrument_control --leg strategy
```

`TransferCurve`, `CrossRepo` and `summary.StrategyTable` all take the verdict as a **required,
non-defaulted** field, so no figure and no summary row can be produced — and no verdict quoted —
without stating whether the instrument that produced it has cleared both legs. Every row
`write_summary_csv` emits carries `instrument_admissible` and `instrument_verdict`. An instrument
that has not cleared them is a coverage-gap, not a falsification: a negative result from it says
nothing about routing.

## How weak a signal could this corpus see? (`sensitivity.py`)

A passed positive control licenses one claim: the committed `NULL RESULT` is a null on an
instrument **proven to detect something — at the strength probed**. The planted signal is far
larger than any plausible real routing signal, so whether the null means "nothing is there" or
"something is there, below our floor" is a separate question.

`sensitivity.py` answers it. It re-assigns the **real** outcome rows to the **real** tasks so
that a fraction `rho` of them line up with a direction in the real embedding space, sweeps `rho`
downward, and reports the smallest effect the published test still flags at 80% power — as an
interval, not a point. Because planting only re-assigns rows, the permutation null is unchanged,
so the bar the planted signal must clear is the published analysis's own bar.

`rho` converts to a portable unit: a perfect reader of the planted signal separates
cheap-sufficient from escalation-needed tasks at an AUROC of one half plus half of `rho`. The
sweep reports both, over both splits (the published ungrouped one, which leaks same-repo
siblings into a held-out task's index, and a repo-grouped one), both k-rules (the config `k` and
the figure's selection-corrected best-over-k), and both plantable geometries.

```sh
python3 -m benchmark.routing.sensitivity          # prints; writes nothing
```

## Model registry (`src/shunt/config/models.yaml`) — the cost + routing source of truth

The model registry is shared with the shipped router: one `providers` table (access
channels) and one `models` table. It is the cost source of truth. Prices are the
**Requesty router listing** (the rate actually paid for requesty-routed models;
direct providers list the same published rate).

A model's **required core** — `model_id`, `provider`, `supports_streaming`,
`supports_cache_control` — makes it routable. The **optional `pricing` block** makes it
benchmarkable: a model with no `pricing` is routable but invisible here, so it can never
be scored against a fabricated price.

| Field | Meaning |
|-------|---------|
| `model_id` / `provider` | Model identity and the `providers` row used to reach it |
| `pricing.input_cost_per_1m` / `.output_cost_per_1m` | Price, USD per 1M tokens (Requesty router listing) |
| `pricing.cache_read_cost_per_1m` / `.cache_write_cost_per_1m` | Optional — cache-read/write rate where the provider lists one |
| `pricing.price_provider` | Where the price is quoted from (`requesty` — the router listing) |
| `pricing.price_source` | The pricing-listing URL the number came from |
| `pricing.price_as_of` | Date the price was recorded |
| `pricing.price_note` | Provenance note — the listing the rate came from + cache rates |
| `version` | Stable model-version string (feeds `results.csv` `model_version` staleness); a model-level field, not a pricing field |

The **litellm route is derived**, not stored: `<litellm_prefix>/<model_id>`, e.g.
`deepseek/deepseek-v4-flash`, `openai/alibaba/qwen3.7-plus`. `base_url` and
`api_key_env_var` come from the model's `providers` row.

**Model row order is NOT semantic** — the router ranks models by price for capability ranking. Never re-serialize the registry with a key-sorting dumper.

**Cost model.** `config._pricing_dict` / `config.models_matrix` /
`integrity.estimated_cost` read `input_cost_per_1m` / `output_cost_per_1m` × token
counts. The optional `cache_read_cost_per_1m` / `cache_write_cost_per_1m` record
the provider's cache pricing where listed but are **not yet consumed** by the cost
model (the strategy layer applies its own cache-hit discount). Each entry's
`price_source` / `price_as_of` / `price_note` document provenance; several
near-future model-version strings use the provider's closest published rate and
say so in `price_note`.

## Outcome cache schema (`results.csv`)

The file is populated by live matrix runs (`run_matrix.py --live`), which append real outcome rows.
Header:

```
challenge_id,model,reasoning,pass,cost,in_tok,out_tok,calls,version_hash,model_version,arm_hash,real_cost,estimated_cost,timeout_flag,image_digest,computed_at
```

One row per **current** `(challenge, model, reasoning)` cell (the cache is upserted —
one row per key; superseded rows move to the history log, below):

| Column | Meaning |
|--------|---------|
| `challenge_id` | Instance id = spec file stem under `challenges/swebench_verified/` |
| `model` | Model key (matches the model registry) |
| `reasoning` | Reasoning arm id (per-model effort level); `"default"` on legacy rows aliases to the model's default arm |
| `arm_hash` | Hash of the arm's request params — a staleness anchor; a re-mapped arm recomputes |
| `pass` / `cost` / `in_tok` / `out_tok` / `calls` | Verified outcome + token usage |
| `version_hash` | SHA256 of the instance spec's canonical content **at compute time** (staleness anchor) |
| `model_version` | The model's `version` field (from the registry) **at compute time** (staleness anchor) |
| `real_cost` | Actual measured cost (USD); equals `cost` for cached rows |
| `estimated_cost` | Cost derived from the registry's prices × token counts |
| `timeout_flag` | True if the run hit the per-cell timeout |
| `image_digest` | Canonical **manifest** digest (`sha256:…`) of the SWE-bench image the cell was produced with (staleness anchor) |
| `computed_at` | ISO-8601 timestamp the row was computed — **AUDIT ONLY, never a staleness key** |

Sample row (after a live run):

```
astropy__astropy-7166,deepseek-v4-flash,high,True,0.0239,65928,1078,6,fd811481…,deepseek-v4-flash,3c9a7e02…,0.0239,0.0239,False,sha256:9b0b13…,2026-07-15T12:00:00+00:00
```

### Anchors, staleness & the run-twice-zero guarantee

Staleness is decided by **string-equality on immutable anchors** — no git or
registry lookup happens when *reading* the cache. Four anchors are stored per
row:

- **`version_hash`** — deterministic **SHA256** of the instance spec's
  canonicalized content (`json.dumps(sort_keys=True)`, order-independent). Because
  the spec holds `base_commit`, `FAIL_TO_PASS`, `PASS_TO_PASS`, and
  `dataset_revision`, that hash *is* the git-pinned problem version — no directory
  hashing. **Selection metadata is excluded** (`_HASH_EXCLUDED_KEYS`, currently
  `difficulty_stratum`): a label the model never sees is not execution identity, so
  correcting it must not stale a paid result cell. `challenge_hash(id)` /
  `all_hashes()` expose it.
- **`model_version`** — the model's `version` field from the registry.
- **`arm_hash`** — SHA256 of the reasoning arm's resolved API params
  (`integrity.arm_hash_value`). Re-mapping an arm's native request params (a
  changed thinking budget, say) changes the hash, so only that arm's cells
  recompute. A model with no `reasoning:` block has no arm anchor, and a legacy
  row with an empty `arm_hash` degrades to a no-op rather than restaging a paid
  cell.
- **`image_digest`** — the **manifest** digest (never the config digest) of the
  instance's SWE-bench image, resolved via `docker buildx imagetools inspect`
  (registry query, **no pull**) and canonicalized to a bare `sha256:…`. The
  `:latest` tag is only a *lookup key*; the image's identity is its manifest
  digest. At run time the harness pulls by namespace+tag, so the runner records
  the digest the image **actually** used (`docker inspect` RepoDigest) — stored
  therefore equals produced.

A cell is **STALE** iff current spec `version_hash` ≠ stored **OR** current
`image_digest` ≠ stored **OR** current `model_version` ≠ stored **OR** the
current `arm_hash` for that `(model, reasoning)` ≠ a non-empty stored one;
**MISSING** iff no current row. Missing means *compute new*; stale means
*recompute and archive the old row*. Invalidation is per-cell: an **image
rebuild** invalidates every `(model, reasoning)` cell for that challenge; a
**model bump** invalidates only that model's cells; an **arm re-map**
invalidates only that arm's.

**Never invalidate on resolution failure.** If a digest can't be resolved
(offline / unreachable / yanked tag), the image axis is *skipped* with a warning —
the cell is **not** marked stale. Invalidating on failure would mean
recompute-forever whenever the registry is unreachable.

The **run-twice-zero guarantee** (`test_run_twice_computes_zero`): given a
populated `results.csv` with correct anchors and no changes, the `run_matrix`
planning pass classifies **0** cells as missing-or-stale. This is the invariant
"stored == resolved for unchanged content"; it catches digest-mismatch,
canonicalization, and offline-invalidation bugs as a class.

`check_integrity.py` reports spec-hash drift, removed challenges, and stale model
versions; with `--check-images` it also resolves manifest digests and reports
image-digest drift (offline-safe — an unresolved digest is never drift).

### Append-only history (`artifacts/results_history.csv`)

`results.csv` keeps **only current rows**. When a cell is superseded (recomputed
because it went stale), its old row is appended — with a `superseded_at`
timestamp — to `routing/artifacts/results_history.csv` (**gitignored**, keeping
the public repo lean). Nothing is discarded; the owner can compact or promote the
history later.

## Caching loop (`../runner/run_matrix.py`)

The benchmark is a **backtest over the cache**: strategies are scored by looking
up cached `(challenge × model)` cells (flattened to each model's default reasoning
arm). `run_matrix.py` keeps the cache current:

1. Compute current challenge hashes + read current model versions.
2. Load the `results.csv` cache.
3. Classify every enabled cell: **missing** (no row), **stale** (hash/version
   mismatch), or **present**.
4. **Simulated by default** — with no `--live` (or no API keys) it logs
   *"would run N cells"* and **fabricates nothing**. With `--live` **and** keys,
   it delegates each uncached cell to the orchestrator's real Docker executor.
5. Writes new/updated rows back (live mode only).
6. Writes the derived `reports/strategy_summary.csv` (gitignored; unless `--no-summary`)
   and regenerates plots (unless `--no-plots`).

Per-strategy **coverage** is reported: a strategy whose decision needs an
uncached cell is flagged (can't be backtested) rather than silently skipped.
Respects `benchmark.yaml`'s `sample_size` for local subset debugging.

```sh
# --strategy full = exhaustive matrix (the caching loop below). Default is cost_optimal (adaptive).
python3 -m benchmark.runner.run_matrix --strategy full              # simulated: report gaps, refresh summary + plots
python3 -m benchmark.runner.run_matrix --strategy full --no-summary # cache report + plots only (container default)
python3 -m benchmark.runner.run_matrix --strategy full --live       # real execution (needs Docker + API keys)
```

## Integrity check (`../runner/check_integrity.py`)

Fails (non-zero exit) on any **changed** challenge (content hash ≠ stored
`version_hash`), **removed** challenge (rows remain but the file is gone), or
**stale** model version. `--check-derived` additionally recomputes the per-strategy
summary from `results.csv` **twice** and fails if the derivation is non-deterministic
(there is no committed summary to diff — it is regenerable). Wired into CI
(`benchmark-integrity` job) — light, no model calls.

```sh
python3 -m benchmark.runner.check_integrity --check-derived
```

## Container

A reproducible image (`benchmark/Dockerfile`) runs the loop identically anywhere.
Code is mounted **read-only**; only `results.csv` and `docs/assets/figures/` (both halves'
subdirectories) are
writable. Build from the repo root (BuildKit reads `benchmark/Dockerfile.dockerignore`):

```sh
docker build -f benchmark/Dockerfile -t shunt-benchmark .
docker compose -f benchmark/compose.yaml run --rm benchmark  # simulated loop + plots
```


## Metric definitions

| Metric | Meaning |
|--------|---------|
| AvgPerf% | Tasks solved correctly |
| AvgPerf_ci_lower / AvgPerf_ci_upper | 95% bootstrap CI on AvgPerf% |
| TotalCost | Total backend model cost (USD) — for the difficulty rows this is **model + judge label cost** (the per-task judge bill is folded into the total) |
| Reward | `Σ(1.0 × passed − γ × cost)` per task (γ=0.1 default) |
| CumReg | `total(oracle_reward) − total(strategy_reward)` |
| CumReg_ci_lower / CumReg_ci_upper | 95% bootstrap CI on CumReg |
| rAcc | Fraction of tasks where strategy picked same model as oracle |
| Pareto | True if strategy is on the Pareto frontier (no other strategy has higher AvgPerf% AND lower TotalCost) |
| judge_label_cost | The MEASURED per-task judge label cost a difficulty row paid over its scored tasks (0.0 for every other row); published beside the totals it is folded into |
| instrument_admissible / instrument_verdict | The two-sided instrument verdict for the SHIPPED selection path (`run_strategy_control`), stamped on every row |

## Baselines

| Strategy | Description |
|----------|-------------|
| Oracle | Upper bound: cheapest pass-per-task |
| Always-Cheap | Route all to cheapest model (derived from pricing matrix) |
| Always-Frontier | Route all to most expensive model (derived from pricing matrix) |
| Random | Uniform random (mean over seeds) |
| kNN-semantic | Embed task → retrieve similar → cheapest capable (a CONTROL: the pick without the ladder) |
| kNN-semantic-cascade | The opt-in routing strategy (`router.strategy: knn_semantic_cascade`): the kNN pick, then the session-cadence escalation ladder |
| kNN-semantic-cascade (within-task) | kNN-informed try-verify-escalate INSIDE one task (blocked) |
| kNN-difficulty | Judge-difficulty selection rule with the ladder removed (a CONTROL — the pick without the ladder). Judge labels from `gpt-5.6-terra` (committed `judge_difficulty.json`) |
| kNN-difficulty-cascade | Judge-difficulty pick, then the session ladder (blocked — needs a per-task judge call at inference) |
| Difficulty-Band-cascade | "Just the judge label + escalation": same-difficulty-band members vote, the cheapest in-band model whose pass rate clears the bar opens the ladder (blocked) |
| Price-Cascade | Try-verify-escalate in ascending price order — no embeddings, no kNN |
| Session-Cascade | **The shipped default.** The escalation ladder at session cadence: one decision per session, effort rung then rank rung, climbed rank persisting, cache-safe analogue |
| kNN-semantic-tier | Single-shot: predict the crossover tier, route there directly |

## Challenge store

The **challenge sources** are **SWE-bench Verified** (swebench_verified, 500 Python instances) and **SWE-bench Multimodal** (swebench_multimodal, 102 multi-language instances). Each task is a minimal
spec under `benchmark/challenges/<source>/{instance_id}.json`
(`instance_id, repo, base_commit, version, difficulty_stratum, FAIL_TO_PASS,
PASS_TO_PASS, image_ref, dataset_revision`) whose repo/patch content is pulled on
demand by the official harness — nothing is vendored. The swebench_verified suite spans 12 repos with a spread of difficulty strata; every one has a
verified prebuilt `swebench/sweb.eval.x86_64.*` image. The swebench_multimodal suite covers repositories in multiple languages. Live runs cover a nested
partial subset set by `sample_size` and configured by the manifest source (see the harness README).
`integrity.swebench_spec_hash()` hashes each spec; live `(instance, model)`
outcomes flow into `results.csv` with the spec hash as `version_hash`.
See the benchmark README's *SWE-bench Verified execution* section for the
spec → image → ephemeral-container run flow and the gold-smoke / `--live` commands.

The canonical indices are `benchmark/routing/data/challenges.json` (swebench_verified) and `benchmark/routing/data/challenges_multimodal.json` (swebench_multimodal):
- `challenges` — lightweight index (id, source, language, difficulty)
- `tasks` — metadata dict (id → description, repo, base_commit, difficulty, spec
  path). `routing_text()` prefers `problem_statement` and every committed entry
  carries it, so strategies embed the issue text
  rather than the `description` label
  (`<repo>@<commit12> - resolve <test-id>`, median 106 characters — kept as a
  contrast row so the input change is visible, not asserted)
- top-level `source`, `source_dataset`, `dataset_revision` — the HF provenance

Model pricing and per-model outcomes are kept **out** of challenges.json to
avoid duplication:
- **Model pricing** is sourced from the model registry (`src/shunt/config/models.yaml` —
  the single source of truth). `config.load_matrix()` reads it and exposes it as `matrix["models"]`
  (`{model: {input_price, output_price}}`) for backward compatibility.
- **Per-model outcomes** live in `results.csv` (long/tidy).
  `config.load_matrix()` reconstructs them as `matrix["results"]`. **Until a live
  run appends rows the cache is empty**, so `run_eval.py` and `kill_gate.py` print
  *"no results yet — run the live matrix"* (no crash, no divide-by-zero) and the
  kNN strategies fall back to a cheap default.

Consumers should load the matrix via `config.load_matrix(path)` rather than
reading challenges.json directly, so `models` and `results` are stitched back
in from their sources of truth.

## Data provenance

Challenges are **real SWE-bench Verified instances** pulled from
[`princeton-nlp/SWE-bench_Verified`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
at dataset revision `c104f840` (pinned per spec as `dataset_revision`). Each spec
carries the upstream `repo`, `base_commit`, `version`, and `FAIL_TO_PASS` /
`PASS_TO_PASS` test sets verbatim; the gold-patch smoke confirms the harness loop
end-to-end at $0.

The predecessor OOD176/ACRouter cached matrix was **dropped** — its numbers were
another router's measurements on other models, not trustworthy for our kill gate.
No legacy rows ship with the repo; all real `(instance, model)` outcomes are
now self-collected via the live harness.

## Cost decomposition

The kill gate (`benchmark/runner/kill_gate.py`) decomposes cost savings
using an Oaxaca-Blinder three-factor decomposition:

| Component | Formula | Meaning |
|-----------|---------|---------|
| Price effect | `(F_price − S_price) × S_tok` | Savings from cheaper per-token pricing |
| Volume effect | `(F_tok − S_tok) × S_price` | Savings/cost from token count differences |
| Interaction | `(F_price − S_price) × (F_tok − S_tok)` | Joint effect of price and volume differences |

Only tasks where both arms pass contribute to the decomposition
(equal-quality comparison). Per-task `in_tok`, `out_tok`, and `calls`
are tracked in the matrix.

## Citation

Challenge data is SWE-bench Verified:

```
@inproceedings{jimenez2024swebench,
  title     = {{SWE-bench}: Can Language Models Resolve Real-World GitHub Issues?},
  author    = {Jimenez, Carlos E. and Yang, John and others},
  booktitle = {ICLR},
  year      = {2024}
}
```

The reward weight (`γ = 0.1`) follows the ACRouter/CodeRouterBench convention
([LanceZPF/agent-as-a-router](https://github.com/LanceZPF/agent-as-a-router));
their OOD176 outcome data is no longer used (see *Data provenance*).
