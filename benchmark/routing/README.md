# Routing Benchmark

Evaluates routing strategies against a task×model performance matrix to answer: *which routing algorithm maximizes reward (performance − λ·cost)?*

## Layout

```
routing/
  results.csv                 # THE committed source of truth — per-cell outcomes from live runs
  data/                       # Curated read-only inputs
    challenges.json           # Index of the 500 swebench_verified specs (challenges, tasks)
    external_swebench.csv     # Per-instance resolve rates from SWE-bench/experiments (separate table)
  strategies/
    __init__.py               # Strategy protocol
    oracle.py                 # Upper bound: perfect per-task selection
    fixed.py                  # Always-cheap, always-frontier, random
    knn.py                    # kNN retrieval (shunt's approach)
    knn_cascade.py            # kNN-informed verify-and-escalate
    external_prior.py         # Escalate on external p_solve difficulty (in-sample lookup)
    knn_blended.py            # kNN over our verified runs ∪ external Verified priors (down-weighted)
  heldout_eval.py             # Out-of-sample generalization over held-out instances (no live result yet)
  exploration_replay.py       # Direct-Method replay of the SHIPPED exploration policy on the dense slice
  run_eval.py                 # Evaluate all strategies
  metrics.py                  # Metric definitions
  report.py                   # Comparison tables and plots (derived from results.csv)
  scripts/                    # Analysis + figure producers (all read results.csv, write reports/)
    compute_costs.py          # Per-model cost/pass rollup from the outcome cache
    embedding_compare.py      # TF-IDF vs embedding neighbourhoods (retrieval quality)
    plot_exploration.py       # Exploit-only vs exploit+exploration cost/quality + explore share
    plot_external.py          # External-signal plots (difficulty, ours-vs-external, held-out)
    plot_strategies.py        # Strategy Pareto scatter (same rows as report.py)
    threshold_sweep.py        # kNN (k, success_rate, min_samples) sweep + reward heatmap
    viz_knn.py                # kNN neighbourhood / routing-map visualisations
  artifacts/                  # gitignored — parameterized run_eval outputs + embedding cache
  reports/                    # gitignored — regenerable plots (PNG) + derived strategy_summary.csv
../runner/build_external_prior.py  # Regenerates data/external_swebench.csv from the experiments clone
benchmark/
  challenges/
    swebench_verified/        # The 500 instance specs (the sole challenge source)
```

There is a **single committed data source of truth**:
`routing/results.csv` — the **per-cell** outcome cache in long/tidy form
(the raw benchmark data, see schema below). Everything else is **derived and
regenerable, never committed**: the **per-strategy** summary
(`strategy_summary.csv`) is computed in-memory by `summary.py` (used by
`report.py`, `run_matrix.py`, `run_eval.py`) and written to the gitignored
`reports/` dir; plots and parameter sweeps likewise regenerate from `results.csv`.

## Model registry (`src/shunt/config/models.yaml`) — the cost + routing source of truth

The model registry is shared with the shipped router: one `providers` table (access
channels) and one `models` table. It is the cost source of truth. Prices are the
**Requesty router listing** (the rate actually paid for requesty-routed models;
direct providers list the same published rate).

A model's **required core** — `model_id`, `tier`, `provider`, `supports_streaming`,
`supports_cache_control` — makes it routable. The **optional `pricing` block** makes it
benchmarkable: a model with no `pricing` is routable but invisible here, so it can never
be scored against a fabricated price.

| Field | Meaning |
|-------|---------|
| `model_id` / `tier` / `provider` | Model identity, routing tier (cheap/mid/high/frontier), and the `providers` row used to reach it |
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

**Model row order is semantic** — `SelectionRule._escalate` returns the last row of a
tier in file order. Never re-serialize the registry with a key-sorting dumper.

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
Code is mounted **read-only**; only `results.csv` and `reports/` are
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
| TotalCost | Total backend model cost (USD) |
| Reward | `Σ(1.0 × passed − γ × cost)` per task (γ=0.1 default) |
| CumReg | `total(oracle_reward) − total(strategy_reward)` |
| CumReg_ci_lower / CumReg_ci_upper | 95% bootstrap CI on CumReg |
| rAcc | Fraction of tasks where strategy picked same model as oracle |
| Pareto | True if strategy is on the Pareto frontier (no other strategy has higher AvgPerf% AND lower TotalCost) |

## Baselines

| Strategy | Description |
|----------|-------------|
| Oracle | Upper bound: cheapest pass-per-task |
| Always-Cheap | Route all to cheapest model (derived from pricing matrix) |
| Always-Frontier | Route all to most expensive model (derived from pricing matrix) |
| Random | Uniform random (mean over seeds) |
| kNN | Embed task → retrieve similar → cheapest capable |
| kNN-cascade | kNN-informed try-verify-escalate |
| External-Prior | SWE-bench leaderboard per-task difficulty prior; escalate on external p_solve signal |

## Challenge store

The **sole** challenge source is **SWE-bench Verified**. Each task is a minimal
spec under `benchmark/challenges/swebench_verified/{instance_id}.json`
(`instance_id, repo, base_commit, version, difficulty_stratum, FAIL_TO_PASS,
PASS_TO_PASS, image_ref, dataset_revision`) whose repo/patch content is pulled on
demand by the official harness — nothing is vendored. The suite is the full **500
instances** across 12 repos with a spread of difficulty strata; every one has a
verified prebuilt `swebench/sweb.eval.x86_64.*` image. Live runs cover a nested
partial subset set by `sample_size` (see the harness README).
`integrity.swebench_spec_hash()` hashes each spec; live `(instance, model)`
outcomes flow into `results.csv` with the spec hash as `version_hash`.
See the benchmark README's *SWE-bench Verified execution* section for the
spec → image → ephemeral-container run flow and the gold-smoke / `--live` commands.

The canonical index is `benchmark/routing/data/challenges.json`:
- `challenges` — lightweight index (id, source, language, difficulty)
- `tasks` — metadata dict (id → description, repo, base_commit, difficulty, spec path)
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
