# Routing Benchmark

Offline evaluation of routing strategies — which model should handle a given task, given the task's prompt and known model performance on similar tasks.

## Limitations

Treat the current numbers as a **pilot, not a verdict.** They come from a small, narrow sample and should not be read as a settled result.

- **Small measured N.** The challenge suite is the full 500-task Verified set, but **live `results.csv` coverage is a nested partial subset** (set by `benchmark.sample_size`) — start it small (10 → 20 → 200 → 500) so cost grows deliberately. At the current pilot coverage (~tens of tasks) pass-rate confidence intervals are ±10–14 points — wide enough that most strategy differences are not statistically distinguishable.
- **Single sample per cell (pass@1).** Each task×model outcome is one stochastic agentic run; a lone pass or fail can flip on a rerun. Decisions that hinge on one cell are noisy.
- **The routable tail is small.** On SWE-bench Verified the field already solves ~80% of tasks with cheap models, so only ~15–20% of tasks carry any routing headroom. Aggregate metrics are dominated by easy tasks where every model succeeds and routing is irrelevant.
- **Frontier coverage is censored under cascade collection.** A tiered/cascade collector would only run a model at the tier a task needs, so easy tasks never get a frontier data point — the fixed-frontier baseline (the kill-gate reference) would be unmeasurable without a full-matrix run or an adaptive collection strategy. An **adaptive frontier-collection mode** is available via `python -m benchmark.runner.run_matrix --strategy cost_optimal` (or the deprecated alias `python -m benchmark.runner.collect`): it runs cheap+mid models on every task, then routes frontier runs to disputed tasks plus a uniform random audit, and estimates the baseline with a doubly-robust (PPI++/AIPW) estimator. See `docs/benchmark.md` for the method and assumptions.

**Why more samples:** to make a defensible claim (tight CIs, a routable subset large enough to measure, robustness to pass@1 noise), run the full 500-task suite (already materialised) with ≥2 samples at escalation-boundary decisions. For a **fixed manifest**, the nested run order (`runner/sampling.py`) makes each step up in `sample_size` *add* tasks rather than reshuffle, so earlier `results.csv` cells are reused, not re-spent (see the caveat below — regenerating the manifest can move the order).

## What this harness offers offline — a checklist

Live data is collected **once**; everything after that should run from the committed corpus, on
any machine, with no API keys. Each line is a capability with the status it has **today**,
checked against the code — not a roadmap. **YES** holds · **PARTLY** holds with a caveat that
changes how you read the result · **NO** does not hold.

- **YES — replay a collected run in a container.** `runner/offline_replay.py` rebuilds each
  captured step inside that instance's prebuilt SWE-bench image and re-derives its verified
  outcome through SWE-bench's own grader. No model call, no spend. Cost is measured rather than
  guessed (~3.5 s median per step; ~104 worker-hours for the full 799-trajectory corpus) — see
  *What a re-replay costs* in [`docs/benchmark.md`](../docs/benchmark.md). Not to be confused with
  `benchmark/Dockerfile` + `compose.yaml`, which containerise the **harness itself** for a
  simulated `run_matrix`.
- **PARTLY — from a fresh clone, once you supply what git cannot hold.** Replay reads the per-step
  `git diff HEAD` captures from `runner/artifacts/step_snapshots/`, which is **gitignored**
  (`step_snapshots.SNAPSHOT_ROOT`; ~74 MB / 792 trajectories on the collecting host), and a
  checkout without them raises `SnapshotsMissingError` — deliberately, since filesystem absence
  cannot be told from "this run captured nothing". `runner/snapshot_archive.py` closes that half:
  `make state-export` packs the captures into `escalation/data/live/state/` as one deterministic
  `.tar.gz` per trajectory (34 105 506 B of diffs → 1 647 548 B on disk, ~1.5 MB in a packfile;
  plain git, not LFS), and `make state-import` restores them byte-identically on any checkout.
  **Two inputs remain outside git and always will:** the instance images (~100 GB) and the gold
  `patch`/`test_patch` rows, which `offline_replay._dataset_row` fetches from the HF dataset at
  replay time *without* pinning `swebench_specs.DATASET_REVISION`. `make replay-inputs` enumerates
  every one of them and exits non-zero — a partial reproduction that silently produces different
  numbers is worse than a refusal. So a clone can re-score policies unconditionally, and re-derive
  outcomes once it has Docker, the images, and HF access.
- **YES — backtest a policy over the corpus.** The loop you actually iterate in.
  `make escalation-eval` re-scores the escalation detector over all 799 trajectories in ~90 s;
  `benchmark.routing.run_eval` does the same for routing strategies over `results.csv`. No
  containers, no requests. Train/eval splits are offline too: escalation uses grouped CV
  (`prefix_eval`, grouped by challenge so no instance straddles a fold), routing has a
  deterministic hash-thresholded holdout (`runner/calibration.py`) and a held-out kNN sweep.
- **PARTLY — backtest one model's runs.** The data separates per model, and the cost of doing so
  is tabulated per model in [`docs/benchmark.md`](../docs/benchmark.md), but neither evaluator
  has a `--model` selector. You do it by feeding a filtered input: `offline_replay` takes one
  trajectory id at a time, `escalation.run_eval` takes a `--live-dir`, `routing.run_eval` takes a
  self-contained `--matrix`. Workable, not first-class. A single-model re-replay is also the
  worst case for cost — it touches nearly as many instances but puts one trajectory in each, so
  most instances pay their admissibility gate for a single trajectory instead of ~4.8.
- **NO — evaluate a model that has never run.** Impossible offline, and not a gap that can be
  closed with more container time: a model with no trajectories has no steps to replay and no
  cells to look up. Adding a model to the comparison is a live collection pass.
- **PARTLY — evaluate routing offline.** Mechanically fine (`routing/run_eval.py` reads only
  cached cells and flags a decision needing an uncached one instead of guessing). **But the input
  is degraded:** `problem_statement` is absent from all 500 specs under
  `challenges/swebench_verified/` and from all 500 `tasks` entries in
  `routing/data/challenges.json`, so `strategies.routing_text()` falls back to `description` — a
  ~106-character `<repo>@<commit12> — resolve <test-node-id>` label. The embedding channel is fed an
  identifier, not the task. Every embedding-based number is pending re-measurement on a manifest
  that carries the statement; the zero-ML strategies are unaffected.
- **PARTLY — evaluate escalation offline.** `make escalation-eval` scores the detector over the
  stamped corpus with permutation nulls. Its report carries a `deployability` verdict, currently
  **OFFLINE-ONLY UPPER BOUND**: 2 of the 5 features (`infra_rate`, `max_action_repeat_rate`) read
  fields that do not exist at the production decision point, and the eval scores one decision per
  step where production decides once per session (`escalation/deployability.py`). The number is
  real; it bounds a policy production does not run.
- **PARTLY — all benchmark data in git.** Tracked: `routing/results.csv`, the 500 instance
  specs, `routing/data/challenges.json`, the 799 trajectory JSONL files with their per-step
  stamps, `manifest.json`, `admissibility.json`, `stamp_ledger.json`, and the report PNGs.
  Untracked: the per-step diffs above, the ~100 GB image set, and the HF dataset rows. Everything
  needed to **re-score** is in git; what is needed to **re-derive** is not.
- **YES — flag a model the collection has not covered.** `benchmark.model_coverage` (`make
  model-coverage`) enumerates the models in `benchmark.yaml`'s `models:` list, not the models the
  data happens to contain, and exits nonzero when one of them is `ABSENT` (no data) or `THIN`
  (below the floor its analysis needs). Both floors are the ones already pinned elsewhere:
  `capability_rank.K` routing cells, below which a model's capability rank is a price prior rather
  than a measurement, and `prefix_eval.MIN_ROWS` trajectories admissible at the shallowest
  evaluated depth. Add a model and the check says so on the next run. On the current corpus
  `zai-glm-5.2` and `kimi-k3` are already `THIN` on escalation.

## Layout

```
benchmark/
  admissibility.py             # Instrument-validity adjudicator (positive control + destroyed-signal null)
  model_coverage.py            # Per-model corpus coverage — flags enabled models the collection missed
  challenges/                  # Individual challenge files
    swebench_verified/         # SWE-bench Verified instance SPECS (the sole source, runnable)
  runner/
    swebench_specs.py          # Spec load/materialise (repos pulled on demand, not vendored)
    infer.py                   # predictions.jsonl: gold (keyless) | live (key-gated)
    swebench_harness.py        # Thin wrapper over `python -m swebench.harness.run_evaluation`
    swebench_smoke.py          # $0 gold-patch smoke driver
    select_swebench.py         # Rank Verified instances from SWE-bench/experiments
    build_challenges.py        # Reproducible producer: materialise all 500 specs + rebuild challenges.json
    sampling.py                # Nested, diversity-first run order (partial runs reuse cached cells)
    calibration.py             # Deterministic full-matrix calibration holdout (hash-thresholding)
  routing/                     # Active: routing strategy benchmark
    results.csv                # THE committed source of truth — per-cell outcomes from live runs
    data/                      # Curated read-only inputs
      challenges.json          # Challenge index of the 500 swebench_verified specs (challenges, tasks)
    strategies/                # Routing strategies (one file per strategy)
      __init__.py
      oracle.py                # Perfect-information upper bound
      fixed.py                 # Fixed-model baselines
    run_eval.py                # Evaluate all strategies against a matrix
    instrument_control.py      # Positive control + destroyed-signal null for the routing pipeline
    metrics.py                 # Metric definitions (cost, quality, trade-offs)
    report.py                  # Comparison tables and plots (derived from results.csv)
    scripts/                   # Analysis + figure producers (read results.csv, write reports/)
      compute_costs.py         # Per-model cost/pass rollup
      embedding_compare.py     # Arctic vs Jina-code neighbourhoods
      knn_nulls.py             # Permutation nulls + the shared kNN selection rule
      plot_exploration.py      # Exploit-only vs exploit+exploration cost/quality
      plot_knn_nulls.py        # Transfer-vs-k curve + cross-repo transfer matrix
      plot_strategies.py       # Strategy Pareto scatter (plotted FROM strategy_summary.csv)
      plot_timing.py           # API calls per task, per model and per routed strategy
      threshold_sweep.py       # kNN hyperparameter held-out sweep + allocation panel
      viz_knn.py               # kNN neighbourhood / routing-map visualisations
    reports/                   # Regenerable plots (PNG, tracked) + derived strategy_summary.csv (gitignored)
  .gitignore
  README.md                    # This file
```

Everything except `results.csv` is derived: the per-strategy summary, plots, and
parameter sweeps are all regenerated from it into `reports/` (the PNGs are
tracked; the derived `strategy_summary.csv` is not) — there is a **single committed source of truth**.

## Run

```sh
# Is the routing pipeline measuring anything? Plants a known-learnable signal in the task
# text and asks the assembled Embedder -> neighbourhood -> selection path to recover it, then
# destroys the signal and asks it to collapse to chance. Exits non-zero when it does not clear
# both legs — no routing verdict is quotable until it does.
python3 -m benchmark.routing.instrument_control

# Evaluate strategies against the cached matrix (writes parameterized CSV to artifacts/)
python3 -m benchmark.routing.run_eval

# Runner (simulated by default). --strategy cost_optimal is the DEFAULT: adaptive
# cheap+mid on all tasks, frontier only on disputed tasks + a random audit (minimises
# frontier spend; savings are scale-dependent — measured ρ²≈0.04 is low, so the gain
# over `full` is modest at small task counts).
python3 -m benchmark.runner.run_matrix

# --strategy full = exhaustive every-enabled-model × every-sampled-challenge matrix.
python3 -m benchmark.runner.run_matrix --strategy full

# `python -m benchmark.runner.collect` is a DEPRECATED alias for --strategy cost_optimal.

# Integrity gate: hashes match, no removed challenges, versions current, no drift
python3 -m benchmark.runner.check_integrity --check-derived

# Coverage gate: does every model in `benchmark.yaml` have enough collected data to be
# evaluated? Exits nonzero on a model that is ABSENT or THIN — the signal that a live
# collection is incomplete rather than finished. Reads committed data only.
python3 -m benchmark.model_coverage
```

### Scaling the suite / cost-safe partial runs

The 500-challenge suite is regenerable from the pinned dataset:

```sh
python3 -m benchmark.runner.build_challenges         # materialise 500 specs + rebuild challenges.json
python3 -m benchmark.runner.build_challenges --limit 5   # cheap dry build of the first few
```

Live runs are gated by `benchmark.sample_size` in `benchmark.yaml` (0 = all). Because
`runner/sampling.py` orders challenges into a **fixed, diversity-first, nested**
sequence (round-robin across repos × difficulty strata), raising `sample_size`
`10 → 20 → 200 → 500` only *adds* tasks — already-computed `results.csv` cells are
reused, never re-spent. Start small to bound cost before committing to the full matrix.

**Caveat — the nesting guarantee holds for a _fixed_ manifest.** The order stratifies by
`difficulty_stratum`, so regenerating `challenges.json` (`build_challenges.py`) after a
difficulty label changes can move a task's position and
shift which tasks fall in a given `sample_size` prefix. A relabel also changes a spec's
content hash, which stales that task's cached cells (`version_hash` mismatch). Freeze the
manifest across a partial-run ramp; treat a manifest regen as a new baseline.

The order is **cheapest-difficulty-first** (each repo leads with its easy tasks), so a
small prefix (~first 20) is intentionally easy/medium-heavy — `hard` tasks (~9% of the
suite) surface later. A small partial run is a repo-diverse smoke of the suite, **not**
a sample of the frontier-headroom tail where routing decisions carry the most value;
don't over-read it. The kill-gate claim uses the full set (or the calibration holdout).

## Cache + integrity infrastructure

The per-model outcome cache (`routing/results.csv`) is a long/tidy
`(challenge × model)` table, **upserted** (one current row per
`(challenge, model, reasoning)`). Its header is
`challenge_id,model,reasoning,pass,cost,in_tok,out_tok,calls,version_hash,model_version,arm_hash,real_cost,estimated_cost,timeout_flag,image_digest,computed_at`
— the verified outcome plus integrity columns. Staleness is decided by
**string-equality on four immutable anchors**: `version_hash` (SHA256 of the
instance spec's canonical JSON — base_commit + F2P/P2P + provenance),
`model_version` (from the model registry), `arm_hash` (SHA256 of the reasoning
arm's resolved API params, so re-mapping an arm's knobs invalidates only that
arm's cells), and `image_digest` (the canonical **manifest** digest of the SWE-bench
image, resolved via `docker buildx imagetools inspect` — `:latest` is only a
lookup key, identity is the manifest digest). `computed_at` is **audit-only,
never a staleness key**. A cell is
**stale** iff any anchor drifted, **missing** iff no current row; an image
rebuild invalidates a challenge's cells, a model bump only that model's. **A
digest that can't be resolved (offline) never marks a cell stale.** Superseded
rows move to an append-only history log (`routing/artifacts/results_history.csv`,
gitignored) — nothing is discarded. The **run-twice-zero guarantee**: a populated
cache with unchanged content classifies **0** cells to recompute
(`test_run_twice_computes_zero`). The caching loop (`runner/run_matrix.py`) runs
only **missing/stale** cells and is **simulated by default** (fabricates nothing
without `--live` + API keys); with no cached rows the eval scripts print *"no
results yet — run the live matrix"* instead of crashing. See
[`routing/README.md`](routing/README.md) for the full schema, anchor set, and the
container invocation.

## SWE-bench Verified execution (the real executor)

The routing benchmark's runnable tasks are **SWE-bench Verified** instances, scored
by the **official `swebench` harness** — we reuse it, we do not reimplement test
parsing. Needs Docker + the `benchmark` extra: `pip install -e '.[dev,benchmark]'`.

### Instance-spec setup (not vendored repos)

Each task is a small **spec** under `challenges/swebench_verified/<instance_id>.json`
holding only what's needed to run + identify it: `instance_id, repo, base_commit,
version, difficulty_stratum, FAIL_TO_PASS, PASS_TO_PASS, image_ref, dataset_revision`.
`dataset_revision` pins the HF dataset commit the fields were pulled from — the spec's
provenance. `swebench_specs.py` also writes a `problem_statement` — the upstream issue
text, the same string the harness hands the agent — so that routing embeds the task
rather than the `description` label; it is excluded from the spec content hash (it adds
no execution identity), so backfilling it stales no cached result cell. **The 500
committed specs predate that field. The key is absent entirely — not present-but-empty —
and so is every `tasks` entry in `routing/data/challenges.json`, so
`strategies.routing_text()` falls back to `description` on 500/500 tasks** (a
~106-character `<repo>@<commit12> — resolve <test-node-id>` label). The repo snapshot,
environment, and patches are **pulled on
demand** from the HF dataset (`princeton-nlp/SWE-bench_Verified`) and the prebuilt
instance image — no repos are copied into the tree. The suite is the full **500
Verified instances** across 12 repos (django, sympy, sphinx, matplotlib,
scikit-learn, astropy, xarray, pytest, pylint, requests, seaborn, flask), each
with a verified prebuilt `swebench/sweb.eval.x86_64.*` image and a spread of
difficulty strata; live runs cover a nested partial subset (`sample_size`).
Materialise specs by id. This writes **spec files only** — it does not touch
`routing/data/challenges.json`, and routing reads only that manifest, so backfilling specs
alone still leaves `routing_text()` falling back to `description` on every task. Rebuilding
the manifest is the second half of the fix: `build_challenges` (above) re-materialises all
500 specs from the pinned revision *and* rewrites the manifest in one pass.

```sh
python -m benchmark.runner.swebench_specs astropy__astropy-7166 psf__requests-1142 …
python -m benchmark.runner.build_challenges   # then rebuild the manifest routing reads
```

### Gold-patch smoke ($0, no API keys)

Proves the whole harness loop: apply each instance's **gold** patch and confirm it
RESOLVES. Costs $0 and needs no model keys. Gold is a **pipeline check, not routing
data** — it is written to a separate gitignored report
(`runner/artifacts/smoke_gold.json`), never to `results.csv`.

```sh
python -m benchmark.runner.swebench_smoke            # all materialised specs; target N/N resolved
python -m benchmark.runner.swebench_harness preds.jsonl --run-id r   # score an existing preds file
```

### Live inference (`--live`, needs API keys)

Live mode runs one **consistent agent scaffold** (`mini-swe-agent`) per
`(instance, model)` to produce a patch, then scores it through the same harness.
Scaffold consistency across models is required for a valid cheap-vs-frontier
comparison. It is **gated on provider keys** — a keyless environment raises
`MissingApiKeysError` and **never fabricates a patch**. Live outcomes DO go into
`results.csv` (version_hash from the instance spec, model_version from
the model registry), driven by the caching loop:

```sh
export DEEPSEEK_API_KEY=…      # models on the `deepseek` provider (read from env by the SDK)
export REQUESTY_API_KEY=…      # models on the `requesty` provider (routed via router.requesty.ai)
python -m benchmark.runner.run_matrix --live    # cost_optimal (default), real execution; no keys ⇒ simulated
python -m benchmark.runner.run_matrix --strategy full --live --max-cost 20   # capped exhaustive run
```

`--strategy full --live` with **no** `--max-cost` prompts for interactive confirmation
before spending (uncapped live spend is dangerous); a non-interactive stdin aborts.

Keys are read from **environment variables** (the OpenAI SDK convention) — set them
however you like (shell export, direnv, a secrets manager). At startup the runner
also loads a gitignored `.env` from the working directory (or `$SHUNT_ENV_FILE`);
a real environment variable always wins over a value in it. Each model in
the registry (`src/shunt/config/models.yaml`) names a `provider`, and that provider's row carries the
`base_url` and `api_key_env_var` used to reach it. The litellm route is derived as
`<litellm_prefix>/<model_id>`. No credentials file is ever committed.

### Container / image lifecycle

The harness **pulls** prebuilt images from a namespace (default `swebench` on Docker
Hub; Epoch AI's GHCR mirror is the alternative) — it never builds the ~2,000 GB image
set locally. Runs are **ephemeral** (`--rm` containers); `--cache_level env` keeps the
shared base/env images (~100 GB) and prunes per-instance images so disk stays flat.
All harness logs, predictions, and reports land in the gitignored
`runner/artifacts/` working dir.

## Container

```sh
docker build -f benchmark/Dockerfile -t shunt-benchmark .   # from repo root
docker compose -f benchmark/compose.yaml run --rm benchmark  # code read-only, cache read-write
```
