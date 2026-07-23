---
title: Configuration
description: Add provider credentials and register new models, with or without a benchmark run.
---

# Configuration

Shunt ships with a registry of providers and models. Configuring it means two
things: giving it keys, and telling it about models you want it to route to.

## Add credentials

Every provider reads its key from one environment variable. Set the variable for
the providers you use; shunt ignores the rest.

```bash
export REQUESTY_API_KEY=...
export DEEPSEEK_API_KEY=...
```

`.env.example` lists every provider variable — the two the shipped registry
routes to out of the box (Requesty, DeepSeek) plus the wider catalog in
`examples/providers/`. Copy it to `.env` and fill in what you need — shunt loads
that file at startup, and a real environment variable always wins over a value in
it. `.env` is gitignored; keep it that way.

To find the variable for a provider, look at its `api_key_env_var` — in
`src/shunt/config/models.yaml` for the two shipped providers, or in that
provider's `examples/providers/<name>.yaml` fragment for the rest. `OPENAI_API_KEY`
for OpenAI, `GROQ_API_KEY` for Groq, and so on. Two of the providers are
aggregators — Requesty and OpenRouter — where one key reaches many vendors. Local
models (Ollama, vLLM) need no key at all.

## Add a model

The registry lives at `src/shunt/config/models.yaml` inside the package.
To change it, write your own at `~/.config/shunt/models.yaml`, or point
`SHUNT_CONFIG_DIR` somewhere else.

**Your file replaces the packaged registry. It is not merged with it.** If your
config lists one model, shunt knows one model. To keep the shipped models and add
your own, start from a copy of the packaged file.

### Without a benchmark run

Three fields make a model registerable — `model_id`, `tier`, and `provider`. A
model row is picked up the moment it exists; `tier` is the prior the routing is
designed to start from before real outcomes accumulate. (Pre-alpha note: the live
proxy now calls `engine.decide()` to choose a model on the first turn. Outcomes can be
recorded manually via `shunt flag`, or captured automatically at session close once you
configure a capture work_dir (see [Tune the router](#tune-the-router)); with neither, the
router typically cold-starts every session to the cheap default — see [architecture.md](architecture.md).
Registering models sets up the pool the router uses for decision seeding and
makes them scoreable in the offline benchmark.) The two `supports_*` fields below
are optional; they default to streaming on, cache control off.

```yaml
providers:
  groq:
    base_url: https://api.groq.com/openai/v1
    api_key_env_var: GROQ_API_KEY
    litellm_prefix: groq

models:
  gpt-oss-120b-groq:
    model_id: openai/gpt-oss-120b   # the id the provider knows it by
    tier: cheap                     # cheap | mid | high | frontier
    provider: groq                  # must name a row in `providers:`
    supports_streaming: true
    supports_cache_control: false   # true only if you've confirmed it
```

Set `supports_cache_control` to `true` only when you know the provider accepts
cache breakpoints. Claiming support that isn't there earns a 400 mid-request;
claiming less than the truth just costs you the discount. Guess low.

The `examples/providers/` directory has one of these per provider, ready to copy.

Adding a model to the registry makes it *known*, not *live*. To have the running
router actually pick it, also add its name to `router.yaml`'s `models:` list — see
[Choose which models are live-routable](#choose-which-models-are-live-routable).

**Order matters.** Models are read in file order, and that order is load-bearing
for the routing being built: when the router escalates, it is designed to try the
first model it hasn't tested yet, starting from the cheapest tier, and to fall
back to the *last* model of the highest tier once everything has been tried.
(Escalation is not on the live path yet — it exists as an offline benchmark
strategy.) Reordering rows changes that intended behavior, so add new rows
deliberately rather than sorting the file.

### With a benchmark run

The benchmark scores models on cost, so it needs prices. Add an optional
`pricing:` block and the model becomes scoreable:

```yaml
models:
  gpt-oss-120b-groq:
    model_id: openai/gpt-oss-120b
    tier: cheap
    provider: groq
    version: gpt-oss-120b            # model identity — see "A model id is immutable"
    supports_streaming: true
    supports_cache_control: false
    pricing:
      input_cost_per_1m: 0.15
      output_cost_per_1m: 0.6
      cache_read_cost_per_1m: 0.075   # omit if the provider has no cache discount
      price_provider: groq
      price_source: https://groq.com/pricing
      price_as_of: "2026-07-17"
      price_note: Optional — anything a reader needs to trust the number above.
```

A model without `pricing:` is routable but invisible to the benchmark. That's on
purpose: a model can't be compared on a price nobody looked up. The provenance
fields exist for the same reason — `price_source` and `price_as_of` are what let
a future reader tell a checked price from a remembered one. Prices move; a number
without a date is a number you can't audit.

Watch for models with no cache-read discount. They resend the full context at
full price every turn, which shows up as a benchmark bill rather than an error.

Once a model is registered, score it with `python -m benchmark.runner.run_matrix`. The
default `--strategy cost_optimal` runs the cheap adaptive collection (frontier only where
tiers disagree, plus a random audit); `--strategy full` runs the exhaustive matrix. Both
are simulated unless you pass `--live`. See [benchmark.md](benchmark.md) for the details.

### A model id is immutable — new version, new id

`version` sits on the model row, next to `tier` and `provider` — not inside
`pricing:`. It records model *identity*, and that distinction is a rule worth
following:

- **A model id is a fixed behavior identity.** Every benchmark result is a fact
  about one `(task, model-at-that-version)` pair. When a provider ships genuinely
  new weights, give it a **new registry id** — `kimi-k3` becomes `kimi-k3-2026-09`,
  a new row — rather than editing the old one in place. Old results stay valid:
  they describe the old model, which still existed.
- **`version` is provenance plus an opt-in re-run switch.** The benchmark treats it
  as a staleness key: bump it only when you knowingly want to recompute every cell
  for that id under the same name. It is the deliberate escape hatch, not the
  default path — the default path for a real model change is a new id.
- **Price changes never invalidate data.** A stored result records the cost you were
  actually billed. Editing `input_cost_per_1m`, a cache price, or `price_as_of`
  re-scores current routing against the current price but leaves every stored
  result untouched — which is exactly why `version` is a model attribute and not a
  pricing one. Correcting a price is not a model change.

A priced model must declare a `version`; an unpriced one (the `examples/providers/`
fragments) may omit it, since nothing benchmarks it.

### Reasoning effort (optional)

Some models expose a reasoning/thinking effort knob, and each one exposes it
differently — a label (`reasoning_effort: high`), a boolean (`thinking: {type:
enabled}`), or a mix. Rather than invent a fake shared scale, a model lists its own
**arms**: each arm is a named effort level with the exact request params to send and
a `rank` (0 = least effort) that orders arms *within that model only*. Arms are not
comparable across models — one model's `high` is not another's.

```yaml
models:
  gpt-oss-120b-groq:
    model_id: openai/gpt-oss-120b
    tier: cheap
    provider: groq
    reasoning:
      default_arm: medium          # must match one arm id below; used when nothing else decides
      arms:
        - id: low
          rank: 0
          api: { reasoning_effort: low }     # merged verbatim into the request
        - id: medium
          rank: 1
          api: { reasoning_effort: medium }
        - id: high
          rank: 2
          api: { reasoning_effort: high }
```

A model with no `reasoning:` block runs at a single implicit `default` arm, exactly
as before — the field is optional and backward-compatible. The benchmark scores each
`(model, arm)` as its own cell, so `results.csv` keys on `(challenge, model,
reasoning)`; `default_arm` is the arm a new model routes to until real outcomes
accumulate. Effort is chosen once per task and held for the session — never switched
mid-conversation, which would break the provider's prompt cache.

## Tune the router

Which models shunt knows is one question; how it picks between them is another.
The routing policy ships at `src/shunt/config/router.yaml` inside the package —
the active strategy, its knobs, and the exploration settings. Override it the same
way as the registry: write your own `router.yaml` in `~/.config/shunt/`, or point
`SHUNT_CONFIG_DIR` at a directory holding one. Both files resolve from the same
place, so a single directory holds everything you edit.

### Choose which models are live-routable

The registry (`models.yaml`) defines every model shunt *knows*. `router.yaml`'s
`models:` list decides which of those the running router may actually pick:

```yaml
router:
  models:                 # live-routable models; each name must exist in the registry
    - deepseek-v4-flash
    - qwen3.7-plus
    - gpt-5-mini
    - kimi-k2.5
    - zai-glm-5.2
    - kimi-k3
```

Omit the key, or leave it empty, and every registry model is live-routable — the
backward-compatible default. A name in the list that isn't in the registry fails
loudly at startup, naming the offender, the same way an unregistered benchmark
model does (see [Choose which models the benchmark runs](#choose-which-models-the-benchmark-runs)).
That benchmark list is a separate setting — it decides what the offline benchmark
scores, not what the live proxy routes to.

If you supply your **own** `models.yaml` (a custom registry), keep the two in sync:
the packaged `router.yaml` names the shipped models, so against a different registry
its list won't match and startup fails. Ship a matching `router.yaml`, or set an
empty `models:` list to route over whatever registry is active.

To restrict live routing to a smaller set — say, the core cheap/mid models plus
one frontier model you trust — write your own `router.yaml` at
`$SHUNT_CONFIG_DIR/router.yaml` (or mount one into the container at that path).
Like the registry, this replaces the packaged file wholesale, not a per-key merge,
so restate every setting you care about:

```yaml
router:
  strategy: knn
  models:
    - qwen3.7-plus
    - deepseek-v4-flash
    - gpt-5-mini
    - kimi-k2.5
    - zai-glm-5.2
    - claude-opus-4-8   # the one frontier model this deployment allows
```

Because those overrides stack, the config actually in force is not always the file you
last edited. Shunt prints it at startup, so you never have to guess:

```
Shunt config | strategy=knn
Shunt config | knn: k=20 success_rate_threshold=0.60 min_samples=3
Shunt config | exploration: enabled=True budget_frac=0.15 conservative_alpha=0.10 ...
Shunt config | models: cheap:qwen3.7-plus, mid:gpt-5-mini, frontier:kimi-k3
Shunt config | session: inactivity_timeout=900s grace_period=120s retry_count=3
```

Only names and chosen values are printed — never a credential, and never the value of
an API-key variable.

Three settings are worth flipping without opening a file, and each has a flag and an
env var:

| What | Flag on `shunt start` | Environment variable |
|------|----------------------|----------------------|
| Active strategy | `--strategy knn` | `SHUNT_ROUTER_STRATEGY` |
| Exploration on/off | `--explore` / `--no-explore` | `SHUNT_EXPLORATION_ENABLED` |
| Exploration budget | `--explore-budget-frac 0.2` | `SHUNT_EXPLORE_BUDGET_FRAC` |
| Log verbosity | `--log-level debug` | `SHUNT_LOG_LEVEL` |

You don't need debug for the headline outcome: at the default `info` level, Shunt
logs one line per session the first time it routes — `Session <id> routed to
model=<name> reason=<source>` — so which model handled a session is always visible
without opening a file or flipping a flag.

`--log-level debug` traces the decision that produced it: which config file was
loaded, the cold-start counts, the neighbours the kNN query returned, and why each
candidate model passed or failed the success threshold. Third-party HTTP libraries
deliberately stay at INFO even then — their debug output includes `Authorization`
headers, and Shunt holds your provider keys.

They resolve in one order, most specific first:

**CLI flags → environment variables → `$SHUNT_CONFIG_DIR/router.yaml` → the packaged
`router.yaml`.**

A flag beats an env var, an env var beats your file, and your file replaces the
shipped one wholesale — it is not merged key by key, so copy the packaged file
before editing it.

Exploration ships on, but its effect is limited today. Outcomes can be recorded
manually via `shunt flag <session_id> good|bad`, or automatically once you configure a
capture work_dir (below); with neither, the outcome count typically stays near zero and
the exploration branch rarely fires. Exploration costs nothing extra today because it
rarely fires. Read the rest of this section as configured behaviour that grows as
verified outcomes accumulate.

### Record verified outcomes automatically

Exploration and the kNN neighbourhood only learn from *verified* outcomes. By default
those are recorded by hand (`shunt flag <session_id> good|bad`). To capture them
automatically, point Shunt at the repo it should test when a session goes idle:

```yaml
router:
  capture:
    work_dir: /path/to/your/repo        # single repo — the dogfooding default
    # work_dirs:                        # or several repos, keyed by tool identity:
    #   <tool_identity>: /path/to/repo-a
```

or set `SHUNT_WORK_DIR=/path/to/your/repo` (it overrides the file's single `work_dir`).
At session close Shunt re-runs the repo's test suite off the request path — pytest /
jest / `go test` / `cargo test`, auto-detected — and records the pass/fail as a verified
outcome. A session with no configured work_dir, or whose tests can't be detected or run,
is left unlabeled: Shunt never guesses an outcome, and the test path comes only from your
config, never from a request. Startup states which mode is in force (`Shunt capture is
ON` / `MANUAL-ONLY`).

A router that never tries a model it is unsure about never learns which ones it
can trust, so the shipped default spends a bounded slice of your budget probing
alternatives. `explore_budget_frac` is that bound: at the
default 0.4, the router holds exploratory spend to 40% of exploit spend, putting your
bill around **~1.4× what pure exploitation would cost**. Read that as a target rather
than a hard ceiling — the cap counts the router's own confidence-weighted
neighbourhood costs, not realized ones, so the *realized* ratio can overshoot the cap
substantially on an unlucky seed (measured up to 1.29 against a 0.4 cap in the offline
replay — roughly 3× the bound). In practice it usually
runs looser than the bound (replaying the shipped policy over the benchmark's
measured outcome matrix averages 1.10×, worst seed 1.22×; see
[benchmark](benchmark.md#evaluating-the-exploration-policy-without-spending-money)).

Two honest caveats. The cap is enforced against the **cost the provider reports**
for each call (`usage.cost` on OpenAI-compatible responses); a provider that does
not report a cost contributes nothing to either side of the ratio, so the cap
cannot bind on that traffic. Measured 2026-07-20: Requesty **does** report
`usage.cost`; DeepSeek's direct API **does not** (it returns token counts only) —
so traffic routed to `deepseek-v4-flash` through the direct provider is cost-blind.
Note the consequence for selection: a model whose cost is never reported reads as
`0.0`, i.e. *free*, to the cheapest-first rule. That happens to be harmless for
deepseek (it genuinely is the cheapest model), but it would mis-rank any pricier
model on a provider that omits the field. And exploration is not free on quality: in the same
offline replay, exploring cost **−2.8 pp pass rate** against exploration-off on
the paired per-task comparison (95% CI −6.5 to +0.3, n=43), measured with
exploration's learning benefit set to zero. To turn it off entirely:

```bash
shunt start --no-explore
```

or `exploration.enabled: false` in your `router.yaml` for a permanent setting.

### Prior seeding from offline model estimates

The exploration layer initializes Thompson priors from offline per-model success-rate
estimates, improving inference when outcomes are sparse. A model's prior is seeded with
its global confidence-weighted success rate from Tier-2 (verified) outcomes, with the
strength capped by `exploration.prior_strength_cap` (default 20.0 pseudo-observations).
This empirical-Bayes regularization means a model with historic evidence starts close to
its learned rate, while one without evidence falls back to the flat `Beta(1, 1)`. Adjust
the strength cap if you have strong offline data and want faster learning, or if you want
the priors to regularize more conservatively:

```yaml
router:
  exploration:
    prior_strength_cap: 20.0      # default; raise to trust offline estimates more
```

### Batch offline re-fit

The kNN index is rebuilt from the append-only outcome log periodically, not on every
outcome. This batch-first design trades real-time precision for robustness (HNSW cannot
delete in place; rebuild is cheaper than per-outcome updates at scale). By default, the
index rebuilds every 50 captured outcomes:

```yaml
router:
  refit:
    every_n_outcomes: 50          # 0 disables (only boot-time rebuild runs)
```

The index always rebuilds on startup. If you want no runtime re-fit (frozen index after
boot), set `every_n_outcomes: 0`; if you want tighter coupling, lower the threshold
(beware: frequent rebuilds are CPU-intensive). Monitor logs for `index rebuild` messages.

## Choose the embedding model (and stay swap-safe)

The embedder turns each task into the vector every kNN neighbour is measured against, so
it is the corpus's foundation. It has its own config file, `embedding.yaml`, resolved with
the same precedence as `router.yaml` (explicit path → `$SHUNT_CONFIG_DIR/embedding.yaml` →
the packaged default). Your file **replaces** the packaged one wholesale — it is not merged
key by key.

```yaml
embedding:
  active: jina-code            # a KEY into models below, not a raw repo
  max_chars: 4000              # part of the fingerprint (see below)
  models:
    jina-code: { repo: jinaai/jina-embeddings-v2-base-code, dim: 768, context_length: 8192 }
    arctic:    { repo: Snowflake/snowflake-arctic-embed-m-long, dim: 768, context_length: 2048 }
  cache_dir: null              # null → SHUNT_EMBED_CACHE_DIR / SHUNT_DATA_DIR resolution
```

`SHUNT_EMBEDDER_MODEL` still wins over the file and now selects a **key** (`jina-code`) —
or, for back-compat, any model's full `repo` string. An unresolvable value is a loud error
listing the valid keys, never a silent fallback. `SHUNT_EMBED_MAX_CHARS` overrides
`max_chars`, and `SHUNT_EMBED_CACHE_DIR` overrides `cache_dir`.

### Swap-safety: the fingerprint and `shunt reindex`

The active model's `repo`, its `dim`, and `max_chars` form the corpus **fingerprint** — the
tuple that fully determines the vector space. Two models can share a dimension yet produce
vectors in completely different geometries, so switching the model silently would leave the
stored corpus and every new query in disagreeing spaces, and the router would route on
garbage.

To prevent that, Shunt stores the fingerprint alongside the corpus and compares it at
startup:

- **Match (or a genuinely fresh database — no fingerprint *and* no embeddings):** the index
  is trusted and kNN routing serves as normal. A fresh corpus adopts the current config.
- **Legacy database (embeddings present but no stored fingerprint):** the vectors predate
  fingerprinting, so their space can't be proven to match the configured embedder. Shunt
  refuses kNN neighbours (cold-start) and logs one line asking you to `shunt reindex`, which
  re-embeds into the current space and stamps the fingerprint.
- **Mismatch:** the stored vectors are in a foreign space. Shunt still starts and stays
  healthy, but **refuses to serve kNN neighbours** — it routes every request via the
  cold-start / cheap default and logs one line telling you to reindex. It never
  auto-reindexes on boot (re-embedding the whole corpus is a heavy, surprising side effect).

When you deliberately change the embedding model or `max_chars`, re-embed the corpus into
the new space with the server **stopped**:

```bash
shunt reindex
```

This re-embeds every stored task, rebuilds the index atomically, and advances the
fingerprint last (as the commit marker). If it is interrupted, the old fingerprint remains,
so the next boot safely refuses neighbours and asks you to reindex again — it never serves a
half-migrated corpus. Restart the server afterward to pick up the new space.

> **Residual risk:** upstream **revision drift** is unguarded. If a model repo re-publishes
> different weights under the same name, the fingerprint still matches and the swap goes
> undetected. Pin the cache or re-benchmark if that matters for your deployment.

## The rest of the environment variables

Defaults that are fine to leave alone, but which you can override without editing
a file. Each is read once at startup.

**Where things live**

| Variable | Default | Effect |
|---|---|---|
| `SHUNT_CONFIG_DIR` | `~/.config/shunt` | Directory holding your `models.yaml`, `router.yaml`, and `embedding.yaml` overrides |
| `SHUNT_MODEL_CONFIG_PATH` | unset | Path to a single registry file, bypassing the config-directory lookup |
| `SHUNT_DATA_DIR` | `~/.local/share/shunt` | Directory for the outcomes database (`outcomes.db`) |
| `SHUNT_ENV_FILE` | `./.env` | The `.env` file loaded at startup; real environment variables still win |

**Where it listens**

| Variable | Default | Effect |
|---|---|---|
| `SHUNT_HOST` | `127.0.0.1` | Address the proxy binds to. It defaults to loopback because Shunt holds your provider keys and does not authenticate its callers — change it only behind a network you trust |
| `SHUNT_PORT` | `8080` | Port the proxy listens on |

**Sessions and upstream calls**

| Variable | Default | Effect |
|---|---|---|
| `SHUNT_SESSION_INACTIVITY_TIMEOUT` | `900` | Seconds before an idle open session is closed |
| `SHUNT_SESSION_GRACE_PERIOD` | `120` | Seconds reserved after a session closes for outcome verification; configured but not yet acted on |
| `SHUNT_RETRY_COUNT` | `3` | Attempts per upstream model before falling back, with exponential backoff |

**Routing and embeddings**

| Variable | Default | Effect |
|---|---|---|
| `SHUNT_COLD_START_THRESHOLD_TIER2` | `20` | Effective sample size (nₑ) of Tier-2 outcomes to leave cold start (either threshold ends it) |
| `SHUNT_COLD_START_THRESHOLD_TIER1` | `50` | Effective sample size (nₑ) of all labelled outcomes to leave cold start (either threshold ends it) |
| `SHUNT_EMBEDDER_MODEL` | `jina-code` | Active embedding model — a key (or `repo`) from `embedding.yaml`; overrides the file. See [Choose the embedding model](#choose-the-embedding-model-and-stay-swap-safe) |
| `SHUNT_EMBED_MAX_CHARS` | `4000` | Prompt characters fed to the embedder; overrides `embedding.yaml`'s `max_chars` |
| `SHUNT_EMBED_CACHE_DIR` | `$SHUNT_DATA_DIR/models` | Where the ~600MB embedding model is cached. Shunt downloads it once at startup and reuses it; keep this on durable storage or every restart re-downloads it |
| `SHUNT_RESPONSE_MODEL_LABEL` | unset | Prefix added to the response `model` field (e.g. `shunt:` → `shunt:qwen3.7-plus`), so a client shows which model actually served the turn |
| `SHUNT_LOG_LEVEL` | `info` | Log verbosity; `debug` traces the routing decision |

`SHUNT_EMBED_MAX_CHARS` bounds only the text the router embeds to make its
decision. The prompt itself is forwarded upstream untouched — truncation never
reaches the model.

## Choose which models the benchmark runs

The registry above defines every model shunt knows. The benchmark harness runs a
subset of them, chosen by the `models` list in `benchmark/benchmark.yaml` — a
separate list from `router.yaml`'s `models:` (see
[Choose which models are live-routable](#choose-which-models-are-live-routable)),
since what the benchmark scores and what the live proxy routes to are independent
decisions:

```yaml
models:                 # enabled models; each name must exist in the registry
  - deepseek-v4-flash
  - qwen3.7-plus
  - gpt-5-mini
  - kimi-k2.5
  - zai-glm-5.2
  - kimi-k3
```

The list decides enablement three ways:

- **In the list** — the model is enabled and runs.
- **In the registry but not the list** — disabled. It stays available (drop its
  name back in to turn it on), it just sits out the current runs. That is how
  `claude-opus-4-6` is priced for provenance yet excluded from the sweep.
- **In the list but not the registry** — a hard error at config load, naming the
  offender. A model you run must exist, so a typo fails loudly instead of silently
  routing to nothing.

Enabled models are always scored cheapest-tier-first (cheap → mid → high →
frontier), so list order is for readability only — it does not affect results.

```bash
shunt start
```

A misspelled field, a missing required one, or a model naming a provider that
isn't in the `providers:` table fails at startup and names the offender. A wrong
`base_url` or a bad key can't be caught that way — those surface on the first
request.

To check a `base_url` and key ahead of that first request, the repo ships a
provider probe (developer tool, not part of the installed package):

```bash
# Wiring check — no key needed. Sends a deliberately bogus key and confirms the
# provider rejects it the way it should (proves base_url + auth are wired right).
python tools/provider_probe.py

# Credential check — needs a real key in the provider's env var. Confirms the
# key is ACCEPTED (200). Providers with no key set are skipped, not failed.
DEEPSEEK_API_KEY=sk-... python tools/provider_probe.py --authenticated
```

Both checks are free. The keyless check fails before billing; the authenticated
check only ever does a GET (a model listing, or a key-info endpoint) — never a
completion — so it cannot cost anything. A provider with no free authenticated
endpoint (currently Requesty, whose model list is public) is skipped rather than
billed. CI runs the keyless check on every push and the authenticated check on a
secrets-gated schedule — add a provider's key as a repo secret of the same name
and it starts being checked; see `tools/provider_auth_signatures.yaml`.
