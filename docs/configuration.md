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
`src/shunt/models/default_config.yaml` for the two shipped providers, or in that
provider's `examples/providers/<name>.yaml` fragment for the rest. `OPENAI_API_KEY`
for OpenAI, `GROQ_API_KEY` for Groq, and so on. Two of the providers are
aggregators — Requesty and OpenRouter — where one key reaches many vendors. Local
models (Ollama, vLLM) need no key at all.

## Add a model

The registry lives at `src/shunt/models/default_config.yaml` inside the package.
To change it, write your own at `~/.config/shunt/models.yaml`, or point
`SHUNT_CONFIG_DIR` somewhere else.

**Your file replaces the packaged registry. It is not merged with it.** If your
config lists one model, shunt knows one model. To keep the shipped models and add
your own, start from a copy of the packaged file.

### Without a benchmark run

Three fields make a model registerable — `model_id`, `tier`, and `provider`. A
model row is picked up the moment it exists; `tier` is the prior the routing is
designed to start from before real outcomes accumulate. (Pre-alpha note: the live
proxy forwards to a single cheap default today and does not yet choose between
registered models — see [architecture.md](architecture.md). Registering models
now sets up the pool the router will use once routing is wired in, and makes them
scoreable in the offline benchmark.) The two `supports_*` fields below are
optional; they default to streaming on, cache control off.

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

## Choose which models the benchmark runs

The registry above defines every model shunt knows. The benchmark harness runs a
subset of them, chosen by the `models` list in `benchmark/config.yaml`:

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
