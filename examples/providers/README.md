# Provider examples

One file per provider. Each is a working registry fragment — the same schema
`~/.config/shunt/models.yaml` uses — so you can copy it, set one environment
variable, and route.

## Use one

```bash
cp examples/providers/groq.yaml ~/.config/shunt/models.yaml
export GROQ_API_KEY=...
shunt start
```

That replaces your registry with a single provider and a single model. **A config
file at `~/.config/shunt/models.yaml` replaces the packaged registry — it is not
merged into it.** To keep the shipped models and add one of these, copy the
fragment's `providers:` row and `models:` row into your existing file rather than
overwriting it. Full walkthrough: [`docs/configuration.md`](../../docs/configuration.md).

Every model here omits `pricing:`, which makes it routable but invisible to the
benchmark — a model can never be scored against a price nobody measured. Add a
`pricing:` block when you want it benchmarked.

## What's here

| File | Key | Notes |
|---|---|---|
| `openai.yaml` | `OPENAI_API_KEY` | |
| `groq.yaml` | `GROQ_API_KEY` | |
| `mistral.yaml` | `MISTRAL_API_KEY` | |
| `together.yaml` | `TOGETHER_API_KEY` | litellm prefix is `together_ai` |
| `cerebras.yaml` | `CEREBRAS_API_KEY` | |
| `nebius.yaml` | `NEBIUS_API_KEY` | |
| `openrouter.yaml` | `OPENROUTER_API_KEY` | aggregator |
| `xai.yaml` | `XAI_API_KEY` | probes `/v1/models`, expects 400 |
| `fireworks.yaml` | `FIREWORKS_API_KEY` | probes `/v1/models`; litellm prefix is `fireworks_ai` |
| `requesty.yaml` | `REQUESTY_API_KEY` | aggregator; rejects with 403 |
| `deepseek.yaml` | `DEEPSEEK_API_KEY` | |
| `local.yaml` | — | Ollama / vLLM; no key, no probe |

## The `shunt-ci:` marker (wired)

Line 1 of every file is either:

```yaml
# shunt-ci: probe   # eligible for the live auth-probe
# shunt-ci: skip    # not eligible
```

It declares whether a file is eligible for the live auth-probe, and the probe
**reads it**: `tools/provider_probe.py` globs `examples/providers/*.yaml`, keeps
the `# shunt-ci: probe` files, and probes each. Adding a file opts it in; changing
one word opts it out; both show up in the same diff — no side-car manifest to keep
in sync (a second list is a second source of truth, stale the first time someone
adds an example).

It's a comment rather than a YAML key because the registry schema sets
`extra="forbid"` — an `x-ci:` key would make the example fail to parse, which
would defeat the point of shipping examples that parse.

The live probe is **opt-in and never PR-blocking** (the `provider-probe.yml`
workflow, run on demand or weekly). It sends a deliberately bogus key and asserts
the provider rejects it in the auth-shaped way its measured signature declares.
That proves the `base_url` and the auth wiring are right; it does not prove a real
completion works, and it costs nothing (auth fails before billing). Providers
change their rejection codes without warning, so a failure there is news about the
provider — not a reason to redden an unrelated pull request. `local.yaml` is
`skip`: a local server has no auth to reject.

A second workflow, `provider-auth-check.yml`, does the complementary **positive**
check: with a REAL key (from a repo secret named like the provider's
`api_key_env_var`) it proves the provider ACCEPTS the key (200). It is
secrets-gated (never runs on fork PRs), secret-optional (a provider with no key
set is skipped with a warning, not failed), and **always free** — a GET of a model
listing or key-info endpoint, never a billed completion. Add keys gradually; each
one starts being checked as soon as its secret exists.

## Where the pieces live

These fragments carry only **connection facts** (`base_url`, `api_key_env_var`,
`litellm_prefix`) and a sample model — everything a user needs to copy and route.
Two other files complete the picture, each the single owner of its part:

- `src/shunt/config/models.yaml` — the runtime registry, listing only the
  providers a shipped model routes to (Requesty, DeepSeek). The wider catalog here
  is never loaded by the router.
- `tools/provider_auth_signatures.yaml` — the measured auth-rejection signatures
  (status/body/endpoint) the live probe checks against. Perishable validation
  metadata, not user config, so it does not live in these fragments. A test asserts
  every `# shunt-ci: probe` file has exactly one signature and vice versa.
