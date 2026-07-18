# Shunt

Your coding agent pays frontier prices for every request — even the routine ones a
small open-weight model would nail. Shunt is a local router that sends the easy
majority to a cheap model and the hard tail to a frontier one. It's built to learn
that line from your own passing tests, and never breaks your prompt cache doing it.
One environment variable, and nothing else changes. (Pre-alpha — see status below.)

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Status](https://img.shields.io/badge/status-pre--alpha-orange)
![Telemetry](https://img.shields.io/badge/telemetry-none-brightgreen)
![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)

<!-- TODO: hero image — a simple diagram of "coding agent → Shunt → {cheap model | frontier model}". Add before launch. -->

<!-- TODO: demo — a short asciinema cast showing Shunt routing a real request and printing the X-Shunt-Decision reason. Add before launch. -->

## Why this project

Existing routers make you choose: cloud-only with a take-rate (OpenRouter),
licensed so enterprises can't touch them (NadirClaw), proxy-only with no real
routing (Portkey, Kong), or a research artifact never built to ship (ACRouter).
None are cache-safe, outcome-grounded, tool-agnostic, self-hosted, and Apache 2.0
at once. Shunt is all five.

The hard part is the *decision*, not the plumbing — which model handles which
task. Shunt learns that from your own verified outcomes, and everything below is
in service of getting it right.

**What makes it different:**

- **ML-powered, not hand-coded heuristics.** At the core is a k-nearest-neighbours
  model over task embeddings — it learns which model works for which task from
  past verified outcomes. No brittle keyword lists, no hand-authored utterance
  patterns, no magic-number thresholds. When kNN is uncertain, a cascade
  mechanism tries a cheap model first, verifies the result (does it typecheck?
  do the tests pass?), and escalates to a smarter model if the cheap answer
  fails. The opposite of fusion or ensembling — those call multiple models and
  cost more, which defeats the purpose. Shunt calls *one* model per request
  and only escalates on verified need.
- **Cache-safe by design.** Switching models mid-session re-reads the whole
  history at full price — that alone can wipe out the savings. Shunt routes at
  task and session boundaries and never swaps the model out from under a cached
  conversation.
- **Outcome-grounded, not guess-grounded.** Every decision is checked afterward
  against a real signal — does the diff apply, does it typecheck, do the tests
  pass. That result feeds the kNN index for the next decision.
- **Local-first, zero telemetry, Apache 2.0.** Your data stays yours. You own
  the model pool, the decision method, the API keys, and the learning data.
  No phone-home, no take-rate, no CLA — a DCO sign-off is all we ask.

**Measured, not marketed.** The decision is the whole product, so Shunt ships a
benchmark that scores it. Six models from cheap to frontier — over 50× apart on
output price — are scored against the 500-task SWE-bench Verified suite, each task
judged by its own tests (partial live coverage so far, and growing). Every routing
strategy is scored offline on reward (quality minus cost), with bootstrap
confidence intervals and a Pareto check against a perfect-oracle baseline. No
cherry-picked demo.

And we name the bar we have to clear: beat fixed-frontier-with-caching at equal
quality on a real workflow, or don't ship. Published evidence puts single-turn
code-gen savings at 15–30%; the one study on *agentic* Claude Code found no
benefit, so we measure our own workflow before quoting you a number. Zero
telemetry, and we publish the real result — honesty is the feature.

## Works with the tools you already use

One line, and your agent talks to Shunt instead of the model API. Shunt speaks both
wire formats and translates between them, so the same router sits in front of
either.

**Claude Code** — and any Anthropic-wire client:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
```

**opencode, aider, Continue** — and any OpenAI-compatible client:

```
base_url = http://127.0.0.1:8080/v1
```

That one change covers the whole ecosystem: coding agents (Claude Code, opencode,
aider, Continue, Cline, Zed), agent frameworks (LangChain, Pydantic AI, LiteLLM —
and LlamaIndex, CrewAI, or AutoGen with a one-line adapter), no-code builders (n8n,
Flowise), and any raw SDK or `curl`. Clients that auto-discover models hit a
`/v1/models` endpoint and just work.

Copy-paste config for each tool — plus a dry-run handshake that proves the wiring
end-to-end without spending a cent — lives in
[`examples/integrations/`](examples/integrations/README.md). The verified
integrations gate CI; the third-party-CLI legs run best-effort. Either way, the
examples that matter don't rot.

## How it decides

The routing is the hard part and the whole point. The multi-provider plumbing is
becoming free; the value is in getting the *decision* right.

> **Pre-alpha status.** The pipeline below is implemented and validated *offline*
> against the benchmark. In the live proxy today, routing forwards to a cheap
> default; the kNN decision path is **not yet** wired into the request path —
> [`docs/architecture.md`](docs/architecture.md) tracks exactly what runs live.

Shunt's decision pipeline has two layers:

**Primary: kNN over task embeddings (ML, not rules).** Every task is embedded
into a vector using a local ONNX model (CPU, ~10ms). A k-nearest-neighbours
lookup finds past tasks with verified outcomes that resemble the current one.
If the cheap model in those neighbours has a strong success rate on similar
work, it gets picked. No hand-authored keyword lists, no brittle heuristics,
no regular expressions — the model learns the boundary from your actual
outcomes.

**Fallback: cascade on uncertainty.** When kNN has too few neighbours or the
success rate is borderline, Shunt falls back to a cascade: try the cheap model,
verify the result (does the diff apply? does it typecheck? do the tests pass?),
and escalate to a smarter model only if verification fails. This is the
opposite of fusion or ensembling — those call multiple models simultaneously
and cost 4–5× more, which works against the goal of cutting your bill. Shunt
calls *one* model per request and pays for a second only when the first one
can't do the job.

**No heuristic rule-based routing.** Other projects use complexity scoring
(keyword counts, substring matches, `len//4` magic numbers) or hand-authored
utterance patterns. These don't generalise across tasks, languages, or user
workflows. Shunt uses neither.

**The loop closes on every request.** After routing, an async verifier checks
the outcome and writes it to a local store. The kNN index reads from that store
on the next request. Over time, the router gets better at distinguishing
routine work from the hard tail.

**Cache-safe.** Switching models mid-session re-reads the whole conversation at
full price — that alone can wipe out the savings. Shunt routes at task and
session boundaries and never swaps the model out from under a cached
conversation. Every decision returns an `X-Shunt-Decision` header so you know
which model was chosen and why.

**Secure because it holds your keys.** Localhost-bind by default, no exposed
control plane, keys kept out of logs, dependencies pinned and locked — the
posture a credential-handling tool in the request path has to be built to.

## Bring your own keys

Shunt routes through your own provider accounts, so you keep full control of spend
and nothing is replayed or resold. Set one environment variable per provider, add a
model with a few lines of YAML, and you're done.
[`docs/configuration.md`](docs/configuration.md) walks through both;
[`examples/providers/`](examples/providers/README.md) ships a ready-to-copy config
for each supported provider.

## Roadmap

Where Shunt is headed, in order:

1. **The core router.** Embedding + k-NN routing, cache-aware task-level decisions,
   and typecheck/test verifiers, dogfooded on a real coding workflow.
2. **The learning loop.** Outcome logging, per-key spend caps, graceful handling of
   models added to or pulled from the pool, and a streaming benchmark.
3. **Reach and control.** Mid-session escalation with an upfront cost quote, a
   pluggable-policy extension API, and bring-your-own eval metric.

Further out: a plugin ecosystem for third-party policies and verifiers, more
providers on demand, and a faster runtime if concurrency ever calls for it.

## Repository layout

```
├── src/shunt/             Core router engine
│   ├── cli.py             CLI entry point (shunt start, explain, version; flag planned)
│   ├── proxy/             HTTP server: /health, /v1/chat/completions, /v1/messages, /v1/models
│   ├── router/            Decision core: embed → nearest-neighbour → selection rule
│   ├── verifiers/         Async outcome backfill (auto-detected tests, typecheck)
│   ├── db/                SQLite persistence for sessions, outcomes, index
│   ├── session/           Session lifecycle, inactivity timeout, model lock
│   └── models/            Provider config, capability tiers, fallback chain
├── benchmark/             Model-capability and routing evaluation
├── docs/                  User documentation (MkDocs)
├── examples/providers/    Copy-paste registry config, one file per provider
├── examples/integrations/ Tool integration examples (CLI agents, frameworks, gateways)
└── tests/                 Test suite
```

Distribution: `shunt-router` on PyPI (import as `shunt`); `ghcr.io/kookas/shunt-router` on Docker.

## Contributing

Shunt is a one-person project in the open, and early is the best time to shape it.

- ⭐ **Star the repo** if you want to follow where it goes.
- 💬 **Open a discussion or issue** with your workflow, your cost pain, or an idea.
- 📝 **Docs and typo fixes** make a low-friction first pull request. Contributions
  sign off under the [DCO](CONTRIBUTING.md); there's no CLA.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how changes get merged.

## Commercial support

Shunt's router core is Apache-2.0, free for everyone including companies, and it
stays that way. If your organization later needs priority support, custom
integration, or governance features built around the free core, that will be a
separate offering — never a gate on the core routing itself. If that's ever you,
open an issue to start the conversation.

## License

**[Apache-2.0](LICENSE)** — free for everyone, with a patent grant.

Security disclosures: [SECURITY.md](SECURITY.md) ·
Community standards: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
