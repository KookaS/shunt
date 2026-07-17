# Shunt

Tired of paying for multiple frontier-model subscriptions and watching your API
bill climb every month? So are we all.

Shunt is a smart, adaptable router that finds the cheapest model that can
actually handle your task and it learns from your own experience. Plug it in with one environment variable and let it do the rest.

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Status](https://img.shields.io/badge/status-pre--alpha-orange)
![Telemetry](https://img.shields.io/badge/telemetry-none-brightgreen)
![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)

<!-- TODO: hero image — a simple diagram of "coding agent → Shunt → {cheap model | frontier model}". Add before launch. -->

<!-- TODO: demo — a short asciinema cast showing Shunt routing a real request and printing the X-Shunt-Decision reason. Add before launch. -->

## Why this project

Coding agents bill you frontier-model prices on every request, even the routine
ones a small open-weight model would answer just as well. Existing solutions are
either cloud-only with a take-rate (OpenRouter), licensed so enterprises cannot
touch them (NadirClaw), proxy-only with no real routing (Portkey, Kong), or
research artifacts not built to ship (ACRouter). None are simultaneously
cache-safe, outcome-grounded, tool-agnostic, self-hosted, and Apache 2.0.

Shunt is a proxy you drop in front of the agent. It reads each task, sends the
routine majority to a cheap model and the rest to a frontier one, and learns
where that line falls from your own passing tests and typechecks — not from a
guess. You point one environment variable at it, and nothing else changes.

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

**On cutting your bill.** Shunt reduces cost by routing verified-easy work to
cheaper models. The published evidence shows 15–30% savings for single-turn
code-gen tasks, but the one study isolating *agentic* Claude Code found no
routing benefit there. Whether Shunt saves you money on agentic coding depends
on your workflow and requires broader testing — the initial dogfood experiment
exists to measure this honestly. We will publish the real number before asking
you to adopt it. If agentic coding nulls, Shunt's primary wedge shifts to
high-volume, stateless, and single-turn workloads where the evidence is positive.

## Drop-in integration

The integration is a one-line change.

**Claude Code**

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:<port>
```

**opencode, aider, Cursor, and other OpenAI-compatible clients**

```
base_url = http://127.0.0.1:<port>/v1
```

Your agent talks to Shunt exactly as it talked to the model API. Shunt speaks both
the Anthropic and OpenAI wire formats and translates between them, so the same
router sits in front of either.

## How it decides

The routing is the hard part and the whole point. The multi-provider plumbing is
becoming free; the value is in getting the *decision* right.

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

## Saving money

Shunt cuts cost by routing the verified-easy work to a cheaper model and keeping
the frontier model for the hard tail — measured *at equal quality*, gated on tests
and typechecks, so a cheaper answer that breaks the build gets escalated instead of
shipped.

Bring your own keys. Shunt routes through your own provider accounts, so you keep
full control of spend and nothing is replayed or resold. Set one environment
variable per provider you use, and add models with a few lines of YAML —
[`docs/configuration.md`](docs/configuration.md) walks through both, and
[`examples/providers/`](examples/providers/README.md) has a ready-to-copy config
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
│   ├── proxy/             HTTP server: /v1/chat/completions, /v1/messages, admin API
│   ├── router/            Decision core: embed → nearest-neighbour → selection rule
│   ├── verifiers/         Async outcome backfill (auto-detected tests, typecheck)
│   ├── db/                SQLite persistence for sessions, outcomes, index
│   ├── session/           Session lifecycle, inactivity timeout, model lock
│   └── models/            Provider config, capability tiers, fallback chain
├── benchmark/             Model-capability and routing evaluation
├── docs/                  User documentation (MkDocs)
├── examples/providers/    Copy-paste registry config, one file per provider
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
