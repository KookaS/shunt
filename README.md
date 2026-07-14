# Shunt

**A local router that puts a cheaper model on the routine work and keeps the expensive one for the hard tail — built to cut your coding-agent bill without you babysitting the model picker.**

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Status](https://img.shields.io/badge/status-pre--alpha-orange)
![Telemetry](https://img.shields.io/badge/telemetry-none-brightgreen)
![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)

<!-- TODO: hero image — a simple diagram of "coding agent → Shunt → {cheap model | frontier model}". Add before launch. -->

<!-- TODO: demo — a short asciinema cast showing Shunt routing a real request and printing the X-Shunt-Decision reason. Add before launch. -->

Coding agents bill you frontier-model prices on every request, even the ones a
small open-weight model would answer just as well. Most of the work — renaming a
variable, writing a test, fixing an obvious type error — is routine. The expensive
model earns its keep on the hard tail.

Shunt is a proxy you drop in front of the agent. It reads each task, sends the
routine majority to a cheap model and the rest to a frontier one, and learns where
that line falls from your own passing tests and typechecks — not from a guess. You
point one environment variable at it, and nothing else in your setup changes.

You own the whole thing: the model pool, the decision method, your API keys, and
the data it learns from. It runs on your machine, binds to localhost, keeps no
telemetry, and is Apache-2.0.

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

- **It learns from outcomes, not guesses.** Every task Shunt routes gets checked
  afterward against an objective signal — does the diff apply, does it typecheck,
  do the touched tests pass. That result is written down next to the task, and the
  next similar task reads it. The check runs off the hot path and never delays your
  response.
- **It's cache-safe by design.** Switching models mid-session throws away the
  prompt cache and re-reads the whole history at full price — that alone can wipe
  out the savings. Shunt routes at task and session boundaries and never swaps the
  model out from under a cached conversation. When escalating to a stronger model
  is worth it, the recompute cost (roughly 4× the context) is part of the decision,
  not a surprise on your bill.
- **The policy is yours to inspect and swap.** Rules first, because they're cheap
  and predictable. A k-nearest-neighbours lookup over task embeddings by default.
  A logistic-regression or an LLM-as-judge tier if you want them — the judge is
  opt-in and never on by default. Every decision comes back with an
  `X-Shunt-Decision` header that tells you which model was chosen and why.
- **Secure because it holds your keys.** Localhost-bind by default, no exposed
  control plane, keys kept out of logs, dependencies pinned and locked — the
  posture a credential-handling tool in the request path has to be built to.

## Saving money

Shunt cuts cost by routing the verified-easy work to a cheaper model and keeping
the frontier model for the hard tail — measured *at equal quality*, gated on tests
and typechecks, so a cheaper answer that breaks the build gets escalated instead of
shipped.

Bring your own keys. Shunt routes through your own provider accounts, so you keep
full control of spend and nothing is replayed or resold.

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
│   ├── cli.py             CLI entry point (shunt start, explain, flag, version)
│   ├── proxy/             HTTP server: /v1/chat/completions, /v1/messages, admin API
│   ├── router/            Decision core: embed → nearest-neighbour → selection rule
│   ├── verifiers/         Async outcome backfill (auto-detected tests, typecheck)
│   ├── db/                SQLite persistence for sessions, outcomes, index
│   ├── session/           Session lifecycle, inactivity timeout, model lock
│   └── models/            Provider config, capability tiers, fallback chain
├── benchmark/             Model-capability and routing evaluation
├── docs/                  User documentation (MkDocs)
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
