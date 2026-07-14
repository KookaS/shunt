# Shunt

A tool-agnostic, cache-safe LLM router. Shunt decides which model handles a
request based on the task, learns from verified outcomes (tests / typecheck /
your own eval metric), and is cache-safe by design — it routes at
task/session boundaries, never silently mid-cached-turn.

Cheap / open-weight models handle the ~70–80% of routine work; frontier models
are reserved for the hard tail. You configure both the model pool **and** the
decision method.

> Status: **pre-alpha.** The core hypothesis — does cheap-first routing beat always-frontier at equal quality? — has not been validated yet. If it doesn't, the project stops.

## Why it's different

No shipped OSS project is simultaneously pluggable-by-policy **+** outcome-grounded
**+** tool-agnostic **+** embeddable **+** cache-aware. Each incumbent has at most
2–3 of those legs. The hard, valuable part is the **decision** (which task needs
the smart model), not the multi-provider plumbing (which is commoditizing to free).

## Design center

- **Cache-boundary-aware routing** — Shunt controls `cache_control` placement and
  never switches models mid-session. It does not observe live server-side cache
  occupancy; post-hoc `usage.cache_read_input_tokens` measures the switch-tax,
  not the decision.
- **Pluggable, inspectable policy** — rule-first (cheap, trustworthy), kNN default,
  logreg / LLM-judge optional. Every decision emits an `X-Shunt-Decision` header.
  LLM-judge is opt-in, never default.
- **OpenAI ↔ Anthropic translation** — Anthropic + OpenAI first, not 100+ providers.
- **Verifier + memory loop** — log `(task → model → verified outcome)` and learn
  from it. Verification is async, backfill only — never on the hot path. As a
  `base_url` proxy the signals are wire-visible only: repeat/rephrase, next-turn
  errors, session end. Objective verification, not prediction.
- **Secure by default** — localhost-bind, no exposed control plane, no key logging.
  Apache-2.0, zero telemetry.

## Status

**Pre-alpha.** The hypothesis (cheap-first routing beats always-frontier at equal quality) is unproven. Implementation follows validation.

Integration:

- **Claude Code:** `ANTHROPIC_BASE_URL=http://127.0.0.1:<port>`
- **OpenAI-compatible clients:** `base_url = http://127.0.0.1:<port>/v1`

Distribution: `shunt-router` on PyPI (`shunt` is taken); import as `shunt`.
Docker: `ghcr.io/kookas/shunt-router`.

## Repo layout

```
├── src/shunt/             Core router engine
│   ├── cli.py             CLI entry point (shunt start, explain, flag, version)
│   ├── proxy/             HTTP server: /v1/chat/completions, /v1/messages, admin API
│   ├── router/            Decision core: fastembed → hnswlib → selection rule
│   ├── verifiers/         Async outcome backfill (output mining, auto-detected tests)
│   ├── db/                SQLite persistence for sessions, outcomes, HNSW index
│   ├── session/           Session lifecycle, inactivity timeout, model lock
│   └── models/            Provider config, capability tiers, fallback chain
├── benchmark/             Model-capability and routing evaluation
│   └── routing/           Strategy evaluation: kNN, cascade, bandit, baselines
├── docs/                  Documentation (MkDocs → readthedocs)
├── tests/                 Test suite
├── .github/workflows/     CI + release workflows (PyPI + Docker)
├── Dockerfile             Production Docker image (multi-stage python build)
├── docker-compose.yml     Quickstart: proxy + SQLite with one command
├── mkdocs.yaml            MkDocs configuration
└── .readthedocs.yaml      Read the Docs build config
```

## Supporting the project

Shunt is a one-person project. If this is useful or interesting and you would like to support development, you can:

- ⭐ **Star the repo and share it.**
- 🐛 **Open a GitHub issue** with benchmark numbers from your setup, ideas, or questions.
- 📬 **Email me directly** if you are a business interested in sponsoring development.

## License

Shunt is **[Apache-2.0](LICENSE)** — free for everyone. See [CONTRIBUTING.md](CONTRIBUTING.md).

Security disclosures: [SECURITY.md](SECURITY.md).

Community: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
