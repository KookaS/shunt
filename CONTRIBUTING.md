# Contributing to Shunt

Thanks for your interest. Shunt is **Apache-2.0** — free for everyone, always.

## Before you write code — confirm scope

For anything non-trivial, **open an issue or discussion first** to confirm the change
fits before you invest in a PR. This protects your time and the reviewer's. Bug fixes,
docs, and tests don't need this — just send them.

## What belongs in Shunt

Four criteria decide whether a proposal fits. Read them as scope rather than slogans:
a change that breaks one is out, however well it scores.

- **Data driven.** The label is a verified outcome from our own run: did your
  tests pass? Not a model's self-reported confidence, and not someone's guess
  about what "hard" looks like.
- **Lightweight.** It runs on any laptop. A router that needs a big machine to
  save you money defeats its own purpose.
- **Divide and conquer.** Routing and escalation are separate supervised
  problems with separate datasets, so each can be replaced and beaten on its own.
- **Open.** A tool that runs on your laptop holding your provider keys has to be
  inspectable to be trustworthy. You own the model pool, the keys and the
  learning data. Nothing phones home.

Routing is a statistics problem before it is a systems problem. Making the pipeline
faster buys nothing until the decision rule is right, so we take that rule from wherever
it comes — deterministic, statistical or learned — and judge it only on our own data.

## Contributor sign-off — DCO, not a CLA

We use the **Developer Certificate of Origin (DCO)**, *not* a CLA. You keep the copyright
to your contribution; you just certify (via a sign-off) that you have the right to submit
it under the project's license.

Sign off every commit by adding a line (git does this with `-s`):

```
git commit -s -m "your message"
```

which appends:

```
Signed-off-by: Your Name <your.email@example.com>
```

By signing off you agree to the DCO (https://developercertificate.org/).

## Standards

- Match the surrounding code; style is enforced by `ruff` (config in `pyproject.toml`).
- Add/adjust tests for behavior changes; keep the project **zero-telemetry** and
  **secure-by-default** (localhost-bind, never log keys).

## Non-code contributions are first-class

Docs, examples, recipes, benchmarks, triage, and bug reports are as valued as code — and
are the easiest way to start.

## Community

Be excellent to each other — see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
