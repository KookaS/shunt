# Strategy templates — the two routing strategies you can run

Shunt offers **two** routing strategies. They climb the *same* escalation ladder and
differ in exactly one thing: which rung the first session of a task opens on.

| File | `router.strategy` | Opens on | Embeds? |
|---|---|---|---|
| [`session-cascade.yaml`](session-cascade.yaml) | `session_cascade` (**default**) | the cheapest healthy model | no |
| [`knn-semantic-cascade.yaml`](knn-semantic-cascade.yaml) | `knn_semantic_cascade` (**opt-in**) | the model the kNN pick chooses | yes — a local ONNX model, in the request path |

Each file is a complete, runnable `router.yaml`. Nothing in it is a secret: provider
credentials are read from environment variables named by the registry, never from this
file. See [`../providers/README.md`](../providers/README.md) for the key per provider.

## Try one

```bash
mkdir -p ~/.config/shunt
cp examples/strategies/session-cascade.yaml ~/.config/shunt/router.yaml

export DEEPSEEK_API_KEY=...      # whichever keys your registry's models need
cd ~/code/my-project             # so the ladder has a repo whose tests it can verify
shunt start
```

Then point your tool at `http://127.0.0.1:8080` — see
[`../integrations/README.md`](../integrations/README.md). Shunt prints the config it
actually resolved at boot (`Shunt config | strategy=…`), and every response carries an
`X-Shunt-Decision` header naming the model and the reason.

To switch without editing a file: `shunt start --strategy knn_semantic_cascade`, or set
`SHUNT_ROUTER_STRATEGY`. A flag beats an environment variable, which beats your file,
which **replaces** the packaged one wholesale — it is not merged key by key, which is
why these templates spell out every block that matters.

## Which one

**`session-cascade.yaml` — the default. Pick it when you want the cheapest bill.**
On the offline corpus it resolves the same 178 of 184 scored tasks as the routing <!-- frozen-value: n=184, date=2026-08-11, run=49b8362 -->
model, at 96.74%, for **$33.56** of naive corpus cost against **$43.28** — the <!-- frozen-value: n=184, date=2026-08-11, run=49b8362 -->
classifier costs about 29% more for no measured quality gain. It never embeds, so it
carries no extra dependency and no extra failure mode.

**`knn-semantic-cascade.yaml` — the opt-in. Pick it when you want a more predictable
bill, or when you have reason to believe model choice changes outcomes on your
workload.** What it measurably buys is *variance*, not quality: cost CV **1.894**
against the default's **2.416**, and a slightly shorter ladder (**2.076** sessions per <!-- frozen-value: n=184, date=2026-08-11, run=49b8362 -->
task against **2.217**). But the p95 session tail is **identical — 7 against 7**: the <!-- frozen-value: n=184, date=2026-08-11, run=49b8362 -->
classifier smooths the middle of the distribution, not the tail. It also loads an ONNX
embedding model (~600MB) in the request path, which is memory, startup time, and a way
for routing to fail that the default does not have.

**The honest limit on all of the above.** This corpus cannot discriminate *quality*:
every escalating strategy measured on it — including a hindsight Oracle that cannot be
deployed — ties at the same pass rate. So these numbers say which strategy is cheaper
and which is steadier, and they say nothing about whether the classifier helps on a
workload where the model choice changes the outcome. That question is open.

Full numbers, intervals and method: [`../../docs/results.md`](../../docs/results.md).
Every knob in these files: [`../../docs/configuration.md`](../../docs/configuration.md).

## What is deliberately not here

`always_cheap` and `always_frontier` are also accepted by `router.strategy`, and no
template ships for them. They pin one model and never climb — they are the baselines a
routing comparison is read against, not operating points you would run a router for.

## Tested

`tests/test_strategy_examples_sync.py` keeps these files in step with the shipped
defaults and the live-strategy allowlist. `tests/integrations/test_strategy_handshake.py`
boots Shunt on each template and drives a real request through it, over both wire
formats, against the same hermetic fake upstream the tool handshakes use — proving the
template loads, the strategy named is the strategy that decides, and the default embeds
nothing while the opt-in does. No provider is called and nothing is billed; a green run
proves the wiring, not model quality.
