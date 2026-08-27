# Probe trajectories — committed, but outside the analysed corpus

One-off probe collections live here rather than in `../live/`, because `../live/` is not
just an archive: it is the **population** every escalation statistic is computed over
(`benchmark.escalation.corpus.census` counts the `*.jsonl` files it holds, and
`run_eval` / `metrics.json` / every escalation figure are derived from that count).

Dropping a probe run into `../live/` therefore silently re-bases every published
escalation number — corpus size, stamped share, per-model tables, join rates and the
figure captions that quote them.

## What is here

- **`zai-glm-5.3-flash`, 41 trajectories** (2026-08-25 → 2026-08-26). The revealed identity
  of the retired `stealth/ox-alpha`, collected during OpenRouter's $0 free window. The model
  is not in `benchmark.yaml`'s enabled set and is not routed by `router.yaml`, so it is not
  part of the six-model escalation study; folding 41 partial-coverage runs into it would have
  restated ~20 published measurement claims for a model the product never uses.

These runs are **real measured data and cannot be regenerated offline** — the capture happens
during a live agent run (`benchmark.runner.infer._capture_escalation_trajectory`) and the free
window has closed, so re-collecting them would cost money. They are committed and verifiable:
`manifest.json` here carries the same `content_sha256` ledger as the live corpus, and
`benchmark.escalation.authenticity.verify_manifest` validates this directory independently.

The corresponding per-cell outcomes DO remain in `benchmark/routing/results.csv`, where the
routing analyses read them; `tests/escalation/test_results_set_b_coherence.py` documents why
that set is derived with a superset predicate rather than exact model-set equality.
