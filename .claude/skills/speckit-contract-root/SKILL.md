---
name: speckit-contract-root
description: 'Write the root or branch contract of a backlog item: objective, north star, guardrails, kill criteria, pre-registration, budget and roll-up — never inventing a threshold.'
compatibility: Requires spec-kit project structure with .specify/ directory
metadata:
  author: agentic
  source: contract:commands/speckit.contract.root.md
---

## User Input

```text
$ARGUMENTS
```

The input names the backlog item (`TASK-7`) and, optionally, the level (`root` or
`branch`). If no item is named, ask which one; never pick one yourself.

## What this step is

A backlog item that groups work — an epic (**root**) or a feature with children
(**branch**) — gets its contract as a file, `<backlog_dir>/contracts/<TASK-id>.yaml`, never
as task frontmatter (the backlog CLI drops keys it does not know). The spec features under
it carry leaf contracts (`/speckit-contract`); this contract owns what no leaf
can: the outcome, the guardrails every descendant inherits, when to kill the whole effort,
and how the children roll up.

Work falsifier-first: decide what would prove this effort wrong or infeasible BEFORE
polishing what success looks like. Never invent a number: a value is quoted from the item,
its spec, or the owner; `THRESHOLD-UNSET` is allowed only on a `target` or the
`north_star`; a `gate`, `threshold` or `guardrail` without a number is asked about, or left
out and reported open. Never produce a composite score or a "goal quality" number.

## Outline

1. **Setup.** Run `.specify/extensions/contract/scripts/bash/contract-context.sh --json --root` from the project root and parse `PROJECT_DIR`,
   `BACKLOG_DIR`, `CONTRACTS_DIR`, `CONTRACT_CLI`. On failure (no backlog), report and stop.
   If `CONTRACT_CLI` is empty, say so and stop.

2. **Read the item.** Find its task file under `BACKLOG_DIR` (its id is in the frontmatter
   `id:`), its parent and children (`parent_task_id`, subtasks, dependencies), any spec it
   references, and any existing contract at `CONTRACTS_DIR/<id>.yaml` (if committed, it is
   pre-registered: change it only on request, as a logged change — a commit with a
   `Contract-Change:` trailer). Read the parent's contract too: its guardrails are inherited
   and its budget bounds this one.

3. **Choose the level.** `root` when the item has no contracted ancestor and owns an
   outcome; `branch` when it sits under a contracted parent and splits into children.

4. **Falsifier first** (both levels):
   - `kill`: the observable results that mean the approach is wrong — and one
     **feasibility** kill (the goal cannot be reached at all, e.g. the target is below the
     measured noise floor) — each with a `decision_rule` that ends in continue or
     stop-and-escalate, never "ask".
   - `prereg` (root required): `validates_if`, `falsifies_if`, `timeout`, `decision_rule`.
     Frozen once committed; a later change is a logged contract change.
   - `budget`: `tokens`, `wall` (`40h`, `2d`), `attempts`, `plateau_k`, and `reserve` (the
     fraction held back from children) — only numbers the owner gave; ask otherwise.

5. **Outcome and guardrails.**
   - **Root:** exactly one `north_star` metric — held out from the builders or judged by a
     human (`trust: B`), never just the sum of the leaves; `value` may be `THRESHOLD-UNSET`
     when the judgement is a sign-off. At least one held-out `acceptance` test if one can be
     written (`held_out: true`). At least one `guardrail` metric (must not regress;
     inherited by every descendant) with a measured or stated bound.
   - **Branch:** an `integration` or `acceptance` test written before the children start,
     one `contract`-level test per interface, `interfaces: [{id, provider, consumer,
     artifact, test}]` naming child ids, and a re-plan kill ("a child fails its gates twice →
     split it or escalate").
   - Both: `rollup: {gates: all, thresholds: remeasure}` (or `min` when a threshold can only
     come from the children — the worst child counts, never an average).

6. **Classify every proposed metric before keeping it.** Run this checklist and drop or
   re-tier what fails: is it a predicate or a threshold on a fixed fixture (not a bare
   scalar)? Can a degenerate edit maximise it without the work? Who computes it — a harness
   outside the builders' reach (`trust: A`), a blind judge (`trust: B`), or the builder
   itself (`trust: A'`, never a gate)? Is it measured against the incumbent? A volume proxy
   (lines, files, tokens, tickets closed, coverage as a target) is never a gate, threshold,
   guardrail or north star — at most a `target` that is only reported. Prose intent with no
   predicate (`trust: C`) stays in the objective, never a metric.

7. **Ask, don't guess.** At most 5 questions, one at a time, clarify-style (question, options
   table with implications and your recommendation, or a short answer): the north star's
   judge, missing committed numbers, the budget, the kill thresholds. Record unanswered ones
   as open decisions; route a fork you cannot pre-decide to `PIV-0`.

8. **Write** `CONTRACTS_DIR/<id>.yaml` (schema 2). For example, for a root `TASK-1`:

   ```yaml
   schema: 2
   id: TASK-1
   level: root                      # or branch
   objective: Close the monthly books from the bank export without manual fixes.
   non_goals: [Replacing the accounting system]
   rationale: ingest feeds the report; the report is what finance signs off
   tests:
     - {id: AT-close, path: tests/acceptance/test_close.py, level: acceptance,
        category: acceptance, covers: [NS-signoff], sealed: true, held_out: true, trust: A}
   metrics:
     - {id: NS-signoff, tier: north_star, via: finance signs off the close (human judge),
        comparator: "==", value: 1, trust: B, local: "judged here, never rolled up",
        state: proposed}
     - {id: GR-p95-import, tier: guardrail, via: python3 bench/import_p95.py,
        comparator: "<=", value: 2.0, unit: s, direction: lower, trust: A,
        local: re-measured at the root on the whole pipeline, state: proposed,
        suggested: {value: 2.0, reason: "baseline 1.8 s + 10%", source: baseline}}
   kill:
     - {id: K1, trigger: no sign-off after two closes, decision_rule: stop the epic and write a retro}
     - {id: K-feasible, trigger: the bank export lacks the fields the close needs,
        decision_rule: stop and escalate to the owner}
   prereg: {validates_if: finance signs off one close with zero manual fixes,
            falsifies_if: two closes need manual fixes to the imported totals,
            timeout: two monthly closes, decision_rule: kill per K1 on falsification}
   budget: {tokens: 2000000, wall: 40h, reserve: 0.25}
   rollup: {gates: all, thresholds: remeasure}
   abort: {conflict: escalate}
   decisions:
     - {id: PIV-0, trigger: none of the anticipated forks matches,
        default: "STOP, escalate to the owner", rationale: unknowns are never guessed,
        decided_by: owner, date: "2026-10-16"}
   rollback: {ref: v1.4.0, procedure: revert the epic's merge commits}
   dod: [north star signed off, every child done, guardrails held]
   ```

   Schema-2 rules for every metric and test here:
   - **Proposed, never accepted, by you:** `state: proposed`, with `suggested: {value,
     reason, source}` (`default` — the owner or the item stated it; `catalog`; `sibling`;
     `baseline` — `python3 "$CONTRACT_CLI" baseline "$PROJECT_DIR" --unit <TASK-id>`
     measures each runnable `via` once). The owner accepts them at approval.
   - **Catalogue first:** a measure `<backlog_dir>/metrics.yaml` defines is `ref: <id>` plus
     only what differs.
   - **Lineage:** a metric this node measures itself is `local: <why>`; on a branch, one
     that rolls into its parent's metric is `feeds: <parent metric id>` with `combine:`
     (never `sum`/`mean` on a percentile, ratio or rate). Every committed metric of a parent
     needs a feeder among its children or its own `via`.
   - **Tests** carry `category` and `covers:` (the parent metric or requirement ids they
     prove). A branch also lists `requirements:` it owns and a `rationale:`.
   - **A branch under a contracted parent** gets `parent_hash` from
     `python3 "$CONTRACT_CLI" rehash "$PROJECT_DIR" --unit <TASK-id>`, run after writing.

9. **Validate until clean**:
   `python3 "$CONTRACT_CLI" validate "$PROJECT_DIR" --unit <TASK-id>` — fix errors and
   re-run until the only error left is `metric_not_accepted` (cleared by the owner's
   acceptance; never by setting `state: accepted` yourself, and never by deleting what a
   check asks for); also run it without
   `--unit` to catch tree errors (a child budget exceeding this one minus its reserve).

10. **Report**: the contract path and level, the north star and its judge, each guardrail,
    each kill rule, the budget split available to children, every `THRESHOLD-UNSET` and open
    decision, the validation warnings, and the next step: commit the contract for the
    owner's approval (that commit pre-registers it), then contract each child.

## Done When

- [ ] `<backlog_dir>/contracts/<TASK-id>.yaml` (schema 2) exists and `contract.py validate` reports no error but `metric_not_accepted` for it and for the tree
- [ ] Every metric is `state: proposed` with a suggestion, `ref`s the catalogue where it fits, and is `local` or `feeds` a parent metric
- [ ] At least one kill rule is a feasibility check; every decision rule ends in continue or stop-and-escalate
- [ ] No number was invented; open decisions are listed
