---
description: 'From a plain request to an approval card: backlog item, spec, plan, contract with failing tests, tasks, completeness gate and review card — every spec-kit step run for you. `continue` implements and proves one unit; `continue all` runs the whole ready set.'
---


<!-- Extension: work -->
<!-- Config: .specify/extensions/work/ -->
## User Input

```text
$ARGUMENTS
```

The input is one of:

- a **plain request** in the owner's words ("let members export their workout history as
  CSV") — the INTAKE below;
- a request that **groups work** — a sub-goal, a split of an item into children, a new
  epic or branch ("under TASK-4, add a sub-goal with two children and an integration test")
  — the BRANCH INTAKE below: an item with children is a branch, never a spec feature;
- `continue [unit]` — the CONTINUE section (the unit is `specs/NNN-x`; without one, the
  active feature);
- `continue all` / `all remaining` / `entire plan` — the CONTINUE ALL section: run every
  ready unit until the plan is done, stopping only for a named owner blocker;
- **answers to open assumptions** — the bridge's `Answers to the open assumptions …` message,
  a free-text thread reply or a web comment about a unit whose card is out — the RESUME
  section;
- an **edit of existing work** in plain words, one or many units at once ("set p95 ≤ 200 ms
  on every leaf under TASK-3", "drop the cost guardrail", "move TASK-5 to in progress"), a
  **digest of web comments** (the bridge's `N comments from the specs-site inbox …` message),
  or **feedback on a pending change set** — the EDIT MODE section: one change set and one
  diff card, never a new spec.

Empty input is the one blocking case: there is nothing to work on, so ask what to work on
with a single blocking question (below) and stop.

## What this command is

The one entry point of this project's spec-kit flow. The owner gives a request and gets
back an **approval card**; they never type a step's name. You run every step yourself, in
order, by reading each step's installed command file and following it:

specify → clarify → plan → contract → checklist → tasks → analyze → **completeness gate**
→ card → STOP for approval. After approval, `continue`: implement → converge → harness →
done lint.

**Intake never waits for an answer.** Every open question becomes an assumption with a
recommended default; you take the default, record it, keep going to the card, and list the
questions once, together, in your final message. **The only hard stop is approval.**

Four rules hold throughout:

- **Never invent a number.** A metric's number comes from the spec, the owner, the
  project's metric catalogue, or a measured baseline — and is only ever PROPOSED here.
- **Never commit the spec, plan, contract or tests by hand.** At `warn`/`required` the
  owner's approval commits them together and seals the tests; at `auto` step 11 commits them
  through `.specify/extensions/work/scripts/auto_approve.py` (never a raw `git commit`). An agent commit outside that
  path is not an approval.
- **Never weaken a test or the spec to make them agree.** After approval (implement,
  converge, the harness), a conflict stops the work and is escalated (the contract's
  `abort` path). At intake nothing is approved yet, so a conflict is a question, not a stop
  (next rule).
- **A conflict found at intake is an assumption, never a stop.** The request contradicts a
  sealed test of a shipped feature, the constitution, or another feature's contract → take
  as the default the closest variant of the request that keeps every sealed test and gate
  as they are (an opt-in flag instead of a new default, a new code path instead of a changed
  one), record it as an assumption naming the conflict (the test id and file), and carry on
  to the card; the batch offers the literal request as an option, noting it needs the owner
  to re-approve that test (a `Contract-Change:` commit). You never stop here to ask: the
  QUESTIONS batch at the end carries it, and only empty input blocks.
- **Never approve or commit on the owner's behalf.** An approval in the thread is handled
  by the bridge, never by you — even when the owner's message says "approved".

## Talking to the owner (the protocol)

The owner may be reading on Slack, which turns these exact shapes into buttons. Keep them
exact.

- **The question batch (non-blocking)** — every question the run raised, collected into
  ONE block in your FINAL message, before the `WORK-CARD:` line. You never stop for it: by
  the time you print it, each default is already taken and recorded as an assumption.

  ```text
  QUESTIONS: 2 open — continuing on the recommended defaults (recorded as assumptions A-1, A-2)
  QUESTION A-1: <the question, one sentence>
  1. <option> (Recommended) — <implication in a few words>
  2. <option>
  3. <option>

  QUESTION A-2: <question>
  1. <option>
  2. <option> (Recommended)
  ```

  The `QUESTIONS:` header marks the batch as non-blocking. Each `QUESTION A-n:` id is the
  assumption's id in `spec.md`. Exactly ONE option per question carries `(Recommended)` —
  the default you took. At most 5 questions × 5 options; a blank line between questions.
  Anything beyond 5 questions becomes an assumption with no question (recorded, still on
  the card). No open question → no block. This format REPLACES the option tables the
  clarify and contract steps describe, and every "ask … and wait" in a step's own file.
- **The one blocking question** — only for empty input (nothing to work on):

  ```text
  QUESTION: <the question, one sentence>
  1. <option — your recommendation first, with its implication in a few words>
  2. <option>
  ```

  then stop.
- **The card** — when intake (or a resume) reaches it, the LAST line of your final message
  is exactly:

  `WORK-CARD: <absolute path of contract.yaml> <absolute path of review-card.svg>`

- **Done** — when `continue` completes one unit, the LAST line is exactly:

  `WORK-DONE: <absolute path of contract.yaml>`

- **Done, all units** — when `continue all` completes the plan, the LAST line is exactly:

  `WORK-DONE-ALL: <absolute path of the branch contract or feature dir>`

- **Blocked on the owner** — when the only way forward needs an owner action (an approval,
  a merge, a credential, a decision no assumption may take), the LAST line is exactly:

  `BLOCKER: owner <what is needed>`

  This is the one legitimate stop. Never use it for a missing checklist, `covers:`,
  a red stub, a failing test, or any other agent-executable blocker — fix those and
  continue.

- **The change set** — when EDIT MODE reaches its card, the LAST line is exactly:

  `CHANGE-SET: <absolute path of changeset.json> <absolute path of card.svg>`

## Step 0 — draft in a worktree, never the shared checkout

Before Setup, run `git rev-parse --git-dir --git-common-dir`. When both print the same
directory, this is the repository's MAIN checkout: `require_spec` denies every edit in the
project there, and a pending proposal would stall the host's sync. Run `python3
<substrate>/scripts/spec_draft.py create --cwd .` (`<substrate>`: the repository root's
`agentic/`) and do Setup and every step below in the printed `cwd` — `PROJECT_DIR`, and a
change set's project root (`--project`), are then that worktree's paths. In a linked worktree
(a draft or a session worktree) skip this step.

## Setup

Run `.specify/extensions/work/scripts/bash/work-context.sh --json --agent opencode` from the project root and parse its JSON: `PROJECT_DIR`, `AGENT`,
`SUBSTRATE`, `CONTRACT_CLI`, `CONTRACT_RUN`, `WORK_GATE`, `SPEC_CHARTS`, `SPEC_LINT`,
`CHANGESET_CLI`, `DIFF_CARD`,
`BACKLOG_DIR`, `BACKLOG_CLI`, `CATALOG`, `APPROVAL`, `ACTIVE_FEATURE`, `STEPS` (step → the
command file to follow) and `MISSING_STEPS`. Stop and report when `CONTRACT_CLI` or
`WORK_GATE` is empty (the substrate was not found), or when a step this run needs is in
`MISSING_STEPS` (the project's spec-kit install is incomplete: re-sync it).

**Running a step** means: read the file `STEPS[<step>]` — that step's own instructions —
and follow it exactly, with the input this section gives it as its `$ARGUMENTS`, then come
back here for the next step. Every path a step names is relative to `PROJECT_DIR`. When a
step says to "suggest" or "hand off" to the next command, do not stop: the next step is
yours to run. When it would ask the owner something, do not ask: record an assumption
(below) with your recommended default, continue, and add the question to the batch.
**Run every step, every time** — a small request included. Never skip one or write its
output by hand (`analyze.md` is the analyze step's report, the checklist is the checklist
step's file): a hand-written report is not the step, and the card would claim checks that
never ran.

## Assumptions (how a default is recorded)

Every default you take on the owner's behalf is a line in `spec.md` under
`## Assumptions` (spec-kit's spec template has that section — reuse it; create it at the
end of the spec when it is missing). Prose assumptions already there stay as they are; a
tracked one is exactly this shape, so the card and the bridge can parse it:

```text
- **A-1** (assumed) <question> → <chosen default>. Affects: clarify, plan, contract
```

- Ids are `A-1`, `A-2`, … in order, never reused within a unit.
- The state is `assumed` (default taken, the owner has not answered) or `confirmed` (the
  owner answered; the text after `→` records their answer).
- `Affects:` lists the steps whose output depended on it, from: specify, clarify, plan,
  contract, checklist, tasks, analyze. Add a step to it whenever that step relied on the
  assumption.
- A `[NEEDS CLARIFICATION` marker is never left in `spec.md`: resolve each into such an
  assumption (asked in the batch or not), which is what keeps the gate green.
- A metric whose suggested value rests on an assumption names it in `suggested.reason`
  (`reason: "A-2: assumed 200 ms budget"`) — the source stays `default`, `catalog`,
  `sibling` or `baseline`.

## INTAKE (`/work <request>`)

1. **The backlog item.** Skip this step when `BACKLOG_DIR` is empty (no backlog). Otherwise:
   - Read the backlog: `backlog task list --plain` from `PROJECT_DIR` (or, without
     `BACKLOG_CLI`, the task files under `BACKLOG_DIR/tasks/`). Note the open items, their
     parents, and the roots (items with no parent).
   - If the request names an existing item (`TASK-12`) or plainly IS one (the same work,
     not done), link to it: that item is the unit's backlog item.
   - Otherwise pick its **parent**: the root (or a branch under it) whose objective this
     request serves. One clear fit → use it. Several plausible, or none → take the best fit
     as the default and record it as an assumption once `spec.md` exists (`Affects:` —
     nothing downstream depends on the parent, so list `specify`); its batch entry reads:

     ```text
     QUESTION A-1: Which goal does "<short request>" belong under?
     1. TASK-3 — <its title> (Recommended) — <why>
     2. TASK-7 — <its title>
     3. A new top-level goal
     ```

   - Create the item as that parent's child:
     `backlog task create "<short title>" -p <parent> -d "<the request, verbatim>" --plain`
     (with no parent chosen, omit `-p`). Never write a task file by hand. Record its id.

2. **specify.** Run `specify` with the owner's request (verbatim) as its input. It
   creates the feature directory (`specs/NNN-x`) and its `spec.md`; from here on that
   directory is `FEATURE_DIR` and `specs/NNN-x` is the unit id. When step 1 has an item,
   link it now: `backlog task edit <id> --ref specs/NNN-x --plain`.

3. **clarify.** Run `clarify`, but answer its questions yourself: for each, pick the
   recommended option, write it into `spec.md` where clarify would record the answer, and
   record it as an assumption (`Affects: clarify` plus every later step that uses it). The
   questions go to the batch (at most 5 across the whole run). Then make sure no
   `[NEEDS CLARIFICATION` marker is left in `spec.md`: each one becomes an assumption.

4. **plan.** Run `plan`. Its Task Contract section lists the tests to write first (each with
   the `FR-###`/`SC-###` it covers) and the metrics from the spec's measurable `SC-###`.

5. **contract.** Run `contract` (the feature contract). It drafts `FEATURE_DIR/contract.yaml`
   in schema 2, writes the tests, confirms they fail and records that red run. While it
   drafts the metrics, these rules apply:
   - **Every test `covers:`** the `FR-###`/`SC-###` ids it proves; together the tests cover
     every requirement the spec defines.
   - **Every metric is `state: proposed`**, with `suggested: {value, reason, source}`.
     Prefer a catalogue entry: when `CATALOG` exists, read it, and for a measure it already
     defines (p95 latency, error rate, bundle size, …) write `ref: <catalogue id>` and only
     what differs (tier, value, lineage, state, suggested). A number the spec states →
     `source: default` with the `SC-###` as the reason; a catalogue default →
     `source: catalog`; a sibling feature's accepted value → `source: sibling`.
   - **Lineage:** under a parent with a contract, a metric that rolls into a parent metric
     says `feeds: <parent metric id>` and `combine: max|min|all|remeasure|sum|mean` (never
     `sum`/`mean` on a percentile, ratio or rate); otherwise `local: <why it is this
     feature's own>`. With a contracted parent, also run
     `python3 "$CONTRACT_CLI" rehash "$PROJECT_DIR" --unit <unit>` to write `parent_hash`.
   - **Measure where you can:** when a metric has a runnable `via`, run
     `python3 "$CONTRACT_CLI" baseline "$PROJECT_DIR" --unit <unit>` — it measures each
     proposed metric once on the current code and writes `suggested` with
     `source: baseline` (a threshold or guardrail gets 10% headroom). A `via` that cannot
     run yet (the feature does not exist) stays a spec or catalogue suggestion.
   - **A number, `via` or fixture the step would ask for** is not asked: record an
     assumption whose default comes from the catalogue, a baseline, a sibling or the spec
     (never invented), name it in that metric's `suggested.reason` (`"A-2: assumed 200 ms
     budget"`), and batch the question. With no source at all, the metric stays proposed
     with no suggested number and the question says so.
   - Its validation is clean when the only errors left are `metric_not_accepted`: a
     proposed metric is accepted by the owner at approval, never by you.

6. **checklist.** Run `checklist` for the feature's riskiest quality dimension (its
   requirements' completeness and clarity by default); fix in `spec.md` what it finds
   unclear; what the spec cannot answer becomes an assumption (default + batch entry).

7. **tasks.** Run `tasks`. Every contract test maps to exactly one test task.

8. **analyze.** Run `analyze`, and save its full report (the findings table and the
   metrics block) as `FEATURE_DIR/analyze.md` — the gate reads it there.

9. **The completeness gate.** Run `python3 "$WORK_GATE" "$FEATURE_DIR"`. It exits 0 when
   the feature is complete: no `[NEEDS CLARIFICATION` left, every `FR-###`/`SC-###`
   covered by a contract test, no CRITICAL/HIGH analyze finding, no validator error
   (proposed metrics are listed as pending, never as blockers). On exit 1, fix each blocker
   at its source — the spec, the plan, the contract (a missing test or `covers:`), the
   tasks — re-run the step that owns what you changed (re-run `analyze` after a spec, plan
   or tasks change, and save its report again), and re-run the gate. Loop until it exits
   0. A blocker only the owner can resolve becomes an assumption (its default taken, the
   question batched); one that would need weakening a test or the spec is escalated, never
   "fixed". A gate that CRASHES (not exit 0/1) is a tooling failure: Self-heal below.

10. **The card.** Render it:
    `python3 "$SPEC_CHARTS" card "$FEATURE_DIR/contract.yaml" -o "$FEATURE_DIR/review-card.svg" --png`
    (the evidence log and spec.md beside the contract are read by default). Exit 3 means
    no PNG converter here: run it again without `--png` — the SVG is the card.

11. **Approve.** The approval level is `APPROVAL`.
    - **`auto` (the v6 default): approve it yourself, now.** Run
      `python3 "$SUBSTRATE/scripts/auto_approve.py" unit --feature "$FEATURE_DIR" --project "$PROJECT_DIR" --tests-seen-red`.
      It re-runs `work_gate` and the DAG validator, selects the critic (a skipped critic is
      a named degraded mode, never a silent one), writes `.agentic/auto-approval.json`,
      commits the items with `Approved-By: auto`, and prints
      `AUTO-APPROVED: <absolute contract> <absolute card> <sha>`. **Remember that line.**
      Continue with the implementation (`continue`) in this same turn — do not stop, do not
      ask — and **end the turn with the remembered `AUTO-APPROVED:` line** (before or beside
      `WORK-DONE`). The bridge posts the receipt only when the final message carries it.
      If the script also prints `ROOT-ALERT:`, end with that line too: it opens the 24 h
      veto window for a root contract change.
    - **`warn` / `required`: STOP for approval.** Do not implement, and do not commit
      anything the approval commits (spec.md, plan.md, tasks.md, contract.yaml, the tests,
      the backlog item's link). Report, briefly: the backlog item and its parent; the
      feature and its one-line objective; the tests (id, category, what they cover) and
      that they were seen failing; each metric with its suggested value, reason and source;
      the assumptions (id, state, default); and how to approve — the card's Approve button
      on Slack, or `make approve UNIT=<unit>` on a terminal. Then the QUESTIONS block for
      the still-`assumed` ones (if any), then the WORK-CARD line, last.

## BRANCH INTAKE (a node that groups work)

For a request that adds a node with children, or splits an item into children. It is work
like any other — never "only backlog items, so no flow": the branch's contract is what its
children are checked against.

1. **The items.** Find the parent the request names (or the best fit, as an assumption).
   Create the branch item as its child (`backlog task create "<title>" -p <parent> -d
   "<the request, verbatim>" --plain`), then each child as the branch's child
   (`backlog task create "<child title>" -p <the new item> --plain`). Never write a task
   file by hand.
2. **The branch contract.** Run `STEPS[contract-root]` with `<the new item> branch` as its
   input. Its "ask, don't guess" step is answered here, not asked: each question gets your
   recommended default, recorded in the contract's `decisions:` (`{id: A-1, trigger: <the
   question>, default: <the default>, rationale: …, decided_by: agent (assumed)}`) and added
   to the batch; the card lists them as its assumptions. An owner's answer later (RESUME)
   sets that decision's `default:` to the answer and `decided_by: owner`. The contract carries the `rationale:` (why this split), `requirements:`,
   `interfaces:` naming the children, and at least one `integration` (or `acceptance`) test
   that `covers:` them.
3. **The test, seen failing.** Write the integration test file the contract names — at a
   test-shaped path (`test_*.py`, `*.test.ts`, under `tests/`): the edit gate opens exactly
   those while the branch contract is unapproved — and run it: it must fail now (the children are not built). Record the red run
   (`python3 "$CONTRACT_CLI" red "$PROJECT_DIR" --unit <the new item> --check <test id>`)
   and say in the report which command failed and how.
4. **Lineage.** Under a contracted ancestor, run `python3 "$CONTRACT_CLI" rehash
   "$PROJECT_DIR" --unit <the new item>` so the contract carries `parent_hash`.
5. **The gate.** `python3 "$CONTRACT_CLI" validate "$PROJECT_DIR" --unit <the new item>` and
   `python3 "$SUBSTRATE/spec/dag_validate.py" "$PROJECT_DIR" --unit <the new item>`: no
   error but `metric_not_accepted`. Fix each finding at its source and re-run; one only the
   owner can resolve becomes an assumption.
6. **The card.** `python3 "$SPEC_CHARTS" card <the branch contract> -o
   <BACKLOG_DIR>/contracts/<the new item>.review-card.svg --png` (exit 3: again without
   `--png`).
7. **STOP for approval**, exactly as intake's step 11: never commit the items, the contract
   or the test; report the tree (item, children, depth), the rationale, the tests and the
   red run, the metrics as proposed, the assumptions; then the QUESTIONS block and the
   `WORK-CARD: <contract path> <card path>` line, last. Each child is later run through
   INTAKE on its own (a leaf) or this section again (another branch).

## EDIT MODE (one change set)

For an edit of EXISTING work — a plain-words change to one or many units, a digest of web
comments, feedback on a pending change set. The owner gets ONE proposal and approves it with
ONE click (with a per-item opt-out); nothing you do here is a new feature or a new spec.

1. **Resolve the node set from the DAG.** Name every unit the input touches: an id it gives
   (`TASK-3`, `specs/002-x`), and for "every leaf under X" / "all of X" the subtree from
   `python3 "$CHANGESET_CLI" subtree "$PROJECT_DIR" X` (`leaf: true` rows for "leaves").
   A comment's `node` plus its `anchor` (`{kind: node|spec|metric|test|assumption|status,
   ref}`) says what it is about: a metric id, a spec section or requirement, a test id, an
   assumption `A-n`, or — kind `status` — the backlog status the item was dragged to.
   "drop the cost guardrail on <project>" means every unit whose contract carries that
   metric. A node that does not exist is skipped, never invented.
2. **Edit the working tree, per unit**, with the same rules as intake:
   - a metric: set `tier`/`comparator`/`value` and `state: proposed` (or remove the entry for
     a drop). **Never invent a number** — the number comes from the input, the catalogue, a
     sibling or a measured baseline; with none, `value: THRESHOLD-UNSET` plus a
     `suggested` with its source, and an assumption in the QUESTIONS batch;
   - a spec section, a test (its contract entry and its failing test file — never weakened
     to pass), an assumption (`(assumed)` → `(confirmed)`, or a new `A-n`): edit them as the
     owner asked; a spec edit that changes requirements re-runs `tasks` → `analyze` → the
     completeness gate for that feature (RESUME's rule);
   - a status: edit the backlog item's `status:` (`backlog task edit <id> -s "<Status>"`, or
     the file). The **status gate** applies: an item moved to In Progress or Done needs a
     spec'd leaf with a valid contract (or children that cover it); one that would fail is
     NOT edited — it goes in `skipped` with the reason;
   - validate every touched contract (`python3 "$CONTRACT_CLI" validate .`).
   Never commit — the owner's approval commits the whole set.
3. **Write the change set** — the schema is the docstring of `$CHANGESET_CLI`. ONE item per
   independent change (`C1`, `C2`, …): kind `metric|test|spec|assumption|status|backlog`, its
   `unit`, a one-line `title`, the project-relative `files` it edited (items that can be
   approved separately should not share a file), the inbox row ids it answers in `rows`, a
   metric's `before`/`after`, a status's `before`/`after`. Comments you could not turn into a
   change go in `skipped` with the reason (never silently dropped); open questions in
   `assumptions`. Where: the path the input names (a digest names it), else
   `${TMPDIR:-/tmp}/agentic-changesets/<project>/chat-<8 hex>/changeset.json`. On feedback,
   rewrite the SAME file and keep unchanged items' ids.
4. **Check and render.** `python3 "$CHANGESET_CLI" validate <changeset.json> --project
   "$PROJECT_DIR"` must exit 0; then `python3 "$DIFF_CARD" <changeset.json> -o <dir>/card.svg
   --png` (a before→after dumbbell per metric, then tests, spec, assumptions and status).
5. **Report** in a few lines what changed per unit and what was skipped, then the QUESTIONS
   batch (if any), and end with the CHANGE-SET line. The bridge posts the card with Approve
   all / Approve selected / Request change; a free-text reply in its thread comes back to you
   as feedback (step 3's "rewrite the same file"). "approve all leaves under X" is the owner
   approving pending cards — the bridge does it, never you.

## RESUME (answers to open assumptions)

The owner answers after the card is out — the bridge's message (unanswered ones keep their
recommended default):

```text
Answers to the open assumptions (unanswered ones keep their recommended default):
A-1: <answer>
```

or a free-text thread reply, or a web comment. Find the unit (the one named, else
`ACTIVE_FEATURE`) and its `spec.md` assumptions, map each answer to its `A-n` (a free-text
reply: by its content; ambiguous → keep the default and say so), then:

1. **Record.** Each answered assumption becomes `(confirmed)`, with the owner's answer after
   `→`. An unanswered one stays `(assumed)`.
2. **Answer equals the default** → nothing re-runs for it.
3. **Answer differs** → re-run ONLY the steps in its `Affects:` list, in flow order, each
   with the new answer as its input, plus their downstream consumers — never the whole
   flow, never `specify` from scratch. Whenever anything changed, `tasks` → `analyze` →
   the completeness gate → the card always re-run. A changed number in a metric's
   suggestion stays `state: proposed`.
4. **Report** what changed and which steps re-ran, re-render the card, and end with the
   QUESTIONS block for the still-`assumed` ones (if any) and the WORK-CARD line.

A message that approves ("approved", "ship it") is not an answer: the bridge handles
approval. Do not commit or approve; say the card is waiting for the Approve button.

## Self-heal (a tooling step fails)

A tooling step is `WORK_GATE`, `CONTRACT_RUN`, `CONTRACT_CLI`, `SPEC_CHARTS`, the sync, the
done lint (`SPEC_LINT`), the backlog CLI, or a script a step runs. A failure is a crash, a
traceback or an error that is not the tool's documented verdict (a gate exit 1 with
blockers is a verdict — fix the blockers; a traceback is a failure). On a failure:

1. **Log it** as an observability event, every time (`<step>` the tool, `<unit>` the unit):

   ```bash
   AGENTIC_HOST_ROOT="$(git -C "$PROJECT_DIR" rev-parse --show-toplevel)" python3 -c 'import sys; sys.path.insert(0, sys.argv[1] + "/enforcement"); import lib; lib.log_event("fail", sys.argv[4], gate="work", tool=sys.argv[2], extra={"unit": sys.argv[3]})' "$SUBSTRATE" <step> <unit> "<one-line error>"
   ```

   It appends one row to the host's event log (`.agentic/observability/events.jsonl`, or
   `AGENTIC_EVENT_LOG`) and never fails the turn.
2. **Diagnose and fix it in the same turn.**
   - A bug in THIS project's code → it is a request like any other: it goes through this
     flow (its own unit), not a side edit.
   - A bug in substrate tooling → fix it in a substrate worktree
     (`git -C "$SUBSTRATE" worktree add <path> -b fix/<name>`), red test first, then the
     fix, land it, and re-sync the host (`make sync` from the host root); then re-run the
     failed step.
   - A broken environment (a missing tool, a stale install) → repair it (re-sync, install
     the pinned tool) and re-run.
3. **Twice is a backlog item.** After the 2nd failure of the same step in one unit, open
   one: `backlog task create "Fix <step> failure: <short cause>" -d "<step>, unit <unit>:
   <error>, what was tried" --plain` (from `PROJECT_DIR`; without a backlog, say it in the
   report).
4. **Never weaken a test, a gate, a contract or the spec** to get past a failure — no
   skipped check, no loosened threshold, no deleted `covers:`. If the only way past is a
   weaker check, stop and escalate.

## CONTINUE (`/work continue [unit]`)

1. **Find the unit.** The named `specs/NNN-x`, else `ACTIVE_FEATURE`. Its `contract.yaml`
   must be committed (`git log -1 -- <contract>` prints a commit): that commit is the
   approval. Not approved yet → say so, point at the card, and stop. Every metric must be
   `state: accepted`; a proposed one means the approval is incomplete — stop and say so.

2. **implement.** Run `implement` for the unit. The sealed tests are read-only: if one
   seems wrong, stop and escalate (QUESTION), never edit it.

3. **converge.** Run `converge`; when it appends tasks, run `implement` again. Repeat until
   converge appends nothing.

4. **The harness.** Commit the implementation (the project's commit conventions, its
   `Spec:` trailer), then run
   `python3 "$CONTRACT_RUN" run "$PROJECT_DIR" --unit <unit>`. It is the only producer of
   counted evidence. A REJECT names what failed: fix the code (never a sealed test), commit,
   and re-run. Follow its stop decision — `revert`, `redirect`, `stop` and `abort` end the
   loop with a report to the owner instead of another attempt.

5. **The done lint.** Commit the evidence log the harness appended, then mark the work
   done — tick every task in `tasks.md` and, with a backlog item,
   `backlog task edit <id> -s Done --plain` — and run
   `python3 "$SPEC_LINT" --lint` on the staged change before committing it. It refuses a
   unit marked done without green harness evidence for exactly the sealed files; fix the
   cause and re-run, never bypass it.

6. **Report** the evidence (each check and its outcome, the metrics' measured values), and
   end with the WORK-DONE line.

## CONTINUE ALL (`/work continue all`)

The owner's "execute everything / all remaining / entire plan" is ONE run, not one unit per
message. After the first unit's `WORK-DONE`, do not summarize and do not stop: select the
next ready unit and run the CONTINUE steps above again.

1. **Select the ready set.** From the backlog/DAG, take every unit whose contract is
   committed, whose dependencies are `Done`, and whose metrics are `accepted`. Skip any
   unit the owner explicitly excluded (the current v6 wave excludes poker-coach and travel).
2. **Run it.** Run CONTINUE steps 2–6 for that unit. A unit whose approval is missing, or
   whose metric is still `proposed`, is an owner gate: post that unit's card and end with
   `BLOCKER: owner approve <unit>`.
3. **Agent-executable blockers self-heal.** A missing checklist, uncovered requirement,
   red stub or failing test is fixed at its source and the owning step re-run; it is never a
   reason to stop. A `revert`/`redirect`/`stop`/`abort` decision from the harness is a real
   stop: report it as `BLOCKER: owner <reason>`.
4. **Loop.** Repeat 1–3 until the ready set is empty. Emit `REMAINING: <ids>` each
   iteration (or `REMAINING: none` at the end), then the final `WORK-DONE-ALL:` line. Keep
   the per-unit `WORK-DONE:` lines in the report so the bridge and the owner can audit the
   whole wave.

If the owner named a subset ("all except X"), that subset is the ready set; do not expand it.

## Done When

- [ ] Intake: the gate exits 0, the card is rendered, every default taken is recorded in `spec.md` as an `A-n` assumption, the open questions are batched in ONE `QUESTIONS:` block (none asked and waited on), nothing the approval commits was committed, and the last line is `WORK-CARD: …`
- [ ] Resume: answered assumptions are `confirmed`, only the affected steps (plus tasks → analyze → gate → card) re-ran, and the message ends with the remaining `QUESTIONS:` block (if any) and `WORK-CARD: …`
- [ ] Self-heal: every tooling failure is logged as an event, fixed (never by weakening a check), and a repeat (2nd failure of a step in a unit) has a backlog item
- [ ] Edit mode: every touched unit resolved from the DAG, every edit drafted in the tree (metrics `proposed`, no invented number), the change set validates, the diff card is rendered, nothing committed, and the last line is `CHANGE-SET: …`
- [ ] Continue: the harness ACCEPTED the unit, the done lint passed, and the last line is `WORK-DONE: …`
- [ ] Continue all: every ready unit ran, the final ready set is empty, and the last line is `WORK-DONE-ALL: …` (or a named `BLOCKER: owner …`)
