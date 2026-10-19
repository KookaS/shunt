---
name: speckit-contract-feature
description: Write the feature's contract.yaml from spec.md and plan.md (tests first, metrics tiered from SC-###, abort path), validate it, then write its tests and confirm they fail.
compatibility: Requires spec-kit project structure with .specify/ directory
metadata:
  author: agentic
  source: contract:commands/speckit.contract.feature.md
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty). It may name the
level (`leaf` or `branch`), answer an open question, or say "update".

## What this step is

The step between plan and tasks: specify → clarify → plan → **contract** → tasks →
analyze → implement → converge. It turns the spec's requirements and success criteria into
a **task contract** (`contract.yaml`): the tests that decide "done" — written NOW, before
any code, and seen failing — the metrics that decide when to keep iterating and when to
stop, and the path taken when the spec and the tests disagree. Not to be confused with the
feature's `contracts/` directory, which holds interface definitions.

Three rules are never broken:

- **Never invent a number.** A metric's value is copied from the spec (a `SC-###`, an FR,
  a clarification answer). With no number: a `target` may say `THRESHOLD-UNSET` (it stays a
  visible open decision); a `gate`, `threshold` or `guardrail` may NOT — ask the owner
  (below), and if unanswered, leave that metric out and report it as open.
- **Never write implementation code here.** Only the contract and the failing tests.
- **Never weaken a test to make it fail or pass.** A test encodes the spec; when they
  conflict, stop and ask.

## Outline

1. **Setup.** Run `.specify/extensions/contract/scripts/bash/contract-context.sh --json` from the project root and parse its JSON: `PROJECT_DIR`,
   `FEATURE_DIR`, `UNIT_ID` (`specs/NNN-x`), `FEATURE_SPEC`, `IMPL_PLAN`, `CONTRACT_FILE`,
   `EVIDENCE_FILE`, `HAS_CONTRACT`, `CONTRACT_CLI`. If it fails (no feature, no plan.md),
   report its message and stop: the contract is written after the plan. If `CONTRACT_CLI`
   is empty, say the substrate's contract CLI was not found (`agentic/scripts/contract.py`
   above the project, or `$AGENTIC_CONTRACT_CLI`) and stop — an unvalidated contract gates
   nothing.

2. **Existing contract.** If `HAS_CONTRACT` is true, read it. If it is already committed
   (`git log -1 -- <CONTRACT_FILE>` prints a commit), it is approved and its tests are
   sealed: do not rewrite it. Report its validation (step 6) and stop, unless the user said
   "update" — then make only the requested change and tell the user that committing it is a
   logged contract change (a commit carrying a `Contract-Change:` trailer saying what changed
   and why), never a silent edit.

3. **Read the inputs.**
   - `spec.md`: every functional requirement `FR-###`, every success criterion `SC-###`
     (its number, unit, and wording — "must" vs "should"), every user story's acceptance
     scenarios (Given/When/Then), the edge cases, the assumptions and anything marked out of
     scope.
   - `plan.md`: the language, the test runner and command, the source and test layout, and
     its **Task Contract** section if present (test rows, metric rows, level).
   - `.specify/memory/constitution.md` if present (it may require test levels or sealing).

4. **Choose the level.** `leaf` by default. `branch` when the feature is split into child
   features with interfaces between them (or the user says so). Never `root` here: a root
   contract belongs to a backlog item (`$speckit-contract-root`).

5. **Draft `contract.yaml`** (schema 2) at `CONTRACT_FILE`. For example, for
   `specs/004-export`:

   ```yaml
   schema: 2
   id: specs/004-export                 # UNIT_ID, exactly
   level: leaf                          # or branch
   objective: Export a month of orders as one CSV file the accountant can open.
   non_goals: [XLSX or PDF output]
   context: [specs/004-export/spec.md, specs/004-export/plan.md]
   tests:                               # every test is a gate; written now, failing now
     - {id: UT-export-rows, path: tests/unit/test_export.py, level: unit, category: happy,
        covers: [FR-001], sealed: true, trust: A}
     - {id: UT-export-empty, path: tests/unit/test_export_empty.py, level: unit,
        category: error, covers: [FR-002], sealed: true, trust: A}
     - {id: AT-export-month, path: tests/acceptance/test_export_month.py, level: acceptance,
        category: acceptance, covers: [FR-001, SC-001], sealed: true, trust: A}
   metrics:                             # leaf: at most one (the primary measurable SC-###)
     - {id: TH-export-p95-ms, tier: threshold, via: python3 bench/export_p95.py,
        comparator: "<=", value: 2000, unit: ms, direction: lower,
        dataset: fixtures/orders-10k.csv, trust: A,
        local: "SC-001 is this feature's own: nothing above it measures export time",
        state: proposed,
        suggested: {value: 2000, reason: "SC-001: under 2 s", source: default}}
   abort: {conflict: escalate}          # spec vs tests conflict → stop and escalate
   decisions:
     - {id: PIV-0, trigger: none of the anticipated forks matches,
        default: "STOP, escalate to the owner", rationale: unknowns are never guessed,
        decided_by: owner, date: "2026-10-16"}
   rollback: {ref: a1b2c3d, procedure: revert the feature's commits}   # current HEAD short sha
   dod: [every contract test green under the harness, the project's full check green]
   ```

   Mapping rules:
   - **Tests.** At least one per user story; prefer one per acceptance scenario or FR that
     is observable. Ids are prefixed by level: `UT-` unit, `PT-` property, `CT-` contract,
     `IT-` integration, `AT-` acceptance. `path` follows the plan's test layout. Every test
     is `sealed: true` and `trust: A` (an executable oracle). A check only a human can judge
     is not a test: make it a metric with `trust: B`.
   - **Category and coverage.** Every test has a `category` — `happy`, `edge`, `error`,
     `security`, `perf`, `regression`, `integration` or `acceptance` — and `covers:`, the
     `FR-###`/`SC-###` ids it proves (a branch test may name a parent metric or test id).
     Together the tests cover EVERY `FR-###` and `SC-###` the spec defines; an uncovered one
     is a missing test, never a silent gap.
   - **Branch** additionally needs: at least one `integration` or `acceptance` test written
     before the children start; one `contract`-level test per interface; `interfaces:
     [{id, provider, consumer, artifact, test}]` naming child unit ids; `rollup: {gates: all,
     thresholds: remeasure}`; `rationale:` (why the children cover this feature); and a kill
     rule such as "a child fails its gates twice → split it or escalate".
   - **Metrics come only from measurable `SC-###`.** Tier by the spec's commitment:
     `threshold` — a committed number ("must", "at least", "within"); `target` — a stretch
     ("should", "aim", "ideally"); `guardrail` — must not regress ("no worse than", "still
     under"), with `tolerance` only if the spec states one; `gate` — a pass/fail criterion
     with no number. `north_star` is root-only — never here. Quote every comparator
     (`">="`, `"<="`, `"=="`), or use `ge`/`le`/`eq`/`gt`/`lt`. A success criterion that is
     not measurable by a command becomes a test (if executable) or is reported as
     unmeasurable — it is not dropped silently.
   - **Every metric is proposed, never accepted, by you.** Write `state: proposed` and
     `suggested: {value, reason, source}` — where the number came from: `default` (the spec
     states it; the reason names the `SC-###`), `catalog` (the project's catalogue default),
     `sibling` (another feature's accepted value) or `baseline` (measured, below). The owner
     accepts it at approval; only an `accepted` metric is sealed.
   - **Prefer the catalogue.** When `<backlog_dir>/metrics.yaml` exists, read it; for a
     measure it defines, write `ref: <catalogue id>` and only what differs — the entry
     supplies `unit`, `direction`, `via`, `comparator`, `trust` and the default tier and
     value. Define a metric in full only when the measure is the feature's own.
   - **Lineage.** Each metric either feeds a parent metric — `feeds: <parent metric id>`
     with `combine: max | min | all | remeasure | sum | mean` (never `sum` or `mean` on a
     percentile, a ratio or a rate) — or is `local: <why it is this feature's own>`. When
     the feature has a contracted parent (a backlog item's contract), run
     `python3 "$CONTRACT_CLI" rehash "$PROJECT_DIR" --unit "$UNIT_ID"` after writing the
     file: it records `parent_hash`, the parent version this contract was written against.
   - **Measure where you can.** For a metric whose `via` already runs on the current code,
     run `python3 "$CONTRACT_CLI" baseline "$PROJECT_DIR" --unit "$UNIT_ID"`: it measures
     each proposed metric once and writes `suggested` with `source: baseline` (10% headroom
     on a threshold or guardrail). A `via` that needs the feature stays a spec or catalogue
     suggestion.
   - **Budget, kill, prereg** are optional on a leaf: write them only from numbers the spec,
     plan or user gave. A missing budget is a warning, not a reason to invent one.

6. **Ask, don't guess.** Collect every committed metric (gate/threshold/guardrail) whose
   number, `via` command or fixture the inputs do not give, plus a missing objective or
   non-goal. Ask at most 5 questions, one at a time, in the clarify style: the question, a
   short table of options (A/B/C with the implication of each, your recommendation first),
   and "or a short answer". Record each answer in `spec.md` (under `## Clarifications`, as
   `- Q: … → A: …`, and in the success criterion it settles) and in the contract. Anything
   still unanswered: a target becomes `THRESHOLD-UNSET`; a committed metric is left out
   and listed as an open decision in the report.

7. **Validate until clean.** Run
   `python3 "$CONTRACT_CLI" validate "$PROJECT_DIR" --unit "$UNIT_ID"`.
   Fix every error and re-run until the only errors left are `metric_not_accepted` (a
   proposed committed metric: the owner's acceptance at approval clears it) and
   `leaf_no_red_test` before step 9 has recorded the red run. Do not silence a check by
   deleting what it asks for (a test, the abort path, `PIV-0`), and never set `state:
   accepted` yourself; fix the contract instead. Report warnings (`missing_budget`,
   `threshold_unset`, `leaf_many_metrics`, …) to the user as-is.

8. **Write the tests first.** For every test with a `path`, create the test file in the
   project's test framework. Each test encodes its acceptance scenario or FR as executable
   assertions against the code the feature WILL add (import the planned module, call the
   planned function, hit the planned route). Write no implementation code, no stubs that
   make a test pass, and no `skip`/`xfail` markers.

9. **Confirm red.** Run the test command from the plan for exactly those files. Every
   contract test must FAIL, and fail for the right reason: an assertion, or the missing
   feature itself (the planned module or route does not exist yet) — not a syntax error, a
   typo, or a broken fixture in the test. Fix any test that fails for the wrong reason, and
   treat a test that PASSES as a defect (it does not test the new behaviour): rewrite it or
   tell the user. Then record the red run:
   `python3 "$CONTRACT_CLI" red "$PROJECT_DIR" --unit "$UNIT_ID" --check <id> [--check <id> …] --note "<one-line failure summary>"`.
   These rows are `produced_by=self` — history that the tests were seen failing, never a
   verdict any gate counts.

10. **Report.** Print: the contract path and level; each test (id, level, path, red
    reason, what it covers); each metric (id, tier, `SC-###` source, value or
    `THRESHOLD-UNSET`, its suggestion and source, `ref`, lineage); open decisions
    (unanswered committed metrics); validation warnings; and the next steps — run
    `$speckit-tasks`, then the owner approves: the approval commit carries
    `spec.md`, `plan.md`, `contract.yaml` (its metrics accepted) and the new tests together,
    and seals the tests. Do not make that commit yourself.

## Done When

- [ ] `contract.yaml` (schema 2) exists at `CONTRACT_FILE` and `contract.py validate` reports no error for `UNIT_ID` but `metric_not_accepted`
- [ ] Every `FR-###`/`SC-###` of the spec is in some test's `covers:`
- [ ] Every metric is `state: proposed` with a `suggested` value and source, `ref`s a catalogue entry where one fits, and `feeds` a parent metric or is `local`
- [ ] Every metric value comes from the spec, an owner's answer, the catalogue or a baseline; no committed metric is `THRESHOLD-UNSET`
- [ ] Every contract test with a path exists and was seen failing for the right reason, recorded with `contract.py red`
- [ ] No implementation code was written
- [ ] Open decisions and the next step reported to the user
