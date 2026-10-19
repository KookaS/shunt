# shunt Constitution

The rules every change to this project follows. Amend it deliberately: a change here is a
change to how all future work is judged, so it is committed on its own, with the reason.

## Core Principles

### I. Tests first

Behaviour changes start with a failing test — a unit test for logic, a smoke check for a
route or a page. Tests are mandatory, not optional: every user story's tasks open with its
tests, which are run and seen FAILING for the expected reason before any production code
is written. A change is not done until those tests pass unedited and the project's full
check (lint, typecheck, tests, build) passes.

### II. Specs flow forward

A spec describes a change at the time it was approved. It is not rewritten to match what
was later built: new work gets a new spec that references the old one. Old specs stay as
history, which keeps every decision traceable to the text that authorised it.

### III. Approval is a committed spec and plan

Implementation starts only once the feature's `spec.md` and `plan.md` are committed and
its `tasks.md` exists. Committing spec and plan is the approval; an uncommitted plan is a
draft, and code written against a draft has no agreed target.

### IV. Every code commit names its spec

A commit that changes project code carries a `Spec: <path to the feature's spec
directory>` trailer, so each change can be traced to the spec that approved it.

### V. Every feature carries a task contract

Between plan and tasks, each feature gets a `contract.yaml`: its objective and non-goals,
the tests written first, the metrics taken from the spec's measurable success criteria
with their tier (gate, threshold, target, guardrail), the abort path and the rollback
point. A number is never invented: an unknown target is `THRESHOLD-UNSET`, an unknown
committed threshold is a question for the owner. A grouping backlog item carries a root or
branch contract with a north star, guardrails, kill criteria and a budget.

### VI. Tests are sealed at approval

The commit that approves a feature (spec, plan, contract and its tests together) seals the
contract's tests. Implementation makes them pass; it never edits them. Changing a sealed
test or the contract afterwards is a logged change of its own (a `Contract-Change:`
trailer saying what changed and why), never part of an implementation commit. Done is
judged by the harness's evidence, not by the builder's report.

### VII. Abort on conflict

When a test and the spec contradict each other, or a kill criterion fires, work stops and
is escalated to the owner. Neither the test nor the spec is edited to make them agree.

## Governance

This constitution supersedes ad-hoc practice. Amendments state what changed and why, bump
the version below, and are reviewed like code.

**Version**: 1.1.0
