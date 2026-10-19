---
description: Generate an actionable, dependency-ordered, TEST-FIRST tasks.md for the feature from its spec, plan and task contract.
strategy: wrap
---

## Test-first project rules (read before the core instructions below; they win on conflict)

This project generates tasks **test-first**. The core instructions below were written for
projects where tests are optional; here every "tests are OPTIONAL", "if tests requested" or
"(if included)" is void — **tests are MANDATORY**.

1. **The task contract comes first.** Read `FEATURE_DIR/contract.yaml`. If it is missing,
   STOP: tell the user to run `__SPECKIT_COMMAND_CONTRACT__` first, and generate nothing.
   If it exists, every entry of its `tests[]` becomes exactly one test task, with the same
   `path`, naming its id in the description — `(contract: UT-parse)`. A test the contract
   does not list but a story needs is added as a task AND reported, so the contract can be
   amended (do not edit contract.yaml from this command).
2. **Each user-story phase opens with its test tasks**, then a task that runs them and
   confirms they FAIL for the expected reason (red), then the implementation tasks, then a
   task that runs them again and confirms they PASS without editing them (green).
3. **Branch features** (`level: branch` in contract.yaml): the contract's integration,
   acceptance and contract tests (one per `interfaces[]` entry) go in the Foundational
   phase, before any story, outside-in, followed by their red run.
4. **Sealed tests.** Tests are sealed once spec, plan and tasks are approved. No
   implementation task may edit a contract test; if a test and the spec conflict, the task
   is "STOP and escalate to the owner" (the contract's `abort` path), never "fix the test".
5. **Metrics.** The Polish phase carries one task per `metrics[]` entry of contract.yaml:
   measure it with its `via:` command and record the value. Never write a threshold into
   tasks.md that contract.yaml does not carry.
6. **Only a task line may START with a task ID.** tasks.md is linted at commit: a line whose
   first token (after an optional bullet or checkbox) is `T###` must read exactly
   `- [ ] T### description`. Dependency notes, parallel examples and every other prose line
   name tasks mid-sentence — write `- Task T019 can run in parallel with T006`, `After T005,
   …`, never `- T019 can run …` (this exact slip failed two approval commits).
7. In the Completion Report, also report: contract tests covered (n of n), any test added
   that the contract lacks, and the red-run tasks per phase.

{CORE_TEMPLATE}
