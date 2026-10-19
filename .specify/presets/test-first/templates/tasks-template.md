---

description: "Task list template for feature implementation (test-first)"
---

# Tasks: [FEATURE NAME]

**Input**: Design documents from `/specs/[###-feature-name]/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), contract.yaml (required: the task contract), research.md, data-model.md, contracts/

**Tests**: MANDATORY and written FIRST. Every test the feature's `contract.yaml` lists
(`tests[]`) has a task here with the same id and path. Each user-story phase OPENS with its
test tasks; they are run and seen FAILING (red) before any implementation task of that phase
starts. Once the spec, plan and tasks are approved (committed), those tests are SEALED:
implementation makes them pass, it never edits them.

**Abort path**: if a test and the spec contradict each other, STOP and escalate to the
owner (`abort: {conflict: escalate}` in contract.yaml). Never edit a test or the spec to
make them agree.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions
- A test task names its contract test id in the description, e.g. `(contract: UT-parse)`

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

<!--
  ============================================================================
  IMPORTANT: The tasks below are SAMPLE TASKS for illustration purposes only.

  The tasks command MUST replace these with actual tasks based on:
  - Tests from contract.yaml (`tests[]`: id, path, level) — one test task each
  - User stories from spec.md (with their priorities P1, P2, P3...)
  - Feature requirements from plan.md
  - Entities from data-model.md
  - Endpoints from contracts/

  Tasks MUST be organized by user story so each story can be:
  - Implemented independently
  - Tested independently, by tests written before its code
  - Delivered as an MVP increment

  DO NOT keep these sample tasks in the generated tasks.md file.
  ============================================================================
-->

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project structure per implementation plan
- [ ] T002 Initialize [language] project with [framework] dependencies
- [ ] T003 [P] Configure linting, formatting and the test runner named in plan.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

**Branch features** (`level: branch` in contract.yaml): the contract's integration,
acceptance and contract tests (one per `interfaces[]` entry) are written HERE, outside-in,
and seen failing before any child story starts.

- [ ] T004 [P] Integration test for [end-to-end journey] in tests/integration/test_[name].py (contract: IT-[name]) — branch features only
- [ ] T005 [P] Contract test for [interface between children] in tests/contract/test_[name].py (contract: CT-[name]) — branch features only
- [ ] T006 Run T004–T005 and confirm they FAIL for the expected reason (red); record the red run
- [ ] T007 Setup database schema and migrations framework
- [ ] T008 [P] Implement authentication/authorization framework
- [ ] T009 Create base models/entities that all stories depend on

**Checkpoint**: Foundation ready, branch-level tests red - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - [Title] (Priority: P1) 🎯 MVP

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own — the contract test ids below]

### Tests for User Story 1 (MANDATORY — write first, see them fail) ⚠️

- [ ] T010 [P] [US1] Unit test for [behaviour from FR-001] in tests/unit/test_[name].py (contract: UT-[name])
- [ ] T011 [P] [US1] Acceptance test for [acceptance scenario 1] in tests/acceptance/test_[name].py (contract: AT-[name])
- [ ] T012 [US1] Run T010–T011 and confirm each FAILS for the expected reason (red, not a syntax or import error in the test itself); record the red run

### Implementation for User Story 1

- [ ] T013 [P] [US1] Create [Entity1] model in src/models/[entity1].py
- [ ] T014 [US1] Implement [Service] in src/services/[service].py (depends on T013)
- [ ] T015 [US1] Implement [endpoint/feature] in src/[location]/[file].py
- [ ] T016 [US1] Run T010–T011 and confirm they PASS (green) without editing them

**Checkpoint**: At this point, User Story 1 should be fully functional, with its contract tests green

---

## Phase 4: User Story 2 - [Title] (Priority: P2)

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Tests for User Story 2 (MANDATORY — write first, see them fail) ⚠️

- [ ] T017 [P] [US2] Unit test for [behaviour from FR-00X] in tests/unit/test_[name].py (contract: UT-[name])
- [ ] T018 [US2] Run T017 and confirm it FAILS (red); record the red run

### Implementation for User Story 2

- [ ] T019 [P] [US2] Create [Entity] model in src/models/[entity].py
- [ ] T020 [US2] Implement [Service] in src/services/[service].py
- [ ] T021 [US2] Run T017 and confirm it PASSES (green) without editing it

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

[Add more user story phases as needed, following the same pattern: tests → red → implementation → green]

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] TXXX [P] Documentation updates in docs/
- [ ] TXXX Code cleanup and refactoring (every test still green)
- [ ] TXXX Measure the contract's metrics (`metrics[]` in contract.yaml, each by its `via:`) and record the values
- [ ] TXXX Run the project's full check (lint, typecheck, tests, build) and the contract's checks

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### Within Each User Story

- Tests are written FIRST and seen FAILING before any implementation task of the story
- Models before services
- Services before endpoints
- Core implementation before integration
- The story's tests go green without being edited before moving to the next priority

### Task-level dependencies

- Task T014 depends on T013 (same service, model first)
- Task T020 can run in parallel with T014 once T019 is done

(Only a checklist line may START with a task ID; in prose, name tasks mid-sentence:
`Task T014 …`, `After T013, …`.)

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- All test tasks for a user story marked [P] can run in parallel
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Models within a story marked [P] can run in parallel

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 — tests red, then code, then tests green
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → red → green → Deploy/Demo (MVP!)
3. Add User Story 2 → red → green → Deploy/Demo
4. Each story adds value without breaking previous stories

---

## Notes

- [P] tasks = different files, no dependencies
- A line starting with `T###` (after a bullet) is a task line and must read `- [ ] T### …`; prose writes `Task T###`
- [Story] label maps task to specific user story for traceability
- Every contract test id in contract.yaml appears in exactly one test task
- A sealed test is never edited to pass; a spec/test conflict is escalated, not resolved in code
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
