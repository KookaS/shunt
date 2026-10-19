---
description: Execute the implementation planning workflow using the plan template to generate design artifacts, then hand over to the task contract.
strategy: wrap
---

{CORE_TEMPLATE}

## Test-first project rules (apply after the core instructions above)

- Fill the plan's **Task Contract** section: the test runner and command, one row per test
  to write first (id, level, category, path, and the FR-###/SC-### ids it covers — every
  requirement covered by some row), and one row per measurable success criterion `SC-###`
  (proposed, with its suggested value and source, a catalogue `ref` where one fits, and
  its lineage). Copy numbers from the spec verbatim;
  where the spec gives none, write `THRESHOLD-UNSET` (a target) or `NEEDS CLARIFICATION`
  (a committed threshold, gate or guardrail) — never a plausible constant.
- The **next step is the task contract**, not tasks: in the completion report, name
  `__SPECKIT_COMMAND_CONTRACT__` as the next command, then `__SPECKIT_COMMAND_TASKS__`.
