# Test-first (spec-kit preset)

Makes tests mandatory and written first, through spec-kit's override stack:

- `tasks-template` — replaced: each user-story phase opens with its test tasks and a red run;
  branch features write their integration/contract tests in the Foundational phase; every
  test in the feature's `contract.yaml` maps to one task.
- `plan-template` — appended: a Task Contract section (tests to write first, metrics from the
  spec's success criteria).
- `speckit.tasks`, `speckit.plan` — wrapped (`{CORE_TEMPLATE}` keeps the core body): tasks
  requires `contract.yaml`; plan names the contract step next.

Requires the `contract` extension. Installed and kept current by the substrate's `speckit`
project kit (`docs/kits.md`); do not edit the installed copy under `.specify/presets/`.
