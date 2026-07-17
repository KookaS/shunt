# AGENTS.md — how to write code in this repo

Rules a human or model must follow here. The linter is the backstop: if a rule
matters, CI enforces it (`pre-commit run --all-files`). This file says *why* so
you get it right the first time.

## Layout

- `src/shunt/` — the shipped router (product). Strictest rules apply here.
- `benchmark/` — the eval harness (installed package `benchmark`). Absolute
  imports only: `from benchmark import config`, never `sys.path` hacks.
- `tools/lint/` — the custom `SH0xx` AST checks.
- `tests/` — pytest suite.
- `examples/providers/` — one copy-paste registry fragment per provider. Adding a
  provider or model? Read `examples/providers/README.md` and `docs/configuration.md`
  first; the registry is `src/shunt/models/default_config.yaml` and its row order
  is semantic.

Install once: `pip install -e '.[dev,benchmark]'` — then `benchmark` imports
resolve everywhere with no path munging.

**Run everything through `uv run` — never bare `python3`.** In a worktree, bare
`python3 -m pytest` resolves `import shunt` to a *different* worktree's source and
reports pass/fail for code you aren't editing. Check with
`uv run python -c "import shunt.models as m; print(m.__file__)"`.

## The rules (enforced by ruff + mypy — one manifest in `pyproject.toml`)

- **Types.** mypy `--strict` on `src/`; no untyped defs, no bare `dict`
  (use `dict[str, X]`), no un-coded `# type: ignore` (write `# type: ignore[code]`).
  `benchmark/` runs a relaxed rung (see pyproject) — a ratchet target, not a licence to skip types.
- **Short functions.** ≤ 40 statements, cyclomatic ≤ 10, ≤ 6 args, ≤ 12 branches
  (blocking on `src/`). Extract, don't inline-grow.
- **No `sys.path` mutation** (ruff `TID251` + `SH003`). Use absolute imports.
- **No module-level mutable global state** (ruff `PLW0603` + `SH001`). Inject state
  or use a class. Lazy singletons need an explicit opt-out (below).
- **Docstrings ≤ 3 lines** (`SH002`, advisory). One intent line; put detail in prose near the code.
- **No `print` in `src/`** (ruff `T20`) — use `logging`. Benchmark/CLI stdout is fine.
- Naming (`N`), bug patterns (`B`), no commented-out code (`ERA001`), no `Any` in
  signatures (`ANN401`). Ruff auto-fixes imports/formatting — don't hand-fix those.

## Escape hatches (explicit + greppable — never silent)

- ruff: `# noqa: <CODE> (reason)` on the line.
- custom checks: `# noqa: SH001` / `SH002` / `SH003` (declared `external` in ruff so they coexist).
- dead code: add a documented entry to `whitelist_vulture.py`.
- A ceiling change is a reviewed edit to `pyproject.toml`, not a per-file workaround.

## Adding a routing strategy

Copy `benchmark/routing/strategies/_template.py` — it shows the canonical shape:
fully typed, short methods, no module globals, subclass `Strategy`, implement
`name` + `select`. Fill the blanks; don't invent structure.
