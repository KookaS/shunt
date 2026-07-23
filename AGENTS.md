# AGENTS.md — how to write code in this repo

Rules a human or model must follow here. This file says *why*; the linter is the
backstop that says *whether* — if a rule matters, CI enforces it
(`pre-commit run --all-files`), so get it right the first time.

It is **thin and pointer-first by design.** For anything concrete — layout, docs,
config, examples, the exact lint ceilings — follow the *entrypoint* in the map
below rather than trusting a copy here to stay current. When you add an area, add a
pointer row, not a paragraph. That is what lets this file scale with the codebase
instead of rotting.

## Where to look (entrypoints — the source of truth, not this file)

Follow the entrypoint; it is the maintained copy. This table only routes you there,
so it stays correct as the codebase grows.

| Need | Start here |
|------|-----------|
| What Shunt is · status · quick start | `README.md` |
| The repo & package layout | `README.md` → "Repository layout" |
| **The docs map** — every doc, in order | `mkdocs.yml` (`nav:`) and `docs/index.md` (Contents) |
| How the pieces fit at runtime | `docs/architecture.md` |
| The feedback / learning loop (Context → Action → Feedback) | `docs/feedback.md` |
| Configure providers, models, the router, the embedder | `docs/configuration.md` + `src/shunt/config/{models,router,embedding}.yaml` |
| Add a provider or model | `examples/providers/README.md` — registry is `src/shunt/config/models.yaml`, **row order is semantic** |
| Hook up a tool (Claude Code, opencode, aider, n8n, …) | `examples/integrations/README.md` + the shared handshake harness (`tests/integrations/`) |
| The benchmark / eval harness | `docs/benchmark.md`, `docs/benchmark-design.md`, `benchmark/` |
| Add a routing strategy | `benchmark/routing/strategies/_template.py` — copy it, don't invent structure |
| The exact lint / type ceilings | `pyproject.toml` (the one manifest) |
| The custom AST gates (`SH0xx`) | `tools/lint/` |

New area? Add a row here pointing at its entrypoint — never restate what the
entrypoint already says.

### The tree, one line each

- `src/shunt/` — the shipped router (product); **strictest rules apply here**.
- `benchmark/` — the eval harness (not installed; tests reach it via pytest
  `pythonpath = ["."]`). Absolute imports only: `from benchmark import config`,
  never `sys.path` hacks.
- `tools/lint/` — custom `SH0xx` AST checks · `tests/` — pytest suite · `examples/`
  — provider + integration configs.

Install once: `pip install -e '.[dev,benchmark]'` — then `benchmark` imports resolve
everywhere with no path munging.

**Run everything through `uv run` — never bare `python3`.** In a worktree, bare
`python3 -m pytest` resolves `import shunt` to a *different* worktree's source and
reports pass/fail for code you aren't editing. Check with
`uv run python -c "import shunt.models as m; print(m.__file__)"`.

## The rules — the *why* (the exact ceilings live in `pyproject.toml`; ruff + mypy enforce them)

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
