#!/usr/bin/env node
// Gate: every ```mermaid fence in a Markdown file parses under the real mermaid grammar.
//
// The docs site renders mermaid client-side (mkdocs-material + pymdownx.superfences),
// so a syntax error is invisible to every gate we have: ruff, mypy, pytest and the
// docs-integrity scanner all pass, mkdocs builds clean, and the defect only appears as
// a red error box in a reader's browser. This runs the SAME parser the browser runs
// (mermaid's own `mermaid.parse`, driven headlessly under a jsdom shim), so a pass here
// means the published page will at least not fail to parse.
//
// WHAT IT CANNOT SEE, stated so nobody reads a pass as more than it is:
//   * render-time failures. `mermaid.parse` stops after the grammar+semantic pass; it
//     never runs layout. A diagram that parses but throws inside dagre/cytoscape, or
//     references a missing icon pack, an unregistered `layout:`/`theme:`, or an
//     external shape, passes here and still breaks in the browser. Only a real
//     browser render (mmdc/puppeteer) sees those, at ~100x the cost.
//   * anything visual. Overlapping nodes, an edge routed through a label, text
//     clipped out of a node, a 40-node graph nobody can read — all parse fine.
//   * truth. It cannot tell you the diagram describes the wrong architecture, names a
//     module that no longer exists, or contradicts the prose above it. That is the
//     `docs-drift` reviewer's job, not this one's.
//   * mermaid-version drift. It validates against the mermaid pinned in the hook's
//     `additional_dependencies`, NOT against whatever mermaid.js the docs site loads
//     from its CDN. Keep the two in the same major, or a pass here can still be a
//     failure there (and vice-versa).
//   * non-mermaid fences, and mermaid outside a fence (raw <div class="mermaid">).
//
// Reported line numbers are the fence's own line plus mermaid's in-block line number.
// Mermaid may normalise the block (directives, in-diagram `---` frontmatter) before
// reporting, so on those blocks the line can be off by the number of stripped lines.

import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

// pre-commit installs `additional_dependencies` into its node env's GLOBAL
// lib/node_modules and points NODE_PATH at it. CommonJS honours NODE_PATH; ESM
// bare-specifier resolution does not — it only walks node_modules upward from this
// file, which under pre-commit contains neither dep. So resolve to an absolute path
// through NODE_PATH first (require.resolve applies the package's exports map, so this
// works for mermaid's ESM-only entry too), and fall back to a plain bare import for a
// local `npm install` checkout.
const require = createRequire(import.meta.url);
const NODE_PATH_DIRS = (process.env.NODE_PATH ?? '')
  .split(process.platform === 'win32' ? ';' : ':')
  .filter(Boolean);

async function importDep(name) {
  if (NODE_PATH_DIRS.length > 0) {
    try {
      return await import(pathToFileURL(require.resolve(name, { paths: NODE_PATH_DIRS })).href);
    } catch {
      // fall through to normal resolution
    }
  }
  return await import(name);
}

// Importing jsdom + mermaid costs ~0.9s; extracting fences costs ~1ms. Most commits
// touch no diagram at all, so both imports are deferred until a mermaid fence is
// actually found — that is the difference between a hook people keep and a hook
// people bypass with --no-verify.
async function loadMermaid() {
  // mermaid pulls in DOMPurify at import time, which needs a real `window` with
  // createHTMLDocument. Without it every parse throws "DOMPurify.addHook is not a
  // function" — i.e. the gate would fail EVERY block, including valid ones. jsdom is
  // the dependency that makes the pass side of this gate meaningful.
  const { JSDOM } = await importDep('jsdom');
  const dom = new JSDOM('<!DOCTYPE html><body></body>');
  globalThis.window = dom.window;
  globalThis.document = dom.window.document;
  // Node >=21 defines `navigator` as a getter-only global; plain assignment throws.
  Object.defineProperty(globalThis, 'navigator', { value: dom.window.navigator, configurable: true });
  globalThis.Element = dom.window.Element;
  globalThis.HTMLElement = dom.window.HTMLElement;
  globalThis.SVGElement = dom.window.SVGElement;
  globalThis.Node = dom.window.Node;
  globalThis.DocumentFragment = dom.window.DocumentFragment;
  globalThis.getComputedStyle = dom.window.getComputedStyle;
  globalThis.MutationObserver = dom.window.MutationObserver;
  globalThis.requestAnimationFrame = (cb) => setTimeout(cb, 0);
  return (await importDep('mermaid')).default;
}

// --- fence extraction ----------------------------------------------------------
// CommonMark fenced-code scanning, which is what both python-markdown/superfences and
// GitHub implement. Handled deliberately:
//   * ``` and ~~~ fences
//   * fences of 4+ chars, so a ````markdown block QUOTING a ```mermaid example is
//     content of the outer fence and is correctly NOT validated
//   * indented fences (a diagram inside a list item or admonition), with the opening
//     indent stripped from each content line
//   * superfences' brace info string: ```{.mermaid ...} as well as ```mermaid title="x"
//   * an unterminated fence at EOF (treated as running to EOF, as CommonMark does)
const FENCE_OPEN = /^(\s*)(`{3,}|~{3,})\s*(.*)$/;

/** @returns {{lang: string, startLine: number, text: string}[]} */
export function extractFences(source) {
  const lines = source.split('\n');
  const blocks = [];
  let open = null; // {char, len, indent, lang, startLine, body[]}

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (open) {
      const close = line.match(/^(\s*)(`{3,}|~{3,})\s*$/);
      if (close && close[2][0] === open.char && close[2].length >= open.len) {
        blocks.push({ lang: open.lang, startLine: open.startLine, text: open.body.join('\n') });
        open = null;
        continue;
      }
      // Strip the opening fence's indent, no more (a deeper-indented line keeps its
      // extra indent, which matters for mermaid `subgraph` bodies).
      open.body.push(line.slice(0, open.indent).trim() === '' ? line.slice(open.indent) : line);
      continue;
    }
    const m = line.match(FENCE_OPEN);
    if (!m) continue;
    const [, indent, fence, info] = m;
    // A backtick fence's info string may not contain a backtick (CommonMark), which is
    // what stops an inline `code span` from being read as a fence.
    if (fence[0] === '`' && info.includes('`')) continue;
    open = {
      char: fence[0],
      len: fence.length,
      indent: indent.length,
      lang: normaliseLang(info),
      startLine: i + 1, // 1-based line of the opening fence
      body: [],
    };
  }
  if (open) blocks.push({ lang: open.lang, startLine: open.startLine, text: open.body.join('\n') });
  return blocks;
}

function normaliseLang(info) {
  const t = info.trim();
  if (!t) return '';
  if (t.startsWith('{')) {
    // superfences brace form: {.mermaid #id attr=1} -> mermaid
    const dot = t.slice(1).match(/\.([\w-]+)/);
    return dot ? dot[1].toLowerCase() : '';
  }
  return t.split(/[\s,;]/)[0].toLowerCase();
}

// --- validation ----------------------------------------------------------------
function formatError(err, block) {
  const msg = String(err && err.message ? err.message : err);
  // Mermaid's jison errors lead with "Parse error on line N:" (grammar) or "Lexical
  // error on line N." (tokeniser), numbered relative to the block body. Rewrite to an
  // absolute file line so an editor can jump straight to it.
  const m = msg.match(/^((?:Parse|Lexical) error on line )(\d+)([:.])/);
  if (!m) return { line: block.startLine, msg };
  return {
    line: block.startLine + Number(m[2]),
    msg: msg.replace(m[0], `${m[1]}${m[2]} of the block${m[3]}`),
  };
}

async function main() {
  const files = process.argv.slice(2);
  if (files.length === 0) return 0;

  let mermaid = null;
  let failures = 0;
  let checked = 0;

  for (const file of files) {
    let source;
    try {
      source = readFileSync(file, 'utf8');
    } catch (e) {
      console.error(`${file}: cannot read (${e.message})`);
      failures++;
      continue;
    }
    for (const block of extractFences(source)) {
      if (block.lang !== 'mermaid') continue;
      checked++;
      if (block.text.trim() === '') {
        console.error(`${file}:${block.startLine}: empty mermaid block`);
        failures++;
        continue;
      }
      // Deferred until the first real block, so a commit touching no diagram never
      // pays the ~0.9s jsdom+mermaid import.
      mermaid ??= await loadMermaid();
      try {
        await mermaid.parse(block.text);
      } catch (err) {
        const { line, msg } = formatError(err, block);
        console.error(`${file}:${line}: mermaid parse error`);
        for (const l of msg.split('\n')) console.error(`    ${l}`);
        failures++;
      }
    }
  }

  if (failures > 0) {
    console.error(`\n${failures} of ${checked} mermaid block(s) failed to parse.`);
    return 1;
  }
  if (process.env.MERMAID_CHECK_VERBOSE) {
    console.log(`${checked} mermaid block(s) parsed cleanly.`);
  }
  return 0;
}

process.exitCode = await main();
