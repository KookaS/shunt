#!/usr/bin/env python3
"""SH009: every committed figure has a manifest row and a docs section, and vice versa."""

# The canvas carries a claim, a subtitle and at most one red caveat (SH007 forces every
# figure through benchmark/plot_frame.py, which draws exactly that). Everything a reader
# needs beyond it — how to read the axes, what to look for, the terms, the method, every
# limitation — lives in the half's docs page (see `_HALVES`).
#
# That split only works if a half's figures, manifest and prose cannot drift apart, so this gate
# holds each half in a BIJECTION: PNG in docs/assets/figures/<half>/ <-> row in the half's
# <package>/figures.json <-> section in the half's doc. One-way presence would miss the failure
# that actually bites — a RETIRED figure whose docs section survives it, describing a plot nobody
# can look at. It also checks
# the five rendered strings are byte-identical between the manifest and the prose — title,
# subtitle, caveat, NOTES and LIMITS — so the doc quotes the figure rather than an older draft
# of it.
# The per-model/per-strategy note rows carry every data-adjacent claim the canvas does not, so
# they are the strings most likely to be hand-corrected in one place and left stale in the
# other; byte-locking them is what makes a hand-written correction a gate failure instead of a
# silent drift.
#
# Semantic freshness is NOT in scope and is not checkable here; that is the docs-drift
# subagent's job. This gate proves the sections exist, are complete, and quote the canvas.

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _shared import Finding

_CODE = "SH009"
_ROOT = Path(__file__).resolve().parents[2]

# The whole-tree scan is the point (same rationale as SH004-SH006, NOT SH007's
# `types: [python]`): the failure this catches is a DELETION, and no staged-file
# filter ever sees a file that is no longer there.
#
# Each row is (half, package, doc): the package is an arbitrary repo-relative directory holding
# that half's `figures.json` — it is NOT required to sit under benchmark/, and every consumer
# below reads it from the row rather than rebuilding a path. Halves are open-ended; nothing here
# assumes how many there are.
_HALVES: tuple[tuple[str, str, str], ...] = (
    ("routing", "benchmark/routing", "docs/routing.md"),
    ("escalation", "benchmark/escalation", "docs/escalation.md"),
    ("inference", "src/shunt/inspect/inference", "docs/inference.md"),
    # The illustrative half. Its own row is what enforces the one rule the synthetic figures
    # live under: never mixed with a measurement figure. Because membership is keyed on the
    # half, a demo PNG dropped into docs/assets/figures/inference/ is an orphan finding rather
    # than a figure that quietly acquires a measurement page's credibility.
    ("demo", "benchmark/demo", "docs/inference-demo.md"),
)

# The PNGs live inside the published docs tree so the docs can link them relatively, ONE
# SUBDIRECTORY PER HALF (`docs/assets/figures/<half>/`). So membership is claimed twice — by
# the directory a PNG sits in and by the half's `figures.json` — and `_check_orphans` is what
# makes the two agree: it walks the whole figures tree, so a PNG left in the old flat root, or
# dropped into the wrong half's directory, is a finding rather than an invisible file.
_FIGURES_ROOT = "docs/assets/figures"


def figures_dir(half: str) -> str:
    """A half's committed PNG directory, repo-relative."""
    return f"{_FIGURES_ROOT}/{half}"


_MAX_CAVEAT = 120
_README_FIGURE_BUDGET = 3

_SECTION = re.compile(r"^###\s+(?P<title>.+?)\s+\{#fig-(?P<slug>[a-z0-9-]+)\}\s*$", re.M)
_IMAGE = re.compile(r"^!\[[^\]]*\]\((?P<url>\S+?\.png)\)\s*$", re.M)
_SUBTITLE = re.compile(r"^\*(?P<text>[^*].*?)\*\s*$", re.M)
_CAVEAT = re.compile(r"^>\s+\*\*Caveat\.\*\*\s+(?P<text>.+?)\s*$", re.M)
_BLOCK = re.compile(r"^\*\*(?P<label>Reading|What to look for|Terms|Notes|Limits)\.\*\*", re.M)
# A Notes block runs from its label to the next `**Label.**` block (or the section end); it may
# legitimately span lines, because every note is joined onto its own line.
_NOTES_TEXT = re.compile(r"^\*\*Notes\.\*\*\s*(?P<text>.*?)(?=^\*\*|\Z)", re.M | re.S)
# The Limits block, same shape. It is normally the LAST block of a section, so it runs to the
# section end and swallows the trailing `<!-- n: … -->` provenance comment the renderer emits;
# `_HTML_COMMENT` is stripped before the comparison so the byte-lock compares prose to prose.
_LIMITS_TEXT = re.compile(r"^\*\*Limits\.\*\*\s*(?P<text>.*?)(?=^\*\*|\Z)", re.M | re.S)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)

_REQUIRED_BLOCKS = ("Reading", "What to look for")
_BLOCK_ORDER = ("Reading", "What to look for", "Terms", "Notes", "Limits")


def slug_of(png: str) -> str:
    """The docs anchor for a figure: `kill_gate.png` -> `kill-gate`."""
    return png.removesuffix(".png").replace("_", "-")


@dataclass(frozen=True)
class Section:
    """One parsed `### … {#fig-slug}` block of a half's docs page."""

    slug: str
    title: str
    line: int
    body: str


def _sections(text: str) -> dict[str, Section]:
    """Parse the '## Figures' region into one Section per '### … {#fig-slug}' heading."""
    start = text.find("\n## Figures")
    if start < 0:
        return {}
    region = text[start:]
    # A later H2 ends the figure region.
    nxt = re.search(r"^##\s+(?!Figures)", region[1:], re.M)
    if nxt:
        region = region[: nxt.start() + 1]
    offset = start
    found: dict[str, Section] = {}
    matches = list(_SECTION.finditer(region))
    for i, m in enumerate(matches):
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(region)
        found[m.group("slug")] = Section(
            slug=m.group("slug"),
            title=m.group("title").strip(),
            line=text.count("\n", 0, offset + m.start()) + 1,
            body=region[m.end() : body_end],
        )
    return found


def _check_section(
    doc: str, section: Section, half: str, png: str, row: dict[str, Any]
) -> list[Finding]:
    """The section must be complete, ordered, and quote what the canvas renders."""
    out: list[Finding] = []

    def bad(message: str) -> None:
        out.append(Finding(doc, section.line, 0, message))

    if section.title != row.get("title", ""):
        bad(
            f"section {{#fig-{section.slug}}} heading does not match the rendered title.\n"
            f"    canvas: {row.get('title', '')!r}\n"
            f"    docs:   {section.title!r}"
        )

    # The link must carry the half's subdirectory, not just the basename: the docs are
    # rendered by mkdocs from `docs/`, so `assets/figures/<half>/<png>` is the only relative
    # URL that resolves — and it is the one thing that would have caught the flat-path link
    # left behind by the per-half split.
    want = f"{half}/{png}"
    image = _IMAGE.search(section.body)
    if image is None:
        bad(f"section {{#fig-{section.slug}}} has no image line — add ![alt](<url to {want}>)")
    elif not image.group("url").endswith(want):
        bad(
            f"section {{#fig-{section.slug}}} links {image.group('url')!r}; "
            f"expected a url ending in {want!r}"
        )

    subtitle = _SUBTITLE.search(section.body)
    expected_subtitle = row.get("subtitle", "")
    if subtitle is None:
        bad(f"section {{#fig-{section.slug}}} has no italic subtitle line: *{expected_subtitle}*")
    elif subtitle.group("text").strip() != expected_subtitle:
        bad(
            f"section {{#fig-{section.slug}}} subtitle does not match the canvas.\n"
            f"    canvas: {expected_subtitle!r}\n"
            f"    docs:   {subtitle.group('text').strip()!r}"
        )

    out.extend(_check_caveat(doc, section, row))
    out.extend(_check_notes(doc, section, row))
    out.extend(_check_limits(doc, section, row))
    out.extend(_check_blocks(doc, section, row))
    return out


def _check_notes(doc: str, section: Section, row: dict[str, Any]) -> list[Finding]:
    """The docs '**Notes.**' block must quote the manifest's notes, one note per line."""
    # Notes are the manifest field with the least structural pressure: title/subtitle/caveat
    # are all checked, but a per-model note row can be rewritten in the docs with nothing red
    # — which is exactly how the note rows went factually wrong while every other gate stayed
    # green. Joining on the newline keeps each note's provenance visible in the diff.
    out: list[Finding] = []
    notes = row.get("notes") or []
    expected = "\n".join(str(n) for n in notes) if notes else None
    block = _NOTES_TEXT.search(section.body)
    if expected and block is None:
        out.append(
            Finding(
                doc,
                section.line,
                0,
                f"section {{#fig-{section.slug}}} has no '**Notes.**' block but the figure "
                f"renders {len(notes)} note(s)",
            )
        )
    elif expected and block and block.group("text").strip() != expected:
        out.append(
            Finding(
                doc,
                section.line,
                0,
                f"section {{#fig-{section.slug}}} Notes block does not quote the canvas notes.\n"
                f"    canvas: {expected!r}\n"
                f"    docs:   {block.group('text').strip()!r}",
            )
        )
    elif expected is None and block is not None:
        out.append(
            Finding(
                doc,
                section.line,
                0,
                f"section {{#fig-{section.slug}}} documents notes the figure does not render "
                f"— delete the Notes block or add the notes to the FigureSpec",
            )
        )
    return out


def _check_limits(doc: str, section: Section, row: dict[str, Any]) -> list[Finding]:
    """The docs '**Limits.**' block must quote the manifest's limitations, space-joined."""
    # Byte-locked for the same reason the notes are, and it is the field where the drift was
    # actually found: the gate used to check only that a Limits block EXISTED when the manifest
    # carried limitations, so every WORD inside it was unenforced. Two stale-in-production
    # defects lived in exactly that gap — a docs page publishing an imputed-cell count and a
    # sweep argmax the manifests no longer say. The join is a single space because that is what
    # `shunt.inspect.inference.docs_section` renders ("**Limits.** " + " ".join(...)); the notes
    # join on a newline, and the two must not be conflated.
    out: list[Finding] = []
    limitations = row.get("limitations") or []
    expected = " ".join(str(item) for item in limitations) if limitations else None
    block = _LIMITS_TEXT.search(section.body)
    found = _HTML_COMMENT.sub("", block.group("text")).strip() if block else None
    if expected and block is None:
        out.append(
            Finding(
                doc,
                section.line,
                0,
                f"section {{#fig-{section.slug}}} is missing '**Limits.**' but the figure "
                f"records {len(limitations)} limitation(s)",
            )
        )
    elif expected and found != expected:
        out.append(
            Finding(
                doc,
                section.line,
                0,
                f"section {{#fig-{section.slug}}} Limits block does not quote the canvas "
                f"limitations.\n"
                f"    canvas: {expected!r}\n"
                f"    docs:   {found!r}",
            )
        )
    elif expected is None and block is not None:
        out.append(
            Finding(
                doc,
                section.line,
                0,
                f"section {{#fig-{section.slug}}} documents limitations the figure does not "
                f"record — delete the Limits block or add them to the FigureSpec",
            )
        )
    return out


def _check_caveat(doc: str, section: Section, row: dict[str, Any]) -> list[Finding]:
    out: list[Finding] = []
    caveat = _CAVEAT.search(section.body)
    expected = row.get("caveat")
    if expected and caveat is None:
        out.append(
            Finding(
                doc,
                section.line,
                0,
                f"section {{#fig-{section.slug}}} has no '> **Caveat.**' line but the figure "
                f"renders one: {expected!r}",
            )
        )
    elif expected and caveat and caveat.group("text").strip() != expected:
        out.append(
            Finding(
                doc,
                section.line,
                0,
                f"section {{#fig-{section.slug}}} caveat does not match the canvas.\n"
                f"    canvas: {expected!r}\n"
                f"    docs:   {caveat.group('text').strip()!r}",
            )
        )
    elif expected is None and caveat is not None:
        out.append(
            Finding(
                doc,
                section.line,
                0,
                f"section {{#fig-{section.slug}}} documents a caveat the figure does not render "
                f"— delete the blockquote or add the caveat to the FigureSpec",
            )
        )
    if expected and len(expected) > _MAX_CAVEAT:
        out.append(
            Finding(
                doc,
                section.line,
                0,
                f"section {{#fig-{section.slug}}} caveat is {len(expected)} chars, max "
                f"{_MAX_CAVEAT}",
            )
        )
    return out


def _check_blocks(doc: str, section: Section, row: dict[str, Any]) -> list[Finding]:
    out: list[Finding] = []
    present = [m.group("label") for m in _BLOCK.finditer(section.body)]
    for required in _REQUIRED_BLOCKS:
        if required not in present:
            out.append(
                Finding(
                    doc,
                    section.line,
                    0,
                    f"section {{#fig-{section.slug}}} is missing the required "
                    f"'**{required}.**' block",
                )
            )
    # The Limits block's presence AND its bytes are `_check_limits`'s: an existence-only check
    # here would report the same finding twice and lock nothing the byte-lock does not.
    ranks = [_BLOCK_ORDER.index(label) for label in present if label in _BLOCK_ORDER]
    if ranks != sorted(ranks):
        out.append(
            Finding(
                doc,
                section.line,
                0,
                f"section {{#fig-{section.slug}}}: blocks are out of order. Expected "
                f"{' -> '.join(_BLOCK_ORDER)}, found {' -> '.join(present)}",
            )
        )
    return out


def _check_manifest(
    doc: str, pkg: str, half: str, pngs: list[str], rows: dict[str, Any]
) -> list[Finding]:
    """Manifest row -> PNG on disk. The other direction is `_check_orphans`, once, globally."""
    return [
        Finding(
            f"{pkg}/figures.json",
            1,
            0,
            f"entry {name!r} has no PNG in {figures_dir(half)}/ — the figure was retired (or "
            f"landed in the wrong half's directory); remove its manifest entry and its "
            f"{doc} section, or write the PNG where the half keeps its figures",
        )
        for name in sorted(set(rows) - set(pngs))
    ]


def _claimed_by_half(root: Path) -> dict[str, tuple[str, set[str]]]:
    """half -> (its package, the PNG names its figures.json claims)."""
    # The package travels with the claim so a finding can name the manifest that is actually
    # missing the row; a half's figures.json does not necessarily live under benchmark/.
    claimed: dict[str, tuple[str, set[str]]] = {}
    for half, pkg, _doc in _HALVES:
        manifest = root / pkg / "figures.json"
        rows = json.loads(manifest.read_text()).get("figures", {}) if manifest.exists() else {}
        claimed[half] = (pkg, set(rows))
    return claimed


def _check_orphans(root: Path) -> list[Finding]:
    """Every PNG in the figures tree sits in a half's directory AND is claimed by that half."""
    # Walked recursively from the ROOT, not per-half: a PNG left behind in the old flat
    # `docs/assets/figures/`, or written into the other half's directory, is exactly the
    # regression a per-half glob would never see.
    tree = root / _FIGURES_ROOT
    if not tree.is_dir():
        return []
    claimed = _claimed_by_half(root)
    homes = {root / figures_dir(half): half for half in claimed}
    out: list[Finding] = []
    for path in sorted(tree.rglob("*.png")):
        rel = path.relative_to(root).as_posix()
        half = homes.get(path.parent)
        if half is None:
            out.append(
                Finding(
                    rel,
                    1,
                    0,
                    f"sits outside every half's figure directory "
                    f"({', '.join(figures_dir(h) + '/' for h in claimed)}) — move it into the "
                    f"half that owns it, or delete it if the figure is retired",
                )
            )
        elif path.name not in claimed[half][1]:
            out.append(
                Finding(
                    rel,
                    1,
                    0,
                    f"no entry in {claimed[half][0]}/figures.json — regenerate the figures, or "
                    f"delete the PNG if the figure is retired",
                )
            )
    return out


def _check_empty_half(
    half: str, pkg: str, doc: str, manifest_path: Path, doc_path: Path
) -> list[Finding]:
    """A half with no PNGs is only consistent if nothing still describes one.

    Passing quietly here would mean deleting every figure in a half retires the whole
    docs requirement with it — the inverse of the deletion failure this gate exists for.
    """
    stale = []
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("figures"):
        stale.append(f"{pkg}/figures.json")
    if doc_path.exists() and _sections(doc_path.read_text()):
        stale.append(doc)
    if not stale:
        return []
    return [
        Finding(
            figures_dir(half),
            1,
            0,
            f"holds no {half} PNGs, but {' and '.join(stale)} still describe {half} figures "
            f"— either the figures were lost, or their records were left behind",
        )
    ]


def _check_half(root: Path, half: str, pkg: str, doc: str) -> list[Finding]:
    out: list[Finding] = []
    home = root / figures_dir(half)
    manifest_path = root / pkg / "figures.json"
    doc_path = root / doc

    rows: dict[str, Any] = {}
    if manifest_path.exists():
        rows = json.loads(manifest_path.read_text()).get("figures", {})
    # The half's live figures: a manifest row whose PNG is on disk IN THIS HALF'S directory.
    # Both conditions matter — a PNG written to the other half's home is not this half's
    # figure, and `_check_orphans` reports it from the other side.
    pngs = sorted(name for name in rows if (home / name).is_file())
    if not pngs:
        return _check_empty_half(half, pkg, doc, manifest_path, doc_path)

    out.extend(_check_manifest(doc, pkg, half, pngs, rows))

    if not doc_path.exists():
        return out + [Finding(doc, 1, 0, f"missing, but {len(pngs)} {half} figure(s) need it")]

    text = doc_path.read_text()
    sections = _sections(text)
    for png in pngs:
        slug = slug_of(png)
        if slug not in sections:
            out.append(
                Finding(
                    doc,
                    1,
                    0,
                    f"no section for {png!r} — add '### <claim title> {{#fig-{slug}}}' under "
                    f"'## Figures'",
                )
            )
        elif png in rows:
            out.extend(_check_section(doc, sections[slug], half, png, rows[png]))
    live = {slug_of(p) for p in pngs}
    for slug, section in sections.items():
        if slug not in live:
            out.append(
                Finding(
                    doc,
                    section.line,
                    0,
                    f"section {{#fig-{slug}}} describes a figure that no longer exists — "
                    f"delete this section",
                )
            )
    return out


def _check_readme(root: Path) -> list[Finding]:
    """The top-level README shows the headline figures and links out for the rest."""
    readme = root / "README.md"
    if not readme.exists():
        return []
    out: list[Finding] = []
    text = readme.read_text()
    embeds = [
        (i + 1, m.group(1))
        for i, line in enumerate(text.splitlines())
        for m in [re.match(r"^!\[[^\]]*\]\((\S+?\.png)\)", line.strip())]
        if m
    ]
    benchmark_embeds = [(ln, url) for ln, url in embeds if _FIGURES_ROOT in url]
    if len(benchmark_embeds) > _README_FIGURE_BUDGET:
        out.append(
            Finding(
                "README.md",
                benchmark_embeds[0][0],
                0,
                f"embeds {len(benchmark_embeds)} benchmark figures; the top-level README "
                f"carries at most {_README_FIGURE_BUDGET} — move the rest to "
                f"{', '.join(doc for _half, _pkg, doc in _HALVES)}",
            )
        )
    for line, url in benchmark_embeds:
        rel = url.split(_FIGURES_ROOT + "/", 1)[1]
        if not (root / _FIGURES_ROOT / rel).exists():
            out.append(Finding("README.md", line, 0, f"embeds {url!r}, which does not exist"))
    return out


def main(argv: list[str]) -> int:
    """Scan every half plus the README; print every finding and exit non-zero on any."""
    # --root exists so the gate is testable against a fixture tree; the hook passes no
    # arguments, so the repo root IS the coverage (same shape as SH004).
    root = _ROOT
    if "--root" in argv:
        root = Path(argv[argv.index("--root") + 1]).resolve()
    findings: list[Finding] = []
    for half, pkg, doc in _HALVES:
        findings.extend(_check_half(root, half, pkg, doc))
    findings.extend(_check_orphans(root))
    findings.extend(_check_readme(root))
    for f in findings:
        print(f"{f.path}:{f.line}:{f.col}: [{_CODE} ERROR] {f.message}")  # noqa: T201
    if findings:
        print(  # noqa: T201
            f"\n{_CODE}: {len(findings)} problem(s). Every committed figure needs a "
            f"figures.json row and a docs section, and every section needs a live figure."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
