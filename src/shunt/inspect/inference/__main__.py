"""Container entrypoint: render the inference family, or print its docs sections."""

# This module is the ONLY render path the rig image has. The image carries no `benchmark/`
# tree (and SH006 forbids importing it from `src/shunt/` anyway), so everything here reads the
# ambient `OutcomeStore` and the shipped registry — nothing benchmark-side.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shunt.inspect.inference import (
    MANIFEST,
    _manifest_for,
    docs_section,
    docs_sections,
    render,
)


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m shunt.inspect.inference",
        description="Render the seven inference figures from the live outcome store.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="directory to write the PNGs into; the manifest lands beside it unless this "
        "is the committed figure directory",
    )
    parser.add_argument(
        "--emit-docs-section",
        metavar="FIGURE",
        help="print the SH009 markdown block for one figure (or `all`) from the manifest "
        "and exit without rendering",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.emit_docs_section:
        return _emit(args.emit_docs_section, args.out_dir)
    if args.out_dir is None:
        print("--out-dir is required unless --emit-docs-section is given", file=sys.stderr)
        return 2
    return _render(args.out_dir)


def _emit(figure: str, out_dir: Path | None) -> int:
    manifest = MANIFEST if out_dir is None else _manifest_for(out_dir)
    if not manifest.exists():
        print(f"no manifest at {manifest} — render the figures first", file=sys.stderr)
        return 1
    sections = docs_sections(manifest)
    if figure == "all":
        print(sections, end="")
        return 0
    name = figure if figure.endswith(".png") else f"{figure}.png"
    rows = json.loads(manifest.read_text()).get("figures", {})
    if name not in rows:
        print(f"{name} has no manifest row in {manifest}", file=sys.stderr)
        return 1
    print(docs_section(name, rows[name]), end="")
    return 0


def _render(out_dir: Path) -> int:
    from shunt.db.store import OutcomeStore

    report = render(OutcomeStore(), out_dir)
    for path in report.figures:
        print(path)
    print(f"manifest: {report.manifest}")
    if report.inadmissible is not None:
        # Non-zero: the off-policy figure did not ship, and a caller that treats a partial
        # family as a success is exactly how an unproven estimator gets quoted.
        print(f"off-policy figure refused: {report.inadmissible}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
