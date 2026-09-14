"""One canonicaliser: model_validity and model_universe must not fork on a label.

`model_validity._bare`/`_resolve` used to return a channel-prefixed listing unchanged while
`model_universe.display_name`/`canonical_label` stripped it, so the identity a figure keyed on
could differ from the identity the display path rendered. Both now route through the shared
`display_name`, and this pins it over the registry, overlay and identity maps.
"""

from __future__ import annotations

import csv
from contextlib import suppress

import yaml

from benchmark import config
from benchmark.routing import model_universe, model_validity
from benchmark.routing.scripts.scan_free_models import load_identity

OVERLAY = "configs/free-tier/overlay.yaml"


def _registry_and_identity_labels() -> set[str]:
    """Canonical slugs from the paid registry and the curated identity map."""
    labels: set[str] = set()
    with suppress(Exception):
        labels |= {str(name) for name in config.load_pricing()}
    identity = load_identity()
    labels |= set(identity.entries)
    labels |= set(identity.version_aliases)
    labels |= set(identity.version_aliases.values())
    labels.discard("")
    return labels


def _overlay_rows() -> list[tuple[str, str]]:
    with open(OVERLAY, encoding="utf-8") as handle:
        overlay = yaml.safe_load(handle)
    return [
        (str(name), str(row.get("version") or ""))
        for name, row in overlay.get("models", {}).items()
    ]


def _corpus_labels() -> set[str]:
    labels: set[str] = set()
    for path in (config.results_csv_path(), config.free_results_csv_path()):
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                labels.add(str(row.get("lane") or row.get("model") or ""))
                labels.add(str(row.get("model") or ""))
                version = str(row.get("model_version") or "")
                if version:
                    labels.add(version)
    labels.discard("")
    return labels


def _fixpoint(label: str) -> bool:
    return model_universe.display_name(label) == label


class TestValidityCanonicaliserIsTheSharedOne:
    def test_every_evidence_identity_renders_as_the_universe_labels_it(self):
        # The join key model_validity hands the universe must equal what the universe renders
        # for the same listing, or a figure keyed on one drops a model shown by the other.
        evidence = model_validity.gather_evidence()
        for listing, identity in evidence.identities.items():
            assert model_universe.display_name(identity) == model_universe.canonical_label(
                listing
            ), (
                listing,
                identity,
            )

    def test_registry_and_identity_slugs_are_display_stable(self):
        labels = _registry_and_identity_labels() | _corpus_labels()
        for label in sorted(labels):
            bare = model_validity._bare(label)
            assert _fixpoint(bare), (label, bare)
            resolved = model_validity._resolve(label, label)
            assert _fixpoint(resolved), (label, resolved)

    def test_overlay_rows_resolve_through_their_declared_version(self):
        for name, version in _overlay_rows():
            if not version:
                continue
            resolved = model_validity._resolve(name, version)
            assert resolved == model_universe.display_name(version), (name, version)
            assert _fixpoint(resolved), (name, version, resolved)
