"""The inference package's wiring: render(), the manifest diversion, and the docs emitter."""

# The figures themselves are covered by `test_inference_figures.py`. What is under test here is
# the integration: that all eight land in one call, that a scratch render can never reach the
# committed manifest, that an inadmissible instrument costs exactly one figure, and that the
# markdown the docs page is built from is generated from the manifest rather than retyped.

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shunt.db.store import OutcomeEvent, OutcomeStore, SessionProvenance
from shunt.inspect import inference
from shunt.inspect.inference import estimators, specs
from shunt.inspect.inference.__main__ import main

_DIM = 8


@pytest.fixture(autouse=True)
def _strict_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_PLOT_STRICT", "1")


@pytest.fixture
def seeded_store(tmp_path: Path) -> OutcomeStore:
    """A seed-only corpus — the shape the committed docs render actually sees."""
    store = OutcomeStore(
        db_path=str(tmp_path / "outcomes.db"), index_path=str(tmp_path / "index.hnsw2")
    )
    epoch = datetime(2020, 1, 1, tzinfo=UTC)
    for index in range(6):
        rng = np.random.default_rng(index)
        vector = rng.normal(size=_DIM).astype(np.float32)
        store.store_session(
            session_id=f"bench:{index}",
            prompt_text=f"prompt {index}",
            embedding=vector / float(np.linalg.norm(vector)),
            model_chosen="cheap" if index % 2 else "frontier",
            cost=0.1 * (index + 1),
            cache_stats={},
            duration=1.0,
            timestamp=epoch.isoformat(),
            decision_provenance={"selection_rule_used": "benchmark_seed"},
            provenance=SessionProvenance(selection_propensity=None, cost_known=True),
        )
        store.append_outcome_event(
            OutcomeEvent(
                session_id=f"bench:{index}",
                tier=2,
                source="benchmark_seed",
                outcome="success" if index % 3 else "failure",
                confidence=0.9,
                run_signature=f"test:{index}",
                created_at=epoch.isoformat(),
            )
        )
    return store


def _stub_certificate(monkeypatch: pytest.MonkeyPatch, *, admissible: bool) -> None:
    """Supply the instrument verdict — the real control costs ~11 s and has its own test."""
    from shunt.analysis.admissibility import admissibility_verdict

    verdict = admissibility_verdict(
        1.0 if admissible else 0.0, 0.0, chance_level=0.0, chance_band=0.2
    )
    certificates = tuple(
        estimators.EstimatorCertificate(
            admissibility=verdict, leg=leg, estimator=name, n_rows=400, n_shuffles=200, seed=1
        )
        for leg in (estimators.ROUTING, estimators.ESCALATION)
        for name in estimators.ESTIMATORS
    )

    def _certify(store: OutcomeStore, **_kwargs: Any) -> estimators.CertifiedEstimates:
        routing, escalation = estimators.live_estimates(store)
        result = estimators.CertifiedEstimates(
            admissibility=verdict,
            certificates=certificates,
            routing=routing,
            escalation=escalation,
        )
        result.require_admissible()
        return result

    monkeypatch.setattr(estimators, "certify", _certify)


def test_render_writes_all_eight_figures_and_their_manifest(
    seeded_store: OutcomeStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_certificate(monkeypatch, admissible=True)
    report = inference.render(seeded_store, tmp_path / "figures")
    assert [path.name for path in report.figures] == [text.filename for text in specs.FIGURES]
    assert report.inadmissible is None
    rows = json.loads(report.manifest.read_text())["figures"]
    assert set(rows) == {text.filename for text in specs.FIGURES}


def test_an_inadmissible_instrument_costs_one_figure_not_the_family(
    seeded_store: OutcomeStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole point of catching it: a broken estimator must not take down the six figures
    # that read the store rather than the instrument.
    _stub_certificate(monkeypatch, admissible=False)
    report = inference.render(seeded_store, tmp_path / "figures")
    assert report.inadmissible is not None
    assert len(report.figures) == len(specs.FIGURES) - 1
    assert not (tmp_path / "figures" / specs.OPE.filename).exists()
    assert specs.OPE.filename not in json.loads(report.manifest.read_text())["figures"]


def test_a_scratch_render_cannot_reach_the_committed_manifest(tmp_path: Path) -> None:
    # Asserted through `_manifest_for`/`Family.manifest_for`, the one surviving implementation
    # of "is this the committed home". The predicate used to exist twice — a private helper
    # here and the comparison inside `Family` — and two copies of one boolean is how a scratch
    # render eventually reaches the committed manifest.
    assert inference._manifest_for(tmp_path) == tmp_path.parent / "figures.json"
    assert inference._manifest_for(inference.CANONICAL_PLOTS_DIR) == inference.MANIFEST
    assert inference.INFERENCE.manifest_for(tmp_path) == tmp_path.parent / "figures.json"
    assert inference.INFERENCE.manifest_for(inference.CANONICAL_PLOTS_DIR) == inference.MANIFEST


def test_the_committed_manifest_sits_beside_the_producer() -> None:
    assert inference.MANIFEST.parent == Path(inference.__file__).resolve().parent


def test_digest_keys_on_content_not_on_the_database_file(
    seeded_store: OutcomeStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_certificate(monkeypatch, admissible=True)
    first = inference.render(seeded_store, tmp_path / "a").data_digest
    second = inference.render(seeded_store, tmp_path / "b").data_digest
    assert first == second


def test_docs_section_is_generated_from_the_manifest_row() -> None:
    row = {
        "title": "A title",
        "subtitle": "a subtitle",
        "caveat": "a caveat",
        "reading": "how to read it",
        "goal": "what to look for",
        "terms": [["term", "meaning"]],
        "notes": ["one", "two"],
        "limitations": ["a limit"],
        "n": {"sessions": 6},
    }
    section = inference.docs_section("inference_cost.png", row)
    assert section.startswith("### A title {#fig-inference-cost}\n")
    assert "![A title](assets/figures/inference/inference_cost.png)" in section
    assert "\n*a subtitle*\n" in section
    assert "> **Caveat.** a caveat" in section
    assert "**Notes.** one\ntwo" in section
    assert "**Limits.** a limit" in section
    assert f"generated-by: {inference.GENERATOR}" in section


def test_docs_section_omits_the_blocks_the_figure_does_not_render() -> None:
    # SH009 fails a section documenting a caveat or notes the canvas never drew, so the
    # emitter must drop them rather than print an empty block.
    row = {
        "title": "T",
        "subtitle": "s",
        "caveat": None,
        "reading": "r",
        "goal": "g",
        "terms": [],
        "notes": [],
        "limitations": [],
        "n": {},
    }
    section = inference.docs_section("inference_ope.png", row)
    assert "Caveat" not in section
    assert "**Notes.**" not in section
    assert "**Limits.**" not in section


def test_emit_docs_section_prints_every_figure_in_family_order(
    seeded_store: OutcomeStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_certificate(monkeypatch, admissible=True)
    out_dir = tmp_path / "figures"
    inference.render(seeded_store, out_dir)
    assert main(["--out-dir", str(out_dir), "--emit-docs-section", "all"]) == 0
    printed = capsys.readouterr().out
    headings = [line for line in printed.splitlines() if line.startswith("### ")]
    assert headings == [f"### {text.title} {{#fig-{text.slug}}}" for text in specs.FIGURES]


def test_the_module_entrypoint_requires_an_out_dir(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "--out-dir is required" in capsys.readouterr().err


def test_docs_section_marker_names_the_rows_own_generator() -> None:
    # The emitter serves more than one manifest: the routing family's `model_grid.png` is
    # drawn by `benchmark.routing.figures.model_grid` and only its markdown block is
    # rendered here. A marker naming this module would point at an entrypoint that cannot
    # produce that figure — and SH012 would still pass it, because it checks a marker
    # EXISTS, never that it resolves to the producer.
    row = {
        "title": "T",
        "subtitle": "s",
        "reading": "r",
        "goal": "g",
        "generator": "benchmark.routing.figures.model_grid",
        "n": {"models": 8},
    }
    section = inference.docs_section("model_grid.png", row, half="routing")
    assert "<!-- generated-by: benchmark.routing.figures.model_grid -->" in section
    assert inference.GENERATOR not in section
