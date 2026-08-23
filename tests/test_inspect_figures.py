"""`shunt inspect` figures: PCA positive control, end-to-end render, empty corpus, CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shunt.cli import _inspect, main
from shunt.db.store import OutcomeStore
from shunt.inspect import figures, plot_frame
from shunt.inspect.figures import project, render

_DIM = 16


def _canvas_texts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every string the next `plot_frame.save` puts on the canvas, band text included."""
    strings: list[str] = []
    original = plot_frame.save

    def spy(fig: Any, *args: Any, **kwargs: Any) -> Any:
        saved = original(fig, *args, **kwargs)
        strings.extend(t.get_text() for ax in fig.axes for t in ax.texts)
        strings.extend(t.get_text() for t in fig.texts)
        return saved

    monkeypatch.setattr(plot_frame, "save", spy)
    return strings


def _store_session(
    store: OutcomeStore,
    sid: str,
    base: float,
    model: str,
    timestamp: str,
    cost: float = 0.01,
) -> None:
    store.store_session(
        session_id=sid,
        prompt_text=f"task {sid}",
        embedding=np.full(_DIM, base, dtype=np.float32),
        model_chosen=model,
        cost=cost,
        cache_stats={},
        duration=1.0,
        timestamp=timestamp,
    )


def _seed(store: OutcomeStore) -> None:
    """Live + `bench:` seeded sessions in two clusters; three labeled, one open."""
    _store_session(store, "bench:seed-1", 1.0, "qwen3.7-plus", "2026-01-01T00:00:00")
    _store_session(store, "sess-live-a", 1.0, "qwen3.7-plus", "2026-01-01T00:01:00")
    _store_session(store, "sess-live-b", -1.0, "cheap", "2026-01-01T00:02:00")
    _store_session(store, "sess-live-open", 0.0, "mid", "2026-01-01T00:03:00")
    for sid, outcome in (
        ("bench:seed-1", "success"),
        ("sess-live-a", "success"),
        ("sess-live-b", "failure"),
    ):
        store.store_outcome(
            session_id=sid,
            tier1_outcome=outcome,
            tier1_confidence=0.9,
            tier2_outcome=outcome,
            tier2_confidence=0.9,
        )


def test_project_separates_two_clusters() -> None:
    """Positive control: PC1 must split two well-separated clusters, variance must match SVD."""
    a = np.zeros((12, 8), dtype=np.float64)
    a[:, 0] = 1.0
    a += np.arange(12)[:, None] * 0.01
    b = a.copy()
    b[:, 0] = 11.0
    vectors = np.vstack([a, b])

    projected, explained = project(vectors)

    assert projected.shape == (24, 2)
    assert explained.shape == (2,)
    group_a = projected[:12, 0]
    group_b = projected[12:, 0]
    assert min(group_a) > max(group_b) or min(group_b) > max(group_a)

    centered = vectors - vectors.mean(axis=0)
    _u, singular, _vt = np.linalg.svd(centered, full_matrices=False)
    expected = (singular[:2] ** 2) / np.sum(singular**2)
    assert np.allclose(explained, expected)


def test_render_writes_a_nonempty_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end against a real temp store: PNG exists, census numbers add up."""
    store = OutcomeStore(db_path=str(tmp_path / "outcomes.db"))
    _seed(store)
    texts = _canvas_texts(monkeypatch)
    try:
        result = render(store, tmp_path / "figs")
    finally:
        store.close()

    # The census panel is the figure's claim; a returned count that the canvas contradicts is
    # the failure `st_size > 0` cannot see.
    assert "Shunt inference corpus — live outcome store" in texts
    assert any("embedded:       4" in text for text in texts)
    assert any("seeded (bench:):1    live: 3" in text for text in texts)

    assert result.figure_path is not None
    assert result.figure_path.exists()
    assert result.figure_path.stat().st_size > 0
    assert result.n_embedded == 4
    assert result.n_labeled == 3
    assert result.n_tier2 == 3
    assert result.seeded == 1
    assert result.live == 3
    assert result.pc1_share is not None
    assert result.pc2_share is not None


def test_render_empty_corpus_is_a_clean_noop(tmp_path: Path) -> None:
    """No sessions: no figure, no crash, nothing written."""
    store = OutcomeStore(db_path=str(tmp_path / "empty.db"))
    try:
        result = render(store, tmp_path / "figs")
    finally:
        store.close()

    assert result.figure_path is None
    assert result.n_embedded == 0
    assert not (tmp_path / "figs").exists()


def test_render_single_session_renders_the_cold_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One session: no PCA, but a message + census figure still saves."""
    store = OutcomeStore(db_path=str(tmp_path / "one.db"))
    _store_session(store, "s1", 1.0, "m", "2026-01-01T00:00:00")
    texts = _canvas_texts(monkeypatch)
    try:
        result = render(store, tmp_path / "figs")
    finally:
        store.close()

    # The cold canvas must say WHY there is no projection; the live-corpus title on this
    # figure would claim a projection that was never drawn.
    assert "Shunt inference corpus — too small to project" in texts
    assert "corpus too small — n=1 (still cold-start?)" in texts

    assert result.figure_path is not None
    assert result.figure_path.exists()
    assert result.pc1_share is None
    assert result.n_embedded == 1


def test_render_prompt_pin_uses_the_shipped_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The --prompt path embeds via a faked shipped Embedder (no real ONNX, SH008-exempt)."""
    store = OutcomeStore(db_path=str(tmp_path / "pin.db"))
    _seed(store)

    class _FakeEmbedder:
        def embed(self, text: str) -> np.ndarray:
            return np.full(_DIM, 0.25, dtype=np.float32)

    monkeypatch.setattr("shunt.router.embedder.Embedder", _FakeEmbedder)
    captured: dict[str, object] = {}

    def _capture(ax: object, data: object) -> None:
        captured["data"] = data

    monkeypatch.setattr(figures, "_draw_projection", _capture)
    try:
        result = render(store, tmp_path / "figs", prompt="make a payment")
        data = captured["data"]
        assert data is not None
        pin_point = data.pin_point
        assert pin_point is not None, "the pin must be projected in the corpus basis"
        px, py = pin_point
        assert np.isfinite(px) and np.isfinite(py)
        xmin, xmax = float(data.projected[:, 0].min()), float(data.projected[:, 0].max())
        ymin, ymax = float(data.projected[:, 1].min()), float(data.projected[:, 1].max())
        margin = 1e-9 * max(1.0, float(np.abs(data.projected).max()))
        assert xmin - margin <= px <= xmax + margin
        assert ymin - margin <= py <= ymax + margin
    finally:
        store.close()

    assert result.figure_path is not None
    assert result.pin_warning is None


def test_render_prompt_pin_failure_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Embedder unavailable: warn and keep rendering, never crash."""
    store = OutcomeStore(db_path=str(tmp_path / "pinfail.db"))
    _seed(store)

    class _BrokenEmbedder:
        def embed(self, text: str) -> np.ndarray:
            raise RuntimeError("offline")

    monkeypatch.setattr("shunt.router.embedder.Embedder", _BrokenEmbedder)
    try:
        result = render(store, tmp_path / "figs", prompt="hello")
    finally:
        store.close()

    assert result.figure_path is not None
    assert result.pin_warning is not None
    assert "unavailable" in result.pin_warning


def test_render_knn_overlay_consults_index_and_skips_self(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--k overlay: index consulted, no self-edge, the overlay origin is the query session."""
    store = OutcomeStore(db_path=str(tmp_path / "knn.db"))
    _seed(store)
    calls: list[int] = []
    real_query = store.query_index

    def _spy_query(embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        calls.append(k)
        return real_query(embedding, k)

    monkeypatch.setattr(store, "query_index", _spy_query)
    captured: dict[str, object] = {}

    def _capture(ax: object, data: object) -> None:
        captured["data"] = data

    monkeypatch.setattr(figures, "_draw_projection", _capture)
    try:
        result = render(store, tmp_path / "figs", k=3)
        query_sid = max(store.labeled_outcome_rows(), key=lambda r: r["created_at"])["session_id"]
    finally:
        store.close()

    assert result.figure_path is not None
    assert calls == [3], "store.query_index must be consulted once with the requested k"
    data = captured["data"]
    assert data is not None
    assert data.label_lines, "the kNN overlay must produce segments"
    assert data.query_idx is not None, "the overlay origin must be marked"
    assert data.ids[data.query_idx] == query_sid
    qx, qy = data.projected[data.query_idx]
    for x0, y0, x1, y1 in data.label_lines:
        assert (x0, y0) == (qx, qy)
        assert (x1, y1) != (qx, qy), "a segment must never connect a session to itself"


def test_render_knn_overlay_large_k_is_graceful(tmp_path: Path) -> None:
    """--k beyond the index size: no crash, fewer segments, figure still saves."""
    store = OutcomeStore(db_path=str(tmp_path / "knlarge.db"))
    _seed(store)
    try:
        result = render(store, tmp_path / "figs", k=100)
    finally:
        store.close()

    assert result.figure_path is not None
    assert result.figure_path.exists()


def test_inspect_cli_renders_and_prints_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = OutcomeStore(db_path=str(tmp_path / "cli.db"))
    _seed(store)
    monkeypatch.setattr("shunt.db.store.OutcomeStore", lambda *a, **k: store)

    out_dir = tmp_path / "out"
    _inspect(argparse.Namespace(output_dir=str(out_dir), prompt=None, k=5))

    out = capsys.readouterr().out
    assert "embedded: 4" in out
    assert "1 seeded vs 3 live" in out
    assert "figure:" in out
    assert (tmp_path / "out" / "inspect_corpus.png").exists()


def test_main_dispatches_inspect(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr("shunt.cli._inspect", lambda args: called.append(True))
    monkeypatch.setattr("sys.argv", ["shunt", "inspect", "--output-dir", "x", "--k", "3"])
    main()
    assert called == [True]


def test_inspect_subcommand_exists_in_the_parser() -> None:
    """The real CLI exposes `shunt inspect --output-dir/--prompt/--k` (help exits 0)."""
    proc = subprocess.run(
        [sys.executable, "-m", "shunt.cli", "inspect", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--output-dir" in proc.stdout
    assert "--prompt" in proc.stdout
    assert "--k" in proc.stdout
