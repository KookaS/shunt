"""The runtime wall: only the plot frame may write a figure, whatever the call site looks like."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from benchmark import plot_frame, plot_guard


def _fig() -> Figure:
    return plot_frame.new_figure(plot_frame.SINGLE)


def test_the_guard_is_installed_for_the_whole_suite() -> None:
    """Non-vacuity floor: every assertion below is meaningless if conftest did not install."""
    assert plot_guard._INSTALLED, "tests/conftest.py must install the plot guard"


def test_a_bare_savefig_is_refused(tmp_path: Path) -> None:
    fig = _fig()
    try:
        with pytest.raises(plot_guard.FigureEscapedFrameError) as caught:
            fig.savefig(tmp_path / "escaped.png")  # noqa: SH007 (the planted violation)
    finally:
        fig.clear()
    assert "test_plot_guard.py" in str(caught.value)
    assert "plot_frame.save" in str(caught.value)
    assert not (tmp_path / "escaped.png").exists()


def test_an_aliased_savefig_is_refused(tmp_path: Path) -> None:
    """The spelling SH007 could not see until this session — the guard never had to."""
    fig = _fig()
    writer = fig.savefig  # noqa: SH007 (the planted violation)
    try:
        with pytest.raises(plot_guard.FigureEscapedFrameError):
            writer(tmp_path / "aliased.png")
    finally:
        fig.clear()
    assert not (tmp_path / "aliased.png").exists()


def test_a_getattr_reached_savefig_is_refused(tmp_path: Path) -> None:
    fig = _fig()
    try:
        with pytest.raises(plot_guard.FigureEscapedFrameError):
            getattr(fig, "sav" + "efig")(tmp_path / "dynamic.png")
    finally:
        fig.clear()
    assert not (tmp_path / "dynamic.png").exists()


def test_a_canvas_print_figure_is_refused(tmp_path: Path) -> None:
    """One layer below savefig — the writer matplotlib itself calls."""
    fig = _fig()
    try:
        with pytest.raises(plot_guard.FigureEscapedFrameError):
            fig.canvas.print_figure(tmp_path / "canvas.png")  # noqa: SH007 (planted)
    finally:
        fig.clear()
    assert not (tmp_path / "canvas.png").exists()


def test_the_frame_still_saves(tmp_path: Path) -> None:
    """The other half of non-vacuity: the guard refuses violations, not the supported path."""
    out = tmp_path / "framed.png"
    fig = _fig()
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])
    written = plot_frame.save(
        fig,
        out,
        plot_frame.FigureSpec(
            title="A framed figure", reading="The line rises.", goal="Prove the frame saves."
        ),
    )
    assert written.exists() and written.stat().st_size > 0


def test_uninstall_restores_matplotlib(tmp_path: Path) -> None:
    """Proves the refusal above comes from the patch and not from something else in the env."""
    plot_guard.uninstall()
    try:
        fig = _fig()
        fig.savefig(tmp_path / "unguarded.png")  # noqa: SH007 (guard deliberately removed)
        fig.clear()
        assert (tmp_path / "unguarded.png").exists()
    finally:
        plot_guard.install()
    with pytest.raises(plot_guard.FigureEscapedFrameError):
        _fig().savefig(tmp_path / "again.png")  # noqa: SH007 (the planted violation)


def _launch(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    """Run a one-module producer through the `python -m benchmark.plot_guard` launcher."""
    (tmp_path / "producer_probe.py").write_text(body)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tmp_path}:{env.get('PYTHONPATH', '')}"
    env["PROBE_OUT"] = str(tmp_path)
    return subprocess.run(
        [sys.executable, "-m", "benchmark.plot_guard", "producer_probe"],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )


_UNFRAMED = (
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "from matplotlib.figure import Figure\n"
    "fig = Figure()\n"
    "fig.savefig('escaped.png')\n"
)

_FRAMED = (
    "from pathlib import Path\n"
    "from benchmark import plot_frame\n"
    "fig = plot_frame.new_figure(plot_frame.SINGLE)\n"
    "fig.subplots().plot([0, 1], [0, 1])\n"
    "spec = plot_frame.FigureSpec(title='T', reading='R.', goal='G.')\n"
    "import os\n"
    "plot_frame.save(fig, Path(os.environ['PROBE_OUT']) / 'framed.png', spec)\n"
)


def test_the_launcher_refuses_an_unframed_producer(tmp_path: Path) -> None:
    """The Makefile's $(DRAW) prefix is the guard for a directly-invoked producer."""
    result = _launch(tmp_path, _UNFRAMED)
    assert result.returncode != 0
    assert "FigureEscapedFrameError" in result.stderr


def test_the_launcher_runs_a_framed_producer(tmp_path: Path) -> None:
    """...and only the violation fails: the supported path still draws."""
    result = _launch(tmp_path, _FRAMED)
    assert result.returncode == 0, result.stderr


def test_an_unguarded_run_of_the_same_producer_succeeds(tmp_path: Path) -> None:
    """Non-vacuity: without the launcher the identical producer writes the PNG happily."""
    (tmp_path / "producer_probe.py").write_text(_UNFRAMED)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tmp_path}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [sys.executable, "-m", "producer_probe"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "escaped.png").exists()
