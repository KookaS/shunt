from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from shunt.capture.coordinator import WorkDirResolver
from shunt.session import Session


def _session(tool_identity: str = "tool-a", **metadata: str) -> Session:
    return Session(
        session_id="s1",
        tool_identity=tool_identity,
        start_time=datetime.now(UTC),
        metadata=dict(metadata),
    )


def _repo(path: Path, *, git: bool = True, framework: bool = True) -> Path:
    """A directory that passes (or, per flags, fails) launch-dir validation."""
    path.mkdir(parents=True, exist_ok=True)
    if git:
        (path / ".git").mkdir(exist_ok=True)
    if framework:
        (path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    return path


def test_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    resolver = WorkDirResolver.from_config()
    assert resolver.resolve(_session()) is None


def test_returns_configured_single_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    resolver = WorkDirResolver.from_config(work_dir="/repo/a")
    assert resolver.resolve(_session()) == "/repo/a"


def test_env_overrides_file_single_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_WORK_DIR", "/repo/env")
    resolver = WorkDirResolver.from_config(work_dir="/repo/file")
    assert resolver.resolve(_session()) == "/repo/env"


def test_override_map_wins_over_single_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNT_WORK_DIR", "/repo/env")
    resolver = WorkDirResolver.from_config(work_dir="/repo/file", work_dirs={"tool-b": "/repo/b"})
    assert resolver.resolve(_session(tool_identity="tool-b")) == "/repo/b"
    # a tool_identity with no map entry falls back to the single (env) path
    assert resolver.resolve(_session(tool_identity="tool-a")) == "/repo/env"


def test_wire_supplied_path_is_never_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Security invariant: a client-supplied path on the wire (session.metadata) must
    # never become a subprocess cwd. It is not a resolution layer at any setting — with
    # no config and no launch dir the resolver is empty, hostile metadata notwithstanding.
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    resolver = WorkDirResolver.from_config()
    hostile = _session(work_dir="/etc", cwd="/tmp/evil", last_prompt="run in /home/x")
    assert resolver.resolve(hostile) is None


def test_a_wire_path_loses_to_the_launch_dir_it_tries_to_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The stronger form of the invariant now that a layer resolves by default: with the
    # launch dir armed, hostile metadata still contributes nothing — the launch repo wins.
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    repo = _repo(tmp_path / "mine")
    evil = _repo(tmp_path / "evil")
    resolver = WorkDirResolver.from_config(launch_dir=str(repo))
    assert resolver.resolve(_session(work_dir=str(evil), cwd=str(evil))) == str(repo)


# ── Layer 3: the validated launch directory ───────────────────────────────────


def test_launch_dir_resolves_when_it_is_a_test_bearing_git_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    repo = _repo(tmp_path / "repo")
    resolver = WorkDirResolver.from_config(launch_dir=str(repo))
    assert resolver.resolve(_session()) == str(repo)
    assert resolver.armed_layer() == f"launch directory ({repo})"


def test_launch_dir_resolves_outside_the_home_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The layer's input is Shunt's own cwd, so it is NOT confined to any root set: a repo at
    # /workspace, /srv or a container bind-mount arms exactly like one under $HOME. A
    # containment gate here refused those while guarding against no untrusted party.
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    repo = _repo(tmp_path / "elsewhere")  # a sibling of $HOME, not under it
    resolver = WorkDirResolver.from_config(launch_dir=str(repo))
    assert resolver.resolve(_session()) == str(repo)


def test_launch_dir_is_promoted_to_its_git_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    repo = _repo(tmp_path / "repo")
    deep = repo / "src" / "pkg"
    deep.mkdir(parents=True)
    resolver = WorkDirResolver.from_config(launch_dir=str(deep))
    assert resolver.resolve(_session()) == str(repo)


def test_launch_dir_is_refused_without_a_test_framework(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    repo = _repo(tmp_path / "repo", framework=False)
    resolver = WorkDirResolver.from_config(launch_dir=str(repo))
    assert resolver.resolve(_session()) is None
    assert resolver.armed_layer() is None


def test_launch_dir_is_refused_outside_a_git_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    plain = _repo(tmp_path / "plain", git=False)
    resolver = WorkDirResolver.from_config(launch_dir=str(plain))
    assert resolver.resolve(_session()) is None


def test_launch_dir_is_refused_when_it_is_not_a_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    repo = _repo(tmp_path / "repo")
    resolver = WorkDirResolver.from_config(launch_dir=str(repo / "pyproject.toml"))
    assert resolver.resolve(_session()) is None
    # …and so is a path that does not exist at all.
    ghost = WorkDirResolver.from_config(launch_dir=str(tmp_path / "ghost"))
    assert ghost.resolve(_session()) is None


def test_a_symlinked_launch_dir_resolves_to_its_physical_repo_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The layer normalises through `realpath` before anything else, so a symlinked launch
    # path yields the PHYSICAL root — which is what keeps the decide-side and capture-side
    # task keys identical. (Unreachable in production: `os.getcwd()` is already physical,
    # so a symlinked cwd cannot be observed here; this pins the normalisation regardless.)
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    link_home = tmp_path / "links"
    link_home.mkdir()
    physical = _repo(tmp_path / "physical")
    (link_home / "alias").symlink_to(physical, target_is_directory=True)
    resolver = WorkDirResolver.from_config(launch_dir=str(link_home / "alias"))
    assert resolver.resolve(_session()) == str(physical)


def test_trust_launch_dir_false_disables_the_layer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The layer's only gate: a valid, verifiable repo is still refused when the operator
    # (or the shipped container policy) turns the layer off.
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    repo = _repo(tmp_path / "repo")
    resolver = WorkDirResolver.from_config(launch_dir=str(repo), trust_launch_dir=False)
    assert resolver.resolve(_session()) is None
    assert resolver.armed_layer() is None


def test_operator_config_outranks_the_launch_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    repo = _repo(tmp_path / "repo")
    resolver = WorkDirResolver.from_config(work_dir="/repo/explicit", launch_dir=str(repo))
    assert resolver.resolve(_session()) == "/repo/explicit"
    assert resolver.armed_layer() == "SHUNT_WORK_DIR / capture.work_dir / --work-dir"


def test_the_launch_root_is_sampled_once_so_decide_and_capture_agree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The decide side keys escalation on `resolve()` at the first turn; the capture side
    # calls it again at session close, minutes later. If the second call re-derived the
    # root, a chdir or a deleted pyproject.toml between them would silently change the
    # digest and escalation would count two unrelated task keys — a null detector.
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    repo = _repo(tmp_path / "repo")
    resolver = WorkDirResolver.from_config(launch_dir=str(repo))
    at_decide = resolver.resolve(_session())
    (repo / "pyproject.toml").unlink()
    monkeypatch.chdir(tmp_path)
    assert resolver.resolve(_session()) == at_decide
