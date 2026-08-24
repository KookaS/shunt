"""`shunt doctor` against the REAL registry, policy and work-dir resolver."""

# Nothing here hand-builds a report: every check runs against the packaged registry, the
# packaged router policy and the real WorkDirResolver/detect_framework pair the capture path
# uses. The point of the command is to tell a stranger whether their install actually works,
# so a fixture's idea of "a resolved work dir" would test the wrong thing entirely.

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import pytest

from shunt.cli import _doctor
from shunt.router.diagnostics import Check, DoctorReport, doctor_report

_FAKE_KEY = "sk-not-a-real-key-0123456789"


def _registry_key_vars() -> set[str]:
    """Every api_key_env_var the PACKAGED registry names, read from the registry itself."""
    # Derived, never hardcoded: a hardcoded pair rots silently the moment a provider is added,
    # and the failure mode is a suite that passes because a real key leaked in from the shell.
    from shunt.models.config import load_registry

    return {p.api_key_env_var for p in load_registry().providers.values() if p.api_key_env_var}


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A hermetic data dir + the PACKAGED registry/policy (no dev-machine ~/.config/shunt)."""
    monkeypatch.setenv("SHUNT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("SHUNT_MODEL_CONFIG_PATH", raising=False)
    monkeypatch.delenv("SHUNT_WORK_DIR", raising=False)
    for var in _registry_key_vars():
        monkeypatch.delenv(var, raising=False)
    # These four leaked before and each broke the suite differently on a developer machine:
    # SHUNT_EMBED_CACHE_DIR silently flipped which embedder branch was under test WHILE STAYING
    # GREEN (the worst kind), SHUNT_HOST/SHUNT_PORT turned every test red, and
    # SHUNT_EMBEDDER_MODEL failed eight. Pin the cache dir rather than merely clearing it, so
    # the "not cached" branch is the deterministic default.
    for var in ("SHUNT_HOST", "SHUNT_PORT", "SHUNT_EMBEDDER_MODEL", "SHUNT_ROUTER_STRATEGY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SHUNT_EMBED_CACHE_DIR", str(tmp_path / "embed-cache"))
    # chdir is load-bearing, not tidiness: `doctor` calls load_dotenv_file() exactly as
    # `start` does, so running from the repo root would read shunt's OWN .env and report
    # keys as set that the test just cleared. It also makes the launch-directory work-dir
    # resolution land somewhere inert instead of on this very repo.
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(work_dir: str | None = None, as_json: bool = False) -> None:
    _doctor(argparse.Namespace(work_dir=work_dir, as_json=as_json))


def _repo_with_tests(root: Path) -> str:
    """A directory the resolver accepts: a git root that declares a detectable framework."""
    (root / ".git").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n", encoding="utf-8")
    (root / "tests").mkdir()
    return str(root)


def _repo_without_tests(root: Path) -> str:
    """A git root with NO test framework — resolvable, but nothing for the verifier to run."""
    (root / ".git").mkdir(parents=True)
    (root / "README.md").write_text("no tests here\n", encoding="utf-8")
    return str(root)


def _check(report: DoctorReport, name: str) -> Check:
    """The one check by name — a KeyError here means the report shape changed."""
    return next(c for c in report.checks if c.name == name)


def _plant_model_cache(cache: Path) -> None:
    """A cache dir shaped like a COMPLETE fastembed download, without downloading one."""
    from shunt.router.embedding_config import load_embedding_config

    repo = load_embedding_config().resolve_active({}).repo
    target = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / "deadbeef"
    target.mkdir(parents=True)
    # Non-trivial size: the check must reject a truncated file, so a 1-byte stub would not do.
    (target / "model.onnx").write_bytes(b"\0" * (2 * 1024 * 1024))


def _tree_fingerprint(root: Path) -> list[tuple[str, int]]:
    """Every file under *root* as (relative path, size) — enough to catch any write."""
    return sorted(
        (str(p.relative_to(root)), p.stat().st_size) for p in root.rglob("*") if p.is_file()
    )


# ── credentials: the check that decides the exit code ────────────────────────


def test_exits_nonzero_when_no_provider_key_resolves(cli_env: Path) -> None:
    # With every key unset the router cannot serve a single request, which is exactly the
    # state a new user is in before they read configuration.md. A zero exit here would be
    # the command telling them their broken install is fine.
    with pytest.raises(SystemExit) as exc:
        _run()
    assert exc.value.code != 0


def test_exits_zero_once_a_key_resolves(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    # DEEPSEEK too: the shipped default is `session_cascade`, which routes its first request
    # deterministically to the cheapest model (deepseek-v4-flash), so its key is REQUIRED for
    # the install to be serviceable at all. Setting only REQUESTY fails this for a reason
    # unrelated to what the test is about.
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    _run()  # must not raise SystemExit


def test_names_the_env_var_that_is_missing(
    cli_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _run()
    out = capsys.readouterr().out
    # The whole point is actionability: naming the variable is the fix instruction.
    assert "REQUESTY_API_KEY" in out
    assert "DEEPSEEK_API_KEY" in out


def test_never_prints_a_key_value(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A doctor command is the single most likely thing a user pastes into an issue.
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    _run()
    out = capsys.readouterr().out
    assert _FAKE_KEY not in out
    assert "set" in out.lower()


def test_json_output_carries_no_key_value(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    # DEEPSEEK too: the shipped default is `session_cascade`, which routes its first request
    # deterministically to the cheapest model (deepseek-v4-flash), so its key is REQUIRED for
    # the install to be serviceable at all. Setting only REQUESTY fails this for a reason
    # unrelated to what the test is about.
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    _run(as_json=True)
    out = capsys.readouterr().out
    assert _FAKE_KEY not in out
    import json

    payload = json.loads(out)
    assert payload["serviceable"] is True
    assert any(c["name"] == "credentials" for c in payload["checks"])


# ── escalation: ARMED vs ENABLED-BUT-INERT, the defect this command exists for ──


def test_reports_escalation_inert_when_no_work_dir_resolves(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    # DEEPSEEK too: the shipped default is `session_cascade`, which routes its first request
    # deterministically to the cheapest model (deepseek-v4-flash), so its key is REQUIRED for
    # the install to be serviceable at all. Setting only REQUESTY fails this for a reason
    # unrelated to what the test is about.
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    # Launch somewhere that is not a git repo with tests, so nothing resolves.
    monkeypatch.chdir(cli_env)
    _run()
    out = capsys.readouterr().out
    assert "INERT" in out
    # Enabled and armed are different states and the text must not conflate them.
    assert "enabled" in out.lower()


def test_reports_escalation_armed_for_a_repo_with_tests(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    # DEEPSEEK too: the shipped default is `session_cascade`, which routes its first request
    # deterministically to the cheapest model (deepseek-v4-flash), so its key is REQUIRED for
    # the install to be serviceable at all. Setting only REQUESTY fails this for a reason
    # unrelated to what the test is about.
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    repo = _repo_with_tests(cli_env / "repo")
    _run(work_dir=repo)
    out = capsys.readouterr().out
    assert "ARMED" in out
    assert "INERT" not in out


def test_resolved_work_dir_without_a_test_framework_is_inert_not_armed(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The case the boot warning misses: an EXPLICIT work_dir is not framework-validated the
    # way a launch dir is, so escalation can resolve a repo and still have nothing to run.
    # Reporting that as ARMED would be the exact false-confidence this command must prevent.
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    # DEEPSEEK too: the shipped default is `session_cascade`, which routes its first request
    # deterministically to the cheapest model (deepseek-v4-flash), so its key is REQUIRED for
    # the install to be serviceable at all. Setting only REQUESTY fails this for a reason
    # unrelated to what the test is about.
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    repo = _repo_without_tests(cli_env / "bare")
    _run(work_dir=repo)
    out = capsys.readouterr().out
    assert "INERT" in out
    assert "no test framework" in out.lower()


def test_escalation_inert_does_not_make_the_install_unserviceable(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Inert escalation is a degraded state, not a broken one — the router still routes.
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    # DEEPSEEK too: the shipped default is `session_cascade`, which routes its first request
    # deterministically to the cheapest model (deepseek-v4-flash), so its key is REQUIRED for
    # the install to be serviceable at all. Setting only REQUESTY fails this for a reason
    # unrelated to what the test is about.
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    monkeypatch.chdir(cli_env)
    report = doctor_report(work_dir=None, launch_dir=str(cli_env))
    assert report.serviceable is True
    assert any(c.name == "escalation" and c.warn for c in report.checks)


# ── the remaining surfaces ───────────────────────────────────────────────────


def test_registry_reports_the_exact_live_model_set(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Asserting only that the word "registry" appears would survive a +99 count mutation.
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    from shunt.models import ModelPool
    from shunt.router.policy import load_router_policy

    pool = ModelPool()
    pool.restrict_to_live(load_router_policy().models)
    expected = pool.model_names()

    check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "registry")
    assert f"{len(expected)} live model(s)" in check.detail
    for name in expected:
        assert name in check.detail


def test_embedder_reports_not_cached_on_an_empty_cache(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    # The embedder verdict depends on whether the ACTIVE strategy consults it, and the shipped
    # default (`session_cascade`) does not — so this branch has to name the strategy that does.
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "knn_cascade")
    monkeypatch.setenv("SHUNT_EMBED_CACHE_DIR", str(cli_env / "empty-cache"))
    check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "embedder")
    assert check.warn is True
    assert "NOT cached" in check.detail


def test_embedder_reports_cached_on_a_real_looking_cache(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    cache = cli_env / "full-cache"
    _plant_model_cache(cache)
    monkeypatch.setenv("SHUNT_EMBED_CACHE_DIR", str(cache))
    check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "embedder")
    assert check.warn is False
    assert "cached" in check.detail and "NOT cached" not in check.detail


def test_embedder_reports_a_partial_download_as_incomplete(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A non-empty cache is NOT a usable cache: fastembed writes into it before completing, so
    # "directory has something in it" reported a truncated ONNX file as healthy — inverting the
    # verdict on a real failure state, for the command run to diagnose exactly that.
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    cache = cli_env / "partial-cache"
    cache.mkdir()
    (cache / "garbage.bin").write_bytes(b"x")
    monkeypatch.setenv("SHUNT_EMBED_CACHE_DIR", str(cache))
    check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "embedder")
    assert check.warn is True
    assert "incomplete" in check.detail.lower()


def test_embedder_check_survives_an_unreadable_cache_dir(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    # PIN THE STRATEGY. This assertion is only true when the router actually embeds, and the
    # test used to rely on the PACKAGED DEFAULT being knn — so it passed identically under
    # always_cheap, the strategy the assertion is false for. That made it a green test encoding
    # the defect: change the shipped default and it flips from correct to wrong, silently.
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "knn")
    cache = cli_env / "noperm"
    cache.mkdir()
    (cache / "something").write_bytes(b"x")
    cache.chmod(0o000)
    monkeypatch.setenv("SHUNT_EMBED_CACHE_DIR", str(cache))
    try:
        check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "embedder")
    finally:
        cache.chmod(0o755)
    # Unreadable is a hard FAIL, not a warning: unlike an absent cache it does not recover by
    # downloading — the first routed request hits the same EACCES.
    assert check.ok is False


def test_does_not_touch_the_embedding_cache(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The observable half: no bytes change under the cache directory.
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    cache = cli_env / "watch-cache"
    _plant_model_cache(cache)
    monkeypatch.setenv("SHUNT_EMBED_CACHE_DIR", str(cache))

    before = _tree_fingerprint(cache)
    doctor_report(work_dir=None, launch_dir=str(cli_env))
    assert _tree_fingerprint(cache) == before


def test_never_constructs_the_embedder(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The CONTRACT half, and it needs its own test. The observable check above cannot catch a
    # bare `Embedder()` because __init__ is lazy=True and so writes nothing — mutation testing
    # confirmed that exact mutation survived it. Constructing one is harmless TODAY and the
    # module promises not to; if the lazy default ever flips, this is what fails instead of a
    # user's disk filling up.
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("doctor must not construct the embedding model")

    monkeypatch.setattr("shunt.router.embedder.Embedder.__init__", _boom)
    doctor_report(work_dir=None, launch_dir=str(cli_env))


def test_config_provenance_distinguishes_file_from_default(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This is the assertion that would have caught the provenance lie: `strategy` was always
    # attributed to the config file, even when the file never mentioned it.
    config_dir = cli_env / "config"
    config_dir.mkdir(exist_ok=True)
    policy = config_dir / "router.yaml"
    policy.write_text("router:\n  escalation:\n    escalate_after_n: 7\n", encoding="utf-8")
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)

    detail = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "config").detail
    rows = {line.split()[0]: line for line in detail.splitlines() if line.split()}
    assert str(policy) in rows["escalate_after_n"]  # the file really set it
    assert "built-in default" in rows["stale_window"]  # the file did not
    assert "built-in default" in rows["strategy"]  # nor this — the old code claimed the file


def test_config_reports_the_env_override_not_the_file_value(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # load_router_policy() deliberately does NOT apply env overlays (the server layer does), so
    # doctor reported the FILE's strategy while the server would run the env's. Wrong value, in
    # the one command whose job is reporting the effective config.
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "always_cheap")
    detail = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "config").detail
    strategy_row = next(line for line in detail.splitlines() if line.startswith("strategy"))
    assert "always_cheap" in strategy_row
    assert "SHUNT_ROUTER_STRATEGY" in strategy_row


# ── C1: a broken install must be DIAGNOSED, never raised ─────────────────────
# These are the states `shunt doctor` exists for, and every one of them used to reach the
# user as a Python traceback. A report with ok=False is the contract; an exception is not.


def test_malformed_router_yaml_is_reported_not_raised(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = cli_env / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "router.yaml").write_text("router:\n  strategy: [unclosed\n", encoding="utf-8")
    report = doctor_report(work_dir=None, launch_dir=str(cli_env))
    assert report.serviceable is False
    assert any(not c.ok for c in report.checks)


def test_malformed_models_yaml_is_reported_not_raised(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = cli_env / "models.yaml"
    bad.write_text("providers: {oops\n", encoding="utf-8")
    monkeypatch.setenv("SHUNT_MODEL_CONFIG_PATH", str(bad))
    report = doctor_report(work_dir=None, launch_dir=str(cli_env))
    assert report.serviceable is False
    assert not _check(report, "registry").ok


def test_unknown_live_model_is_reported_not_raised(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The nastiest of the set: restrict_to_live raises a perfectly-worded diagnostic that the
    # user only ever saw as a crash. The message must reach them as a finding instead.
    config_dir = cli_env / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "router.yaml").write_text(
        "router:\n  models:\n    - not-a-real-model\n", encoding="utf-8"
    )
    report = doctor_report(work_dir=None, launch_dir=str(cli_env))
    assert report.serviceable is False
    assert "not-a-real-model" in _check(report, "registry").detail


def test_missing_work_dir_is_reported_not_raised(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    report = doctor_report(work_dir="/definitely/not/here", launch_dir=str(cli_env))
    check = _check(report, "escalation")
    assert check.warn is True
    assert "INERT" in check.detail


def test_work_dir_pointing_at_a_file_is_reported_not_raised(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    target = cli_env / "a-file.txt"
    target.write_text("not a directory\n", encoding="utf-8")
    check = _check(doctor_report(work_dir=str(target), launch_dir=str(cli_env)), "escalation")
    assert check.warn is True


# ── C2/C3: the port probe ────────────────────────────────────────────────────


def test_port_reports_warn_when_the_address_is_occupied(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # _port_check had ZERO tests, which is why a no-op mutation of it survived the whole suite.
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        monkeypatch.setenv("SHUNT_PORT", str(srv.getsockname()[1]))
        check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "port")
    assert check.warn is True


def test_port_reports_free_when_nothing_listens(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    monkeypatch.setenv("SHUNT_PORT", str(free_port))
    check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "port")
    assert check.warn is False and check.ok is True


@pytest.mark.parametrize("bad_port", ["99999", "-1", "not-a-number"])
def test_out_of_range_or_junk_port_is_reported_not_raised(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, bad_port: str
) -> None:
    # int() caught junk but NOT out-of-range: connect_ex raised OverflowError for 99999/-1.
    monkeypatch.setenv("SHUNT_PORT", bad_port)
    check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "port")
    assert check.ok is False
    assert bad_port in check.detail


def test_ipv6_host_does_not_raise(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # ::1 is a documented bind address for a localhost-only service; AF_INET-only raised gaierror.
    monkeypatch.setenv("SHUNT_HOST", "::1")
    check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "port")
    assert check.name == "port"


def test_ipv6_occupied_port_is_not_reported_free(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The AF_INET-only probe said "free" while a listener held ::1, so `shunt start` then
    # failed to bind — doctor actively misled the user about the one thing it checked.
    import socket

    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as srv:
        srv.bind(("::1", 0))
        srv.listen(1)
        monkeypatch.setenv("SHUNT_HOST", "::1")
        monkeypatch.setenv("SHUNT_PORT", str(srv.getsockname()[1]))
        check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "port")
    assert check.warn is True


def test_unresolvable_host_is_reported_not_raised(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHUNT_HOST", "no-such-host.invalid")
    check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "port")
    assert check.ok is False


# ── C8: a provider that needs no key ─────────────────────────────────────────


def test_a_keyless_provider_does_not_make_the_install_unserviceable(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A local/ollama-style fragment declares no api_key_env_var. Treating "" as MISSING made
    # doctor exit 1 on a working install and print a bare ": MISSING" with no variable name.
    registry = cli_env / "keyless.yaml"
    registry.write_text(
        "providers:\n"
        "  local:\n"
        "    base_url: http://127.0.0.1:11434/v1\n"
        "    api_key_env_var: ''\n"
        "    litellm_prefix: openai\n"
        "models:\n"
        "  local-llama:\n"
        "    provider: local\n"
        "    model_id: llama3\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SHUNT_MODEL_CONFIG_PATH", str(registry))
    config_dir = cli_env / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "router.yaml").write_text(
        "router:\n  models:\n    - local-llama\n", encoding="utf-8"
    )
    report = doctor_report(work_dir=None, launch_dir=str(cli_env))
    creds = _check(report, "credentials")
    assert creds.ok is True
    assert ": MISSING" not in creds.detail
    assert report.serviceable is True


# ── second-round regressions: a valid-but-falsy config, and cache mislabels ──


@pytest.mark.parametrize("body", ["", "# only a comment\n", "false\n"])
def test_falsy_router_yaml_is_reported_not_raised(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    # strict_yaml_load is TYPED -> dict but returns None for an empty/comment-only file and
    # False for `false`. parse_router_policy accepts those as "use the built-in defaults", so
    # the policy loaded fine and the AttributeError landed in a render helper past every guard.
    config_dir = cli_env / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "router.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    report = doctor_report(work_dir=None, launch_dir=str(cli_env))
    assert _check(report, "config").detail.strip()


def test_falsy_router_yaml_does_not_break_shunt_escalate_either(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same unchecked parse lives in inspection.escalation_sources, which `shunt escalate`
    # has always used and `shunt doctor` now shares. Fixing only the diagnostics copy would
    # have left the original crash in place.
    from shunt.router.inspection import escalation_sources

    policy = cli_env / "router.yaml"
    policy.write_text("# nothing here\n", encoding="utf-8")
    sources = escalation_sources(policy)
    assert sources and all(isinstance(v, str) for v in sources.values())


def test_unreadable_populated_cache_is_not_reported_as_absent(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The mislabel that gave actively false advice: a real 600MB cache that this user cannot
    # read was reported "NOT cached ... will download ... not an error". All three clauses
    # wrong — it IS cached, the download will hit the same EACCES, and it IS an error.
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    # The embedder verdict depends on whether the ACTIVE strategy consults it, and the shipped
    # default (`session_cascade`) does not — so this branch has to name the strategy that does.
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "knn_cascade")
    cache = cli_env / "locked-cache"
    _plant_model_cache(cache)
    monkeypatch.setenv("SHUNT_EMBED_CACHE_DIR", str(cache))
    cache.chmod(0o000)
    try:
        check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "embedder")
    finally:
        cache.chmod(0o755)
    assert "NOT cached" not in check.detail
    assert "NOT READABLE" in check.detail
    assert check.ok is False


def test_model_inside_an_unreadable_subdir_is_not_called_incomplete(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # os.walk swallows per-directory errors by default, so a VALID model file under an
    # unreadable subdirectory read as "incomplete — delete the directory and re-download":
    # advice to throw away a good 600MB cache.
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    cache = cli_env / "part-locked"
    _plant_model_cache(cache)
    monkeypatch.setenv("SHUNT_EMBED_CACHE_DIR", str(cache))
    inner = next(p for p in cache.iterdir() if p.is_dir())
    inner.chmod(0o000)
    try:
        check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "embedder")
    finally:
        inner.chmod(0o755)
    assert "delete the directory" not in check.detail


def test_a_bad_embedder_env_var_does_not_blame_the_file(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Telling the user embedding.yaml is unreadable when the ENV VAR is wrong misdirects the
    # fix — and env-vs-file drift is the exact thing this check exists to surface.
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("SHUNT_EMBEDDER_MODEL", "no/such/model")
    check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "embedder")
    assert check.ok is False
    assert "could not read embedding.yaml" not in check.detail


def test_overlong_host_label_is_reported_not_raised(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # getaddrinfo raises UnicodeError — a ValueError, NOT an OSError — for a label over 63
    # bytes, so the gaierror-only guard missed a plausible pasted/garbled SHUNT_HOST.
    monkeypatch.setenv("SHUNT_HOST", "a" * 300 + ".example")
    check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "port")
    assert check.ok is False


def test_empty_host_probes_loopback_like_the_server_binds(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An empty SHUNT_HOST binds every interface under uvicorn exactly as 0.0.0.0 does. Probing
    # it literally made doctor declare a working config unserviceable.
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    # DEEPSEEK too: the shipped default is `session_cascade`, which routes its first request
    # deterministically to the cheapest model (deepseek-v4-flash), so its key is REQUIRED for
    # the install to be serviceable at all. Setting only REQUESTY fails this for a reason
    # unrelated to what the test is about.
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("SHUNT_HOST", "")
    report = doctor_report(work_dir=None, launch_dir=str(cli_env))
    assert _check(report, "port").ok is True
    assert report.serviceable is True


# ── F-06: the embedder verdict must depend on whether the strategy consults it ──
# `_decide_fixed` routes a fixed strategy from the pool alone — "no cold-start, embedding, or
# query" — so an unreadable or absent embed cache cannot stop that install serving a request.
# Reporting it as a FAIL made doctor exit 1 and print "Router CANNOT serve a request" on an
# install that serves every request perfectly: a false negative in the one command whose exit
# code is the whole verdict. The tests below VARY THE STRATEGY while varying cache state — the
# combination that was untested, which is exactly why this shipped.


def _unreadable_cache(root: Path) -> Path:
    cache = root / "locked"
    _plant_model_cache(cache)
    cache.chmod(0o000)
    return cache


def test_unreadable_cache_is_not_a_failure_under_a_fixed_strategy(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    # DEEPSEEK too: this test asserts `report.serviceable`, and always_cheap routes its first
    # request to deepseek-v4-flash. Setting only REQUESTY made the install genuinely
    # unserviceable for an unrelated reason (its first pick has no key), which would have made
    # the assertion pass or fail for reasons nothing to do with the embedder cache.
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "always_cheap")
    cache = _unreadable_cache(cli_env)
    monkeypatch.setenv("SHUNT_EMBED_CACHE_DIR", str(cache))
    try:
        report = doctor_report(work_dir=None, launch_dir=str(cli_env))
    finally:
        cache.chmod(0o755)
    check = _check(report, "embedder")
    assert check.ok is True
    assert check.warn is True
    assert report.serviceable is True


def test_unreadable_cache_is_still_a_failure_under_knn(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The other half of the same contract: knn cannot route without neighbours, so the same
    # filesystem state IS fatal there. Without this pair a fix could flip the verdict globally.
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "knn")
    cache = _unreadable_cache(cli_env)
    monkeypatch.setenv("SHUNT_EMBED_CACHE_DIR", str(cache))
    try:
        report = doctor_report(work_dir=None, launch_dir=str(cli_env))
    finally:
        cache.chmod(0o755)
    assert _check(report, "embedder").ok is False
    assert report.serviceable is False


def test_absent_cache_does_not_promise_a_download_under_a_fixed_strategy(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "The first routed request downloads it (~600MB)" is false under always_cheap: no routed
    # request ever embeds anything, so the download never happens and the advice is noise.
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "always_frontier")
    monkeypatch.setenv("SHUNT_EMBED_CACHE_DIR", str(cli_env / "no-cache"))
    check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "embedder")
    assert "600MB" not in check.detail
    assert "downloads it" not in check.detail
    assert "always_frontier" in check.detail


def test_absent_cache_still_promises_a_download_under_knn(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "knn")
    monkeypatch.setenv("SHUNT_EMBED_CACHE_DIR", str(cli_env / "no-cache"))
    check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "embedder")
    assert "600MB" in check.detail
    assert check.warn is True


# ── F-07: a stock install must not attribute its values to a file nobody wrote ──


def test_stock_install_attributes_every_value_to_the_built_in_default(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # resolved_policy_path() falls back to the PACKAGED router.yaml, so on a stock install every
    # provenance cell named a path inside site-packages and the words "built-in default" never
    # appeared — §2 asks doctor to say which values are defaults vs overridden, and it said
    # "all overridden, by a file you have never seen".
    from shunt.router.policy import packaged_policy_path

    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    # cli_env points SHUNT_CONFIG_DIR at a directory with no router.yaml — a stock install.
    detail = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "config").detail
    assert str(packaged_policy_path()) not in detail
    rows = [line for line in detail.splitlines() if line.strip()]
    assert rows
    for row in rows:
        assert "built-in default" in row, row


def test_a_user_config_is_still_named_on_a_stock_style_install(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The relabel must not swallow a real user file — that would be the opposite lie.
    config_dir = cli_env / "config"
    config_dir.mkdir(exist_ok=True)
    policy = config_dir / "router.yaml"
    policy.write_text("router:\n  strategy: always_cheap\n", encoding="utf-8")
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    detail = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "config").detail
    strategy_row = next(line for line in detail.splitlines() if line.startswith("strategy"))
    assert str(policy) in strategy_row


# ── F-08: --json must have ONE shape, whatever went wrong ────────────────────


def _json_payload(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    """Run `doctor --json` and parse stdout, whichever exit code the branch produces."""
    import json

    with contextlib.suppress(SystemExit):
        _run(as_json=True)
    return dict(json.loads(capsys.readouterr().out))


def _break_models_yaml(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = root / "models.yaml"
    bad.write_text("providers: {oops\n", encoding="utf-8")
    monkeypatch.setenv("SHUNT_MODEL_CONFIG_PATH", str(bad))


def _break_router_yaml(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = root / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "router.yaml").write_text("router:\n  strategy: [unclosed\n", encoding="utf-8")


@pytest.mark.parametrize("break_it", [None, _break_models_yaml, _break_router_yaml])
def test_json_check_names_are_identical_across_every_assembly_branch(
    cli_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    break_it: object,
) -> None:
    # A machine consumer keying on `credentials` used to KeyError the moment models.yaml was
    # malformed — the branch it most needs to read. Membership AND order are the contract.
    from shunt.router.diagnostics import CHECK_ORDER

    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    if break_it is not None:
        break_it(cli_env, monkeypatch)  # type: ignore[operator]
    payload = _json_payload(capsys)
    checks = payload["checks"]
    assert isinstance(checks, list)
    assert [c["name"] for c in checks] == list(CHECK_ORDER)
    for check in checks:
        assert check["status"] in {"ok", "warn", "fail", "skipped"}
        assert isinstance(check["detail"], str) and check["detail"]


# ── F-09: the redaction property, proven stronger than redaction ─────────────


def test_no_credential_env_value_ever_reaches_any_output(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Proves the property redaction would only approximate: the value is never read at all."""
    # The spec asked this test to reuse proxy/redaction.py's `header_safe`/`redact_secrets`.
    # It deliberately does not, and reusing them here would be a REGRESSION. Redaction is a
    # filter applied to a string that already contains the secret; the credentials check calls
    # `bool(os.environ.get(var))` and never binds the value, so there is nothing to filter.
    # Swapping in redaction would weaken the guarantee from "the value is never read" to "the
    # value is read and then scrubbed" — one missed call site away from a leak. So the test
    # asserts the stronger property directly, over EVERY key variable the packaged registry
    # names and over BOTH renderings, which is what the spec was actually reaching for.
    sentinels = {var: f"sk-sentinel-{i}-abcdef" for i, var in enumerate(_registry_key_vars())}
    assert sentinels, "the packaged registry must name at least one key variable"
    for var, value in sentinels.items():
        monkeypatch.setenv(var, value)

    _run()
    text_out = capsys.readouterr().out
    _run(as_json=True)
    json_out = capsys.readouterr().out

    for var, value in sentinels.items():
        assert var in text_out, f"{var} must still be NAMED — presence reporting is the point"
        for rendered in (text_out, json_out):
            assert value not in rendered
            # Not even a prefix: a truncated secret survives a naive `value not in` check.
            assert value[:8] not in rendered


# ── F-12: model-pool health and loop health, the two §2 reuses that were absent ──


def test_circuit_broken_models_are_not_reported_as_a_healthy_registry(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # §2 names ModelPool.is_healthy; nothing in diagnostics called it, so a pool with every
    # breaker open still printed "[ ok ] registry — 10 live model(s)" and "[ ok ] credentials".
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    monkeypatch.setattr("shunt.models.ModelPool.is_healthy", lambda self, name: False)
    report = doctor_report(work_dir=None, launch_dir=str(cli_env))
    check = _check(report, "registry")
    assert check.ok is False
    assert "0" in check.detail
    assert report.serviceable is False


def test_a_healthy_pool_reports_how_many_models_are_routable(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    from shunt.models import ModelPool
    from shunt.router.policy import load_router_policy

    pool = ModelPool()
    pool.restrict_to_live(load_router_policy().models)
    n = len(pool.model_names())
    check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "registry")
    assert check.ok is True
    assert f"{n} of {n}" in check.detail


def test_loop_health_is_absent_without_an_outcome_corpus_and_creates_nothing(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The store's constructor MIGRATES a database into existence. doctor promises not to mutate,
    # so the no-corpus branch must report the absence rather than manufacture the file.
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    db = cli_env / "data" / "outcomes.db"
    check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "loop")
    assert not db.exists()
    assert check.warn is True
    assert check.ok is True


def test_loop_health_is_reported_when_a_corpus_exists(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    from shunt.db.store import OutcomeStore

    store = OutcomeStore()
    store.close()
    check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "loop")
    assert "0" in check.detail
    assert check.ok is True


# ── R10: SHUNT_PORT=0 is not an address ──────────────────────────────────────


def test_port_zero_is_not_reported_as_a_free_address(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 0 means "ask the kernel for an ephemeral port": there is no address to probe and no
    # address a tool can be pointed at. "127.0.0.1:0 is free" is true of nothing.
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("SHUNT_PORT", "0")
    check = _check(doctor_report(work_dir=None, launch_dir=str(cli_env)), "port")
    assert "is free" not in check.detail
    assert check.warn is True
    assert check.ok is True


# ── F2: the credentials check must know which model the router actually picks ──
# Doctor held both facts — the active strategy, and which key is unset — and never joined them,
# so it printed "Router is serviceable" on an install whose first request dies on auth. This is
# the mirror image of the embedder/strategy bug: same root cause, opposite sign.


def test_missing_key_for_the_model_a_fixed_strategy_picks_is_a_failure(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # always_cheap takes ranked[0] == deepseek-v4-flash, which needs DEEPSEEK_API_KEY.
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "always_cheap")
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    report = doctor_report(work_dir=None, launch_dir=str(cli_env))
    creds = _check(report, "credentials")
    assert creds.ok is False
    assert "deepseek-v4-flash" in creds.detail
    assert "DEEPSEEK_API_KEY" in creds.detail
    assert report.serviceable is False


def test_missing_key_for_a_model_the_strategy_does_not_pick_is_only_a_warning(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The complement, and the reason this cannot just be "all keys must be set": always_frontier
    # never routes to deepseek-v4-flash, so its missing key does not stop the router serving.
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "always_frontier")
    monkeypatch.setenv("REQUESTY_API_KEY", _FAKE_KEY)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    report = doctor_report(work_dir=None, launch_dir=str(cli_env))
    creds = _check(report, "credentials")
    assert creds.ok is True
    assert creds.warn is True
    assert report.serviceable is True


def test_missing_key_for_the_cold_start_model_is_a_failure_under_knn(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A neighbour-consulting strategy has no verified outcomes on a fresh install, so the engine
    # cold-starts — and the cold-start model is a Requesty one. Checking only "some key is set"
    # would call this serviceable while every first request fails.
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "knn")
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    monkeypatch.delenv("REQUESTY_API_KEY", raising=False)
    report = doctor_report(work_dir=None, launch_dir=str(cli_env))
    creds = _check(report, "credentials")
    assert creds.ok is False
    assert "REQUESTY_API_KEY" in creds.detail
    assert report.serviceable is False


def test_first_pick_reachable_leaves_the_install_serviceable(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "always_cheap")
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE_KEY)
    monkeypatch.delenv("REQUESTY_API_KEY", raising=False)
    report = doctor_report(work_dir=None, launch_dir=str(cli_env))
    assert _check(report, "credentials").ok is True
    assert report.serviceable is True


def test_the_first_pick_is_derived_from_the_real_strategy_object(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Guards against the fix drifting into a hardcoded name list: the answer must come from
    # build_strategy(...).select(...) / ColdStartStrategy, so a new live strategy inherits it.
    from shunt.models import ModelPool
    from shunt.router.diagnostics import _first_routed_model
    from shunt.router.policy import apply_env_overrides, load_router_policy
    from shunt.router.selection import SelectionRule
    from shunt.router.strategies.registry import build_strategy

    monkeypatch.setenv("SHUNT_ROUTER_STRATEGY", "always_frontier")
    policy = apply_env_overrides(load_router_policy())
    pool = ModelPool()
    pool.restrict_to_live(policy.models)
    expected, _rule = build_strategy("always_frontier", SelectionRule()).select([], pool)
    assert _first_routed_model(policy, pool)[0] == expected
