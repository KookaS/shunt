"""Shared test fixtures and configuration for the whole project."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest

from tests.mock_openai_server import MockOpenAIServer, MockSignature


@pytest.fixture(autouse=True)
def _disallow_real_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse a real ONNX load for the whole unit suite (real-only in benchmark/live)."""
    # A unit test that reaches ``Embedder._load_model`` instead of injecting a fake hits a
    # loud error rather than silently downloading ~600MB. A test that legitimately exercises
    # the load path with a mocked fastembed unsets this locally.
    from shunt.router.embedder import DISALLOW_REAL_EMBEDDER_ENV

    monkeypatch.setenv(DISALLOW_REAL_EMBEDDER_ENV, "1")


@pytest.fixture(autouse=True)
def _isolate_benchmark_config_caches() -> Iterator[None]:
    """Snapshot/restore ``benchmark.config``'s load-once module caches per test."""
    # _config/_pricing are intentional load-once globals in production, but a test
    # that calls load()/validate() with a custom path leaves a stale config behind —
    # which the next test reads (select_pilot._models_by_tier then sees a disabled
    # model as enabled). Snapshotting keeps the suite order-independent without
    # changing production's caching.
    from benchmark import config as _bcfg

    saved_config, saved_pricing = _bcfg._config, _bcfg._pricing
    try:
        yield
    finally:
        _bcfg._config, _bcfg._pricing = saved_config, saved_pricing


@pytest.fixture(scope="session")
def provider_probe() -> ModuleType:
    """The `tools/provider_probe.py` executor, imported by path."""
    # tools/ is CI/hook-invoked script code, not an installed package, so there
    # is no import path to it. Loading by spec keeps that true — the alternative
    # is a sys.path mutation, which SH003/TID251 ban outright.
    path = Path(__file__).resolve().parents[1] / "tools" / "provider_probe.py"
    spec = importlib.util.spec_from_file_location("provider_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves a class's annotations through
    # sys.modules[cls.__module__], which blows up on a module that was never
    # registered. This is the documented module_from_spec recipe, not a hack.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mock_openai_server() -> Iterator[Callable[[MockSignature], MockOpenAIServer]]:
    """Factory for signature-replaying OpenAI stubs; all are stopped at teardown."""
    servers: list[MockOpenAIServer] = []

    def _start(signature: MockSignature) -> MockOpenAIServer:
        server = MockOpenAIServer(signature)
        server.start()
        servers.append(server)
        return server

    yield _start

    for server in servers:
        server.stop()
