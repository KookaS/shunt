"""Install-anywhere guarantee: the shipped configs resolve from the INSTALLED package."""

# A `pip install shunt-router` on a fresh machine must find its bundled defaults through
# importlib.resources — never a source-tree or CWD-relative path. If the wheel omits
# embedding.yaml the router cannot build its Embedder at boot. These tests lock that the
# file is bundled (the `*.yaml` package-data glob), parses, and wires into an Embedder
# without downloading the ~600MB ONNX model.

from __future__ import annotations

import importlib.resources
import os
from pathlib import Path

import pytest

from shunt.models.config import strict_yaml_load
from shunt.router.embedder import (
    DISALLOW_REAL_EMBEDDER_ENV,
    Embedder,
    RealEmbedderBlockedError,
)
from shunt.router.embedding_config import (
    load_embedding_config,
    packaged_embedding_path,
    parse_embedding_config,
)

# The shipped active model — these are the values a fresh install must resolve to.
_ACTIVE_KEY = "jina-code"
_ACTIVE_REPO = "jinaai/jina-embeddings-v2-base-code"
_ACTIVE_DIM = 768
_MAX_CHARS = 4000

# Every default the wheel must ship (the `*.yaml` package-data glob). models.yaml and
# router.yaml are loaded the same way; if the glob ever narrows, this list catches it.
_BUNDLED = ("embedding.yaml", "models.yaml", "router.yaml")


class TestBundledInWheel:
    """importlib.resources resolution — the installed-package angle, not a source path."""

    @pytest.mark.parametrize("filename", _BUNDLED)
    def test_bundled_config_resolves_and_is_readable(self, filename: str) -> None:
        ref = importlib.resources.files("shunt.config") / filename
        assert ref.is_file(), f"{filename} is not bundled in shunt.config"
        assert ref.read_text().strip(), f"{filename} is bundled but empty"

    def test_packaged_embedding_path_points_into_the_package(self) -> None:
        path = packaged_embedding_path()
        assert path.exists()
        assert path.name == "embedding.yaml"
        assert path.parent.name == "config"


class TestPackagedEmbeddingLoads:
    """The REAL packaged embedding.yaml parses, validates, and resolves its active model."""

    @staticmethod
    def _packaged_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Point SHUNT_CONFIG_DIR at an empty dir so the user-config layer misses and the
        # PACKAGED default is deterministically what loads — independent of ~/.config/shunt.
        monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path))

    def test_default_resolves_to_shipped_active_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._packaged_default(monkeypatch, tmp_path)
        cfg = load_embedding_config()  # no explicit path — the packaged default
        assert cfg.active == _ACTIVE_KEY
        assert cfg.max_chars == _MAX_CHARS
        active = cfg.resolve_active({})
        assert active.repo == _ACTIVE_REPO
        assert active.dim == _ACTIVE_DIM

    def test_shipped_schema_forbids_unknown_key(self) -> None:
        # Prove the SHIPPED file's schema is strict: append a bogus key to the real
        # packaged text and confirm extra="forbid" rejects it, not just a fixture.
        text = packaged_embedding_path().read_text().rstrip() + "\n  bogus: 1\n"
        with pytest.raises(Exception):  # noqa: B017,PT011 (pydantic ValidationError, fail-loud)
            parse_embedding_config(strict_yaml_load(text))

    def test_resolution_is_cwd_independent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # A fresh install runs from an arbitrary CWD; resolution must come from
        # importlib.resources, never a Path("src/...") relative to the working dir.
        self._packaged_default(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        cfg = load_embedding_config()
        assert cfg.resolve_active({}).repo == _ACTIVE_REPO


class TestConfigToEmbedderWiring:
    """The packaged config builds an Embedder on a clean env — without loading ONNX."""

    def test_embedder_reads_packaged_config_without_download(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("SHUNT_CONFIG_DIR", str(tmp_path))
        # The suite's autouse fixture already blocks a real ONNX load — assert it, so this
        # test's no-download claim is structural, not incidental.
        assert os.environ.get(DISALLOW_REAL_EMBEDDER_ENV)
        embedder = Embedder()  # lazy: parses packaged embedding.yaml, no model load
        assert embedder.model_name == _ACTIVE_REPO
        assert embedder.dims == _ACTIVE_DIM
        assert embedder.max_chars == _MAX_CHARS
        # Forcing the load proves the config→embedder path reaches the ONNX step and that
        # the guard — not a network round-trip — is what stops it on a clean machine.
        with pytest.raises(RealEmbedderBlockedError):
            embedder.warm()
