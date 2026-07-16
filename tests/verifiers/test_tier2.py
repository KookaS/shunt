import tempfile
from pathlib import Path

from shunt.verifiers.tier2 import AutoDetectVerifier


class TestAutoDetectVerifier:
    def setup_method(self) -> None:
        self.v = AutoDetectVerifier()

    def test_unknown_when_no_work_dir(self) -> None:
        result = self.v.verify(work_dir=None)
        assert result.outcome == "unknown"
        assert result.confidence == 0.0

    def test_unknown_when_dir_not_exist(self) -> None:
        result = self.v.verify(work_dir="/nonexistent/path/xyz")
        assert result.outcome == "unknown"
        assert result.confidence == 0.0

    def test_detect_python_pyproject_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "pyproject.toml").write_text(
                '[build-system]\nrequires = ["setuptools", "pytest"]'
            )
            lang = self.v.detect(tmpdir)
            assert lang == "python"

    def test_detect_python_setup_cfg(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "setup.cfg").write_text("[tool:pytest]\naddopts = -v")
            lang = self.v.detect(tmpdir)
            assert lang == "python"

    def test_detect_python_requirements_dev(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "requirements-dev.txt").write_text("pytest>=7.0")
            lang = self.v.detect(tmpdir)
            assert lang == "python"

    def test_detect_typescript(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "package.json").write_text(
                '{\n  "devDependencies": { "jest": "^29.0.0" }\n}'
            )
            lang = self.v.detect(tmpdir)
            assert lang == "typescript"

    def test_detect_go(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "go.mod").write_text("module example.com/myproject\ngo 1.21\n")
            lang = self.v.detect(tmpdir)
            assert lang == "go"

    def test_detect_rust(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "Cargo.toml").write_text(
                '[package]\nname = "myproject"\nversion = "0.1.0"\n'
            )
            lang = self.v.detect(tmpdir)
            assert lang == "rust"

    def test_detect_none_when_no_framework(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lang = self.v.detect(tmpdir)
            assert lang is None

    def test_detect_prefers_pytest_over_other_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "pyproject.toml").write_text(
                '[build-system]\nrequires = ["setuptools", "pytest"]'
            )
            (Path(tmpdir) / "go.mod").write_text("module example\n")
            lang = self.v.detect(tmpdir)
            assert lang == "python"

    def test_empty_work_dir_returns_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.v.verify(work_dir=tmpdir)
            assert result.outcome == "unknown"
            assert result.confidence == 0.0
            assert "no test framework detected" in result.detail

    def test_timeout_becomes_infra_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "Cargo.toml").write_text('[package]\nname = "x"\n')
            v = AutoDetectVerifier(timeout=0.000001)
            result = v.verify(work_dir=tmpdir)
            ok = (result.outcome == "unknown" and result.is_infra_failure) or (
                result.outcome == "unknown" and "not found" in result.detail
            )
            assert ok, f"expected infra failure, got {result}"

    def test_detect_vitest_as_typescript(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "package.json").write_text(
                '{\n  "devDependencies": { "vitest": "^1.0.0" }\n}'
            )
            lang = self.v.detect(tmpdir)
            assert lang == "typescript"
