"""Unit tests for the committed per-step state plane: determinism, round-trip, refusals."""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from typing import TYPE_CHECKING, Final

import pytest

from benchmark.runner import snapshot_archive, step_snapshots

if TYPE_CHECKING:
    from pathlib import Path

SNAPSHOTS: Final[dict[int, str]] = {
    0: "",
    1: "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n",
    2: "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+c\n",
    12: "diff --git a/ü.py b/ü.py\n+é\n",
}


def _seed_scratch(root: Path, trajectory_id: str, snapshots: dict[int, str]) -> None:
    step_snapshots.write_snapshots(trajectory_id, snapshots, root)


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


class TestArchiveDeterminism:
    """The property that decides committability: same payload in, same bytes out."""

    def test_two_builds_are_byte_identical(self) -> None:
        assert snapshot_archive.build_archive(SNAPSHOTS) == snapshot_archive.build_archive(
            SNAPSHOTS
        )

    def test_member_order_does_not_depend_on_dict_order(self) -> None:
        shuffled = dict(reversed(list(SNAPSHOTS.items())))
        assert snapshot_archive.build_archive(shuffled) == snapshot_archive.build_archive(SNAPSHOTS)

    def test_no_timestamp_in_the_gzip_header(self) -> None:
        # Bytes 4:8 are MTIME. A non-zero value would re-dirty the tree on every export.
        assert snapshot_archive.build_archive(SNAPSHOTS)[4:8] == b"\x00\x00\x00\x00"

    def test_members_carry_no_environment_fields(self) -> None:
        blob = snapshot_archive.build_archive(SNAPSHOTS)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            infos = tar.getmembers()
        assert [i.name for i in infos] == [
            "step_0000.diff",
            "step_0001.diff",
            "step_0002.diff",
            "step_0012.diff",
        ]
        assert all(i.mtime == 0 and i.uid == 0 and i.gid == 0 for i in infos)
        assert all(i.uname == "" and i.gname == "" for i in infos)


class TestArchiveRoundTrip:
    """Unpacking must return exactly what was packed, and refuse anything else."""

    def test_round_trip(self) -> None:
        assert snapshot_archive.read_archive(snapshot_archive.build_archive(SNAPSHOTS)) == SNAPSHOTS

    def test_digest_is_stable_and_order_free(self) -> None:
        shuffled = dict(reversed(list(SNAPSHOTS.items())))
        assert snapshot_archive.content_digest(shuffled) == snapshot_archive.content_digest(
            SNAPSHOTS
        )

    def test_digest_separates_a_renumbering(self) -> None:
        moved = {k + 1: v for k, v in SNAPSHOTS.items()}
        assert snapshot_archive.content_digest(moved) != snapshot_archive.content_digest(SNAPSHOTS)

    def test_traversal_member_is_refused(self) -> None:
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w") as tar:
            info = tarfile.TarInfo("../../escape.diff")
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
        blob = io.BytesIO()
        with gzip.GzipFile(fileobj=blob, mode="wb", mtime=0) as gz:
            gz.write(raw.getvalue())
        with pytest.raises(ValueError, match="unexpected archive member"):
            snapshot_archive.read_archive(blob.getvalue())


class TestExportImport:
    """Export writes the committed plane; import restores a scratch that is byte-identical."""

    def test_restore_into_an_empty_root_is_byte_identical(self, tmp_path: Path) -> None:
        origin, data, restored = tmp_path / "a", tmp_path / "d", tmp_path / "b"
        _seed_scratch(origin, "inst__model__arm", SNAPSHOTS)
        _seed_scratch(origin, "inst2__model__arm", {0: "x\n"})
        snapshot_archive.export_corpus(data, origin)
        summary = snapshot_archive.import_corpus(data, restored)
        assert summary.restored == 2
        assert _tree(restored) == _tree(origin)

    def test_export_is_idempotent_on_content(self, tmp_path: Path) -> None:
        origin, data = tmp_path / "a", tmp_path / "d"
        _seed_scratch(origin, "inst__model__arm", SNAPSHOTS)
        first = snapshot_archive.export_corpus(data, origin)
        archive = snapshot_archive.archive_path("inst__model__arm", data)
        before = archive.stat().st_mtime_ns
        second = snapshot_archive.export_corpus(data, origin)
        assert (first.written, first.unchanged) == (1, 0)
        assert (second.written, second.unchanged) == (0, 1)
        assert archive.stat().st_mtime_ns == before

    def test_index_is_deterministic_text(self, tmp_path: Path) -> None:
        origin, data = tmp_path / "a", tmp_path / "d"
        _seed_scratch(origin, "inst__model__arm", SNAPSHOTS)
        snapshot_archive.export_corpus(data, origin)
        text = snapshot_archive.index_path(data).read_text(encoding="utf-8")
        entry = json.loads(text)["inst__model__arm"]
        assert text.endswith("\n")
        assert entry["steps"] == len(SNAPSHOTS)
        assert entry["content_sha256"] == snapshot_archive.content_digest(SNAPSHOTS)

    def test_import_without_a_committed_plane_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no committed state plane"):
            snapshot_archive.import_corpus(tmp_path / "d", tmp_path / "b")

    def test_import_refuses_to_clobber_a_diverging_scratch(self, tmp_path: Path) -> None:
        origin, data, other = tmp_path / "a", tmp_path / "d", tmp_path / "b"
        _seed_scratch(origin, "inst__model__arm", SNAPSHOTS)
        snapshot_archive.export_corpus(data, origin)
        _seed_scratch(other, "inst__model__arm", {0: "something else\n"})
        with pytest.raises(ValueError, match="refusing to overwrite"):
            snapshot_archive.import_corpus(data, other)
        assert snapshot_archive.import_corpus(data, other, force=True).restored == 1

    def test_import_is_a_no_op_when_the_scratch_already_matches(self, tmp_path: Path) -> None:
        origin, data = tmp_path / "a", tmp_path / "d"
        _seed_scratch(origin, "inst__model__arm", SNAPSHOTS)
        snapshot_archive.export_corpus(data, origin)
        summary = snapshot_archive.import_corpus(data, origin)
        assert (summary.restored, summary.already_present) == (0, 1)

    def test_import_rejects_a_tampered_archive(self, tmp_path: Path) -> None:
        origin, data, restored = tmp_path / "a", tmp_path / "d", tmp_path / "b"
        _seed_scratch(origin, "inst__model__arm", SNAPSHOTS)
        snapshot_archive.export_corpus(data, origin)
        path = snapshot_archive.archive_path("inst__model__arm", data)
        path.write_bytes(snapshot_archive.build_archive({0: "forged\n"}))
        with pytest.raises(ValueError, match="the state plane is corrupt"):
            snapshot_archive.import_corpus(data, restored)


class TestVerify:
    """The re-runnable guard: it must fail on absence, corruption, and scratch drift."""

    def test_absent_plane_is_an_error(self, tmp_path: Path) -> None:
        findings = snapshot_archive.verify_archives(tmp_path / "d", tmp_path / "b")
        assert [f.rule for f in findings] == ["state_archive.absent"]

    def test_exported_plane_verifies_clean(self, tmp_path: Path) -> None:
        origin, data = tmp_path / "a", tmp_path / "d"
        _seed_scratch(origin, "inst__model__arm", SNAPSHOTS)
        snapshot_archive.export_corpus(data, origin)
        assert snapshot_archive.verify_archives(data, origin) == []

    def test_missing_archive_is_caught(self, tmp_path: Path) -> None:
        origin, data = tmp_path / "a", tmp_path / "d"
        _seed_scratch(origin, "inst__model__arm", SNAPSHOTS)
        snapshot_archive.export_corpus(data, origin)
        snapshot_archive.archive_path("inst__model__arm", data).unlink()
        findings = snapshot_archive.verify_archives(data, tmp_path / "b")
        assert [f.rule for f in findings] == ["state_archive.missing"]

    def test_scratch_drift_is_caught(self, tmp_path: Path) -> None:
        origin, data = tmp_path / "a", tmp_path / "d"
        _seed_scratch(origin, "inst__model__arm", SNAPSHOTS)
        snapshot_archive.export_corpus(data, origin)
        _seed_scratch(origin, "inst__model__arm", {0: "drifted\n", 1: "x\n", 2: "y\n", 12: "z\n"})
        rules = [f.rule for f in snapshot_archive.verify_archives(data, origin)]
        assert "state_archive.scratch_drift" in rules

    def test_unexported_scratch_is_a_warning_not_a_pass(self, tmp_path: Path) -> None:
        origin, data = tmp_path / "a", tmp_path / "d"
        _seed_scratch(origin, "inst__model__arm", SNAPSHOTS)
        snapshot_archive.export_corpus(data, origin)
        _seed_scratch(origin, "later__model__arm", {0: "new\n"})
        findings = snapshot_archive.verify_archives(data, origin)
        assert [(f.severity, f.rule) for f in findings] == [("WARN", "state_archive.unexported")]


class TestCloneRequirements:
    """A clone must be told every input it still lacks — never a partial go-ahead."""

    def test_absent_plane_reports_both_state_requirements_unmet(self, tmp_path: Path) -> None:
        reqs = {r.name: r for r in snapshot_archive.clone_requirements(tmp_path, tmp_path / "b")}
        assert not reqs["state.archives"].satisfied
        assert not reqs["state.scratch"].satisfied

    def test_gold_rows_are_never_reported_satisfied(self, tmp_path: Path) -> None:
        reqs = {r.name: r for r in snapshot_archive.clone_requirements(tmp_path, tmp_path / "b")}
        assert not reqs["dataset.gold_rows"].satisfied
        assert "not committed" in reqs["dataset.gold_rows"].detail

    def test_exported_and_restored_plane_satisfies_the_state_requirements(
        self, tmp_path: Path
    ) -> None:
        origin, data = tmp_path / "a", tmp_path / "d"
        _seed_scratch(origin, "inst__model__arm", SNAPSHOTS)
        snapshot_archive.export_corpus(data, origin)
        reqs = {r.name: r for r in snapshot_archive.clone_requirements(data, origin)}
        assert reqs["state.archives"].satisfied
        assert reqs["state.scratch"].satisfied

    def test_corpus_instances_reads_only_the_header(self, tmp_path: Path) -> None:
        (tmp_path / "t.jsonl").write_text(
            json.dumps({"instance_id": "astropy__astropy-12907"}) + "\n" + "{bad json\n",
            encoding="utf-8",
        )
        assert snapshot_archive.corpus_instances(tmp_path) == {"astropy__astropy-12907"}
