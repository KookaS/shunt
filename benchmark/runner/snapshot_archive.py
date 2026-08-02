"""Commit the per-step state capture, and restore it on a fresh clone."""

# WHY THIS EXISTS. The offline replay derives every per-step verified outcome from the per-step
# `git diff HEAD` captures under `step_snapshots.SNAPSHOT_ROOT`, and that directory is gitignored
# (`benchmark/runner/artifacts/`). So the corpus in git records the *outcomes* but not the raw
# material they were derived from: on any other checkout `offline_replay._handle_absent_snapshots`
# raises `SnapshotsMissingError`, and the corpus is re-scoreable but not re-derivable. Measured on
# the collection host: 792 trajectories, 28 440 `.diff` files, 34 105 506 B of content.
#
# THE SIZE OBJECTION DOES NOT SURVIVE MEASUREMENT. Cumulative diffs are near-duplicates step to
# step, so they compress ~21:1. Measured, on this corpus, as bytes git would actually store in a
# packfile: 1 495 295 B as 792 per-trajectory `.tar.gz`; 1 144 501 B as one archive; 1 054 448 B as
# 28 440 loose files. Every layout is under 2 MB against an 87 MB corpus already in LFS — the
# choice is therefore decided by determinism and churn, not by size.
#
# WHY PER-TRAJECTORY, PLAIN GIT, NOT LFS. LFS stores each object whole with no delta and keys the
# pointer on the content hash, so a re-collection that changes one trajectory would push a fresh
# whole object for it, forever, and the `.jsonl` LFS rule (`.gitattributes`) deliberately does not
# match `.tar.gz`. At ~1.7 MB total, LFS buys nothing and costs delta compression. One archive per
# trajectory keeps a re-collection's diff scoped to the trajectory that changed, and keeps the
# working tree at ~1.7 MB rather than the 52 MB an uncompressed tar of the same content needs.
#
# DETERMINISM IS THE PROPERTY THAT DECIDES COMMITTABILITY, AND IT IS ENFORCED TWICE. `build_archive`
# fixes every field a tar/gzip writer would otherwise draw from the environment — member order,
# mtime, mode, uid/gid, uname/gname, tar format, name encoding, compression level, gzip header
# mtime and filename — so two exports of the same snapshots produce byte-identical archives. That
# is proven, not assumed (`benchmark/tests/test_snapshot_archive.py`). But byte-determinism of a
# COMPRESSED stream is a property of the zlib build, not of this file, and a machine linking a
# different zlib could emit a valid archive with different bytes. So the committed index records
# only content-derived facts (`steps`, `bytes`, `content_sha256` over the restored payload) and
# `export_corpus` is idempotent on CONTENT: an archive whose payload already matches is left
# untouched on disk. A re-export can therefore only dirty the tree when the snapshots themselves
# changed, whatever zlib is underneath.
#
# WHAT THIS DOES NOT CLOSE. The per-step state is one of four replay inputs. The instance images
# (~100 GB) and the gold `patch`/`test_patch` rows (fetched from the HF dataset at replay time) are
# not candidates for committing, so "reproducible on any machine" stays conditional on pulling
# both. `clone_requirements` enumerates every one of them and the CLI exits non-zero listing the
# full set, because a partial reproduction that silently produces different numbers is worse than a
# refusal.

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import logging
import os
import re
import subprocess
import tarfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from benchmark import corpus_lock
from benchmark.routing.authenticity import ERROR, WARN, Finding
from benchmark.runner import step_snapshots, swebench_specs

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# The committed home for the state plane: one subdirectory of the corpus dir, so an export creates
# exactly one new directory and never rewrites a file the stamping stage is holding.
STATE_DIRNAME: Final[str] = "state"
INDEX_FILENAME: Final[str] = "index.json"
ARCHIVE_SUFFIX: Final[str] = ".tar.gz"

# Every knob a tar/gzip writer would otherwise take from the environment, pinned. `GNU_FORMAT`
# rather than the `PAX_FORMAT` default: pax records mtime as a decimal string with sub-second
# precision and adds a per-member extended header, both of which are more surface for a future
# stdlib change to move. Level 9 because the archive is written once and read many times.
_GZIP_LEVEL: Final[int] = 9
_TAR_FORMAT: Final[int] = tarfile.GNU_FORMAT
_TAR_ENCODING: Final[str] = "utf-8"
_MEMBER_MODE: Final[int] = 0o644

# A member name this regex rejects is refused rather than sanitised. Restore never calls
# `tar.extract`/`extractall`: it derives the output path from the parsed integer, so a traversal
# name (`../../x`) cannot name a file even in principle — it simply fails to parse.
_MEMBER_RE: Final[re.Pattern[str]] = re.compile(r"^step_(\d{4,})\.diff$")

_DOCKER_TIMEOUT_S: Final[int] = 20
_SWEBENCH_IMAGE_PREFIX: Final[str] = "swebench/sweb.eval."


# ---------------------------------------------------------------------------
# Paths and the content digest.
# ---------------------------------------------------------------------------


def state_dir(data_dir: Path) -> Path:
    """The committed state plane for a corpus directory."""
    return data_dir / STATE_DIRNAME


def archive_path(trajectory_id: str, data_dir: Path) -> Path:
    """Where one trajectory's committed state archive lives."""
    return state_dir(data_dir) / f"{trajectory_id}{ARCHIVE_SUFFIX}"


def index_path(data_dir: Path) -> Path:
    """The committed, diffable index binding each trajectory to its content digest."""
    return state_dir(data_dir) / INDEX_FILENAME


def content_digest(snapshots: Mapping[int, str]) -> str:
    """Digest the RESTORED payload, independent of how it happens to be packed or compressed."""
    # Length-prefixed per member so no concatenation of two payloads can collide with a third, and
    # keyed on the step index so a renumbering is a difference rather than a silent match. This is
    # the value the index commits to: it is a pure function of the snapshots, with no dependence on
    # tar layout, compression level, or the zlib build.
    digest = hashlib.sha256()
    for index in sorted(snapshots):
        body = snapshots[index].encode("utf-8")
        digest.update(f"{index} {len(body)}\n".encode("ascii"))
        digest.update(body)
    return digest.hexdigest()


def content_bytes(snapshots: Mapping[int, str]) -> int:
    """Total payload size, recorded so the index states what a restore will cost on disk."""
    return sum(len(body.encode("utf-8")) for body in snapshots.values())


# ---------------------------------------------------------------------------
# The archive format: deterministic in, validated out.
# ---------------------------------------------------------------------------


def build_archive(snapshots: Mapping[int, str]) -> bytes:
    """Pack per-step diffs into a byte-reproducible gzipped tar (same input ⇒ same bytes)."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=_TAR_FORMAT, encoding=_TAR_ENCODING) as tar:
        for index in sorted(snapshots):
            body = snapshots[index].encode("utf-8")
            info = tarfile.TarInfo(f"step_{index:04d}.diff")
            info.size = len(body)
            info.mtime = 0
            info.mode = _MEMBER_MODE
            info.type = tarfile.REGTYPE
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            tar.addfile(info, io.BytesIO(body))
    out = io.BytesIO()
    # `mtime=0` kills the gzip header timestamp; a BytesIO has no `.name`, so no FNAME field is
    # written either. Those two are the whole non-determinism budget of the gzip container.
    with gzip.GzipFile(fileobj=out, mode="wb", compresslevel=_GZIP_LEVEL, mtime=0) as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def read_archive(blob: bytes) -> dict[int, str]:
    """Unpack an archive back to per-step diffs, refusing any member that is not a step file."""
    out: dict[int, str] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz", encoding=_TAR_ENCODING) as tar:
        for info in tar.getmembers():
            match = _MEMBER_RE.match(info.name)
            if not info.isfile() or match is None:
                raise ValueError(f"unexpected archive member {info.name!r}")
            handle = tar.extractfile(info)
            if handle is None:
                raise ValueError(f"unreadable archive member {info.name!r}")
            out[int(match.group(1))] = handle.read().decode("utf-8")
    return out


# ---------------------------------------------------------------------------
# The committed index.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexEntry:
    """One trajectory's committed state facts — all of them pure functions of the payload."""

    steps: int
    bytes: int
    content_sha256: str


def entry_for(snapshots: Mapping[int, str]) -> IndexEntry:
    """The index entry a set of snapshots implies."""
    return IndexEntry(
        steps=len(snapshots),
        bytes=content_bytes(snapshots),
        content_sha256=content_digest(snapshots),
    )


def read_index(data_dir: Path) -> dict[str, IndexEntry]:
    """Load the committed index (empty when the state plane was never exported)."""
    path = index_path(data_dir)
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {}
    return {
        str(tid): IndexEntry(
            steps=int(body["steps"]),
            bytes=int(body["bytes"]),
            content_sha256=str(body["content_sha256"]),
        )
        for tid, body in sorted(loaded.items())
    }


def _index_json(index: Mapping[str, IndexEntry]) -> str:
    """Render the index deterministically: sorted keys, fixed indent, trailing newline."""
    body = {
        tid: {"steps": e.steps, "bytes": e.bytes, "content_sha256": e.content_sha256}
        for tid, e in index.items()
    }
    return json.dumps(body, indent=2, sort_keys=True) + "\n"


def _atomic_write_bytes(path: Path, blob: bytes) -> None:
    """Binary sibling of `corpus_lock.atomic_write_text` — a reader sees old or new, never half."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("wb") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Export: scratch -> committed artifact.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportSummary:
    """What one `export_corpus` pass did."""

    trajectories: int
    written: int
    unchanged: int
    orphaned: tuple[str, ...]


def scratch_trajectories(snapshot_root: Path) -> list[str]:
    """Every trajectory id the local scratch holds, sorted."""
    if not snapshot_root.is_dir():
        return []
    return sorted(p.name for p in snapshot_root.iterdir() if p.is_dir())


def export_corpus(
    data_dir: Path, snapshot_root: Path = step_snapshots.SNAPSHOT_ROOT, *, dry_run: bool = False
) -> ExportSummary:
    """Write the committed state plane from the local scratch; idempotent on content."""
    out_dir = state_dir(data_dir)
    index: dict[str, IndexEntry] = {}
    written = unchanged = 0
    trajectories = scratch_trajectories(snapshot_root)
    # The lock scope is the STATE directory, not the corpus directory: an export must not contend
    # with a stamping pass that is holding the corpus lock next door.
    with corpus_lock.corpus_lock(out_dir):
        previous = read_index(data_dir)
        for tid in trajectories:
            snapshots = step_snapshots.read_snapshots(tid, snapshot_root)
            entry = entry_for(snapshots)
            index[tid] = entry
            if _archive_is_current(tid, data_dir, entry):
                unchanged += 1
                continue
            written += 1
            if not dry_run:
                _atomic_write_bytes(archive_path(tid, data_dir), build_archive(snapshots))
        orphaned = tuple(sorted(set(previous) - set(index)))
        if not dry_run:
            corpus_lock.atomic_write_text(index_path(data_dir), _index_json(index))
    return ExportSummary(
        trajectories=len(trajectories), written=written, unchanged=unchanged, orphaned=orphaned
    )


def _archive_is_current(trajectory_id: str, data_dir: Path, entry: IndexEntry) -> bool:
    """True when the committed archive already restores to exactly this payload."""
    # IDEMPOTENCE ON CONTENT, NOT ON BYTES. Comparing the archive's own bytes would make a
    # re-export on a machine with a different zlib rewrite all 792 files for no content change.
    # Comparing what it RESTORES to cannot: only a real snapshot change rewrites anything.
    path = archive_path(trajectory_id, data_dir)
    if not path.is_file():
        return False
    try:
        return content_digest(read_archive(path.read_bytes())) == entry.content_sha256
    except (OSError, ValueError, tarfile.TarError, gzip.BadGzipFile, UnicodeDecodeError):
        return False


# ---------------------------------------------------------------------------
# Import: committed artifact -> scratch.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportSummary:
    """What one `import_corpus` pass did."""

    trajectories: int
    restored: int
    already_present: int
    conflicts: tuple[str, ...]


def import_corpus(
    data_dir: Path, snapshot_root: Path = step_snapshots.SNAPSHOT_ROOT, *, force: bool = False
) -> ImportSummary:
    """Restore the local scratch from the committed state plane so a replay can run."""
    index = read_index(data_dir)
    if not index:
        raise FileNotFoundError(
            f"no committed state plane at {state_dir(data_dir)} — this checkout cannot re-derive "
            "per-step outcomes. Export it on the collection host first "
            "(`python -m benchmark.runner.snapshot_archive export`)."
        )
    restored = present = 0
    conflicts: list[str] = []
    for tid, entry in index.items():
        blob = _require_archive(tid, data_dir)
        snapshots = read_archive(blob)
        actual = content_digest(snapshots)
        if actual != entry.content_sha256:
            raise ValueError(
                f"{tid}: committed archive restores to {actual} but the index binds "
                f"{entry.content_sha256} — the state plane is corrupt or the index is stale"
            )
        local = step_snapshots.read_snapshots(tid, snapshot_root)
        if local and content_digest(local) == actual:
            present += 1
            continue
        if local and not force:
            conflicts.append(tid)
            continue
        step_snapshots.write_snapshots(tid, snapshots, snapshot_root)
        restored += 1
    if conflicts:
        raise ValueError(
            f"{len(conflicts)} trajectories already have a DIFFERENT scratch on this host "
            f"({', '.join(conflicts[:5])}…); refusing to overwrite a local capture with a "
            "committed one. Re-run with --force only if the committed plane is authoritative."
        )
    return ImportSummary(
        trajectories=len(index),
        restored=restored,
        already_present=present,
        conflicts=tuple(conflicts),
    )


def _require_archive(trajectory_id: str, data_dir: Path) -> bytes:
    """Read one committed archive, or say precisely which file the checkout is missing."""
    path = archive_path(trajectory_id, data_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"{trajectory_id}: the index binds a state archive but {path} is absent. If this "
            "corpus is LFS-backed, `git lfs pull` first; otherwise the commit is incomplete."
        )
    return path.read_bytes()


# ---------------------------------------------------------------------------
# Verify: the re-runnable guard over whatever this checkout has.
# ---------------------------------------------------------------------------


def verify_archives(
    data_dir: Path, snapshot_root: Path = step_snapshots.SNAPSHOT_ROOT
) -> list[Finding]:
    """Prove the committed state plane restores to what it claims, and matches any local scratch."""
    out: list[Finding] = []
    index = read_index(data_dir)
    if not index:
        return [
            Finding(
                ERROR,
                "state_archive.absent",
                str(state_dir(data_dir)),
                "no committed state plane — the corpus's per-step outcomes cannot be re-derived "
                "on any checkout but the collection host",
            )
        ]
    for tid, entry in sorted(index.items()):
        out.extend(_archive_findings(tid, data_dir, entry))
    out.extend(_scratch_findings(index, snapshot_root))
    return out


def _archive_findings(trajectory_id: str, data_dir: Path, entry: IndexEntry) -> list[Finding]:
    """One indexed trajectory: the archive exists and restores to the digest the index binds."""
    path = archive_path(trajectory_id, data_dir)
    if not path.is_file():
        return [
            Finding(
                ERROR,
                "state_archive.missing",
                trajectory_id,
                f"the index binds a state archive but {path.name} is absent",
            )
        ]
    try:
        snapshots = read_archive(path.read_bytes())
    except (OSError, ValueError, tarfile.TarError, gzip.BadGzipFile, UnicodeDecodeError) as exc:
        return [Finding(ERROR, "state_archive.unreadable", trajectory_id, str(exc))]
    actual = entry_for(snapshots)
    if actual != entry:
        return [
            Finding(
                ERROR,
                "state_archive.digest_mismatch",
                trajectory_id,
                f"archive restores to {actual.steps} steps / {actual.content_sha256} but the "
                f"index binds {entry.steps} steps / {entry.content_sha256}",
            )
        ]
    return []


def _scratch_findings(index: Mapping[str, IndexEntry], snapshot_root: Path) -> list[Finding]:
    """On a host that HAS the scratch, the committed plane must still describe it."""
    # Absence of a scratch is the normal case off the collection host and is never a failure —
    # that is the whole point of committing the plane. Divergence is: it means the committed
    # artifact and the local capture are two different instruments.
    out: list[Finding] = []
    local = set(scratch_trajectories(snapshot_root))
    if not local:
        return out
    for tid in sorted(local - set(index)):
        out.append(
            Finding(
                WARN,
                "state_archive.unexported",
                tid,
                "present in the local scratch but absent from the committed index — a re-export "
                "would add it",
            )
        )
    for tid in sorted(local & set(index)):
        actual = content_digest(step_snapshots.read_snapshots(tid, snapshot_root))
        if actual != index[tid].content_sha256:
            out.append(
                Finding(
                    ERROR,
                    "state_archive.scratch_drift",
                    tid,
                    f"local scratch digests {actual} but the committed index binds "
                    f"{index[tid].content_sha256} — re-export before trusting either",
                )
            )
    return out


# ---------------------------------------------------------------------------
# What a fresh clone still needs — enumerated, never half-run.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Requirement:
    """One input the offline replay needs, whether this checkout has it, and how to get it."""

    name: str
    satisfied: bool
    detail: str
    remedy: str


def clone_requirements(
    data_dir: Path, snapshot_root: Path = step_snapshots.SNAPSHOT_ROOT
) -> list[Requirement]:
    """Every replay input, checked. Committing the state plane closes ONE of these, not all."""
    return [
        _state_plane_requirement(data_dir),
        _scratch_requirement(data_dir, snapshot_root),
        _module_requirement("swebench", "the SWE-bench grader and its test-command map"),
        _module_requirement("datasets", "the HF client that fetches the gold patch rows"),
        _docker_requirement(),
        _images_requirement(data_dir),
        _gold_rows_requirement(),
    ]


def _state_plane_requirement(data_dir: Path) -> Requirement:
    """Is the per-step state capture committed and self-consistent in this checkout?"""
    # Archive integrity ONLY, not the scratch cross-check `verify_archives` also runs: whether a
    # collection host's live capture has moved on is `state.scratch`'s question, and letting it
    # answer this one would report an un-exported trajectory as "the plane is not committed".
    index = read_index(data_dir)
    failed = [
        finding
        for tid, entry in index.items()
        for finding in _archive_findings(tid, data_dir, entry)
        if finding.severity == ERROR
    ]
    return Requirement(
        name="state.archives",
        satisfied=bool(index) and not failed,
        detail=(
            f"{len(failed)} problems, first: {failed[0].rule} {failed[0].key}"
            if failed
            else f"{len(index)} trajectories in {index_path(data_dir)}"
            if index
            else f"no committed state plane at {state_dir(data_dir)}"
        ),
        remedy="export it on the collection host: `snapshot_archive export` (~1.7 MB, in git)",
    )


def _scratch_requirement(data_dir: Path, snapshot_root: Path) -> Requirement:
    """Is the scratch the replay actually reads present, and does it match the committed plane?"""
    index = read_index(data_dir)
    local = set(scratch_trajectories(snapshot_root))
    missing = sorted(set(index) - local)
    return Requirement(
        name="state.scratch",
        satisfied=bool(index) and not missing,
        detail=(
            f"{len(local)} of {len(index)} indexed trajectories present under {snapshot_root}"
            if index
            else f"{len(local)} local scratch trajectories, but no committed plane to check them "
            f"against ({snapshot_root})"
        ),
        remedy="`python -m benchmark.runner.snapshot_archive import` (restores ~74 MB of scratch)",
    )


def _module_requirement(module: str, what: str) -> Requirement:
    """Is an import-time dependency of the replay actually installed here?"""
    found = importlib.util.find_spec(module) is not None
    return Requirement(
        name=f"deps.{module}",
        satisfied=found,
        detail=f"{module} ({what}) {'importable' if found else 'NOT importable'}",
        remedy="run through `uv run --extra benchmark …` or a `make` target that wraps it",
    )


def _docker_requirement() -> Requirement:
    """Is a Docker daemon reachable? Every re-derivation runs inside the instance image."""
    ok, detail = _docker_info()
    return Requirement(
        name="docker.daemon",
        satisfied=ok,
        detail=detail,
        remedy="start Docker and `docker login` (unauthenticated pulls of swebench/* hit 429)",
    )


def _docker_info() -> tuple[bool, str]:
    """Probe the daemon without raising when docker is absent or unreachable."""
    try:
        done = subprocess.run(  # noqa: S603
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=_DOCKER_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"docker not runnable: {exc}"
    if done.returncode != 0:
        return False, f"docker daemon unreachable (exit {done.returncode})"
    return True, f"docker server {done.stdout.strip()}"


def _images_requirement(data_dir: Path) -> Requirement:
    """Are the ~100 GB of SWE-bench instance images present? They are NEVER committable."""
    # `swebench_specs.image_ref` owns the tag mangling (`__` -> `_1776_`, lowercased). Re-deriving
    # it here would be a second spelling of the same rule, and the first thing to drift.
    needed = corpus_instances(data_dir)
    refs = swebench_specs.spec_image_refs(sorted(needed))
    local = _local_swebench_images()
    have = {i for i in needed if refs.get(i, "").lower() in local}
    return Requirement(
        name="docker.images",
        satisfied=bool(needed) and len(have) == len(needed),
        detail=(
            f"{len(have)} of {len(needed)} instance images present locally "
            f"(~100 GB for the full set; not committable, pulled per instance)"
        ),
        remedy="the replay pulls `swebench/sweb.eval.x86_64.<instance>` on demand; pre-pull to "
        "avoid rate limits",
    )


def _local_swebench_images() -> set[str]:
    """Every SWE-bench instance image tag this host already holds."""
    try:
        done = subprocess.run(  # noqa: S603
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=_DOCKER_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    if done.returncode != 0:
        return set()
    return {
        line.strip().lower()
        for line in done.stdout.splitlines()
        if line.startswith(_SWEBENCH_IMAGE_PREFIX)
    }


def _gold_rows_requirement() -> Requirement:
    """The gold patch/test_patch rows are fetched at replay time and are not in the tree."""
    # Deliberately NOT probed by hitting the network: a requirements check that makes a request is
    # a check that fails differently on a flaky link. This states the dependency instead, which is
    # what a clone actually needs to be told.
    return Requirement(
        name="dataset.gold_rows",
        satisfied=False,
        detail="gold `patch`/`test_patch` are fetched from the HF dataset at replay time and are "
        "not committed; `offline_replay._dataset_row` passes no `revision=`, so the fetch is not "
        "pinned to `swebench_specs.DATASET_REVISION`",
        remedy="network + HF access at replay time; pin the revision (and vendor the rows) to "
        "remove the dependency",
    )


def corpus_instances(data_dir: Path) -> set[str]:
    """The SWE-bench instances this corpus needs images for, read from the trajectory headers."""
    # Header line only: `schema.load_jsonl` would parse every step of an 87 MB LFS corpus to
    # answer a question the first line already answers.
    out: set[str] = set()
    for path in sorted(data_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            first = handle.readline()
        if not first.strip():
            continue
        header = json.loads(first)
        instance = header.get("instance_id")
        if isinstance(instance, str) and instance:
            out.add(instance)
    return out


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _print_requirements(requirements: Iterable[Requirement]) -> int:
    """Print every requirement and fail on the first unmet one — never a partial go-ahead."""
    unmet = []
    for req in requirements:
        mark = "OK  " if req.satisfied else "MISS"
        print(f"{mark} {req.name}: {req.detail}")
        if not req.satisfied:
            unmet.append(req)
    if not unmet:
        print("replay inputs: all present")
        return 0
    print(f"\n{len(unmet)} replay inputs are MISSING — a replay run would not be reproducible:")
    for req in unmet:
        print(f"  {req.name}: {req.remedy}")
    return 1


def _main() -> int:
    import argparse  # noqa: PLC0415

    from benchmark.escalation.live_capture import LIVE_DIR  # noqa: PLC0415
    from benchmark.routing.authenticity import errors, warnings  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("export", "import", "verify", "requirements"))
    ap.add_argument("--data-dir", type=Path, default=LIVE_DIR, help="committed corpus directory")
    ap.add_argument(
        "--snapshot-root",
        type=Path,
        default=step_snapshots.SNAPSHOT_ROOT,
        help="per-step diff scratch (gitignored; absent on any other checkout)",
    )
    ap.add_argument("--dry-run", action="store_true", help="export: report without writing")
    ap.add_argument("--force", action="store_true", help="import: overwrite a diverging scratch")
    args = ap.parse_args()

    if args.command == "export":
        print(export_corpus(args.data_dir, args.snapshot_root, dry_run=args.dry_run))
        return 0
    if args.command == "import":
        print(import_corpus(args.data_dir, args.snapshot_root, force=args.force))
        return 0
    if args.command == "requirements":
        return _print_requirements(clone_requirements(args.data_dir, args.snapshot_root))

    findings = verify_archives(args.data_dir, args.snapshot_root)
    for finding in warnings(findings):
        print(f"WARN  {finding.rule} {finding.key}: {finding.detail}")
    failures = errors(findings)
    for finding in failures:
        print(f"ERROR {finding.rule} {finding.key}: {finding.detail}")
    print(f"state-archive check: {len(failures)} errors, {len(warnings(findings))} warnings")
    return 1 if failures else 0


__all__ = [
    "ARCHIVE_SUFFIX",
    "INDEX_FILENAME",
    "STATE_DIRNAME",
    "ExportSummary",
    "ImportSummary",
    "IndexEntry",
    "Requirement",
    "archive_path",
    "build_archive",
    "clone_requirements",
    "content_bytes",
    "content_digest",
    "corpus_instances",
    "entry_for",
    "export_corpus",
    "import_corpus",
    "index_path",
    "read_archive",
    "read_index",
    "scratch_trajectories",
    "state_dir",
    "verify_archives",
]


if __name__ == "__main__":
    raise SystemExit(_main())
