"""The stamping stage's shared state under REAL concurrency — every claim proved by execution.

Each test reproduces a race measured against the unsynchronised code, so reverting the fix it
pins turns the test red rather than merely making the stage slower.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

from benchmark import corpus_lock, pipeline
from benchmark.escalation import authenticity, schema
from benchmark.runner import offline_replay, replay_admissibility
from tests.escalation.factories import make_step, make_trajectory

# Enough contention to break the unsynchronised version reliably (measured on it: 8x60 concurrent
# `record_verdict` calls raised 282 uncaught JSONDecodeErrors and lost 413 of 480 verdicts) while
# keeping the suite fast.
_PROCS = 6
_PER_PROC = 20

# fork, not spawn: the child must inherit the already-imported benchmark package rather than
# re-resolve it, which in a worktree can bind a DIFFERENT checkout's source.
_CTX = mp.get_context("fork")


def _verdict(instance_id: str) -> replay_admissibility.AdmissibilityVerdict:
    base = replay_admissibility.LegOutcome("base", "failure", 1, False, "d" * 400)
    gold = replay_admissibility.LegOutcome("gold", "success", 0, False, "d" * 400)
    return replay_admissibility.AdmissibilityVerdict(
        instance_id=instance_id,
        admissible=False,
        reason="probe " + "x" * 200,
        base=base,
        gold=gold,
        gate_key="k",
    )


def _run(targets: list[tuple[object, tuple[object, ...]]]) -> None:
    """Start every (target, args) as a real process and require all of them to exit clean."""
    procs = [_CTX.Process(target=t, args=a) for t, a in targets]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=180)
    assert [p.exitcode for p in procs] == [0] * len(procs)


# --------------------------------------------------------------------------- the primitive


def test_atomic_write_replaces_whole_and_leaves_no_temp(tmp_path):
    target = tmp_path / "f.json"
    corpus_lock.atomic_write_text(target, "one")
    corpus_lock.atomic_write_text(target, "two")
    assert target.read_text() == "two"
    assert list(tmp_path.iterdir()) == [target]  # the sibling temp is renamed, never left behind


def _hold_lock(data_dir: str, hold_s: float, log: str) -> None:
    with corpus_lock.corpus_lock(Path(data_dir)):
        with Path(log).open("a") as handle:
            handle.write("enter\n")
        time.sleep(hold_s)
        with Path(log).open("a") as handle:
            handle.write("exit\n")


def test_the_lock_excludes_a_second_process(tmp_path):
    log = tmp_path / "order.log"
    _run([(_hold_lock, (str(tmp_path), 0.3, str(log))) for _ in range(3)])
    # Strict alternation: no process was ever inside while another was.
    assert log.read_text().split() == ["enter", "exit"] * 3


def test_the_lock_is_reentrant_within_one_thread(tmp_path):
    # `clear_rejected` records a verdict (one transaction) and then commits the cleared trajectory
    # (another). A non-re-entrant flock would deadlock the replay child against itself.
    with corpus_lock.corpus_lock(tmp_path), corpus_lock.corpus_lock(tmp_path):
        pass


def test_a_second_directory_still_takes_its_own_lock(tmp_path):
    # Re-entrancy is keyed on the directory: holding one must not let an unrelated directory's
    # acquisition silently skip its flock.
    a, b = tmp_path / "a", tmp_path / "b"
    with corpus_lock.corpus_lock(a), corpus_lock.corpus_lock(b):
        assert (a / corpus_lock.LOCK_NAME).exists()
        assert (b / corpus_lock.LOCK_NAME).exists()


# --------------------------------------------------------------------------- admissibility.json


def _record_many(data_dir: str, lo: int, hi: int) -> None:
    for i in range(lo, hi):
        replay_admissibility.record_verdict(_verdict(f"inst-{i:04d}"), Path(data_dir))


def test_concurrent_verdicts_lose_nothing_and_leave_a_parseable_file(tmp_path):
    # UNSYNCHRONISED THIS WAS NOT SLOW, IT WAS WRONG. `record_verdict` did a bare
    # `json.loads(read_text())`: it raised an uncaught JSONDecodeError on 282 of 480 concurrent
    # calls, lost 413 of 480 verdicts, and left the file itself unparseable — after which
    # `load_verdicts` returns {} and every clearing in the corpus loses the record saying WHY its
    # stamps were removed. A cleared trajectory with no recorded reason is the failure this file
    # exists to prevent.
    _run(
        [(_record_many, (str(tmp_path), w * _PER_PROC, (w + 1) * _PER_PROC)) for w in range(_PROCS)]
    )
    expected = _PROCS * _PER_PROC
    assert len(json.loads((tmp_path / replay_admissibility.VERDICT_FILENAME).read_text())) == (
        expected
    )
    assert len(replay_admissibility.load_verdicts(tmp_path)) == expected


def _ledger_many(data_dir: str, lo: int, hi: int) -> None:
    for i in range(lo, hi):
        pipeline.record_stamped(Path(data_dir), f"traj-{i:04d}", "digest")


def test_concurrent_ledger_writes_lose_no_completed_trajectory(tmp_path):
    # The ledger is the resume point, written by every worker as its replay lands. A lost update
    # re-queues a trajectory that is already done (hours of redundant container time on a 137 h
    # rebuild); a torn write reads back as "nothing is done" and restarts the whole thing.
    _run(
        [(_ledger_many, (str(tmp_path), w * _PER_PROC, (w + 1) * _PER_PROC)) for w in range(_PROCS)]
    )
    assert len(pipeline.load_stamp_ledger(tmp_path)) == _PROCS * _PER_PROC


def test_recording_one_verdict_preserves_records_it_cannot_parse(tmp_path):
    # A read-modify-write must carry OTHER instances through byte-for-byte; dropping what this
    # module cannot decode would silently delete the audit trail of a clearing it knows nothing
    # about.
    path = tmp_path / replay_admissibility.VERDICT_FILENAME
    path.write_text(json.dumps({"future-instance": {"shape": "unknown"}}))
    replay_admissibility.record_verdict(_verdict("inst-1"), tmp_path)
    assert json.loads(path.read_text())["future-instance"] == {"shape": "unknown"}


# --------------------------------------------------------------------------- the corpus files


def _seed(out: Path, count: int, steps: int = 4, filler: int = 0) -> None:
    for i in range(count):
        schema.dump_jsonl(
            make_trajectory(
                [make_step(step_index=s, result="x" * filler) for s in range(steps)],
                trajectory_id=f"t{i}",
            ),
            out / f"t{i}.jsonl",
        )


def _flip(path: Path) -> schema.Trajectory:
    """The trajectory at *path* with every step stamped — a real restamp's shape."""
    traj = schema.load_jsonl(path)
    steps = [
        replace(s, confirmed=True, success=False, blocking=True, exit_code=1) for s in traj.steps
    ]
    return schema.Trajectory(
        header=replace(traj.header, content_sha256=schema.content_sha256(steps)), steps=steps
    )


def _slow_commit(data_dir: str, name: str, pause_on: str, hold_s: float) -> None:
    """Commit *name*, pausing the manifest rebuild AFTER it read the sibling a rival rewrites."""
    # The pause is on a file that sorts LAST, so by the time it fires this worker has already read
    # the rival's trajectory in its pre-rewrite state — which is exactly the shape of the real
    # window (`manifest()` reads 799 files in 0.45 s while other workers are writing into them).
    real_load = schema.load_jsonl

    def slow_load(path: Path) -> schema.Trajectory:
        traj = real_load(path)
        if path.name == pause_on:
            time.sleep(hold_s)
        return traj

    target = Path(data_dir) / name
    traj = _flip(target)
    schema.load_jsonl = slow_load
    offline_replay.commit_trajectory(traj, target)


def _quick_commit(data_dir: str, name: str, delay_s: float) -> None:
    time.sleep(delay_s)
    target = Path(data_dir) / name
    offline_replay.commit_trajectory(_flip(target), target)


def test_two_workers_committing_at_once_keep_the_manifest_bound_to_what_is_on_disk(tmp_path):
    # THE WRONG-DATA RACE. `authenticity.manifest` hashes every *.jsonl in the directory, so a
    # worker rebuilding it from a snapshot a sibling has already superseded records a STALE
    # `content_sha256`. Measured on the unpaired code: `verify_manifest` reported
    # `manifest.hash_mismatch` — the corpus failing its own Layer-1 integrity check. Pairing the
    # trajectory write and the manifest rebuild inside ONE `corpus_lock` transaction is what makes
    # that unreachable; atomic writes alone do NOT (both writes are individually whole).
    _seed(tmp_path, 6)
    _run(
        [
            (_slow_commit, (str(tmp_path), "t0.jsonl", "t5.jsonl", 1.0)),
            (_quick_commit, (str(tmp_path), "t1.jsonl", 0.3)),
        ]
    )
    findings = authenticity.verify_manifest(tmp_path)
    assert [f for f in findings if f.severity == authenticity.ERROR] == []
    # And the manifest is not merely self-consistent: it lists BOTH workers' new hashes.
    recorded = json.loads((tmp_path / authenticity.MANIFEST_NAME).read_text())["trajectories"]
    for name in ("t0", "t1"):
        header = schema.load_jsonl(tmp_path / f"{name}.jsonl").header
        assert recorded[name]["content_sha256"] == header.content_sha256
        assert all(step.confirmed for step in schema.load_jsonl(tmp_path / f"{name}.jsonl").steps)


def _rewrite_forever(data_dir: str, name: str, rounds: int) -> None:
    path = Path(data_dir) / name
    small = schema.load_jsonl(path)
    steps = [replace(s, result="y" * 200_000) for s in small.steps]
    big = schema.Trajectory(
        header=replace(small.header, content_sha256=schema.content_sha256(steps)), steps=steps
    )
    for _ in range(rounds):
        schema.dump_jsonl(small, path)
        schema.dump_jsonl(big, path)


def _read_forever(data_dir: str, rounds: int, crash_log: str) -> None:
    crashes: list[str] = []
    for _ in range(rounds):
        try:
            authenticity.manifest(Path(data_dir))
        except Exception as exc:  # noqa: BLE001 — the failure this test exists to forbid
            crashes.append(f"{type(exc).__name__}: {exc}"[:160])
    Path(crash_log).write_text("\n".join(crashes))


def test_a_trajectory_being_rewritten_is_never_read_half_written(tmp_path):
    # `dump_jsonl` used a plain `write_text`, so a concurrent `manifest()` read saw a truncated
    # file (`IndexError` on an empty read, `Unterminated string` on a partial line). One large
    # trajectory alone in the directory makes the reader's window and the writer's overlap, so the
    # old behaviour fails this in the first few rounds. Temp+rename means a reader sees the whole
    # old file or the whole new one, never a prefix — and a `kill -9` mid-rebuild cannot leave a
    # truncated trajectory in the committed corpus.
    corpus = tmp_path / "live"
    corpus.mkdir()
    _seed(corpus, 1, steps=20, filler=1_000)
    crash_log = tmp_path / "crashes.txt"
    _run(
        [
            (_rewrite_forever, (str(corpus), "t0.jsonl", 60)),
            (_read_forever, (str(corpus), 60, str(crash_log))),
        ]
    )
    assert crash_log.read_text() == ""
