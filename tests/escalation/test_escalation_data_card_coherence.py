"""Every count in docs/escalation-data-card.md is pinned to its source, never editorial."""

# The data card publishes the corpus's census, its per-model stamping coverage, its
# unmeasured-state share and its field census. Each of those rots the moment the corpus
# moves, and a data card that silently states a stale population is worse than none — it
# is the "under this data" half of every escalation claim. So each number is re-derived
# here from `corpus.census()`, the committed `reports/metrics.json`, or the corpus itself,
# and a disagreement fails.

from __future__ import annotations

import dataclasses
import json
import re
import statistics
from functools import cache
from pathlib import Path

from benchmark.escalation import features, schema
from benchmark.escalation.corpus import LIVE_DIR, census

_ROOT = Path(__file__).resolve().parents[2]
_DOC = _ROOT / "docs" / "escalation-data-card.md"
_METRICS = _ROOT / "benchmark" / "escalation" / "reports" / "metrics.json"


def _doc_text() -> str:
    return _DOC.read_text(encoding="utf-8")


@cache
def _trajectories() -> tuple[schema.Trajectory, ...]:
    """The committed corpus, loaded once — the same population `census()` counts."""
    return tuple(schema.load_jsonl(path) for path in sorted(LIVE_DIR.glob("*.jsonl")))


@cache
def _metrics() -> dict[str, object]:
    return dict(json.loads(_METRICS.read_text(encoding="utf-8")))


def _run_block() -> dict[str, object]:
    run = _metrics()["run"]
    assert isinstance(run, dict)
    return run


def _summary_row(label: str) -> str:
    """The right-hand cell of the doc's `| label | value |` summary row."""
    m = re.search(rf"^\|\s*{re.escape(label)}\s*\|([^|]+)\|\s*$", _doc_text(), re.MULTILINE)
    assert m, f"summary row {label!r} not found in the data card"
    return m.group(1).strip()


def _int(text: str) -> int:
    return int(text.replace(",", "").replace("**", "").strip())


def test_headline_census_matches_corpus_census() -> None:
    # The two numbers every other count is a subset of. `census()` is the ONE place the
    # corpus size is derived; a literal here that disagrees with it is the rot this guards.
    got = census()
    assert _int(_summary_row("Trajectories (one per live session)")) == got.trajectories
    assert _int(_summary_row("Steps")) == got.steps


def test_stamped_count_matches_metrics_and_the_live_corpus() -> None:
    stamped = sum(features.is_stamped(t) for t in _trajectories())
    assert _run_block()["n_stamped"] == stamped, "metrics.json is stale against the corpus"
    assert _run_block()["n_trajectories"] == census().trajectories
    assert _int(_summary_row("Per-step stamped trajectories")) == stamped


def test_structural_summary_rows_match_the_corpus() -> None:
    trajs = _trajectories()
    instances = {t.header.instance_id for t in trajs}
    repos = {(t.header.instance_id or "").split("__")[0] for t in trajs}
    models = {features.model_of(t) for t in trajs}
    efforts = {t.header.trajectory_id.rsplit("__", 1)[-1] for t in trajs}
    assert _int(_summary_row("Distinct SWE-bench Verified instances")) == len(instances)
    assert _int(_summary_row("Upstream repositories")) == len(repos)
    assert _int(_summary_row("Models")) == len(models)
    assert _int(_summary_row("Reasoning arms")) == len(efforts)


def test_run_length_row_matches_the_corpus() -> None:
    lengths = sorted(len(t.steps) for t in _trajectories())
    stated = _summary_row("Median steps per run (range)")
    m = re.fullmatch(r"\*\*(\d+)\*\* \((\d+)–(\d+)\)", stated)
    assert m, f"unparseable run-length row: {stated!r}"
    median, low, high = (int(g) for g in m.groups())
    assert median == statistics.median(lengths)
    assert (low, high) == (lengths[0], lengths[-1])


def test_failure_rates_match_the_corpus() -> None:
    trajs = _trajectories()
    stamped = [t for t in trajs if features.is_stamped(t)]
    m = re.search(
        r"is \*\*([\d.]+)\*\* over all \d+ trajectories and \*\*([\d.]+)\*\* over the \d+ stamped",
        _doc_text(),
    )
    assert m, "the two terminal-failure-rate claims were not found in the data card"
    corpus_rate = sum(not t.header.terminal_resolved for t in trajs) / len(trajs)
    stamped_rate = sum(not t.header.terminal_resolved for t in stamped) / len(stamped)
    assert float(m.group(1)) == round(corpus_rate, 3)
    assert float(m.group(2)) == round(stamped_rate, 3)


def _doc_per_model_rows() -> dict[str, tuple[int, int, int, float]]:
    """The data card's per-model table: model -> (total, stamped, unstamped, share)."""
    rows: dict[str, tuple[int, int, int, float]] = {}
    for line in _doc_text().splitlines():
        m = re.fullmatch(
            r"\|\s*([a-z0-9.\-]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|",
            line.strip(),
        )
        if m:
            total, stamped, unstamped = (int(m.group(i)) for i in (2, 3, 4))
            rows[m.group(1)] = (total, stamped, unstamped, float(m.group(5)))
    assert rows, "the per-model coverage table was not found in the data card"
    return rows


def test_per_model_table_matches_the_live_corpus() -> None:
    totals: dict[str, int] = {}
    stamped: dict[str, int] = {}
    for traj in _trajectories():
        model = features.model_of(traj)
        totals[model] = totals.get(model, 0) + 1
        stamped[model] = stamped.get(model, 0) + features.is_stamped(traj)
    rows = _doc_per_model_rows()
    assert set(rows) == set(totals), "the data card's model set differs from the corpus's"
    for model, (doc_total, doc_stamped, doc_unstamped, doc_share) in rows.items():
        assert doc_total == totals[model], model
        assert doc_stamped == stamped[model], model
        assert doc_unstamped == totals[model] - stamped[model], model
        assert doc_share == round(stamped[model] / totals[model], 3), model


def test_per_model_table_matches_the_committed_metrics() -> None:
    # The published figures are drawn from metrics.json, so the card must agree with THAT
    # too — otherwise the page and the plots could quote different populations.
    coverage = _metrics()["corpus_and_coverage.png"]
    assert isinstance(coverage, dict)
    rows = _doc_per_model_rows()
    for entry in coverage["coverage"]:
        model = entry["model"]
        assert rows[model][0] == entry["n_trajectories"], model
        assert rows[model][1] == entry["n_stamped"], model


def test_totals_row_matches_the_sum_of_the_table() -> None:
    m = re.search(
        r"\|\s*\*\*Total\*\*\s*\|\s*\*\*(\d+)\*\*\s*\|\s*\*\*(\d+)\*\*\s*\|"
        r"\s*\*\*(\d+)\*\*\s*\|\s*\*\*([\d.]+)\*\*\s*\|",
        _doc_text(),
    )
    assert m, "the per-model totals row was not found"
    rows = _doc_per_model_rows()
    total = sum(r[0] for r in rows.values())
    stamped = sum(r[1] for r in rows.values())
    assert (int(m.group(1)), int(m.group(2)), int(m.group(3))) == (
        total,
        stamped,
        total - stamped,
    )
    assert float(m.group(4)) == round(stamped / total, 3)


def test_unstamped_share_range_matches_the_corpus() -> None:
    # The card names the extremes of the model-correlated drop; those two percentages are
    # the whole point of the claim, so they are pinned rather than described.
    shares = []
    for traj in _trajectories():
        shares.append((features.model_of(traj), features.is_stamped(traj)))
    by_model: dict[str, list[bool]] = {}
    for model, is_st in shares:
        by_model.setdefault(model, []).append(is_st)
    drops = {m: 1 - sum(v) / len(v) for m, v in by_model.items()}
    pattern = r"ranges from ([\d.]+)% of `([\w.\-]+)` runs\s*to ([\d.]+)% of `([\w.\-]+)` runs"
    m = re.search(pattern, _doc_text())
    assert m, "the model-correlated drop-rate claim was not found"
    assert float(m.group(1)) == round(100 * drops[m.group(2)], 1)
    assert float(m.group(3)) == round(100 * drops[m.group(4)], 1)
    assert drops[m.group(2)] == min(drops.values())
    assert drops[m.group(4)] == max(drops.values())


def _unmeasured() -> tuple[int, int]:
    """Steps carrying the unmeasured-state sentinel, and the runs they fall on."""
    # `state_capture_audit` writes (success=True, confirmed=False, is_infra_failure=True)
    # and no other path does; this is the same predicate that module's `_is_unmeasured` uses.
    steps = 0
    runs = 0
    for traj in _trajectories():
        hit = sum(
            step.success and not step.confirmed and step.is_infra_failure for step in traj.steps
        )
        steps += hit
        runs += bool(hit)
    return steps, runs


def test_unmeasured_state_counts_match_the_corpus() -> None:
    steps, runs = _unmeasured()
    m = re.search(
        r"\*\*([\d,]+) steps — ([\d.]+)% of the corpus — across (\d+) of the (\d+) runs",
        _doc_text(),
    )
    assert m, "the unmeasured-state claim was not found in the data card"
    assert _int(m.group(1)) == steps
    assert float(m.group(2)) == round(100 * steps / census().steps, 1)
    assert int(m.group(3)) == runs
    assert int(m.group(4)) == census().trajectories


def test_field_census_matches_the_corpus() -> None:
    # Naming a field as 0%-populated is a licence to never build a feature on it, so the
    # list must be measured, not remembered.
    names = [f.name for f in dataclasses.fields(schema.StepView)]
    populated = dict.fromkeys(names, 0)
    total = 0
    for traj in _trajectories():
        for step in traj.steps:
            total += 1
            for name in names:
                if getattr(step, name) is not None:
                    populated[name] += 1
    empty = {n for n in names if populated[n] == 0}
    m = re.search(r"are \*\*0% populated\*\*.*?steps:\n?(.+?)\.\s*Three more", _doc_text(), re.S)
    assert m, "the 0%-populated field list was not found"
    stated = set(re.findall(r"`(\w+)`", m.group(1)))
    assert stated == empty, f"card says {sorted(stated)}, corpus says {sorted(empty)}"
    assert total == census().steps
    # The three constants named beside them.
    assert {s.is_revert for t in _trajectories() for s in t.steps} == {False}
    assert {s.retry_count for t in _trajectories() for s in t.steps} == {0}
    assert {s.loop_signal for t in _trajectories() for s in t.steps} == {False}


def test_corpus_digest_matches_the_committed_metrics() -> None:
    digest = _run_block()["corpus_digest"]
    assert isinstance(digest, str)
    assert f"`{digest}`" in _doc_text(), "the data card quotes a stale corpus digest"


def test_card_cites_its_sources() -> None:
    # A number without a producer cannot be re-derived, which is what lets one go stale
    # unnoticed. These are the entrypoints the card's numbers actually come from.
    text = _doc_text()
    for cited in (
        "benchmark.escalation.corpus:census",
        "benchmark/runner/offline_replay.py",
        "benchmark/runner/swebench_grading.py",
        "benchmark/runner/state_capture_audit.py",
        "benchmark/escalation/authenticity.py",
    ):
        assert cited in text, f"the data card must cite {cited}"


def test_every_committed_trajectory_is_a_single_session() -> None:
    # The card's structural claim: no trajectory spans sessions, which is why a
    # session-cadence replay is impossible on this corpus rather than merely unrun.
    assert "No multi-session trajectories" in _doc_text()
    assert census().trajectories == len(list(LIVE_DIR.glob("*.jsonl")))
