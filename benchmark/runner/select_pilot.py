#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

from benchmark import config
from shunt.models import TIER_ORDER


def _models_by_tier(tier: str) -> set[str]:
    pricing = config.load_pricing()
    enabled = set(config.enabled_models())
    return {
        name
        for name, info in pricing.items()
        if isinstance(info, dict) and info.get("tier") == tier and name in enabled
    }


def _escalation_models() -> set[str]:
    """Enabled models in any tier above ``mid`` — the escalation band (high, frontier, …).

    Derived from ``TIER_ORDER``; the ``frontier-only`` pattern means "a top-tier model
    passed", not literally the ``frontier`` label.
    """
    above = TIER_ORDER[TIER_ORDER.index("mid") + 1 :]
    return set().union(*(_models_by_tier(t) for t in above)) if above else set()


def classify_pattern(task_id: str, results: dict) -> str:
    cheap = _models_by_tier("cheap")
    mid = _models_by_tier("mid")
    escalation = _escalation_models()
    cheap_pass = all(results.get(m, {}).get("pass", False) for m in cheap if m in results)
    mid_pass = all(results.get(m, {}).get("pass", False) for m in mid if m in results)
    escalation_pass = any(results.get(m, {}).get("pass", False) for m in escalation if m in results)

    if cheap_pass and mid_pass:
        return "all-pass"
    if not cheap_pass and mid_pass:
        return "cheap-fail-mid-pass"
    if not cheap_pass and not mid_pass and escalation_pass:
        return "frontier-only"
    return "other"


def load_matrix(path: Path) -> dict:
    return config.load_matrix(path)


def main(config_path: str = "benchmark/config.yaml") -> None:
    config.load(config_path)
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        path = Path(sys.argv[1])
    else:
        path = config.challenges_path()
    matrix = load_matrix(path)
    tasks = matrix.get("tasks", {})
    results = matrix.get("results", {})

    target_languages = {"python", "typescript", "go", "rust"}
    target_count = 10

    by_pattern: dict[str, list[tuple[str, dict, str, list[str]]]] = {
        "cheap-fail-mid-pass": [],
        "frontier-only": [],
        "all-pass": [],
        "other": [],
    }

    for tid in sorted(results):
        task_meta = tasks.get(tid, {})
        lang = task_meta.get("language", "")
        if lang not in target_languages:
            continue
        pattern = classify_pattern(tid, results[tid])
        tags = task_meta.get("tags", [])
        by_pattern[pattern].append((tid, task_meta, lang, tags))

    selected: list[tuple[str, dict, str, str, list[str]]] = []
    used_lang_pattern: set[tuple[str, str]] = set()
    used_langs: set[str] = set()
    used_patterns: set[str] = set()

    def pick_one(pattern: str, lang: str | None = None) -> bool:
        candidates = [c for c in by_pattern[pattern] if c[0] not in {s[0] for s in selected}]
        if lang:
            candidates = [c for c in candidates if c[2] == lang]
        if not candidates:
            return False
        candidates.sort(key=lambda x: x[1].get("tags", [""]))
        tid, meta, clang, tags = candidates[0]
        selected.append((tid, meta, clang, pattern, tags))
        used_lang_pattern.add((clang, pattern))
        used_langs.add(clang)
        used_patterns.add(pattern)
        return True

    # Phase 1: one cheap-fail-mid-pass per language
    for lang in sorted(target_languages):
        if len(selected) >= target_count:
            break
        pick_one("cheap-fail-mid-pass", lang)

    # Phase 2: one frontier-only per language
    for lang in sorted(target_languages):
        if len(selected) >= target_count:
            break
        if (lang, "frontier-only") not in used_lang_pattern:
            pick_one("frontier-only", lang)

    # Phase 3: one all-pass per language (avoid-over-escalation test)
    for lang in sorted(target_languages):
        if len(selected) >= target_count:
            break
        if (lang, "all-pass") not in used_lang_pattern:
            pick_one("all-pass", lang)

    # Phase 4: fill remaining with discriminating tasks
    need = target_count - len(selected)
    if need > 0:
        for lang in sorted(target_languages):
            for pat in ["cheap-fail-mid-pass", "frontier-only"]:
                if need <= 0:
                    break
                candidates = [
                    c
                    for c in by_pattern[pat]
                    if c[0] not in {s[0] for s in selected} and c[2] == lang
                ]
                for tid, meta, clang, tags in candidates:
                    if need <= 0:
                        break
                    selected.append((tid, meta, clang, pat, tags))
                    need -= 1

    cheap = _models_by_tier("cheap")
    frontier = _escalation_models()

    print("=" * 72)
    print("Pilot Task Selection — 10 Discriminating Tasks")
    print("=" * 72)

    for i, (tid, meta, lang, pattern, tags) in enumerate(selected, 1):
        desc = meta.get("description", "")
        tag_str = ", ".join(tags)
        rationale_map = {
            "cheap-fail-mid-pass": "Cheap fail; mid/top pass — tests escape-to-escalation.",
            "frontier-only": "Only a top-tier model passes — tests max-cost escalation.",
            "all-pass": "All models pass — tests avoid-over-escalation.",
            "other": "Mixed pattern.",
        }
        rationale = rationale_map.get(pattern, "")
        cost_map = results.get(tid, {})
        cheap_p = any(cost_map.get(m, {}).get("pass") for m in cheap if m in cost_map)
        top_tier_p = any(cost_map.get(m, {}).get("pass") for m in frontier if m in cost_map)

        print(f"\n  {i}. {tid} ({lang} — {pattern})")
        print(f"     Task: {desc}")
        print(f"     Tags: {tag_str}")
        print(f"     Rationale: {rationale}")
        print(f"     Cheap-pass={cheap_p}, Top-tier-pass={top_tier_p}")

    print(f"\n  Language coverage: {sorted(set(s[2] for s in selected))}")
    print(f"  Pattern coverage: {sorted(set(s[3] for s in selected))}")

    ids = [s[0] for s in selected]
    print(f"\n  Selected IDs: {' '.join(ids)}")


if __name__ == "__main__":
    main()
