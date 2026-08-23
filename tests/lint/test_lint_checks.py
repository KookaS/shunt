"""Contract tests for the custom SH0xx lint checks (subprocess = real CLI)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Final

_ROOT = Path(__file__).resolve().parents[2]
_LINT_DIR = _ROOT / "tools" / "lint"


def _run(script: str, *args: str) -> int:
    return subprocess.run(
        [sys.executable, str(_LINT_DIR / script), *args],
        capture_output=True,
        text=True,
    ).returncode


def _output(script: str, *args: str) -> str:
    """A check's stdout — for the findings whose VALUE is the message, not the exit code."""
    return subprocess.run(
        [sys.executable, str(_LINT_DIR / script), *args],
        capture_output=True,
        text=True,
    ).stdout


def test_sh001_catches_uppercase_mutable_container(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    f.write_text("CACHE = {}\n")
    assert _run("check_mutable_globals.py", str(f)) == 1


def test_sh001_ignores_immutable_uppercase_constants(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    f.write_text('MAX = 5\nNAMES = ("a",)\nPATH = "x"\nFROZEN = frozenset({1})\n')
    assert _run("check_mutable_globals.py", str(f)) == 0


def test_sh001_ignores_final_annotated_container(tmp_path: Path) -> None:
    f = tmp_path / "mod.py"
    f.write_text("from typing import Final\n\nTABLE: Final = {}\n")
    assert _run("check_mutable_globals.py", str(f)) == 0


def test_sh004_catches_planted_story_ref(tmp_path: Path) -> None:
    f = tmp_path / "leak.py"
    f.write_text("# tracked in STORY-9.9\nx = 1\n")  # noqa: SHUNT-ISO
    assert _run("check_internal_refs.py", str(f)) == 1


def test_sh004_passes_clean_public_vocab(tmp_path: Path) -> None:
    f = tmp_path / "clean.md"
    f.write_text("# Roadmap\n\nUses kill_gate and dogfood on claude-opus-4-6.\n")
    assert _run("check_internal_refs.py", str(f)) == 0


def test_sh004_default_tree_is_clean() -> None:
    assert _run("check_internal_refs.py") == 0


def test_sh004_scans_yaml_when_walking_a_directory(tmp_path: Path) -> None:
    f = tmp_path / "provider.yaml"
    f.write_text("# see backlog for the rollout\nbase_url: https://example.com/v1\n")  # noqa: SHUNT-ISO
    assert _run("check_internal_refs.py", str(tmp_path)) == 1


def test_sh004_default_scan_covers_examples() -> None:
    # examples/ ships to users but was scanned by NOTHING: this checker skipped
    # both the tree and the .yaml suffix, and check-docs-integrity.sh only walks
    # docs/*.md. Planting a real leak is the only way to prove the tree is wired
    # into the DEFAULT target list, which is the thing that silently regresses.
    examples = _ROOT / "examples"
    created = not examples.exists()
    examples.mkdir(parents=True, exist_ok=True)
    planted = examples / "_sh004_contract_probe.yaml"
    planted.write_text("# see backlog\n")  # noqa: SHUNT-ISO
    try:
        assert _run("check_internal_refs.py") == 1
    finally:
        planted.unlink()
        if created:
            examples.rmdir()


def _src_shunt_file(tmp_path: Path, rel: str, body: str) -> Path:
    """Write a file at a real-looking src/shunt/<rel> path (SH005 keys on the path)."""
    f = tmp_path / "src" / "shunt" / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(body)
    return f


def test_sh005_catches_pricing_attribute_in_router(tmp_path: Path) -> None:
    f = _src_shunt_file(tmp_path, "router/pick.py", "def f(cfg):\n    return cfg.pricing\n")
    assert _run("check_pricing_isolation.py", str(f)) == 1


def test_sh005_catches_cost_field_string_key(tmp_path: Path) -> None:
    # The string-subscript back door must not bypass the gate.
    f = _src_shunt_file(
        tmp_path, "router/cost.py", 'def f(d):\n    return d["input_cost_per_1m"]\n'
    )
    assert _run("check_pricing_isolation.py", str(f)) == 1


def test_sh005_exempts_the_registry_loader(tmp_path: Path) -> None:
    f = _src_shunt_file(tmp_path, "models/config.py", "def f(cfg):\n    return cfg.pricing\n")
    assert _run("check_pricing_isolation.py", str(f)) == 0


def test_sh005_ignores_modules_outside_src_shunt(tmp_path: Path) -> None:
    # The benchmark is the legitimate pricing consumer.
    f = tmp_path / "benchmark" / "config.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("def f(cfg):\n    return cfg.pricing\n")
    assert _run("check_pricing_isolation.py", str(f)) == 0


def test_sh005_passes_a_clean_router_module(tmp_path: Path) -> None:
    f = _src_shunt_file(tmp_path, "router/clean.py", "def f(cfg):\n    return cfg.provider\n")
    assert _run("check_pricing_isolation.py", str(f)) == 0


def test_sh004_default_scan_covers_root_level_files() -> None:
    # The default target list used to name five subtrees, so every root-level
    # file except README.md shipped unscanned — a planted leak in CONTRIBUTING.md
    # exited 0. The hook passes no filenames, so the default IS the coverage.
    planted = _ROOT / "_sh004_root_probe.md"
    planted.write_text("# tracked in STORY-9.9\n")  # noqa: SHUNT-ISO
    try:
        assert _run("check_internal_refs.py") == 1
    finally:
        planted.unlink()
    assert _run("check_internal_refs.py") == 0


def test_sh004_honours_the_wrapper_isolation_noqa_token() -> None:
    # One suppression token serves both gates; two would drift apart.
    planted = _ROOT / "_sh004_noqa_probe.py"
    body = "x = 1  # tracked in STORY" + "-9.9  noqa: SHUNT-ISO\n"  # split: fixture, not a ref
    planted.write_text(body)
    try:
        assert _run("check_internal_refs.py") == 0
    finally:
        planted.unlink()


def test_sh004_catches_the_missing_vocab_families(tmp_path: Path) -> None:
    """The audit: SH004 saw 1 of 6 internal-vocab families. Plant each family the
    OLD gate missed and require a hit (noqa kept OFF the planted file so the hit is
    real; the literal here is suppressed for the whole-repo scan)."""
    leaks = [
        "# tracked in SG-1",  # noqa: SHUNT-ISO
        "# tracked in PIV-3",  # noqa: SHUNT-ISO
        "# tracked in AC-3",  # noqa: SHUNT-ISO
        "# tracked in AC12",  # noqa: SHUNT-ISO
        "# tracked in ADR-0003",  # noqa: SHUNT-ISO
        "# tracked in (D3)",  # noqa: SHUNT-ISO
        "# tracked in decision 0002",  # noqa: SHUNT-ISO
        "# tracked in decisions/0002",  # noqa: SHUNT-ISO
        "# tracked in project/shunt/backlog",  # noqa: SHUNT-ISO
    ]
    for i, line in enumerate(leaks):
        f = tmp_path / f"leak_{i}.py"
        f.write_text(line + "\nx = 1\n")
        assert _run("check_internal_refs.py", str(f)) == 1, line


def test_sh004_does_not_flag_public_english_uses(tmp_path: Path) -> None:
    # Tight \b-anchored patterns must not flag ordinary English: a lowercase "story",
    # "AC-" buried in "MAC-10" (no word boundary before it), "decision process"
    # (no year digits), and "PACK-1" (no "AC-" at all).
    f = tmp_path / "public.md"
    f.write_text(
        "# Roadmap\n\n"
        "A short story about the shipping plan.\n"
        "PACK-1 and MAC-10 cover the west.\n"
        "Our decision process is documented in the README.\n"
    )
    assert _run("check_internal_refs.py", str(f)) == 0


def test_sh006_catches_src_importing_benchmark(tmp_path: Path) -> None:
    f = _src_shunt_file(tmp_path, "router/leak.py", "import benchmark.config\n")
    assert _run("check_import_direction.py", str(f)) == 1


def test_sh006_allows_benchmark_importing_src(tmp_path: Path) -> None:
    f = tmp_path / "benchmark" / "uses_shunt.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("import shunt.router\n")
    assert _run("check_import_direction.py", str(f)) == 0


def test_sh006_passes_a_clean_src_module(tmp_path: Path) -> None:
    f = _src_shunt_file(tmp_path, "router/clean2.py", "import logging\n")
    assert _run("check_import_direction.py", str(f)) == 0


def _bench_file(tmp_path: Path, rel: str, body: str) -> Path:
    f = tmp_path / "benchmark" / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(body)
    return f


def test_sh006_catches_routing_importing_escalation(tmp_path: Path) -> None:
    f = _bench_file(tmp_path, "routing/leak.py", "from benchmark.escalation import schema\n")
    assert _run("check_import_direction.py", str(f)) == 1


def test_sh006_catches_calibration_importing_escalation(tmp_path: Path) -> None:
    f = _bench_file(tmp_path, "calibration/leak.py", "import benchmark.escalation.metrics\n")
    assert _run("check_import_direction.py", str(f)) == 1


def test_sh006_allows_escalation_importing_routing(tmp_path: Path) -> None:
    f = _bench_file(tmp_path, "escalation/uses_spine.py", "from benchmark.routing import metrics\n")
    assert _run("check_import_direction.py", str(f)) == 0


# --- SH008: real embeddings only -------------------------------------------------
# The real-only embedding rule was 100% prose-enforced until this gate existed, and a
# raw `TextEmbedding(model_name=...)` in benchmark/routing/strategies/knn_cascade.py
# lived on the kill-gate path for exactly that reason. These tests are the wall's wall:
# delete the gate and they go red.


def test_sh008_catches_raw_fastembed_import_in_benchmark(tmp_path: Path) -> None:
    f = _bench_file(tmp_path, "routing/strategies/raw.py", "from fastembed import TextEmbedding\n")
    assert _run("check_embedder_isolation.py", str(f)) == 1


def test_sh008_catches_plain_fastembed_module_import(tmp_path: Path) -> None:
    f = _bench_file(tmp_path, "routing/raw2.py", "import fastembed\n")
    assert _run("check_embedder_isolation.py", str(f)) == 1


def test_sh008_exempts_the_shipped_embedder_module(tmp_path: Path) -> None:
    f = _src_shunt_file(tmp_path, "router/embedder.py", "from fastembed import TextEmbedding\n")
    assert _run("check_embedder_isolation.py", str(f)) == 0


def test_sh008_catches_tfidf_vectorizer(tmp_path: Path) -> None:
    f = _bench_file(
        tmp_path,
        "routing/proxy.py",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    )
    assert _run("check_embedder_isolation.py", str(f)) == 1


def test_sh008_catches_a_vectorizer_reexported_from_elsewhere(tmp_path: Path) -> None:
    # The name is banned, not just the sklearn path — a re-export must not be a back door.
    f = _bench_file(tmp_path, "routing/proxy2.py", "from myshim import HashingVectorizer\n")
    assert _run("check_embedder_isolation.py", str(f)) == 1


def test_sh008_ignores_tests(tmp_path: Path) -> None:
    f = tmp_path / "tests" / "test_fake.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("from fastembed import TextEmbedding\nfrom x import CountVectorizer\n")
    assert _run("check_embedder_isolation.py", str(f)) == 0


def test_sh008_passes_the_shipped_embedder_indirection(tmp_path: Path) -> None:
    f = _bench_file(tmp_path, "routing/clean.py", "from shunt.router.embedder import Embedder\n")
    assert _run("check_embedder_isolation.py", str(f)) == 0


def test_sh008_default_tree_is_clean() -> None:
    # The default scan IS the coverage (the hook passes no filenames), and this is the
    # assertion that knn_cascade.py's raw TextEmbedding stays gone.
    assert _run("check_embedder_isolation.py") == 0


def test_sh008_catches_hashlib_digest_converted_to_vector(tmp_path: Path) -> None:
    # A sha256 digest fed to np.frombuffer IS an embedding build, not integrity hashing.
    f = _bench_file(
        tmp_path,
        "routing/hash_emb.py",
        "import hashlib\n"
        "import numpy as np\n"
        "def emb(t):\n"
        "    return np.frombuffer(hashlib.sha256(t.encode()).digest(), dtype=np.float32)\n",
    )
    assert _run("check_embedder_isolation.py", str(f)) == 1


def test_sh008_catches_hashlib_digest_sliced_to_vector(tmp_path: Path) -> None:
    f = _bench_file(
        tmp_path,
        "routing/hash_slice.py",
        "import hashlib\ndef emb(t):\n    return hashlib.sha256(t.encode()).digest()[:384]\n",
    )
    assert _run("check_embedder_isolation.py", str(f)) == 1


def test_sh008_catches_hashlib_digest_listed(tmp_path: Path) -> None:
    f = _bench_file(
        tmp_path,
        "routing/hash_list.py",
        "import hashlib\ndef emb(t):\n    return list(hashlib.sha256(t.encode()).digest())\n",
    )
    assert _run("check_embedder_isolation.py", str(f)) == 1


def test_sh008_passes_legitimate_integrity_hashing(tmp_path: Path) -> None:
    # A bare hexdigest/state-key slice (as src/shunt/router/engine.py does) is integrity
    # hashing, not an embedding — must NOT trip the proxy-vector detector.
    f = _src_shunt_file(
        tmp_path,
        "router/digest.py",
        "import hashlib\n"
        "def state_key(k):\n"
        "    return hashlib.sha256(k.encode('utf-8')).hexdigest()[:16]\n",
    )
    assert _run("check_embedder_isolation.py", str(f)) == 0


def test_sh008_catches_np_random_draw_as_embedding(tmp_path: Path) -> None:
    f = _bench_file(
        tmp_path,
        "routing/rand_emb.py",
        "import numpy as np\ndef emb():\n    return np.random.rand(384)\n",
    )
    assert _run("check_embedder_isolation.py", str(f)) == 1


def test_sh008_catches_default_rng_inline_draw(tmp_path: Path) -> None:
    f = _bench_file(
        tmp_path,
        "routing/rng_emb.py",
        "import numpy as np\ndef emb():\n    return np.random.default_rng(0).random(384)\n",
    )
    assert _run("check_embedder_isolation.py", str(f)) == 1


def test_sh008_catches_default_rng_draw_on_bound_name(tmp_path: Path) -> None:
    f = _bench_file(
        tmp_path,
        "routing/rng_bound.py",
        "import numpy as np\n"
        "def emb():\n"
        "    rng = np.random.default_rng(0)\n"
        "    return rng.normal(size=384)\n",
    )
    assert _run("check_embedder_isolation.py", str(f)) == 1


def test_sh008_catches_random_rand_constructed_in_vector_context(tmp_path: Path) -> None:
    f = _bench_file(
        tmp_path,
        "routing/randobj_emb.py",
        "import random\ndef emb():\n    return [random.Random(0).random() for _ in range(384)]\n",
    )
    assert _run("check_embedder_isolation.py", str(f)) == 1


def test_sh008_passes_legitimate_seeded_rng_sampler(tmp_path: Path) -> None:
    # A random.Random instance seeded for EXPLORATION (as src/shunt/router/escalation.py
    # does) is not an embedding; a scalar draw on a field receiver must not trip.
    f = _src_shunt_file(
        tmp_path,
        "router/sampler.py",
        "import random\n"
        "class S:\n"
        "    rng: random.Random\n"
        "    def __init__(self, seed):\n"
        "        self.rng = random.Random(seed)\n"
        "    def explore(self, eps):\n"
        "        return self.rng.random() < eps\n",
    )
    assert _run("check_embedder_isolation.py", str(f)) == 0


def test_sh008_catches_importlib_import_module_banned_module(tmp_path: Path) -> None:
    f = _bench_file(
        tmp_path,
        "routing/dyn_sklearn.py",
        "import importlib\n"
        "def f():\n"
        "    return importlib.import_module('sklearn.feature_extraction.text')\n",
    )
    assert _run("check_embedder_isolation.py", str(f)) == 1


def test_sh008_catches_importlib_import_module_fastembed(tmp_path: Path) -> None:
    f = _bench_file(
        tmp_path,
        "routing/dyn_fastembed.py",
        "import importlib\ndef f():\n    return importlib.import_module('fastembed')\n",
    )
    assert _run("check_embedder_isolation.py", str(f)) == 1


def test_sh008_catches_importlib_import_module_via_alias(tmp_path: Path) -> None:
    f = _bench_file(
        tmp_path,
        "routing/dyn_alias.py",
        "from importlib import import_module as load\n"
        "def f():\n"
        "    return load('sklearn.feature_extraction')\n",
    )
    assert _run("check_embedder_isolation.py", str(f)) == 1


def test_sh008_passes_importlib_import_module_of_benign_module(tmp_path: Path) -> None:
    f = _bench_file(
        tmp_path,
        "routing/dyn_ok.py",
        "import importlib\ndef f():\n    return importlib.import_module('shunt.router.embedder')\n",
    )
    assert _run("check_embedder_isolation.py", str(f)) == 0


def test_sh008_exemption_is_anchored_to_a_tests_segment(tmp_path: Path) -> None:
    # A file under benchmark/latests/ is production code: the old `tests/` substring
    # exemption let it ride. Segment anchoring keeps it scanned.
    f = _bench_file(tmp_path, "latests/leak.py", "from fastembed import TextEmbedding\n")
    assert _run("check_embedder_isolation.py", str(f)) == 1
    g = _bench_file(tmp_path, "mytests/leak.py", "from fastembed import TextEmbedding\n")
    assert _run("check_embedder_isolation.py", str(g)) == 1


def test_sh008_still_exempts_a_tests_segment(tmp_path: Path) -> None:
    f = tmp_path / "benchmark" / "tests" / "test_fake.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("from fastembed import TextEmbedding\n")
    assert _run("check_embedder_isolation.py", str(f)) == 0


# --- SH009: the figure <-> manifest <-> docs bijection ----------------------------------


# The halves the gate knows, and where each keeps its manifest. `inference` is the proof that
# a half's package need not sit under benchmark/.
_HALF_PKGS: Final[dict[str, str]] = {
    "routing": "benchmark/routing",
    "escalation": "benchmark/escalation",
    "inference": "src/shunt/inspect/inference",
}


def _figure_tree(
    tmp_path: Path,
    *,
    pngs: tuple[str, ...] = ("kill_gate.png",),
    manifest: tuple[str, ...] | None = None,
    section_slugs: tuple[str, ...] | None = None,
    notes: tuple[str, ...] = ("x is cost.", "175 tasks"),
    half: str = "routing",
) -> Path:
    """A minimal repo shaped like shunt: one half's figure, its row, and its section."""
    title = "The gate is untested"
    # One PNG directory per half inside the docs tree, so a figure's home and its
    # figures.json row both have to name the same half.
    figures = tmp_path / "docs" / "assets" / "figures" / half
    figures.mkdir(parents=True)
    (tmp_path / _HALF_PKGS[half]).mkdir(parents=True)
    for png in pngs:
        (figures / png).write_bytes(b"\x89PNG")
    rows = {
        name: {
            "title": title,
            "subtitle": "175 tasks",
            "caveat": None,
            "reading": "x is cost.",
            "goal": "Aim top-left.",
            "limitations": [],
            "notes": list(notes),
        }
        for name in (manifest if manifest is not None else pngs)
    }
    (tmp_path / _HALF_PKGS[half] / "figures.json").write_text(
        json.dumps({"schema": 1, "half": half, "figures": rows})
    )
    docs = tmp_path / "docs"
    slugs = (
        section_slugs
        if section_slugs is not None
        else tuple(s[:-4].replace("_", "-") for s in pngs)
    )
    body = "\n".join(
        f"""
### {title} {{#fig-{slug}}}

![Cost versus quality](assets/figures/{half}/{slug.replace("-", "_")}.png)

*175 tasks*

**Reading.** x is cost.

**What to look for.** Aim top-left.

**Notes.** {"\n".join(notes)}
"""
        for slug in slugs
    )
    (docs / f"{half}.md").write_text(f"# {half.title()}\n\n## Figures\n{body}\n")
    # Every other half stays entirely absent — no directory, no manifest, no doc — which is
    # what a half looks like before its producer lands.
    return tmp_path


def test_sh009_accepts_a_complete_bijection(tmp_path: Path) -> None:
    assert _run("check_figure_docs.py", "--root", str(_figure_tree(tmp_path))) == 0


def test_sh009_catches_a_png_with_no_manifest_row(tmp_path: Path) -> None:
    root = _figure_tree(tmp_path, pngs=("kill_gate.png",), manifest=())
    assert _run("check_figure_docs.py", "--root", str(root)) == 1


def test_sh009_catches_a_retired_figure_left_in_the_manifest(tmp_path: Path) -> None:
    """The one-way check misses this, and it is the failure that actually bites."""
    root = _figure_tree(tmp_path, pngs=("kill_gate.png",), manifest=("kill_gate.png", "gone.png"))
    assert _run("check_figure_docs.py", "--root", str(root)) == 1


def test_sh009_catches_a_retired_figure_left_in_the_docs(tmp_path: Path) -> None:
    root = _figure_tree(tmp_path, section_slugs=("kill-gate", "cumulative-regret"))
    assert _run("check_figure_docs.py", "--root", str(root)) == 1


def test_sh009_catches_a_figure_with_no_docs_section(tmp_path: Path) -> None:
    root = _figure_tree(tmp_path, section_slugs=())
    assert _run("check_figure_docs.py", "--root", str(root)) == 1


def test_sh009_catches_a_heading_that_does_not_quote_the_canvas_title(tmp_path: Path) -> None:
    root = _figure_tree(tmp_path)
    doc = root / "docs" / "routing.md"
    doc.write_text(doc.read_text().replace("### The gate is untested", "### Cost vs quality"))
    assert _run("check_figure_docs.py", "--root", str(root)) == 1


def test_sh009_catches_a_missing_required_block(tmp_path: Path) -> None:
    root = _figure_tree(tmp_path)
    doc = root / "docs" / "routing.md"
    doc.write_text(doc.read_text().replace("**What to look for.** Aim top-left.\n", ""))
    assert _run("check_figure_docs.py", "--root", str(root)) == 1


def test_sh009_catches_a_png_left_outside_every_halfs_directory(tmp_path: Path) -> None:
    """The regression the per-half split can cause: a producer still writing to the flat root."""
    root = _figure_tree(tmp_path)
    (root / "docs" / "assets" / "figures" / "kill_gate.png").write_bytes(b"\x89PNG")
    assert _run("check_figure_docs.py", "--root", str(root)) == 1


def test_sh009_catches_a_png_in_the_wrong_halfs_directory(tmp_path: Path) -> None:
    root = _figure_tree(tmp_path)
    wrong = root / "docs" / "assets" / "figures" / "escalation"
    wrong.mkdir(parents=True)
    (wrong / "kill_gate.png").write_bytes(b"\x89PNG")
    assert _run("check_figure_docs.py", "--root", str(root)) == 1


def test_sh009_catches_a_docs_link_that_skips_the_halfs_directory(tmp_path: Path) -> None:
    root = _figure_tree(tmp_path)
    doc = root / "docs" / "routing.md"
    doc.write_text(doc.read_text().replace("assets/figures/routing/", "assets/figures/"))
    assert _run("check_figure_docs.py", "--root", str(root)) == 1


def test_sh009_catches_a_mutated_notes_string(tmp_path: Path) -> None:
    """The notes byte-lock's positive control: a per-model note row rewritten in the docs
    must redden the gate, because that is precisely the silent drift it exists to stop."""
    root = _figure_tree(tmp_path, notes=("x is cost.", "175 tasks"))
    doc = root / "docs" / "routing.md"
    doc.write_text(
        doc.read_text().replace("**Notes.** x is cost.\n175 tasks", "**Notes.** y is free.")
    )
    assert _run("check_figure_docs.py", "--root", str(root)) == 1


def test_sh009_catches_a_notes_block_for_notes_the_figure_does_not_render(tmp_path: Path) -> None:
    """A Notes block describing claims the manifest no longer carries is a hand-written
    note row with nothing derived behind it — the gate must refuse to bless it."""
    root = _figure_tree(tmp_path, notes=())
    assert _run("check_figure_docs.py", "--root", str(root)) == 1


def test_sh009_catches_a_figure_whose_notes_block_is_missing(tmp_path: Path) -> None:
    root = _figure_tree(tmp_path, notes=("x is cost.", "175 tasks"))
    doc = root / "docs" / "routing.md"
    doc.write_text(doc.read_text().replace("**Notes.** x is cost.\n175 tasks", ""))
    assert _run("check_figure_docs.py", "--root", str(root)) == 1


def test_sh009_accepts_a_complete_bijection_for_the_inference_half(tmp_path: Path) -> None:
    """The inference half keeps its manifest outside benchmark/, so this is the proof that
    nothing downstream rebuilds `benchmark/<half>` instead of reading the half's package."""
    root = _figure_tree(tmp_path, half="inference")
    assert (root / "src" / "shunt" / "inspect" / "inference" / "figures.json").exists()
    assert _run("check_figure_docs.py", "--root", str(root)) == 0


def test_sh009_orphan_message_names_the_halfs_real_manifest(tmp_path: Path) -> None:
    """The message must point at a path that exists; `benchmark/inference/figures.json` is a
    debugging trap, because nothing will ever be there."""
    root = _figure_tree(tmp_path, half="inference", manifest=())
    out = _output("check_figure_docs.py", "--root", str(root))
    assert "src/shunt/inspect/inference/figures.json" in out
    assert "benchmark/inference" not in out


def test_sh009_an_entirely_absent_half_is_consistent(tmp_path: Path) -> None:
    """Before a half's producer lands it has no PNG directory, no manifest and no doc page.
    That is not a finding — a half only reddens once something describes a figure."""
    root = _figure_tree(tmp_path)
    assert not (root / "docs" / "assets" / "figures" / "inference").exists()
    assert not (root / "docs" / "inference.md").exists()
    assert _run("check_figure_docs.py", "--root", str(root)) == 0


# --- SH007: the frame owns figure creation, titling and saving ---------------------------


def _py(tmp_path: Path, body: str, name: str = "mod.py") -> Path:
    f = tmp_path / name
    f.write_text(body)
    return f


def test_sh007_catches_a_bare_savefig(tmp_path: Path) -> None:
    f = _py(tmp_path, "import matplotlib.pyplot as plt\nfig = object()\nfig.savefig('x.png')\n")
    assert _run("check_plot_frame.py", str(f)) == 1


def test_sh007_catches_the_bare_name_import_bypass(tmp_path: Path) -> None:
    f = _py(tmp_path, "from matplotlib.pyplot import savefig\nsavefig('x.png')\n")
    assert _run("check_plot_frame.py", str(f)) == 1


def test_sh007_catches_a_raw_pyplot_figure(tmp_path: Path) -> None:
    """An ad-hoc figsize with no layout engine is how the set got 15 canvas sizes."""
    f = _py(tmp_path, "import matplotlib.pyplot as plt\nfig, ax = plt.subplots(figsize=(9, 5))\n")
    assert _run("check_plot_frame.py", str(f)) == 1


def test_sh007_catches_a_caller_owned_title(tmp_path: Path) -> None:
    """A title the frame cannot measure is a title drawn over the content."""
    f = _py(tmp_path, "def draw(ax):\n    ax.set_title('mine')\n")
    assert _run("check_plot_frame.py", str(f)) == 1
    g = _py(tmp_path, "def draw(fig):\n    fig.suptitle('mine')\n", name="mod2.py")
    assert _run("check_plot_frame.py", str(g)) == 1


def test_sh007_allows_subplots_on_a_frame_made_figure(tmp_path: Path) -> None:
    """`fig.subplots(...)` is the supported way to lay panels out inside the frame."""
    f = _py(tmp_path, "def draw(fig):\n    axes = fig.subplots(2, 2)\n    return axes\n")
    assert _run("check_plot_frame.py", str(f)) == 0


def test_sh007_lets_tests_build_a_deliberately_broken_figure(tmp_path: Path) -> None:
    """The layout-audit tests must be able to construct the defect they assert against."""
    f = _py(tmp_path, "import matplotlib.pyplot as plt\nfig = plt.figure()\n", name="test_x.py")
    assert _run("check_plot_frame.py", str(f)) == 0


def test_sh007_still_blocks_a_test_that_writes_a_png(tmp_path: Path) -> None:
    """The creation exemption is not a savefig exemption."""
    f = _py(tmp_path, "def test_x(fig):\n    fig.savefig('x.png')\n", name="test_y.py")
    assert _run("check_plot_frame.py", str(f)) == 1


def test_sh007_allows_the_inspect_frame_module(tmp_path: Path) -> None:
    """src/shunt/inspect/plot_frame.py is a legal frame — raw savefig inside it is fine."""
    frame = tmp_path / "src" / "shunt" / "inspect" / "plot_frame.py"
    frame.parent.mkdir(parents=True)
    frame.write_text("import matplotlib.pyplot as plt\nfig = object()\nfig.savefig('x.png')\n")
    assert _run("check_plot_frame.py", str(frame)) == 0


def test_sh007_allows_calling_the_inspect_frame(tmp_path: Path) -> None:
    """A draw module may create/title/save through the inspect frame's public API."""
    f = _py(
        tmp_path,
        "from shunt.inspect import plot_frame\n"
        "fig = plot_frame.new_figure(plot_frame.WIDE)\n"
        "axes = fig.subplots(1, 2)\n"
        "plot_frame.save(fig, '/tmp/x.png', plot_frame.FigureSpec())\n",
        name="figures_x.py",
    )
    assert _run("check_plot_frame.py", str(f)) == 0


def test_sh007_default_tree_is_clean() -> None:
    import subprocess as _sp

    files = _sp.run(
        ["git", "ls-files", "*.py"], cwd=_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert _run("check_plot_frame.py", *files) == 0


def test_sh009_an_empty_half_does_not_silently_pass(tmp_path: Path) -> None:
    """Deleting every figure must not retire the docs requirement along with them."""
    root = _figure_tree(tmp_path)
    for png in (root / "docs" / "assets" / "figures").rglob("*.png"):
        png.unlink()
    assert _run("check_figure_docs.py", "--root", str(root)) == 1


def test_sh009_a_genuinely_empty_half_is_fine(tmp_path: Path) -> None:
    """A half with no figures, no manifest rows and no sections is consistent."""
    root = _figure_tree(tmp_path, pngs=(), manifest=(), section_slugs=())
    assert _run("check_figure_docs.py", "--root", str(root)) == 0


# --- SH012: number-bearing sites carry a provenance marker ---------------------


def _number_tree(
    tmp_path: Path,
    *,
    results: str | None = None,
    benchmark: str | None = None,
    benchmark_data: str | None = None,
) -> Path:
    """A minimal repo shaped like shunt: the three main result docs."""
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    if results is not None:
        (docs / "results.md").write_text(results)
    if benchmark is not None:
        (docs / "benchmark.md").write_text(benchmark)
    if benchmark_data is not None:
        (docs / "benchmark-data.md").write_text(benchmark_data)
    return tmp_path


_MARKED_RANK0 = (
    "| Session-Cascade, `rank_shortlist=0` (pre-shortlist) | 96.57% | "
    "93.71–98.86 | $48.19 | [29.32, 69.57] | $35.79 | "
    "<!-- frozen-value: n=180, date=2026-08-10, run=49b8362 -->"
)
_MARKED_PAIRED = (
    "| `sl=2` vs `sl=3` | +$0.61 | [−0.42, +1.47] | not distinguishable | "
    "<!-- frozen-value: n=175, date=2026-08-10, run=49b8362 -->"
)
_MARKED_PP = (
    "matches fixed-frontier quality is `Price-Cascade` (**+1.6 pp, CI crosses "
    "zero → statistically equal**, at roughly **72% lower cost** on the shared "
    "measurable set) <!-- generated-by: benchmark.routing.report:paired_quality_contrast -->"
)
_MARKED_USABLE = (
    "at matched quality — has a tighter coverage limit than the 177 usable tasks "
    "suggest. It <!-- frozen-value: n=177, date=2026-07-28, run=cece0fd -->"
)


def test_sh012_clean_tree_passes(tmp_path: Path) -> None:
    root = _number_tree(
        tmp_path,
        results=f"# Results\n\n{_MARKED_RANK0}\n\n{_MARKED_PAIRED}\n",
        benchmark=f"# Benchmark\n\n{_MARKED_PP}\n",
        benchmark_data=f"# Data\n\n{_MARKED_USABLE}\n",
    )
    assert _run("check_number_provenance.py", "--root", str(root)) == 0


def test_sh012_negative_control_real_tree_is_clean() -> None:
    # The committed docs carry a marker on every number-bearing line — the tree the
    # hook scans must pass without a staged-file filter (pass_filenames: false).
    assert _run("check_number_provenance.py") == 0


def test_sh012_unmarked_rank_shortlist_row_fails(tmp_path: Path) -> None:
    root = _number_tree(
        tmp_path,
        results=f"# Results\n\n{_MARKED_RANK0.replace(' <!-- frozen-value', '')}\n",
    )
    assert _run("check_number_provenance.py", "--root", str(root)) == 1


def test_sh012_unmarked_paired_row_fails(tmp_path: Path) -> None:
    root = _number_tree(
        tmp_path,
        results=f"# Results\n\n{_MARKED_PAIRED.replace(' <!-- frozen-value', '')}\n",
    )
    assert _run("check_number_provenance.py", "--root", str(root)) == 1


def test_sh012_unmarked_pp_claim_fails(tmp_path: Path) -> None:
    root = _number_tree(
        tmp_path,
        benchmark=f"# Benchmark\n\n{_MARKED_PP.replace(' <!-- generated-by', '')}\n",
    )
    assert _run("check_number_provenance.py", "--root", str(root)) == 1


def test_sh012_frozen_marker_without_corpus_fails(tmp_path: Path) -> None:
    # the marker is present but n/date/run are missing — a frozen value that
    # is not allowed to be anonymous.
    stripped = _MARKED_RANK0.replace(
        "frozen-value: n=180, date=2026-08-10, run=49b8362", "frozen-value:"
    )
    root = _number_tree(tmp_path, results=f"# Results\n\n{stripped}\n")
    assert _run("check_number_provenance.py", "--root", str(root)) == 1


def test_sh012_frozen_marker_with_partial_corpus_fails(tmp_path: Path) -> None:
    root = _number_tree(
        tmp_path,
        results=f"# Results\n\n{_MARKED_RANK0.replace('date=2026-08-10, ', '')}\n",
    )
    assert _run("check_number_provenance.py", "--root", str(root)) == 1


def test_sh012_unmarked_task_count_claim_fails(tmp_path: Path) -> None:
    root = _number_tree(
        tmp_path,
        results="# Results\n\nThe router sends **167 of 175 tasks** to `deepseek-v4-flash`.\n",
    )
    assert _run("check_number_provenance.py", "--root", str(root)) == 1


def test_sh012_unmarked_scorable_claim_fails(tmp_path: Path) -> None:
    root = _number_tree(
        tmp_path,
        results=(
            "# Results\n\n**Set B — raw, un-imputed, fully-measured tasks only: 74 scorable.**\n"
        ),
    )
    assert _run("check_number_provenance.py", "--root", str(root)) == 1


def test_sh012_unmarked_site_outside_the_old_three_doc_list_fails(tmp_path: Path) -> None:
    # The scan set used to be a hardcoded (results, benchmark, benchmark-data) list, so a
    # result statistic published in CHANGELOG.md or README.md was unenforced. The walk is
    # now every *.md in the tree; this is the regression test for that widening.
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\nOn the 184-task set the figure is ~70%.\n"
    )
    (tmp_path / "README.md").write_text("# Shunt\n\nMeasured on the 74-task set.\n")
    assert _run("check_number_provenance.py", "--root", str(tmp_path)) == 1


def test_sh012_marked_site_outside_the_old_three_doc_list_passes(tmp_path: Path) -> None:
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\nOn the 184-task set the figure is ~70%. "
        "<!-- frozen-value: n=184, date=2026-08-11, run=49b8362 -->\n"
    )
    assert _run("check_number_provenance.py", "--root", str(tmp_path)) == 0


def test_sh012_generator_marker_accepts_a_generated_number(tmp_path: Path) -> None:
    # A generator marker needs no corpus — byte-for-byte regeneration
    # regeneration; the lint only requires the marker to be present.
    root = _number_tree(
        tmp_path,
        benchmark=f"# Benchmark\n\n{_MARKED_PP}\n",
    )
    assert _run("check_number_provenance.py", "--root", str(root)) == 0


# ── SH013 — every workflow job is time-bounded ────────────────────────────────
#
# The gate exists because a job with no `timeout-minutes` inherits GitHub's 360-minute
# default, and three integration-handshake legs rode that default for 2h16m.

_BOUNDED_JOB = "jobs:\n  build:\n    runs-on: ubuntu-latest\n    timeout-minutes: 10\n"


def test_sh013_catches_a_job_with_no_timeout(tmp_path: Path) -> None:
    f = tmp_path / "wf.yml"
    f.write_text("on: push\njobs:\n  build:\n    runs-on: ubuntu-latest\n")
    assert _run("check_workflow_timeouts.py", str(f)) == 1


def test_sh013_accepts_a_bounded_job(tmp_path: Path) -> None:
    f = tmp_path / "wf.yml"
    f.write_text(f"on: push\n{_BOUNDED_JOB}")
    assert _run("check_workflow_timeouts.py", str(f)) == 0


def test_sh013_rejects_a_bound_over_the_ceiling(tmp_path: Path) -> None:
    f = tmp_path / "wf.yml"
    f.write_text(
        "on: push\njobs:\n  build:\n    runs-on: ubuntu-latest\n    timeout-minutes: 360\n"
    )
    assert _run("check_workflow_timeouts.py", str(f)) == 1


def test_sh013_exempts_a_reusable_workflow_call(tmp_path: Path) -> None:
    # `timeout-minutes` is not a valid key on a `uses:` job — the callee carries its own.
    f = tmp_path / "wf.yml"
    f.write_text("on: push\njobs:\n  call:\n    uses: ./.github/workflows/other.yml\n")
    assert _run("check_workflow_timeouts.py", str(f)) == 0


def test_sh013_default_scan_of_this_repo_is_clean() -> None:
    assert _run("check_workflow_timeouts.py") == 0


# ── SH014 — relative docs links resolve inside the docs tree ──────────────────
#
# The gate exists because `docs/free-tier-smoke.md` linked `examples/integrations/README.md`
# — repo content, not site content — and `mkdocs build --strict` only ran on push to main,
# so it failed on the merge commit rather than on the PR.


def _docs_tree(tmp_path: Path, body: str, *, extra: str | None = None) -> Path:
    """A minimal repo root: mkdocs.yml naming docs_dir, plus one page."""
    (tmp_path / "mkdocs.yml").write_text("site_name: t\ndocs_dir: docs\n")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "page.md").write_text(body)
    if extra is not None:
        (docs / extra).write_text("# other\n")
    return tmp_path


def test_sh014_catches_a_link_escaping_the_docs_tree(tmp_path: Path) -> None:
    root = _docs_tree(tmp_path, "See [it](../examples/integrations/README.md).\n")
    assert _run("check_docs_links.py", "--root", str(root)) == 1


def test_sh014_catches_a_missing_in_tree_target(tmp_path: Path) -> None:
    # The shape of the real bug: resolves inside docs/, but nothing is there.
    root = _docs_tree(tmp_path, "See [it](examples/integrations/README.md).\n")
    assert _run("check_docs_links.py", "--root", str(root)) == 1


def test_sh014_accepts_an_in_tree_link(tmp_path: Path) -> None:
    root = _docs_tree(tmp_path, "See [it](other.md).\n", extra="other.md")
    assert _run("check_docs_links.py", "--root", str(root)) == 0


def test_sh014_resolves_a_link_carrying_a_fragment(tmp_path: Path) -> None:
    root = _docs_tree(tmp_path, "See [it](other.md#what-a-pass-means).\n", extra="other.md")
    assert _run("check_docs_links.py", "--root", str(root)) == 0


def test_sh014_ignores_absolute_and_anchor_targets(tmp_path: Path) -> None:
    # The prescribed fix must itself pass, as must mailto/anchor-only links.
    root = _docs_tree(
        tmp_path,
        "[a](https://github.com/KookaS/shunt/blob/main/examples/integrations/README.md)\n"
        "[b](#a-heading) [c](mailto:x@y.z)\n",
    )
    assert _run("check_docs_links.py", "--root", str(root)) == 0


def test_sh014_checks_image_targets_too(tmp_path: Path) -> None:
    # A moved asset is the same failure with a worse symptom: green build, broken image.
    root = _docs_tree(tmp_path, "![chart](assets/gone.webp)\n")
    assert _run("check_docs_links.py", "--root", str(root)) == 1


def test_sh014_honours_the_same_line_noqa(tmp_path: Path) -> None:
    root = _docs_tree(tmp_path, "See [it](nope.md). <!-- noqa: SH014 -->\n")
    assert _run("check_docs_links.py", "--root", str(root)) == 0


def test_sh014_default_scan_of_this_repo_is_clean() -> None:
    assert _run("check_docs_links.py") == 0
