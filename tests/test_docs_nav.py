"""The docs site signposts a getting-started page right after Home.

These read the sources (``mkdocs.yml`` and ``docs/*.md``) directly, so they run offline in
seconds. The strict site build and the docs-link lint remain the authoritative link gates.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# The navigation before the Quickstart entry existed, in order.
PRIOR_NAV = (
    ("Home", "index.md"),
    ("Architecture", "architecture.md"),
    ("Configuration", "configuration.md"),
    ("Feedback", "feedback.md"),
    ("The routing model", "routing.md"),
    ("Model triage", "model-selection.md"),
    ("Error detection & auto-escalation", "escalation.md"),
    ("The escalation claim", "escalation-claim.md"),
    ("Escalation dataset", "escalation-data-card.md"),
    ("Reproducing the escalation eval", "escalation-reproduction.md"),
    ("The live router", "inference.md"),
    ("The same eight figures, on invented data", "inference-demo.md"),
    ("Results", "results.md"),
    ("Research log", "research-log.md"),
    ("Benchmark", "benchmark.md"),
    ("Benchmark dataset", "benchmark-data.md"),
    ("Benchmark design", "benchmark-design.md"),
    ("Live smoke", "live-smoke-runbook.md"),
    ("Free-tier smoke", "free-tier-smoke.md"),
)

SOURCE_INSTALL = "git clone https://github.com/KookaS/shunt.git"
DOCKER_BUILD = "docker build -t shunt-router ."
LINK = re.compile(r"\]\(([^)\s]+)\)")


def _nav() -> list[tuple[str, str]]:
    raw = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    config = yaml.safe_load(raw.replace("!!python/name:", "!!str "))
    return [(label, target) for entry in config["nav"] for label, target in entry.items()]


def _read(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def _frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert match, "page has no YAML frontmatter"
    return yaml.safe_load(match.group(1))


def _shell_lines(text: str) -> list[str]:
    """Every line inside a fenced bash block, in document order."""
    return [
        line
        for block in re.findall(r"```bash\n(.*?)```", text, re.S)
        for line in block.splitlines()
    ]


def test_nav_signpost() -> None:
    nav = _nav()
    assert nav[0] == ("Home", "index.md")
    assert nav[1] == ("Quickstart", "quickstart.md"), "Quickstart is not the entry right after Home"
    assert [e for e in nav if e != ("Quickstart", "quickstart.md")] == list(PRIOR_NAV)
    assert all("#" not in target for _, target in nav), "a nav target carries a #fragment"


def test_install_steps() -> None:
    page = DOCS / "quickstart.md"
    assert page.is_file(), "docs/quickstart.md does not exist"
    text = page.read_text(encoding="utf-8")
    front = _frontmatter(text)
    assert front.get("title") and front.get("description")
    assert re.search(r"^# Quickstart\s*$", text, re.M), "page title is not '# Quickstart'"
    lines = _shell_lines(text)
    assert SOURCE_INSTALL in lines and "pip install -e ." in " ".join(lines)
    assert DOCKER_BUILD in lines
    assert any(line.startswith("docker run ") for line in lines)


def test_doctor_first() -> None:
    text = _read("quickstart.md")
    lines = _shell_lines(text)
    assert "shunt doctor" in lines, "the doctor command is not shown"
    doctor = lines.index("shunt doctor")
    starts = [i for i, line in enumerate(lines) if line.strip() == "shunt"]
    assert starts, "the start-the-proxy command is not shown"
    assert doctor < min(starts), "the proxy is started before `shunt doctor`"
    equivalent = text.index("are equivalent")
    assert text.index("shunt doctor") < equivalent, "the 'equivalent commands' text precedes doctor"
    after = text[text.index("### Check the install first") :]
    for claim in ("no provider call", "no spend", "will not download the"):
        assert claim in after, f"doctor description lost: {claim!r}"


def test_single_home() -> None:
    def homes(marker: str) -> list[str]:
        return [p.name for p in DOCS.glob("*.md") if marker in p.read_text(encoding="utf-8")]

    # Other pages (the architecture page, the reproduction guide) legitimately mention a clone;
    # the install blocks that moved are the ones the home page must no longer hold.
    assert homes(DOCKER_BUILD) == ["quickstart.md"], f"Docker build lives in {homes(DOCKER_BUILD)}"
    source = homes(SOURCE_INSTALL)
    assert "quickstart.md" in source and "index.md" not in source, f"install lives in {source}"
    assert not re.search(r"^## Quickstart\s*$", _read("index.md"), re.M)
    assert "quickstart.md" in _read("index.md"), "the home page keeps no pointer"


def test_links_intact() -> None:
    index = _read("index.md")
    contents = index[index.index("## Contents") :]
    first = next(line for line in contents.splitlines() if line.startswith("- "))
    assert "[Quickstart](quickstart.md)" in first, "Contents does not lead with Quickstart"
    assert "## An honest result" in index
    assert "index.md#an-honest-result" in _read("benchmark-data.md")
    for path in [*DOCS.glob("*.md"), ROOT / "README.md"]:
        assert "#quickstart" not in path.read_text(encoding="utf-8"), f"stale anchor in {path.name}"
    for name in ("index.md", "quickstart.md"):
        for target in LINK.findall(_read(name)):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            assert (DOCS / target.split("#")[0]).exists(), f"{name}: broken link {target}"
