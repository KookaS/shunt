"""Tests for the .env credential loader (env-wins, no-op if missing)."""

from __future__ import annotations

from shunt.secrets import load_dotenv_file


def test_missing_file_is_noop(tmp_path):
    applied = load_dotenv_file(tmp_path / "absent.env")
    assert applied == {}


def test_loads_key_values_into_env(tmp_path, monkeypatch):
    monkeypatch.delenv("FOO_KEY", raising=False)
    p = tmp_path / ".env"
    p.write_text("FOO_KEY=abc123\n")
    applied = load_dotenv_file(p)
    assert applied == {"FOO_KEY": "abc123"}
    import os

    assert os.environ["FOO_KEY"] == "abc123"


def test_existing_env_var_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("FOO_KEY", "from-env")
    p = tmp_path / ".env"
    p.write_text("FOO_KEY=from-file\n")
    load_dotenv_file(p)
    import os

    # env-provided value is preserved (not clobbered by the file)
    assert os.environ["FOO_KEY"] == "from-env"


def test_override_replaces_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FOO_KEY", "from-env")
    p = tmp_path / ".env"
    p.write_text("FOO_KEY=from-file\n")
    load_dotenv_file(p, override=True)
    import os

    assert os.environ["FOO_KEY"] == "from-file"


def test_comments_blanks_and_malformed_skipped(tmp_path, monkeypatch):
    monkeypatch.delenv("REAL_KEY", raising=False)
    p = tmp_path / ".env"
    p.write_text("# a comment\n\nnot-a-pair\nREAL_KEY=value\n   \n")
    applied = load_dotenv_file(p)
    assert applied == {"REAL_KEY": "value"}


def test_quotes_and_whitespace_stripped(tmp_path, monkeypatch):
    monkeypatch.delenv("Q_KEY", raising=False)
    monkeypatch.delenv("S_KEY", raising=False)
    p = tmp_path / ".env"
    p.write_text('Q_KEY="quoted"\nS_KEY =  spaced-value  \n')
    applied = load_dotenv_file(p)
    assert applied["Q_KEY"] == "quoted"
    assert applied["S_KEY"] == "spaced-value"


def test_value_with_equals_is_preserved(tmp_path, monkeypatch):
    # A base64/url value containing '=' must keep everything after the first '='.
    monkeypatch.delenv("TOKEN", raising=False)
    p = tmp_path / ".env"
    p.write_text("TOKEN=ab=cd==\n")
    applied = load_dotenv_file(p)
    assert applied["TOKEN"] == "ab=cd=="


def test_export_prefixed_lines_are_tolerated(tmp_path, monkeypatch):
    # A `.env` copied from a shell profile may use `export KEY=value`.
    monkeypatch.delenv("EXPORTED_KEY", raising=False)
    p = tmp_path / ".env"
    p.write_text("export EXPORTED_KEY=val9\n")
    applied = load_dotenv_file(p)
    assert applied == {"EXPORTED_KEY": "val9"}
    import os

    assert "export EXPORTED_KEY" not in os.environ
    assert os.environ["EXPORTED_KEY"] == "val9"


def test_env_var_path_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("PATHED_KEY", raising=False)
    p = tmp_path / "custom.env"
    p.write_text("PATHED_KEY=xyz\n")
    monkeypatch.setenv("SHUNT_ENV_FILE", str(p))
    applied = load_dotenv_file()
    assert applied == {"PATHED_KEY": "xyz"}


def test_duplicate_key_last_wins_and_applied_agrees(tmp_path, monkeypatch):
    # dotenv/shell convention: the LAST assignment wins; and the returned map must
    # agree with what actually reached the environment (never misreport).
    monkeypatch.delenv("DUP", raising=False)
    p = tmp_path / ".env"
    p.write_text("DUP=first\nDUP=second\n")
    applied = load_dotenv_file(p)
    import os

    assert os.environ["DUP"] == "second"
    assert applied == {"DUP": "second"}


def test_env_var_wins_excluded_from_applied(tmp_path, monkeypatch):
    # A pre-existing env var wins AND must not appear in the returned applied map.
    monkeypatch.setenv("KEPT", "from-env")
    p = tmp_path / ".env"
    p.write_text("KEPT=from-file\nNEWK=n\n")
    monkeypatch.delenv("NEWK", raising=False)
    applied = load_dotenv_file(p)
    import os

    assert os.environ["KEPT"] == "from-env"
    assert applied == {"NEWK": "n"}  # KEPT was not applied (env won)


def test_crlf_line_endings_leave_no_carriage_return(tmp_path, monkeypatch):
    monkeypatch.delenv("CRLFK", raising=False)
    p = tmp_path / ".env"
    p.write_bytes(b"CRLFK=value\r\n")
    applied = load_dotenv_file(p)
    assert applied["CRLFK"] == "value"  # no trailing \r


def test_empty_value_stays_empty_and_falsy(tmp_path, monkeypatch):
    # The ".env.example copied but not filled in" path: KEY= sets "" (falsy), so
    # has_api_keys / the sk-missing fallback both trigger correctly.
    monkeypatch.delenv("BLANKK", raising=False)
    p = tmp_path / ".env"
    p.write_text("BLANKK=\n")
    applied = load_dotenv_file(p)
    import os

    assert os.environ["BLANKK"] == ""
    assert applied == {"BLANKK": ""}
    assert not os.environ["BLANKK"]
