"""Tests for pafbench.m3report (stage 34)."""
import html as _html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pafbench import m2report, m3report
from pafbench.m3report import (Registry, _md_to_html, build, digits_in_prose, higher, lower, m2_block, render, SECTIONS)
from tests.test_m2report import _full_base


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def base(tmp_path):
    return _full_base(tmp_path)


# ---------------------------------------------------------------------------------------------- helpers for the digit test

def _strip_m2_blocks(doc):
    return re.sub(r'<section[^>]*data-m2="[^"]*"[^>]*>.*?</section>', " ", doc, flags=re.DOTALL)


# ---------------------------------------------------------------------------------------------- test cases

def test_build_on_synthetic(base, tmp_path):
    out = tmp_path / "out"
    paths = build(base, out)
    assert paths["html"].exists() and paths["numbers"].exists()
    text = paths["html"].read_text()
    # every section 0..20 is rendered
    for i in range(21):
        assert f'id="section-{i}"' in text, f"section {i} missing"
    # every digit in milestone 3 prose resolves to a registry entry (outside the wrapped m2 blocks and the allow list)
    bad = digits_in_prose(text)
    assert bad == [], f"unregistered digits in m3 prose: {bad[:5]}"
    # numbers.json is machine readable, has at least a few entries, every one of them names a stage and a function
    entries = json.loads(paths["numbers"].read_text())["entries"]
    assert len(entries) >= 10
    for e in entries:
        assert e["source"].get("stage")
        assert e["source"].get("function")


def test_build_on_empty_base(tmp_path):
    base = tmp_path / "empty"
    base.mkdir()
    out = tmp_path / "out"
    paths = build(base, out)
    text = paths["html"].read_text()
    for i in range(21):
        assert f'id="section-{i}"' in text
    assert text.count("not run (missing") >= 15
    # the digit test still passes
    assert digits_in_prose(text) == []


def test_digit_test_catches_hand_written_digit(base, tmp_path, monkeypatch):
    """Monkey-patch one m3 template to add a hard-coded '42' in prose; the digit test fails and names it."""
    def bad_section_1(b, reg):
        return (f'<section id="section-1" class="s-m3"><h2>1. The problem and the tool</h2>'
                f'<p>Here is a hand-written number: 42.</p></section>')
    patched = tuple(bad_section_1 if fn.__name__ == "section_1" else fn for fn in SECTIONS)
    monkeypatch.setattr(m3report, "SECTIONS", patched)
    out = tmp_path / "out"
    paths = build(base, out)
    bad = digits_in_prose(paths["html"].read_text())
    assert any("42" in digit or "42" in ctx for digit, ctx in bad)


def test_wrapped_m2_block_passes(base):
    """An m2 wrapped block that carries a digit in its prose must not fail the digit test."""
    html = m2_block(m2report.truth_section, base, title="Milestone 2 truth")
    assert "data-m2=" in html
    # confirm it does carry a plain digit
    stripped_of_m2 = _strip_m2_blocks(html)
    assert re.search(r"\d", stripped_of_m2) is None
    # and digits_in_prose on the wrapper alone returns nothing
    assert digits_in_prose(html) == []


def test_changed_input_changes_prose(base, tmp_path):
    out1 = tmp_path / "out1"
    paths1 = build(base, out1)
    text1 = paths1["html"].read_text()
    # change one row in a truth table: lift the templated share in burden/templated_share.parquet
    bp = base / "burden" / "templated_share.parquet"
    df = pd.read_parquet(bp)
    df.loc[df.stratum_kind == "all", "share_all"] = 0.42000
    df.to_parquet(bp, index=False)
    out2 = tmp_path / "out2"
    paths2 = build(base, out2)
    text2 = paths2["html"].read_text()
    assert text1 != text2
    # the registry entry for the share changes between renders
    e1 = json.loads(paths1["numbers"].read_text())["entries"]
    e2 = json.loads(paths2["numbers"].read_text())["entries"]
    s1 = {e["source"].get("column"): e["value"] for e in e1 if e["source"].get("column") == "share_all"}
    s2 = {e["source"].get("column"): e["value"] for e in e2 if e["source"].get("column") == "share_all"}
    assert s1 and s2 and s1 != s2


def test_registry_entries_name_stage_and_function(base, tmp_path):
    paths = build(base, tmp_path / "out")
    entries = json.loads(paths["numbers"].read_text())["entries"]
    assert entries
    for e in entries:
        assert e["source"].get("stage"), e
        assert e["source"].get("function"), e
        assert e["id"].startswith("n_")


def test_section_missing_input_prints_not_run(base, tmp_path):
    """For each m3 section whose input exists, remove one of its inputs and assert 'not run (missing <rel path>)'."""
    # the exact relative paths this stage checks per section
    cases = [
        ("burden/templated_share.parquet", 5, "burden/templated_share.parquet"),
        ("validation/exact_match_m2.json", 18, "validation/exact_match_m2.json"),
    ]
    for rel, section_num, missing_rel in cases:
        p = base / rel
        assert p.exists(), rel
        p.unlink()
        text = render(base)[0]
        assert f'id="section-{section_num}"' in text
        want = f"not run (missing {missing_rel})"
        assert want in text, f"section {section_num} did not print {want!r}"
        # restore so later cases do not cascade
        _full_base(base.parent)  # rebuild siblings; the fixture is independent per test


def test_markdown_converter_on_m2_shapes(base):
    """Feed the converter the output of each of the eleven m2 functions and assert the HTML parses and has no stray | or **."""
    from html.parser import HTMLParser

    class _P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ok = True

        def error(self, message):
            self.ok = False

    for fn in m2report.SECTIONS + (m2report.headline,):
        md = fn(base)
        html = _md_to_html(md)
        p = _P()
        p.feed(html)
        # the converter must not leak Markdown table pipes or bold markers into text
        stripped = re.sub(r"<[^>]+>", " ", html)
        stripped = _html.unescape(stripped)
        # a stray `|` or `**` would mean a line was not handled
        assert "**" not in stripped, f"{fn.__name__}: stray ** in {stripped[:200]!r}"
        # pipes inside code/text would mean a Markdown table row leaked through
        for line in stripped.splitlines():
            trimmed = line.strip()
            assert not (trimmed.startswith("| ") and trimmed.endswith(" |")), f"{fn.__name__}: raw md row: {trimmed!r}"


def test_md_to_html_tables_and_lists():
    md = "## Title\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n- one\n- two\n\n**bold** and `code`."
    html = _md_to_html(md)
    assert "<h2>Title</h2>" in html
    assert "<table>" in html and "<th>A</th>" in html and "<td>1</td>" in html
    assert "<ul>" in html and "<li>one</li>" in html
    assert "<strong>bold</strong>" in html and "<code>code</code>" in html


def test_registry_formats_and_sources():
    reg = Registry()
    s = reg.num(1234, source={"file": "x.parquet", "column": "y", "function": "test_fn"})
    assert "1,234" in s and 'data-num="n_1"' in s
    s2 = reg.num(0.6667, source={"file": "x.parquet", "column": "z", "function": "test_fn"}, fmt="{:.3f}")
    assert "0.667" in s2 and 'n_2' in s2
    assert reg.dump()[0]["source"]["stage"]
    # a None value is written "n/a"
    s3 = reg.num(None, source={"file": "x.parquet", "column": "w", "function": "test_fn"})
    assert "n/a" in s3


def test_helper_words():
    assert higher(5, 3) == "higher"
    assert higher(3, 5) == "lower"
    assert higher(3, 3) == "the same"
    assert lower(3, 5) == "higher"
    # a None in either slot returns n/a
    assert higher(None, 1) == "n/a"


def test_digits_in_prose_allow_list_and_registry():
    html = ('<section id="section-1" class="s-m3"><p>library 1417 has <span data-num="n_1">500</span> molecules '
            'and the member cap is 1,000 nt.</p></section>')
    assert digits_in_prose(html) == []


def test_m2_block_marks_the_wrapper(base):
    html = m2_block(m2report.grid_section, base, title="Milestone 2 grid")
    assert 'data-m2="grid_section"' in html
    assert "<table" in html
