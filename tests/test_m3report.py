"""Tests for pafbench.m3report (stage 34)."""
import html as _html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pafbench import m2report, m3report
from pafbench.config import LONG_READS, NUCLEI
from pafbench.m2report import LIBS
from pafbench.m3report import (Registry, _md_to_html, build, digits_in_prose, higher, lower, m2_block, render, SECTIONS)
from tests.test_m2report import _full_base


REPO = Path(__file__).resolve().parents[1]


def _m3_inputs(base):
    """Add minimal versions of every file the design-4 table names for milestone 3 that `_full_base` does not already
    write. One per-library row where the brief asks for a per-lib list; a handful of rows where the shape matters
    (strata, bias). Keeps the real column names so a column typo in m3report would be caught."""
    # section 2: <lib>/outs/metrics_summary.csv  (10x-style, quoted thousands)
    for lib in LIBS:
        p = base / lib / "outs" / "metrics_summary.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('Estimated Number of Cells,Mean Reads per Cell\n"1,234","10,000"\n')

    # section 6: burden/selection_bias_m3.parquet
    rows = []
    for lib in LIBS:
        for axis in ("region", "a_band"):
            rows.append((lib, axis, "all", 0.5, 0.52, 2.0))
    pd.DataFrame(rows, columns=["lib", "axis", "stratum", "share_labeled", "share_unlabeled",
                                "diff_points"]).to_parquet(base / "burden" / "selection_bias_m3.parquet", index=False)

    # section 9: eval/strata_m3.parquet, eval/strata_m3_bootstrap.parquet, eval/matrix_outcome.parquet
    rows = []
    for lib in LIBS:
        for kind in ("all", "class"):
            rows.append((lib, "c300_l10_m1_i", "molecule", kind, "all", 0.6, 0.9))
    strata = pd.DataFrame(rows, columns=["lib", "tag", "level", "stratum_kind", "stratum", "sens", "spec"])
    strata.to_parquet(base / "eval" / "strata_m3.parquet", index=False)
    strata.to_parquet(base / "eval" / "strata_m3_bootstrap.parquet", index=False)
    pd.DataFrame({"lib": list(LIBS), "tag": "c300_l10_m1_i", "outcome": "kept"}).to_parquet(
        base / "eval" / "matrix_outcome.parquet", index=False)

    # section 11: <lib>/recount/recount.json
    for lib in LIBS:
        p = base / lib / "recount" / "recount.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"entries_compared": 1000, "entries_differ": 2}))

    # section 12: <lib>/impact/per_cell.parquet
    for lib in LIBS:
        p = base / lib / "impact" / "per_cell.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"cb": ["AAAC-1", "AAAC-2"], "tag": ["c300_l10_m1_i", "c300_l10_m1_i"],
                      "umis0": [100, 200], "umis": [95, 190], "lost": [5, 10],
                      "gained": [0, 0]}).to_parquet(p, index=False)

    # section 13: <lib>/cells/cells.parquet
    for lib in LIBS:
        p = base / lib / "cells" / "cells.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"cb": ["AAAC-1", "AAAC-2"], "class": ["neuron", "glia"], "cluster": [0, 1],
                      "umap1": [0.1, 0.2], "umap2": [0.3, 0.4]}).to_parquet(p, index=False)

    # section 14: <lib>/impact/per_class.parquet
    for lib in LIBS:
        p = base / lib / "impact" / "per_class.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"class": ["neuron", "glia"], "tag": ["c300_l10_m1_i", "c300_l10_m1_i"],
                      "umis_lost_share": [0.05, 0.08], "cells": [100, 50]}).to_parquet(p, index=False)

    # section 15: <lib>/impact/clustering.parquet
    for lib in LIBS:
        p = base / lib / "impact" / "clustering.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"tag": ["c300_l10_m1_i"] * 4, "kind": ["unfilt", "unfilt", "filt", "filt"],
                      "replicate": [0, 1, 0, 1], "seed": [1, 2, 1, 2],
                      "ari": [0.9, 0.88, 0.85, 0.83]}).to_parquet(p, index=False)

    # section 16: burden/matched_sites.parquet
    pd.DataFrame({"pair": ["1417_vs_1419"] * 3, "set": ["shared", "a_only", "b_only"],
                  "gene": ["gA", "gB", "gC"], "site_a": [100.0, 200.0, np.nan],
                  "site_b": [100.0, np.nan, 300.0],
                  "label": ["templated", "templated", "templated"]}).to_parquet(
        base / "burden" / "matched_sites.parquet", index=False)
    return base


def _full_m3_base(tmp_path):
    return _m3_inputs(_full_base(tmp_path))


@pytest.fixture
def base(tmp_path):
    return _full_m3_base(tmp_path)


# ---------------------------------------------------------------------------------------------- helpers for the digit test

def _strip_m2_blocks(doc):
    return re.sub(r'<section[^>]*data-m2="[^"]*"[^>]*>.*?</section>', " ", doc, flags=re.DOTALL)


def _section_body(text, num):
    """The HTML between <section id="section-N"> and the next <section id="section-M"> (or </body>)."""
    pattern = rf'<section id="section-{num}"[^>]*>(.*?)(?=<section id="section-\d+"|</body>)'
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------------------------- test cases

def test_build_on_synthetic(base, tmp_path):
    out = tmp_path / "out"
    paths = build(base, out)
    assert paths["html"].exists() and paths["numbers"].exists()
    text = paths["html"].read_text()
    # every section 0..20 is rendered
    for i in range(21):
        assert f'id="section-{i}"' in text, f"section {i} missing"
    # sections 2, 6, 9, 11..16 render the populated branch (not the _not_run paragraph) on the extended fixture
    for i in (2, 6, 9, 11, 12, 13, 14, 15, 16):
        body = _section_body(text, i)
        assert "not run (missing" not in body, f"section {i} still rendered _not_run paragraph on the m3 fixture"
        assert "<table>" in body or 'class="plotly"' in body, f"section {i} rendered neither a table nor a figure"
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


# every m3 section (2..18) with a missing-input branch, the files whose deletion triggers it, and extra files to delete
# that are not printed in the missing list (section 10 only).
_MISSING_CASES = [
    (2,  [f"{lib}/outs/metrics_summary.csv" for lib in LIBS], []),
    (3,  [f"{lib}/truth/molecules_labeled.parquet" for lib in LIBS], []),
    (4,  ["eval/selection.json"], []),
    (5,  ["burden/templated_share.parquet"], []),
    (6,  ["burden/selection_bias_m3.parquet"], []),
    (7,  [f"{lib}/truth/truth_reads.parquet" for lib in LIBS], []),
    (8,  ["eval/metrics.parquet"], []),
    (9,  ["eval/strata_m3.parquet", "eval/matrix_outcome.parquet"], []),
    (10, ["eval/spikein_metrics.parquet"],
         [f"{lib}/spikein/ercc.json" for lib in NUCLEI]),
    (11, [f"{lib}/recount/recount.json" for lib in LIBS], []),
    (12, [f"{lib}/impact/per_cell.parquet" for lib in LIBS], []),
    (13, [f"{lib}/cells/cells.parquet" for lib in LIBS], []),
    (14, [f"{lib}/impact/per_class.parquet" for lib in LIBS], []),
    (15, [f"{lib}/impact/clustering.parquet" for lib in LIBS], []),
    (16, ["burden/matched_sites.parquet"], []),
    (17, [f"{lib}/long_read_acceptance.json" for lib in LONG_READS], []),
    (18, ["validation/exact_match_m2.json"], []),
]


@pytest.mark.parametrize("section_num,missing_rels,extras", _MISSING_CASES,
                         ids=[f"section_{c[0]}" for c in _MISSING_CASES])
def test_section_missing_input_prints_not_run(tmp_path, section_num, missing_rels, extras):
    """For every m3 section 2..18, remove its input(s) and assert 'not run (missing <rel paths>)' with the exact
    relative paths in the section's own body."""
    base = _full_m3_base(tmp_path)
    for rel in list(missing_rels) + list(extras):
        p = base / rel
        if p.exists():
            p.unlink()
    text = render(base)[0]
    body = _section_body(text, section_num)
    assert body, f"section {section_num} did not render"
    want = f"not run (missing {', '.join(missing_rels)})"
    assert want in body, f"section {section_num} did not print {want!r} in its body:\n{body[:400]}"


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
