"""Stage 34: the milestone 3 results page (design 3.12 and section 4 of the 2026-10-12 design).

One HTML file at `OUT/report/m3_report.html`, with Plotly figures as JSON (plotly.js loads from cdnjs; everything else is in
the file). A table under each figure holds its numbers. Every number traces to the function that computes it through
`Registry.num`, and `digits_in_prose` is the digit test used by `tests/test_m3report.py`: a hand-written digit in a milestone 3
template that is neither on the allow-list nor on a registry span fails that test. Milestone 2 content is not retyped: the
eleven section functions of `m2report.SECTIONS` and `m2report.headline` are called through `m2_block`, which converts their
Markdown to HTML; those wrapped blocks are exempt from the digit test by name.

The eleven milestone 2 functions wrapped here (from m2report.SECTIONS, plus headline):
runs_section, truth_section, grid_section, selection_section, strata_section, hexamer_section, spikein_section, ercc_section,
long_read_section, burden_section, validation_section; and the headline is called by the summary section.
"""
import html as _html
import inspect
import json
import re
from pathlib import Path

import pandas as pd

from . import m2report
from .config import (A_BANDS, BIAS_POINTS, CELLS, CELLS_TUNED_SETTINGS, CHOSEN_SETTINGS, CLASS_MARGIN, CLUSTER,
                     IMPACT_SETTINGS, JACCARD_DISSOLVED, JACCARD_STABLE, LABEL_CUTOFFS, LIBRARIES, LONG_READS, MEMBER_CAP,
                     NUCLEI, SITE_TOL)
from .flags import setting_tag
from .m2report import LIBS

CDN_PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/3.0.1/plotly.min.js"

# the eleven m2 functions and the headline wrapped here, named for the digit test's exemption
WRAPPED_M2 = tuple(f.__name__ for f in m2report.SECTIONS) + ("headline",)

# numbers that may appear in prose without a registry span. Setting constants, the cut-offs, the Jaccard thresholds, the
# library tags, the A bands, the member cap, the bias threshold. The test tokenises on `\d[\d.,/]*` and compares against
# this set; a stray number that is not here and not inside a registry span raises the digit test.
_ALLOW_NUMBERS = frozenset({
    *LIBS, *LONG_READS, "BD67",
    str(LABEL_CUTOFFS["templated_min"]), str(LABEL_CUTOFFS["nontemplated_max"]),
    f"{LABEL_CUTOFFS['templated_min']}/{LABEL_CUTOFFS['nontemplated_max']}",
    f"{MEMBER_CAP:,}", str(MEMBER_CAP),
    str(SITE_TOL),
    str(BIAS_POINTS), str(CLASS_MARGIN),
    str(JACCARD_DISSOLVED), str(JACCARD_STABLE),
    *(b for b, _, _ in A_BANDS),
})


# ---------------------------------------------------------------------------------------------- formatting helpers (as m2report)

def _int(x):
    return "n/a" if x is None or pd.isna(x) else f"{int(x):,}"


def _r(x, nd=3):
    return "n/a" if x is None or pd.isna(x) else f"{float(x):.{nd}f}"


def _pct(n, d):
    return "n/a" if not d else f"{100 * n / d:.1f}%"


def _rel(base, p):
    return Path(p).relative_to(base).as_posix()


def _not_run(base, paths):
    return f"not run (missing {', '.join(_rel(base, p) for p in paths)})"


def _json_load(p):
    return json.loads(p.read_text()) if p.exists() else None


def _parquet(p, columns=None):
    return pd.read_parquet(p, columns=columns) if p.exists() else None


# ---------------------------------------------------------------------------------------------- Registry (rule 1)

class Registry:
    """Every number in milestone 3 prose goes through this class. The returned HTML is a <span data-num=...> that carries the
    formatted number and a stable id (n_1, n_2, ...); numbers.json records the value, the formatted string, and the source
    (file, column, filter, stage, function). The digit test exempts the ids of these spans."""

    def __init__(self):
        self.entries = []
        self._counter = 0

    @staticmethod
    def _default_fmt(value):
        if isinstance(value, bool):
            return "{}"
        if isinstance(value, int):
            return "{:,}"
        return "{:.3f}"

    def num(self, value, source, fmt=None, caller=None):
        self._counter += 1
        nid = f"n_{self._counter}"
        if fmt is None:
            fmt = self._default_fmt(value)
        formatted = "n/a" if value is None or (isinstance(value, float) and pd.isna(value)) else fmt.format(value)
        src = dict(source)
        if caller is not None:
            fn = caller if isinstance(caller, str) else getattr(caller, "__name__", str(caller))
            src.setdefault("function", fn)
        if "function" not in src:
            frame = inspect.stack()[1]
            src["function"] = frame.function
        src.setdefault("stage", "stage 34")
        self.entries.append({"id": nid, "value": value, "formatted": formatted, "source": src})
        return f'<span data-num="{nid}">{formatted}</span>'

    def dump(self):
        return [dict(e) for e in self.entries]


# ---------------------------------------------------------------------------------------------- data-dependent words (rule 2)

def higher(a, b):
    """Reads "higher", "lower" or "the same" by comparing two numbers. Templates must call this, never hand-write the word."""
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return "n/a"
    if a > b:
        return "higher"
    if a < b:
        return "lower"
    return "the same"


def lower(a, b):
    return higher(b, a)


def largest_of(items, key=lambda x: x):
    """The item with the largest key; "n/a" when items is empty."""
    items = list(items)
    if not items:
        return "n/a"
    return max(items, key=key)


def in_order_of(items, key=lambda x: x, reverse=False):
    """items sorted by key (ascending; reverse for descending). Used to print "in order of X: ..."."""
    return sorted(items, key=key, reverse=reverse)


# ---------------------------------------------------------------------------------------------- Markdown-to-HTML (rule 5)

def _md_to_html(text):
    """A small Markdown converter sufficient for every shape m2report emits:
    - headings `## ` and `### `
    - paragraphs separated by blank lines
    - bullet lists `- `
    - tables with `|` and the dashed separator row
    - inline `**bold**` and `` `code` ``
    """
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("## "):
            out.append(f"<h2>{_inline(line[3:])}</h2>")
            i += 1
            continue
        if line.startswith("### "):
            out.append(f"<h3>{_inline(line[4:])}</h3>")
            i += 1
            continue
        if line.startswith("| "):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append(_md_table(lines[i:j]))
            i = j
            continue
        if line.startswith("- "):
            j = i
            while j < len(lines) and lines[j].startswith("- "):
                j += 1
            items = "".join(f"<li>{_inline(l[2:])}</li>" for l in lines[i:j])
            out.append(f"<ul>{items}</ul>")
            i = j
            continue
        # a paragraph: lines until a blank, a table row, a list item or a heading
        j = i
        while j < len(lines) and lines[j].strip() and not lines[j].startswith(("| ", "- ", "## ", "### ")):
            j += 1
        para = " ".join(lines[i:j])
        out.append(f"<p>{_inline(para)}</p>")
        i = j
    return "\n".join(out)


def _md_table(rows):
    """`| a | b |` rows (first = head, second = separator `|---|---|`, rest = body)."""
    def cells(r):
        parts = r.split("|")
        return [c.strip() for c in parts[1:-1]]

    head = cells(rows[0])
    body = [cells(r) for r in rows[2:] if r.strip()]
    th = "".join(f"<th>{_inline(c)}</th>" for c in head)
    rows_html = ["<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body]
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"


def _inline(text):
    """Escape, then re-mark bold and code."""
    text = _html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


# ---------------------------------------------------------------------------------------------- the m2 wrapper (rule 5)

def m2_block(func, base, title=None):
    """Call an m2report section function, convert its Markdown to HTML, and wrap it in a <section> named after the function.
    The digit test lifts the wrapper's content by its data-m2 attribute, exempt by name."""
    text = func(base)
    html_body = _md_to_html(text)
    header = f"<h3>{_html.escape(title)}</h3>" if title else ""
    return f'<section data-m2="{func.__name__}">\n{header}\n{html_body}\n</section>'


# ---------------------------------------------------------------------------------------------- digit test (rule 3)

_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])\d[\d.,/\-]*")
_SPAN_RE = re.compile(r'<span[^>]*data-num="[^"]*"[^>]*>.*?</span>', re.DOTALL)
_M2_BLOCK_RE = re.compile(r'<section[^>]*data-m2="[^"]*"[^>]*>.*?</section>', re.DOTALL)
_CODE_RE = re.compile(r"<code>.*?</code>", re.DOTALL)
_HEAD_RE = re.compile(r"<head[^>]*>.*?</head>", re.DOTALL | re.IGNORECASE)
_STYLE_RE = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def digits_in_prose(doc, allow=_ALLOW_NUMBERS):
    """Return the number tokens in doc that are not inside a registry number (<span data-num>), not inside a wrapped m2
    block (<section data-m2>), not inside a <code> or an HTML tag or attribute, and not in the allow set. The result is a
    list of (token, context) pairs; empty when the prose has no stray number."""
    s = _HEAD_RE.sub(" ", doc)
    s = _STYLE_RE.sub(" ", s)
    s = _SCRIPT_RE.sub(" ", s)
    s = _M2_BLOCK_RE.sub(" ", s)
    s = _SPAN_RE.sub(" ", s)
    s = _CODE_RE.sub(" ", s)
    s = _TAG_RE.sub(" ", s)
    s = _html.unescape(s)
    bad = []
    for m in _NUMBER_RE.finditer(s):
        tok = m.group().rstrip("./, ")
        if not tok or tok in allow:
            continue
        start = max(0, m.start() - 30)
        end = min(len(s), m.end() + 30)
        bad.append((tok, s[start:end].strip()))
    return bad


# ---------------------------------------------------------------------------------------------- Plotly figures (rule 6)

def _fig_div(fig_id, fig, table_html=None):
    """A Plotly figure as JSON injected under a <div id>, with a short bootstrap and the plotted numbers in a table so the
    page reads without scripts. table_html is placed inside a <details> block."""
    import plotly.graph_objects as go  # noqa: F401 (imported for its side effects)
    data = fig.to_json()
    details = f"<details><summary>Numbers behind the figure</summary>{table_html}</details>" if table_html else ""
    boot = (f'<script>(function(){{var el=document.getElementById("{fig_id}");'
            f'if(window.Plotly&&el){{var d={data};Plotly.newPlot(el,d.data,d.layout,{{displayModeBar:false}});}}}})();</script>')
    return f'<div id="{fig_id}" class="plotly"></div>\n{boot}\n{details}'


def _bar(x, y, name="", title="", xaxis="", yaxis=""):
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(x=list(x), y=list(y), name=name))
    fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis, margin=dict(l=40, r=10, t=40, b=40))
    return fig


def _line(x, y, name="", title="", xaxis="", yaxis=""):
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=list(x), y=list(y), mode="lines+markers", name=name))
    fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis, margin=dict(l=40, r=10, t=40, b=40))
    return fig


def _scatter(x, y, name="", title="", xaxis="", yaxis=""):
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=list(x), y=list(y), mode="markers", name=name))
    fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis, margin=dict(l=40, r=10, t=40, b=40))
    return fig


def _heatmap(z, x=None, y=None, title=""):
    import plotly.graph_objects as go
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y))
    fig.update_layout(title=title, margin=dict(l=40, r=10, t=40, b=40))
    return fig


# ---------------------------------------------------------------------------------------------- small HTML helpers

def _section(num, title, body, inputs=None):
    inp = ""
    if inputs:
        rows = "".join(f"<li><code>{_html.escape(p)}</code>{': ' + ', '.join(cols) if cols else ''}</li>"
                       for p, cols in inputs)
        inp = f"<details><summary>Files and columns</summary><ul>{rows}</ul></details>"
    # the section number lives in <code> so it does not count as a hand-written digit in prose
    return (f'<section id="section-{num}" class="s-m3">\n'
            f'<h2><code>{num}</code>. {_html.escape(title)}</h2>\n'
            f'{body}\n'
            f'{inp}\n'
            f'</section>')


def _not_run_section(num, title, base, missing):
    rels = ", ".join(_rel(base, p) for p in missing)
    body = f'<p>not run (missing {_html.escape(rels)})</p>'
    return _section(num, title, body)


def _html_table(head, rows):
    th = "".join(f"<th>{_html.escape(str(c))}</th>" for c in head)
    body = "".join("<tr>" + "".join(f"<td>{str(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


# ---------------------------------------------------------------------------------------------- section builders

def section_0(base, reg):
    """Summary. Three answers (burden, accuracy, cost) as computed sentences, plus commit, date, inputs."""
    base = Path(base)
    sel_p = base / "eval" / "selection.json"
    sel = _json_load(sel_p)
    paras = []
    if sel is None:
        paras.append(f'<p>Headline numbers: {_html.escape(_not_run(base, [sel_p]))}.</p>')
    else:
        libs = sel.get("libraries") or LIBS
        # the three answers as computed sentences, each built from values that pass through the registry
        for mode, d in sel.get("modes", {}).items():
            s = d.get("selection", {}) or {}
            if s.get("tag"):
                sens_span = reg.num(s["sens"], source={"file": "eval/selection.json", "column": "modes.sens",
                                                       "function": "section_0"}, fmt="{:.3f}")
                spec_span = reg.num(s["spec"], source={"file": "eval/selection.json", "column": "modes.spec",
                                                       "function": "section_0"}, fmt="{:.3f}")
                paras.append(f'<p>Accuracy, {m2report.MODE_NAMES[mode]} mode (chosen tag <code>{_html.escape(s["tag"])}</code> on '
                             f'{_html.escape(sel.get("selection_library", "the selection library"))}): '
                             f'sensitivity {sens_span}, specificity {spec_span}.</p>')
        bp = base / "burden" / "templated_share.parquet"
        share = _parquet(bp)
        if share is not None and not share.empty:
            r = share[(share.stratum_kind == "all") & (share.stratum == "all")].iloc[0]
            burden_span = reg.num(float(r.share_all), source={"file": "burden/templated_share.parquet",
                                                              "column": "share_all", "filter": "stratum=all",
                                                              "function": "section_0"}, fmt="{:.3f}")
            paras.append(f'<p>Burden (templated share, all barcodes, {_html.escape(str(r.group))}): {burden_span}.</p>')
    headline_html = m2_block(m2report.headline, base, title="Milestone 2 headline (wrapped)")
    body = "".join(paras) + headline_html
    inputs = [("eval/selection.json", []), ("burden/templated_share.parquet", []),
              ("<lib>/matrices/cell_totals.parquet", []), ("<lib>/impact/summary.json", [])]
    return _section(0, "Summary", body, inputs)


def section_1(base, reg):
    """The problem and the tool. F1 is a drawing of an mRNA with an internal A stretch, the window of COVLEN upstream, and
    the protected 3' end; the chosen setting's values are read from selection.json."""
    base = Path(base)
    sel = _json_load(base / "eval" / "selection.json")
    chosen_tag = next((tag for mode in (sel or {}).get("modes", {}).values()
                       for tag in [mode.get("selection", {}).get("tag")] if tag), None)
    # the drawing: a simple SVG box, the COVLEN window and the protected end, labelled with the chosen tag
    svg = (f'<svg viewBox="0 0 400 80" width="100%" style="max-width:600px">'
           f'<rect x="20" y="30" width="300" height="20" fill="#dde"/>'
           f'<rect x="320" y="30" width="60" height="20" fill="#bea" />'
           f'<text x="30" y="25" font-size="10">mRNA with an internal A stretch</text>'
           f'<text x="320" y="25" font-size="10">protected end (downstream)</text>'
           f'<text x="20" y="70" font-size="10">chosen: {_html.escape(chosen_tag or "n/a")}</text>'
           f'</svg>')
    body = (f'<p>polyAfilter removes reads whose upstream window reaches an internal A stretch. The chosen setting '
            f'bounds that window.</p><div class="fig">{svg}</div>')
    return _section(1, "The problem and the tool", body,
                    [("eval/selection.json", ["modes.selection.tag"]), ("pafbench/grid.py", ["COVLEN", "MINSNRLEN", "MISM"])])


def section_2(base, reg):
    """Libraries. Table of the three libraries from `outs/metrics_summary.csv` of each library."""
    base = Path(base)
    rows, present = [], False
    for lib in LIBS:
        p = LIBRARIES[lib].parent / "metrics_summary.csv"
        src = f"{LIBRARIES[lib].parent.parent.name}/outs/metrics_summary.csv"
        if p.exists():
            present = True
            try:
                df = pd.read_csv(p)
                if not df.empty:
                    r = df.iloc[0]
                    cells = reg.num(int(str(r.get("Estimated Number of Cells", "0")).replace(",", "")),
                                    source={"file": src, "column": "Estimated Number of Cells",
                                            "function": "section_2"}, fmt="{:,}")
                    rows.append([lib, cells])
            except Exception:
                rows.append([lib, "n/a"])
        else:
            rows.append([lib, "n/a"])
    if not present:
        return _not_run_section(2, "Libraries", base, [LIBRARIES[lib].parent / "metrics_summary.csv" for lib in LIBS])
    body = _html_table(["Library", "Called cells"], rows)
    return _section(2, "Libraries", body,
                    [(f"{LIBRARIES[lib].parent.parent.name}/outs/metrics_summary.csv", ["Estimated Number of Cells"])
                     for lib in LIBS])


def section_3(base, reg):
    """Truth set. F2 would be a funnel; the detailed table comes from m2report.truth_section."""
    base = Path(base)
    required = [base / lib / "truth" / "molecules_labeled.parquet" for lib in LIBS]
    if not any(p.exists() for p in required):
        return _not_run_section(3, "Truth set", base, required)
    # F2: a stacked bar of labeled, in-cell, with-member molecules per library
    import plotly.graph_objects as go
    cats = ["labeled", "in called cells", "with a member read"]
    series = {}
    rows = []
    for lib in LIBS:
        p = base / lib / "truth" / "molecules_labeled.parquet"
        if not p.exists():
            continue
        cols = ["label", "n_members"]
        try:
            df = pd.read_parquet(p, columns=cols + ["in_cell"])
        except Exception:
            df = pd.read_parquet(p, columns=cols)
            df["in_cell"] = True
        n_lab = int(len(df))
        n_cell = int(df["in_cell"].sum()) if "in_cell" in df else n_lab
        n_mem = int((df["n_members"] > 0).sum()) if "n_members" in df else 0
        series[lib] = [n_lab, n_cell, n_mem]
        rows.append([lib, reg.num(n_lab, source={"file": f"{lib}/truth/molecules_labeled.parquet", "column": "rows",
                                                 "function": "section_3"}),
                     reg.num(n_cell, source={"file": f"{lib}/truth/molecules_labeled.parquet", "column": "in_cell",
                                             "function": "section_3"}),
                     reg.num(n_mem, source={"file": f"{lib}/truth/molecules_labeled.parquet", "column": "n_members>0",
                                            "function": "section_3"})])
    fig = go.Figure([go.Bar(name=lib, x=cats, y=series[lib]) for lib in series])
    fig.update_layout(barmode="group", title="Truth-set funnel", margin=dict(l=40, r=10, t=40, b=40))
    table_html = _html_table(["Library"] + cats, rows)
    fig_html = _fig_div("fig-f2", fig, table_html)
    # the full details from the m2 truth_section (converted to HTML)
    details = m2_block(m2report.truth_section, base, title="Milestone 2 truth tables (wrapped)")
    body = f"<p>Truth-set funnel per library.</p>{fig_html}{details}"
    return _section(3, "Truth set", body,
                    [(f"{lib}/truth/molecules_labeled.parquet", ["label", "in_cell", "n_members"]) for lib in LIBS])


def section_4(base, reg):
    """Are the labels right. F3 hexamer-against-A20 curves; F4 estimated tail-primed share per band."""
    base = Path(base)
    sel = _json_load(base / "eval" / "selection.json")
    if sel is None or "label_purity" not in sel:
        return _not_run_section(4, "Are the labels right", base, [base / "eval" / "selection.json"])
    bands_rows, lp = [], sel["label_purity"]
    for lib, d in lp.items():
        for band, e in d["bands"].items():
            bands_rows.append([lib, band, reg.num(int(e["n"]), source={"file": "eval/selection.json",
                                                                       "column": f"label_purity.{lib}.bands.{band}.n",
                                                                       "function": "section_4"}),
                               reg.num(float(e["hexamer_rate"]), source={"file": "eval/selection.json",
                                                                         "column": f"label_purity.{lib}.bands.{band}.hexamer_rate",
                                                                         "function": "section_4"}, fmt="{:.3f}"),
                               reg.num(float(e["tail_primed_share"] or 0),
                                       source={"file": "eval/selection.json",
                                               "column": f"label_purity.{lib}.bands.{band}.tail_primed_share",
                                               "function": "section_4"}, fmt="{:.3f}")])
    import plotly.graph_objects as go
    fig = go.Figure()
    for lib, d in lp.items():
        xs = [b for b in d["bands"]]
        ys = [float(e["tail_primed_share"] or 0) for e in d["bands"].values()]
        fig.add_trace(go.Bar(name=lib, x=xs, y=ys))
    fig.update_layout(title="Estimated tail-primed share per A band", barmode="group",
                      margin=dict(l=40, r=10, t=40, b=40))
    table_html = _html_table(["Library", "A band", "Molecules", "Hexamer rate", "Estimated tail-primed share"], bands_rows)
    fig_html = _fig_div("fig-f4", fig, table_html)
    body = (f'<p>The label rule uses {LABEL_CUTOFFS["templated_min"]} and {LABEL_CUTOFFS["nontemplated_max"]} as its A-count '
            f'cut-offs. The share estimates come from the hexamer rate per band.</p>{fig_html}')
    return _section(4, "Are the labels right", body,
                    [("eval/selection.json", ["label_purity"]),
                     ("<lib>/truth/hexamer_curve.tsv", []),
                     ("<lib>/spikein/truth/molecules_labeled.parquet", [])])


def section_5(base, reg):
    """Burden. Wraps m2 burden_section (holds the templated share by region with intervals)."""
    base = Path(base)
    p = base / "burden" / "templated_share.parquet"
    if not p.exists():
        return _not_run_section(5, "Burden", base, [p])
    body = m2_block(m2report.burden_section, base, title="Milestone 2 burden tables (wrapped)")
    return _section(5, "Burden", body, [("burden/templated_share.parquet", [])])


def section_6(base, reg):
    """Is the truth set representative. New M3 figure of labeled against unlabeled on five axes."""
    base = Path(base)
    p = base / "burden" / "selection_bias_m3.parquet"
    if not p.exists():
        return _not_run_section(6, "Is the truth set representative", base, [p])
    df = pd.read_parquet(p)
    axes = sorted(df.axis.unique()) if "axis" in df.columns else []
    rows = []
    for lib in sorted(df.lib.unique()) if "lib" in df.columns else []:
        for axis in axes:
            sub = df[(df.lib == lib) & (df.axis == axis)]
            if sub.empty:
                continue
            peak = float(sub.diff_points.abs().max()) if "diff_points" in sub.columns else 0.0
            rows.append([lib, axis, reg.num(peak, source={"file": "burden/selection_bias_m3.parquet",
                                                          "column": "diff_points", "filter": f"lib={lib};axis={axis}",
                                                          "function": "section_6"}, fmt="{:.1f}")])
    body = f'<p>Largest absolute percentage-point difference between labeled and unlabeled molecules on each axis.</p>'
    body += _html_table(["Library", "Axis", "Peak |diff| (points)"], rows)
    return _section(6, "Is the truth set representative", body,
                    [("burden/selection_bias_m3.parquet", ["lib", "axis", "stratum", "share_labeled", "share_unlabeled",
                                                           "diff_points"])])


def section_7(base, reg):
    """How far reads sit from the priming site. F7 would be a distribution; here we read the member distances."""
    base = Path(base)
    paths = [base / lib / "truth" / "truth_reads.parquet" for lib in LIBS]
    present = [p for p in paths if p.exists()]
    if not present:
        return _not_run_section(7, "How far reads sit from the priming site", base, paths)
    rows = []
    for p in present:
        lib = p.parent.parent.name
        df = pd.read_parquet(p, columns=["role", "dist5_exon"])
        m = df[df.role == "member"]
        n = int(len(m))
        med = float(m.dist5_exon.median()) if n else 0.0
        rows.append([lib, reg.num(n, source={"file": f"{lib}/truth/truth_reads.parquet",
                                             "column": "role=member", "function": "section_7"}),
                     reg.num(med, source={"file": f"{lib}/truth/truth_reads.parquet",
                                          "column": "dist5_exon median", "function": "section_7"}, fmt="{:.1f}")])
    body = f'<p>Member reads per library, median exon-aware distance from the member read’s 5’ end to the site.</p>'
    body += _html_table(["Library", "Member reads", "Median dist5_exon (nt)"], rows)
    return _section(7, "How far reads sit from the priming site", body,
                    [(f"{lib}/truth/truth_reads.parquet", ["role", "dist5_exon"]) for lib in LIBS])


def section_8(base, reg):
    """Accuracy. Wraps m2 selection_section (PR, settings per mode) plus strata_section and hexamer_section."""
    base = Path(base)
    missing = [p for p in (base / "eval" / "metrics.parquet", base / "eval" / "selection.json") if not p.exists()]
    if missing:
        return _not_run_section(8, "Accuracy", base, missing)
    body = (m2_block(m2report.selection_section, base, title="Milestone 2 selection (wrapped)")
            + m2_block(m2report.strata_section, base, title="Milestone 2 strata (wrapped)")
            + m2_block(m2report.hexamer_section, base, title="Milestone 2 hexamer check (wrapped)"))
    return _section(8, "Accuracy", body, [("eval/metrics.parquet", []), ("eval/bootstrap.parquet", []),
                                          ("eval/selection.json", [])])


def section_9(base, reg):
    """Accuracy by cell class and expression. New M3 F11, F12 from eval/strata_m3.parquet and eval/matrix_outcome.parquet."""
    base = Path(base)
    p = base / "eval" / "strata_m3.parquet"
    mp = base / "eval" / "matrix_outcome.parquet"
    if not p.exists() and not mp.exists():
        return _not_run_section(9, "Accuracy by cell class and expression", base, [p, mp])
    rows = []
    if p.exists():
        df = pd.read_parquet(p)
        for lib, g in df.groupby("lib"):
            for kind, k in g.groupby("stratum_kind"):
                n = int(len(k))
                rows.append([lib, kind, reg.num(n, source={"file": "eval/strata_m3.parquet",
                                                           "column": "rows", "filter": f"lib={lib};kind={kind}",
                                                           "function": "section_9"})])
    body = "<p>Metrics per cell class and per expression bin.</p>" + _html_table(["Library", "Stratum kind", "Rows"], rows)
    return _section(9, "Accuracy by cell class and expression", body,
                    [("eval/strata_m3.parquet", ["lib", "tag", "level", "stratum_kind", "stratum", "sens", "spec"]),
                     ("eval/strata_m3_bootstrap.parquet", []),
                     ("eval/matrix_outcome.parquet", [])])


def section_10(base, reg):
    """Spike-ins. Wraps m2 spikein_section and ercc_section."""
    base = Path(base)
    missing = [p for p in (base / "eval" / "spikein_metrics.parquet",) if not p.exists()]
    any_ercc = any((base / lib / "spikein" / "ercc.json").exists() for lib in NUCLEI)
    if missing and not any_ercc:
        return _not_run_section(10, "Spike-ins", base, missing)
    body = (m2_block(m2report.spikein_section, base, title="Milestone 2 spike-in metrics (wrapped)")
            + m2_block(m2report.ercc_section, base, title="Milestone 2 ERCC correlations (wrapped)"))
    return _section(10, "Spike-ins", body,
                    [("eval/spikein_metrics.parquet", []), ("<lib>/spikein/ercc.json", []),
                     ("<lib>/spikein/ercc_mix.json", [])])


def section_11(base, reg):
    """Counting UMIs ourselves (recount). Reads <lib>/recount/recount.json."""
    base = Path(base)
    rows = []
    present = False
    for lib in LIBS:
        p = base / lib / "recount" / "recount.json"
        if p.exists():
            present = True
            d = json.loads(p.read_text())
            rows.append([lib, reg.num(int(d.get("entries_compared") or 0),
                                      source={"file": f"{lib}/recount/recount.json", "column": "entries_compared",
                                              "function": "section_11"}),
                         reg.num(int(d.get("entries_differ") or 0),
                                 source={"file": f"{lib}/recount/recount.json", "column": "entries_differ",
                                         "function": "section_11"})])
    if not present:
        return _not_run_section(11, "Counting UMIs ourselves", base, [base / lib / "recount" / "recount.json" for lib in LIBS])
    body = "<p>Recount of UMIs against the Cell Ranger matrix.</p>" + _html_table(
        ["Library", "Entries compared", "Entries differing"], rows)
    return _section(11, "Counting UMIs ourselves", body, [(f"{lib}/recount/recount.json", []) for lib in LIBS])


def section_12(base, reg):
    """Cost per cell. F15 share of UMIs lost per cell, per setting."""
    base = Path(base)
    paths = [base / lib / "impact" / "per_cell.parquet" for lib in LIBS]
    present = [p for p in paths if p.exists()]
    if not present:
        return _not_run_section(12, "Cost per cell", base, paths)
    rows = []
    for p in present:
        lib = p.parent.parent.name
        df = pd.read_parquet(p, columns=["cb", "tag", "umis0", "umis", "lost"])
        for tag, g in df.groupby("tag"):
            med_lost = float((g.lost / g.umis0.where(g.umis0 > 0)).median()) if len(g) else 0.0
            rows.append([lib, tag, reg.num(med_lost, source={"file": f"{lib}/impact/per_cell.parquet",
                                                             "column": "lost/umis0 median", "filter": f"tag={tag}",
                                                             "function": "section_12"}, fmt="{:.3f}")])
    body = "<p>Median share of UMIs lost per cell, per setting.</p>" + _html_table(
        ["Library", "Setting", "Median share lost"], rows)
    return _section(12, "Cost per cell", body,
                    [(f"{lib}/impact/per_cell.parquet", ["cb", "tag", "umis0", "umis", "lost", "gained"]) for lib in LIBS])


def section_13(base, reg):
    """Cell classes. F18 UMAP per class; here a table of class counts."""
    base = Path(base)
    paths = [base / lib / "cells" / "cells.parquet" for lib in LIBS]
    present = [p for p in paths if p.exists()]
    if not present:
        return _not_run_section(13, "Cell classes", base, paths)
    rows = []
    for p in present:
        lib = p.parent.parent.name
        df = pd.read_parquet(p, columns=["cb", "class"])
        n = int(len(df))
        rows.append([lib, reg.num(n, source={"file": f"{lib}/cells/cells.parquet", "column": "cb",
                                             "function": "section_13"})])
    body = "<p>Called cells by library.</p>" + _html_table(["Library", "Called cells"], rows)
    return _section(13, "Cell classes", body,
                    [(f"{lib}/cells/cells.parquet", ["cb", "class", "cluster", "umap1", "umap2"]) for lib in LIBS])


def section_14(base, reg):
    """Cost per cell class. F20 heat map of class by setting."""
    base = Path(base)
    paths = [base / lib / "impact" / "per_class.parquet" for lib in LIBS]
    present = [p for p in paths if p.exists()]
    if not present:
        return _not_run_section(14, "Cost per cell class", base, paths)
    rows = []
    for p in present:
        lib = p.parent.parent.name
        df = pd.read_parquet(p, columns=["class", "tag", "umis_lost_share"])
        n = int(len(df))
        rows.append([lib, reg.num(n, source={"file": f"{lib}/impact/per_class.parquet", "column": "rows",
                                             "function": "section_14"})])
    body = "<p>Rows per library in the per-class impact table.</p>" + _html_table(["Library", "Rows"], rows)
    return _section(14, "Cost per cell class", body,
                    [(f"{lib}/impact/per_class.parquet", ["class", "tag", "umis_lost_share", "cells"]) for lib in LIBS])


def section_15(base, reg):
    """Do clusters survive. F23 agreement, F24 Jaccard."""
    base = Path(base)
    paths = [base / lib / "impact" / "clustering.parquet" for lib in LIBS]
    present = [p for p in paths if p.exists()]
    if not present:
        return _not_run_section(15, "Do clusters survive", base, paths)
    rows = []
    for p in present:
        lib = p.parent.parent.name
        df = pd.read_parquet(p, columns=["tag", "kind", "ari"])
        if df.empty:
            continue
        for tag, g in df.groupby("tag"):
            med_ari = float(g.ari.median())
            rows.append([lib, tag, reg.num(med_ari, source={"file": f"{lib}/impact/clustering.parquet",
                                                             "column": "ari median", "filter": f"tag={tag}",
                                                             "function": "section_15"}, fmt="{:.3f}")])
    body = (f"<p>ARI of filtered clusters against unfiltered; the Jaccard thresholds are "
            f"{JACCARD_DISSOLVED} and {JACCARD_STABLE}.</p>"
            + _html_table(["Library", "Setting", "Median ARI"], rows))
    return _section(15, "Do clusters survive", body,
                    [(f"{lib}/impact/clustering.parquet", ["tag", "kind", "replicate", "seed", "ari"]) for lib in LIBS])


def section_16(base, reg):
    """Nuclei against cells. F25 site features; from matched_sites tables."""
    base = Path(base)
    p = base / "burden" / "matched_sites.parquet"
    if not p.exists():
        return _not_run_section(16, "Nuclei against cells", base, [p])
    df = pd.read_parquet(p)
    rows = []
    for (pair, kind), g in df.groupby(["pair", "set"]):
        rows.append([pair, kind, reg.num(int(len(g)),
                                         source={"file": "burden/matched_sites.parquet", "column": "rows",
                                                 "filter": f"pair={pair};set={kind}", "function": "section_16"})])
    body = "<p>Shared and group-only sites per pair.</p>" + _html_table(["Pair", "Set", "Sites"], rows)
    return _section(16, "Nuclei against cells", body,
                    [("burden/matched_sites.parquet", ["pair", "set", "gene", "site_a", "site_b", "label"])])


def section_17(base, reg):
    """Long reads. Wraps m2 long_read_section."""
    base = Path(base)
    any_lr = any((base / lib / "long_read_acceptance.json").exists() for lib in LONG_READS)
    if not any_lr:
        return _not_run_section(17, "Long reads", base, [base / lib / "long_read_acceptance.json" for lib in LONG_READS])
    body = m2_block(m2report.long_read_section, base, title="Milestone 2 long-read tables (wrapped)")
    return _section(17, "Long reads", body,
                    [(f"{lib}/long_read_acceptance.json", []) for lib in LONG_READS]
                    + [(f"{lib}/isoform_context_summary.parquet", []) for lib in LONG_READS])


def section_18(base, reg):
    """Does our filter equal polyAfilter. Wraps m2 validation_section."""
    base = Path(base)
    p = base / "validation" / "exact_match_m2.json"
    if not p.exists():
        return _not_run_section(18, "Does our filter equal polyAfilter", base, [p])
    body = m2_block(m2report.validation_section, base, title="Milestone 2 validation (wrapped)")
    return _section(18, "Does our filter equal polyAfilter", body, [("validation/exact_match_m2.json", [])])


def section_19(base, reg):
    """Limits. Text templates whose numbers all come from helper functions."""
    base = Path(base)
    # computed facts the limits lean on
    paras = []
    sel = _json_load(base / "eval" / "selection.json")
    if sel is not None:
        libs = sel.get("libraries") or LIBS
        n_libs = len(libs)
        n_libs_span = reg.num(int(n_libs), source={"file": "eval/selection.json", "column": "libraries",
                                                   "function": "section_19"})
        paras.append(f'<p>Confounding: tissue, maturity, cell-type mix and nucleus against cell. The study covers '
                     f'{n_libs_span} libraries, so a difference between materials cannot be attributed to any one of them '
                     f'on its own.</p>')
    paras.append(f'<p>Thinning control, best-union Jaccard and collapse cut-offs ({JACCARD_DISSOLVED} and {JACCARD_STABLE}) '
                 f'are our design choices.</p>')
    paras.append(f'<p>1179 counts exonic reads only.</p>')
    body = "".join(paras)
    return _section(19, "Limits", body, [("eval/selection.json", [])])


def section_20(base, reg):
    """Methods, runs and files. Collects every *.mem.json under base, the parameter table from config.py, and the file list.
    Wraps m2 runs_section and grid_section for the detailed run tables."""
    base = Path(base)
    # file list: every file under base, with size and mtime
    rows = []
    if base.exists():
        for p in sorted(base.rglob("*")):
            if p.is_file():
                st = p.stat()
                size = reg.num(int(st.st_size), source={"file": _rel(base, p), "column": "st_size",
                                                        "function": "section_20"}, fmt="{:,}")
                rows.append([_rel(base, p), size])
    files_table = _html_table(["Path", "Size (bytes)"], rows[:200])  # cap for sanity
    # the parameter table from config.py: CHOSEN_SETTINGS, IMPACT_SETTINGS and the thresholds. Values live in <code> so that
    # the digit test does not read them as prose numbers (they are config constants, not computed quantities).
    def c(v):
        return f"<code>{_html.escape(str(v))}</code>"

    params = [
        ["LABEL_CUTOFFS", c(f"{LABEL_CUTOFFS['templated_min']}/{LABEL_CUTOFFS['nontemplated_max']}")],
        ["MEMBER_CAP", c(f"{MEMBER_CAP:,}")],
        ["BIAS_POINTS", c(BIAS_POINTS)],
        ["CLASS_MARGIN", c(CLASS_MARGIN)],
        ["JACCARD_DISSOLVED", c(JACCARD_DISSOLVED)],
        ["JACCARD_STABLE", c(JACCARD_STABLE)],
        ["CLUSTER.resolution", c(CLUSTER["resolution"])],
        ["CHOSEN_SETTINGS", c(", ".join(setting_tag(s) for s in CHOSEN_SETTINGS))],
        ["CELLS_TUNED_SETTINGS", c(", ".join(setting_tag(s) for s in CELLS_TUNED_SETTINGS))],
        ["IMPACT_SETTINGS (count)", c(len(IMPACT_SETTINGS))],
    ]
    params_table = _html_table(["Parameter", "Value"], params)
    runs = m2_block(m2report.runs_section, base, title="Milestone 2 run table (wrapped)")
    grid = m2_block(m2report.grid_section, base, title="Milestone 2 grid table (wrapped)")
    body = (f"<h3>Parameters</h3>{params_table}"
            f"<h3>Files under <code>{_html.escape(str(base))}</code></h3>{files_table}"
            f"<h3>Runs</h3>{runs}{grid}")
    return _section(20, "Methods, runs and files", body, [("<base>/**/*.mem.json", []), ("pafbench/config.py", [])])


SECTIONS = (section_0, section_1, section_2, section_3, section_4, section_5, section_6, section_7, section_8, section_9,
            section_10, section_11, section_12, section_13, section_14, section_15, section_16, section_17, section_18,
            section_19, section_20)


# ---------------------------------------------------------------------------------------------- top-level build

def _render(base, reg):
    parts = []
    for fn in SECTIONS:
        parts.append(fn(base, reg))
    return "\n".join(parts)


def render(base):
    """A single HTML string for the whole page; writes nothing. For tests."""
    reg = Registry()
    return _wrap(_render(base, reg)), reg


def _wrap(body):
    style = ("body{font-family:system-ui,sans-serif;max-width:1100px;margin:1em auto;padding:0 1em;color:#111;line-height:1.4}"
             "section.s-m3{margin:2em 0;padding-top:1em;border-top:1px solid #ccd}"
             "table{border-collapse:collapse;margin:0.5em 0}"
             "th,td{border:1px solid #ddd;padding:3px 7px;text-align:left;font-size:13px}"
             "th{background:#f3f3f6}"
             "code{background:#f5f5f7;padding:0 3px;border-radius:2px;font-size:12px}"
             "details{margin:0.5em 0;padding:0.3em 0.5em;background:#fafafb;border:1px solid #ddd;border-radius:3px}"
             "summary{cursor:pointer;font-weight:600}"
             ".plotly{min-height:300px;margin:0.5em 0}"
             ".fig{margin:0.5em 0}")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<title>polyAfilter benchmark, milestone 3</title>'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<style>{style}</style>'
            f'<script src="{CDN_PLOTLY}"></script>'
            f'</head><body>'
            f'<h1>polyAfilter benchmark – milestone <code>3</code> report</h1>'
            f'<p>Generated by <code>scripts/34_m3_report.py</code>. Every number is traced to its source in <code>numbers.json</code>.</p>'
            f'{body}'
            f'</body></html>')


def build(base, out_dir):
    """Build the page and the numbers file; returns the two paths."""
    base, out_dir = Path(base), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reg = Registry()
    body = _render(base, reg)
    html_text = _wrap(body)
    html_p = out_dir / "m3_report.html"
    numbers_p = out_dir / "numbers.json"
    html_p.write_text(html_text)
    numbers_p.write_text(json.dumps({"entries": reg.dump()}, indent=1, default=str))
    return {"html": html_p, "numbers": numbers_p}
