"""Stage 21: the milestone 2 results page. One function per section; each reads its files under a base directory (OUT) and
prints "not run (missing ...)" with the absent file names instead of numbers."""
import inspect
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .config import A_BANDS, CELLS, LABEL_CUTOFFS, LONG_READS, MEMBER_CAP, NUCLEI, POLYAFILTER_COMMIT, SENS_CAPS
from .evaluate import MODULES
from .flags import Setting, setting_tag
from .grid import COVLENS, MINLENS, MISMS, PUBLISHED, published
from .polyasite import BINS, NULL_LABEL
from .regions import REGION_NAMES
from .bd67 import read_outcome
from .spikein import label_spike_site
from .truth import in_population
from .validation import parse_tag, run_stats

LIBS = NUCLEI + CELLS
_SPIKE = inspect.signature(label_spike_site).parameters
_READ = inspect.signature(read_outcome).parameters
LABELS = ("templated", "ambiguous", "nontemplated")
LABEL_NAMES = {"templated": "templated", "ambiguous": "ambiguous", "nontemplated": "non-templated", "all": "all"}
MODE_NAMES = {"e": "default", "i": "-i"}
BANDS = tuple(b[0] for b in A_BANDS)
POP_NAMES = {"all": "all barcodes", "cells": "called cells", "noncells": "other barcodes"}
UNIT_NAMES = {"with_member": "molecules with a member read within the cap", "all_labeled": "all labeled molecules"}
SPIKE_LABELS = ("internal", "tail", "other")
EXPR_TERTILES = ("low", "mid", "high", "no_gene")
READING = f"our reading of the polyAfilter code at commit {POLYAFILTER_COMMIT}, spec 3"
BAM_TAGS_URL = "https://www.10xgenomics.com/support/software/cell-ranger/latest/analysis/outputs/cr-outputs-bam"
CUTOFFS = "{templated_min}/{nontemplated_max}".format(**LABEL_CUTOFFS)
PUBLISHED_TAGS = [setting_tag(s) for s in published()]
N_CANDIDATES = len(COVLENS) * len(MINLENS) * len(MISMS)
PUBLISHED_SOURCES = {(300, 10, 1): "README example, not a tuned value", (275, 9, 1): "paper Fig. 4A optimum, 10x PBMC1",
                     (180, 11, 2): "paper Fig. 4A optimum, 10x PBMC2",
                     (375, 3, 2): "paper Fig. 4A optimum, 10x mouse cortex nuclei with intronic counting"}
CONTROL_TAGS = [setting_tag(Setting(c, None, 0, introns)) for introns in (False, True) for c in COVLENS]


def _int(x):
    return "n/a" if x is None or pd.isna(x) else f"{int(x):,}"


def _r(x, nd=3):
    return "n/a" if x is None or pd.isna(x) else f"{float(x):.{nd}f}"


def _pct(n, d):
    return "n/a" if not d else f"{100 * n / d:.1f}%"


def _share(n, d):
    return f"{_int(n)} ({_r(n / d if d else None)})"


def _ci(est, lo, hi):
    return "n/a" if est is None or pd.isna(est) else f"{_r(est)} [{_r(lo)}, {_r(hi)}]"


def _sec(x):
    return "n/a" if x is None else f"{x:,.0f}"


def _value(x):
    """A JSON scalar for a table cell: ints with thousands separators, floats to 3 decimals, null as n/a."""
    if isinstance(x, (bool, str)):
        return str(x)
    if isinstance(x, int):
        return _int(x)
    return _r(x)


def _table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def _notes(out, notes):
    """out plus notes as a bullet list, after a blank line (a bare line right after a table would be read as a table row)."""
    return out + ([""] if out[-1] else []) + [f"- {n}" for n in notes] if notes else out


def _rel(base, p):
    return Path(p).relative_to(base).as_posix()


def _not_run(base, paths):
    return f"not run (missing {', '.join(_rel(base, p) for p in paths)})"


def _json(p):
    return json.loads(p.read_text()) if p.exists() else None


def _parquet(p, columns=None):
    return pd.read_parquet(p, columns=columns) if p.exists() else None


def _command(cmd):
    """The command from its script on (or whole), with paths reduced to their last component."""
    i = next((k for k, a in enumerate(cmd) if a.endswith((".py", ".sh"))), 0)
    return " ".join(Path(a).name if "/" in a else a for a in cmd[i:])


def _chosen(sel):
    """{tag: [roles]} of the chosen settings (per mode) and the published ones, chosen first."""
    roles = {}
    for mode, d in sel["modes"].items():
        if d["selection"]["tag"]:
            roles.setdefault(d["selection"]["tag"], []).append(f"chosen ({MODE_NAMES[mode]})")
    for tag in sel["published"]:
        roles.setdefault(tag, []).append("published")
    return roles


def _entry(sel, tag, lib):
    """Per-level metrics of a chosen or published tag on one library ({} when the tag was not in its truth_flags)."""
    for d in sel["modes"].values():
        if d["selection"]["tag"] == tag:
            return d["metrics"].get(lib, {})
    return sel["published"].get(tag, {}).get(lib, {})


def _pop_text(sel):
    """"Computed on: <population>, <units>." from a loaded eval/selection.json, "" when it is absent."""
    if sel is None:
        return ""
    return (f"Computed on: {POP_NAMES[sel.get('selection_population', 'all')]}, "
            f"{UNIT_NAMES[sel.get('selection_units', 'with_member')]}.")


def _computed_on(base):
    return _pop_text(_json(base / "eval" / "selection.json"))


def _report_tags(base):
    """Chosen tags from eval/selection.json (when present) then the published tags, plus a note when the selection is absent."""
    p = base / "eval" / "selection.json"
    sel = _json(p)
    if sel is None:
        return list(PUBLISHED_TAGS), [f"chosen settings: {_not_run(base, [p])}; published settings only", ""]
    chosen = [d["selection"]["tag"] for d in sel["modes"].values() if d["selection"]["tag"]]
    return list(dict.fromkeys(chosen + PUBLISHED_TAGS)), []


def runs_section(base):
    base = Path(base)
    out = ["## 1. Runs", "",
           "Stage runs under memwatch (every `*.mem.json` under the base directory outside `*/original/`; a JSON written "
           "elsewhere is not found). This includes stage 20's createTRANS run for the remove-all control "
           "(`validation/none_control/`), an original-tool run, and its createTRANS and original-tool runs on the 1179 subset "
           "(`validation/subset_1179/`). Peak RSS is the largest sum over the process tree at a sample, so pages shared by "
           "forked workers count once per process and a peak between two samples is missed, so the peak of a shorter run "
           "reads near zero. SAMPLING", ""]
    mem = [p for p in base.rglob("*.mem.json") if "original" not in p.relative_to(base).parts[:-1]]
    rows, intervals = [], set()
    for p in mem:
        d = json.loads(p.read_text())
        intervals.add(d.get("interval_s"))
        rows.append([_command(d["cmd"]), _rel(base, p), _sec(d["elapsed_s"]), _r(d["peak_tree_rss_gib"], 2), d["exit_status"]])
    shown = sorted(f"{x:g} s" for x in intervals if x is not None)
    out[-2] = out[-2].replace("SAMPLING", f"memwatch samples every {' and '.join(shown)} (interval_s in the files)." if shown else
                              "The sampling interval is recorded as interval_s in each file; no file is available here.")
    out += _table(["Command", "File", "Wall (s)", "Peak tree RSS (GiB)", "Exit"], sorted(rows)) if rows else \
        ["not run (missing *.mem.json)"]

    out += ["", "Stage 13 grid scans (`<lib>/grid/summary.json`; wall time is that of the last invocation, so a resumed run "
                "counts only its own references; the per-reference sum covers all parts):", ""]
    rows, notes = [], []
    for lib in LIBS:
        p = base / lib / "grid" / "summary.json"
        d = _json(p)
        if d is None:
            notes.append(f"{lib}: {_not_run(base, [p])}")
            continue
        per = d["per_ref"]
        slow = max(per, key=lambda r: per[r]["seconds"]) if per else None
        rows.append([lib, _int(len(per)), _int(d["records"]), _int(d["primary"]), d["workers"], d["scanned"], _sec(d["seconds"]),
                     _sec(sum(v["seconds"] for v in per.values())), f"{slow} ({_sec(per[slow]['seconds'])} s)" if slow else "n/a"])
    if rows:
        out += _table(["Library", "References", "Records", "Primary records", "Workers", "Scanned in last run", "Wall of last run (s)",
                       "Sum over references (s)", "Slowest reference"], rows)
    out = _notes(out, notes)

    out += ["", "Original polyAfilter runs (`<lib>/original/<step>.time.txt`, `.mem.json`). Max RSS is the largest single "
                "process (GNU time); exit 128 + N means killed by signal N.", ""]
    rows = []
    for p in sorted(base.glob("*/original/*.time.txt")):
        st = run_stats(str(p)[:-len(".time.txt")])
        wall = "running" if st["elapsed_s"] is None else f"{st['elapsed_s'] / 3600:.2f}"
        rows.append([p.parent.parent.name, p.name[:-len(".time.txt")], wall, _r(st["max_rss_gib"], 2),
                     _r(st["peak_tree_rss_gib"], 2), "n/a" if st["exit_status"] is None else st["exit_status"]])
    out += _table(["Library", "Step", "Wall (h)", "Max RSS (GiB)", "Peak tree RSS (GiB)", "Exit"], rows) if rows else \
        ["not run (missing */original/*.time.txt)"]
    return "\n".join(out)


def _band_lines(base, sel, sel_path):
    """A band 8-9 in the truth section: the estimated tail-primed share per library from label_purity, the reading of it
    where the estimate is above 0.5, and the largest sites of band 8-9 in each whole-cell library."""
    band = BANDS[2]
    out = [""]
    purity = (sel or {}).get("label_purity")
    if purity is None:
        out.append(f"A band {band} tail-primed share: " + (_not_run(base, [sel_path]) if sel is None else
                                                          "not run (eval/selection.json has no label_purity)"))
    else:
        out += [f"A band {band}, estimated tail-primed share (`eval/selection.json` label_purity):", ""]
        out += _table(["Library", f"Molecules in A band {band}", "Hexamer rate", "Estimated tail-primed share"],
                      [[lib, _int(d["bands"][band]["n"]), _r(d["bands"][band]["hexamer_rate"]), _r(d["bands"][band]["tail_primed_share"])]
                       for lib, d in purity.items()])
        for lib, d in purity.items():
            if d["bands"][band]["tail_primed_share"] > 0.5:
                out += ["", f"By this estimate most A band {band} molecules of {lib} are real poly(A) sites that follow "
                            f"genome-encoded A's, and the {CUTOFFS} rule labels them templated, so read {lib} by A band."]
    pop = (sel or {}).get("selection_population", "all")
    for lib in CELLS:
        p = base / lib / "truth" / "molecules_labeled.parquet"
        out.append("")
        if not p.exists():
            out.append(f"Largest A band {band} sites of {lib}: {_not_run(base, [p])}")
            continue
        m = pd.read_parquet(p, columns=["ref", "strand", "site", "a_band", "gene", "hexamer"] + (["in_cell"] if "in_cell" in pq.read_schema(p).names else []))
        if pop != "all" and "in_cell" not in m:
            out.append(f"Largest A band {band} sites of {lib}: not run (no in_cell column in {_rel(base, p)}, rerun stage 12)")
            continue
        m = in_population(m, pop)
        g = m[m.a_band == band].groupby(["ref", "site", "strand"]).agg(n=("hexamer", "size"), hexamer=("hexamer", "mean"),
                                                                       gene=("gene", "first")).reset_index()
        g = g.sort_values(["n", "ref", "site"], ascending=[False, True, True]).head(5)
        out += [f"Largest A band {band} sites of {lib} in {POP_NAMES[pop]} (`{_rel(base, p)}`; site is the priming-site boundary, "
                "gene the alphabetically first gene id at the last cDNA base, share with the hexamer the share of the site's "
                "molecules with AATAAA or ATTAAA upstream):", ""]
        out += _table(["Ref", "Site", "Strand", "Gene id", "Molecules", "Share with the hexamer"],
                      [[r.ref, _int(r.site), r.strand, r.gene or "none", _int(r.n), _r(r.hexamer)] for r in g.itertuples()])
    return out


def truth_section(base):
    base = Path(base)
    out = ["## 2. Truth sets", "",
           "Source: `<lib>/truth/molecules_labeled.parquet` and `<lib>/truth/truth_reads.parquet` (stage 12). "
           f"Labels at {CUTOFFS} A's in the 20 nt after the site; members are selected member reads (gene match, "
           "dist3_exon >= -25) by exon-aware distance. The truth table and the region table below cover all barcodes (every row "
           "of molecules_labeled.parquet); the tables that come from `eval/selection.json` name their population.", "",
           f"has_ts: a pa read of the molecule has a 10x ts tag above 0. Cell Ranger documentation, Barcoded BAM Tags ({BAM_TAGS_URL}): "
           "ts is \"The number of template switch oligo (TSO) nucleotides trimmed from the 5' end of read 2.\" Our reading: such a "
           "read reaches the 5' end of the cDNA.", ""]
    data, notes = {}, []
    for lib in LIBS:
        paths = [base / lib / "truth" / "molecules_labeled.parquet", base / lib / "truth" / "truth_reads.parquet"]
        absent = [p for p in paths if not p.exists()]
        if absent:
            notes.append(f"{lib}: {_not_run(base, absent)}")
            continue
        sites = base / lib / "pa_sites.parquet"
        data[lib] = (pd.read_parquet(paths[0], columns=["label", "region", "has_ts", "a20", "hexamer", "genes", "n_members"]),
                     pd.read_parquet(paths[1], columns=["role", "dist5_exon"]),
                     pq.ParquetFile(sites).metadata.num_rows if sites.exists() else None)
    if data:
        def row(name, f):
            return [name] + [f(m, r) for m, r, _ in data.values()]

        def hexa(s):
            return f"{_r(s.mean() if len(s) else None)} (n {_int(len(s))})"

        rows = [row("Molecules", lambda m, r: _int(len(m)))]
        rows += [row(LABEL_NAMES[k], lambda m, r, k=k: f"{_int((m.label == k).sum())} ({_pct((m.label == k).sum(), len(m))})")
                 for k in LABELS]
        rows += [row("With a pa read with ts > 0 (has_ts)", lambda m, r: _int(m.has_ts.sum())),
                 row("Multi-gene (gene = alphabetically first id)", lambda m, r: _int(m.genes.str.contains(";").sum())),
                 row("pa reads in labeled molecules", lambda m, r: _int((r.role == "pa").sum()))]
        rows += [["pa reads in pa_sites.parquet"] + [_int(n) for _, _, n in data.values()]]
        rows += [row(f"Members within {cap:,} nt", lambda m, r, cap=cap: _int(((r.role == "member") & (r.dist5_exon <= cap)).sum()))
                 for cap in SENS_CAPS]
        rows += [row(f"Molecules with a member within {MEMBER_CAP:,} nt", lambda m, r: _int((m.n_members > 0).sum())),
                 row("Hexamer rate at A 0-4", lambda m, r: hexa(m.hexamer[m.a20 <= 4])),
                 row("Hexamer rate at A 18-20", lambda m, r: hexa(m.hexamer[m.a20 >= 18]))]
        out += _table(["Quantity"] + list(data), rows)
    out = _notes(out, notes)
    if data:
        out += ["", "Region at the last cDNA base, by label (molecules):", ""]
        rows = []
        for lib, (m, _, _) in data.items():
            c = pd.crosstab(m.region, m.label).reindex(index=list(REGION_NAMES), columns=list(LABELS), fill_value=0)
            for region, t in c.iterrows():
                resolved = t.templated + t.nontemplated
                rows.append([lib, region, _int(t.templated), _int(t.ambiguous), _int(t.nontemplated), _int(t.sum()),
                             _r(t.templated / resolved if resolved else None)])
        out += _table(["Library", "Region", "templated", "ambiguous", "non-templated", "total",
                       "templated / (templated + non-templated)"], rows)

    p = base / "eval" / "selection.json"
    sel = _json(p)
    out += ["", "Ambiguous molecules: hexamer-based estimate of their tail-primed share over the labeled molecules of "
                f"{POP_NAMES[(sel or {}).get('selection_population', 'all')]}, with and without a member read, not only "
                f"the cap {MEMBER_CAP:,} units the metrics use (`eval/selection.json` ambiguous_tail_share, stage 17). Two "
                "estimates, each (rate ambiguous - rate of the reference) / (rate non-templated - rate of the reference), "
                "clipped to [0, 1]; they differ in the reference: the templated class, or the molecules with 14 or more A's "
                "(a20 >= 14), because in whole cells the templated class itself holds tail-primed molecules:", ""]
    if sel is None:
        out.append(_not_run(base, [p]))
    elif "ambiguous_tail_share" not in sel:
        out.append("not run (eval/selection.json has no ambiguous_tail_share)")
    else:
        ats = sel["ambiguous_tail_share"]
        nkeys = list(dict.fromkeys(k for d in ats.values() for k in d if k.startswith("n_")))
        rows = [[lib, _r(d.get("estimate")), _r(d.get("estimate_ref14")), _r(d.get("rate_templated")),
                 _r(d.get("rate_nontemplated")), _r(d.get("rate_ambiguous")), _r(d.get("rate_ref14"))]
                + [_int(d.get(k)) for k in nkeys] for lib, d in ats.items()]
        out += _table(["Library", "Estimated tail-primed share (reference: templated class)",
                       "Estimated tail-primed share (reference: a20 >= 14)", "Hexamer rate templated",
                       "Hexamer rate non-templated", "Hexamer rate ambiguous", "Hexamer rate a20 >= 14"] + nkeys, rows)

    out += ["", "Label purity: molecules, AATAAA/ATTAAA rate and estimated tail-primed share per A band over the labeled "
                f"molecules of {POP_NAMES[(sel or {}).get('selection_population', 'all')]} (`eval/selection.json` label_purity, "
                "stage 17); the last row of a library is the templated class "
                "as a whole:", ""]
    if sel is None:
        out[-2:] = [f"Label purity: {_not_run(base, [p])}"]
    elif "label_purity" not in sel:
        out[-2:] = ["Label purity: not run (eval/selection.json has no label_purity)"]
    else:
        rows = []
        for lib, d in sel["label_purity"].items():
            rows += [[lib, band, _int(e["n"]), _r(e["hexamer_rate"]), _r(e["tail_primed_share"])] for band, e in d["bands"].items()]
            t = d["classes"]["templated"]
            rows.append([lib, "templated class", _int(t["n"]), _r(t["hexamer_rate"]), _r(t["tail_primed_share"])])
        out += _table(["Library", "A band", "Molecules", "Hexamer rate", "Estimated tail-primed share"], rows)
        out += ["", "The estimated tail-primed share is an estimate, not a count: it compares the hexamer rate of the band against "
                    "the rate at 14 or more A's, scaled by the rate at 0 to 4 A's, and is limited to 0 to 1."]
    out += _band_lines(base, sel, p)
    return "\n".join(out)


def grid_section(base):
    base = Path(base)
    out = ["## 3. Grid: removed primary records", "",
           "Source: `<lib>/grid/summary.json` (stage 13). Share of primary records in brackets. The remove-all control "
           "(MINSNRLEN none) removes every non-terminal read on an expressed transcript at that COVLEN (" + READING + "); it "
           "bounds removals only within its own mode.", ""]
    data, notes = {}, []
    for lib in LIBS:
        p = base / lib / "grid" / "summary.json"
        d = _json(p)
        if d is None:
            notes.append(f"{lib}: {_not_run(base, [p])}")
        else:
            data[lib] = d
    if data:
        rows = [["records", "Records fetched"] + [_int(d["records"]) for d in data.values()],
                ["records", "Primary records"] + [_int(d["primary"]) for d in data.values()]]
        for kind, tags in (("published", PUBLISHED_TAGS), ("remove-all control", CONTROL_TAGS)):
            for tag in tags:
                cells = []
                for d in data.values():
                    v = d["removed"].get(tag)
                    cells.append("n/a" if v is None else f"{_int(v)} ({_pct(v, d['primary'])})")
                rows.append([kind, tag] + cells)
        out += _table(["Kind", "Setting"] + list(data), rows)
    out = _notes(out, notes)
    return "\n".join(out)


def _populations_lines(base, sel):
    """The "Called cells and other barcodes" table and the per-population picks, or a not-run line."""
    if "populations" not in sel or "molecules" not in sel:
        return ["Called cells and other barcodes: not run (eval/selection.json has no populations)"]
    libs, sl = sel["libraries"], sel["selection_library"]
    p = base / "burden" / "templated_share_population.parquet"
    share = _parquet(p)
    look = {} if share is None else {(r.lib, r.population): r for r in share[share.stratum_kind == "all"].itertuples()}
    rows = []
    for lib in libs:
        m = sel["molecules"].get(lib, {})
        cells = [f"{_r(look[lib, pop].share_all)} ({_int(look[lib, pop].templated)} of {_int(look[lib, pop].total)})"
                 if (lib, pop) in look else "n/a" for pop in POP_NAMES]
        rows.append([lib, _int(m.get("labeled")), "n/a" if m.get("in_cell") is None else
                     f"{_int(m['in_cell'])} ({_pct(m['in_cell'], m['labeled'])})"] + cells)
    out = ["Called cells and other barcodes: labeled molecules (stage 12), the share of them in cells called by Cell Ranger, "
           "and the templated share of all labeled molecules in each population (templated molecules, then all, in "
           "parentheses):", ""]
    out += _table(["Library", "Labeled molecules", "In called cells"] + [f"Templated share, {n}" for n in POP_NAMES.values()], rows)
    if share is None:
        out += ["", f"Templated share by population: {_not_run(base, [p])}"]
    out += ["", f"The selection rule run on {sl} in each population and with each unit definition (member cap {MEMBER_CAP:,} nt, "
                f"{CUTOFFS} labels), with the pick's molecule-level sens / spec on every library in the same population and units:", ""]
    rows = []
    for pop, units_ in sel["populations"].items():
        for um, modes in units_.items():
            for mode, d in modes.items():
                s = d["selection"]
                rows.append([POP_NAMES[pop], UNIT_NAMES[um], MODE_NAMES[mode], s["tag"] or "none", s["n_qualifying"]] + [
                    f"{_r(m['sens'])} / {_r(m['spec'])}" if (m := d["metrics"].get(lib, {}).get("molecule")) else "n/a" for lib in libs])
    out += _table(["Population", "Units", "Mode", f"Chosen (on {sl})", "Qualifying"] + [f"{lib} sens / spec" for lib in libs], rows)
    return out


def _mapq_lines(sel):
    if "mapq_sensitivity" not in sel:
        return ["Molecules with mapq_min below 255 left out: not run (eval/selection.json has no mapq_sensitivity)"]
    mq, min_spec = sel["mapq_sensitivity"], sel["min_spec"]
    out = ["Molecules with mapq_min below 255 left out (" + "; ".join(f"{k}: {_int(v)}" for k, v in mq["left_out"].items()) +
           f"), selection rerun on {sel['selection_library']} and the chosen and published settings, molecule level. "
           f"{_pop_text(sel)}", ""]
    rows = []
    for mode, d in mq["modes"].items():
        s = d["selection"]
        none = "no candidates" if s["n_candidates"] == 0 else f"none reached specificity {min_spec:g}"
        rows.append([MODE_NAMES[mode], s["tag"] or none, _r(s["sens"]), _r(s["spec"]), f"{s['n_qualifying']} / {s['n_candidates']}"])
    out += _table(["Mode", "Chosen setting", "Sens", "Spec", "Qualifying / candidates"], rows)
    rows = []
    for tag, rs in _chosen(mq).items():
        for lib in mq["left_out"]:
            m = _entry(mq, tag, lib).get("molecule")
            rows.append([tag, ", ".join(rs), lib] + (["n/a"] * 4 if not m else [
                _int(m["tp"] + m["fn"]), _int(m["tn"] + m["fp"]), _r(m["sens"]), _r(m["spec"])]))
    return out + [""] + _table(["Setting", "Role", "Library", "Templated", "Non-templated", "Sens", "Spec"], rows)


def _cells_tuned_lines(sel):
    """The "Setting tuned on whole cells" table (stage 17 cells_tuned) or a not-run line."""
    if "cells_tuned" not in sel:
        return ["Setting tuned on whole cells: not run (eval/selection.json has no cells_tuned)"]
    sl = sel["selection_library"]
    rows, tuned_libs, said = [], [], []
    for tuned, modes in sel["cells_tuned"].items():
        tuned_libs.append(tuned)
        for mode, d in modes.items():
            s, other = d["selection"], sel["modes"][mode]
            m, o = d["metrics"].get(tuned, {}).get("molecule"), other["metrics"].get(tuned, {}).get("molecule")
            rows.append([MODE_NAMES[mode], tuned, s["tag"] or "none", _r(m and m["sens"]), _r(m and m["spec"])])
            rows.append([MODE_NAMES[mode], sl, other["selection"]["tag"] or "none", _r(o and o["sens"]), _r(o and o["spec"])])
            said.append(f"{MODE_NAMES[mode]} {_r(o and o['spec'])} and {_r(o and o['sens'])} against {_r(m and m['spec'])} and "
                        f"{_r(m and m['sens'])}")
    t = " and ".join(tuned_libs)
    out = [f"Setting tuned on whole cells (descriptive): the selection rule run on {t} called cells in place of {sl}, next to "
           f"the setting picked on {sl}; sensitivity and specificity at molecule level on {t}:", ""]
    out += _table(["Mode", "Tuned on", "Setting", f"Sens on {t}", f"Spec on {t}"], rows)
    out += ["", f"On {t}, specificity and sensitivity of the setting picked on {sl} next to the setting picked on {t}: " +
                "; ".join(said) + ".",
            f"This is not a recommendation: {t} has no second cell library to hold out, and tissue (organoid against adult "
            "retina) is confounded with material."]
    return out


def selection_section(base):
    base = Path(base)
    out = ["## 4. Setting selection", ""]
    p = base / "eval" / "selection.json"
    sel = _json(p)
    if sel is None:
        return "\n".join(out + [_not_run(base, [p])])
    libs, sl, min_spec = sel["libraries"], sel["selection_library"], sel["min_spec"]
    out += [f"Source: `eval/selection.json`, `eval/bootstrap.parquet`, `eval/metrics.parquet` (stage 17). Rule: {sel['rule']}", "",
            f"Computed on: {POP_NAMES[sel.get('selection_population', 'all')]}, "
            f"{UNIT_NAMES[sel.get('selection_units', 'with_member')]}.", ""]
    rows, fronts = [], []
    for mode, d in sel["modes"].items():
        s = d["selection"]
        none = "no candidates" if s["n_candidates"] == 0 else f"none reached specificity {min_spec:g}"
        rows.append([MODE_NAMES[mode], s["tag"] or none, _r(s["sens"]), _r(s["spec"]), _r(s["prec"]),
                     f"{s['n_qualifying']} / {s['n_candidates']}"])
        if s["n_candidates"] and s["frontier"]:
            fronts.append((mode, s["tag"], s["frontier"]))
    out += _table(["Mode", f"Chosen setting (on {sl})", "Sens", "Spec", "Prec", "Qualifying / candidates"], rows)
    out += ["", f"{N_CANDIDATES} candidates per mode: {len(COVLENS)} COVLEN x {len(MINLENS)} MINSNRLEN x {len(MISMS)} MISM (the "
                "remove-all controls and the published settings off the grid are not candidates)."]
    roles = _chosen(sel)
    for mode, tag, front in fronts:
        out += ["", f"Frontier, {MODE_NAMES[mode]} mode ({'no setting chosen; ' if tag is None else ''}candidates not dominated in "
                    f"sensitivity and specificity on {sl}; chosen and published settings marked):", ""]
        out += _table(["Setting", "Role", "Sens", "Spec"], [[t, ", ".join(roles.get(t, [])), _r(a), _r(b)] for t, a, b in front])
    out += ["", "Published settings (spec 3), COVLEN/MINSNRLEN/MISM, each run in both modes:", ""]
    out += _table(["Setting", "Source"], [[f"{c}/{l}/{m}", PUBLISHED_SOURCES[c, l, m]] for c, l, m in PUBLISHED])

    mp = base / "eval" / "metrics.parquet"
    metrics = _parquet(mp)
    bp = base / "eval" / "bootstrap.parquet"
    boot = _parquet(bp)
    ci = {}
    if boot is not None:
        b = boot[(boot.level == "molecule") & (boot.cap == MEMBER_CAP)]
        ci = {(r.lib, r.tag, r.metric): (r.lo, r.hi) for r in b.itertuples()}
    out += ["", f"Chosen and published settings, molecule level (all reads removed), member cap {MEMBER_CAP:,} nt, {CUTOFFS} "
                f"labels, 95% gene-bootstrap interval in brackets:", ""]
    roles = _chosen(sel)
    rows, rows_lvl = [], []
    wt = {} if metrics is None else {(r.lib, r.tag): r for r in metrics[
        (metrics.cap == MEMBER_CAP) & (metrics.cutoffs == CUTOFFS) & (metrics.level == "molecule_weighted")
        & (metrics.stratum_kind == "all")].itertuples()}

    def weighted(tag, lib, e):
        r = e.get("molecule_weighted") or ({"sens": wt[lib, tag].sens, "spec": wt[lib, tag].spec} if (lib, tag) in wt else None)
        return "n/a" if r is None else f"{_r(r['sens'])} / {_r(r['spec'])}"

    for tag, rs in roles.items():
        for lib in libs:
            e = _entry(sel, tag, lib)
            m = e.get("molecule")
            if not m:
                rows.append([tag, ", ".join(rs), lib] + ["n/a"] * 5)
                rows_lvl.append([tag, lib] + ["n/a"] * 4)
                continue
            cells = [_ci(m[k], *ci.get((lib, tag, k), (None, None))) for k in ("sens", "spec", "prec")]
            rows.append([tag, ", ".join(rs), lib, _int(m["tp"] + m["fn"]), _int(m["tn"] + m["fp"])] + cells)
            rows_lvl.append([tag, lib] + [f"{_r(e[lv]['sens'])} / {_r(e[lv]['spec'])}" if lv in e else "n/a"
                                          for lv in ("molecule", "molecule_majority")] + [weighted(tag, lib, e)] +
                            [f"{_r(e['read']['sens'])} / {_r(e['read']['spec'])}" if "read" in e else "n/a"])
    out += _table(["Setting", "Role", "Library", "Templated", "Non-templated", "Sens", "Spec", "Prec"], rows)
    if boot is None:
        out += ["", f"Bootstrap: {_not_run(base, [bp])}"]
    else:
        clusters = boot[boot.level == "molecule"].groupby("lib", sort=False).n_clusters.first()
        out += ["", "Bootstrap: 2.5th and 97.5th percentiles over gene-cluster resamples (cluster = gene, or a 10 kb locus "
                    "window for molecules without a gene (intergenic and antisense)); clusters " + "; ".join(f"{k} {_int(v)}" for k, v in clusters.items()) + "."]
    out += ["", "Majority rule (molecule removed when most of its reads are; our extension, spec 5.3), the weighted level "
                "(each molecule counts 1 / (the number of its truth reads inside the units: pa reads and members within the cap), "
                "which approximates 1 / UMI family size; spec 4.1 selection bias) and read level, sens / spec:", ""]
    out += _table(["Setting", "Library", "Molecule (all reads)", "Molecule (majority)", "Molecule (weighted 1 / reads in units)",
                   "Read"], rows_lvl)

    out += ["", "Remove-all control (MINSNRLEN none, every non-terminal read on an expressed transcript is removed; " + READING + "), molecule "
                f"level, cap {MEMBER_CAP:,} nt, {CUTOFFS} labels, stratum all of the selection population "
                f"({POP_NAMES[sel.get('selection_population', 'all')]}, {UNIT_NAMES[sel.get('selection_units', 'with_member')]}); "
                "95% gene-bootstrap interval in brackets where "
                "`eval/bootstrap.parquet` has one. The control ignores A content, so it bounds what a mode can reach:", ""]
    if metrics is None:
        out[-2:] = [f"Remove-all control: {_not_run(base, [mp])}"]
    else:
        ctl = {(r.lib, r.tag): r for r in metrics[(metrics.cap == MEMBER_CAP) & (metrics.cutoffs == CUTOFFS)
                                                  & (metrics.level == "molecule") & (metrics.stratum_kind == "all")].itertuples()}
        rows = []
        for tag in CONTROL_TAGS:
            if any((lib, tag) in ctl for lib in libs):
                rows.append([tag] + [c for lib in libs for c in (
                    "n/a" if (lib, tag) not in ctl else
                    (_ci(ctl[lib, tag].sens, *ci[lib, tag, "sens"]) if (lib, tag, "sens") in ci else _r(ctl[lib, tag].sens)),
                    "n/a" if (lib, tag) not in ctl else
                    (_ci(ctl[lib, tag].spec, *ci[lib, tag, "spec"]) if (lib, tag, "spec") in ci else _r(ctl[lib, tag].spec)))])
        out += _table(["Setting"] + [f"{lib} {k}" for lib in libs for k in ("sens", "spec")], rows)

    out += ["", "Units per member cap (molecules with a truth read missing from truth_flags are dropped). " +
                _pop_text(sel), ""]
    out += _table(["Library", "Cap (nt)", "Read units", "Molecule units", "Missing reads", "Dropped molecules"],
                  [[lib, _int(cap), _int(u["read_units"]), _int(u["molecule_units"]), _int(u["missing_reads"]),
                    _int(u["dropped_molecules"])] for lib, caps in sel["units"].items() for cap, u in caps.items()])
    out += ["", f"Member cap sensitivity (selection rerun on {sl} at each cap):", ""]
    out += _table(["Cap (nt)", "Mode", "Chosen", "Sens", "Spec", "Qualifying / candidates"],
                  [[_int(cap), MODE_NAMES[mode], s["tag"] or "none", _r(s["sens"]), _r(s["spec"]),
                    f"{s['n_qualifying']} / {s['n_candidates']}"] for cap, modes in sel["cap_sensitivity"].items()
                   for mode, s in modes.items()])
    out += ["", f"Label cut-off sweep (selection on {sl} at each templated / non-templated pair; molecule level on every "
                "library):", ""]
    rows = []
    for e in sel["cutoff_sweep"]:
        by = e["by_library"]
        rows.append([e["templated_min"], e["nontemplated_max"], MODE_NAMES[e["mode"]], e["tag"] or "none", e["n_qualifying"]]
                    + [f"{_r(by[lib]['sens'])} / {_r(by[lib]['spec'])}" if by.get(lib) else "n/a" for lib in libs])
    out += _table(["Templated min", "Non-templated max", "Mode", "Chosen", "Qualifying"] + [f"{lib} sens / spec" for lib in libs], rows)

    out += [""] + _populations_lines(base, sel) + [""] + _mapq_lines(sel)

    out += [""] + _cells_tuned_lines(sel) + [""]
    out += ["Intergenic and antisense molecules. 'Removed at most' is the largest count over all settings in "
            f"metrics.parquet (molecule level, cap {MEMBER_CAP:,}, {CUTOFFS}); 'never removed' is the templated count in these "
            "two regions minus that, and the ceiling is the stratum-all sensitivity if every other templated molecule were "
            "removed. Intergenic and antisense non-templated molecules count as retained negatives. Spec 3 (from the "
            "polyAfilter code, commit 5ce0098) says reads in these regions are never removed; the columns show what this data "
            "gives.", ""]
    if metrics is None:
        out[-2:] = [f"Intergenic and antisense molecules: {_not_run(base, [mp])}"]
        return "\n".join(out)
    m = metrics[(metrics.cap == MEMBER_CAP) & (metrics.cutoffs == CUTOFFS) & (metrics.level == "molecule")]
    rows, notes = [], []
    for lib in libs:
        a = m[(m.lib == lib) & (m.stratum_kind == "all")]
        g = m[(m.lib == lib) & (m.stratum_kind == "region") & m.stratum.isin(["intergenic", "antisense"])]
        g = g.groupby("tag")[["tp", "fn", "tn", "fp"]].sum()
        if a.empty or g.empty:
            continue
        pos, gpos, gneg = a.tp.iloc[0] + a.fn.iloc[0], g.tp.iloc[0] + g.fn.iloc[0], g.tn.iloc[0] + g.fp.iloc[0]
        rows.append([lib, _int(pos), _int(gpos), _int(g.tp.max()), _int(gpos - g.tp.max()),
                     _r((pos - gpos + g.tp.max()) / pos if pos else None), _int(gneg), _int(g.fp.max())])
        neg = a.tn.iloc[0] + a.fp.iloc[0]
        if lib == sl and neg > gneg:
            notes.append(f"While a setting keeps every intergenic and antisense non-templated molecule, specificity >= "
                         f"{min_spec:g} over all {_int(neg)} non-templated units on {sl} is at least ({min_spec:g} x {_int(neg)} - "
                         f"{_int(gneg)}) / {_int(neg - gneg)} = {_r((min_spec * neg - gneg) / (neg - gneg))} over the "
                         f"{_int(neg - gneg)} genic ones.")
    out += _table(["Library", "Templated units", "Intergenic and antisense templated", "Templated removed at most",
                   "Never removed (intergenic and antisense)", "Sensitivity ceiling", "Intergenic and antisense non-templated",
                   "Non-templated removed at most"], rows)
    return "\n".join(_notes(out, notes))


def strata_section(base):
    base = Path(base)
    out = ["## 5. Metrics by A band and region", ""]
    p = base / "eval" / "metrics.parquet"
    metrics = _parquet(p)
    if metrics is None:
        return "\n".join(out + [_not_run(base, [p])])
    tags, note = _report_tags(base)
    out += [f"Source: `eval/metrics.parquet` (stage 17), member cap {MEMBER_CAP:,} nt, {CUTOFFS} labels; molecule level unless "
            f"stated. Specificity exists only in band {BANDS[0]} and sensitivity only in bands {BANDS[2]} and up; band {BANDS[1]} is "
            "ambiguous. " + _computed_on(base), ""] + note
    m = metrics[(metrics.cap == MEMBER_CAP) & (metrics.cutoffs == CUTOFFS) & metrics.tag.isin(tags)]
    libs = list(dict.fromkeys(m.lib))
    looks = {lv: {(r.tag, r.lib, r.stratum_kind, r.stratum): r for r in g.itertuples()} for lv, g in m.groupby("level")}
    look = looks.get("molecule", {})

    def get(lv, tag, lib, kind, stratum, metric):
        r = looks.get(lv, {}).get((tag, lib, kind, stratum))
        return None if r is None else getattr(r, metric)

    present = [(tag, lib) for tag in tags for lib in libs if (tag, lib, "all", "all") in look]

    def sens_spec(kind, strata):
        return _table(["Setting", "Library"] + list(strata),
                      [[tag, lib] + [f"{_r(get('molecule', tag, lib, kind, g, 'sens'))} / {_r(get('molecule', tag, lib, kind, g, 'spec'))}"
                                     if (tag, lib, kind, g) in look else "n/a" for g in strata] for tag, lib in present])

    def a_band_table(lv):
        return _table(["Setting", "Library", f"Spec {BANDS[0]}"] + [f"Sens {b}" for b in BANDS[2:]],
                      [[tag, lib, _r(get(lv, tag, lib, "a_band", BANDS[0], "spec"))]
                       + [_r(get(lv, tag, lib, "a_band", b, "sens")) for b in BANDS[2:]] for tag, lib in present])

    out += ["By A band at the molecule level:", ""] + a_band_table("molecule")
    out += ["", "By A band at the read level (a read counts as a unit):", ""] + a_band_table("read")
    out += ["", "By A band at the molecule-majority level (molecule removed when most of its reads are):", ""]
    out += a_band_table("molecule_majority")
    out += ["", "By region, molecule level, sens / spec:", ""] + sens_spec("region", REGION_NAMES)
    out += ["", "By expression tertile of the gene, molecule level, sens / spec (no_gene: molecules without a gene). Expression tertile: "
                "genes ranked by the number of their labeled molecules in the selection units (ties by gene id) and split into three "
                "equal gene groups, so the groups hold very different numbers of molecules; each molecule takes its gene's group.", ""]
    out += sens_spec("expr_tertile", EXPR_TERTILES)
    out += ["", "By has_ts (a pa read of the molecule has a ts tag above 0, section 2), molecule level, sens / spec:", ""]
    out += sens_spec("has_ts", ("True", "False"))
    out += ["", "Units per stratum at the molecule level (the same for every setting):", ""]
    rows = []
    for lib in libs:
        mol = m[(m.level == "molecule") & (m.lib == lib)]
        first = mol[(mol.tag == mol.tag.iloc[0]) & mol.stratum_kind.isin(["a_band", "region", "expr_tertile", "has_ts"])]
        rows += [[lib, r.stratum_kind, r.stratum, _int(r.tp + r.fn), _int(r.tn + r.fp)] for r in first.itertuples()]
    out += _table(["Library", "Stratum kind", "Stratum", "Templated units", "Non-templated units"], rows)
    return "\n".join(out)


def hexamer_section(base):
    base = Path(base)
    out = ["## 6. Label-free check: hexamer rate in removed and kept molecules", ""]
    p = base / "eval" / "hexamer_check.parquet"
    h = _parquet(p)
    if h is None:
        return "\n".join(out + [_not_run(base, [p])])
    out += [f"Source: `eval/hexamer_check.parquet` (stage 17), molecule level, member cap {MEMBER_CAP:,} nt. Hexamer: "
            "AATAAA/ATTAAA fully inside the window 6 to 35 nt upstream of the site (sites.has_hexamer; the 5 bases just before the "
            "site are not searched). " + _computed_on(base), ""]
    h = h[h.cap == MEMBER_CAP]
    out += _table(["Setting", "Library", "Label", "Removed", "Removed hexamer rate", "Kept", "Kept hexamer rate"],
                  [[r.tag, r.lib, LABEL_NAMES.get(r.label, r.label), _int(r.removed_n), _r(r.removed_hex_rate), _int(r.kept_n),
                    _r(r.kept_hex_rate)] for r in h.itertuples()])
    return "\n".join(out)


def spikein_section(base):
    base = Path(base)
    out = ["## 7. Spike-ins", "",
           "Source: `<lib>/spikein/truth/molecules_labeled.parquet` (stage 14), `eval/spikein_metrics.parquet` (stage 17). "
           "spike_label comes from the distance to the annotated 3' end; label from the A count after the site. Spike label "
           f"(spikein.label_spike_site) tail: within {_SPIKE['tail_tol'].default} nt of an annotated 3' end; internal: not tail and "
           f"at least {_SPIKE['internal_min'].default} nt upstream of one; other: neither, so other molecules are left out of the "
           "agreement and of the metrics (positives are internal, negatives tail). Spike-in metrics "
           "use every barcode, whatever population the setting selection uses: spike-ins are free RNA in every droplet. 1179 has no "
           "spike-ins, so this section has no 1179 rows.", ""]
    data, notes = {}, []
    for lib in NUCLEI:
        p = base / lib / "spikein" / "truth" / "molecules_labeled.parquet"
        if p.exists():
            data[lib] = pd.read_parquet(p, columns=["label", "spike_label", "module"])
        else:
            notes.append(f"{lib}: {_not_run(base, [p])}")
    if data:
        def counts(m):
            return pd.crosstab(m.spike_label, m.label).reindex(index=list(SPIKE_LABELS), columns=list(LABELS), fill_value=0)

        def agreement(c):
            hit = c.loc["internal", "templated"] + c.loc["tail", "nontemplated"]
            n = c.loc[["internal", "tail"], ["templated", "nontemplated"]].to_numpy().sum()
            return f"{_r(hit / n if n else None)} ({_int(hit)} of {_int(n)})"

        rows, agree = [], []
        for lib, m in data.items():
            c = counts(m)
            rows += [[lib, k, _int(t.templated), _int(t.ambiguous), _int(t.nontemplated), _int(t.sum())] for k, t in c.iterrows()]
            agree.append(f"{lib}: agreement {agreement(c)}, SIRV_E0 only {agreement(counts(m[m.module == 'SIRV_E0']))}")
        out += _table(["Library", "Spike label", "templated", "ambiguous", "non-templated", "total"], rows)
        out += ["", "Agreement = (internal and templated + tail and non-templated) / internal or tail molecules labeled "
                    "templated or non-templated: " + "; ".join(agree) + ".", "",
                "- ERCC and long SIRV transcripts end at the contig end, so at their tail sites the 20 nt after the site are "
                f"truncated and a20 cannot reach {LABEL_CUTOFFS['templated_min']}: their non-templated (or ambiguous) calls are "
                "forced, and they lift the pooled agreement. Only SIRV_E0 tails have full windows.",
                "- A SIRV_E0 tail site can also be internal priming of a longer isoform that runs through a shorter isoform's "
                "annotated end; the annotation cannot separate the two.", ""]
        rows = []
        for lib, m in data.items():
            for module in ("all",) + MODULES:
                g = m if module == "all" else m[m.module == module]
                i, t = g[g.spike_label == "internal"], g[g.spike_label == "tail"]
                rows.append([lib, module, _int(len(i)), _share((i.label == "templated").sum(), len(i)), _int(len(t)),
                             _share((t.label == "nontemplated").sum(), len(t))])
        out += _table(["Library", "Module", "Internal", "Internal and templated", "Tail", "Tail and non-templated"], rows)
    out = _notes(out, notes)

    out += ["", f"Metrics per module (positives: internal, negatives: tail; molecule level, cap {MEMBER_CAP:,} nt), "
                "sens / spec:", ""]
    p = base / "eval" / "spikein_metrics.parquet"
    s = _parquet(p)
    if s is None:
        out[-2:] = [f"Metrics per module: {_not_run(base, [p])}"]
        return "\n".join(out)
    tags, note = _report_tags(base)
    tags = tags + CONTROL_TAGS
    out += note
    s = s[s.cap == MEMBER_CAP]

    def table(level):
        look = {(r.lib, r.tag, r.stratum): r for r in s[s.level == level].itertuples()}
        rows = []
        for lib in dict.fromkeys(s.lib):
            for tag in tags:
                if (lib, tag, "all") in look:
                    rows.append([lib, tag] + [f"{_r(look[lib, tag, k].sens)} / {_r(look[lib, tag, k].spec)}"
                                              if (lib, tag, k) in look else "n/a" for k in ("all",) + MODULES])
        return _table(["Library", "Setting", "all"] + list(MODULES), rows)

    out += table("molecule")
    out += ["", f"Metrics per module at the read level (a read counts as a unit; same file, cap {MEMBER_CAP:,} nt), sens / spec:", ""]
    out += table("read")
    rows = []
    for lib in dict.fromkeys(s.lib):
        ctl = s[(s.lib == lib) & (s.stratum == "all") & s.tag.isin(CONTROL_TAGS)]
        mol = ctl[(ctl.level == "molecule") & ctl.spec.notna()]
        if mol.empty:
            continue
        r = mol.loc[mol.spec.idxmin()]
        rd = ctl[(ctl.level == "read") & (ctl.tag == r.tag)].spec
        rows.append([lib, r.tag, f"{_int(r.fp)} of {_int(r.tn + r.fp)}", _r(r.spec), _r(rd.iloc[0] if len(rd) else None)])
    out += ["", "Remove-all controls on spike-in tail molecules, our observation in this data (stratum all; per library the "
                "control with the lowest molecule-level specificity, and the read level at the same setting):", ""]
    out += _table(["Library", "Setting", "Tail molecules removed (fp of tn + fp)", "Spec, molecule level", "Spec, read level"],
                  rows) if rows else ["not run (no remove-all control rows in eval/spikein_metrics.parquet)"]
    return "\n".join(out)


def _flat(d):
    """Scalar values of a nested dict, keyed by their own name (the first occurrence wins)."""
    out = {}
    for k, v in d.items():
        for kk, vv in (_flat(v).items() if isinstance(v, dict) else [(k, v)]):
            out.setdefault(kk, vv)
    return out


def ercc_section(base):
    base = Path(base)
    out = ["## 8. ERCC count accuracy (provisional)", "",
           "Source: `<lib>/spikein/ercc.json` (stage 15): correlation of ERCC UMI counts with input concentration, "
           "unfiltered and per setting. 1179 has no spike-ins, so this section has no 1179 rows.", ""]
    entries, notes, flags, recorded, barcodes = [], [], [], {}, {}
    for lib in NUCLEI:
        p = base / lib / "spikein" / "ercc.json"
        d = _json(p)
        if d is None:
            notes.append(f"{lib}: {_not_run(base, [p])}")
            continue
        if d.get("provisional"):
            flags.append(f"Provisional ({lib}): {d.get('note', '')}")
        recorded[lib] = d.get("selection")
        barcodes[lib] = d.get("barcodes")
        src = d.get("settings", d)
        entries += [(lib, k, _flat(v)) for k, v in src.items() if isinstance(v, dict)]
    out += [f"- {f}" for f in flags] + ([""] if flags else [])
    sel = _json(base / "eval" / "selection.json")
    now = None if sel is None else {m: d["selection"]["tag"] for m, d in sel["modes"].items()}
    for lib, rec in recorded.items():
        if now is not None and rec is not None and rec != now:
            show = lambda x: ", ".join(f"{MODE_NAMES[m]} {t or 'none'}" for m, t in x.items())  # noqa: E731
            notes.append(f"{lib}: ercc.json was written with the chosen settings {show(rec)}; eval/selection.json now chooses "
                         f"{show(now)}, so stage 15 predates the current selection and the table lacks the chosen setting.")
    if entries:
        out += ["Columns: n: ERCCs with a UMI unfiltered and a Mix 1 concentration; pearson_log10: Pearson correlation of "
                "log10(UMIs + 1) with log10(concentration); spearman: Spearman correlation of the UMIs with the concentration; "
                "umis: UMIs left after the setting; umis_kept_share: umis over the unfiltered umis. UMIs are counted over primary "
                "records with a UB and a CB of the barcode set recorded in ercc.json (stage 15): "
                + "; ".join(f"{lib} {POP_NAMES[b] if b in POP_NAMES else b}" if b else f"{lib} not recorded"
                            for lib, b in barcodes.items()) + ". Section 7 counts every barcode.", ""]
    if entries:
        cols = list(dict.fromkeys(k for _, _, f in entries for k in f))
        out += _table(["Library", "Setting"] + cols,
                      [[lib, k] + [_value(f[c]) if c in f else "n/a" for c in cols] for lib, k, f in entries])
    out = _notes(out, notes)
    return "\n".join(out)


def _per_read(sites, cutoffs):
    """(templated rate, non-templated rate, ratio) of the hexamer weighted by accepted reads, at the given cut-offs."""
    s = sites[sites.n_accepted >= 1]
    w = s.n_accepted.to_numpy(float)
    rate = []
    for m in (s.a20 >= cutoffs["templated_min"], s.a20 <= cutoffs["nontemplated_max"]):
        m = m.to_numpy()
        rate.append((w[m] * s.hexamer.to_numpy()[m]).sum() / w[m].sum() if w[m].sum() else None)
    return rate[0], rate[1], rate[1] / rate[0] if rate[0] and rate[1] is not None else None


def long_read_section(base):
    base = Path(base)
    out = ["## 9. Long reads (BD67, BD70)", "",
           "Source: `<lib>/long_read_acceptance.json` and `<lib>/long_read_sites.parquet` (stage 16). Hexamer rates per unique "
           "site with an accepted read (binding, spec 4.3: passes at a per-site ratio >= 3) and per accepted read. Outcomes of the "
           f"used reads: Accepted, a terminal A run of at least {_READ['min_run'].default} nt followed by at least "
           f"{_READ['min_after'].default} nt of clip; Short run, the run is shorter than {_READ['min_run'].default} nt; Short "
           f"after, the run is long enough but fewer than {_READ['min_after'].default} nt of clip follow; Both, neither.", ""]
    data, notes = {}, []
    for lib in LONG_READS:
        p = base / lib / "long_read_acceptance.json"
        d = _json(p)
        if d is None:
            notes.append(f"{lib}: {_not_run(base, [p])}")
        else:
            data[lib] = (d, _parquet(base / lib / "long_read_sites.parquet", ["n_accepted", "a20", "hexamer", "label", "region"]))
    if not data:
        return "\n".join(_notes(out, notes))
    rows = []
    for lib, (d, _) in data.items():
        rec, oc = d["records"], d["outcomes"]
        rows.append([lib, _int(len(d["refs"])), _int(rec["records"]), _int(rec["secondary_supplementary"]), _int(rec["low_mapq"]),
                     _int(rec["used"]), _int(oc["accepted"]), _int(oc["short_run"]), _int(oc["short_after"]), _int(oc["both"]),
                     _pct(d["dropped_fraction"], 1), _int(d["duplicate_qnames"])])
    out += _table(["Library", "References", "Records", "Secondary or supplementary", "MAPQ < 20", "Used", "Accepted", "Short run",
                   "Short after", "Both", "Dropped", "Duplicate qnames"], rows)
    (first, (d0, _)), *rest = data.items()
    drops = f"{first} dropped {_pct(d0['dropped_fraction'], 1)} of the reads used" + "".join(
        f" and {lib} {_pct(d['dropped_fraction'], 1)}" for lib, (d, _) in rest)
    out += ["", f"- The spec expects about 7% of reads dropped. In this run {drops} (`dropped_fraction` in each acceptance file, "
                "over the references stage 16 processed)."]
    out += ["", "Sites with an accepted read by region and label "
                "(" + ", ".join(f"`{lib}/long_read_sites.parquet`" for lib, (_, sites) in data.items() if sites is not None) +
                ", unique sites):", ""]
    rows = []
    for lib, (_, sites) in data.items():
        if sites is None:
            continue
        c = pd.crosstab(sites[sites.n_accepted >= 1].region, sites[sites.n_accepted >= 1].label)
        c = c.reindex(index=list(REGION_NAMES), columns=list(LABELS), fill_value=0)
        rows += [[lib, region, _int(t.templated), _int(t.ambiguous), _int(t.nontemplated), _int(t.sum())] for region, t in c.iterrows()]
    out += _table(["Library", "Region", "templated", "ambiguous", "non-templated", "total"], rows) if rows else \
        ["not run (missing long_read_sites.parquet)"]
    out += ["", "Hexamer rates (sites T / NT / amb counted per unique site with an accepted read):", ""]
    rows = []
    for lib, (d, sites) in data.items():
        for key in ("cutoffs_8_4", "cutoffs_14_8"):
            h = d[key]
            c = h["cutoffs"]
            t, nt, ratio = _per_read(sites, c) if sites is not None else (None, None, None)
            rows.append([lib, f"{c['templated_min']}/{c['nontemplated_max']}",
                         " / ".join(_int(h["n"][k]) for k in ("templated", "nontemplated", "ambiguous")),
                         _r(h["hexamer_rate"]["templated"]), _r(h["hexamer_rate"]["nontemplated"]),
                         _r(h["hexamer_rate"]["ambiguous"]), _r(h["ratio"], 2), h["passes"], _r(t), _r(nt), _r(ratio, 2)])
    out += _table(["Library", "Cut-offs", "Sites T / NT / amb", "Rate T", "Rate NT", "Rate amb", "Ratio per site",
                   "Passes (per-site ratio >= 3)", "Per-read rate T", "Per-read rate NT", "Ratio per read"], rows)
    out += ["", "Agreement with 1179 (1179 molecules on the references stage 16 processed; a long-read site within 10 nt on "
                "the same strand). Agreement T / NT / pooled: among 1179 molecules matched to a long-read site, the share with the "
                "same label (T: of the matched templated molecules, NT: of the matched non-templated, pooled: of the matched "
                "templated or non-templated). Agreement within 10 nt may mostly measure co-location (not tested here):", ""]
    rows = []
    for lib, (d, _) in data.items():
        v = d.get("vs_1179")
        if v is None:
            notes.append(f"{lib} vs 1179: not run (long_read_acceptance.json has no vs_1179)")
            continue
        by, ag = v.get("share_matched_by_label", {}), v["agreement"]
        rows.append([lib, _int(v["molecules"]), _r(v["share_matched"]),
                     " / ".join(_r(by.get(k)) for k in ("templated", "nontemplated", "ambiguous")),
                     " / ".join(_r(ag[k]) for k in ("templated", "nontemplated", "pooled"))])
    if rows:
        out += _table(["Library", "1179 molecules", "Matched share", "Matched share T / NT / amb", "Agreement T / NT / pooled"], rows)
    return "\n".join(_notes(out, notes))


def _a_content_line(base):
    """Our observation of the A content downstream of the atlas sites (stage 22), or the not-run line."""
    p = base / "burden" / "polyasite_a_content.json"
    d = _json(p)
    if d is None:
        return f"A content downstream of the atlas sites: {_not_run(base, [p])}"
    t, w, k, shift = d["total"], d["window_nt"], d["min_a"], d["null_shift_nt"]
    real, null = t["n_a14"] / t["n_full"] if t["n_full"] else None, t["n_a14_null"] / t["n_full_null"] if t["n_full_null"] else None
    kind = "n/a" if real is None or null is None else "is not depleted of" if real >= null else "is depleted of"
    return (f"A content downstream of the atlas sites: our observation (`{_rel(base, p)}`, stage 22, {len(d['refs'])} of "
            f"{len(d['refs']) + len(d['atlas_refs_not_counted'])} atlas references): {_int(t['n_a14'])} of {_int(t['n_full'])} atlas "
            f"sites with a full {w} nt downstream window ({100 * t['n_a14'] / t['n_full']:.2f}%) have {k} or more A's in it, against "
            f"{_int(t['n_a14_null'])} of {_int(t['n_full_null'])} ({100 * t['n_a14_null'] / t['n_full_null']:.2f}%) for the same sites "
            f"moved {shift:,} nt upstream; the clusters file {kind} A-rich downstream sequence by this count.")


def _re_counts_lines(base):
    """Primary records and records without the duplicate flag per RE tag and library, from stage 13's region_counts.parquet."""
    paths = [base / lib / "grid" / "region_counts.parquet" for lib in LIBS]
    absent = [p for p in paths if not p.exists()]
    if len(absent) == len(paths):
        return [f"Primary records by RE tag: {_not_run(base, absent)}"]
    out = ["Primary records by RE tag, our observation in these BAMs (`<lib>/grid/region_counts.parquet`, stage 13; the counts of one "
           "tag, the same for every tag; a record is without the duplicate flag when SAM flag 0x400 is not set; no explanation "
           "is attempted):", ""]
    rows, notes = [], []
    for lib, p in zip(LIBS, paths):
        if not p.exists():
            notes.append(f"{lib}: {_not_run(base, [p])}")
            continue
        rc = pd.read_parquet(p)
        has = "total_nondup" in rc
        if not has:
            notes.append(f"{_rel(base, p)} has no total_nondup column: stage 13 predates it, so the records without the "
                         "duplicate flag are not counted")
        one = rc[rc.tag == rc.tag.iloc[0]].groupby("re")[["total"] + (["total_nondup"] if has else [])].sum()
        rows += [[lib, re or "none", _int(r.total), _int(r.total_nondup) if has else "n/a"] for re, r in one.iterrows()]
    out += _table(["Library", "RE", "Primary records", "Primary records without the duplicate flag"], rows)
    return _notes(out, notes)


def _burden_population(base):
    """Lines naming the population of the burden tables (burden/summary.json "population", written by stage 18) and the
    groups of libraries."""
    summary = _json(base / "burden" / "summary.json")
    lines = []
    if summary is None:
        lines.append(f"Population of the burden tables: {_not_run(base, [base / 'burden' / 'summary.json'])}")
    elif "population" not in summary:
        lines.append("Population of the burden tables: not recorded (stage 18 predates it).")
    else:
        lines.append(f"Population of the burden tables (burden/summary.json): {POP_NAMES[summary['population']]}.")
    if summary is None:
        lines.append(f"Groups: {_not_run(base, [base / 'burden' / 'summary.json'])}")
    else:
        g = summary["groups"]
        grp = {k: v for k, v in g.items() if k in ("nuclei", "cells")}
        same = [f"{k} = {v[0]}" for k, v in grp.items() if len(v) == 1]
        lines.append("Groups (burden/summary.json): " + "; ".join(f"{k} = {' + '.join(v)}" for k, v in grp.items()) + "." + (
            " A group with one library holds the same molecules as that library: " + ", ".join(same) + "." if same else ""))
    return lines


def burden_section(base):
    base = Path(base)
    out = ["## 10. Burden: templated share of labeled molecules", ""]
    p = base / "burden" / "templated_share.parquet"
    share = _parquet(p)
    if share is None:
        return "\n".join(out + [_not_run(base, [p])])
    out += ["Source: `burden/*.parquet` (stage 18). Descriptive only: tissue, maturity, cell-type mix and the cell/nucleus "
            "difference are confounded. share = templated / all labeled molecules, 95% gene-bootstrap interval in brackets, "
            "molecules in parentheses; resolved = templated / (templated + non-templated).", ""]
    groups = list(dict.fromkeys(share.group))
    pop_lines = _burden_population(base)
    out += [x for line in pop_lines for x in (line, "")]
    look = {(r.group, r.stratum_kind, r.stratum): r for r in share.itertuples()}
    strata = [("all", "all")] + [("region", g) for g in REGION_NAMES] + [("has_ts", "True"), ("has_ts", "False")]

    def cell(g, k, s, f):
        r = look.get((g, k, s))
        return "n/a" if r is None else f(r)

    out += _table(["Stratum kind", "Stratum"] + groups,
                  [[k, s] + [cell(g, k, s, lambda r: f"{_ci(r.share_all, r.lo, r.hi)} ({_int(r.total)})") for g in groups]
                   for k, s in strata])
    out += ["", "Resolved share:", ""]
    out += _table(["Stratum kind", "Stratum"] + groups,
                  [[k, s] + [cell(g, k, s, lambda r: _r(r.share_resolved)) for g in groups] for k, s in strata])
    ps = base / "eval" / "selection.json"
    sel = _json(ps)
    out += ["", "Templated share by label and corrected (`eval/selection.json` label_purity, stage 17; labeled molecules of "
                f"{POP_NAMES[(sel or {}).get('selection_population', 'all')]}). The corrected share removes the estimated tail-primed part of the templated class, an estimate "
                "from the hexamer rates (see the label purity table in section 2):", ""]
    if sel is None:
        out[-2:] = [f"Templated share corrected: {_not_run(base, [ps])}"]
    elif "label_purity" not in sel:
        out[-2:] = ["Templated share corrected: not run (eval/selection.json has no label_purity)"]
    else:
        out += _table(["Library", "Labeled molecules", "Templated share by label", "Templated share, corrected"],
                      [[lib, _int(d["n_labeled"]), _r(d["templated_share"]), _r(d["templated_share_corrected"])]
                       for lib, d in sel["label_purity"].items()])
    out += ["", "Distribution over A bands (share of labeled molecules in the band, 95% gene-bootstrap interval in brackets; on "
                "these rows stage 18's lo and hi bound this share, not share_all, because the labels are defined by the band "
                "and fix share_all within it):", ""]
    out += _table(["A band"] + groups, [[b] + [cell(g, "a_band", b, lambda r: _ci(r.frac, r.lo, r.hi)) for g in groups]
                                        for b in BANDS])

    no_pair = "(stage 18 needs 1417 and 1419, or nuclei and cells)"
    p = base / "burden" / "same_site.parquet"
    ss = _parquet(p)
    if ss is None:
        out += ["", f"Same site: {_not_run(base, [p])}"]
    elif ss.empty:
        out += ["", f"Same site: no pair in {_rel(base, p)} {no_pair}"]
    else:
        out += ["", "Same site (each site of the first group matched to the nearest site of the second within 5 nt on the same "
                    "strand; site label = most common label of its molecules):", ""]
        c = ss.pivot_table(index=["pair", "label_a"], columns="label_b", values="n", aggfunc="sum", fill_value=0)
        rows = pd.MultiIndex.from_tuples([(p, k) for p in dict.fromkeys(ss.pair) for k in LABELS if (p, k) in c.index])
        c = c.reindex(index=rows, columns=list(LABELS) + ["none"], fill_value=0)
        out += _table(["Pair", "Label in first", "templated", "ambiguous", "non-templated", "none"],
                      [[pair, LABEL_NAMES.get(la, la)] + [_int(v) for v in t] for (pair, la), t in c.iterrows()])
        rows = []
        for pair, g in ss.groupby("pair", sort=False):
            n, matched = g.n.sum(), g[g.label_b != "none"]
            same = matched[matched.label_a == matched.label_b].n.sum()
            rows.append([pair, _int(n), f"{_int(matched.n.sum())} ({_pct(matched.n.sum(), n)})",
                         f"{_int(same)} ({_pct(same, matched.n.sum())})"])
        out += [""] + _table(["Pair", "Sites in first", "Matched", "Same label (of matched)"], rows)
    p = base / "burden" / "same_gene.parquet"
    sg = _parquet(p)
    if sg is None:
        out += ["", f"Same gene: {_not_run(base, [p])}"]
    elif sg.empty:
        out += ["", f"Same gene: no pair in {_rel(base, p)} {no_pair}"]
    else:
        out += ["", "Same gene (Spearman correlation of the templated share over genes with enough molecules in both):", ""]
        out += _table(["Pair", "Genes", "Spearman rho", "p"],
                      [[r.pair, _int(r.n_genes), _r(r.spearman_rho), "n/a" if pd.isna(r.p_value) else f"{r.p_value:.2e}"]
                       for r in sg.itertuples()])
    p = base / "burden" / "selection_bias.parquet"
    sb = _parquet(p)
    if sb is None:
        out += ["", f"Selection bias: {_not_run(base, [p])}"]
    else:
        has = "records_nondup_share" in sb
        out += ["", "Selection bias: share of labeled molecules versus share of primary records, by RE tag (RE is the region type "
                    f"of the alignment (E = exonic, N = intronic, I = intergenic; Cell Ranger documentation, Barcoded BAM Tags, "
                    f"{BAM_TAGS_URL}). records_share is the share among all primary records; the duplicate-free share is the "
                    "share among primary records without the PCR-duplicate flag. Both shares cover all barcodes: the records are those of every "
                    "barcode, so stage 18 takes the labeled molecules of all barcodes whatever its population. The check is not "
                    "repeated within called cells because stage 13 does not split the records by barcode (left for milestone 3, "
                    "which builds per-cell counts):", ""]
        out += _table(["Library", "RE", "Labeled share", "Records share"] + (["Records share without PCR duplicates"] if has else []),
                      [[r.lib, r.re, _r(r.labeled_share), _r(r.records_share)] + ([_r(r.records_nondup_share)] if has else [])
                       for r in sb.itertuples()])
        if not has:
            out += ["", "burden/selection_bias.parquet has no records_nondup_share column: stage 13 predates it, so the spec 11 "
                        "rule below cannot be applied."]
        else:
            off = [r for r in sb.itertuples() if abs(r.labeled_share - r.records_nondup_share) > 0.05]
            out += ["", "Spec 11 says that where labeled and unlabeled molecules differ strongly, conclusions are stated for the "
                        "labeled population only; the threshold used here, a difference of more than 5 percentage points, is our "
                        "choice (the milestone 1 region check uses the same value). "]
            out[-1] += ("Pairs with a difference of more than 5 percentage points: " + "; ".join(
                f"{r.lib} {r.re} (labeled {_r(r.labeled_share)}, without duplicates {_r(r.records_nondup_share)}, "
                f"{100 * abs(r.labeled_share - r.records_nondup_share):.1f} points)" for r in off) + "." if off else
                        "No library and region pair differs by more than 5 percentage points.")
    out += [""] + _re_counts_lines(base)
    p = base / "burden" / "polyasite.parquet"
    px = _parquet(p)
    if px is None:
        out += ["", f"PolyASite proximity: {_not_run(base, [p])}"]
    else:
        out += ["", "PolyASite 3.0 proximity (share of labeled molecules by the distance in nt from the site to the nearest atlas "
                    "cleavage site on the same strand; templated molecules also by A band). Proximity is reported, not used to "
                    "validate the labels: the atlas is built from 10x Genomics scRNA-seq data (PolyASite 3.0, "
                    "https://polyasite.unibas.ch/atlas_sc), so internally primed sites can be in it. The v3.0 documentation describes no filter on A "
                    "content downstream of a site. The null row takes every molecule of the group with its site moved 1,000 nt "
                    "upstream in transcript sense on the same ref and strand; its share within 10 nt is the local background at "
                    "that offset, not a uniform chance level."]
        out += ["", _a_content_line(base)]
        for g, d in px.groupby("group", sort=False):
            n = d.groupby(["label", "a_band", "bin"]).n.sum()
            keys = ([(lab, "all") for lab in LABELS] + [(NULL_LABEL, "all")]
                    + [("templated", b) for b in BANDS if ("templated", b) in n.index])
            rows = []
            for lab, band in keys:
                c = n.loc[(lab, band)].reindex(list(BINS), fill_value=0)
                name = ("null (sites 1 kb upstream)" if lab == NULL_LABEL else LABEL_NAMES[lab] if band == "all"
                        else f"templated, A band {band}")
                rows.append([name, _int(c.sum())] + [_r(x / c.sum() if c.sum() else None) for x in c])
            out += ["", f"Group {g}:", ""] + _table(["Label", "Molecules"] + list(BINS), rows)
    return "\n".join(out)


def validation_section(base):
    base = Path(base)
    out = ["## 11. Validation against the original tool", ""]
    p = base / "validation" / "exact_match_m2.json"
    em = _json(p)
    if em is None:
        return "\n".join(out + [_not_run(base, [p])])
    out += ["Source: `validation/exact_match_m2.json` (stage 20). Removed-record sets (every record the original fetches, by our reading of its code, spec 5: "
            "secondary and supplementary included) of the original polyAfilter and the reimplementation. Max RSS is the "
            "largest single process; peak tree RSS the memwatch sum.", ""]
    rows, notes = [], []
    for lib in LIBS:
        d, sub = em.get(lib), em.get(f"{lib}_subset")
        if d is None and sub is None:
            notes.append(f"{lib}: not in exact_match_m2.json" + ("; the 1179 grid has no original-versus-reimplementation "
                                                                 "evidence" if lib in CELLS else ""))
        elif d is not None and not d:
            notes.append(f"{lib}: no completed original run")
        if sub is not None:
            notes.append(f"{lib}: original tool on the {', '.join(sub['refs'])} subset only ({_int(sub['n_records'])} records)")
        runs = [(lib, d or {})] + ([(f"{lib} ({', '.join(sub['refs'])})", sub["settings"])] if sub is not None else [])
        for name, by_tag in runs:
            for tag, r in by_tag.items():
                rows.append([name, tag, _int(r["n_original"]), _int(r["n_reimpl"]), _int(r["only_original"]),
                             _int(r["only_reimpl"]), r["identical"],
                             _r(r["original_elapsed_s"] / 3600 if r["original_elapsed_s"] is not None else None, 2),
                             _r(r["original_max_rss_gib"], 2), _r(r["original_peak_tree_rss_gib"], 2), _int(r.get("n_grid")),
                             "n/a" if r.get("grid_equal") is None else r["grid_equal"]])
    if rows:
        out += _table(["Library", "Setting", "Original removed", "Reimpl removed", "Only original", "Only reimpl", "Identical",
                       "Original wall (h)", "Max RSS (GiB)", "Peak tree RSS (GiB)", "Stage 13 removed", "Equal to stage 13"], rows)
    out = _notes(out, notes)
    nc = em.get("none_control")
    out += ["", "Remove-all control (MINSNRLEN none) on a reference subset, original run in process:", ""]
    if nc is None:
        out[-2:] = [f"Remove-all control: not run (no none_control entry in {_rel(base, p)})"]
    else:
        ct = nc["createTRANS"]
        out += _table(["Setting", "References", "Records", "Original removed", "Reimpl removed", "Only original", "Only reimpl",
                       "Identical", "createTRANS wall (s)", "createTRANS max RSS (GiB)", "Total (s)"],
                      [[nc["setting"], ", ".join(nc["refs"]), _int(nc["n_records"]), _int(nc["n_original"]), _int(nc["n_reimpl"]),
                        _int(nc["only_original"]), _int(nc["only_reimpl"]), nc["identical"], _sec(ct["elapsed_s"]),
                        _r(ct["max_rss_gib"], 2), _sec(sum(nc["seconds"].values()))]])
    out += ["", f"Compared: {em.get('n_compared', 'n/a')}; all identical: {em.get('all_identical', 'n/a')}"]
    if "complete" not in em:
        out += ["", "completeness not recorded (file written before 2026-10-10)"]
        return "\n".join(out)
    out += ["", f"complete: {'yes' if em['complete'] else 'no'}"]
    miss = [[lib, m["tag"], "n/a" if m["exit_status"] is None else m["exit_status"], m["reason"]]
            for lib, ms in em.get("missing", {}).items() for m in ms]
    if miss:
        out += ["", "Original runs that are missing (expected for every setting of the library):", ""]
        out += _table(["Library", "Setting", "Exit status", "Reason"], miss)
    tie = em.get("engine_tie", {})
    if tie:
        out += ["", "Engine tie: stage 13 truth flags against the stage 03 flags of the same primary records (a difference is "
                    "counted per record and setting):", ""]
        out += _table(["Library", "Records compared", "Differences (records x settings)", "Missing"],
                      [[lib, "not computed", "n/a", "n/a"] if t is None else
                       [lib, _int(t["rows"]), _int(sum(t["mismatches"].values())), _int(t["missing_rows"])]
                       for lib, t in tie.items()])
    if em.get("reused"):
        out += ["", "Stages 03 and 04 reused: " + ", ".join(f"{lib} {'yes' if v else 'no'}" for lib, v in em["reused"].items())]
    ok = em.get("all_identical") and em["complete"]
    out += ["", "The reimplementation is validated only when all_identical and complete are both true: " +
            ("both are true, so the reimplementation is validated on these runs." if ok else "not validated on these runs.")]
    return "\n".join(out)


SECTIONS = (runs_section, truth_section, grid_section, selection_section, strata_section, hexamer_section, spikein_section,
            ercc_section, long_read_section, burden_section, validation_section)


def headline(base):
    """Chosen setting per mode with sensitivity and specificity and their 95% interval on each library, the population and
    units, and the validation outcome; from eval/selection.json, eval/bootstrap.parquet and validation/exact_match_m2.json."""
    base = Path(base)
    out = ["## Headline", ""]
    p = base / "eval" / "selection.json"
    sel = _json(p)
    if sel is None:
        out.append(f"Headline: {_not_run(base, [p])}")
    else:
        libs, min_spec = sel["libraries"], sel["min_spec"]
        out += [f"{_pop_text(sel)[:-1]}; member cap {sel['cap']:,} nt, {sel['cutoffs']} labels. Rule: {sel['rule']}", ""]
        bp = base / "eval" / "bootstrap.parquet"
        boot = _parquet(bp)
        ci = {}
        if boot is not None:
            b = boot[(boot.level == "molecule") & (boot.cap == sel["cap"])]
            ci = {(r.lib, r.tag, r.metric): (r.lo, r.hi) for r in b.itertuples()}
        rows = []
        for mode, d in sel["modes"].items():
            s = d["selection"]
            q = f"{s['n_qualifying']} / {s['n_candidates']}"
            if s["tag"] is None:
                none = "no candidates" if s["n_candidates"] == 0 else f"none reached specificity {min_spec:g}"
                rows.append([MODE_NAMES[mode], none, "n/a", "n/a", "n/a", q] + ["n/a"] * 5)
                continue
            t = parse_tag(s["tag"])
            for lib in libs:
                m = d["metrics"].get(lib, {}).get("molecule")
                rows.append([MODE_NAMES[mode], s["tag"], t.covlen, t.min_len, t.mism, q, lib] + (["n/a"] * 4 if not m else [
                    _int(m["tp"] + m["fn"]), _int(m["tn"] + m["fp"])] + [_ci(m[k], *ci.get((lib, s["tag"], k), (None, None)))
                                                                         for k in ("sens", "spec")]))
        out += _table(["Mode", "Setting", "COVLEN", "MINSNRLEN", "MISM", "Qualifying / candidates", "Library", "Templated",
                       "Non-templated", "Sens (95% interval)", "Spec (95% interval)"], rows)
        if boot is None:
            out += ["", f"Bootstrap: {_not_run(base, [bp])}"]
    out += [""]
    p = base / "validation" / "exact_match_m2.json"
    em = _json(p)
    if em is None:
        out.append(f"Validation against the original tool: {_not_run(base, [p])}")
    elif "complete" not in em:
        out.append(f"Validation against the original tool (section 11): compared {em.get('n_compared', 'n/a')}, all identical "
                   f"{em.get('all_identical', 'n/a')}, completeness not recorded.")
    else:
        ok = em.get("all_identical") and em["complete"]
        out.append(f"Validation against the original tool (section 11): compared {em.get('n_compared', 'n/a')}, all identical "
                   f"{em.get('all_identical', 'n/a')}, complete {'yes' if em['complete'] else 'no'}: "
                   f"{'validated' if ok else 'not validated'} on these runs.")
    return "\n".join(out)


def _missing(sections):
    """The header line on missing inputs: places per section and the files, from the rendered sections."""
    per, files = {}, []
    for name, text in sections:
        n = text.count("not run")
        if n:
            per[name] = n
        files += [f for m in re.findall(r"not run \(missing ([^)]*)\)", text) for f in m.split(", ") if f not in files]
    if not per:
        return "Missing inputs: none."
    return (f"Missing inputs: {sum(per.values())} places ({'; '.join(f'{k}: {v}' for k, v in per.items())}). Files: "
            + ", ".join(files) + ".")


def render(base):
    texts = [("headline", headline(base))] + [(f"section {i}", f(base)) for i, f in enumerate(SECTIONS, 1)]
    head = ["# Milestone 2 results (generated by scripts/21_m2_summary.py)", "",
            f"Inputs under `{base}`. Each section names the files its numbers come from; \"not run\" lists the files that "
            "are absent.", "",
            "Setting tags read c<COVLEN>_l<MINSNRLEN>_m<MISM>_<e or i>: COVLEN, MINSNRLEN and MISM are polyAfilter's parameters "
            "(spec 3); e is the default mode and i the -i mode (introns); l none is the remove-all control.", "",
            _missing(texts), ""]
    return "\n".join(head) + "\n" + "\n\n".join(t for _, t in texts) + "\n"
