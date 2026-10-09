"""Stage 21: the milestone 2 results page. One function per section; each reads its files under a base directory (OUT) and
prints "not run (missing ...)" with the absent file names instead of numbers."""
import json
from pathlib import Path

import pandas as pd

from .config import A_BANDS, CELLS, LABEL_CUTOFFS, LONG_READS, MEMBER_CAP, NUCLEI, SENS_CAPS
from .evaluate import MODULES
from .flags import Setting, setting_tag
from .grid import COVLENS, published
from .polyasite import BINS, NULL_LABEL
from .regions import REGION_NAMES
from .validation import run_stats

LIBS = NUCLEI + CELLS
LABELS = ("templated", "ambiguous", "nontemplated")
LABEL_NAMES = {"templated": "templated", "ambiguous": "ambiguous", "nontemplated": "non-templated", "all": "all"}
MODE_NAMES = {"e": "default", "i": "-i"}
BANDS = tuple(b[0] for b in A_BANDS)
SPIKE_LABELS = ("internal", "tail", "other")
CUTOFFS = "{templated_min}/{nontemplated_max}".format(**LABEL_CUTOFFS)
PUBLISHED_TAGS = [setting_tag(s) for s in published()]
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
           "(`validation/subset_1179/`). Peak RSS is summed over the process tree, so pages shared by "
           "forked workers count once per process (an upper bound). memwatch samples every 2 s (interval_s), so the peak of a "
           "shorter run reads near zero.", ""]
    mem = [p for p in base.rglob("*.mem.json") if "original" not in p.relative_to(base).parts[:-1]]
    rows = []
    for p in mem:
        d = json.loads(p.read_text())
        rows.append([_command(d["cmd"]), _rel(base, p), _sec(d["elapsed_s"]), _r(d["peak_tree_rss_gib"], 2), d["exit_status"]])
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


def truth_section(base):
    base = Path(base)
    out = ["## 2. Truth sets", "",
           "Source: `<lib>/truth/molecules_labeled.parquet` and `<lib>/truth/truth_reads.parquet` (stage 12). "
           f"Labels at {CUTOFFS} A's in the 20 nt after the site; members are selected member reads (gene match, "
           "dist3_exon >= -25) by exon-aware distance.", ""]
    data, notes = {}, []
    for lib in LIBS:
        paths = [base / lib / "truth" / "molecules_labeled.parquet", base / lib / "truth" / "truth_reads.parquet"]
        absent = [p for p in paths if not p.exists()]
        if absent:
            notes.append(f"{lib}: {_not_run(base, absent)}")
            continue
        data[lib] = (pd.read_parquet(paths[0], columns=["label", "region", "has_ts", "a20", "hexamer", "genes", "n_members"]),
                     pd.read_parquet(paths[1], columns=["role", "dist5_exon"]))
    if data:
        def row(name, f):
            return [name] + [f(m, r) for m, r in data.values()]

        def hexa(s):
            return f"{_r(s.mean() if len(s) else None)} (n {_int(len(s))})"

        rows = [row("Molecules", lambda m, r: _int(len(m)))]
        rows += [row(LABEL_NAMES[k], lambda m, r, k=k: f"{_int((m.label == k).sum())} ({_pct((m.label == k).sum(), len(m))})")
                 for k in LABELS]
        rows += [row("With a ts read (has_ts)", lambda m, r: _int(m.has_ts.sum())),
                 row("Multi-gene (gene = alphabetically first id)", lambda m, r: _int(m.genes.str.contains(";").sum())),
                 row("pa reads", lambda m, r: _int((r.role == "pa").sum()))]
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
        for lib, (m, _) in data.items():
            c = pd.crosstab(m.region, m.label).reindex(index=list(REGION_NAMES), columns=list(LABELS), fill_value=0)
            for region, t in c.iterrows():
                resolved = t.templated + t.nontemplated
                rows.append([lib, region, _int(t.templated), _int(t.ambiguous), _int(t.nontemplated), _int(t.sum()),
                             _r(t.templated / resolved if resolved else None)])
        out += _table(["Library", "Region", "templated", "ambiguous", "non-templated", "total",
                       "templated / (templated + non-templated)"], rows)

    out += ["", "Ambiguous molecules: hexamer-based estimate of their tail-primed share over all labeled molecules, not only "
                f"the cap {MEMBER_CAP:,} units the metrics use (`eval/selection.json` ambiguous_tail_share, stage 17; "
                "(rate ambiguous - rate templated) / (rate non-templated - rate templated), clipped to [0, 1]):", ""]
    p = base / "eval" / "selection.json"
    sel = _json(p)
    if sel is None:
        out.append(_not_run(base, [p]))
    elif "ambiguous_tail_share" not in sel:
        out.append("not run (eval/selection.json has no ambiguous_tail_share)")
    else:
        ats = sel["ambiguous_tail_share"]
        nkeys = list(dict.fromkeys(k for d in ats.values() for k in d if k.startswith("n_")))
        rows = [[lib, _r(d.get("estimate")), _r(d.get("rate_templated")), _r(d.get("rate_nontemplated")),
                 _r(d.get("rate_ambiguous"))] + [_int(d.get(k)) for k in nkeys] for lib, d in ats.items()]
        out += _table(["Library", "Estimated tail-primed share", "Hexamer rate templated", "Hexamer rate non-templated",
                       "Hexamer rate ambiguous"] + nkeys, rows)
    return "\n".join(out)


def grid_section(base):
    base = Path(base)
    out = ["## 3. Grid: removed primary records", "",
           "Source: `<lib>/grid/summary.json` (stage 13). Share of primary records in brackets. The remove-all control "
           "(MINSNRLEN none) removes every non-terminal read on an expressed transcript at that COVLEN; it bounds removals "
           "only within its own mode.", ""]
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


def selection_section(base):
    base = Path(base)
    out = ["## 4. Setting selection", ""]
    p = base / "eval" / "selection.json"
    sel = _json(p)
    if sel is None:
        return "\n".join(out + [_not_run(base, [p])])
    libs, sl, min_spec = sel["libraries"], sel["selection_library"], sel["min_spec"]
    out += [f"Source: `eval/selection.json`, `eval/bootstrap.parquet`, `eval/metrics.parquet` (stage 17). Rule: {sel['rule']}", ""]
    rows, fronts = [], []
    for mode, d in sel["modes"].items():
        s = d["selection"]
        none = "no candidates" if s["n_candidates"] == 0 else f"none reached specificity {min_spec:g}"
        rows.append([MODE_NAMES[mode], s["tag"] or none, _r(s["sens"]), _r(s["spec"]), _r(s["prec"]),
                     f"{s['n_qualifying']} / {s['n_candidates']}"])
        if s["tag"] is None and s["n_candidates"]:
            fronts.append((mode, s["frontier"]))
    out += _table(["Mode", f"Chosen setting (on {sl})", "Sens", "Spec", "Prec", "Qualifying / candidates"], rows)
    for mode, front in fronts:
        out += ["", f"Frontier, {MODE_NAMES[mode]} mode (no setting chosen; candidates not dominated in sensitivity and "
                    f"specificity):", ""]
        out += _table(["Setting", "Sens", "Spec"], [[t, _r(a), _r(b)] for t, a, b in front])

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
    for tag, rs in roles.items():
        for lib in libs:
            e = _entry(sel, tag, lib)
            m = e.get("molecule")
            if not m:
                rows.append([tag, ", ".join(rs), lib] + ["n/a"] * 5)
                rows_lvl.append([tag, lib] + ["n/a"] * 3)
                continue
            cells = [_ci(m[k], *ci.get((lib, tag, k), (None, None))) for k in ("sens", "spec", "prec")]
            rows.append([tag, ", ".join(rs), lib, _int(m["tp"] + m["fn"]), _int(m["tn"] + m["fp"])] + cells)
            rows_lvl.append([tag, lib] + [f"{_r(e[lv]['sens'])} / {_r(e[lv]['spec'])}" if lv in e else "n/a"
                                          for lv in ("molecule", "molecule_majority", "read")])
    out += _table(["Setting", "Role", "Library", "Templated", "Non-templated", "Sens", "Spec", "Prec"], rows)
    if boot is None:
        out += ["", f"Bootstrap: {_not_run(base, [bp])}"]
    else:
        clusters = boot[boot.level == "molecule"].groupby("lib", sort=False).n_clusters.first()
        out += ["", "Bootstrap: 2.5th and 97.5th percentiles over gene-cluster resamples (cluster = gene, or a 10 kb locus "
                    "window for intergenic molecules); clusters " + "; ".join(f"{k} {_int(v)}" for k, v in clusters.items()) + "."]
    out += ["", "Majority rule (molecule removed when most of its reads are; our extension, spec 5.3) and read level, "
                "sens / spec:", ""]
    out += _table(["Setting", "Library", "Molecule (all reads)", "Molecule (majority)", "Read"], rows_lvl)

    out += ["", "Units per member cap (molecules with a truth read missing from truth_flags are dropped):", ""]
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

    out += ["", "Intergenic molecules: polyAfilter's windows lie on expressed transcripts, so an intergenic molecule is "
                "removed only through reads elsewhere. 'Removed at most' is the largest count over all settings in "
                f"metrics.parquet (molecule level, cap {MEMBER_CAP:,}, {CUTOFFS}); the ceiling is the stratum-all sensitivity if "
                "every other templated molecule were removed. Intergenic non-templated molecules count as retained negatives.", ""]
    mp = base / "eval" / "metrics.parquet"
    metrics = _parquet(mp)
    if metrics is None:
        out[-2:] = [f"Intergenic molecules: {_not_run(base, [mp])}"]
        return "\n".join(out)
    m = metrics[(metrics.cap == MEMBER_CAP) & (metrics.cutoffs == CUTOFFS) & (metrics.level == "molecule")]
    rows, notes = [], []
    for lib in libs:
        a = m[(m.lib == lib) & (m.stratum_kind == "all")]
        g = m[(m.lib == lib) & (m.stratum_kind == "region") & (m.stratum == "intergenic")]
        if a.empty or g.empty:
            continue
        pos, gpos, gneg = a.tp.iloc[0] + a.fn.iloc[0], g.tp.iloc[0] + g.fn.iloc[0], g.tn.iloc[0] + g.fp.iloc[0]
        rows.append([lib, _int(pos), _int(gpos), _int(g.tp.max()), _r((pos - gpos + g.tp.max()) / pos if pos else None),
                     _int(gneg), _int(g.fp.max())])
        neg = a.tn.iloc[0] + a.fp.iloc[0]
        if lib == sl and neg > gneg:
            notes.append(f"While a setting keeps every intergenic non-templated molecule, specificity >= {min_spec:g} over all "
                         f"{_int(neg)} non-templated units on {sl} is at least ({min_spec:g} x {_int(neg)} - {_int(gneg)}) / "
                         f"{_int(neg - gneg)} = {_r((min_spec * neg - gneg) / (neg - gneg))} over the {_int(neg - gneg)} genic "
                         "ones.")
    out += _table(["Library", "Templated units", "Intergenic templated", "Removed at most", "Sensitivity ceiling",
                   "Intergenic non-templated", "Removed at most"], rows)
    return "\n".join(_notes(out, notes))


def strata_section(base):
    base = Path(base)
    out = ["## 5. Metrics by A band and region", ""]
    p = base / "eval" / "metrics.parquet"
    metrics = _parquet(p)
    if metrics is None:
        return "\n".join(out + [_not_run(base, [p])])
    tags, note = _report_tags(base)
    out += [f"Source: `eval/metrics.parquet` (stage 17), molecule level, member cap {MEMBER_CAP:,} nt, {CUTOFFS} labels. "
            f"Specificity exists only in band {BANDS[0]} and sensitivity only in bands {BANDS[2]} and up; band {BANDS[1]} is "
            "ambiguous.", ""] + note
    m = metrics[(metrics.cap == MEMBER_CAP) & (metrics.cutoffs == CUTOFFS) & (metrics.level == "molecule")
                & metrics.tag.isin(tags)]
    libs = list(dict.fromkeys(m.lib))
    look = {(r.tag, r.lib, r.stratum_kind, r.stratum): r for r in m.itertuples()}

    def get(tag, lib, kind, stratum, metric):
        r = look.get((tag, lib, kind, stratum))
        return None if r is None else getattr(r, metric)

    present = [(tag, lib) for tag in tags for lib in libs if (tag, lib, "all", "all") in look]
    out += _table(["Setting", "Library", f"Spec {BANDS[0]}"] + [f"Sens {b}" for b in BANDS[2:]],
                  [[tag, lib, _r(get(tag, lib, "a_band", BANDS[0], "spec"))]
                   + [_r(get(tag, lib, "a_band", b, "sens")) for b in BANDS[2:]] for tag, lib in present])
    out += ["", "By region, sens / spec:", ""]
    out += _table(["Setting", "Library"] + list(REGION_NAMES),
                  [[tag, lib] + [f"{_r(get(tag, lib, 'region', g, 'sens'))} / {_r(get(tag, lib, 'region', g, 'spec'))}"
                                 if (tag, lib, "region", g) in look else "n/a" for g in REGION_NAMES] for tag, lib in present])
    out += ["", "Units per stratum (the same for every setting):", ""]
    rows = []
    for lib in libs:
        first = m[(m.lib == lib) & (m.tag == m[m.lib == lib].tag.iloc[0]) & m.stratum_kind.isin(["a_band", "region"])]
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
            "AATAAA/ATTAAA 5 to 35 nt upstream of the site.", ""]
    h = h[h.cap == MEMBER_CAP]
    out += _table(["Setting", "Library", "Label", "Removed", "Removed hexamer rate", "Kept", "Kept hexamer rate"],
                  [[r.tag, r.lib, LABEL_NAMES.get(r.label, r.label), _int(r.removed_n), _r(r.removed_hex_rate), _int(r.kept_n),
                    _r(r.kept_hex_rate)] for r in h.itertuples()])
    return "\n".join(out)


def spikein_section(base):
    base = Path(base)
    out = ["## 7. Spike-ins", "",
           "Source: `<lib>/spikein/truth/molecules_labeled.parquet` (stage 14), `eval/spikein_metrics.parquet` (stage 17). "
           "spike_label comes from the distance to the annotated 3' end; label from the A count after the site.", ""]
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
    out += note
    s = s[(s.level == "molecule") & (s.cap == MEMBER_CAP)]
    look = {(r.lib, r.tag, r.stratum): r for r in s.itertuples()}
    rows = []
    for lib in dict.fromkeys(s.lib):
        for tag in tags:
            if (lib, tag, "all") in look:
                rows.append([lib, tag] + [f"{_r(look[lib, tag, k].sens)} / {_r(look[lib, tag, k].spec)}"
                                          if (lib, tag, k) in look else "n/a" for k in ("all",) + MODULES])
    out += _table(["Library", "Setting", "all"] + list(MODULES), rows)
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
           "unfiltered and per setting.", ""]
    entries, notes, flags = [], [], []
    for lib in NUCLEI:
        p = base / lib / "spikein" / "ercc.json"
        d = _json(p)
        if d is None:
            notes.append(f"{lib}: {_not_run(base, [p])}")
            continue
        if d.get("provisional"):
            flags.append(f"Provisional ({lib}): {d.get('note', '')}")
        src = d.get("settings", d)
        entries += [(lib, k, _flat(v)) for k, v in src.items() if isinstance(v, dict)]
    out += [f"- {f}" for f in flags] + ([""] if flags else [])
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
           "site with an accepted read (binding, spec 4.3: passes at ratio >= 3) and per accepted read.", ""]
    data, notes = {}, []
    for lib in LONG_READS:
        p = base / lib / "long_read_acceptance.json"
        d = _json(p)
        if d is None:
            notes.append(f"{lib}: {_not_run(base, [p])}")
        else:
            data[lib] = (d, _parquet(base / lib / "long_read_sites.parquet", ["n_accepted", "a20", "hexamer"]))
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
    out += ["", "- The spec expects about 7% of reads dropped. BD67 drops more in every build check (11.1% on chr21 and chr22, "
                "12.4% on chr1; BD70 7.7% and 9.1%), from more short-run and both-fail reads, so expect about 12% genome-wide; "
                "that is the library, not a failure of the acceptance rule."]
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
                   "Passes (ratio >= 3)", "Per-read rate T", "Per-read rate NT", "Ratio per read"], rows)
    out += ["", "Agreement with 1179 (1179 molecules on the references stage 16 processed; a long-read site within 10 nt on "
                "the same strand; agreement within 10 nt mostly measures co-location, the matched share per label is the "
                "informative number):", ""]
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
        out += ["", "Selection bias (share of labeled molecules versus share of primary records, by RE tag):", ""]
        out += _table(["Library", "RE", "Labeled share", "Records share"],
                      [[r.lib, r.re, _r(r.labeled_share), _r(r.records_share)] for r in sb.itertuples()])
    p = base / "burden" / "polyasite.parquet"
    px = _parquet(p)
    if px is None:
        out += ["", f"PolyASite proximity: {_not_run(base, [p])}"]
    else:
        out += ["", "PolyASite 3.0 proximity (share of labeled molecules by the distance in nt from the site to the nearest atlas "
                    "cleavage site on the same strand; templated molecules also by A band). Proximity is reported, not used to "
                    "validate the labels: the atlas is built from 10x 3' single-cell reads, oligo(dT) primed like these "
                    "libraries, and does not screen out candidate sites with A-rich genome downstream, so internally primed "
                    "sites can be in it. The null row takes every molecule of the group with its site moved 1,000 nt upstream "
                    "in transcript sense on the same ref and strand; its share within 10 nt is the share expected by chance "
                    "at this atlas density."]
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
    out += ["Source: `validation/exact_match_m2.json` (stage 20). Removed-record sets (every record the original fetches, "
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
                             _r(r["original_max_rss_gib"], 2), _r(r["original_peak_tree_rss_gib"], 2)])
    if rows:
        out += _table(["Library", "Setting", "Original removed", "Reimpl removed", "Only original", "Only reimpl", "Identical",
                       "Original wall (h)", "Max RSS (GiB)", "Peak tree RSS (GiB)"], rows)
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
    return "\n".join(out)


SECTIONS = (runs_section, truth_section, grid_section, selection_section, strata_section, hexamer_section, spikein_section,
            ercc_section, long_read_section, burden_section, validation_section)


def render(base):
    head = ["# Milestone 2 results (generated by scripts/21_m2_summary.py)", "",
            f"Inputs under `{base}`. Each section names the files its numbers come from; \"not run\" lists the files that "
            "are absent.", ""]
    return "\n".join(head) + "\n" + "\n\n".join(f(base) for f in SECTIONS) + "\n"
