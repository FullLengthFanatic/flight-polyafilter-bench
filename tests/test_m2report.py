import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pafbench import m2report
from pafbench.evaluate import METRIC_COLUMNS
from pafbench.labels import a_band
from pafbench.molecules import MOL_COLUMNS
from pafbench.validation import parse_tag

REPO = Path(__file__).resolve().parents[1]
I1, I3, E1 = "c300_l10_m1_i", "c375_l3_m2_i", "c300_l10_m1_e"
NOTE = "Lexogen's ERCC mix identity in SIRV Set 4 is not verified; concentrations are ERCC Mix 1 from Thermo Fisher"


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(path, index=False)
    elif isinstance(obj, str):
        path.write_text(obj)
    else:
        path.write_text(json.dumps(obj, indent=1))


def _mol(rows):
    """Labeled molecules with the real columns; rows = (a20, label, region, genes, has_ts, hexamer, n_members)."""
    n = len(rows)
    a20, label, region, genes, has_ts, hexamer, n_members = (list(c) for c in zip(*rows))
    df = pd.DataFrame({"mol": np.arange(n), "cb": "AAAC-1", "ub": [f"U{i}" for i in range(n)], "strand": "+", "ref": "chr1",
                       "site": 1000 + 100 * np.arange(n), "n_pa": 1, "spread": 0, "a20": a20, "down20": "A" * 20,
                       "up40": "C" * 40, "hexamer": hexamer, "genes": genes, "re": "E", "mapq_min": 255})[MOL_COLUMNS]
    return df.assign(label=label, a_band=[a_band(x) for x in a20], region=region, gene=[g.split(";")[0] for g in genes],
                     has_ts=has_ts, n_members=n_members)


def _rates(tp, fn, tn, fp):
    div = lambda a, b: a / b if b else np.nan  # noqa: E731
    return div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)


def _metric(lib, tag, kind, stratum, tp, fn, tn, fp, cap=1000, cutoffs="8/4", level="molecule"):
    s = parse_tag(tag)
    sens, spec, prec = _rates(tp, fn, tn, fp)
    return dict(lib=lib, cap=cap, cutoffs=cutoffs, tag=tag, covlen=s.covlen, min_len=s.min_len, mism=s.mism,
                introns=s.introns, level=level, stratum_kind=kind, stratum=stratum, tp=float(tp), fn=float(fn), tn=float(tn),
                fp=float(fp), n_amb=0.0, amb_removed=0.0, sens=sens, spec=spec, prec=prec)


def _lvl(tp, fn, tn, fp):
    sens, spec, prec = _rates(tp, fn, tn, fp)
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "n_amb": 10, "amb_removed": 1, "sens": sens, "spec": spec, "prec": prec}


def _sel(tag, sens, spec, prec, n_q):
    return {"tag": tag, "sens": sens, "spec": spec, "prec": prec, "n_candidates": 189, "n_qualifying": n_q, "frontier": []}


NONE_E = {"tag": None, "sens": None, "spec": None, "prec": None, "n_candidates": 189, "n_qualifying": 0,
          "frontier": [["c150_l3_m2_e", 0.41, 0.9], [E1, 0.0356, 0.9943]]}
# chosen in -i on 1417: tp 9,795 fn 31,574 tn 2,757 fp 29 -> sens 0.2368, spec 0.9896, prec 0.9970
I1_1417 = {"molecule": _lvl(9795, 31574, 2757, 29), "molecule_majority": {"sens": 0.2587, "spec": 0.9856, "prec": 0.997},
           "read": {"sens": 0.2780, "spec": 0.9752, "prec": 0.9937}}
I1_1179 = {"molecule": _lvl(300, 700, 95, 5), "molecule_majority": {"sens": 0.35, "spec": 0.9, "prec": 0.98},
           "read": {"sens": 0.4, "spec": 0.8, "prec": 0.97}}


def _full_base(tmp_path):
    b = tmp_path / "base"
    # ---- stage 12, 1417 only; 1419 and 1179 absent
    mol = _mol([(15, "templated", "3UTR", "gA", False, False, 2),
                (9, "templated", "intron", "gA;gB", True, False, 1),
                (12, "templated", "intergenic", "", False, True, 0),
                (4, "nontemplated", "3UTR", "gC", False, True, 1),
                (6, "ambiguous", "3UTR", "gC", False, True, 0),
                (18, "templated", "antisense", "", False, False, 0)])
    _write(b / "1417/truth/molecules_labeled.parquet", mol)
    pa = pd.DataFrame({"qname": [f"p{i}" for i in range(6)], "mol": np.arange(6), "role": "pa", "dist5_exon": np.nan,
                       "dist_mode": "", "spliced": False})
    mem = pd.DataFrame({"qname": [f"m{i}" for i in range(6)], "mol": [0, 0, 1, 1, 3, 4], "role": "member",
                        "dist5_exon": [100.0, 700.0, 1000.0, 1500.0, 400.0, 2000.0], "dist_mode": "exon", "spliced": False})
    _write(b / "1417/truth/truth_reads.parquet", pd.concat([pa, mem], ignore_index=True))
    # ---- stage 13
    _write(b / "1417/grid/summary.json", {
        "library": "1417", "settings": 396, "workers": 8, "scanned": 116, "seconds": 1234.4,
        "per_ref": {"chr1": {"records": 6962527, "primary": 6962527, "seconds": 186.0},
                    "chr21": {"records": 1837866, "primary": 1837866, "seconds": 28.6}},
        "records": 85000000, "primary": 83140741,
        "removed": {I1: 17068514, I3: 46808565, E1: 3729928, "c300_lnone_m0_i": 50000000}})
    _write(b / "1417/grid/stage13.mem.json", {
        "cmd": ["/x/.venv/bin/python", "/x/polyafilter_bench/scripts/13_grid_flags.py", "1417", "--workers", "8"],
        "exit_status": 0, "elapsed_s": 1240.2, "peak_tree_rss_gib": 19.123, "interval_s": 2.0})
    # ---- stage 19 original runs: one finished (with memwatch), one killed (GNU time writes Exit status 0)
    _write(b / "1417/original/c275_l9_m1_i.time.txt",
           "\tCommand being timed: \"x\"\n\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:57:00\n"
           "\tMaximum resident set size (kbytes): 16238356\n\tExit status: 0\n")
    _write(b / "1417/original/c275_l9_m1_i.mem.json", {
        "cmd": ["/x/scripts/02_run_original.sh", "1417", "275", "9", "1", "i"], "exit_status": 0, "elapsed_s": 7021.0,
        "peak_tree_rss_gib": 60.5, "interval_s": 2.0})
    _write(b / "1417/original/c180_l11_m2_i.time.txt",
           "Command terminated by signal 9\n\tElapsed (wall clock) time (h:mm:ss or m:ss): 10:00\n"
           "\tMaximum resident set size (kbytes): 1048576\n\tExit status: 0\n")
    _write(b / "1417/original/c300_l10_m0_i.time.txt", "")
    # ---- stage 17
    sel = {
        "rule": "highest molecule-level sensitivity with specificity >= 0.95 on 1417", "min_spec": 0.95,
        "selection_library": "1417", "cap": 1000, "cutoffs": "8/4", "libraries": ["1417", "1179"], "missing": {},
        "units": {"1417": {"500": {"read_units": 73200, "molecule_units": 47402, "missing_reads": 0, "dropped_molecules": 0},
                           "1000": {"read_units": 76245, "molecule_units": 48936, "missing_reads": 0, "dropped_molecules": 0},
                           "2000": {"read_units": 77966, "molecule_units": 49827, "missing_reads": 3, "dropped_molecules": 2}}},
        "modes": {"e": {"introns": False, "selection": NONE_E, "metrics": {}},
                  "i": {"introns": True, "selection": _sel(I1, 0.2368, 0.9896, 0.9970, 120),
                        "metrics": {"1417": I1_1417, "1179": I1_1179}}},
        "published": {I1: {"1417": I1_1417, "1179": I1_1179},
                      I3: {"1417": {"molecule": _lvl(26356, 15013, 2399, 387), "molecule_majority": {"sens": 0.6559, "spec": 0.8482, "prec": 0.98},
                                    "read": {"sens": 0.6771, "spec": 0.828, "prec": 0.9823}},
                           "1179": {}}},
        "cap_sensitivity": {c: {"e": NONE_E, "i": _sel(I1, s, 0.991, 0.99, q)}
                            for c, s, q in (("500", 0.25, 118), ("1000", 0.2368, 120), ("2000", 0.22, 121))},
        "cutoff_sweep": [{"templated_min": 7, "nontemplated_max": 2, "mode": "i", "tag": I1, "n_qualifying": 100,
                          "by_library": {"1417": {"sens": 0.2, "spec": 0.99, "prec": 0.97}, "1179": None}},
                         {"templated_min": 10, "nontemplated_max": 6, "mode": "e", "tag": None, "n_qualifying": 0,
                          "by_library": {}}],
        "bootstrapped_tags": [I1, I3],
        "ambiguous_tail_share": {"1417": {"estimate": 0.458, "rate_templated": 0.069, "rate_nontemplated": 0.772,
                                          "rate_ambiguous": 0.391, "n_templated": 4, "n_nontemplated": 1, "n_ambiguous": 1}},
    }
    _write(b / "eval/selection.json", sel)
    # the read-level row comes first and has its own cluster count (clusters without a unit at a level are left out)
    boot = [("1417", 1000, "read", I1, "sens", 0.278, 0.1, 0.9, 13500),
            ("1417", 1000, "molecule", I1, "sens", 0.2368, 0.2074, 0.2588, 13271),
            ("1417", 1000, "molecule", I1, "spec", 0.9896, 0.9854, 0.9932, 13271),
            ("1417", 1000, "molecule", I1, "prec", 0.9970, 0.9959, 0.9981, 13271),
            ("1417", 1000, "molecule", I3, "sens", 0.6371, 0.5623, 0.6904, 13271)]
    _write(b / "eval/bootstrap.parquet", pd.DataFrame(
        [r[:8] + (1000, r[8]) for r in boot], columns=["lib", "cap", "level", "tag", "metric", "estimate", "lo", "hi", "n_boot", "n_clusters"]))
    rows = []
    for tag in (I1, I3):
        rows += [_metric("1417", tag, "all", "all", 9795, 31574, 2757, 29),
                 _metric("1417", tag, "a_band", "0-4", 0, 0, 2757, 29), _metric("1417", tag, "a_band", "5-7", 0, 0, 0, 0),
                 _metric("1417", tag, "a_band", "8-9", 100, 1900, 0, 0), _metric("1417", tag, "a_band", "10-13", 1000, 5400, 0, 0),
                 _metric("1417", tag, "a_band", "14-20", 8695, 24274, 0, 0),
                 _metric("1417", tag, "region", "intergenic", 0 if tag == I1 else 500, 5182 if tag == I1 else 4682,
                         125 if tag == I1 else 120, 3 if tag == I1 else 8),
                 _metric("1417", tag, "region", "3UTR", 5000, 10000, 2000, 20)]
    # rows the sections must ignore: another cap, other cut-offs, the read level
    rows += [_metric("1417", I1, "region", "intergenic", 99, 5083, 100, 28, cap=500),
             _metric("1417", I1, "region", "intergenic", 77, 5105, 100, 28, cutoffs="7/2"),
             _metric("1417", I1, "a_band", "8-9", 2000, 0, 0, 0, level="read")]
    metrics = pd.DataFrame(rows)[METRIC_COLUMNS]
    metrics["min_len"] = metrics.min_len.astype("Int64")
    _write(b / "eval/metrics.parquet", metrics)
    _write(b / "eval/hexamer_check.parquet", pd.DataFrame(
        [("1417", 1000, I3, "all", 28762, 0.061, 20174, 0.229), ("1417", 1000, I3, "nontemplated", 387, 0.305, 2399, 0.836),
         ("1417", 1000, I3, "ambiguous", 0, np.nan, 10, 0.4), ("1417", 500, I3, "all", 1, 0.9, 1, 0.9)],
        columns=["lib", "cap", "tag", "label", "removed_n", "removed_hex_rate", "kept_n", "kept_hex_rate"]))
    spike = [dict(_metric("1417", I1, "all", "all", 10, 30, 50, 2), level="molecule"),
             dict(_metric("1417", I1, "module", "ERCC", 5, 5, 20, 0)),
             dict(_metric("1417", I1, "all", "all", 9, 1, 0, 0, level="read"))]
    spike = pd.DataFrame(spike).drop(columns=["cutoffs"])
    spike["min_len"] = spike.min_len.astype("Int64")
    _write(b / "eval/spikein_metrics.parquet", spike)
    # ---- stage 14, 1417 only
    sm = _mol([(12, "templated", "ncexon", "ERCC-00002", False, False, 1), (15, "templated", "ncexon", "SIRV1", False, False, 1),
               (1, "nontemplated", "ncexon", "SIRV1", False, True, 1), (0, "nontemplated", "ncexon", "ERCC-00002", False, True, 1),
               (3, "nontemplated", "ncexon", "SIRV4001", False, True, 1), (10, "templated", "ncexon", "SIRV4001", False, False, 1),
               (6, "ambiguous", "ncexon", "ERCC-00002", False, False, 1), (5, "ambiguous", "ncexon", "SIRV1", False, False, 1),
               (7, "ambiguous", "ncexon", "SIRV1", False, False, 1), (9, "templated", "ncexon", "SIRV1", False, False, 1)])
    sm = sm.assign(spike_label=["internal", "internal", "internal", "tail", "tail", "tail", "tail", "other", "internal", "tail"],
                   module=["ERCC", "SIRV_E0", "SIRV_E0", "ERCC", "long_SIRV", "long_SIRV", "ERCC", "SIRV_E0", "SIRV_E0", "SIRV_E0"],
                   dist_to_end=[300, 400, 500, 0, 2, -3, 5, 20, 600, 1])
    _write(b / "1417/spikein/truth/molecules_labeled.parquet", sm)
    # ---- stage 15: 1417 in stage 15's layout (settings -> correlation dict, UMIs, kept share; null correlations for a
    # constant input), 1419 with the settings at the top level
    _write(b / "1417/spikein/ercc.json", {
        "library": "1417", "provisional": True, "note": NOTE, "concentration_source": "https://x",
        "selection": {"e": None, "i": I1}, "ercc_contigs": 92, "reads": 120000, "ercc_detected": 61,
        "ercc_with_concentration": 61, "settings": {
            "unfiltered": {"correlation": {"n": 61, "pearson_log10": 0.9123, "spearman": 0.8876}, "umis": 52000,
                           "umis_kept_share": 1.0},
            I1: {"correlation": {"n": 61, "pearson_log10": 0.9201, "spearman": 0.8899}, "umis": 50502, "umis_kept_share": 0.9712},
            I3: {"correlation": {"n": 3, "pearson_log10": None, "spearman": None}, "umis": 3, "umis_kept_share": 0.0001}}})
    _write(b / "1419/spikein/ercc.json", {"provisional": True, "note": NOTE, "unfiltered": {
        "correlation": {"n": 60, "pearson_log10": 0.9, "spearman": 0.8}, "umis": 4000, "umis_kept_share": 1.0}})
    # ---- stage 16, BD67 only
    used, acc = 1610467, 1432000
    _write(b / "BD67/long_read_acceptance.json", {
        "library": "BD67", "bam": "/x/BD67.bam", "refs": ["chr21", "chr22"],
        "records": {"records": 1735308, "secondary_supplementary": 14097, "low_mapq": 110744, "used": used},
        "duplicate_qnames": 0, "outcomes": {"accepted": acc, "short_run": 77500, "short_after": 113, "both": 100854},
        "outcome_fractions": {}, "dropped_fraction": 1 - acc / used, "unique_sites": 70136, "sites_with_accepted_read": 42170,
        "cutoffs_8_4": {"cutoffs": {"templated_min": 8, "nontemplated_max": 4},
                        "n": {"templated": 28473, "nontemplated": 6469, "ambiguous": 7228},
                        "hexamer_rate": {"templated": 0.106, "nontemplated": 0.541, "ambiguous": 0.443}, "ratio": 5.12, "passes": True},
        "cutoffs_14_8": {"cutoffs": {"templated_min": 14, "nontemplated_max": 8},
                         "n": {"templated": 11551, "nontemplated": 16279, "ambiguous": 14340},
                         "hexamer_rate": {"templated": 0.046, "nontemplated": 0.456, "ambiguous": 0.2}, "ratio": 9.95, "passes": True},
        "per_ref": {},
        "vs_1179": {"path": "/x/1179.parquet", "molecules": 16408, "matched": 10091, "share_matched": 0.615,
                    "share_matched_by_label": {"templated": 0.491, "nontemplated": 0.907, "ambiguous": 0.805},
                    "agreement": {"templated": 0.993, "nontemplated": 0.996, "pooled": 0.994}}})
    sites = pd.DataFrame({"ref": "chr21", "strand": "+", "site": [10, 20, 30, 40, 50, 60, 70, 80, 90],
                          "n_reads": [3, 1, 2, 2, 1, 1, 2, 1, 1], "n_accepted": [3, 1, 2, 2, 0, 0, 2, 1, 1],
                          "a20": [15, 9, 2, 1, 3, 12, 4, 8, 14],
                          "label": ["templated", "templated", "nontemplated", "nontemplated", "nontemplated", "templated",
                                    "nontemplated", "templated", "templated"],
                          "a_band": "", "hexamer": [False, True, True, False, True, True, True, False, True], "region": "3UTR",
                          "gene": "gA"})
    _write(b / "BD67/long_read_sites.parquet", sites)
    # ---- stage 18
    share = pd.DataFrame(
        [("1417", "all", "all", 118278, 10822, 15374, 144474, 1.0, 0.81868, 0.91617, 0.8038, 0.8346),
         ("1417", "region", "3UTR", 15773, 8607, 6749, 31129, 0.2155, 0.5067, 0.6470, 0.479, 0.532),
         ("1417", "a_band", "14-20", 40103, 0, 0, 40103, 0.2776, 1.0, 1.0, 0.2705, 0.2851),
         ("1417", "has_ts", "True", 524, 50, 150, 724, 0.005, 0.7238, 0.9129, 0.638, 0.794),
         ("cells", "all", "all", 900, 50, 50, 1000, 1.0, 0.9, 0.9474, 0.88, 0.92)],
        columns=["group", "stratum_kind", "stratum", "templated", "nontemplated", "ambiguous", "total", "frac", "share_all",
                 "share_resolved", "lo", "hi"])
    _write(b / "burden/templated_share.parquet", share)
    _write(b / "burden/same_site.parquet", pd.DataFrame(
        [("1417_vs_1419", "templated", "templated", 95920), ("1417_vs_1419", "templated", "none", 500),
         ("1417_vs_1419", "nontemplated", "nontemplated", 6261), ("1417_vs_1419", "ambiguous", "ambiguous", 11219),
         ("1417_vs_1419", "ambiguous", "templated", 100)],
        columns=["pair", "label_a", "label_b", "n"]))
    _write(b / "burden/same_gene.parquet", pd.DataFrame([("1417_vs_1419", 1212, 1.0, 0.0)],
                                                        columns=["pair", "n_genes", "spearman_rho", "p_value"]))
    _write(b / "burden/selection_bias.parquet", pd.DataFrame(
        [("1417", "E", 0.421, 0.407), ("1417", "N", 0.48, 0.5), ("1417", "I", 0.099, 0.093)],
        columns=["lib", "re", "labeled_share", "records_share"]))
    prox = {"1417": {("templated", "all"): [9000, 20000, 40000, 44000, 5278],
                     ("nontemplated", "all"): [4000, 2000, 2000, 2681, 141],
                     ("ambiguous", "all"): [3000, 3000, 4000, 3674, 1700],
                     ("null_shift1kb", "all"): [31500, 40000, 50000, 15000, 7974],
                     ("templated", "8-9"): [1000, 3000, 6000, 7577, 1000],
                     ("templated", "10-13"): [5000, 10000, 20000, 22598, 2000],
                     ("templated", "14-20"): [3000, 7000, 14000, 14103, 2000]},
            "cells": {("templated", "all"): [100, 200, 300, 300, 100], ("nontemplated", "all"): [0, 0, 0, 0, 0],
                      ("ambiguous", "all"): [5, 5, 0, 6, 4], ("null_shift1kb", "all"): [255, 255, 255, 153, 102],
                      ("templated", "8-9"): [0, 0, 0, 0, 0], ("templated", "10-13"): [50, 100, 150, 150, 50],
                      ("templated", "14-20"): [50, 100, 150, 150, 50]}}
    _write(b / "burden/polyasite.parquet", pd.DataFrame(
        [(g, lab, band, bin_, n) for g, d in prox.items() for (lab, band), ns in d.items()
         for bin_, n in zip(("0-10", "11-50", "51-200", ">200", "no atlas site on this ref and strand"), ns)],
        columns=["group", "label", "a_band", "bin", "n"]))
    # ---- stage 20
    em = {"n_original": 17000000, "n_reimpl": 17000000, "only_original": 0, "only_reimpl": 0, "identical": True, "examples": []}
    _write(b / "validation/exact_match_m2.json", {
        "1417": {"c275_l9_m1_i": em | {"original_elapsed_s": 7020.0, "original_max_rss_gib": 15.486, "original_peak_tree_rss_gib": 60.5},
                 I1: em | {"n_original": 17068514, "n_reimpl": 17068514, "original_elapsed_s": 7020.0,
                           "original_max_rss_gib": 15.5, "original_peak_tree_rss_gib": None}},
        "1419": {},
        "none_control": {"refs": ["chr21", "chr22"], "n_records": 2781871, "setting": "c300_lnone_m0_i", "n_original": 1894573,
                         "n_reimpl": 1894573, "only_original": 0, "only_reimpl": 0, "identical": True, "examples": [],
                         "createTRANS": {"elapsed_s": 412.0, "max_rss_gib": 0.26, "exit_status": 0, "peak_tree_rss_gib": 0.23},
                         "seconds": {"subset": 20.0, "createTRANS": 412.0, "original": 160.0, "compare": 80.0}},
        "n_compared": 3, "all_identical": True})
    return b


@pytest.fixture
def base(tmp_path):
    return _full_base(tmp_path)


def _sections(text):
    """{section number: its text} from a rendered page."""
    parts = text.split("\n## ")[1:]
    return {int(p.split(".", 1)[0]): p for p in parts}


def test_empty_base_renders_every_section_as_not_run(tmp_path):
    text = m2report.render(tmp_path)
    s = _sections(text)
    assert sorted(s) == list(range(1, 12))
    for k, body in s.items():
        assert "not run" in body, k
    assert "1417: not run (missing 1417/truth/molecules_labeled.parquet, 1417/truth/truth_reads.parquet)" in s[2]
    assert "1179: not run (missing 1179/grid/summary.json)" in s[3]
    assert "not run (missing eval/selection.json)" in s[4]
    assert "not run (missing eval/metrics.parquet)" in s[5]
    assert "not run (missing eval/hexamer_check.parquet)" in s[6]
    assert "1419: not run (missing 1419/spikein/truth/molecules_labeled.parquet)" in s[7]
    assert "1417: not run (missing 1417/spikein/ercc.json)" in s[8]
    assert "BD70: not run (missing BD70/long_read_acceptance.json)" in s[9]
    assert "not run (missing burden/templated_share.parquet)" in s[10]
    assert "not run (missing validation/exact_match_m2.json)" in s[11]
    # a base that does not exist at all behaves the same
    assert m2report.render(tmp_path / "absent") == text.replace(str(tmp_path), str(tmp_path / "absent"))


def test_runs_section(base):
    s = m2report.runs_section(base)
    # memwatch outside original/: command from the script on, paths reduced to names; 1,240.2 s; 19.123 GiB
    assert "| 13_grid_flags.py 1417 --workers 8 | 1417/grid/stage13.mem.json | 1,240 | 19.12 | 0 |" in s
    assert "c275_l9_m1_i.mem.json |" not in s.split("Stage 13")[0]
    # grid: 2 references, 85,000,000 records of which 83,140,741 primary, sum of per-reference seconds 186.0 + 28.6 = 214.6,
    # slowest chr1
    assert "| 1417 | 2 | 85,000,000 | 83,140,741 | 8 | 116 | 1,234 | 215 | chr1 (186 s) |" in s
    assert "1419: not run (missing 1419/grid/summary.json)" in s
    # 1:57:00 = 1.95 h; 16,238,356 KiB = 15.49 GiB; tree peak 60.5 from the mem.json
    assert "| 1417 | c275_l9_m1_i | 1.95 | 15.49 | 60.50 | 0 |" in s
    # killed by signal 9: exit 137 although GNU time wrote 0; 10:00 = 0.17 h; 1,048,576 KiB = 1.00 GiB; no mem.json
    assert "| 1417 | c180_l11_m2_i | 0.17 | 1.00 | n/a | 137 |" in s
    # an empty time file (GNU time killed before writing): no exit status
    assert "| 1417 | c300_l10_m0_i | running | n/a | n/a | n/a |" in s
    assert "validation/none_control/" in s.split("Stage 13")[0]


def test_truth_section(base):
    s = m2report.truth_section(base)
    assert "| Quantity | 1417 |" in s
    assert "| Molecules | 6 |" in s
    # 4 templated, 1 ambiguous, 1 non-templated of 6
    assert "| templated | 4 (66.7%) |" in s
    assert "| ambiguous | 1 (16.7%) |" in s
    assert "| non-templated | 1 (16.7%) |" in s
    assert "| With a ts read (has_ts) | 1 |" in s
    assert "| Multi-gene (gene = alphabetically first id) | 1 |" in s
    assert "| pa reads | 6 |" in s
    # member dist5_exon 100, 700, 1000, 1500, 400, 2000
    assert "| Members within 500 nt | 2 |" in s
    assert "| Members within 1,000 nt | 4 |" in s
    assert "| Members within 2,000 nt | 6 |" in s
    assert "| Molecules with a member within 1,000 nt | 3 |" in s
    # A 0-4: only a20 4 (hexamer); A 18-20: only a20 18 (no hexamer); both on the band edge
    assert "| Hexamer rate at A 0-4 | 1.000 (n 1) |" in s
    assert "| Hexamer rate at A 18-20 | 0.000 (n 1) |" in s
    # region by label: 3UTR holds templated, ambiguous and non-templated, 1 / (1 + 1)
    assert "| 1417 | 3UTR | 1 | 1 | 1 | 3 | 0.500 |" in s
    assert "| 1417 | antisense | 1 | 0 | 0 | 1 | 1.000 |" in s
    assert "| 1417 | CDS | 0 | 0 | 0 | 0 | n/a |" in s
    assert "1179: not run (missing 1179/truth/molecules_labeled.parquet, 1179/truth/truth_reads.parquet)" in s
    assert "| 1417 | 0.458 | 0.069 | 0.772 | 0.391 | 4 | 1 | 1 |" in s
    assert "over all labeled molecules" in s
    assert "| Library | Estimated tail-primed share | Hexamer rate templated | Hexamer rate non-templated | " \
           "Hexamer rate ambiguous | n_templated | n_nontemplated | n_ambiguous |" in s


def test_truth_section_without_ambiguous_estimate(base):
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text())
    del sel["ambiguous_tail_share"]
    p.write_text(json.dumps(sel))
    assert "not run (eval/selection.json has no ambiguous_tail_share)" in m2report.truth_section(base)
    p.unlink()
    assert "Ambiguous molecules" in m2report.truth_section(base)
    assert "not run (missing eval/selection.json)" in m2report.truth_section(base)


def test_grid_section(base):
    s = m2report.grid_section(base)
    assert "| Kind | Setting | 1417 |" in s
    assert "| records | Records fetched | 85,000,000 |" in s
    assert "| records | Primary records | 83,140,741 |" in s
    # over primary records, not the 85,000,000 records (20.08%, 4.39%, 58.82%): 17,068,514 / 83,140,741 = 20.53%;
    # 3,729,928 / 83,140,741 = 4.49%; 50,000,000 / 83,140,741 = 60.14%
    assert "| published | c300_l10_m1_i | 17,068,514 (20.5%) |" in s
    assert "| published | c300_l10_m1_e | 3,729,928 (4.5%) |" in s
    assert "| published | c275_l9_m1_e | n/a |" in s
    assert "| remove-all control | c300_lnone_m0_i | 50,000,000 (60.1%) |" in s
    assert "| remove-all control | c150_lnone_m0_e | n/a |" in s
    assert "1179: not run (missing 1179/grid/summary.json)" in s


def test_selection_section(base):
    s = m2report.selection_section(base)
    assert "| Mode | Chosen setting (on 1417) |" in s
    assert "| -i | c300_l10_m1_i | 0.237 | 0.990 | 0.997 | 120 / 189 |" in s
    assert "| default | none reached specificity 0.95 | n/a | n/a | n/a | 0 / 189 |" in s
    assert "| c150_l3_m2_e | 0.410 | 0.900 |" in s and "| c300_l10_m1_e | 0.036 | 0.994 |" in s
    # chosen and published at once; molecule-level CIs only (the read-level row is not used)
    assert ("| c300_l10_m1_i | chosen (-i), published | 1417 | 41,369 | 2,786 | 0.237 [0.207, 0.259] | "
            "0.990 [0.985, 0.993] | 0.997 [0.996, 0.998] |") in s
    # 1179 has no bootstrap rows; 300 / 1,000 and 95 / 100
    assert "| c300_l10_m1_i | chosen (-i), published | 1179 | 1,000 | 100 | 0.300 [n/a, n/a] | 0.950 [n/a, n/a] |" in s
    # sens 26,356 / 41,369 = 0.637 with its CI; spec and prec without bootstrap rows
    assert "| c375_l3_m2_i | published | 1417 | 41,369 | 2,786 | 0.637 [0.562, 0.690] | 0.861 [n/a, n/a] |" in s
    assert "| c375_l3_m2_i | published | 1179 | n/a | n/a | n/a | n/a | n/a |" in s
    assert "| c300_l10_m1_i | 1417 | 0.237 / 0.990 | 0.259 / 0.986 | 0.278 / 0.975 |" in s
    # molecule-level clusters only (the read-level row, first in the file, has 13,500)
    assert "clusters 1417 13,271." in s
    assert "| 1417 | 2,000 | 77,966 | 49,827 | 3 | 2 |" in s
    assert "| 500 | -i | c300_l10_m1_i | 0.250 | 0.991 | 118 / 189 |" in s
    assert "| 500 | default | none | n/a | n/a | 0 / 189 |" in s
    assert "| Templated min | Non-templated max | Mode | Chosen | Qualifying | 1417 sens / spec | 1179 sens / spec |" in s
    # sens / spec, not prec (0.970)
    assert "| 7 | 2 | -i | c300_l10_m1_i | 100 | 0.200 / 0.990 | n/a |" in s
    assert "| 10 | 6 | default | none | 0 | n/a | n/a |" in s
    # intergenic: 5,182 of 41,369 templated units, at most 500 removed (c375_l3_m2_i; c300_l10_m1_i removes none) at cap
    # 1,000 and 8/4 (the cap 500 and 7/2 rows, with 99 and 77 removed, are ignored); ceiling (41,369 - 5,182 + 500) / 41,369
    # = 0.8868; non-templated tn + fp = 125 + 3 = 128, at most 8 removed
    assert "| 1417 | 41,369 | 5,182 | 500 | 0.887 | 128 | 8 |" in s
    # (0.95 x 2,786 - 128) / (2,786 - 128) = 2,518.7 / 2,658 = 0.9476
    assert ("specificity >= 0.95 over all 2,786 non-templated units on 1417 is at least (0.95 x 2,786 - 128) / 2,658 = "
            "0.948 over the 2,658 genic ones") in s


def test_selection_section_reads_threshold_and_library_from_the_file(base):
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text())
    sel["min_spec"] = 0.9
    p.write_text(json.dumps(sel))
    s = m2report.selection_section(base)
    assert "| default | none reached specificity 0.9 | n/a | n/a | n/a | 0 / 189 |" in s
    # (0.9 x 2,786 - 128) / 2,658 = 2,379.4 / 2,658 = 0.8952
    assert "specificity >= 0.9 over all 2,786 non-templated units on 1417 is at least (0.9 x 2,786 - 128) / 2,658 = 0.895" in s
    # no candidate in a mode: no threshold claim and no frontier; another selection library in the headers
    sel["modes"]["e"]["selection"] |= {"n_candidates": 0, "frontier": []}
    sel["selection_library"] = "1419"
    p.write_text(json.dumps(sel))
    s = m2report.selection_section(base)
    assert "| default | no candidates | n/a | n/a | n/a | 0 / 0 |" in s
    assert "Frontier" not in s
    assert "| Mode | Chosen setting (on 1419) |" in s
    assert "Member cap sensitivity (selection rerun on 1419 at each cap)" in s
    assert "Label cut-off sweep (selection on 1419 at each" in s
    # metrics.parquet has no 1419 rows, so there is no genic-specificity line
    assert "genic ones" not in s


def test_selection_section_without_bootstrap_or_metrics(base):
    (base / "eval/bootstrap.parquet").unlink()
    (base / "eval/metrics.parquet").unlink()
    s = m2report.selection_section(base)
    assert "| -i | c300_l10_m1_i | 0.237 | 0.990 | 0.997 | 120 / 189 |" in s
    assert "| c300_l10_m1_i | chosen (-i), published | 1417 | 41,369 | 2,786 | 0.237 [n/a, n/a] |" in s
    assert "Bootstrap: not run (missing eval/bootstrap.parquet)" in s
    assert "Intergenic molecules: not run (missing eval/metrics.parquet)" in s


def test_selection_section_chosen_setting_outside_the_published_ones(base):
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text())
    sel["modes"]["i"]["selection"]["tag"] = "c450_l9_m1_i"
    sel["modes"]["i"]["metrics"] = {"1417": I1_1417}
    p.write_text(json.dumps(sel))
    s = m2report.selection_section(base)
    # its metrics come from modes, not from published; no bootstrap rows for it
    assert "| c450_l9_m1_i | chosen (-i) | 1417 | 41,369 | 2,786 | 0.237 [n/a, n/a] | 0.990 [n/a, n/a] |" in s
    assert "| c450_l9_m1_i | 1179 | n/a | n/a | n/a |" in s
    assert "| c300_l10_m1_i | published | 1417 | 41,369 | 2,786 | 0.237 [0.207, 0.259] |" in s


def test_strata_section(base):
    s = m2report.strata_section(base)
    # spec 2,757 / 2,786 = 0.990; sens 100 / 2,000, 1,000 / 6,400 = 0.156, 8,695 / 32,969 = 0.264
    assert "| c300_l10_m1_i | 1417 | 0.990 | 0.050 | 0.156 | 0.264 |" in s
    # intergenic 0 / 5,182 and 125 / 128; 3UTR 5,000 / 15,000 and 2,000 / 2,020; other regions absent
    assert "| c300_l10_m1_i | 1417 | 0.000 / 0.977 | n/a | n/a | n/a | n/a | 0.333 / 0.990 | n/a |" in s
    assert "| Setting | Library | intergenic | intron | ncexon | 5UTR | CDS | 3UTR | antisense |" in s
    assert "| 1417 | a_band | 8-9 | 2,000 | 0 |" in s
    assert "| 1417 | region | 3UTR | 15,000 | 2,020 |" in s
    assert "| 1417 | region | intergenic | 5,182 | 128 |" in s
    (base / "eval/selection.json").unlink()
    s = m2report.strata_section(base)
    assert "chosen settings: not run (missing eval/selection.json)" in s
    assert "| c300_l10_m1_i | 1417 | 0.990 | 0.050 | 0.156 | 0.264 |" in s


def test_hexamer_section(base):
    s = m2report.hexamer_section(base)
    assert "| c375_l3_m2_i | 1417 | all | 28,762 | 0.061 | 20,174 | 0.229 |" in s
    assert "| c375_l3_m2_i | 1417 | non-templated | 387 | 0.305 | 2,399 | 0.836 |" in s
    assert "| c375_l3_m2_i | 1417 | ambiguous | 0 | n/a | 10 | 0.400 |" in s
    assert "| 1 | 0.900 |" not in s


def test_spikein_section(base):
    s = m2report.spikein_section(base)
    assert "| 1417 | internal | 2 | 1 | 1 | 4 |" in s
    assert "| 1417 | tail | 2 | 1 | 2 | 5 |" in s
    assert "| 1417 | other | 0 | 1 | 0 | 1 |" in s
    # internal and templated 2 + tail and non-templated 2 over the 7 internal or tail molecules labeled T or NT (the
    # ambiguous ones left out); SIRV_E0 alone: internal T 1 (a20 15), internal NT 1 (a20 1), tail T 1 (a20 9): 1 of 3
    assert "1417: agreement 0.571 (4 of 7), SIRV_E0 only 0.333 (1 of 3)" in s
    assert "forced" in s
    # the ambiguous internal molecules (ERCC none, SIRV_E0 a20 7) count in the denominator, not as templated
    assert "| 1417 | all | 4 | 2 (0.500) | 5 | 2 (0.400) |" in s
    assert "| 1417 | ERCC | 1 | 1 (1.000) | 2 | 1 (0.500) |" in s
    assert "| 1417 | SIRV_E0 | 3 | 1 (0.333) | 1 | 0 (0.000) |" in s
    assert "| 1417 | long_SIRV | 0 | 0 (n/a) | 2 | 1 (0.500) |" in s
    assert "1419: not run (missing 1419/spikein/truth/molecules_labeled.parquet)" in s
    # molecule level only: 10 / 40, 50 / 52; ERCC 5 / 10, 20 / 20; the read-level row (9 / 10) is not used
    assert "| 1417 | c300_l10_m1_i | 0.250 / 0.962 | 0.500 / 1.000 | n/a | n/a |" in s
    (base / "eval/spikein_metrics.parquet").unlink()
    assert "Metrics per module: not run (missing eval/spikein_metrics.parquet)" in m2report.spikein_section(base)


def test_ercc_section(base):
    s = m2report.ercc_section(base)
    assert s.startswith("## 8. ERCC count accuracy (provisional)")
    assert f"Provisional (1417): {NOTE}" in s
    assert "| Library | Setting | n | pearson_log10 | spearman | umis | umis_kept_share |" in s
    assert "| 1417 | unfiltered | 61 | 0.912 | 0.888 | 52,000 | 1.000 |" in s
    assert "| 1417 | c300_l10_m1_i | 61 | 0.920 | 0.890 | 50,502 | 0.971 |" in s
    assert "| 1417 | c375_l3_m2_i | 3 | n/a | n/a | 3 | 0.000 |" in s
    assert "| 1419 | unfiltered | 60 | 0.900 | 0.800 | 4,000 | 1.000 |" in s


def test_long_read_section(base):
    s = m2report.long_read_section(base)
    # dropped 178,467 / 1,610,467 = 11.08%
    assert "| BD67 | 2 | 1,735,308 | 14,097 | 110,744 | 1,610,467 | 1,432,000 | 77,500 | 113 | 100,854 | 11.1% | 0 |" in s
    # per read over sites with an accepted read (a20 3 and 12 have none), cut-offs inclusive. At 8/4, templated a20 15 (3, no
    # hexamer), 9 (1, hexamer), 8 (1, no), 14 (1, hexamer): 2 / 6 = 0.333; non-templated a20 2 (2, hexamer), 1 (2, no), 4 (2,
    # hexamer): 4 / 6 = 0.667; ratio 2.00. At 14/8, templated a20 15 and 14: 1 / 4 = 0.25; non-templated a20 2, 1, 4 and 8:
    # 4 / 7 = 0.571; ratio 2.29. Exclusive cut-offs would give 0.400, 0.500, 0.000 and 0.667.
    assert "| BD67 | 8/4 | 28,473 / 6,469 / 7,228 | 0.106 | 0.541 | 0.443 | 5.12 | True | 0.333 | 0.667 | 2.00 |" in s
    assert "| BD67 | 14/8 | 11,551 / 16,279 / 14,340 | 0.046 | 0.456 | 0.200 | 9.95 | True | 0.250 | 0.571 | 2.29 |" in s
    assert "about 7%" in s
    assert "| BD67 | 16,408 | 0.615 | 0.491 / 0.907 / 0.805 | 0.993 / 0.996 / 0.994 |" in s
    assert "BD70: not run (missing BD70/long_read_acceptance.json)" in s


def test_burden_section(base):
    s = m2report.burden_section(base)
    assert "| Stratum kind | Stratum | 1417 | cells |" in s
    assert "| all | all | 0.819 [0.804, 0.835] (144,474) | 0.900 [0.880, 0.920] (1,000) |" in s
    assert "| region | 3UTR | 0.507 [0.479, 0.532] (31,129) | n/a |" in s
    assert "| region | intron | n/a | n/a |" in s
    assert "| has_ts | True | 0.724 [0.638, 0.794] (724) | n/a |" in s
    assert "| all | all | 0.916 | 0.947 |" in s
    # frac with its own interval
    assert "| 14-20 | 0.278 [0.271, 0.285] | n/a |" in s
    assert "within 5 nt on the same strand" in s
    assert "| 1417_vs_1419 | templated | 95,920 | 0 | 0 | 500 |" in s
    assert "| 1417_vs_1419 | ambiguous | 100 | 11,219 | 0 | 0 |" in s
    # rows in label order, not alphabetical
    assert s.index("| 1417_vs_1419 | templated |") < s.index("| 1417_vs_1419 | ambiguous |") < \
        s.index("| 1417_vs_1419 | non-templated |")
    # 114,000 sites, 113,500 matched (99.56%), 113,400 of them with the same label (99.91%)
    assert "| 1417_vs_1419 | 114,000 | 113,500 (99.6%) | 113,400 (99.9%) |" in s
    assert "| 1417_vs_1419 | 1,212 | 1.000 | 0.00e+00 |" in s
    assert "| 1417 | E | 0.421 | 0.407 |" in s
    (base / "burden/same_gene.parquet").unlink()
    assert "Same gene: not run (missing burden/same_gene.parquet)" in m2report.burden_section(base)


def test_burden_section_without_a_pair(base):
    # stage 18 with one library (1417 alone): empty same-site and same-gene frames
    pd.DataFrame(columns=["pair", "label_a", "label_b", "n"]).to_parquet(base / "burden/same_site.parquet", index=False)
    pd.DataFrame([], columns=["pair", "n_genes", "spearman_rho", "p_value"]).to_parquet(base / "burden/same_gene.parquet",
                                                                                         index=False)
    s = m2report.burden_section(base)
    assert ("Same site: no pair in burden/same_site.parquet (stage 18 needs 1417 and 1419, or nuclei and cells)") in s
    assert ("Same gene: no pair in burden/same_gene.parquet (stage 18 needs 1417 and 1419, or nuclei and cells)") in s
    assert "| Pair |" not in s
    assert "| 1417 | E | 0.421 | 0.407 |" in s
    assert "Same site" in m2report.render(base)


def test_burden_section_polyasite(base):
    s = m2report.burden_section(base)
    assert "does not screen out candidate sites with A-rich genome downstream" in s
    assert ("The null row takes every molecule of the group with its site moved 1,000 nt upstream in transcript sense on the "
            "same ref and strand; its share within 10 nt is the share expected by chance at this atlas density.") in s
    assert "| Label | Molecules | 0-10 | 11-50 | 51-200 | >200 | no atlas site on this ref and strand |" in s
    one, cells = s.split("Group 1417:")[1].split("Group cells:")
    # 9,000 / 118,278 = 0.0761, 20,000 / 118,278 = 0.1691, 40,000 / 118,278 = 0.3382, 44,000 / 118,278 = 0.3720,
    # 5,278 / 118,278 = 0.0446
    assert "| templated | 118,278 | 0.076 | 0.169 | 0.338 | 0.372 | 0.045 |" in one
    # 3,000 / 15,374 = 0.1951, 4,000 / 15,374 = 0.2602, 3,674 / 15,374 = 0.2390, 1,700 / 15,374 = 0.1106
    assert "| ambiguous | 15,374 | 0.195 | 0.195 | 0.260 | 0.239 | 0.111 |" in one
    # 4,000 / 10,822 = 0.3696, 2,000 / 10,822 = 0.1848, 2,681 / 10,822 = 0.2477, 141 / 10,822 = 0.0130
    assert "| non-templated | 10,822 | 0.370 | 0.185 | 0.185 | 0.248 | 0.013 |" in one
    # 144,474 molecules: 31,500 = 0.2180, 40,000 = 0.2769, 50,000 = 0.3461, 15,000 = 0.1038, 7,974 = 0.0552
    assert "| null (sites 1 kb upstream) | 144,474 | 0.218 | 0.277 | 0.346 | 0.104 | 0.055 |" in one
    # 1,000 / 18,577 = 0.0538, 3,000 / 18,577 = 0.1615, 6,000 / 18,577 = 0.3230, 7,577 / 18,577 = 0.4079
    assert "| templated, A band 8-9 | 18,577 | 0.054 | 0.161 | 0.323 | 0.408 | 0.054 |" in one
    # 3,000 / 40,103 = 0.0748, 7,000 = 0.1746, 14,000 = 0.3491, 14,103 = 0.3517, 2,000 = 0.0499
    assert "| templated, A band 14-20 | 40,103 | 0.075 | 0.175 | 0.349 | 0.352 | 0.050 |" in one
    # rows: labels in page order, the null, then the bands
    assert one.index("| templated |") < one.index("| ambiguous |") < one.index("| non-templated |") < \
        one.index("| null (sites 1 kb upstream) |") < one.index("| templated, A band 8-9 |") < \
        one.index("| templated, A band 10-13 |") < one.index("| templated, A band 14-20 |")
    assert "| templated | 1,000 | 0.100 | 0.200 | 0.300 | 0.300 | 0.100 |" in cells
    assert "| non-templated | 0 | n/a | n/a | n/a | n/a | n/a |" in cells
    assert "| ambiguous | 20 | 0.250 | 0.250 | 0.000 | 0.300 | 0.200 |" in cells
    # 255, 255, 255, 153, 102 of 1,020
    assert "| null (sites 1 kb upstream) | 1,020 | 0.250 | 0.250 | 0.250 | 0.150 | 0.100 |" in cells
    assert "| templated, A band 8-9 | 0 | n/a | n/a | n/a | n/a | n/a |" in cells
    (base / "burden/polyasite.parquet").unlink()
    s = m2report.burden_section(base)
    assert "PolyASite proximity: not run (missing burden/polyasite.parquet)" in s and "Group 1417:" not in s


def test_validation_section(base):
    s = m2report.validation_section(base)
    assert "| 1417 | c275_l9_m1_i | 17,000,000 | 17,000,000 | 0 | 0 | True | 1.95 | 15.49 | 60.50 |" in s
    assert "| 1417 | c300_l10_m1_i | 17,068,514 | 17,068,514 | 0 | 0 | True | 1.95 | 15.50 | n/a |" in s
    assert "1419: no completed original run" in s
    assert "1179: not in exact_match_m2.json; the 1179 grid has no original-versus-reimplementation evidence" in s
    # 20 + 412 + 160 + 80 = 672 s
    assert "| c300_lnone_m0_i | chr21, chr22 | 2,781,871 | 1,894,573 | 1,894,573 | 0 | 0 | True | 412 | 0.26 | 672 |" in s
    assert "Compared: 3; all identical: True" in s
    assert "Removed-record sets (every record the original fetches" in s


def test_validation_section_with_the_1179_subset(base):
    p = base / "validation/exact_match_m2.json"
    em = json.loads(p.read_text())
    r = {"setting": I1, "n_original": 2000000, "n_reimpl": 2000000, "only_original": 0, "only_reimpl": 0, "identical": True,
         "examples": [], "original_elapsed_s": 1800.0, "original_max_rss_gib": 3.25, "original_peak_tree_rss_gib": 6.5,
         "compare_s": 300.0}
    em["1179_subset"] = {"refs": ["chr21", "chr22"], "n_records": 18382751,
                         "createTRANS": {m: {"elapsed_s": 900.0, "max_rss_gib": 0.3, "exit_status": 0, "peak_tree_rss_gib": 0.3}
                                         for m in "ie"},
                         "settings": {I1: r, E1: r | {"setting": E1, "only_original": 2, "identical": False}}}
    p.write_text(json.dumps(em))
    s = m2report.validation_section(base)
    # the subset rows join the library table: 1800 s = 0.50 h
    assert "| 1179 (chr21, chr22) | c300_l10_m1_i | 2,000,000 | 2,000,000 | 0 | 0 | True | 0.50 | 3.25 | 6.50 |" in s
    assert "| 1179 (chr21, chr22) | c300_l10_m1_e | 2,000,000 | 2,000,000 | 2 | 0 | False | 0.50 | 3.25 | 6.50 |" in s
    assert "1179: original tool on the chr21, chr22 subset only (18,382,751 records)" in s
    assert "the 1179 grid has no original-versus-reimplementation evidence" not in s
    # the nuclei rows are unchanged
    assert "| 1417 | c275_l9_m1_i | 17,000,000 | 17,000,000 | 0 | 0 | True | 1.95 | 15.49 | 60.50 |" in s


def test_validation_section_on_a_stale_file(base):
    # the schema an earlier draft left in the real OUT: no library keys, no none_control, no totals
    (base / "validation/exact_match_m2.json").write_text(json.dumps({"none_control_chr21_chr22": {"error": "createTRANS failed"}}))
    s = m2report.validation_section(base)
    assert "1417: not in exact_match_m2.json" in s
    assert "Remove-all control: not run (no none_control entry in validation/exact_match_m2.json)" in s
    assert "Compared: n/a; all identical: n/a" in s


def test_render_and_script(base, tmp_path):
    text = m2report.render(base)
    s = _sections(text)
    assert sorted(s) == list(range(1, 12))
    assert text.startswith("# Milestone 2 results (generated by scripts/21_m2_summary.py)")
    assert "not run (missing eval/selection.json)" not in text
    out = tmp_path / "docs" / "m2_results.md"
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "21_m2_summary.py"), "--base", str(base), "--out", str(out)],
                       capture_output=True, text=True, env=env, check=True)
    assert out.read_text() == text
    line = json.loads(r.stdout.strip().splitlines()[-1])
    assert line == {"out": str(out), "sections": 11, "not_run": text.count("not run")}
