import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pafbench import m2report
from pafbench.config import MEMBER_CAP
from pafbench.evaluate import METRIC_COLUMNS
from pafbench.labels import a_band
from pafbench.molecules import MOL_COLUMNS
from pafbench.validation import parse_tag

REPO = Path(__file__).resolve().parents[1]
I1, I3, E1 = "c300_l10_m1_i", "c375_l3_m2_i", "c300_l10_m1_e"
C1, C2 = "c150_lnone_m0_i", "c600_lnone_m0_e"
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


def _purity(scale=1):
    """label_purity of one library as stage 17 writes it; scale lifts the hexamer rates (a cell-like library)."""
    rates = {"0-4": 0.6, "5-7": 0.2, "8-9": 0.09, "10-13": 0.06, "14-20": 0.04}
    bands = {b: {"n": 100 * (k + 1), "hexamer_rate": min(1.0, r * scale), "tail_primed_share":
                 min(1.0, max(0.0, (r * scale - 0.04) / (0.6 - 0.04)))} for k, (b, r) in enumerate(rates.items())}
    return {"bands": bands, "reference": {"n": 500, "hexamer_rate": 0.04},
            "classes": {"templated": {"n": 1200, "hexamer_rate": 0.056, "tail_primed_share": 0.0279 * scale},
                        "ambiguous": {"n": 200, "hexamer_rate": 0.2, "tail_primed_share": 0.2857}},
            "n_labeled": 1500, "templated_share": 0.8, "templated_share_corrected": 0.8 * (1 - 0.0279 * scale)}


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
    _write(b / "1417/pa_sites.parquet", pd.DataFrame({"qname": [f"s{i}" for i in range(8)]}))
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
        "ambiguous_tail_share": {"1417": {"estimate": 0.458, "estimate_ref14": 0.481, "rate_templated": 0.069,
                                          "rate_nontemplated": 0.772, "rate_ambiguous": 0.391, "rate_ref14": 0.04,
                                          "n_templated": 4, "n_nontemplated": 1, "n_ambiguous": 1}},
        "label_purity": {"1417": _purity(), "1179": _purity(scale=3)},
        "cells_tuned": {"1179": {
            "e": {"introns": False, "selection": _sel("c250_l7_m2_e", 0.2232, 0.9507, 0.93, 168),
                  "metrics": {"1179": {"molecule": _lvl(40, 139, 58, 3)}}},
            "i": {"introns": True, "selection": _sel("c450_l5_m1_i", 0.4782, 0.9501, 0.9658, 155),
                  "metrics": {"1179": {"molecule": _lvl(478, 522, 95, 5)}}}}},
    }
    _write(b / "eval/selection.json", sel)
    # the read-level row comes first and has its own cluster count (clusters without a unit at a level are left out)
    boot = [("1417", 1000, "read", I1, "sens", 0.278, 0.1, 0.9, 13500),
            ("1417", 1000, "molecule", I1, "sens", 0.2368, 0.2074, 0.2588, 13271),
            ("1417", 1000, "molecule", I1, "spec", 0.9896, 0.9854, 0.9932, 13271),
            ("1417", 1000, "molecule", I1, "prec", 0.9970, 0.9959, 0.9981, 13271),
            ("1417", 1000, "molecule", I3, "sens", 0.6371, 0.5623, 0.6904, 13271),
            ("1417", 1000, "molecule", C1, "spec", 0.1436, 0.1301, 0.1572, 13271)]
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
                 _metric("1417", tag, "region", "antisense", 0 if tag == I1 else 100, 2000 if tag == I1 else 1900,
                         50 if tag == I1 else 48, 0 if tag == I1 else 2),
                 _metric("1417", tag, "region", "3UTR", 5000, 10000, 2000, 20)]
    rows += [_metric("1417", I1, "expr_tertile", "low", 100, 900, 40, 1), _metric("1417", I1, "expr_tertile", "high", 300, 700, 90, 10),
             _metric("1417", I1, "expr_tertile", "no_gene", 0, 500, 20, 0), _metric("1417", I1, "has_ts", "True", 5, 95, 10, 0),
             _metric("1417", I1, "has_ts", "False", 9790, 31479, 2747, 29),
             _metric("1417", I1, "a_band", "0-4", 0, 0, 2700, 100, level="read"),
             _metric("1417", I1, "a_band", "14-20", 12000, 24000, 0, 0, level="read"),
             _metric("1417", I1, "a_band", "0-4", 0, 0, 2650, 150, level="molecule_majority"),
             _metric("1417", I1, "a_band", "8-9", 300, 1700, 0, 0, level="molecule_majority")]
    rows += [_metric("1417", I1, "all", "all", 8000, 33369, 2700, 86, level="molecule_weighted"),
             _metric("1417", C1, "all", "all", 30000, 11369, 400, 2386), _metric("1417", C2, "all", "all", 20000, 21369, 900, 1886),
             _metric("1417", C1, "all", "all", 1, 1, 1, 1, cap=500)]
    # rows the molecule-level tables must ignore: another cap, other cut-offs, the read level
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
             dict(_metric("1417", I1, "all", "all", 9, 1, 0, 0, level="read")),
             dict(_metric("1417", C1, "all", "all", 50, 50, 984, 16)),
             dict(_metric("1417", C1, "all", "all", 60, 40, 156, 844, level="read"))]
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
        "selection": {"e": None, "i": I1}, "barcodes": "cells", "ercc_contigs": 92, "reads": 120000, "ercc_detected": 61,
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
        "1417": {"c275_l9_m1_i": em | {"original_elapsed_s": 7020.0, "original_max_rss_gib": 15.486, "original_peak_tree_rss_gib": 60.5,
                                       "n_grid": 17000000, "grid_equal": True},
                 I1: em | {"n_original": 17068514, "n_reimpl": 17068514, "original_elapsed_s": 7020.0,
                           "original_max_rss_gib": 15.5, "original_peak_tree_rss_gib": None, "n_grid": 17068514,
                           "grid_equal": True}},
        "1419": {},
        "complete": False,
        "missing": {"1417": [], "1419": [{"tag": "c275_l9_m1_i", "exit_status": 137, "reason": "killed or failed"},
                                         {"tag": I1, "exit_status": None, "reason": "no output"}]},
        "engine_tie": {"1417": {"rows": 1000, "tags": ["c275_l9_m1_i", I1], "mismatches": {"c275_l9_m1_i": 0, I1: 3},
                                "missing_rows": 2}, "1419": None},
        "reused": {"1417": True, "1419": False},
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
    return {int(p.split(".", 1)[0]): p for p in parts if p.split(".", 1)[0].isdigit()}


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
    head = text.split("\n## 1.")[0]
    assert "## Headline" in head and "Headline: not run (missing eval/selection.json)" in head
    assert "Validation against the original tool: not run (missing validation/exact_match_m2.json)" in head
    # every missing input is listed once in the header, with the number of places
    assert "Missing inputs:" in head and "eval/selection.json" in head
    # a base that does not exist at all behaves the same
    assert m2report.render(tmp_path / "absent") == text.replace(str(tmp_path), str(tmp_path / "absent"))


def test_runs_section_skips_kept_aside_folders(base):
    n = m2report.runs_section(base).count("stage13.mem.json")
    aside = base / "1417/grid_pre_fix_20261010/stage13.mem.json"
    aside.parent.mkdir()
    aside.write_text((base / "1417/grid/stage13.mem.json").read_text())
    s = m2report.runs_section(base)
    assert s.count("stage13.mem.json") == n and "pre_fix_20261010" not in s
    assert "folders kept aside from earlier runs (name containing `_pre_fix_`) are left out" in s


def test_runs_section(base):
    s = m2report.runs_section(base)
    assert "memwatch samples every 2 s (interval_s in the files)" in s
    assert "upper bound" not in s and "a peak between two samples is missed" in s
    (base / "1417/grid/stage13.mem.json").unlink()
    (base / "1417/original/c275_l9_m1_i.mem.json").unlink()
    assert "no file is available here" in m2report.runs_section(base) and "samples every" not in m2report.runs_section(base)
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
    assert "| With a pa read with ts > 0 (has_ts) | 1 |" in s
    assert "With a ts read" not in s
    # has_ts is defined as truth.py sets it, with the 10x wording and URL for the tag
    assert ("has_ts: a pa read of the molecule has a 10x ts tag above 0. Cell Ranger documentation, Barcoded BAM Tags "
            "(https://www.10xgenomics.com/support/software/cell-ranger/latest/analysis/outputs/cr-outputs-bam): ts is \"The number "
            "of template switch oligo (TSO) nucleotides trimmed from the 5' end of read 2.\" Our reading: such a read reaches the "
            "5' end of the cDNA.") in s
    assert "| Multi-gene (gene = alphabetically first id) | 1 |" in s
    assert "| pa reads in labeled molecules | 6 |" in s
    # pa_sites.parquet row count from the parquet metadata
    assert "| pa reads in pa_sites.parquet | 8 |" in s
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
    assert "| 1417 | 0.458 | 0.481 | 0.069 | 0.772 | 0.391 | 0.040 | 4 | 1 | 1 |" in s
    assert "over all labeled molecules" not in s
    # the fixture selection.json has no selection_population: all barcodes, as the label purity caption
    assert "tail-primed share over the labeled molecules of all barcodes, with and without a member read, not only" in s
    assert "The truth table and the region table below cover all barcodes (every row of molecules_labeled.parquet)" in s
    assert "| Library | Estimated tail-primed share (reference: templated class) | " \
           "Estimated tail-primed share (reference: a20 >= 14) | Hexamer rate templated | Hexamer rate non-templated | " \
           "Hexamer rate ambiguous | Hexamer rate a20 >= 14 | n_templated | n_nontemplated | n_ambiguous |" in s


def test_truth_section_label_purity(base):
    s = m2report.truth_section(base)
    assert "| Library | A band | Molecules | Hexamer rate | Estimated tail-primed share |" in s
    # 1417 band 8-9: 300 molecules, rate 0.09, (0.09 - 0.04) / (0.6 - 0.04) = 0.089; 1179 (rates x 3): 0.27, 0.411
    assert "| 1417 | 8-9 | 300 | 0.090 | 0.089 |" in s
    assert "| 1179 | 8-9 | 300 | 0.270 | 0.411 |" in s
    assert "| 1417 | 14-20 | 500 | 0.040 | 0.000 |" in s
    assert "| 1417 | templated class | 1,200 | 0.056 | 0.028 |" in s
    assert "| 1179 | templated class | 1,200 | 0.056 | 0.084 |" in s
    assert "hexamer rate of the band against the rate at 14 or more A's, scaled by the rate at 0 to 4 A's" in s
    assert "an estimate" in s
    assert "over the labeled molecules of all barcodes" in s
    sel = json.loads((base / "eval/selection.json").read_text()) | {"selection_population": "cells"}
    (base / "eval/selection.json").write_text(json.dumps(sel))
    assert "over the labeled molecules of called cells" in m2report.truth_section(base)
    assert "labeled molecules of called cells)" in m2report.burden_section(base)
    assert "tail-primed share over the labeled molecules of called cells, with and without a member read" in \
        m2report.truth_section(base)
    assert "ACTB" not in s and "1,976" not in s and "annotated 3' end" not in s
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text())
    del sel["label_purity"]
    p.write_text(json.dumps(sel))
    assert "Label purity: not run (eval/selection.json has no label_purity)" in m2report.truth_section(base)
    p.unlink()
    assert "Label purity: not run (missing eval/selection.json)" in m2report.truth_section(base)


def _cells_truth(base, scale):
    """1179 labeled molecules for the band 8-9 site table: 5 sites in called cells, a larger one only in other barcodes, one
    band 5-7 site; sel label_purity of 1179 lifted by scale."""
    rows, k = [], 0
    for ref, site, strand, gene, n, n_hex in (("chr8", 5000, "-", "G1", 7, 7), ("chr12", 600, "+", "G2", 6, 3), ("chr7", 70, "-", "", 5, 0),
                                              ("chr1", 10, "+", "G4", 4, 4), ("chr2", 20, "+", "G5", 3, 1), ("chr3", 30, "+", "G6", 2, 2)):
        rows += [dict(ref=ref, site=site, strand=strand, gene=gene, a_band="8-9", hexamer=i < n_hex, in_cell=True) for i in range(n)]
    rows += [dict(ref="chr9", site=1, strand="+", gene="G7", a_band="8-9", hexamer=True, in_cell=False) for _ in range(50)]
    rows += [dict(ref="chr4", site=1, strand="+", gene="G8", a_band="5-7", hexamer=True, in_cell=True) for _ in range(30)]
    df = pd.DataFrame(rows)
    _write(base / "1179/truth/molecules_labeled.parquet", df)
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text())
    sel["label_purity"]["1179"] = _purity(scale=scale)
    sel["selection_population"] = "cells"
    p.write_text(json.dumps(sel))


def test_truth_section_band_8_9_is_computed(base):
    _cells_truth(base, scale=5)
    s = m2report.truth_section(base)
    # band 8-9 share: 1417 (0.09 - 0.04) / 0.56 = 0.089; 1179 (0.45 - 0.04) / 0.56 = 0.732
    assert "| Library | Molecules in A band 8-9 | Hexamer rate | Estimated tail-primed share |" in s
    assert "| 1417 | 300 | 0.090 | 0.089 |" in s and "| 1179 | 300 | 0.450 | 0.732 |" in s
    assert ("By this estimate most A band 8-9 molecules of 1179 are real poly(A) sites that follow genome-encoded A's, and the 8/4 "
            "rule labels them templated, so read 1179 by A band.") in s
    assert "molecules of 1417 are real poly(A)" not in s and "read 1417 by A band" not in s
    # five largest sites of called cells, in the order of their molecule count; the 50 molecules of other barcodes and the
    # band 5-7 site are not counted; no gene gives n/a
    t = s.split("Largest A band 8-9 sites of 1179 in called cells")[1]
    assert "| Ref | Site | Strand | Gene id | Molecules | Share with the hexamer |" in t
    rows = [l for l in t.splitlines() if l.startswith("| chr")]
    assert rows[:5] == ["| chr8 | 5,000 | - | G1 | 7 | 1.000 |", "| chr12 | 600 | + | G2 | 6 | 0.500 |", "| chr7 | 70 | - | none | 5 | 0.000 |",
                        "| chr1 | 10 | + | G4 | 4 | 1.000 |", "| chr2 | 20 | + | G5 | 3 | 0.333 |"]
    assert not any(l.startswith(("| chr3", "| chr9", "| chr4")) for l in rows)
    assert "ACTB" not in s and "annotated 3' end" not in s
    # no stage 12 population column for the selected population: not run, no numbers
    df = pd.read_parquet(base / "1179/truth/molecules_labeled.parquet").drop(columns="in_cell")
    _write(base / "1179/truth/molecules_labeled.parquet", df)
    assert "Largest A band 8-9 sites of 1179: not run (no in_cell column" in m2report.truth_section(base)
    (base / "1179/truth/molecules_labeled.parquet").unlink()
    assert "Largest A band 8-9 sites of 1179: not run (missing 1179/truth/molecules_labeled.parquet)" in m2report.truth_section(base)
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text())
    del sel["label_purity"]
    p.write_text(json.dumps(sel))
    s = m2report.truth_section(base)
    assert "A band 8-9 tail-primed share: not run (eval/selection.json has no label_purity)" in s


def test_truth_section_without_pa_sites(base):
    (base / "1417/pa_sites.parquet").unlink()
    assert "| pa reads in pa_sites.parquet | n/a |" in m2report.truth_section(base)


def test_truth_section_members_beyond_the_cap(base):
    row = f"| Member candidates beyond {MEMBER_CAP:,} nt (left out of the truth set) |"
    assert f"{row} n/a |" in m2report.truth_section(base)
    # candidates: gene match and dist3_exon >= -25; two of the three lie beyond the cap
    _write(base / "1417/members.parquet", pd.DataFrame({
        "gene_match": [True, True, True, False, True], "dist3_exon": [0, 5, -25, 0, -26],
        "dist5_exon": [MEMBER_CAP, MEMBER_CAP + 1, 5000, 9000, 9000]}))
    assert f"{row} 2 of 3 (66.7%) |" in m2report.truth_section(base)


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
    assert "| c150_l3_m2_e |  | 0.410 | 0.900 |" in s and "| c300_l10_m1_e |  | 0.036 | 0.994 |" in s
    # chosen and published at once; molecule-level CIs only (the read-level row is not used)
    assert ("| c300_l10_m1_i | chosen (-i), published | 1417 | 41,369 | 2,786 | 0.237 [0.207, 0.259] | "
            "0.990 [0.985, 0.993] | 0.997 [0.996, 0.998] |") in s
    # 1179 has no bootstrap rows; 300 / 1,000 and 95 / 100
    assert "| c300_l10_m1_i | chosen (-i), published | 1179 | 1,000 | 100 | 0.300 [n/a, n/a] | 0.950 [n/a, n/a] |" in s
    # sens 26,356 / 41,369 = 0.637 with its CI; spec and prec without bootstrap rows
    assert "| c375_l3_m2_i | published | 1417 | 41,369 | 2,786 | 0.637 [0.562, 0.690] | 0.861 [n/a, n/a] |" in s
    assert "| c375_l3_m2_i | published | 1179 | n/a | n/a | n/a | n/a | n/a |" in s
    # molecule_weighted from metrics.parquet: 8,000 / 41,369 = 0.193, 2,700 / 2,786 = 0.969
    assert "| Setting | Library | Molecule (all reads) | Molecule (majority) | Molecule (weighted 1 / reads in units) | Read |" in s
    assert "| c300_l10_m1_i | 1417 | 0.237 / 0.990 | 0.259 / 0.986 | 0.193 / 0.969 | 0.278 / 0.975 |" in s
    assert "| c375_l3_m2_i | 1417 | 0.637 / 0.861 | 0.656 / 0.848 | n/a | 0.677 / 0.828 |" in s
    # molecule-level clusters only (the read-level row, first in the file, has 13,500)
    assert "clusters 1417 13,271." in s
    assert "| 1417 | 2,000 | 77,966 | 49,827 | 3 | 2 |" in s
    assert "| 500 | -i | c300_l10_m1_i | 0.250 | 0.991 | 118 / 189 |" in s
    assert "| 500 | default | none | n/a | n/a | 0 / 189 |" in s
    assert "| Templated min | Non-templated max | Mode | Chosen | Qualifying | 1417 sens / spec | 1179 sens / spec |" in s
    # sens / spec, not prec (0.970)
    assert "| 7 | 2 | -i | c300_l10_m1_i | 100 | 0.200 / 0.990 | n/a |" in s
    assert "| 10 | 6 | default | none | 0 | n/a | n/a |" in s
    # intergenic and antisense: 5,182 + 2,000 = 7,182 of 41,369 templated units; at most 500 + 100 = 600 removed (c375_l3_m2_i;
    # c300_l10_m1_i removes none) at cap 1,000 and 8/4 (the cap 500 and 7/2 rows are ignored); never removed 7,182 - 600 =
    # 6,582; ceiling (41,369 - 7,182 + 600) / 41,369 = 0.841; non-templated 125 + 3 + 50 + 0 = 178, at most 8 + 2 = 10 removed
    assert "Never removed (intergenic and antisense)" in s
    assert "| 1417 | 41,369 | 7,182 | 600 | 6,582 | 0.841 | 178 | 10 |" in s
    # (0.95 x 2,786 - 178) / (2,786 - 178) = 2,468.7 / 2,608 = 0.9466
    assert ("specificity >= 0.95 over all 2,786 non-templated units on 1417 is at least (0.95 x 2,786 - 178) / 2,608 = "
            "0.947 over the 2,608 genic ones") in s


def test_selection_section_wording_matches_the_code(base):
    s = m2report.selection_section(base)
    # weighted level: 1 / number of the molecule's truth reads in the units (evaluate.units n_reads), not the UMI family size
    assert "Molecule (weighted 1 / reads in units)" in s and "1 / family size" not in s
    assert "each molecule counts 1 / (the number of its truth reads inside the units: pa reads and members within the cap)" in s
    # molecules without a gene are intergenic and antisense (evaluate.gene_clusters)
    assert "a 10 kb locus window for molecules without a gene (intergenic and antisense)" in s
    # candidates per mode from the grid constants: 7 COVLEN x 9 MINSNRLEN x 3 MISM
    assert "189 candidates per mode: 7 COVLEN x 9 MINSNRLEN x 3 MISM" in s
    # column names say which class they count
    assert "Templated removed at most" in s and "Non-templated removed at most" in s and "| Removed at most |" not in s
    assert "Intergenic and antisense non-templated" in s
    # published settings and where each comes from
    assert "| Setting | Source |" in s
    assert "| 300/10/1 | README example, not a tuned value |" in s
    assert "| 275/9/1 | paper Fig. 4A optimum, 10x PBMC1 |" in s
    assert "| 180/11/2 | paper Fig. 4A optimum, 10x PBMC2 |" in s
    assert "| 375/3/2 | paper Fig. 4A optimum, 10x mouse cortex nuclei with intronic counting |" in s
    assert "only through reads elsewhere" not in s and "polyAfilter's windows lie on expressed transcripts" not in s
    # the chosen setting's frontier is printed too, with the chosen and published settings marked
    sel = json.loads((base / "eval/selection.json").read_text())
    sel["modes"]["i"]["selection"]["frontier"] = [["c150_l3_m2_i", 0.7, 0.7], [I1, 0.2368, 0.9896]]
    (base / "eval/selection.json").write_text(json.dumps(sel))
    s = m2report.selection_section(base)
    t = s.split("Frontier, -i mode")[1]
    assert "| Setting | Role | Sens | Spec |" in t
    assert "| c150_l3_m2_i |  | 0.700 | 0.700 |" in t
    assert "| c300_l10_m1_i | chosen (-i), published | 0.237 | 0.990 |" in t


def test_selection_section_remove_all_control(base):
    s = m2report.selection_section(base)
    assert "| Setting | 1417 sens | 1417 spec | 1179 sens | 1179 spec |" in s
    # 30,000 / 41,369 = 0.725 and 400 / 2,786 = 0.144 with its interval; 20,000 / 41,369 = 0.483, 900 / 2,786 = 0.323
    assert "| c150_lnone_m0_i | 0.725 | 0.144 [0.130, 0.157] | n/a | n/a |" in s
    assert "| c600_lnone_m0_e | 0.483 | 0.323 | n/a | n/a |" in s
    assert "c300_lnone_m0_i |" not in s.split("Remove-all control")[1].split("Units per member cap")[0]
    assert "ignores A content" in s and "bounds what a mode can reach" in s
    cap = s.split("Remove-all control")[1].split("Units per member cap")[0]
    assert "all molecules" not in cap
    assert "stratum all of the selection population (all barcodes, molecules with a member read within the cap)" in cap
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text()) | {"selection_population": "cells"}
    p.write_text(json.dumps(sel))
    assert "stratum all of the selection population (called cells, molecules with a member read" in m2report.selection_section(base)
    (base / "eval/metrics.parquet").unlink()
    assert "Remove-all control: not run (missing eval/metrics.parquet)" in m2report.selection_section(base)


def test_selection_section_cells_tuned(base):
    s = m2report.selection_section(base)
    assert "| Mode | Tuned on | Setting | Sens on 1179 | Spec on 1179 |" in s
    assert "Picked on" not in s
    # 478 / 1,000 and 95 / 100; the 1417 pick c300_l10_m1_i on 1179: 300 / 1,000 and 95 / 100; default mode: 40 / 179, 58 / 61
    assert "| -i | 1179 | c450_l5_m1_i | 0.478 | 0.950 |" in s
    assert "| -i | 1417 | c300_l10_m1_i | 0.300 | 0.950 |" in s
    assert "| default | 1179 | c250_l7_m2_e | 0.223 | 0.951 |" in s
    assert "| default | 1417 | none | n/a | n/a |" in s
    assert s.index("| default | 1179 |") < s.index("| default | 1417 |") < s.index("| -i | 1179 |") < s.index("| -i | 1417 |")
    assert ("On 1179, specificity and sensitivity of the setting picked on 1417 next to the setting picked on 1179: default n/a "
            "and n/a against 0.951 and 0.223; -i 0.950 and 0.300 against 0.950 and 0.478.") in s
    assert "not a recommendation" in s and "no second cell library to hold out" in s
    assert "leaves sensitivity unused" not in s and "is safe on cells" not in s
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text())
    del sel["cells_tuned"]
    p.write_text(json.dumps(sel))
    assert "Setting tuned on whole cells: not run (eval/selection.json has no cells_tuned)" in m2report.selection_section(base)


def test_selection_section_reads_threshold_and_library_from_the_file(base):
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text())
    sel["min_spec"] = 0.9
    p.write_text(json.dumps(sel))
    s = m2report.selection_section(base)
    assert "| default | none reached specificity 0.9 | n/a | n/a | n/a | 0 / 189 |" in s
    # (0.9 x 2,786 - 178) / 2,608 = 2,329.4 / 2,608 = 0.8932
    assert "specificity >= 0.9 over all 2,786 non-templated units on 1417 is at least (0.9 x 2,786 - 178) / 2,608 = 0.893" in s
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
    assert "Intergenic and antisense molecules: not run (missing eval/metrics.parquet)" in s


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


def _pop_sel(tag, sens, spec, n_q):
    head = "c250_l7_m1_e" if not tag else tag
    return {"introns": bool(tag), "selection": _sel(tag, sens, spec, 0.99, n_q) if tag else NONE_E,
            "metrics": {"1417": {"molecule": _lvl(80, 20, 90, 10)}, "1179": {"molecule": _lvl(30, 70, 95, 5)}} if tag else {},
            "headline": {"tag": head, "metrics": {"1417": {"molecule": _lvl(60, 40, 95, 5)}, "1179": {"molecule": _lvl(10, 90, 99, 1)}}}}


def _add_populations(base):
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text())
    sel |= {"selection_population": "cells", "selection_units": "all_labeled", "spikein_barcodes": "all",
            "molecules": {"1417": {"labeled": 144474, "in_cell": 90099, "units": {}},
                          "1179": {"labeled": 1402589, "in_cell": 1174777, "units": {}}},
            "populations": {pop: {um: {"e": _pop_sel(None, 0, 0, 0), "i": _pop_sel(f"c{150 + 10 * k}_l9_m2_i", 0.4, 0.96, 100 + k)}
                                    for k, um in enumerate(("with_member", "all_labeled"))}
                            for pop in ("all", "cells", "noncells")},
            "mapq_sensitivity": {"left_out": {"1417": 1200, "1179": 30},
                                 "modes": {"e": _pop_sel(None, 0, 0, 0),
                                           "i": {"introns": True, "selection": _sel(I1, 0.25, 0.99, 0.99, 119),
                                                 "metrics": {"1417": {"molecule": _lvl(100, 300, 200, 10)}}}},
                                 "published": {I1: {"1417": {"molecule": _lvl(100, 300, 200, 10)}, "1179": {}},
                                               I3: {"1417": {"molecule": _lvl(300, 100, 150, 60)}, "1179": {}}}}}
    p.write_text(json.dumps(sel))
    rows = []
    for lib, n in (("1417", 100), ("1179", 60)):
        for pop, f in (("all", 1.0), ("cells", 0.5), ("noncells", 0.5)):
            t, nt, am = int(n * f * 0.6), int(n * f * 0.3), int(n * f * 0.1)
            rows.append((lib, pop, "all", "all", t, nt, am, t + nt + am, t / (t + nt + am)))
    _write(base / "burden/templated_share_population.parquet", pd.DataFrame(
        rows, columns=["lib", "population", "stratum_kind", "stratum", "templated", "nontemplated", "ambiguous", "total", "share_all"]))


def test_selection_section_states_population_and_units(base):
    s = m2report.selection_section(base)
    # a selection.json without the keys reads as all barcodes and molecules with a member read
    assert "Computed on: all barcodes, molecules with a member read within the cap." in s
    _add_populations(base)
    s = m2report.selection_section(base)
    assert "Computed on: called cells, all labeled molecules." in s


def test_selection_section_called_cells_table(base):
    s = m2report.selection_section(base)
    assert "Called cells and other barcodes: not run (eval/selection.json has no populations)" in s
    assert "Molecules with mapq_min below 255 left out: not run (eval/selection.json has no mapq_sensitivity)" in s
    _add_populations(base)
    s = m2report.selection_section(base)
    assert "| Library | Labeled molecules | In called cells | Templated share, all barcodes | Templated share, called cells | " \
        "Templated share, other barcodes |" in s
    # 90,099 / 144,474 = 62.4%; the templated shares come from burden/templated_share_population.parquet (60 / 100 ...)
    assert "| 1417 | 144,474 | 90,099 (62.4%) | 0.600 (60 of 100) | 0.600 (30 of 50) | 0.600 (30 of 50) |" in s
    assert "| 1179 | 1,402,589 | 1,174,777 (83.8%) | 0.600 (36 of 60) | 0.600 (18 of 30) | 0.600 (18 of 30) |" in s
    assert "| Population | Units | Mode | Chosen (on 1417) | Qualifying | 1417 sens / spec | 1179 sens / spec |" in s
    assert "| called cells | all labeled molecules | -i | c160_l9_m2_i | 101 | 0.800 / 0.900 | 0.300 / 0.950 |" in s
    assert "| other barcodes | molecules with a member read within the cap | default | none | 0 | n/a | n/a |" in s
    assert "| all barcodes | molecules with a member read within the cap | -i | c150_l9_m2_i | 100 | 0.800 / 0.900 | 0.300 / 0.950 |" in s
    # headline settings under every population and units: sens / spec = tp / (tp + fn), tn / (tn + fp) of _lvl
    assert ("Headline settings (the tags chosen per mode in called cells, all labeled molecules) scored in each population and "
            "with each unit definition, molecule level (member cap 1,000 nt, 8/4 labels):") in s
    assert "| Population | Units | Mode | Headline setting | 1417 sens / spec | 1179 sens / spec |" in s
    assert "| called cells | all labeled molecules | -i | c160_l9_m2_i | 0.600 / 0.950 | 0.100 / 0.990 |" in s
    assert "| other barcodes | molecules with a member read within the cap | default | c250_l7_m1_e | 0.600 / 0.950 | 0.100 / 0.990 |" in s
    sel = json.loads((base / "eval/selection.json").read_text())
    for units_ in sel["populations"].values():
        for modes in units_.values():
            for d in modes.values():
                del d["headline"]
    (base / "eval/selection.json").write_text(json.dumps(sel))
    assert "Headline settings in each population: not run (eval/selection.json populations have no headline, stage 17 predates it)" \
        in m2report.selection_section(base)
    (base / "burden/templated_share_population.parquet").unlink()
    s = m2report.selection_section(base)
    assert "Templated share by population: not run (missing burden/templated_share_population.parquet)" in s
    assert "| 1417 | 144,474 | 90,099 (62.4%) | n/a | n/a | n/a |" in s


def test_selection_section_mapq_sensitivity(base):
    _add_populations(base)
    s = m2report.selection_section(base)
    assert "Molecules with mapq_min below 255 left out (1417: 1,200; 1179: 30)" in s
    assert "| -i | c300_l10_m1_i | 0.250 | 0.990 | 119 / 189 |" in s
    assert "| default | none reached specificity 0.95 | n/a | n/a | 0 / 189 |" in s
    # published settings on the reduced set: tp 300 fn 100 tn 150 fp 60
    assert "| c375_l3_m2_i | published | 1417 | 400 | 210 | 0.750 | 0.714 |" in s
    assert "| c375_l3_m2_i | published | 1179 | n/a | n/a | n/a | n/a |" in s
    assert "| c300_l10_m1_i | chosen (-i), published | 1417 | 400 | 210 | 0.250 | 0.952 |" in s


def test_spikein_section_states_that_every_barcode_is_used(base):
    _add_populations(base)
    assert "Spike-in metrics count all barcodes" in m2report.spikein_section(base)


def test_strata_section(base):
    s = m2report.strata_section(base)
    # spec 2,757 / 2,786 = 0.990; sens 100 / 2,000, 1,000 / 6,400 = 0.156, 8,695 / 32,969 = 0.264
    assert "| c300_l10_m1_i | 1417 | 0.990 | 0.050 | 0.156 | 0.264 |" in s
    # intergenic 0 / 5,182 and 125 / 128; 3UTR 5,000 / 15,000 and 2,000 / 2,020; other regions absent
    assert "| c300_l10_m1_i | 1417 | 0.000 / 0.977 | n/a | n/a | n/a | n/a | 0.333 / 0.990 | 0.000 / 1.000 |" in s
    assert "| Setting | Library | intergenic | intron | ncexon | 5UTR | CDS | 3UTR | antisense |" in s
    assert "| 1417 | a_band | 8-9 | 2,000 | 0 |" in s
    assert "| 1417 | region | 3UTR | 15,000 | 2,020 |" in s
    assert "| 1417 | region | intergenic | 5,182 | 128 |" in s
    (base / "eval/selection.json").unlink()
    s = m2report.strata_section(base)
    assert "chosen settings: not run (missing eval/selection.json)" in s
    assert "| c300_l10_m1_i | 1417 | 0.990 | 0.050 | 0.156 | 0.264 |" in s


def test_strata_section_expression_has_ts_and_levels(base):
    s = m2report.strata_section(base)
    # molecule level: sens 100 / 1,000 = 0.100 and spec 40 / 41 = 0.976; 300 / 1,000 and 90 / 100; 0 / 500 and 20 / 20
    assert "| Setting | Library | low | mid | high | no_gene |" in s
    assert "| c300_l10_m1_i | 1417 | 0.100 / 0.976 | n/a | 0.300 / 0.900 | 0.000 / 1.000 |" in s
    # has_ts True 5 / 100 and 10 / 10; False 9,790 / 41,269 = 0.237 and 2,747 / 2,776 = 0.990
    assert "| Setting | Library | True | False |" in s
    assert "| c300_l10_m1_i | 1417 | 0.050 / 1.000 | 0.237 / 0.990 |" in s
    assert "| c375_l3_m2_i | 1417 | n/a | n/a |" in s
    assert "| 1417 | expr_tertile | no_gene | 500 | 20 |" in s
    assert "| 1417 | has_ts | True | 100 | 10 |" in s
    assert "By has_ts (a pa read of the molecule has a ts tag above 0, section 2), molecule level" in s
    assert "short RNA" not in s
    # A band at the read level: spec 2,700 / 2,800 = 0.964, sens 12,000 / 36,000 = 0.333; 8-9 is the 2,000 / 2,000 fixture row
    assert "at the read level" in s and "at the molecule-majority level" in s
    assert "| c300_l10_m1_i | 1417 | 0.964 | 1.000 | n/a | 0.333 |" in s
    # majority: spec 2,650 / 2,800 = 0.946, sens 8-9 300 / 2,000 = 0.150
    assert "| c300_l10_m1_i | 1417 | 0.946 | 0.150 | n/a | n/a |" in s


def test_strata_and_hexamer_sections_name_their_population_and_definitions(base):
    pop = "Computed on: all barcodes, molecules with a member read within the cap."
    s = m2report.strata_section(base)
    assert pop in s
    assert ("Expression tertile: genes ranked by the number of their labeled molecules in the selection population, with or without a member read (ties by gene id) "
            "and split into three equal gene groups, so the groups hold very different numbers of molecules; each molecule takes "
            "its gene's group.") in s
    h = m2report.hexamer_section(base)
    assert pop in h
    assert "AATAAA/ATTAAA fully inside the window 6 to 35 nt upstream of the site" in h and "5 to 35" not in h
    sel = json.loads((base / "eval/selection.json").read_text()) | {"selection_population": "cells"}
    (base / "eval/selection.json").write_text(json.dumps(sel))
    assert "Computed on: called cells, molecules" in m2report.strata_section(base)
    (base / "eval/selection.json").unlink()
    assert "Computed on" not in m2report.strata_section(base)


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
    assert "tail: within 10 nt of an annotated 3' end; internal: not tail and at least 30 nt upstream of one; other: neither" \
           in s and "other molecules are left out of the agreement and of the metrics" in s
    # the ambiguous internal molecules (ERCC none, SIRV_E0 a20 7) count in the denominator, not as templated
    assert "| 1417 | all | 4 | 2 (0.500) | 5 | 2 (0.400) |" in s
    assert "| 1417 | ERCC | 1 | 1 (1.000) | 2 | 1 (0.500) |" in s
    assert "| 1417 | SIRV_E0 | 3 | 1 (0.333) | 1 | 0 (0.000) |" in s
    assert "| 1417 | long_SIRV | 0 | 0 (n/a) | 2 | 1 (0.500) |" in s
    assert "1419: not run (missing 1419/spikein/truth/molecules_labeled.parquet)" in s
    # molecule level only: 10 / 40, 50 / 52; ERCC 5 / 10, 20 / 20; the read-level row (9 / 10) is not used
    assert "| 1417 | c300_l10_m1_i | 0.250 / 0.962 | 0.500 / 1.000 | n/a | n/a |" in s
    # the control, molecule level: 50 / 100, 984 / 1,000
    assert "| 1417 | c150_lnone_m0_i | 0.500 / 0.984 | n/a | n/a | n/a |" in s
    (base / "eval/spikein_metrics.parquet").unlink()
    assert "Metrics per module: not run (missing eval/spikein_metrics.parquet)" in m2report.spikein_section(base)


def test_spikein_section_read_level_and_no_1179(base):
    s = m2report.spikein_section(base)
    read = s.split("Metrics per module at the read level", 1)[1]
    # read level from the same file: 9 / 10 and no negatives; the control 60 / 100 and 156 / 1,000
    assert "| 1417 | c300_l10_m1_i | 0.900 / n/a | n/a | n/a | n/a |" in read
    assert "| 1417 | c150_lnone_m0_i | 0.600 / 0.156 | n/a | n/a | n/a |" in read
    assert "cannot fail" not in s and "protected 3' stretch" not in s and "never fully removed" not in s
    # the remove-all control with the lowest molecule-level specificity per library, counted from the control rows: fp 16 of
    # tn + fp 1,000 (0.984) at molecule level, 0.156 at read level
    assert "| Library | Setting | Tail molecules removed (fp of tn + fp) | Spec, molecule level | Spec, read level |" in s
    assert "| 1417 | c150_lnone_m0_i | 16 of 1,000 | 0.984 | 0.156 |" in s
    assert "our observation in this data" in s
    assert "1179 has no spike-ins, so this section has no 1179 rows" in s
    (base / "eval/spikein_metrics.parquet").unlink()
    assert "at the read level" not in m2report.spikein_section(base)


def test_ercc_section_says_1179_has_no_spikeins(base):
    assert "1179 has no spike-ins, so this section has no 1179 rows" in m2report.ercc_section(base)


def test_ercc_section(base):
    s = m2report.ercc_section(base)
    assert s.startswith("## 8. ERCC count accuracy (provisional)")
    assert f"Provisional (1417): {NOTE}" in s
    assert "| Library | Setting | n | pearson_log10 | spearman | umis | umis_kept_share |" in s
    assert "| 1417 | unfiltered | 61 | 0.912 | 0.888 | 52,000 | 1.000 |" in s
    assert "| 1417 | c300_l10_m1_i | 61 | 0.920 | 0.890 | 50,502 | 0.971 |" in s
    assert "| 1417 | c375_l3_m2_i | 3 | n/a | n/a | 3 | 0.000 |" in s
    assert "| 1419 | unfiltered | 60 | 0.900 | 0.800 | 4,000 | 1.000 |" in s
    # column definitions from evaluate/ercc.py, and the barcode set of stage 15
    assert "n: ERCCs with a UMI unfiltered and a Mix 1 concentration" in s
    assert "pearson_log10: Pearson correlation of log10(UMIs + 1) with log10(concentration)" in s
    assert "umis_kept_share: umis over the unfiltered umis" in s
    assert ("UMIs are counted over primary records with a UB and a CB of the barcode set recorded in ercc.json (stage 15): "
            "1417 called cells; 1419 not recorded. The spike-in metrics of section 7 use a barcode set that is not recorded "
            "(stage 17 predates it).") in s
    p = base / "1417/spikein/ercc.json"
    d = json.loads(p.read_text())
    for pop, name in (("all", "all barcodes"), ("noncells", "other barcodes")):
        p.write_text(json.dumps(d | {"barcodes": pop}))
        assert f"1417 {name}; 1419 not recorded." in m2report.ercc_section(base)
    # the recorded selection (default none, -i c300_l10_m1_i) is the one in eval/selection.json: no note
    assert "predates the current selection" not in s


def test_ercc_section_flags_a_selection_that_changed(base):
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text())
    sel["modes"]["i"]["selection"]["tag"] = "c200_l10_m2_i"
    p.write_text(json.dumps(sel))
    s = m2report.ercc_section(base)
    assert ("1417: ercc.json was written with the chosen settings default none, -i c300_l10_m1_i; eval/selection.json now "
            "chooses default none, -i c200_l10_m2_i, so stage 15 predates the current selection and the table lacks the "
            "chosen setting.") in s
    assert "1419:" not in s.split("predates")[0].split("- ")[-1]   # 1419 records no selection: no note
    p.unlink()
    assert "predates the current selection" not in m2report.ercc_section(base)


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
    assert "Spec 4.3 gives about 7% for milestone 1. In this run BD67 dropped 11.1% of the reads used" in s
    assert "about 12%" not in s
    assert "| BD67 | 16,408 | 0.615 | 0.491 / 0.907 / 0.805 | 0.993 / 0.996 / 0.994 |" in s
    assert "BD70: not run (missing BD70/long_read_acceptance.json)" in s
    # outcome and agreement columns defined, thresholds from bd67.read_outcome
    assert ("Outcomes of the used reads: Accepted, a terminal A run of at least 10 nt followed by at least 30 nt of clip; Short run, "
            "the run is shorter than 10 nt; Short after, the run is long enough but fewer than 30 nt of clip follow; Both, "
            "neither.") in s
    assert "Passes (per-site ratio >= 3)" in s and "Passes (ratio >= 3)" not in s
    assert "Agreement T / NT / pooled: among 1179 molecules matched to a long-read site" in s
    assert "may mostly measure co-location (not tested here)" in s and "mostly measures co-location" not in s


def test_long_read_section_sites_by_region_and_label(base):
    s = m2report.long_read_section(base)
    # the fixture sites (all region 3UTR): sites with an accepted read are 10, 20, 30, 40, 70, 80, 90; labels templated 10, 20,
    # 80, 90 and non-templated 30, 40, 70
    assert "Sites with an accepted read by region and label (`BD67/long_read_sites.parquet`, unique sites):" in s
    assert "| BD67 | 3UTR | 4 | 0 | 3 | 7 |" in s
    assert "| BD67 | CDS | 0 | 0 | 0 | 0 |" in s
    assert "| Library | Region | templated | ambiguous | non-templated | total |" in s.split("Sites with an accepted read")[1]


def test_long_read_section_drop_rates_come_from_the_files(base):
    d = json.loads((base / "BD67/long_read_acceptance.json").read_text())
    d["dropped_fraction"] = 0.1188
    (base / "BD67/long_read_acceptance.json").write_text(json.dumps(d))
    _write(base / "BD70/long_read_acceptance.json", d | {"library": "BD70", "dropped_fraction": 0.0953})
    s = m2report.long_read_section(base)
    assert "BD67 dropped 11.9% of the reads used and BD70 9.5%" in s


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


def test_tool_statements_are_cited_or_labeled(base):
    reading = "our reading of the polyAfilter code at commit 5ce0098, spec 3"
    assert reading in m2report.grid_section(base)
    assert reading in m2report.selection_section(base)
    b = m2report.burden_section(base)
    assert "10x Genomics scRNA-seq data (PolyASite 3.0, https://polyasite.unibas.ch/atlas_sc)" in b
    assert "oligo(dT) primed like these libraries" not in b


def test_polyasite_wording_in_the_sources():
    for f, n in (("pafbench/polyasite.py", 5), ("scripts/18_burden.py", 5)):
        head = " ".join((REPO / f).read_text().splitlines()[:n])
        assert "describes no filter on A content downstream of a site" in head, f
        assert "22_polyasite_a_content.py" in head, f
        assert "9,544" not in head and "465,261" not in head and "keeps sites" not in head and "not screened" not in head, f


def test_burden_section_templated_share_by_label_and_corrected(base):
    s = m2report.burden_section(base)
    assert "| Library | Labeled molecules | Templated share by label | Templated share, corrected |" in s
    # 1,500 labeled; 0.8 x (1 - 0.0279) = 0.778; 1179: 0.8 x (1 - 0.0837) = 0.733
    assert "| 1417 | 1,500 | 0.800 | 0.778 |" in s
    assert "| 1179 | 1,500 | 0.800 | 0.733 |" in s
    assert "removes the estimated tail-primed part of the templated class" in s
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text())
    del sel["label_purity"]
    p.write_text(json.dumps(sel))
    assert "Templated share corrected: not run (eval/selection.json has no label_purity)" in m2report.burden_section(base)
    p.unlink()
    assert "Templated share corrected: not run (missing eval/selection.json)" in m2report.burden_section(base)


def test_burden_section_selection_bias_without_the_duplicate_free_baseline(base):
    s = m2report.burden_section(base)
    assert "| Library | RE | Labeled share | Records share |" in s
    assert "records_nondup_share" in s and "stage 13 predates it" in s
    assert "cannot be applied" in s
    assert "Cell Ranger sets the duplicate flag" not in s   # the caveat belongs to the duplicate-free column


def test_burden_section_selection_bias_rule(base):
    _write(base / "burden/summary.json", {"groups": {"nuclei": ["1417"]}, "selection_bias_population": "all"})
    sb = pd.DataFrame([("1417", "E", 0.421, 0.407, 0.350), ("1417", "N", 0.48, 0.5, 0.49), ("1419", "I", 0.099, 0.093, 0.099)],
                      columns=["lib", "re", "labeled_share", "records_share", "records_nondup_share"])
    _write(base / "burden/selection_bias.parquet", sb)
    s = m2report.burden_section(base)
    assert "| Library | RE | Labeled share | Records share | Records share without PCR duplicates |" in s
    assert "| 1417 | E | 0.421 | 0.407 | 0.350 |" in s
    assert "stage 13 predates it" not in s
    assert "all primary records" in s and "without the PCR-duplicate flag" in s
    assert "never carry" not in s
    assert ("Both shares cover all barcodes: the records are those of every barcode, so stage 18 takes the labeled molecules of "
            "all barcodes whatever its population. The check is not repeated within called cells because stage 13 does not "
            "split the records by barcode (left for milestone 3, which builds per-cell counts):") in s
    assert f"RE is the region type of the alignment (E = exonic, N = intronic, I = intergenic; Cell Ranger documentation, " \
           f"Barcoded BAM Tags, {m2report.BAM_TAGS_URL})" in s
    # 0.421 - 0.350 = 7.1 points; 0.48 - 0.49 = 1.0 and 0.099 - 0.099 = 0.0 are within 5
    assert "more than 5 percentage points: 1417 E (labeled 0.421, without duplicates 0.350, 7.1 points)." in s
    assert ("Spec 11 says that where labeled and unlabeled molecules differ strongly, conclusions are stated for the labeled "
            "population only; the threshold used here, a difference of more than 5 percentage points, is our choice (the "
            "milestone 1 region check uses the same value).") in s
    sb["records_nondup_share"] = sb.labeled_share
    _write(base / "burden/selection_bias.parquet", sb)
    assert "No library and region pair differs by more than 5 percentage points" in m2report.burden_section(base)


def test_burden_section_duplicate_flag_caveat(base):
    sb = pd.DataFrame([("1417", "E", 0.421, 0.407, 0.350)],
                      columns=["lib", "re", "labeled_share", "records_share", "records_nondup_share"])
    _write(base / "burden/selection_bias.parquet", sb)
    source = ("Cell Ranger sets the duplicate flag only on reads that it counts, that is reads with a valid UMI that are "
              "confidently mapped to one gene, and among those not on the read that stands for the UMI (cellranger 9.0.1 source, "
              "lib/rust/tx_annotation/src/mark_dups.rs lines 288 to 291 and read.rs lines 560 to 568).")
    last = ("Milestone 3 repeats the comparison within called cells with a unit defined the same way in every region.")
    # no region_counts.parquet: the source statement and its consequence, no claim about these BAMs
    s = m2report.burden_section(base)
    assert f"{source} The shares without PCR duplicates therefore remove duplicates only among counted reads. {last}" in s
    assert "absent on every" not in s
    assert s.index(source) < s.index("Spec 11 says")   # under the table, before the rule
    # the regions without one flagged record are read from stage 13's counts: I everywhere, N in 1179 only
    for lib in ("1417", "1419"):
        _write(base / lib / "grid/region_counts.parquet", _region_counts())
    cells = _region_counts()
    cells.loc[cells.re == "N", "total_nondup"] = cells.loc[cells.re == "N", "total"]
    _write(base / "1179/grid/region_counts.parquet", cells)
    s = m2report.burden_section(base)
    assert (f"{source} The flag is therefore absent on every intergenic record and, in 1179, on every intronic record, so "
            f"the shares without PCR duplicates remove duplicates only among counted reads. {last}") in s
    # one library only: the claim names it
    (base / "1419/grid/region_counts.parquet").unlink()
    (base / "1179/grid/region_counts.parquet").unlink()
    s = m2report.burden_section(base)
    assert "The flag is therefore absent on every intergenic record of 1417, so the shares" in s
    # a library whose intergenic records do carry the flag is not named
    flagged = _region_counts()
    flagged.loc[flagged.re == "I", "total_nondup"] = 80
    _write(base / "1417/grid/region_counts.parquet", flagged)
    assert "absent on every" not in m2report.burden_section(base)


def _region_counts(total_nondup=True):
    rows = [(t, re, rem, tot, nd) for t in ("c150_lnone_m0_e", "c300_l10_m1_i")
            for re, rem, tot, nd in (("E", 10, 400, 350), ("I", 0, 90, 90), ("N", 5, 510, 400))]
    df = pd.DataFrame(rows, columns=["tag", "re", "removed", "total", "total_nondup"])
    return df if total_nondup else df.drop(columns="total_nondup")


def test_burden_section_records_by_re_tag_is_an_observation_from_region_counts(base):
    s = m2report.burden_section(base)
    # fixture base has no region_counts.parquet
    assert "Primary records by RE tag: not run (missing 1417/grid/region_counts.parquet, 1419/grid/region_counts.parquet, " \
           "1179/grid/region_counts.parquet)" in s
    # without total_nondup (stage 13 predates it): the primary records only, and the line saying so
    _write(base / "1417/grid/region_counts.parquet", _region_counts(total_nondup=False))
    s = m2report.burden_section(base)
    assert "| 1417 | E | 400 | n/a |" in s and "| 1417 | I | 90 | n/a |" in s
    assert "1417/grid/region_counts.parquet has no total_nondup column: stage 13 predates it" in s
    # with it: the counts of one tag (not summed over tags), the records without the duplicate flag beside them
    _write(base / "1417/grid/region_counts.parquet", _region_counts())
    s = m2report.burden_section(base)
    assert "| Library | RE | Primary records | Primary records without the duplicate flag |" in s
    assert "| 1417 | E | 400 | 350 |" in s and "| 1417 | I | 90 | 90 |" in s and "| 1417 | N | 510 | 400 |" in s
    assert "stage 13 predates it" in s   # the selection_bias.parquet line of the fixture
    assert "has no total_nondup column" not in s
    assert "our observation in these BAMs" in s


def test_burden_section_population_and_groups(base):
    s = m2report.burden_section(base)
    assert "Population of the burden tables: not run (missing burden/summary.json)" in s
    assert "Groups: not run (missing burden/summary.json)" in s
    groups = {"1417": ["1417"], "nuclei": ["1417", "1419"], "cells": ["1179"]}
    _write(base / "burden/summary.json", {"groups": groups})
    s = m2report.burden_section(base)
    assert "Population of the burden tables: not recorded (stage 18 predates it)." in s
    assert "Groups (burden/summary.json): nuclei = 1417 + 1419; cells = 1179. A group with one library holds the same " \
           "molecules as that library: cells = 1179." in s
    for pop, name in (("all", "all barcodes"), ("cells", "called cells"), ("noncells", "other barcodes")):
        _write(base / "burden/summary.json", {"groups": groups, "population": pop, "selection_bias_population": "all"})
        s = m2report.burden_section(base)
        assert f"Population of the burden tables (burden/summary.json): {name}." in s
        assert "not recorded" not in s and "inferred" not in s
    # no inference from eval/selection.json any more, and no mismatch warning
    _add_populations(base)
    s = m2report.burden_section(base)
    assert "not over the same molecules" not in s and "matches no population" not in s


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
    assert "describes no filter on A content downstream of a site" in s
    assert "describes no filter on A content downstream of a site" in s
    assert "9,544" not in s and "465,261" not in s and "chr21" not in s
    # no json: the not-run line with its file, and no numbers
    assert "A content downstream of the atlas sites: not run (missing burden/polyasite_a_content.json)" in s
    tot = {"n_sites": 1000, "n_full": 990, "n_a14": 99, "n_full_null": 980, "n_a14_null": 49}
    _write(base / "burden/polyasite_a_content.json", {"window_nt": 20, "min_a": 14, "null_shift_nt": 1000, "refs": ["chr1", "chr2"],
                                                       "atlas_refs_not_counted": ["chrX"], "total": tot, "per_ref": {}})
    s = m2report.burden_section(base)
    # 99 / 990 = 10.0%, 49 / 980 = 5.0%
    assert ("our observation (`burden/polyasite_a_content.json`, stage 22, 2 of 3 atlas references): 99 of 990 atlas sites with a "
            "full 20 nt downstream window (10.00%) have 14 or more A's in it, against 49 of 980 (5.00%) for the same sites moved 1,000 "
            "nt upstream; the clusters file is not depleted of A-rich downstream sequence by this count.") in s
    _write(base / "burden/polyasite_a_content.json", {"window_nt": 20, "min_a": 14, "null_shift_nt": 1000, "refs": ["chr1"],
                                                       "atlas_refs_not_counted": [], "total": tot | {"n_a14": 10}, "per_ref": {}})
    s = m2report.burden_section(base)
    assert "1 of 1 atlas references" in s and "10 of 990 atlas sites" in s and "(1.01%)" in s
    assert "the clusters file is depleted of A-rich downstream sequence by this count" in s
    assert "screen" not in s
    assert ("The null row takes every molecule of the group with its site moved 1,000 nt upstream in transcript sense on the "
            "same ref and strand; its share within 10 nt is the local background at that offset, not a uniform chance level.") in s
    assert "expected by chance" not in s
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
    assert "| 1417 | c275_l9_m1_i | 17,000,000 | 17,000,000 | 0 | 0 | True | 1.95 | 15.49 | 60.50 | 17,000,000 | True |" in s
    assert "| 1417 | c300_l10_m1_i | 17,068,514 | 17,068,514 | 0 | 0 | True | 1.95 | 15.50 | n/a | 17,068,514 | True |" in s
    assert "1419: no completed original run" in s
    assert "1179: not in exact_match_m2.json; the 1179 grid has no original-versus-reimplementation evidence" in s
    # 20 + 412 + 160 + 80 = 672 s
    assert "| c300_lnone_m0_i | chr21, chr22 | 2,781,871 | 1,894,573 | 1,894,573 | 0 | 0 | True | 412 | 0.26 | 672 |" in s
    assert "Removed-record comparisons: 3 (the table rows and the control); checks counted by stage 20 (these comparisons plus, where it records them, one completeness and one engine-tie check per library): 3; all identical: True" in s
    assert "Removed-record sets (every record the original fetches, by our reading of its code, spec 5" in s


def test_validation_section_completeness(base):
    s = m2report.validation_section(base)
    assert "| Library | Setting | Original removed | Reimpl removed | Only original | Only reimpl | Identical | Original wall (h) | " \
           "Max RSS (GiB) | Peak tree RSS (GiB) | Stage 13 removed | Equal to stage 13 |" in s
    assert "complete: no" in s
    # missing runs with their exit status and reason; a run without a time file has no exit status
    assert "| Library | Setting | Exit status | Reason |" in s
    assert "| 1419 | c275_l9_m1_i | 137 | killed or failed |" in s
    assert f"| 1419 | {I1} | n/a | no output |" in s
    # engine tie: 1,000 records, 0 + 3 differences, 2 without a counterpart; 1419 not computed
    assert "| Library | Records compared | Differences (records x settings) | Missing |" in s
    assert "| 1417 | 1,000 | 3 | 2 |" in s
    assert "| 1419 | not computed | n/a | n/a |" in s
    assert "validated only when all_identical and complete are both true" in s
    assert "not validated" in s
    assert "Stages 03 and 04 reused: 1417 yes, 1419 no" in s
    p = base / "validation/exact_match_m2.json"
    em = json.loads(p.read_text())
    em["complete"], em["missing"] = True, {"1417": [], "1419": []}
    p.write_text(json.dumps(em))
    s = m2report.validation_section(base)
    assert "complete: yes" in s and "checks counted by stage 20 (these comparisons plus, where it records them, one completeness and one engine-tie check per library): 3; all identical: True" in s
    assert "| Library | Setting | Exit status | Reason |" not in s
    assert "the reimplementation is validated" in s and "not validated" not in s


def test_validation_section_without_completeness_keys(base):
    p = base / "validation/exact_match_m2.json"
    em = json.loads(p.read_text())
    for k in ("complete", "missing", "engine_tie", "reused"):
        del em[k]
    for v in em["1417"].values():
        del v["n_grid"], v["grid_equal"]
    p.write_text(json.dumps(em))
    s = m2report.validation_section(base)
    assert "completeness not recorded (stage 20 predates the key)" in s
    assert "| 1417 | c275_l9_m1_i | 17,000,000 | 17,000,000 | 0 | 0 | True | 1.95 | 15.49 | 60.50 | n/a | n/a |" in s
    assert "Engine tie" not in s and "validated" not in s.split("completeness not recorded")[1]


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
    assert "| 1179 (chr21, chr22) | c300_l10_m1_i | 2,000,000 | 2,000,000 | 0 | 0 | True | 0.50 | 3.25 | 6.50 | n/a | n/a |" in s
    assert "| 1179 (chr21, chr22) | c300_l10_m1_e | 2,000,000 | 2,000,000 | 2 | 0 | False | 0.50 | 3.25 | 6.50 | n/a | n/a |" in s
    assert "1179: original tool on the chr21, chr22 subset only (18,382,751 records)" in s
    assert "the 1179 grid has no original-versus-reimplementation evidence" not in s
    # the nuclei rows are unchanged
    assert "| 1417 | c275_l9_m1_i | 17,000,000 | 17,000,000 | 0 | 0 | True | 1.95 | 15.49 | 60.50 | 17,000,000 | True |" in s


def test_validation_section_on_a_stale_file(base):
    # the schema an earlier draft left in the real OUT: no library keys, no none_control, no totals
    (base / "validation/exact_match_m2.json").write_text(json.dumps({"none_control_chr21_chr22": {"error": "createTRANS failed"}}))
    s = m2report.validation_section(base)
    assert "1417: not in exact_match_m2.json" in s
    assert "Remove-all control: not run (no none_control entry in validation/exact_match_m2.json)" in s
    assert "Removed-record comparisons: 0 (the table rows and the control); checks counted by stage 20 (these comparisons plus, where it records them, one completeness and one engine-tie check per library): n/a; all identical: n/a" in s


def test_render_and_script(base, tmp_path):
    text = m2report.render(base)
    s = _sections(text)
    assert sorted(s) == list(range(1, 12))
    assert text.startswith("# Milestone 2 results (generated by scripts/21_m2_summary.py)")
    assert "not run (missing eval/selection.json)" not in text
    assert "{" not in text and "}" not in text  # no unformatted placeholder reaches the page
    out = tmp_path / "docs" / "m2_results.md"
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "21_m2_summary.py"), "--base", str(base), "--out", str(out)],
                       capture_output=True, text=True, env=env, check=True)
    assert out.read_text() == text
    line = json.loads(r.stdout.strip().splitlines()[-1])
    assert line == {"out": str(out), "sections": 11, "not_run": text.split("\n## Headline", 1)[1].count("not run")}


def test_headline_block(base):
    text = m2report.render(base)
    head = text.split("\n## 1.")[0]
    assert head.index("## Headline") > head.index("# Milestone 2 results")
    # population, units, cap, cut-offs and the rule text of selection.json
    assert ("Computed on: all barcodes, molecules with a member read within the cap; member cap 1,000 nt, 8/4 labels. "
            "Rule: highest molecule-level sensitivity with specificity >= 0.95 on 1417") in head
    # chosen -i setting split into its parameters, its library rows with the bootstrap interval; the default mode chose none
    assert "| Mode | Setting | COVLEN | MINSNRLEN | MISM | Qualifying / candidates | Library | Templated | Non-templated | " \
           "Sens (95% interval) | Spec (95% interval) |" in head
    assert "| -i | c300_l10_m1_i | 300 | 10 | 1 | 120 / 189 | 1417 | 41,369 | 2,786 | 0.237 [0.207, 0.259] | 0.990 [0.985, 0.993] |" in head
    assert "| -i | c300_l10_m1_i | 300 | 10 | 1 | 120 / 189 | 1179 | 1,000 | 100 | 0.300 [n/a, n/a] | 0.950 [n/a, n/a] |" in head
    assert "| default | none reached specificity 0.95 | n/a | n/a | n/a | 0 / 189 | n/a | n/a | n/a | n/a | n/a |" in head
    # validation outcome from exact_match_m2.json (the fixture file is complete: false)
    assert "Validation against the original tool (section 11): 3 removed-record comparisons, all identical True, complete no: not validated on these runs." \
        in head
    (base / "eval/bootstrap.parquet").unlink()
    assert "Bootstrap: not run (missing eval/bootstrap.parquet)" in m2report.headline(base)
    p = base / "validation/exact_match_m2.json"
    em = json.loads(p.read_text()) | {"complete": True}
    p.write_text(json.dumps(em))
    assert "complete yes: validated on these runs." in m2report.headline(base)
    del em["complete"]
    p.write_text(json.dumps(em))
    assert "completeness not recorded" in m2report.headline(base)


def test_header_defines_setting_tags_and_lists_missing_inputs(base):
    text = m2report.render(base)
    head = text.split("\n## Headline")[0]
    assert ("Setting tags read c<COVLEN>_l<MINSNRLEN>_m<MISM>_<e or i>: COVLEN, MINSNRLEN and MISM are polyAfilter's parameters "
            "(spec 3); e is the default mode and i the -i mode (introns); l none is the remove-all control.") in head
    # the fixture lacks 1419 and 1179 inputs: counted per section, files listed once
    assert "Missing inputs:" in head and "1419/grid/summary.json" in head and "section 3: " in head
    assert "not run" not in head.split("Missing inputs:")[1]


def test_selection_bias_caption_follows_the_recorded_population(base):
    groups = {"1417": ["1417"], "nuclei": ["1417", "1419"], "cells": ["1179"]}
    _write(base / "burden/summary.json", {"groups": groups, "population": "cells"})
    s = m2report.burden_section(base)
    assert "Population of the labeled share: not recorded (stage 18 predates it)." in s
    assert "Both shares cover all barcodes" not in s
    _write(base / "burden/summary.json", {"groups": groups, "population": "cells", "selection_bias_population": "all"})
    s = m2report.burden_section(base)
    assert "Both shares cover all barcodes: the records are those of every barcode" in s and "not recorded (stage 18" not in \
        s.split("Selection bias:")[1]
    _write(base / "burden/summary.json", {"groups": groups, "selection_bias_population": "cells"})
    s = m2report.burden_section(base)
    assert "The labeled share covers called cells (burden/summary.json)" in s and "Both shares cover all barcodes" not in s


def test_band_lines_survive_an_undefined_estimate(base):
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text())
    sel["label_purity"]["1417"]["bands"]["8-9"] = {"n": 0, "hexamer_rate": None, "tail_primed_share": None}
    p.write_text(json.dumps(sel))
    s = m2report.truth_section(base)
    assert "| 1417 | 8-9 | 0 | n/a | n/a |" in s


def test_a_content_line_with_no_full_window(base):
    z = {"n_sites": 5, "n_full": 0, "n_a14": 0, "n_full_null": 0, "n_a14_null": 0}
    _write(base / "burden/polyasite_a_content.json", {"window_nt": 20, "min_a": 14, "null_shift_nt": 1000, "refs": ["chr1"],
                                                       "atlas_refs_not_counted": [], "total": z, "per_ref": {}})
    s = m2report.burden_section(base)
    assert "0 of 0 atlas sites with a full 20 nt downstream window (n/a)" in s


def test_population_words_follow_the_inputs(base):
    _add_populations(base)
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text())
    sel |= {"selection_population": "noncells", "spikein_barcodes": "cells"}
    p.write_text(json.dumps(sel))
    assert "run on 1179 other barcodes in place of" in m2report.selection_section(base)
    assert "Spike-in metrics count called cells" in m2report.spikein_section(base)
    del sel["spikein_barcodes"]
    p.write_text(json.dumps(sel))
    assert "Spike-in metrics use a barcode set that is not recorded (stage 17 predates it)" in m2report.spikein_section(base)
