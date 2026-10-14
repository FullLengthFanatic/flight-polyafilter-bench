import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pafbench.config import REPO
from pafbench.evaluate import (METRIC_COLUMNS, STRATA, ambiguous_tail_share, bootstrap, confusion, evaluate, expr_tertiles,
                               gene_clusters, hexamer_check, label_purity, rates, relabel, resample_counts, select_setting, spikein_metrics,
                               tail_share, units)
from pafbench.flags import Setting, setting_tag
from pafbench.grid import published

T0, T1, T2 = "c300_l10_m1_i", "c375_l3_m2_i", "c300_lnone_m0_i"
TAGS = [T0, T1, T2]

MOL_COLS = ["mol", "label", "a20", "a_band", "region", "gene", "has_ts", "hexamer", "ref", "strand", "site"]
MOLS = [
    (0, "templated", 15, "14-20", "3UTR", "gA", False, False, "chrA", "+", 1000),
    (1, "templated", 9, "8-9", "CDS", "gA", True, True, "chrA", "+", 900),
    (2, "nontemplated", 2, "0-4", "3UTR", "gB", False, True, "chrA", "-", 5000),
    (3, "ambiguous", 6, "5-7", "intron", "gB", False, False, "chrA", "-", 5600),
    (4, "nontemplated", 0, "0-4", "intergenic", "", False, False, "chrA", "+", 12345),
    (5, "templated", 12, "10-13", "3UTR", "gC", False, False, "chrB", "+", 300),
    (6, "templated", 20, "14-20", "3UTR", "gA", False, True, "chrA", "+", 1200),
]
# members: qname, mol, dist5_exon, dist_mode, spliced
MEMBERS = [
    ("r0a", 0, 100, "exon", False), ("r0b", 0, 300, "genome", False), ("r0c", 0, 900, "exon", False),
    ("r1a", 1, 400, "exon", True), ("r1b", 1, 200, "genome", False),
    ("r2a", 2, 150, "exon", False), ("r3a", 3, 250, "genome", False),
    ("r4a", 4, 1500, "genome", False), ("r5a", 5, 100, "exon", False),
]
# primary flags per qname for T0, T1 (T2 = remove-all control, True everywhere); r5a is missing
FLAGS = {"p0": (1, 0), "r0a": (1, 1), "r0b": (1, 0), "r0c": (1, 0), "p1": (0, 0), "r1a": (1, 0), "r1b": (1, 0),
         "p2": (0, 1), "r2a": (0, 1), "p3": (1, 0), "r3a": (1, 0), "p4": (0, 0), "r4a": (1, 0), "p5": (0, 0), "p6": (0, 0)}


def _tables():
    mol = pd.DataFrame(MOLS, columns=MOL_COLS)
    pa = pd.DataFrame({"qname": [f"p{m}" for m in range(7)], "mol": range(7), "role": "pa", "dist5_exon": np.nan,
                       "dist_mode": "", "spliced": False})
    mem = pd.DataFrame([(q, m, "member", float(d), mode, s) for q, m, d, mode, s in MEMBERS], columns=pa.columns)
    tr = pd.concat([pa, mem], ignore_index=True)
    rows = [(q, "chrA", 0, 0, bool(a), bool(b), True) for q, (a, b) in FLAGS.items()]
    rows.append(("r2a", "chrA", 0, 256, True, False, True))  # secondary record of r2a: must be ignored
    fl = pd.DataFrame(rows, columns=["qname", "ref", "pos", "flag"] + TAGS)
    return mol, tr, fl


def _row(m, level, kind, stratum, tag):
    r = m[(m.level == level) & (m.stratum_kind == kind) & (m.stratum == stratum) & (m.tag == tag)]
    assert len(r) == 1
    return r.iloc[0]


def _counts(r):
    return [r.tp, r.fn, r.tn, r.fp, r.n_amb, r.amb_removed]


def test_confusion_and_rates():
    removed = np.array([[1, 0], [0, 0], [1, 1], [0, 1], [1, 1]], bool)
    label = np.array(["templated", "templated", "nontemplated", "nontemplated", "ambiguous"], object)
    tp, fn, tn, fp, n_amb, amb_rem = confusion(removed, label)
    # tag 0: positives removed 1 of 2; negatives removed 1 of 2; the ambiguous one removed
    assert tp.tolist() == [1, 0] and fn.tolist() == [1, 2] and tn.tolist() == [1, 0] and fp.tolist() == [1, 2]
    assert n_amb.tolist() == [1, 1] and amb_rem.tolist() == [1, 1]
    sens, spec, prec = rates(tp, fn, tn, fp)
    assert sens.tolist() == [0.5, 0.0] and spec.tolist() == [0.5, 0.0] and prec.tolist() == [0.5, 0.0]
    w = np.array([0.5, 0.25, 1.0, 1.0, 2.0])
    tp, fn, tn, fp, n_amb, amb_rem = confusion(removed, label, w)
    assert tp.tolist() == [0.5, 0.0] and fn.tolist() == [0.25, 0.75] and n_amb.tolist() == [2.0, 2.0]
    # empty denominators give NaN: no positives, no negatives, nothing removed
    sens, spec, prec = rates(np.array([0]), np.array([0]), np.array([0]), np.array([0]))
    assert math.isnan(sens[0]) and math.isnan(spec[0]) and math.isnan(prec[0])
    sens, spec, prec = rates(np.array([0]), np.array([3]), np.array([2]), np.array([0]))
    assert sens[0] == 0.0 and spec[0] == 1.0 and math.isnan(prec[0])
    # other positive / negative values (spike-ins)
    tp, fn, tn, fp, n_amb, _ = confusion(removed, np.array(["internal", "tail", "tail", "other", "internal"], object),
                                         pos="internal", neg="tail")
    assert tp.tolist() == [2, 1] and fn.tolist() == [0, 1] and tn.tolist() == [1, 1] and fp.tolist() == [1, 1]
    assert n_amb.tolist() == [1, 1]


def test_units_reads_molecules_and_missing():
    mol, tr, fl = _tables()
    u = units(mol, tr, fl, TAGS, cap=1000)
    # m4 has its only member beyond 1000 nt, m6 has no member, m5 loses r5a (absent from the flags) and drops out
    assert u.mols.mol.tolist() == [0, 1, 2, 3]
    assert u.missing_reads == 1 and u.dropped_mols == 1
    assert u.mols.n_reads.tolist() == [4, 3, 2, 2]
    assert u.K[:, 0].tolist() == [4, 2, 0, 2] and u.K[:, 1].tolist() == [1, 0, 2, 0] and u.K[:, 2].tolist() == [4, 3, 2, 2]
    assert sorted(u.reads.qname) == ["r0a", "r0b", "r0c", "r1a", "r1b", "r2a", "r3a"]
    r = dict(zip(u.reads.qname, u.R[:, 0]))
    assert r["r2a"] == False and r["r1a"] == True  # noqa: E712  (primary row of r2a, not the secondary one)
    r1a = u.reads[u.reads.qname == "r1a"].iloc[0]
    assert (r1a.label, r1a.a_band, r1a.region, r1a.gene, r1a.has_ts, r1a.dist_mode, r1a.spliced) == \
        ("templated", "8-9", "CDS", "gA", True, "exon", True)
    u2 = units(mol, tr, fl, TAGS, cap=2000)
    assert u2.mols.mol.tolist() == [0, 1, 2, 3, 4] and u2.mols.n_reads.tolist()[-1] == 2 and u2.K[-1].tolist() == [1, 0, 2]
    # the cap is inclusive: r0c sits at dist5_exon 900
    assert units(mol, tr, fl, TAGS, cap=900).mols.n_reads.tolist()[0] == 4
    assert units(mol, tr, fl, TAGS, cap=899).mols.n_reads.tolist()[0] == 3


def test_units_all_labeled_adds_molecules_without_member_reads():
    mol, tr, fl = _tables()
    u, a = units(mol, tr, fl, TAGS, cap=1000), units(mol, tr, fl, TAGS, cap=1000, all_labeled=True)
    # m4 (only member beyond the cap) and m6 (no member) become units with their pa read; m5 still drops out
    assert a.mols.mol.tolist() == [0, 1, 2, 3, 4, 6] and a.mols.n_reads.tolist() == [4, 3, 2, 2, 1, 1]
    assert a.K[4:].tolist() == [[0, 0, 1], [0, 0, 1]] and a.K[:4].tolist() == u.K.tolist()
    assert a.missing_reads == 1 and a.dropped_mols == 1
    # read-level units do not change
    pd.testing.assert_frame_equal(a.reads.drop(columns="mi"), u.reads.drop(columns="mi"))
    assert (a.R == u.R).all()
    assert units(mol, tr, fl, TAGS, cap=1000, all_labeled=False).mols.mol.tolist() == [0, 1, 2, 3]


def test_expr_tertiles_match_qcut():
    mol, _, _ = _tables()
    # molecules per gene: gA 3, gB 2, gC 1 -> gC low, gB mid, gA high; no gene (intergenic or antisense) -> no_gene
    assert expr_tertiles(mol).tolist() == ["high", "high", "mid", "mid", "no_gene", "low", "high"]
    assert expr_tertiles(mol.iloc[0:0]).tolist() == [] and STRATA["expr_tertile"] == ("low", "mid", "high", "no_gene")
    genes = [f"g{i}" for i in range(5) for _ in range(i + 1)]  # g0 x1 ... g4 x5
    t = expr_tertiles(pd.DataFrame({"gene": genes}))
    ref = pd.qcut(pd.Series(np.arange(1, 6)), 3, labels=["low", "mid", "high"]).astype(str).tolist()
    assert [t[genes.index(f"g{i}")] for i in range(5)] == ref == ["low", "low", "mid", "high", "high"]
    assert expr_tertiles(pd.DataFrame({"gene": ["g1", "g1"]})).tolist() == ["low", "low"]


def test_levels_all_reads_versus_majority_and_weighted():
    mol, tr, fl = _tables()
    m = evaluate(units(mol, tr, fl, TAGS, cap=1000), "1417", 1000)
    # molecule rule: m0 removed (4/4), m1 not (2/3), m2 kept, m3 (ambiguous) removed
    assert _counts(_row(m, "molecule", "all", "all", T0)) == [1, 1, 1, 0, 1, 1]
    assert _counts(_row(m, "molecule", "all", "all", T1)) == [0, 2, 0, 1, 1, 0]
    # majority rule: m1 has 2 of 3 reads removed, so it counts as removed
    assert _counts(_row(m, "molecule_majority", "all", "all", T0)) == [2, 0, 1, 0, 1, 1]
    assert _counts(_row(m, "molecule_majority", "all", "all", T1)) == [0, 2, 0, 1, 1, 0]
    # weights 1 / n_reads: tp 1/4 (m0), fn 1/3 (m1), tn 1/2 (m2), ambiguous 1/2 removed
    w = _row(m, "molecule_weighted", "all", "all", T0)
    assert np.allclose(_counts(w), [0.25, 1 / 3, 0.5, 0, 0.5, 0.5]) and w.sens == pytest.approx(3 / 7) and w.spec == 1.0
    r = _row(m, "read", "all", "all", T1)
    assert _counts(r) == [1, 4, 0, 1, 1, 0] and r.sens == 0.2 and r.spec == 0.0 and r.prec == 0.5
    c = _row(m, "molecule", "all", "all", T2)
    assert _counts(c) == [2, 0, 0, 1, 1, 1] and c.prec == pytest.approx(2 / 3)
    m2 = evaluate(units(mol, tr, fl, TAGS, cap=2000), "1417", 2000)
    assert _counts(_row(m2, "molecule", "all", "all", T0))[2:4] == [2, 0]  # m4: 1 of 2 reads removed, kept
    assert _counts(_row(m2, "read", "all", "all", T0))[2:4] == [1, 1]  # r4a removed at read level
    assert _counts(_row(m2, "molecule_majority", "all", "all", T0))[2:4] == [2, 0]  # m4: half its reads is not a majority


def test_strata_and_columns():
    mol, tr, fl = _tables()
    m = evaluate(units(mol, tr, fl, TAGS, cap=1000), "1417", 1000)
    assert list(m.columns) == METRIC_COLUMNS
    assert str(m.min_len.dtype) == "Int64"
    ctrl = m[m.tag == T2].iloc[0]
    assert ctrl.covlen == 300 and pd.isna(ctrl.min_len) and ctrl.mism == 0 and ctrl.introns == True  # noqa: E712
    assert _row(m, "molecule", "all", "all", T0).min_len == 10
    assert set(m.cutoffs) == {"8/4"} and set(m.lib) == {"1417"} and set(m.cap) == {1000}
    assert _counts(_row(m, "molecule", "region", "3UTR", T0)) == [1, 0, 1, 0, 0, 0]
    assert _counts(_row(m, "molecule", "region", "CDS", T0)) == [0, 1, 0, 0, 0, 0]
    assert _counts(_row(m, "molecule", "region", "intron", T0)) == [0, 0, 0, 0, 1, 1]
    assert math.isnan(_row(m, "molecule", "region", "intergenic", T0).sens)
    assert _counts(_row(m, "molecule", "a_band", "8-9", T0)) == [0, 1, 0, 0, 0, 0]
    assert _counts(_row(m, "molecule", "expr_tertile", "high", T0)) == [1, 1, 0, 0, 0, 0]
    assert _counts(_row(m, "molecule", "expr_tertile", "mid", T0)) == [0, 0, 1, 0, 1, 1]
    assert _counts(_row(m, "molecule", "expr_tertile", "low", T0)) == [0] * 6
    assert _counts(_row(m, "molecule", "has_ts", "True", T0)) == [0, 1, 0, 0, 0, 0]
    assert _counts(_row(m, "read", "dist_mode", "exon", T1)) == [1, 2, 0, 1, 0, 0]
    assert _counts(_row(m, "read", "dist_mode", "genome", T1)) == [0, 2, 0, 0, 1, 0]
    assert _counts(_row(m, "read", "spliced", "True", T0)) == [1, 0, 0, 0, 0, 0]
    assert m[(m.level != "read") & m.stratum_kind.isin(["dist_mode", "spliced"])].empty
    per_level = m[m.tag == T0].groupby("level").size().to_dict()
    # all 1 + a_band 5 + region 7 + expr 4 + has_ts 2 = 19; read adds dist_mode 2 + spliced 2
    assert per_level == {"molecule": 19, "molecule_majority": 19, "molecule_weighted": 19, "read": 23}
    only = evaluate(units(mol, tr, fl, TAGS, cap=1000), "1417", 1000, levels=("molecule",), kinds=("all",))
    assert len(only) == 3 and _counts(only[only.tag == T0].iloc[0]) == [1, 1, 1, 0, 1, 1]


def test_relabel_changes_labels_only():
    mol, tr, fl = _tables()
    u = relabel(units(mol, tr, fl, TAGS, cap=1000), {"templated_min": 10, "nontemplated_max": 6})
    # a20 15, 9, 2, 6 -> templated, ambiguous, nontemplated, nontemplated
    assert u.mols.label.tolist() == ["templated", "ambiguous", "nontemplated", "nontemplated"]
    assert u.reads.set_index("qname").label["r1a"] == "ambiguous"
    m = evaluate(u, "1417", 1000, "10/6", levels=("molecule",), kinds=("all",))
    assert _counts(m[m.tag == T0].iloc[0]) == [1, 0, 1, 1, 1, 0]


def _metrics(rows):
    return pd.DataFrame([dict(lib=lib, cap=cap, cutoffs="8/4", level=level, stratum_kind="all", stratum="all", tag=tag,
                              sens=se, spec=sp, prec=0.9) for lib, cap, level, tag, se, sp in rows])


SEL_ROWS = [
    ("1417", 1000, "molecule", "c300_l10_m1_i", 0.6, 0.96),
    ("1417", 1000, "molecule", "c300_l9_m1_i", 0.7, 0.97),
    ("1417", 1000, "molecule", "c450_l9_m1_i", 0.7, 0.99),
    ("1417", 1000, "molecule", "c300_l3_m2_i", 0.9, 0.90),
    ("1417", 1000, "molecule", "c300_lnone_m0_i", 0.99, 0.99),  # control: never a candidate
    ("1417", 1000, "molecule", "c275_l9_m1_i", 0.95, 0.99),  # published but not on the grid
    ("1417", 1000, "molecule", "c300_l9_m1_e", 0.99, 0.99),  # other mode
    ("1419", 1000, "molecule", "c300_l3_m2_i", 0.99, 0.99),  # other library
    ("1417", 500, "molecule", "c300_l3_m2_i", 0.99, 0.99),  # other cap
    ("1417", 1000, "read", "c300_l3_m2_i", 0.99, 0.99),  # other level
]


def test_select_setting():
    m = _metrics(SEL_ROWS)
    removed = {"c300_l9_m1_i": 500, "c450_l9_m1_i": 400, "c300_l10_m1_i": 100, "c300_l3_m2_i": 900}
    s = select_setting(m, removed, introns=True)
    # c300_l9 and c450_l9 tie on sensitivity 0.7; c450 removes fewer records
    assert s["tag"] == "c450_l9_m1_i" and s["sens"] == 0.7 and s["spec"] == 0.99 and s["prec"] == 0.9
    assert s["n_candidates"] == 4 and s["n_qualifying"] == 3
    assert s["frontier"] == [["c300_l3_m2_i", 0.9, 0.90], ["c450_l9_m1_i", 0.7, 0.99]]
    s = select_setting(m, {"c300_l9_m1_i": 400, "c450_l9_m1_i": 400}, introns=True)
    assert s["tag"] == "c300_l9_m1_i"  # same removed count: tag name decides
    s = select_setting(m, removed, introns=True, min_spec=0.995)
    assert s["tag"] is None and s["sens"] is None and s["n_qualifying"] == 0
    assert s["frontier"] == [["c300_l3_m2_i", 0.9, 0.90], ["c450_l9_m1_i", 0.7, 0.99]]
    # specificity exactly at the threshold qualifies
    edge = _metrics(SEL_ROWS + [("1417", 1000, "molecule", "c600_l20_m0_i", 0.8, 0.95)])
    s = select_setting(edge, removed, introns=True)
    assert s["tag"] == "c600_l20_m0_i" and s["n_candidates"] == 5 and s["n_qualifying"] == 4
    assert s["frontier"] == [["c300_l3_m2_i", 0.9, 0.90], ["c600_l20_m0_i", 0.8, 0.95], ["c450_l9_m1_i", 0.7, 0.99]]
    e = select_setting(m, {}, introns=False)
    assert e["tag"] == "c300_l9_m1_e" and e["n_candidates"] == 1


def _boot_units(genes, removed, labels):
    """One pa read and one member per molecule, both removed or both kept, so a molecule is removed iff `removed`."""
    n = len(genes)
    mol = pd.DataFrame({"mol": range(n), "label": labels, "a20": 10, "a_band": "10-13", "region": "3UTR", "gene": genes,
                        "has_ts": False, "hexamer": False, "ref": "chrA", "strand": "+", "site": np.arange(n) * 10_000})
    q = [f"p{i}" for i in range(n)] + [f"r{i}" for i in range(n)]
    tr = pd.DataFrame({"qname": q, "mol": list(range(n)) * 2, "role": ["pa"] * n + ["member"] * n,
                       "dist5_exon": [np.nan] * n + [100.0] * n, "dist_mode": [""] * n + ["exon"] * n, "spliced": False})
    fl = pd.DataFrame({"qname": q, "ref": "chrA", "pos": 0, "flag": 0, T0: list(removed) * 2})
    return units(mol, tr, fl, [T0], cap=1000)


def test_gene_clusters():
    mol = pd.DataFrame({"gene": ["gA", "", ""], "ref": ["chr1", "chr1", "chr2"], "strand": ["+", "-", "+"],
                        "site": [5, 12345, 9999]})
    assert gene_clusters(mol).tolist() == ["gA", "chr1:-:1", "chr2:+:0"]


def test_bootstrap_matches_brute_force_and_is_reproducible():
    rng = np.random.default_rng(1)
    genes = [f"g{i % 7}" for i in range(60)] + ["", ""]
    labels = rng.choice(["templated", "nontemplated", "ambiguous"], 62, p=[0.6, 0.3, 0.1])
    removed = rng.random(62) < 0.5
    u = _boot_units(genes, removed, labels)
    b = bootstrap(u, [T0], n=200, seed=3)
    assert b.equals(bootstrap(u, [T0], n=200, seed=3))
    assert not b.equals(bootstrap(u, [T0], n=200, seed=4))
    assert list(b.columns) == ["tag", "metric", "estimate", "lo", "hi", "n_boot", "n_clusters"]
    assert b.metric.tolist() == ["sens", "spec", "prec"] and set(b.n_clusters) == {9}  # 7 genes + 2 intergenic loci
    # brute force: weight each molecule by the draw count of its cluster (clusters in sorted order)
    cl = gene_clusters(u.mols)
    names = np.unique(cl)
    counts = resample_counts(len(names), 200, 3)
    w = counts[:, np.searchsorted(names, cl)].astype(float)
    rem = (u.K[:, 0] == u.mols.n_reads.to_numpy())
    lab = u.mols.label.to_numpy()
    P, N = lab == "templated", lab == "nontemplated"
    sens = (w * (P & rem)).sum(1) / (w * P).sum(1)
    spec = (w * (N & ~rem)).sum(1) / (w * N).sum(1)
    for metric, vals in (("sens", sens), ("spec", spec)):
        r = b[b.metric == metric].iloc[0]
        assert [r.lo, r.hi] == pytest.approx(np.nanpercentile(vals, [2.5, 97.5]).tolist())
        assert r.lo <= r.estimate <= r.hi
    est = evaluate(u, "x", 1000, levels=("molecule",), kinds=("all",)).iloc[0]
    assert b[b.metric == "sens"].estimate.iloc[0] == est.sens and b[b.metric == "prec"].estimate.iloc[0] == est.prec


def test_bootstrap_single_gene_is_degenerate():
    u = _boot_units(["g1"] * 4, [True, False, True, False], ["templated", "templated", "nontemplated", "nontemplated"])
    b = bootstrap(u, [T0], n=50, seed=0)
    assert (b.lo == b.hi).all() and (b.lo == b.estimate).all() and set(b.n_clusters) == {1}
    assert b.estimate.tolist() == [0.5, 0.5, 0.5]
    r = bootstrap(u, [T0], level="read", n=20, seed=0)
    assert r.estimate.tolist() == [0.5, 0.5, 0.5] and (r.lo == r.hi).all()


def test_hexamer_check():
    mol, tr, fl = _tables()
    h = hexamer_check(units(mol, tr, fl, TAGS, cap=1000), [T0]).set_index("label")
    # T0 removes m0 (hexamer False) and m3 (False); keeps m1 (True) and m2 (True)
    assert list(h.columns) == ["tag", "removed_n", "removed_hex_rate", "kept_n", "kept_hex_rate"]
    assert h.loc["all"].tolist() == [T0, 2, 0.0, 2, 1.0]
    assert h.loc["templated"].tolist() == [T0, 1, 0.0, 1, 1.0]
    assert h.loc["nontemplated", "removed_n"] == 0 and math.isnan(h.loc["nontemplated", "removed_hex_rate"])
    assert h.loc["nontemplated", "kept_hex_rate"] == 1.0
    assert h.loc["ambiguous", "removed_n"] == 1 and h.loc["ambiguous", "kept_n"] == 0


def test_spikein_labels_and_modules():
    mol, tr, fl = _tables()
    mol = mol.assign(spike_label=["internal", "internal", "tail", "other", "tail", "internal", "internal"],
                     module=["ERCC", "SIRV_E0", "ERCC", "ERCC", "ERCC", "ERCC", "ERCC"])
    u = units(mol, tr, fl, TAGS, cap=1000, label="spike_label", pos="internal", neg="tail", amb=None)
    assert u.mols.mol.tolist() == [0, 1, 2]  # m3 ("other") excluded entirely
    s = spikein_metrics(u, [T0, T1])
    assert set(s.level) == {"read", "molecule"} and set(s.stratum_kind) == {"all", "module"}
    assert set(s.stratum) == {"all", "ERCC", "SIRV_E0", "long_SIRV"}
    r = s[(s.level == "molecule") & (s.stratum == "all") & (s.tag == T0)].iloc[0]
    assert _counts(r) == [1, 1, 1, 0, 0, 0]
    r = s[(s.level == "molecule") & (s.stratum == "SIRV_E0") & (s.tag == T0)].iloc[0]
    assert _counts(r) == [0, 1, 0, 0, 0, 0]
    r = s[(s.level == "read") & (s.stratum == "ERCC") & (s.tag == T1)].iloc[0]
    assert _counts(r) == [1, 2, 0, 1, 0, 0]  # r0a removed, r0b, r0c kept; r2a (tail) removed
    assert s[(s.stratum == "long_SIRV")][["tp", "fn", "tn", "fp"]].to_numpy().sum() == 0


def _run_stage(script, base, out, *extra, check=True):
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / script), "--base", str(base), "--out", str(out), *extra],
                       capture_output=True, text=True, env=env, cwd=REPO)
    if not check:
        return r
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_stage17_end_to_end(tmp_path):
    mol, tr, fl = _tables()
    mol = mol.assign(in_cell=True)
    base, out = tmp_path / "base", tmp_path / "eval"
    for lib in ("1417", "1419"):
        (base / lib / "truth").mkdir(parents=True)
        (base / lib / "grid").mkdir()
        mol.to_parquet(base / lib / "truth" / "molecules_labeled.parquet")
        tr.to_parquet(base / lib / "truth" / "truth_reads.parquet")
        fl.to_parquet(base / lib / "grid" / "truth_flags.parquet")
        pd.DataFrame({"tag": TAGS, "re": "E", "removed": [10, 20, 30], "total": 100}).to_parquet(
            base / lib / "grid" / "region_counts.parquet")
    spike = mol.assign(spike_label=["internal", "internal", "tail", "other", "tail", "internal", "internal"],
                       module=["ERCC", "SIRV_E0", "ERCC", "ERCC", "ERCC", "ERCC", "long_SIRV"])
    (base / "1417" / "spikein" / "truth").mkdir(parents=True)
    (base / "1417" / "spikein" / "grid").mkdir()
    spike.to_parquet(base / "1417" / "spikein" / "truth" / "molecules_labeled.parquet")
    tr.to_parquet(base / "1417" / "spikein" / "truth" / "truth_reads.parquet")
    fl.to_parquet(base / "1417" / "spikein" / "grid" / "truth_flags.parquet")
    stdout = _run_stage("17_evaluate.py", base, out)
    assert "1179: skipped" in stdout and "1419: spike-in metrics skipped" in stdout
    m = pd.read_parquet(out / "metrics.parquet")
    assert set(m.lib) == {"1417", "1419"} and set(m.cap) == {500, 1000, 2000}
    got = m[(m.lib == "1417") & (m.cap == 1000)].reset_index(drop=True)
    # the stage orders tags as all_settings() does: the control precedes the other settings of its COVLEN
    pd.testing.assert_frame_equal(got, evaluate(units(mol, tr, fl, [T2, T0, T1], 1000), "1417", 1000), check_dtype=False)
    sel = json.loads((out / "selection.json").read_text())
    # mode i: T0 (sens 0.5, spec 1.0) qualifies, T1 (spec 0) does not, T2 is the control; mode e has no candidate
    assert sel["modes"]["i"]["selection"]["tag"] == T0 and sel["modes"]["e"]["selection"]["tag"] is None
    assert sel["modes"]["e"]["selection"]["n_candidates"] == 0
    assert sel["modes"]["i"]["metrics"]["1419"]["molecule"]["sens"] == 0.5
    assert sel["cells_tuned"] == {}  # 1179 is absent
    assert set(sel["label_purity"]) == {"1417", "1419"}
    assert sel["modes"]["i"]["metrics"]["1417"]["molecule_majority"]["tp"] == 2.0
    assert sel["published"][T1]["1417"]["read"]["prec"] == 0.5 and sel["published"]["c180_l11_m2_i"]["1417"] == {}
    assert sel["cap_sensitivity"]["2000"]["i"]["tag"] == T0
    assert len(sel["cutoff_sweep"]) == 40
    sw = {(d["templated_min"], d["nontemplated_max"], d["mode"]): d for d in sel["cutoff_sweep"]}
    assert sw[(8, 4, "i")]["tag"] == T0 and sw[(8, 4, "i")]["by_library"]["1419"]["sens"] == 0.5
    assert sw[(10, 6, "i")]["tag"] is None  # T0 spec 0.5 and T1 spec 0.5 at 10/6
    assert sel["units"]["1417"]["1000"] == {"read_units": 7, "molecule_units": 4, "missing_reads": 1, "dropped_molecules": 1}
    assert sel["bootstrapped_tags"][0] == T0 and sel["bootstrapped_tags"][-1] == T2
    b = pd.read_parquet(out / "bootstrap.parquet")
    assert len(b) == 2 * 2 * 3 * 3 and set(b.tag) == {T0, T1, T2}
    r = b[(b.lib == "1417") & (b.level == "molecule") & (b.tag == T0) & (b.metric == "sens")].iloc[0]
    assert r.estimate == 0.5
    h = pd.read_parquet(out / "hexamer_check.parquet")
    assert len(h) == 2 * 3 * 4
    s = pd.read_parquet(out / "spikein_metrics.parquet")
    assert set(s.lib) == {"1417"} and set(s.cap) == {1000}
    ref = spikein_metrics(units(spike, tr, fl, [T2, T0, T1], 1000, label="spike_label", pos="internal", neg="tail", amb=None),
                          [T2, T0, T1])
    pd.testing.assert_frame_equal(s.drop(columns=["lib", "cap"]), ref, check_dtype=False)
    r = s[(s.level == "molecule") & (s.stratum == "all") & (s.tag == T0)].iloc[0]
    assert _counts(r) == [1, 1, 1, 0, 0, 0]  # m0 internal removed, m1 internal kept, m2 tail kept


IN_CELL = [True, True, False, False, True, True, False]
POP_MOLS = {"all": [0, 1, 2, 3, 4, 5, 6], "cells": [0, 1, 4, 5], "noncells": [2, 3, 6]}
COMBOS = [(p, u) for p in ("all", "cells", "noncells") for u in ("with_member", "all_labeled")]


def _pop_base(tmp_path, mol, tr, fl):
    base = tmp_path / "base"
    for lib in ("1417", "1419"):
        (base / lib / "truth").mkdir(parents=True)
        (base / lib / "grid").mkdir()
        mol.to_parquet(base / lib / "truth" / "molecules_labeled.parquet")
        tr.to_parquet(base / lib / "truth" / "truth_reads.parquet")
        fl.to_parquet(base / lib / "grid" / "truth_flags.parquet")
        pd.DataFrame({"tag": TAGS, "re": "E", "removed": [10, 20, 30], "total": 100}).to_parquet(
            base / lib / "grid" / "region_counts.parquet")
    # spike-in molecules sit in no called cell: the spike-in metrics must not depend on --population
    spike = mol.assign(spike_label=["internal", "internal", "tail", "other", "tail", "internal", "internal"], module="ERCC",
                       in_cell=False)
    for sub in ("truth", "grid"):
        (base / "1417" / "spikein" / sub).mkdir(parents=True)
    spike.to_parquet(base / "1417" / "spikein" / "truth" / "molecules_labeled.parquet")
    tr.to_parquet(base / "1417" / "spikein" / "truth" / "truth_reads.parquet")
    fl.to_parquet(base / "1417" / "spikein" / "grid" / "truth_flags.parquet")
    return base, spike


def test_stage17_population_and_units_options(tmp_path):
    mol, tr, fl = _tables()
    mol = mol.assign(in_cell=IN_CELL, mapq_min=[255, 255, 3, 255, 255, 255, 255])
    base, spike = _pop_base(tmp_path, mol, tr, fl)
    tags = [T2, T0, T1]
    runs = {}
    for pop, um in (("all", "with_member"), ("cells", "all_labeled"), ("noncells", "with_member")):
        out = tmp_path / f"eval_{pop}_{um}"
        _run_stage("17_evaluate.py", base, out, "--population", pop, "--units", um)
        runs[pop, um] = sel = json.loads((out / "selection.json").read_text())
        assert (sel["selection_population"], sel["selection_units"], sel["spikein_barcodes"]) == (pop, um, "all")
        sub = mol[mol.mol.isin(POP_MOLS[pop])].reset_index(drop=True)
        m = pd.read_parquet(out / "metrics.parquet")
        got = m[(m.lib == "1419") & (m.cap == 1000)].reset_index(drop=True)
        u = units(sub, tr, fl, tags, 1000, all_labeled=um == "all_labeled")
        pd.testing.assert_frame_equal(got, evaluate(u, "1419", 1000), check_dtype=False)
        assert sel["units"]["1419"]["1000"]["molecule_units"] == len(u.mols)
        s = pd.read_parquet(out / "spikein_metrics.parquet")
        us = units(spike, tr, fl, tags, 1000, label="spike_label", pos="internal", neg="tail", amb=None, all_labeled=um == "all_labeled")
        pd.testing.assert_frame_equal(s.drop(columns=["lib", "cap"]), spikein_metrics(us, tags), check_dtype=False)
        assert sel["spikein_units"] == {"1417": {"read_units": len(us.reads), "molecule_units": len(us.mols),
                                                 "missing_reads": us.missing_reads, "dropped_molecules": us.dropped_mols}}
    assert runs["all", "with_member"]["spikein_units"]["1417"]["molecule_units"] == 3  # m0, m1, m2: m4 and m6 have no member
    assert runs["cells", "all_labeled"]["spikein_units"]["1417"]["molecule_units"] == 5  # plus m4 and m6, not m5 (read missing)
    # populations.parquet and the "populations" block do not depend on the selection population and units
    pops = [pd.read_parquet(tmp_path / f"eval_{p}_{u}" / "populations.parquet") for p, u in runs]
    pd.testing.assert_frame_equal(pops[0], pops[1])
    pd.testing.assert_frame_equal(pops[0], pops[2])
    assert list(pops[0].columns) == METRIC_COLUMNS + ["population", "units"]
    assert set(zip(pops[0].population, pops[0].units)) == set(COMBOS)
    assert set(pops[0].cap) == {1000} and set(pops[0].cutoffs) == {"8/4"} and set(pops[0].lib) == {"1417", "1419"}
    assert set(pops[0].level) == {"molecule", "read"} and set(pops[0].stratum_kind) == {"all", "a_band", "region"}
    for pop, um in COMBOS:
        sub = mol[mol.mol.isin(POP_MOLS[pop])].reset_index(drop=True)
        ref = evaluate(units(sub, tr, fl, tags, 1000, all_labeled=um == "all_labeled"), "1417", 1000, levels=("molecule", "read"),
                       kinds=("all", "a_band", "region"))
        got = pops[0][(pops[0].population == pop) & (pops[0].units == um) & (pops[0].lib == "1417")]
        pd.testing.assert_frame_equal(got.drop(columns=["population", "units"]).reset_index(drop=True), ref, check_dtype=False)
    def bare(sel):
        return {pop: {um: {mode: {k: v for k, v in d.items() if k != "headline"} for mode, d in modes.items()}
                      for um, modes in v.items()} for pop, v in sel["populations"].items()}

    assert bare(runs["all", "with_member"]) == bare(runs["cells", "all_labeled"]) == bare(runs["noncells", "with_member"])
    # headline: the tag chosen per mode in the run's population and units, scored in every population and units
    for (hp, hu), sel in runs.items():
        for pop, um in COMBOS:
            for mode, d in sel["populations"][pop][um].items():
                tag = sel["modes"][mode]["selection"]["tag"]
                assert d["headline"]["tag"] == tag
                if tag is None:
                    assert d["headline"]["metrics"] == {}
                    continue
                assert set(d["headline"]["metrics"]) == {"1417", "1419"}
                for lib, lv in d["headline"]["metrics"].items():
                    sub = mol[mol.mol.isin(POP_MOLS[pop])].reset_index(drop=True)
                    ref = evaluate(units(sub, tr, fl, tags, 1000, all_labeled=um == "all_labeled"), lib, 1000, levels=("molecule",),
                                   kinds=("all",))
                    r = ref[(ref.tag == tag) & (ref.cutoffs == "8/4") & (ref.stratum_kind == "all")].iloc[0]
                    assert lv["molecule"]["tp"] == r.tp and lv["molecule"]["tn"] == r.tn
                    assert lv["molecule"]["sens"] == (None if pd.isna(r.sens) else pytest.approx(r.sens))
                    assert set(lv) == {"molecule"}
                if (pop, um) == (hp, hu):
                    assert d["headline"]["metrics"] == {lib: {"molecule": v["molecule"]}
                                                        for lib, v in sel["modes"][mode]["metrics"].items()}
    block = runs["all", "with_member"]["populations"]
    assert set(block) == {"all", "cells", "noncells"} and all(set(v) == {"with_member", "all_labeled"} for v in block.values())
    assert block["all"]["with_member"]["i"]["selection"] == runs["all", "with_member"]["modes"]["i"]["selection"]
    # cells with a member: no negative unit, so no setting has a specificity; cells with all labeled: m4 is a kept negative
    assert block["cells"]["with_member"]["i"]["selection"]["tag"] is None and block["cells"]["with_member"]["i"]["metrics"] == {}
    c = block["cells"]["all_labeled"]["i"]
    assert c["selection"]["tag"] == T0 and c["introns"] is True and c["selection"]["n_qualifying"] == 2  # T1 keeps the negative too, sens 0
    assert c["metrics"]["1419"]["molecule"] == {"tp": 1.0, "fn": 1.0, "tn": 1.0, "fp": 0.0, "n_amb": 0.0, "amb_removed": 0.0,
                                                "sens": 0.5, "spec": 1.0, "prec": 1.0}
    # molecule counts per library: labeled, in called cells, molecule units by label for each population and units
    cnt = runs["all", "with_member"]["molecules"]["1417"]
    assert cnt["labeled"] == 7 and cnt["in_cell"] == 4
    lab = lambda t, n, a: {"templated": t, "nontemplated": n, "ambiguous": a}  # noqa: E731
    assert cnt["units"] == {"all": {"with_member": lab(2, 1, 1), "all_labeled": lab(3, 2, 1)},
                            "cells": {"with_member": lab(2, 0, 0), "all_labeled": lab(2, 1, 0)},
                            "noncells": {"with_member": lab(0, 1, 1), "all_labeled": lab(1, 1, 1)}}
    # molecules with mapq_min below 255 (m2) left out: the tail unit that T1 removed is gone
    mq = runs["all", "with_member"]["mapq_sensitivity"]
    assert mq["left_out"] == {"1417": 1, "1419": 1} and mq["modes"]["i"]["selection"]["tag"] is None
    assert runs["all", "with_member"]["published"][T1]["1417"]["molecule"]["fp"] == 1.0
    assert mq["published"][T1]["1417"]["molecule"]["fp"] == 0.0 and mq["published"][T1]["1417"]["molecule"]["fn"] == 2.0
    assert set(mq["published"][T1]["1417"]) == {"molecule"}


def test_stage17_population_without_in_cell_is_refused(tmp_path):
    mol, tr, fl = _tables()
    base, _ = _pop_base(tmp_path, mol, tr, fl)
    r = _run_stage("17_evaluate.py", base, tmp_path / "eval", "--population", "cells", check=False)
    assert r.returncode != 0 and "in_cell" in r.stderr and not (tmp_path / "eval" / "selection.json").exists()
    r = _run_stage("17_evaluate.py", base, tmp_path / "eval_default", check=False)  # the default is called cells
    assert r.returncode != 0 and "in_cell" in r.stderr and not (tmp_path / "eval_default" / "selection.json").exists()
    out = tmp_path / "eval_all"
    _run_stage("17_evaluate.py", base, out, "--population", "all")
    sel = json.loads((out / "selection.json").read_text())
    assert set(pd.read_parquet(out / "populations.parquet").population) == {"all"}
    assert set(sel["populations"]) == {"all"} and "mapq_sensitivity" not in sel
    assert sel["molecules"]["1417"]["in_cell"] is None


def test_stage17_defaults_to_called_cells(tmp_path):
    from pafbench.config import SEL_POPULATION, SEL_UNITS
    assert (SEL_POPULATION, SEL_UNITS) == ("cells", "with_member")
    mol, tr, fl = _tables()
    mol = mol.assign(in_cell=IN_CELL)
    base, _ = _pop_base(tmp_path, mol, tr, fl)
    out = tmp_path / "eval"
    _run_stage("17_evaluate.py", base, out)
    sel = json.loads((out / "selection.json").read_text())
    assert (sel["selection_population"], sel["selection_units"]) == ("cells", "with_member")
    sub = mol[mol.mol.isin(POP_MOLS["cells"])].reset_index(drop=True)
    m = pd.read_parquet(out / "metrics.parquet")
    got = m[(m.lib == "1417") & (m.cap == 1000)].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, evaluate(units(sub, tr, fl, [T2, T0, T1], 1000), "1417", 1000), check_dtype=False)


def test_select_setting_reads_only_stratum_all_and_its_cutoffs():
    m = _metrics(SEL_ROWS)
    # qualifying rows with a higher sensitivity than the choice, in another stratum and at other cut-offs
    extra = _metrics([("1417", 1000, "molecule", "c300_l5_m1_i", 0.95, 0.99), ("1417", 1000, "molecule", "c200_l5_m1_i", 0.96, 0.99)])
    extra.loc[0, ["stratum_kind", "stratum"]] = ["region", "3UTR"]
    extra.loc[1, "cutoffs"] = "7/2"
    m = pd.concat([m, extra], ignore_index=True)
    removed = {"c300_l9_m1_i": 500, "c450_l9_m1_i": 400, "c300_l10_m1_i": 100, "c300_l3_m2_i": 900}
    s = select_setting(m, removed, introns=True)
    assert s["tag"] == "c450_l9_m1_i" and s["n_candidates"] == 4 and s["n_qualifying"] == 3
    s = select_setting(m, removed, introns=True, cutoffs="7/2")
    assert s["tag"] == "c200_l5_m1_i" and s["n_candidates"] == 1 and s["frontier"] == [["c200_l5_m1_i", 0.96, 0.99]]


def test_bootstrap_drops_clusters_without_positive_or_negative_units():
    labels = ["templated", "nontemplated", "templated", "nontemplated", "templated", "ambiguous", "ambiguous", "templated",
              "nontemplated"]
    genes = ["g1", "g1", "g2", "g2", "g3", "g4", "g4", "", ""]
    removed = [True, False, False, True, True, True, False, False, True]
    u = _boot_units(genes, removed, labels)
    # g4 holds only ambiguous molecules: it adds nothing to tp, fn, tn or fp and is left out of the resampled clusters,
    # so the draws equal those of the same units without g4 (g1, g2, g3 and two intergenic loci, in the same sorted order)
    keep = [i for i, g in enumerate(genes) if g != "g4"]
    v = _boot_units([genes[i] for i in keep], [removed[i] for i in keep], [labels[i] for i in keep])
    for level in ("molecule", "read", "molecule_weighted"):
        b = bootstrap(u, [T0], level=level, n=200, seed=7)
        assert set(b.n_clusters) == {5}
        pd.testing.assert_frame_equal(b, bootstrap(v, [T0], level=level, n=200, seed=7))
    # a cluster with only positive units stays: without g1's negative, g1 still counts (5 clusters)
    w = _boot_units(genes[:1] + genes[2:], removed[:1] + removed[2:], labels[:1] + labels[2:])
    assert set(bootstrap(w, [T0], n=50, seed=1).n_clusters) == {5}


def test_ambiguous_tail_share():
    mol = pd.DataFrame({"label": ["templated"] * 10 + ["nontemplated"] * 5 + ["ambiguous"] * 4,
                        "hexamer": [True] + [False] * 9 + [True] * 4 + [False] + [True, True, False, False],
                        "a20": [20] * 4 + [9] * 6 + [1] * 5 + [6] * 4})
    # rates: templated 1/10, non-templated 4/5, ambiguous 2/4; (0.5 - 0.1) / (0.8 - 0.1) = 4/7
    # a20 >= 14 reference: the first four molecules, rate 1/4: (0.5 - 0.25) / (0.8 - 0.25) = 5/11
    s = ambiguous_tail_share(mol)
    assert s == {"estimate": pytest.approx(4 / 7), "estimate_ref14": pytest.approx(5 / 11), "rate_templated": 0.1,
                 "rate_nontemplated": 0.8, "rate_ambiguous": 0.5, "rate_ref14": 0.25,
                 "n_templated": 10, "n_nontemplated": 5, "n_ambiguous": 4}
    # clipped to [0, 1]: ambiguous rates above the non-templated and below the templated rate
    hi = mol.assign(hexamer=[True] + [False] * 9 + [True] * 4 + [False] + [True] * 4)
    assert ambiguous_tail_share(hi)["estimate"] == 1.0 and ambiguous_tail_share(hi)["estimate_ref14"] == 1.0
    lo = mol.assign(hexamer=[True] * 2 + [False] * 8 + [True] * 4 + [False] + [False] * 4)
    assert ambiguous_tail_share(lo)["estimate"] == 0.0 and ambiguous_tail_share(lo)["estimate_ref14"] == 0.0
    # undefined: no ambiguous molecule, or equal templated and non-templated rates
    s = ambiguous_tail_share(mol[mol.label != "ambiguous"])
    assert s["estimate"] is None and s["estimate_ref14"] is None and s["rate_ambiguous"] is None and s["n_ambiguous"] == 0
    s = ambiguous_tail_share(mol.assign(hexamer=True))
    assert s["estimate"] is None and s["estimate_ref14"] is None
    # no molecule with a20 >= 14: no reference rate
    s = ambiguous_tail_share(mol.assign(a20=5))
    assert s["rate_ref14"] is None and s["estimate_ref14"] is None and s["estimate"] == pytest.approx(4 / 7)


def _purity_mol(spec):
    """Molecules from (a20, label, n, n with a hexamer) rows."""
    rows = [(a, lab, i < k) for a, lab, n, k in spec for i in range(n)]
    return pd.DataFrame(rows, columns=["a20", "label", "hexamer"]).astype({"hexamer": bool})


PURITY = [(2, "nontemplated", 10, 6), (6, "ambiguous", 4, 2), (8, "templated", 10, 2), (11, "templated", 10, 1),
          (16, "templated", 10, 1), (20, "templated", 10, 0)]


def test_label_purity():
    p = label_purity(_purity_mol(PURITY))
    # rates per band: 0.6, 0.5, 0.2, 0.1, 0.05; reference (a20 >= 14) 0.05; tail-primed share (rate - 0.05) / 0.55
    assert list(p["bands"]) == ["0-4", "5-7", "8-9", "10-13", "14-20"]
    assert {b: v["n"] for b, v in p["bands"].items()} == {"0-4": 10, "5-7": 4, "8-9": 10, "10-13": 10, "14-20": 20}
    assert [v["hexamer_rate"] for v in p["bands"].values()] == pytest.approx([0.6, 0.5, 0.2, 0.1, 0.05])
    assert [v["tail_primed_share"] for v in p["bands"].values()] == pytest.approx([1, 9 / 11, 3 / 11, 1 / 11, 0])
    assert p["reference"] == {"n": 20, "hexamer_rate": pytest.approx(0.05)}
    # classes by label: templated 40 molecules, 4 hexamers (rate 0.1); ambiguous is the 5-7 band
    assert p["classes"]["templated"] == {"n": 40, "hexamer_rate": pytest.approx(0.1), "tail_primed_share": pytest.approx(1 / 11)}
    assert p["classes"]["ambiguous"] == {"n": 4, "hexamer_rate": 0.5, "tail_primed_share": pytest.approx(9 / 11)}
    # templated share over the 54 labeled molecules, then times (1 - 1/11)
    assert p["n_labeled"] == 54 and p["templated_share"] == pytest.approx(40 / 54)
    assert p["templated_share_corrected"] == pytest.approx(40 / 54 * 10 / 11)


def test_label_purity_clips_and_undefined_cases():
    # a band above the 0-4 rate clips to 1, below the reference to 0
    p = label_purity(_purity_mol([(2, "nontemplated", 10, 5), (6, "ambiguous", 4, 4), (8, "templated", 10, 0),
                                  (16, "templated", 10, 2)]))
    assert p["bands"]["5-7"]["tail_primed_share"] == 1.0 and p["bands"]["8-9"]["tail_primed_share"] == 0.0
    # no 0-4 molecule, or a 0-4 rate equal to the reference rate: no estimates, no corrected share
    for spec in (PURITY[1:], [(2, "nontemplated", 10, 1)] + PURITY[1:4] + [(16, "templated", 10, 1)]):
        p = label_purity(_purity_mol(spec))
        assert all(v["tail_primed_share"] is None for v in p["bands"].values())
        assert p["classes"]["templated"]["tail_primed_share"] is None and p["templated_share_corrected"] is None
        assert p["templated_share"] == pytest.approx(sum(n for _, lab, n, _ in spec if lab == "templated") / sum(n for _, _, n, _ in spec))
    # no a20 >= 14 molecule: no reference; an empty band has no rate
    p = label_purity(_purity_mol(PURITY[:3]))
    assert p["reference"] == {"n": 0, "hexamer_rate": None} and p["bands"]["14-20"] == {"n": 0, "hexamer_rate": None, "tail_primed_share": None}
    assert p["bands"]["8-9"]["tail_primed_share"] is None
    # no molecules at all
    e = label_purity(_purity_mol([]))
    assert e["n_labeled"] == 0 and e["templated_share"] is None and e["templated_share_corrected"] is None
    assert e["classes"]["templated"] == {"n": 0, "hexamer_rate": None, "tail_primed_share": None}


def test_tail_share_with_the_frames_own_rates_is_label_purity():
    for spec in (PURITY, [(2, "nontemplated", 10, 5), (6, "ambiguous", 4, 4), (8, "templated", 10, 0), (16, "templated", 10, 2)]):
        mol = _purity_mol(spec)
        p = label_purity(mol)
        own = tail_share(mol, p["reference"]["hexamer_rate"], p["bands"]["0-4"]["hexamer_rate"])
        assert own == {k: v for k, v in p.items() if k != "reference"}


def test_tail_share_takes_the_reference_rates_as_given():
    mol = _purity_mol(PURITY)
    # band rates 0.6, 0.5, 0.2, 0.1, 0.05; given rates 0.1 (reference) and 0.6 (0-4): (rate - 0.1) / 0.5, clipped to [0, 1]
    t = tail_share(mol, 0.1, 0.6)
    assert {b: v["n"] for b, v in t["bands"].items()} == {"0-4": 10, "5-7": 4, "8-9": 10, "10-13": 10, "14-20": 20}
    assert [v["tail_primed_share"] for v in t["bands"].values()] == pytest.approx([1, 0.8, 0.2, 0, 0])
    # the templated class (rate 0.1) equals the given reference: nothing to remove
    assert t["classes"]["templated"]["tail_primed_share"] == 0.0 and t["templated_share_corrected"] == pytest.approx(40 / 54)
    # a rate missing, or the two rates equal: no estimates, the templated share stays
    for ref, tail in ((None, 0.6), (0.1, None), (0.3, 0.3)):
        t = tail_share(mol, ref, tail)
        assert all(v["tail_primed_share"] is None for v in t["bands"].values())
        assert t["templated_share_corrected"] is None and t["templated_share"] == pytest.approx(40 / 54)
    e = tail_share(_purity_mol([]), 0.1, 0.6)
    assert e["n_labeled"] == 0 and e["templated_share"] is None and e["templated_share_corrected"] is None
    assert e["bands"]["0-4"] == {"n": 0, "hexamer_rate": None, "tail_primed_share": None}


E1, E2, E3, CE, I1, I2 ="c300_l10_m1_e", "c450_l9_m1_e", "c150_l20_m0_e", "c450_lnone_m0_e", "c300_l10_m1_i", "c375_l3_m2_i"
SEL_TAGS = [E1, E2, E3, CE, I1, I2]
# id, label, a20, gene, member dist5_exon (None: no member), hexamer; Tm's member is absent from truth_flags
SEL_MOLS = [(0, "templated", 15, "gA", 100, False), (1, "templated", 15, "gB", 100, False), (2, "templated", 15, "gC", 700, False),
            (3, "templated", 15, "gD", 1500, False), (4, "templated", 15, "gE", 100, False),
            (5, "nontemplated", 2, "gA", 100, True), (6, "nontemplated", 2, "gB", 100, True), (7, "ambiguous", 6, "gC", 100, True),
            (8, "templated", 15, "gF", None, True), (9, "templated", 15, "gF", None, False), (10, "nontemplated", 2, "gF", None, True),
            (11, "nontemplated", 2, "gF", None, False), (12, "ambiguous", 6, "gF", None, False)]
# molecules removed (both reads) per tag: E1 Ta Tc Td; E2 Tb Tc; E3 Ta; CE all; I1 Ta Na; I2 Ta Tb Tc Nb
SEL_REMOVED = {E1: {0, 2, 3}, E2: {1, 2}, E3: {0}, CE: set(range(13)), I1: {0, 5}, I2: {0, 1, 2, 6}}


def _sel_tables():
    mol = pd.DataFrame([(i, lab, a20, "14-20" if lab == "templated" else "0-4" if lab == "nontemplated" else "5-7", "3UTR", g,
                         False, h, "chrA", "+", 1000 * (i + 1)) for i, lab, a20, g, _, h in SEL_MOLS], columns=MOL_COLS).assign(in_cell=True)
    rows = [(f"p{i}", i, "pa", np.nan, "", False) for i, *_ in SEL_MOLS]
    rows += [(f"r{i}", i, "member", float(d), "exon", False) for i, _, _, _, d, _ in SEL_MOLS if d is not None]
    tr = pd.DataFrame(rows, columns=["qname", "mol", "role", "dist5_exon", "dist_mode", "spliced"])
    q = [x for x in tr.qname if x != "r4"]
    fl = pd.DataFrame({"qname": q, "ref": "chrA", "pos": 0, "flag": 0}
                      | {t: [int(x[1:]) in SEL_REMOVED[t] for x in q] for t in SEL_TAGS})
    return mol, tr, fl


def _region_counts(removed):
    return pd.DataFrame({"tag": list(removed), "re": "E", "removed": list(removed.values()), "total": 1000})


def test_stage17_selection_without_qualifier_tie_breaks_caps_and_spikeins(tmp_path):
    mol, tr, fl = _sel_tables()
    base, out = tmp_path / "base", tmp_path / "eval"
    # 1417's record counts break the E1/E2 tie at cap 1000 toward E2 and the three-way tie at cap 500 toward E3;
    # 1419's would break them the other way
    counts = {"1417": {E1: 50, E2: 40, E3: 30, CE: 100, I1: 60, I2: 70}, "1419": {E1: 40, E2: 50, E3: 60, CE: 100, I1: 60, I2: 70}}
    counts["1179"] = counts["1419"]  # the tie that 1417 breaks toward E2 goes to E1 on 1179
    for lib in ("1417", "1419", "1179"):
        (base / lib / "truth").mkdir(parents=True)
        (base / lib / "grid").mkdir()
        mol.to_parquet(base / lib / "truth" / "molecules_labeled.parquet")
        tr.to_parquet(base / lib / "truth" / "truth_reads.parquet")
        fl.to_parquet(base / lib / "grid" / "truth_flags.parquet")
        _region_counts(counts[lib]).to_parquet(base / lib / "grid" / "region_counts.parquet")
    spike = mol.assign(spike_label=np.where(mol.label == "templated", "internal", "tail"), module="ERCC")
    for sub in ("truth", "grid"):
        (base / "1417" / "spikein" / sub).mkdir(parents=True)
    spike.to_parquet(base / "1417" / "spikein" / "truth" / "molecules_labeled.parquet")
    tr.to_parquet(base / "1417" / "spikein" / "truth" / "truth_reads.parquet")
    fl.to_parquet(base / "1417" / "spikein" / "grid" / "truth_flags.parquet")
    stdout = _run_stage("17_evaluate.py", base, out)
    assert not [line for line in stdout.splitlines() if ": skipped, missing" in line]  # all three libraries used
    assert "1419: spike-in metrics skipped" in stdout and "1179: spike-in" not in stdout
    assert "mode i: no setting chosen" in stdout
    sel = json.loads((out / "selection.json").read_text())
    assert sel["libraries"] == ["1417", "1419", "1179"]
    # mode e at cap 1000: E1 and E2 have sens 2/3, spec 1 (E3 1/3); E2 removes fewer 1417 records
    e = sel["modes"]["e"]
    assert e["selection"]["tag"] == E2 and e["selection"]["sens"] == pytest.approx(2 / 3) and e["selection"]["spec"] == 1.0
    assert e["selection"]["n_candidates"] == 3 and e["selection"]["n_qualifying"] == 3
    # cap 1000 values: Ta fn, Tb and Tc tp, Tm (member missing from truth_flags) dropped, not counted as kept
    assert e["metrics"]["1417"]["molecule"] == {"tp": 2.0, "fn": 1.0, "tn": 2.0, "fp": 0.0, "n_amb": 1.0, "amb_removed": 0.0,
                                                 "sens": pytest.approx(2 / 3), "spec": 1.0, "prec": 1.0}
    assert sel["units"]["1417"]["1000"] == {"read_units": 6, "molecule_units": 6, "missing_reads": 1, "dropped_molecules": 1}
    # mode i: I1 (1/3, 1/2) and I2 (1, 1/2) both miss spec 0.95; the frontier is I2 alone
    i = sel["modes"]["i"]
    assert i["selection"]["tag"] is None and i["selection"]["n_candidates"] == 2 and i["selection"]["n_qualifying"] == 0
    assert i["selection"]["frontier"] == [[I2, 1.0, 0.5]] and i["metrics"] == {}
    # caps 500 (three-way tie at 1/2: E3 removes fewest records) and 2000 (Td enters: E1 3/4) choose differently
    assert {c: sel["cap_sensitivity"][c]["e"]["tag"] for c in ("500", "1000", "2000")} == {"500": E3, "1000": E2, "2000": E1}
    assert sel["cap_sensitivity"]["2000"]["i"]["tag"] is None
    # bootstrap and hexamer check: the chosen tag, the published settings, the control at the chosen COVLEN, and, for mode i
    # (nothing chosen), the controls at the COVLENs of its published settings
    assert sel["bootstrapped_tags"] == [E2] + [setting_tag(p) for p in published()] + [CE] + \
        [f"c{c}_lnone_m0_i" for c in (300, 275, 180, 375)]
    b = pd.read_parquet(out / "bootstrap.parquet")
    assert set(b.tag) == {E2, E1, I1, I2, CE} and set(b.lib) == {"1417", "1419", "1179"} and len(b) == 3 * 2 * 5 * 3
    h = pd.read_parquet(out / "hexamer_check.parquet")
    assert set(h.tag) == {E2, E1, I1, I2, CE} and len(h) == 3 * 5 * 4
    s = pd.read_parquet(out / "spikein_metrics.parquet")
    assert set(s.lib) == {"1417"}
    # hexamer rates over all labeled molecules: templated 1/7, non-templated 3/4, ambiguous 1/2 -> (1/2 - 1/7) / (3/4 - 1/7)
    a = sel["ambiguous_tail_share"]
    assert set(a) == {"1417", "1419", "1179"} and a["1417"]["estimate"] == pytest.approx(10 / 17)
    assert (a["1417"]["n_templated"], a["1417"]["n_nontemplated"], a["1417"]["n_ambiguous"]) == (7, 4, 2)
    # a20 >= 14 reference: the 7 templated molecules (a20 15), rate 1/7, so the same value here
    assert a["1417"]["rate_ref14"] == pytest.approx(1 / 7) and a["1417"]["estimate_ref14"] == pytest.approx(10 / 17)
    # label purity per library on the headline population, as label_purity gives it
    assert set(sel["label_purity"]) == {"1417", "1419", "1179"}
    assert sel["label_purity"]["1179"] == json.loads(json.dumps(label_purity(mol)))
    assert sel["label_purity"]["1417"]["bands"]["14-20"] == {"n": 7, "hexamer_rate": pytest.approx(1 / 7), "tail_primed_share": 0.0}
    # cells_tuned: the same rule run on 1179 (descriptive), ties broken by 1179's own removed records, scored on every library
    ct = sel["cells_tuned"]
    assert set(ct) == {"1179"} and set(ct["1179"]) == {"e", "i"}
    e = ct["1179"]["e"]
    assert e["introns"] is False and e["selection"]["tag"] == E1 and e["selection"]["n_qualifying"] == 3
    assert set(e["metrics"]) == {"1417", "1419", "1179"}
    assert e["metrics"]["1179"]["molecule"]["sens"] == pytest.approx(2 / 3) and e["metrics"]["1417"]["molecule"]["tp"] == 2.0
    assert ct["1179"]["i"]["selection"]["tag"] is None and ct["1179"]["i"]["metrics"] == {}
