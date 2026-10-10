import gzip
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from pafbench.config import REPO
from pafbench.burden import (near_shares, proximity_table, same_gene, same_site, selection_bias, share_ci, share_table,
                             site_labels)
from pafbench.evaluate import gene_clusters, resample_counts
from pafbench.polyasite import BINS


def _mol(labels, region=None, gene=None, **cols):
    n = len(labels)
    d = {"label": labels, "region": region or ["3UTR"] * n, "gene": gene or ["g1"] * n, "ref": "chr1", "strand": "+",
         "site": np.arange(n) * 10_000}
    d.update(cols)
    return pd.DataFrame(d)


def test_share_table():
    mol = _mol(["templated", "templated", "nontemplated", "ambiguous", "templated", "templated", "templated"],
               region=["3UTR"] * 4 + ["intron"] * 3)
    a = share_table(mol).iloc[0]
    assert (a.stratum, a.templated, a.nontemplated, a.ambiguous, a.total, a.frac) == ("all", 5, 1, 1, 7, 1.0)
    assert a.share_all == 5 / 7 and a.share_resolved == 5 / 6
    t = share_table(mol, "region").set_index("stratum")
    assert t.loc["3UTR", ["templated", "nontemplated", "ambiguous", "total"]].tolist() == [2, 1, 1, 4]
    assert t.loc["3UTR", "share_all"] == 0.5 and t.loc["3UTR", "share_resolved"] == 2 / 3 and t.loc["3UTR", "frac"] == 4 / 7
    assert t.loc["intron", "share_all"] == 1.0 and t.loc["intron", "share_resolved"] == 1.0
    only_amb = share_table(_mol(["ambiguous"])).iloc[0]
    assert only_amb.share_all == 0.0 and math.isnan(only_amb.share_resolved)


def test_share_ci_matches_brute_force_and_degenerate_case():
    rng = np.random.default_rng(0)
    genes = [f"g{i % 9}" for i in range(80)]
    labels = list(rng.choice(["templated", "nontemplated", "ambiguous"], 80, p=[0.7, 0.2, 0.1]))
    regions = list(rng.choice(["3UTR", "intron"], 80))
    mol = _mol(labels, region=regions, gene=genes)
    ci = share_ci(mol, "region", n=300, seed=5).set_index("stratum")
    assert ci.equals(share_ci(mol, "region", n=300, seed=5).set_index("stratum"))
    names = np.unique(gene_clusters(mol))
    w = resample_counts(len(names), 300, 5)[:, np.searchsorted(names, gene_clusters(mol))].astype(float)
    t = mol.label.to_numpy() == "templated"
    for reg in ("3UTR", "intron"):
        m = mol.region.to_numpy() == reg
        share = (w * (t & m)).sum(1) / (w * m).sum(1)
        assert [ci.loc[reg, "lo"], ci.loc[reg, "hi"]] == pytest.approx(np.percentile(share, [2.5, 97.5]).tolist())
    one = share_ci(_mol(["templated", "nontemplated", "templated", "ambiguous"]), n=50, seed=1).iloc[0]
    assert one.stratum == "all" and one.lo == one.hi == 0.5


def test_site_labels_take_the_most_common_label():
    mol = pd.DataFrame({"ref": ["chr3"] * 4, "strand": "+", "site": [10, 10, 10, 20],
                        "label": ["templated", "nontemplated", "templated", "ambiguous"]})
    assert site_labels(mol).values.tolist() == [["chr3", "+", 10, "templated"], ["chr3", "+", 20, "ambiguous"]]


def test_same_site_nearest_within_tol_and_strand():
    a = pd.DataFrame([
        ("chr1", "+", 100, "templated"), ("chr1", "+", 100, "templated"),  # one site
        ("chr1", "+", 200, "nontemplated"),  # b at 195 (5 nt) and 206 (6 nt): nearest is 195
        ("chr1", "-", 300, "templated"),  # b has 300 only on +: unmatched
        ("chr2", "+", 50, "ambiguous"),  # b at 48 and 52: tie, the lower coordinate wins
        ("chr3", "+", 10, "templated"), ("chr3", "+", 10, "templated"), ("chr3", "+", 10, "nontemplated"),
        ("chr1", "+", 1000, "templated"),  # b at 1006: beyond tol
    ], columns=["ref", "strand", "site", "label"])
    b = pd.DataFrame([
        ("chr1", "+", 104, "templated"), ("chr1", "+", 195, "nontemplated"), ("chr1", "+", 206, "templated"),
        ("chr1", "+", 300, "templated"), ("chr2", "+", 48, "ambiguous"), ("chr2", "+", 52, "templated"),
        ("chr1", "+", 1006, "templated"),
    ], columns=["ref", "strand", "site", "label"])
    s = same_site(a, b)
    assert list(s.columns) == ["label_a", "label_b", "n"]
    got = {(r.label_a, r.label_b): r.n for r in s.itertuples()}
    assert got == {("templated", "templated"): 1, ("nontemplated", "nontemplated"): 1, ("templated", "none"): 3,
                   ("ambiguous", "ambiguous"): 1}
    wide = {(r.label_a, r.label_b): r.n for r in same_site(a, b, tol=6).itertuples()}
    assert wide[("templated", "templated")] == 2 and wide[("templated", "none")] == 2
    assert same_site(a, b.iloc[:0]).n.sum() == 6


def _genes(shares, n, label_prefix="g"):
    rows = []
    for i, (s, k) in enumerate(zip(shares, n)):
        t = round(s * k)
        rows += [(f"{label_prefix}{i}", "templated")] * t + [(f"{label_prefix}{i}", "nontemplated")] * (k - t)
    return pd.DataFrame(rows, columns=["gene", "label"])


def test_same_gene_spearman_and_min_n():
    a = _genes([0.1, 0.2, 0.3, 0.4, 0.9], [20, 20, 20, 20, 19])
    b = _genes([0.15, 0.3, 0.5, 0.9, 0.0], [20, 40, 20, 20, 40])
    a = pd.concat([a, pd.DataFrame({"gene": [""] * 30, "label": "templated"})], ignore_index=True)
    b = pd.concat([b, pd.DataFrame({"gene": [""] * 30, "label": "nontemplated"})], ignore_index=True)
    r = same_gene(a, b)
    # g4 has 19 molecules in a, intergenic molecules never count
    assert r == {"n_genes": 4, "spearman_rho": 1.0, "p_value": pytest.approx(spearmanr([1, 2, 3, 4], [1, 2, 3, 4]).pvalue)}
    r = same_gene(a, b, min_n=19)
    assert r["n_genes"] == 5 and r["spearman_rho"] == pytest.approx(spearmanr([0.1, 0.2, 0.3, 0.4, 0.9],
                                                                               [0.15, 0.3, 0.5, 0.9, 0.0]).statistic)


def test_selection_bias():
    mol = pd.DataFrame({"re": ["E", "E", "N", "I", None]})
    rc = pd.DataFrame({"tag": ["t1"] * 4 + ["t2"] * 4, "re": ["E", "N", "I", ""] * 2, "removed": [5, 1, 1, 0] * 2,
                       "total": [60, 20, 15, 5] * 2, "total_nondup": [30, 20, 10, 0] * 2})
    s = selection_bias(mol, rc).set_index("re")
    assert list(s.columns) == ["labeled_share", "records_share", "records_nondup_share"]
    assert s.labeled_share.to_dict() == pytest.approx({"": 0.2, "E": 0.4, "I": 0.2, "N": 0.2})
    assert s.records_share.to_dict() == pytest.approx({"": 0.05, "E": 0.6, "I": 0.15, "N": 0.2})
    assert s.records_nondup_share.to_dict() == pytest.approx({"": 0.0, "E": 0.5, "I": 1 / 6, "N": 1 / 3})
    s = selection_bias(pd.DataFrame({"re": ["E"]}), rc).set_index("re")
    assert s.loc["N", "labeled_share"] == 0.0 and s.loc["E", "labeled_share"] == 1.0


def _lib_mol(shift):
    """60 molecules, genes g0, g1, g2 of 20 each with 5, 10, 15 templated; the rest alternate nontemplated / ambiguous."""
    i = np.arange(60)
    k = i % 20
    lab = np.where(k < 5 * (i // 20 + 1), "templated", np.where(k % 2 == 0, "nontemplated", "ambiguous")).astype(object)
    return pd.DataFrame({"mol": i, "ref": "chr1", "strand": "+", "site": 100 + 50 * i + shift, "label": lab,
                         "a_band": np.where(lab == "templated", "14-20", np.where(lab == "nontemplated", "0-4", "5-7")),
                         "region": np.where(i < 40, "3UTR", "intron"), "gene": [f"g{x}" for x in i // 20],
                         "has_ts": i % 5 == 0, "re": np.where(i < 40, "E", "I")})


def _atlas(tmp_path):
    """PolyASite rows on chr1 plus at 100 and 1105; a minus site at 1150 (on a plus molecule's site) and chr2 do not count."""
    p = tmp_path / "atlas.bed.gz"
    rows = [("chr1", 99, 100, "chr1:100:+", 0.5, "+"), ("chr1", 1104, 1105, "chr1:1105:+", 0.1, "+"),
            ("chr1", 1148, 1149, "chr1:1149:-", 0.1, "-"), ("chr2", 999, 1000, "chr2:1000:+", 0.1, "+")]
    with gzip.open(p, "wt") as fh:
        fh.writelines("\t".join(map(str, r + (11.1, 1, 47, "TE", "NaN"))) + "\n" for r in rows)
    return p


def test_stage18_end_to_end(tmp_path):
    base, out = tmp_path / "base", tmp_path / "burden"
    libs = {"1417": _lib_mol(0), "1419": _lib_mol(0), "1179": _lib_mol(3)}
    for lib, mol in libs.items():
        (base / lib / "truth").mkdir(parents=True)
        mol.to_parquet(base / lib / "truth" / "molecules_labeled.parquet")
    (base / "1417" / "grid").mkdir()
    pd.DataFrame({"tag": "t1", "re": ["E", "I"], "removed": [1, 2], "total": [30, 70],
                       "total_nondup": [15, 70]}).to_parquet(
        base / "1417" / "grid" / "region_counts.parquet")
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "18_burden.py"), "--base", str(base), "--out", str(out),
                        "--atlas", str(_atlas(tmp_path))], capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 0, r.stderr
    assert "1419: selection bias skipped" in r.stdout
    t = pd.read_parquet(out / "templated_share.parquet")
    assert set(t.group) == {"1417", "1419", "1179", "nuclei", "cells"}
    a = t[(t.group == "1417") & (t.stratum_kind == "all")].iloc[0]
    # non-templated at even k: g0 k 6..18 (7), g1 k 10..18 (5), g2 k 16, 18 (2); the other 16 ambiguous
    assert (a.templated, a.nontemplated, a.ambiguous, a.total, a.share_all) == (30, 14, 16, 60, 0.5)
    assert t[(t.group == "nuclei") & (t.stratum_kind == "all")].total.iloc[0] == 120
    reg = share_table(libs["1417"], "region").merge(share_ci(libs["1417"], "region"), on="stratum")
    got = t[(t.group == "1417") & (t.stratum_kind == "region")].drop(columns=["group", "stratum_kind"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, reg, check_dtype=False)
    assert set(t.stratum_kind) == {"all", "region", "a_band", "has_ts"}
    s = pd.read_parquet(out / "same_site.parquet")
    same = s[s.pair == "1417_vs_1419"]
    assert (same.label_a == same.label_b).all() and same.n.sum() == 60
    nc = s[s.pair == "nuclei_vs_cells"]
    assert (nc.label_a == nc.label_b).all() and nc.n.sum() == 60  # the 1179 sites lie 3 nt away
    g = pd.read_parquet(out / "same_gene.parquet").set_index("pair")
    assert g.loc["1417_vs_1419", "n_genes"] == 3 and g.loc["1417_vs_1419", "spearman_rho"] == 1.0
    assert g.loc["nuclei_vs_cells", "n_genes"] == 3
    b = pd.read_parquet(out / "selection_bias.parquet")
    assert set(b.lib) == {"1417"} and b.set_index("re").records_share.to_dict() == {"E": 0.3, "I": 0.7}
    assert b.set_index("re").records_nondup_share.to_dict() == pytest.approx({"E": 15 / 85, "I": 70 / 85})
    summ = json.loads((out / "summary.json").read_text())
    assert summ["groups"]["nuclei"] == ["1417", "1419"] and summ["share"]["1417"]["all"]["share_all"] == 0.5
    assert summ["share"]["cells"]["region"]["3UTR"]["total"] == 40
    assert summ["same_site"]["1417_vs_1419"] == {"sites_a": 60, "matched": 60, "same_label": 60}


def _stage18(base, out, tmp_path, *extra):
    env = dict(os.environ, PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(REPO / "scripts" / "18_burden.py"), "--base", str(base), "--out", str(out),
                           "--atlas", str(_atlas(tmp_path)), *extra], capture_output=True, text=True, env=env, cwd=REPO)


def _pop_libs(base, in_cell=True):
    libs = {"1417": _lib_mol(0), "1419": _lib_mol(0), "1179": _lib_mol(3)}
    for lib, mol in libs.items():
        libs[lib] = mol.assign(in_cell=(mol.mol % 3 != 0) if lib != "1179" else mol.mol < 40) if in_cell else mol
        (base / lib / "truth").mkdir(parents=True)
        libs[lib].to_parquet(base / lib / "truth" / "molecules_labeled.parquet")
    return libs


def test_stage18_population_selects_barcodes_and_reports_all_three(tmp_path):
    base = tmp_path / "base"
    libs = _pop_libs(base)
    shares = {}
    for pop in ("all", "cells", "noncells"):
        r = _stage18(base, tmp_path / pop, tmp_path, "--population", pop)
        assert r.returncode == 0, r.stderr
        t = pd.read_parquet(tmp_path / pop / "templated_share.parquet")
        a = t[(t.group == "1417") & (t.stratum_kind == "all")].iloc[0]
        sub = libs["1417"][{"all": slice(None), "cells": libs["1417"].in_cell, "noncells": ~libs["1417"].in_cell}[pop]]
        assert (a.templated, a.nontemplated, a.ambiguous, a.total) == tuple(
            int(x) for x in (sub.label.eq("templated").sum(), sub.label.eq("nontemplated").sum(), sub.label.eq("ambiguous").sum(),
                             len(sub)))
        shares[pop] = pd.read_parquet(tmp_path / pop / "templated_share_population.parquet")
    # the per-population table does not depend on --population
    pd.testing.assert_frame_equal(shares["all"], shares["cells"])
    pd.testing.assert_frame_equal(shares["all"], shares["noncells"])
    t = shares["all"]
    assert list(t.columns) == ["lib", "population", "stratum_kind", "stratum", "templated", "nontemplated", "ambiguous", "total",
                               "share_all"]
    assert set(zip(t.lib, t.population)) == {(lib, p) for lib in libs for p in ("all", "cells", "noncells")}
    assert set(t.stratum_kind) == {"all", "region"}
    for lib, mol in libs.items():
        for pop, sub in (("all", mol), ("cells", mol[mol.in_cell]), ("noncells", mol[~mol.in_cell])):
            row = t[(t.lib == lib) & (t.population == pop) & (t.stratum_kind == "all")].iloc[0]
            n = sub.label.value_counts()
            assert (row.templated, row.nontemplated, row.ambiguous, row.total) == (
                n.get("templated", 0), n.get("nontemplated", 0), n.get("ambiguous", 0), len(sub))
            assert row.share_all == n.get("templated", 0) / len(sub)
    r = t[(t.lib == "1179") & (t.population == "cells") & (t.stratum_kind == "region") & (t.stratum == "intron")]
    assert len(r) == 0  # 1179 cells are mol < 40: all 3UTR
    r = t[(t.lib == "1179") & (t.population == "noncells") & (t.stratum_kind == "region")]
    assert r.stratum.tolist() == ["intron"] and r.total.tolist() == [20]


def test_stage18_population_needs_in_cell(tmp_path):
    base = tmp_path / "base"
    _pop_libs(base, in_cell=False)
    r = _stage18(base, tmp_path / "out", tmp_path, "--population", "cells")
    assert r.returncode != 0 and "in_cell" in r.stderr
    r = _stage18(base, tmp_path / "out_all", tmp_path)
    assert r.returncode == 0, r.stderr
    t = pd.read_parquet(tmp_path / "out_all" / "templated_share_population.parquet")
    assert set(t.population) == {"all"}


def test_share_ci_of_stratum_fraction_matches_brute_force():
    rng = np.random.default_rng(2)
    genes = [f"g{i % 9}" for i in range(80)]
    bands = list(rng.choice(["0-4", "8-9", "14-20"], 80))
    labels = ["nontemplated" if b == "0-4" else "templated" for b in bands]
    mol = _mol(labels, gene=genes, a_band=bands)
    ci = share_ci(mol, "a_band", n=300, seed=5, of="frac").set_index("stratum")
    names = np.unique(gene_clusters(mol))
    w = resample_counts(len(names), 300, 5)[:, np.searchsorted(names, gene_clusters(mol))].astype(float)
    for band in ("0-4", "8-9", "14-20"):
        frac = (w * (mol.a_band.to_numpy() == band)).sum(1) / w.sum(1)
        assert [ci.loc[band, "lo"], ci.loc[band, "hi"]] == pytest.approx(np.percentile(frac, [2.5, 97.5]).tolist())
        assert ci.loc[band, "lo"] < ci.loc[band, "hi"]
    # share_all is still the default, and it is fixed within a band by the labels
    assert share_ci(mol, "a_band", n=300, seed=5).set_index("stratum").loc["14-20"].tolist() == [1.0, 1.0]


def test_stage18_a_band_interval_is_of_the_band_fraction(tmp_path):
    base, out = tmp_path / "base", tmp_path / "burden"
    mol = _lib_mol(0)
    (base / "1417" / "truth").mkdir(parents=True)
    mol.to_parquet(base / "1417" / "truth" / "molecules_labeled.parquet")
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "18_burden.py"), "--base", str(base), "--out", str(out),
                        "--atlas", str(_atlas(tmp_path))], capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 0, r.stderr
    t = pd.read_parquet(out / "templated_share.parquet")
    band = t[(t.group == "1417") & (t.stratum_kind == "a_band")].set_index("stratum")
    want = share_ci(mol, "a_band", of="frac").set_index("stratum")
    assert band[["lo", "hi"]].equals(want[["lo", "hi"]].loc[band.index])
    # 14-20 holds the 30 templated molecules: frac 0.5, with genes of 5, 10 and 15 templated the interval is not a point
    assert band.loc["14-20", "frac"] == 0.5 and band.loc["14-20", "lo"] < 0.5 < band.loc["14-20", "hi"]
    region = t[(t.group == "1417") & (t.stratum_kind == "region")].set_index("stratum")
    assert region[["lo", "hi"]].equals(share_ci(mol, "region").set_index("stratum")[["lo", "hi"]].loc[region.index])


def _prox(t):
    """{(label, a_band): [n per bin in BINS order]} of a proximity table."""
    return {k: g.set_index("bin").n.reindex(list(BINS)).tolist() for k, g in t.groupby(["label", "a_band"], sort=False)}


def test_proximity_table_and_near_shares():
    mol = pd.DataFrame([("templated", "14-20"), ("templated", "14-20"), ("templated", "8-9"), ("templated", "10-13"),
                        ("templated", "10-13"), ("nontemplated", "0-4"), ("nontemplated", "0-4"), ("ambiguous", "5-7")],
                       columns=["label", "a_band"])
    dist = np.array([0, -11, 10, np.nan, 201, -10, 51, 200])
    null = np.array([5, -300, np.nan, 12, -10, 100, 2000, -8])
    t = proximity_table(mol, dist, null)
    assert list(t.columns) == ["label", "a_band", "bin", "n"] and len(t) == 35
    # labels in burden order with every bin (zeros kept), the null over all molecules, then the templated A bands
    assert list(dict.fromkeys(zip(t.label, t.a_band))) == [
        ("templated", "all"), ("nontemplated", "all"), ("ambiguous", "all"), ("null_shift1kb", "all"), ("templated", "8-9"),
        ("templated", "10-13"), ("templated", "14-20")]
    # NaN (no atlas site on the ref and strand) has its own bin, apart from > 200; null: 5, -10, -8 within 10, 12, 100,
    # -300 and 2000 beyond 200, NaN
    assert _prox(t) == {("templated", "all"): [2, 1, 0, 1, 1], ("nontemplated", "all"): [1, 0, 1, 0, 0],
                        ("ambiguous", "all"): [0, 0, 1, 0, 0], ("null_shift1kb", "all"): [3, 1, 1, 2, 1],
                        ("templated", "8-9"): [1, 0, 0, 0, 0], ("templated", "10-13"): [0, 0, 0, 1, 1],
                        ("templated", "14-20"): [1, 1, 0, 0, 0]}
    assert near_shares(t) == {"label": {"templated": 0.4, "nontemplated": 0.5, "ambiguous": 0.0, "null_shift1kb": 3 / 8},
                              "templated_a_band": {"8-9": 1.0, "10-13": 0.0, "14-20": 0.5}}
    # a label without molecules has no share; the null covers the molecules given (5, -10 of the first five)
    s = near_shares(proximity_table(mol[mol.label == "templated"], dist[:5], null[:5]))
    assert s["label"] == {"templated": 0.4, "nontemplated": None, "ambiguous": None, "null_shift1kb": 0.4}


def test_stage18_polyasite(tmp_path):
    base, out = tmp_path / "base", tmp_path / "burden"
    # 1419: sites 100 nt further, i 55 on chr2 minus (chr2 has a plus atlas site only), i 57-59 on chrM (absent from the
    # atlas), i 56 on chr1 minus at 150
    m19 = _lib_mol(100)
    m19.loc[55, ["ref", "strand"]] = ["chr2", "-"]
    m19.loc[56, ["strand", "site"]] = ["-", 150]
    m19.loc[[57, 58, 59], "ref"] = "chrM"
    for lib, mol in {"1417": _lib_mol(0), "1419": m19, "1179": _lib_mol(11)}.items():
        (base / lib / "truth").mkdir(parents=True)
        mol.to_parquet(base / lib / "truth" / "molecules_labeled.parquet")
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "18_burden.py"), "--base", str(base), "--out", str(out),
                        "--atlas", str(_atlas(tmp_path))], capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 0, r.stderr
    t = pd.read_parquet(out / "polyasite.parquet")
    assert list(t.columns) == ["group", "label", "a_band", "bin", "n"]
    assert list(dict.fromkeys(t.group)) == ["1417", "1419", "1179", "nuclei", "cells"]
    prox = {g: _prox(t[t.group == g].drop(columns="group")) for g in dict.fromkeys(t.group)}
    # Templated: i 0-4, 20-29, 40-54; non-templated i 6-18, 30-38 even, 56, 58; ambiguous the other i.
    # 1417, sites 100 + 50 i on chr1 plus, atlas at 100 and 1105. 0-10: i 0 (0), i 20 (+5); 11-50: i 1 (-50), i 21 (-45);
    # 51-200: i 2-4 (-100, -150, -200), i 17-19 (+155, +105, +55), i 22-24 (-95, -145, -195); the rest > 200.
    # Null, sites 1,000 nt upstream (-900 + 50 i): 0-10: i 20 (0), i 40 (+5); 11-50: i 19 (+50), i 21 (-50), i 41 (-45);
    # 51-200: i 16-18, 22-24, 37-39, 42-44; the other 43 > 200.
    want = {("templated", "all"): [2, 2, 6, 20, 0], ("nontemplated", "all"): [0, 0, 1, 13, 0],
            ("ambiguous", "all"): [0, 0, 2, 14, 0], ("null_shift1kb", "all"): [2, 3, 12, 43, 0],
            ("templated", "8-9"): [0] * 5, ("templated", "10-13"): [0] * 5, ("templated", "14-20"): [2, 2, 6, 20, 0]}
    assert prox["1417"] == want
    # 1419, sites 200 + 50 i. 0-10: i 18 (+5); 11-50: i 19 (-45); 51-200: i 0-2 (-100, -150, -200), i 15-17 (+155, +105,
    # +55), i 20-22 (-95, -145, -195); i 56 on minus: the minus atlas site 1150 lies 1,000 nt upstream (-1000, > 200);
    # i 55 and 57-59: no atlas site on their ref and strand.
    # Null (-800 + 50 i; i 56 on minus moves up to 1150): 0-10: i 18 (0), i 38 (+5), i 56 (0); 11-50: i 17, 19, 39;
    # 51-200: i 14-16, 20-22, 35-37, 40-42; none: i 55, 57-59; the other 38 > 200.
    assert prox["1419"] == {("templated", "all"): [0, 0, 6, 24, 0], ("nontemplated", "all"): [1, 0, 1, 11, 1],
                            ("ambiguous", "all"): [0, 1, 2, 10, 3], ("null_shift1kb", "all"): [3, 3, 12, 38, 4],
                            ("templated", "8-9"): [0] * 5, ("templated", "10-13"): [0] * 5,
                            ("templated", "14-20"): [0, 0, 6, 24, 0]}
    # 1179, sites 111 + 50 i. 0-10: i 20 (-6); 11-50: i 0 (-11), i 19 (+44); 51-200: i 1-3 (-61, -111, -161), i 16-18
    # (+194, +144, +94), i 21-23 (-56, -106, -156). Null (-889 + 50 i): 0-10: i 40 (-6); 11-50: i 19 (+39), i 20 (-11),
    # i 39 (+44); 51-200: i 16-18, 21-23, 36-38, 41-43; the other 44 > 200.
    assert prox["1179"] == {("templated", "all"): [1, 1, 6, 22, 0], ("nontemplated", "all"): [0, 0, 2, 12, 0],
                            ("ambiguous", "all"): [0, 1, 1, 14, 0], ("null_shift1kb", "all"): [1, 3, 12, 44, 0],
                            ("templated", "8-9"): [0] * 5, ("templated", "10-13"): [0] * 5,
                            ("templated", "14-20"): [1, 1, 6, 22, 0]}
    assert prox["nuclei"] == {k: [a + b for a, b in zip(v, prox["1419"][k])] for k, v in prox["1417"].items()}
    assert prox["cells"] == prox["1179"]
    p = json.loads((out / "summary.json").read_text())["polyasite"]
    assert p["atlas"] == str(_atlas(tmp_path)) and p["atlas_sites"] == 4
    # within 10 nt per group, from the counts above (bands 8-9 and 10-13 hold no molecule)
    within = {"1417": ([2 / 30, 0.0, 0.0, 2 / 60], 2 / 30), "1419": ([0.0, 1 / 14, 0.0, 3 / 60], 0.0),
              "1179": ([1 / 30, 0.0, 0.0, 1 / 60], 1 / 30), "nuclei": ([2 / 60, 1 / 28, 0.0, 5 / 120], 2 / 60),
              "cells": ([1 / 30, 0.0, 0.0, 1 / 60], 1 / 30)}
    assert list(p["within_10nt"]) == list(within)
    for g, (labels, band) in within.items():
        assert p["within_10nt"][g] == {
            "label": dict(zip(("templated", "nontemplated", "ambiguous", "null_shift1kb"), labels)),
            "templated_a_band": {"8-9": None, "10-13": None, "14-20": band}}, g
