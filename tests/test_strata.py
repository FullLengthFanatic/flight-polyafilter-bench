"""Stage 27 (pafbench.strata): truth-set strata by cell class and expression level, the join of truth molecules to the
molecules counted before filtering, accuracy as the matrix sees it and the acceptance, on synthetic truth sets and on the
synthetic 10x library of tests/synth.build_10x."""
import importlib.util
import json
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pafbench import recount
from pafbench.config import A_BANDS, CHOSEN_SETTINGS, MIN_STRATUM, REPO
from pafbench.evaluate import COUNT_COLUMNS, METRIC_COLUMNS, bootstrap, evaluate, subset_units, units
from pafbench.flags import setting_tag
from pafbench.grid import published
from pafbench.regions import REGION_NAMES
from pafbench.strata import (JOIN_COLUMNS, KINDS, LEVELS, NO_UMI, OUTCOME_COLUMNS, UNASSIGNED, acceptance, cell_classes,
                             expr_bins, few_molecules, join_counted, join_summary, matrix_outcome, strata_bootstrap, with_strata)
from tests.synth import CELLS_10X, SETTINGS_10X
from tests.test_recount import stage13, stage24  # noqa: F401  (fixtures: stage 24 on the synthetic 10x library)

TAGS = ["c300_l10_m1_i", "c375_l3_m2_i", "c300_lnone_m0_i"]
BARCODES = ["AAACGG-1", "AAAGTT-1", "AACATG-1", "AACCGA-1", "AACGTC-1"]
# AACCGA-1 has a row without a class, AACGTC-1 none at all
CELLS = pd.DataFrame({"cb": BARCODES[:4], "class": ["rod", "rod", "cone", None]})
GENE_UMIS = pd.Series({"gD": 0} | {f"g{i}": 7 * i * i + 3 for i in range(20)})


def _truth(seed, n=400):
    """Labeled molecules of called cells with a pa read and 0 to 3 member reads (some beyond the 1,000 nt cap), and random
    removal flags at TAGS. Molecule ids are not row numbers. Genes: '' (none), gG (no row in GENE_UMIS) and the genes of
    GENE_UMIS, gD among them with no counted UMI."""
    rng = np.random.default_rng(seed)
    a20 = rng.integers(0, 21, n)
    band = np.array([next(name for name, lo, hi in A_BANDS if lo <= a <= hi) for a in a20], object)
    label = np.where(a20 >= 8, "templated", np.where(a20 <= 4, "nontemplated", "ambiguous"))
    mol = pd.DataFrame({"mol": np.arange(n) * 3 + 7, "cb": rng.choice(BARCODES, n), "label": label, "a20": a20, "a_band": band,
                        "region": rng.choice(REGION_NAMES, n), "gene": rng.choice(["", "gG", *GENE_UMIS.index], n),
                        "has_ts": False, "hexamer": rng.random(n) < 0.3, "ref": "chrA", "strand": "+",
                        "site": rng.integers(0, 10**6, n), "in_cell": True})
    reads = [(f"p{m}", m, "pa", np.nan, "", False) for m in mol.mol]
    for m in mol.mol:
        reads += [(f"r{m}_{k}", m, "member", float(rng.integers(0, 1500)), "exon", False) for k in range(rng.integers(0, 4))]
    tr = pd.DataFrame(reads, columns=["qname", "mol", "role", "dist5_exon", "dist_mode", "spliced"])
    fl = pd.DataFrame({"qname": tr.qname, "ref": "chrA", "pos": 0, "flag": 0} | {t: rng.random(len(tr)) < 0.4 for t in TAGS})
    return mol, tr, fl


def _stratified(seed):
    mol, tr, fl = _truth(seed)
    return with_strata(units(mol, tr, fl, TAGS, cap=1000), CELLS, GENE_UMIS)


def test_a_molecule_whose_barcode_has_no_class_lands_in_unassigned():
    assert cell_classes(np.array(BARCODES + ["AAACGG-1"], object), CELLS).tolist() == [
        "rod", "rod", "cone", UNASSIGNED, UNASSIGNED, "rod"]
    assert UNASSIGNED == "unassigned"
    u, _ = _stratified(0)
    m = evaluate(u, "L", 1000, levels=LEVELS, kinds=("cell_class",))
    assert set(m.stratum) == {"rod", "cone", UNASSIGNED}
    cb = u.mols.cb.to_numpy()
    assert (u.mols.cell_class.to_numpy() == UNASSIGNED).sum() == np.isin(cb, ["AACCGA-1", "AACGTC-1"]).sum() > 0


def test_expr_bins_are_quartiles_of_counted_umis_over_the_molecules():
    gene = np.array(["g1", "g2", "g2", "g3", "", "g4", "g5", "g6", "g7", "g8", "g9"], object)
    umis = pd.Series({"g1": 10, "g2": 20, "g3": 30, "g4": 0, "g5": 40, "g6": 50, "g7": 60, "g8": 70})
    bins, bounds = expr_bins(gene, umis)
    # 10 20 20 30 40 50 60 70 over the 8 molecules with a counted UMI: numpy.quantile 20, 35 and 52.5; a molecule at a
    # boundary falls in the lower bin, as in pandas.qcut. No gene (''), no counted UMI (g4) and no row (g9): their own stratum
    assert bounds.tolist() == [20.0, 35.0, 52.5]
    assert bins.tolist() == ["q1", "q1", "q1", "q2", NO_UMI, NO_UMI, "q3", "q3", "q4", "q4", NO_UMI]
    has = ~np.isin(bins, [NO_UMI])
    ref = pd.qcut(pd.Series(gene[has]).map(umis), 4, labels=["q1", "q2", "q3", "q4"]).astype(str).tolist()
    assert bins[has].tolist() == ref
    # reproducible: the same call again, and the molecules in another order
    again, b2 = expr_bins(gene, umis)
    assert again.tolist() == bins.tolist() and b2.tolist() == bounds.tolist()
    order = np.random.default_rng(3).permutation(len(gene))
    shuffled, b3 = expr_bins(gene[order], umis)
    assert shuffled.tolist() == bins[order].tolist() and b3.tolist() == bounds.tolist()
    nothing, b4 = expr_bins(np.array(["", "g4"], object), umis)
    assert nothing.tolist() == [NO_UMI, NO_UMI] and np.isnan(b4).all() and len(b4) == 3


def test_with_strata_bins_the_molecule_units():
    u, bounds = _stratified(1)
    x = u.mols.gene.map(GENE_UMIS).fillna(0).to_numpy()
    assert bounds.tolist() == np.quantile(x[x > 0], [0.25, 0.5, 0.75]).tolist()
    assert u.mols.expr_bin.tolist() == expr_bins(u.mols.gene.to_numpy(object), GENE_UMIS)[0].tolist()
    assert u.mols.cell_class.tolist() == cell_classes(u.mols.cb.to_numpy(object), CELLS).tolist()
    assert set(u.mols.expr_bin) == {"q1", "q2", "q3", "q4", NO_UMI} and KINDS == ("cell_class", "expr_bin")
    assert (u.mols.expr_bin == NO_UMI).sum() == u.mols.gene.isin(["", "gD", "gG"]).sum()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_strata_add_up_to_the_whole_at_both_levels(seed):
    u, _ = _stratified(seed)
    m = evaluate(u, "L", 1000, levels=LEVELS, kinds=("all",) + KINDS)
    assert LEVELS == ("read", "molecule")
    for level in LEVELS:
        whole = m[(m.level == level) & (m.stratum_kind == "all")].set_index("tag")[COUNT_COLUMNS]
        for kind in KINDS:
            parts = m[(m.level == level) & (m.stratum_kind == kind)]
            assert len(parts) == len(TAGS) * parts.stratum.nunique() and parts.stratum.nunique() >= 3
            pd.testing.assert_frame_equal(parts.groupby("tag")[COUNT_COLUMNS].sum().loc[TAGS], whole.loc[TAGS])
    # the units: a molecule level row per label, a read level row per member read
    lab = u.mols.label.value_counts()
    r = m[(m.level == "molecule") & (m.stratum_kind == "all")].iloc[0]
    assert (r.tp + r.fn, r.tn + r.fp, r.n_amb) == (lab["templated"], lab["nontemplated"], lab["ambiguous"])


def test_subset_units_of_every_stratum_put_back_together_give_the_counts_of_the_whole():
    u, _ = _stratified(4)
    whole = evaluate(u, "L", 1000, levels=LEVELS, kinds=KINDS)
    for kind in KINDS:
        col = u.mols[kind].to_numpy()
        subsets = [(value, subset_units(u, col == value)) for value in sorted(set(col))]
        assert sum(len(s.mols) for _, s in subsets) == len(u.mols) and sum(len(s.reads) for _, s in subsets) == len(u.reads)
        assert sorted(np.concatenate([s.reads.qname for _, s in subsets])) == sorted(u.reads.qname)
        sums = 0
        for value, s in subsets:
            e = evaluate(s, "L", 1000, levels=LEVELS, kinds=("all",)).set_index(["level", "tag"])[COUNT_COLUMNS]
            ref = whole[(whole.stratum_kind == kind) & (whole.stratum == value)].set_index(["level", "tag"])[COUNT_COLUMNS]
            pd.testing.assert_frame_equal(e, ref)
            sums = sums + e
        total = evaluate(u, "L", 1000, levels=LEVELS, kinds=("all",)).set_index(["level", "tag"])[COUNT_COLUMNS]
        pd.testing.assert_frame_equal(sums, total)


def test_strata_bootstrap_is_the_bootstrap_of_each_stratum():
    u, _ = _stratified(5)
    b = strata_bootstrap(u, TAGS[:2], "L", n=50, seed=2)
    assert list(b.columns) == ["lib", "tag", "level", "stratum_kind", "stratum", "metric", "estimate", "lo", "hi", "n_boot",
                               "n_clusters"]
    assert set(b.lib) == {"L"} and set(b.level) == set(LEVELS) and set(b.stratum_kind) == set(KINDS)
    for kind in KINDS:
        for value in sorted(set(u.mols[kind])):
            for level in LEVELS:
                got = b[(b.stratum_kind == kind) & (b.stratum == value) & (b.level == level)]
                ref = bootstrap(subset_units(u, u.mols[kind].to_numpy() == value), TAGS[:2], level, n=50, seed=2)
                pd.testing.assert_frame_equal(got[ref.columns].reset_index(drop=True), ref)
    assert len(b) == 2 * 3 * 2 * (u.mols.cell_class.nunique() + u.mols.expr_bin.nunique())
    # a stratum without a templated or non-templated unit has nothing to resample and gets no rows
    amb = u.mols.assign(cell_class=np.where(u.mols.label == "ambiguous", "amb_only", u.mols.cell_class))
    b2 = strata_bootstrap(replace(u, mols=amb), TAGS[:1], "L", n=20, seed=0)
    assert "amb_only" not in set(b2.stratum) and "rod" in set(b2.stratum)


def test_few_molecules_marks_the_rates_with_a_denominator_under_min_stratum():
    u, _ = _stratified(6)
    few = few_molecules(u)
    lab = u.mols.label.to_numpy()
    assert [(r["stratum_kind"], r["stratum"]) for r in few] == [(k, v) for k in KINDS for v in sorted(set(u.mols[k]))]
    for r in few:
        m = u.mols[r["stratum_kind"]].to_numpy() == r["stratum"]
        assert (r["templated"], r["nontemplated"]) == ((lab[m] == "templated").sum(), (lab[m] == "nontemplated").sum())
        assert (r["sens_few"], r["spec_few"]) == (r["templated"] < MIN_STRATUM, r["nontemplated"] < MIN_STRATUM)
    assert any(r["sens_few"] for r in few) and not all(r["sens_few"] for r in few)
    assert few_molecules(u, min_n=1) == [r | {"sens_few": r["templated"] < 1, "spec_few": r["nontemplated"] < 1} for r in few]


def _m2_metrics(u, lib="L"):
    """What stage 17 writes for these units: every stratum kind and level at cap 1,000 and cut-offs 8/4."""
    return evaluate(u, lib, 1000, "8/4")


def test_acceptance_compares_the_sums_over_strata_with_the_whole_of_milestone_2():
    u, _ = _stratified(7)
    strata = evaluate(u, "L", 1000, levels=LEVELS, kinds=KINDS)
    metrics = pd.concat([_m2_metrics(u), _m2_metrics(u, "M"), evaluate(u, "L", 500)], ignore_index=True)
    a = acceptance(strata, metrics, "L", TAGS)
    assert a["passed"] is True and a["compared"] == len(TAGS) * len(LEVELS) * len(KINDS) and a["differing"] == []
    lab = u.mols.label.value_counts()
    assert a["units"]["molecule"] == {"templated": int(lab["templated"]), "nontemplated": int(lab["nontemplated"])}
    assert a["units"]["read"]["templated"] == int((u.reads.label == "templated").sum())
    # one templated unit more in the whole of milestone 2: tp + fn and the sensitivity differ, for both kinds
    bad = metrics.copy()
    hit = (bad.lib == "L") & (bad.cap == 1000) & (bad.level == "molecule") & (bad.stratum_kind == "all") & (bad.tag == TAGS[1])
    bad.loc[hit, "fn"] += 1
    a = acceptance(strata, bad, "L", TAGS)
    assert a["passed"] is False and a["compared"] == len(TAGS) * len(LEVELS) * len(KINDS)
    assert [(d["tag"], d["level"], d["stratum_kind"]) for d in a["differing"]] == [
        (TAGS[1], "molecule", "cell_class"), (TAGS[1], "molecule", "expr_bin")]
    assert a["differing"][0]["templated"] == a["differing"][0]["templated_m2"] - 1
    # a sensitivity that differs at equal unit counts fails too
    bad = metrics.copy()
    bad.loc[hit, "sens"] = 0.123
    assert acceptance(strata, bad, "L", TAGS)["passed"] is False
    # a setting or level missing from milestone 2 fails
    a = acceptance(strata, metrics[~((metrics.tag == TAGS[2]) & (metrics.level == "read"))], "L", TAGS)
    assert a["passed"] is False and {(d["tag"], d["level"]) for d in a["differing"]} == {(TAGS[2], "read")}
    # so does a setting, level or stratum kind missing from the strata
    a = acceptance(strata[~((strata.tag == TAGS[0]) & (strata.stratum_kind == "expr_bin"))], metrics, "L", TAGS)
    assert a["passed"] is False and a["compared"] == len(TAGS) * len(LEVELS) * len(KINDS)
    assert [(d["tag"], d["stratum_kind"], d["templated"]) for d in a["differing"]] == [
        (TAGS[0], "expr_bin", None), (TAGS[0], "expr_bin", None)]
    assert acceptance(strata, metrics, "L", TAGS + ["c180_l11_m2_i"])["passed"] is False


UMIS = ["ACGTACGTACGT", "CCCCAAAAGGGG", "TTTTGGGGCCCA", "GATTACAGATTA", "AAAACCCCGGGG", "ACACACACACAC"]
GENE_IDS = np.array(["gA", "gB", "gC"], object)


def _code(umi):
    return int(umi.translate(str.maketrans("ACGT", "0123")), 4)


def _counted(path, rows):
    """counted_molecules.parquet with the columns the join reads, in the dtypes of stage 24."""
    pq.write_table(pa.table({"cell": pa.array([r[0] for r in rows], pa.int32()),
                             "gene": pa.array([r[1] for r in rows], pa.int32()),
                             "umi": pa.array([_code(r[2]) for r in rows], pa.uint32()),
                             "lost": pa.array([r[3] for r in rows], pa.uint32())}), path)
    return path


def _join_tables():
    """Labeled molecules of called cells and their pa reads (with a member read and the pa read of a molecule that is not
    among them), pa_sites rows and the counted molecules of the two cells."""
    mol = pd.DataFrame([(10, 0, 0, "gA", "3UTR", "templated"), (11, 0, 1, "gC", "CDS", "templated"),
                        (12, 1, 0, "gB", "intron", "nontemplated"), (13, 1, 2, "", "intergenic", "templated"),
                        (14, 1, 3, "gC", "3UTR", "ambiguous"), (15, 0, 4, "gB", "3UTR", "templated")],
                       columns=["mol", "cell", "u", "gene", "region", "label"])
    mol = mol.assign(cb=[BARCODES[c] for c in mol.cell], ub=[UMIS[u] for u in mol.u]).drop(columns=["cell", "u"])
    pa_reads = [("p10", 10, "gB", 25),                        # our gene gA, GX gB: the group of gB is counted
                ("p11a", 11, "gA", 19), ("p11b", 11, "gC", 17),  # one of two pa reads in the counted group
                ("p12", 12, "gB", 19),                        # the cell and UMI counts gA, the pa read is in the group of gB
                ("p13", 13, None, 0),                         # no GX
                ("p14a", 14, "gA;gC", 0), ("p14b", 14, "gC", 0),  # xf bit 1 unset, although the group of gC is counted
                ("p15a", 15, "gB", 19), ("p15b", 15, "gA", 19),  # two groups, neither counted
                ("p99", 99, "gA", 25)]                         # a molecule that is not among the called-cell molecules
    tr = pd.DataFrame({"qname": [r[0] for r in pa_reads] + ["m10"], "mol": [r[1] for r in pa_reads] + [10],
                       "role": ["pa"] * len(pa_reads) + ["member"], "dist5_exon": [np.nan] * len(pa_reads) + [80.0],
                       "dist_mode": [""] * len(pa_reads) + ["exon"], "spliced": False})
    sites = pd.DataFrame({"qname": [r[0] for r in pa_reads][::-1], "gx": [r[2] for r in pa_reads][::-1],
                          "xf": [r[3] for r in pa_reads][::-1], "cb": "x", "ub": "x"})
    # cell, gene row, UMI, lost bits
    counted = [(0, 1, UMIS[0], 2), (0, 2, UMIS[1], 1), (1, 0, UMIS[0], 4), (1, 2, UMIS[3], 0), (0, 0, UMIS[5], 7),
               (1, 1, UMIS[5], 0), (1, 0, UMIS[4], 0)]
    return mol, tr, sites, counted


@pytest.mark.parametrize("batch_rows", [2, 1000])
def test_join_takes_gx_of_the_pa_read_when_our_gene_differs(tmp_path, batch_rows):
    mol, tr, sites, counted = _join_tables()
    path = _counted(tmp_path / "counted_molecules.parquet", counted)
    j = join_counted(mol, tr, sites, BARCODES, GENE_IDS, path, "L", batch_rows=batch_rows)
    assert JOIN_COLUMNS == ["lib", "mol", "cell", "gene", "umi", "joined", "region"]
    assert list(j.columns) == JOIN_COLUMNS + ["lost"]
    assert j.dtypes.astype(str).tolist() == ["str", "int64", "int32", "int32", "uint32", "bool", "str", "uint32"]
    assert j.mol.tolist() == [10, 11, 12, 13, 14, 15] and set(j.lib) == {"L"}
    assert j.cell.tolist() == [0, 0, 1, 1, 1, 0] and j.umi.tolist() == [_code(UMIS[u]) for u in (0, 1, 0, 2, 3, 4)]
    # m10 takes gB (GX of its pa read), m11 the counted group of its second pa read; m12 and m15 lie in groups that are
    # not counted (m15: the smaller gene row of its two); m13 and m14 have no pa read with xf bit 1 and a GX
    assert j.gene.tolist() == [1, 2, 1, -1, -1, 0]
    assert j.joined.tolist() == [True, True, False, False, False, False]
    assert j.lost.tolist() == [2, 1, 0, 0, 0, 0] and j.region.tolist() == mol.region.tolist()
    s = join_summary(j, mol, GENE_IDS)
    assert s == {"molecules": 6, "joined": 2, "group_not_counted": 2, "no_group": 2, "counted_molecules_joined": 2,
                 "joined_gene_differs": 1, "joined_gene_differs_by_label": {"templated": 1},
                 "by_region": {"3UTR": {"joined": 1, "group_not_counted": 1, "no_group": 1},
                               "CDS": {"joined": 1, "group_not_counted": 0, "no_group": 0},
                               "intergenic": {"joined": 0, "group_not_counted": 0, "no_group": 1},
                               "intron": {"joined": 0, "group_not_counted": 1, "no_group": 0}}}
    with pytest.raises(ValueError, match="1 molecules have a barcode that is not a row of barcodes.tsv"):
        join_counted(mol.assign(cb=lambda d: d.cb.where(d.mol != 12, "TTTTTT-1")), tr, sites, BARCODES, GENE_IDS, path, "L")


def test_two_truth_molecules_can_join_one_counted_molecule(tmp_path):
    mol, tr, sites, counted = _join_tables()
    # a second molecule of the cell and UMI of m10 (another strand at its site, say) with a pa read in the group of gB
    mol = pd.concat([mol, mol[mol.mol == 10].assign(mol=16, region="antisense")], ignore_index=True)
    tr = pd.concat([tr, pd.DataFrame({"qname": ["p16"], "mol": [16], "role": ["pa"], "dist5_exon": [np.nan], "dist_mode": [""],
                                      "spliced": [False]})], ignore_index=True)
    sites = pd.concat([sites, pd.DataFrame({"qname": ["p16"], "gx": ["gB"], "xf": [17], "cb": "x", "ub": "x"})],
                      ignore_index=True)
    j = join_counted(mol, tr, sites, BARCODES, GENE_IDS, _counted(tmp_path / "c.parquet", counted), "L")
    s = join_summary(j, mol, GENE_IDS)
    assert j.joined.sum() == 3 and s["joined"] == 3 and s["counted_molecules_joined"] == 2


def _synthetic_truth(s):
    """Truth molecules of the synthetic 10x library: their pa reads are records of it, chosen per case, and pa_sites holds
    every record of a called cell with its GX and xf."""
    rec = s.reads.assign(qname=s.reads.qname.astype(str))
    pick = {"A": rec[rec.case == "pa_only"], "B": rec[(rec.case == "pa_and_member") & (rec.pa > 0)],
            "C": rec[(rec.case == "minus_pa") & (rec.pa > 0)], "D": rec[(rec.case == "three_vs_one") & rec.xf.isin([19, 25])],
            "E": rec[(rec.case == "leads_by_one") & (rec.xf == 19)], "F": rec[(rec.case == "antisense") & (rec.re == "E")]}
    rows, reads = [], []
    for k, (name, r) in enumerate(pick.items()):
        gene = "g2" if name == "B" else r.gx.dropna().iloc[0] if r.gx.notna().any() else "g1"
        rows.append((k, name, r.cb.iloc[0], r.ub.iloc[0], gene, "3UTR" if name in "ABC" else "CDS"))
        reads += [(q, k) for q in r.qname]
    mol = pd.DataFrame(rows, columns=["mol", "name", "cb", "ub", "gene", "region"]).assign(label="templated")
    tr = pd.DataFrame(reads, columns=["qname", "mol"]).assign(role="pa", dist5_exon=np.nan, dist_mode="", spliced=False)
    sites = rec[rec.cb.isin(list(CELLS_10X.values()))][["qname", "cb", "ub", "gx", "xf"]]
    return mol, tr, sites


def test_join_on_the_synthetic_10x_library(synth10x, stage24):
    d, _, _ = stage24
    mol, tr, sites = _synthetic_truth(synth10x)
    called = sorted(CELLS_10X.values())
    genes = np.array(["g1", "g2", "g3", "g4"], object)
    j = join_counted(mol, tr, sites, called, genes, d / "recount" / "counted_molecules.parquet", "X", batch_rows=3)
    assert dict(zip(mol.name, j.joined)) == {"A": True, "B": True, "C": True, "D": True, "E": False, "F": False}
    # B: our gene g2, GX g1
    assert dict(zip(mol.name, [genes[g] if g >= 0 else "" for g in j.gene])) == {
        "A": "g1", "B": "g1", "C": "g2", "D": "g1", "E": "g2", "F": ""}
    # lost bits: the designed events of the cell and UMI at each setting (bit t: SETTINGS_10X[t]); pa_only keeps g1 in both
    # modes and loses it at the controls, three_vs_one loses g1 at every setting (a tie, then a switch to g3)
    ev = synth10x.events
    for r, lost in zip(mol[mol.name != "F"].itertuples(), j.lost):  # F: its cell and UMI has no read group
        e = ev[(ev.cb == r.cb) & (ev.ub == r.ub)].set_index("tag").loc[[setting_tag(x) for x in SETTINGS_10X]]
        assert lost == sum(int(b) << t for t, b in enumerate(e.lost & j.joined[r.mol]))
    assert dict(zip(mol.name, j.lost)) == {"A": 12, "B": 0, "C": 0, "D": 15, "E": 0, "F": 0}
    # the joined keys are rows of counted_molecules, the group of E one of other_read_groups
    cm = pd.read_parquet(d / "recount" / "counted_molecules.parquet")
    key = ["cell", "gene", "umi"]
    assert len(j[j.joined].merge(cm[key], on=key)) == 4
    other = pd.read_parquet(d / "recount" / "other_read_groups.parquet")
    assert j[mol.name == "E"].merge(other[key], on=key).mol.tolist() == [4]
    assert j.umi.tolist() == [_code(u) for u in mol.ub] and j.cell.tolist() == [called.index(c) for c in mol.cb]


def _outcome_units():
    """Five labeled molecules of called cells: m1 to m3 with a member read within the cap, m4 with its pa read alone and m5
    with a member read beyond it. Removal at T0 and T1 per read."""
    mol = pd.DataFrame([(1, "templated", 15, "14-20"), (2, "templated", 11, "10-13"), (3, "nontemplated", 2, "0-4"),
                        (4, "templated", 18, "14-20"), (5, "nontemplated", 0, "0-4")], columns=["mol", "label", "a20", "a_band"])
    mol = mol.assign(cb=BARCODES[0], region="3UTR", gene="gA", has_ts=False, hexamer=False, ref="chrA", strand="+", site=100)
    reads = [("p1", 1, "pa", np.nan), ("r1", 1, "member", 100.0), ("p2", 2, "pa", np.nan), ("r2", 2, "member", 200.0),
             ("p3", 3, "pa", np.nan), ("r3", 3, "member", 50.0), ("p4", 4, "pa", np.nan), ("p5", 5, "pa", np.nan),
             ("r5", 5, "member", 1500.0)]
    tr = pd.DataFrame(reads, columns=["qname", "mol", "role", "dist5_exon"]).assign(dist_mode="exon", spliced=False)
    removed = {"p1": (1, 0), "r1": (1, 1), "p2": (0, 1), "r2": (1, 1), "p3": (1, 0), "r3": (1, 0), "p4": (1, 0), "p5": (0, 0),
               "r5": (1, 1)}
    fl = pd.DataFrame([(q, "chrA", 0, 0, bool(a), bool(b)) for q, (a, b) in removed.items()],
                      columns=["qname", "ref", "pos", "flag", "T0x", "T1x"])
    return mol, tr, fl


def test_matrix_outcome_counts_the_lost_bit_against_the_unit_rule_on_joined_molecules():
    mol, tr, fl = _outcome_units()
    tags = ["c300_l10_m1_i", "c375_l3_m2_i"]
    fl = fl.rename(columns={"T0x": tags[0], "T1x": tags[1]})
    sets = {"with_member": units(mol, tr, fl, tags, cap=1000),
            "all_labeled": units(mol, tr, fl, tags, cap=1000, all_labeled=True)}
    # m3 does not join; lost bits: the first tag is bit 1, the second bit 0
    join = pd.DataFrame({"mol": [1, 2, 3, 4, 5], "joined": [True, True, False, True, True],
                         "lost": np.array([2, 1, 0, 2, 0], np.uint32)})
    o = matrix_outcome(join, sets, tags, {tags[0]: 1, tags[1]: 0}, "L")
    assert list(o.columns) == OUTCOME_COLUMNS == ["lib", "tag", "units", "label", "a_band", "joined", "not_joined",
                                                  "lost_in_matrix", "removed_as_unit", "both"]
    got = {(r.tag, r.units, r.label, r.a_band): (r.joined, r.not_joined, r.lost_in_matrix, r.removed_as_unit, r.both)
           for r in o.itertuples()}
    assert set(o.lib) == {"L"} and len(o) == len(got) == 2 * (3 + 3)
    t0, t1 = tags
    # m1: lost at t0 and removed (pa and member read); m2: removed at t1 only, lost at t1; m3 is removed at t0 but not joined
    assert got[t0, "with_member", "templated", "14-20"] == (1, 0, 1, 1, 1)
    assert got[t0, "with_member", "templated", "10-13"] == (1, 0, 0, 0, 0)
    assert got[t0, "with_member", "nontemplated", "0-4"] == (0, 1, 0, 0, 0)
    assert got[t1, "with_member", "templated", "14-20"] == (1, 0, 0, 0, 0)
    assert got[t1, "with_member", "templated", "10-13"] == (1, 0, 1, 1, 1)
    # all labeled: m4 (its pa read removed at t0, lost at t0) and m5 (pa read kept, nothing lost) join as well
    assert got[t0, "all_labeled", "templated", "14-20"] == (2, 0, 2, 2, 2)
    assert got[t0, "all_labeled", "nontemplated", "0-4"] == (1, 1, 0, 0, 0)
    assert got[t1, "all_labeled", "templated", "14-20"] == (2, 0, 0, 0, 0)
    assert o.dtypes[["joined", "not_joined", "lost_in_matrix", "removed_as_unit", "both"]].astype(str).tolist() == ["int64"] * 5


def _stage27():
    spec = importlib.util.spec_from_file_location("stage27", REPO / "scripts" / "27_truth_strata.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    return stage


# label and a20 of the molecules of _synthetic_truth
LABELS_10X = {"A": ("templated", 15, "14-20"), "B": ("nontemplated", 2, "0-4"), "C": ("templated", 9, "8-9"),
              "D": ("templated", 12, "10-13"), "E": ("nontemplated", 0, "0-4"), "F": ("ambiguous", 6, "5-7")}


def _stage27_tree(s, d, root):
    """What stage 27 reads for the synthetic library X, next to the outputs of stage 24 in d: the truth tables of
    _synthetic_truth with member reads (the other records of each cell and UMI) and a molecule of an uncalled barcode,
    truth_flags at SETTINGS_10X, pa_sites, the classes of stage 25 (c5 without a row) and eval/metrics.parquet as stage 17
    writes it. Returns (base, cells template, molecules of called cells, truth reads, truth flags)."""
    mol, tr, sites = _synthetic_truth(s)
    rec = s.reads.assign(qname=s.reads.qname.astype(str))
    tags = [setting_tag(x) for x in SETTINGS_10X]
    unc = rec[rec.case == "uncalled"].iloc[0]
    mol = pd.concat([mol, pd.DataFrame([(len(mol), "U", unc.cb, unc.ub, "g1", "3UTR")], columns=mol.columns[:6])
                     .assign(label="templated")], ignore_index=True)
    tr = pd.concat([tr, pd.DataFrame({"qname": [unc.qname], "mol": [len(mol) - 1], "role": "pa", "dist5_exon": np.nan,
                                      "dist_mode": "", "spliced": False})], ignore_index=True)
    lab = [LABELS_10X.get(n, ("templated", 15, "14-20")) for n in mol.name]
    mol = mol.assign(label=[x[0] for x in lab], a20=[x[1] for x in lab], a_band=[x[2] for x in lab], has_ts=False,
                     hexamer=False, ref="chrA", strand="+", site=np.arange(len(mol)) * 100_000, in_cell=mol.name != "U")
    members = [(q, m.mol, "member", 100.0, "exon", False) for m in mol.itertuples()
               for q in rec[(rec.cb == m.cb) & (rec.ub == m.ub) & ~rec.qname.isin(tr.qname)].qname]
    tr = pd.concat([tr, pd.DataFrame(members, columns=tr.columns)], ignore_index=True)
    fl = rec[rec.qname.isin(tr.qname)][["qname", "ref", "pos"] + tags].assign(flag=0)[["qname", "ref", "pos", "flag"] + tags]
    base = root / "base"
    (base / "X" / "truth").mkdir(parents=True)
    (base / "X" / "grid").mkdir()
    mol.drop(columns="name").to_parquet(base / "X" / "truth" / "molecules_labeled.parquet")
    tr.to_parquet(base / "X" / "truth" / "truth_reads.parquet")
    fl.to_parquet(base / "X" / "grid" / "truth_flags.parquet")
    pd.concat([sites, rec[rec.case == "uncalled"][sites.columns]]).to_parquet(base / "X" / "pa_sites.parquet")
    (root / "cells" / "X").mkdir(parents=True)
    pd.DataFrame({"cb": [CELLS_10X[c] for c in ("c1", "c2", "c3", "c4")], "class": ["rod", "cone", "cone", "rod"],
                  "umis": 1}).to_parquet(root / "cells" / "X" / "cells.parquet")
    called = mol[mol.in_cell].drop(columns="name").reset_index(drop=True)
    (base / "eval").mkdir()
    evaluate(units(called, tr, fl, tags, 1000), "X", 1000, "8/4").to_parquet(base / "eval" / "metrics.parquet")
    return base, str(root / "cells" / "{lib}"), called, tr, fl


def test_stage27_on_the_synthetic_10x_library(synth10x, stage24, tmp_path, monkeypatch, capsys):
    d, _, _ = stage24
    stage = _stage27()
    # the bootstrap runs at the chosen and the published settings, all of them impact settings; the synthetic library has
    # flags at SETTINGS_10X only, so both lists are replaced below
    assert stage.BOOT_TAGS == [setting_tag(s) for s in (*CHOSEN_SETTINGS, *published())]
    assert set(stage.BOOT_TAGS) <= set(stage.TAGS)
    # the lost bit of a setting is its place in recount.json, not in TAGS: the stage gets the settings in reverse order
    bits = {t: i for i, t in enumerate(json.loads((d / "recount" / "recount.json").read_text())["settings"])}
    tags = [setting_tag(x) for x in SETTINGS_10X][::-1]
    assert list(bits) == tags[::-1]
    monkeypatch.setattr(recount, "LIBRARIES", {"X": synth10x.bam})
    monkeypatch.setattr(stage, "TAGS", tags)
    monkeypatch.setattr(stage, "BOOT_TAGS", tags[:2])
    base, cells, mol, tr, fl = _stage27_tree(synth10x, d, tmp_path)
    out = tmp_path / "eval27"
    argv = ["27", "X", "--base", str(base), "--cells", cells, "--matrices", str(d / "matrices"), "--recount", str(d / "recount"),
            "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    stage.main()
    gt = pd.read_parquet(d / "matrices" / "gene_totals.parquet")
    gene_umis = gt[gt.tag == "unfiltered"].set_index("gene_id").umis
    u, bounds = with_strata(units(mol, tr, fl, tags, 1000), pd.read_parquet(tmp_path / "cells" / "X" / "cells.parquet"),
                            gene_umis)
    strata = pd.read_parquet(out / "strata_m3.parquet")
    pd.testing.assert_frame_equal(strata, evaluate(u, "X", 1000, "8/4", levels=LEVELS, kinds=KINDS))
    assert list(strata.columns) == METRIC_COLUMNS and set(strata.stratum_kind) == set(KINDS)
    assert set(strata[strata.stratum_kind == "cell_class"].stratum) == {"rod", "cone", UNASSIGNED}   # C: cell c5
    boot = pd.read_parquet(out / "strata_m3_bootstrap.parquet")
    pd.testing.assert_frame_equal(boot, strata_bootstrap(u, tags[:2], "X"))
    join = pd.read_parquet(out / "truth_to_counted.parquet")
    ref = join_counted(mol, tr, pd.read_parquet(base / "X" / "pa_sites.parquet"), sorted(CELLS_10X.values()),
                       np.array(["g1", "g2", "g3", "g4"], object), d / "recount" / "counted_molecules.parquet", "X")
    pd.testing.assert_frame_equal(join, ref[JOIN_COLUMNS])
    assert join.joined.tolist() == [True, True, True, True, False, False]          # the uncalled molecule is left out
    outcome = pd.read_parquet(out / "matrix_outcome.parquet")
    sets = {"with_member": u, "all_labeled": units(mol, tr, fl, tags, 1000, all_labeled=True)}
    pd.testing.assert_frame_equal(outcome, matrix_outcome(ref, sets, tags, bits, "X"))
    assert set(outcome.units) == {"with_member", "all_labeled"} and set(outcome.tag) == set(tags)
    # A (pa read alone) and F (no member read) are units only among all labeled molecules
    assert outcome[outcome.units == "all_labeled"].groupby("tag")[["joined", "not_joined"]].sum().sum(axis=1).tolist() == [6] * 4
    assert outcome[outcome.units == "with_member"].groupby("tag")[["joined", "not_joined"]].sum().sum(axis=1).tolist() == [4] * 4
    js = json.loads((out / "strata_m3.json").read_text())
    x = js["libraries"]["X"]
    assert x["expr_bins"]["boundaries"] == bounds.tolist() and x["join"] == join_summary(ref, mol, ["g1", "g2", "g3", "g4"])
    assert x["few_molecules"] == few_molecules(u) and x["acceptance"]["passed"] is True and js["acceptance_passed"] is True
    assert x["units"] == {"with_member": {"templated": 2, "nontemplated": 2, "ambiguous": 0},
                          "all_labeled": {"templated": 3, "nontemplated": 2, "ambiguous": 1}}
    assert x["cell_class"] == {"cone": 2, "rod": 1, "unassigned": 1}
    assert x["expr_bins"]["molecules"] == u.mols.expr_bin.value_counts().sort_index().to_dict()
    assert js["bootstrap"] == {"tags": tags[:2], "n": 1000, "seed": 0, "levels": list(LEVELS)}
    assert (js["cap"], js["cutoffs"], js["min_stratum"], js["expr_bins"]) == (1000, "8/4", MIN_STRATUM, 4)
    assert x["inputs"]["counted_molecules"] == str(d / "recount" / "counted_molecules.parquet")
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["libraries"] == ["X"] and line["acceptance_passed"] is True and line["joined"] == {"X": 4}
    assert line["boundaries"] == {"X": bounds.tolist()} and "seconds" in line

    # the whole of milestone 2 with one non-templated unit more: the outputs are written and the stage exits non-zero
    m = pd.read_parquet(base / "eval" / "metrics.parquet")
    m.loc[(m.stratum_kind == "all") & (m.level == "molecule") & (m.tag == tags[0]), "tn"] += 1
    m.to_parquet(base / "eval" / "metrics.parquet")
    (out / "strata_m3.json").unlink()
    with pytest.raises(SystemExit, match="acceptance failed for X"):
        stage.main()
    js = json.loads((out / "strata_m3.json").read_text())
    assert js["acceptance_passed"] is False and js["libraries"]["X"]["acceptance"]["differing"][0]["tag"] == tags[0]

    # stage 24 not passed: the stage refuses to start
    monkeypatch.setattr(sys, "argv", argv[:8] + ["--recount", str(tmp_path / "none"), "--out", str(out)])
    with pytest.raises(SystemExit, match="has not passed its acceptance checks for X"):
        stage.main()
