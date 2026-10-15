"""Stage 27 (pafbench.strata): truth-set strata by cell class and expression level, the join of truth molecules to the
molecules counted before filtering, accuracy as the matrix sees it and the acceptance, on synthetic truth sets and on the
synthetic 10x library of tests/synth.build_10x."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from pafbench.config import A_BANDS, MIN_STRATUM
from pafbench.evaluate import COUNT_COLUMNS, bootstrap, evaluate, subset_units, units
from pafbench.regions import REGION_NAMES
from pafbench.strata import (KINDS, LEVELS, NO_UMI, UNASSIGNED, acceptance, cell_classes, expr_bins, few_molecules,
                             strata_bootstrap, with_strata)

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
    a = acceptance(strata, metrics, "L")
    assert a["passed"] is True and a["compared"] == len(TAGS) * len(LEVELS) * len(KINDS) and a["differing"] == []
    lab = u.mols.label.value_counts()
    assert a["units"]["molecule"] == {"templated": int(lab["templated"]), "nontemplated": int(lab["nontemplated"])}
    assert a["units"]["read"]["templated"] == int((u.reads.label == "templated").sum())
    # one templated unit more in the whole of milestone 2: tp + fn and the sensitivity differ, for both kinds
    bad = metrics.copy()
    hit = (bad.lib == "L") & (bad.cap == 1000) & (bad.level == "molecule") & (bad.stratum_kind == "all") & (bad.tag == TAGS[1])
    bad.loc[hit, "fn"] += 1
    a = acceptance(strata, bad, "L")
    assert a["passed"] is False and a["compared"] == len(TAGS) * len(LEVELS) * len(KINDS)
    assert [(d["tag"], d["level"], d["stratum_kind"]) for d in a["differing"]] == [
        (TAGS[1], "molecule", "cell_class"), (TAGS[1], "molecule", "expr_bin")]
    assert a["differing"][0]["templated"] == a["differing"][0]["templated_m2"] - 1
    # a sensitivity that differs at equal unit counts fails too
    bad = metrics.copy()
    bad.loc[hit, "sens"] = 0.123
    assert acceptance(strata, bad, "L")["passed"] is False
    # a setting or level missing from milestone 2 fails
    a = acceptance(strata, metrics[~((metrics.tag == TAGS[2]) & (metrics.level == "read"))], "L")
    assert a["passed"] is False and {(d["tag"], d["level"]) for d in a["differing"]} == {(TAGS[2], "read")}
