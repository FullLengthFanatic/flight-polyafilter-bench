import pandas as pd
import pysam

from pafbench.molecules import beyond_cap, choose_cap, consolidate_sites, gene_match, member_density, member_distances


def _pa(rows):
    cols = ["cb", "ub", "strand", "ref", "site", "a20", "down20", "up40", "hexamer", "genes", "re"]
    return pd.DataFrame(rows, columns=cols)


def test_consolidate_drops_spread_and_missing_umi():
    pa = _pa([
        ["c1", "u1", "+", "chr1", 100, 15, "A" * 20, "C" * 40, False, "g", "E"],
        ["c1", "u1", "+", "chr1", 104, 15, "A" * 20, "C" * 40, False, "g", "E"],
        ["c1", "u1", "+", "chr1", 109, 15, "A" * 20, "C" * 40, False, "g", "E"],
        ["c2", "u2", "+", "chr1", 100, 3, "C" * 20, "C" * 40, True, "g", "E"],
        ["c2", "u2", "+", "chr1", 200, 3, "C" * 20, "C" * 40, True, "g", "E"],
        ["c3", None, "+", "chr1", 300, 3, "C" * 20, "C" * 40, True, "g", "E"],
    ])
    mol = consolidate_sites(pa)
    assert len(mol) == 1
    assert mol.iloc[0].site == 104 and mol.iloc[0].n_pa == 3


def test_member_distances():
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "chr1", "LN": 10000}]})
    a = pysam.AlignedSegment(hdr)
    a.reference_name, a.reference_start, a.cigarstring, a.flag = "chr1", 700, "90M", 0
    a.query_sequence = "C" * 90
    assert member_distances(a, 1000) == (300, 210)
    a.flag = 16
    assert member_distances(a, 500) == (290, 200)


def test_gene_match():
    assert gene_match("", "") and gene_match("g1;g2", "g2") and not gene_match("g1", "g2") and not gene_match("g1", "")


def test_choose_cap():
    # bins from 750 to 1500 are empty, so d=750 passes; d=500 fails on the 600-nt members
    members = pd.DataFrame({"dist5": [150] * 900 + [600] * 50 + [500000] * 9,
                            "dist3": [0] * 959, "spliced": [False] * 959})
    curve = member_density(members, n_molecules=100)
    background = curve[curve.lo == 100000].density.iloc[0]
    assert choose_cap(curve, background) == 750


def test_member_density_columns_and_spliced():
    members = pd.DataFrame({"dist5": [150, 4000], "dist3": [0, 0], "dist5_exon": [150, 250], "dist3_exon": [0, 0],
                            "spliced": [False, True]})
    default = member_density(members, n_molecules=1)
    assert default.n.sum() == 1
    exon = member_density(members, n_molecules=1, d5="dist5_exon", d3="dist3_exon", include_spliced=True)
    assert exon.loc[exon.lo == 100, "n"].iloc[0] == 1 and exon.loc[exon.lo == 200, "n"].iloc[0] == 1


def test_beyond_cap():
    members = pd.DataFrame({"mol": [1, 1, 2, 3], "dist5_exon": [300, 1500, 2500, -5], "dist3_exon": [0, 0, 0, -30]})
    r = beyond_cap(members)
    assert r["population"] == 3
    assert r["500"] == {"members": 2, "molecules": 2} and r["1000"] == {"members": 2, "molecules": 2}
    assert r["2000"] == {"members": 1, "molecules": 1}
