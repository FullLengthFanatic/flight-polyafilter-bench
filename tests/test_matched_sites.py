
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from pafbench.burden import (END_BINS, SETS, downsample_sets, downsampling_summary, end_bin, end_distances, gene_ends,
                             match_sites, same_site, same_site_check, set_counts, set_summary, shared_genes, site_sets,
                             site_table, usage, usage_comparison)
from pafbench.config import A_BANDS, DOWNSAMPLE_DRAWS
from pafbench.exons import load_collapsed_exons
from pafbench.labels import a_band
from pafbench.regions import REGION_NAMES
from pafbench.sites import classify

# gP on plus: t1 ends at 1300, t2 at 1250 (collapsed exons [1000, 1100) and [1200, 1300)); gM on minus: t3 ends at 2000, t4
# at 2050 (collapsed exons [2000, 2100) and [2200, 2300)); ends in the priming_site convention (GTF end on plus, start - 1
# on minus). gQ on chr2 plus has one exon.
GTF = [
    ("chr1", "gene", 1001, 1300, "+", 'gene_id "gP";'),
    ("chr1", "transcript", 1001, 1300, "+", 'gene_id "gP"; transcript_id "t1";'),
    ("chr1", "exon", 1001, 1100, "+", 'gene_id "gP"; transcript_id "t1";'),
    ("chr1", "exon", 1201, 1300, "+", 'gene_id "gP"; transcript_id "t1";'),
    ("chr1", "transcript", 1001, 1250, "+", 'gene_id "gP"; transcript_id "t2";'),
    ("chr1", "exon", 1001, 1100, "+", 'gene_id "gP"; transcript_id "t2";'),
    ("chr1", "exon", 1201, 1250, "+", 'gene_id "gP"; transcript_id "t2";'),
    ("chr1", "gene", 2001, 2300, "-", 'gene_id "gM";'),
    ("chr1", "transcript", 2001, 2300, "-", 'gene_id "gM"; transcript_id "t3";'),
    ("chr1", "exon", 2001, 2100, "-", 'gene_id "gM"; transcript_id "t3";'),
    ("chr1", "exon", 2201, 2300, "-", 'gene_id "gM"; transcript_id "t3";'),
    ("chr1", "transcript", 2051, 2300, "-", 'gene_id "gM"; transcript_id "t4";'),
    ("chr1", "exon", 2051, 2100, "-", 'gene_id "gM"; transcript_id "t4";'),
    ("chr1", "exon", 2201, 2300, "-", 'gene_id "gM"; transcript_id "t4";'),
    ("chr2", "gene", 101, 600, "+", 'gene_id "gQ";'),
    ("chr2", "transcript", 101, 600, "+", 'gene_id "gQ"; transcript_id "t5";'),
    ("chr2", "exon", 101, 600, "+", 'gene_id "gQ"; transcript_id "t5";'),
]


def _gtf(tmp_path):
    p = tmp_path / "genes.gtf"
    p.write_text("".join(f"{c}\tt\t{ft}\t{s}\t{e}\t.\t{strand}\t.\t{attr}\n" for c, ft, s, e, strand, attr in GTF))
    return p


def _sites(rows):
    return pd.DataFrame(rows, columns=["ref", "strand", "site"])


def _mols(spec, in_cell=True):
    """Molecules from (ref, strand, site, gene, n, a20, region) rows: n copies each, label and a_band from a20."""
    rows = []
    for ref, strand, site, gene, n, a20, region in spec:
        down20 = "A" * a20 + "C" * (20 - a20)
        rows += [(ref, strand, site, gene, a20, down20, "G" * 40, a20 == 0, region, in_cell)] * n
    m = pd.DataFrame(rows, columns=["ref", "strand", "site", "gene", "a20", "down20", "up40", "hexamer", "region", "in_cell"])
    return m.assign(label=[classify(x, 8, 4) for x in m.a20], a_band=[a_band(x) for x in m.a20])


def _brute_mutual(a, b, tol):
    def nearest(x, other):
        cand = [(abs(y.site - x.site), y.site, j) for j, y in enumerate(other.itertuples())
                if (y.ref, y.strand) == (x.ref, x.strand)]
        return min(cand)[2] if cand and min(cand)[0] <= tol else None

    nb = [nearest(x, b) for x in a.itertuples()]
    na = [nearest(y, a) for y in b.itertuples()]
    return {(i, j) for i, j in enumerate(nb) if j is not None and na[j] == i}


def test_match_sites_mutual_nearest_within_tol_strand_and_ref():
    a = _sites([("chr1", "+", 100), ("chr1", "+", 200), ("chr1", "-", 300), ("chr2", "+", 50), ("chr3", "+", 10)])
    b = _sites([("chr1", "+", 104), ("chr1", "+", 102), ("chr1", "+", 206), ("chr1", "+", 300), ("chr2", "+", 52),
                ("chr2", "+", 48), ("chr3", "+", 15)])
    ia, ib = match_sites(a, b, tol=5)
    # 100 with 102 (104 is nearer to 100 than to anything else, but 100 takes 102); 200 and 206 are 6 apart; 300 on the other
    # strand; 50 between 48 and 52 takes the lower; 10 and 15 at exactly tol
    assert list(zip(ia.tolist(), ib.tolist())) == [(0, 1), (3, 5), (4, 6)]
    assert ia.dtype == np.int64 and ib.dtype == np.int64
    ia, ib = match_sites(a, b, tol=6)
    assert (1, 2) in set(zip(ia.tolist(), ib.tolist()))
    assert [len(x) for x in match_sites(a, b.iloc[:0])] == [0, 0]


def test_match_sites_is_symmetric_and_mutual_nearest_on_random_sites():
    rng = np.random.default_rng(3)
    for k in range(20):
        a = _sites(sorted({("chr1", s, int(x)) for s, x in zip(rng.choice(["+", "-"], 40), rng.integers(0, 300, 40))}))
        b = _sites(sorted({("chr1", s, int(x)) for s, x in zip(rng.choice(["+", "-"], 40), rng.integers(0, 300, 40))}))
        a = a.sample(frac=1, random_state=k).reset_index(drop=True)
        b = b.sample(frac=1, random_state=k + 1).reset_index(drop=True)
        ab = set(zip(*(x.tolist() for x in match_sites(a, b, tol=5))))
        ba = set(zip(*(x.tolist() for x in match_sites(b, a, tol=5))))
        assert ab == {(i, j) for j, i in ba}
        assert ab == _brute_mutual(a, b, 5)


def test_a_site_nearest_to_two_sites_of_the_other_group_keeps_one_pair():
    a = _sites([("chr1", "+", 100)])
    b = _sites([("chr1", "+", 98), ("chr1", "+", 103)])
    # 100 is the nearest site of a for both 98 and 103; mutual nearest keeps 100 with 98 (2 nt) and leaves 103 unmatched
    assert [x.tolist() for x in match_sites(a, b)] == [[0], [0]]
    assert [x.tolist() for x in match_sites(b, a)] == [[0], [0]]
    # equal distance on both sides: the lower coordinate
    b = _sites([("chr1", "+", 103), ("chr1", "+", 97)])
    assert [x.tolist() for x in match_sites(a, b)] == [[0], [1]]


def test_gene_ends(tmp_path):
    ends = gene_ends(_gtf(tmp_path))
    assert sorted(ends) == ["gM", "gP", "gQ"]
    assert ends["gP"].tolist() == [1250, 1300] and ends["gM"].tolist() == [2000, 2050] and ends["gQ"].tolist() == [600]
    assert ends["gP"].dtype == np.int64


@pytest.mark.parametrize("strand,site,gene,dist,mode", [
    # plus, exonic: last cDNA base 1049; collapsed-exon bases to 1250 are 50 + 50 (genomic 200), to 1300 150
    ("+", 1050, "gP", 100, "exon"),
    # plus, exonic in the second exon, between the two ends: -30 to 1250, +20 to 1300
    ("+", 1280, "gP", 20, "exon"),
    # plus, exonic, equally far from both ends (-25, +25): the positive one
    ("+", 1275, "gP", 25, "exon"),
    # plus, exonic, past the last end: the last cDNA base 1299 is the last exon base, the site 1300 is d = 0
    ("+", 1300, "gP", 0, "exon"),
    # plus, intronic (last base 1149): genome, 100 to 1250
    ("+", 1150, "gP", 100, "genome"),
    # plus, the site at the exon start 1200 has its last cDNA base 1199 in the intron
    ("+", 1200, "gP", 50, "genome"),
    # minus, exonic: last cDNA base 2250; collapsed-exon bases to 2050 are 50 + 50 (genomic 200), to 2000 150
    ("-", 2250, "gM", 100, "exon"),
    # minus, intronic (last base 2150): genome, 100 to 2050
    ("-", 2150, "gM", 100, "genome"),
    # minus, exonic between the ends (last base 2020): +20 to 2000, -30 to 2050
    ("-", 2020, "gM", 20, "exon"),
    # minus, below the gene (last base 1990, outside its exons): genome, 10 past the end at 2000
    ("-", 1990, "gM", -10, "genome"),
    # minus, last cDNA base 2000 is the first base of the transcript: d = 0
    ("-", 2000, "gM", 0, "exon"),
])
def test_end_distances_both_strands(tmp_path, strand, site, gene, dist, mode):
    gtf = _gtf(tmp_path)
    s = pd.DataFrame({"ref": ["chr1"], "strand": [strand], "site": [site], "gene": [gene]})
    d, m = end_distances(s, load_collapsed_exons(gtf), gene_ends(gtf))
    assert d.tolist() == [dist] and m.tolist() == [mode]


def test_end_distances_mixed_rows_and_no_gene(tmp_path):
    gtf = _gtf(tmp_path)
    s = pd.DataFrame({"ref": ["chr1", "chr1", "chr1", "chr2"], "strand": ["-", "+", "+", "+"], "site": [2250, 1500, 1050, 700],
                      "gene": ["gM", "", "gP", "gQ"]})
    d, m = end_distances(s, load_collapsed_exons(gtf), gene_ends(gtf))
    assert np.isnan(d[1]) and m[1] == ""
    assert d[[0, 2, 3]].tolist() == [100, 100, -100] and m.tolist() == ["exon", "", "exon", "genome"]


def test_end_bin():
    d = np.array([0, 100, 101, -1, -100, -101, 300, 301, 1000, 1001, 5000, 5001, -5001, np.nan])
    assert end_bin(d).tolist() == ["0-100", "0-100", "101-300", "past 0-100", "past 0-100", "past 101-300", "101-300",
                                   "301-1000", "301-1000", "1001-5000", "1001-5000", ">5000", "past >5000", "no gene"]
    assert END_BINS == ("0-100", "101-300", "301-1000", "1001-5000", ">5000", "past 0-100", "past 101-300",
                        "past 301-1000", "past 1001-5000", "past >5000", "no gene")


def test_site_table(tmp_path):
    gtf = _gtf(tmp_path)
    mol = _mols([("chr1", "+", 1280, "gP", 3, 12, "3UTR"), ("chr1", "+", 1050, "gP", 2, 2, "CDS"),
                 ("chr1", "-", 2250, "gM", 1, 6, "3UTR"), ("chr2", "+", 900, "", 4, 9, "intergenic")])
    t = site_table(mol, load_collapsed_exons(gtf), gene_ends(gtf))
    assert list(t.columns) == ["ref", "strand", "site", "n", "label", "a_band", "a20", "down20", "up40", "hexamer", "region",
                               "gene", "dist_end", "dist_mode"]
    assert t[["ref", "strand", "site", "n"]].values.tolist() == [["chr1", "+", 1050, 2], ["chr1", "+", 1280, 3],
                                                                  ["chr1", "-", 2250, 1], ["chr2", "+", 900, 4]]
    assert t.label.tolist() == ["nontemplated", "templated", "ambiguous", "templated"]
    assert t.a_band.tolist() == ["0-4", "10-13", "5-7", "8-9"] and t.a20.tolist() == [2, 12, 6, 9]
    assert t.down20.tolist()[1] == "A" * 12 + "C" * 8 and t.hexamer.tolist() == [False] * 4
    assert t.region.tolist() == ["CDS", "3UTR", "3UTR", "intergenic"] and t.gene.tolist() == ["gP", "gP", "gM", ""]
    assert t.dist_end.tolist()[:3] == [100, 20, 100] and np.isnan(t.dist_end.iloc[3])
    assert t.dist_mode.tolist() == ["exon", "exon", "exon", ""]


def test_usage_shares_sum_to_one_per_gene():
    s = pd.DataFrame({"gene": ["g1", "g1", "g2", "g1", "", ""], "n": [3, 1, 5, 4, 2, 2]})
    u = usage(s)
    assert u[:4].tolist() == [3 / 8, 1 / 8, 1.0, 4 / 8] and np.isnan(u[4:]).all()
    assert pd.Series(u[:4]).groupby(s.gene[:4].to_numpy()).sum().tolist() == [1.0, 1.0]


def test_shared_genes_are_those_of_same_gene():
    a = pd.DataFrame({"gene": ["g1"] * 20 + ["g2"] * 25 + ["g3"] * 19 + [""] * 30, "label": "templated"})
    b = pd.DataFrame({"gene": ["g1"] * 21 + ["g2"] * 20 + ["g3"] * 40 + ["g4"] * 30 + [""] * 30, "label": "nontemplated"})
    assert shared_genes(a, b) == ["g1", "g2"]
    assert shared_genes(a, b, min_n=19) == ["g1", "g2", "g3"]


def _pair_tables(tmp_path):
    gtf = _gtf(tmp_path)
    exons, ends = load_collapsed_exons(gtf), gene_ends(gtf)
    a = site_table(_mols([("chr1", "+", 1050, "gP", 4, 12, "CDS"), ("chr1", "+", 1150, "gP", 1, 2, "intron"),
                          ("chr2", "+", 300, "gQ", 2, 9, "ncexon")]), exons, ends)
    b = site_table(_mols([("chr1", "+", 1052, "gP", 6, 13, "CDS"), ("chr1", "+", 1280, "gP", 2, 6, "3UTR"),
                          ("chr2", "+", 300, "gQ", 2, 9, "ncexon")]), exons, ends)
    return a, b


def test_three_sites_give_one_of_each_set(tmp_path):
    a, b = _pair_tables(tmp_path)
    r = site_sets(a, b, ["gP"])
    assert list(r.columns) == ["gene", "ref", "strand", "site_a", "site_b", "set", "n_a", "n_b", "usage_a", "usage_b", "label",
                               "a_band", "a20", "down20", "hexamer", "region", "dist_end", "dist_mode"]
    assert r.set.tolist() == ["shared", "first_only", "second_only"]
    assert r.site_a.tolist() == [1050, 1150, pd.NA] and r.site_b.tolist() == [1052, pd.NA, 1280]
    assert r.n_a.tolist() == [4, 1, 0] and r.n_b.tolist() == [6, 0, 2]
    assert r.usage_a.tolist() == [0.8, 0.2, 0.0] and r.usage_b.tolist() == [0.75, 0.0, 0.25]
    # features of the first group's site on shared rows, of the site itself otherwise
    assert r.a20.tolist() == [12, 2, 6] and r.label.tolist() == ["templated", "nontemplated", "ambiguous"]
    assert r.region.tolist() == ["CDS", "intron", "3UTR"] and r.dist_end.tolist() == [100, 100, 20]
    assert r.dist_mode.tolist() == ["exon", "genome", "exon"] and set(r.gene) == {"gP"}
    # usage shares sum to 1 per gene and group
    assert r.groupby("gene").usage_a.sum().tolist() == [1.0] and r.groupby("gene").usage_b.sum().tolist() == [1.0]
    both = site_sets(a, b, ["gP", "gQ"])
    assert both.set.tolist() == ["shared", "first_only", "second_only", "shared"]
    assert both.groupby("gene")[["usage_a", "usage_b"]].sum().values.tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_set_summary(tmp_path):
    a, b = _pair_tables(tmp_path)
    s = set_summary(site_sets(a, b, ["gP", "gQ"]))
    assert list(s.columns) == ["set", "feature", "bin", "sites", "molecules"]
    assert list(dict.fromkeys(s.set)) == list(SETS)
    assert list(dict.fromkeys(s.feature)) == ["label", "a_band", "a_pos", "hexamer", "region", "dist_bin"]
    n_bins = 3 + len(A_BANDS) + 20 + 2 + len(REGION_NAMES) + len(END_BINS)
    assert len(s) == 3 * n_bins
    got = {(r.set, r.feature, r.bin): (r.sites, r.molecules) for r in s.itertuples()}
    # shared: gP 1050 with 1052 (a20 12, 4 + 6 molecules) and gQ 300 with 300 (a20 9, 2 + 2 molecules)
    assert got[("shared", "label", "templated")] == (2, 14) and got[("shared", "label", "ambiguous")] == (0, 0)
    assert got[("shared", "a_band", "10-13")] == (1, 10) and got[("shared", "a_band", "8-9")] == (1, 4)
    assert [got[("shared", "a_pos", str(k))] for k in (1, 9, 10, 12, 13)] == [(2, 14), (2, 14), (1, 10), (1, 10), (0, 0)]
    assert got[("shared", "hexamer", "False")] == (2, 14) and got[("shared", "hexamer", "True")] == (0, 0)
    assert got[("shared", "region", "CDS")] == (1, 10) and got[("shared", "region", "ncexon")] == (1, 4)
    # gQ 300 is 300 nt before its end at 600, along the exon; gP 1050 is 100 before 1250
    assert got[("shared", "dist_bin", "0-100")] == (1, 10) and got[("shared", "dist_bin", "101-300")] == (1, 4)
    assert got[("first_only", "label", "nontemplated")] == (1, 1) and got[("first_only", "a_pos", "3")] == (0, 0)
    assert got[("second_only", "a_band", "5-7")] == (1, 2) and got[("second_only", "region", "3UTR")] == (1, 2)
    for st in SETS:
        sub = s[s.set == st]
        totals = sub[sub.feature != "a_pos"].groupby("feature")[["sites", "molecules"]].sum()
        assert totals.nunique().tolist() == [1, 1]


def test_usage_comparison():
    rows = pd.DataFrame({"set": ["shared"] * 4 + ["first_only", "second_only"], "usage_a": [0.1, 0.2, 0.4, 0.3, 0.5, 0.0],
                         "usage_b": [0.2, 0.1, 0.4, 0.6, 0.0, 0.3],
                         "label": ["templated", "templated", "nontemplated", "templated", "templated", "ambiguous"],
                         "region": ["3UTR", "intron", "3UTR", "3UTR", "3UTR", "CDS"]})
    u = usage_comparison(rows)
    rho, p = spearmanr([0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.4, 0.6])
    assert u["n"] == 4 and u["spearman_rho"] == pytest.approx(rho) and u["p_value"] == pytest.approx(p)
    # log2(usage_a / usage_b): -1, 1, 0, -1; quartiles by linear interpolation (numpy.quantile)
    assert u["log2_ratio"]["all"] == {"n": 4, "q25": -1.0, "median": -0.5, "q75": 0.25}
    assert u["log2_ratio"]["label"]["templated"] == {"n": 3, "q25": -1.0, "median": -1.0, "q75": 0.0}
    assert u["log2_ratio"]["label"]["ambiguous"] == {"n": 0, "q25": None, "median": None, "q75": None}
    assert u["log2_ratio"]["region"]["3UTR"]["n"] == 3 and u["log2_ratio"]["region"]["3UTR"]["median"] == -1.0
    assert list(u["log2_ratio"]["region"]) == list(REGION_NAMES)


def _depth_tables():
    """First group: g1 with 3 molecules at 1000; second: g1 with 12 sites of one molecule, every 100 nt from 950 (one of
    them within tol of 1000), and g2 with more molecules in the first group than in the second."""
    def t(rows):
        return pd.DataFrame(rows, columns=["ref", "strand", "site", "gene", "n"])

    a = t([("chr1", "+", 1000, "g1", 3), ("chr1", "+", 9000, "g2", 5)])
    b = t([("chr1", "+", 950 + 100 * k + (52 if k == 0 else 0), "g1", 1) for k in range(12)] + [("chr1", "+", 9001, "g2", 2),
                                                                                                    ("chr1", "+", 9100, "g2", 1)])
    return a, b


def test_downsample_sets_keeps_the_first_group_count_per_gene_and_is_seeded():
    a, b = _depth_tables()
    d = downsample_sets(a, b, ["g1", "g2"], draws=40, seed=1)
    assert list(d.columns) == ["draw", "molecules", "shared", "first_only", "second_only"] and d.draw.tolist() == list(range(40))
    assert (d.molecules == 3 + 3).all()
    # g1: 3 of its 12 cell molecules per draw, each at its own site; g2: both sites kept (3 molecules, fewer than 5)
    assert ((d.shared + d.second_only) == 3 + 2).all()
    assert (d.shared + d.first_only == 2).all()
    # g2 9000 with 9001 always; g1 1000 with 1002 only when that molecule is drawn
    assert set(d.shared) == {1, 2}
    pd.testing.assert_frame_equal(d, downsample_sets(a, b, ["g1", "g2"], draws=40, seed=1))
    assert not d.equals(downsample_sets(a, b, ["g1", "g2"], draws=40, seed=2))
    # only the genes given count
    d1 = downsample_sets(a, b, ["g2"], draws=3, seed=1)
    assert d1[["shared", "first_only", "second_only"]].values.tolist() == [[1, 0, 1]] * 3
    assert len(downsample_sets(a, b, ["g1"])) == DOWNSAMPLE_DRAWS


def test_set_counts_and_shares(tmp_path):
    a, b = _pair_tables(tmp_path)
    c = set_counts(site_sets(a, b, ["gP", "gQ"]))
    assert c == {"shared": {"sites": 2, "molecules_first": 6, "molecules_second": 8},
                 "first_only": {"sites": 1, "molecules_first": 1, "molecules_second": 0},
                 "second_only": {"sites": 1, "molecules_first": 0, "molecules_second": 2},
                 "shares": {"all": 0.5, "first": 2 / 3, "second": 2 / 3}}


def test_downsampling_summary():
    d = pd.DataFrame({"draw": range(4), "molecules": 9, "shared": [1, 2, 3, 2], "first_only": [3, 2, 1, 2],
                      "second_only": [0, 2, 1, 6]})
    s = downsampling_summary(d)
    assert list(s) == ["draws", "molecules", "shared", "first_only", "second_only", "share_all", "share_first", "share_second"]
    assert s["draws"] == 4 and s["molecules"] == 9
    assert s["shared"] == {"mean": 2.0, "lo": pytest.approx(np.percentile([1, 2, 3, 2], 2.5)),
                           "hi": pytest.approx(np.percentile([1, 2, 3, 2], 97.5))}
    share_all = np.array([1 / 4, 2 / 6, 3 / 5, 2 / 10])
    assert s["share_all"]["mean"] == pytest.approx(share_all.mean())
    assert s["share_first"]["mean"] == pytest.approx(0.5) and s["share_second"]["lo"] == pytest.approx(np.percentile(
        [1, 0.5, 0.75, 0.25], 2.5))

def test_same_site_check_reproduces_same_site(tmp_path):
    gtf = _gtf(tmp_path)
    exons, ends = load_collapsed_exons(gtf), gene_ends(gtf)
    ma = _mols([("chr1", "+", 1050, "gP", 2, 12, "CDS"), ("chr1", "+", 1150, "gP", 1, 2, "intron"),
                ("chr1", "+", 1300, "gP", 1, 6, "3UTR"), ("chr2", "+", 900, "", 3, 9, "intergenic")])
    mb = _mols([("chr1", "+", 1048, "gP", 5, 12, "CDS"), ("chr1", "+", 1053, "gP", 1, 12, "CDS"),
                ("chr1", "+", 1306, "gP", 1, 6, "3UTR"), ("chr2", "+", 905, "", 1, 2, "intergenic")])
    expected = same_site(ma, mb)
    sa, sb = site_table(ma, exons, ends), site_table(mb, exons, ends)
    c = same_site_check(sa, sb, expected)
    # one way: 1050 to 1048 (1053 is farther), 900 to 905 at 5 nt; 1150 and 1300 (6 nt from 1306) unmatched
    assert c == {"sites_a": 4, "matched": 2, "expected_sites_a": 4, "expected_matched": 2, "label_pairs_equal": True,
                 "pass": True}
    wrong = expected.assign(n=expected.n + (expected.label_b == "none"))
    c = same_site_check(sa, sb, wrong)
    assert c["pass"] is False and c["label_pairs_equal"] is False and c["expected_sites_a"] == 6
    swapped = expected.assign(label_b=expected.label_b.replace({"templated": "ambiguous"}))
    assert same_site_check(sa, sb, swapped)["label_pairs_equal"] is False

