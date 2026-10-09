import numpy as np
import pandas as pd
import pysam
import pytest

from pafbench.bd67 import terminal_a_run
from pafbench.flags import Setting, setting_tag
from pafbench.genes import load_gene_segments
from pafbench.sites import COMP, PA_COLUMNS, has_hexamer, pa_row, priming_site
from pafbench.spikein import (add_spike_labels, annotated_ends, label_spike_site, merge_scans, spike_pa_row,
                              spike_summary)
from pafbench.stream import merge_parts, scan_reference, scan_to_parts
from pafbench.transcripts import load_transcripts

ENDS_GTF = [
    ("X", "gene", 101, 650, "+", 'gene_id "gX";'),
    ("X", "transcript", 101, 500, "+", 'gene_id "gX"; transcript_id "tx1";'),
    ("X", "exon", 101, 200, "+", 'gene_id "gX"; transcript_id "tx1";'),
    ("X", "exon", 301, 500, "+", 'gene_id "gX"; transcript_id "tx1";'),
    ("X", "transcript", 101, 650, "+", 'gene_id "gX"; transcript_id "tx2";'),
    ("X", "transcript", 300, 650, "+", 'gene_id "gX"; transcript_id "tx3";'),
    ("X", "transcript", 701, 900, "-", 'gene_id "gX"; transcript_id "tx4";'),
    ("X", "exon", 701, 900, "-", 'gene_id "gX"; transcript_id "tx4";'),
    ("X", "transcript", 1001, 1100, "-", 'gene_id "gX"; transcript_id "tx5";'),
    ("Y", "gene", 1, 1045, "+", 'gene_id "gY";'),
    ("Y", "transcript", 1, 1045, "+", 'gene_id "gY"; transcript_id "ty1";'),
]


def _write_gtf(path, rows):
    with open(path, "w") as fh:
        for c, ft, s, e, strand, attr in rows:
            fh.write(f"{c}\tt\t{ft}\t{s}\t{e}\t.\t{strand}\t.\t{attr}\n")
    return path


def test_annotated_ends(tmp_path):
    ends = annotated_ends(_write_gtf(tmp_path / "e.gtf", ENDS_GTF))
    # plus: GTF end (tx1 500, tx2 and tx3 both 650); minus: GTF start - 1 (tx4 700, tx5 1000); exon end 200 and gene lines ignored
    assert sorted(ends) == [("X", False), ("X", True), ("Y", True)]
    assert ends[("X", True)].tolist() == [500, 650] and ends[("X", True)].dtype == np.int64
    assert ends[("X", False)].tolist() == [700, 1000]
    assert ends[("Y", True)].tolist() == [1045]


def test_annotated_ends_match_priming_site(tmp_path):
    # a read whose last cDNA base is the transcript's last base has priming_site == the annotated end (d = 0)
    ends = annotated_ends(_write_gtf(tmp_path / "e.gtf", ENDS_GTF))
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "X", "LN": 2000}]})
    fwd, rev = pysam.AlignedSegment(hdr), pysam.AlignedSegment(hdr)
    fwd.reference_name, fwd.reference_start, fwd.cigarstring, fwd.flag = "X", 450, "50M", 0  # covers GTF 451..500
    rev.reference_name, rev.reference_start, rev.cigarstring, rev.flag = "X", 700, "50M", 16  # covers GTF 701..750
    assert label_spike_site(priming_site(fwd), True, ends[("X", True)]) == ("tail", 0)
    assert label_spike_site(priming_site(rev), False, ends[("X", False)]) == ("tail", 0)


@pytest.mark.parametrize("site,expected", [
    (1010, ("tail", -10)),       # d = 1000 - 1010
    (990, ("tail", 10)),
    (989, ("other", 11)),
    (971, ("other", 29)),
    (970, ("internal", 30)),
    (1011, ("other", -11)),
])
def test_label_spike_site_plus(site, expected):
    assert label_spike_site(site, True, np.array([1000], np.int64)) == expected


@pytest.mark.parametrize("site,expected", [
    (530, ("internal", 30)),     # d = site - end on minus: upstream in transcript sense is a larger coordinate
    (529, ("other", 29)),
    (511, ("other", 11)),
    (510, ("tail", 10)),
    (490, ("tail", -10)),
    (489, ("other", -11)),
])
def test_label_spike_site_minus(site, expected):
    assert label_spike_site(site, False, np.array([500], np.int64)) == expected


def test_label_spike_site_several_ends():
    plus = np.array([1000, 1100], np.int64)
    assert label_spike_site(1050, True, plus) == ("internal", 50)   # d -50 and 50: tie on |d|, the positive one
    assert label_spike_site(1105, True, plus) == ("tail", -5)       # d -105, -5
    assert label_spike_site(1200, True, plus) == ("other", -100)    # d -200, -100: past every end
    assert label_spike_site(995, True, plus) == ("tail", 5)         # d 5 and 105: tail wins over internal
    assert label_spike_site(550, False, np.array([500, 600], np.int64)) == ("internal", 50)  # d 50, -50
    assert label_spike_site(550, True, None) == ("other", None)
    assert label_spike_site(550, True, np.array([], np.int64)) == ("other", None)
    assert label_spike_site(1050, True, plus, tail_tol=50) == ("tail", 50)
    assert label_spike_site(1050, True, plus, internal_min=51) == ("other", 50)


# Contig S1 (200 nt) on a background with no A or T; planted runs give each test read a known terminal A run.
def _s1():
    s = list("CG" * 100)
    s[63:69] = "AATAAA"          # hexamer for the plus run read: in the 5..35 nt window of site 98, not of 104
    s[95:105] = "CATAGAAAAC"     # 95 C, 96 A, 97 T, 98 A, 99 G, 100..103 A, 104 C
    s[150:158] = "TTTTCTGG"      # sense on minus (genomic 157 down to 150): C C A G A A A A
    s[184:190] = "TTTATT"        # sense AATAAA for the minus run read: in the window of site 156, not of 150
    return "".join(s)


S1 = _s1()
SEG_GTF = [
    ("S1", "gene", 1, 98, "+", 'gene_id "gp1";'),      # [0, 98)
    ("S1", "gene", 99, 200, "+", 'gene_id "gp2";'),    # [98, 200)
    ("S1", "gene", 1, 156, "-", 'gene_id "gm1";'),     # [0, 156)
    ("S1", "gene", 157, 200, "-", 'gene_id "gm2";'),   # [156, 200)
]


@pytest.fixture
def s1(tmp_path):
    fasta = tmp_path / "s1.fa"
    fasta.write_text(f">S1\n{S1}\n")
    pysam.faidx(str(fasta))
    seg = load_gene_segments(_write_gtf(tmp_path / "s1.gtf", SEG_GTF))
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "S1", "LN": 200}]})
    with pysam.FastaFile(str(fasta)) as fa:
        yield fa, seg, hdr


def _rec(hdr, name, pos, cigar, seq, rev, tags):
    a = pysam.AlignedSegment(hdr)
    a.query_name, a.reference_name, a.reference_start, a.cigarstring = name, "S1", pos, cigar
    a.flag, a.mapping_quality, a.query_sequence = 16 if rev else 0, 255, seq
    a.set_tags(tags)
    return a


def _revcomp(s):
    return s.translate(COMP)[::-1]


def test_spike_pa_row_plus_run(s1):
    fa, seg, hdr = s1
    # aligned 60..103, then 10 clipped A's; the run takes 103..100 (A), 99 (G, the one allowed non-A, A at 98 after it)
    # and 98 (A); 97 is a second single non-A (T, A at 96), so the run starts at 98. A budget of two non-A's would take
    # 97 and 96 and stop at 95 (C, C at 94): site 96
    a = _rec(hdr, "p_run", 60, "44M10S", S1[60:104] + "A" * 10, False,
             [("pa", 10), ("ts", 5), ("CB", "c1"), ("UB", "u1"), ("xf", 25)])
    row = dict(zip(PA_COLUMNS, spike_pa_row(a, fa, seg)))
    assert row["site"] == 98 and row["strand"] == "+"
    assert terminal_a_run(a, tolerance=2)[0] == 96
    assert row["down20"] == S1[98:118] and row["up40"] == S1[58:98]
    assert row["a20"] == 5                       # 98 and 100..103
    assert row["hexamer"] is True                # AATAAA at 63..68 lies in up40[5:35] = S1[63:93]
    assert row["genes"] == "gp1"                 # last cDNA base 97
    assert (row["pa"], row["ts"], row["cb"], row["ub"], row["gx"], row["re"], row["xf"]) == (10, 5, "c1", "u1", None, None, 25)
    assert (row["flag"], row["dup"], row["mapq"], row["aligned_len"], row["spliced"]) == (0, False, 255, 44, False)
    # pa_row keeps the alignment end: site 104, no A downstream, hexamer outside its window, gene gp2 at 103
    old = dict(zip(PA_COLUMNS, pa_row(a, fa, seg)))
    assert (old["site"], old["a20"], old["hexamer"], old["genes"]) == (104, 0, False, "gp2")
    assert {k for k in PA_COLUMNS if row[k] != old[k]} == {"site", "down20", "up40", "a20", "hexamer", "genes"}


def test_spike_pa_row_minus_run(s1):
    fa, seg, hdr = s1
    # reverse read aligned 150..193 with 10 clipped T's on the left (sense A's); sense run over 150..153 (A),
    # 154 (G, then A at 155) and 155 (A); 156 is sense C with sense C at 157, so the run starts at 156
    a = _rec(hdr, "m_run", 150, "10S44M", "T" * 10 + S1[150:194], True, [("pa", 10), ("CB", "c2"), ("UB", "u2")])
    row = dict(zip(PA_COLUMNS, spike_pa_row(a, fa, seg)))
    assert row["site"] == 156 and row["strand"] == "-"
    assert row["down20"] == _revcomp(S1[136:156]) and row["up40"] == _revcomp(S1[156:196])
    assert row["a20"] == 5                       # genomic T's at 150..153 and 155
    assert row["hexamer"] is True                # TTTATT at 184..189 is sense up40[6:12]
    assert row["genes"] == "gm2"                 # last cDNA base 156
    assert (row["ts"], row["xf"]) == (0, None)
    old = dict(zip(PA_COLUMNS, pa_row(a, fa, seg)))
    assert (old["site"], old["a20"], old["hexamer"], old["genes"]) == (150, 0, False, "gm1")


def test_spike_pa_row_without_aligned_run_equals_pa_row(s1):
    fa, seg, hdr = s1
    tags = [("pa", 8), ("CB", "c3"), ("UB", "u3"), ("ts", 2)]
    fwd = _rec(hdr, "p_norun", 20, "30M8S", S1[20:50] + "A" * 8, False, tags)      # 49 G, 48 C: no aligned A run
    rev = _rec(hdr, "m_norun", 120, "8S20M", "T" * 8 + S1[120:140], True, tags)    # 120 C, 121 G: no aligned T run
    for a, site in ((fwd, 50), (rev, 120)):
        row = spike_pa_row(a, fa, seg)
        assert row == pa_row(a, fa, seg)
        assert row[PA_COLUMNS.index("site")] == site


def test_add_spike_labels():
    ends = {("ERCC-00002", True): np.array([1045], np.int64), ("SIRV1", False): np.array([1000, 1006], np.int64),
            ("SIRV4001", True): np.array([3997], np.int64)}
    mol = pd.DataFrame({"mol": [0, 1, 2, 3, 4], "ref": ["ERCC-00002", "ERCC-00002", "SIRV1", "SIRV1", "SIRV4001"],
                        "strand": ["+", "-", "-", "+", "+"], "site": [1040, 800, 1040, 1040, 2000]})
    out = add_spike_labels(mol, ends)
    assert out.columns.tolist() == ["mol", "ref", "strand", "site", "spike_label", "dist_to_end", "module"]
    pd.testing.assert_frame_equal(out[mol.columns], mol)
    # 1045 - 1040 = 5 tail; no minus ends on ERCC-00002; minus d = 1040 - 1006 = 34 internal; no plus ends on SIRV1;
    # 3997 - 2000 = 1997 internal
    assert out.spike_label.tolist() == ["tail", "other", "internal", "other", "internal"]
    assert out.dist_to_end.tolist()[:1] == [5.0] and np.isnan(out.dist_to_end.iloc[1])
    assert out.dist_to_end.iloc[2] == 34.0 and np.isnan(out.dist_to_end.iloc[3]) and out.dist_to_end.iloc[4] == 1997.0
    assert out.dist_to_end.dtype == np.float64
    assert out.module.tolist() == ["ERCC", "ERCC", "SIRV_E0", "SIRV_E0", "long_SIRV"]


def test_merge_scans_equals_merge_parts(synth, tmp_path):
    tx = load_transcripts(synth.gtf)
    settings = [Setting(300, 10, 1, True), Setting(150, 5, 0, False), Setting(150, None, 0, True), Setting(450, 3, 2, False)]
    tags = [setting_tag(s) for s in settings]
    with pysam.AlignmentFile(str(synth.bam)) as fh:
        names = sorted({a.query_name for a in fh.fetch(until_eof=True)})
    truth = set(names[::3])
    refs = ["chrA", "chrB"]
    scans = []
    for ref in refs:
        t, r, _ = scan_reference(synth.bam, synth.fasta, ref, tx, settings, truth)
        scans.append((t, r))
        scan_to_parts(synth.bam, synth.fasta, ref, tx, settings, truth, None, tmp_path / "parts")
    truth_df, region_df = merge_scans(scans, tags)
    merge_parts(tmp_path / "parts", refs, tags, tmp_path / "merged")
    truth_df.to_parquet(tmp_path / "truth_flags.parquet", index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "truth_flags.parquet"),
                                  pd.read_parquet(tmp_path / "merged" / "truth_flags.parquet"))
    pd.testing.assert_frame_equal(region_df, pd.read_parquet(tmp_path / "merged" / "region_counts.parquet"))
    assert len(truth_df) == sum(len(t) for t, _ in scans) > 10
    assert truth_df.ref.tolist() == ["chrA"] * len(scans[0][0]) + ["chrB"] * len(scans[1][0])
    assert region_df.tag.tolist() == tags and set(region_df.re) == {""}
    assert (region_df.total == sum(int(r.total.iloc[0]) for _, r in scans)).all()


def test_spike_summary():
    pa = pd.DataFrame({"qname": list("abcdefg")})
    labeled = pd.DataFrame({"mol": range(5), "module": ["ERCC", "ERCC", "SIRV_E0", "long_SIRV", "SIRV_E0"],
                            "spike_label": ["tail", "internal", "tail", "other", "internal"],
                            "label": ["nontemplated", "templated", "ambiguous", "templated", "templated"]})
    members = pd.DataFrame({"mol": [0, 0, 1, 2]})
    reads = pd.DataFrame({"role": ["pa"] * 6 + ["member"] * 3})
    assert spike_summary(pa, labeled, members, reads) == {
        "pa_reads": 7, "molecules": 5,
        "by_module": {"ERCC": {"tail": 1, "internal": 1, "other": 0}, "SIRV_E0": {"tail": 1, "internal": 1, "other": 0},
                      "long_SIRV": {"tail": 0, "internal": 0, "other": 1}},
        "spike_label_by_label": {"tail": {"templated": 0, "nontemplated": 1, "ambiguous": 1},
                                 "internal": {"templated": 2, "nontemplated": 0, "ambiguous": 0},
                                 "other": {"templated": 1, "nontemplated": 0, "ambiguous": 0}},
        "members": 4, "truth_reads": {"pa": 6, "member": 3}}


def test_merge_scans_order_with_several_re_values():
    def region(rows):
        return pd.DataFrame(rows, columns=["tag", "re", "removed", "total"])
    # contig 1 has E and N, contig 2 has I and E; first appearance order would be t1E t1N t2E t2N t1I t2I
    r1 = region([("t1", "E", 1, 10), ("t1", "N", 2, 20), ("t2", "E", 3, 10), ("t2", "N", 4, 20)])
    r2 = region([("t1", "E", 5, 7), ("t1", "I", 6, 9), ("t2", "E", 0, 7), ("t2", "I", 1, 9)])
    t1 = pd.DataFrame({"qname": ["a"], "ref": ["x"], "pos": [1], "flag": [0], "t1": [True], "t2": [False]})
    t2 = pd.DataFrame({"qname": ["b", "c"], "ref": ["y", "y"], "pos": [5, 9], "flag": [16, 0], "t1": [False, True],
                       "t2": [True, True]})
    truth, reg = merge_scans([(t1, r1), (t2, r2)], ["t1", "t2"])
    assert truth.qname.tolist() == ["a", "b", "c"] and truth.index.tolist() == [0, 1, 2]
    assert list(reg.itertuples(index=False, name=None)) == [
        ("t1", "E", 6, 17), ("t1", "I", 6, 9), ("t1", "N", 2, 20), ("t2", "E", 3, 17), ("t2", "I", 1, 9), ("t2", "N", 4, 20)]
