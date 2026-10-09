import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest

from pafbench.molecules import MOL_COLUMNS, consolidate_sites
from pafbench.regions import load_region_segments, regions_at
from pafbench.sites import last_aligned_base, priming_site
from pafbench.truth import check_pa_counts, label_truth_molecules, truth_reads, truth_summary

REPO = Path(__file__).resolve().parents[1]

GTF_ROWS = [
    # tc on +: exons 101-200, 301-400, 501-600 (1-based), CDS 151-350; 0-based 199 CDS / 200 intron, 599 3UTR / 600 intergenic
    ("transcript", 101, 600, "+", "tc"), ("exon", 101, 200, "+", "tc"), ("exon", 301, 400, "+", "tc"),
    ("exon", 501, 600, "+", "tc"), ("CDS", 151, 200, "+", "tc"), ("CDS", 301, 350, "+", "tc"),
    # tm on -: exons 1001-1100, 1201-1300, CDS 1051-1250; 0-based 999 intergenic / 1000 3UTR, 1299 5UTR / 1300 intergenic
    ("transcript", 1001, 1300, "-", "tm"), ("exon", 1001, 1100, "-", "tm"), ("exon", 1201, 1300, "-", "tm"),
    ("CDS", 1051, 1100, "-", "tm"), ("CDS", 1201, 1250, "-", "tm"),
]

PA_COLS = ["qname", "cb", "ub", "strand", "ref", "site", "ts", "spliced", "genes", "a20", "down20", "up40", "hexamer", "re", "mapq"]
# Molecules after consolidate_sites (reads r8 without UB and E with spread 20 drop out): A=0, B=1, C=2, D=3, F=4.
PA_ROWS = [
    ["r1", "c1", "u1", "+", "chrZ", 600, 0, False, "g_tc", 15, "A" * 20, "C" * 40, False, "E", 255],          # A
    ["r2", "c1", "u1", "+", "chrZ", 600, 0, False, "g_tc", 15, "A" * 20, "C" * 40, False, "E", 3],            # A
    ["r3", "c1", "u2", "+", "chrZ", 200, 0, False, "g_tc", 3, "C" * 20, "C" * 40, True, "E", 255],            # B
    ["r4", "c2", "u1", "-", "chrZ", 1000, 2, True, "g_tm;g_x", 6, "C" * 20, "C" * 40, True, "E", 255],       # C
    ["r5", "c2", "u3", "-", "chrZ", 1300, 0, False, "", 9, "C" * 20, "C" * 40, False, "N", 255],              # D
    ["r6", "c3", "u1", "+", "chrZ", 300, 4, False, "g_tc", 15, "A" * 20, "C" * 40, False, "E", 255],          # E
    ["r7", "c3", "u1", "+", "chrZ", 320, 4, False, "g_tc", 15, "A" * 20, "C" * 40, False, "E", 255],          # E
    ["r8", "c1", None, "+", "chrZ", 600, 9, False, "g_tc", 15, "A" * 20, "C" * 40, False, "E", 255],          # no UB
    ["r9", "c1", "u1", "+", "chrZ", 560, 7, False, "g_tn", 12, "A" * 20, "C" * 40, False, "E", 255],          # F
]

MEMBER_COLS = ["mol", "qname", "dist5", "dist3", "spliced", "dup", "genes", "gene_match", "dist5_exon", "dist3_exon", "dist_mode"]
# dist5 = dist5_exon + 7 so a function reading the genomic distance gives other answers
MEMBER_ROWS = [
    [0, "m1", 307, 0, False, True, "g_tc", True, 300, 0, "exon"],         # kept at 1000 and 2000
    [0, "m2", 107, 0, False, True, "g_zz", False, 100, 0, "genome"],      # gene mismatch
    [0, "m3", 107, -26, False, True, "g_tc", True, 100, -26, "genome"],   # passes the site by 26
    [0, "m4", 1007, -25, False, True, "g_tc", True, 1000, -25, "genome"], # boundary: kept at both caps
    [0, "m5", 1008, 0, False, True, "g_tc", True, 1001, 0, "exon"],       # only at 2000
    [0, "m6", 2007, 5, False, True, "g_tc", True, 2000, 5, "exon"],       # only at 2000
    [0, "m7", 2008, 5, False, True, "g_tc", True, 2001, 5, "exon"],       # beyond 2000
    [2, "m8", 57, 10, True, False, "g_tm", True, 50, 10, "exon"],         # C
    [4, "m1", 347, 0, False, True, "g_tc", False, 340, 0, "exon"],        # same read, F's gene does not match
]


@pytest.fixture
def data(tmp_path):
    gtf = tmp_path / "t.gtf"
    with open(gtf, "w") as fh:
        for ft, s, e, strand, tid in GTF_ROWS:
            fh.write(f"chrZ\tt\t{ft}\t{s}\t{e}\t.\t{strand}\t.\tgene_id \"g_{tid}\"; transcript_id \"{tid}\";\n")
    pa = pd.DataFrame(PA_ROWS, columns=PA_COLS)
    mol = consolidate_sites(pa)
    members = pd.DataFrame(MEMBER_ROWS, columns=MEMBER_COLS)
    return mol, pa, members, load_region_segments(gtf)


def test_fixture_molecules(data):
    mol = data[0]
    # E (sites 300 and 320, spread 20) is dropped and r8 has no UB
    assert mol.mol.tolist() == [0, 1, 2, 3, 4]
    assert mol.site.tolist() == [600, 200, 1000, 1300, 560] and mol.n_pa.tolist() == [2, 1, 1, 1, 1]


def test_label_columns_and_order(data):
    mol, pa, members, seg = data
    out = label_truth_molecules(mol, pa, members, seg)
    assert out.columns.tolist() == MOL_COLUMNS + ["label", "a_band", "region", "gene", "has_ts", "n_members"]
    pd.testing.assert_frame_equal(out[MOL_COLUMNS], mol)
    # a20 15, 3, 6, 9, 12 at cut-offs 8/4
    assert out.label.tolist() == ["templated", "nontemplated", "ambiguous", "templated", "templated"]
    assert out.a_band.tolist() == ["14-20", "0-4", "5-7", "8-9", "10-13"]
    strict = label_truth_molecules(mol, pa, members, seg, {"templated_min": 14, "nontemplated_max": 8})
    assert strict.label.tolist() == ["templated", "nontemplated", "nontemplated", "ambiguous", "ambiguous"]


def test_region_at_last_cdna_base(data):
    mol, pa, members, seg = data
    out = label_truth_molecules(mol, pa, members, seg)
    # A + site 600 -> base 599 (3UTR; 600 is intergenic); B + site 200 -> 199 (CDS; 200 is intron)
    # C - site 1000 -> 1000 (3UTR; 999 is intergenic); D - site 1300 -> 1300 (intergenic; 1299 is 5UTR); F + 560 -> 559
    assert out.region.tolist() == ["3UTR", "CDS", "3UTR", "intergenic", "3UTR"]
    assert regions_at(seg, "chrZ", True, [600, 200]).tolist() == ["intergenic", "intron"]
    assert regions_at(seg, "chrZ", False, [999, 1299]).tolist() == ["intergenic", "5UTR"]


def test_region_matches_last_aligned_base_of_a_read(data):
    """Independent reference: the region at sites.last_aligned_base of real alignments ending at the molecule site."""
    mol, pa, members, seg = data
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "chrZ", "LN": 6000}]})
    expected = []
    for start, flag in ((570, 0), (170, 0), (1000, 16), (1300, 16), (530, 0)):
        a = pysam.AlignedSegment(hdr)
        a.reference_name, a.reference_start, a.cigarstring, a.flag = "chrZ", start, "30M", flag
        a.query_sequence = "C" * 30
        expected.append((priming_site(a), regions_at(seg, "chrZ", not a.is_reverse, [last_aligned_base(a)])[0]))
    out = label_truth_molecules(mol, pa, members, seg)
    assert list(zip(out.site, out.region)) == expected


def test_gene_has_ts_n_members(data):
    mol, pa, members, seg = data
    out = label_truth_molecules(mol, pa, members, seg)
    assert out.gene.tolist() == ["g_tc", "g_tc", "g_tm", "", "g_tn"]
    # A's reads have ts 0; r9 (ts 7) shares A's CB, UB, strand, ref but not its gene, so only F; r8 (ts 9) has no UB
    assert out.has_ts.tolist() == [False, False, True, False, True]
    # A: m1, m4 (m2 gene, m3 dist3 -26, m5..m7 beyond 1000); C: m8; F: m1 row has gene_match False
    assert out.n_members.tolist() == [2, 0, 1, 0, 0]
    assert out.n_members.dtype == np.int64 and out.has_ts.dtype == bool


def test_truth_reads(data):
    mol, pa, members, seg = data
    reads = truth_reads(mol, pa, members)
    expected = pd.DataFrame({
        "qname": ["r1", "r2", "r3", "r4", "r5", "r9", "m1", "m4", "m5", "m6", "m8"],
        "mol": np.array([0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 2], np.int64),
        "role": ["pa"] * 6 + ["member"] * 5,
        "dist5_exon": [np.nan] * 6 + [300.0, 1000.0, 1001.0, 2000.0, 50.0],
        "dist_mode": [""] * 6 + ["exon", "genome", "exon", "exon", "exon"],
        "spliced": [False, False, False, True, False, False, False, False, False, False, True],
    })
    pd.testing.assert_frame_equal(reads, expected)


def test_truth_reads_cap_1000(data):
    mol, pa, members, seg = data
    reads = truth_reads(mol, pa, members, cap=1000)
    m = reads[reads.role == "member"]
    assert m.qname.tolist() == ["m1", "m4", "m8"] and m.dist5_exon.tolist() == [300.0, 1000.0, 50.0]
    assert reads[reads.role == "pa"].qname.tolist() == ["r1", "r2", "r3", "r4", "r5", "r9"]


def test_pa_count_check(data):
    mol, pa, members, seg = data
    reads = truth_reads(mol, pa, members)
    check_pa_counts(mol, reads)
    with pytest.raises(ValueError, match="n_pa"):
        check_pa_counts(mol, reads.drop(index=1))
    # B loses its only pa read: zero reads for a molecule with n_pa 1
    with pytest.raises(ValueError, match="n_pa"):
        check_pa_counts(mol, truth_reads(mol, pa[pa.qname != "r3"], members))


def test_truth_summary(data):
    mol, pa, members, seg = data
    labeled = label_truth_molecules(mol, pa, members, seg)
    s = truth_summary(labeled, truth_reads(mol, pa, members))
    assert s == {
        "molecules": 5,
        "labels": {"templated": 3, "nontemplated": 1, "ambiguous": 1},
        "regions": {"intergenic": 1, "intron": 0, "ncexon": 0, "5UTR": 0, "CDS": 1, "3UTR": 3, "antisense": 0},
        "has_ts": 2,
        "truth_reads": {"pa": 6, "member": 5},
        # member dist5_exon 300, 1000, 1001, 2000, 50
        "members_within_cap": {"500": 2, "1000": 3, "2000": 5},
        # A (templated) and C (ambiguous) have members at 1000
        "molecules_with_members": {"templated": 1, "nontemplated": 0, "ambiguous": 1},
    }


def test_script_refuses_old_molecules(tmp_path):
    base = tmp_path / "lib"
    base.mkdir()
    pa = pd.DataFrame(PA_ROWS, columns=PA_COLS)
    pa.to_parquet(base / "pa_sites.parquet")
    consolidate_sites(pa).drop(columns="mapq_min").to_parquet(base / "molecules.parquet")
    pd.DataFrame(MEMBER_ROWS, columns=MEMBER_COLS).to_parquet(base / "members.parquet")
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "12_truth_sets.py"), "X1", "--base", str(base),
                        "--out", str(tmp_path / "out")], env=env, capture_output=True, text=True)
    assert r.returncode == 1
    assert "molecules.parquet predates milestone 2; rerun scripts/06_molecules.py X1" in r.stderr
    assert not (tmp_path / "out").exists()


# Cases the main fixture does not separate. After consolidate_sites: G=0 (s1, s2), H=1, I=2, J=3, K=4, L=5.
PA_EXTRA = [
    # G: two pa reads, the first with ts 5 and the last with ts 0, so has_ts is True ("any read", not "all" or "last")
    ["s1", "c5", "u5", "+", "chrZ", 560, 5, False, "g_tc", 15, "A" * 20, "C" * 40, False, "E", 255],
    ["s2", "c5", "u5", "+", "chrZ", 560, 0, False, "g_tc", 15, "A" * 20, "C" * 40, False, "E", 255],
    # H, I: same CB, UB and genes "" on opposite strands; J, K: same CB, UB, strand and genes "" on two references
    ["s3", "c6", "u6", "+", "chrZ", 3000, 0, False, "", 15, "A" * 20, "C" * 40, False, "N", 255],
    ["s4", "c6", "u6", "-", "chrZ", 3500, 0, False, "", 15, "A" * 20, "C" * 40, False, "N", 255],
    ["s5", "c7", "u7", "+", "chrZ", 4000, 0, False, "", 15, "A" * 20, "C" * 40, False, "N", 255],
    ["s6", "c7", "u7", "+", "chrY", 4000, 0, False, "", 15, "A" * 20, "C" * 40, False, "N", 255],
    # L: minus-strand site 1099, last cDNA base 1099 = CDS (tm CDS 1050..1099 0-based); 1100 is intron, 1098 CDS
    ["s7", "c8", "u8", "-", "chrZ", 1099, 0, False, "g_tm", 3, "C" * 20, "C" * 40, True, "E", 255],
]


def test_has_ts_join_key_and_minus_boundary(data):
    seg = data[3]
    pa = pd.DataFrame(PA_EXTRA, columns=PA_COLS)
    mol = consolidate_sites(pa)
    assert mol.mol.tolist() == [0, 1, 2, 3, 4, 5] and mol.n_pa.tolist() == [2, 1, 1, 1, 1, 1]
    members = pd.DataFrame(MEMBER_ROWS, columns=MEMBER_COLS).iloc[:0]
    out = label_truth_molecules(mol, pa, members, seg)
    assert out.has_ts.tolist() == [True, False, False, False, False, False]
    assert out.region.tolist() == ["3UTR", "intergenic", "intergenic", "intergenic", "intergenic", "CDS"]
    assert regions_at(seg, "chrZ", False, [1098, 1099, 1100]).tolist() == ["CDS", "CDS", "intron"]
    reads = truth_reads(mol, pa, members)
    # each pa read joins its own molecule only: the key holds strand and ref
    assert reads.qname.tolist() == ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    assert reads.mol.tolist() == [0, 0, 1, 2, 3, 4, 5] and (reads.role == "pa").all()
    check_pa_counts(mol, reads)
    # a duplicated pa row gives L n_pa + 1 rows
    with pytest.raises(ValueError, match="1 molecules do not receive n_pa pa reads \\(first mol 5\\)"):
        check_pa_counts(mol, pd.concat([reads, reads[reads.qname == "s7"]], ignore_index=True))


# Runs a stage script with pafbench.config.GTF pointed at a test GTF: GTF SCRIPT ARGS...
GTF_RUNNER = """
import runpy, sys
from pathlib import Path
import pafbench.config as c
c.GTF = Path(sys.argv[1])
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def _stage12(tmp_path, mol, *extra):
    base = tmp_path / "lib"
    base.mkdir()
    pa = pd.DataFrame(PA_ROWS, columns=PA_COLS)
    pa.to_parquet(base / "pa_sites.parquet")
    mol.to_parquet(base / "molecules.parquet")
    pd.DataFrame(MEMBER_ROWS, columns=MEMBER_COLS).to_parquet(base / "members.parquet")
    gtf = tmp_path / "t.gtf"
    with open(gtf, "w") as fh:
        for ft, s, e, strand, tid in GTF_ROWS:
            fh.write(f"chrZ\tt\t{ft}\t{s}\t{e}\t.\t{strand}\t.\tgene_id \"g_{tid}\"; transcript_id \"{tid}\";\n")
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, "-c", GTF_RUNNER, str(gtf), str(REPO / "scripts" / "12_truth_sets.py"), "X1",
                        "--base", str(base), *extra], env=env, capture_output=True, text=True)
    return base, r


def test_script_writes_truth_sets_under_base(tmp_path, data):
    mol, pa, members, seg = data
    base, r = _stage12(tmp_path, mol)
    assert r.returncode == 0, r.stderr
    out = base / "truth"  # --base without --out writes to <base>/truth
    labeled = label_truth_molecules(mol, pa, members, seg)
    reads = truth_reads(mol, pa, members)
    pd.testing.assert_frame_equal(pd.read_parquet(out / "molecules_labeled.parquet"), labeled)
    pd.testing.assert_frame_equal(pd.read_parquet(out / "truth_reads.parquet"), reads)
    curve = pd.read_csv(out / "hexamer_curve.tsv", sep="\t")
    assert curve.n.sum() == 5 and curve.set_index("a20").n[[3, 6, 9, 12, 15]].tolist() == [1, 1, 1, 1, 1]
    assert r.stdout.strip().splitlines() == [json.dumps({"library": "X1"} | truth_summary(labeled, reads))]


def test_script_checks_pa_counts_before_writing(tmp_path, data):
    mol = data[0]
    base, r = _stage12(tmp_path, mol.assign(n_pa=mol.n_pa + (mol.mol == 1)), "--out", str(tmp_path / "out"))
    assert r.returncode == 1 and "1 molecules do not receive n_pa pa reads (first mol 1)" in r.stderr
    assert not (tmp_path / "out").exists()
