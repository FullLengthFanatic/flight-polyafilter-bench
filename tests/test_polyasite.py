import gzip

import numpy as np
import pandas as pd
import pysam

from pafbench.polyasite import BINS, distance_bin, load_polyasite, nearest_signed, site_distances
from pafbench.sites import priming_site
from pafbench.spikein import label_spike_site

# PolyASite 3.0 BED rows: chrom, start, end, id chrom:pos:strand (pos 1-based), mean RPM, strand, % tissues, protocols,
# stringency, class, signals. Out of order on purpose. chr2 has a plus site only and chrX a minus site only; chr3, which
# test_site_distances_per_ref_and_strand also uses, is absent.
BED = [
    ("chr1", 1999, 2000, "chr1:2000:+", 2.0, "+", 50.0, 1, 98, "TE", "AUUAAA;AAUAAA"),
    ("chr1", 99, 100, "chr1:100:+", 0.5, "+", 11.1, 1, 47, "TE", "AAUAAA"),
    ("chr1", 480, 520, "chr1:500:+", 0.01, "+", 5.6, 1, 23, "IN", "NaN"),  # cluster span: the id gives the cleavage site
    ("chr1", 199, 200, "chr1:200:-", 0.2, "-", 5.6, 1, 26, "DI", "NaN"),
    ("chr1", 690, 720, "chr1:700:-", 0.001, "-", 5.6, 1, 20, "IN", "NaN"),
    ("chr2", 49, 50, "chr2:50:+", 1.0, "+", 22.2, 1, 36, "UI", "NaN"),
    ("chrX", 9, 10, "chrX:10:-", 0.05, "-", 5.6, 1, 21, "DI", "NaN"),
]


def _bed(path, rows=BED):
    with gzip.open(path, "wt") as fh:
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    return path


def _eq(got, want):
    assert sorted(got) == sorted(want)
    for k in want:
        assert got[k].dtype == np.int64 and got[k].tolist() == want[k], k


def test_load_polyasite_converts_to_the_priming_site_boundary(tmp_path):
    p = _bed(tmp_path / "atlas.bed.gz")
    # plus: the id position itself; minus: id position + 1
    _eq(load_polyasite(p), {("chr1", True): [100, 500, 2000], ("chr1", False): [201, 701], ("chr2", True): [50],
                            ("chrX", False): [11]})
    # score (column 5) at or above min_score: 0.2 is kept, 0.05, 0.01 and 0.001 go, and chrX loses its only site
    _eq(load_polyasite(p, min_score=0.2), {("chr1", True): [100, 2000], ("chr1", False): [201], ("chr2", True): [50]})


def test_conversion_puts_the_atlas_site_on_the_priming_site_of_a_read(tmp_path):
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "chr1", "LN": 5000}]})
    reads = []
    for start, rev in ((90, False), (201, True)):
        a = pysam.AlignedSegment(hdr)
        a.query_name, a.reference_id, a.reference_start, a.cigarstring = f"r{start}", 0, start, "10M"
        a.query_sequence, a.is_reverse = "C" * 10, rev
        reads.append(a)
    # plus read covers 0-based 90..99: last cDNA base 99 = 1-based 100 = chr1:100:+.
    # minus read covers 201..210: last cDNA base 201 = 1-based 202, yet chr1:200:- maps onto it. With the symmetric BED
    # reading, the atlas sits 2 nt further downstream on minus than on plus, both from 1417 tail-primed pa sites (a20 <= 4)
    # and from GENCODE 44 transcript ends, so the minus id is taken 2 nt downstream of the last templated base.
    mol = pd.DataFrame({"ref": "chr1", "strand": ["+", "-"], "site": [priming_site(a) for a in reads]})
    assert mol.site.tolist() == [100, 201]
    assert site_distances(mol, load_polyasite(_bed(tmp_path / "a.bed.gz"))).tolist() == [0.0, 0.0]


def test_nearest_signed_plus():
    ends = np.array([100, 500, 2000], np.int64)
    sites = np.array([50, 100, 101, 300, 299, 301, 2500, 1250])
    # 50: next end 100 downstream (+50); 101: 100 is 1 nt upstream (-1); 300: 100 and 500 both 200 away, the positive wins;
    # 299: -199 beats +201; 301: +199 beats -201; 2500: last end 500 nt upstream; 1250: tie of 750, positive
    got = nearest_signed(sites, True, ends)
    assert got.dtype == np.float64 and got.tolist() == [50, 0, -1, 200, -199, 199, -500, 750]


def test_nearest_signed_minus():
    ends = np.array([201, 701], np.int64)
    # minus: downstream is the lower coordinate, d = site - end. 150: the end at 201 lies upstream (-51); 451: tie of 250,
    # positive; 452: 701 is 249 upstream; 800: 701 lies 99 downstream
    assert nearest_signed(np.array([201, 150, 451, 452, 800]), False, ends).tolist() == [0, -51, 250, -249, 99]


def test_nearest_signed_without_atlas_sites():
    for ends in (None, np.array([], np.int64)):
        got = nearest_signed(np.array([5, 6]), True, ends)
        assert got.shape == (2,) and np.isnan(got).all()


def test_nearest_signed_matches_label_spike_site():
    rng = np.random.default_rng(3)
    ends = np.unique(rng.integers(0, 20_000, 150))
    sites = np.concatenate([rng.integers(-500, 20_500, 3000), ends, ends + 1, ends - 1])
    for strd in (True, False):
        want = [label_spike_site(int(s), strd, ends)[1] for s in sites]
        assert nearest_signed(sites, strd, ends).tolist() == want


def test_site_distances_per_ref_and_strand(tmp_path):
    atlas = load_polyasite(_bed(tmp_path / "atlas.bed.gz"))
    mol = pd.DataFrame([("chr1", "+", 101), ("chr1", "-", 800), ("chr2", "+", 40), ("chr2", "-", 40), ("chr3", "+", 5),
                        ("chr1", "+", 300), ("chrX", "-", 30)], columns=["ref", "strand", "site"])
    # chr1 +: 100 is 1 nt upstream; chr1 -: 701 lies 99 nt downstream; chr2 +: 50 lies 10 downstream; chr2 has no minus
    # site and chr3 no site at all; chr1 + 300: tie, positive; chrX -: 11 is 19 nt downstream
    got = site_distances(mol, atlas)
    assert got[[0, 1, 2, 5, 6]].tolist() == [-1, 99, 10, 200, 19]
    assert np.isnan(got[[3, 4]]).all()


def test_site_distances_moved_upstream():
    atlas = {("chr1", True): np.array([100, 2000], np.int64), ("chr1", False): np.array([3000], np.int64)}
    mol = pd.DataFrame([("chr1", "+", 1100), ("chr1", "-", 1990), ("chr1", "+", 3005), ("chrM", "+", 1100)],
                       columns=["ref", "strand", "site"])
    # 1,000 nt upstream in transcript sense: plus 1100 -> 100 (0); minus 1990 -> 2990 (upstream is the higher coordinate on
    # minus), the atlas site 3000 lies 10 nt upstream (-10); plus 3005 -> 2005, 2000 lies 5 nt upstream (-5); chrM: none.
    # Moved the wrong way the minus site would be 990, 2,010 nt from 3000.
    got = site_distances(mol, atlas, upstream=1000)
    assert got[:3].tolist() == [0, -10, -5] and np.isnan(got[3])
    # unmoved: 1100 is 900 upstream of 2000 and 1,000 downstream of 100 (+900); 1990 against 3000 (-1010); 3005 (-1005)
    got = site_distances(mol, atlas)
    assert got[:3].tolist() == [900, -1010, -1005] and np.isnan(got[3])


def test_distance_bin_edges():
    d = np.array([0, 10, -10, 11, -11, 50, -50, 51, -51, 200, -200, 201, -201, 5000, np.nan])
    want = ["0-10"] * 3 + ["11-50"] * 4 + ["51-200"] * 4 + [">200"] * 3 + ["no atlas site on this ref and strand"]
    assert BINS == ("0-10", "11-50", "51-200", ">200", "no atlas site on this ref and strand")
    assert distance_bin(d).tolist() == want
