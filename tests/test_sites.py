import pysam

from pafbench.genes import genes_at, load_gene_segments
from pafbench.sites import classify, has_hexamer, priming_site, sense_context


def _read(hdr, pos, cigar, rev):
    a = pysam.AlignedSegment(hdr)
    a.query_name = "x"
    a.reference_name = "chrA"
    a.reference_start = pos
    a.cigarstring = cigar
    a.flag = 16 if rev else 0
    a.query_sequence = "C" * sum(n for op, n in a.cigartuples if op in (0, 1, 4, 7, 8))
    return a


def test_site_and_context(synth):
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "chrA", "LN": 6000}, {"SN": "chrB", "LN": 2000}]})
    with pysam.FastaFile(str(synth.fasta)) as fa:
        fwd = _read(hdr, 1209, "90M", False)        # ends at 1299, where A*12 is planted
        assert priming_site(fwd) == 1299
        down, up = sense_context(fa, "chrA", 1299, False)
        assert down.startswith("A" * 12) and len(down) == 20 and len(up) == 40
        rev = _read(hdr, 4509, "90M", True)          # minus strand; T*10 at 4499..4508 lies downstream
        assert priming_site(rev) == 4509
        down, _ = sense_context(fa, "chrA", 4509, True)
        assert down.startswith("A" * 10)


def test_hexamer_window():
    up = "C" * 20 + "AATAAA" + "C" * 14            # hexamer ends 14 nt before the site
    assert has_hexamer(up)
    assert not has_hexamer("C" * 30 + "AATAAA" + "C" * 4)   # ends 4 nt before the site, too close
    assert not has_hexamer("AATAAA" + "C" * 34)     # further than 35 nt


def test_classify():
    assert classify(14) == "templated" and classify(8) == "nontemplated" and classify(11) == "ambiguous"


def test_gene_segments(synth):
    seg = load_gene_segments(synth.gtf)
    assert genes_at(seg, "chrA", True, 1500) == "g1"
    assert genes_at(seg, "chrA", True, 4100) == ""
    assert genes_at(seg, "chrA", False, 4300) == "g2"
    assert genes_at(seg, "chrB", True, 1700) == ""
