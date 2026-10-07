import pysam

from pafbench.bd67 import a_run_site, leading_a, sense_3p_clip

HDR = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "chr1", "LN": 100000}]})


def _read(pos, cigar, seq, rev):
    a = pysam.AlignedSegment(HDR)
    a.query_name, a.reference_name, a.reference_start = "x", "chr1", pos
    a.cigarstring, a.flag, a.query_sequence = cigar, 16 if rev else 0, seq
    return a


def test_forward_read():
    a = _read(100, "10M6S", "CCCCCAAAAA" + "AAAGGG", False)
    assert sense_3p_clip(a) == "AAAGGG" and leading_a("AAAGGG") == 3
    assert a_run_site(a) == (105, 5)


def test_reverse_read():
    a = _read(200, "4S10M", "CCTT" + "TTTGGGGGGG", True)
    assert sense_3p_clip(a) == "AAGG"
    assert a_run_site(a) == (203, 3)


def test_no_a_run():
    a = _read(100, "10M", "CCCCCCCCCG", False)
    assert a_run_site(a) == (110, 0) and sense_3p_clip(a) == ""
