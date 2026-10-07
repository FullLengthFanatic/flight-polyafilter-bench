import numpy as np

from pafbench.transcripts import coverage_prefix, expressed, load_transcripts, start_pieces
from pafbench.vendor import polyafilter_modules

_, RNA, SNR = polyafilter_modules()


def test_start_pieces_match_original(synth):
    tx = load_transcripts(synth.gtf)
    for introns in (False, True):
        bl = RNA.getBaselineData(str(synth.tmp / f"bl_{int(introns)}.pkl"), str(synth.db), str(synth.bam), includeIntrons=introns)
        for covlen in (50, 100, 300):
            orig = RNA.getTransEndSensSpec(covlen, str(synth.bam), bl, introns, getSensSpec=False)[4]
            for ref in ("chrA", "chrB"):
                prefix = coverage_prefix(synth.bam, ref)
                for strd in (True, False):
                    ex = expressed(tx.get((ref, strd), []), prefix[strd], introns)
                    s, e = start_pieces(ex, covlen, introns, strd)
                    flat = []
                    for t in orig.get(strd, {}).get(ref, []):
                        flat.extend(t)
                    want = SNR.flattenIntervals(flat)
                    assert list(zip(s.tolist(), e.tolist())) == want, (ref, strd, covlen, introns)


def test_coverage_matches_pysam_count_coverage(tmp_path):
    import pysam
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "c", "LN": 100}]})
    path = tmp_path / "x.bam"
    with pysam.AlignmentFile(str(path), "wb", header=hdr) as out:
        a = pysam.AlignedSegment(hdr)
        a.query_name, a.reference_name, a.reference_start, a.cigarstring, a.flag = "s", "c", 10, "5M", 256  # no SEQ
        out.write(a)
        b = pysam.AlignedSegment(hdr)
        b.query_name, b.reference_name, b.reference_start, b.cigarstring, b.flag = "n", "c", 50, "3M", 0
        b.query_sequence, b.query_qualities = "ANA", pysam.qualitystring_to_array("III")
        out.write(b)
    pysam.index(str(path))
    p = coverage_prefix(path, "c")[True]
    assert p[15] - p[10] == 0          # record without SEQ adds no coverage
    assert p[53] - p[50] == 2          # N is not counted
    with pysam.AlignmentFile(str(path)) as bam:
        cov = np.sum(bam.count_coverage("c", 0, 100, quality_threshold=0, read_callback=lambda r: not r.is_reverse), axis=0)
    assert ((p[1:] - p[:-1]) == (cov > 0)).all()


def test_short_transcript_fully_protected_with_introns(synth):
    tx = load_transcripts(synth.gtf)
    prefix = coverage_prefix(synth.bam, "chrB")
    ex = expressed(tx[("chrB", False)], prefix[False], True)
    s, e = start_pieces(ex, 100, True, False)
    assert s.size == 0
