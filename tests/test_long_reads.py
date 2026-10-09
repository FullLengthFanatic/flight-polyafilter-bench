import random

import numpy as np
import pandas as pd
import pysam
import pytest

from pafbench.bd67 import (accept_long_read, annotate_sites, a_run_site, compare_labels, count_duplicates,
                           hexamer_summary, inward_pairs, read_outcome, scan_long_reads, sense_3p_clip,
                           terminal_a_run)
from pafbench.genes import load_gene_segments
from pafbench.regions import load_region_segments
from pafbench.sites import COMP, context_from_seq, has_hexamer, priming_site, sense_context

HDR = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 100000}]})


def _read(pos, cigar, seq, rev, name="x", extra=0, mapq=60):
    a = pysam.AlignedSegment(HDR)
    a.query_name, a.reference_name, a.reference_start = name, "chr1", pos
    a.cigarstring, a.flag, a.query_sequence, a.mapping_quality = cigar, (16 if rev else 0) | extra, seq, mapq
    return a


# name: (record args, expected at tolerance 1, expected at tolerance 0); expected = (site, aligned_a, clip_a, after_len).
# Forward reads are in sense already. Reverse reads hold the reverse complement of the sense sequence, so the
# 3' clip is the leading soft clip and the aligned base next to it (reference_start) is the sense S[b-1].
CASES = {
    # pure clip run; also the aligned non-A at the run start (G, with C further inward) is not taken
    "pure_clip_fwd": ((100, "10M45S", "C" * 9 + "G" + "A" * 12 + "G" * 33, False), (110, 0, 12, 33), (110, 0, 12, 33)),
    # sense clip A12 G33 -> leading clip C33 T12; sense aligned C9 G -> ref-orientation C G9 (genomic 200 = sense G)
    "pure_clip_rev": ((200, "45S10M", "C" * 33 + "T" * 12 + "C" + "G" * 9, True), (200, 0, 12, 33), (200, 0, 12, 33)),
    # aligned A's 110..114 absorbed plus 8 clip A's: run starts at 110
    "aligned_plus_clip_fwd": ((100, "15M40S", "C" * 10 + "A" * 5 + "A" * 8 + "T" * 32, False), (110, 5, 8, 32), (110, 5, 8, 32)),
    # genomic 200..204 are T (sense A), 205..214 G (sense C); run's first base r = 204 -> site 205
    "aligned_plus_clip_rev": ((200, "40S15M", "A" * 32 + "T" * 8 + "T" * 5 + "G" * 10, True), (205, 5, 8, 32), (205, 5, 8, 32)),
    # clip A5 G A6 C38: the G is taken at tolerance 1 (11 A's), the run stops at it at tolerance 0
    "clip_mismatch_fwd": ((100, "10M50S", "C" * 10 + "AAAAAGAAAAAA" + "C" * 38, False), (110, 0, 11, 38), (110, 0, 5, 45)),
    "clip_mismatch_rev": ((300, "50S10M", "G" * 38 + "TTTTTTCTTTTT" + "G" * 10, True), (300, 0, 11, 38), (300, 0, 5, 45)),
    # aligned C10 AAA G AAAAAA (110..119) + clip A4 C36: tolerance 1 takes G at 113 and the A's at 110..112
    "aligned_mismatch_fwd": ((100, "20M40S", "C" * 10 + "AAAGAAAAAA" + "AAAA" + "C" * 36, False), (110, 9, 4, 36), (114, 6, 4, 36)),
    # genomic 300..305 sense A, 306 sense G, 307..309 sense A, 310..319 sense C
    "aligned_mismatch_rev": ((300, "40S20M", "G" * 36 + "TTTT" + "TTTTTTCTTT" + "G" * 10, True), (310, 9, 4, 36), (306, 6, 4, 36)),
    # aligned C7 G AAAA: the G at 107 has C further inward, so it would start the run and is not taken
    "nonA_run_start_fwd": ((100, "12M40S", "C" * 7 + "G" + "AAAA" + "A" * 8 + "C" * 32, False), (108, 4, 8, 32), (108, 4, 8, 32)),
    # first clip base G after aligned A's: taken at tolerance 1 (aligned base S[b-1] is A)
    "first_clip_nonA_after_A": ((100, "13M41S", "C" * 10 + "AAA" + "G" + "A" * 10 + "C" * 30, False), (110, 3, 10, 30), (110, 3, 0, 41)),
    # first clip base G after aligned C: not taken, the run is empty and the clip is all after-run sequence
    "first_clip_nonA_after_nonA_fwd": ((100, "10M43S", "C" * 10 + "G" + "A" * 12 + "C" * 30, False), (110, 0, 0, 43), (110, 0, 0, 43)),
    "first_clip_nonA_after_nonA_rev": ((200, "43S10M", "G" * 30 + "T" * 12 + "C" + "G" * 10, True), (200, 0, 0, 43), (200, 0, 0, 43)),
    # clip A4 G A4 G A4: the second G finds the budget spent
    "second_nonA_clip": ((100, "10M44S", "C" * 10 + "AAAAGAAAAGAAAA" + "C" * 30, False), (110, 0, 8, 35), (110, 0, 4, 40)),
    # aligned C6 AAGAAA (109..111 A, 108 G) + clip AAAGAAAA C30: the clip G spends the budget, the aligned G stops the run
    "budget_shared": ((100, "12M38S", "C" * 6 + "AAGAAA" + "AAAGAAAA" + "C" * 30, False), (109, 3, 7, 30), (109, 3, 3, 35)),
    # no clip: left growth may still take a non-A (the run has aligned A's on its right)
    "no_clip_aligned_mismatch": ((100, "10M", "CCAAGAAAAA", False), (102, 7, 0, 0), (105, 5, 0, 0)),
    "no_clip_no_a": ((100, "10M", "CCCCCCCCCG", False), (110, 0, 0, 0), (110, 0, 0, 0)),
    # inserted GG inside the aligned run is not an aligned base: run 110..116 (7 A's)
    "insertion_fwd": ((100, "13M2I4M40S", "C" * 10 + "AAA" + "GG" + "AAAA" + "A" * 8 + "T" * 32, False), (110, 7, 8, 32), (110, 7, 8, 32)),
    # sense A at genomic 300..303 and 304..306 (CC inserted between), 307 deleted, 308..317 sense C: r = 306 -> site 307
    "insertion_deletion_rev": ((300, "40S4M2I3M1D10M", "A" * 32 + "T" * 8 + "TTTT" + "CC" + "TTT" + "G" * 10, True), (307, 7, 8, 32), (307, 7, 8, 32)),
    # 5 + 4 = 9 run A's with 30 after: one short of the run threshold
    "run_9": ((100, "15M34S", "C" * 10 + "A" * 5 + "A" * 4 + "C" * 30, False), (110, 5, 4, 30), (110, 5, 4, 30)),
    # 5 + 5 = 10 run A's with 29 after: one short of the clip threshold
    "after_29": ((100, "15M34S", "C" * 10 + "A" * 5 + "A" * 5 + "C" * 29, False), (110, 5, 5, 29), (110, 5, 5, 29)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_terminal_a_run_cases(name):
    args, exp1, exp0 = CASES[name]
    a = _read(*args)
    assert terminal_a_run(a) == exp1
    assert terminal_a_run(a, tolerance=0) == exp0
    assert (exp0[0], exp0[1]) == a_run_site(a)


def test_acceptance_and_outcome():
    assert not accept_long_read(4, 5, 30) and accept_long_read(4, 6, 30)
    assert not accept_long_read(10, 0, 29) and accept_long_read(0, 10, 30)
    outcome = lambda name: read_outcome(*terminal_a_run(_read(*CASES[name][0]))[1:])
    assert outcome("budget_shared") == "accepted"            # 3 + 7 = 10 run A's, 30 after
    assert outcome("run_9") == "short_run"
    assert outcome("after_29") == "short_after"
    assert outcome("no_clip_aligned_mismatch") == "both"     # 7 run A's, 0 after
    assert outcome("first_clip_nonA_after_nonA_fwd") == "short_run"


def _literal_run(a, tol):
    """The brief's rule written out on the full sense string S from get_aligned_pairs."""
    seq, pairs = a.query_sequence, a.get_aligned_pairs(matches_only=True)
    al = [(seq[q].translate(COMP), r) for q, r in reversed(pairs)] if a.is_reverse else [(seq[q], r) for q, r in pairs]
    c = sense_3p_clip(a)
    s, b = "".join(x for x, _ in al) + c, len(al)
    budget, j = tol, b
    while j < len(s):
        if s[j] == "A":
            j += 1
        elif budget and j + 1 < len(s) and s[j + 1] == "A" and (j > b or s[b - 1] == "A"):
            budget, j = budget - 1, j + 1
        else:
            break
    i = b
    while i > 0:
        if s[i - 1] == "A":
            i -= 1
        elif budget and i >= 2 and s[i - 2] == "A" and (i < b or j > b):
            budget, i = budget - 1, i - 1
        else:
            break
    site = priming_site(a) if i == b else (al[i][1] + 1 if a.is_reverse else al[i][1])
    return site, s[i:b].count("A"), s[b:j].count("A"), len(c) - (j - b)


def _random_reads(n, seed=3):
    rng = random.Random(seed)
    for k in range(n):
        rev = rng.random() < 0.5
        ops = [(5, rng.randint(1, 5))] if rng.random() < 0.1 else []
        if rng.random() < 0.7:
            ops.append((4, rng.randint(1, 60)))
        ops.append((rng.choice([0, 7, 8]), rng.randint(1, 30)))
        for _ in range(rng.randint(0, 4)):
            ops += [(rng.choice([1, 2, 3]), rng.randint(1, 4)), (rng.choice([0, 7, 8]), rng.randint(1, 30))]
        if rng.random() < 0.7:
            ops.append((4, rng.randint(1, 60)))
        if rng.random() < 0.1:
            ops.append((5, rng.randint(1, 5)))
        qlen = sum(n for op, n in ops if op in (0, 1, 4, 7, 8))
        a = pysam.AlignedSegment(HDR)
        a.query_name, a.reference_name, a.reference_start = f"q{k}", "chr1", 1000
        a.cigartuples, a.flag = ops, 16 if rev else 0
        a.query_sequence = "".join(rng.choice("TTTTTGCAN" if rev else "AAAAACGTN") for _ in range(qlen))
        yield a


def _expected_inward(a):
    pairs = a.get_aligned_pairs(matches_only=True)
    return pairs if a.is_reverse else pairs[::-1]


def test_random_reads_match_literal_rule_and_a_run_site():
    for a in _random_reads(3000):
        assert list(inward_pairs(a)) == _expected_inward(a)
        assert terminal_a_run(a, 0)[:2] == a_run_site(a)
        for tol in (0, 1, 2):
            assert terminal_a_run(a, tol) == _literal_run(a, tol)


def test_synthetic_bam_matches_a_run_site(synth):
    n = 0
    with pysam.AlignmentFile(str(synth.bam)) as bam:
        for a in bam.fetch():
            if a.query_sequence is None:
                continue
            assert list(inward_pairs(a)) == _expected_inward(a)
            assert terminal_a_run(a, 0)[:2] == a_run_site(a)
            assert terminal_a_run(a) == _literal_run(a, 1)
            n += 1
    assert n == 144     # 149 synthetic records (75 + 5 + 41 + 9 + 5 + 4 + 4 + 2 + 4) minus 5 secondaries without a sequence


def test_context_from_seq_matches_sense_context(synth):
    with pysam.FastaFile(str(synth.fasta)) as fa:
        for ref in fa.references:
            seq, L = fa.fetch(ref).upper(), fa.get_reference_length(ref)
            for site in (0, 1, 19, 20, 21, 39, 40, 41, 1299, 1500, L - 41, L - 40, L - 21, L - 20, L - 1, L):
                for rev in (False, True):
                    assert context_from_seq(seq, site, rev) == sense_context(fa, ref, site, rev)
        seq = fa.fetch("chrA").upper()
        assert context_from_seq(seq, 1299, False)[0].startswith("A" * 12)      # planted A12 at 1299
        assert context_from_seq(seq, 4509, True)[0].startswith("A" * 10)       # planted T10 at 4499..4508


def _bam(tmp_path, reads):
    unsorted, path = tmp_path / "u.bam", tmp_path / "r.bam"
    with pysam.AlignmentFile(str(unsorted), "wb", header=HDR) as out:
        for a in reads:
            out.write(a)
    pysam.sort("-o", str(path), str(unsorted))
    pysam.index(str(path))
    return path


def test_scan_long_reads(tmp_path):
    rec = lambda case, **kw: _read(*CASES[case][0][:4], **kw)
    reads = [
        rec("aligned_plus_clip_fwd", name="f1"),                    # +110 accepted
        rec("aligned_plus_clip_fwd", name="f2", mapq=20),           # +110 accepted, MAPQ 20 is kept
        rec("run_9", name="f3"),                                    # +110 short_run
        rec("after_29", name="f4"),                                 # +110 short_after
        rec("no_clip_aligned_mismatch", name="f5"),                 # +102 both
        rec("aligned_plus_clip_rev", name="r1"),                    # -205 accepted
        rec("aligned_plus_clip_fwd", name="s1", extra=256),         # secondary: skipped
        rec("aligned_plus_clip_rev", name="s2", extra=2048),        # supplementary: skipped
        rec("aligned_plus_clip_fwd", name="lq", mapq=19),           # MAPQ 19: skipped
        rec("first_clip_nonA_after_A", name="f1"),                  # +110 accepted (3 + 10, 30 after), repeated qname
    ]
    sites, counts, digests = scan_long_reads(_bam(tmp_path, reads), "chr1")
    assert sites.strand.tolist() == ["+", "+", "-"] and sites.site.tolist() == [102, 110, 205]
    assert sites.n_reads.tolist() == [1, 5, 1] and sites.n_accepted.tolist() == [0, 3, 1]
    assert counts == {"records": 10, "secondary_supplementary": 2, "low_mapq": 1, "used": 7,
                      "accepted": 4, "short_run": 1, "short_after": 1, "both": 1}
    assert len(digests) == 7 and count_duplicates(digests) == 1


def test_annotate_sites(synth):
    sites = pd.DataFrame({"strand": ["+", "+", "+", "+", "-", "-"], "site": [0, 1299, 1600, 4100, 4509, 6000],
                          "n_reads": [1, 2, 3, 4, 5, 6], "n_accepted": [0, 1, 2, 3, 4, 5]})
    with pysam.FastaFile(str(synth.fasta)) as fa:
        seq = fa.fetch("chrA").upper()
        out = annotate_sites(sites, "chrA", seq, load_region_segments(synth.gtf), load_gene_segments(synth.gtf))
        ctx = [sense_context(fa, "chrA", s, st == "-") for s, st in zip(sites.site, sites.strand)]
    assert list(out.columns) == ["ref", "strand", "site", "n_reads", "n_accepted", "a20", "label", "a_band",
                                 "hexamer", "region", "gene"]
    assert (out.ref == "chrA").all() and out.n_reads.tolist() == [1, 2, 3, 4, 5, 6]
    a20 = [d.count("A") for d, _ in ctx]
    assert out.a20.tolist() == a20 and out.hexamer.tolist() == [has_hexamer(u) for _, u in ctx]
    # planted runs: A6 at 0, A12 at 1299, A10 at 1599 (9 of them after 1600), T10 before 4509 on -, T6 at 5994
    assert a20[0] >= 6 and a20[1] >= 12 and a20[2] >= 9 and a20[4] >= 10 and a20[5] >= 6
    assert out.label.tolist() == ["templated" if x >= 8 else "nontemplated" if x <= 4 else "ambiguous" for x in a20]
    assert out.a_band.iloc[1] == ("14-20" if a20[1] >= 14 else "10-13")
    # last cDNA base: -1 (off contig), 1298 (t1/t2 exon 1, no CDS), 1599 (g1 intron), 4099 (past g1),
    # 4509 on - (t3 exon 4201-4700), 6000 on - (past g2)
    assert out.region.tolist() == ["intergenic", "ncexon", "intron", "intergenic", "ncexon", "intergenic"]
    assert out.gene.tolist() == ["", "g1", "g1", "", "g2", ""]


def test_annotate_sites_last_cdna_base_at_boundaries(synth):
    # 0-based: g1 exon 1 1000..1399 (+); g2 = t3 4200..5799 (-), t3 exon 4200..4699, intron 4700..4999.
    # + 1400: last 1399 exon end (1400 intron); + 1000: last 999 intergenic (1000 exon);
    # - 4200: last 4200 first exon base (4199 intergenic); - 4699: last 4699 exon (4700 intron);
    # - 5800: last 5800 intergenic (5799 g2 exon)
    sites = pd.DataFrame({"strand": ["+", "+", "-", "-", "-"], "site": [1400, 1000, 4200, 4699, 5800],
                          "n_reads": [1] * 5, "n_accepted": [1] * 5})
    with pysam.FastaFile(str(synth.fasta)) as fa:
        out = annotate_sites(sites, "chrA", fa.fetch("chrA").upper(), load_region_segments(synth.gtf),
                             load_gene_segments(synth.gtf))
    assert out.region.tolist() == ["ncexon", "intergenic", "ncexon", "ncexon", "intergenic"]
    assert out.gene.tolist() == ["g1", "", "g2", "g2", ""]


def test_hexamer_summary():
    sites = pd.DataFrame({"a20": [0, 2, 3, 4, 8, 9, 12, 15, 6, 0],
                          "hexamer": [True, True, False, False, True, False, False, False, True, True],
                          "n_accepted": [1, 1, 1, 2, 1, 1, 3, 1, 1, 0]})
    s = hexamer_summary(sites, {"templated_min": 8, "nontemplated_max": 4})
    # last row has no accepted read; non-templated a20 0,2,3,4 -> 2/4; templated 8,9,12,15 -> 1/4; ambiguous 6 -> 1/1
    assert s["n"] == {"templated": 4, "nontemplated": 4, "ambiguous": 1}
    assert s["hexamer_rate"] == {"templated": 0.25, "nontemplated": 0.5, "ambiguous": 1.0}
    assert s["ratio"] == 2.0 and s["passes"] is False
    s = hexamer_summary(sites, {"templated_min": 14, "nontemplated_max": 8})
    # templated only a20 15 (no hexamer): rate 0, ratio undefined
    assert s["n"] == {"templated": 1, "nontemplated": 6, "ambiguous": 2}
    assert s["hexamer_rate"]["nontemplated"] == 4 / 6 and s["ratio"] is None and s["passes"] is False
    sites = pd.DataFrame({"a20": [0, 1, 2, 3, 10, 11, 12, 13], "hexamer": [True] * 3 + [False] + [True] + [False] * 3,
                          "n_accepted": [1] * 8})
    s = hexamer_summary(sites, {"templated_min": 8, "nontemplated_max": 4})
    assert s["ratio"] == 3.0 and s["passes"] is True        # 0.75 / 0.25


def test_compare_labels():
    sites = pd.DataFrame({"ref": ["chr1", "chr1", "chr1", "chr1", "chr2"], "strand": ["+", "+", "+", "-", "+"],
                          "site": [100, 110, 104, 100, 500],
                          "label": ["templated", "nontemplated", "ambiguous", "nontemplated", "templated"],
                          "n_accepted": [1, 2, 0, 1, 1]})
    mol = pd.DataFrame({
        "ref": ["chr1", "chr1", "chr1", "chr1", "chr1", "chr2", "chr3", "chr1", "chr1"],
        "strand": ["+", "+", "+", "+", "-", "-", "+", "+", "+"],
        "site": [105, 107, 121, 120, 100, 500, 10, 90, 200],
        "label": ["templated", "nontemplated", "templated", "ambiguous", "templated", "nontemplated",
                  "nontemplated", "templated", "nontemplated"]})
    table, agree = compare_labels(mol, sites)
    # 105: 100 and 110 both 5 away, the lower site wins (104 is closer but has no accepted read); 107 -> 110;
    # 121 is 11 from 110 (none); 120 is 10 from 110 (matched); chr1 - 100 exact; chr2 - and chr3 have no site;
    # 90 is 10 below 100; 200 is 90 above 110
    exp = {("templated", "templated"): 2, ("nontemplated", "nontemplated"): 1, ("templated", "none"): 1,
           ("ambiguous", "nontemplated"): 1, ("templated", "nontemplated"): 1, ("nontemplated", "none"): 3}
    assert list(table.columns) == ["label_1179", "label_long", "n"]
    assert dict(zip(zip(table.label_1179, table.label_long), table.n)) == exp
    assert agree["molecules"] == 9 and agree["matched"] == 5 and agree["share_matched"] == 5 / 9
    # matched templated: 105 (agree), chr1 - 100 (disagree), 90 (agree); matched non-templated: 107 (agree)
    assert agree["agreement"] == {"templated": 2 / 3, "nontemplated": 1.0, "pooled": 0.75}
    # templated 105, 121, chr1 - 100, 90 -> 3 of 4 matched; non-templated 107, chr2, chr3, 200 -> 1 of 4; ambiguous 120
    assert agree["share_matched_by_label"] == {"templated": 0.75, "nontemplated": 0.25, "ambiguous": 1.0}


def test_count_duplicates():
    assert count_duplicates(np.array([5, 7, 5, 5, 9], np.uint64)) == 2
    assert count_duplicates(np.array([], np.uint64)) == 0
