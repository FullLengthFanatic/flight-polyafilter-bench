import importlib.util
import json
import random
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pysam
import pytest

from pafbench.bd67 import (accept_long_read, annotate_sites, a_run_site, compare_labels, count_duplicates,
                           hexamer_summary, inward_pairs, read_outcome, scan_long_reads, sense_3p_clip,
                           terminal_a_run)
from pafbench import tecap
from pafbench.config import REPO
from pafbench.genes import load_gene_segments
from pafbench.lrcontext import CLASSES, CONTEXT_COLUMNS, READ_COLUMNS, SUB_CAUSES, load_models, scan_context
from pafbench.regions import load_region_segments
from pafbench.sites import COMP, context_from_seq, has_hexamer, priming_site, sense_context
from pafbench.tecap import TAG
from tests.synth_long import AS, DESIGNED, G, IG, NJ, ctx, expected, long_read, write_bam, write_fasta, write_gtf

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


def _context_reads():
    """The designed reads of stage 30 on both strands, a read whose aligned bases all belong to the run, and five records
    that stage 16 does not accept or does not use."""
    reads = [long_read(blocks, site, minus, name=f"{name}{'-' if minus else '+'}")
             for name, (blocks, site, _) in DESIGNED.items() for minus in (False, True)]
    blocks = DESIGNED["full_chain_3UTR"][0]
    body = blocks[:-1] + [(1600, 1800)]
    return reads + [long_read([(1050, 1090)], 1050, name="all_in_run"),
                    long_read(body, 1800, name="short_run", run=9),         # no aligned A and 9 clipped A's
                    long_read(body, 1800, name="short_after", after=29),
                    long_read(blocks, 1800, name="low_mapq", mapq=19),
                    long_read(blocks, 1800, name="secondary", flag=256),
                    long_read(blocks, 1800, minus=True, name="supplementary", flag=2048)]


def _read_row(a, site, want):
    return (a.is_reverse, site, CLASSES.index(want.klass), want.matched, want.chain_novel, want.starts_outside,
            want.retained_after_junction, want.end_nf, want.complete_5p, want.covers_start, want.other_class,
            SUB_CAUSES.index(want.sub_cause) if want.sub_cause else -1, a.query_length, "N" not in a.cigarstring)


def test_scan_context_takes_the_reads_and_sites_of_stage_16(tmp_path):
    records = _context_reads()
    bam = write_bam(tmp_path / "long.bam", records)
    models = load_models(write_gtf(tmp_path / "genes.gtf"), ["chrS", "chrE"])
    reads, counts = scan_context(bam, "chrS", models["chrS"], tagged=tmp_path / "tagged.bam")
    sites16, counts16, _ = scan_long_reads(bam, "chrS")
    assert counts == {k: counts16[k] for k in ("records", "secondary_supplementary", "low_mapq", "used", "accepted")} | {"empty_body": 1}
    assert counts["records"] == 2 * len(DESIGNED) + 6 and counts["accepted"] == 2 * len(DESIGNED) + 1 == len(reads)
    # the sites with an accepted read and their read counts are those of stage 16
    acc = sites16[sites16.n_accepted >= 1]
    assert reads.groupby(["minus", "site"]).size().to_dict() == {(s == "-", p): n for s, p, n in zip(acc.strand, acc.site, acc.n_accepted)}
    # one row per accepted read with its class and flags. The read without a body is classed at its last cDNA base, 1049,
    # as an unspliced read: 5' UTR of TA, 49 nt from its 5' end, and in an intron of TB
    want = [_read_row(a, G - site if minus else site, expected(name, minus)) for a, (name, minus, site) in zip(
        records, [(name, minus, site) for name, (_, site, _) in DESIGNED.items() for minus in (False, True)])]
    want.append(_read_row(records[2 * len(DESIGNED)], 1050, ctx("5UTR", "TA", complete_5p=True, other_class=True)))
    assert list(reads.columns) == READ_COLUMNS
    assert sorted(reads.itertuples(index=False, name=None)) == sorted(want)
    assert reads.dtypes.astype(str).to_dict() == {
        "minus": "bool", "site": "int64", "klass": "int8", "matched": "int16", "chain_novel": "int16", "starts_outside": "bool",
        "retained_after_junction": "bool", "end_nf": "bool", "complete_5p": "bool", "covers_start": "bool", "other_class": "bool",
        "sub_cause": "int8", "length": "int32", "no_splice": "bool"}
    # the accepted records, in the order of the BAM and unchanged but for the site key in the tag
    site_of = {a.query_name: f"chrS:{'-' if a.is_reverse else '+'}:{row[1]}" for a, row in zip(records, want)} | {"all_in_run": "chrS:+:1050"}
    with pysam.AlignmentFile(str(bam)) as src, pysam.AlignmentFile(str(tmp_path / "tagged.bam")) as out:
        assert out.header.to_dict() == src.header.to_dict()
        kept = [a for a in src.fetch("chrS") if a.query_name in site_of]
        got = list(out.fetch(until_eof=True))
        assert [a.query_name for a in got] == [a.query_name for a in kept] and len(got) == counts["accepted"]
        for a, b in zip(got, kept):
            assert a.get_tags() == b.get_tags() + [(TAG, site_of[a.query_name])]
            a.set_tag(TAG, None)
            assert a.to_string() == b.to_string()
    # a reference without a read, and no tagged BAM unless asked for
    reads, counts = scan_context(bam, "chrE", models["chrE"])
    assert list(reads.columns) == READ_COLUMNS and not len(reads) and counts["records"] == 0 and counts["accepted"] == 0
    assert sorted(p.name for p in tmp_path.glob("*.bam")) == ["long.bam", "long.unsorted.bam", "tagged.bam"]


def _stage30(monkeypatch):
    """The stage script as module 'stage30' (registered so that pool workers can unpickle its functions)."""
    spec = importlib.util.spec_from_file_location("stage30", REPO / "scripts" / "30_long_read_context.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "stage30", mod)
    spec.loader.exec_module(mod)
    return mod


def _fake_tecap(bam, sample, out_dir, threads):
    """Stands in for tecap.run_classify: every record with a CIGAR N goes to the bucket "spliced" of its tag value, as
    tecap counts only reads with a splice, and the result is written as per_cell_metrics."""
    cells = {}
    with pysam.AlignmentFile(str(bam)) as fh:
        assert fh.has_index()
        for a in fh.fetch():
            if "N" in a.cigarstring:
                cells.setdefault(a.get_tag(TAG), {"counts": {"spliced": 0}})["counts"]["spliced"] += 1
    out_dir.mkdir(parents=True)
    path = out_dir / f"{sample}_terminal_exon.json"
    path.write_text(json.dumps({"sample": sample, "cell_barcode_tag": TAG, "threads": threads, "per_cell_metrics": cells}))
    return path


@pytest.fixture
def stage30_inputs(tmp_path, monkeypatch):
    """Inputs of stage 30 for the designed reads: the BAM as BD67 and as BD70, the stage 16 site table of each made with
    stage 16's own functions, 1179 truth molecules, and a pigeon isoform GTF with its classification."""
    base = tmp_path / "base"
    bam, gtf, fasta = write_bam(tmp_path / "long.bam", _context_reads()), write_gtf(tmp_path / "genes.gtf"), write_fasta(tmp_path / "genome.fa")
    seg = (load_region_segments(gtf), load_gene_segments(gtf))
    with pysam.FastaFile(str(fasta)) as fa:
        s16 = pd.concat([annotate_sites(scan_long_reads(bam, ref)[0], ref, fa.fetch(ref).upper(), *seg) for ref in ("chrS", "chrE")],
                        ignore_index=True)
    for lib in ("BD67", "BD70"):
        (base / lib).mkdir(parents=True)
        s16.to_parquet(base / lib / "long_read_sites.parquet", index=False)
    (base / "1179" / "truth").mkdir(parents=True)
    # at the site of three designed reads, 10 nt from another, 11 nt from it, on the other strand of the first, and on
    # a reference that is not scanned
    mol = pd.DataFrame({"mol": [3, 5, 8, 9, 11], "ref": ["chrS"] * 4 + ["chrX"], "strand": ["+", "+", "+", "-", "+"],
                        "site": [1650, 1340, 1341, 1650, 1650], "label": ["templated", "nontemplated", "templated", "ambiguous", "templated"],
                        "a_band": "x"})
    mol.to_parquet(base / "1179" / "truth" / "molecules_labeled.parquet", index=False)
    pigeon = tmp_path / "pigeon.gtf"
    pigeon.write_text("##pacbio-collapse-version 1.0\n" + "".join(
        f'{ref}\tPacBio\ttranscript\t{s}\t{e}\t.\t{strand}\t.\tgene_id "PB.{i}"; transcript_id "PB.{i}.1";\n'
        for i, (ref, s, e, strand) in enumerate([("chrS", 1001, 1805, "+"), ("chrS", 1001, 1811, "+"), ("chrE", 5, 90, "+"),
                                                 ("chrS", G - 1995, G - 1000, "-"), ("chrX", 1001, 1800, "+")])))
    cls = tmp_path / "classification.txt"
    cls.write_text("isoform\tchrom\tstructural_category\tsubcategory\nPB.0.1\tchrS\tfull-splice_match\treference_match\n"
                   "PB.1.1\tchrS\tfull-splice_match\treference_match\nPB.3.1\tchrS\tincomplete-splice_match\t5prime_fragment\n")
    stage = _stage30(monkeypatch)
    monkeypatch.setattr(stage, "LONG_READS", {"BD67": bam, "BD70": bam})
    monkeypatch.setattr(stage, "GTF", gtf)
    monkeypatch.setattr(stage, "PIGEON_GTF", pigeon)
    monkeypatch.setattr(stage, "PIGEON_CLASS", cls)
    monkeypatch.setattr(tecap, "run_classify", _fake_tecap)
    return stage, base, s16


def _run30(stage, monkeypatch, capsys, base, out, *extra):
    monkeypatch.setattr(sys, "argv", ["30", "--refs", "chrS", "chrE", "--base", str(base), "--out", str(out), "--workers", "2", *extra])
    stage.main()
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def _types(path):
    return {f.name: str(f.type).replace("large_", "") for f in pq.read_schema(path)}


def test_stage30_on_synthetic_long_reads(stage30_inputs, tmp_path, monkeypatch, capsys):
    stage, base, s16 = stage30_inputs
    out = tmp_path / "out"
    printed = _run30(stage, monkeypatch, capsys, base, out)
    acc = s16[s16.n_accepted >= 1].reset_index(drop=True)
    n_reads, n_spliced = 2 * len(DESIGNED) + 1, 2 * sum(len(blocks) > 1 for blocks, _, _ in DESIGNED.values())
    for lib in ("BD67", "BD70"):
        d = out / lib
        sites = pd.read_parquet(d / "isoform_context_sites.parquet")
        # the sites and accepted reads of stage 16, with its label, band, hexamer and region
        assert list(sites.columns) == CONTEXT_COLUMNS and len(sites) == len(acc) and sites.n_accepted.sum() == n_reads
        for c in ("ref", "strand", "site", "label", "a_band", "hexamer", "region", "n_accepted"):
            assert sites[c].tolist() == acc[c].tolist(), c
        assert (sites[[f"n_{c}" for c in CLASSES]].sum(axis=1) == sites.n_accepted).all()
        t = _types(d / "isoform_context_sites.parquet")
        assert {t[c] for c in ("ref", "strand", "label", "a_band", "region", "context", "context_without_chain_novel")} == {"string"}
        assert {t[c] for c in CONTEXT_COLUMNS if c.startswith("n_")} | {t["site"]} == {"int64"}
        assert (t["hexamer"], t["tied"]) == ("bool", "bool")
        assert {t[c] for c in ("top_share", "share_spliced", "share_5p_complete", "median_length")} == {"double"}
        # five designed reads end at +1800: two 3UTR, one of them placed by its last intron, and three novel junctions with
        # three sub-causes; three end at +1650: CDS, one of them placed by its last two introns and one with a retained intron
        at = lambda strand, site: sites[(sites.strand == strand) & (sites.site == site)].iloc[0]
        r = at("+", 1800)
        assert (r.n_3UTR, r[f"n_{NJ}"], r.n_chain_novel, r.n_chain_novel_suffix1) == (2, 3, 1, 1)
        assert (r.context, r.context_without_chain_novel, r.top_share, r.tied) == (NJ, NJ, 0.6, False)
        assert (r.share_spliced, r.share_5p_complete) == (0.4, 0.2)
        r = at("-", G - 1650)
        assert (r.n_CDS, r.n_chain_novel, r.n_chain_novel_suffix1, r.n_retained_after_junction) == (3, 1, 0, 1)
        assert (r.context, r.context_without_chain_novel, r.top_share) == ("CDS", "CDS", 1.0)
        assert at("+", 3497).n_end_NF == 1 and at("+", 930).n_starts_outside == 1 and at("+", 1050).context == "5UTR"
        # median query length of the records at a site, clips included: five designed reads at +1800 (518, 468, 515, 417 and
        # 417 nt), two at +1550 (266 and 377 nt), and the read without a body (7S40M45S)
        assert (at("+", 1800).median_length, at("+", 1550).median_length, at("+", 1050).median_length) == (468, 321.5, 92)
        summary = pd.read_parquet(d / "isoform_context_summary.parquet")
        assert list(summary.columns) == ["version", "label", "a_band", "context", "sites", "reads", "share_spliced", "share_5p_complete"]
        assert _types(d / "isoform_context_summary.parquet") == dict(zip(summary.columns, ["string"] * 4 + ["int64"] * 2 + ["double"] * 2))
        region = pd.read_parquet(d / "region_by_context.parquet")
        assert list(region.columns) == ["version", "region", "context", "sites", "reads"]
        assert _types(d / "region_by_context.parquet") == dict(zip(region.columns, ["string"] * 3 + ["int64"] * 2))
        for table in (summary, region):
            assert table.groupby("version")[["sites", "reads"]].sum().to_dict("index") == {
                v: {"sites": len(acc), "reads": n_reads} for v in ("placed", "strict")}
        # placed: 2 x 5 reads keep their class; strict: they are novel junction reads
        by = lambda v: summary[summary.version == v].groupby("context").reads.sum()
        assert by("strict")[NJ] - by("placed")[NJ] == 10 and by("placed")[NJ] == 6 and by("placed")[AS] == 4 and by("placed")[IG] == 2
        assert set(region.region) == set(acc.region) and "intron" in set(acc.region)
        no_model = pd.read_parquet(d / "sites_no_model.parquet")
        assert list(no_model.columns) == ["ref", "strand", "site", "region", "label", "n_accepted", "reason", "sub_cause"]
        assert _types(d / "sites_no_model.parquet") == dict(zip(no_model.columns, ["string"] * 2 + ["int64"] + ["string"] * 2 + ["int64"]
                                                                + ["string"] * 2))
        # near, donor_only and sites_not_joined have one read each at 1800: the first of SUB_CAUSES is the main one
        assert sorted(zip(no_model.strand, no_model.site, no_model.reason, no_model.sub_cause)) == sorted(
            [("+", 1800, "novel_junction", "near_annotated_intron"), ("-", G - 1800, "novel_junction", "near_annotated_intron"),
             ("+", 8080, "antisense", ""), ("-", G - 8080, "antisense", ""), ("+", 8400, "antisense", ""),
             ("-", G - 8400, "antisense", ""), ("+", 5080, "intergenic", ""), ("-", G - 5080, "intergenic", "")])
        merged = no_model.merge(sites, on=["ref", "strand", "site"], suffixes=("", "_s"))
        assert (merged.region == merged.region_s).all() and (merged.label == merged.label_s).all() and len(merged) == 8
        j = json.loads((d / "isoform_context.json").read_text())
        assert j["library"] == lib and j["refs"] == ["chrS", "chrE"] and j["accepted_reads"] == n_reads and j["sites"] == len(acc)
        assert j["records"] == {"records": n_reads + 5, "secondary_supplementary": 2, "low_mapq": 1, "used": n_reads + 2,
                                "accepted": n_reads, "empty_body": 1}
        assert j["tolerances"] == {"LONG_END_TOL": 10, "NEAR_INTRON_TOL": 5, "MIN_INTRON": 50, "FIVE_PRIME_TOL": 50, "NULL_SHIFT": 1000}
        assert j["acceptance"] == {"sites_and_accepted_reads_equal_stage16": True, "one_class_per_accepted_read": True,
                                   "intron_sites_off_model": 0}
        assert j["placed"]["reads_by_class"][NJ] == 6 and j["strict"]["reads_by_class"][NJ] == 16
        assert j["flagged_reads"]["chain_novel"] == 10 and j["per_ref"]["chrE"]["records"] == 0 and j["per_ref"]["chrS"]["seconds"] > 0
        # link to 1179: the molecules on the scanned references; 1340 is 10 nt from the site 1330, 1341 is 11 nt from it
        v = j["vs_1179"]
        assert v["real"]["molecules"] == 4 and v["real"]["matched"] == 2 and v["null_shift_nt"] == 1000 and v["null"]["matched"] == 0
        # tecap: the tagged BAM holds the accepted reads; the stand-in puts the spliced ones into one bucket
        with pysam.AlignmentFile(str(d / "tecap" / "accepted.bam")) as fh:
            keys = [a.get_tag(TAG) for a in fh.fetch()]
        assert len(keys) == n_reads and set(keys) == {f"{r}:{s}:{p}" for r, s, p in zip(acc.ref, acc.strand, acc.site)}
        assert sorted(p.name for p in (d / "tecap").iterdir()) == ["accepted.bam", "accepted.bam.bai", "out"]
        buckets, by_label = pd.read_parquet(d / "tecap_site_buckets.parquet"), pd.read_parquet(d / "tecap_buckets.parquet")
        assert list(buckets.columns) == ["ref", "strand", "site", "bucket", "reads"] and list(by_label.columns) == ["label", "bucket", "reads"]
        assert _types(d / "tecap_site_buckets.parquet") == dict(zip(buckets.columns, ["string"] * 2 + ["int64", "string", "int64"]))
        assert _types(d / "tecap_buckets.parquet") == {"label": "string", "bucket": "string", "reads": "int64"}
        assert buckets.reads.sum() == by_label.reads.sum() == n_spliced and set(buckets.bucket) == {"spliced"}
        per_label = j["tecap"]["per_label"]
        assert per_label == {k: {"written": int(g.n_accepted.sum()), "without_splice": w, "classified": int(g.n_accepted.sum()) - w}
                             for k, g in acc.groupby("label") for w in [per_label[k]["without_splice"]]}
        assert sum(v["without_splice"] for v in per_label.values()) == n_reads - n_spliced
        assert j["tecap"]["json_bytes"] == (d / "tecap" / "out" / f"{lib}_terminal_exon.json").stat().st_size
        assert j["tecap"]["sites_with_a_classified_read"] == buckets[["strand", "site"]].drop_duplicates().shape[0]
        # top class a no_model reason: the eight sites of sites_no_model.parquet; strict, also the two sites at 1900, where both
        # reads are chain_novel (at 1550 one of two is, and the tie stays with intron_after_junction)
        assert printed[lib] == {"accepted_reads": n_reads, "sites": len(acc), "sites_top_no_model": {"placed": 8, "strict": 10},
                                "tecap": per_label}
        assert at("+", 1900).context_without_chain_novel == NJ and at("+", 1550).context_without_chain_novel == "intron_after_junction"
    # pigeon cross-check, BD70 only: ends at +1805 (5 nt from the site 1800), +1811 (11 nt), chrE (no site) and, on minus,
    # at the site of the mirrored full_chain_annotated_end read; the isoform of chrX is not on a scanned reference
    assert not (out / "BD67" / "pigeon_crosscheck.parquet").exists()
    pigeon = pd.read_parquet(out / "BD70" / "pigeon_crosscheck.parquet")
    assert list(pigeon.itertuples(index=False, name=None)) == [
        ("full-splice_match", "reference_match", NJ, 1), ("full-splice_match", "reference_match", "none", 1),
        ("incomplete-splice_match", "5prime_fragment", "annotated_end", 1), ("not_classified", "not_classified", "none", 1)]
    assert _types(out / "BD70" / "pigeon_crosscheck.parquet") == {"structural_category": "string", "subcategory": "string",
                                                                  "context": "string", "isoforms": "int64"}
    j = json.loads((out / "BD70" / "isoform_context.json").read_text())
    assert j["pigeon"] == {"gtf": str(stage.PIGEON_GTF), "classification": str(stage.PIGEON_CLASS), "isoform_ends": 4,
                           "without_classification": 1, "matched": 2, "share_matched": 0.5}
    assert "pigeon" not in json.loads((out / "BD67" / "isoform_context.json").read_text())
    link = pd.read_parquet(out / "1179" / "truth" / "long_read_context.parquet")
    assert list(link.columns) == ["mol", "context_BD67", "context_BD70"] and link.mol.tolist() == [3, 5, 8, 9]
    assert link.context_BD67.tolist() == link.context_BD70.tolist() == ["CDS", "intron_unspliced", "none", "none"]
    assert _types(out / "1179" / "truth" / "long_read_context.parquet") == {"mol": "int64", "context_BD67": "string",
                                                                           "context_BD70": "string"}
    assert not list(base.rglob("isoform_context*")) and not list(base.rglob("tecap*"))       # the inputs folder is only read


def test_stage30_without_tecap_and_one_library(stage30_inputs, tmp_path, monkeypatch, capsys):
    stage, base, s16 = stage30_inputs
    out = tmp_path / "out"
    printed = _run30(stage, monkeypatch, capsys, base, out, "--libs", "BD67", "--no-tecap")
    assert list(printed) == ["BD67", "seconds"] and "tecap" not in printed["BD67"]
    assert sorted(p.name for p in (out / "BD67").iterdir()) == ["isoform_context.json", "isoform_context_sites.parquet",
                                                                "isoform_context_summary.parquet", "region_by_context.parquet",
                                                                "sites_no_model.parquet"]
    assert "tecap" not in json.loads((out / "BD67" / "isoform_context.json").read_text()) and not (out / "BD70").exists()
    assert list(pd.read_parquet(out / "1179" / "truth" / "long_read_context.parquet").columns) == ["mol", "context_BD67"]


def test_stage30_stops_when_an_acceptance_condition_fails(stage30_inputs, tmp_path, monkeypatch, capsys):
    stage, base, s16 = stage30_inputs
    out = tmp_path / "out"
    # stage 16 knows a site of BD70 that this scan does not find: nothing is written for either library
    extra = s16.iloc[:1].assign(site=77, n_accepted=1)
    pd.concat([s16, extra], ignore_index=True).to_parquet(base / "BD70" / "long_read_sites.parquet", index=False)
    with pytest.raises(SystemExit, match="BD70: 1 sites of stage 16 without reads here, 0 sites here that stage 16 lacks"):
        _run30(stage, monkeypatch, capsys, base, out, "--no-tecap")
    assert not list(out.rglob("*.parquet")) and not list(out.rglob("*.json"))
    # a site of the region intron whose top class is antisense is listed and stops the stage
    s16.assign(region=np.where((s16.strand == "+") & (s16.site == 8080), "intron", s16.region)).to_parquet(
        base / "BD70" / "long_read_sites.parquet", index=False)
    with pytest.raises(SystemExit, match=r"(?s)BD70: 1 sites of the region intron .*chrS +\+ +8080 +1 +" + AS):
        _run30(stage, monkeypatch, capsys, base, out, "--no-tecap")
    assert not list(out.rglob("*.parquet")) and not list(out.rglob("*.json"))
