"""Synthetic locus and designed long reads for stage 30 (context of long-read 3' ends): one read per class and per rule of
design 3.8, on both strands. The placement reads and the transcripts TA, TB and TC are those of the prototype's self-test
(design_m3_draft/design/checks/b1_tier2_longread_context.py)."""
import random

import pysam

from pafbench.lrcontext import CLASSES, SUB_CAUSES, ReadContext
from pafbench.sites import COMP

G = 10000   # the minus-strand loci are the plus-strand loci mirrored: x -> G - x
CHROMS = {"chrS": 12000, "chrE": 3000, "chrA": 2000}   # chrE and chrA have no transcript
HDR = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6", "SO": "coordinate"},
                                       "SQ": [{"SN": c, "LN": n} for c, n in CHROMS.items()]})
NJ, AS, IG = CLASSES[8:]
NEAR, SHORT, OPPOSITE, NOT_JOINED, DONOR, ACCEPTOR, NEITHER = SUB_CAUSES

# transcript: (exons, CDS span or None, tagged mRNA_end_NF); 0-based half-open, as on the plus strand. TC is the decoy that
# holds the last intron of TB and ends where two of the designed reads end. TS, TL and TN share one intron and differ in
# their 3' end; TN is tagged mRNA_end_NF.
LOCUS = {
    "TA": ([(1000, 1100), (1200, 1300), (1400, 1500), (1600, 2000)], (1050, 1700), False),
    "TB": ([(900, 950), (1200, 1300), (1400, 1500), (1700, 2100)], (1220, 1750), False),
    "TC": ([(1350, 1500), (1700, 1900)], None, False),
    "TS": ([(3000, 3200), (3400, 3600)], (3050, 3450), False),
    "TL": ([(3000, 3200), (3400, 3900)], (3050, 3450), False),
    "TN": ([(3000, 3200), (3400, 3500)], (3050, 3450), True),
}


def ctx(klass, transcript="", matched=0, chain_novel=0, **flags):
    return ReadContext(klass, transcript, matched, chain_novel, **flags)


# name: (aligned blocks as on the plus strand, site, expected context). The aligned bases from the site on are A's and
# belong to the terminal run, as do 12 clipped A's; 33 clipped bases follow (see long_read).
DESIGNED = {
    # one read per class with a model. TA: |1050 - 1000| = 50, so the 5' end is just complete, and 1050 is its first CDS base
    "full_chain_3UTR": ([(1050, 1100), (1200, 1300), (1400, 1500), (1600, 1812)], 1800,
                        ctx("3UTR", "TA", 3, complete_5p=True, covers_start=True)),
    # 5' truncated (starts in exon 2): classed by its end, not 5' complete
    "full_chain_annotated_end": ([(1250, 1300), (1400, 1500), (1600, 2008)], 1996, ctx("annotated_end", "TA", 2)),
    "full_chain_CDS": ([(1050, 1100), (1200, 1300), (1400, 1500), (1600, 1662)], 1650,
                       ctx("CDS", "TA", 3, complete_5p=True, covers_start=True)),
    # TA and TB contain it; TA has 25 exonic nt of it and TB none (the end is in TB's first intron)
    "unspliced_5UTR": ([(1005, 1042)], 1030, ctx("5UTR", "TA", complete_5p=True, other_class=True)),
    # in the exon of TC that TA and TB have as an intron: TC has the larger exonic overlap
    "unspliced_ncexon": ([(1360, 1407)], 1395, ctx("ncexon", "TC", complete_5p=True, other_class=True)),
    # wholly inside the intron 1300..1400 of TA and TB (TC starts at 1350 and does not contain it): TA has the nearer end
    "unspliced_in_intron": ([(1310, 1392)], 1380, ctx("intron_unspliced", "TA")),
    # from exon 2 into the intron
    "unspliced_exon_into_intron": ([(1260, 1342)], 1330, ctx("intron_unspliced", "TA")),
    # spliced, runs past the donor at 1500 and ends in the next intron
    "spliced_past_donor": ([(1250, 1300), (1400, 1562)], 1550, ctx("intron_after_junction", "TA", 1)),
    # TB holds both introns, TC only the last one; the end is 30 nt past TB's last base
    "spliced_downstream": ([(1250, 1300), (1400, 1500), (1700, 2142)], 2130, ctx("downstream", "TB", 2)),
    # overlaps the span of TB and ends past it: downstream, no transcript taken
    "unspliced_downstream": ([(2080, 2162)], 2150, ctx("downstream")),
    # no span contains the read, the span of TB contains its end; 900 - 860 = 40, within the 5' tolerance
    "unspliced_starts_outside": ([(860, 942)], 930, ctx("5UTR", "TB", starts_outside=True, complete_5p=True)),
    # the block after the last junction runs over the intron 1500..1600 of TA and ends in its last exon
    "retained_after_junction": ([(1050, 1100), (1200, 1300), (1400, 1662)], 1650,
                                ctx("CDS", "TA", 2, retained_after_junction=True, complete_5p=True, covers_start=True)),
    # the terminal A run crosses the last junction: the body ends at 1490 and has two introns
    "run_across_junction": ([(1050, 1100), (1200, 1300), (1400, 1500), (1600, 1612)], 1490,
                            ctx("CDS", "TA", 2, complete_5p=True, covers_start=True)),
    # TA, TB and TC have 40 exonic nt each and none ends here: the nearest annotated 3' end (TC, 1900) decides
    "unspliced_nearest_end": ([(1410, 1462)], 1450, ctx("ncexon", "TC", other_class=True)),
    # the reads of the prototype's self-test for the placement by the 3' part. The first one starts 20 nt from the 5' end
    # of TA and covers its first CDS base, and is still neither 5' complete nor counted as covering the start
    "novel_first_intron_last_two_annotated": ([(1020, 1080), (1200, 1300), (1400, 1500), (1600, 1662)], 1650,
                                              ctx("CDS", "TA", 2, 2)),
    "last_intron_novel": ([(1050, 1100), (1200, 1300), (1400, 1500), (1650, 1812)], 1800, ctx(NJ, sub_cause=DONOR)),
    "last_intron_3nt_off": ([(1050, 1100), (1200, 1300), (1400, 1500), (1603, 1812)], 1800, ctx(NJ, sub_cause=NEAR)),
    # introns of TA (first two) then of TB (last). TC also holds the last intron and ends within the tolerance of the
    # read end, so a rule that took every transcript with the last intron would answer annotated_end
    "all_annotated_two_transcripts": ([(1050, 1100), (1200, 1300), (1400, 1500), (1700, 1912)], 1900, ctx("3UTR", "TB", 2, 2)),
    "suffix_1_two_holders": ([(1050, 1100), (1200, 1300), (1420, 1500), (1700, 1912)], 1900,
                             ctx("annotated_end", "TC", 1, 1, other_class=True)),
    "anchored_ends_in_intron": ([(1020, 1080), (1200, 1300), (1400, 1562)], 1550, ctx("intron_after_junction", "TA", 1, 1)),
    # exon 2 of TA skipped: not compatible with TA, placed by its last intron only
    "skips_upstream_exon": ([(1050, 1100), (1400, 1500), (1600, 1812)], 1800, ctx("3UTR", "TA", 1, 1)),
    # exon 3 skipped: the last intron joins two annotated splice sites that no transcript joins
    "skips_last_internal_exon": ([(1050, 1100), (1200, 1300), (1600, 1812)], 1800, ctx(NJ, sub_cause=NOT_JOINED)),
    # no transcript of the strand, inside the mirrored locus of the other strand
    "antisense": ([(8000, 8092)], 8080, ctx(AS)),
    "antisense_spliced": ([(8000, 8100), (8300, 8412)], 8400, ctx(AS)),
    "intergenic": ([(5000, 5092)], 5080, ctx(IG)),
    # ends 5 nt past the last base of TS (3599): annotated_end, although TL covers the end with more exonic nt (3UTR)
    "end_5nt_past_short_transcript": ([(3010, 3200), (3400, 3617)], 3605,
                                      ctx("annotated_end", "TS", 1, complete_5p=True, covers_start=True, other_class=True)),
    # ends 3 nt before the last base of TN (3499), which is tagged mRNA_end_NF
    "end_of_NF_transcript": ([(3010, 3200), (3400, 3509)], 3497,
                             ctx("annotated_end", "TN", 1, end_nf=True, complete_5p=True, covers_start=True, other_class=True)),
}


def write_gtf(path):
    lines = ["#synthetic\n"]
    for minus in (False, True):
        for tid, (exons, cds, nf) in LOCUS.items():
            name = tid + ("m" if minus else "")
            attrs = f'gene_id "g{name}"; transcript_id "{name}"; tag "basic";' + (' tag "mRNA_end_NF";' if nf else "")

            def row(feature, s, e, attrs=attrs):
                s, e = (G - e, G - s) if minus else (s, e)
                return f"chrS\tsyn\t{feature}\t{s + 1}\t{e}\t.\t{'-' if minus else '+'}\t.\t{attrs}\n"

            lines.append(row("gene", exons[0][0], exons[-1][1], f'gene_id "g{name}";'))
            lines.append(row("transcript", exons[0][0], exons[-1][1]))
            lines += [row("exon", s, e) for s, e in exons]
            if cds:
                lines += [row("CDS", max(s, cds[0]), min(e, cds[1])) for s, e in exons if s < cds[1] and e > cds[0]]
    path.write_text("".join(lines))
    return path


def write_fasta(path, seed=11):
    rng = random.Random(seed)
    with open(path, "w") as fh:
        for c, n in CHROMS.items():
            fh.write(f">{c}\n" + "".join(rng.choice("ACGT") for _ in range(n)) + "\n")
    pysam.faidx(str(path))
    return path


def long_read(blocks, site, minus=False, name="r", run=12, after=33, mapq=60, flag=0):
    """Long read (transcript-oriented, as FLNC reads are) with these aligned blocks and this site, given as on the plus
    strand and mirrored for minus. Sense sequence: 7 clipped bases, the aligned bases (C before the site, A from it on),
    then a clip of `run` A's and `after` C's; the defaults make an accepted read. A block longer than 20 nt whose first half
    lies before the site holds one deletion and one insertion, which must not split it."""
    ops, seq, last = [(4, 7)], ["G" * 7], None
    for s, e in blocks:
        if last is not None:
            ops.append((3, s - last))
        h, last = (e - s) // 2, e
        bases = "".join("A" if x >= site else "C" for x in range(s, e))
        if e - s > 20 and s + h <= site:
            ops += [(0, h - 2), (2, 2), (1, 3), (0, e - s - h)]
            bases = bases[:h - 2] + "CCC" + bases[h:]
        else:
            ops.append((0, e - s))
        seq.append(bases)
    ops.append((4, run + after))
    seq = "".join(seq) + "A" * run + "C" * after
    a = pysam.AlignedSegment(HDR)
    a.query_name, a.reference_name, a.flag, a.mapping_quality = name, "chrS", (16 if minus else 0) | flag, mapq
    a.reference_start = G - blocks[-1][1] if minus else blocks[0][0]
    a.cigartuples = ops[::-1] if minus else ops
    a.query_sequence = seq.translate(COMP)[::-1] if minus else seq
    return a


def expected(name, minus):
    want = DESIGNED[name][2]
    return want._replace(transcript=want.transcript + "m") if minus and want.transcript else want


def write_bam(path, reads):
    """Coordinate-sorted, indexed BAM of these records."""
    unsorted = path.with_suffix(".unsorted.bam")
    with pysam.AlignmentFile(str(unsorted), "wb", header=HDR) as out:
        for a in reads:
            out.write(a)
    pysam.sort("-o", str(path), str(unsorted))
    pysam.index(str(path))
    return path
