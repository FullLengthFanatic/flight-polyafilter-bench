"""Small synthetic genome, annotation and BAM for exactness tests against the original polyAfilter, and a 10x variant of
the BAM with a Cell Ranger outs folder (build_10x) for the read-group stages of milestone 3."""
import gzip
import random
from dataclasses import dataclass
from pathlib import Path

import gffutils
import h5py
import numpy as np
import pandas as pd
import pysam

from pafbench.flags import Setting, flag_reference, removed_keys, setting_tag
from pafbench.transcripts import load_transcripts
from pafbench.vendor import polyafilter_modules

CHROMS = {"chrA": 6000, "chrB": 2000}
HEADER = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": c, "LN": n} for c, n in CHROMS.items()]}

GTF_ROWS = [
    ("chrA", "gene", 1001, 4000, "+", 'gene_id "g1";'),
    ("chrA", "transcript", 1001, 4000, "+", 'gene_id "g1"; transcript_id "t1";'),
    ("chrA", "exon", 1001, 1400, "+", 'gene_id "g1"; transcript_id "t1";'),
    ("chrA", "exon", 1801, 2600, "+", 'gene_id "g1"; transcript_id "t1";'),
    ("chrA", "exon", 3001, 4000, "+", 'gene_id "g1"; transcript_id "t1";'),
    ("chrA", "transcript", 1001, 2700, "+", 'gene_id "g1"; transcript_id "t2";'),
    ("chrA", "exon", 1001, 1400, "+", 'gene_id "g1"; transcript_id "t2";'),
    ("chrA", "exon", 1801, 2700, "+", 'gene_id "g1"; transcript_id "t2";'),
    ("chrA", "gene", 4201, 5800, "-", 'gene_id "g2";'),
    ("chrA", "transcript", 4201, 5800, "-", 'gene_id "g2"; transcript_id "t3";'),
    ("chrA", "exon", 4201, 4700, "-", 'gene_id "g2"; transcript_id "t3";'),
    ("chrA", "exon", 5001, 5800, "-", 'gene_id "g2"; transcript_id "t3";'),
    ("chrB", "gene", 201, 1500, "+", 'gene_id "g3";'),
    ("chrB", "transcript", 201, 1500, "+", 'gene_id "g3"; transcript_id "t4";'),
    ("chrB", "exon", 201, 1500, "+", 'gene_id "g3"; transcript_id "t4";'),
    ("chrB", "gene", 1601, 1680, "-", 'gene_id "g4";'),
    ("chrB", "transcript", 1601, 1680, "-", 'gene_id "g4"; transcript_id "t5";'),
    ("chrB", "exon", 1601, 1680, "-", 'gene_id "g4"; transcript_id "t5";'),
]

# (contig, 0-based position, sequence) planted over the random background
PLANTS = [
    ("chrA", 0, "AAAAAA"),
    ("chrA", 1299, "A" * 12),
    ("chrA", 1599, "A" * 10),
    ("chrA", 1999, "AAAAAGAAAAAA"),
    ("chrA", 2299, "aaaaaaaaaa"),
    ("chrA", 2549, "A" * 8),
    ("chrA", 3199, "NNNNN"),
    ("chrA", 3499, "A" * 15),
    ("chrA", 4499, "T" * 10),
    ("chrA", 5299, "T" * 12),
    ("chrA", 5994, "TTTTTT"),
    ("chrB", 1640, "T" * 9),
]


@dataclass
class Synth:
    tmp: Path
    fasta: Path
    gtf: Path
    db: Path
    bam: Path


def _reads(rng, hdr):
    reads = []

    def add(ref, pos, cigar, rev, length=90, extra=0, seq=True, n_base=False):
        a = pysam.AlignedSegment(hdr)
        a.query_name = f"r{len(reads)}"
        a.reference_name = ref
        a.reference_start = pos
        a.cigarstring = cigar
        a.flag = (16 if rev else 0) | extra
        a.mapping_quality = 255
        if seq:
            s = "".join(rng.choice("ACGT") for _ in range(length))
            if n_base:
                s = s[:10] + "N" + s[11:]
            a.query_sequence = s
            a.query_qualities = pysam.qualitystring_to_array("I" * length)
        reads.append(a)

    # Stops short of 4000 (not 4000, per the plan) so g1's 3' end (3778-4000) has
    # no coverage: the original's getTransEndSensSpec divides by Neg (uncovered
    # exonic bp across all expressed transcripts) and crashes if every base of
    # every expressed transcript is covered, which full-range tiling guarantees.
    # eachTransStartByStrdRef (what scanBAMfilter actually uses) depends only on
    # the GTF, not on this coverage, so the gap has no effect on which reads are
    # identified for removal.
    for pos in range(950, 3700, 37):
        add("chrA", pos, "90M", False)
    for pos in range(1340, 1390, 10):
        left = 1400 - pos
        add("chrA", pos, f"{left}M400N{90 - left}M", False)
    for pos in range(4150, 5800, 41):
        add("chrA", pos, "90M", True)
    for pos in range(1200, 3900, 300):
        add("chrA", pos, "90M", True)
    for pos in range(1250, 1550, 60):
        add("chrA", pos, "90M", False, extra=256, seq=False)
    for pos in range(3400, 3600, 50):
        add("chrA", pos, "90M", False, extra=1024)
    for pos in range(2900, 3100, 50):
        add("chrA", pos, "90M", False, n_base=True)
    add("chrA", 1950, "40M2I48M", False)
    add("chrA", 2450, "40M3D50M", False)
    for pos in range(1590, 1660, 20):
        add("chrB", pos, "60M", True, length=60)
    return reads


def _genome(tmp, rng):
    """genome.fa (indexed), genes.gtf and genes.db under tmp; rng gives the background between the planted stretches."""
    seqs = {c: [rng.choice("ACGT") for _ in range(n)] for c, n in CHROMS.items()}
    for c, p, s in PLANTS:
        seqs[c][p:p + len(s)] = list(s)
    fasta = tmp / "genome.fa"
    with open(fasta, "w") as fh:
        for c, s in seqs.items():
            s = "".join(s)
            fh.write(f">{c}\n")
            for k in range(0, len(s), 60):
                fh.write(s[k:k + 60] + "\n")
    pysam.faidx(str(fasta))
    gtf = tmp / "genes.gtf"
    with open(gtf, "w") as fh:
        for c, ft, s, e, strand, attr in GTF_ROWS:
            fh.write(f"{c}\tsynth\t{ft}\t{s}\t{e}\t.\t{strand}\t.\t{attr}\n")
    db = tmp / "genes.db"
    gffutils.create_db(str(gtf), str(db), merge_strategy="create_unique")
    return fasta, gtf, db


def _sorted_bam(tmp, hdr, reads, name="reads.bam"):
    unsorted = tmp / "unsorted.bam"
    with pysam.AlignmentFile(str(unsorted), "wb", header=hdr) as out:
        for a in reads:
            out.write(a)
    bam = tmp / name
    pysam.sort("-o", str(bam), str(unsorted))
    pysam.index(str(bam))
    return bam


def build(tmp: Path, seed: int = 7) -> Synth:
    rng = random.Random(seed)
    fasta, gtf, db = _genome(tmp, rng)
    hdr = pysam.AlignmentHeader.from_dict(HEADER)
    return Synth(tmp, fasta, gtf, db, _sorted_bam(tmp, hdr, _reads(rng, hdr)))


CELLS_10X = {"c1": "AAACCCAAGGAGAGTA-1", "c2": "AACGGGATCCTTGACC-1", "c3": "ACGTTGCACATGGTAC-1", "c4": "CCTTGGAACGTATCGA-1",
             "c5": "GGATCCAGTCAATGCC-1"}  # the called cells
# sorts between two called barcodes: in molecule_info.h5 the barcode index of a called cell is then not its row in
# barcodes.tsv.gz (our observation: pass_filter of 1417 starts 281, 346, 363 among 1,006,747 barcodes)
UNCALLED_10X = "AAAGGGCCAACATTGA-1"
# Test settings of the 10x fixture: default mode, -i, and the remove-all control of each. With MINSNRLEN 8 and no mismatch
# the windows of seed 7 come from the planted stretches alone (chrA only), and -i adds the intronic stretch at 1599.
SETTINGS_10X = (Setting(200, 8, 0, False), Setting(200, 8, 0, True), Setting(200, None, 0, False), Setting(200, None, 0, True))
# @CO lines of a Cell Ranger 9.0.1 BAM (our observation: samtools view -H on 1417's possorted_genome_bam.bam)
CO_10X = ["10x_bam_to_fastq:R1(CR:CY,UR:UY)", "10x_bam_to_fastq:R2(SEQ:QUAL)",
          'library_info:{"library_id":0,"library_type":"Gene Expression","gem_group":1,"target_set_name":null}']
# Gene the documented counting rule gives each case (one cell and one corrected UMI): before filtering, then at each of
# SETTINGS_10X. Written by hand from the design of the reads; tests/test_synth10x.py checks it against a naive count.
RULE_10X = {
    "one_inside": ("g1", None, None, None, None),
    "two_one_inside": ("g1", "g1", "g1", None, None),
    "spliced": ("g1", "g1", "g1", None, None),
    "protected": ("g1", "g1", "g1", "g1", "g1"),
    "intronic": ("g1", "g1", None, "g1", None),
    "exon_intron": ("g1", "g1", "g1", "g1", None),
    "pa_only": ("g1", "g1", "g1", None, None),
    "pa_and_member": ("g1", "g1", "g1", "g1", "g1"),
    "three_vs_one": ("g1", None, "g3", None, "g3"),
    "leads_by_one": ("g1", None, None, "g2", "g2"),
    "tie": (None, None, None, None, None),
    "tie_broken": (None, None, "g2", None, "g2"),
    "g3_two": ("g3", "g3", "g3", None, None),
    "three_genes": ("g1", None, "g3", None, "g3"),
    "three_vs_two": ("g1", "g2", "g2", "g2", "g2"),
    "g2_inside": ("g2", None, None, None, None),
    "alone_xf19": ("g3", "g3", "g3", "g3", "g3"),
    "alone_xf19_partner": ("g1", "g1", "g1", "g1", "g1"),
    "g2_partial": ("g2", "g2", "g2", None, None),
    "winner_keeps": ("g1", "g1", "g1", "g1", "g1"),
    "chain": (None, None, None, None, None),
    "chain_end": ("g1", "g1", "g1", "g1", "g1"),
    "minus_pa": ("g2", "g2", "g2", "g2", "g2"),
}


@dataclass
class Synth10x(Synth):
    outs: Path           # the BAM's folder, with filtered_feature_bc_matrix/ and molecule_info.h5 as Cell Ranger writes them
    reads: pd.DataFrame  # one row per record: qname, case, ref, pos, end, reverse, cb, ur, ub, gx, re, xf, pa, removed per tag
    # one row per read group: case, cb, ub, gene, n_reads, n_winner_reads, n_pa, re_class, end3, end3_nonpa, rm_<tag>
    groups: pd.DataFrame
    events: pd.DataFrame  # designed, one row per cell, UMI and tag: case, cb, ub, tag, before, after, lost, gained, switched


def _reads_10x(rng, hdr):
    """Designed records with the tags of a Cell Ranger 9.0.1 BAM, and one table row per record.

    Positions are 0-based starts. Pieces at SETTINGS_10X (seed 7; e default mode, i with -i, then the two controls):
    chrA forward (g1)  e (1098,1298) (1397,1400) (2099,2299) (2349,2500) (3298,3498); i the same with (1397,1597);
                       controls (1000,1400) (1800,2500) (3000,3800) and (1000,2500) (2700,3800)
    chrA reverse (g2)  e (4509,4700) (5311,5511); i (4509,4709) (5311,5511); controls (4400,4700) (5000,5800) and (4400,5800)
    chrB forward (g3)  none; both controls (200,1300). chrB reverse (g4): none.
    So a 90 nt read of g1 at 1150, 2150, 2400 or 3350 is removed at all four, at 1005, 1810, 3020, 3600 or 3700 by the
    controls only, at 1450 (intron) with -i only, and at 2505 or from 3800 on never (protected 3' ends of t2 and t1).

    xf bits, Cell Ranger 9.0 BAM page (copy in design_m3_draft/research/dl/bam.txt;
    https://www.10xgenomics.com/support/software/cell-ranger/9.0/analysis/outputs/cr-outputs-bam): "1 - The read is
    confidently mapped to the transcriptome 2 - This read's barcode, UMI, and feature combination was discarded in favor of
    a different feature with higher read support", "8 - This read is representative for a molecule and is treated as a UMI
    count 16 - This read maps to exactly one feature". So 25 is on the representative read of a counted group, 17 on its
    other reads, 19 on every read of a discarded group and 0 on a read that is not confidently mapped to one gene; our
    observation: the called cells of the three libraries carry these four values only (design of 2026-10-12, section 2.3).
    A UMI with a base below quality 10 is not valid and gets no UB (cellranger 9.0.1 source, `const UMI_MIN_QV: u8 = 10;`,
    lib/rust/umi/src/info.rs line 6, and `if self.umi_info.is_valid {` before the UB tag, lib/rust/tx_annotation/src/read.rs
    line 549); our observation: such a record keeps xf 17. The duplicate flag is on the xf 17 records with a CB and a UB
    (our observation, design_m3_draft/design/checks/dupflag_vs_xf_<lib>.json).

    pa is "The number of poly-A nucleotides trimmed from the 3' end of read 2." (same page). Our observation on 1417,
    chr21:33 to 36 Mb: the trimmed bases stay in the record as a soft clip, all 175 pa reads are 90 nt long with a soft clip
    of at least pa bases at their 3' end (153 of exactly pa), for example 53M37S with pa 36 and, on the reverse strand,
    21S16M1D53M with pa 21. AN, same page: "Present for reads that are aligned to the antisense strand of annotated
    transcripts. If intron counts are not included (with include-introns=false ), this tag is the same as the TX tag but
    with - values for the strand identifier."; a TX entry is "[transcript_id] , [strand][pos] , [cigar]" and "pos is the
    alignment offset in transcript coordinates". Our observation in that region of 1417, a library with counted intronic
    reads: an exonic antisense record carries the transcript form (ENST00000342101,-130,89M1S;...), an intronic one the
    gene form (ENSG00000227757,-). The pos written here is the 0-based offset of the record's first base in the
    transcript; how Cell Ranger counts it on an antisense record was not checked.

    Left out on purpose, although every real record or header has them (our observation, tag order of 1417: NH HI AS nM,
    ts or pa, RG TX GX GN fx RE, AN, mm, xf CR CY CB UR UY UB): the tags NH, HI, AS, nM, RG, TX and fx, the tags ts and mm
    of some records, and the @RG header line; the header says VN 1.6 (HEADER, shared with build()) where the real one says
    1.4.
    """
    place = {attr.split('"')[1]: (c, strand == "-") for c, ft, _, _, strand, attr in GTF_ROWS if ft == "gene"}
    cells = CELLS_10X | {"u": UNCALLED_10X}
    reads, rows = [], []

    def add(case, cell, umi, gene, pos, xf, cigar="90M", re="E", ub=None, uy="F" * 12, pa=0, gx=None, an=None, antisense=False,
            mapq=255):
        """gene places the record (its contig and strand) and is its GX unless gx, antisense or RE I say otherwise (gx=""
        leaves the tag out); an is the AN tag, the gene form on an antisense record unless given; ub is given only where
        the corrected UMI differs from the raw one."""
        ref, reverse = place[gene]
        reverse ^= antisense
        cb = cells.get(cell)
        gx = None if antisense or re == "I" else gene if gx is None else gx or None
        an = an or (f"{gene},-" if antisense else None)
        ub = (ub or umi) if min(ord(q) - 33 for q in uy) >= 10 else None
        a = pysam.AlignedSegment(hdr)
        a.query_name = f"{case}:{len(reads):03d}"
        a.reference_name = ref
        a.reference_start = pos
        a.cigarstring = cigar
        a.flag = (16 if reverse else 0) | (1024 if xf == 17 and cb and ub else 0)
        a.mapping_quality = mapq
        length = a.infer_query_length()
        a.query_sequence = "".join(rng.choice("ACGT") for _ in range(length))
        a.query_qualities = pysam.qualitystring_to_array("I" * length)
        if pa:
            a.set_tag("pa", pa, value_type="i")
        if gx:
            a.set_tag("GX", gx)
            a.set_tag("GN", gx.upper())
        a.set_tag("RE", re, value_type="A")
        if an:
            a.set_tag("AN", an)
        a.set_tag("xf", xf, value_type="i")
        a.set_tag("CR", cb[:-2] if cb else "AAACCCAAGGAGNGTA")
        a.set_tag("CY", "F" * 16)
        if cb:
            a.set_tag("CB", cb)
        a.set_tag("UR", umi)
        a.set_tag("UY", uy)
        if ub:
            a.set_tag("UB", ub)
        reads.append(a)
        rows.append({"qname": a.query_name, "case": case, "ref": ref, "pos": pos, "end": a.reference_end, "reverse": reverse,
                     "cb": cb, "ur": umi, "ub": ub, "gx": gx, "re": re, "xf": xf, "pa": pa})

    # One read inside a removal window: the UMI is lost at every setting.
    add("one_inside", "c1", "AACCGGTTAACC", "g1", 1150, 25)
    # Two reads with one inside: kept by the settings, lost at the controls. A third record of the same cell and raw UMI
    # has a base below quality 10 in UY and belongs to no read group.
    add("two_one_inside", "c1", "ACACACGTGTGT", "g1", 2150, 25)
    add("two_one_inside", "c1", "ACACACGTGTGT", "g1", 1810, 17)
    add("no_ub", "c1", "ACACACGTGTGT", "g1", 3350, 17, uy="FFFFFF$FFFFF")
    # Spliced molecule: a read inside a window in exon 2, one over the 400 nt intron and one in exon 3.
    add("spliced", "c1", "AGTCAGTCAGTC", "g1", 2400, 25)
    add("spliced", "c1", "AGTCAGTCAGTC", "g1", 2560, 17, cigar="40M400N50M")
    add("spliced", "c1", "AGTCAGTCAGTC", "g1", 3020, 17)
    # A read in the protected 3' end of t2, over the planted stretch at 2549: never removed.
    add("protected", "c1", "CATGCATGGTAC", "g1", 2505, 25)
    # Intronic molecule that only -i removes, and a molecule with an exonic and an intronic read (re_class 3). The first
    # carries an AN next to its GX, as a counted read that is antisense to another gene does (our observation on 1417,
    # chr21:33 to 36 Mb: 5,025 of 88,073 records carry GX and AN, 4,946 of them with xf 25 or 17, for example GX
    # ENSG00000227757, RE N, AN ENSG00000205930,-, xf 25). No gene of this annotation overlaps g1; the tag names g2.
    add("intronic", "c1", "CCGGAATTCCGG", "g1", 1450, 25, re="N", an="g2,-")
    # the intronic read has the smaller name; the exonic one is still the UMI count: UmiSelectKey compares the UmiType
    # before the read name (cellranger 9.0.1, lib/rust/tx_annotation/src/mark_dups.rs lines 110 to 114 and 137 to 146), and
    # Txomic is declared before NonTxomic (lib/rust/umi/src/lib.rs lines 102 to 106)
    add("exon_intron", "c1", "CGATCGATTAGC", "g1", 1450, 17, re="N")
    add("exon_intron", "c1", "CGATCGATTAGC", "g1", 1005, 25)
    # pa reads: alone in their group (no read without pa), and as the 3'-most read of a group on each strand. 60 nt are
    # aligned and the 30 at the 3' end of the read soft-clipped, so the 3'-most aligned base is not start + 89.
    add("pa_only", "c1", "GAGATCTCTAGA", "g1", 3600, 25, cigar="60M30S", pa=30)
    add("pa_and_member", "c1", "GTTGCAACGTTG", "g1", 3880, 25, cigar="60M30S", pa=30)
    add("pa_and_member", "c1", "GTTGCAACGTTG", "g1", 3700, 17)
    add("minus_pa", "c5", "TCGATCGAAGCT", "g2", 4210, 25, cigar="30S60M", pa=28)
    add("minus_pa", "c5", "TCGATCGAAGCT", "g2", 4290, 17)
    # Records of called cells that are in no read group: two genes in GX; one gene but MAPQ below 255 and xf 0; antisense
    # (AN, no GX); intronic on the sense strand without GX and AN, with and without a UB, which -i removes (the intronic
    # records of a library counted without introns; our observation on 1179, chr21:33.2 to 33.3 Mb: of 19,319 records 8,480
    # are RE N with xf 0, a UB and neither GX nor AN, and 213 the same without a UB); intergenic. All sit where a gene read
    # would be removed or kept, so they count in the record totals.
    add("multi_gene", "c1", "TACGTACGATCG", "g1", 1150, 0, gx="g1;g2")
    add("multimapped", "c1", "TCAGGACTTCAG", "g1", 2150, 0, mapq=3)
    add("antisense", "c1", "TGCATGCAACGT", "g1", 1200, 0, antisense=True, an="t1,-200,90M;t2,-200,90M")
    add("antisense", "c1", "TTGGCCAAGGTT", "g1", 1500, 0, re="N", antisense=True)
    add("intronic_no_gx", "c1", "GGCCTTAAGGCC", "g1", 1450, 0, re="N", gx="")
    add("intronic_no_gx", "c1", "GTACCATGGTAC", "g1", 1450, 0, re="N", gx="", uy="FFF$FFFFFFFF")
    add("intergenic", "c1", "ATATGCGCATAT", "g1", 4050, 0, re="I")
    add("intergenic", "c2", "ATTACGGCATTA", "g1", 100, 0, re="I")
    # Not a called cell: a barcode outside barcodes.tsv.gz with counted molecules, and a record without CB (our observation
    # on 1417, chr22:20.0 to 21.5 Mb: such records carry xf 17, a UB equal to UR and no duplicate flag).
    add("uncalled", "u", "CTAGCTAGGATC", "g1", 1150, 25)
    add("uncalled", "u", "CTAGCTAGGATC", "g1", 1810, 17)
    add("uncalled", "u", "GACTGACTCAGT", "g4", 1610, 25, cigar="60M")
    add("no_cb", None, "CAGTCAGTTGAC", "g1", 1005, 17)

    # Two genes on one cell and UMI. Winner with three reads against a loser with one on the other reference: default
    # mode removes two reads of the winner (a tie, the UMI is lost), -i the third as well (the UMI moves to g3).
    add("three_vs_one", "c2", "AAGGTTCCAAGG", "g1", 1150, 25)
    add("three_vs_one", "c2", "AAGGTTCCAAGG", "g1", 3350, 17)
    add("three_vs_one", "c2", "AAGGTTCCAAGG", "g1", 1450, 17, re="N")
    add("three_vs_one", "c2", "AAGGTTCCAAGG", "g3", 1400, 19)
    # Winner that leads by one read: one read removed makes a tie; the controls remove both and the UMI moves to g2.
    add("leads_by_one", "c2", "ACGGTACCGTAA", "g1", 2150, 25)
    add("leads_by_one", "c2", "ACGGTACCGTAA", "g1", 3020, 17)
    add("leads_by_one", "c2", "ACGGTACCGTAA", "g2", 4290, 19)
    # Tie that counts nothing: two reads each, all in protected 3' ends.
    add("tie", "c2", "AGCTTCGAAGCT", "g1", 3810, 19)
    add("tie", "c2", "AGCTTCGAAGCT", "g1", 3880, 19)
    add("tie", "c2", "AGCTTCGAAGCT", "g2", 4210, 19)
    add("tie", "c2", "AGCTTCGAAGCT", "g2", 4290, 19)
    # Tie that the filter breaks: -i removes the intronic read of g1 and g2 gains the UMI.
    add("tie_broken", "c2", "CAACGTTGGCAA", "g1", 1450, 19, re="N")
    add("tie_broken", "c2", "CAACGTTGGCAA", "g1", 2505, 19)
    add("tie_broken", "c2", "CAACGTTGGCAA", "g2", 4210, 19)
    add("tie_broken", "c2", "CAACGTTGGCAA", "g2", 4290, 19)
    add("g3_two", "c2", "CCATGGTTAACC", "g3", 600, 25)
    add("g3_two", "c2", "CCATGGTTAACC", "g3", 900, 17)

    # Three genes with 3, 2 and 2 reads: default mode removes one read of g1 (a tie of three); -i removes two of g1 and
    # the intronic read of g2, so g3 leads (a switch).
    add("three_genes", "c3", "ACGTACGTACGT", "g1", 1150, 25)
    add("three_genes", "c3", "ACGTACGTACGT", "g1", 1450, 17, re="N")
    add("three_genes", "c3", "ACGTACGTACGT", "g1", 2505, 17)
    add("three_genes", "c3", "ACGTACGTACGT", "g3", 1310, 19)
    add("three_genes", "c3", "ACGTACGTACGT", "g3", 1400, 19)
    add("three_genes", "c3", "ACGTACGTACGT", "g2", 4702, 19, re="N")
    add("three_genes", "c3", "ACGTACGTACGT", "g2", 4250, 19)
    # 3 reads against 2, two reads of the winner removed: the UMI moves to g2 while a read remains on g1.
    add("three_vs_two", "c3", "CGTAGCTACGAT", "g1", 1150, 25)
    add("three_vs_two", "c3", "CGTAGCTACGAT", "g1", 2400, 17)
    add("three_vs_two", "c3", "CGTAGCTACGAT", "g1", 3880, 17)
    add("three_vs_two", "c3", "CGTAGCTACGAT", "g2", 4210, 19)
    add("three_vs_two", "c3", "CGTAGCTACGAT", "g2", 4290, 19)
    add("g2_inside", "c3", "GATTACAGATTA", "g2", 4550, 25)

    # Pattern of the Cell Ranger source, a group alone at its corrected UMI that still carries xf 19 (as 1417
    # ACTCCTTAGTGAATGA-1, design_m3_draft/review/v2/a1b_exception_records_1417.json). g1 has two reads at ..GTT and two at
    # ..GTG, which are corrected to ..GTT ("If there's a 1-HD UMI w/ equal count, move to the lexicographically larger UMI.",
    # cellranger 9.0.1, lib/rust/tx_annotation/src/mark_dups.rs line 43). Low support is decided after one read of the
    # corrected group has moved (lines 226 to 239): at ..GTG g1 still has one read against one of g3, a tie, so the g3 read
    # is discarded although no other gene keeps that UB. The documented rule on UB counts it.
    add("alone_xf19_partner", "c4", "GATCCTGAGGTT", "g1", 3800, 17)
    add("alone_xf19_partner", "c4", "GATCCTGAGGTT", "g1", 3830, 17)
    add("alone_xf19_partner", "c4", "GATCCTGAGGTG", "g1", 3860, 25, ub="GATCCTGAGGTT")
    add("alone_xf19_partner", "c4", "GATCCTGAGGTG", "g1", 3890, 17, ub="GATCCTGAGGTT")
    add("alone_xf19", "c4", "GATCCTGAGGTG", "g3", 1400, 19)
    add("g2_partial", "c4", "TGACCAGTTGAC", "g2", 5350, 25)
    add("g2_partial", "c4", "TGACCAGTTGAC", "g2", 5600, 17)
    # Winner with three reads against a loser with one, one read of the winner inside a window: at every setting a read is
    # removed and the UMI stays on g1. In every other case with two genes a removed read changes the count.
    add("winner_keeps", "c4", "TTAACCGGTTAA", "g1", 1150, 25)
    add("winner_keeps", "c4", "TTAACCGGTTAA", "g1", 2505, 17)
    add("winner_keeps", "c4", "TTAACCGGTTAA", "g1", 3810, 17)
    add("winner_keeps", "c4", "TTAACCGGTTAA", "g3", 1400, 19)

    # Pattern of the Cell Ranger source, a chain of corrections (as 1417 CACATTTCAACCGTAG-1, same file): g1 has one read at
    # CTGTTCTTGTAT, two at CTGTTCTTGTTT and two at GTGTTCTTGTTT; the first goes to the second UMI and the second to the
    # third. With one read of each moved, g1 has two reads at CTGTTCTTGTTT against one of g2 and one of g3, so both are
    # discarded and the read of g1 that arrives there is counted: three groups of one read, one with xf 25. The documented
    # rule on UB sees a tie of three and counts nothing.
    add("chain", "c5", "CTGTTCTTGTAT", "g1", 3800, 25, ub="CTGTTCTTGTTT")
    add("chain", "c5", "CTGTTCTTGTTT", "g2", 4250, 19)
    add("chain", "c5", "CTGTTCTTGTTT", "g3", 1400, 19)
    add("chain_end", "c5", "CTGTTCTTGTTT", "g1", 3830, 25, ub="GTGTTCTTGTTT")
    add("chain_end", "c5", "CTGTTCTTGTTT", "g1", 3860, 17, ub="GTGTTCTTGTTT")
    add("chain_end", "c5", "GTGTTCTTGTTT", "g1", 3890, 17)
    add("chain_end", "c5", "GTGTTCTTGTTT", "g1", 3905, 17)
    return reads, pd.DataFrame(rows)


def _outs_10x(outs, reads, genes):
    """The designed Cell Ranger result: a UMI is counted for each (cell, UB, gene) with a record that has xf bit 1 set and
    bit 2 unset, the count that reproduces the filtered matrix of the three libraries (design of 2026-10-12, section 2.3).

    filtered_feature_bc_matrix holds the called cells and molecule_info.h5 every barcode. Its count is the number of such
    records: "count uint32 Number of reads associated with this putative molecule that were confidently mapped to the
    assigned feature." (https://www.10xgenomics.com/support/software/cell-ranger/9.0/analysis/outputs/cr-outputs-molecule-info).
    Our observation on the outs of 1417 (zcat and h5py): the three matrix files as written here; in molecule_info.h5 the
    barcodes without the "-1" (it is gem_group), these dataset types, pass_filter rows of barcode, library and genome
    index, and rows sorted by barcode, feature and UMI.
    """
    mol = reads[reads.cb.notna() & reads.ub.notna() & ((reads.xf & 3) == 1)].groupby(["cb", "gx", "ub"]).size()
    called = sorted(CELLS_10X.values())
    matrix = mol[mol.index.get_level_values("cb").isin(called)].groupby(["cb", "gx"]).size()
    entries = sorted((called.index(cb) + 1, genes.index(gx) + 1, n) for (cb, gx), n in matrix.items())
    d = outs / "filtered_feature_bc_matrix"
    d.mkdir()
    with gzip.open(d / "barcodes.tsv.gz", "wt") as fh:
        fh.write("".join(f"{cb}\n" for cb in called))
    with gzip.open(d / "features.tsv.gz", "wt") as fh:
        fh.write("".join(f"{g}\t{g.upper()}\tGene Expression\n" for g in genes))
    with gzip.open(d / "matrix.mtx.gz", "wt") as fh:
        fh.write("%%MatrixMarket matrix coordinate integer general\n"
                 '%metadata_json: {"software_version": "cellranger-9.0.1", "format_version": 2}\n'
                 f"{len(genes)} {len(called)} {len(entries)}\n" + "".join(f"{g} {b} {n}\n" for b, g, n in entries))
    barcodes = sorted({cb[:-2] for cb in reads.cb.dropna()})
    # UMI in 2-bit code, A 0, C 1, G 2, T 3, first base highest (our observation: with this code all 467,969 molecules of
    # 1417 chr21 and chr22 join the BAM, design_m3_draft/design/checks/count_vs_support_1417.json)
    rows = sorted((barcodes.index(cb[:-2]), genes.index(gx), int(ub.translate(str.maketrans("ACGT", "0123")), 4), n)
                  for (cb, gx, ub), n in mol.items())
    with h5py.File(outs / "molecule_info.h5", "w") as f:
        f["barcodes"] = np.array(barcodes, "S")
        f["features/id"] = np.array(genes, "S")
        f["barcode_info/pass_filter"] = np.array([(barcodes.index(cb[:-2]), 0, 0) for cb in called], np.uint64)
        for k, (name, dtype) in enumerate((("barcode_idx", np.uint64), ("feature_idx", np.uint32), ("umi", np.uint32),
                                           ("count", np.uint32))):
            f[name] = np.array([r[k] for r in rows], dtype)
        f["gem_group"] = np.ones(len(rows), np.uint16)


def build_10x(tmp: Path) -> Synth10x:
    """The genome and annotation of build() with a designed Cell Ranger outs folder: possorted_genome_bam.bam, the filtered
    matrix and molecule_info.h5 (design of 2026-10-12, section 3.0, Tests). reads, groups and events are the designed truth
    for the stages that read them; the removed columns come from flags.flag_reference at SETTINGS_10X.

    The cases are designed on the genome of seed 7, the default of build(), and there is no seed argument: another seed can
    put 8 A's in a row somewhere and add a window (our observation: seeds 4 and 15 of 0 to 24 change a removed read), and
    events then no longer fits the removed columns.
    """
    rng = random.Random(7)
    fasta, gtf, db = _genome(tmp, rng)
    hdr = pysam.AlignmentHeader.from_dict(HEADER | {"CO": CO_10X})
    records, reads = _reads_10x(rng, hdr)
    outs = tmp / "outs"
    outs.mkdir()
    bam = _sorted_bam(tmp, hdr, records, "outs/possorted_genome_bam.bam")
    tags = [setting_tag(s) for s in SETTINGS_10X]
    tx = load_transcripts(gtf)
    flags = pd.concat([flag_reference(bam, fasta, ref, tx, SETTINGS_10X) for ref in CHROMS])
    reads = reads.merge(flags[["qname"] + tags], on="qname", validate="one_to_one")
    genes = [attr.split('"')[1] for _, ft, _, _, _, attr in GTF_ROWS if ft == "gene"]
    _outs_10x(outs, reads, genes)
    # read groups (design 3.1): called cell, UB, one gene of the feature list in GX, xf bit 1
    g = reads[reads.cb.isin(list(CELLS_10X.values())) & reads.ub.notna() & reads.gx.isin(genes) & ((reads.xf & 1) == 1)]
    # end3 (design 3.1) is the 3'-most aligned base in transcript sense, 0-based: the last base of a forward record, the
    # first of a reverse one. Negated on the reverse strand, so that the 3'-most base of a group is a maximum on both.
    last = (g.end - 1).where(~g.reverse, -g.pos)
    g = g.assign(winner=(g.xf & 2) == 0, has_pa=g.pa > 0, e=g.re == "E", n=g.re == "N", last=last,
                 last_nonpa=last.where(g.pa == 0))
    groups = g.groupby(["case", "cb", "ub", "gx"], sort=False).agg(
        n_reads=("xf", "size"), n_winner_reads=("winner", "sum"), n_pa=("has_pa", "sum"), e=("e", "any"), n=("n", "any"),
        end3=("last", "max"), end3_nonpa=("last_nonpa", "max"),
        **{f"rm_{t}": (t, "sum") for t in tags}).reset_index().rename(columns={"gx": "gene"})
    groups.insert(7, "re_class", groups.pop("e") + 2 * groups.pop("n"))
    for c in ("end3", "end3_nonpa"):  # -1: no record without pa
        groups[c] = groups[c].abs().fillna(-1).astype(int)
    units = groups.drop_duplicates(["cb", "ub"])
    events = pd.DataFrame([(u.case, u.cb, u.ub, tag, RULE_10X[u.case][0], after) for u in units.itertuples()
                           for tag, after in zip(tags, RULE_10X[u.case][1:])],
                          columns=["case", "cb", "ub", "tag", "before", "after"])
    changed = events.before.fillna("") != events.after.fillna("")
    events["lost"] = events.before.notna() & changed
    events["gained"] = events.after.notna() & changed
    events["switched"] = events.lost & events.gained
    return Synth10x(tmp, fasta, gtf, db, bam, outs, reads, groups, events)


def run_original(synth: Synth, covlen, min_len, mism, introns):
    """Removed record keys from the original scanBAMfilter (serial mode, fresh tROC)."""
    BAMfilter, _, _ = polyafilter_modules()
    tag = f"{covlen}_{min_len}_{mism}_{int(introns)}"
    trans = synth.tmp / f"trans_{int(introns)}.pkl"
    out = synth.tmp / f"orig_{tag}.bam"
    BAMfilter.scanBAMfilter(
        covlen, min_len, str(synth.bam), str(synth.fasta), str(trans),
        out_db=str(synth.db), mism=mism, out_bamfile=str(out), tROC={}, includeIntrons=introns,
    )
    return removed_keys(synth.bam, out)
