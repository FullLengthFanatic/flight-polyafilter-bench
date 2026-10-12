"""Small synthetic genome, annotation and BAM for exactness tests against the original polyAfilter."""
import random
from dataclasses import dataclass
from pathlib import Path

import gffutils
import pysam

from pafbench.flags import removed_keys
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


def _sorted_bam(tmp, hdr, reads):
    unsorted = tmp / "unsorted.bam"
    with pysam.AlignmentFile(str(unsorted), "wb", header=hdr) as out:
        for a in reads:
            out.write(a)
    bam = tmp / "reads.bam"
    pysam.sort("-o", str(bam), str(unsorted))
    pysam.index(str(bam))
    return bam


def build(tmp: Path, seed: int = 7) -> Synth:
    rng = random.Random(seed)
    fasta, gtf, db = _genome(tmp, rng)
    hdr = pysam.AlignmentHeader.from_dict(HEADER)
    return Synth(tmp, fasta, gtf, db, _sorted_bam(tmp, hdr, _reads(rng, hdr)))


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
