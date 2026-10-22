"""Stage 01: rewrite BD067 Bioproduct.bam (BD Rhapsody WTA v3.0) as a 10x-compatible BAM.

The Rhapsody BAM carries CB:Z (cell-label integer index), MR:Z (8nt UMI), XF:Z (feature). It
lacks the UB, UR, GX, RE, xf and pa tags that pafbench reads from 10x BAMs. This stage adds them
so every downstream stage can treat BD67 like any other library:

  CB:Z  kept as-is (integer-string, opaque in pafbench)
  UB:Z  renamed from MR, right-padded with 'AAAA' from 8nt to 12nt to match pafbench's UMI_LEN=12
  UR:Z  = UB (Rhapsody does not store the uncorrected UMI; naive_UR baseline collapses to naive_UB)
  xf:i  25 (bit 0 pass, bit 3 feature-assigned, bit 4 CB whitelisted; what Cell Ranger emits)
  GX:Z  ENSG id of the sense-exon gene (features.tsv col 1), blank otherwise
  RE:A  E/N/I from XF prefix (see classify_xf)
  pa:i  length of soft-clipped polyA at the original read 3' end, threshold >=20 (Cell Ranger
        convention; flag=0 reads carry it as a 3' S-op with A-only content, flag=16 reads carry
        it as a 5' S-op with T-only content). Written only when count>=20, matching 10x STAR.

For sense-exon reads BD Rhapsody writes the gene symbol into XF (features.tsv col 2); the matrix
is keyed by ENSG (col 1), so we build a col2 -> col1 lookup from features.tsv and translate once
per read. Secondary/supplementary records and reads without CB or MR are dropped.
"""
import gzip
import sys
from pathlib import Path

import pysam

IN_BAM = Path("/data/element_runs/BD67/short_read/BD067_Lib1189_Bioproduct.bam")
OUT_BAM = Path("/data/element_runs/BD67/short_read/BD067_as10x.bam")
FEATURES_TSV = Path("/data/element_runs/BD67/short_read/filtered_feature_bc_matrix/features.tsv.gz")


def classify_xf(xf):
    """Return (gene_name_or_empty, 'E'|'N'|'I') from a BD Rhapsody XF tag value.

    For sense exon hits (no '__' prefix) the gene name is the whole XF value, matching
    features.tsv column 2. For other prefixes we drop the gene name.
    """
    if not xf:
        return "", "I"
    if not xf.startswith("__"):
        return xf, "E"
    body = xf[2:]
    if body.startswith("antisense") or body.startswith("ambiguousantisense"):
        return "", "I"
    if body.startswith("intergenic") or body == "no_feature":
        return "", "I"
    if body.startswith("ambiguousintrons"):
        return "", "N"
    if body.startswith("introns") or body.startswith("intron"):
        return "", "N"
    if body.startswith("ambiguous"):
        return "", "E"
    return "", "I"


def load_sym_to_ensg(features_tsv_gz):
    """{col2: col1} from features.tsv.gz, so gene_name (as written in XF) -> ENSG id."""
    out = {}
    with gzip.open(features_tsv_gz, "rt") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[parts[1]] = parts[0]
    return out


def polya_softclip_length(a, min_len=20):
    """Length of the soft-clipped polyA at the ORIGINAL read 3' end, or 0 when below threshold.

    Cell Ranger STAR emits pa:i:N for N>=20 soft-clipped A's trimmed from the read's 3' end before
    alignment. In BAM space: flag=0 (forward map) keeps the polyA as a 3' S-op with A-only content;
    flag=16 (reverse map) keeps it as a 5' S-op with T-only content (sequence reverse-complemented).
    We require >=90% A/T content so that reads with a bit of adapter contamination still match.
    """
    cig = a.cigartuples
    if not cig:
        return 0
    seq = a.query_sequence
    if a.is_reverse:
        if cig[0][0] != 4:
            return 0
        n = cig[0][1]
        if n < min_len:
            return 0
        clip = seq[:n]
        t_count = clip.count("T")
        return t_count if t_count >= min_len and t_count >= 0.9 * n else 0
    if cig[-1][0] != 4:
        return 0
    n = cig[-1][1]
    if n < min_len:
        return 0
    clip = seq[-n:]
    a_count = clip.count("A")
    return a_count if a_count >= min_len and a_count >= 0.9 * n else 0


def process(in_bam=IN_BAM, out_bam=OUT_BAM, features_tsv=FEATURES_TSV, threads=4):
    sym_to_ensg = load_sym_to_ensg(features_tsv)
    print(f"loaded {len(sym_to_ensg):,} gene_name -> ENSG mappings", file=sys.stderr, flush=True)
    n = kept = skipped = missed = pa_tagged = 0
    with pysam.AlignmentFile(str(in_bam), "rb", threads=threads) as bin_, \
         pysam.AlignmentFile(str(out_bam), "wb", template=bin_, threads=threads) as bout:
        for a in bin_:
            n += 1
            if a.is_secondary or a.is_supplementary or not a.has_tag("CB") or not a.has_tag("MR"):
                skipped += 1
                continue
            cb = a.get_tag("CB")
            mr = a.get_tag("MR")
            ub = (mr + "AAAA")[:12]
            xf = a.get_tag("XF") if a.has_tag("XF") else ""
            gene, re_val = classify_xf(xf)
            if gene:
                ensg = sym_to_ensg.get(gene, "")
                if not ensg:
                    missed += 1
            else:
                ensg = ""
            a.set_tag("CB", str(cb), value_type="Z")
            a.set_tag("UB", ub, value_type="Z")
            a.set_tag("UR", ub, value_type="Z")
            a.set_tag("GX", ensg, value_type="Z")
            a.set_tag("RE", re_val, value_type="A")
            a.set_tag("xf", 25, value_type="i")
            pa = polya_softclip_length(a)
            if pa:
                a.set_tag("pa", pa, value_type="i")
                pa_tagged += 1
            bout.write(a)
            kept += 1
            if n % 5_000_000 == 0:
                print(f"  {n:,} reads processed ({kept:,} kept, {missed:,} gene misses, "
                      f"{pa_tagged:,} pa-tagged)", file=sys.stderr, flush=True)
    print(f"total={n:,} kept={kept:,} skipped={skipped:,} gene_lookup_misses={missed:,} "
          f"pa_tagged={pa_tagged:,}")
    print("indexing...")
    pysam.index(str(out_bam))


if __name__ == "__main__":
    process()
