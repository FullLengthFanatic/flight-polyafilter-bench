"""Stage 01a: rewrite the nine Pool278 BD Bioproduct BAMs as 10x-compatible BAMs and merge
per sample.

Pool278 is BD Rhapsody WTA 3.0 Enhanced V2/V3 on human retina nuclei. Nine libraries: three
samples (211, 213, 215) crossed with three bead-split replicates (a, b, c). Each BAM is coord
sorted, STAR-aligned against RhapRef_Human_WTA_2023-02 (GRCh38 primary assembly, chr1 naming)
and carries CB:Z (cell-label integer index), MR:Z (8nt UMI) and XF:Z (feature). It lacks the
UB, UR, GX, RE, xf and pa tags pafbench reads.

This stage mirrors `01_bd67_preprocess.py` for every library, adds a replicate-level RG tag per
read, and merges the three retagged BAMs of each sample into one `BD_nuc_<sample>_as10x.bam`
with three @RG header lines.

Usage: `python scripts/01a_pool278_preprocess.py --sample 211`
"""
import argparse
import gzip
import io
import shutil
import sys
import zipfile
from pathlib import Path

import pysam

POOL = Path("/data/element_runs/Pool278_FLIGHTseq_NucleiHsRetinaDonation")
BD_DIR = POOL / "bd_pipeline"
OUT_DIR = POOL / "merged"
PARTS_DIR = OUT_DIR / "parts"

SAMPLES = {
    "211": [("1488_BD-211a", "Pool278-1488-BD-211a"),
            ("1489_BD-211b", "Pool278-1489-BD-211b"),
            ("1490_BD-211c", "Pool278-1490-BD-211c")],
    "213": [("1492_BD-213a", "Pool278-1492-BD-213a"),
            ("1493_BD-213b", "Pool278-1493-BD-213b"),
            ("1494_BD-213c", "Pool278-1494-BD-213c")],
    "215": [("1496_BD-215a", "Pool278-1496-BD-215a"),
            ("1497_BD-215b", "Pool278-1497-BD-215b"),
            ("1498_BD-215c", "Pool278-1498-BD-215c")],
}


def classify_xf(xf):
    """Mirror of `01_bd67_preprocess.classify_xf`."""
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


def load_sym_to_ensg_from_mex_zip(mex_zip):
    """{col2: col1} from the features.tsv.gz inside a BD RSEC_MolsPerCell_MEX.zip."""
    out = {}
    with zipfile.ZipFile(mex_zip) as z:
        name = next(n for n in z.namelist() if n.endswith("features.tsv.gz"))
        with z.open(name) as inner, gzip.open(io.BufferedReader(inner), "rt") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    out[parts[1]] = parts[0]
    return out


def polya_softclip_length(a, min_len=20):
    """Mirror of `01_bd67_preprocess.polya_softclip_length`."""
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


def retag_one(in_bam, out_bam, rg_id, sm, sym_to_ensg, threads=4):
    """Rewrite one Pool278 Bioproduct BAM with 10x-compatible tags and an RG tag per read."""
    with pysam.AlignmentFile(str(in_bam), "rb", threads=threads) as bin_:
        header = bin_.header.to_dict()
        header.setdefault("RG", [])
        if not any(r.get("ID") == rg_id for r in header["RG"]):
            header["RG"].append({"ID": rg_id, "SM": sm, "LB": rg_id, "PL": "ILLUMINA"})
        n = kept = skipped = missed = pa_tagged = 0
        with pysam.AlignmentFile(str(out_bam), "wb", header=header, threads=threads) as bout:
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
                ensg = sym_to_ensg.get(gene, "") if gene else ""
                if gene and not ensg:
                    missed += 1
                a.set_tag("CB", str(cb), value_type="Z")
                a.set_tag("UB", ub, value_type="Z")
                a.set_tag("UR", ub, value_type="Z")
                a.set_tag("GX", ensg, value_type="Z")
                a.set_tag("RE", re_val, value_type="A")
                a.set_tag("xf", 25, value_type="i")
                a.set_tag("RG", rg_id, value_type="Z")
                pa = polya_softclip_length(a)
                if pa:
                    a.set_tag("pa", pa, value_type="i")
                    pa_tagged += 1
                bout.write(a)
                kept += 1
                if n % 10_000_000 == 0:
                    print(f"  [{rg_id}] {n:,} reads processed ({kept:,} kept, "
                          f"{missed:,} gene misses, {pa_tagged:,} pa-tagged)",
                          file=sys.stderr, flush=True)
    print(f"[{rg_id}] total={n:,} kept={kept:,} skipped={skipped:,} gene_lookup_misses={missed:,} "
          f"pa_tagged={pa_tagged:,}")


def process_sample(sample, threads=4):
    libs = SAMPLES[sample]
    sm = f"BD_nuc_{sample}"
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    first_lib_dir = BD_DIR / libs[0][0]
    mex = first_lib_dir / f"{libs[0][1]}_RSEC_MolsPerCell_MEX.zip"
    sym_to_ensg = load_sym_to_ensg_from_mex_zip(mex)
    print(f"[{sm}] loaded {len(sym_to_ensg):,} gene_name -> ENSG mappings from {mex.name}",
          file=sys.stderr, flush=True)
    part_bams = []
    for lib_dir_name, prefix in libs:
        rg_id = lib_dir_name.split("_", 1)[1]
        in_bam = BD_DIR / lib_dir_name / f"{prefix}_Bioproduct.bam"
        out_bam = PARTS_DIR / f"{rg_id}_as10x.bam"
        if out_bam.exists():
            print(f"[{sm}] {out_bam.name} exists, skipping retag", file=sys.stderr, flush=True)
        else:
            retag_one(in_bam, out_bam, rg_id, sm, sym_to_ensg, threads=threads)
        part_bams.append(str(out_bam))
    merged = OUT_DIR / f"{sm}_as10x.bam"
    tmp = merged.with_suffix(".bam.inprog")
    print(f"[{sm}] merging {len(part_bams)} parts -> {merged.name}", file=sys.stderr, flush=True)
    pysam.merge("-c", "-p", f"-@{threads}", "-f", str(tmp), *part_bams)
    shutil.move(tmp, merged)
    print(f"[{sm}] indexing {merged.name}", file=sys.stderr, flush=True)
    pysam.index(str(merged))
    print(f"[{sm}] done -> {merged}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", required=True, choices=sorted(SAMPLES))
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    process_sample(args.sample, threads=args.threads)


if __name__ == "__main__":
    main()
