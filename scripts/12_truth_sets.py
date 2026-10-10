"""Stage 12: labeled molecules and truth reads (pa reads plus selected members)."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from pafbench.config import GTF, LIBRARIES, OUT
from pafbench.labels import hexamer_curve
from pafbench.regions import load_region_segments
from pafbench.truth import check_pa_counts, label_truth_molecules, mark_in_cell, truth_reads, truth_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--base", help="directory with molecules, pa_sites and members.parquet (default OUT/LIB)")
    ap.add_argument("--out", help="output directory (default <base>/truth: OUT/LIB/truth without --base, and a redirected "
                                  "--base never writes into OUT)")
    ap.add_argument("--barcodes", help="called-cell barcodes.tsv.gz (default: filtered_feature_bc_matrix next to the library's BAM)")
    args = ap.parse_args()
    base = Path(args.base) if args.base else OUT / args.library
    out = Path(args.out) if args.out else base / "truth"
    mol = pd.read_parquet(base / "molecules.parquet")
    if "mapq_min" not in mol:
        sys.exit(f"molecules.parquet predates milestone 2; rerun scripts/06_molecules.py {args.library}")
    if args.barcodes is None and args.library not in LIBRARIES:
        sys.exit(f"{args.library} is not in config.LIBRARIES; pass --barcodes")
    barcodes = Path(args.barcodes) if args.barcodes else LIBRARIES[args.library].parent / "filtered_feature_bc_matrix" / "barcodes.tsv.gz"
    pa = pd.read_parquet(base / "pa_sites.parquet")
    members = pd.read_parquet(base / "members.parquet")
    labeled = label_truth_molecules(mol, pa, members, load_region_segments(GTF))
    labeled = mark_in_cell(labeled, barcodes)
    reads = truth_reads(mol, pa, members)
    check_pa_counts(mol, reads)
    out.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(out / "molecules_labeled.parquet", index=False)
    reads.to_parquet(out / "truth_reads.parquet", index=False)
    hexamer_curve(labeled).to_csv(out / "hexamer_curve.tsv", sep="\t", index=False)
    print(json.dumps({"library": args.library} | truth_summary(labeled, reads)))


if __name__ == "__main__":
    main()
