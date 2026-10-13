"""Stage 24: UMIs by the documented counting rule on the read groups of stage 23, before filtering and at each setting of
IMPACT_SETTINGS, with the acceptance checks A1 to A6 (pafbench.recount; design of 2026-10-12, section 3.2).

Reads OUT/LIB/recount/parts/ (stage 23; the references are those of the BAM index with mapped reads, and the stage stops
when a part file of one of them is missing or was made from other inputs), the Cell Ranger outs next to the BAM
(filtered_feature_bc_matrix, molecule_info.h5), OUT/LIB/grid/ (stage 13) and the BAMs the original tool wrote at the
published settings: OUT/LIB/original/<tag>.bam, and for a library of SUBSET the BAMs of its reference subset.

Writes under OUT/LIB/recount/: counted_molecules.parquet (one row per molecule the rule counts before filtering),
other_read_groups.parquet, rule_vs_xf.parquet and recount.json; under OUT/LIB/matrices/: unfiltered.npz, lost/<tag>.npz,
gained/<tag>.npz, barcodes.tsv, features.tsv, manifest.json, cell_totals.parquet and gene_totals.parquet. recount.json gets
its field passed last, after every check has run and passed; the later stages refuse to start without it
(recount.require_passed). A failed check is printed with its offending entries and the exit status is 1.
"""
import argparse
import json
import sys
from pathlib import Path

import pysam

from pafbench.config import IMPACT_SETTINGS, LIBRARIES, OUT
from pafbench.grid import published
from pafbench.recount import original_bams, run

# check A4 of these libraries runs on the BAMs of a reference subset (spec 5.1: the original tool does not hold 1179 in
# memory in full; scripts/20_validation_compare.py wrote them)
SUBSET = {"1179": (OUT / "validation" / "subset_1179", ("chr21", "chr22"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--out", type=Path, help="recount directory, with parts/ of stage 23 (default OUT/LIB/recount)")
    ap.add_argument("--matrices", type=Path, help="matrices directory (default OUT/LIB/matrices)")
    args = ap.parse_args()
    lib = args.library
    bam = LIBRARIES[lib]
    with pysam.AlignmentFile(str(bam)) as fh:
        refs = [s.contig for s in fh.get_index_statistics() if s.mapped > 0]
    folder, a4_refs = SUBSET.get(lib, (OUT / lib / "original", None))
    bams = original_bams(folder, "" if a4_refs is None else "_".join(a4_refs) + ".", [s for s in published() if s in IMPACT_SETTINGS])
    check = OUT / "design_m3_draft" / "review" / "v2" / f"a1b_genomewide_{lib}.json"
    res = run(lib, bam, refs, IMPACT_SETTINGS, args.out or OUT / lib / "recount", args.matrices or OUT / lib / "matrices",
              OUT / lib / "grid", bams, a4_refs, check if check.exists() else None)
    keys = ("library", "refs", "read_groups", "counted_molecules", "other_read_groups", "cell_umis_two_or_more_groups", "seconds")
    print(json.dumps({k: res[k] for k in keys if k in res} | {
        name: res[name]["passed"] for name in ("A1", "A1b", "A2", "A3", "A4", "A5") if name in res} | {
        "passed": res.get("passed", False)}))
    sys.exit(0 if res.get("passed") else 1)


if __name__ == "__main__":
    main()
