"""Stage 16: long-read (BD67, BD70) priming sites, read acceptance by terminal A run and bead clip, comparison with 1179."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from pafbench.bd67 import OUTCOMES, annotate_sites, compare_labels, count_duplicates, hexamer_summary, scan_long_reads
from pafbench.config import FASTA, GTF, LABEL_CUTOFFS, LONG_READS, OUT
from pafbench.genes import load_gene_segments
from pafbench.pool import map_forked
from pafbench.regions import load_region_segments

REFS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
M1_CUTOFFS = {"templated_min": 14, "nontemplated_max": 8}
RECORD_KEYS = ("records", "secondary_supplementary", "low_mapq", "used")
_SEG = None


def _init(segments):
    global _SEG
    _SEG = segments


def _work(job):
    lib, ref = job
    t0 = time.monotonic()
    with pysam.FastaFile(str(FASTA)) as fa:
        seq = fa.fetch(ref).upper()
    sites, counts, digests = scan_long_reads(LONG_READS[lib], ref)
    sites = annotate_sites(sites, ref, seq, *_SEG)
    return lib, ref, sites, counts | {"seconds": time.monotonic() - t0}, digests


def _summary(lib, parts, sites):
    counts = {k: sum(p[3][k] for p in parts) for k in RECORD_KEYS + OUTCOMES}
    used = counts["used"]
    return {"library": lib, "bam": str(LONG_READS[lib]), "refs": [p[1] for p in parts],
            "records": {k: counts[k] for k in RECORD_KEYS},
            "duplicate_qnames": count_duplicates(np.concatenate([p[4] for p in parts])),
            "outcomes": {k: counts[k] for k in OUTCOMES},
            "outcome_fractions": {k: counts[k] / used for k in OUTCOMES},
            "dropped_fraction": 1 - counts["accepted"] / used,
            "unique_sites": len(sites), "sites_with_accepted_read": int((sites.n_accepted >= 1).sum()),
            "cutoffs_8_4": hexamer_summary(sites, LABEL_CUTOFFS),
            "cutoffs_14_8": hexamer_summary(sites, M1_CUTOFFS),
            "per_ref": {p[1]: p[3] for p in parts}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", nargs="+", default=list(LONG_READS), choices=list(LONG_READS))
    ap.add_argument("--refs", nargs="+", default=REFS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--compare", type=Path, default=OUT / "1179" / "truth" / "molecules_labeled.parquet")
    args = ap.parse_args()
    size = {}
    for lib in args.libs:
        with pysam.AlignmentFile(str(LONG_READS[lib])) as fh:
            size |= {(lib, s.contig): s.mapped for s in fh.get_index_statistics()}
    jobs = sorted(((lib, ref) for lib in args.libs for ref in args.refs), key=lambda j: -size.get(j, 0))
    segments = (load_region_segments(GTF), load_gene_segments(GTF))
    res = map_forked(_work, jobs, min(args.workers, len(jobs)), initializer=_init, initargs=(segments,))
    mol, line = None, {"compare": f"skipped, {args.compare} not found", "libraries": []}
    if args.compare.exists():
        mol = pd.read_parquet(args.compare, columns=["ref", "strand", "site", "label"])
        mol = mol[mol.ref.isin(args.refs)].reset_index(drop=True)
        line["compare"] = str(args.compare)
    for lib in args.libs:
        parts = sorted((r for r in res if r[0] == lib), key=lambda r: args.refs.index(r[1]))
        sites = pd.concat([p[2] for p in parts], ignore_index=True)
        out = args.out / lib
        out.mkdir(parents=True, exist_ok=True)
        sites.to_parquet(out / "long_read_sites.parquet", index=False)
        summary = _summary(lib, parts, sites)
        if mol is not None:
            table, agreement = compare_labels(mol, sites)
            table.to_parquet(out / "vs_1179.parquet", index=False)
            summary["vs_1179"] = {"path": str(args.compare)} | agreement
        (out / "long_read_acceptance.json").write_text(json.dumps(summary, indent=1))
        line["libraries"].append({"library": lib, "used": summary["records"]["used"],
                                  "dropped": summary["dropped_fraction"], "duplicate_qnames": summary["duplicate_qnames"],
                                  "accepted_sites": summary["sites_with_accepted_read"],
                                  "ratio_8_4": summary["cutoffs_8_4"]["ratio"], "passes_8_4": summary["cutoffs_8_4"]["passes"],
                                  "ratio_14_8": summary["cutoffs_14_8"]["ratio"]})
    print(json.dumps(line))


if __name__ == "__main__":
    main()
