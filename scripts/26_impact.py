"""Stage 26: impact of filtering on cells and cell classes, one library per call (pafbench.impact; design of 2026-10-12,
section 3.4).

Starts only after stage 24 passed (recount.require_passed with the recount and matrices folders given). Reads the matrices
of stage 24 under <matrices> (unfiltered.npz, lost/ and gained/ of each setting, barcodes.tsv, features.tsv, manifest.json,
cell_totals.parquet) and the classes of stage 25 under <cells> (cells.parquet, model.npz, classes.json). Runs every setting
of the manifest, or those of --settings (for trials; the outputs then hold those settings only and summary.json lists
them), with --workers processes. Writes under <out>: per_cell, per_class, markers, knn, knn_confusion, clustering and
class_jaccard (.parquet) and summary.json (design 3.4, Outputs); prints the seconds of every matrix by step on stderr and a
json line at the end. Exits non-zero after writing when the acceptance of summary.json failed.

Measured on 2026-10-16 on the trial outputs of stages 24 and 25 (/usr/bin/time -v and memwatch, every setting, 33 matrices):
1417 (5,571 cells) 2 min 51 s wall with 8 workers, memwatch 10.02 GiB, the largest process 1.48 GB; again with 4 workers
3 min 47 s and 5.67 GiB, the same files byte for byte; 1419 (5,114 cells) 2 min 45 s with 8 workers, 9.79 GiB; 1179
(24,378 cells) 15 min 10 s with 7 workers, 26.50 GiB, the largest process 4.28 GB. A matrix takes about 15 s (nuclei) or
117 s (1179) in a worker that has run one before; the first in a worker adds the numba compilation (about 25 s and 65 s),
and the unfiltered matrix the Wilcoxon ranking of the marker genes (35 s and 140 s).
"""
import argparse
import json
import sys
import time
from pathlib import Path

from pafbench.config import BD_NUC, CELLS, CLUSTER, NUCLEI, OUT
from pafbench.impact import run
from pafbench.recount import require_passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library", choices=NUCLEI + CELLS + BD_NUC)
    ap.add_argument("--cells", type=Path, help="stage 25 folder (default OUT/LIB/cells)")
    ap.add_argument("--matrices", type=Path, help="matrices directory of stage 24 (default OUT/LIB/matrices)")
    ap.add_argument("--recount", type=Path, help="recount directory of stage 24 (default OUT/LIB/recount)")
    ap.add_argument("--out", type=Path, help="output directory (default OUT/LIB/impact)")
    ap.add_argument("--workers", type=int, default=4, help="processes for the matrices after the unfiltered one")
    ap.add_argument("--settings", nargs="+", help="setting tags to run, for trials (default every tag of the manifest)")
    args = ap.parse_args()
    t0 = time.monotonic()
    lib = args.library
    matrix_dir = args.matrices or OUT / lib / "matrices"
    require_passed(lib, args.recount or OUT / lib / "recount", matrix_dir)
    res = run(lib, matrix_dir, args.cells or OUT / lib / "cells", args.out or OUT / lib / "impact", args.settings, args.workers)
    for s in res.seconds:
        print(json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in s.items()}), file=sys.stderr)
    s = res.summary
    per_matrix = sorted(m["total"] for m in res.seconds)
    print(json.dumps({"library": lib, "settings": s["settings"], "matrices": len(res.seconds),
                      "calls": {f"{e['class']} {e['tag']}": e["call"] for e in s["calls"]
                                if e["resolution"] == CLUSTER["resolution"]},
                      "acceptance_passed": s["acceptance"]["passed"],
                      "seconds_per_matrix": {"median": round(per_matrix[len(per_matrix) // 2], 1),
                                             "max": round(per_matrix[-1], 1), "sum": round(sum(per_matrix), 1)},
                      "seconds": round(time.monotonic() - t0, 1)}))
    if not s["acceptance"]["passed"]:
        sys.exit(f"acceptance failed for {lib}: see acceptance in {args.out or OUT / lib / 'impact'}/summary.json")


if __name__ == "__main__":
    main()
