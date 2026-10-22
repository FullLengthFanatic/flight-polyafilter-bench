"""Stage 25: cell classes from marker genes on the unfiltered matrix of one library (pafbench.classes; design of
2026-10-12, section 3.3, decisions D1 and D9). The donor comparison of 1417 and 1419 (check 5) is scripts/25b_donor_replicate.py,
run once both libraries are done.

Starts only after stage 24 passed (recount.require_passed with the recount and matrices folders given). Reads
<matrices>/unfiltered.npz, barcodes.tsv and features.tsv, and Cell Ranger's graph clusters and their differential
expression under <analysis> (default: outs/analysis next to the library's BAM). Writes under <cells>: cells.parquet,
clusters.parquet, marker_table.parquet, model.npz and classes.json (3.3, outputs), and for the review gate class_table.csv
and marker_plot.png.

Measured on 2026-10-15 after the review fixes (10 Leiden seeds, under /usr/bin/time -v and memwatch, each process limited
to 1 thread per pool; the four runs of 1417 and 1419 at once, the two of 1179 one after the other), each library run twice
with the same files byte for byte: 1417 (5,571 cells) 2 min 20 s and 2 min 21 s wall, at most 3.10 GiB resident
(/usr/bin/time; memwatch 2.97 and 2.96 GiB); 1419 (5,114 cells) 2 min 19 s and 2 min 21 s, 3.50 GiB (memwatch 3.33 and
3.33 GiB); 1179 (24,378 cells, a neighbour graph by pynndescent above 8,192 cells) 6 min 33 s and 6 min 30 s, 13.04 GiB
(memwatch 12.64 and 12.41 GiB). Under cProfile on 1417 the scrublet doublet score took 107 s of 164 s and the pipeline
with UMAP 52 s (measured before the review fixes, which leave cells.parquet and model.npz as they were).
"""
import argparse
import json
import time
from pathlib import Path

from pafbench.classes import UNASSIGNED, run
from pafbench.config import CELLS, LIBRARIES, NUCLEI, OUT
from pafbench.recount import require_passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library", choices=NUCLEI + CELLS)
    ap.add_argument("--cells", type=Path, help="output directory (default OUT/LIB/cells)")
    ap.add_argument("--matrices", type=Path, help="matrices directory of stage 24 (default OUT/LIB/matrices)")
    ap.add_argument("--recount", type=Path, help="recount directory of stage 24 (default OUT/LIB/recount)")
    ap.add_argument("--analysis", type=Path, help="Cell Ranger outs/analysis (default: next to the BAM of LIB)")
    args = ap.parse_args()
    t0 = time.monotonic()
    lib = args.library
    matrix_dir = args.matrices or OUT / lib / "matrices"
    require_passed(lib, args.recount or OUT / lib / "recount", matrix_dir)
    res = run(lib, matrix_dir, args.cells or OUT / lib / "cells", args.analysis or LIBRARIES[lib].parent / "analysis")
    s = res.summary
    print(json.dumps({"library": lib, "cells": s["cells"], "clusters": s["clusters"],
                      "classes": {c: v["cells"] for c, v in s["classes"].items() if c != UNASSIGNED},
                      "unassigned_cells": s["unassigned"]["cells"], "ari_cell_ranger": s["cell_ranger"].get("ari"),
                      "seed_ari_min": s["seed_stability"]["ari_min"], "panel_genes_missing": s["panel_genes_missing"],
                      "seconds": round(time.monotonic() - t0, 1)}))


if __name__ == "__main__":
    main()
