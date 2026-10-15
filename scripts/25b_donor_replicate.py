"""Stage 25, check 5: the donor replicate, class proportions and class-mean expression of 1417 against 1419
(pafbench.classes.donor_replicate; design of 2026-10-12, section 3.3). Run after scripts/25_cell_classes.py on both.

Starts only after stage 24 passed for both libraries (recount.require_passed). Reads cells.parquet of stage 25 and
unfiltered.npz and features.tsv of stage 24 for each library, and stops when the two features files differ. Writes
<out> (default OUT/cells/donor_replicate.json).
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from pafbench import matrices
from pafbench.classes import donor_replicate
from pafbench.config import NUCLEI, OUT
from pafbench.recount import require_passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, nargs=2, default=[OUT / lib / "cells" for lib in NUCLEI],
                    help="cells directories of stage 25, 1417 then 1419 (default OUT/LIB/cells)")
    ap.add_argument("--matrices", type=Path, nargs=2, default=[OUT / lib / "matrices" for lib in NUCLEI],
                    help="matrices directories of stage 24, 1417 then 1419 (default OUT/LIB/matrices)")
    ap.add_argument("--recount", type=Path, nargs=2, default=[OUT / lib / "recount" for lib in NUCLEI],
                    help="recount directories of stage 24, 1417 then 1419 (default OUT/LIB/recount)")
    ap.add_argument("--out", type=Path, default=OUT / "cells" / "donor_replicate.json")
    args = ap.parse_args()
    t0 = time.monotonic()
    for lib, r, m in zip(NUCLEI, args.recount, args.matrices):
        require_passed(lib, r, m)
    features = [(m / "features.tsv").read_text() for m in args.matrices]
    if features[0] != features[1]:
        raise SystemExit(f"{args.matrices[0]} and {args.matrices[1]}: features.tsv differ")
    symbols = [line.split("\t")[1] for line in features[0].splitlines()]
    res = donor_replicate(NUCLEI, [pd.read_parquet(d / "cells.parquet") for d in args.cells],
                          [matrices.load(lib, "unfiltered", m) for lib, m in zip(NUCLEI, args.matrices)], symbols)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"libraries": res["libraries"], "cells": res["cells"],
                      "classes": {c["class"]: {"share": c["share"], "pearson_class_mean": c["pearson_class_mean"]}
                                  for c in res["classes"]}, "seconds": round(time.monotonic() - t0, 1)}))


if __name__ == "__main__":
    main()
