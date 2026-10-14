"""Stage 29: the templated share corrected for label purity, by region (design of 2026-10-12, section 3.7).

On the labeled molecules of called cells (in_cell) of each library and group (nuclei = 1417 + 1419, cells = 1179), over all
regions ("all") and per region value that truth/molecules_labeled.parquet holds: the reference rates h_tail (A band 0-4)
and h_ref (a20 >= 14), the tail-primed share per A band and the corrected templated share, once with the library-wide
rates and once with the region's own, a gene bootstrap of 1,000 resamples (seed 0) with every rate recomputed in each
resample, and the headline version of each region by the rule of 3.7 (config.REGION_REF_MIN). The library-wide rates of a
group are those of the group. pafbench.burden.purity_by_region.

Writes under --out (default OUT/burden): purity_reference_by_region.parquet (group, region, n_ref_tail, n_ref_internal,
h_tail, h_tail_lo, h_tail_hi, h_ref, h_ref_lo, h_ref_hi, headline_version), templated_share_corrected.parquet (group, region,
n_labeled, share_label, share_corrected, lo, hi, share_corrected_regionref, lo_regionref, hi_regionref, headline_version)
and purity_by_region.parquet (group, region, a_band, n, hexamer_rate, tail_primed_share, tail_primed_share_regionref,
clipped, clipped_regionref). share_corrected, lo, hi and tail_primed_share are the library-wide version, the _regionref
columns the region's own; clipped marks a band estimate that was clipped to [0, 1].

Acceptance of 3.7, per library, in the summary line: the all-regions corrected share by the class-level formula of
evaluate.label_purity equals templated_share_corrected of eval/selection.json to 9 decimals, and so does the per-band sum
unless a templated band estimate is clipped (pafbench.burden.purity_acceptance). After writing its outputs the stage exits
1 when a library fails.

Measured on 2026-10-14 on the three libraries (90,099, 105,400 and 1,174,777 labeled molecules of called cells): 40 to
45 s wall, 1.6 to 1.7 GiB peak tree memory (memwatch). The bootstrap multiplies whole-number counts, exact in float64, and
our observation agrees: the tables were the same with the BLAS thread count left free and set to 4 (OPENBLAS_NUM_THREADS).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pafbench.burden import purity_acceptance, purity_by_region
from pafbench.config import CELLS, NUCLEI, OUT
from pafbench.truth import in_population

COLUMNS = ["a20", "hexamer", "label", "region", "gene", "ref", "strand", "site"]
OUTPUTS = ("purity_reference_by_region", "templated_share_corrected", "purity_by_region")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=OUT)
    ap.add_argument("--out", type=Path, default=OUT / "burden")
    args = ap.parse_args()
    t0 = time.time()
    mols = {}
    for lib in NUCLEI + CELLS:
        p = args.base / lib / "truth" / "molecules_labeled.parquet"
        if not p.exists():
            print(f"{lib}: skipped, missing {p}")
            continue
        m = pd.read_parquet(p, columns=COLUMNS + (["in_cell"] if "in_cell" in pq.read_schema(p).names else []))
        try:
            mols[lib] = in_population(m, "cells")
        except ValueError as e:
            raise SystemExit(f"{lib}: {e}")
    if not mols:
        raise SystemExit("no library has truth files")
    groups, members = dict(mols), {lib: [lib] for lib in mols}
    for name, libs in (("nuclei", [lib for lib in NUCLEI if lib in mols]), ("cells", [lib for lib in CELLS if lib in mols])):
        if libs:
            groups[name] = pd.concat([mols[lib] for lib in libs], ignore_index=True)
            members[name] = libs

    tables = ([], [], [])
    for g, mol in groups.items():
        for parts, t in zip(tables, purity_by_region(mol)):
            t.insert(0, "group", g)
            parts.append(t)
    tables = [pd.concat(parts, ignore_index=True) for parts in tables]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, t in zip(OUTPUTS, tables):
        t.to_parquet(args.out / f"{name}.parquet", index=False)

    expected = json.loads((args.base / "eval" / "selection.json").read_text())["label_purity"]
    acceptance = {lib: purity_acceptance(mol, expected[lib]["templated_share_corrected"]) for lib, mol in mols.items()}
    ref = tables[0]
    print(json.dumps({"groups": members, "molecules": {lib: len(m) for lib, m in mols.items()},
                      "regions": {lib: sorted(m.region.unique()) for lib, m in mols.items()},
                      "headline_version": {g: dict(zip(d.region, d.headline_version))
                                           for g, d in ref.groupby("group", sort=False)},
                      "acceptance": acceptance, "seconds": round(time.time() - t0, 1)}))
    failed = [lib for lib, a in acceptance.items() if not a["pass"]]
    if failed:
        sys.exit(f"acceptance failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
