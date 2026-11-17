"""Stage 29b: the internal-priming share of each library under the conventional rule (pafbench.conventional: 6 or more
consecutive adenines anywhere in the 20 nt window downstream of the site, or 12 or more adenines in it), beside the
templated share under the 8/4 labels of the same molecules.

Molecules as stage 18 at its default --population: uniquely mapped (mapq_min 255, pafbench.truth.uniquely_mapped) on called
cells (config.SEL_POPULATION), so n is the total and share_all_8_4 the share_all of the library's "all" row in
burden/templated_share.parquet.

Writes under --out (default OUT/burden): conventional_rule.parquet (lib, n, n_conventional, share, share_all_8_4; one row
per library of NUCLEI + CELLS + BD_NUC that has truth files) and conventional_rule.json (the rule, the population, the
mapq_min and the same values per library).
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pafbench.config import BD_NUC, CELLS, NUCLEI, OUT, SEL_POPULATION, SITE_WINDOW, UNIQUE_MAPQ
from pafbench.conventional import COUNT_MIN, RUN_MIN, conventional_share
from pafbench.truth import in_population, uniquely_mapped

COLUMNS = ["down20", "a20", "mapq_min", "label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=OUT)
    ap.add_argument("--out", type=Path, default=OUT / "burden")
    args = ap.parse_args()
    t0 = time.time()
    rows = []
    for lib in NUCLEI + CELLS + BD_NUC:
        p = args.base / lib / "truth" / "molecules_labeled.parquet"
        if not p.exists():
            print(f"{lib}: skipped, missing {p}")
            continue
        m = uniquely_mapped(pd.read_parquet(p, columns=COLUMNS + (["in_cell"] if "in_cell" in pq.read_schema(p).names else [])))
        try:
            m = in_population(m, SEL_POPULATION)
        except ValueError as e:
            raise SystemExit(f"{lib}: {e}")
        rows.append({"lib": lib} | conventional_share(m))
    if not rows:
        raise SystemExit("no library has truth files")
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out / "conventional_rule.parquet", index=False)
    out = {"rule": {"window": SITE_WINDOW, "run_min": RUN_MIN, "count_min": COUNT_MIN}, "population": SEL_POPULATION,
           "mapq_min": UNIQUE_MAPQ, "libraries": {r["lib"]: {k: v for k, v in r.items() if k != "lib"} for r in rows}}
    (args.out / "conventional_rule.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"out": str(args.out), "share": {r["lib"]: r["share"] for r in rows},
                      "share_all_8_4": {r["lib"]: r["share_all_8_4"] for r in rows}, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
