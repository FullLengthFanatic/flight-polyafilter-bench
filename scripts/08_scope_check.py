"""Stage 08: does the 3k aliquot (1417) carry the same priming bias as the 20k library (1416)? chr1, by region."""
import json

import pandas as pd

from pafbench.config import OUT
from pafbench.labels import compare_shares, templated_share
from pafbench.molecules import consolidate_sites


def main():
    a = consolidate_sites(pd.read_parquet(OUT / "1416" / "pa_sites_chr1.parquet"))
    p17 = pd.read_parquet(OUT / "1417" / "pa_sites.parquet")
    b = consolidate_sites(p17[p17.ref == "chr1"])
    cut = json.loads((OUT / "1417" / "label_cutoffs.json").read_text())
    res = {}
    for name, (tmin, nmax) in {"start_14_8": (14, 8), "fixed": (cut["templated_min"], cut["nontemplated_max"])}.items():
        cmp = compare_shares(templated_share(a, tmin, nmax), templated_share(b, tmin, nmax))
        res[name] = cmp.reset_index().to_dict(orient="records")
    go = not any(r["flag"] for rows in res.values() for r in rows)
    res["go"] = go
    res["molecules"] = {"1416_chr1": len(a), "1417_chr1": len(b)}
    (OUT / "scope_check.json").write_text(json.dumps(res, indent=1, default=float))
    print("go" if go else "NO-GO: a region differs by more than 5 points; take it to the user")


if __name__ == "__main__":
    main()
