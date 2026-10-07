"""Stage 04: compare removed-record sets, original polyAfilter vs reimplementation."""
import argparse
import json
import sys

import pandas as pd

from pafbench.config import LIBRARIES, OUT
from pafbench.flags import parse_setting, removed_from_flags, removed_keys_by_ref, setting_tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--setting", action="append", required=True)
    args = ap.parse_args()
    bam = LIBRARIES[args.library]
    flags_dir = OUT / args.library / "reimpl_flags"
    results = {}
    for st in (parse_setting(s) for s in args.setting):
        tag = setting_tag(st)
        orig_bam = OUT / args.library / "original" / f"{tag}.bam"
        n_o = n_r = only_o = only_r = 0
        examples = []
        for ref, want in removed_keys_by_ref(bam, orig_bam):
            pq = flags_dir / f"{ref}.parquet"
            got = removed_from_flags(pd.read_parquet(pq), tag) if ref is not None and pq.exists() else set()
            n_o += len(want)
            n_r += len(got)
            a, b = want - got, got - want
            only_o += len(a)
            only_r += len(b)
            examples.extend([["only_original", *k] for k in list(a)[:5]] + [["only_reimpl", *k] for k in list(b)[:5]])
        results[tag] = {"n_original": n_o, "n_reimpl": n_r, "only_original": only_o, "only_reimpl": only_r,
                        "identical": only_o == 0 and only_r == 0, "examples": examples[:50]}
    out = OUT / args.library / "exact_match.json"
    out.write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v["identical"] for k, v in results.items()}))
    sys.exit(0 if all(v["identical"] for v in results.values()) else 1)


if __name__ == "__main__":
    main()
