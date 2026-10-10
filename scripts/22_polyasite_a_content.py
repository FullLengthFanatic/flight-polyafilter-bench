"""Stage 22: A content downstream of the PolyASite 3.0 atlas sites, per reference and in total.

For every atlas site (the loader and site convention of pafbench.polyasite, the atlas file stage 18 uses) with a full 20 nt
window after it in transcript sense: how many have 14 or more A's in that window, and the same for the sites moved 1,000 nt
upstream (pafbench.polyasite NULL_SHIFT). Writes <out>/polyasite_a_content.json, which the results page reads (section 10).

Options: --refs (default: every atlas reference present in the genome FASTA), --base, --out (default <base>/burden),
--atlas, --fasta.
"""
import argparse
import json
import time
from pathlib import Path

import pysam

from pafbench.config import FASTA, OUT, POLYASITE
from pafbench.polyasite import A_MIN, A_WINDOW, NULL_SHIFT, a_content, load_polyasite

KEYS = ("n_sites", "n_full", "n_a14", "n_full_null", "n_a14_null")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=OUT)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default <base>/burden)")
    ap.add_argument("--atlas", type=Path, default=POLYASITE)
    ap.add_argument("--fasta", type=Path, default=FASTA)
    ap.add_argument("--refs", nargs="+", default=None, help="references to count (default: all atlas references in the FASTA)")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = args.out or args.base / "burden"
    atlas = load_polyasite(args.atlas)
    atlas_refs = sorted({ref for ref, _ in atlas})
    with pysam.FastaFile(str(args.fasta)) as fa:
        in_fasta = set(fa.references)
        refs = args.refs if args.refs else [r for r in atlas_refs if r in in_fasta]
        missing = [r for r in refs if r not in in_fasta or r not in atlas_refs]
        if missing:
            raise SystemExit(f"references missing from the FASTA or the atlas: {', '.join(missing)}")
        per_ref = {}
        for ref in refs:
            seq = fa.fetch(ref).encode()
            counts = [a_content(seq, atlas[ref, plus], plus) for plus in (True, False) if (ref, plus) in atlas]
            per_ref[ref] = {k: sum(c[k] for c in counts) for k in KEYS}
    total = {k: sum(v[k] for v in per_ref.values()) for k in KEYS}
    result = {"atlas": str(args.atlas), "fasta": str(args.fasta), "window_nt": A_WINDOW, "min_a": A_MIN, "null_shift_nt": NULL_SHIFT,
              "refs": refs, "atlas_refs_not_counted": [r for r in atlas_refs if r not in refs], "total": total, "per_ref": per_ref}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "polyasite_a_content.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"refs": len(refs), "total": total, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
