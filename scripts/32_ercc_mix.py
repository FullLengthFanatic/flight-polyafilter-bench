"""Stage 32: which ERCC mix is in the tube, from the unfiltered ERCC UMIs of stage 15 (design of 2026-10-12, section 3.10,
step 1; decision D7).

Reads <base>/ercc_counts.parquet (setting "unfiltered") and <base>/ercc.json of stage 15, <ref>/ercc_conc_raw.txt (the
Thermo Fisher table with both mixes and the subgroups, downloaded by stage 15) and the ERCC lengths from the SIRV FASTA.
Writes <out>/ercc_mix.json: the inputs, pafbench.ercc.ercc_mix (n, Pearson r of log2 UMIs against log2 Mix 1 and Mix 2,
the slope of the residuals of the Mix 1 fit on the subgroup's log2 ratio with its bootstrap interval, mean residual and
median length per subgroup, the length check, the verdict and its reason) and the acceptance of the design (n equals
ercc_with_concentration of ercc.json). Stage 15 files are not rewritten.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pysam

from pafbench.config import OUT, SIRV_FASTA
from pafbench.ercc import ercc_mix, mix_acceptance, parse_ercc_mixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--base", type=Path, help="dir with stage 15's ercc_counts.parquet and ercc.json (default OUT/LIB/spikein)")
    ap.add_argument("--out", type=Path, help="output dir (default: base)")
    ap.add_argument("--ref", type=Path, default=OUT / "ref", help="dir with ercc_conc_raw.txt (default OUT/ref)")
    args = ap.parse_args()
    t0 = time.monotonic()
    base = args.base or OUT / args.library / "spikein"
    out = args.out or base
    raw = args.ref / "ercc_conc_raw.txt"
    counts = pd.read_parquet(base / "ercc_counts.parquet")
    unfiltered = counts[counts.setting == "unfiltered"].set_index("ercc_id").umis
    with pysam.FastaFile(str(SIRV_FASTA)) as fa:
        lengths = dict(zip(fa.references, fa.lengths))
    mix = ercc_mix(unfiltered, parse_ercc_mixes(raw.read_text()), lengths)
    acceptance = mix_acceptance(mix, json.loads((base / "ercc.json").read_text()))
    summary = {"library": args.library, "counts": str(base / "ercc_counts.parquet"), "setting": "unfiltered", "table": str(raw),
               "fasta": str(SIRV_FASTA)} | mix | {"acceptance": acceptance}
    out.mkdir(parents=True, exist_ok=True)
    (out / "ercc_mix.json").write_text(json.dumps(summary, indent=1))
    rs = mix["residual_slope"] or {}
    print(json.dumps({"library": args.library, "n": mix["n"], "n_equal": acceptance["n_equal"],
                      "pearson_log2_mix1": mix["pearson_log2_mix1"], "pearson_log2_mix2": mix["pearson_log2_mix2"],
                      "residual_slope": [rs.get(k) for k in ("estimate", "lo", "hi")],
                      "length_p": (mix["length_check"] or {}).get("p"), "verdict": mix["verdict"], "reason": mix["reason"],
                      "seconds": round(time.monotonic() - t0, 1)}))


if __name__ == "__main__":
    main()
