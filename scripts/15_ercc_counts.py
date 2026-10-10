"""Stage 15: ERCC UMI counts against ERCC Mix 1 input, unfiltered and after polyAfilter (spec 4.2), provisional.

Per ERCC contig of <base>/star_sorted.bam (stage 14), flags.flag_reference at the 8 published settings plus the tags chosen
in OUT/eval/selection.json (skipped with a note when the file is absent), joined by fetch order to CB and UB. UMIs count
(contig, CB, UB) over primary records with a called-cell CB (a barcode of filtered_feature_bc_matrix) and a UB; ercc.json
records this in the key "barcodes": "cells" (config.POPULATIONS vocabulary). The concentration table is downloaded once to
<ref>/ercc_conc_raw.txt. Writes <out>/ercc_counts.parquet (ercc_id, setting, umis) and <out>/ercc.json.
"""
import argparse
import gzip
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd
import pysam

from pafbench.config import ERCC_CONC_URL, LIBRARIES, OUT, SIRV_FASTA, SIRV_GTF
from pafbench.ercc import ercc_reads, ercc_settings, ercc_tables, parse_ercc_conc
from pafbench.flags import setting_tag
from pafbench.transcripts import load_transcripts

NOTE = ("Lexogen's ERCC mix identity in SIRV Set 4 is not verified; concentrations are ERCC Mix 1 from Thermo Fisher")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--base", type=Path, help="dir with star_sorted.bam (default OUT/LIB/spikein)")
    ap.add_argument("--out", type=Path, help="output dir (default: base)")
    ap.add_argument("--ref", type=Path, default=OUT / "ref", help="dir for the concentration table (default OUT/ref)")
    ap.add_argument("--selection", type=Path, default=OUT / "eval" / "selection.json")
    args = ap.parse_args()
    t0 = time.monotonic()
    base = args.base or OUT / args.library / "spikein"
    out = args.out or base
    raw = args.ref / "ercc_conc_raw.txt"
    if not raw.exists():
        args.ref.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(ERCC_CONC_URL) as r:
            raw.write_bytes(r.read())
    conc = parse_ercc_conc(raw.read_text())
    conc.to_csv(args.ref / "ercc_conc.tsv", sep="\t", index=False)

    settings, chosen = ercc_settings(args.selection)
    if chosen is None:
        print(f"{args.selection} absent: published settings only")
    with gzip.open(LIBRARIES[args.library].parent / "filtered_feature_bc_matrix" / "barcodes.tsv.gz", "rt") as fh:
        cells = {line.strip() for line in fh}
    bam = base / "star_sorted.bam"
    with pysam.AlignmentFile(str(bam)) as fh:
        contigs = [s.contig for s in fh.get_index_statistics() if s.mapped > 0 and s.contig.startswith("ERCC")]
    tx = load_transcripts(SIRV_GTF)
    reads = pd.concat([ercc_reads(bam, SIRV_FASTA, c, tx, settings, cells) for c in contigs], ignore_index=True)
    tags = [setting_tag(s) for s in settings]
    counts, per = ercc_tables(reads, tags, conc)

    out.mkdir(parents=True, exist_ok=True)
    counts.to_parquet(out / "ercc_counts.parquet", index=False)
    summary = {"library": args.library, "provisional": True, "note": NOTE, "concentration_source": ERCC_CONC_URL,
               "selection": chosen, "barcodes": "cells", "ercc_contigs": len(contigs),
               "reads": len(reads), "ercc_detected": int((counts.setting == "unfiltered").sum()),
               "ercc_with_concentration": per["unfiltered"]["correlation"]["n"], "settings": per}
    (out / "ercc.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("library", "ercc_contigs", "reads", "ercc_detected", "ercc_with_concentration")}
                     | {t: [per[t]["correlation"]["pearson_log10"], per[t]["umis_kept_share"]] for t in ["unfiltered"] + tags}
                     | {"seconds": round(time.monotonic() - t0, 1)}))


if __name__ == "__main__":
    main()
