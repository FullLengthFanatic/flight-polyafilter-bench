"""Stage 10: confirm how BD67 and BD70 FLNC reads mark priming sites (chr1 sample)."""
import json

import pandas as pd
import pysam

from pafbench.bd67 import a_run_site, leading_a, sense_3p_clip
from pafbench.config import FASTA, LONG_READS, OUT
from pafbench.sites import classify, has_hexamer, sense_context


def check(name, bam_path, n_max=50000):
    rows = []
    with pysam.AlignmentFile(str(bam_path)) as bam, pysam.FastaFile(str(FASTA)) as fa:
        for a in bam.fetch("chr1"):
            if a.is_secondary or a.is_supplementary or a.mapping_quality < 20:
                continue
            site, k = a_run_site(a)
            down, up = sense_context(fa, "chr1", site, a.is_reverse)
            rows.append([leading_a(sense_3p_clip(a)), k, down.count("A"), has_hexamer(up)])
            if len(rows) >= n_max:
                break
    d = pd.DataFrame(rows, columns=["clip_a", "aligned_a", "a20", "hexamer"])
    d["cls"] = d.a20.map(classify)
    rate = d.groupby("cls").hexamer.mean().to_dict()
    frac_clip = float((d.clip_a >= 10).mean())
    ratio = rate.get("nontemplated", 0) / rate["templated"] if rate.get("templated") else None
    res = {"reads": len(d), "frac_clip_a_ge10": frac_clip, "hexamer_rate": rate, "ratio_nontempl_templ": ratio,
           "class_counts": d.cls.value_counts().to_dict(), "aligned_a_quantiles": d.aligned_a.quantile([0.5, 0.9, 0.99]).to_dict(),
           "accept": bool(frac_clip >= 0.8 and ratio is not None and ratio >= 3)}
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "ends_check.json").write_text(json.dumps(res, indent=1, default=float))
    print(name, json.dumps({k: res[k] for k in ("reads", "frac_clip_a_ge10", "ratio_nontempl_templ", "accept")}, default=float))


if __name__ == "__main__":
    for name, path in LONG_READS.items():
        check(name, path)
