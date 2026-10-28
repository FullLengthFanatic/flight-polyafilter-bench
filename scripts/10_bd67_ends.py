"""Stage 10: confirm how BD67 FLNC reads mark priming sites (genome-wide sample)."""
import json
import random

import pandas as pd
import pysam

from pafbench.bd67 import a_run_site, leading_a, sense_3p_clip
from pafbench.config import FASTA, LONG_READS, OUT
from pafbench.sites import classify, has_hexamer, sense_context

PRIMARY_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
WINDOW = 1_000_000
N_WINDOWS = 200
PER_WINDOW_MAX = 250


def _windows(bam, seed=0, n_windows=N_WINDOWS, window=WINDOW):
    """n_windows windows of size window, drawn with random.Random(seed): chromosome chosen weighted
    by length (so the sample is uniform over the genome), start uniform in [0, length - window)."""
    lengths = {c: bam.get_reference_length(c) for c in PRIMARY_CHROMS}
    chroms = list(lengths)
    rng = random.Random(seed)
    out = []
    for _ in range(n_windows):
        c = rng.choices(chroms, weights=[lengths[c] for c in chroms], k=1)[0]
        start = int(rng.uniform(0, lengths[c] - window))
        out.append((c, start, start + window))
    return out


def check(name, bam_path, n_max=50000):
    rows = []
    windows_used = 0
    with pysam.AlignmentFile(str(bam_path)) as bam, pysam.FastaFile(str(FASTA)) as fa:
        for ref, start, end in _windows(bam):
            windows_used += 1
            n_window = 0
            for a in bam.fetch(ref, start, end):
                if not (start <= a.reference_start < end) or a.is_secondary or a.is_supplementary or a.mapping_quality < 20:
                    continue
                site, k = a_run_site(a)
                down, up = sense_context(fa, ref, site, a.is_reverse)
                strand = "-" if a.is_reverse else "+"
                rows.append([ref, site, strand, leading_a(sense_3p_clip(a)), k, down.count("A"), has_hexamer(up)])
                n_window += 1
                if n_window >= PER_WINDOW_MAX or len(rows) >= n_max:
                    break
            if len(rows) >= n_max:
                break
    d = pd.DataFrame(rows, columns=["chrom", "site", "strand", "clip_a", "aligned_a", "a20", "hexamer"])
    d["cls"] = d.a20.map(classify)
    rate = d.groupby("cls").hexamer.mean().to_dict()
    frac_clip = float((d.clip_a >= 10).mean())
    ratio = rate.get("nontemplated", 0) / rate["templated"] if rate.get("templated") else None
    sites = d.drop_duplicates(subset=["chrom", "site", "strand"])
    rate_sites = sites.groupby("cls").hexamer.mean().to_dict()
    res = {"reads": len(d), "frac_clip_a_ge10": frac_clip, "hexamer_rate": rate, "ratio_nontempl_templ": ratio,
           "class_counts": d.cls.value_counts().to_dict(), "aligned_a_quantiles": d.aligned_a.quantile([0.5, 0.9, 0.99]).to_dict(),
           "accept": bool(frac_clip >= 0.8 and ratio is not None and ratio >= 3),
           "unique_sites": int(len(sites)), "class_counts_sites": sites.cls.value_counts().to_dict(),
           "hexamer_rate_sites": rate_sites, "windows_used": windows_used}
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "ends_check.json").write_text(json.dumps(res, indent=1, default=float))
    print(name, json.dumps({k: res[k] for k in ("reads", "frac_clip_a_ge10", "ratio_nontempl_templ", "accept",
                                                  "unique_sites", "hexamer_rate_sites", "windows_used")}, default=float))


if __name__ == "__main__":
    for name, path in LONG_READS.items():
        check(name, path)
