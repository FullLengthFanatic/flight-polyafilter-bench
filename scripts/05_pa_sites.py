"""Stage 05: primary reads with a Cell Ranger pa tag, with priming-site context."""
import argparse
from multiprocessing import Pool

import pandas as pd
import pysam

from pafbench.config import FASTA, GTF, LIBRARIES, OUT
from pafbench.genes import load_gene_segments
from pafbench.sites import PA_COLUMNS, pa_row

_SEG = None


def _init():
    global _SEG
    _SEG = load_gene_segments(GTF)


def _work(job):
    bam_path, ref = job
    rows = []
    with pysam.AlignmentFile(str(bam_path)) as bam, pysam.FastaFile(str(FASTA)) as fa:
        for a in bam.fetch(ref):
            if a.is_secondary or a.is_supplementary or not a.has_tag("pa"):
                continue
            rows.append(pa_row(a, fa, _SEG))
    return pd.DataFrame(rows, columns=PA_COLUMNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--refs", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    bam = LIBRARIES[args.library]
    with pysam.AlignmentFile(str(bam)) as fh:
        refs = args.refs or [s.contig for s in fh.get_index_statistics() if s.mapped > 0]
    with Pool(args.workers, initializer=_init) as pool:
        parts = pool.map(_work, [(bam, r) for r in refs], chunksize=1)
    df = pd.concat(parts, ignore_index=True)
    out = args.out or (OUT / args.library / "pa_sites.parquet")
    df.to_parquet(out, index=False)
    print(out, len(df))


if __name__ == "__main__":
    main()
