"""Stage 03: per-record removal flags for polyAfilter settings (reimplementation)."""
import argparse
import json
from multiprocessing import Pool

import pysam

from pafbench.config import FASTA, GTF, LIBRARIES, OUT
from pafbench.flags import flag_reference, parse_setting, setting_tag
from pafbench.transcripts import load_transcripts

_TX = None


def _init():
    global _TX
    _TX = load_transcripts(GTF)


def _work(job):
    bam, ref, settings, out_dir = job
    df = flag_reference(bam, FASTA, ref, _TX, settings)
    df.to_parquet(out_dir / f"{ref}.parquet", index=False)
    return ref, len(df), {setting_tag(s): int(df[setting_tag(s)].sum()) for s in settings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--setting", action="append", required=True, help="COVLEN,MINSNRLEN|none,MISM,i|e")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    settings = [parse_setting(s) for s in args.setting]
    bam = LIBRARIES[args.library]
    out_dir = OUT / args.library / "reimpl_flags"
    out_dir.mkdir(parents=True, exist_ok=True)
    with pysam.AlignmentFile(str(bam)) as fh:
        stats = sorted((s for s in fh.get_index_statistics() if s.mapped > 0), key=lambda s: -s.mapped)
    jobs = [(bam, s.contig, settings, out_dir) for s in stats]
    with Pool(args.workers, initializer=_init) as pool:
        res = pool.map(_work, jobs, chunksize=1)
    summary = {
        "library": args.library,
        "settings": [setting_tag(s) for s in settings],
        "per_ref": {ref: {"records": n, "removed": rem} for ref, n, rem in res},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
