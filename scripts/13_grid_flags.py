"""Stage 13: polyAfilter removal at every grid setting, streamed per reference (pafbench.stream).

Writes OUT/LIB/grid/truth_flags.parquet (primary records of truth reads), region_counts.parquet, cell_counts.parquet
and summary.json. Per-reference parts go to grid/parts/; a reference whose json exists is not scanned again. Each part
json names its inputs (truth file path and row count, number of settings); when they differ from this run's, the stage
stops and asks for parts/ to be trashed (after a stage 12 rerun, for example).

Measured on 1417 chr1 (6.96M records, 396 settings, one worker, 2026-10-08): 202 s wall; worker peak RSS 2.71 GiB, of
which 0.6 GiB is shared with the parent through fork (interpreter, GTF transcripts, truth qnames); memwatch tree peak
3.31 GiB. The peak is set by chromosome length, not records: two int32 coverage arrays plus the sequence, about 9 bytes
per bp (chunk buffers are fixed). Measured on 1179 chr1 (49.9M records, 396 settings, one worker, stage 13 review):
740 s wall, worker VmHWM 2.75 GiB, memwatch tree peak 3.36 GiB. Seven 1179 references at once (96.6M records, seven
workers) ran each job 1.2 to 1.35 times slower than alone (concurrent BAM reads). Whole 1179 BAM (574M records, 134
references) at 8 workers: about 22 to 25 min wall, about 14 GiB physical (about 18 GiB as a memwatch sum, which counts
the shared pages once per worker).
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pysam

from pafbench.config import FASTA, GTF, LIBRARIES, OUT
from pafbench.flags import setting_tag
from pafbench.grid import all_settings
from pafbench.pool import map_forked
from pafbench.stream import merge_parts, scan_to_parts, stale_parts
from pafbench.transcripts import load_transcripts

_TX = _TRUTH = _CELLS = None


def _init(tx, truth, cells):
    # loaded once in the parent and handed over by fork (no pickling), not reloaded for each reference
    global _TX, _TRUTH, _CELLS
    _TX, _TRUTH, _CELLS = tx, truth, cells


def _work(job):
    bam, ref, settings, part_dir, chunk, meta = job
    return scan_to_parts(bam, FASTA, ref, _TX, settings, _TRUTH, _CELLS, part_dir, chunk, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--refs", nargs="+")
    ap.add_argument("--out", type=Path, help="grid directory (default OUT/LIB/grid)")
    ap.add_argument("--truth", type=Path, help="parquet with a qname column (default OUT/LIB/truth/truth_reads.parquet)")
    ap.add_argument("--chunk", type=int, default=1_000_000)
    args = ap.parse_args()
    t0 = time.monotonic()
    bam = LIBRARIES[args.library]
    out_dir = args.out or OUT / args.library / "grid"
    part_dir = out_dir / "parts"
    truth = args.truth or OUT / args.library / "truth" / "truth_reads.parquet"
    cells = bam.parent / "filtered_feature_bc_matrix" / "barcodes.tsv.gz"
    settings = all_settings()
    with pysam.AlignmentFile(str(bam)) as fh:
        stats = [s for s in fh.get_index_statistics() if s.mapped > 0 and (args.refs is None or s.contig in args.refs)]
    refs = [s.contig for s in stats]
    if args.refs:
        absent = [r for r in args.refs if r not in refs]
        if absent:
            print(f"--refs not in the BAM index or without mapped reads, ignored: {', '.join(absent)}", file=sys.stderr)
    meta = {"truth": str(truth.resolve()), "truth_rows": pq.ParquetFile(truth).metadata.num_rows, "settings": len(settings)}
    stale = stale_parts(part_dir, refs, meta)
    if stale:
        sys.exit(f"{part_dir}: the parts of {', '.join(stale)} were made from another truth file or settings list "
                 f"(this run: {json.dumps(meta)}); trash them with `gio trash {part_dir}` and rerun")
    todo = [s for s in sorted(stats, key=lambda s: -s.mapped) if not (part_dir / f"{s.contig}.json").exists()]
    if todo:
        tx = load_transcripts(GTF)
        truth_qnames = set(pd.read_parquet(truth, columns=["qname"]).qname)
        with gzip.open(cells, "rt") as fh:
            called = [line.strip() for line in fh]
        jobs = [(bam, s.contig, settings, part_dir, args.chunk, meta) for s in todo]
        # fork hands the parent's transcripts, truth set and cells to the workers without pickling (see _init)
        map_forked(_work, jobs, args.workers, initializer=_init, initargs=(tx, truth_qnames, called))
    summary = merge_parts(part_dir, refs, [setting_tag(s) for s in settings], out_dir)
    summary = {"library": args.library, "settings": len(settings), "workers": args.workers, "scanned": len(todo),
               "seconds": round(time.monotonic() - t0, 1), **summary}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("library", "settings", "scanned", "records", "primary", "seconds")} | {"refs": len(refs)}))


if __name__ == "__main__":
    main()
