"""Stage 23: the read groups of the called cells and the reads each setting of IMPACT_SETTINGS removes from them, streamed
per reference (pafbench.molstream; design of 2026-10-12, section 3.1).

Writes per reference, under OUT/LIB/recount/parts/: <ref>.read_groups.parquet (one row per cell, UMI and gene: reads,
winner reads, pa reads, region class, 3' ends and the reads removed at each setting), <ref>.cell_records.parquet (primary
records per called cell, RE and presence of a GX tag: total, with a UB, units and removed at each setting; cell -1 holds
all barcodes) and <ref>.json. OUT/LIB/recount/read_groups.json holds the counts per reference and their sums. A reference
whose json exists is not scanned again. Each part json names its inputs (BAM path, size and mtime, sha256 of the barcode
and feature files, the settings in mask bit order), the rule version and the sha256 of pafbench/molstream.py; when they
differ from this run's, the stage stops and asks for parts/ to be trashed (after an edit of the module, for example).

Measured on 1417 (14 settings, 2026-10-13, /usr/bin/time -v, resident size of the largest process). chr1 alone, one
worker, run twice with the same bytes in both parts: 6.96M records and 1.98M read groups, 134 and 128 s in the reference
(52,000 and 54,000 records per second over the two passes), 150 and 142 s wall, 2.61 GiB (stage 13 on the same reference:
2.71 GiB, set by chromosome length). chr21 and chr22, two workers: 2.78M records, 44 s wall, 0.93 GiB. On chr22 pass 1
takes 4.9 s, the pieces 4.3 s and pass 2 with the reduction 8.7 s.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pysam

from pafbench.config import FASTA, GTF, IMPACT_SETTINGS, LIBRARIES, OUT
from pafbench.molstream import COUNTS, part_meta, read_groups_to_parts, row_index
from pafbench.pool import map_forked
from pafbench.stream import stale_parts
from pafbench.transcripts import load_transcripts

_TX = _CELLS = _GENES = None


def _init(tx, cells, genes):
    # loaded once in the parent and handed over by fork (no pickling), not reloaded for each reference
    global _TX, _CELLS, _GENES
    _TX, _CELLS, _GENES = tx, cells, genes


def _work(job):
    bam, ref, part_dir, meta = job
    return read_groups_to_parts(bam, FASTA, ref, _TX, IMPACT_SETTINGS, _CELLS, _GENES, part_dir, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--refs", nargs="+")
    ap.add_argument("--out", type=Path, help="recount directory (default OUT/LIB/recount)")
    args = ap.parse_args()
    t0 = time.monotonic()
    bam = LIBRARIES[args.library]
    out_dir = args.out or OUT / args.library / "recount"
    part_dir = out_dir / "parts"
    matrix = bam.parent / "filtered_feature_bc_matrix"
    with pysam.AlignmentFile(str(bam)) as fh:
        stats = [s for s in fh.get_index_statistics() if s.mapped > 0 and (args.refs is None or s.contig in args.refs)]
    refs = [s.contig for s in stats]
    if args.refs:
        absent = [r for r in args.refs if r not in refs]
        if absent:
            print(f"--refs not in the BAM index or without mapped reads, ignored: {', '.join(absent)}", file=sys.stderr)
    meta = part_meta(bam, matrix, IMPACT_SETTINGS)
    stale = stale_parts(part_dir, refs, meta)
    if stale:
        sys.exit(f"{part_dir}: the parts of {', '.join(stale)} were made from other inputs, with another rule version or by "
                 f"another pafbench/molstream.py (this run: {json.dumps(meta)}); trash them with `gio trash {part_dir}` and "
                 "rerun")
    todo = [s for s in sorted(stats, key=lambda s: -s.mapped) if not (part_dir / f"{s.contig}.json").exists()]
    if todo:
        jobs = [(bam, s.contig, part_dir, meta) for s in todo]
        # fork hands the parent's transcripts and the barcode and feature rows to the workers without pickling (see _init)
        map_forked(_work, jobs, args.workers, initializer=_init,
                   initargs=(load_transcripts(GTF), row_index(matrix / "barcodes.tsv.gz"), row_index(matrix / "features.tsv.gz")))
    per_ref = {ref: json.loads((part_dir / f"{ref}.json").read_text()) for ref in refs}
    summary = {"library": args.library, "settings": meta["settings"], "workers": args.workers, "scanned": len(todo),
               "seconds": round(time.monotonic() - t0, 1), **{k: sum(st[k] for st in per_ref.values()) for k in COUNTS},
               "per_ref": {ref: {k: st[k] for k in (*COUNTS, "seconds")} for ref, st in per_ref.items()}}
    (out_dir / "read_groups.json").write_text(json.dumps(summary, indent=1))
    keys = ("library", "scanned", "records", "primary", "called", "in_groups", "read_groups", "seconds")
    print(json.dumps({k: summary[k] for k in keys} | {"settings": len(IMPACT_SETTINGS), "refs": len(refs)}))


if __name__ == "__main__":
    main()
