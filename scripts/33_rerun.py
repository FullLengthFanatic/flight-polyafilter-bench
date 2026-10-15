"""Stage 33 (optional, decision D8): one Cell Ranger rerun on the BAM the original tool wrote for 1417 at c375_l3_m2_i, and
its comparison with our matrix at that setting (pafbench.rerun; design of 2026-10-12, section 3.11).

run: prints the two commands of the rerun and, with --execute, runs them, each logged to <folder>/<name>.log:
  bamtofastq (~/cellranger-9.0.1/lib/bin) on OUT/1417/original/c375_l3_m2_i.bam into <folder>/fastq, then cellranger count
  on those FASTQs into <folder>/count, with the flags of the original call in the _cmdline of the Cell Ranger run folder of
  1417 (--include-introns=true --create-bam=true --localcores=16 --localmem=64, checked against its _invocation). The
  original reference path, /data/references/refdata-gex-GRCh38-2024-A, does not exist on this machine; REF_DIR is used in
  its place after its reference.json is found to hold the FASTA and GTF hashes of the original run's molecule_info.h5.
  <folder> is <rerun>/c375_l3_m2_i, <rerun> OUT/1417/rerun or --rerun. The stage stops unless the BAM's run ended with exit
  status 0 (recount.original_bams).
compare: once <folder>/count/outs exists (or --outs), its raw_feature_bc_matrix on the called cells of the original run,
  its filtered_feature_bc_matrix and molecule_info.h5 against our matrix at the tag (pafbench.rerun.compare), written to
  <rerun>/compare.json (or --out); at a --tag other than c375_l3_m2_i the default is <rerun>/compare_<tag>.json, so a check
  at another setting does not take the place of the D8 comparison. Reads the outputs of stage 24 (OUT/1417/recount and
  OUT/1417/matrices, or --recount and --matrices) after recount.require_passed.

Where the files go. Every path of the commands is absolute (rerun.rerun_commands resolves a relative --rerun) and what they
write is under <folder>, which rerun.check_rerun_dir refuses inside the repository or inside the Cell Ranger run folder of
any library, and both commands run with <folder> as their working directory: cellranger count writes __<id>.mro into the
directory it runs in (our observation, a --dry run on synthetic FASTQs). bamtofastq stops when <folder>/fastq exists; a
command whose <folder>/<name>.json exists (written after exit status 0) is not run again. The FASTQs stay until someone
trashes them (gio trash <folder>/fastq).

Telemetry, to settle with the user before run --execute. `cellranger telemetry --help` (cellranger-9.0.1): "collect: Collect
telemetry data, if enabled.", "disable: Disable telemetry collection for this user." and "For more information about what
data is collected and how it's used, visit https://10xgen.com/pipeline-telemetry". Our observation, 2026-10-15: `cellranger
telemetry check` says "Collected telemetry data is being saved to" ~/.cache/tenx/telemetry/cellranger/cellranger-9.0.1, and
a cellranger count --dry on synthetic FASTQs wrote a count_success record to extras/telemetry of its output folder. Whether
Cell Ranger sends such records to 10x is not verified. The user decides whether to run `cellranger telemetry disable` first;
this script leaves the setting as it is.

Disk, 2026-10-15: /data has 6.1 TB free (df -h /data), the BAM is 6.1 GB; FASTQs and working files are estimated at a few
times that while it runs, with about 12 GB kept, the size of the original outs (design 3.11). Time and memory are not
measured: hours of wall time, memory within Cell Ranger's 64 GB limit (estimates of the design). Not to be started while
stage 23 or 24 runs on 1179 or while stage 30 runs.

compare, measured on 2026-10-15 under /usr/bin/time -v and memwatch with the outs of the original run of 1417 in place of
the rerun's: at "unfiltered" 30 s wall, 3.34 GiB resident (memwatch 3.12 GiB); at c375_l3_m2_i, which also reads the
removed reads of the parts, 24 s and 2.36 GiB (memwatch 2.12 GiB).
"""
import argparse
import json
import shlex
import shutil
import sys
import time
from pathlib import Path

from pafbench import rerun
from pafbench.config import LIBRARIES, OUT, REF_DIR
from pafbench.flags import Setting, setting_tag
from pafbench.recount import original_bams, require_passed

LIB, SETTING = "1417", Setting(375, 3, 2, True)          # decision D8: the paper's nuclei setting


def run(args, tag):
    bam = original_bams(OUT / LIB / "original", "", [SETTING])[tag]
    rerun.check_reference(REF_DIR, LIBRARIES[LIB].parent / "molecule_info.h5")
    folder = (args.rerun or OUT / LIB / "rerun") / tag
    commands = rerun.rerun_commands(LIBRARIES[LIB].parent.parent, bam, REF_DIR, folder)
    for cmd in commands:
        print(shlex.join(cmd))
    seconds = rerun.run_rerun(commands, folder) if args.execute else {}
    free = shutil.disk_usage(next(p for p in (folder, *folder.parents) if p.exists())).free
    print(json.dumps({"library": LIB, "tag": tag, "bam": str(bam), "bam_bytes": bam.stat().st_size, "folder": str(folder),
                      "free_bytes": free, "executed": args.execute, "seconds": seconds}))


def compare(args, tag):
    t0 = time.monotonic()
    root = args.rerun or OUT / LIB / "rerun"
    outs = args.outs or root / args.tag / "count" / "outs"
    recount_dir, matrix_dir = args.recount or OUT / LIB / "recount", args.matrices or OUT / LIB / "matrices"
    res = require_passed(LIB, recount_dir, matrix_dir)
    if not (outs / "raw_feature_bc_matrix").is_dir():
        sys.exit(f"{outs}: no raw_feature_bc_matrix; run scripts/33_rerun.py run --execute first")
    got = rerun.compare(LIB, args.tag, res["settings"], outs, recount_dir, matrix_dir)
    out = args.out or root / ("compare.json" if args.tag == tag else f"compare_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(got, indent=1))
    keys = ("entries_compared", "entries_differing", "umis_excess_ours", "umis_excess_rerun", "above_rerun_diff")
    print(json.dumps({"library": LIB, "tag": args.tag, "compare": str(out)} | {k: got[k] for k in keys} | {
        "cells_called": {k: got["cells_called"][k] for k in ("original", "rerun", "both")},
        "seconds": round(time.monotonic() - t0, 1)}))


def main():
    tag = setting_tag(SETTING)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="part", required=True)
    r = sub.add_parser("run", help="print the commands of the rerun; run them with --execute")
    r.add_argument("--rerun", type=Path,
                   help=f"folder of the rerun; the commands write to <rerun>/{tag} (default OUT/1417/rerun)")
    r.add_argument("--execute", action="store_true", help="run the commands (hours, up to 64 GB), not only print them")
    c = sub.add_parser("compare", help="compare the rerun's outs with our matrix and write compare.json")
    c.add_argument("--rerun", type=Path, help="folder of the rerun (default OUT/1417/rerun)")
    c.add_argument("--tag", default=tag, help=f"setting of our matrix (default {tag})")
    c.add_argument("--outs", type=Path, help="Cell Ranger outs of the rerun (default <rerun>/<tag>/count/outs)")
    c.add_argument("--recount", type=Path, help="recount directory of stage 24 (default OUT/1417/recount)")
    c.add_argument("--matrices", type=Path, help="matrices directory of stage 24 (default OUT/1417/matrices)")
    c.add_argument("--out", type=Path,
                   help=f"output json (default <rerun>/compare.json at {tag}, <rerun>/compare_<tag>.json at another tag)")
    args = ap.parse_args()
    if args.part == "run":
        run(args, tag)
    else:
        compare(args, tag)


if __name__ == "__main__":
    main()
