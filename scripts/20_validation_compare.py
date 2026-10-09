"""Stage 20: exact match of the original polyAfilter and the reimplementation on every completed original run, the
remove-all-non-terminal control on 1417 chr21 + chr22, and the original tool at the 8 published settings on 1179
chr21 + chr22 (spec 5.1: 1179 does not fit in memory in full, so it is validated by chromosome).

Original runs and stage 03/04 outputs live under OUT/<lib>/ (stages 03 and 04 have no --out); --out moves only this
stage's own files: validation/exact_match_m2.json, validation/none_control/ and validation/subset_1179/. The 1179 block
resumes: a setting whose result file exists in subset_1179/ is not rerun. Its cost, scaled by record count from a 1179
chr22-only check (8.3M of the 18.4M records): subset about 6 min, createTRANS e about 76 min with i (about 36 min) alongside,
then about 30 min per setting (original 24, comparison 6), about 5.4 h in all; original under 8 GiB, comparison about 10 GiB.
The exact_match_m2.json that stage 21 reads must come from a call without skip flags (each call rewrites it with only the
parts it ran); once the 1179 block has run, such a call only reads the block's result files.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from pafbench.config import FASTA, GTF, LIBRARIES, NUCLEI, OUT, REPO
from pafbench.flags import setting_tag
from pafbench.grid import published
from pafbench.validation import discover_settings, none_control, run_stats, setting_arg, subset_validation

DB = OUT / "ref" / "genes_2024A.db"
SUBSET_1179_REFS = ("chr21", "chr22")


def compare_library(lib, settings, workers):
    """Stage 03 with all settings (it rewrites every reimpl_flags/<ref>.parquet), then stage 04; per-tag match and run cost."""
    args = [a for s in settings for a in ("--setting", setting_arg(s))]
    subprocess.run([sys.executable, str(REPO / "scripts" / "03_reimpl_flags.py"), lib, *args, "--workers", str(workers)],
                   check=True)
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "04_compare_removed.py"), lib, *args], stdout=subprocess.PIPE,
                       text=True)
    # stage 04 prints its summary line only after writing exact_match.json: exit 1 without it is a crash, not a difference
    lines = r.stdout.splitlines()
    if r.returncode not in (0, 1) or not lines or set(json.loads(lines[-1])) != {setting_tag(s) for s in settings}:
        raise RuntimeError(f"stage 04 on {lib} exited {r.returncode} without its summary line")
    match = json.loads((OUT / lib / "exact_match.json").read_text())
    out = {}
    for s in settings:
        tag = setting_tag(s)
        st = run_stats(OUT / lib / "original" / tag)
        out[tag] = {**match[tag], "original_elapsed_s": st["elapsed_s"], "original_max_rss_gib": st["max_rss_gib"],
                    "original_peak_tree_rss_gib": st["peak_tree_rss_gib"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", nargs="*", default=list(NUCLEI))
    ap.add_argument("--skip-none-control", action="store_true")
    ap.add_argument("--skip-1179", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    res = {}
    for lib in args.libs:
        settings = discover_settings(OUT / lib / "original")
        res[lib] = compare_library(lib, settings, args.workers) if settings else {}
    flags = [v["identical"] for lib in args.libs for v in res[lib].values()]
    if not args.skip_none_control:
        res["none_control"] = none_control(LIBRARIES["1417"], ["chr21", "chr22"], args.out / "validation" / "none_control",
                                           DB, FASTA, GTF)
        flags.append(res["none_control"]["identical"])
    if not args.skip_1179:
        res["1179_subset"] = subset_validation(LIBRARIES["1179"], list(SUBSET_1179_REFS),
                                               args.out / "validation" / "subset_1179", DB, FASTA, GTF, published())
        flags.extend(v["identical"] for v in res["1179_subset"]["settings"].values())
    res["n_compared"] = len(flags)
    res["all_identical"] = all(flags)
    dest = args.out / "validation" / "exact_match_m2.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1))
    print(json.dumps({**{lib: {t: v["identical"] for t, v in res[lib].items()} for lib in args.libs},
                      "none_control": res.get("none_control", {}).get("identical"),
                      "1179_subset": ({t: v["identical"] for t, v in res["1179_subset"]["settings"].items()}
                                      if "1179_subset" in res else None),
                      "n_compared": res["n_compared"],
                      "all_identical": res["all_identical"]}))
    sys.exit(0 if res["all_identical"] else 1)


if __name__ == "__main__":
    main()
