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
all_identical is also false when an expected original run (validation.EXPECTED_RUNS) is missing ("missing", per library), when
a stage 13 removed count differs from the reimplementation's ("n_grid", "grid_equal" per tag) or when the stage 13 truth flags
differ from the stage 03 flags ("engine_tie", per library). "complete" is true only for a call over exactly the nuclei libraries,
without skip flags, with nothing missing and the engine tie present. --reuse skips stages 03 and 04 of a library whose
exact_match.json is current (validation.reusable); "reused" records it.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from pafbench.config import FASTA, GTF, LIBRARIES, NUCLEI, OUT, REPO
from pafbench.flags import setting_tag
from pafbench.grid import published
from pafbench.validation import (
    EXPECTED_RUNS, discover_settings, engine_tie, grid_removed, missing_runs, none_control, reusable, run_stats, setting_arg,
    subset_validation,
)

DB = OUT / "ref" / "genes_2024A.db"
SUBSET_1179_REFS = ("chr21", "chr22")


def compare_library(lib, settings, workers, run=True):
    """Stage 03 with all settings (it rewrites every reimpl_flags/<ref>.parquet), then stage 04; per-tag match and run cost.

    run False reads the exact_match.json of an earlier comparison instead (see validation.reusable).
    """
    args = [a for s in settings for a in ("--setting", setting_arg(s))]
    if run:
        subprocess.run([sys.executable, str(REPO / "scripts" / "03_reimpl_flags.py"), lib, *args, "--workers", str(workers)],
                       check=True)
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "04_compare_removed.py"), lib, *args],
                           stdout=subprocess.PIPE, text=True)
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
    ap.add_argument("--reuse", action="store_true", help="skip stages 03 and 04 for a library whose exact_match.json is current")
    args = ap.parse_args()
    res, missing, tie, reused = {}, {}, {}, {}
    flags = []
    for lib in args.libs:
        lib_dir = OUT / lib
        settings = discover_settings(lib_dir / "original")
        missing[lib] = missing_runs(lib_dir / "original", EXPECTED_RUNS.get(lib, ()))
        reused[lib] = bool(args.reuse and settings and reusable(lib_dir, settings))
        res[lib] = compare_library(lib, settings, args.workers, run=not reused[lib]) if settings else {}
        for tag, v in res[lib].items():
            v["n_grid"] = grid_removed(lib_dir / "grid", tag)
            v["grid_equal"] = None if v["n_grid"] is None else v["n_grid"] == v["n_reimpl"]
        grid = lib_dir / "grid"
        tie[lib] = (engine_tie(lib_dir, res[lib]) if settings and (grid / "summary.json").exists()
                    and (grid / "truth_flags.parquet").exists() else None)
        flags.append(not missing[lib])
        flags.extend(v["identical"] and v["grid_equal"] is not False for v in res[lib].values())
        if tie[lib] is not None:
            flags.append(tie[lib]["missing_rows"] == 0 and not any(tie[lib]["mismatches"].values()))
    if not args.skip_none_control:
        res["none_control"] = none_control(LIBRARIES["1417"], ["chr21", "chr22"], args.out / "validation" / "none_control",
                                           DB, FASTA, GTF)
        flags.append(res["none_control"]["identical"])
    if not args.skip_1179:
        res["1179_subset"] = subset_validation(LIBRARIES["1179"], list(SUBSET_1179_REFS),
                                               args.out / "validation" / "subset_1179", DB, FASTA, GTF, published())
        for tag, v in res["1179_subset"]["settings"].items():
            v["n_grid"] = grid_removed(OUT / "1179" / "grid", tag, SUBSET_1179_REFS)
            v["grid_equal"] = None if v["n_grid"] is None else v["n_grid"] == v["n_reimpl"]
        flags.extend(v["identical"] and v["grid_equal"] is not False for v in res["1179_subset"]["settings"].values())
    res["n_compared"] = len(flags)
    res["all_identical"] = all(flags)
    res["missing"], res["engine_tie"], res["reused"] = missing, tie, reused
    res["complete"] = (not any(missing.values()) and sorted(args.libs) == sorted(NUCLEI)
                       and not (args.skip_none_control or args.skip_1179) and all(t is not None for t in tie.values()))
    dest = args.out / "validation" / "exact_match_m2.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1))
    print(json.dumps({**{lib: {t: v["identical"] for t, v in res[lib].items()} for lib in args.libs},
                      "complete": res["complete"], "missing": {lib: [m["tag"] for m in missing[lib]] for lib in args.libs},
                      "none_control": res.get("none_control", {}).get("identical"),
                      "1179_subset": ({t: v["identical"] for t, v in res["1179_subset"]["settings"].items()}
                                      if "1179_subset" in res else None),
                      "n_compared": res["n_compared"],
                      "all_identical": res["all_identical"]}))
    sys.exit(0 if res["all_identical"] else 1)


if __name__ == "__main__":
    main()
