"""Stage 17: grid metrics per library and member cap, setting selection (spec 7), bootstrap, hexamer check, spike-in metrics."""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from pafbench.config import CELLS, MEMBER_CAP, NUCLEI, OUT, SENS_CAPS
from pafbench.evaluate import (COUNT_COLUMNS, ambiguous_tail_share, bootstrap, evaluate, hexamer_check, relabel, select_setting,
                               spikein_metrics, tag_setting, units)
from pafbench.flags import Setting, setting_tag
from pafbench.grid import all_settings, published

RULE = ("On 1417, in each mode, the grid setting (remove-all control and off-grid published settings excluded) with the "
        "highest molecule-level sensitivity subject to specificity >= 0.95, at member cap 1000 nt and 8/4 labels; ties broken "
        "by fewer removed primary records, then by tag name. Reported on 1419 and 1179 without retuning. If no setting "
        "reaches 0.95, the frontier is reported and none is chosen.")
MIN_SPEC = 0.95
SEL_LIB = "1417"
CUTOFFS = "8/4"
MODES = {"e": False, "i": True}
SUMMARY_LEVELS = ("molecule", "molecule_majority", "read")
BOOT_LEVELS = ("molecule", "read")


def _num(x):
    return None if pd.isna(x) else float(x)


def _files(base, lib, sub=""):
    d = base / lib / sub
    return {"mol": d / "truth" / "molecules_labeled.parquet", "reads": d / "truth" / "truth_reads.parquet",
            "flags": d / "grid" / "truth_flags.parquet"}


def _load(f):
    fl = pd.read_parquet(f["flags"])
    tags = [setting_tag(s) for s in all_settings() if setting_tag(s) in fl.columns]
    return pd.read_parquet(f["mol"]), pd.read_parquet(f["reads"]), fl, tags


def _summary(metrics, tag, lib):
    m = metrics[(metrics.tag == tag) & (metrics.lib == lib) & (metrics.cap == MEMBER_CAP) & (metrics.cutoffs == CUTOFFS)
                & (metrics.stratum_kind == "all") & metrics.level.isin(SUMMARY_LEVELS)]
    return {r["level"]: {k: _num(r[k]) for k in COUNT_COLUMNS + ["sens", "spec", "prec"]} for _, r in m.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=OUT)
    ap.add_argument("--out", type=Path, default=OUT / "eval")
    args = ap.parse_args()
    t0 = time.time()
    libs, missing = [], {}
    for lib in NUCLEI + CELLS:
        need = list(_files(args.base, lib).values()) + [args.base / lib / "grid" / "region_counts.parquet"]
        absent = [str(p) for p in need if not p.exists()]
        if absent:
            missing[lib] = absent
            print(f"{lib}: skipped, missing {', '.join(absent)}")
        else:
            libs.append(lib)
    if not libs:
        raise SystemExit("no library has truth and grid files")
    args.out.mkdir(parents=True, exist_ok=True)

    metrics, u_main, removed_by_tag, unit_stats, tail_share = [], {}, {}, {}, {}
    for lib in libs:
        mol, tr, fl, tags = _load(_files(args.base, lib))
        tail_share[lib] = ambiguous_tail_share(mol)
        unit_stats[lib] = {}
        for cap in SENS_CAPS:
            u = units(mol, tr, fl, tags, cap)
            metrics.append(evaluate(u, lib, cap, CUTOFFS))
            unit_stats[lib][str(cap)] = {"read_units": len(u.reads), "molecule_units": len(u.mols),
                                         "missing_reads": u.missing_reads, "dropped_molecules": u.dropped_mols}
            if cap == MEMBER_CAP:
                u_main[lib] = u
        rc = pd.read_parquet(args.base / lib / "grid" / "region_counts.parquet")
        removed_by_tag[lib] = {k: int(v) for k, v in rc.groupby("tag").removed.sum().items()}
        print(f"{lib}: {len(tags)} tags, units {json.dumps(unit_stats[lib])}")
    metrics = pd.concat(metrics, ignore_index=True)
    metrics.to_parquet(args.out / "metrics.parquet", index=False)

    removed_sel = removed_by_tag.get(SEL_LIB, {})
    modes = {}
    for mode, introns in MODES.items():
        s = select_setting(metrics, removed_sel, introns, SEL_LIB, MEMBER_CAP, CUTOFFS, MIN_SPEC)
        modes[mode] = {"introns": introns, "selection": s,
                       "metrics": {lib: _summary(metrics, s["tag"], lib) for lib in libs} if s["tag"] else {}}
    pub = {setting_tag(p): {lib: _summary(metrics, setting_tag(p), lib) for lib in libs} for p in published()}
    cap_sens = {str(cap): {mode: select_setting(metrics, removed_sel, introns, SEL_LIB, cap, CUTOFFS, MIN_SPEC)
                           for mode, introns in MODES.items()} for cap in SENS_CAPS}
    sweep = []
    for tmin in range(7, 11):
        for nmax in range(2, 7):
            if nmax >= tmin:
                continue
            name = f"{tmin}/{nmax}"
            cut = {"templated_min": tmin, "nontemplated_max": nmax}
            m = pd.concat([evaluate(relabel(u_main[lib], cut), lib, MEMBER_CAP, name, levels=("molecule",), kinds=("all",))
                           for lib in libs], ignore_index=True)
            for mode, introns in MODES.items():
                s = select_setting(m, removed_sel, introns, SEL_LIB, MEMBER_CAP, name, MIN_SPEC)
                held = {}
                if s["tag"]:
                    for lib in libs:
                        r = m[(m.lib == lib) & (m.tag == s["tag"])]
                        held[lib] = {k: _num(r[k].iloc[0]) for k in ("sens", "spec", "prec")} if len(r) else None
                sweep.append({"templated_min": tmin, "nontemplated_max": nmax, "mode": mode, "tag": s["tag"],
                              "n_qualifying": s["n_qualifying"], "by_library": held})

    chosen = {mode: d["selection"]["tag"] for mode, d in modes.items()}
    boot_tags = [t for t in chosen.values() if t]
    boot_tags += [setting_tag(p) for p in published()]
    for mode, introns in MODES.items():
        if chosen[mode]:
            boot_tags.append(setting_tag(Setting(tag_setting(chosen[mode]).covlen, None, 0, introns)))
        else:
            print(f"mode {mode}: no setting chosen, so no chosen tag and no remove-all control at its COVLEN are bootstrapped")
    boot_tags = list(dict.fromkeys(boot_tags))
    boots, hexes = [], []
    for lib in libs:
        u = u_main[lib]
        here = [t for t in boot_tags if t in u.tags]
        absent = [t for t in boot_tags if t not in u.tags]
        if absent:
            print(f"{lib}: not in truth_flags, not bootstrapped: {', '.join(absent)}")
        for level in BOOT_LEVELS:
            b = bootstrap(u, here, level=level)
            b.insert(0, "lib", lib)
            b.insert(1, "cap", MEMBER_CAP)
            b.insert(2, "level", level)
            boots.append(b)
        h = hexamer_check(u, here)
        h.insert(0, "lib", lib)
        h.insert(1, "cap", MEMBER_CAP)
        hexes.append(h)
    pd.concat(boots, ignore_index=True).to_parquet(args.out / "bootstrap.parquet", index=False)
    pd.concat(hexes, ignore_index=True).to_parquet(args.out / "hexamer_check.parquet", index=False)

    spikes = []
    for lib in NUCLEI:
        f = _files(args.base, lib, "spikein")
        absent = [str(p) for p in f.values() if not p.exists()]
        if absent:
            print(f"{lib}: spike-in metrics skipped, missing {', '.join(absent)}")
            continue
        mol, tr, fl, tags = _load(f)
        u = units(mol, tr, fl, tags, MEMBER_CAP, label="spike_label", pos="internal", neg="tail", amb=None)
        s = spikein_metrics(u, tags)
        s.insert(0, "lib", lib)
        s.insert(1, "cap", MEMBER_CAP)
        spikes.append(s)
    if spikes:
        pd.concat(spikes, ignore_index=True).to_parquet(args.out / "spikein_metrics.parquet", index=False)

    out = {"rule": RULE, "min_spec": MIN_SPEC, "selection_library": SEL_LIB, "cap": MEMBER_CAP, "cutoffs": CUTOFFS,
           "libraries": libs, "missing": missing, "units": unit_stats, "modes": modes, "published": pub,
           "cap_sensitivity": cap_sens, "cutoff_sweep": sweep, "bootstrapped_tags": boot_tags,
           "ambiguous_tail_share": tail_share}
    (args.out / "selection.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"libraries": libs, "chosen": chosen, "metrics_rows": len(metrics), "bootstrap_tags": len(boot_tags),
                      "spikein_libraries": [s.lib.iloc[0] for s in spikes], "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
