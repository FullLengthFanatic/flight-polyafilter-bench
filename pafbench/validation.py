"""Exact-match validation against the original polyAfilter: run bookkeeping, the remove-all-non-terminal control and the
original tool on a reference subset."""
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pysam

from .config import REPO, VENDOR_POLYAFILTER
from .flags import Setting, flag_reference, removed_from_flags, removed_keys_by_ref, setting_tag
from .report import parse_gnu_time
from .transcripts import load_transcripts
from .vendor import polyafilter_modules

_TAG = re.compile(r"c(\d+)_l(none|[1-9]\d*)_m(\d+)_([ie])")
_SIGNAL = re.compile(r"^Command terminated by signal (\d+)$", re.M)
TRASH = ("gio", "trash")
# the original runs stage 20 must find: milestone 1's three on 1417 plus the chains of scripts/19_validation_runs.sh
EXPECTED_RUNS = {
    "1417": ("c300_l10_m1_i", "c375_l3_m2_i", "c300_l10_m1_e", "c275_l9_m1_i", "c275_l9_m1_e", "c180_l11_m2_i", "c180_l11_m2_e",
             "c300_l10_m0_i", "c600_l10_m1_i", "c375_l3_m2_e"),
    "1419": ("c300_l10_m1_i", "c375_l3_m2_i", "c300_l10_m1_e", "c275_l9_m1_i", "c275_l9_m1_e", "c180_l11_m2_i", "c180_l11_m2_e",
             "c375_l3_m2_e"),
}


def parse_tag(tag):
    """Inverse of flags.setting_tag; 'lnone' is the remove-all control, 'l0' is refused like parse_setting does."""
    m = _TAG.fullmatch(tag)
    if m is None:
        raise ValueError(f"not a setting tag: {tag}")
    c, l, mism, mode = m.groups()
    return Setting(int(c), None if l == "none" else int(l), int(mism), mode == "i")


def setting_arg(s):
    """Inverse of flags.parse_setting: the --setting argument of stages 03 and 04."""
    return f"{s.covlen},{'none' if s.min_len is None else s.min_len},{s.mism},{'i' if s.introns else 'e'}"


def time_exit_status(tfile):
    """Exit status from a GNU time file; 128 + N for a command killed by signal N, for which GNU time writes 'Exit status: 0'."""
    m = _SIGNAL.search(Path(tfile).read_text())
    return 128 + int(m.group(1)) if m else parse_gnu_time(tfile).get("exit_status")


def run_stats(prefix):
    """Wall time, largest-process max RSS and exit status from <prefix>.time.txt; whole-tree peak from <prefix>.mem.json if present."""
    t = parse_gnu_time(f"{prefix}.time.txt")
    mem = Path(f"{prefix}.mem.json")
    peak = json.loads(mem.read_text())["peak_tree_rss_gib"] if mem.exists() else None
    return {"elapsed_s": t.get("elapsed_s"), "max_rss_gib": t.get("max_rss_gib"),
            "exit_status": time_exit_status(f"{prefix}.time.txt"), "peak_tree_rss_gib": peak}


def discover_settings(orig_dir):
    """Settings of every <tag>.bam in orig_dir whose <tag>.time.txt reports exit status 0 (not killed), in file-name order."""
    out = []
    for bam in sorted(Path(orig_dir).glob("*.bam")):
        tfile = bam.with_suffix(".time.txt")
        if tfile.exists() and time_exit_status(tfile) == 0:
            out.append(parse_tag(bam.stem))
    return out


def missing_runs(orig_dir, tags):
    """{tag, exit_status, reason} of every tag in tags that discover_settings does not return for orig_dir, in tags order.

    reason: "no output" (no BAM), "no time file" (BAM but no time file; still running or lost), "killed or failed".
    exit_status is that of the time file, None without one.
    """
    found = {setting_tag(s) for s in discover_settings(orig_dir)}
    out = []
    for tag in tags:
        if tag in found:
            continue
        bam, prefix = Path(orig_dir) / f"{tag}.bam", Path(orig_dir) / tag
        has_time = Path(f"{prefix}.time.txt").exists()
        reason = "no output" if not bam.exists() else "no time file" if not has_time else "killed or failed"
        out.append({"tag": tag, "exit_status": run_stats(prefix)["exit_status"] if has_time else None, "reason": reason})
    return out


def reusable(lib_dir, settings):
    """True when lib_dir/exact_match.json holds exactly the tags of settings, reimpl_flags/summary.json lists the same
    settings, and exact_match.json is newer than that summary and than every BAM in original/ (stages 03 and 04 can be skipped)."""
    lib_dir = Path(lib_dir)
    summary, match = lib_dir / "reimpl_flags" / "summary.json", lib_dir / "exact_match.json"
    if not summary.exists() or not match.exists():
        return False
    tags = {setting_tag(s) for s in settings}
    if set(json.loads(match.read_text())) != tags or set(json.loads(summary.read_text())["settings"]) != tags:
        return False
    newest = max([summary.stat().st_mtime, *(b.stat().st_mtime for b in (lib_dir / "original").glob("*.bam"))])
    return match.stat().st_mtime > newest


def grid_removed(grid_dir, tag, refs=None):
    """Removed primary records of tag in a stage 13 grid: summary.json "removed" (refs None), or the sum over the per-reference
    parts/<ref>.region.parquet of refs. None when a file is missing or the tag is not in the summary or a part."""
    grid_dir = Path(grid_dir)
    if refs is None:
        path = grid_dir / "summary.json"
        return json.loads(path.read_text())["removed"].get(tag) if path.exists() else None
    paths = [grid_dir / "parts" / f"{ref}.region.parquet" for ref in refs]
    if not all(p.exists() for p in paths):
        return None
    total = 0
    for p in paths:
        reg = pq.read_table(p, columns=["tag", "removed"]).to_pandas()
        if not (reg.tag == tag).any():
            return None
        total += int(reg.removed[reg.tag == tag].sum())
    return total


def engine_tie(lib_dir, tags):
    """Stage 13 against stage 03 on the primary record of every truth read: the tag columns of grid/truth_flags.parquet
    against reimpl_flags/<ref>.parquet, matched on (qname, flag, pos) within the reference (equal keys by order of
    appearance), one reference at a time and only the needed columns.

    Returns {rows, tags, mismatches: {tag: n}, missing_rows}. Rows without a counterpart are missing_rows and are not
    compared; a tag column absent from either table counts every row as a mismatch.
    """
    lib_dir, tags = Path(lib_dir), list(tags)
    truth_path = lib_dir / "grid" / "truth_flags.parquet"
    refs = pc.unique(pq.read_table(truth_path, columns=["ref"]).column("ref")).to_pylist()
    key = ["qname", "flag", "pos"]
    rows, missing, mism = 0, 0, dict.fromkeys(tags, 0)
    for ref in refs:
        truth = pq.read_table(truth_path, filters=[("ref", "==", ref)]).drop_columns(["ref"])
        rows += truth.num_rows
        path = lib_dir / "reimpl_flags" / f"{ref}.parquet"
        have = set(truth.column_names) - set(key)
        if path.exists():
            have &= set(pq.read_schema(path).names)
        use = [t for t in tags if t in have]
        for t in tags:
            if t not in use:
                mism[t] += truth.num_rows
        if not path.exists():
            missing += truth.num_rows
            continue
        got = pq.read_table(path, columns=[*key, *use])
        got = got.filter(pc.and_(pc.equal(pc.bit_wise_and(got.column("flag"), 0x900), 0),
                                 pc.is_in(got.column("qname"), value_set=truth.column("qname").combine_chunks())))
        want = truth.select([*key, *use]).to_pandas()
        got = got.to_pandas()
        for df in (want, got):
            df["_n"] = df.groupby(key).cumcount()
        m = want.merge(got, on=[*key, "_n"], how="left", suffixes=("", "_r"), indicator=True)
        both = m._merge == "both"
        missing += int((~both).sum())
        for t in use:
            mism[t] += int((m.loc[both, t] != m.loc[both, f"{t}_r"]).sum())
    return {"rows": rows, "tags": tags, "mismatches": mism, "missing_rows": missing}


def subset_bam(bam, refs, out_bam):
    """Records of the given references (samtools view -b -M keeps file order), indexed; the header keeps every reference."""
    pysam.view("-b", "-M", "-o", str(out_bam), str(bam), *refs, catch_stdout=False)
    pysam.index(str(out_bam))


def _timed(cmd, prefix):
    """cmd under memwatch and /usr/bin/time, run from REPO so a child python imports this checkout's pafbench:
    <prefix>.mem.json, .time.txt, .log (stdout and stderr)."""
    full = [sys.executable, "-m", "pafbench.memwatch", f"{prefix}.mem.json", "--",
            "/usr/bin/time", "-v", "-o", f"{prefix}.time.txt", *cmd]
    with open(f"{prefix}.log", "w") as log:
        subprocess.run(full, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)


def create_trans(db, bam, trans, include_introns, prefix):
    """polyAfilter createTRANS with stage 02's command line, under /usr/bin/time and memwatch: <prefix>.time.txt, .mem.json, .log."""
    _timed([sys.executable, str(VENDOR_POLYAFILTER / "polyAfilter.py"), "createTRANS", *(["-i"] if include_introns else []),
            str(db), str(bam), str(trans)], prefix)


def run_original(bam, trans_pkl, db, fasta, out_bam, setting, threads=4):
    """The original scanBAMfilter as the CLI's BAMfilter calls it, at setting (min_len None is the remove-all control), fresh tROC.

    trans_pkl is loaded if it exists, otherwise built from db; the -p temp files go next to out_bam and are deleted by the tool.
    Like the CLI, it does nothing when out_bam already exists.
    """
    BAMfilter, _, _ = polyafilter_modules()
    BAMfilter.scanBAMfilter(setting.covlen, setting.min_len, str(bam), str(fasta), str(trans_pkl), out_db=str(db), base="A",
                            mism=setting.mism, out_bamfile=str(out_bam), tROC={}, includeIntrons=setting.introns, cbFile=None,
                            out_cbFile=None, weightedCov=True, verbose=True, nThreads=threads)


def run_original_timed(bam, trans_pkl, db, fasta, out_bam, setting, prefix, threads=4):
    """run_original in a child process (python -m pafbench.validation) under memwatch and /usr/bin/time, files as create_trans."""
    _timed([sys.executable, "-m", "pafbench.validation", str(bam), str(trans_pkl), str(db), str(fasta), str(out_bam),
            setting_tag(setting), str(threads)], prefix)


def compare_subset(bam, out_bam, fasta, transcripts, setting):
    """Removed records of the original (bam minus out_bam) against flag_reference at setting, per reference."""
    tag = setting_tag(setting)
    n_o = n_r = only_o = only_r = 0
    examples = []
    for ref, want in removed_keys_by_ref(bam, out_bam):
        got = set() if ref is None else removed_from_flags(flag_reference(bam, fasta, ref, transcripts, [setting]), tag)
        a, b = want - got, got - want
        n_o += len(want)
        n_r += len(got)
        only_o += len(a)
        only_r += len(b)
        examples.extend([["only_original", *k] for k in sorted(a, key=str)[:20]]
                        + [["only_reimpl", *k] for k in sorted(b, key=str)[:20]])
    return {"setting": tag, "n_original": n_o, "n_reimpl": n_r, "only_original": only_o, "only_reimpl": only_r,
            "identical": only_o == 0 and only_r == 0, "examples": examples[:20]}


def none_control(bam, refs, out_dir, db, fasta, gtf, covlen=300, threads=4):
    """Remove-all control in intron mode on a reference subset: subset BAM, createTRANS -i, original in-process, comparison.

    Files in out_dir are prefixed with the joined reference names; existing subset, TRANS and filtered BAM are reused.
    The original writes <label>.<tag>.tmp.bam, renamed when it returns, so a filtered BAM is never a truncated one.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = "_".join(refs)
    st = Setting(covlen, None, 0, True)
    sub, trans = out_dir / f"{label}.bam", out_dir / f"{label}.trans_i.pkl"
    filtered = out_dir / f"{label}.{setting_tag(st)}.bam"
    partial = out_dir / f"{label}.{setting_tag(st)}.tmp.bam"
    secs = {}
    t0 = time.monotonic()
    if not Path(f"{sub}.bai").exists():
        subset_bam(bam, refs, sub)
    secs["subset"] = time.monotonic() - t0
    t0 = time.monotonic()
    if not trans.exists():
        create_trans(db, sub, trans, True, out_dir / f"{label}.createTRANS_i")
    secs["createTRANS"] = time.monotonic() - t0
    t0 = time.monotonic()
    if not filtered.exists():
        if partial.exists():
            subprocess.run([*TRASH, str(partial)], check=True)
        run_original(sub, trans, db, fasta, partial, st, threads=threads)
        partial.rename(filtered)
    secs["original"] = time.monotonic() - t0
    t0 = time.monotonic()
    res = compare_subset(sub, filtered, fasta, load_transcripts(gtf), st)
    secs["compare"] = time.monotonic() - t0
    with pysam.AlignmentFile(str(sub)) as fh:
        n = sum(s.total for s in fh.get_index_statistics())
    return {"refs": list(refs), "n_records": n, **res, "createTRANS": run_stats(out_dir / f"{label}.createTRANS_i"),
            "seconds": secs}


def _completed(out, prefix):
    """True when out exists and <prefix>.time.txt reports exit 0; an out left by a failed or killed run goes to the trash
    (the original tool and createTRANS return early on an existing output)."""
    out, tfile = Path(out), Path(f"{prefix}.time.txt")
    if out.exists() and tfile.exists() and time_exit_status(tfile) == 0:
        return True
    if out.exists():
        subprocess.run([*TRASH, str(out)], check=True)
    return False


def subset_validation(bam, refs, out_dir, db, fasta, gtf, settings, threads=4):
    """Original tool against the reimplementation on a reference subset: subset BAM, createTRANS per mode, then per setting
    the original in a measured child process and the comparison, saved as <label>.<tag>.json.

    Files in out_dir are prefixed with the joined reference names. Resume: an existing subset (with .bai) is reused, TRANS
    files and filtered BAMs are reused when their time file reports exit 0, and a setting with a result file is not rerun.
    A mode's TRANS file is checked and built only while one of its settings lacks its result file; its createTRANS cost is
    None when the time file is gone.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = "_".join(refs)
    sub = out_dir / f"{label}.bam"
    if not Path(f"{sub}.bai").exists():
        subset_bam(bam, refs, sub)
    modes = {"i" if s.introns else "e": s.introns for s in settings}
    trans = {m: out_dir / f"{label}.trans_{m}.pkl" for m in modes}
    ct = {m: out_dir / f"{label}.createTRANS_{m}" for m in modes}
    need = {"i" if s.introns else "e" for s in settings if not (out_dir / f"{label}.{setting_tag(s)}.json").exists()}
    todo = [m for m in modes if m in need and not _completed(trans[m], ct[m])]
    with ThreadPoolExecutor(max(len(todo), 1)) as ex:
        list(ex.map(lambda m: create_trans(db, sub, trans[m], modes[m], ct[m]), todo))
    transcripts = None
    res = {}
    for st in settings:
        tag = setting_tag(st)
        done = out_dir / f"{label}.{tag}.json"
        if not done.exists():
            filtered, prefix = out_dir / f"{label}.{tag}.bam", out_dir / f"{label}.{tag}"
            if not _completed(filtered, prefix):
                run_original_timed(sub, trans["i" if st.introns else "e"], db, fasta, filtered, st, prefix, threads)
            transcripts = transcripts or load_transcripts(gtf)
            t0 = time.monotonic()
            cmp = compare_subset(sub, filtered, fasta, transcripts, st)
            cost = run_stats(prefix)
            done.write_text(json.dumps({**cmp, "original_elapsed_s": cost["elapsed_s"],
                                        "original_max_rss_gib": cost["max_rss_gib"],
                                        "original_peak_tree_rss_gib": cost["peak_tree_rss_gib"],
                                        "compare_s": time.monotonic() - t0}, indent=1))
        res[tag] = json.loads(done.read_text())
    with pysam.AlignmentFile(str(sub)) as fh:
        n = sum(s.total for s in fh.get_index_statistics())
    return {"refs": list(refs), "n_records": n,
            "createTRANS": {m: run_stats(ct[m]) if Path(f"{ct[m]}.time.txt").exists() else None for m in modes},
            "settings": res}


def main(argv):
    """python -m pafbench.validation BAM TRANS_PKL DB FASTA OUT_BAM TAG THREADS: run_original in this process."""
    bam, trans_pkl, db, fasta, out_bam, tag, threads = argv
    run_original(bam, trans_pkl, db, fasta, out_bam, parse_tag(tag), int(threads))


if __name__ == "__main__":
    main(sys.argv[1:])

