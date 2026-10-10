import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pysam
import pytest

import pafbench.validation as validation
from pafbench.config import LIBRARIES, OUT
from pafbench.flags import Setting, parse_setting, record_key, removed_keys, setting_tag
from pafbench.grid import all_settings, published
from pafbench.transcripts import load_transcripts
from pafbench.validation import (
    EXPECTED_RUNS, compare_subset, create_trans, discover_settings, engine_tie, grid_removed, missing_runs, none_control,
    parse_tag, reusable, run_original, run_stats, setting_arg, subset_bam, subset_validation,
)
from tests.synth import run_original as serial_original

REPO = Path(__file__).resolve().parents[1]
OK_TIME = "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03\n\tMaximum resident set size (kbytes): 2097152\n\tExit status: 0\n"
# GNU time on a command killed by a signal (OOM kill, SIGKILL): a "terminated by signal" line, yet "Exit status: 0"
KILLED_TIME = ("Command terminated by signal 9\n\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:10.00\n"
               "\tMaximum resident set size (kbytes): 1048576\n\tExit status: 0\n")
FAILED_TIME = ("Command exited with non-zero status 3\n\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:10.00\n"
               "\tMaximum resident set size (kbytes): 1048576\n\tExit status: 3\n")


def _keys(bam, ref=None):
    with pysam.AlignmentFile(str(bam)) as fh:
        return [record_key(a) for a in fh.fetch(until_eof=True) if ref is None or a.reference_name == ref]


# ---------- stage 19 ----------

def _repo19(tmp_path):
    """A repo layout around a copy of stage 19, so $REPO/.venv is the running venv and pafbench is this checkout."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "19_validation_runs.sh", repo / "scripts")
    (repo / ".venv").symlink_to(Path(sys.executable).parents[1])
    (repo / "pafbench").symlink_to(REPO / "pafbench")
    return repo


def _dryrun(repo, out):
    # the dry run prints the trash command, so a PAFBENCH_TRASH exported by the caller must not reach it
    env = {k: v for k, v in os.environ.items() if k != "PAFBENCH_TRASH"}
    r = subprocess.run(["bash", str(repo / "scripts" / "19_validation_runs.sh")], capture_output=True, text=True,
                       check=True, env=env | {"DRYRUN": "1", "PAFBENCH_OUT": str(out)})
    return r.stdout.splitlines()


def _filter_line(repo, out, lib, c, l, m, mode):
    return (f"{repo}/.venv/bin/python -m pafbench.memwatch {out}/{lib}/original/c{c}_l{l}_m{m}_{mode}.mem.json -- "
            f"{repo}/scripts/02_run_original.sh {lib} {c} {l} {m} {mode}")


def _trans_line(repo, out, mode):
    py, d = f"{repo}/.venv/bin/python", f"{out}/1419/original"
    flag = " -i" if mode == "i" else ""
    return (f"{py} -m pafbench.memwatch {d}/createTRANS_{mode}.mem.json -- /usr/bin/time -v -o {d}/createTRANS_{mode}.time.txt "
            f"{py} {repo}/vendor/polyAfilter/polyAfilter.py createTRANS{flag} {OUT}/ref/genes_2024A.db {LIBRARIES['1419']} "
            f"{d}/trans_{mode}.pkl > {d}/createTRANS_{mode}.log 2>&1 &")


CHAIN_1417 = [(275, 9, 1, "i"), (275, 9, 1, "e"), (180, 11, 2, "i"), (180, 11, 2, "e"), (300, 10, 0, "i"), (600, 10, 1, "i"),
              (375, 3, 2, "e")]
CHAIN_1419 = [(300, 10, 1, "i"), (375, 3, 2, "i"), (300, 10, 1, "e"), (275, 9, 1, "i"), (275, 9, 1, "e"), (180, 11, 2, "i"),
              (180, 11, 2, "e"), (375, 3, 2, "e")]


def test_stage19_dryrun_order(tmp_path):
    repo, out = _repo19(tmp_path), tmp_path / "out"
    out.mkdir()
    lines = _dryrun(repo, out)
    want = (["# chain 1417"] + [_filter_line(repo, out, "1417", *s) for s in CHAIN_1417]
            + ["# chain 1419", _trans_line(repo, out, "i"), _trans_line(repo, out, "e"), "wait"]
            + [_filter_line(repo, out, "1419", *s) for s in CHAIN_1419])
    assert lines == want
    # a dry run writes nothing: no log, no original/ directories
    assert list(out.iterdir()) == []


MILESTONE1_1417 = ["c300_l10_m1_i", "c375_l3_m2_i", "c300_l10_m1_e"]


def test_expected_runs_are_pinned_to_stage19(tmp_path):
    repo, out = _repo19(tmp_path), tmp_path / "out"
    out.mkdir()
    listed = {"1417": list(MILESTONE1_1417), "1419": []}
    for line in _dryrun(repo, out):
        if "02_run_original.sh" in line and not line.startswith("#"):
            lib, c, l, m, mode = line.split("02_run_original.sh ")[1].split()
            listed[lib].append(f"c{c}_l{l}_m{m}_{mode}")
    assert {lib: sorted(tags) for lib, tags in EXPECTED_RUNS.items()} == {lib: sorted(tags) for lib, tags in listed.items()}
    assert [len(EXPECTED_RUNS[lib]) for lib in ("1417", "1419")] == [10, 8]
    assert all(len(set(tags)) == len(tags) for tags in EXPECTED_RUNS.values())


def test_stage19_skip_rule(tmp_path):
    repo, out = _repo19(tmp_path), tmp_path / "out"
    o17, o19 = out / "1417" / "original", out / "1419" / "original"
    o17.mkdir(parents=True)
    o19.mkdir(parents=True)
    # done: BAM plus a successful time file -> skipped
    (o17 / "c275_l9_m1_i.bam").touch()
    (o17 / "c275_l9_m1_i.time.txt").write_text(OK_TIME)
    # killed by a signal: BAM exists, time file says "Exit status: 0" after a "terminated by signal" line -> stale BAM
    # trashed (BAMfilter returns early on an existing BAM), then rerun
    (o17 / "c275_l9_m1_e.bam").touch()
    (o17 / "c275_l9_m1_e.time.txt").write_text(KILLED_TIME)
    # failed with a non-zero exit: same
    (o17 / "c180_l11_m2_e.bam").touch()
    (o17 / "c180_l11_m2_e.time.txt").write_text(FAILED_TIME)
    # successful time file but no BAM -> rerun
    (o17 / "c180_l11_m2_i.time.txt").write_text(OK_TIME)
    # 1419 createTRANS i done -> skipped; e still runs, and the pair is still followed by wait
    (o19 / "trans_i.pkl").touch()
    (o19 / "createTRANS_i.time.txt").write_text(OK_TIME)
    lines = _dryrun(repo, out)
    want = (["# chain 1417", "# skip 1417 c275_l9_m1_i", f"gio trash {o17}/c275_l9_m1_e.bam"]
            + [_filter_line(repo, out, "1417", *s) for s in CHAIN_1417[1:3]]
            + [f"gio trash {o17}/c180_l11_m2_e.bam"] + [_filter_line(repo, out, "1417", *s) for s in CHAIN_1417[3:]]
            + ["# chain 1419", "# skip 1419 createTRANS_i", _trans_line(repo, out, "e"), "wait"]
            + [_filter_line(repo, out, "1419", *s) for s in CHAIN_1419])
    assert lines == want
    assert (o17 / "c275_l9_m1_e.bam").exists() and (o17 / "c180_l11_m2_e.bam").exists()


FAKE_02 = """#!/usr/bin/env bash
# stand-in for 02_run_original.sh: writes the BAM and a GNU time file under PAFBENCH_OUT; FAIL=LIB/TAG exits 3
set -euo pipefail
TAG=c$2_l$3_m$4_$5
D=$PAFBENCH_OUT/$1/original
RC=0
[ "$1/$TAG" = "${FAIL:-}" ] && RC=3
[ -e "$D/$TAG.bam" ] || echo partial > "$D/$TAG.bam"   # like BAMfilter, leave an existing output alone
printf '\\tExit status: %s\\n' "$RC" > "$D/$TAG.time.txt"
exit "$RC"
"""

FAKE_POLYAFILTER = """import sys, time
time.sleep(1)
open(sys.argv[-1], "w").write(" ".join(sys.argv[1:]))
"""


def _log(path):
    """(event, name, exit) per line of a runs_<lib>.log after the timestamp."""
    out = []
    for line in path.read_text().splitlines():
        words = line.split()
        out.append((words[2], words[3].rstrip(":"), words[5] if words[2] == "end" else None))
    return out


def _repo19_fakes(tmp_path):
    """_repo19 with stand-ins for 02_run_original.sh and polyAfilter.py; env runs it for real with trash into tmp/trash."""
    repo, out = _repo19(tmp_path), tmp_path / "out"
    (repo / "scripts" / "02_run_original.sh").write_text(FAKE_02)
    (repo / "scripts" / "02_run_original.sh").chmod(0o755)
    (repo / "vendor" / "polyAfilter").mkdir(parents=True)
    (repo / "vendor" / "polyAfilter" / "polyAfilter.py").write_text(FAKE_POLYAFILTER)
    (tmp_path / "trash").mkdir()
    env = {**os.environ, "PAFBENCH_OUT": str(out), "PAFBENCH_TRASH": f"mv -t {tmp_path / 'trash'}"}
    env.pop("DRYRUN", None)
    return repo, out, env, ["bash", str(repo / "scripts" / "19_validation_runs.sh")]


def test_stage19_runs_resumes_and_reports_failure(tmp_path):
    repo, out, env, script = _repo19_fakes(tmp_path)
    env["FAIL"] = "1417/c180_l11_m2_e"
    assert subprocess.run(script, env=env).returncode == 1
    o17, o19 = out / "1417" / "original", out / "1419" / "original"
    tags17 = [f"c{c}_l{l}_m{m}_{mode}" for c, l, m, mode in CHAIN_1417]
    tags19 = [f"c{c}_l{l}_m{m}_{mode}" for c, l, m, mode in CHAIN_1419]
    # the failed 1417 step does not stop the chain; both chains ran every step, each under memwatch
    assert all((o17 / f"{t}.bam").exists() and (o17 / f"{t}.mem.json").exists() for t in tags17)
    assert all((o19 / f"{t}.bam").exists() for t in tags19)
    assert json.loads((o17 / "c180_l11_m2_e.mem.json").read_text())["exit_status"] == 3
    # createTRANS got 02_run_original.sh's arguments; its time file and log sit next to the TRANS file
    db_bam = f"{OUT}/ref/genes_2024A.db {LIBRARIES['1419']}"
    assert (o19 / "trans_i.pkl").read_text() == f"createTRANS -i {db_bam} {o19}/trans_i.pkl"
    assert (o19 / "trans_e.pkl").read_text() == f"createTRANS {db_bam} {o19}/trans_e.pkl"
    assert run_stats(o19 / "createTRANS_i")["exit_status"] == 0 and (o19 / "createTRANS_e.log").exists()
    log17 = _log(out / "validation" / "runs_1417.log")
    assert log17 == [e for t in tags17 for e in (("start", t, None), ("end", t, "3" if t == "c180_l11_m2_e" else "0"))]
    log19 = _log(out / "validation" / "runs_1419.log")
    # the createTRANS pair runs in parallel: both start before either ends, and the filter steps follow
    assert {e[0] for e in log19[:2]} == {"start"} and {e[0] for e in log19[2:4]} == {"end"}
    assert {e[1] for e in log19[:4]} == {"createTRANS_i", "createTRANS_e"}
    assert log19[4:] == [e for t in tags19 for e in (("start", t, None), ("end", t, "0"))]
    # resume: completed steps are skipped, the failed one is trashed and rerun
    env.pop("FAIL")
    (o17 / "c180_l11_m2_e.bam").write_text("stale")
    assert subprocess.run(script, env=env).returncode == 0
    assert (o17 / "c180_l11_m2_e.bam").read_text() == "partial\n"
    assert [p.read_text() for p in (tmp_path / "trash").iterdir()] == ["stale"]
    assert _log(out / "validation" / "runs_1417.log")[len(log17):] == (
        [("skip", t, None) for t in tags17[:3]] + [("start", "c180_l11_m2_e", None), ("end", "c180_l11_m2_e", "0")]
        + [("skip", t, None) for t in tags17[4:]])
    assert {e[0] for e in _log(out / "validation" / "runs_1419.log")[len(log19):]} == {"skip"}


def test_stage19_refuses_a_second_concurrent_run(tmp_path):
    repo, out, env, script = _repo19_fakes(tmp_path)
    (out / "validation").mkdir(parents=True)
    with open(out / "validation" / "runs.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)   # a running instance holds the lock
        r = subprocess.run(script, env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "runs.lock" in r.stderr
    # nothing ran: no log, no output, no memwatch file, nothing trashed
    assert sorted(p.name for p in out.rglob("*") if p.is_file()) == ["runs.lock"]
    assert list((tmp_path / "trash").iterdir()) == []
    # once the lock is free the same command runs every step
    assert subprocess.run(script, env=env).returncode == 0
    assert len(list((out / "1417" / "original").glob("*.bam"))) == len(CHAIN_1417)


# ---------- tags and discovery ----------

def test_parse_tag_and_setting_arg_invert_all_settings():
    for s in all_settings():
        assert parse_tag(setting_tag(s)) == s
        assert parse_setting(setting_arg(s)) == s
    assert parse_tag("c300_lnone_m0_i") == Setting(300, None, 0, True)
    assert setting_arg(Setting(375, 3, 2, False)) == "375,3,2,e"
    for bad in ("c300_l0_m0_i", "c300_l10_m1_x", "createTRANS_i", "c300_l10_m1_i.bam_True_chr1"):
        with pytest.raises(ValueError):
            parse_tag(bad)


def test_discover_settings(tmp_path):
    for tag, txt in (("c300_l10_m1_i", OK_TIME), ("c375_l3_m2_i", KILLED_TIME), ("c300_lnone_m0_e", OK_TIME),
                     ("c180_l11_m2_i", FAILED_TIME)):
        (tmp_path / f"{tag}.bam").touch()
        (tmp_path / f"{tag}.time.txt").write_text(txt)
    (tmp_path / "c180_l11_m2_e.bam").touch()                  # still running: no time file yet
    (tmp_path / "c300_l10_m1_i.bam_True_chr1.bam").touch()    # BAMfilter -p temp file
    (tmp_path / "trans_i.pkl").touch()
    (tmp_path / "createTRANS_i.time.txt").write_text(OK_TIME)
    (tmp_path / "c275_l9_m1_e.time.txt").write_text(OK_TIME)  # time file without BAM
    # sorted by file name: "c300_l1..." < "c300_ln..."
    assert discover_settings(tmp_path) == [Setting(300, 10, 1, True), Setting(300, None, 0, False)]
    assert discover_settings(tmp_path / "missing") == []


def test_missing_runs_names_the_reason(tmp_path):
    for tag, txt in (("c300_l10_m1_i", OK_TIME), ("c375_l3_m2_i", KILLED_TIME), ("c180_l11_m2_i", FAILED_TIME)):
        (tmp_path / f"{tag}.bam").touch()
        (tmp_path / f"{tag}.time.txt").write_text(txt)
    (tmp_path / "c300_l10_m1_e.bam").touch()                  # still running or lost its time file
    (tmp_path / "c275_l9_m1_i.time.txt").write_text(OK_TIME)  # time file without BAM
    tags = ["c300_l10_m1_i", "c375_l3_m2_i", "c300_l10_m1_e", "c180_l11_m2_i", "c275_l9_m1_i", "c275_l9_m1_e"]
    assert missing_runs(tmp_path, tags) == [
        {"tag": "c375_l3_m2_i", "exit_status": 137, "reason": "killed or failed"},
        {"tag": "c300_l10_m1_e", "exit_status": None, "reason": "no time file"},
        {"tag": "c180_l11_m2_i", "exit_status": 3, "reason": "killed or failed"},
        {"tag": "c275_l9_m1_i", "exit_status": 0, "reason": "no output"},
        {"tag": "c275_l9_m1_e", "exit_status": None, "reason": "no output"},
    ]
    assert missing_runs(tmp_path, ["c300_l10_m1_i"]) == []
    assert [m["reason"] for m in missing_runs(tmp_path / "missing", ["c300_l10_m1_i"])] == ["no output"]


def test_run_stats(tmp_path):
    (tmp_path / "x.time.txt").write_text(OK_TIME)
    # 1:02:03 = 3723 s; 2097152 kB = 2 GiB
    assert run_stats(tmp_path / "x") == {"elapsed_s": 3723.0, "max_rss_gib": 2.0, "exit_status": 0, "peak_tree_rss_gib": None}
    (tmp_path / "x.mem.json").write_text(json.dumps({"exit_status": 0, "elapsed_s": 3724.5, "peak_tree_rss_gib": 3.5}))
    assert run_stats(tmp_path / "x")["peak_tree_rss_gib"] == 3.5
    # killed by signal 9: GNU time writes "Exit status: 0", the run's status is 128 + 9
    (tmp_path / "k.time.txt").write_text(KILLED_TIME)
    assert run_stats(tmp_path / "k") == {"elapsed_s": 10.0, "max_rss_gib": 1.0, "exit_status": 137, "peak_tree_rss_gib": None}
    (tmp_path / "f.time.txt").write_text(FAILED_TIME)
    assert run_stats(tmp_path / "f")["exit_status"] == 3


def test_killed_fixture_is_what_gnu_time_writes(tmp_path):
    # the premise of KILLED_TIME, checked against this machine's /usr/bin/time: a SIGKILLed command
    r = subprocess.run(["/usr/bin/time", "-v", "-o", str(tmp_path / "k.time.txt"), sys.executable, "-c",
                        "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"])
    text = (tmp_path / "k.time.txt").read_text()
    assert r.returncode == 137
    assert text.startswith("Command terminated by signal 9\n") and text.endswith("\tExit status: 0\n")
    assert run_stats(tmp_path / "k")["exit_status"] == 137
    (tmp_path / "c300_l10_m1_i.bam").touch()
    (tmp_path / "c300_l10_m1_i.time.txt").write_text(text)
    assert discover_settings(tmp_path) == []


# ---------- subset and the remove-all control ----------

def test_subset_bam(synth, tmp_path):
    sub = tmp_path / "b.bam"
    subset_bam(synth.bam, ["chrB"], sub)
    assert _keys(sub) == _keys(synth.bam, "chrB")
    with pysam.AlignmentFile(str(sub)) as fh:
        assert fh.references == ("chrA", "chrB")
        assert sum(1 for _ in fh.fetch("chrA")) == 0
        assert sum(1 for _ in fh.fetch("chrB")) == len(_keys(synth.bam, "chrB"))
    both = tmp_path / "ba.bam"
    subset_bam(synth.bam, ["chrB", "chrA"], both)   # reference order as given is ignored: output stays in file order
    assert _keys(both) == _keys(synth.bam)


# removed-record counts of the original's serial path on the synthetic BAM (tests.synth.run_original), pinned so a
# mode or MISM that is not passed through changes the count: 300/10/1 removes 79 in -i mode and 66 in exon mode
SERIAL_COUNTS = {"c300_lnone_m0_i": 123, "c150_lnone_m0_e": 108, "c300_l10_m1_i": 79, "c300_l10_m1_e": 66}


@pytest.mark.parametrize("st", [Setting(300, None, 0, True), Setting(150, None, 0, False), Setting(300, 10, 1, True),
                                Setting(300, 10, 1, False)], ids=setting_tag)
def test_run_original_matches_reimpl_and_serial_original(synth, tmp_path, st):
    trans = tmp_path / "trans.pkl"
    create_trans(synth.db, synth.bam, trans, st.introns, tmp_path / "createTRANS")
    assert run_stats(tmp_path / "createTRANS")["exit_status"] == 0
    out = tmp_path / "orig.bam"
    run_original(synth.bam, trans, synth.db, synth.fasta, out, st, threads=2)
    res = compare_subset(synth.bam, out, synth.fasta, load_transcripts(synth.gtf), st)
    # independent reference: the original's serial path, fresh tROC, TRANS built in-process
    want = serial_original(synth, st.covlen, st.min_len, st.mism, st.introns)
    assert len(want) == SERIAL_COUNTS[setting_tag(st)]
    assert removed_keys(synth.bam, out) == want
    assert res["setting"] == setting_tag(st) and res["identical"] is True
    assert res["n_original"] == res["n_reimpl"] == len(want)
    assert res["only_original"] == res["only_reimpl"] == 0 and res["examples"] == []
    # the -p temp files next to out_bam were merged and deleted by the tool
    assert sorted(p.name for p in tmp_path.glob("orig.bam*")) == ["orig.bam"]


def test_compare_subset_reports_one_sided_differences(synth):
    # out_bam = the input itself: the "original" removed nothing, so every reimpl removal is only_reimpl
    for st in (Setting(300, None, 0, True), Setting(300, 10, 1, False)):
        want = serial_original(synth, st.covlen, st.min_len, st.mism, st.introns)
        res = compare_subset(synth.bam, synth.bam, synth.fasta, load_transcripts(synth.gtf), st)
        assert res["setting"] == setting_tag(st)
        assert res["identical"] is False and res["n_original"] == 0 and res["only_original"] == 0
        assert res["n_reimpl"] == res["only_reimpl"] == len(want) > 20
        assert len(res["examples"]) == 20 and all(e[0] == "only_reimpl" and tuple(e[1:]) in want for e in res["examples"])


def test_none_control_on_a_reference_subset(synth, tmp_path):
    nc = tmp_path / "nc"
    res = none_control(synth.bam, ["chrA"], nc, synth.db, synth.fasta, synth.gtf, threads=2)
    # the TRANS of a chrA-only BAM has the same chrA transcripts, so the result is the full run restricted to chrA
    want = {k for k in serial_original(synth, 300, None, 0, True) if k[2] == "chrA"}
    assert res["refs"] == ["chrA"] and res["setting"] == "c300_lnone_m0_i"
    assert res["n_records"] == len(_keys(synth.bam, "chrA"))
    assert res["identical"] is True and res["n_original"] == res["n_reimpl"] == len(want) > 0
    assert res["createTRANS"]["exit_status"] == 0 and res["createTRANS"]["peak_tree_rss_gib"] > 0
    assert set(res["seconds"]) == {"subset", "createTRANS", "original", "compare"}
    assert sorted(p.name for p in nc.iterdir()) == [
        "chrA.bam", "chrA.bam.bai", "chrA.c300_lnone_m0_i.bam", "chrA.createTRANS_i.log", "chrA.createTRANS_i.mem.json",
        "chrA.createTRANS_i.time.txt", "chrA.trans_i.pkl",
    ]


def test_none_control_filters_under_a_temporary_name(synth, tmp_path, monkeypatch):
    trash = tmp_path / "trash"
    trash.mkdir()
    monkeypatch.setattr(validation, "TRASH", ("mv", "-t", str(trash)))
    nc = tmp_path / "nc"
    nc.mkdir()
    seen = []
    real = validation.run_original

    def spy(bam, trans_pkl, db, fasta, out_bam, setting, threads=4):
        seen.append(Path(out_bam).name)
        real(bam, trans_pkl, db, fasta, out_bam, setting, threads=threads)

    monkeypatch.setattr(validation, "run_original", spy)
    # a truncated leftover of a killed run under the temporary name is trashed, never mistaken for the result
    tmp = nc / "chrA.c300_lnone_m0_i.tmp.bam"
    tmp.write_text("truncated")
    res = none_control(synth.bam, ["chrA"], nc, synth.db, synth.fasta, synth.gtf, threads=2)
    assert seen == ["chrA.c300_lnone_m0_i.tmp.bam"] and (trash / tmp.name).read_text() == "truncated"
    assert not tmp.exists() and (nc / "chrA.c300_lnone_m0_i.bam").exists() and res["identical"] is True
    # a finished filtered BAM only ever appears by rename, so it is reused
    seen.clear()
    assert none_control(synth.bam, ["chrA"], nc, synth.db, synth.fasta, synth.gtf, threads=2)["identical"] is True
    assert seen == []
    # the original dying leaves the temporary file and no final BAM
    (nc / "chrA.c300_lnone_m0_i.bam").unlink()

    def die(bam, trans_pkl, db, fasta, out_bam, setting, threads=4):
        Path(out_bam).write_text("truncated")
        raise RuntimeError("killed")

    monkeypatch.setattr(validation, "run_original", die)
    with pytest.raises(RuntimeError):
        none_control(synth.bam, ["chrA"], nc, synth.db, synth.fasta, synth.gtf, threads=2)
    assert tmp.exists() and not (nc / "chrA.c300_lnone_m0_i.bam").exists()


# ---------- original tool on a reference subset at given settings (1179 block of stage 20) ----------

def _recorded(monkeypatch):
    """Wrap create_trans and run_original_timed so each call is logged as ("trans", mode) or ("run", tag)."""
    calls = []
    real_trans, real_run = validation.create_trans, validation.run_original_timed

    def trans(db, bam, out, introns, prefix):
        calls.append(("trans", "i" if introns else "e"))
        real_trans(db, bam, out, introns, prefix)

    def run(bam, trans_pkl, db, fasta, out_bam, setting, prefix, threads=4):
        calls.append(("run", setting_tag(setting)))
        real_run(bam, trans_pkl, db, fasta, out_bam, setting, prefix, threads)

    monkeypatch.setattr(validation, "create_trans", trans)
    monkeypatch.setattr(validation, "run_original_timed", run)
    return calls


def _without(res, *keys):
    """res with the given keys dropped from every setting's entry."""
    return {**res, "settings": {t: {k: v for k, v in r.items() if k not in keys} for t, r in res["settings"].items()}}


COST = ("original_elapsed_s", "original_max_rss_gib", "original_peak_tree_rss_gib")


def test_subset_validation_runs_compares_and_resumes(synth, tmp_path, monkeypatch):
    trash = tmp_path / "trash"
    trash.mkdir()
    monkeypatch.setattr(validation, "TRASH", ("mv", "-t", str(trash)))
    calls = _recorded(monkeypatch)
    st_i, st_e = Setting(300, 10, 1, True), Setting(300, 10, 1, False)
    d = tmp_path / "sub"
    res = subset_validation(synth.bam, ["chrA"], d, synth.db, synth.fasta, synth.gtf, [st_i, st_e], threads=2)
    # one createTRANS per mode on the subset, then one original run per setting
    assert sorted(calls[:2]) == [("trans", "e"), ("trans", "i")]
    assert calls[2:] == [("run", "c300_l10_m1_i"), ("run", "c300_l10_m1_e")]
    assert res["refs"] == ["chrA"] and res["n_records"] == len(_keys(synth.bam, "chrA"))
    assert set(res["createTRANS"]) == {"i", "e"} and all(v["exit_status"] == 0 for v in res["createTRANS"].values())
    assert list(res["settings"]) == ["c300_l10_m1_i", "c300_l10_m1_e"]
    for st in (st_i, st_e):
        r = res["settings"][setting_tag(st)]
        # every synthetic removal is on chrA, so the chrA subset removes what the full serial run removes (79 i, 66 e)
        want = serial_original(synth, st.covlen, st.min_len, st.mism, st.introns)
        assert r["identical"] is True and r["n_original"] == r["n_reimpl"] == len(want) == SERIAL_COUNTS[setting_tag(st)]
        assert r["only_original"] == r["only_reimpl"] == 0 and r["examples"] == []
        # the filtered BAM is the original's output for this setting
        assert removed_keys(d / "chrA.bam", d / f"chrA.{setting_tag(st)}.bam") == want
        # run cost of the measured child process: GNU time and memwatch both written, exit 0
        cost = run_stats(d / f"chrA.{setting_tag(st)}")
        assert cost["exit_status"] == 0
        assert (r["original_elapsed_s"], r["original_max_rss_gib"], r["original_peak_tree_rss_gib"]) == (
            cost["elapsed_s"], cost["max_rss_gib"], cost["peak_tree_rss_gib"])
        assert r["original_max_rss_gib"] > 0 and r["original_peak_tree_rss_gib"] > 0 and r["compare_s"] >= 0
        # the child runs this module on the subset with the TRANS file of its own mode (the synthetic TRANS files of the
        # two modes give the same removals, so only the command line shows a swap)
        m = "i" if st.introns else "e"
        assert json.loads((d / f"chrA.{setting_tag(st)}.mem.json").read_text())["cmd"][-9:] == [
            "-m", "pafbench.validation", str(d / "chrA.bam"), str(d / f"chrA.trans_{m}.pkl"), str(synth.db), str(synth.fasta),
            str(d / f"chrA.{setting_tag(st)}.bam"), setting_tag(st), "2"]
        # and THREADS reaches the tool: its log line names 2 pool processes (run_original_timed's default is 4)
        assert ("Identifying alignments 300 bp upstream of non-terminal SNR10+ with up to 1 mismatches "
                f"({'including' if st.introns else 'excluding'} introns) to be removed across 2 parallel processes") in (
            d / f"chrA.{setting_tag(st)}.log").read_text()
        assert json.loads((d / f"chrA.{setting_tag(st)}.json").read_text()) == r
    assert sorted(p.name for p in d.iterdir()) == sorted(
        ["chrA.bam", "chrA.bam.bai"]
        + [f"chrA.trans_{m}.pkl" for m in "ie"] + [f"chrA.createTRANS_{m}.{x}" for m in "ie" for x in ("log", "mem.json", "time.txt")]
        + [f"chrA.{t}.{x}" for t in ("c300_l10_m1_i", "c300_l10_m1_e") for x in ("bam", "json", "log", "mem.json", "time.txt")])

    def again():
        calls.clear()
        return subset_validation(synth.bam, ["chrA"], d, synth.db, synth.fasta, synth.gtf, [st_i, st_e], threads=2)

    # everything done: nothing reruns, the result files are read back
    assert again() == res and calls == []
    # result file gone but the run finished (BAM plus a successful time file): compared again, not rerun
    os.replace(d / "chrA.c300_l10_m1_i.json", trash / "i.json")
    assert _without(again(), "compare_s") == _without(res, "compare_s") and calls == []
    # killed run: a partial BAM next to a time file with a signal line goes to the trash and the setting reruns
    os.replace(d / "chrA.c300_l10_m1_e.json", trash / "e.json")
    (d / "chrA.c300_l10_m1_e.bam").write_text("partial")
    (d / "chrA.c300_l10_m1_e.time.txt").write_text(KILLED_TIME)
    assert _without(again(), "compare_s", *COST) == _without(res, "compare_s", *COST) and calls == [("run", "c300_l10_m1_e")]
    assert (trash / "chrA.c300_l10_m1_e.bam").read_text() == "partial"
    # stopped before the child wrote anything (GNU time truncates its file at start): a partial BAM next to an empty time
    # file goes to the trash and the setting reruns
    os.replace(d / "chrA.c300_l10_m1_i.json", trash / "i2.json")
    (d / "chrA.c300_l10_m1_i.bam").write_text("partial")
    (d / "chrA.c300_l10_m1_i.time.txt").write_text("")
    assert _without(again(), "compare_s", *COST) == _without(res, "compare_s", *COST) and calls == [("run", "c300_l10_m1_i")]
    assert (trash / "chrA.c300_l10_m1_i.bam").read_text() == "partial"
    # every setting has its result file: a failed createTRANS e and a missing createTRANS i time file trigger no rebuild,
    # and the cost entries show the failure and the gap
    (d / "chrA.createTRANS_e.time.txt").write_text(FAILED_TIME)
    os.replace(d / "chrA.createTRANS_i.time.txt", trash / "createTRANS_i.time.txt")
    got = again()
    assert calls == [] and (d / "chrA.trans_e.pkl").exists() and (d / "chrA.trans_i.pkl").exists()
    assert got["createTRANS"]["e"]["exit_status"] == 3 and got["createTRANS"]["i"] is None
    assert {**_without(got, "compare_s", *COST), "createTRANS": None} == {**_without(res, "compare_s", *COST), "createTRANS": None}
    # once an e setting lacks its result file, the failed TRANS file is trashed and rebuilt; the filtered BAM is complete, so
    # the setting is compared again, not rerun, and mode i (all results present) is left alone
    os.replace(d / "chrA.c300_l10_m1_e.json", trash / "e2.json")
    got = again()
    assert calls == [("trans", "e")] and (trash / "chrA.trans_e.pkl").exists() and got["createTRANS"]["e"]["exit_status"] == 0
    assert got["createTRANS"]["i"] is None
    assert {**_without(got, "compare_s", *COST), "createTRANS": None} == {**_without(res, "compare_s", *COST), "createTRANS": None}


# ---------- stage 20 wiring ----------

FAKE_03 = """import os, sys
from pathlib import Path
(Path(os.environ["FAKE_OUT"]) / sys.argv[1] / "args03.txt").write_text(" ".join(sys.argv[2:]))
"""

# stand-in for stage 04: FAKE04=crash dies like a truncated BAM does; otherwise writes FAKE04_RESULT as exact_match.json,
# prints the summary line and exits as stage 04 does
FAKE_04 = """import json, os, sys
from pathlib import Path
d = Path(os.environ["FAKE_OUT"]) / sys.argv[1]
(d / "args04.txt").write_text(" ".join(sys.argv[2:]))
if os.environ["FAKE04"] == "crash":
    raise OSError("no BGZF EOF marker; file may be truncated")
res = json.loads(os.environ["FAKE04_RESULT"])
(d / "exact_match.json").write_text(json.dumps(res))
print(json.dumps({k: v["identical"] for k, v in res.items()}))
sys.exit(0 if all(v["identical"] for v in res.values()) else 1)
"""


def _match(n_o, n_r, only_o, only_r):
    return {"n_original": n_o, "n_reimpl": n_r, "only_original": only_o, "only_reimpl": only_r,
            "identical": only_o == 0 and only_r == 0, "examples": []}


def _stage20(tmp_path, monkeypatch):
    """Stage 20 loaded as a module with REPO and OUT pointed at a stand-in layout (fake stages 03 and 04)."""
    repo, out = tmp_path / "repo", tmp_path / "out"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "03_reimpl_flags.py").write_text(FAKE_03)
    (repo / "scripts" / "04_compare_removed.py").write_text(FAKE_04)
    orig = out / "1417" / "original"
    orig.mkdir(parents=True)
    for tag in ("c300_l10_m1_i", "c300_lnone_m0_e"):
        (orig / f"{tag}.bam").touch()
        (orig / f"{tag}.time.txt").write_text(OK_TIME)
    spec = importlib.util.spec_from_file_location("stage20", REPO / "scripts" / "20_validation_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "REPO", repo)
    monkeypatch.setattr(mod, "OUT", out)
    monkeypatch.setenv("FAKE_OUT", str(out))
    return mod, out / "1417"


def test_compare_library_records_a_difference(tmp_path, monkeypatch):
    mod, d = _stage20(tmp_path, monkeypatch)
    result = {"c300_l10_m1_i": _match(10, 9, 1, 0), "c300_lnone_m0_e": _match(5, 5, 0, 0)}
    monkeypatch.setenv("FAKE04", "ok")
    monkeypatch.setenv("FAKE04_RESULT", json.dumps(result))
    got = mod.compare_library("1417", discover_settings(d / "original"), 7)
    # stage 03 gets every setting once, plus the worker count; stage 04 the same settings
    assert (d / "args03.txt").read_text() == "--setting 300,10,1,i --setting 300,none,0,e --workers 7"
    assert (d / "args04.txt").read_text() == "--setting 300,10,1,i --setting 300,none,0,e"
    # stage 04's exit 1 (a difference) is recorded; run cost from OK_TIME: 1:02:03 = 3723 s, 2 GiB, no mem.json
    cost = {"original_elapsed_s": 3723.0, "original_max_rss_gib": 2.0, "original_peak_tree_rss_gib": None}
    assert got == {tag: {**v, **cost} for tag, v in result.items()}
    assert got["c300_l10_m1_i"]["identical"] is False


def test_compare_library_raises_when_stage04_crashes(tmp_path, monkeypatch):
    mod, d = _stage20(tmp_path, monkeypatch)
    # an earlier run left an all-identical exact_match.json; this run's stage 04 dies with exit 1 and writes nothing
    (d / "exact_match.json").write_text(json.dumps({"c300_l10_m1_i": _match(10, 10, 0, 0),
                                                   "c300_lnone_m0_e": _match(5, 5, 0, 0)}))
    monkeypatch.setenv("FAKE04", "crash")
    with pytest.raises(RuntimeError, match="stage 04 on 1417"):
        mod.compare_library("1417", discover_settings(d / "original"), 7)

def test_stage20_without_libraries_or_control(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "20_validation_compare.py"), "--libs", "--skip-none-control",
                        "--skip-1179", "--out", str(tmp_path)], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0
    # no requested library: nothing missing, nothing tied, never complete
    assert json.loads((tmp_path / "validation" / "exact_match_m2.json").read_text()) == {
        "n_compared": 0, "all_identical": True, "missing": {}, "engine_tie": {}, "reused": {}, "complete": False}
    assert json.loads(r.stdout) == {"complete": False, "missing": {}, "none_control": None, "1179_subset": None, "n_compared": 0,
                                    "all_identical": True}


def test_stage20_1179_block_on_a_stand_in(synth, tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("stage20", REPO / "scripts" / "20_validation_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # the real block: chr21 + chr22 of 1179 at the 8 published settings (4 in each mode)
    assert mod.SUBSET_1179_REFS == ("chr21", "chr22") and mod.LIBRARIES["1179"] == LIBRARIES["1179"]
    assert len(published()) == 8 and sum(s.introns for s in published()) == 4
    monkeypatch.setattr(mod, "LIBRARIES", {**LIBRARIES, "1179": synth.bam})
    monkeypatch.setattr(mod, "SUBSET_1179_REFS", ("chrA", "chrB"))
    monkeypatch.setattr(mod, "DB", synth.db)
    monkeypatch.setattr(mod, "FASTA", synth.fasta)
    monkeypatch.setattr(mod, "GTF", synth.gtf)
    monkeypatch.setattr(sys, "argv", ["20", "--libs", "--skip-none-control", "--out", str(tmp_path)])
    # stage 13 parts of the subset references: both tags' removals split over the two references
    monkeypatch.setattr(mod, "OUT", tmp_path / "out")
    parts = tmp_path / "out" / "1179" / "grid" / "parts"
    parts.mkdir(parents=True)
    t0, t1 = (setting_tag(s) for s in published()[:2])
    n0, n1 = (len(serial_original(synth, s.covlen, s.min_len, s.mism, s.introns)) for s in published()[:2])
    for ref, a, b in (("chrA", n0 - 1, n1), ("chrB", 1, 0)):
        pd.DataFrame({"tag": [t0, t1], "re": ["x", "x"], "removed": [a, b], "total": [99, 99]}).to_parquet(
            parts / f"{ref}.region.parquet", index=False)
    calls = _recorded(monkeypatch)
    with pytest.raises(SystemExit) as ex:
        mod.main()
    tags = [setting_tag(s) for s in published()]
    assert sorted(calls[:2]) == [("trans", "e"), ("trans", "i")] and calls[2:] == [("run", t) for t in tags]
    em = json.loads((tmp_path / "validation" / "exact_match_m2.json").read_text())
    assert set(em) == {"1179_subset", "n_compared", "all_identical", "missing", "engine_tie", "reused", "complete"}
    assert em["n_compared"] == 8 and em["complete"] is False
    # grid counts of the subset references sum to the original's; tags without a part have none
    blk = em["1179_subset"]["settings"]
    assert (blk[t0]["n_grid"], blk[t0]["grid_equal"]) == (n0, True) and (blk[t1]["n_grid"], blk[t1]["grid_equal"]) == (n1, True)
    assert all(v["n_grid"] is None and v["grid_equal"] is None for t, v in blk.items() if t not in (t0, t1))
    assert em["all_identical"] is True and ex.value.code == 0
    blk = em["1179_subset"]
    assert blk["refs"] == ["chrA", "chrB"] and blk["n_records"] == len(_keys(synth.bam))
    assert list(blk["settings"]) == tags
    for s in published():
        r = blk["settings"][setting_tag(s)]
        want = serial_original(synth, s.covlen, s.min_len, s.mism, s.introns)
        assert r["identical"] is True and r["n_original"] == r["n_reimpl"] == len(want) > 0
    assert sorted(p.name for p in (tmp_path / "validation" / "subset_1179").glob("*.json")) == sorted(
        [f"chrA_chrB.{t}.json" for t in tags] + [f"chrA_chrB.{t}.mem.json" for t in tags]
        + [f"chrA_chrB.createTRANS_{m}.mem.json" for m in "ie"])
    assert json.loads(capsys.readouterr().out) == {"complete": False, "missing": {}, "none_control": None,
                                                   "1179_subset": {t: True for t in tags}, "n_compared": 8, "all_identical": True}
    # resume: every setting has its result file, so nothing is subset, built, run or compared again
    for name in ("subset_bam", "compare_subset"):
        monkeypatch.setattr(validation, name, lambda *a, **k: pytest.fail("rerun"))
    calls.clear()
    with pytest.raises(SystemExit) as ex:
        mod.main()
    assert ex.value.code == 0 and calls == []
    assert json.loads((tmp_path / "validation" / "exact_match_m2.json").read_text()) == em
    # a difference in one setting makes the stage fail and is folded into all_identical
    p = tmp_path / "validation" / "subset_1179" / f"chrA_chrB.{tags[0]}.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), "only_reimpl": 1, "identical": False}))
    with pytest.raises(SystemExit) as ex:
        mod.main()
    em = json.loads((tmp_path / "validation" / "exact_match_m2.json").read_text())
    assert ex.value.code == 1 and em["all_identical"] is False and em["n_compared"] == 8
    # a grid count that differs from the original's fails the stage too
    p.write_text(json.dumps({**json.loads(p.read_text()), "only_reimpl": 0, "identical": True}))
    pd.DataFrame({"tag": [t0], "re": ["x"], "removed": [0], "total": [99]}).to_parquet(parts / "chrB.region.parquet", index=False)
    with pytest.raises(SystemExit) as ex:
        mod.main()
    em = json.loads((tmp_path / "validation" / "exact_match_m2.json").read_text())
    assert ex.value.code == 1 and em["1179_subset"]["settings"][t0]["grid_equal"] is False and em["all_identical"] is False


# ---------- engine tie: stage 13 grid against stage 03 flags ----------

TAGS = ["c300_l10_m1_i", "c300_lnone_m0_e"]


def _tie_dirs(lib_dir, truth, reimpl):
    """grid/truth_flags.parquet from truth rows (qname, ref, pos, flag, *tags) and reimpl_flags/<ref>.parquet from
    {ref: rows (qname, flag, ref, pos, cigar, *tags)}."""
    (lib_dir / "grid").mkdir(parents=True, exist_ok=True)
    (lib_dir / "reimpl_flags").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(truth, columns=["qname", "ref", "pos", "flag", *TAGS]).to_parquet(lib_dir / "grid" / "truth_flags.parquet", index=False)
    for ref, rows in reimpl.items():
        pd.DataFrame(rows, columns=["qname", "flag", "ref", "pos", "cigar", *TAGS]).to_parquet(
            lib_dir / "reimpl_flags" / f"{ref}.parquet", index=False)


def test_engine_tie_counts_mismatches_and_missing_rows(tmp_path):
    truth = [("r1", "chr1", 10, 0, True, False), ("r2", "chr1", 20, 16, False, False), ("r3", "chr1", 30, 0, True, True),
             ("r4", "chr2", 5, 0, False, True), ("r5", "chr3", 7, 0, True, True)]
    # reimpl has non-truth reads, a secondary record sharing a truth qname, and no file for chr3; r2 differs in tag 0,
    # r4 in both tags
    reimpl = {"chr1": [("x", 0, "chr1", 1, "5M", True, True), ("r1", 0, "chr1", 10, "5M", True, False),
                       ("r1", 256, "chr1", 99, "5M", False, True), ("r2", 16, "chr1", 20, "5M", True, False),
                       ("r3", 0, "chr1", 30, "5M", True, True)],
              "chr2": [("r4", 0, "chr2", 5, "5M", True, False)]}
    _tie_dirs(tmp_path, truth, reimpl)
    assert engine_tie(tmp_path, TAGS) == {"rows": 5, "tags": TAGS, "mismatches": {TAGS[0]: 2, TAGS[1]: 1}, "missing_rows": 1}
    # only the asked tags are compared
    assert engine_tie(tmp_path, TAGS[1:])["mismatches"] == {TAGS[1]: 1}
    # a row of the same read at another position is not the same record
    reimpl["chr1"][4] = ("r3", 0, "chr1", 31, "5M", True, True)
    _tie_dirs(tmp_path, truth, reimpl)
    assert engine_tie(tmp_path, TAGS)["missing_rows"] == 2
    # identical tables tie
    truth = [("r1", "chr1", 10, 0, True, False)]
    _tie_dirs(tmp_path, truth, {"chr1": [("r1", 0, "chr1", 10, "5M", True, False)]})
    assert engine_tie(tmp_path, TAGS) == {"rows": 1, "tags": TAGS, "mismatches": {t: 0 for t in TAGS}, "missing_rows": 0}


def test_engine_tie_tells_duplicate_keys_apart_and_flags_an_absent_tag_column(tmp_path):
    truth = [("r1", "chr1", 10, 0, True, False), ("r1", "chr1", 10, 0, False, False)]
    _tie_dirs(tmp_path, truth, {"chr1": [("r1", 0, "chr1", 10, "5M", True, False), ("r1", 0, "chr1", 10, "4M1S", False, False)]})
    assert engine_tie(tmp_path, TAGS)["mismatches"] == {t: 0 for t in TAGS}
    got = engine_tie(tmp_path, [*TAGS, "c150_l3_m0_e"])
    assert got["mismatches"]["c150_l3_m0_e"] == 2 and got["missing_rows"] == 0


def test_grid_removed(tmp_path):
    assert grid_removed(tmp_path, "c300_l10_m1_i") is None
    (tmp_path / "parts").mkdir()
    (tmp_path / "summary.json").write_text(json.dumps({"removed": {"c300_l10_m1_i": 7}}))
    assert grid_removed(tmp_path, "c300_l10_m1_i") == 7 and grid_removed(tmp_path, "c180_l11_m2_e") is None
    for ref, a, b in (("chr21", 3, 4), ("chr22", 5, 6), ("chr1", 100, 100)):
        pd.DataFrame({"tag": ["t1", "t1", "t2"], "re": ["x", "y", "x"], "removed": [a, b, 1], "total": [9, 9, 9]}).to_parquet(
            tmp_path / "parts" / f"{ref}.region.parquet", index=False)
    assert grid_removed(tmp_path, "t1", ("chr21", "chr22")) == 18 and grid_removed(tmp_path, "t2", ("chr21", "chr22")) == 2
    assert grid_removed(tmp_path, "t1", ("chr21", "chrX")) is None


def test_reusable(tmp_path):
    d = tmp_path / "1417"
    (d / "original").mkdir(parents=True)
    (d / "reimpl_flags").mkdir()
    settings = [parse_tag(t) for t in TAGS]
    for t in TAGS:
        (d / "original" / f"{t}.bam").touch()
    summ, em = d / "reimpl_flags" / "summary.json", d / "exact_match.json"
    assert reusable(d, settings) is False
    summ.write_text(json.dumps({"settings": TAGS[::-1]}))
    em.write_text(json.dumps({t: {} for t in TAGS}))
    t0 = time.time()

    def stamp(path, dt):
        os.utime(path, (t0 + dt, t0 + dt))

    for p, dt in ((d / "original" / f"{TAGS[0]}.bam", -30), (d / "original" / f"{TAGS[1]}.bam", -20), (summ, -10), (em, 0)):
        stamp(p, dt)
    assert reusable(d, settings) is True
    # a different tag set in either file
    em.write_text(json.dumps({TAGS[0]: {}}))
    stamp(em, 0)
    assert reusable(d, settings) is False
    em.write_text(json.dumps({t: {} for t in TAGS}))
    stamp(em, 0)
    summ.write_text(json.dumps({"settings": TAGS + ["c180_l11_m2_e"]}))
    stamp(summ, -10)
    assert reusable(d, settings) is False
    summ.write_text(json.dumps({"settings": TAGS}))
    # exact_match.json older than the summary, or than any original BAM (also one that is not a discovered setting)
    stamp(summ, 5)
    assert reusable(d, settings) is False
    stamp(summ, -10)
    (d / "original" / "c180_l11_m2_e.bam").touch()
    stamp(d / "original" / "c180_l11_m2_e.bam", 5)
    assert reusable(d, settings) is False
    stamp(d / "original" / "c180_l11_m2_e.bam", -5)
    assert reusable(d, settings) is True


# ---------- stage 20: completeness, engine tie, reuse ----------

def _stage20_run(mod, monkeypatch, out, *argv):
    """Stage 20 with its two reference-subset blocks stubbed (they have their own tests); skip flags go in argv."""
    monkeypatch.setattr(mod, "none_control", lambda *a, **k: {"identical": True})
    monkeypatch.setattr(mod, "subset_validation", lambda *a, **k: {"settings": {}})
    monkeypatch.setattr(sys, "argv", ["20", "--out", str(out), *argv])
    with pytest.raises(SystemExit) as ex:
        mod.main()
    return ex.value.code, json.loads((out / "validation" / "exact_match_m2.json").read_text())


def _stage20_tie(d, tags, removed, mismatch=False):
    """Grid and reimpl outputs for one stand-in library: two truth reads on chr1 that tie (or differ in the first tag)."""
    truth = [("r1", "chr1", 10, 0, True, False), ("r2", "chr1", 20, 0, False, False)]
    reimpl = {"chr1": [("r1", 0, "chr1", 10, "5M", not mismatch, False), ("r2", 0, "chr1", 20, "5M", False, False)]}
    _tie_dirs(d, truth, reimpl)
    (d / "grid" / "summary.json").write_text(json.dumps({"removed": removed}))


def test_stage20_incomplete_when_a_run_is_missing(tmp_path, monkeypatch, capsys):
    mod, d = _stage20(tmp_path, monkeypatch)
    out = tmp_path / "out"
    tags = ["c300_l10_m1_i", "c300_lnone_m0_e"]
    monkeypatch.setattr(mod, "EXPECTED_RUNS", {"1417": tags + ["c375_l3_m2_i", "c180_l11_m2_e"]})
    (d / "original" / "c375_l3_m2_i.bam").touch()
    (d / "original" / "c375_l3_m2_i.time.txt").write_text(KILLED_TIME)
    monkeypatch.setattr(mod, "NUCLEI", ("1417",))
    monkeypatch.setenv("FAKE04", "ok")
    monkeypatch.setenv("FAKE04_RESULT", json.dumps({t: _match(10, 10, 0, 0) for t in tags}))
    monkeypatch.setattr(mod, "engine_tie", lambda lib_dir, tags: {"rows": 1, "tags": tags, "mismatches": {}, "missing_rows": 0})
    code, em = _stage20_run(mod, monkeypatch, tmp_path, "--libs", "1417")
    # every compared run is identical, yet two expected runs are absent: not identical, not complete, exit 1
    assert code == 1 and em["all_identical"] is False and em["complete"] is False
    assert em["missing"] == {"1417": [{"tag": "c375_l3_m2_i", "exit_status": 137, "reason": "killed or failed"},
                                      {"tag": "c180_l11_m2_e", "exit_status": None, "reason": "no output"}]}
    printed = json.loads(capsys.readouterr().out)
    assert printed["complete"] is False and printed["missing"] == {"1417": ["c375_l3_m2_i", "c180_l11_m2_e"]}
    assert printed["all_identical"] is False


def test_stage20_engine_tie_and_complete(tmp_path, monkeypatch, capsys):
    mod, d = _stage20(tmp_path, monkeypatch)
    tags = ["c300_l10_m1_i", "c300_lnone_m0_e"]
    monkeypatch.setattr(mod, "EXPECTED_RUNS", {"1417": tags})
    monkeypatch.setattr(mod, "NUCLEI", ("1417",))
    monkeypatch.setenv("FAKE04", "ok")
    monkeypatch.setenv("FAKE04_RESULT", json.dumps({t: _match(10, 10, 0, 0) for t in tags}))
    # no grid outputs: tie fields are null, complete is false, but nothing differs
    code, em = _stage20_run(mod, monkeypatch, tmp_path, "--libs", "1417")
    assert code == 0 and em["all_identical"] is True and em["complete"] is False and em["engine_tie"] == {"1417": None}
    assert em["missing"] == {"1417": []} and em["reused"] == {"1417": False}
    assert all(em["1417"][t]["n_grid"] is None and em["1417"][t]["grid_equal"] is None for t in tags)
    # grid counts equal n_reimpl and the records tie
    _stage20_tie(d, tags, {tags[0]: 10, tags[1]: 10})
    code, em = _stage20_run(mod, monkeypatch, tmp_path, "--libs", "1417")
    assert code == 0 and em["all_identical"] is True and em["complete"] is True
    assert em["engine_tie"]["1417"] == {"rows": 2, "tags": tags, "mismatches": {t: 0 for t in tags}, "missing_rows": 0}
    assert all(em["1417"][t]["n_grid"] == 10 and em["1417"][t]["grid_equal"] is True for t in tags)
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["complete"] is True
    # a count difference in one tag
    _stage20_tie(d, tags, {tags[0]: 11, tags[1]: 10})
    code, em = _stage20_run(mod, monkeypatch, tmp_path, "--libs", "1417")
    assert code == 1 and em["all_identical"] is False and em["1417"][tags[0]]["grid_equal"] is False
    assert em["1417"][tags[0]]["identical"] is True and em["1417"][tags[1]]["grid_equal"] is True
    # a record mismatch
    _stage20_tie(d, tags, {tags[0]: 10, tags[1]: 10}, mismatch=True)
    code, em = _stage20_run(mod, monkeypatch, tmp_path, "--libs", "1417")
    assert code == 1 and em["all_identical"] is False and em["engine_tie"]["1417"]["mismatches"] == {tags[0]: 1, tags[1]: 0}
    # a missing row
    _stage20_tie(d, tags, {tags[0]: 10, tags[1]: 10})
    (d / "reimpl_flags" / "chr1.parquet").unlink()
    code, em = _stage20_run(mod, monkeypatch, tmp_path, "--libs", "1417")
    assert code == 1 and em["engine_tie"]["1417"]["missing_rows"] == 2
    # skip flags and a library list other than the nuclei libraries never complete
    _stage20_tie(d, tags, {tags[0]: 10, tags[1]: 10})
    monkeypatch.setattr(mod, "NUCLEI", ("1417", "1419"))
    code, em = _stage20_run(mod, monkeypatch, tmp_path, "--libs", "1417")
    assert code == 0 and em["all_identical"] is True and em["complete"] is False
    monkeypatch.setattr(mod, "NUCLEI", ("1417",))
    for skip in ("--skip-none-control", "--skip-1179"):
        code, em = _stage20_run(mod, monkeypatch, tmp_path, "--libs", "1417", skip)
        assert code == 0 and em["all_identical"] is True and em["complete"] is False


def test_stage20_reuse_skips_stages_03_and_04(tmp_path, monkeypatch):
    mod, d = _stage20(tmp_path, monkeypatch)
    tags = ["c300_l10_m1_i", "c300_lnone_m0_e"]
    monkeypatch.setattr(mod, "EXPECTED_RUNS", {"1417": tags})
    monkeypatch.setattr(mod, "NUCLEI", ("1417",))
    monkeypatch.setenv("FAKE04", "ok")
    monkeypatch.setenv("FAKE04_RESULT", json.dumps({t: _match(10, 10, 0, 0) for t in tags}))
    _stage20_tie(d, tags, {tags[0]: 10, tags[1]: 10})
    # without a previous comparison --reuse runs both stages
    code, em = _stage20_run(mod, monkeypatch, tmp_path, "--libs", "1417", "--reuse")
    assert code == 0 and em["reused"] == {"1417": False} and (d / "args03.txt").exists() and (d / "args04.txt").exists()
    # stage 03's summary, then stage 04's result: newer than both, so the next call reuses them
    (d / "reimpl_flags" / "summary.json").write_text(json.dumps({"settings": tags}))
    (d / "exact_match.json").write_text(json.dumps({t: _match(10, 10, 0, 0) for t in tags}))
    t0 = time.time()
    for p, dt in ((d / "original" / f"{tags[0]}.bam", -30), (d / "original" / f"{tags[1]}.bam", -30),
                  (d / "reimpl_flags" / "summary.json", -10), (d / "exact_match.json", 0)):
        os.utime(p, (t0 + dt, t0 + dt))
    (d / "args03.txt").unlink()
    (d / "args04.txt").unlink()
    first = em
    code, em = _stage20_run(mod, monkeypatch, tmp_path, "--libs", "1417", "--reuse")
    assert code == 0 and em["reused"] == {"1417": True} and not (d / "args03.txt").exists() and not (d / "args04.txt").exists()
    assert em["1417"] == first["1417"] and em["complete"] is True
    # without the flag the stages run as before
    code, em = _stage20_run(mod, monkeypatch, tmp_path, "--libs", "1417")
    assert em["reused"] == {"1417": False} and (d / "args03.txt").exists()
