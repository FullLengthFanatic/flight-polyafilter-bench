"""Run a command and record its wall time and the peak RSS summed over its whole process tree.

usage: python -m pafbench.memwatch OUT.json -- CMD [ARG ...]
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _proc_table():
    """{pid: (ppid, rss_kib)} for every process readable in /proc."""
    out = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/status") as fh:
                ppid, rss = None, 0
                for line in fh:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                    elif line.startswith("VmRSS:"):
                        rss = int(line.split()[1])
        except (FileNotFoundError, ProcessLookupError):
            continue
        out[int(name)] = (ppid, rss)
    return out


def tree_rss_bytes(pid):
    table = _proc_table()
    kids = {}
    for p, (pp, _) in table.items():
        kids.setdefault(pp, []).append(p)
    total, stack = 0, [pid]
    while stack:
        p = stack.pop()
        total += table.get(p, (None, 0))[1]
        stack.extend(kids.get(p, []))
    return total * 1024


def run(cmd, interval=2.0):
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd)
    peak = 0
    while True:
        peak = max(peak, tree_rss_bytes(proc.pid))
        try:
            proc.wait(timeout=interval)
            break
        except subprocess.TimeoutExpired:
            pass
    return {"cmd": cmd, "exit_status": proc.returncode, "elapsed_s": time.monotonic() - t0,
            "peak_tree_rss_gib": peak / 1024 ** 3, "interval_s": interval}


def main(argv):
    if len(argv) < 3 or argv[1] != "--":
        raise SystemExit(__doc__)
    res = run(argv[2:])
    Path(argv[0]).write_text(json.dumps(res, indent=1))
    sys.exit(res["exit_status"])


if __name__ == "__main__":
    main(sys.argv[1:])
