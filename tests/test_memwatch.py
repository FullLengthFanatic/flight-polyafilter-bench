import json
import subprocess
import sys

from pafbench.memwatch import run


def test_peak_counts_children_and_exit_status():
    child = "import time; x = bytearray(300 * 1024 * 1024); time.sleep(2)"
    res = run(["bash", "-c", f"{sys.executable} -c '{child}' & wait; exit 3"], interval=0.2)
    assert res["exit_status"] == 3
    assert res["peak_tree_rss_gib"] > 0.25
    assert res["elapsed_s"] >= 2


def test_cli_writes_json(tmp_path):
    out = tmp_path / "m.json"
    r = subprocess.run([sys.executable, "-m", "pafbench.memwatch", str(out), "--", "true"])
    assert r.returncode == 0 and json.loads(out.read_text())["exit_status"] == 0
