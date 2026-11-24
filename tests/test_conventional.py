import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from pafbench.config import REPO
from pafbench.conventional import conventional_flag, conventional_share
from tests.test_burden import _lib_mol, _stage18


def _mol(down20):
    return pd.DataFrame({"down20": down20, "a20": [s.count("A") for s in down20]})


def test_six_consecutive_anywhere_in_the_window():
    assert conventional_flag(_mol(["CCCCAAAAAACCCCCCCCCC"])).tolist() == [True]
    assert conventional_flag(_mol(["AAAAACAAAAACCCCCCCCC"])).tolist() == [False]      # 10 A, longest run 5


def test_twelve_of_twenty_without_a_run():
    assert conventional_flag(_mol(["AACAACAACAACAACAACAA"])).tolist() == [True]       # 14 A, longest run 2
    assert conventional_flag(_mol(["AACAACAACAACAACCCCCC"])).tolist() == [False]      # 10 A
    assert conventional_flag(_mol(["AAAAACAAAAACAACCCCCC"])).tolist() == [True]       # 12 A, longest run 5
    assert conventional_flag(_mol(["AAAAACAAAAACACCCCCCC"])).tolist() == [False]      # 11 A, longest run 5


def test_share_uses_all_molecules_as_denominator():
    m = _mol(["AAAAAAAACCCCCCCCCCCC", "CCCCCCCCCCCCCCCCCCCC", "AAAAACCCCCCCCCCCCCCC", "ACACACACACACCCCCCCCC"])
    m["label"] = ["templated", "nontemplated", "ambiguous", "ambiguous"]
    assert conventional_share(m) == {"n": 4, "n_conventional": 1, "share": 0.25, "share_all_8_4": 0.25}


def test_stage29b_counts_uniquely_mapped_molecules_of_called_cells(tmp_path):
    """The four molecules above, plus a templated A run with mapq_min 3 and one outside called cells: neither enters."""
    base, out = tmp_path / "base", tmp_path / "burden"
    m = _mol(["AAAAAAAACCCCCCCCCCCC", "CCCCCCCCCCCCCCCCCCCC", "AAAAACCCCCCCCCCCCCCC", "ACACACACACACCCCCCCCC",
              "AAAAAAAACCCCCCCCCCCC", "AAAAAAAACCCCCCCCCCCC"])
    m["label"] = ["templated", "nontemplated", "ambiguous", "ambiguous", "templated", "templated"]
    m["mapq_min"] = [255, 255, 255, 255, 3, 255]
    m["in_cell"] = [True, True, True, True, True, False]
    (base / "1417" / "truth").mkdir(parents=True)
    m.to_parquet(base / "1417" / "truth" / "molecules_labeled.parquet")
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "29b_conventional_rule.py"), "--base", str(base), "--out", str(out)],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(REPO)), cwd=REPO)
    assert r.returncode == 0, r.stderr
    want = {"n": 4, "n_conventional": 1, "share": 0.25, "share_all_8_4": 0.25}
    assert pd.read_parquet(out / "conventional_rule.parquet").to_dict("records") == [{"lib": "1417"} | want]
    assert json.loads((out / "conventional_rule.json").read_text())["libraries"] == {"1417": want}


def test_stage29b_selects_the_molecules_of_stage18(tmp_path):
    """Section 22 prints share_all_8_4 as the share_all of the burden table. Stages 18 and 29b select their molecules with
    separate code, so on one base with multi-mapped and non-cell molecules, n and share_all_8_4 of every library must be
    the total and share_all of its "all" row in templated_share.parquet."""
    base = tmp_path / "base"
    for lib, shift in (("1417", 0), ("1419", 0), ("1179", 3)):
        m = _lib_mol(shift)
        a = np.select([m.label == "templated", m.label == "ambiguous"], [8, 5], 0)
        m["down20"] = ["A" * k + "C" * (20 - k) for k in a]
        m["a20"] = a
        m["in_cell"] = m.mol % 3 != 0
        m["mapq_min"] = np.where(m.mol % 7 == 0, 3, 255)
        (base / lib / "truth").mkdir(parents=True)
        m.to_parquet(base / lib / "truth" / "molecules_labeled.parquet")
    r = _stage18(base, tmp_path / "burden18", tmp_path)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "29b_conventional_rule.py"), "--base", str(base),
                        "--out", str(tmp_path / "burden29b")], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(REPO)), cwd=REPO)
    assert r.returncode == 0, r.stderr
    t = pd.read_parquet(tmp_path / "burden18" / "templated_share.parquet")
    t = t[t.stratum_kind == "all"].set_index("group")
    c = pd.read_parquet(tmp_path / "burden29b" / "conventional_rule.parquet").set_index("lib")
    assert list(c.index) == ["1417", "1419", "1179"] and (c.n < 60).all()
    assert c.n.tolist() == t.loc[c.index, "total"].tolist()
    assert c.share_all_8_4.tolist() == t.loc[c.index, "share_all"].tolist()
