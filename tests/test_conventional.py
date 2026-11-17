import json
import os
import subprocess
import sys

import pandas as pd

from pafbench.config import REPO
from pafbench.conventional import conventional_flag, conventional_share


def _mol(down20):
    return pd.DataFrame({"down20": down20, "a20": [s.count("A") for s in down20]})


def test_six_consecutive_anywhere_in_the_window():
    assert conventional_flag(_mol(["CCCCAAAAAACCCCCCCCCC"])).tolist() == [True]
    assert conventional_flag(_mol(["AAAAACAAAAACCCCCCCCC"])).tolist() == [False]      # 10 A, longest run 5


def test_twelve_of_twenty_without_a_run():
    assert conventional_flag(_mol(["AACAACAACAACAACAACAA"])).tolist() == [True]       # 14 A, longest run 2
    assert conventional_flag(_mol(["AACAACAACAACAACCCCCC"])).tolist() == [False]      # 10 A


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
