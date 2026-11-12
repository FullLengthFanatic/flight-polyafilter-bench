import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pafbench.config import REPO

# templated molecules among the 40 uniquely mapped molecules of called cells of each library
TEMPLATED = {"1417": 30, "1419": 28, "1179": 20, "BD67": 10, "BD_nuc_211": 16, "BD_nuc_213": 14, "BD_nuc_215": 15}


def _mol(n_templated):
    """40 uniquely mapped molecules of called cells over 8 genes, n_templated of them templated; 6 templated molecules of
    called cells with mapq_min 0, 1 or 3; 5 uniquely mapped templated molecules outside called cells."""
    n = 40 + 6 + 5
    i = np.arange(n)
    return pd.DataFrame({"gene": [f"g{x % 8}" for x in i], "ref": "chr1", "strand": "+", "site": 1000 * i,
                         "label": np.where((i < n_templated) | (i >= 40), "templated", "nontemplated"),
                         "in_cell": i < 46, "mapq_min": np.where((i >= 40) & (i < 46), np.tile([0, 1, 3], n)[:n], 255)})


def test_stage29a_counts_uniquely_mapped_molecules_of_called_cells(tmp_path):
    base, out = tmp_path / "base", tmp_path / "burden"
    for lib, k in TEMPLATED.items():
        (base / lib / "truth").mkdir(parents=True)
        _mol(k).to_parquet(base / lib / "truth" / "molecules_labeled.parquet")
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "29a_interaction.py"), "--base", str(base), "--out", str(out),
                        "--n-draws", "20"], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(REPO)), cwd=REPO)
    assert r.returncode == 0, r.stderr
    libs = json.loads((out / "interaction.json").read_text())["libraries"]
    assert {lib: d["n_labeled"] for lib, d in libs.items()} == dict.fromkeys(TEMPLATED, 40)
    assert {lib: d["share_point"] for lib, d in libs.items()} == pytest.approx({lib: k / 40 for lib, k in TEMPLATED.items()})
