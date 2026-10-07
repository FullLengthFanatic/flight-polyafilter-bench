"""Stage 07: hexamer rate by downstream A count; fixed label cut-offs."""
import argparse
import json

import pandas as pd

from pafbench.config import OUT
from pafbench.labels import choose_cutoffs, hexamer_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    args = ap.parse_args()
    base = OUT / args.library
    mol = pd.read_parquet(base / "molecules.parquet")
    curve = hexamer_curve(mol)
    curve.to_csv(base / "hexamer_curve.tsv", sep="\t", index=False)
    cut = choose_cutoffs(curve)
    (base / "label_cutoffs.json").write_text(json.dumps(cut, indent=1))
    print(json.dumps(cut))


if __name__ == "__main__":
    main()
