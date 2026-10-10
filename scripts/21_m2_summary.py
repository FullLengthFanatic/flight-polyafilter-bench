"""Stage 21: collect milestone 2 numbers into docs/m2_results.md (sections from pafbench.m2report)."""
import argparse
import json
from pathlib import Path

from pafbench.config import OUT, REPO
from pafbench.m2report import SECTIONS, render_counted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=OUT)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "m2_results.md")
    args = ap.parse_args()
    text, n_not_run = render_counted(args.base)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(json.dumps({"out": str(args.out), "sections": len(SECTIONS), "not_run": n_not_run}))


if __name__ == "__main__":
    main()
