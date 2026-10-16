"""Stage 34: write the milestone 3 report page (OUT/report/m3_report.html and numbers.json)."""
import argparse
import json
import time
from pathlib import Path

from pafbench.config import OUT, REPORT_DIR
from pafbench.m3report import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=OUT)
    ap.add_argument("--out", type=Path, default=REPORT_DIR)
    ap.add_argument("--memwatch", action="store_true", help="placeholder; the pipeline wraps this in memwatch itself")
    args = ap.parse_args()
    t0 = time.time()
    paths = build(args.base, args.out)
    text = paths["html"].read_text()
    n_sections = text.count('class="s-m3"')
    n_not_run = text.count("not run (missing")
    print(json.dumps({
        "html": str(paths["html"]), "html_bytes": paths["html"].stat().st_size,
        "numbers": str(paths["numbers"]), "numbers_bytes": paths["numbers"].stat().st_size,
        "sections": n_sections, "not_run": n_not_run, "seconds": round(time.time() - t0, 3),
    }))


if __name__ == "__main__":
    main()
