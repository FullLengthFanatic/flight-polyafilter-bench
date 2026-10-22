"""Stage 90: export the SVG figures of the summary artifact as standalone PNGs.

Reads scratchpad/m3_summary.html (or an explicit --html), pulls every <figure><svg>, extracts the Figure N from the
<figcaption>, and writes outputs/figures/figure_<N>_<slug>.png at 2x the viewBox size for print clarity. Figures 3 and
6 of the artifact are HTML tables, not SVG, and are not exported; they live in m3_report.html instead.

Dependencies: beautifulsoup4, cairosvg. Install once per environment:
    uv pip install beautifulsoup4 cairosvg
or run through uv:
    uv run --with beautifulsoup4,cairosvg python scripts/90_export_figures.py --html <path> --out <dir>
"""
import argparse
import re
from pathlib import Path

import cairosvg
from bs4 import BeautifulSoup

HTML = Path("/tmp/claude-1001/-data-element-runs/b3824f3c-a825-42b4-8a9f-c3b00896d118/scratchpad/m3_summary.html")
OUT = Path("/data/element_runs/polyafilter_bench/report/figures")
SCALE = 2
SVG_HEADER = '<?xml version="1.0" encoding="UTF-8"?>\n'


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def figure_number(cap):
    m = re.search(r"Figure\s*(\d+)", cap or "")
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", type=Path, default=HTML)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    soup = BeautifulSoup(args.html.read_text(), "html.parser")
    exported = []
    for fig in soup.find_all("figure"):
        svg = fig.find("svg")
        if svg is None:
            continue
        n = figure_number(fig.figcaption.get_text() if fig.figcaption else "")
        name = slug(svg.get("aria-label") or f"figure_{n or len(exported) + 1}")
        if svg.get("xmlns") is None:
            svg["xmlns"] = "http://www.w3.org/2000/svg"
        vb = svg.get("viewBox", "0 0 640 320").split()
        w, h = int(float(vb[2])), int(float(vb[3]))
        svg["width"], svg["height"] = str(w), str(h)
        path = args.out / f"figure_{n:d}_{name}.png" if n else args.out / f"{name}.png"
        cairosvg.svg2png(bytestring=(SVG_HEADER + str(svg)).encode("utf-8"), write_to=str(path),
                         output_width=w * SCALE, output_height=h * SCALE)
        exported.append({"figure": n, "name": name, "path": str(path), "size": f"{w * SCALE}x{h * SCALE}"})
    for e in exported:
        print(e)
    print(f"{len(exported)} figures exported to {args.out}")


if __name__ == "__main__":
    main()
