"""Stage 92: build the manuscript's main-text figure (two panels, publication-ready PNG + SVG).

Panel A: Templated / non-templated / ambiguous burden across four libraries, read from
<lib>/truth/molecules_labeled.parquet.
Panel B: Clustering ARI across the four libraries at unfiltered, mode-e, mode-i, and paper default,
median across seeds x resolutions, read from <lib>/impact/clustering.parquet.

The script does not use the artifact's SVG; it rebuilds from the parquet files so the manuscript
figure traces to the pipeline, not to a hand-authored page.

Dependencies: pandas, matplotlib. In the pafbench environment these are already in.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path("/data/element_runs/polyafilter_bench")
OUT = BASE / "report" / "figures"
LIBS = ("1417", "1419", "1179", "BD67")
LIB_LABELS = {"1417": "1417\nnuclei", "1419": "1419\nnuclei", "1179": "1179\ncells", "BD67": "BD67\nFLIGHT-seq cells"}
TAGS = ("unfiltered", "c300_l10_m1_e", "c250_l7_m1_e", "c200_l10_m2_i")
TAG_LABELS = {"unfiltered": "unfiltered", "c300_l10_m1_e": "paper default", "c250_l7_m1_e": "mode-e", "c200_l10_m2_i": "mode-i"}
TAG_COLORS = {"unfiltered": "#bdbdbd", "c300_l10_m1_e": "#9ecae1", "c250_l7_m1_e": "#4292c6", "c200_l10_m2_i": "#08519c"}
LABEL_COLORS = {"templated": "#d62728", "nontemplated": "#2ca02c", "ambiguous": "#999999"}


def load_burden(base):
    rows = []
    for lib in LIBS:
        d = pd.read_parquet(base / lib / "truth" / "molecules_labeled.parquet", columns=["label"])
        c = d.label.value_counts(normalize=True)
        rows.append({"library": lib, "templated": c.get("templated", 0), "nontemplated": c.get("nontemplated", 0),
                     "ambiguous": c.get("ambiguous", 0), "n": len(d)})
    return pd.DataFrame(rows)


def load_ari(base):
    rows = []
    for lib in LIBS:
        d = pd.read_parquet(base / lib / "impact" / "clustering.parquet", columns=["tag", "ari"])
        for tag in TAGS:
            sub = d[d.tag == tag]
            if len(sub):
                rows.append({"library": lib, "tag": tag, "ari_median": sub.ari.median(), "n": len(sub)})
    return pd.DataFrame(rows)


def plot(base, out):
    burden, ari = load_burden(base), load_ari(base)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.3]})

    ax = axes[0]
    x = range(len(LIBS))
    tpl = [burden.set_index("library").loc[l, "templated"] for l in LIBS]
    ntpl = [burden.set_index("library").loc[l, "nontemplated"] for l in LIBS]
    amb = [burden.set_index("library").loc[l, "ambiguous"] for l in LIBS]
    ax.bar(x, tpl, color=LABEL_COLORS["templated"], label="templated (internal priming)")
    ax.bar(x, ntpl, bottom=tpl, color=LABEL_COLORS["nontemplated"], label="non-templated (real poly-A)")
    ax.bar(x, amb, bottom=[t + n for t, n in zip(tpl, ntpl)], color=LABEL_COLORS["ambiguous"], label="ambiguous")
    ax.set_xticks(x)
    ax.set_xticklabels([LIB_LABELS[l] for l in LIBS])
    ax.set_ylabel("Share of labeled molecules")
    ax.set_ylim(0, 1)
    ax.set_title("A. Internal-priming burden")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    for i, (t, _lib) in enumerate(zip(tpl, LIBS)):
        ax.text(i, t / 2, f"{t * 100:.1f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")

    ax = axes[1]
    width = 0.2
    for j, tag in enumerate(TAGS):
        vals = [ari[(ari.library == l) & (ari.tag == tag)].ari_median.iloc[0] for l in LIBS]
        ax.bar([i + (j - 1.5) * width for i in range(len(LIBS))], vals, width,
               label=TAG_LABELS[tag], color=TAG_COLORS[tag])
    ax.set_xticks(range(len(LIBS)))
    ax.set_xticklabels([LIB_LABELS[l] for l in LIBS])
    ax.set_ylabel("Clustering ARI (median)")
    ax.set_ylim(0, 1)
    ax.set_title("B. Clustering reproducibility by filter setting")
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"main_text_figure.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote main_text_figure.{{png,svg,pdf}} to {out}")
    return burden, ari


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    plot(args.base, args.out)


if __name__ == "__main__":
    main()
