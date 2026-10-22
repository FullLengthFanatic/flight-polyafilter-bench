"""Stage 90: render the four supplementary figures as publication-ready PNG + SVG + PDF.

Rebuilds every figure from pafbench parquet outputs, not from the hand-authored SVG in the summary
artifact. The artifact's SVGs depend on CSS defined in the surrounding HTML page and lose their
styling when extracted standalone; this script bypasses that failure mode entirely.

Figures:
  1. Internal-priming burden by library (stacked bars, 4 libraries, 3 labels).
  2. Internal-priming share by compartment (grouped bars, 4 libraries x region).
  4. UMI loss by library and setting (grouped bars, 4 libraries x 3 settings).

Numbering matches the m3_summary artifact (figures 3 and 6 are HTML tables; figure 5 was dropped
because its intron-share source in per_cell.parquet is empty for this benchmark run).

Dependencies: pandas, matplotlib. Both are in the pafbench environment.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path("/data/element_runs/polyafilter_bench")
OUT = BASE / "report" / "figures"
LIBS = ("1417", "1419", "1179", "BD67")
LIB_LABELS = {"1417": "1417\nnuclei", "1419": "1419\nnuclei", "1179": "1179\n10x cells", "BD67": "BD67\nFLIGHT-seq"}
TAGS = ("c300_l10_m1_e", "c250_l7_m1_e", "c200_l10_m2_i")
TAG_LABELS = {"c300_l10_m1_e": "paper default", "c250_l7_m1_e": "mode-e", "c200_l10_m2_i": "mode-i"}
TAG_COLORS = {"c300_l10_m1_e": "#9ecae1", "c250_l7_m1_e": "#4292c6", "c200_l10_m2_i": "#08519c"}
LABEL_COLORS = {"templated": "#d62728", "nontemplated": "#2ca02c", "ambiguous": "#999999"}
LABEL_ORDER = ("templated", "nontemplated", "ambiguous")
LABEL_PRETTY = {"templated": "templated (internal priming)", "nontemplated": "non-templated (real poly-A)",
                "ambiguous": "ambiguous (5-7 A's downstream)"}
REGIONS = ("3UTR", "CDS", "5UTR", "ncexon", "intron", "intergenic", "antisense")
REGION_LABELS = {"3UTR": "3' UTR", "CDS": "CDS", "5UTR": "5' UTR", "ncexon": "ncRNA exon",
                 "intron": "intron", "intergenic": "intergenic", "antisense": "antisense"}


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_1(base, out):
    rows = []
    for lib in LIBS:
        d = pd.read_parquet(base / lib / "truth" / "molecules_labeled.parquet", columns=["label"])
        c = d.label.value_counts(normalize=True)
        rows.append({"library": lib} | {k: c.get(k, 0) for k in LABEL_ORDER})
    df = pd.DataFrame(rows).set_index("library")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = [0] * len(LIBS)
    for lab in LABEL_ORDER:
        vals = [df.loc[l, lab] for l in LIBS]
        ax.bar(range(len(LIBS)), vals, bottom=bottom, color=LABEL_COLORS[lab], label=LABEL_PRETTY[lab],
               edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            if v >= 0.08:
                ax.text(i, bottom[i] + v / 2, f"{v * 100:.1f}%", ha="center", va="center", color="white",
                        fontsize=9, fontweight="bold")
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(len(LIBS)))
    ax.set_xticklabels([LIB_LABELS[l] for l in LIBS])
    ax.set_ylabel("Share of labeled molecules")
    ax.set_ylim(0, 1)
    ax.set_title("Figure 1. Internal-priming burden by library")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9)
    _despine(ax)
    fig.tight_layout()
    save(fig, out, "figure_1_burden_by_library")
    plt.close(fig)


def figure_2(base, out):
    rows = []
    for lib in LIBS:
        d = pd.read_parquet(base / lib / "truth" / "molecules_labeled.parquet", columns=["label", "region"])
        for region in REGIONS:
            sub = d[d.region == region]
            if len(sub) < 100:
                continue
            tpl = (sub.label == "templated").mean()
            rows.append({"library": lib, "region": region, "templated": tpl, "n": len(sub)})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.18
    for j, lib in enumerate(LIBS):
        sub = df[df.library == lib].set_index("region")
        vals = [sub.loc[r, "templated"] if r in sub.index else 0 for r in REGIONS]
        x = [i + (j - 1.5) * width for i in range(len(REGIONS))]
        colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
        ax.bar(x, vals, width, label=LIB_LABELS[lib].replace("\n", " "), color=colors[j])
    ax.set_xticks(range(len(REGIONS)))
    ax.set_xticklabels([REGION_LABELS[r] for r in REGIONS])
    ax.set_ylabel("Share templated (internal priming)")
    ax.set_ylim(0, 1)
    ax.set_title("Figure 2. Internal-priming share by compartment")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=9)
    _despine(ax)
    fig.tight_layout()
    save(fig, out, "figure_2_burden_by_compartment")
    plt.close(fig)


def _umi_cost(base, lib, tag):
    d = pd.read_parquet(base / lib / "impact" / "per_cell.parquet", columns=["tag", "umis0", "lost"])
    sub = d[d.tag == tag]
    return sub.lost.sum() / sub.umis0.sum() if len(sub) else None


def figure_4(base, out):
    rows = [{"library": lib, "tag": tag, "cost": _umi_cost(base, lib, tag)}
            for lib in LIBS for tag in TAGS]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    width = 0.26
    for j, tag in enumerate(TAGS):
        vals = [df[(df.library == lib) & (df.tag == tag)].cost.iloc[0] for lib in LIBS]
        x = [i + (j - 1) * width for i in range(len(LIBS))]
        ax.bar(x, vals, width, label=TAG_LABELS[tag], color=TAG_COLORS[tag])
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.01, f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(LIBS)))
    ax.set_xticklabels([LIB_LABELS[l] for l in LIBS])
    ax.set_ylabel("Fraction of UMIs removed")
    ax.set_ylim(0, max(df.cost) * 1.15)
    ax.set_title("Figure 4. UMI loss by library and setting")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    _despine(ax)
    fig.tight_layout()
    save(fig, out, "figure_4_umi_loss_by_setting")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    for fn in (figure_1, figure_2, figure_4):
        fn(args.base, args.out)
        print(f"wrote {fn.__name__}")


if __name__ == "__main__":
    main()
