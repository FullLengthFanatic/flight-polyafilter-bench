"""Stage 18: templated share by library and group with gene-bootstrap CIs, same-site and same-gene comparisons, selection bias,
PolyASite 3.0 proximity of the labeled sites (reported only, not a label check: the atlas is built from 10x scRNA-seq data,
https://polyasite.unibas.ch/atlas_sc; the v3.0 documentation describes no filter on A content downstream of a site; the A content downstream
of the atlas sites is counted by scripts/22_polyasite_a_content.py).

--population (default cells, config.SEL_POPULATION) picks the barcodes of all tables except templated_share_population.parquet,
which holds every population. summary.json records it in the key "population". Other files: same_site.parquet,
same_gene.parquet, selection_bias.parquet (labeled_share, records_share and, when stage 13 wrote total_nondup,
records_nondup_share) and summary.json. grid/region_counts.parquet counts the primary records of
every barcode, so selection_bias.parquet compares them with the labeled molecules of all barcodes whatever --population is
(summary.json key "selection_bias_population": "all"). The comparison is not repeated within called cells because stage 13
does not split the records by barcode (left for milestone 3, which builds per-cell counts).

In templated_share.parquet, lo and hi bound share_all, except on the a_band rows: there they bound frac (the band's share
of the molecules), because the labels are defined from the A count and fix share_all within a band.

polyasite.parquet (group, label, a_band, bin, n): molecules per bin of |d|, the distance from the site to the nearest atlas
site on the same ref and strand: "0-10", "11-50", "51-200", ">200", "no atlas site on this ref and strand" (contigs the
atlas lacks, such as chrM and unplaced contigs). Per group the label rows (a_band "all"), then label "null_shift1kb"
(a_band "all": every molecule of the group with its site moved 1,000 nt upstream in transcript sense on the same ref and
strand, binned the same way; the share expected by chance at this atlas density), then the templated molecules per
templated A band. summary.json polyasite.within_10nt[group]: the "0-10" share per label, null_shift1kb beside them, and
per templated A band.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pafbench.burden import near_shares, proximity_table, same_gene, same_site, selection_bias, share_ci, share_table
from pafbench.config import CELLS, NUCLEI, OUT, POLYASITE, POPULATIONS, SEL_POPULATION
from pafbench.polyasite import NULL_SHIFT, load_polyasite, site_distances
from pafbench.truth import in_population

KINDS = (None, "region", "a_band", "has_ts")
COLUMNS = ["mol", "ref", "strand", "site", "label", "a_band", "region", "gene", "has_ts", "re"]


def _num(x):
    return None if pd.isna(x) else float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=OUT)
    ap.add_argument("--out", type=Path, default=OUT / "burden")
    ap.add_argument("--atlas", type=Path, default=POLYASITE)
    ap.add_argument("--population", choices=POPULATIONS, default=SEL_POPULATION,
                    help="which barcodes to use: all, cells (in_cell=True), or noncells (in_cell=False)")
    args = ap.parse_args()
    t0 = time.time()
    mols, by_population = {}, {}
    for lib in NUCLEI + CELLS:
        p = args.base / lib / "truth" / "molecules_labeled.parquet"
        if p.exists():
            m = pd.read_parquet(p, columns=COLUMNS + (["in_cell"] if "in_cell" in pq.read_schema(p).names else []))
            by_population[lib] = {pop: in_population(m, pop) for pop in POPULATIONS} if "in_cell" in m else {"all": m}
            try:
                m = in_population(m, args.population)
            except ValueError as e:
                raise SystemExit(f"{lib}: {e}")
            mols[lib] = m
        else:
            print(f"{lib}: skipped, missing {p}")
    if not mols:
        raise SystemExit("no library has truth files")
    args.out.mkdir(parents=True, exist_ok=True)
    groups, members = dict(mols), {lib: [lib] for lib in mols}
    for name, libs in (("nuclei", [lib for lib in NUCLEI if lib in mols]), ("cells", [lib for lib in CELLS if lib in mols])):
        if libs:
            groups[name] = pd.concat([mols[lib] for lib in libs], ignore_index=True)
            members[name] = libs

    tables = []
    for g, mol in groups.items():
        for by in KINDS:
            t = share_table(mol, by).merge(share_ci(mol, by, of="frac" if by == "a_band" else "share_all"), on="stratum")
            t.insert(0, "group", g)
            t.insert(1, "stratum_kind", by or "all")
            tables.append(t)
    share = pd.concat(tables, ignore_index=True)
    share.to_parquet(args.out / "templated_share.parquet", index=False)

    pop_share = []
    for lib, pops in by_population.items():
        for pop, mol in pops.items():
            for by in (None, "region"):
                t = share_table(mol, by).drop(columns=["frac", "share_resolved"])
                t.insert(0, "lib", lib)
                t.insert(1, "population", pop)
                t.insert(2, "stratum_kind", by or "all")
                pop_share.append(t)
    pd.concat(pop_share, ignore_index=True).to_parquet(args.out / "templated_share_population.parquet", index=False)

    pairs = []
    if "1417" in mols and "1419" in mols:
        pairs.append(("1417_vs_1419", mols["1417"], mols["1419"]))
    if "nuclei" in groups and "cells" in groups:
        pairs.append(("nuclei_vs_cells", groups["nuclei"], groups["cells"]))
    sites, genes = [], []
    for name, a, b in pairs:
        s = same_site(a, b)
        s.insert(0, "pair", name)
        sites.append(s)
        genes.append({"pair": name} | same_gene(a, b))
    site_cols = ["pair", "label_a", "label_b", "n"]
    (pd.concat(sites, ignore_index=True) if sites else pd.DataFrame(columns=site_cols)).to_parquet(
        args.out / "same_site.parquet", index=False)
    pd.DataFrame(genes, columns=["pair", "n_genes", "spearman_rho", "p_value"]).to_parquet(args.out / "same_gene.parquet", index=False)

    bias = []
    for lib in mols:
        p = args.base / lib / "grid" / "region_counts.parquet"
        if not p.exists():
            print(f"{lib}: selection bias skipped, missing {p}")
            continue
        b = selection_bias(by_population[lib]["all"], pd.read_parquet(p))
        b.insert(0, "lib", lib)
        bias.append(b)
    if bias:
        pd.concat(bias, ignore_index=True).to_parquet(args.out / "selection_bias.parquet", index=False)

    t1 = time.time()
    atlas = load_polyasite(args.atlas)
    atlas_seconds = time.time() - t1
    prox = []
    for g, mol in groups.items():
        t = proximity_table(mol, site_distances(mol, atlas), site_distances(mol, atlas, upstream=NULL_SHIFT))
        t.insert(0, "group", g)
        prox.append(t)
    prox = pd.concat(prox, ignore_index=True)
    prox.to_parquet(args.out / "polyasite.parquet", index=False)

    def entry(r):
        return {"share_all": _num(r.share_all), "lo": _num(r.lo), "hi": _num(r.hi), "share_resolved": _num(r.share_resolved),
                "total": int(r.total)}

    headline = {}
    for g in groups:
        d = share[share.group == g]
        headline[g] = {"all": entry(d[d.stratum_kind == "all"].iloc[0])}
        for kind in ("region", "has_ts"):
            headline[g][kind] = {r.stratum: entry(r) for r in d[d.stratum_kind == kind].itertuples()}
    site_summary = {}
    for s in sites:
        n = int(s.n.sum())
        matched = s[s.label_b != "none"]
        site_summary[s.pair.iloc[0]] = {"sites_a": n, "matched": int(matched.n.sum()),
                                        "same_label": int(matched[matched.label_a == matched.label_b].n.sum())}
    out = {"population": args.population, "selection_bias_population": "all", "libraries": list(mols), "groups": members, "share": headline,
           "same_gene": {d["pair"]: {"n_genes": d["n_genes"], "spearman_rho": _num(d["spearman_rho"]),
                                     "p_value": _num(d["p_value"])} for d in genes},
           "same_site": site_summary,
           "polyasite": {"atlas": str(args.atlas), "atlas_sites": int(sum(len(v) for v in atlas.values())),
                         "within_10nt": {g: near_shares(prox[prox.group == g]) for g in groups}}}
    (args.out / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"groups": members, "share_all": {g: headline[g]["all"]["share_all"] for g in groups},
                      "pairs": [p[0] for p in pairs], "atlas_seconds": round(atlas_seconds, 1),
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
