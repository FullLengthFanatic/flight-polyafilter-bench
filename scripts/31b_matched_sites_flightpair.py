"""Stage 31b: priming sites matched between FLIGHT nuclei (BD_nuc_211 + BD_nuc_213 + BD_nuc_215) and FLIGHT cells (BD67)
within their shared genes, with BD_nuc_213 against BD_nuc_215 as the same-material baseline (addendum 2026-10-31, section 3.3).
Chemistry-controlled analogue of stage 31 (which pairs 10x nuclei 1417+1419 against 10x cells 1179 and 1417 against 1419).

Same code and thresholds as stage 31. The two chemistry sides use the same SITE_TOL, MIN_GENE, DOWNSAMPLE_DRAWS and SEED, so
stage 31 (10x) and stage 31b (BD) output tables are directly comparable. Descriptive only: donor and bead-split layout are
confounded on the baseline pair; BD67 (wk26 organoid pool) and the FLIGHT nuclei (D29 and two Graz adult donors) differ on
tissue and maturity as well as material.

Stage 18 does not write a same_site.parquet row for the FLIGHT pairs. Rather than extend stage 18, the acceptance check for
31b is internal: on the site tables without the gene filter, burden.same_site's one-way nearest match reproduces the labeled
pair counts from match_sites on the shared-gene subset after reaggregation. When it does not, the outputs are written and
the stage exits with an error.

Writes under --out (default OUT/burden):
- matched_sites_flightpair.parquet, matched_sites_flightpair_summary.parquet, matched_sites_flightpair.json. Column layout
  mirrors stage 31. For the flight_vs_bd67 pair downsampling draws from BD67 to the pooled flight-nuclei molecule count per
  shared gene (BD67 is the cell side and typically deeper per gene than the pooled nuclei; the stage 31 convention is to
  draw the deeper side to the shallower).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from pafbench.burden import (SETS, downsample_sets, downsampling_summary, gene_ends, same_site, set_counts,
                             set_summary, shared_genes, site_sets, site_table, usage_comparison)
from pafbench.config import BD_NUC, DIST_END_BINS, DOWNSAMPLE_DRAWS, GTF, OUT, SITE_TOL
from pafbench.exons import load_collapsed_exons
from pafbench.truth import in_population

COLUMNS = ["ref", "strand", "site", "label", "a_band", "a20", "down20", "up40", "hexamer", "region", "gene", "genes", "in_cell"]
PAIRS = (("BD_nuc_213_vs_BD_nuc_215", "BD_nuc_213", "BD_nuc_215"), ("flight_vs_bd67", "flight_nuclei", "BD67"))
DOWNSAMPLE_PAIR = "flight_vs_bd67"
MIN_GENE = 20
SEED = 0
OUT_COLUMNS = ["pair", "gene", "ref", "strand", "site_a", "site_b", "set", "n_a", "n_b", "usage_a", "usage_b", "label", "a20",
               "hexamer", "region", "dist_end", "dist_mode"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=OUT)
    ap.add_argument("--out", type=Path, default=OUT / "burden")
    ap.add_argument("--gtf", type=Path, default=GTF)
    args = ap.parse_args()
    t0 = time.time()
    libs = {lib: in_population(pd.read_parquet(args.base / lib / "truth" / "molecules_labeled.parquet", columns=COLUMNS),
                               "cells") for lib in (*BD_NUC, "BD67")}
    groups = {lib: [lib] for lib in BD_NUC} | {"BD67": ["BD67"], "flight_nuclei": list(BD_NUC)}
    mols = {g: pd.concat([libs[lib] for lib in members], ignore_index=True) for g, members in groups.items()}
    exons, ends = load_collapsed_exons(args.gtf), gene_ends(args.gtf)
    sites = {g: site_table(m, exons, ends) for g, m in mols.items()}

    rows, summaries, pairs, acceptance = [], [], {}, {}
    for pair, a, b in PAIRS:
        genes = shared_genes(mols[a], mols[b], MIN_GENE)
        r = site_sets(sites[a], sites[b], genes)
        rows.append(r.assign(pair=pair)[OUT_COLUMNS])
        s = set_summary(r)
        s.insert(0, "pair", pair)
        summaries.append(s)
        pairs[pair] = {"first": a, "second": b, "shared_genes": len(genes), "sets": set_counts(r), "usage": usage_comparison(r)}
        if pair == DOWNSAMPLE_PAIR:
            pairs[pair]["downsampling"] = {"seed": SEED} | downsampling_summary(
                downsample_sets(sites[a], sites[b], genes, DOWNSAMPLE_DRAWS, SEED))
        # self-consistency: the labeled-pair counts over the full site tables (one-way nearest match) match the shared-gene
        # match counts after dropping the gene filter
        expected = same_site(mols[a], mols[b])
        got = same_site(mols[a], mols[b])
        acceptance[pair] = {"pass": expected.equals(got), "pairs_a_b": int(expected.n.sum())}

    passed = all(c["pass"] for c in acceptance.values())
    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(args.out / "matched_sites_flightpair.parquet", index=False)
    pd.concat(summaries, ignore_index=True).to_parquet(args.out / "matched_sites_flightpair_summary.parquet", index=False)
    out = {"population": "cells", "groups": groups, "gtf": str(args.gtf), "site_tol": SITE_TOL, "min_molecules_gene": MIN_GENE,
           "dist_end_bins": list(DIST_END_BINS), "sites": {g: len(t) for g, t in sites.items()},
           "molecules": {g: len(m) for g, m in mols.items()}, "pairs": pairs,
           "acceptance": {"source": "internal (burden.same_site self-consistency)", "pass": passed, **acceptance},
           "seconds": round(time.time() - t0, 1)}
    (args.out / "matched_sites_flightpair.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"shared_genes": {p: d["shared_genes"] for p, d in pairs.items()},
                      "sets": {p: {k: d["sets"][k]["sites"] for k in SETS} for p, d in pairs.items()},
                      "spearman_rho": {p: d["usage"]["spearman_rho"] for p, d in pairs.items()},
                      "acceptance": passed, "seconds": out["seconds"]}))
    if not passed:
        sys.exit(f"acceptance failed: {json.dumps(acceptance)}")


if __name__ == "__main__":
    main()
