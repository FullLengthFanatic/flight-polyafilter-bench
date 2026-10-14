"""Stage 31: priming sites matched between nuclei (1417 + 1419) and cells (1179) within their shared genes, with 1417 against
1419 as the same-material baseline (design of 2026-10-12, section 3.9). Descriptive only: tissue, maturity, cell mix and
nucleus against cell are confounded.

Site tables (burden.site_table), one per group (1417, 1419, nuclei, cells), from the labeled molecules of called cells
(<lib>/truth/molecules_labeled.parquet, in_cell): one row per site (ref, strand, site) with its molecules, label, A band,
a20, down20, up40, hexamer, region and gene, which take one value per site in these tables (our observation, 2026-10-14:
no site of 1417, 1419 or 1179 has two values of any of them), and dist_end, dist_mode (burden.end_distances): the signed
distance to the nearest annotated 3' end of the site's gene, > 0 when that end lies downstream in transcript sense, equal
|d| the positive one. The ends are the 3' ends of every transcript line of the gene in genes.gtf (GTF end on plus, start - 1
on minus: the priming_site convention, where the last cDNA base is site - 1 on plus and site on minus). The collapsed exons
of a gene are the union of the exon lines of all its transcripts, overlapping or abutting exons merged
(exons.load_collapsed_exons). A site is exonic, dist_mode "exon", when its last cDNA base lies in a collapsed exon of its
own gene; the distance then counts collapsed-exon bases between the site and the end. Otherwise dist_mode is "genome" and
the distance counts genomic bases. Sites without a gene: NaN and "". region is the strand-wide annotation of milestone 2
(any transcript on the strand), so a site can be in a CDS of another gene and intronic for its own.

Shared genes: at least MIN_GENE labeled molecules in both groups (the threshold of burden.same_gene). Within them each pair
of sites that are each other's nearest on the same ref and strand within SITE_TOL nt is shared (burden.match_sites); the
other sites are first_only or second_only. Our observation on 2026-10-14: no shared pair has sites of two genes (0 of
18,209 nuclei against cells, 0 of 4,076 for 1417 against 1419).

Writes under --out (default OUT/burden):
- matched_sites.parquet: one row per shared pair or unmatched site of a shared gene, both pairs (pair 1417_vs_1419 and
  nuclei_vs_cells, first group a, second b): pair, gene, ref, strand, site_a, site_b (<NA> on the side without a site), set
  (shared, first_only, second_only), n_a, n_b (labeled molecules at the site), usage_a, usage_b (the site's share of its
  gene's labeled molecules in that group, 0 on the side without a site), label, a20, hexamer, region, dist_end, dist_mode.
  On shared rows gene and site features are those of site_a; site_b lies within SITE_TOL nt of it.
- matched_sites_summary.parquet: pair, set, feature, bin, sites, molecules (n_a + n_b) per bin of label, a_band, a_pos (an A
  at position 1 to 20 after the site, from down20), hexamer, region and dist_bin (burden.end_bin: |dist_end| at
  DIST_END_BINS, "past" when the site lies downstream of its nearest end); every bin kept, zeros included.
- matched_sites.json: groups, sites and molecules per group; per pair the shared genes, sites and molecules per set with the
  shares of shared sites (among all, the first group's and the second group's sites), the Spearman correlation of usage_a
  and usage_b over shared sites with the quartiles of log2(usage_a / usage_b) by label and region, and for nuclei_vs_cells
  the downsampling: the cell molecules of each shared gene drawn without replacement to the nuclei count (all of them where
  the cells have fewer), DOWNSAMPLE_DRAWS draws from numpy.random.default_rng(SEED), mean and 95% range of each set size
  and share; the acceptance.

Acceptance (burden.same_site_check): on the site tables without the gene filter, the one-way nearest match of
burden.same_site reproduces every label-pair count of <base>/burden/same_site.parquet. When it does not, the outputs are
written and the stage exits with an error.

Measured on 2026-10-14 on the full data (/usr/bin/time -v): 49.8 s wall, 1.15 GiB maximum resident size. A prototype run
of the same steps took 12 s to read genes.gtf twice, 12 s for the four site tables and 1.5 s for 5 draws.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from pafbench.burden import (SETS, downsample_sets, downsampling_summary, gene_ends, same_site_check, set_counts,
                             set_summary, shared_genes, site_sets, site_table, usage_comparison)
from pafbench.config import CELLS, DIST_END_BINS, DOWNSAMPLE_DRAWS, GTF, NUCLEI, OUT, SITE_TOL
from pafbench.exons import load_collapsed_exons
from pafbench.truth import in_population

COLUMNS = ["ref", "strand", "site", "label", "a_band", "a20", "down20", "up40", "hexamer", "region", "gene", "in_cell"]
PAIRS = (("1417_vs_1419", "1417", "1419"), ("nuclei_vs_cells", "nuclei", "cells"))
MIN_GENE = 20  # burden.same_gene's min_n
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
                               "cells") for lib in NUCLEI + CELLS}
    groups = {lib: [lib] for lib in NUCLEI} | {"nuclei": list(NUCLEI), "cells": list(CELLS)}
    mols = {g: pd.concat([libs[lib] for lib in members], ignore_index=True) for g, members in groups.items()}
    exons, ends = load_collapsed_exons(args.gtf), gene_ends(args.gtf)
    sites = {g: site_table(m, exons, ends) for g, m in mols.items()}
    expected = pd.read_parquet(args.base / "burden" / "same_site.parquet")

    rows, summaries, pairs, acceptance = [], [], {}, {}
    for pair, a, b in PAIRS:
        genes = shared_genes(mols[a], mols[b], MIN_GENE)
        r = site_sets(sites[a], sites[b], genes)
        rows.append(r.assign(pair=pair)[OUT_COLUMNS])
        s = set_summary(r)
        s.insert(0, "pair", pair)
        summaries.append(s)
        pairs[pair] = {"first": a, "second": b, "shared_genes": len(genes), "sets": set_counts(r), "usage": usage_comparison(r)}
        if pair == "nuclei_vs_cells":
            pairs[pair]["downsampling"] = {"seed": SEED} | downsampling_summary(
                downsample_sets(sites[a], sites[b], genes, DOWNSAMPLE_DRAWS, SEED))
        acceptance[pair] = same_site_check(sites[a], sites[b], expected[expected.pair == pair])
    passed = all(c["pass"] for c in acceptance.values())

    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(args.out / "matched_sites.parquet", index=False)
    pd.concat(summaries, ignore_index=True).to_parquet(args.out / "matched_sites_summary.parquet", index=False)
    out = {"population": "cells", "groups": groups, "gtf": str(args.gtf), "site_tol": SITE_TOL, "min_molecules_gene": MIN_GENE,
           "dist_end_bins": list(DIST_END_BINS), "sites": {g: len(t) for g, t in sites.items()},
           "molecules": {g: len(m) for g, m in mols.items()}, "pairs": pairs,
           "acceptance": {"source": str(args.base / "burden" / "same_site.parquet"), "pass": passed, **acceptance},
           "seconds": round(time.time() - t0, 1)}
    (args.out / "matched_sites.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"shared_genes": {p: d["shared_genes"] for p, d in pairs.items()},
                      "sets": {p: {k: d["sets"][k]["sites"] for k in SETS} for p, d in pairs.items()},
                      "spearman_rho": {p: d["usage"]["spearman_rho"] for p, d in pairs.items()},
                      "acceptance": passed, "seconds": out["seconds"]}))
    if not passed:
        sys.exit(f"acceptance failed: {json.dumps(acceptance)}")


if __name__ == "__main__":
    main()
