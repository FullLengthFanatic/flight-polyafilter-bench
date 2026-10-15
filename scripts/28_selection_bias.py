"""Stage 28: selection bias, molecules with a polyA-trimmed read (labeled) against the other counted molecules
(pafbench.selection; design of 2026-10-12, section 3.6).

Population: the molecules of called cells counted before filtering, one row of <recount>/counted_molecules.parquet per UMI
of M0. Labeled: a truth molecule of stage 27's eval/truth_to_counted.parquet joins it (joined, key cell, gene, UMI; the
gene is GX of the pa read). n_pa above 0, the pass's own definition, is cross-tabulated with it. The truth molecules that
join no counted molecule are tabulated by region.

Axes (shares of labeled and of unlabeled molecules per category):
1. region: E, N, EN from re_class (records of the molecule exonic only, intronic only, both).
2. cell_class: the class of the barcode in <cells>/cells.parquet of stage 25, "unassigned" without one; the classes are
   read from that file at run time.
3. expr_bin: quartile q1 to q4 of the gene's unfiltered UMIs in called cells (<matrices>/gene_totals.parquet), cut over the
   counted molecules so that each bin holds about a quarter of them (selection.expr_bins_counted; stage 27 cuts the same
   quantity over its truth molecule units). Per gene, the labeled fraction (selection_bias_genes.parquet), with the
   TOP_GENES genes furthest above and below the library's fraction among the genes with at least MIN_STRATUM UMIs.
4. position: end3_nonpa, the 3'-most base of the reads without a pa tag, against the nearest annotated transcript end of
   the gene at or downstream (selection.downstream_end: collapsed-exon bases when the base is exonic, genomic bases
   otherwise; ends and exons from genes.gtf as in stage 31), in the bins of DIST_END_BINS, "past every annotated end", and
   "pa read only" for the molecules none of whose reads lacks a pa tag (end3_nonpa -1). Stratum "all", and the strata 1, 2
   and 3+ of the number of reads without a pa tag; the rule reads stratum 1.
5. family_size: n_reads 1, 2, 3+.
Record table: labeled molecules of called cells by the RE of their pa read (<base>/LIB/truth/molecules_labeled.parquet)
against the called-cell primary records and the duplicate-free units of stage 23 (<matrices>/cell_totals.parquet at
unfiltered), with units per read group next to it.

The 5 point rule (selection.rule, BIAS_POINTS): an axis fires when a category differs by more than 5 points; the record
table fires on the unit share. What that decides (selection.decision) is in the json. The reweighted share among counted
molecules (selection.reweight) is computed when an axis among region, cell class, expression bin and family size fires:
strata the cross of those axes, pooling under POOL_MIN labeled molecules on the first of them, gene bootstrap (BOOT_N
resamples, seed BOOT_SEED) with the weights recomputed in every resample.

Reads, per library LIB (the folders of --cells, --matrices and --recount are templates in which {lib} is replaced):
<recount>/counted_molecules.parquet (cell, gene, umi, n_reads, n_pa, re_class, end3_nonpa), <matrices>/barcodes.tsv,
features.tsv, gene_totals.parquet, cell_totals.parquet, unfiltered.npz (the acceptance), <cells>/cells.parquet,
<eval>/truth_to_counted.parquet, <base>/LIB/truth/molecules_labeled.parquet, <base>/LIB/grid/cell_counts.parquet (the
acceptance) and --gtf. The stage starts only when stage 24 has passed for LIB with these two folders (recount.require_passed).

Writes under --out (default OUT/burden), one block of rows per library:
- selection_bias_m3.parquet: lib, axis, stratum, category, n_labeled, n_unlabeled, share_labeled, share_unlabeled,
  diff_points (share_labeled - share_unlabeled in points).
- selection_bias_records.parquet: lib, re, labeled, records, units, share_labeled, share_records, share_units,
  diff_points_records, diff_points_units, units_rg, read_groups, units_per_read_group.
- selection_bias_genes.parquet: lib, gene_id, gene_name, umis, labeled, labeled_fraction, top ("above", "below" or "").
- templated_share_reweighted.parquet: lib, axes (the fired axes among 1, 2, 3 and 5, comma-separated, "" when none),
  share_headline (templated share of the labeled molecules of called cells), share_counted_unweighted (of the truth
  molecules that join a counted molecule), share_counted_reweighted, lo, hi (NaN when no such axis fires), strata,
  strata_pooled, pooled_share.
- selection_bias_m3.json: threshold, parameters, and per library the inputs, counts, classes, expression bin boundaries,
  the join by region, the n_pa cross-tabulation, the rule outcome per axis and for the record table, the decision, the
  reweighting, the burden numbers, the gene ranking and the acceptance.

Acceptance (selection.acceptance): labeled plus unlabeled counted molecules equal the sum of M0, every joined key is a
counted molecule and the counted molecules per gene equal the gene totals; joined plus not joined truth molecules equal
the labeled molecules of called cells; per called cell, the primary records summed over RE equal the total of stage 13
(grid/cell_counts.parquet), and so do their sums. When it fails for a library, every output is written and the stage exits
non-zero.

Measured on 2026-10-15, the three libraries in one run (stage 24 outputs of the trial runs, classes of the stage 25 trial
before the review gate, truth_to_counted.parquet of the stage 27 trial; BLAS and OpenMP limited to 8 threads; /usr/bin/time
-v and memwatch): 5 min 58 s wall and 4.11 GiB peak tree memory (4.26 GiB maximum resident size); 1417 alone 60 s and
1.97 GiB. Reading genes.gtf three times takes 16 s of it. A second run of 1417 wrote the same four tables and the same json
block. Acceptance passed for all three libraries: labeled plus unlabeled counted molecules 70,466 + 22,966,456,
82,359 + 22,621,876 and 851,305 + 218,541,780, the sums of M0; joined plus not joined truth molecules 70,466 + 19,633,
82,359 + 23,041 and 851,305 + 323,472, the 90,099, 105,400 and 1,174,777 labeled molecules of called cells; records summed
over RE equal to stage 13 in every called cell (56,730,829, 56,602,795 and 491,713,467 in all). Two checks of our own on the
same day: selection.downstream_end equals a per-molecule computation with burden._exon_coord on 3,000 random counted
molecules of 1417 and 3,000 of 1179; and all 63,032 primary records with xf bit 1 and one GX in 1417 chr22:20,000,000 to
24,000,000 lie on the strand of their gene, the strand downstream_end takes for the reads of a molecule.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pafbench.burden import gene_ends
from pafbench.config import (BIAS_POINTS, CELLS, DIST_END_BINS, EXPR_BINS, GTF, MIN_STRATUM, NUCLEI, OUT, POOL_MIN,
                             TOP_GENES)
from pafbench.exons import load_collapsed_exons
from pafbench.matrices import load
from pafbench.recount import require_passed
from pafbench.selection import (AXES, EXPR, acceptance, axis_table, decision, end_index, expr_bins_counted, gene_strands,
                                gene_table, labeled_keys, records_table, reweight, rule, tally)
from pafbench.strata import cell_classes
from pafbench.truth import in_population

BOOT_N, BOOT_SEED = 1000, 0
SHARE_COLUMNS = ["lib", "axes", "share_headline", "share_counted_unweighted", "share_counted_reweighted", "lo", "hi", "strata",
                 "strata_pooled", "pooled_share"]


def _json(x):
    return None if isinstance(x, float) and np.isnan(x) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("libraries", nargs="*", default=[*NUCLEI, *CELLS])
    ap.add_argument("--base", type=Path, default=OUT,
                    help="outputs of milestones 1 and 2: LIB/truth/molecules_labeled.parquet, LIB/grid/cell_counts.parquet")
    ap.add_argument("--cells", default=str(OUT / "{lib}" / "cells"), help="stage 25 folder, {lib} replaced")
    ap.add_argument("--matrices", default=str(OUT / "{lib}" / "matrices"), help="stage 24 matrices folder, {lib} replaced")
    ap.add_argument("--recount", default=str(OUT / "{lib}" / "recount"), help="stage 24 recount folder, {lib} replaced")
    ap.add_argument("--eval", type=Path, default=OUT / "eval", help="stage 27 folder with truth_to_counted.parquet")
    ap.add_argument("--gtf", type=Path, default=GTF)
    ap.add_argument("--out", type=Path, default=OUT / "burden")
    args = ap.parse_args()
    t0 = time.monotonic()
    joins = pd.read_parquet(args.eval / "truth_to_counted.parquet",
                            columns=["lib", "mol", "cell", "gene", "umi", "joined", "region"])
    exons, ends, strands = load_collapsed_exons(args.gtf), gene_ends(args.gtf), gene_strands(args.gtf)
    axes, records, genes, shares, libs = [], [], [], [], {}
    for lib in args.libraries:
        cells_dir, matrix_dir, recount_dir = (Path(t.format(lib=lib)) for t in (args.cells, args.matrices, args.recount))
        require_passed(lib, recount_dir, matrix_dir)
        counted = recount_dir / "counted_molecules.parquet"
        truth = args.base / lib / "truth" / "molecules_labeled.parquet"
        stage13 = args.base / lib / "grid" / "cell_counts.parquet"
        inputs = {"counted_molecules": str(counted), "matrices": str(matrix_dir), "cells": str(cells_dir / "cells.parquet"),
                  "truth_to_counted": str(args.eval / "truth_to_counted.parquet"), "molecules_labeled": str(truth),
                  "cell_counts": str(stage13), "gtf": str(args.gtf)}
        barcodes = (matrix_dir / "barcodes.tsv").read_text().split()
        features = [line.split("\t") for line in (matrix_dir / "features.tsv").read_text().splitlines()]
        gene_ids, gene_names = [f[0] for f in features], [f[1] for f in features]
        cls = cell_classes(np.array(barcodes, object), pd.read_parquet(inputs["cells"], columns=["cb", "class"]))
        classes, class_code = np.unique(cls, return_inverse=True)
        gt = pd.read_parquet(matrix_dir / "gene_totals.parquet", columns=["gene_id", "tag", "umis"])
        umis = gt[gt.tag == "unfiltered"].set_index("gene_id").umis.reindex(gene_ids, fill_value=0).to_numpy(np.int64)
        expr_code, bounds = expr_bins_counted(umis)
        join = joins[joins.lib == lib].reset_index(drop=True)
        keys = labeled_keys(join)
        t = tally(counted, keys, class_code, len(classes), end_index(gene_ids, exons, ends, strands))
        mol = in_population(pd.read_parquet(truth, columns=["mol", "re", "label", "in_cell"]), "cells")

        tab = axis_table(t, classes.tolist(), expr_code, lib)
        ct = pd.read_parquet(matrix_dir / "cell_totals.parquet")
        ct = ct[ct.tag == "unfiltered"]
        rec = records_table(mol.re.to_numpy(object), ct, lib)
        r = rule(tab, rec)
        dec = decision(r["fired"], r["records"]["fired"])

        # each joined truth molecule with its label and the strata of the counted molecule it joins
        joined = join.loc[join.joined, ["mol", "cell", "gene", "umi"]].merge(mol[["mol", "label"]], on="mol", how="left",
                                                                             validate="one_to_one")
        lab = joined.merge(t.labeled, on=["cell", "gene", "umi"], validate="many_to_one")
        lab = lab.assign(expr_bin=expr_code[lab.gene.to_numpy()], templated=(lab.label == "templated").to_numpy())
        gene, region, cls, fam = np.nonzero(t.by_gene)
        per_stratum = pd.DataFrame({"gene": gene, "region": region, "cell_class": cls, "expr_bin": expr_code[gene],
                                    "family_size": fam, "n": t.by_gene[gene, region, cls, fam]})
        rw = reweight(per_stratum, lab, dec["reweight_axes"], BOOT_N, BOOT_SEED) if dec["reweight_axes"] else None
        headline = float((mol.label == "templated").mean())
        unweighted = float(lab.templated.mean())
        shares.append({"lib": lib, "axes": ",".join(dec["reweight_axes"]), "share_headline": headline,
                       "share_counted_unweighted": unweighted} |
                      ({k: rw[k] for k in SHARE_COLUMNS[4:]} if rw else
                       {"share_counted_reweighted": np.nan, "lo": np.nan, "hi": np.nan, "strata": 0, "strata_pooled": 0,
                        "pooled_share": 0.0}))

        n_lab = np.bincount(t.labeled.gene.to_numpy(np.intp), minlength=len(gene_ids))
        gtab = gene_table(gene_ids, gene_names, umis, n_lab, lib)
        cells_total = ct.set_index("cb")[["rec_E", "rec_N", "rec_I", "rec_none"]].sum(axis=1)
        acc = acceptance(t, keys, load(lib, "unfiltered", matrix_dir).sum(), umis, join,
                         len(mol), cells_total, pd.read_parquet(stage13, columns=["total"]).total)
        npa = t.full.sum(axis=tuple(range(1, t.full.ndim - 1)))
        by_region = pd.crosstab(join.region, join.joined).reindex(columns=[True, False], fill_value=0)
        libs[lib] = {
            "inputs": inputs, "counted_molecules": int(t.full.sum()), "labeled": int(t.full[1].sum()),
            "classes": classes.tolist(),
            "expr_bins": {"boundaries": bounds.tolist(),
                          "molecules": dict(zip(EXPR, np.bincount(expr_code[umis > 0], weights=umis[umis > 0],
                                                                  minlength=len(EXPR)).astype(int).tolist()))},
            "join_by_region": {k: {"joined": int(v[True]), "not_joined": int(v[False])} for k, v in by_region.iterrows()},
            "n_pa": {g: {"n_pa_above_0": int(npa[i, 1]), "n_pa_0": int(npa[i, 0])}
                     for i, g in ((1, "labeled"), (0, "unlabeled"))},
            "rule": r, "decision": dec, "reweighting": {k: _json(v) for k, v in rw.items()} if rw else None,
            "max_abs_diff_points": max((v["max_abs_diff_points"] for v in r["axes"].values()
                                        if v["max_abs_diff_points"] is not None), default=None),
            "burden": {k: _json(shares[-1][k]) for k in SHARE_COLUMNS[2:7]},
            "genes": {"library_fraction": float(n_lab.sum() / umis.sum()), "min_umis": MIN_STRATUM, "top": TOP_GENES,
                      "ranked": int((gtab.umis >= MIN_STRATUM).sum())},
            "acceptance": acc}
        axes.append(tab)
        records.append(rec)
        genes.append(gtab)
        print(f"{lib}: {libs[lib]['labeled']} of {libs[lib]['counted_molecules']} counted molecules labeled, fired "
              f"{r['fired'] or 'none'}, records {r['records']['fired']}, acceptance {acc['passed']}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(axes, ignore_index=True).to_parquet(args.out / "selection_bias_m3.parquet", index=False)
    pd.concat(records, ignore_index=True).to_parquet(args.out / "selection_bias_records.parquet", index=False)
    pd.concat(genes, ignore_index=True).to_parquet(args.out / "selection_bias_genes.parquet", index=False)
    pd.DataFrame(shares, columns=SHARE_COLUMNS).to_parquet(args.out / "templated_share_reweighted.parquet", index=False)
    failed = [lib for lib, d in libs.items() if not d["acceptance"]["passed"]]
    seconds = round(time.monotonic() - t0, 1)
    (args.out / "selection_bias_m3.json").write_text(json.dumps(
        {"threshold_points": BIAS_POINTS, "pool_min": POOL_MIN, "dist_end_bins": list(DIST_END_BINS), "expr_bins": EXPR_BINS,
         "axes": dict(zip(map(str, range(1, len(AXES) + 1)), AXES)), "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED},
         "libraries": libs, "acceptance_passed": not failed, "seconds": seconds}, indent=1))
    print(json.dumps({"libraries": list(libs), "fired": {lib: d["rule"]["fired"] for lib, d in libs.items()},
                      "records_fired": {lib: d["rule"]["records"]["fired"] for lib, d in libs.items()},
                      "acceptance_passed": not failed, "seconds": seconds}))
    if failed:
        sys.exit(f"acceptance failed for {', '.join(failed)}: see the acceptance of each library in selection_bias_m3.json")


if __name__ == "__main__":
    main()
