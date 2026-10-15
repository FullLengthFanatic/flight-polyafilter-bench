"""Stage 27: sensitivity and specificity of the impact settings by cell class and by expression level, the join of truth
molecules to the molecules counted before filtering, and accuracy as the matrix sees it (pafbench.strata; design of
2026-10-12, section 3.5).

Units as in stage 17 (evaluate.units): labeled molecules of called cells (in_cell) with a member read within MEMBER_CAP,
8/4 labels, read and molecule level. Two more molecule attributes: cell_class, the class of the molecule's barcode in
stage 25's cells.parquet ("unassigned" without one), and expr_bin, the quartile of the unfiltered UMIs of the molecule's
gene in called cells (gene_totals.parquet of stage 24, tag "unfiltered"), cut over the molecule units; molecules without a
gene or whose gene has no counted UMI form the stratum no_umi.

Reads, per library LIB (the folders of --cells, --matrices and --recount are templates in which {lib} is replaced):
<base>/LIB/truth/molecules_labeled.parquet and truth_reads.parquet, <base>/LIB/grid/truth_flags.parquet,
<base>/LIB/pa_sites.parquet, <base>/eval/metrics.parquet (the acceptance), <cells>/cells.parquet,
<matrices>/gene_totals.parquet, barcodes.tsv and features.tsv, <recount>/counted_molecules.parquet. The stage starts only
when stage 24 has passed for LIB with these two folders (recount.require_passed).

Writes under --out (default OUT/eval), one block of rows per library:
- strata_m3.parquet: the columns of metrics.parquet for every setting of IMPACT_SETTINGS, level read and molecule,
  stratum_kind cell_class and expr_bin.
- strata_m3_bootstrap.parquet: lib, tag, level, stratum_kind, stratum, metric, estimate, lo, hi, n_boot, n_clusters; the
  gene bootstrap of stage 17 (1,000 resamples, seed 0) on each stratum, for the chosen and the published settings.
- truth_to_counted.parquet: lib, mol, cell, gene, umi, joined, region; one row per labeled molecule of a called cell. A
  molecule joins the read group (cell, UMI, GX) of one of its pa reads with xf bit 1 and a GX of the feature list, and is
  joined when that group is a molecule counted before filtering; cell, gene and umi are rows of barcodes.tsv and
  features.tsv and the UMI in 2-bit code, as in counted_molecules.parquet. A molecule that is not joined keeps the smallest
  gene row among its read groups, or -1 without one (strata.join_counted). Stage 28 reads this file.
- matrix_outcome.parquet: lib, tag, units (with_member, the headline units, or all_labeled), label, a_band, joined,
  not_joined, lost_in_matrix, removed_as_unit, both. Among the joined molecule units: the UMI lost from the gene's count at
  the tag (the lost bit of counted_molecules, switches included), all truth reads removed (the unit rule of stage 17), and
  both (strata.matrix_outcome).
- strata_m3.json: parameters, and per library the inputs, the units, molecule units per class, the expression bin
  boundaries in UMIs with the molecule units per bin, the join counts (joined, in a read group that is not counted, without
  a read group; by region; joined molecules whose own gene differs from GX, by label), the strata whose sensitivity or
  specificity rests on fewer than MIN_STRATUM molecule units (marked at both levels: the read units of a stratum are member
  reads of its molecule units), and the acceptance.

Acceptance (strata.acceptance): for each setting, level and stratum kind, tp + fn and tn + fp summed over the strata equal
the "all" stratum of metrics.parquet at cap 1,000 and 8/4, and the sensitivity and specificity of the sums equal its. When
it fails for a library, every output is written and the stage exits non-zero.

Trial on 1417 on 2026-10-15 (stage 24 outputs of the trial run, classes standing in for stage 25: the Cell Ranger graph
cluster of each barcode as "cr<k>"; /usr/bin/time -v and memwatch): 13 s wall, 1.35 GB maximum resident size. Acceptance
passed for all 56 combinations, with 31,627 templated and 941 non-templated molecule units; expression bins at 1,609,
4,816 and 15,635 UMIs (7,698, 7,688, 7,692 and 7,692 molecule units, 4,314 in no_umi); 70,466 of 90,099 molecules
joined, 283 of them with our gene other than GX, as in design 2.3.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from pafbench.config import CELLS, CHOSEN_SETTINGS, EXPR_BINS, IMPACT_SETTINGS, MEMBER_CAP, MIN_STRATUM, NUCLEI, OUT
from pafbench.evaluate import evaluate, units
from pafbench.flags import setting_tag
from pafbench.grid import published
from pafbench.recount import require_passed
from pafbench.strata import (JOIN_COLUMNS, KINDS, LEVELS, NO_UMI, acceptance, few_molecules, join_counted, join_summary,
                             matrix_outcome, strata_bootstrap, with_strata)
from pafbench.truth import LABELS, in_population

CUTOFFS = "8/4"
TAGS = [setting_tag(s) for s in IMPACT_SETTINGS]
BOOT_TAGS = [setting_tag(s) for s in (*CHOSEN_SETTINGS, *published())]
BOOT_N, BOOT_SEED = 1000, 0


def _labels(u):
    return {k: int((u.mols.label == k).sum()) for k in LABELS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("libraries", nargs="*", default=[*NUCLEI, *CELLS])
    ap.add_argument("--base", type=Path, default=OUT,
                    help="outputs of milestones 1 and 2: LIB/truth, LIB/grid, LIB/pa_sites.parquet, eval/metrics.parquet")
    ap.add_argument("--cells", default=str(OUT / "{lib}" / "cells"), help="stage 25 folder, {lib} replaced")
    ap.add_argument("--matrices", default=str(OUT / "{lib}" / "matrices"), help="stage 24 matrices folder, {lib} replaced")
    ap.add_argument("--recount", default=str(OUT / "{lib}" / "recount"), help="stage 24 recount folder, {lib} replaced")
    ap.add_argument("--out", type=Path, default=OUT / "eval")
    args = ap.parse_args()
    t0 = time.monotonic()
    metrics = pd.read_parquet(args.base / "eval" / "metrics.parquet")
    strata, boots, joins, outcomes, libs = [], [], [], [], {}
    for lib in args.libraries:
        cells_dir, matrix_dir, recount_dir = (Path(t.format(lib=lib)) for t in (args.cells, args.matrices, args.recount))
        bits = {t: i for i, t in enumerate(require_passed(lib, recount_dir, matrix_dir)["settings"])}
        truth, counted = args.base / lib / "truth", recount_dir / "counted_molecules.parquet"
        inputs = {"truth": str(truth), "truth_flags": str(args.base / lib / "grid" / "truth_flags.parquet"),
                  "pa_sites": str(args.base / lib / "pa_sites.parquet"), "cells": str(cells_dir / "cells.parquet"),
                  "matrices": str(matrix_dir), "counted_molecules": str(counted)}
        mol = in_population(pd.read_parquet(truth / "molecules_labeled.parquet"), "cells")
        tr = pd.read_parquet(truth / "truth_reads.parquet")
        fl = pd.read_parquet(inputs["truth_flags"], columns=["qname", "flag"] + TAGS)
        gt = pd.read_parquet(matrix_dir / "gene_totals.parquet", columns=["gene_id", "tag", "umis"])
        u, bounds = with_strata(units(mol, tr, fl, TAGS, MEMBER_CAP), pd.read_parquet(inputs["cells"], columns=["cb", "class"]),
                                gt[gt.tag == "unfiltered"].set_index("gene_id").umis)
        strata.append(evaluate(u, lib, MEMBER_CAP, CUTOFFS, levels=LEVELS, kinds=KINDS))
        boots.append(strata_bootstrap(u, BOOT_TAGS, lib, n=BOOT_N, seed=BOOT_SEED))
        gene_ids = [line.split("\t")[0] for line in (matrix_dir / "features.tsv").read_text().splitlines()]
        join = join_counted(mol, tr, pd.read_parquet(inputs["pa_sites"], columns=["qname", "gx", "xf"]),
                            (matrix_dir / "barcodes.tsv").read_text().split(), gene_ids, counted, lib)
        joins.append(join[JOIN_COLUMNS])
        u_all = units(mol, tr, fl, TAGS, MEMBER_CAP, all_labeled=True)
        outcomes.append(matrix_outcome(join, {"with_member": u, "all_labeled": u_all}, TAGS, bits, lib))
        no_umi = (u.mols.expr_bin == NO_UMI).to_numpy()
        no_gene = (u.mols.gene == "").to_numpy()
        libs[lib] = {"inputs": inputs, "units": {"with_member": _labels(u), "all_labeled": _labels(u_all)},
                     "read_units": len(u.reads), "missing_reads": u.missing_reads, "dropped_molecules": u.dropped_mols,
                     "cell_class": {k: int(v) for k, v in u.mols.cell_class.value_counts().sort_index().items()},
                     "expr_bins": {"boundaries": bounds.tolist(),
                                   "molecules": {k: int(v) for k, v in u.mols.expr_bin.value_counts().sort_index().items()},
                                   "no_umi": {"no_gene": int((no_umi & no_gene).sum()),
                                              "gene_without_umi": int((no_umi & ~no_gene).sum())}},
                     "join": join_summary(join, mol, gene_ids), "few_molecules": few_molecules(u),
                     "acceptance": acceptance(strata[-1], metrics, lib, TAGS)}
        print(f"{lib}: {len(u.mols)} molecule units, {int(join.joined.sum())} of {len(join)} molecules joined, expression "
              f"bins at {bounds.tolist()} UMIs, acceptance {libs[lib]['acceptance']['passed']}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(strata, ignore_index=True).to_parquet(args.out / "strata_m3.parquet", index=False)
    pd.concat(boots, ignore_index=True).to_parquet(args.out / "strata_m3_bootstrap.parquet", index=False)
    pd.concat(joins, ignore_index=True).to_parquet(args.out / "truth_to_counted.parquet", index=False)
    pd.concat(outcomes, ignore_index=True).to_parquet(args.out / "matrix_outcome.parquet", index=False)
    failed = [lib for lib, d in libs.items() if not d["acceptance"]["passed"]]
    seconds = round(time.monotonic() - t0, 1)
    (args.out / "strata_m3.json").write_text(json.dumps(
        {"cap": MEMBER_CAP, "cutoffs": CUTOFFS, "min_stratum": MIN_STRATUM, "expr_bins": EXPR_BINS, "tags": TAGS,
         "bootstrap": {"tags": BOOT_TAGS, "n": BOOT_N, "seed": BOOT_SEED, "levels": list(LEVELS)}, "libraries": libs,
         "acceptance_passed": not failed, "seconds": seconds}, indent=1))
    print(json.dumps({"libraries": list(libs), "units": {lib: d["units"]["with_member"] for lib, d in libs.items()},
                      "boundaries": {lib: d["expr_bins"]["boundaries"] for lib, d in libs.items()},
                      "joined": {lib: d["join"]["joined"] for lib, d in libs.items()}, "acceptance_passed": not failed,
                      "seconds": seconds}))
    if failed:
        sys.exit(f"acceptance failed for {', '.join(failed)}: the sums over the strata differ from eval/metrics.parquet "
                 f"(differing rows in strata_m3.json)")


if __name__ == "__main__":
    main()
