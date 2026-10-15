"""Stage 27: truth-set strata by cell class and expression level, the join of truth molecules to the molecules counted
before filtering, and accuracy as the matrix sees it (design of 2026-10-12, section 3.5).

The units are those of stage 17 (evaluate.units: called cells, molecules with a member read within the cap, 8/4 labels)
with two more molecule attributes, cell_class and expr_bin; the metrics come from evaluate.evaluate and the intervals from
evaluate.bootstrap on each stratum (evaluate.subset_units).
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import EXPR_BINS, MEMBER_CAP, MIN_STRATUM
from .evaluate import bootstrap, rates, subset_units

KINDS = ("cell_class", "expr_bin")
LEVELS = ("read", "molecule")
UNASSIGNED = "unassigned"   # barcodes without a class of stage 25
NO_UMI = "no_umi"           # molecules without a gene, or whose gene has no counted UMI
BOOT_COLUMNS = ["lib", "tag", "level", "stratum_kind", "stratum", "metric", "estimate", "lo", "hi", "n_boot", "n_clusters"]


def cell_classes(cb, cells):
    """Class of each barcode of cb from cells (stage 25's cells.parquet: cb, class); UNASSIGNED where the barcode has no
    row or its row no class."""
    return pd.Series(cb).map(cells.set_index("cb")["class"]).fillna(UNASSIGNED).to_numpy(object)


def expr_bins(gene, gene_umis, n_bins=EXPR_BINS):
    """(bin of each molecule, the n_bins - 1 boundaries in UMIs).

    gene: the molecule's gene ('' for none); gene_umis: unfiltered UMIs in called cells by gene id. The boundaries are
    numpy.quantile (linear) of the gene UMIs over the molecules whose gene has a counted UMI, so each bin holds about a
    quarter of them at n_bins 4; the bins are right-closed, as in pandas.qcut, so a molecule at a boundary falls in the lower
    one, and are named q1 (lowest) to q<n_bins>. The other molecules get NO_UMI.
    """
    x = pd.Series(gene).map(gene_umis).fillna(0).to_numpy(float)
    has = x > 0
    bounds = np.quantile(x[has], np.arange(1, n_bins) / n_bins) if has.any() else np.full(n_bins - 1, np.nan)
    out = np.full(len(x), NO_UMI, object)
    out[has] = np.array([f"q{i}" for i in range(1, n_bins + 1)], object)[np.searchsorted(bounds, x[has])]
    return out, bounds


def with_strata(u, cells, gene_umis):
    """(u with cell_class and expr_bin on u.mols, the expression bin boundaries); the bins are cut over the molecule
    units of u."""
    bins, bounds = expr_bins(u.mols.gene.to_numpy(object), gene_umis)
    return replace(u, mols=u.mols.assign(cell_class=cell_classes(u.mols.cb.to_numpy(object), cells), expr_bin=bins)), bounds


def strata_bootstrap(u, tags, lib, kinds=KINDS, levels=LEVELS, n=1000, seed=0):
    """evaluate.bootstrap on the units of each stratum (evaluate.subset_units), per level, in BOOT_COLUMNS. A stratum
    without a positive or negative unit has nothing to resample and gets no rows."""
    rows = []
    label = u.mols.label.to_numpy()
    for kind in kinds:
        col = u.mols[kind].to_numpy().astype(str)
        for value in sorted(set(col)):
            m = col == value
            if not ((label[m] == u.pos) | (label[m] == u.neg)).any():
                continue
            s = subset_units(u, m)
            for level in levels:
                rows.append(bootstrap(s, tags, level, n, seed).assign(lib=lib, level=level, stratum_kind=kind, stratum=value))
    return pd.concat(rows, ignore_index=True)[BOOT_COLUMNS]


def few_molecules(u, kinds=KINDS, min_n=MIN_STRATUM):
    """Per stratum the molecule units of the two labels, and whether the sensitivity (sens_few) or the specificity
    (spec_few) rests on fewer than min_n of them. This holds for the read level too, whose units are member reads of these
    molecules."""
    label = u.mols.label.to_numpy()
    rows = []
    for kind in kinds:
        col = u.mols[kind].to_numpy().astype(str)
        for value in sorted(set(col)):
            n_pos, n_neg = (int((label[col == value] == x).sum()) for x in (u.pos, u.neg))
            rows.append({"stratum_kind": kind, "stratum": value, u.pos: n_pos, u.neg: n_neg, "sens_few": n_pos < min_n,
                         "spec_few": n_neg < min_n})
    return rows


def acceptance(strata, metrics, lib, cap=MEMBER_CAP, cutoffs="8/4"):
    """Design 3.5, Acceptance: per setting, level and stratum kind, tp + fn and tn + fp summed over the strata equal the
    "all" stratum of metrics (stage 17's eval/metrics.parquet) at cap and cutoffs, and the sensitivity and specificity of
    the sums equal its. Returns passed, the rows compared, the differing ones and the units per level."""
    s = strata[strata.lib == lib].groupby(["tag", "level", "stratum_kind"], sort=True)[["tp", "fn", "tn", "fp"]].sum()
    s = s.reset_index()
    sens, spec, _ = rates(s.tp.to_numpy(), s.fn.to_numpy(), s.tn.to_numpy(), s.fp.to_numpy())
    s = s.assign(templated=s.tp + s.fn, nontemplated=s.tn + s.fp, sens=sens, spec=spec)
    m = metrics[(metrics.lib == lib) & (metrics.cap == cap) & (metrics.cutoffs == cutoffs) & (metrics.stratum_kind == "all")]
    m = m.assign(templated=m.tp + m.fn, nontemplated=m.tn + m.fp)
    cols = ["templated", "nontemplated", "sens", "spec"]
    j = s.merge(m[["tag", "level"] + cols], on=["tag", "level"], how="left", suffixes=("", "_m2"))
    same = np.ones(len(j), bool)
    for c in cols:
        a, b = j[c].to_numpy(float), j[f"{c}_m2"].to_numpy(float)
        same &= (a == b) | (np.isnan(a) & np.isnan(b) & j.templated_m2.notna().to_numpy())
    keep = ["tag", "level", "stratum_kind"] + cols + [f"{c}_m2" for c in cols]
    differing = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in j.loc[~same, keep].to_dict("records")]
    first = s.drop_duplicates("level").set_index("level")
    return {"passed": bool(same.all()) and len(j) > 0, "compared": len(j), "differing": differing,
            "units": {lv: {"templated": int(r.templated), "nontemplated": int(r.nontemplated)} for lv, r in first.iterrows()}}
