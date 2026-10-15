"""Stage 27: truth-set strata by cell class and expression level, the join of truth molecules to the molecules counted
before filtering, and accuracy as the matrix sees it (design of 2026-10-12, section 3.5).

The units are those of stage 17 (evaluate.units: called cells, molecules with a member read within the cap, 8/4 labels)
with two more molecule attributes, cell_class and expr_bin; the metrics come from evaluate.evaluate and the intervals from
evaluate.bootstrap on each stratum (evaluate.subset_units).
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import EXPR_BINS, MEMBER_CAP, MIN_STRATUM
from .evaluate import bootstrap, rates, subset_units
from .molstream import _umi_codes
from .recount import _in_sorted

KINDS = ("cell_class", "expr_bin")
LEVELS = ("read", "molecule")
UNASSIGNED = "unassigned"   # barcodes without a class of stage 25
NO_UMI = "no_umi"           # molecules without a gene, or whose gene has no counted UMI
BOOT_COLUMNS = ["lib", "tag", "level", "stratum_kind", "stratum", "metric", "estimate", "lo", "hi", "n_boot", "n_clusters"]
JOIN_COLUMNS = ["lib", "mol", "cell", "gene", "umi", "joined", "region"]
OUTCOME_COLUMNS = ["lib", "tag", "units", "label", "a_band", "joined", "not_joined", "lost_in_matrix", "removed_as_unit", "both"]


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


def _key(cell, umi, gene):
    """The read-group key of stage 23: cell row (24 bits), UMI code (24 bits), gene row (16 bits), highest first."""
    return (cell.astype(np.uint64) << np.uint64(40)) | (umi.astype(np.uint64) << np.uint64(16)) | gene.astype(np.uint64)


def join_counted(mol, truth_reads, pa_sites, barcodes, gene_ids, counted_path, lib, batch_rows=1 << 22):
    """The join of design 3.5: one row per molecule of mol (the labeled molecules of called cells), in JOIN_COLUMNS plus lost.

    A molecule's pa reads come from truth_reads (role pa), their GX and xf from pa_sites (by qname). A pa read with xf bit 1
    and a GX that is a row of gene_ids lies in the read group (cell, UMI, GX) of stage 23; cell and UMI are those of the
    molecule, which its pa reads share (truth.MOL_KEY). The molecule is joined when one of these groups is a row of
    counted_molecules.parquet (counted_path, the molecules counted before filtering; a cell and UMI has at most one);
    cell, gene and umi are then that row's and lost its lost bits. Otherwise gene is the smallest gene row among its groups,
    or -1 when it has none, and lost is 0. counted_molecules.parquet is read in batches of batch_rows rows, four columns.
    """
    cell = pd.Index(barcodes).get_indexer(mol.cb)
    if (cell < 0).any():
        raise ValueError(f"{int((cell < 0).sum())} molecules have a barcode that is not a row of barcodes.tsv (first "
                         f"{mol.cb.to_numpy()[cell < 0][0]})")
    umi = _umi_codes(mol.ub.tolist(), "truth molecules").astype(np.uint32)
    pa = truth_reads.loc[truth_reads.role.to_numpy() == "pa", ["qname", "mol"]]
    pa = pa.merge(pa_sites[["qname", "gx", "xf"]], on="qname", validate="one_to_one")
    mi = pd.Index(mol.mol).get_indexer(pa.mol)
    gene = pd.Index(gene_ids).get_indexer(pa.gx)
    grouped = (mi >= 0) & (gene >= 0) & ((pa.xf.to_numpy() & 1) == 1)
    mi, gene = mi[grouped], gene[grouped]
    key = _key(cell[mi], umi[mi], gene)
    want = np.unique(key)
    found, lost = [], []
    for b in pq.ParquetFile(counted_path).iter_batches(batch_size=batch_rows, columns=["cell", "gene", "umi", "lost"]):
        k = _key(*(b.column(c).to_numpy() for c in ("cell", "umi", "gene")))
        hit = _in_sorted(want, k)
        found.append(k[hit])
        lost.append(b.column("lost").to_numpy()[hit])
    row = pd.Series(np.concatenate(lost), index=np.concatenate(found)).reindex(key)
    g = pd.DataFrame({"mi": mi, "gene": gene, "counted": row.notna().to_numpy(), "lost": row.fillna(0).to_numpy(np.uint32)})
    g = g.sort_values(["mi", "counted", "gene"], ascending=[True, False, True]).drop_duplicates("mi")
    out_gene, joined, out_lost = np.full(len(mol), -1, np.int32), np.zeros(len(mol), bool), np.zeros(len(mol), np.uint32)
    i = g.mi.to_numpy()
    out_gene[i], joined[i], out_lost[i] = g.gene.to_numpy(), g.counted.to_numpy(), g.lost.to_numpy()
    return pd.DataFrame({"lib": lib, "mol": mol.mol.to_numpy(), "cell": cell.astype(np.int32), "gene": out_gene, "umi": umi,
                         "joined": joined, "region": mol.region.to_numpy(object), "lost": out_lost})


def join_summary(join, mol, gene_ids):
    """Counts of the join (rows of join and mol in the same order): joined, in a read group that is not counted
    (group_not_counted), without a pa read in a read group (no_group), all three by region; the counted molecules joined;
    and the joined molecules whose own gene (mol.gene) differs from the GX of the group, by label (design 2.3)."""
    joined, gene = join.joined.to_numpy(), join.gene.to_numpy()
    status = np.where(joined, "joined", np.where(gene >= 0, "group_not_counted", "no_group"))
    names = ("joined", "group_not_counted", "no_group")
    differs = joined & (np.asarray(gene_ids, object)[gene.clip(min=0)] != mol.gene.to_numpy(object))
    by_region = pd.crosstab(mol.region.to_numpy(object), status).reindex(columns=list(names), fill_value=0)
    return {"molecules": len(join)} | {k: int((status == k).sum()) for k in names} | {
        "counted_molecules_joined": len(join[joined].drop_duplicates(["cell", "gene", "umi"])),
        "joined_gene_differs": int(differs.sum()),
        "joined_gene_differs_by_label": {k: int(v) for k, v in pd.Series(mol.label.to_numpy()[differs]).value_counts().items()},
        "by_region": {r: {k: int(v) for k, v in row.items()} for r, row in by_region.iterrows()}}


def matrix_outcome(join, unit_sets, tags, bits, lib):
    """Accuracy as the matrix sees it (design 3.5): per tag, units (unit_sets: {name: evaluate.Units}, built at a superset of
    tags from the molecules of join), label and A band, the molecule units that join a counted molecule and those that do
    not; among the joined, those whose UMI is lost from the gene's count at the tag (the lost bit bits[tag] of
    counted_molecules, switches included), those removed by the unit rule of stage 17 (all their truth reads removed) and
    those both. In OUTCOME_COLUMNS."""
    rows = []
    for name, u in unit_sets.items():
        i = pd.Index(join.mol).get_indexer(u.mols.mol)
        joined, lost = join.joined.to_numpy()[i], join.lost.to_numpy()[i]
        removed = u.K == u.mols.n_reads.to_numpy()[:, None]
        for tag in tags:
            gone = joined & ((lost >> np.uint32(bits[tag])) & 1).astype(bool)
            rm = joined & removed[:, u.tags.index(tag)]
            f = u.mols[["label", "a_band"]].assign(joined=joined, not_joined=~joined, lost_in_matrix=gone, removed_as_unit=rm,
                                                   both=gone & rm)
            g = f.groupby(["label", "a_band"], sort=True).sum().astype(np.int64).reset_index()
            rows.append(g.assign(lib=lib, tag=tag, units=name))
    return pd.concat(rows, ignore_index=True)[OUTCOME_COLUMNS]
