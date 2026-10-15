"""Stage 28: selection bias, labeled against unlabeled molecules (design of 2026-10-12, section 3.6).

Population: the molecules of called cells that the counting rule of stage 24 counts before filtering
(recount/counted_molecules.parquet, one row per UMI of M0). A counted molecule is labeled when a truth molecule joins it
(eval/truth_to_counted.parquet of stage 27, joined): the join goes through the read group (cell, UMI, GX of a pa read), so
the gene is GX, not our own gene. The pass's n_pa above 0 is a second definition of labeled, cross-tabulated with it.

Five axes (AXES), each a table of category shares among labeled and among unlabeled molecules (tally, axis_table), and a
record table by RE (records_table). An axis fires when a share differs by more than BIAS_POINTS points between the two
groups (rule); what that decides is decision. The templated share among counted molecules reweighted over the fired axes
among 1, 2, 3 and 5 is reweight; the acceptance of 3.6 is acceptance.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .burden import END_BINS, _exon_coord
from .config import BIAS_POINTS, DIST_END_BINS, EXPR_BINS, MIN_STRATUM, POOL_MIN, TOP_GENES
from .evaluate import interval, resample_counts
from .recount import _in_sorted
from .strata import _key

AXES = ("region", "cell_class", "expr_bin", "position", "family_size")   # axes 1 to 5 of design 3.6, in its order
REWEIGHT_AXES = ("region", "cell_class", "expr_bin", "family_size")       # the merge order of the pooling
REGIONS = ("E", "N", "EN")      # re_class 1, 2 and 3 of stage 23: exonic records only, intronic only, both
FAMILY = ("1", "2", "3+")       # n_reads
NONPA = ("0", "1", "2", "3+")   # reads of the molecule without a pa tag, n_reads - n_pa
EXPR = tuple(f"q{i}" for i in range(1, EXPR_BINS + 1))   # the names of strata.expr_bins
PAST, PA_ONLY = "past every annotated end", "pa read only"
POSITIONS = END_BINS[:len(DIST_END_BINS) + 1] + (PAST, PA_ONLY)   # categories of axis 4
RULE_STRATUM = {"position": "1"}  # the rule reads axis 4 among molecules with one read without a pa tag, the others in "all"
AXIS_COLUMNS = ["lib", "axis", "stratum", "category", "n_labeled", "n_unlabeled", "share_labeled", "share_unlabeled",
                "diff_points"]
RE = ("E", "N", "I")
RECORD_COLUMNS = ["lib", "re", "labeled", "records", "units", "share_labeled", "share_records", "share_units",
                  "diff_points_records", "diff_points_units", "units_rg", "read_groups", "units_per_read_group"]
GENE_COLUMNS = ["lib", "gene_id", "gene_name", "umis", "labeled", "labeled_fraction", "top"]
# the accuracy table of a fired axis that moves from the details block into the main text (design 3.6, point 3)
PLACEMENT = {"region": "eval/metrics.parquet", "cell_class": "eval/strata_m3.parquet", "expr_bin": "eval/strata_m3.parquet"}
_TOL = 1e-9  # a difference of exactly BIAS_POINTS, up to floating point rounding, does not fire


def _share(n):
    n = np.asarray(n, np.float64)
    total = n.sum()
    return n / total if total else np.full(len(n), np.nan)


def axis_rows(lib, axis, stratum, categories, n_labeled, n_unlabeled):
    """Rows of one axis and stratum in AXIS_COLUMNS: molecules per category in each group, the category's share of the
    group (NaN when the group is empty) and the difference share_labeled - share_unlabeled in percentage points."""
    sl, su = _share(n_labeled), _share(n_unlabeled)
    return pd.DataFrame({"lib": lib, "axis": axis, "stratum": stratum, "category": list(categories),
                         "n_labeled": np.asarray(n_labeled, np.int64), "n_unlabeled": np.asarray(n_unlabeled, np.int64),
                         "share_labeled": sl, "share_unlabeled": su, "diff_points": 100 * (sl - su)})


def _largest(diff, names, points):
    d = np.abs(np.asarray(diff, np.float64))
    if not np.isfinite(d).any():
        return {"fired": False, "max_abs_diff_points": None}, None
    k = int(np.nanargmax(d))
    return {"fired": bool(d[k] - points > _TOL), "max_abs_diff_points": float(d[k])}, names[k]


def rule(table, records, points=BIAS_POINTS):
    """The 5 point rule of design 3.6 on one library's axis table (AXIS_COLUMNS) and record table (records_table, or None).
    An axis fires when a category of its rule stratum (RULE_STRATUM, else "all") differs by more than points between labeled
    and unlabeled molecules; the record table fires when the share of labeled molecules of an RE differs by more than points
    from the share of duplicate-free units (diff_points_units). Per axis found in table: the stratum read, fired, the
    largest |diff_points| and its category; fired lists the fired axes in the order of AXES."""
    out = {"axes": {}}
    for axis in [a for a in AXES if a in set(table.axis)]:
        stratum = RULE_STRATUM.get(axis, "all")
        t = table[(table.axis == axis) & (table.stratum == stratum)]
        res, cat = _largest(t.diff_points, t.category.tolist(), points)
        out["axes"][axis] = {"stratum": stratum, **res, "category": cat}
    out["fired"] = [a for a, r in out["axes"].items() if r["fired"]]
    if records is not None:
        res, re = _largest(records.diff_points_units, records.re.tolist(), points)
        out["records"] = {**res, "re": re}
    return out


def decision(fired, records_fired):
    """What the fired axes (names of AXES) and the record table decide, and nothing else (design 3.6, points 1 to 3):
    wording, every headline sentence on burden and accuracy says "among molecules with a polyA-trimmed read" and names
    axes_named; burden_numbers, the headline and the unweighted share among counted labeled molecules always, the share
    reweighted over reweight_axes (the fired axes among 1, 2, 3 and 5) when there is one; main_text_accuracy, the accuracy
    tables of the fired axes among region, cell class and expression that move into the main text."""
    named = [a for a in AXES if a in fired]
    reweight = [a for a in REWEIGHT_AXES if a in fired]
    return {"wording": bool(named) or bool(records_fired), "axes_named": named + (["records"] if records_fired else []),
            "burden_numbers": ["headline", "counted_unweighted"] + (["counted_reweighted"] if reweight else []),
            "reweight_axes": reweight,
            "main_text_accuracy": [{"axis": a, "table": PLACEMENT[a], "stratum_kind": a} for a in named if a in PLACEMENT]}


def gene_strands(gtf):
    """{gene_id: True on plus} from the GTF transcript lines, the lines burden.gene_ends takes the ends from."""
    out = {}
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if f[2] == "transcript":
                out[f[8].split('gene_id "', 1)[1].split('"', 1)[0]] = f[6] == "+"
    return out


_LOW = np.int64((1 << 32) - 1)


@dataclass
class EndIndex:
    """Collapsed exons and annotated 3' ends of the genes of a features.tsv, gene row g, as keys (g << 32) | coordinate
    sorted over all genes, so that one numpy.searchsorted finds a molecule's exon or end within its own gene."""
    plus: np.ndarray          # bool per gene row
    exon_key: np.ndarray      # start of each collapsed exon
    exon_end: np.ndarray      # its end
    exon_before: np.ndarray   # collapsed-exon bases of the gene before the exon
    end_key: np.ndarray       # annotated end (burden.gene_ends)
    end_exon_key: np.ndarray  # annotated end in collapsed-exon bases (burden._exon_coord)


def end_index(gene_ids, exons, ends, strands):
    """EndIndex of the genes gene_ids (features.tsv order) from exons.load_collapsed_exons, burden.gene_ends and
    gene_strands."""
    keys = [[] for _ in range(5)]
    for row, gid in enumerate(gene_ids):
        (s, e), g, x = exons[gid], np.int64(row) << 32, ends[gid]
        for k, v in enumerate((g | s, e, np.concatenate([[0], np.cumsum(e - s)[:-1]]), g | x,
                               g | np.unique(_exon_coord(s, e, x)))):
            keys[k].append(v)
    return EndIndex(np.array([strands[g] for g in gene_ids], bool), *(np.concatenate(k).astype(np.int64) for k in keys))


def exon_coord(ix, gene, x):
    """burden._exon_coord of each boundary x within the collapsed exons of its gene (row of the EndIndex ix): the
    collapsed-exon bases before x, 0 before the gene's first exon."""
    g = np.asarray(gene, np.int64)
    x = np.asarray(x, np.int64)
    j = np.searchsorted(ix.exon_key, (g << 32) | x, side="right") - 1
    k = np.maximum(j, 0)
    start = ix.exon_key[k] & _LOW
    same = (j >= 0) & ((ix.exon_key[k] >> 32) == g)
    return np.where(same, ix.exon_before[k] + np.minimum(x - start, ix.exon_end[k] - start), 0)


def downstream_end(ix, gene, base):
    """(d, exonic) per molecule: d the distance from base, the 0-based 3'-most aligned base of a read in transcript sense,
    to the nearest annotated 3' end of the molecule's gene (row of ix) that lies at or downstream of it; NaN when every
    annotated end lies upstream (past every annotated end). The definition of design 3.6, axis 4.

    As burden.end_distances of stage 31, the base is a boundary in the priming_site convention of burden.gene_ends (base + 1
    on plus, base on minus), and when the base lies in a collapsed exon of the gene (exonic) d counts collapsed-exon bases,
    otherwise genomic bases; d is 0 when the base is the last base of a transcript. Two differences: end_distances takes the
    nearest end on either side (d < 0 when it lies upstream) and the best of the genes of a site, this function the nearest
    end downstream within the one gene of the counted molecule."""
    g = np.asarray(gene, np.int64)
    base = np.asarray(base, np.int64)
    plus = ix.plus[g]
    k = np.searchsorted(ix.exon_key, (g << 32) | base, side="right") - 1
    kk = np.maximum(k, 0)
    exonic = (k >= 0) & ((ix.exon_key[kk] >> 32) == g) & (base < ix.exon_end[kk])
    x = base + plus
    c = np.where(exonic, exon_coord(ix, g, x), x)
    d = np.full(len(g), np.nan)
    for m, keys in ((exonic, ix.end_exon_key), (~exonic, ix.end_key)):
        q, p = (g[m] << 32) | c[m], plus[m]
        i = np.where(p, np.searchsorted(keys, q, side="left"), np.searchsorted(keys, q, side="right") - 1)
        ii = np.clip(i, 0, len(keys) - 1)
        ok = (i >= 0) & (i < len(keys)) & ((keys[ii] >> 32) == g[m])
        e = keys[ii] & _LOW
        d[m] = np.where(ok, np.where(p, e - c[m], c[m] - e), np.nan)
    return d, exonic


def position_codes(ix, gene, end3_nonpa):
    """Index into POSITIONS (axis 4) of each counted molecule from its end3_nonpa: the bin of downstream_end, by
    burden.end_bin's rule (|d| at DIST_END_BINS, right-closed), PAST without an annotated end at or downstream, PA_ONLY when
    end3_nonpa is -1, that is every read of the molecule has a pa tag (stage 23)."""
    end3_nonpa = np.asarray(end3_nonpa)
    code = np.full(len(end3_nonpa), POSITIONS.index(PA_ONLY), np.int8)
    m = end3_nonpa >= 0
    d, _ = downstream_end(ix, np.asarray(gene)[m], end3_nonpa[m])
    code[m] = np.where(np.isnan(d), POSITIONS.index(PAST),
                       np.searchsorted(np.asarray(DIST_END_BINS, np.float64), np.nan_to_num(d), side="left"))
    return code


def expr_bins_counted(umis):
    """(bin of each gene as an index into EXPR, the EXPR_BINS - 1 boundaries in UMIs): the bins of stage 27
    (strata.expr_bins) cut over the counted molecules, each of which carries the unfiltered UMIs of its gene, so that each
    bin holds about a quarter of them. umis: unfiltered UMIs per gene in called cells (gene_totals); a gene with u UMIs has u
    counted molecules, so the quantiles come from the gene totals by numpy.quantile's default method without the repeated
    array: virtual index (n - 1) q ("'get_virtual_index': lambda n, quantiles: (n - 1) * quantiles", numpy 2.4.6,
    numpy/lib/_function_base_impl.py line 132), the values at its floor and at the next index (lines 4691 to 4697) and
    numpy's _lerp between them (lines 4596 to 4599). A gene goes to its bin by numpy.searchsorted with side left, as in
    strata.expr_bins; -1 for a gene without UMIs."""
    x = np.asarray(umis, np.int64)
    v = np.sort(x[x > 0])
    cum = np.cumsum(v)  # index k of the sorted molecules holds the value v[i] of the first i with cum[i] > k
    h = (cum[-1] - 1) * (np.arange(1, EXPR_BINS) / EXPR_BINS)
    lo = np.floor(h)
    a, b = (v[np.searchsorted(cum, np.minimum(i, cum[-1] - 1), side="right")] for i in (lo, lo + 1))
    t = h - lo
    bounds = np.where(t >= 0.5, b - (b - a) * (1 - t), a + (b - a) * t)
    return np.where(x > 0, np.searchsorted(bounds, x, side="left"), -1), bounds


COUNTED_COLUMNS = ["cell", "gene", "umi", "n_reads", "n_pa", "re_class", "end3_nonpa"]


@dataclass
class Tally:
    """The counted molecules of one library, counted once (tally)."""
    full: np.ndarray        # [labeled, REGIONS, cell class, FAMILY, NONPA, POSITIONS, n_pa above 0]
    by_gene: np.ndarray     # [gene row, REGIONS, cell class, FAMILY]
    labeled: pd.DataFrame   # one row per labeled counted molecule: cell, gene, umi and the codes region, cell_class, family_size


def labeled_keys(join):
    """Sorted read-group keys (strata._key) of the counted molecules that a truth molecule joins: the rows of stage 27's
    truth_to_counted with joined True, whose gene is the GX of the pa read."""
    j = join[join.joined.to_numpy(bool)]
    return np.unique(_key(j.cell.to_numpy(), j.umi.to_numpy(), j.gene.to_numpy()))


def tally(counted_path, keys, cell_class, n_classes, ix, batch_rows=1 << 22):
    """One pass over counted_molecules.parquet (counted_path), COUNTED_COLUMNS only, in batches of batch_rows rows. A
    molecule is labeled when its key is one of keys (labeled_keys). Codes: REGIONS from re_class, the class code of its
    barcode row (cell_class, n_classes classes), FAMILY from n_reads, NONPA from n_reads - n_pa, position_codes (ix, an
    EndIndex of the gene rows) from end3_nonpa, and n_pa above 0, the second definition of labeled."""
    shape = (2, len(REGIONS), n_classes, len(FAMILY), len(NONPA), len(POSITIONS), 2)
    gshape = (len(ix.plus), len(REGIONS), n_classes, len(FAMILY))
    full, by_gene, lab = np.zeros(np.prod(shape), np.int64), np.zeros(np.prod(gshape), np.int64), []
    for b in pq.ParquetFile(counted_path).iter_batches(batch_size=batch_rows, columns=COUNTED_COLUMNS):
        c = {k: b.column(k).to_numpy() for k in COUNTED_COLUMNS}
        cell, gene, umi = c["cell"], c["gene"], c["umi"]
        labeled = _in_sorted(keys, _key(cell, umi, gene))
        n_reads, n_pa = c["n_reads"].astype(np.intp), c["n_pa"].astype(np.intp)
        codes = {"region": c["re_class"].astype(np.intp) - 1, "cell_class": cell_class[cell],
                 "family_size": np.minimum(n_reads, len(FAMILY)) - 1}
        idx = (labeled, *codes.values(), np.minimum(n_reads - n_pa, len(NONPA) - 1), position_codes(ix, gene, c["end3_nonpa"]),
               n_pa > 0)
        full += np.bincount(np.ravel_multi_index([np.asarray(i, np.intp) for i in idx], shape), minlength=full.size)
        by_gene += np.bincount(np.ravel_multi_index([gene.astype(np.intp), *codes.values()], gshape), minlength=by_gene.size)
        lab.append(pd.DataFrame({"cell": cell[labeled], "gene": gene[labeled], "umi": umi[labeled]} |
                                {k: v[labeled] for k, v in codes.items()}))
    return Tally(full.reshape(shape), by_gene.reshape(gshape), pd.concat(lab, ignore_index=True))


def axis_table(t, classes, expr_code, lib):
    """The five axes of one library from its Tally t, in AXIS_COLUMNS: region, cell class (classes: the names of the
    codes), expression bin (expr_code: the EXPR index of each gene row, expr_bins_counted) and family size in stratum "all";
    position in stratum "all" with every category of POSITIONS, and in the strata 1, 2 and 3+ of NONPA without PA_ONLY,
    which has a read without pa there."""
    f = t.full

    def counts(*keep):
        s = f.sum(axis=tuple(i for i in range(1, f.ndim) if i not in keep))
        return s[1], s[0]

    n_gene = t.by_gene.sum(axis=(1, 2, 3))
    m = n_gene > 0
    lab = np.bincount(t.labeled.gene.to_numpy(np.intp), minlength=len(n_gene))[m]
    expr = [np.bincount(expr_code[m], weights=w, minlength=len(EXPR)).astype(np.int64) for w in (lab, n_gene[m] - lab)]
    pos_l, pos_u = counts(4, 5)
    rows = [axis_rows(lib, "region", "all", REGIONS, *counts(1)), axis_rows(lib, "cell_class", "all", classes, *counts(2)),
            axis_rows(lib, "expr_bin", "all", EXPR, *expr),
            axis_rows(lib, "position", "all", POSITIONS, pos_l.sum(axis=0), pos_u.sum(axis=0))]
    rows += [axis_rows(lib, "position", s, POSITIONS[:-1], pos_l[k, :-1], pos_u[k, :-1]) for k, s in enumerate(NONPA) if k]
    rows.append(axis_rows(lib, "family_size", "all", FAMILY, *counts(3)))
    return pd.concat(rows, ignore_index=True)


def records_table(re, totals, lib):
    """The record table of design 3.6 in RECORD_COLUMNS, one row per RE (E, N, I): labeled molecules of called cells by
    the RE of their pa read (re, from molecules_labeled.parquet) against the called-cell primary records (rec_<RE>) and the
    duplicate-free units (units_<RE>) of cell_totals at unfiltered (totals, one row per cell). Shares are over the three
    RE; diff_points_records and diff_points_units are share_labeled minus the other share, in points. Printed next to it:
    units over read-group records (units_rg) per read group with a record of that RE (read_groups), the distance of a unit
    from a molecule where both are known; RE I has no read group (0 and NaN)."""
    re = np.asarray(re, object)
    labeled = np.array([(re == x).sum() for x in RE], np.int64)
    records, units = (np.array([totals[f"{k}_{x}"].sum() for x in RE], np.int64) for k in ("rec", "units"))
    rg = {k: np.array([totals[f"{k}_{x}"].sum() if f"{k}_{x}" in totals else 0 for x in RE], np.int64)
          for k in ("units_rg", "rg")}
    sl, sr, su = _share(labeled), _share(records), _share(units)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_group = rg["units_rg"] / rg["rg"]
    return pd.DataFrame({"lib": lib, "re": list(RE), "labeled": labeled, "records": records, "units": units,
                         "share_labeled": sl, "share_records": sr, "share_units": su, "diff_points_records": 100 * (sl - sr),
                         "diff_points_units": 100 * (sl - su), "units_rg": rg["units_rg"], "read_groups": rg["rg"],
                         "units_per_read_group": np.where(rg["rg"] > 0, per_group, np.nan)})


def gene_table(gene_ids, gene_names, umis, labeled, lib, min_umis=MIN_STRATUM, top=TOP_GENES):
    """Per gene with a counted molecule, in GENE_COLUMNS and gene row order: unfiltered UMIs (umis, the counted molecules
    of the gene), labeled counted molecules and their fraction. top marks the top genes furthest above ("above") and below
    ("below") the library's fraction (all labeled over all counted molecules) by labeled_fraction minus it, among the genes
    with at least min_umis UMIs; on equal distance the gene with more UMIs first, then the gene id."""
    umis, labeled = np.asarray(umis, np.int64), np.asarray(labeled, np.int64)
    keep = umis > 0
    t = pd.DataFrame({"lib": lib, "gene_id": np.asarray(gene_ids, object)[keep],
                      "gene_name": np.asarray(gene_names, object)[keep], "umis": umis[keep], "labeled": labeled[keep],
                      "labeled_fraction": labeled[keep] / umis[keep], "top": ""})
    d = t.labeled_fraction - labeled.sum() / umis.sum()
    ranked = t.assign(d=d)[t.umis >= min_umis]
    for name, side, up in (("above", ranked.d > 0, False), ("below", ranked.d < 0, True)):
        r = ranked[side].sort_values(["d", "umis", "gene_id"], ascending=[up, False, True])
        t.loc[r.index[:top], "top"] = name
    return t.reset_index(drop=True)


def acceptance(t, keys, m0_sum, gene_umis, join, n_called, records, stage13):
    """Design 3.6, Acceptance, for one library. molecules: labeled plus unlabeled counted molecules (the Tally t) equal the
    sum of M0 (m0_sum); every joined key (keys, labeled_keys) is a counted molecule; and the counted molecules per gene
    equal gene_umis, the unfiltered UMIs of gene_totals that the expression bins are cut on. truth: joined plus not joined
    truth molecules (join, the library's rows of truth_to_counted) equal the labeled molecules of called cells (n_called).
    records: per called cell, the primary records summed over RE (records: by barcode, rec_E + rec_N + rec_I + rec_none of
    cell_totals at unfiltered) equal the total of stage 13 (stage13: grid/cell_counts.parquet total, by barcode), and so do
    their sums over cells."""
    lab = t.full.sum(axis=tuple(range(1, t.full.ndim)))
    genes_differing = int((t.by_gene.sum(axis=(1, 2, 3)) != np.asarray(gene_umis)).sum())
    mol = {"labeled": int(lab[1]), "unlabeled": int(lab[0]), "m0_sum": int(m0_sum), "joined_keys": len(keys),
           "genes_differing": genes_differing}
    mol["passed"] = (mol["labeled"] + mol["unlabeled"] == mol["m0_sum"] and mol["joined_keys"] == mol["labeled"]
                     and not genes_differing)
    joined = int(join.joined.sum())
    truth = {"joined": joined, "not_joined": len(join) - joined, "labeled_in_called_cells": int(n_called)}
    truth["passed"] = len(join) == truth["labeled_in_called_cells"]
    both = pd.concat([records.rename("ours"), stage13.rename("stage13")], axis=1)
    differing = int((both.ours != both.stage13).sum())
    rec = {"records": int(records.sum()), "stage13": int(stage13.sum()), "cells": len(both), "cells_differing": differing}
    rec["passed"] = rec["records"] == rec["stage13"] and not differing
    return {"molecules": mol, "truth": truth, "records": rec, "passed": mol["passed"] and truth["passed"] and rec["passed"]}


def pool(first, n_labeled, min_n=POOL_MIN):
    """Pooled stratum of each stratum (design 3.6): the strata with fewer than min_n labeled molecules (n_labeled) are
    merged with the other such strata of the same category of the first fired axis (first, one value per stratum); a merged
    stratum still under min_n joins the stratum of that category with the most labeled molecules (the first of them on a
    tie), when the category has one with min_n or more. Returns per stratum the index of the stratum it is pooled into."""
    first, n_labeled = np.asarray(first), np.asarray(n_labeled)
    group = np.arange(len(first))
    for c in pd.unique(first):
        idx = np.flatnonzero(first == c)
        small, big = idx[n_labeled[idx] < min_n], idx[n_labeled[idx] >= min_n]
        if not len(small):
            continue
        group[small] = small[0]
        if n_labeled[small].sum() < min_n and len(big):
            group[small] = big[np.argmax(n_labeled[big])]
    return group


def _reweighted(N, n, t):
    """sum over strata (last axis) of N_s / sum(N) x t_s / n_s; NaN where a stratum with counted molecules has no labeled
    one."""
    with np.errstate(invalid="ignore", divide="ignore"):
        part = np.where(N > 0, N / N.sum(axis=-1, keepdims=True) * t / n, 0.0)
    return np.where(((N > 0) & (n == 0)).any(axis=-1), np.nan, part.sum(axis=-1))


def reweight(counted, labeled, axes, n_boot=1000, seed=0, min_n=POOL_MIN):
    """The templated share among counted molecules reweighted over axes (design 3.6, point 2).

    counted: the counted molecules n per gene and stratum (columns gene, the axes, n); labeled: one row per truth molecule
    that joins a counted molecule, with the gene and the strata of that molecule and templated (bool). Strata are the cross
    of axes, taken in the merge order REWEIGHT_AXES; axis 4 (position) and anything else outside REWEIGHT_AXES is refused.
    Weight of a stratum: its share among all counted molecules over its share among the labeled ones, so that the share is
    the sum over strata of P_s t_s / n_s, which assumes that within a stratum the labeled molecules have the templated share
    of the unlabeled ones. Pooling: pool on the category of the first axis. Interval: gene bootstrap (n_boot resamples of
    evaluate.resample_counts from seed over the genes of counted and labeled), the counted and labeled molecules of a drawn
    gene drawn together and the weights recomputed in every resample over the pooled strata; a resample in which a pooled
    stratum with counted molecules has no labeled one has no value and is left out (evaluate.interval, n_finite); so has the
    share itself, NaN, when a pooled stratum of the full data has none.
    strata: strata with counted molecules; strata_small: of them with fewer than min_n labeled; strata_pooled and
    pooled_share: the small ones merged with another stratum, and their share of the counted molecules; strata_final:
    strata after pooling."""
    bad = [a for a in axes if a not in REWEIGHT_AXES]
    if bad:
        raise ValueError(f"{bad} cannot enter the reweighting (design 3.6): axis 4 compares read positions and its "
                         f"category {PA_ONLY!r} has no counterpart among unlabeled molecules; the axes are {REWEIGHT_AXES}")
    cols = [a for a in REWEIGHT_AXES if a in axes]
    index = pd.MultiIndex.from_frame(counted[cols].drop_duplicates().sort_values(cols))
    sc, sl = (index.get_indexer(pd.MultiIndex.from_frame(f[cols])) for f in (counted, labeled))
    n_strata, templated, size = len(index), labeled.templated.to_numpy(np.float64), counted.n.to_numpy(np.float64)
    N, n = np.bincount(sc, weights=size, minlength=n_strata), np.bincount(sl, minlength=n_strata)
    group = pool(index.get_level_values(0), n, min_n)
    _, final, members = np.unique(group, return_inverse=True, return_counts=True)
    merged = (n < min_n) & (members[final] > 1)
    genes, code = np.unique(np.concatenate([counted.gene.to_numpy(), labeled.gene.to_numpy()]), return_inverse=True)
    gc, gl, F = code[:len(counted)], code[len(counted):], len(members)

    def per_gene(g, s, w):
        return np.bincount(g * F + final[s], weights=w, minlength=len(genes) * F).reshape(len(genes), F)

    Ng, ng, tg = per_gene(gc, sc, size), per_gene(gl, sl, None), per_gene(gl, sl, templated)
    W = resample_counts(len(genes), n_boot, seed).astype(np.float64)
    lo, hi, finite = interval(_reweighted(W @ Ng, W @ ng, W @ tg))
    share = float(_reweighted(Ng.sum(axis=0), ng.sum(axis=0), tg.sum(axis=0)))
    return {"axes": cols, "share_counted_unweighted": float(templated.mean()),
            "share_counted_reweighted": share, "lo": lo, "hi": hi, "n_boot": n_boot,
            "n_finite": finite, "strata": n_strata, "strata_small": int((n < min_n).sum()), "strata_pooled": int(merged.sum()),
            "pooled_share": float(N[merged].sum() / N.sum()), "strata_final": F}
