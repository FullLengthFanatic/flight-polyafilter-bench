"""UMIs by the documented counting rule on the read groups of stage 23, before filtering and at each setting (stage 24;
design of 2026-10-12, section 3.2).

The rule, Cell Ranger 9.0 algorithm page (https://www.10xgenomics.com/support/software/cell-ranger/9.0/algorithms-overview/
cr-gex-algorithm, wording in design 2.3): "If two or more groups of reads have the same barcode and UMI, but different gene
annotations, the gene annotation with the most supporting reads is kept for UMI counting, and the other read groups are
discarded. In case of a tie for maximal read support, all read groups are discarded". It is applied to the corrected UMIs
(UB) of the unfiltered run: before filtering a group has n_reads reads, at a setting n_reads minus the reads removed.

Parts are read reference by reference. The competing gene of a cell and UMI can sit on another reference, so the keys of
all parts are sorted first to find the cells and UMIs with two or more groups (multi_units); their groups are collected
from every part and go through the rule together (rule_on_multi); a group that is alone holds the UMI while it has a read
(ref_block).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from .matrices import filtered
from .molstream import GROUP_MAX
from .stream import RE_CODES

_COUNTS = ("n_reads", "n_winner_reads", "n_pa")       # summed when a key sits on two references, like the rm columns


def counted(reads, cu_start):
    """True for the group whose gene the documented rule counts for its cell and UMI.

    reads: reads of each group, the groups of a cell and UMI next to each other; cu_start: first row of each cell and UMI.
    Among the groups that have a read, the one with strictly the most reads; a tie for the maximum counts none.
    """
    if reads.size == 0:
        return np.zeros(0, np.bool_)
    n = np.diff(np.append(cu_start, reads.size))
    top = (reads == np.repeat(np.maximum.reduceat(reads, cu_start), n)) & (reads > 0)
    return top & (np.repeat(np.add.reduceat(top, cu_start, dtype=np.int64), n) == 1)


def events(c0, ct, cu_start):
    """(lost, gained, switched) per group at one setting, from counted() before filtering (c0) and at the setting (ct).

    Design 3.2 step 4: lost, the group held the UMI and no longer does; gained, the group holds it and did not before;
    switched, the cell and UMI has both, the UMI moved to another gene (set on the group that loses and the one that gains).
    """
    lost, gained = c0 & ~ct, ct & ~c0
    if c0.size == 0:
        return lost, gained, lost
    n = np.diff(np.append(cu_start, c0.size))
    both = np.logical_or.reduceat(lost, cu_start) & np.logical_or.reduceat(gained, cu_start)
    return lost, gained, (lost | gained) & np.repeat(both, n)


def _in_sorted(values, x):
    """True for the elements of x that are in the sorted array values."""
    if values.size == 0:
        return np.zeros(x.size, np.bool_)
    i = np.searchsorted(values, x)
    i[i == values.size] = 0
    return values[i] == x


def multi_units(part_dir, refs):
    """(cells and UMIs with two or more read-group rows over the parts of refs, number of cells and UMIs with a read group,
    number of keys that sit on two references).

    The first is sorted, uint64, key >> 16 (cell row 24 bits, UMI code 24 bits). A key on two references makes two rows,
    so its cell and UMI is among them.
    """
    keys = np.concatenate([pq.read_table(Path(part_dir) / f"{ref}.read_groups.parquet", columns=["key"]).column("key").to_numpy()
                           for ref in refs])
    keys.sort()
    n_dup = int((keys[1:] == keys[:-1]).sum())
    keys >>= np.uint64(16)
    again = keys[1:] == keys[:-1]
    return np.unique(keys[1:][again]), keys.size - int(again.sum()), n_dup


def rule_on_multi(part_dir, refs, multi_cu, tags):
    """The read groups of the cells and UMIs in multi_cu from the parts of every reference, one row per key, sorted by key,
    with the rule applied before filtering and at each tag.

    Columns of the parts, then ref (int16 index into refs: the first reference that holds the key), counted (bool: the
    group holds the UMI before filtering), multi (True), any_removed, lost, gained and switched (uint32, bit t for tags[t];
    events()). A key on two references (design 3.2 step 1): its counts are summed, re_class is the union and both 3' ends
    are set to -1, missing.
    """
    frames = []
    for r, ref in enumerate(refs):
        part = pq.read_table(Path(part_dir) / f"{ref}.read_groups.parquet")
        sel = _in_sorted(multi_cu, part.column("key").to_numpy() >> np.uint64(16))
        frames.append(part.filter(pa.array(sel)).to_pandas().assign(ref=np.int16(r)))
    g = pd.concat(frames, ignore_index=True)
    g = g.iloc[np.argsort(g.key.to_numpy(), kind="stable")].reset_index(drop=True)
    key = g.key.to_numpy()
    new = np.ones(len(g), np.bool_)
    new[1:] = key[1:] != key[:-1]
    if not new.all():
        start = np.flatnonzero(new)
        twice = np.diff(np.append(start, len(g))) > 1
        cols = {}
        for c in g.columns:
            x = g[c].to_numpy()
            if c in _COUNTS or c.startswith("rm_"):
                total = np.add.reduceat(x, start, dtype=np.int64)
                if total.max() >= GROUP_MAX:
                    raise ValueError(f"{c} of a read group on two references sums to {total.max():,}; the columns are uint16")
                cols[c] = total.astype(x.dtype)
            elif c == "re_class":
                cols[c] = np.bitwise_or.reduceat(x, start)
            elif c in ("end3", "end3_nonpa"):
                cols[c] = np.where(twice, -1, x[start]).astype(x.dtype)
            else:               # key, and ref: the rows of a key are in refs order
                cols[c] = x[start]
        g = pd.DataFrame(cols)
        key = g.key.to_numpy()
    cu = key >> np.uint64(16)
    first = np.ones(len(g), np.bool_)
    first[1:] = cu[1:] != cu[:-1]
    cu_start = np.flatnonzero(first)
    n = g.n_reads.to_numpy(np.int64)
    c0 = counted(n, cu_start)
    bits = {name: np.zeros(len(g), np.uint32) for name in ("any_removed", "lost", "gained", "switched")}
    for t, tag in enumerate(tags):
        rm = g[f"rm_{tag}"].to_numpy(np.int64)
        for name, x in zip(bits, (rm > 0, *events(c0, counted(n - rm, cu_start), cu_start))):
            bits[name] |= x.astype(np.uint32) << np.uint32(t)
    return g.assign(counted=c0, multi=True, **bits)


def ref_block(part_dir, refs, r, multi_cu, multi, tags):
    """The read groups of refs[r] with the rule applied: the columns of its part, then counted, multi (the cell and UMI has
    two or more groups), any_removed, lost, gained and switched (rule_on_multi), sorted by key.

    A group that is alone in its cell and UMI holds the UMI while it has a read; the others come from multi. A key on two
    references is in the block of the first one only.
    """
    part = pq.read_table(Path(part_dir) / f"{refs[r]}.read_groups.parquet")
    alone = part.filter(pa.array(~_in_sorted(multi_cu, part.column("key").to_numpy() >> np.uint64(16)))).to_pandas()
    n = alone.n_reads.to_numpy()
    any_removed, lost = np.zeros(len(alone), np.uint32), np.zeros(len(alone), np.uint32)
    for t, tag in enumerate(tags):
        rm = alone[f"rm_{tag}"].to_numpy()
        any_removed |= (rm > 0).astype(np.uint32) << np.uint32(t)
        lost |= (rm >= n).astype(np.uint32) << np.uint32(t)
    none = np.zeros(len(alone), np.uint32)
    alone = alone.assign(counted=n > 0, multi=False, any_removed=any_removed, lost=lost, gained=none, switched=none)
    g = pd.concat([alone, multi[multi.ref == r].drop(columns="ref")], ignore_index=True)
    return g.iloc[np.argsort(g.key.to_numpy(), kind="stable")].reset_index(drop=True)


class Tally:
    """Sums over the blocks of ref_block: the entries of the matrices and the counts per cell and per gene of the totals.

    cell_re, gene_re: UMIs held before filtering (index 0) and at each tag (1 + t), by cell or gene and re_class of the
    group that holds them. cell_events: per tag and cell, UMIs lost, gained and switched and the molecules with a removed
    read that keep their UMI. groups_re: read groups of the cell with a record of RE E, of RE N. gene_ref: the first
    reference (index of the block) with a read group of the gene, -1 without one.
    """

    def __init__(self, n_cells, n_genes, tags):
        self.shape, self.tags = (n_cells, n_genes), list(tags)
        self.pieces = {name: [] for name in ("xf", "unfiltered", *((kind, t) for t in self.tags for kind in ("lost", "gained")))}
        self.cell_re = np.zeros((1 + len(self.tags), n_cells, 4), np.int64)
        self.gene_re = np.zeros((1 + len(self.tags), n_genes, 4), np.int64)
        self.cell_events = np.zeros((len(self.tags), n_cells, 4), np.int64)
        self.groups_re = np.zeros((n_cells, 2), np.int64)
        self.gene_ref = np.full(n_genes, -1, np.int16)

    def add(self, block, r):
        n_cells, n_genes = self.shape
        key = block.key.to_numpy()
        cell, gene = (key >> np.uint64(40)).astype(np.int64), (key & np.uint64(0xFFFF)).astype(np.int64)
        code, re_class, held = cell * n_genes + gene, block.re_class.to_numpy().astype(np.int64), block.counted.to_numpy()
        self.pieces["xf"].append(np.unique(code[block.n_winner_reads.to_numpy() > 0], return_counts=True))
        self.pieces["unfiltered"].append(np.unique(code[held], return_counts=True))
        self._held(0, cell, gene, re_class, held)
        for t, tag in enumerate(self.tags):
            lost, gained, switched, removed = ((block[c].to_numpy() >> np.uint32(t)) & 1 == 1
                                               for c in ("lost", "gained", "switched", "any_removed"))
            self.pieces["lost", tag].append(np.unique(code[lost], return_counts=True))
            self.pieces["gained", tag].append(np.unique(code[gained], return_counts=True))
            self._held(1 + t, cell, gene, re_class, (held & ~lost) | gained)
            # a switch is one UMI: counted on the group that loses it
            for k, x in enumerate((lost, gained, switched & lost, held & ~lost & removed)):
                self.cell_events[t, :, k] += np.bincount(cell[x], minlength=n_cells)
        for k in (0, 1):
            self.groups_re[:, k] += np.bincount(cell[re_class & (1 << k) > 0], minlength=n_cells)
        new = np.unique(gene)
        self.gene_ref[new[self.gene_ref[new] < 0]] = r

    def _held(self, j, cell, gene, re_class, held):
        n_cells, n_genes = self.shape
        self.cell_re[j] += np.bincount(cell[held] * 4 + re_class[held], minlength=4 * n_cells).reshape(n_cells, 4)
        self.gene_re[j] += np.bincount(gene[held] * 4 + re_class[held], minlength=4 * n_genes).reshape(n_genes, 4)

    def matrix(self, name):
        """Cells by genes CSR int32 of "xf" (groups with a winner read), "unfiltered", ("lost", tag) or ("gained", tag);
        the entries of the blocks are handed over, a second call finds none."""
        pieces = self.pieces.pop(name)
        code, n = (np.concatenate([p[k] for p in pieces]) for k in (0, 1))
        m = sp.csr_matrix((n.astype(np.int32), (code // self.shape[1], code % self.shape[1])), shape=self.shape)
        m.sum_duplicates()               # a key on two references aside, an entry comes from one block
        return m


def sum_records(part_dir, refs, n_cells, tags):
    """The cell_records parts of refs summed: int64 array [cell, RE, GX, column]; the last cell holds the rows over all
    barcodes, RE in the order of stream.RE_CODES (E, N, I, none), GX 1 when the records carry the tag, columns total,
    with_ub, units, units_rg, then the records removed at each tag."""
    cols = ["total", "with_ub", "units", "units_rg", *tags]
    rec = np.zeros((n_cells + 1, len(RE_CODES), 2, len(cols)), np.int64)
    for ref in refs:
        part = pd.read_parquet(Path(part_dir) / f"{ref}.cell_records.parquet")
        # a part has one row per cell, RE and GX, and cell -1 is the last row of rec
        rec[part.cell.to_numpy(), part.re.map(RE_CODES.index).to_numpy(), part.gx.to_numpy().astype(np.int64)] += part[cols].to_numpy()
    return rec


def totals(tally, m0, lost, gained, rec, barcodes, features, refs):
    """(cell_totals, gene_totals) in long form, one block of rows per tag: "unfiltered", then tally.tags (design 3.2, Outputs).

    umis and the columns by re_class are counted from the read groups, genes and cells (entries with a UMI) from
    m0 - lost[tag] + gained[tag]; check A5 compares the two routes. rec: sum_records. The record columns hold the primary
    records of the cell kept at the tag. Units and read groups by RE describe the unfiltered library and are empty at the
    settings. features: (gene id, gene name) per row of features.tsv.gz.
    """
    n_cells = len(barcodes)
    removed = np.zeros(rec.shape[:3], np.int64)
    units = {f"units{rg}_{name}": rec[:n_cells, k, :, col].sum(axis=1)
             for rg, col, names in (("", 2, ("E", "N", "I", "none")), ("_rg", 3, ("E", "N"))) for k, name in enumerate(names)}
    units |= {"rg_E": tally.groups_re[:, 0], "rg_N": tally.groups_re[:, 1]}
    gene_ref = pd.Series(np.array([*refs, None], dtype=object)[tally.gene_ref], dtype="str")
    cells, genes = [], []
    for j, tag in enumerate(["unfiltered", *tally.tags]):
        m = m0 if j == 0 else filtered(m0, lost[tag], gained[tag])
        ev = tally.cell_events[j - 1] if j else np.zeros((n_cells, 4), np.int64)
        if j:
            removed = rec[:, :, :, 3 + j]
        kept = (rec[:, :, :, 0] - removed)[:n_cells]
        cre, gre = tally.cell_re[j], tally.gene_re[j]
        cells.append(pd.DataFrame(
            {"cb": barcodes, "tag": tag, "umis": cre.sum(axis=1), "genes": m.getnnz(axis=1), "lost": ev[:, 0], "gained": ev[:, 1],
             "switched": ev[:, 2], "umis_E": cre[:, 1], "umis_N": cre[:, 2], "umis_EN": cre[:, 3], "mol_partial": ev[:, 3]}
            | {f"rec_{name}": kept[:, k].sum(axis=1) for k, name in enumerate(("E", "N", "I", "none"))}
            | {"rec_EN_gx": kept[:, :2, 1].sum(axis=1)}
            | {name: pd.array(x if j == 0 else np.full(n_cells, pd.NA), dtype="Int64") for name, x in units.items()}))
        l, g = (np.zeros(m0.shape[1], np.int64) if j == 0 else np.asarray(x[tag].sum(axis=0)).ravel() for x in (lost, gained))
        genes.append(pd.DataFrame(
            {"gene_id": [f[0] for f in features], "gene_name": [f[1] for f in features], "ref": gene_ref, "tag": tag,
             "umis": gre.sum(axis=1), "cells": m.getnnz(axis=0), "lost": l, "gained": g, "umis_E": gre[:, 1], "umis_N": gre[:, 2],
             "umis_EN": gre[:, 3]}))
    return pd.concat(cells, ignore_index=True), pd.concat(genes, ignore_index=True)
