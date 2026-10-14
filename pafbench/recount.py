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
import gzip
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.io
import scipy.sparse as sp

from . import matrices, naive
from .config import A4_PARALLEL, OUT, RULE_AGREEMENT_MAX, XF_RECOUNT_TOLERANCE
from .flags import Setting, setting_tag
from .matrices import filtered
from .molstream import COUNTS, GROUP_MAX, RULE_VERSION, UMI_LEN, part_meta, row_index
from .pool import map_forked
from .stream import RE_CODES, stale_parts
from .validation import time_exit_status

_COUNTS = ("n_reads", "n_winner_reads", "n_pa")       # summed when a key sits on two references, like the rm columns
EXAMPLES = 100                # offending entries listed per check in recount.json and on stderr; the counts are complete


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
        where = part.cell.to_numpy(), part.re.map(RE_CODES.index).to_numpy(), part.gx.to_numpy().astype(np.int64)
        rec[where] += part[cols].to_numpy()
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


def read_cr_matrix(cr_matrix_dir):
    """Cell Ranger's filtered_feature_bc_matrix/matrix.mtx.gz as cells by genes CSR int32. The file is genes by cells (our
    observation on the outs of 1417, the layout tests/synth._outs_10x writes)."""
    return sp.csr_matrix(scipy.io.mmread(str(Path(cr_matrix_dir) / "matrix.mtx.gz")).T, dtype=np.int32)


def check_entries(ours, theirs, gate):
    """Two cells by genes matrices entry by entry (checks A1, A1b and A4): entries with a count in either, equal and
    differing ones, per_million, and differing, a frame cell, gene, ours, theirs of every entry that differs. passed when
    at most gate entries differ."""
    d = (ours - theirs).tocoo()
    d.eliminate_zeros()
    compared = (abs(ours) + abs(theirs)).nnz
    mine = np.asarray(ours.tocsr()[d.row, d.col]).ravel().astype(np.int64) if d.nnz else np.zeros(0, np.int64)
    differing = pd.DataFrame({"cell": d.row, "gene": d.col, "ours": mine, "theirs": mine - d.data})
    return {"entries_compared": compared, "entries_equal": compared - len(differing), "entries_differing": len(differing),
            "per_million": 1e6 * len(differing) / max(compared, 1), "gate": gate, "passed": len(differing) <= gate,
            "differing": differing.sort_values(["cell", "gene"], ignore_index=True)}


def winner_groups(part_dir, refs):
    """(key, n_reads) of the read groups with a winner read (xf bit 1 set, bit 2 unset) over the parts of refs, sorted by
    key; the reads of a key on two references are summed."""
    cols = [pq.read_table(Path(part_dir) / f"{ref}.read_groups.parquet", columns=["key", "n_reads", "n_winner_reads"])
            for ref in refs]
    key, n, win = (np.concatenate([t.column(c).to_numpy() for t in cols]) for c in ("key", "n_reads", "n_winner_reads"))
    key, n = key[win > 0], n[win > 0].astype(np.int64)
    o = np.argsort(key, kind="stable")
    key, n = key[o], n[o]
    new = np.ones(key.size, np.bool_)
    new[1:] = key[1:] != key[:-1]
    start = np.flatnonzero(new)
    return key[start], np.add.reduceat(n, start) if key.size else n


def molecule_info(h5_path, cell_idx, gene_ids):
    """(key, count) of the molecule_info.h5 rows of pass_filter barcodes, sorted by key (cell row of cell_idx, UMI code,
    gene row, as in the parts).

    Our observation on the three libraries (h5py; tests/synth._outs_10x writes the same layout): barcodes without the
    "-1", which is the gem_group column; barcode_info/pass_filter rows of barcode, library and genome index; features/id in
    the order of features.tsv.gz. ValueError when the feature ids differ from gene_ids or a pass_filter barcode is no row
    of cell_idx.
    """
    with h5py.File(h5_path, "r") as f:
        if f["features/id"][:].astype(str).tolist() != list(gene_ids):
            raise ValueError(f"{h5_path}: features/id is not the first column of features.tsv.gz")
        gem = np.unique(f["gem_group"][:])
        if gem.size > 1:
            raise ValueError(f"{h5_path}: {gem.size} gem groups; the join takes the barcodes of one")
        barcodes = f["barcodes"][:].astype(str)
        called = f["barcode_info/pass_filter"][:, 0].astype(np.int64)
        row = np.full(barcodes.size, -1, np.int64)
        row[called] = [cell_idx.get(f"{b}-{gem[0]}", -1) for b in barcodes[called]] if called.size else []
        if (row[called] < 0).any():
            raise ValueError(f"{h5_path}: {int((row[called] < 0).sum())} pass_filter barcodes are not in barcodes.tsv.gz")
        cell = row[f["barcode_idx"][:]]
        keep = cell >= 0
        key = ((cell[keep].astype(np.uint64) << np.uint64(40)) | (f["umi"][:][keep].astype(np.uint64) << np.uint64(16))
               | f["feature_idx"][:][keep].astype(np.uint64))
        count = f["count"][:][keep].astype(np.int64)
    o = np.argsort(key, kind="stable")
    return key[o], count[o]


def check_a2(key, n_reads, mol_key, mol_count):
    """Check A2: the read groups with a winner read (winner_groups) against the rows of molecule_info (both sorted by key).

    joined, only_read_groups, only_molecule_info, read_count_mismatches (n_reads is not count), and left_over, a frame
    key, n_reads, count of the rows without a partner (the missing side empty) and of the mismatches. passed when nothing
    is left over and no count differs.
    """
    pos = np.minimum(np.searchsorted(mol_key, key), max(mol_key.size - 1, 0))
    found = mol_key[pos] == key if mol_key.size else np.zeros(key.size, np.bool_)
    seen = np.zeros(mol_key.size, np.bool_)
    seen[pos[found]] = True
    differ = found.copy()
    differ[found] = n_reads[found] != mol_count[pos[found]]
    bad = ~found | differ
    only = np.concatenate((np.zeros(int(bad.sum()), np.bool_), np.ones(int((~seen).sum()), np.bool_)))
    left = pd.DataFrame({
        "key": np.concatenate((key[bad], mol_key[~seen])),
        "n_reads": pd.arrays.IntegerArray(np.concatenate((n_reads[bad], np.zeros(only.sum(), np.int64))), only),
        "count": pd.arrays.IntegerArray(np.concatenate((np.where(found[bad], mol_count[pos[bad]], 0) if mol_key.size
                                                        else np.zeros(int(bad.sum()), np.int64), mol_count[~seen])),
                                        np.concatenate((~found[bad], np.zeros(only.sum(), np.bool_))))})
    out = {"read_groups": int(key.size), "molecules": int(mol_key.size), "joined": int(found.sum()),
           "only_read_groups": int((~found).sum()), "only_molecule_info": int((~seen).sum()),
           "read_count_mismatches": int(differ.sum())}
    return out | {"passed": not len(left), "left_over": left}


def check_a3(rec, barcodes, cell_counts, region_counts, tags):
    """Check A3, the tie to the engine of stage 13: the record counters of the pass (sum_records) against
    grid/cell_counts.parquet (index cb; total and removed primary records per called cell and tag, joined on the barcode)
    and grid/region_counts.parquet (tag, re, removed, total over all barcodes). Exact; differing lists every value that
    differs as (what, barcode or RE, column, ours, stage 13)."""
    cols = ["total", *tags]
    per_cell = rec[:-1].sum(axis=(1, 2))[:, [0, *range(4, 4 + len(tags))]]
    ours = pd.DataFrame(per_cell, index=pd.Index(barcodes, name="cb"), columns=cols)
    theirs = cell_counts[cols].astype(np.int64)
    both = ours.index.intersection(theirs.index)
    differing = [("cell only here", cb, "", int(ours.total[cb]), None) for cb in ours.index.difference(theirs.index)]
    differing += [("cell only in stage 13", cb, "", None, int(theirs.total[cb])) for cb in theirs.index.difference(ours.index)]
    a, b = ours.loc[both].to_numpy(), theirs.loc[both].to_numpy()
    differing += [("cell", both[i], cols[j], int(a[i, j]), int(b[i, j])) for i, j in np.argwhere(a != b)]
    region = {(r.tag, r.re): (int(r.removed), int(r.total)) for r in region_counts[region_counts.tag.isin(tags)].itertuples()}
    every = rec[-1].sum(axis=1)
    mine = {(tag, RE_CODES[k]): (int(every[k, 4 + t]), int(every[k, 0])) for t, tag in enumerate(tags)
            for k in range(len(RE_CODES)) if every[k, 0]}
    for k in sorted(set(mine) | set(region)):
        if mine.get(k) != region.get(k):
            differing.append(("region removed, total", k[1], k[0], mine.get(k), region.get(k)))
    return {"cells": len(both), "settings": len(tags), "cell_values_compared": int(a.size), "region_rows_compared": len(region),
            "values_differing": len(differing), "passed": not differing,
            "differing": pd.DataFrame(differing, columns=["what", "name", "column", "ours", "stage13"])}


def check_groups(g, settings):
    """The invariants of check A5 that hold group by group, on a table of whole cells and UMIs sorted by key (the groups of
    a block that are alone, or the table of rule_on_multi). Returns the violations as a frame invariant, key, tag:

    removed_exceeds_reads   more reads removed than the group has
    exceeds_control         more reads removed than at the remove-all control of the same COVLEN and mode (its pieces
                            contain the setting's, stream.reference_pieces)
    event_without_removal   a loss, gain or switch in a cell and UMI none of whose reads is removed
    held_without_reads      the group holds the UMI at the setting and has no read left (what is counted after filtering
                            is among the read groups, with a surviving read)
    gain_in_single_group    a gain in a cell and UMI with one read group
    single_group_loss       a group alone in its cell and UMI is not lost exactly when all its reads are removed
    """
    key = g.key.to_numpy()
    cu = key >> np.uint64(16)
    first = np.ones(len(g), np.bool_)
    first[1:] = cu[1:] != cu[:-1]
    start = np.flatnonzero(first)
    size = np.diff(np.append(start, len(g)))
    single = np.repeat(size, size) == 1
    n, held0 = g.n_reads.to_numpy(np.int64), g.counted.to_numpy()
    tags = [setting_tag(s) for s in settings]
    rm = {tag: g[f"rm_{tag}"].to_numpy(np.int64) for tag in tags}
    found = []
    for t, (s, tag) in enumerate(zip(settings, tags)):
        lost, gained, switched = ((g[c].to_numpy() >> np.uint32(t)) & 1 == 1 for c in ("lost", "gained", "switched"))
        bad = {"removed_exceeds_reads": rm[tag] > n}
        control = setting_tag(Setting(s.covlen, None, 0, s.introns))
        if s.min_len is not None and control in rm:
            bad["exceeds_control"] = rm[tag] > rm[control]
        if len(g):
            touched = np.repeat(np.logical_or.reduceat(rm[tag] > 0, start), size)
            bad["event_without_removal"] = (lost | gained | switched) & ~touched
        bad["held_without_reads"] = ((held0 & ~lost) | gained) & (rm[tag] >= n)
        bad["gain_in_single_group"] = gained & single
        bad["single_group_loss"] = single & (lost != (held0 & (rm[tag] >= n)))
        found += [pd.DataFrame({"invariant": name, "key": key[x], "tag": tag}) for name, x in bad.items() if x.any()]
    if not found:
        return pd.DataFrame({"invariant": [], "key": np.zeros(0, np.uint64), "tag": []})
    return pd.concat(found, ignore_index=True)


def check_matrices(m0, lost, gained):
    """The invariants of check A5 on the matrices: a loss matrix is at most m0 entry by entry and m0 - lost + gained has
    no negative entry. Violations as a frame invariant, cell, gene, tag, value."""
    found = []
    for tag in lost:
        for name, m in (("loss_above_unfiltered", m0 - lost[tag]), ("negative_entry", filtered(m0, lost[tag], gained[tag]))):
            m = m.tocoo()
            neg = m.data < 0
            found.append(pd.DataFrame({"invariant": name, "cell": m.row[neg], "gene": m.col[neg], "tag": tag,
                                       "value": m.data[neg]}))
    return pd.concat(found, ignore_index=True)


def check_totals(cell_totals, gene_totals, sums):
    """The last invariant of check A5: per tag, the UMIs of cell_totals, of gene_totals and of the matrix (sums: {tag:
    matrix sum}) are one number. The tags where they are not, as a frame tag, cells, genes, matrix."""
    t = pd.DataFrame({"cells": cell_totals.groupby("tag").umis.sum(), "genes": gene_totals.groupby("tag").umis.sum(),
                      "matrix": pd.Series(sums)}).rename_axis("tag").reset_index()
    return t[(t.cells != t.genes) | (t.cells != t.matrix)].reset_index(drop=True)


def require_parts(part_dir, refs, meta):
    """Stop (SystemExit) unless every reference of refs has its three part files of stage 23 and all were made from meta
    (molstream.part_meta of this run): stage 23 calls a reference finished on its json alone and describes only its last
    invocation, so the files are looked at here."""
    part_dir = Path(part_dir)
    missing = [f"{ref}.{kind}" for ref in refs for kind in ("json", "read_groups.parquet", "cell_records.parquet")
               if not (part_dir / f"{ref}.{kind}").exists()]
    if missing:
        raise SystemExit(f"{part_dir}: stage 23 is not complete, missing {', '.join(missing)}; run scripts/23_read_groups.py "
                         "for the whole library first")
    stale = stale_parts(part_dir, refs, meta)
    if stale:
        raise SystemExit(f"{part_dir}: the parts of {', '.join(stale)} were made from other inputs, with another unit bin or "
                         f"rule version or by another pafbench/molstream.py (this run: {json.dumps(meta)}); trash them with "
                         f"`gio trash {part_dir}` and rerun stage 23")


def require_passed(lib, recount_dir=None):
    """The recount.json of lib; stop (SystemExit) unless stage 24 wrote it with passed, that is after every acceptance
    check ran and passed. Stages 25 to 28 start with this (design 3.2). recount_dir: default OUT/<lib>/recount."""
    path = Path(recount_dir or OUT / lib / "recount") / "recount.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if res.get("passed") is not True:
        raise SystemExit(f"{path}: stage 24 has not passed its acceptance checks for {lib} (no recount.json, or one without "
                         f"passed); run scripts/24_recount.py {lib} and resolve what it reports")
    return res


def original_bams(folder, prefix, settings):
    """{tag: BAM the original tool wrote at that setting} for check A4: <folder>/<prefix><tag>.bam, as stages 02 and 20
    name them. Stop (SystemExit) when one is missing or its run did not end with exit status 0 in <prefix><tag>.time.txt
    (validation.time_exit_status; our reasoning: a run that did not finish may have left an incomplete BAM)."""
    bams = {setting_tag(s): Path(folder) / f"{prefix}{setting_tag(s)}.bam" for s in settings}
    missing = []
    for path in bams.values():
        time_file = path.with_suffix(".time.txt")
        if not (path.exists() and time_file.exists() and time_exit_status(time_file) == 0):
            missing.append(str(path))
    if missing:
        raise SystemExit(f"check A4 needs the BAMs of the original tool, missing or not finished: {', '.join(missing)}")
    return bams


def groups_of_units(part_dir, refs, units):
    """Every read group of the given cells and UMIs (key >> 16) over the parts of refs, sorted by key: key, n_reads,
    n_winner_reads, n_pa, re_class, ref. The printout of the resolution step (design 3.2): n_winner_reads above 0 is xf bit
    2 unset, re_class the RE of the records; UR and the flags are not in the parts."""
    cols = ["key", "n_reads", "n_winner_reads", "n_pa", "re_class"]
    units, frames = np.unique(units), []
    for ref in refs:
        part = pq.read_table(Path(part_dir) / f"{ref}.read_groups.parquet", columns=cols)
        sel = _in_sorted(units, part.column("key").to_numpy() >> np.uint64(16))
        frames.append(part.filter(pa.array(sel)).to_pandas().assign(ref=ref))
    return pd.concat(frames, ignore_index=True).sort_values("key", ignore_index=True)


def _umi_letters(code):
    return "".join("ACGT"[(code >> 2 * (UMI_LEN - 1 - i)) & 3] for i in range(UMI_LEN))


def _listing(frame, barcodes, gene_ids, limit=EXAMPLES):
    """The first limit rows of a frame of offending entries as json records: a key column taken apart, and cell, gene and
    umi as barcode, gene id and letters."""
    f = frame if limit is None else frame.head(limit)
    if "key" in f:
        key = f.key.to_numpy().astype(np.uint64)
        f = f.drop(columns="key").assign(cell=(key >> np.uint64(40)).astype(np.int64),
                                         umi=((key >> np.uint64(16)) & np.uint64(0xFFFFFF)).astype(np.int64),
                                         gene=(key & np.uint64(0xFFFF)).astype(np.int64))
    names = {"cell": lambda c: barcodes[c], "gene": lambda g: gene_ids[g], "umi": _umi_letters}
    f = f.assign(**{c: [fn(int(x)) for x in f[c]] for c, fn in names.items() if c in f})
    return json.loads(f.to_json(orient="records"))


def _jsonable(check, barcodes, gene_ids, limit=EXAMPLES):
    """A check result with its frames of offending entries as listings."""
    return {k: _listing(v, barcodes, gene_ids, limit) if isinstance(v, pd.DataFrame) else v for k, v in check.items()}


def against_check_script(path, entries_differing, groups):
    """Check A1b next to the genome-wide check of the design (design_m3_draft/review/v2/a1b_genomewide_<lib>.json, script
    a1b_genomewide.py): its number of differing entries, whether the count and the cells and UMIs are the same, and the
    cells and UMIs that are new here or gone, as [barcode, UMI]. groups: the listing of rule_vs_xf."""
    theirs = json.loads(Path(path).read_text())
    known = {(e["barcode"], e["umi"]) for e in theirs["exceptions"]}
    mine = {(g["cell"], g["umi"]) for g in groups}
    n = theirs["A1b_documented_rule"]["entries_differing"]
    return {"file": str(path), "entries_differing": n, "same": n == entries_differing and mine == known,
            "new": sorted(map(list, mine - known)), "gone": sorted(map(list, known - mine))}


def _molecules(g, r, refs, event):
    """Rows of counted_molecules.parquet (event "lost") or other_read_groups.parquet (event "gained") from the groups g of
    refs[r]."""
    key = g.key.to_numpy()
    cols = {"cell": (key >> np.uint64(40)).astype(np.int32), "gene": (key & np.uint64(0xFFFF)).astype(np.int32),
            "umi": ((key >> np.uint64(16)) & np.uint64(0xFFFFFF)).astype(np.uint32),
            "ref": pa.DictionaryArray.from_arrays(pa.array(np.full(len(g), r, np.int16)), pa.array(refs, pa.string()))}
    cols |= {c: g[c].to_numpy() for c in ("n_reads", "n_pa", "re_class", "end3", "end3_nonpa", "any_removed", event, "switched")}
    return pa.table(cols)


def rule_matrices(part_dir, refs, tags, shape):
    """{tag: matrix after filtering} by the rule on the read groups of refs alone, before and after removal. For check A4 on
    a BAM that holds only these references: a group whose winning gene sits elsewhere has no partner there and is counted,
    in the BAM and here (design 3.2, A4)."""
    multi_cu, _, _ = multi_units(part_dir, refs)
    multi = rule_on_multi(part_dir, refs, multi_cu, tags)
    tally = Tally(*shape, tags)
    for r in range(len(refs)):
        tally.add(ref_block(part_dir, refs, r, multi_cu, multi, tags), r)
    m0 = tally.matrix("unfiltered")
    return {tag: filtered(m0, tally.matrix(("lost", tag)), tally.matrix(("gained", tag))) for tag in tags}


def check_a4(job):
    """Checks A4 and A6 for one BAM the original tool wrote. job: (tag, BAM, Cell Ranger's filtered_feature_bc_matrix
    folder, the matrix of the rule at tag). A4: the naive recount by the documented rule equals the matrix, tolerance 0.
    A6, reported only: the entries on which Cell Ranger's counting code on the raw UMIs of the same reads differs from it,
    and the UMIs only the rule or only that code counts. In both listings ours is the matrix of the rule and theirs the
    count from the BAM."""
    tag, bam, cr_matrix_dir, expected = job
    t0 = time.monotonic()
    got = naive.recount_bam(bam, cr_matrix_dir)
    a4, a6 = check_entries(expected, got["rule"], 0), check_entries(expected, got["port"], 0)
    d = a6["differing"].ours - a6["differing"].theirs
    a6 = {k: a6[k] for k in ("entries_compared", "entries_differing", "per_million")} | {
        "umis_rule": int(expected.sum()), "umis_code": int(got["port"].sum()), "umis_only_rule": int(d[d > 0].sum()),
        "umis_only_code": int(-d[d < 0].sum()), "differing": a6["differing"]}
    counts = {k: got[k] for k in ("records", "in_groups", "read_groups", "raw_groups")}
    return tag, {"bam": str(bam)} | counts | a4 | {"seconds": round(time.monotonic() - t0, 1)}, a6


def run(lib, bam, refs, settings, recount_dir, matrix_dir, grid_dir, a4_bams, a4_refs=None, check_script=None):
    """Stage 24 for one library: the rule on the parts of stage 23 under recount_dir/parts, the output tables and matrices,
    and the acceptance checks A1 to A6 in the order of design 3.2. Returns the content of recount.json.

    refs: the references with mapped reads, in the order of the BAM index. grid_dir: the stage 13 folder with
    cell_counts.parquet and region_counts.parquet. a4_bams: {tag: BAM the original tool wrote at that setting}; a4_refs:
    the references those BAMs hold when they are a subset (the rule is then applied to the read groups of these alone).
    check_script: the json of the design's genome-wide check of A1b, when there is one.
    A failed check of A1 to A5 ends the run: recount.json then holds the checks so far, without passed, and the failed
    check is printed to stderr with its offending entries. passed is written last, after every check has run and passed.
    """
    t0 = time.monotonic()
    recount_dir, matrix_dir = Path(recount_dir), Path(matrix_dir)
    part_dir, cr_dir = recount_dir / "parts", Path(bam).parent / "filtered_feature_bc_matrix"
    meta = part_meta(bam, cr_dir, settings)
    require_parts(part_dir, refs, meta)
    tags = meta["settings"]
    cells, genes = row_index(cr_dir / "barcodes.tsv.gz"), row_index(cr_dir / "features.tsv.gz")
    barcodes, gene_ids = list(cells), list(genes)
    stats = [json.loads((part_dir / f"{ref}.json").read_text()) for ref in refs]
    res = {"library": lib, "rule_version": RULE_VERSION, "settings": tags, "refs": len(refs)} | {
        k: sum(st[k] for st in stats) for k in COUNTS} | {"seconds": {}}
    seconds, last = res["seconds"], [t0]

    def lap(name):
        now = time.monotonic()
        seconds[name], last[0] = round(now - last[0], 1), now

    def save():
        (recount_dir / "recount.json").write_text(json.dumps(res, indent=1))
        return res

    def stop(*names):
        for name in names:
            print(f"stage 24, {lib}: {name}\n{json.dumps(res[name], indent=1)}", file=sys.stderr)
        return save()

    save()               # the recount.json of an earlier run is gone, with its passed, before an output is written again
    multi_cu, n_units, n_dup = multi_units(part_dir, refs)
    multi = rule_on_multi(part_dir, refs, multi_cu, tags)
    tally = Tally(len(cells), len(genes), tags)
    violations = [check_groups(multi, settings)]
    cu = multi.key.to_numpy() >> np.uint64(16)
    differ = [multi[_in_sorted(np.unique(cu[(multi.counted != (multi.n_winner_reads > 0)).to_numpy()]), cu)]]
    n_rows = {"lost": 0, "gained": 0}
    files = {"lost": recount_dir / "counted_molecules.parquet", "gained": recount_dir / "other_read_groups.parquet"}
    writers = {}
    try:
        for r in range(len(refs)):
            block = ref_block(part_dir, refs, r, multi_cu, multi, tags)
            tally.add(block, r)
            alone = block[~block.multi]
            violations.append(check_groups(alone, settings))
            differ.append(alone[alone.n_winner_reads == 0])          # counted by the rule, discarded by Cell Ranger
            for event, held in (("lost", True), ("gained", False)):
                table = _molecules(block[block.counted == held], r, refs, event)
                if event not in writers:
                    writers[event] = pq.ParquetWriter(files[event], table.schema)
                writers[event].write_table(table)
                n_rows[event] += table.num_rows
    finally:
        for w in writers.values():
            w.close()
    vs = pd.concat(differ, ignore_index=True).sort_values("key", ignore_index=True)
    vs = vs[["key", "n_reads", "n_winner_reads"]].assign(counted_by_rule=vs.counted, counted_by_xf=vs.n_winner_reads > 0)
    res |= {"duplicate_keys": n_dup, "cell_umis": n_units, "cell_umis_two_or_more_groups": int(multi_cu.size),
            "counted_molecules": n_rows["lost"], "other_read_groups": n_rows["gained"],
            "events": {tag: dict(zip(("lost", "gained", "switched"), map(int, tally.cell_events[t, :, :3].sum(axis=0))))
                       for t, tag in enumerate(tags)}}
    lap("rule")

    m0, cr = tally.matrix("unfiltered"), read_cr_matrix(cr_dir)
    a1 = check_entries(tally.matrix("xf"), cr, XF_RECOUNT_TOLERANCE)
    a1b = check_entries(m0, cr, RULE_AGREEMENT_MAX * cr.nnz)      # at most one differing entry per million of the matrix
    res["A1"] = _jsonable(a1, barcodes, gene_ids)
    groups = _listing(vs, barcodes, gene_ids, None if a1b["passed"] else EXAMPLES)
    res["A1b"] = _jsonable(a1b, barcodes, gene_ids) | {
        "cell_umis_differing": int(np.unique(vs.key.to_numpy() >> np.uint64(16)).size),
        "rule_only_molecules": int((vs.counted_by_rule & ~vs.counted_by_xf).sum()),
        "xf_only_molecules": int((vs.counted_by_xf & ~vs.counted_by_rule).sum()), "read_groups": groups}
    if check_script is not None:
        res["A1b"]["check_script"] = against_check_script(check_script, a1b["entries_differing"], groups)
    h5 = Path(bam).parent / "molecule_info.h5"
    if not (a1["passed"] and a1b["passed"]):
        # resolution step of design 3.2: the join to molecule_info.h5 shows which molecules are missing or extra, and the
        # read groups of their cells and UMIs are listed
        a2 = check_a2(*winner_groups(part_dir, refs), *molecule_info(h5, cells, gene_ids))
        res["A2"] = _jsonable(a2, barcodes, gene_ids)
        units = np.concatenate((a2["left_over"].key.to_numpy(), vs.key.to_numpy())) >> np.uint64(16)
        res["resolution"] = _listing(groups_of_units(part_dir, refs, units[:EXAMPLES]), barcodes, gene_ids)
        return stop("A1", "A1b", "A2", "resolution")

    rec = sum_records(part_dir, refs, len(cells), tags)
    a3 = check_a3(rec, barcodes, pd.read_parquet(Path(grid_dir) / "cell_counts.parquet", columns=["total", *tags]),
                  pd.read_parquet(Path(grid_dir) / "region_counts.parquet"), tags)
    res["A3"] = _jsonable(a3, barcodes, gene_ids)
    if not a3["passed"]:
        return stop("A3")

    lost, gained = ({tag: tally.matrix((kind, tag)) for tag in tags} for kind in ("lost", "gained"))
    with gzip.open(cr_dir / "features.tsv.gz", "rt") as fh:
        features = [line.rstrip("\n").split("\t")[:2] for line in fh]
    cell_totals, gene_totals = totals(tally, m0, lost, gained, rec, barcodes, features, refs)
    sums = {"unfiltered": int(m0.sum())} | {tag: int(m0.sum() - lost[tag].sum() + gained[tag].sum()) for tag in tags}
    bad = [pd.concat(violations, ignore_index=True), check_matrices(m0, lost, gained),
           check_totals(cell_totals, gene_totals, sums)]
    res["A5"] = {"umis": sums, "violations": {name: int(n) for b in bad[:2] for name, n in b.invariant.value_counts().items()}
                 | ({"totals_differ": len(bad[2])} if len(bad[2]) else {}),
                 "passed": not any(len(b) for b in bad), "groups": _listing(bad[0], barcodes, gene_ids),
                 "entries": _listing(bad[1], barcodes, gene_ids), "totals": _listing(bad[2], barcodes, gene_ids)}
    if not res["A5"]["passed"]:
        return stop("A5")
    lap("checks")

    manifest = {"library": lib} | {k: v for k, v in meta.items() if k != "settings"}
    matrices.save(matrix_dir, m0, lost, gained, cr_dir, manifest)
    cell_totals.to_parquet(matrix_dir / "cell_totals.parquet", index=False)
    gene_totals.to_parquet(matrix_dir / "gene_totals.parquet", index=False)
    key = vs.key.to_numpy()
    vs.drop(columns="key").assign(cell=(key >> np.uint64(40)).astype(np.int32), umi=((key >> np.uint64(16)) & np.uint64(0xFFFFFF))
                                  .astype(np.uint32), gene=(key & np.uint64(0xFFFF)).astype(np.int32))[
        ["cell", "umi", "gene", "n_reads", "n_winner_reads", "counted_by_rule", "counted_by_xf"]].to_parquet(
        recount_dir / "rule_vs_xf.parquet", index=False)
    del tally, m0, lost, gained, cr, rec, cell_totals, gene_totals, multi, violations, bad
    lap("write")

    a2 = check_a2(*winner_groups(part_dir, refs), *molecule_info(h5, cells, gene_ids))
    res["A2"] = _jsonable(a2, barcodes, gene_ids)
    if not a2["passed"]:
        res["resolution"] = _listing(groups_of_units(part_dir, refs, a2["left_over"].key.to_numpy()[:EXAMPLES] >> np.uint64(16)),
                                     barcodes, gene_ids)
        return stop("A2", "resolution")
    del a2
    lap("A2")

    if a4_refs is None:
        expected = {tag: matrices.load(lib, tag, matrix_dir) for tag in a4_bams}
    else:
        expected = rule_matrices(part_dir, list(a4_refs), tags, (len(cells), len(genes)))
    jobs = [(tag, path, cr_dir, expected[tag]) for tag, path in a4_bams.items()]
    # the first BAM runs alone, the others A4_PARALLEL at once (design 3.2, Run time and memory)
    done = map_forked(check_a4, jobs[:1], 1) + map_forked(check_a4, jobs[1:], A4_PARALLEL)
    res["A4"] = {"bams": len(done), "refs": "all" if a4_refs is None else list(a4_refs),
                 "passed": bool(done) and all(a4["passed"] for _, a4, _ in done),
                 "settings": {tag: _jsonable(a4, barcodes, gene_ids) for tag, a4, _ in done}}
    res["A6"] = {tag: _jsonable(a6, barcodes, gene_ids) for tag, _, a6 in done}
    lap("A4")
    if not res["A4"]["passed"]:
        return stop("A4")
    res["passed"] = True
    return save()
