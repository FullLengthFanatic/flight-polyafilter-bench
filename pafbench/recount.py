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

from .molstream import GROUP_MAX

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
