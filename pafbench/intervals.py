import numpy as np


def flatten(starts, ends):
    """Merge intervals as polyAfilter's flattenIntervals does (touching intervals merge)."""
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    if s.size == 0:
        return s, e
    if np.any(s >= e):
        raise ValueError("interval with start >= end")
    order = np.argsort(s, kind="stable")
    s, e = s[order], e[order]
    run_end = np.maximum.accumulate(e)
    new = np.ones(s.size, dtype=bool)
    new[1:] = s[1:] > run_end[:-1]
    idx = np.flatnonzero(new)
    return s[idx], np.maximum.reduceat(e, idx)


def intersect(a_s, a_e, b_s, b_e):
    """Positive-length overlaps of two flattened interval sets, as polyAfilter's getOverlaps."""
    a_s, a_e = np.asarray(a_s, np.int64), np.asarray(a_e, np.int64)
    b_s, b_e = np.asarray(b_s, np.int64), np.asarray(b_e, np.int64)
    if a_s.size == 0 or b_s.size == 0:
        z = np.empty(0, np.int64)
        return z, z.copy()
    lo = np.searchsorted(a_e, b_s, side="right")
    hi = np.searchsorted(a_s, b_e, side="left")
    counts = np.maximum(hi - lo, 0)
    b_idx = np.repeat(np.arange(b_s.size), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    a_idx = np.repeat(lo, counts) + offsets
    s = np.maximum(a_s[a_idx], b_s[b_idx])
    e = np.minimum(a_e[a_idx], b_e[b_idx])
    keep = s < e
    s, e = s[keep], e[keep]
    order = np.argsort(s, kind="stable")
    return s[order], e[order]


def blocks_hit(block_s, block_e, piece_s, piece_e):
    """True for each block overlapping at least one base of any piece (pieces sorted and disjoint)."""
    bs = np.asarray(block_s, np.int64)
    be = np.asarray(block_e, np.int64)
    hit = np.zeros(bs.size, dtype=bool)
    if piece_s.size == 0 or bs.size == 0:
        return hit
    idx = np.searchsorted(piece_e, bs, side="right")
    ok = idx < piece_s.size
    hit[ok] = piece_s[idx[ok]] < be[ok]
    return hit
