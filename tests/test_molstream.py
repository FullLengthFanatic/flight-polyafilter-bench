"""The read-group pass of stage 23 (pafbench.molstream) on the synthetic 10x BAM, against the designed tables of
tests/synth.build_10x, flags.flag_reference, stream.scan_reference and plain counts of the BAM."""
import numpy as np

from pafbench import molstream, stream
from pafbench.intervals import blocks_hit, flatten


def _kernel_bits(bs, be, br, n_rec, pieces):
    ps, pe, offs = stream._pack(pieces)
    mask = np.zeros(n_rec, np.uint32)
    molstream._mask_hits(bs, be, br, ps, pe, offs, mask)
    return mask


def test_mask_kernel_edges_by_hand():
    # the blocks of test_stream.test_hit_kernel_edges_by_hand against the pieces [10, 20) and [30, 40) as tag 1; tag 0 has no
    # piece and tag 2 the piece [0, 6), which only the first block touches. Records 0 to 5 have one block each.
    bs = np.array([5, 8, 19, 20, 25, 40], np.int64)
    be = np.array([10, 11, 20, 30, 31, 50], np.int64)
    br = np.arange(6, dtype=np.int64)
    none = (np.empty(0, np.int64), np.empty(0, np.int64))
    mask = _kernel_bits(bs, be, br, 6, [none, (np.array([10, 30]), np.array([20, 40])), (np.array([0]), np.array([6]))])
    assert mask.dtype == np.uint32 and mask.tolist() == [4, 2, 2, 0, 2, 0]
    # two blocks of one record: either one sets the bit, and a bit that is set stays
    mask = _kernel_bits(np.array([5, 25]), np.array([10, 31]), np.array([0, 0]), 1, [(np.array([10, 30]), np.array([20, 40]))])
    assert mask.tolist() == [1]
    assert _kernel_bits(bs[:0], be[:0], br[:0], 2, [none]).tolist() == [0, 0]


def test_mask_kernel_matches_blocks_hit_random():
    rng = np.random.default_rng(3)
    n_rec, n_tags = 400, 32
    starts = np.sort(rng.integers(0, 5000, n_rec))
    blk_s, blk_e, blk_r = [], [], []
    for r, s in enumerate(starts):
        pos = int(s)
        for _ in range(rng.integers(1, 4)):
            ln = int(rng.integers(1, 60))
            blk_s.append(pos)
            blk_e.append(pos + ln)
            blk_r.append(r)
            pos += ln + int(rng.integers(1, 300))
    blk_s, blk_e, blk_r = (np.array(x, np.int64) for x in (blk_s, blk_e, blk_r))
    pieces = []
    for t in range(n_tags):
        k = int(rng.integers(0, 40))
        s = rng.integers(0, 5600, k)
        pieces.append(flatten(s, s + rng.integers(1, 80, k)))
    order = np.argsort(blk_s, kind="stable")
    got = _kernel_bits(blk_s[order], blk_e[order], blk_r[order], n_rec, pieces)
    want = np.zeros(n_rec, np.int64)
    for t, (s, e) in enumerate(pieces):
        want[np.unique(blk_r[blocks_hit(blk_s, blk_e, s, e)])] |= 1 << t
    assert got.tolist() == want.tolist()
    # all 32 bits are in use, the highest included, and no tag hits every record
    hits = [int(((want >> t) & 1).sum()) for t in range(n_tags)]
    assert all(0 < h < n_rec for h, (s, _) in zip(hits, pieces) if s.size) and hits[31] > 0
    assert sum(s.size > 0 for s, _ in pieces) >= 30
