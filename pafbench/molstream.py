"""Read groups of the called cells and their removed reads per setting, in two streaming passes per reference (stage 23).

A sibling of stream.scan_reference (design of 2026-10-12, section 3.1): the pieces that define removal come from stream, the
chunk buffers, the kernel and the part writer are its own. The kernel gives each record one bit per setting.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _mask_hits(bs, be, br, ps, pe, offs, mask):
    """Blocks of one strand sorted by start against each tag's sorted disjoint pieces, the test of stream._scan_hits.

    Bit t of mask[r] is set when a block of record r touches a piece of tag t.
    """
    nb = bs.size
    if nb == 0:
        return
    for t in range(offs.size - 1):
        lo, hi = offs[t], offs[t + 1]
        if lo == hi:
            continue
        bit = np.uint32(1) << np.uint32(t)
        j = lo + np.searchsorted(pe[lo:hi], bs[0], side="right")
        for k in range(nb):
            s = bs[k]
            while j < hi and pe[j] <= s:
                j += 1
            if j == hi:
                break
            if ps[j] < be[k]:
                mask[br[k]] |= bit
