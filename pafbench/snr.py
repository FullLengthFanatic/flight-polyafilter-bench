import numpy as np
from numba import njit

_IS_A = np.zeros(256, dtype=np.bool_)
_IS_A[ord("A")] = True
_IS_A[ord("a")] = True
_IS_T = np.zeros(256, dtype=np.bool_)
_IS_T[ord("T")] = True
_IS_T[ord("t")] = True


def base_mask(seq, strd):
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return (_IS_A if strd else _IS_T)[arr]


@njit(cache=True)
def _scan(is_base, mism):
    # Line-by-line port of BAMfilter.findSNRpieces with minSNRlen = 1 (every candidate is saved).
    n = is_base.shape[0]
    out_first = np.empty(n // 2 + 2, np.int64)
    out_last = np.empty(n // 2 + 2, np.int64)
    k = 0
    cap = mism + 3
    bs = np.empty(cap, np.int64)
    be = np.empty(cap, np.int64)
    nb = 0
    n_mism = 0
    blocked = False
    last_saved = True
    start = 0
    for i in range(n):
        if is_base[i]:
            if not blocked:
                start = i
                blocked = True
        else:
            if blocked:
                bs[nb] = start
                be[nb] = i
                nb += 1
                last_saved = False
                blocked = False
            if nb > 0:
                n_mism += 1
                if n_mism > mism:
                    if not last_saved:
                        out_first[k] = bs[0]
                        out_last[k] = be[nb - 1]
                        k += 1
                        last_saved = True
                    if nb > 1:
                        n_mism -= bs[1] - be[0]
                    else:
                        n_mism = 0
                    for j in range(nb - 1):
                        bs[j] = bs[j + 1]
                        be[j] = be[j + 1]
                    nb -= 1
    if blocked:
        bs[nb] = start
        be[nb] = n
        nb += 1
        last_saved = False
    if nb > 0 and not last_saved:
        out_first[k] = bs[0]
        out_last[k] = be[nb - 1]
        k += 1
    return out_first[:k].copy(), out_last[:k].copy()


def scan_snrs(seq, strd, mism):
    """Stretches of A (plus strand) or T (minus strand) saved by findSNRpieces at minSNRlen 1; 0-based, last exclusive."""
    return _scan(base_mask(seq, strd), mism)


def snr_pieces(first, last, min_len, covlen, strd, ref_len):
    """COVLEN windows upstream (in transcript sense) of stretches of length >= min_len, unflattened."""
    keep = (last - first) >= min_len
    f, l = first[keep], last[keep]
    if strd:
        s = np.maximum(0, f - covlen)
        e = f
    else:
        s = l
        e = np.minimum(ref_len, l + covlen)
    ok = s != e
    return s[ok], e[ok]
