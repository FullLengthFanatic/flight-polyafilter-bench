import random

import numpy as np

from pafbench.intervals import blocks_hit, flatten, intersect
from pafbench.vendor import polyafilter_modules

_, _, SNR = polyafilter_modules()


def _random_intervals(rng, n, span=2000, maxlen=60):
    out = []
    for _ in range(n):
        s = rng.randrange(0, span)
        out.append((s, s + rng.randrange(1, maxlen)))
    return out


def test_flatten_matches_original():
    rng = random.Random(1)
    for _ in range(300):
        iv = _random_intervals(rng, rng.randrange(0, 50))
        want = SNR.flattenIntervals(list(iv))
        s, e = flatten([a for a, _ in iv], [b for _, b in iv])
        assert list(zip(s.tolist(), e.tolist())) == want


def test_flatten_merges_touching():
    s, e = flatten([0, 10], [10, 20])
    assert (s.tolist(), e.tolist()) == ([0], [20])


def test_intersect_matches_original():
    rng = random.Random(2)
    for _ in range(300):
        a = SNR.flattenIntervals(_random_intervals(rng, rng.randrange(0, 30)))
        b = SNR.flattenIntervals(_random_intervals(rng, rng.randrange(0, 30)))
        want = sorted(SNR.getOverlaps(list(a), list(b)))
        s, e = intersect(
            np.array([x for x, _ in a], np.int64), np.array([y for _, y in a], np.int64),
            np.array([x for x, _ in b], np.int64), np.array([y for _, y in b], np.int64),
        )
        assert list(zip(s.tolist(), e.tolist())) == want


def test_blocks_hit_matches_original():
    rng = random.Random(3)
    for _ in range(300):
        pieces = SNR.flattenIntervals(_random_intervals(rng, rng.randrange(0, 20)))
        ps = np.array([x for x, _ in pieces], np.int64)
        pe = np.array([y for _, y in pieces], np.int64)
        blocks = _random_intervals(rng, 40)
        got = blocks_hit(np.array([x for x, _ in blocks]), np.array([y for _, y in blocks]), ps, pe)
        want = [any(SNR.getOverlaps([blk], [p]) for p in pieces) for blk in blocks]
        assert got.tolist() == want


def test_adjacent_is_not_a_hit():
    assert blocks_hit(np.array([0]), np.array([10]), np.array([10]), np.array([20])).tolist() == [False]
