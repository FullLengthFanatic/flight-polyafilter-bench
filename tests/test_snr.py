import random

from pafbench.snr import scan_snrs, snr_pieces
from pafbench.vendor import polyafilter_modules

BAMfilter, _, _ = polyafilter_modules()
ALPHABET = "AAAAAAATTTTTTCGNat"


def _seq(rng, n):
    core = "".join(rng.choice(ALPHABET) for _ in range(n))
    return rng.choice(["", "AAAA", "TTT"]) + core + rng.choice(["", "AAAAA", "TTTT"])


def test_matches_original_on_random_sequences():
    rng = random.Random(11)
    for _ in range(200):
        seq = _seq(rng, rng.randrange(1, 400))
        for strd in (True, False):
            for mism in (0, 1, 2):
                first, last = scan_snrs(seq, strd, mism)
                for min_len in (1, 2, 3, 5, 8, 12):
                    for covlen in (1, 7, 50):
                        want = BAMfilter.findSNRpieces(covlen, seq, strd, mism=mism, minSNRlen=min_len)
                        s, e = snr_pieces(first, last, min_len, covlen, strd, len(seq))
                        assert list(zip(s.tolist(), e.tolist())) == want, (seq, strd, mism, min_len, covlen)


def test_lowercase_counts_as_base():
    first, last = scan_snrs("CCaaaaaCC", True, 0)
    assert (first.tolist(), last.tolist()) == ([2], [7])


def test_contig_edges_give_no_zero_length_window():
    first, last = scan_snrs("AAAAACCCC", True, 0)
    s, e = snr_pieces(first, last, 3, 10, True, 9)
    assert s.size == 0
    first, last = scan_snrs("CCCCTTTTT", False, 0)
    s, e = snr_pieces(first, last, 3, 10, False, 9)
    assert s.size == 0
