from .sites import COMP


def sense_3p_clip(a):
    """Soft-clipped bases after the aligned 3' end, in transcript sense (FLNC reads are transcript-oriented)."""
    cig, seq = a.cigartuples, a.query_sequence
    if a.is_reverse:
        n = cig[0][1] if cig[0][0] == 4 else 0
        return seq[:n].translate(COMP)[::-1]
    n = cig[-1][1] if cig[-1][0] == 4 else 0
    return seq[len(seq) - n:] if n else ""


def leading_a(s):
    return len(s) - len(s.lstrip("A"))


def a_run_site(a):
    """Genomic boundary where the read's terminal A run starts (aligned A's included), and how many aligned A's it has."""
    seq = a.query_sequence
    pairs = a.get_aligned_pairs(matches_only=True)
    k, ref_pos = 0, None
    if a.is_reverse:
        for q, r in pairs:
            if seq[q] != "T":
                break
            k, ref_pos = k + 1, r
        return (ref_pos + 1 if k else a.reference_start), k
    for q, r in reversed(pairs):
        if seq[q] != "A":
            break
        k, ref_pos = k + 1, r
    return (ref_pos if k else a.reference_end), k
