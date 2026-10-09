from .genes import genes_at

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
HEXAMERS = ("AATAAA", "ATTAAA")
PA_COLUMNS = ["qname", "ref", "strand", "site", "pa", "ts", "cb", "ub", "gx", "re", "xf", "flag", "dup",
              "mapq", "aligned_len", "spliced", "down20", "up40", "a20", "hexamer", "genes"]


def priming_site(a):
    """Genomic boundary at the read's 3' end in transcript sense (last cDNA base is just upstream of it)."""
    return a.reference_start if a.is_reverse else a.reference_end


def last_aligned_base(a):
    return a.reference_start if a.is_reverse else a.reference_end - 1


def sense_context(fa, ref, site, is_reverse, down=20, up=40):
    """Genome after (down) and before (up) the site, transcript sense, uppercase."""
    L = fa.get_reference_length(ref)
    if is_reverse:
        d = fa.fetch(ref, max(0, site - down), site).upper().translate(COMP)[::-1]
        u = fa.fetch(ref, site, min(L, site + up)).upper().translate(COMP)[::-1]
    else:
        d = fa.fetch(ref, site, min(L, site + down)).upper()
        u = fa.fetch(ref, max(0, site - up), site).upper()
    return d, u


def context_from_seq(seq, site, is_reverse, down=20, up=40):
    """sense_context on a preloaded uppercase chromosome sequence (seq = fa.fetch(ref).upper())."""
    if is_reverse:
        return seq[max(0, site - down):site].translate(COMP)[::-1], seq[site:site + up].translate(COMP)[::-1]
    return seq[site:site + down], seq[max(0, site - up):site]


def has_hexamer(up40, lo=5, hi=35):
    """AATAAA or ATTAAA fully inside the window lo..hi nt upstream of the site."""
    n = len(up40)
    win = up40[max(0, n - hi):max(0, n - lo)]
    return any(h in win for h in HEXAMERS)


def classify(a20, templated_min=14, nontemplated_max=8):
    if a20 >= templated_min:
        return "templated"
    if a20 <= nontemplated_max:
        return "nontemplated"
    return "ambiguous"


def _tag(a, t, default=None):
    return a.get_tag(t) if a.has_tag(t) else default


def pa_row(a, fa, segments):
    site = priming_site(a)
    down, up = sense_context(fa, a.reference_name, site, a.is_reverse)
    return [a.query_name, a.reference_name, "-" if a.is_reverse else "+", site, a.get_tag("pa"), _tag(a, "ts", 0),
            _tag(a, "CB"), _tag(a, "UB"), _tag(a, "GX"), _tag(a, "RE"), _tag(a, "xf"), a.flag, a.is_duplicate,
            a.mapping_quality, a.query_alignment_length, "N" in a.cigarstring, down, up, down.count("A"),
            has_hexamer(up), genes_at(segments, a.reference_name, not a.is_reverse, last_aligned_base(a))]
