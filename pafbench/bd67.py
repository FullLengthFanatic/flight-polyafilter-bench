import hashlib
from array import array

import numpy as np
import pandas as pd
import pysam

from .config import LABEL_CUTOFFS
from .genes import genes_at
from .labels import label_molecules
from .regions import regions_at
from .sites import COMP, context_from_seq, has_hexamer, priming_site

OUTCOMES = ("accepted", "short_run", "short_after", "both")
LABELS = ("templated", "nontemplated", "ambiguous")
SITE_COLUMNS = ["ref", "strand", "site", "n_reads", "n_accepted", "a20", "label", "a_band", "hexamer", "region", "gene"]
_ALIGNED, _QUERY, _REF = (0, 7, 8), (1, 4), (2, 3)


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


def inward_pairs(a):
    """(query, ref) pairs of the aligned bases from the transcript 3' end inward: get_aligned_pairs(matches_only=True)
    reversed on forward reads, as-is on reverse reads, walked lazily from the cigar."""
    cig = a.cigartuples
    if a.is_reverse:
        q, r, step = 0, a.reference_start, 1
    else:
        q, r, step, cig = a.query_length - 1, a.reference_end - 1, -1, reversed(cig)
    for op, n in cig:
        if op in _ALIGNED:
            for k in range(n):
                yield q + step * k, r + step * k
            q, r = q + step * n, r + step * n
        elif op in _QUERY:
            q += step * n
        elif op in _REF:
            r += step * n


def terminal_a_run(a, tolerance=1):
    """Terminal A run around the aligned 3' end in transcript sense, with up to `tolerance` interior non-A bases.

    Grows rightwards over the 3' clip, then leftwards over the aligned bases, from one shared non-A budget; a non-A is
    taken only between A's. Returns (site, aligned_a, clip_a, after_len): the genomic boundary where the run starts
    (priming_site when the run has no aligned base), the A's in its aligned and clipped parts, and the clipped bases
    after it.
    """
    seq = a.query_sequence.translate(COMP) if a.is_reverse else a.query_sequence
    clip, pairs, left = sense_3p_clip(a), inward_pairs(a), []

    def aligned(t):
        """t-th aligned sense base from the 3' end as (base, ref), None past the 5' end."""
        while len(left) <= t:
            p = next(pairs, None)
            if p is None:
                return None
            left.append((seq[p[0]], p[1]))
        return left[t]

    budget, j, n = tolerance, 0, len(clip)
    while j < n:
        if clip[j] != "A":
            if not (budget and j + 1 < n and clip[j + 1] == "A" and (j or aligned(0)[0] == "A")):
                break
            budget -= 1
        j += 1
    t = aligned_a = 0
    while (cur := aligned(t)) is not None:
        if cur[0] == "A":
            aligned_a += 1
        else:
            nxt = aligned(t + 1)
            if not (budget and nxt is not None and nxt[0] == "A" and (t or j)):
                break
            budget -= 1
        t += 1
    if t == 0:
        site = priming_site(a)
    else:
        site = left[t - 1][1] + 1 if a.is_reverse else left[t - 1][1]
    return site, aligned_a, clip[:j].count("A"), n - j


def accept_long_read(aligned_a, clip_a, after_len, min_run=10, min_after=30):
    return aligned_a + clip_a >= min_run and after_len >= min_after


def read_outcome(aligned_a, clip_a, after_len, min_run=10, min_after=30):
    """accepted, short_run (run too short), short_after (too little clip after the run) or both."""
    if aligned_a + clip_a >= min_run:
        return "accepted" if after_len >= min_after else "short_after"
    return "short_run" if after_len >= min_after else "both"


def _digest(name):
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little")


def scan_long_reads(bam_path, ref, min_mapq=20, min_after=30):
    """Terminal A run (tolerance 1) of every primary record with MAPQ >= min_mapq on one reference.

    Returns the unique sites (strand, site, n_reads, n_accepted), record and read outcome counts, and a 64-bit
    blake2b digest per read name so that duplicate names can be counted across references.
    """
    strand, site, ok, digests = array("b"), array("q"), array("b"), array("Q")
    counts = {"records": 0, "secondary_supplementary": 0, "low_mapq": 0, "used": 0} | dict.fromkeys(OUTCOMES, 0)
    with pysam.AlignmentFile(str(bam_path)) as bam:
        for a in bam.fetch(ref):
            counts["records"] += 1
            if a.is_secondary or a.is_supplementary:
                counts["secondary_supplementary"] += 1
                continue
            if a.mapping_quality < min_mapq:
                counts["low_mapq"] += 1
                continue
            s, aligned_a, clip_a, after_len = terminal_a_run(a)
            outcome = read_outcome(aligned_a, clip_a, after_len, min_after=min_after)
            counts[outcome] += 1
            strand.append(a.is_reverse)
            site.append(s)
            ok.append(outcome == "accepted")
            digests.append(_digest(a.query_name))
    counts["used"] = len(site)
    reads = pd.DataFrame({"strand": np.where(np.asarray(strand, bool), "-", "+"), "site": np.asarray(site, np.int64),
                          "accepted": np.asarray(ok, np.int64)})
    sites = reads.groupby(["strand", "site"]).accepted.agg(n_reads="size", n_accepted="sum").reset_index()
    return sites, counts, np.asarray(digests, np.uint64)


def count_duplicates(digests):
    return int(len(digests) - np.unique(digests).size)


def annotate_sites(sites, ref, seq, region_segments, gene_segments, cutoffs=LABEL_CUTOFFS):
    """Adds a20, label, a_band and hexamer from the genome around each site, and region and first gene id at the last
    cDNA base (site - 1 on '+', site on '-'); seq is the uppercase sequence of ref. Returns SITE_COLUMNS."""
    rev = (sites.strand == "-").to_numpy()
    pos = sites.site.to_numpy(np.int64)
    ctx = [context_from_seq(seq, int(s), bool(r)) for s, r in zip(pos, rev)]
    last = np.where(rev, pos, pos - 1)
    region = np.empty(len(sites), object)
    for strd in (True, False):
        m = rev != strd
        region[m] = regions_at(region_segments, ref, strd, last[m])
    gene = [genes_at(gene_segments, ref, not r, int(p)).split(";")[0] for p, r in zip(last, rev)]
    out = sites.assign(ref=ref, a20=[d.count("A") for d, _ in ctx], hexamer=[has_hexamer(u) for _, u in ctx],
                       region=region, gene=gene)
    return label_molecules(out, cutoffs)[SITE_COLUMNS]


def hexamer_summary(sites, cutoffs):
    """Sites with an accepted read: n and hexamer rate by label at these cut-offs, ratio = rate(nontemplated) /
    rate(templated) (None when the templated rate is 0 or a class is empty), passes = ratio >= 3."""
    acc = label_molecules(sites[sites.n_accepted >= 1], cutoffs)
    g = acc.groupby("label").hexamer
    n, rate = g.size(), g.mean()
    rates = {k: float(rate[k]) if k in rate.index else None for k in LABELS}
    ratio = rates["nontemplated"] / rates["templated"] if rates["templated"] and rates["nontemplated"] is not None else None
    return {"cutoffs": dict(cutoffs), "n": {k: int(n.get(k, 0)) for k in LABELS}, "hexamer_rate": rates,
            "ratio": ratio, "passes": ratio is not None and ratio >= 3}


def nearest_labels(mol, sites, max_dist=10):
    """Label of the nearest site with an accepted read on the same ref and strand within max_dist nt of each
    molecule's site ('none' if there is none); at equal distance the lower site wins."""
    out = np.full(len(mol), "none", object)
    groups = dict(list(sites[sites.n_accepted >= 1].groupby(["ref", "strand"])))
    msite = mol.site.to_numpy(np.int64)
    for key, idx in mol.groupby(["ref", "strand"]).indices.items():
        g = groups.get(key)
        if g is None:
            continue
        g = g.sort_values("site")
        s, lab, x = g.site.to_numpy(np.int64), g.label.to_numpy(object), msite[idx]
        k = np.searchsorted(s, x)
        lo, hi = np.maximum(k - 1, 0), np.minimum(k, s.size - 1)
        dlo = np.where(k > 0, x - s[lo], max_dist + 1)
        dhi = np.where(k < s.size, s[hi] - x, max_dist + 1)
        hit = np.minimum(dlo, dhi) <= max_dist
        out[idx[hit]] = lab[np.where(dlo <= dhi, lo, hi)[hit]]
    return out


def compare_labels(mol, sites, max_dist=10):
    """Molecules (ref, strand, site, label) against long-read sites: the table (label_1179, label_long, n), the
    share of molecules matched overall and per 1179 label, and label agreement among matched templated and
    nontemplated molecules."""
    long, lab = nearest_labels(mol, sites, max_dist), mol.label.to_numpy(object)
    table = (pd.DataFrame({"label_1179": lab, "label_long": long}).value_counts().rename("n").reset_index()
             .sort_values(["label_1179", "label_long"], ignore_index=True))
    matched = long != "none"

    def share(num, den):
        return float(num.sum() / den.sum()) if den.sum() else None

    agreement = {k: share(matched & (lab == k) & (long == k), matched & (lab == k)) for k in ("templated", "nontemplated")}
    both = matched & np.isin(lab, ["templated", "nontemplated"])
    agreement["pooled"] = share(both & (long == lab), both)
    return table, {"molecules": len(mol), "matched": int(matched.sum()),
                   "share_matched": share(matched, np.ones(len(mol), bool)),
                   "share_matched_by_label": {k: share(matched & (lab == k), lab == k) for k in LABELS},
                   "agreement": agreement}
