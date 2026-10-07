from collections import defaultdict

import numpy as np
import pysam

from .vendor import polyafilter_modules

_ACGT = frozenset("ACGT")


def _attr(field, key):
    for part in field.split(";"):
        part = part.strip()
        if part.startswith(key + " "):
            return part.split(" ", 1)[1].strip('"')
    raise KeyError(key)


def load_transcripts(gtf):
    """{(ref, strd): [(start0, end0, exons sorted reverse=strd)]} from GTF transcript and exon lines."""
    tx, exons = {}, defaultdict(list)
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] not in ("transcript", "exon"):
                continue
            tid = _attr(f[8], "transcript_id")
            if f[2] == "transcript":
                tx[tid] = (f[0], f[6] == "+", int(f[3]) - 1, int(f[4]))
            else:
                exons[tid].append((int(f[3]) - 1, int(f[4])))
    out = defaultdict(list)
    for tid, (ref, strd, s, e) in tx.items():
        out[(ref, strd)].append((s, e, sorted(exons[tid], reverse=strd)))
    return dict(out)


def coverage_prefix(bam_path, ref):
    """Per-strand prefix sums of 'position has an aligned A/C/G/T base', as pysam count_coverage with a callable read_callback sees it.

    Every record counts (secondary, duplicate, QC-fail); records without SEQ add nothing; N calls are not counted.
    Strand True collects forward records (polyAfilter's read_callback r.is_reverse != strd).
    """
    with pysam.AlignmentFile(str(bam_path)) as bam:
        L = bam.get_reference_length(ref)
        starts = {True: [], False: []}
        ends = {True: [], False: []}
        not_acgt = {True: [], False: []}
        for a in bam.fetch(ref):
            seq = a.query_sequence
            if seq is None:
                continue
            strd = not a.is_reverse
            for s, e in a.get_blocks():
                starts[strd].append(s)
                ends[strd].append(e)
            if not set(seq) <= _ACGT:
                for q, r in a.get_aligned_pairs(matches_only=True):
                    if seq[q] not in _ACGT:
                        not_acgt[strd].append(r)
    out = {}
    for strd in (True, False):
        diff = np.zeros(L + 1, np.int64)
        np.add.at(diff, np.asarray(starts[strd], np.int64), 1)
        np.add.at(diff, np.asarray(ends[strd], np.int64), -1)
        depth = np.cumsum(diff[:-1])
        if not_acgt[strd]:
            np.add.at(depth, np.asarray(not_acgt[strd], np.int64), -1)
        prefix = np.zeros(L + 1, np.int64)
        np.cumsum(depth > 0, out=prefix[1:])
        out[strd] = prefix
    return out


def expressed(tx_list, prefix, include_introns):
    """Transcripts with any covered base (whole span with introns, exons otherwise)."""
    out = []
    for s, e, exons in tx_list:
        if include_introns:
            hit = prefix[e] - prefix[s] > 0
        else:
            hit = any(prefix[xe] - prefix[xs] > 0 for xs, xe in exons)
        if hit:
            out.append((s, e, exons))
    return out


def start_pieces(expressed_tx, covlen, include_introns, strd):
    """Flattened unprotected body for one (ref, strand), using polyAfilter's own end and overlap functions."""
    _, RNA, SNR = polyafilter_modules()
    each_start, all_ends = [], []
    for s, e, exons in expressed_tx:
        if include_introns:
            if e - s <= covlen:
                ends, starts = [(s, e)], []
            else:
                ends = RNA.getTransEnding(exons, covlen, strd)
                ends.append((e - covlen, e) if strd else (s, s + covlen))
                ends = SNR.flattenIntervals(ends)
                starts = SNR.removeOverlaps([(s, e)], ends)
        else:
            starts, ends = RNA.getTransStartEnding(exons, covlen, strd)
        all_ends.extend(ends)
        if starts != []:
            each_start.append(tuple(starts))
    all_ends = SNR.flattenIntervals(all_ends)
    flat = []
    for st in each_start:
        flat.extend(SNR.removeOverlaps(list(st), all_ends))
    flat = SNR.flattenIntervals(flat)
    return np.array([a for a, _ in flat], np.int64), np.array([b for _, b in flat], np.int64)
