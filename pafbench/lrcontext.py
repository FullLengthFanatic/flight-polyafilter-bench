"""Stage 30: context of long-read 3' ends (design of 2026-10-12, section 3.8). Each accepted read is classed by its own
alignment against the transcripts of the GTF.

All coordinates are in transcript sense: a minus-strand interval [a, b) is mirrored to [-b, -a), so that on both strands
a transcript runs from low to high, exons and introns are sorted, and the 3' end is the high end.
"""
import bisect
from array import array
from collections import defaultdict, namedtuple
from contextlib import ExitStack

import numpy as np
import pandas as pd
import pysam

from .bd67 import accept_long_read, compare_labels, nearest_labels, terminal_a_run
from .config import FIVE_PRIME_TOL, LONG_END_TOL, MIN_INTRON, NEAR_INTRON_TOL, NULL_SHIFT
from .tecap import TAG, site_key

# The order is the tie order of the top class of a site: the eight classes with a model, then the three no_model reasons.
CLASSES = ("annotated_end", "3UTR", "CDS", "5UTR", "ncexon", "intron_unspliced", "intron_after_junction", "downstream",
           "no_model_novel_junction", "no_model_antisense", "no_model_intergenic")
NOVEL_JUNCTION, ANTISENSE, INTERGENIC = CLASSES[8:]
# Why the last intron of a read is in no transcript of its strand; the first that applies, and the tie order of the main
# sub-cause of a site.
SUB_CAUSES = ("near_annotated_intron", "short_intron", "opposite_strand_intron", "sites_not_joined", "donor_only",
              "acceptor_only", "neither_site")
# cds: (first base, end) of the CDS or None; start_codon: its first base or None
Transcript = namedtuple("Transcript", "tid start end exons cds start_codon end_nf")
# matched: introns of the read that the transcript has as a run; chain_novel: 0, or that number for a read placed by its
# 3' part. sub_cause is set for no_model_novel_junction only.
ReadContext = namedtuple("ReadContext", "klass transcript matched chain_novel starts_outside retained_after_junction end_nf "
                                        "complete_5p covers_start other_class sub_cause", defaults=(False,) * 6 + ("",))
# "placed" counts the reads placed by their 3' part (chain_novel) under their class, "strict" as no_model_novel_junction
VERSIONS = ("placed", "strict")
# one row per accepted read: klass and sub_cause are positions in CLASSES and SUB_CAUSES (sub_cause -1 without one), length
# the query length of the record, no_splice says that its CIGAR has no N
READ_COLUMNS = ["minus", "site", "klass", "matched", "chain_novel", "starts_outside", "retained_after_junction", "end_nf",
                "complete_5p", "covers_start", "other_class", "sub_cause", "length", "no_splice"]
_READ_TYPES = ("b", "q", "b", "h", "h", "b", "b", "b", "b", "b", "b", "b", "i", "b")
_READ_BOOL = ["minus"] + READ_COLUMNS[5:11] + ["no_splice"]
_KEY = ["ref", "strand", "site"]
_COUNTS = [f"n_{c}" for c in CLASSES] + ["n_starts_outside", "n_chain_novel", "n_chain_novel_suffix1",
                                         "n_retained_after_junction", "n_end_NF"]
_TOP = ["context", "context_without_chain_novel", "top_share", "tied", "share_spliced", "share_5p_complete", "median_length"]
# reduce_sites: the columns that stage 30 writes per site, then those it needs for its totals
SITE_COLUMNS = _KEY + ["n_accepted"] + _COUNTS + _TOP + ["tied_without_chain_novel", "top_share_without_chain_novel",
                                                         "chain_novel_only", "sub_cause", "n_no_splice"]
BY_CLASS_COLUMNS = _KEY + ["klass", "chain_novel", "sub_cause", "reads", "spliced", "complete_5p", "covers_start", "other_class"]
# isoform_context_sites.parquet (design 3.8, Outputs)
CONTEXT_COLUMNS = _KEY + ["label", "a_band", "hexamer", "region", "n_accepted"] + _COUNTS + _TOP
_BIN = 1 << 14
_FAR = 1 << 62


class Models:
    """Transcripts of one reference with the lookups of the class rule. rows: (transcript id, minus strand, (start, end),
    exons, CDS intervals, first base of the start codon or None, tagged mRNA_end_NF), in transcript sense.

    tx[k] is a Transcript; introns[minus][(donor, acceptor)] lists (k, i) for every transcript k of the strand whose
    i-th intron it is; donors[minus] and acceptors[minus] are the splice sites of the strand."""

    def __init__(self, rows):
        self.tx, self._chains = [], []
        self.introns = {False: defaultdict(list), True: defaultdict(list)}
        self._bins = {False: defaultdict(list), True: defaultdict(list)}
        for tid, minus, (start, end), exons, cds, start_codon, end_nf in rows:
            exons = sorted(exons)
            k = len(self.tx)
            self.tx.append(Transcript(tid, start, end, exons, (min(s for s, _ in cds), max(e for _, e in cds)) if cds else None,
                                      start_codon, end_nf))
            chain = [(exons[i][1], exons[i + 1][0]) for i in range(len(exons) - 1)]
            self._chains.append(chain)
            for i, intron in enumerate(chain):
                self.introns[minus][intron].append((k, i))
            for b in range(start // _BIN, (end - 1) // _BIN + 1):
                self._bins[minus][b].append(k)
        self.introns = {m: dict(v) for m, v in self.introns.items()}
        self.donors = {m: {d for d, _ in v} for m, v in self.introns.items()}
        self.acceptors = {m: {a for _, a in v} for m, v in self.introns.items()}
        self._sorted = {m: sorted(v) for m, v in self.introns.items()}

    def spans(self, minus, a, b):
        """Transcripts of the strand whose span overlaps [a, b)."""
        tx, bins = self.tx, self._bins[minus]
        return {k for bn in range(a // _BIN, (b - 1) // _BIN + 1) for k in bins.get(bn, ()) if tx[k].start < b and tx[k].end > a}

    def chain(self, minus, introns):
        """Transcripts of the strand in which these introns are a run of consecutive introns, coordinates exact."""
        n = len(introns)
        return [k for k, i in self.introns[minus].get(introns[0], ()) if self._chains[k][i:i + n] == introns]

    def suffix(self, minus, introns):
        """(n, transcripts): the longest suffix of the intron chain, the introns nearest the 3' end and at least the last
        one, that is a run of consecutive introns of a transcript of the strand, and the transcripts that hold it.
        (0, []) when no transcript has the last intron."""
        n, best, held = len(introns), 0, []
        for k, i in self.introns[minus].get(introns[-1], ()):
            chain, length = self._chains[k], 1
            while length < n and i - length >= 0 and chain[i - length] == introns[n - 1 - length]:
                length += 1
            if length > best:
                best, held = length, [k]
            elif length == best:
                held.append(k)
        return best, held

    def sub_cause(self, minus, intron):
        """The first of SUB_CAUSES that applies to an intron that no transcript of the strand has: an annotated intron of
        the strand within NEAR_INTRON_TOL nt at both ends; shorter than MIN_INTRON; an annotated intron of the other
        strand; both splice sites annotated on the strand; the donor only; the acceptor only; neither."""
        d, a = intron
        known = self._sorted[minus]
        i = bisect.bisect_left(known, (d - NEAR_INTRON_TOL,))
        while i < len(known) and known[i][0] <= d + NEAR_INTRON_TOL:
            if abs(known[i][1] - a) <= NEAR_INTRON_TOL:
                return SUB_CAUSES[0]
            i += 1
        if a - d < MIN_INTRON:
            return SUB_CAUSES[1]
        if (-a, -d) in self.introns[not minus]:
            return SUB_CAUSES[2]
        donor, acceptor = d in self.donors[minus], a in self.acceptors[minus]
        return SUB_CAUSES[3 if donor and acceptor else 4 if donor else 5 if acceptor else 6]


def load_models(gtf, refs=None):
    """{ref: Models} from the transcript, exon, CDS and start_codon lines of a GTF, transcripts in file order. With refs,
    those references in that order, each with a Models even when the GTF has no transcript on it.

    end_nf is the GENCODE tag mRNA_end_NF on the transcript line ("the mRNA end could not be confirmed",
    https://www.gencodegenes.org/pages/tags.html). The start codon is taken from the start_codon lines, not from the CDS.
    Our observation on the 2024-A genes.gtf (2026-10-14): 13,019 of the 109,751 transcripts with a CDS have no
    start_codon line, 12,958 of them tagged cds_start_NF; in the other 96,732 the start codon begins at the first CDS base."""
    tx, exons, cds, start = {}, defaultdict(list), defaultdict(list), defaultdict(list)
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] not in ("transcript", "exon", "CDS", "start_codon") or (refs is not None and f[0] not in refs):
                continue
            tid = f[8].split('transcript_id "', 1)[1].split('"', 1)[0]
            minus = f[6] == "-"
            iv = (-int(f[4]), 1 - int(f[3])) if minus else (int(f[3]) - 1, int(f[4]))
            if f[2] == "transcript":
                tx[tid] = (f[0], minus, iv, 'tag "mRNA_end_NF"' in f[8])
            elif f[2] == "exon":
                exons[tid].append(iv)
            elif f[2] == "CDS":
                cds[tid].append(iv)
            else:
                start[tid].append(iv[0])
    rows = {ref: [] for ref in refs} if refs is not None else defaultdict(list)
    for tid, (ref, minus, span, end_nf) in tx.items():
        rows[ref].append((tid, minus, span, exons[tid], cds.get(tid, []), min(start[tid]) if tid in start else None, end_nf))
    return {ref: Models(r) for ref, r in rows.items()}


def read_body(a, site):
    """Aligned blocks of a read up to its site (the start of the terminal A run, bd67.terminal_a_run), in transcript
    sense; [] when every aligned base belongs to the run. A CIGAR N separates two blocks and a deletion stays inside its
    block (our convention)."""
    blocks, pos, start = [], a.reference_start, None
    for op, n in a.cigartuples:
        if op in (0, 2, 7, 8):
            if start is None:
                start = pos
            pos += n
        elif op == 3:
            if start is not None:
                blocks.append((start, pos))
                start = None
            pos += n
    if start is not None:
        blocks.append((start, pos))
    if a.is_reverse:
        return [(-e, -max(s, site)) for s, e in reversed(blocks) if e > site]
    return [(s, min(e, site)) for s, e in blocks if s < site]


def _klass(t, p, spliced):
    """Class of the last cDNA base p relative to transcript t (the table of design 3.8, step 4)."""
    if abs(p - (t.end - 1)) <= LONG_END_TOL:
        return "annotated_end"
    if p >= t.end:
        return "downstream"
    i = bisect.bisect_right(t.exons, (p, _FAR)) - 1
    if i >= 0 and p < t.exons[i][1]:
        if t.cds is None:
            return "ncexon"
        return "5UTR" if p < t.cds[0] else "3UTR" if p >= t.cds[1] else "CDS"
    return "intron_after_junction" if spliced else "intron_unspliced"


def _exonic(exons, body):
    """Bases of the body that lie in the exons."""
    n = 0
    for s, e in body:
        i = max(bisect.bisect_right(exons, (s, _FAR)) - 1, 0)
        while i < len(exons) and exons[i][0] < e:
            n += max(0, min(e, exons[i][1]) - max(s, exons[i][0]))
            i += 1
    return n


def classify(models, minus, body):
    """ReadContext of a read from its body (read_body, not empty) on one strand of the reference of models.

    Compatible transcripts of a read with introns: those that have its introns as a run of consecutive introns; failing
    that, when the read overlaps a transcript span of its strand, those that hold the longest suffix of its intron chain
    (Models.suffix; chain_novel is set to its length). Of a read without introns: the transcripts whose span contains
    it, the 3' side with LONG_END_TOL; failing that, those whose span contains its last base (starts_outside); failing
    that, a read that overlaps a span is downstream of it, with no transcript. Among the compatible transcripts the
    choice is: 3' end within LONG_END_TOL of the last base first, then the largest exonic overlap with the whole body,
    then the nearest 3' end, then the transcript id. other_class says that another compatible transcript gives another
    class. retained_after_junction: the block after the last junction covers a whole intron of the chosen transcript.
    complete_5p (the 5' end within FIVE_PRIME_TOL of the transcript's) is never set for a chain_novel read, whose 5' part
    follows no transcript. covers_start: an aligned block holds the first base of the transcript's start codon."""
    tx = models.tx
    p, a0, b0 = body[-1][1] - 1, body[0][0], body[-1][1]
    introns = [(body[i][1], body[i + 1][0]) for i in range(len(body) - 1)]
    chain_novel, starts_outside, cands = 0, False, []
    if introns:
        cands = models.chain(minus, introns)
        if not cands and models.spans(minus, a0, b0):
            chain_novel, cands = models.suffix(minus, introns)
            if not cands:
                return ReadContext(NOVEL_JUNCTION, "", 0, 0, sub_cause=models.sub_cause(minus, introns[-1]))
    else:
        over = models.spans(minus, a0, b0)
        cands = [k for k in over if tx[k].start <= a0 and b0 <= tx[k].end + LONG_END_TOL]
        if not cands:
            cands = [k for k in over if tx[k].start <= p < tx[k].end + LONG_END_TOL]
            starts_outside = bool(cands)
            if over and not cands:
                return ReadContext("downstream", "", 0, 0)
    if not cands:
        return ReadContext(ANTISENSE if models.spans(not minus, -b0, -a0) else INTERGENIC, "", 0, 0)

    def order(k):
        d = abs(p - (tx[k].end - 1))
        return d > LONG_END_TOL, -_exonic(tx[k].exons, body), d, tx[k].tid

    t = tx[min(cands, key=order)]
    klass = _klass(t, p, bool(introns))
    retained = bool(introns) and bisect.bisect_right(t.exons, (p, _FAR)) - 1 > bisect.bisect_left(t.exons, (introns[-1][1],))
    return ReadContext(klass, t.tid, chain_novel or len(introns), chain_novel, starts_outside, retained,
                       klass == "annotated_end" and t.end_nf,
                       not chain_novel and abs(a0 - t.start) <= FIVE_PRIME_TOL,
                       t.start_codon is not None and any(s <= t.start_codon < e for s, e in body),
                       len(cands) > 1 and any(_klass(tx[k], p, bool(introns)) != klass for k in cands))


def scan_context(bam_path, ref, models, min_mapq=20, tagged=None):
    """Class and flags of every accepted read of one reference (READ_COLUMNS, in BAM order), and the record counts.

    The reads and their sites are those of stage 16 (bd67.scan_long_reads): primary records with MAPQ >= min_mapq whose
    terminal A run (bd67.terminal_a_run) passes bd67.accept_long_read; the site is the start of the run. A read whose
    aligned bases all belong to the run has no body; it is classed at the last cDNA base before its site, as a read
    without introns, and counted under empty_body. With tagged, the accepted records are also written to that BAM
    path, each with its site key in the tag tecap.TAG."""
    cols = [array(t) for t in _READ_TYPES]
    counts = dict.fromkeys(("records", "secondary_supplementary", "low_mapq", "used", "accepted", "empty_body"), 0)
    klass_code, cause_code = {c: i for i, c in enumerate(CLASSES)}, {c: i for i, c in enumerate(SUB_CAUSES)} | {"": -1}
    with pysam.AlignmentFile(str(bam_path)) as bam, ExitStack() as stack:
        out = stack.enter_context(pysam.AlignmentFile(str(tagged), "wb", template=bam)) if tagged else None
        for a in bam.fetch(ref):
            counts["records"] += 1
            if a.is_secondary or a.is_supplementary:
                counts["secondary_supplementary"] += 1
                continue
            if a.mapping_quality < min_mapq:
                counts["low_mapq"] += 1
                continue
            counts["used"] += 1
            site, aligned_a, clip_a, after_len = terminal_a_run(a)
            if not accept_long_read(aligned_a, clip_a, after_len):
                continue
            counts["accepted"] += 1
            minus = a.is_reverse
            body = read_body(a, site)
            if not body:
                counts["empty_body"] += 1
                body = [(-site - 1, -site)] if minus else [(site - 1, site)]
            c = classify(models, minus, body)
            row = (minus, site, klass_code[c.klass], c.matched, c.chain_novel, c.starts_outside, c.retained_after_junction,
                   c.end_nf, c.complete_5p, c.covers_start, c.other_class, cause_code[c.sub_cause], a.query_length,
                   all(op != 3 for op, _ in a.cigartuples))
            for col, value in zip(cols, row):
                col.append(value)
            if out is not None:
                a.set_tag(TAG, site_key(ref, minus, site), "Z")
                out.write(a)
    reads = pd.DataFrame({name: np.asarray(col, bool if name in _READ_BOOL else col.typecode)
                          for name, col in zip(READ_COLUMNS, cols)})
    return reads, counts


def top_class(m):
    """Top class per site of a sites x CLASSES matrix of read counts: the class with the most reads, among tied classes
    the first of CLASSES, so that a no_model reason never wins a tie against a class with a model. Returns the class
    names, whether two or more classes share the top count, and the share of the reads that the top class holds."""
    top = m.max(axis=1)
    return np.asarray(CLASSES, object)[m.argmax(axis=1)], (m == top[:, None]).sum(axis=1) > 1, top / m.sum(axis=1)


def reduce_sites(reads, ref):
    """(sites, by_class) of the accepted reads of one reference (READ_COLUMNS), sites ordered by strand and site.

    sites (SITE_COLUMNS): reads per class with the chain_novel reads under their class, the flag counts, the top class
    in both versions (context is "placed", context_without_chain_novel "strict") with tied and top_share for each,
    chain_novel_only (every read of the placed top class is chain_novel), sub_cause (where context is
    no_model_novel_junction, the sub-cause with the most reads, among tied ones the first of SUB_CAUSES; "" elsewhere),
    share_spliced (reads with a matched intron, the chain_novel ones included), share_5p_complete, median_length and
    n_no_splice. by_class (BY_CLASS_COLUMNS): per site, class, chain_novel and sub-cause ("" without one), the reads and
    how many of them have a matched intron, are 5' complete, cover the start and have a compatible transcript that
    gives another class."""
    key, code = np.unique(reads.minus.to_numpy(np.int64) << 40 | reads.site.to_numpy(np.int64), return_inverse=True)
    n, rows = len(key), np.arange(len(key))
    klass, novel, cause = reads.klass.to_numpy(np.int64), reads.chain_novel.to_numpy() > 0, reads.sub_cause.to_numpy(np.int64)

    def table(mask, col, width):
        return np.bincount(code[mask] * width + col[mask], minlength=n * width).reshape(n, width)

    def count(mask):
        return np.bincount(code[mask], minlength=n)

    m, moved = table(slice(None), klass, len(CLASSES)), table(novel, klass, len(CLASSES))
    strict = m - moved
    strict[:, CLASSES.index(NOVEL_JUNCTION)] += moved.sum(axis=1)
    top, tied, share = top_class(m)
    top_strict, tied_strict, share_strict = top_class(strict)
    causes = np.asarray(SUB_CAUSES, object)[table(cause >= 0, cause, len(SUB_CAUSES)).argmax(axis=1)]
    strand, site = np.where(key >> 40, "-", "+"), key & ((1 << 40) - 1)
    spliced = reads.matched.to_numpy() > 0
    sites = pd.DataFrame({"ref": ref, "strand": strand, "site": site, "n_accepted": m.sum(axis=1)}
                         | {f"n_{c}": m[:, i] for i, c in enumerate(CLASSES)}
                         | {"n_starts_outside": count(reads.starts_outside.to_numpy()), "n_chain_novel": count(novel),
                            "n_chain_novel_suffix1": count(reads.chain_novel.to_numpy() == 1),
                            "n_retained_after_junction": count(reads.retained_after_junction.to_numpy()),
                            "n_end_NF": count(reads.end_nf.to_numpy()), "context": top, "context_without_chain_novel": top_strict,
                            "top_share": share, "tied": tied, "share_spliced": count(spliced) / m.sum(axis=1),
                            "share_5p_complete": count(reads.complete_5p.to_numpy()) / m.sum(axis=1),
                            "median_length": reads.length.groupby(code).median().to_numpy(),
                            "tied_without_chain_novel": tied_strict, "top_share_without_chain_novel": share_strict,
                            "chain_novel_only": moved[rows, m.argmax(axis=1)] == m.max(axis=1),
                            "sub_cause": np.where(top == NOVEL_JUNCTION, causes, ""),
                            "n_no_splice": count(reads.no_splice.to_numpy())})
    width = len(SUB_CAUSES) + 1
    cell, inv, n_reads = np.unique(((code * len(CLASSES) + klass) * 2 + novel) * width + cause + 1, return_inverse=True,
                                   return_counts=True)

    def total(mask):
        return np.bincount(inv[mask], minlength=len(cell))

    k = cell // (len(CLASSES) * 2 * width)
    by_class = pd.DataFrame({"ref": ref, "strand": strand[k], "site": site[k],
                             "klass": np.asarray(CLASSES, object)[cell // (2 * width) % len(CLASSES)],
                             "chain_novel": cell // width % 2 == 1,
                             "sub_cause": np.asarray(("",) + SUB_CAUSES, object)[cell % width],
                             "reads": n_reads, "spliced": total(spliced), "complete_5p": total(reads.complete_5p.to_numpy()),
                             "covers_start": total(reads.covers_start.to_numpy()),
                             "other_class": total(reads.other_class.to_numpy())})
    return sites[SITE_COLUMNS], by_class[BY_CLASS_COLUMNS]


def attach_stage16(sites, stage16):
    """sites of reduce_sites with label, a_band, hexamer and region of the stage 16 site table (bd67.SITE_COLUMNS), in the
    order of sites, CONTEXT_COLUMNS first. ValueError unless sites holds exactly the stage 16 sites with an accepted
    read, each with the same number of accepted reads (the first acceptance condition of design 3.8)."""
    s16 = stage16.loc[stage16.n_accepted >= 1, _KEY + ["label", "a_band", "hexamer", "region", "n_accepted"]]
    j = sites.merge(s16.rename(columns={"n_accepted": "n_stage16"}), on=_KEY, how="left", validate="one_to_one")
    extra = int(j.n_stage16.isna().sum())
    missing, differ = len(s16) - (len(j) - extra), int((j.n_stage16.notna() & (j.n_accepted != j.n_stage16)).sum())
    if missing or extra or differ:
        raise ValueError(f"{missing} sites of stage 16 without reads here, {extra} sites here that stage 16 lacks, "
                         f"{differ} sites with another number of accepted reads")
    return j[CONTEXT_COLUMNS + [c for c in SITE_COLUMNS if c not in CONTEXT_COLUMNS]]


def intron_sites_off_model(sites):
    """Sites of the milestone 2 region intron whose top class, in either version, is no_model_antisense or
    no_model_intergenic. The site lies inside a transcript span of its strand, so there should be none (the last
    acceptance condition of design 3.8)."""
    off = sites.context.isin([ANTISENSE, INTERGENIC]) | sites.context_without_chain_novel.isin([ANTISENSE, INTERGENIC])
    return sites[(sites.region == "intron") & off]


def context_tables(sites, by_class, by):
    """version, the columns of by, context, sites, reads, share_spliced, share_5p_complete: per version and per value of
    the site columns `by` (of attach_stage16), the sites whose top class is the context, and the reads whose own class it
    is with the share of them that has a matched intron and that is 5' complete. In the strict version a chain_novel
    read is a no_model_novel_junction read without a matched intron. Rows in the order of VERSIONS, by and CLASSES;
    a context without a site and without a read has no row."""
    reads = by_class.merge(sites[_KEY + by], on=_KEY, validate="many_to_one")
    out = []
    for version, col in zip(VERSIONS, ("context", "context_without_chain_novel")):
        moved = reads.chain_novel.to_numpy() & (version == "strict")
        r = reads.assign(context=np.where(moved, NOVEL_JUNCTION, reads.klass), spliced=np.where(moved, 0, reads.spliced))
        r = r.groupby(by + ["context"])[["reads", "spliced", "complete_5p"]].sum()
        s = sites.groupby(by + [col]).size().rename("sites").rename_axis(by + ["context"])
        t = pd.concat([s, r], axis=1).fillna(0).reset_index()
        t = t.iloc[np.lexsort([t.context.map(CLASSES.index)] + [t[c] for c in reversed(by)])]
        out.append(pd.DataFrame({"version": version} | {c: t[c] for c in by + ["context"]}
                                | {"sites": t.sites.astype(np.int64), "reads": t.reads.astype(np.int64),
                                   "share_spliced": t.spliced / t.reads, "share_5p_complete": t.complete_5p / t.reads}))
    return pd.concat(out, ignore_index=True)


def totals(sites, by_class):
    """Counts of one library for isoform_context.json, from the site table of attach_stage16 and by_class: the sites of
    the milestone 2 region intron; per version the reads by their own class, the sites by top class, the tied sites, the
    sites whose top class holds at most half of the reads and the intron sites whose top class is a no_model reason; the
    sites whose placed top class holds chain_novel reads only; the flagged reads; per class of the placed version its
    reads, those with a matched intron, 5' complete, covering the start and with a compatible transcript of another
    class (other_class among the reads that are not chain_novel, chain_novel_other_class among those that are); the
    novel-junction reads by sub-cause and the sites with a novel-junction top class by main sub-cause."""
    intron = (sites.region == "intron").to_numpy()
    out = {"accepted_reads": int(sites.n_accepted.sum()), "sites": len(sites), "intron_sites": int(intron.sum())}
    novel = by_class.chain_novel.to_numpy()
    for version, col, suffix in zip(VERSIONS, ("context", "context_without_chain_novel"), ("", "_without_chain_novel")):
        own = np.where(novel & (version == "strict"), NOVEL_JUNCTION, by_class.klass)
        by_top = sites[col].value_counts()
        out[version] = {"reads_by_class": {c: int(by_class.reads[own == c].sum()) for c in CLASSES},
                        "sites_by_context": {c: int(by_top.get(c, 0)) for c in CLASSES},
                        "sites_tied": int(sites["tied" + suffix].sum()),
                        "sites_top_share_at_most_half": int((sites["top_share" + suffix] <= 0.5).sum()),
                        "intron_sites_top_no_model": int((intron & sites[col].str.startswith("no_model_").to_numpy()).sum())}
    out["sites_top_class_on_chain_novel_only"] = int(sites.chain_novel_only.sum())
    out["flagged_reads"] = {k: int(sites["n_" + k].sum()) for k in ("starts_outside", "chain_novel", "chain_novel_suffix1",
                                                                    "retained_after_junction", "end_NF", "no_splice")}
    out["by_class"] = {}
    for c in CLASSES:
        g, placed = by_class[by_class.klass == c], by_class[(by_class.klass == c) & novel]
        out["by_class"][c] = {"reads": int(g.reads.sum()), "matched_intron": int(g.spliced.sum()),
                              "complete_5p": int(g.complete_5p.sum()), "covers_start": int(g.covers_start.sum()),
                              "other_class": int(g.other_class.sum() - placed.other_class.sum()),
                              "chain_novel": int(placed.reads.sum()), "chain_novel_other_class": int(placed.other_class.sum())}
    by_cause, main = by_class.groupby("sub_cause").reads.sum(), sites.sub_cause.value_counts()
    out["novel_junction_reads_by_sub_cause"] = {c: int(by_cause.get(c, 0)) for c in SUB_CAUSES}
    out["novel_junction_sites_by_main_sub_cause"] = {c: int(main.get(c, 0)) for c in SUB_CAUSES}
    return out


def isoform_ends(gtf, refs=None):
    """isoform, ref, strand, site of the transcript lines of a collapsed isoform GTF (all references, or refs). The site
    is the boundary after the last base of the isoform in transcript sense, as sites.priming_site gives it for a read:
    the 0-based end on plus, the 0-based start on minus."""
    rows = []
    with open(gtf) as fh:
        for line in fh:
            f = line.split("\t")
            if line.startswith("#") or f[2] != "transcript" or (refs is not None and f[0] not in refs):
                continue
            tid = f[8].split('transcript_id "', 1)[1].split('"', 1)[0]
            rows.append((tid, f[0], f[6], int(f[4]) if f[6] == "+" else int(f[3]) - 1))
    return pd.DataFrame(rows, columns=["isoform", "ref", "strand", "site"])


def pigeon_crosscheck(ends, classification, sites):
    """(table, counts): isoform ends (isoform_ends) against the long-read sites (ref, strand, site, context, n_accepted).
    Each end takes the context of the nearest site on its strand within 10 nt, by the stage 16 match
    (bd67.nearest_labels), or "none". table: structural_category, subcategory, context, isoforms, with the pigeon
    classification of each isoform (isoform, structural_category, subcategory; "not_classified" for an isoform without
    a row), contexts in the order of CLASSES. counts: the isoform ends, those without a classification row, those that
    find a site and their share."""
    context = nearest_labels(ends, sites.assign(label=sites.context))
    j = ends[["isoform"]].assign(context=context).merge(classification, on="isoform", how="left", validate="one_to_one")
    j[["structural_category", "subcategory"]] = j[["structural_category", "subcategory"]].fillna("not_classified")
    t = j.groupby(["structural_category", "subcategory", "context"]).size().rename("isoforms").reset_index()
    t = t.iloc[np.lexsort([t.context.map((CLASSES + ("none",)).index), t.subcategory, t.structural_category])]
    matched = int((context != "none").sum())
    return t.reset_index(drop=True), {"isoform_ends": len(ends),
                                      "without_classification": int(j.structural_category.eq("not_classified").sum()),
                                      "matched": matched, "share_matched": matched / len(ends) if len(ends) else None}


def link_1179(mol, sites, shift=NULL_SHIFT):
    """(context, link) for truth molecules (ref, strand, site, label) against the long-read sites (ref, strand, site,
    label, context, n_accepted). context: per molecule the top class of the nearest site with an accepted read within
    10 nt on its strand, the stage 16 match (bd67.nearest_labels), or "none". link: the matched share and the label
    agreement of bd67.compare_labels as "real", and as "null" the same with every molecule's site moved `shift` nt
    upstream in transcript sense (lower coordinate on plus, higher on minus), which says how much of the match and of
    the agreement nearby positions give by themselves."""
    moved = mol.assign(site=mol.site + np.where(mol.strand == "+", -shift, shift))
    return nearest_labels(mol, sites.assign(label=sites.context)), {
        "real": compare_labels(mol, sites)[1], "null_shift_nt": shift, "null": compare_labels(moved, sites)[1]}
