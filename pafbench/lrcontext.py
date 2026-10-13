"""Stage 30: context of long-read 3' ends (design of 2026-10-12, section 3.8). Each accepted read is classed by its own
alignment against the transcripts of the GTF.

All coordinates are in transcript sense: a minus-strand interval [a, b) is mirrored to [-b, -a), so that on both strands
a transcript runs from low to high, exons and introns are sorted, and the 3' end is the high end.
"""
import bisect
from collections import defaultdict, namedtuple

from .config import FIVE_PRIME_TOL, LONG_END_TOL, MIN_INTRON, NEAR_INTRON_TOL

# The order is the tie order of the top class of a site: the eight classes with a model, then the three no_model reasons.
CLASSES = ("annotated_end", "3UTR", "CDS", "5UTR", "ncexon", "intron_unspliced", "intron_after_junction", "downstream",
           "no_model_novel_junction", "no_model_antisense", "no_model_intergenic")
NOVEL_JUNCTION, ANTISENSE, INTERGENIC = CLASSES[8:]
# Why the last intron of a read is in no transcript of its strand; the first that applies, and the tie order of the main
# sub-cause of a site.
SUB_CAUSES = ("near_annotated_intron", "short_intron", "opposite_strand_intron", "sites_not_joined", "donor_only",
              "acceptor_only", "neither_site")
Transcript = namedtuple("Transcript", "tid start end exons cds end_nf")   # cds: (first base, end) of the CDS or None
# matched: introns of the read that the transcript has as a run; chain_novel: 0, or that number for a read placed by its
# 3' part. sub_cause is set for no_model_novel_junction only.
ReadContext = namedtuple("ReadContext", "klass transcript matched chain_novel starts_outside retained_after_junction end_nf "
                                        "complete_5p covers_start other_class sub_cause", defaults=(False,) * 6 + ("",))
_BIN = 1 << 14
_FAR = 1 << 62


class Models:
    """Transcripts of one reference with the lookups of the class rule. rows: (transcript id, minus strand, (start, end),
    exons, CDS intervals, tagged mRNA_end_NF), intervals in transcript sense.

    tx[k] is a Transcript; introns[minus][(donor, acceptor)] lists (k, i) for every transcript k of the strand whose
    i-th intron it is; donors[minus] and acceptors[minus] are the splice sites of the strand."""

    def __init__(self, rows):
        self.tx, self._strand, self._chains = [], [], []
        self.introns = {False: defaultdict(list), True: defaultdict(list)}
        self._bins = {False: defaultdict(list), True: defaultdict(list)}
        for tid, minus, (start, end), exons, cds, end_nf in rows:
            exons = sorted(exons)
            k = len(self.tx)
            self.tx.append(Transcript(tid, start, end, exons, (min(s for s, _ in cds), max(e for _, e in cds)) if cds else None,
                                      end_nf))
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
    """{ref: Models} from the transcript, exon and CDS lines of a GTF, transcripts in file order. With refs, those
    references in that order, each with a Models even when the GTF has no transcript on it.

    end_nf is the GENCODE tag mRNA_end_NF on the transcript line ("the mRNA end could not be confirmed",
    https://www.gencodegenes.org/pages/tags.html)."""
    tx, exons, cds = {}, defaultdict(list), defaultdict(list)
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] not in ("transcript", "exon", "CDS") or (refs is not None and f[0] not in refs):
                continue
            tid = f[8].split('transcript_id "', 1)[1].split('"', 1)[0]
            minus = f[6] == "-"
            iv = (-int(f[4]), 1 - int(f[3])) if minus else (int(f[3]) - 1, int(f[4]))
            if f[2] == "transcript":
                tx[tid] = (f[0], minus, iv, 'tag "mRNA_end_NF"' in f[8])
            elif f[2] == "exon":
                exons[tid].append(iv)
            else:
                cds[tid].append(iv)
    rows = {ref: [] for ref in refs} if refs is not None else defaultdict(list)
    for tid, (ref, minus, span, end_nf) in tx.items():
        rows[ref].append((tid, minus, span, exons[tid], cds.get(tid, []), end_nf))
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
    complete_5p (the 5' end within FIVE_PRIME_TOL of the transcript's) and covers_start (an aligned block holds the first
    base of its CDS) are never set for a chain_novel read, whose 5' part follows no transcript."""
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
                       not chain_novel and t.cds is not None and any(s <= t.cds[0] < e for s, e in body),
                       len(cands) > 1 and any(_klass(tx[k], p, bool(introns)) != klass for k in cands))
