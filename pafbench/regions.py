from collections import defaultdict

import numpy as np

# Index is the priority: where annotations overlap on a strand, the highest code wins. regions_at adds "antisense".
REGIONS = ("intergenic", "intron", "ncexon", "5UTR", "CDS", "3UTR")
REGION_NAMES = REGIONS + ("antisense",)


def _field(attrs, key):
    return attrs.split(f'{key} "', 1)[1].split('"', 1)[0]


def _paint(bounds, codes, pieces, code):
    if not pieces:
        return
    s, e = np.asarray(pieces, np.int64).T
    diff = np.zeros(bounds.size, np.int64)
    np.add.at(diff, np.searchsorted(bounds, s), 1)
    np.add.at(diff, np.searchsorted(bounds, e), -1)
    codes[np.cumsum(diff)[:-1] > 0] = code


def load_region_segments(gtf):
    """{(ref, strd): (bounds, codes)}: segment [bounds[k], bounds[k+1]) has region REGIONS[codes[k]] on that strand.

    Exon parts before the CDS in transcript sense are 5UTR, after it 3UTR (the stop codon falls in 3UTR, as in
    GENCODE's UTR features); exons of transcripts without CDS are ncexon; the rest of a transcript span is intron.
    """
    tx, exons, cds = {}, defaultdict(list), defaultdict(list)
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] not in ("transcript", "exon", "CDS"):
                continue
            tid = _field(f[8], "transcript_id")
            iv = (int(f[3]) - 1, int(f[4]))
            if f[2] == "transcript":
                tx[tid] = (f[0], f[6] == "+", iv)
            elif f[2] == "exon":
                exons[tid].append(iv)
            else:
                cds[tid].append(iv)
    pieces = defaultdict(lambda: defaultdict(list))
    for tid, (ref, strd, span) in tx.items():
        p = pieces[(ref, strd)]
        p[1].append(span)
        if tid not in cds:
            p[2].extend(exons[tid])
            continue
        cs = min(s for s, _ in cds[tid])
        ce = max(e for _, e in cds[tid])
        left, right = (3, 5) if strd else (5, 3)
        for s, e in exons[tid]:
            if s < cs:
                p[left].append((s, min(e, cs)))
            if e > ce:
                p[right].append((max(s, ce), e))
            if s < ce and e > cs:
                p[4].append((max(s, cs), min(e, ce)))
    out = {}
    for key, by_code in pieces.items():
        bounds = np.unique(np.concatenate([np.asarray(v, np.int64).ravel() for v in by_code.values()]))
        codes = np.zeros(bounds.size - 1, np.int8)
        for code in range(1, len(REGIONS)):
            _paint(bounds, codes, by_code.get(code, []), code)
        out[key] = (bounds, codes)
    return out


def _codes_at(segments, ref, strd, pos):
    c = np.zeros(pos.size, np.int8)
    entry = segments.get((ref, strd))
    if entry is None:
        return c
    bounds, codes = entry
    k = np.searchsorted(bounds, pos, side="right") - 1
    inside = (k >= 0) & (k < codes.size)
    c[inside] = codes[k[inside]]
    return c


def regions_at(segments, ref, strd, positions):
    """Region names at 0-based positions on one strand; unannotated positions inside an opposite-strand transcript are 'antisense'."""
    pos = np.asarray(positions, np.int64)
    names = np.asarray(REGIONS, object)[_codes_at(segments, ref, strd, pos)]
    names[(names == REGIONS[0]) & (_codes_at(segments, ref, not strd, pos) > 0)] = "antisense"
    return names


def region_at(segments, ref, strd, pos):
    return regions_at(segments, ref, strd, [pos])[0]
