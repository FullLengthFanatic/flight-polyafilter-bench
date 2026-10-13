"""Read groups of the called cells and their removed reads per setting, in two streaming passes per reference (stage 23).

A sibling of stream.scan_reference (design of 2026-10-12, section 3.1): pass 1 and the pieces that define removal come from
stream, so coverage and expressed transcripts are those of every record of the BAM; the chunk buffers, the kernel and the
bookkeeping are its own. The kernel gives each record one bit per setting. Kept from the primary records:
- read groups: (cell, UB, gene) over the records of a called cell with a UB, a single-gene GX of the feature list and xf bit 1;
- record counters per called cell, RE and presence of a GX tag, and the same over all barcodes (the tie to stage 13);
- units: distinct (cell, UB, strand, 1 kb bin of the leftmost aligned position) among the records of a called cell with a UB.
"""
import gzip

import numpy as np
import pandas as pd
import pysam
from numba import njit

from .config import UNIT_BIN
from .flags import setting_tag
from .stream import RE_CODES, SECONDARY_OR_SUPPLEMENTARY, _pack, coverage_prefix_chunked, reference_pieces

UMI_LEN = 12
GROUP_MAX = 65_535  # n_reads and the rm columns are uint16
# A 0, C 1, G 2, T 3, first base highest: the code of the umi column of molecule_info.h5 (our observation: with it all 467,969
# molecules of 1417 chr21 and chr22 join the BAM, design_m3_draft/design/checks/count_vs_support_1417.json)
_UMI_CODE = np.full(256, 4, np.uint8)
_UMI_CODE[np.frombuffer(b"ACGT", np.uint8)] = np.arange(4)
_UMI_WEIGHT = 4 ** np.arange(UMI_LEN - 1, -1, -1, dtype=np.int64)


def row_index(tsv_gz):
    """{first column: row} of barcodes.tsv.gz or features.tsv.gz."""
    with gzip.open(tsv_gz, "rt") as fh:
        return {line.rstrip("\n").split("\t")[0]: i for i, line in enumerate(fh)}


@njit(cache=True)
def _mask_hits(bs, be, br, ps, pe, offs, mask):
    """Blocks of one strand sorted by start against each tag's sorted disjoint pieces, the test of stream._scan_hits.

    Bit t of mask[r] is set when a block of record r touches a piece of tag t.
    """
    nb = bs.size
    if nb == 0:
        return
    for t in range(offs.size - 1):
        lo, hi = offs[t], offs[t + 1]
        if lo == hi:
            continue
        bit = np.uint32(1) << np.uint32(t)
        j = lo + np.searchsorted(pe[lo:hi], bs[0], side="right")
        for k in range(nb):
            s = bs[k]
            while j < hi and pe[j] <= s:
                j += 1
            if j == hi:
                break
            if ps[j] < be[k]:
                mask[br[k]] |= bit


@njit(cache=True)
def _count_records(mask, prim, cls, cell, has_ub, allc, celc):
    """Primary records by class into allc[class] and, for a called cell, celc[cell, class].

    Column 0: records; 1: records with a UB; 2 + t: records removed at tag t.
    """
    for r in range(mask.size):
        if not prim[r]:
            continue
        k, j = cls[r], cell[r]
        allc[k, 0] += 1
        if j >= 0:
            celc[j, k, 0] += 1
        if has_ub[r]:
            allc[k, 1] += 1
            if j >= 0:
                celc[j, k, 1] += 1
        m, col = mask[r], 2
        while m:
            if m & 1:
                allc[k, col] += 1
                if j >= 0:
                    celc[j, k, col] += 1
            m >>= 1
            col += 1


class _Chunk:
    """Fixed numpy buffers for one chunk of records and their blocks.

    gene, xf, pa and end3 are written for the primary records of called cells only, and ubs holds the UB of those that
    have one, in record order.
    """

    def __init__(self, chunk):
        self.flag = np.empty(chunk, np.int64)
        self.pos = np.empty(chunk, np.int64)
        self.cls = np.empty(chunk, np.int64)  # 2 * RE code + (the record has a GX tag)
        self.cell = np.empty(chunk, np.int64)
        self.has_ub = np.empty(chunk, np.bool_)
        self.gene = np.empty(chunk, np.int64)
        self.xf = np.empty(chunk, np.int64)
        self.pa = np.empty(chunk, np.bool_)
        self.end3 = np.empty(chunk, np.int32)
        self._blocks(4 * chunk + 64)
        self.nr = self.nb = 0
        self.ubs = []

    def _blocks(self, cap):
        self.bs, self.be, self.br = np.empty(cap, np.int64), np.empty(cap, np.int64), np.empty(cap, np.int64)

    def fits(self, n_blocks):
        if self.nb + n_blocks <= self.bs.size:
            return True
        if self.nb == 0:
            self._blocks(2 * n_blocks)
            return True
        return False

    def reset(self):
        self.nr = self.nb = 0
        self.ubs = []

    def masks(self, packed):
        """uint32 per record: bit t is set when the record is removed at tag t. packed: the pieces of each strand (_pack)."""
        nb = self.nb
        mask = np.zeros(self.nr, np.uint32)
        bs, be, br = self.bs[:nb], self.be[:nb], self.br[:nb]
        bfwd = (self.flag[br] & 16) == 0
        for strd in (True, False):
            sel = np.flatnonzero(bfwd == strd)
            o = sel[np.argsort(bs[sel], kind="stable")]
            _mask_hits(bs[o], be[o], br[o], *packed[strd], mask)
        return mask


def _umi_codes(ubs, ref):
    """The UMIs as integers in 2-bit code; ValueError unless each is 12 bases of A, C, G and T."""
    ok = set(map(len, ubs)) <= {UMI_LEN}
    if ok:
        code = _UMI_CODE[np.frombuffer("".join(ubs).encode("ascii", "replace"), np.uint8)].reshape(-1, UMI_LEN)
        ok = code.max(initial=0) < 4
    if not ok:
        bad = next(u for u in ubs if len(u) != UMI_LEN or set(u) - set("ACGT"))
        raise ValueError(f"{ref}: UB {bad!r} is not {UMI_LEN} bases of A, C, G and T; the key has no code for it")
    return code.astype(np.int64) @ _UMI_WEIGHT


def _reduce_groups(key, removed, flags, end3, tags):
    """One row per read group from the records of its reads, sorted by key.

    key: cell index (24 bits), UMI in 2-bit code (24 bits), gene index (16 bits), highest first, so that sorting brings the
    groups of one cell and UMI together. removed: bit t set when the record is removed at tags[t]. flags: bit 0 RE E, bit 1
    RE N, bit 2 a pa tag above 0, bit 3 xf bit 2. end3: the record's 3'-most aligned base in transcript sense, 0-based (the
    last aligned base of a forward record, the first of a reverse one, as tests/synth.build_10x states it), negated on the
    reverse strand so that the 3'-most base of a group is a maximum on both.
    """
    o = np.argsort(key, kind="stable")
    key, removed, flags, end3 = key[o], removed[o], flags[o], end3[o]
    new = np.ones(key.size, np.bool_)
    new[1:] = key[1:] != key[:-1]
    start = np.flatnonzero(new)
    n_reads = np.diff(np.append(start, key.size))
    if n_reads.max(initial=0) >= GROUP_MAX:
        raise ValueError(f"a read group has {n_reads.max():,} reads; the pass stops at {GROUP_MAX:,}, its counts are uint16")
    lowest = np.iinfo(np.int32).min
    nonpa = np.maximum.reduceat(np.where(flags & 4, lowest, end3), start)
    cols = {
        "key": key[start],
        "n_reads": n_reads,
        "n_winner_reads": np.add.reduceat((flags & 8) == 0, start, dtype=np.int64),
        "n_pa": np.add.reduceat((flags & 4) != 0, start, dtype=np.int64),
        "re_class": np.bitwise_or.reduceat(flags & 3, start),
        "end3": np.abs(np.maximum.reduceat(end3, start)),
        "end3_nonpa": np.where(nonpa == lowest, -1, np.abs(nonpa)),
    }
    for t, tag in enumerate(tags):
        cols[f"rm_{tag}"] = np.add.reduceat((removed >> np.uint32(t)) & np.uint32(1), start, dtype=np.int64)
    dtypes = {c: np.uint16 for c in cols} | {"key": np.uint64, "re_class": np.uint8, "end3": np.int32, "end3_nonpa": np.int32}
    return pd.DataFrame({c: v.astype(dtypes[c]) for c, v in cols.items()})


def scan_read_groups(bam_path, fasta_path, ref, transcripts, settings, cell_idx, gene_idx, chunk=1_000_000):
    """(groups, cell_records, counts) for one reference.

    cell_idx, gene_idx: {barcode: row of barcodes.tsv.gz}, {gene id: row of features.tsv.gz} (row_index).
    groups: one row per read group of a called cell, sorted by key: key (uint64: cell index 24 bits, UMI in 2-bit code 24
    bits, gene index 16 bits), n_reads, n_winner_reads (records with xf bit 2 unset), n_pa (records with a pa tag above 0;
    uint16), re_class (uint8: 1 E only, 2 N only, 3 both), end3 (the 3'-most aligned base of the group in transcript sense,
    0-based), end3_nonpa (the same over the records without pa, -1 when there is none; int32), rm_<tag> per setting
    (uint16: records removed).
    cell_records: primary records per cell, RE ("" without the tag) and presence of a GX tag, rows with records only: cell
    (int32 row of barcodes.tsv.gz, -1 for the row set over all barcodes, called or not, with or without CB), re, gx, total,
    with_ub, units, units_rg (the units with a record of a read group; both 0 in the rows of all barcodes, a unit needs
    a cell), then one column per tag (records removed); uint32.
    counts: records (all fetched), primary, called (primary with a called-cell CB), of those with xf bit 1: xf1_no_ub and
    xf1_no_gene (a UB, but no GX, two genes in GX or a gene outside the feature list), in_groups; read_groups.
    ValueError when a record starts before its predecessor, when a UB of a called cell's primary record is not 12 bases
    of A, C, G and T, and when a read group has 65,535 reads or more; KeyError on an RE other than E, N and I.
    """
    if len(settings) > 32 or len(cell_idx) > 1 << 24 or len(gene_idx) > 1 << 16:
        raise ValueError("the removal mask has 32 bits and the key holds 16,777,216 cells and 65,536 features")
    with pysam.FastaFile(str(fasta_path)) as fa:
        seq = fa.fetch(ref)
    tx_by_strand = {strd: transcripts.get((ref, strd), []) for strd in (True, False)}
    pieces = reference_pieces(seq, tx_by_strand, coverage_prefix_chunked(bam_path, ref), settings)
    del seq
    tags = [setting_tag(s) for s in settings]
    packed = {strd: _pack([pieces.pop((t, strd)) for t in tags]) for strd in (True, False)}
    re_code = {c: i for i, c in enumerate(RE_CODES)}
    n_cells, n_cls = len(cell_idx), 2 * len(RE_CODES)
    allc = np.zeros((n_cls, 2 + len(tags)), np.int64)
    celc = np.zeros((n_cells, n_cls, 2 + len(tags)), np.uint32)
    units = np.zeros((2, n_cells * n_cls), np.int64)  # units, units_rg
    # distinct units of the last bin seen (bin, key, has a read-group record): its records can go on in the next chunk
    pend = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.bool_))
    rg = ([], [], [], [])  # key, removed, flags and end3 of the records of read groups
    counts = {"records": 0, "primary": 0, "called": 0, "xf1_no_ub": 0, "xf1_no_gene": 0}

    def flush(c, last):
        nonlocal pend
        nr = c.nr
        flag, cls, cell, has_ub = c.flag[:nr], c.cls[:nr], c.cell[:nr], c.has_ub[:nr]
        mask = c.masks(packed)
        prim = (flag & SECONDARY_OR_SUPPLEMENTARY) == 0
        _count_records(mask, prim, cls, cell, has_ub, allc, celc)
        unit = prim & (cell >= 0) & has_ub
        xf1 = prim & (cell >= 0) & ((c.xf[:nr] & 1) == 1)
        grp = unit & xf1 & (c.gene[:nr] >= 0)
        counts["xf1_no_ub"] += int((xf1 & ~has_ub).sum())
        counts["xf1_no_gene"] += int((xf1 & has_ub & ~grp).sum())
        umi = np.zeros(nr, np.int64)
        umi[unit] = _umi_codes(c.ubs, ref)

        u = np.flatnonzero(unit)
        b = np.concatenate((pend[0], c.pos[u] // UNIT_BIN))
        k = np.concatenate((pend[1], (cell[u] << 28) | (umi[u] << 4) | ((flag[u] & 16) >> 1) | cls[u]))
        r = np.concatenate((pend[2], grp[u]))
        o = np.lexsort((k, b))
        b, k, r = b[o], k[o], r[o]
        new = np.ones(b.size, np.bool_)
        new[1:] = (b[1:] != b[:-1]) | (k[1:] != k[:-1])
        start = np.flatnonzero(new)
        b, k, r = b[start], k[start], np.logical_or.reduceat(r, start)
        held = np.zeros(b.size, np.bool_) if last else b == b[-1:]
        pend = (b[held], k[held], r[held])
        idx = (k[~held] >> 28) * n_cls + (k[~held] & 7)
        units[0] += np.bincount(idx, minlength=units.shape[1])
        units[1] += np.bincount(idx[r[~held]], minlength=units.shape[1])

        g = np.flatnonzero(grp)
        re_ = cls[g] >> 1
        rg[0].append((cell[g].astype(np.uint64) << np.uint64(40)) | (umi[g].astype(np.uint64) << np.uint64(16))
                     | c.gene[g].astype(np.uint64))
        rg[1].append(mask[g])
        rg[2].append(((re_ == 0) | ((re_ == 1) << 1) | (c.pa[g] << 2) | ((c.xf[g] & 2) << 2)).astype(np.uint8))
        rg[3].append(c.end3[g])
        c.reset()

    c = _Chunk(chunk)
    last_pos = 0
    with pysam.AlignmentFile(str(bam_path)) as bam:
        for a in bam.fetch(ref):
            counts["records"] += 1
            pos = a.reference_start
            if pos < last_pos:
                raise ValueError(f"{bam_path}, {ref}: record {counts['records']:,} starts at {pos}, before its predecessor at "
                                 f"{last_pos}; the pass needs the records in coordinate order")
            last_pos = pos
            blocks = a.get_blocks()
            if c.nr == chunk or not c.fits(len(blocks)):
                flush(c, False)
                c.fits(len(blocks))
            i = c.nr
            flag = a.flag
            c.flag[i] = flag
            c.pos[i] = pos
            try:
                re_ = a.get_tag("RE")
            except KeyError:
                re_ = ""
            has_gx = a.has_tag("GX")
            c.cls[i] = 2 * re_code[re_] + has_gx
            has_ub = c.has_ub[i] = a.has_tag("UB")
            try:
                cell = cell_idx.get(a.get_tag("CB"), -1)
            except KeyError:
                cell = -1
            c.cell[i] = cell
            if cell >= 0 and not flag & SECONDARY_OR_SUPPLEMENTARY:
                if has_ub:
                    c.ubs.append(a.get_tag("UB"))
                c.gene[i] = gene_idx.get(a.get_tag("GX"), -1) if has_gx else -1  # two genes in GX are no feature id
                c.xf[i] = a.get_tag("xf")
                c.pa[i] = a.has_tag("pa") and a.get_tag("pa") > 0
                c.end3[i] = -pos if flag & 16 else a.reference_end - 1
            nb = c.nb
            for s, e in blocks:
                c.bs[nb] = s
                c.be[nb] = e
                c.br[nb] = i
                nb += 1
            c.nb = nb
            c.nr = i + 1
    flush(c, True)

    groups = _reduce_groups(*(np.concatenate(x) for x in rg), tags)
    seen = np.flatnonzero(allc[:, 0] > 0)
    rows = np.argwhere(celc[:, :, 0] > 0)
    cls = np.concatenate((seen, rows[:, 1]))
    num = np.concatenate((allc[seen], celc[rows[:, 0], rows[:, 1]]))
    unit = np.zeros((2, cls.size), np.int64)
    unit[:, seen.size:] = units[:, rows[:, 0] * n_cls + rows[:, 1]]
    records = pd.DataFrame({
        "cell": np.concatenate((np.full(seen.size, -1), rows[:, 0])).astype(np.int32),
        "re": pd.Series(np.array(RE_CODES, dtype=object)[cls >> 1], dtype="str"),
        "gx": (cls & 1).astype(np.bool_),
    } | {name: col.astype(np.uint32) for name, col in zip(["total", "with_ub", "units", "units_rg"] + tags,
                                                           [num[:, 0], num[:, 1], unit[0], unit[1], *num[:, 2:].T])})
    counts["primary"] = int(allc[:, 0].sum())
    counts["called"] = int(celc[:, :, 0].sum(dtype=np.int64))
    counts |= {"in_groups": int(groups.n_reads.sum()), "read_groups": len(groups)}
    return groups, records, counts
