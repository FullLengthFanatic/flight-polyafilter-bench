"""Per-record polyAfilter removal for many settings in two streaming passes per reference (stage 13).

Pass 1 builds the per-strand coverage prefix (only needed to decide which transcripts are expressed).
Pass 2 reads records in chunks and checks each record's blocks against every setting's pieces with a
merge kernel; only aggregates (per RE code, per called cell) and the rows of truth reads are kept.
Results equal flags.flag_reference record for record.
"""
import json
import time
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
from numba import njit

from .flags import setting_tag
from .intervals import flatten, intersect
from .snr import scan_snrs, snr_pieces
from .transcripts import expressed
from .vendor import polyafilter_modules

SECONDARY_OR_SUPPLEMENTARY = 0x900
RE_CODES = ("E", "N", "I", "")


@njit(cache=True)
def _add_blocks(diff_f, diff_r, s, e, fwd, n):
    # Shifted by one: diff[p + 1] holds the depth change at position p. A non-ACGT aligned base at r arrives as
    # (r + 1, r): it undoes its own block's +1 at that one position.
    for i in range(n):
        if fwd[i]:
            diff_f[s[i] + 1] += 1
            diff_f[e[i] + 1] -= 1
        else:
            diff_r[s[i] + 1] += 1
            diff_r[e[i] + 1] -= 1


def coverage_prefix_chunked(bam_path, ref, chunk=2_000_000):
    """Same values as transcripts.coverage_prefix (int32), with blocks buffered and flushed every `chunk` blocks."""
    with pysam.AlignmentFile(str(bam_path)) as bam:
        L = bam.get_reference_length(ref)
        diff_f, diff_r = np.zeros(L + 2, np.int32), np.zeros(L + 2, np.int32)
        bs, be, bf = np.empty(chunk, np.int64), np.empty(chunk, np.int64), np.empty(chunk, np.bool_)
        k = 0
        for a in bam.fetch(ref):
            seq = a.query_sequence
            if seq is None:
                continue
            fwd = not a.is_reverse
            items = a.get_blocks()
            if seq.encode().translate(None, b"ACGT"):
                items = items + [(r + 1, r) for q, r in a.get_aligned_pairs(matches_only=True) if seq[q] not in "ACGT"]
            for s, e in items:
                bs[k] = s
                be[k] = e
                bf[k] = fwd
                k += 1
                if k == chunk:
                    _add_blocks(diff_f, diff_r, bs, be, bf, k)
                    k = 0
        _add_blocks(diff_f, diff_r, bs, be, bf, k)
    out = {}
    for strd, d in ((True, diff_f), (False, diff_r)):
        # in place, no temporaries: d[p + 1] = depth at p, then covered at p, then d[i] = covered positions before i
        np.add.accumulate(d, out=d)
        np.greater(d, 0, out=d)
        np.add.accumulate(d, out=d)
        out[strd] = d[:L + 1]
    return out


def body_pieces(expressed_tx, covlen, include_introns, strd):
    """transcripts.start_pieces, but each transcript's removeOverlaps gets only the protected ends that can touch its span.

    removeOverlaps is a set difference over sorted flattened lists, so the slice gives the same pieces; it turns the
    per-transcript sort and scan of every end on the strand (quadratic in transcripts) into a short list.
    """
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
    end_s = [a for a, _ in all_ends]
    end_e = [b for _, b in all_ends]
    flat = []
    for st in each_start:
        i0 = bisect_right(end_e, min(a for a, _ in st))
        i1 = bisect_left(end_s, max(b for _, b in st))
        flat.extend(SNR.removeOverlaps(list(st), all_ends[i0:i1]))
    flat = SNR.flattenIntervals(flat)
    return np.array([a for a, _ in flat], np.int64), np.array([b for _, b in flat], np.int64)


def reference_pieces(seq, tx_by_strand, prefix, settings):
    """{(tag, strd): (ps, pe)}, the pieces flag_reference intersects blocks with.

    Same caches as flag_reference (bodies by strand, COVLEN and mode; A-stretch scans by strand and MISM). A scan is
    dropped once its settings are done, stretches are pre-filtered by MINSNRLEN (snr_pieces filters again, a no-op),
    and the flattened windows are shared by both modes.
    """
    ref_len = len(seq)
    modes = {s.introns for s in settings}
    ex = {(strd, i): expressed(tx_by_strand[strd], prefix[strd], i) for strd in (True, False) for i in modes}
    del prefix  # scan_reference passes the prefix inline, so this frees it before the A-stretch scans
    out = {}
    for strd in (True, False):
        body = {k: body_pieces(ex[(strd, k[1])], k[0], k[1], strd) for k in sorted({(s.covlen, s.introns) for s in settings})}
        for st in settings:
            if st.min_len is None:
                out[(setting_tag(st), strd)] = body[(st.covlen, st.introns)]
        for mism in sorted({s.mism for s in settings if s.min_len is not None}):
            first, last = scan_snrs(seq, strd, mism)
            group = [s for s in settings if s.min_len is not None and s.mism == mism]
            for min_len in sorted({s.min_len for s in group}):
                keep = (last - first) >= min_len
                first, last = first[keep], last[keep]
                for covlen in sorted({s.covlen for s in group if s.min_len == min_len}):
                    ws, we = flatten(*snr_pieces(first, last, min_len, covlen, strd, ref_len))
                    for st in group:
                        if (st.covlen, st.min_len) == (covlen, min_len):
                            out[(setting_tag(st), strd)] = intersect(*body[(covlen, st.introns)], ws, we)
            del first, last
    return out


def _pack(pairs):
    """Concatenate per-tag (ps, pe) into flat int64 arrays with offsets[t]:offsets[t + 1] per tag."""
    offs = np.zeros(len(pairs) + 1, np.int64)
    offs[1:] = np.cumsum([s.size for s, _ in pairs])
    ps = np.concatenate([np.asarray(s, np.int64) for s, _ in pairs] + [np.empty(0, np.int64)])
    pe = np.concatenate([np.asarray(e, np.int64) for _, e in pairs] + [np.empty(0, np.int64)])
    return ps, pe, offs


@njit(cache=True)
def _scan_hits(bs, be, br, ps, pe, offs, mark, prim, re_code, cell, trow, reg, cel, tru):
    """Blocks of one strand sorted by start against each tag's sorted disjoint pieces; same test as intervals.blocks_hit.

    The first record hit for tag t is marked, and if primary counted once in reg[t, re], cel[t, cell] and tru[row, t].
    """
    nb = bs.size
    if nb == 0:
        return
    for t in range(offs.size - 1):
        lo, hi = offs[t], offs[t + 1]
        if lo == hi:
            continue
        j = lo + np.searchsorted(pe[lo:hi], bs[0], side="right")
        for k in range(nb):
            s = bs[k]
            while j < hi and pe[j] <= s:
                j += 1
            if j == hi:
                break
            if ps[j] < be[k]:
                r = br[k]
                if mark[r] != t:
                    mark[r] = t
                    if prim[r]:
                        reg[t, re_code[r]] += 1
                        if cell[r] >= 0:
                            cel[t, cell[r]] += 1
                        if trow[r] >= 0:
                            tru[trow[r], t] = True


class _Chunk:
    """Fixed numpy buffers for one chunk of records and their blocks."""

    def __init__(self, chunk):
        self.flag = np.empty(chunk, np.int64)
        self.re = np.empty(chunk, np.int64)
        self.cell = np.empty(chunk, np.int64)
        self.trow = np.empty(chunk, np.int64)
        self._blocks(4 * chunk + 64)
        self.nr = self.nb = 0
        self.t_qname, self.t_pos, self.t_flag = [], [], []

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
        self.t_qname, self.t_pos, self.t_flag = [], [], []


def scan_reference(bam_path, fasta_path, ref, transcripts, settings, truth_qnames, cells=None, chunk=1_000_000):
    """(truth_df, region_df, cell_df) for one reference; region_df.attrs["records"] is the number of records fetched.

    truth_df: primary records whose qname is in truth_qnames, fetch order: qname, ref, pos, flag + one bool per tag.
    region_df: tag, re, removed, total over primary records. cell_df: index cb (sorted cells), total then one
    uint32 per tag over primary records with a called-cell CB; None when cells is None.
    """
    with pysam.FastaFile(str(fasta_path)) as fa:
        seq = fa.fetch(ref)
    tx_by_strand = {strd: transcripts.get((ref, strd), []) for strd in (True, False)}
    pieces = reference_pieces(seq, tx_by_strand, coverage_prefix_chunked(bam_path, ref), settings)
    del seq
    tags = [setting_tag(s) for s in settings]
    packed = {strd: _pack([pieces.pop((t, strd)) for t in tags]) for strd in (True, False)}
    cells_sorted = [] if cells is None else sorted(cells)
    cell_idx = {c: i for i, c in enumerate(cells_sorted)}
    n_tags = len(tags)
    re_idx = {c: i for i, c in enumerate(RE_CODES)}
    reg = np.zeros((n_tags, len(RE_CODES)), np.int64)
    reg_total = np.zeros(len(RE_CODES), np.int64)
    cel = np.zeros((n_tags, len(cells_sorted)), np.uint32)
    cel_total = np.zeros(len(cells_sorted), np.int64)
    truth_parts = []
    n_records = 0

    def flush(c):
        nonlocal reg, reg_total
        if len(re_idx) > reg.shape[1]:
            grow = len(re_idx) - reg.shape[1]
            reg = np.pad(reg, ((0, 0), (0, grow)))
            reg_total = np.pad(reg_total, (0, grow))
        nr, nb = c.nr, c.nb
        flags, re_code, cell, trow = c.flag[:nr], c.re[:nr], c.cell[:nr], c.trow[:nr]
        prim = (flags & SECONDARY_OR_SUPPLEMENTARY) == 0
        reg_total += np.bincount(re_code[prim], minlength=reg.shape[1])
        in_cell = prim & (cell >= 0)
        cel_total[:] += np.bincount(cell[in_cell], minlength=len(cells_sorted))
        tru = np.zeros((len(c.t_qname), n_tags), np.bool_)
        mark = np.full(nr, -1, np.int64)
        bs, be, br = c.bs[:nb], c.be[:nb], c.br[:nb]
        bfwd = (flags[br] & 16) == 0
        for strd in (True, False):
            sel = np.flatnonzero(bfwd == strd)
            o = sel[np.argsort(bs[sel], kind="stable")]
            _scan_hits(bs[o], be[o], br[o], *packed[strd], mark, prim, re_code, cell, trow, reg, cel, tru)
        head = pd.DataFrame({
            "qname": pd.Series(c.t_qname, dtype="str"),
            "ref": pd.Series([ref] * len(c.t_qname), dtype="str"),
            "pos": np.asarray(c.t_pos, np.int64),
            "flag": np.asarray(c.t_flag, np.int64),
        })
        truth_parts.append(pd.concat([head, pd.DataFrame(tru, columns=tags)], axis=1))
        c.reset()

    c = _Chunk(chunk)
    with pysam.AlignmentFile(str(bam_path)) as bam:
        for a in bam.fetch(ref):
            n_records += 1
            blocks = a.get_blocks()
            if c.nr == chunk or not c.fits(len(blocks)):
                flush(c)
                c.fits(len(blocks))
            i = c.nr
            flag = a.flag
            c.flag[i] = flag
            try:
                re_ = a.get_tag("RE")
            except KeyError:
                re_ = ""
            code = re_idx.get(re_)
            if code is None:
                code = re_idx[re_] = len(re_idx)
            c.re[i] = code
            if cell_idx:
                try:
                    c.cell[i] = cell_idx.get(a.get_tag("CB"), -1)
                except KeyError:
                    c.cell[i] = -1
            else:
                c.cell[i] = -1
            if not flag & SECONDARY_OR_SUPPLEMENTARY and a.query_name in truth_qnames:
                c.trow[i] = len(c.t_qname)
                c.t_qname.append(a.query_name)
                c.t_pos.append(a.reference_start)
                c.t_flag.append(flag)
            else:
                c.trow[i] = -1
            nb = c.nb
            for s, e in blocks:
                c.bs[nb] = s
                c.be[nb] = e
                c.br[nb] = i
                nb += 1
            c.nb = nb
            c.nr = i + 1
    flush(c)

    truth_df = pd.concat(truth_parts, ignore_index=True)
    codes = sorted(re_idx, key=re_idx.get)
    seen = [k for k in range(len(codes)) if reg_total[k] > 0]
    region_df = pd.DataFrame({
        "tag": np.repeat(np.array(tags, dtype=object), len(seen)),
        "re": [codes[k] for _ in tags for k in seen],
        "removed": reg[:, seen].reshape(-1),
        "total": np.tile(reg_total[seen], n_tags),
    })
    region_df["tag"] = region_df["tag"].astype("str")
    region_df["re"] = region_df["re"].astype("str")
    region_df.attrs["records"] = n_records
    cell_df = None
    if cells is not None:
        cell_df = pd.DataFrame(cel.T, index=pd.Index(cells_sorted, name="cb", dtype="str"), columns=tags)
        cell_df.insert(0, "total", cel_total.astype(np.uint32))
    return truth_df, region_df, cell_df


def scan_to_parts(bam_path, fasta_path, ref, transcripts, settings, truth_qnames, cells, part_dir, chunk=1_000_000,
                  meta=None):
    """scan_reference written to part_dir as <ref>.truth/.region/.cells.parquet, then <ref>.json (records, primary, seconds,
    plus the entries of meta, which name the inputs; see stale_parts).

    The json is written last, so a reference whose json exists is complete: it is not scanned again and its stored
    stats are returned (resume after a crash). No cells part when cells is None.
    """
    part_dir = Path(part_dir)
    done = part_dir / f"{ref}.json"
    if done.exists():
        return json.loads(done.read_text())
    t0 = time.monotonic()
    truth, region, cell = scan_reference(bam_path, fasta_path, ref, transcripts, settings, truth_qnames, cells, chunk)
    part_dir.mkdir(parents=True, exist_ok=True)
    truth.to_parquet(part_dir / f"{ref}.truth.parquet", index=False)
    region.to_parquet(part_dir / f"{ref}.region.parquet", index=False)
    if cell is not None:
        cell.to_parquet(part_dir / f"{ref}.cells.parquet")
    stats = {"ref": ref, "records": region.attrs["records"], "primary": int(region.drop_duplicates("re").total.sum()),
             "seconds": round(time.monotonic() - t0, 1)} | (meta or {})
    done.write_text(json.dumps(stats))
    return stats


def stale_parts(part_dir, refs, meta):
    """refs whose finished part (<ref>.json) was written with other inputs than meta (or without them), in refs order."""
    stale = []
    for ref in refs:
        done = Path(part_dir) / f"{ref}.json"
        if done.exists():
            stored = json.loads(done.read_text())
            if any(stored.get(k) != v for k, v in meta.items()):
                stale.append(ref)
    return stale


def merge_regions(parts, tags):
    """Region rows (tag, re, removed, total) of several references summed per tag and RE, in tags order then RE."""
    region = pd.concat(parts, ignore_index=True)
    region = region.groupby(["tag", "re"], sort=False, as_index=False)[["removed", "total"]].sum()
    rank = region.tag.map({t: i for i, t in enumerate(tags)})
    return region.iloc[np.lexsort((region.re.to_numpy(), rank.to_numpy()))].reset_index(drop=True)


def merge_parts(part_dir, refs, tags, out_dir):
    """Merge the parts of refs into out_dir and return the summary dict (per-ref stats, totals, removed primary per tag).

    truth_flags.parquet: truth rows in refs order, streamed one reference at a time. region_counts.parquet: summed
    over refs, rows in tags order then RE. cell_counts.parquet: summed over refs, only when every ref has a cells part.
    """
    part_dir, out_dir = Path(part_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("qname", pa.string()), ("ref", pa.string()), ("pos", pa.int64()), ("flag", pa.int64())]
                       + [(t, pa.bool_()) for t in tags])
    with pq.ParquetWriter(out_dir / "truth_flags.parquet", schema) as w:
        for ref in refs:
            w.write_table(pq.read_table(part_dir / f"{ref}.truth.parquet").cast(schema))
    region = merge_regions([pd.read_parquet(part_dir / f"{ref}.region.parquet") for ref in refs], tags)
    region.to_parquet(out_dir / "region_counts.parquet", index=False)
    cell_paths = [part_dir / f"{ref}.cells.parquet" for ref in refs]
    if all(p.exists() for p in cell_paths):
        cell = pd.read_parquet(cell_paths[0])
        for p in cell_paths[1:]:
            cell += pd.read_parquet(p)
        cell.to_parquet(out_dir / "cell_counts.parquet")
    per_ref = {}
    for ref in refs:
        st = json.loads((part_dir / f"{ref}.json").read_text())
        per_ref[ref] = {k: st[k] for k in ("records", "primary", "seconds")}
    removed = region.groupby("tag").removed.sum()
    return {
        "per_ref": per_ref,
        "records": sum(v["records"] for v in per_ref.values()),
        "primary": sum(v["primary"] for v in per_ref.values()),
        "removed": {t: int(removed.get(t, 0)) for t in tags},
    }
