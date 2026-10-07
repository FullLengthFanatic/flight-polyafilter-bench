import pysam
from collections import namedtuple

import numpy as np
import pandas as pd

from .intervals import blocks_hit, flatten, intersect
from .snr import scan_snrs, snr_pieces
from .transcripts import coverage_prefix, expressed, start_pieces


def record_key(a):
    return (a.query_name, a.flag, a.reference_name, a.reference_start, a.cigarstring)


def removed_keys_by_ref(in_bam, out_bam):
    """Yield (ref, removed record keys) per reference. out_bam must keep an ordered subsequence of in_bam's records."""
    with pysam.AlignmentFile(str(in_bam)) as fin, pysam.AlignmentFile(str(out_bam)) as fout:
        it = fout.fetch(until_eof=True)
        nxt = next(it, None)
        cur_ref, cur = None, set()
        first = True
        for a in fin.fetch(until_eof=True):
            if first or a.reference_name != cur_ref:
                if not first:
                    yield cur_ref, cur
                cur_ref, cur, first = a.reference_name, set(), False
            if nxt is not None and record_key(a) == record_key(nxt):
                nxt = next(it, None)
            else:
                cur.add(record_key(a))
        if not first:
            yield cur_ref, cur
        if nxt is not None:
            raise ValueError("output BAM is not an ordered subsequence of the input BAM")


def removed_keys(in_bam, out_bam):
    out = set()
    for _, keys in removed_keys_by_ref(in_bam, out_bam):
        out |= keys
    return out


Setting = namedtuple("Setting", "covlen min_len mism introns")


def setting_tag(s):
    lname = "none" if s.min_len is None else str(s.min_len)
    return f"c{s.covlen}_l{lname}_m{s.mism}_{'i' if s.introns else 'e'}"


def parse_setting(text):
    c, l, m, mode = text.split(",")
    return Setting(int(c), None if l == "none" else int(l), int(m), mode == "i")


def flag_reference(bam_path, fasta_path, ref, transcripts, settings):
    """Per-record removal decision on one reference for each setting, in fetch order."""
    prefix = coverage_prefix(bam_path, ref)
    with pysam.FastaFile(str(fasta_path)) as fa:
        seq = fa.fetch(ref)
    ref_len = len(seq)
    rows, rec_fwd, blk_rec, blk_s, blk_e = [], [], [], [], []
    with pysam.AlignmentFile(str(bam_path)) as bam:
        for i, a in enumerate(bam.fetch(ref)):
            rows.append((a.query_name, a.flag, ref, a.reference_start, a.cigarstring))
            rec_fwd.append(not a.is_reverse)
            for s, e in a.get_blocks():
                blk_rec.append(i)
                blk_s.append(s)
                blk_e.append(e)
    df = pd.DataFrame(rows, columns=["qname", "flag", "ref", "pos", "cigar"])
    rec_fwd = np.asarray(rec_fwd, bool)
    blk_rec = np.asarray(blk_rec, np.int64)
    blk_s = np.asarray(blk_s, np.int64)
    blk_e = np.asarray(blk_e, np.int64)
    blk_fwd = rec_fwd[blk_rec] if blk_rec.size else np.zeros(0, bool)
    snr_cache, body_cache = {}, {}
    for st in settings:
        removed = np.zeros(len(df), bool)
        for strd in (True, False):
            bkey = (strd, st.covlen, st.introns)
            if bkey not in body_cache:
                ex = expressed(transcripts.get((ref, strd), []), prefix[strd], st.introns)
                body_cache[bkey] = start_pieces(ex, st.covlen, st.introns, strd)
            body_s, body_e = body_cache[bkey]
            if st.min_len is None:
                ps, pe = body_s, body_e
            else:
                skey = (strd, st.mism)
                if skey not in snr_cache:
                    snr_cache[skey] = scan_snrs(seq, strd, st.mism)
                first, last = snr_cache[skey]
                ws, we = flatten(*snr_pieces(first, last, st.min_len, st.covlen, strd, ref_len))
                ps, pe = intersect(body_s, body_e, ws, we)
            sel = blk_fwd == strd
            hit = blocks_hit(blk_s[sel], blk_e[sel], ps, pe)
            removed[np.unique(blk_rec[sel][hit])] = True
        df[setting_tag(st)] = removed
    return df


def removed_from_flags(df, tag):
    d = df[df[tag]]
    return set(zip(d.qname, d.flag, d.ref, d.pos, d.cigar))
