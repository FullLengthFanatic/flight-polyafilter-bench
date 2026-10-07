import pysam


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
