import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pysam
import pytest

from pafbench import stream
from pafbench.config import REPO
from pafbench.flags import Setting, flag_reference, setting_tag
from pafbench.grid import all_settings
from pafbench.intervals import blocks_hit, flatten, intersect
from pafbench.snr import scan_snrs, snr_pieces
from pafbench.stream import (body_pieces, coverage_prefix_chunked, merge_parts, qname_digest, reference_pieces, scan_reference,
                             scan_to_parts, stale_parts)
from pafbench.transcripts import coverage_prefix, expressed, load_transcripts, start_pieces

SETTINGS = [
    Setting(200, None, 0, True),
    Setting(200, None, 0, False),
    Setting(100, 5, 0, False),
    Setting(300, 8, 1, False),
    Setting(300, 8, 1, True),
    Setting(150, 3, 2, True),
    Setting(150, 3, 2, False),
    Setting(50, 10, 0, True),
    Setting(450, 12, 2, False),
    Setting(600, 9, 1, True),
]
TAGS = [setting_tag(s) for s in SETTINGS]
CELLS = ["GGGTTT-1", "AAACCC-1", "CCCGGG-1"]           # deliberately unsorted
CB_CYCLE = ["AAACCC-1", "CCCGGG-1", "GGGTTT-1", "TTTAAA-1", None]   # TTTAAA-1 is not a called cell
RE_CYCLE = ["E", "N", "I", None]


@pytest.fixture(scope="module")
def tagged(synth, tmp_path_factory):
    """Copy of the synthetic BAM with CB, UB, RE, some supplementary and QC-fail flags, and secondaries sharing a primary's qname."""
    path = tmp_path_factory.mktemp("stream") / "tagged.bam"
    with pysam.AlignmentFile(str(synth.bam)) as fin, pysam.AlignmentFile(str(path), "wb", template=fin) as out:
        prev = None
        for i, a in enumerate(fin.fetch(until_eof=True)):
            cb = CB_CYCLE[i % len(CB_CYCLE)]
            if cb is not None:
                a.set_tag("CB", cb)
            a.set_tag("UB", f"U{i % 7}")
            re_ = "X" if i % 17 == 16 else RE_CYCLE[i % len(RE_CYCLE)]
            if re_ is not None:
                a.set_tag("RE", re_)
            if i % 11 == 5:
                a.flag |= 2048
            if i % 13 == 7:
                a.flag |= 512
            if a.is_secondary and prev is not None:
                a.query_name = prev
            prev = a.query_name
            out.write(a)
    pysam.index(str(path))
    return path


def _expected(bam, fasta, ref, tx, cells):
    """flag_reference rows of primary records, with CB and RE read independently in fetch order."""
    df = flag_reference(bam, fasta, ref, tx, SETTINGS)
    with pysam.AlignmentFile(str(bam)) as fh:
        tags = [(a.get_tag("CB") if a.has_tag("CB") else None, a.get_tag("RE") if a.has_tag("RE") else "") for a in fh.fetch(ref)]
    df["cb"] = [t[0] for t in tags]
    df["re"] = [t[1] for t in tags]
    return df[(df.flag & 0x900) == 0].reset_index(drop=True)


def _region_long(prim):
    rows = []
    for re_, g in prim.groupby("re", sort=False):
        for t in TAGS:
            rows.append((t, re_, int(g[t].sum()), len(g)))
    return sorted(rows)


def _cells_expected(prim, cells):
    sub = prim[prim.cb.isin(cells)]
    idx = sorted(cells)
    want = sub.groupby("cb")[TAGS].sum().reindex(idx, fill_value=0)
    want.insert(0, "total", sub.groupby("cb").size().reindex(idx, fill_value=0))
    return want


def test_coverage_prefix_chunked_equals_coverage_prefix(synth):
    for ref in ("chrA", "chrB"):
        want = coverage_prefix(synth.bam, ref)
        for chunk in (1, 5, 2_000_000):
            got = coverage_prefix_chunked(synth.bam, ref, chunk=chunk)
            for strd in (True, False):
                assert np.array_equal(got[strd], want[strd]), (ref, chunk, strd)


def test_coverage_prefix_chunked_iupac_and_deletions(tmp_path):
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "c", "LN": 60}]})
    path = tmp_path / "x.bam"
    recs = [("a", 5, "4M2D4M", 0, "ACRTACGT"), ("d", 5, "3M", 256, None), ("b", 20, "3M", 16, "ANA"), ("c", 30, "2M10N2M", 0, "ACGT")]
    with pysam.AlignmentFile(str(path), "wb", header=hdr) as out:
        for name, pos, cigar, flag, seq in recs:
            x = pysam.AlignedSegment(hdr)
            x.query_name, x.reference_name, x.reference_start, x.cigarstring, x.flag = name, "c", pos, cigar, flag
            if seq is not None:
                x.query_sequence, x.query_qualities = seq, pysam.qualitystring_to_array("I" * len(seq))
            out.write(x)
    pysam.index(str(path))
    want = coverage_prefix(path, "c")
    for chunk in (1, 2, 100):
        got = coverage_prefix_chunked(path, "c", chunk=chunk)
        for strd in (True, False):
            assert np.array_equal(got[strd], want[strd])
    cov = np.diff(want[True])
    # forward: a covers 5-8 and 11-14 except the R at 7; c covers 30-31 and 42-43; d has no SEQ
    assert np.flatnonzero(cov).tolist() == [5, 6, 8, 11, 12, 13, 14, 30, 31, 42, 43]
    # reverse: b covers 20-22 except the N at 21
    assert np.flatnonzero(np.diff(want[False])).tolist() == [20, 22]


def test_body_pieces_equal_start_pieces(synth):
    tx = load_transcripts(synth.gtf)
    rng = np.random.default_rng(11)
    sizes = []
    for strd in (True, False):
        rand = []
        for _ in range(300):
            pos, exons = int(rng.integers(0, 200_000)), []
            for _ in range(int(rng.integers(1, 6))):
                ln = int(rng.integers(20, 600))
                exons.append((pos, pos + ln))
                pos += ln + int(rng.integers(30, 4000))
            rand.append((exons[0][0], exons[-1][1], sorted(exons, reverse=strd)))
        sets = [rand] + [tx.get((ref, strd), []) for ref in ("chrA", "chrB")]
        for txs in sets:
            for covlen in (50, 150, 300, 600):
                for introns in (False, True):
                    want = start_pieces(txs, covlen, introns, strd)
                    got = body_pieces(txs, covlen, introns, strd)
                    assert got[0].tolist() == want[0].tolist() and got[1].tolist() == want[1].tolist()
                    sizes.append(want[0].size)
    assert min(sizes[:8]) > 50 and max(sizes[8:16]) > 0


def test_reference_pieces_equal_flag_reference_pieces(synth):
    tx = load_transcripts(synth.gtf)
    with pysam.FastaFile(str(synth.fasta)) as fa:
        seqs = {r: fa.fetch(r) for r in ("chrA", "chrB")}
    for ref in ("chrA", "chrB"):
        prefix = coverage_prefix(synth.bam, ref)
        tx_by_strand = {strd: tx.get((ref, strd), []) for strd in (True, False)}
        got = reference_pieces(seqs[ref], tx_by_strand, prefix, SETTINGS)
        assert set(got) == {(t, strd) for t in TAGS for strd in (True, False)}
        n_nonempty = 0
        for st in SETTINGS:
            for strd in (True, False):
                # recomputed exactly as flag_reference does, without caches
                ex = expressed(tx_by_strand[strd], prefix[strd], st.introns)
                bs, be = start_pieces(ex, st.covlen, st.introns, strd)
                if st.min_len is None:
                    ws, we = bs, be
                else:
                    first, last = scan_snrs(seqs[ref], strd, st.mism)
                    fs, fe = flatten(*snr_pieces(first, last, st.min_len, st.covlen, strd, len(seqs[ref])))
                    ws, we = intersect(bs, be, fs, fe)
                ps, pe = got[(setting_tag(st), strd)]
                assert ps.tolist() == ws.tolist() and pe.tolist() == we.tolist(), (ref, st, strd)
                n_nonempty += ps.size > 0
        assert n_nonempty > 0


def test_hit_kernel_edges_by_hand():
    # pieces of one tag: [10, 20) and [30, 40). Blocks (sorted by start) and whether each touches a piece:
    # [5, 10) no (ends where the piece starts); [8, 11) yes; [19, 20) yes; [20, 30) no (touches both, overlaps neither);
    # [25, 31) yes; [40, 50) no.
    bs = np.array([5, 8, 19, 20, 25, 40], np.int64)
    be = np.array([10, 11, 20, 30, 31, 50], np.int64)
    br = np.arange(6, dtype=np.int64)
    ps, pe, offs = stream._pack([(np.array([10, 30]), np.array([20, 40]))])
    tru = _kernel_matrix(bs, be, br, 6, ps, pe, offs)
    assert tru[:, 0].tolist() == [False, True, True, False, True, False]
    assert tru[:, 0].tolist() == blocks_hit(bs, be, np.array([10, 30]), np.array([20, 40])).tolist()


def _kernel_matrix(bs, be, br, n_rec, ps, pe, offs):
    """Per-record hits from the kernel: every record primary, its own truth row, region code 0, no cell."""
    n_tags = offs.size - 1
    tru = np.zeros((n_rec, n_tags), bool)
    reg = np.zeros((n_tags, 1), np.int64)
    cel = np.zeros((n_tags, 0), np.uint32)
    stream._scan_hits(bs, be, br, ps, pe, offs, np.full(n_rec, -1, np.int64), np.ones(n_rec, bool),
                      np.zeros(n_rec, np.int64), np.full(n_rec, -1, np.int64), np.arange(n_rec, dtype=np.int64), reg, cel, tru)
    assert reg[:, 0].tolist() == tru.sum(axis=0).tolist()
    return tru


def test_hit_kernel_matches_blocks_hit_random():
    rng = np.random.default_rng(3)
    n_rec, n_tags = 400, 12
    starts = np.sort(rng.integers(0, 5000, n_rec))
    blk_s, blk_e, blk_r = [], [], []
    for r, s in enumerate(starts):
        pos = int(s)
        for _ in range(rng.integers(1, 4)):
            ln = int(rng.integers(1, 60))
            blk_s.append(pos)
            blk_e.append(pos + ln)
            blk_r.append(r)
            pos += ln + int(rng.integers(1, 300))
    blk_s, blk_e, blk_r = (np.array(x, np.int64) for x in (blk_s, blk_e, blk_r))
    pieces = []
    for t in range(n_tags):
        k = int(rng.integers(0, 60))
        s = rng.integers(0, 5600, k)
        pieces.append(flatten(s, s + rng.integers(1, 80 * (t + 1), k)))
    ps, pe, offs = stream._pack(pieces)
    order = np.argsort(blk_s, kind="stable")
    got = _kernel_matrix(blk_s[order], blk_e[order], blk_r[order], n_rec, ps, pe, offs)
    want = np.zeros((n_rec, n_tags), bool)
    for t, (s, e) in enumerate(pieces):
        want[np.unique(blk_r[blocks_hit(blk_s, blk_e, s, e)]), t] = True
    assert np.array_equal(got, want)
    assert 0 < want.sum() < want.size


@pytest.mark.parametrize("chunk", [3, 1000])
def test_scan_reference_equals_flag_reference(synth, tagged, chunk):
    tx = load_transcripts(synth.gtf)
    with pysam.AlignmentFile(str(tagged)) as fh:
        qnames = {a.query_name for a in fh.fetch(until_eof=True)}
    n_removed = 0
    for ref in ("chrA", "chrB"):
        prim = _expected(tagged, synth.fasta, ref, tx, CELLS)
        truth, region, cells = scan_reference(tagged, synth.fasta, ref, tx, SETTINGS, qnames, cells=CELLS, chunk=chunk)
        assert list(truth.columns) == ["qname", "ref", "pos", "flag"] + TAGS
        assert truth.qname.tolist() == prim.qname.tolist()
        assert (truth.ref == ref).all()
        assert truth.pos.tolist() == prim.pos.tolist() and truth.flag.tolist() == prim.flag.tolist()
        assert np.array_equal(truth[TAGS].to_numpy(), prim[TAGS].to_numpy())
        assert sorted(region[["tag", "re", "removed", "total"]].itertuples(index=False, name=None)) == _region_long(prim)
        want = _cells_expected(prim, CELLS)
        assert cells.index.name == "cb" and cells.index.tolist() == sorted(CELLS)
        assert list(cells.columns) == ["total"] + TAGS
        assert (cells.dtypes == np.uint32).all()
        assert np.array_equal(cells.to_numpy(np.int64), want.to_numpy(np.int64))
        assert region.attrs["records"] == len(flag_reference(tagged, synth.fasta, ref, tx, SETTINGS[:1]))
        n_removed += int(prim[TAGS].to_numpy().sum())
        assert set(region.re) <= {"E", "N", "I", "", "X"}
    assert n_removed > 0


def test_scan_reference_truth_subset_and_no_cells(synth, tagged):
    tx = load_transcripts(synth.gtf)
    full, region_full, _ = scan_reference(tagged, synth.fasta, "chrA", tx, SETTINGS, set(_qnames(tagged, "chrA")), cells=CELLS)
    keep = set(full.qname.iloc[::3]) | {"no_such_read"}
    sub, region_sub, cells = scan_reference(tagged, synth.fasta, "chrA", tx, SETTINGS, keep, chunk=4)
    want = full[full.qname.isin(keep)].reset_index(drop=True)
    pd.testing.assert_frame_equal(sub, want)
    assert 0 < len(sub) < len(full)
    assert cells is None
    pd.testing.assert_frame_equal(region_sub, region_full)


def test_secondary_sharing_a_truth_qname_is_not_a_truth_row(synth, tagged):
    tx = load_transcripts(synth.gtf)
    with pysam.AlignmentFile(str(tagged)) as fh:
        sec = [a for a in fh.fetch("chrA") if a.is_secondary]
        prim_names = {a.query_name for a in fh.fetch("chrA") if not a.is_secondary and not a.is_supplementary}
    shared = {a.query_name for a in sec} & prim_names
    assert shared
    truth, _, _ = scan_reference(tagged, synth.fasta, "chrA", tx, SETTINGS[:2], shared)
    assert sorted(truth.qname) == sorted(shared)
    assert ((truth.flag & 0x900) == 0).all()


def _qnames(bam, ref):
    with pysam.AlignmentFile(str(bam)) as fh:
        return [a.query_name for a in fh.fetch(ref)]


def test_parts_and_merge_equal_flag_reference(synth, tagged, tmp_path):
    tx = load_transcripts(synth.gtf)
    with pysam.AlignmentFile(str(tagged)) as fh:
        qnames = {a.query_name for a in fh.fetch(until_eof=True)}
    part_dir = tmp_path / "parts"
    stats = {ref: scan_to_parts(tagged, synth.fasta, ref, tx, SETTINGS, qnames, CELLS, part_dir, chunk=5) for ref in ("chrB", "chrA")}
    summary = merge_parts(part_dir, ["chrA", "chrB"], TAGS, tmp_path)
    full = {ref: flag_reference(tagged, synth.fasta, ref, tx, SETTINGS) for ref in ("chrA", "chrB")}
    prims = [_expected(tagged, synth.fasta, ref, tx, CELLS) for ref in ("chrA", "chrB")]
    prim = pd.concat(prims, ignore_index=True)

    truth = pd.read_parquet(tmp_path / "truth_flags.parquet")
    assert list(truth.columns) == ["qname", "ref", "pos", "flag"] + TAGS
    assert truth.qname.tolist() == prim.qname.tolist() and truth.ref.tolist() == prim.ref.tolist()
    assert truth.pos.tolist() == prim.pos.tolist() and truth.flag.tolist() == prim.flag.tolist()
    assert np.array_equal(truth[TAGS].to_numpy(), prim[TAGS].to_numpy())

    region = pd.read_parquet(tmp_path / "region_counts.parquet")
    # rows in settings order, then RE in string order
    assert list(region.itertuples(index=False, name=None)) == sorted(_region_long(prim), key=lambda r: (TAGS.index(r[0]), r[1]))

    cells = pd.read_parquet(tmp_path / "cell_counts.parquet")
    assert cells.index.name == "cb" and cells.index.tolist() == sorted(CELLS)
    assert list(cells.columns) == ["total"] + TAGS and (cells.dtypes == np.uint32).all()
    assert np.array_equal(cells.to_numpy(np.int64), _cells_expected(prim, CELLS).to_numpy(np.int64))

    for ref, p, f in zip(("chrA", "chrB"), prims, (full["chrA"], full["chrB"])):
        assert summary["per_ref"][ref]["records"] == stats[ref]["records"] == len(f)
        assert summary["per_ref"][ref]["primary"] == stats[ref]["primary"] == len(p)
    assert summary["records"] == len(full["chrA"]) + len(full["chrB"])
    assert summary["primary"] == len(prim)
    assert summary["removed"] == {t: int(prim[t].sum()) for t in TAGS}
    # chrA has 145 records, 5 of them secondary (flag 256); the tagged copy adds supplementary bits on top
    assert len(full["chrA"]) == 145 and len(prims[0]) < 140


def test_parts_resume_and_no_cells(synth, tagged, tmp_path, monkeypatch):
    tx = load_transcripts(synth.gtf)
    keep = set(_qnames(tagged, "chrA")[::4])
    part_dir = tmp_path / "parts"
    first = {ref: scan_to_parts(tagged, synth.fasta, ref, tx, SETTINGS, keep, None, part_dir) for ref in ("chrA", "chrB")}
    assert not list(part_dir.glob("*.cells.parquet"))

    def boom(*args, **kwargs):
        raise AssertionError("scan_reference ran again")

    monkeypatch.setattr(stream, "scan_reference", boom)
    again = {ref: scan_to_parts(tagged, synth.fasta, ref, tx, SETTINGS, keep, None, part_dir) for ref in ("chrA", "chrB")}
    assert again == first
    summary = merge_parts(part_dir, ["chrA", "chrB"], TAGS, tmp_path)
    assert not (tmp_path / "cell_counts.parquet").exists()
    truth = pd.read_parquet(tmp_path / "truth_flags.parquet")
    want = _expected(tagged, synth.fasta, "chrA", tx, CELLS)
    want = want[want.qname.isin(keep)].reset_index(drop=True)
    # chrB contributes no truth rows; the chrA rows come through with their string and integer types
    assert truth.qname.tolist() == want.qname.tolist() and (truth.ref == "chrA").all()
    assert np.array_equal(truth[TAGS].to_numpy(), want[TAGS].to_numpy())
    assert truth.pos.dtype == np.int64 and truth.flag.dtype == np.int64
    assert summary["primary"] == len(_expected(tagged, synth.fasta, "chrA", tx, CELLS)) + len(_expected(tagged, synth.fasta, "chrB", tx, CELLS))


def _write_bam(path, contigs, recs):
    """Sorted, indexed BAM from (qname, ref, pos, cigar, flag, seq or None, tags) tuples."""
    hdr = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6", "SO": "coordinate"},
                                           "SQ": [{"SN": c, "LN": n} for c, n in contigs.items()]})
    raw = path.with_suffix(".unsorted.bam")
    with pysam.AlignmentFile(str(raw), "wb", header=hdr) as out:
        for name, ref, pos, cigar, flag, seq, tags in recs:
            a = pysam.AlignedSegment(hdr)
            a.query_name, a.reference_name, a.reference_start, a.cigarstring, a.flag = name, ref, pos, cigar, flag
            a.mapping_quality = 255
            if seq is not None:
                a.query_sequence, a.query_qualities = seq, pysam.qualitystring_to_array("I" * len(seq))
            for k, v in tags.items():
                a.set_tag(k, v)
            out.write(a)
    pysam.sort("-o", str(path), str(raw))
    pysam.index(str(path))
    return path


def test_expressed_modes_intron_only_read(tmp_path):
    # One + transcript on contig c: exons [100, 300) and [1000, 1200) (0-based), intron [300, 1000); COVLEN 200.
    # -i: the last 200 nt [1000, 1200) are protected, body [100, 1000). Default mode: the exonic last 200 nt
    # [1000, 1200) are protected, body [100, 300).
    # Forward reads: s0 [150, 210) in exon 1 without SEQ (adds no coverage), i1 [500, 560) inside the intron with SEQ,
    # f2 [2000, 2060) outside the transcript. Only i1 covers bases, so the transcript is expressed with -i (its span is
    # covered) and not in default mode (no exonic base covered).
    rng = np.random.default_rng(5)
    fasta = tmp_path / "c.fa"
    fasta.write_text(">c\n" + "".join(rng.choice(list("ACGT"), 3000)) + "\n")
    pysam.faidx(str(fasta))
    gtf = tmp_path / "c.gtf"
    attr = 'gene_id "gx"; transcript_id "tx";'
    gtf.write_text("".join(f"c\tt\t{ft}\t{s}\t{e}\t.\t+\t.\t{attr}\n"
                           for ft, s, e in (("transcript", 101, 1200), ("exon", 101, 300), ("exon", 1001, 1200))))
    seq = "".join(rng.choice(list("ACGT"), 60))
    bam = _write_bam(tmp_path / "c.bam", {"c": 3000}, [("s0", "c", 150, "60M", 0, None, {}),
                                                         ("i1", "c", 500, "60M", 0, seq, {}),
                                                         ("f2", "c", 2000, "60M", 0, seq, {})])
    tx = load_transcripts(gtf)
    prefix = coverage_prefix(bam, "c")
    assert len(expressed(tx[("c", True)], prefix[True], True)) == 1
    assert expressed(tx[("c", True)], prefix[True], False) == []
    settings = [Setting(200, None, 0, True), Setting(200, None, 0, False), Setting(200, 3, 1, True), Setting(200, 3, 1, False)]
    tags = [setting_tag(s) for s in settings]
    want = flag_reference(bam, fasta, "c", tx, settings)
    for chunk in (1, 1000):
        truth, region, _ = scan_reference(bam, fasta, "c", tx, settings, {"s0", "i1", "f2"}, chunk=chunk)
        assert truth.qname.tolist() == ["s0", "i1", "f2"]
        assert truth["c200_lnone_m0_i"].tolist() == [True, True, False]
        assert truth["c200_lnone_m0_e"].tolist() == [False, False, False]
        for t in tags:
            assert truth[t].tolist() == want[t].tolist(), t
        assert region.set_index("tag").removed.to_dict() == {t: int(want[t].sum()) for t in tags}


@pytest.mark.parametrize("capacity", [1, 2])
def test_scan_reference_with_tiny_block_buffers(synth, tagged, monkeypatch, capacity):
    """Block buffers of 1 or 2 blocks: flushes on a full buffer, and (capacity 1) grows it for a two-block record."""
    caps, flushes = [], []
    init, blocks, hits = stream._Chunk.__init__, stream._Chunk._blocks, stream._scan_hits

    def spy_blocks(self, cap):
        caps.append(cap)
        blocks(self, cap)

    def small(self, chunk):
        init(self, chunk)
        self._blocks(capacity)

    def spy_hits(*args):
        flushes.append(1)
        hits(*args)

    monkeypatch.setattr(stream._Chunk, "_blocks", spy_blocks)
    monkeypatch.setattr(stream._Chunk, "__init__", small)
    monkeypatch.setattr(stream, "_scan_hits", spy_hits)
    tx = load_transcripts(synth.gtf)
    prim = _expected(tagged, synth.fasta, "chrA", tx, CELLS)
    for chunk in (2, 1000):
        caps.clear()
        flushes.clear()
        truth, region, cells = scan_reference(tagged, synth.fasta, "chrA", tx, SETTINGS, set(prim.qname), cells=CELLS,
                                              chunk=chunk)
        assert truth.qname.tolist() == prim.qname.tolist()
        assert np.array_equal(truth[TAGS].to_numpy(), prim[TAGS].to_numpy())
        assert sorted(region[["tag", "re", "removed", "total"]].itertuples(index=False, name=None)) == _region_long(prim)
        assert np.array_equal(cells.to_numpy(np.int64), _cells_expected(prim, CELLS).to_numpy(np.int64))
        # 145 records on chrA; a buffer of at most 4 blocks forces far more flushes (two kernel calls each) than chunk 1000
        assert len(flushes) > 2 * 30
        if capacity == 1:
            assert caps[:2] == [4 * chunk + 64, 1] and 4 in caps[2:]  # spliced and indel records have 2 blocks
        else:
            assert caps == [4 * chunk + 64, 2]


def test_parts_store_their_inputs_and_stale_parts_are_named(synth, tagged, tmp_path):
    tx = load_transcripts(synth.gtf)
    part_dir = tmp_path / "parts"
    meta = {"truth": "/x/truth_reads.parquet", "truth_rows": 12, "truth_qname_sha256": qname_digest(["r1", "r2"]),
            "settings": len(SETTINGS)}
    st = scan_to_parts(tagged, synth.fasta, "chrB", tx, SETTINGS, set(), None, part_dir, meta=meta)
    stored = json.loads((part_dir / "chrB.json").read_text())
    assert stored == st and {k: stored[k] for k in meta} == meta and stored["ref"] == "chrB"
    # chrA has no part yet, so nothing can be stale there
    assert stale_parts(part_dir, ["chrA", "chrB"], meta) == []
    assert stale_parts(part_dir, ["chrA", "chrB"], meta | {"truth_rows": 13}) == ["chrB"]
    assert stale_parts(part_dir, ["chrB"], meta | {"truth": "/y/truth_reads.parquet"}) == ["chrB"]
    assert stale_parts(part_dir, ["chrB"], meta | {"settings": 396}) == ["chrB"]
    assert stale_parts(part_dir, ["chrB"], meta | {"truth_qname_sha256": "0" * 64}) == ["chrB"]
    # a part written without the guard (no inputs stored) is stale as well
    scan_to_parts(tagged, synth.fasta, "chrA", tx, SETTINGS, set(), None, part_dir)
    assert stale_parts(part_dir, ["chrA", "chrB"], meta) == ["chrA"]


def test_qname_digest_is_sha256_of_the_names_in_file_order():
    want = hashlib.sha256(b"r1\nr2\nr3\n").hexdigest()
    assert qname_digest(["r1", "r2", "r3"]) == want
    assert qname_digest(pd.Series(["r1", "r2", "r3"])) == want
    assert qname_digest(["r2", "r1", "r3"]) != want and qname_digest(["r1", "r2", "r4"]) != want
    assert qname_digest(["r1r", "2r3"]) != qname_digest(["r1", "r2r3"])


# Runs a stage script with pafbench.config pointed at test inputs: FASTA GTF BAM(of library X) SCRIPT ARGS...
RUNNER = """
import runpy, sys
from pathlib import Path
import pafbench.config as c
c.FASTA, c.GTF, c.LIBRARIES = Path(sys.argv[1]), Path(sys.argv[2]), {"X": Path(sys.argv[3])}
sys.argv = sys.argv[4:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


@pytest.fixture(scope="module")
def stage13_inputs(synth, tagged, tmp_path_factory):
    """Library X: the tagged synthetic BAM plus chrC (reads, no annotation), its called cells, and a genome with chrC.

    chrB has reads only on the minus strand, so its plus-strand transcript t4 is never expressed.
    """
    d = tmp_path_factory.mktemp("stage13")
    fasta = d / "genome.fa"
    rng = np.random.default_rng(9)
    fasta.write_text(synth.fasta.read_text() + ">chrC\n" + "".join(rng.choice(list("ACGT"), 1000)) + "\n")
    pysam.faidx(str(fasta))
    outs = d / "lib" / "outs"
    (outs / "filtered_feature_bc_matrix").mkdir(parents=True)
    with gzip.open(outs / "filtered_feature_bc_matrix" / "barcodes.tsv.gz", "wt") as fh:
        fh.write("".join(c + "\n" for c in CELLS))
    recs = []
    with pysam.AlignmentFile(str(tagged)) as fh:
        for a in fh.fetch(until_eof=True):
            tags = {k: v for k, v in a.get_tags()}
            recs.append((a.query_name, a.reference_name, a.reference_start, a.cigarstring, a.flag, a.query_sequence, tags))
    seq = "".join(rng.choice(list("ACGT"), 50))
    for i, (pos, flag, re_, cb) in enumerate([(100, 0, "I", "AAACCC-1"), (300, 16, "I", "GGGTTT-1"), (500, 0, None, "TTTAAA-1"),
                                              (700, 1024, "N", "AAACCC-1"), (800, 256, "I", "AAACCC-1")]):
        tags = {"CB": cb} | ({} if re_ is None else {"RE": re_})
        recs.append((f"c{i}", "chrC", pos, "50M", flag, seq, tags))
    bam = _write_bam(outs / "possorted_genome_bam.bam", {"chrA": 6000, "chrB": 2000, "chrC": 1000}, recs)
    chrA = [r[0] for r in recs if r[1] == "chrA" and not r[4] & 0x900]
    truth = d / "truth_reads.parquet"
    pd.DataFrame({"qname": chrA[::2] + ["not_in_bam"]}).to_parquet(truth)
    return fasta, synth.gtf, bam, truth


def _stage13(inputs, out, *extra):
    fasta, gtf, bam, truth = inputs
    script = REPO / "scripts" / "13_grid_flags.py"
    return subprocess.run([sys.executable, "-c", RUNNER, str(fasta), str(gtf), str(bam), str(script), "X", "--workers", "2",
                           "--out", str(out), "--truth", str(truth), *extra],
                          capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(REPO)), cwd=REPO)


def _grid_outputs(out):
    summary = json.loads((out / "summary.json").read_text())
    return (pd.read_parquet(out / "truth_flags.parquet"), pd.read_parquet(out / "region_counts.parquet"),
            pd.read_parquet(out / "cell_counts.parquet"),
            {k: summary[k] for k in ("records", "primary", "removed")} | {"per_ref": {r: {k: v for k, v in s.items() if k != "seconds"}
                                                                                     for r, s in summary["per_ref"].items()}})


@pytest.fixture(scope="module")
def stage13_fresh(stage13_inputs, tmp_path_factory):
    out = tmp_path_factory.mktemp("fresh") / "grid"
    r = _stage13(stage13_inputs, out, "--refs", "chrA", "chrB", "chrC", "chrZ")
    assert r.returncode == 0, r.stderr
    return out, r


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")  # flag_reference adds 396 columns one by one
def test_stage13_empty_parts_keep_their_region_totals(stage13_inputs, stage13_fresh):
    fasta, gtf, bam, truth = stage13_inputs
    out, r = stage13_fresh
    assert "chrZ" in r.stderr  # asked for, but not in the BAM index
    line = json.loads(r.stdout.strip().splitlines()[-1])
    assert (line["library"], line["settings"], line["scanned"], line["refs"]) == ("X", 396, 3, 3)
    settings = all_settings()
    tags = [setting_tag(s) for s in settings]
    tx = load_transcripts(gtf)
    keep = set(pd.read_parquet(truth).qname)
    full = {ref: flag_reference(bam, fasta, ref, tx, settings) for ref in ("chrA", "chrB", "chrC")}
    with pysam.AlignmentFile(str(bam)) as fh:
        for ref, df in full.items():
            recs = list(fh.fetch(ref))
            df["re"] = [a.get_tag("RE") if a.has_tag("RE") else "" for a in recs]
            df["cb"] = [a.get_tag("CB") if a.has_tag("CB") else None for a in recs]
    prim = {ref: df[(df.flag & 0x900) == 0].reset_index(drop=True) for ref, df in full.items()}
    # chrB (no expressed transcript on +, no truth read) and chrC (no annotation, no truth read) give no truth rows
    tf, region, cells, summary = _grid_outputs(out)
    want = prim["chrA"][prim["chrA"].qname.isin(keep)].reset_index(drop=True)
    assert tf.qname.tolist() == want.qname.tolist() and set(tf.ref) == {"chrA"} and len(tf) == len(keep) - 1
    assert np.array_equal(tf[tags].to_numpy(), want[tags].to_numpy())
    # but their primary records stay in the region totals: chrC adds I, I, "" and N (the duplicate counts, the secondary not)
    allp = pd.concat(prim.values(), ignore_index=True)
    want_region = sorted((t, re_, int(g[t].sum()), len(g)) for re_, g in allp.groupby("re") for t in tags)
    assert sorted(region.itertuples(index=False, name=None)) == want_region
    assert prim["chrC"].re.tolist() == ["I", "I", "", "N"] and not prim["chrC"][tags].to_numpy().any()
    # chrB: t4 (+) has no read and t5 (-, 80 nt) is shorter than every COVLEN, so wholly protected: nothing removed
    assert not prim["chrB"][tags].to_numpy().any() and len(prim["chrB"]) > 0
    assert summary["per_ref"]["chrC"] == {"records": 5, "primary": 4} and summary["primary"] == len(allp)
    for ref in ("chrB", "chrC"):
        assert len(pd.read_parquet(out / "parts" / f"{ref}.truth.parquet")) == 0
        part = pd.read_parquet(out / "parts" / f"{ref}.region.parquet")
        assert part[part.tag == tags[0]].total.sum() == len(prim[ref]) and part.removed.sum() == 0
    assert summary["removed"] == {t: int(allp[t].sum()) for t in tags}
    in_cells = allp[allp.cb.isin(CELLS)]
    assert cells.total.to_dict() == in_cells.groupby("cb").size().to_dict()
    assert np.array_equal(cells[tags].to_numpy(np.int64), in_cells.groupby("cb")[tags].sum().reindex(cells.index).to_numpy(np.int64))


def test_stage13_resume_after_crash_and_stale_truth(stage13_inputs, stage13_fresh, tmp_path):
    fresh_out, _ = stage13_fresh
    fasta, gtf, bam, fixture_truth = stage13_inputs
    truth = tmp_path / "truth_reads.parquet"
    shutil.copy(fixture_truth, truth)
    inputs = (fasta, gtf, bam, truth)
    out = tmp_path / "grid"
    # a run that finished chrA only, then a crash in chrC after its truth part was written but before its json
    r = _stage13(inputs, out, "--refs", "chrA")
    assert r.returncode == 0, r.stderr
    pd.read_parquet(out / "parts" / "chrA.truth.parquet").head(3).to_parquet(out / "parts" / "chrC.truth.parquet")
    r = _stage13(inputs, out)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["scanned"] == 2  # chrB and chrC, not chrA again
    got, want = _grid_outputs(out), _grid_outputs(fresh_out)
    pd.testing.assert_frame_equal(got[0], want[0])
    pd.testing.assert_frame_equal(got[1], want[1])
    pd.testing.assert_frame_equal(got[2], want[2])
    assert got[3] == want[3]
    # a second resume scans nothing and changes nothing
    r = _stage13(inputs, out)
    assert r.returncode == 0 and json.loads(r.stdout.strip().splitlines()[-1])["scanned"] == 0
    assert _grid_outputs(out)[3] == want[3]
    # stage 12 reran and rewrote the truth file at the same path with the same number of other reads: refused, nothing rewritten
    before = {p.name: p.stat().st_mtime_ns for p in out.iterdir() if p.is_file()}
    same_n = pd.read_parquet(fixture_truth)
    same_n.loc[same_n.index[0], "qname"] = "another_read"
    same_n.to_parquet(truth)
    assert len(same_n) == len(pd.read_parquet(fixture_truth))
    r = _stage13(inputs, out)
    assert r.returncode == 1 and "chrA, chrB, chrC" in r.stderr
    assert {p.name: p.stat().st_mtime_ns for p in out.iterdir() if p.is_file()} == before
    # and with one more row
    pd.concat([pd.read_parquet(truth), pd.DataFrame({"qname": ["extra"]})]).to_parquet(truth)
    r = _stage13(inputs, out)
    assert r.returncode == 1
    assert "chrA, chrB, chrC" in r.stderr and f"gio trash {out / 'parts'}" in r.stderr
    assert {p.name: p.stat().st_mtime_ns for p in out.iterdir() if p.is_file()} == before


def test_json_is_last(synth, tagged, tmp_path, monkeypatch):
    """The json that marks a reference done is written after the parquet parts: a failed cells write leaves no json."""
    tx = load_transcripts(synth.gtf)
    part_dir = tmp_path / "parts"
    real = pd.DataFrame.to_parquet

    def boom(self, path, *a, **k):
        if str(path).endswith(".cells.parquet"):
            raise OSError("disk full")
        return real(self, path, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        scan_to_parts(tagged, synth.fasta, "chrA", tx, SETTINGS, set(_qnames(tagged, "chrA")[::4]), CELLS, part_dir)
    assert not (part_dir / "chrA.json").exists()
