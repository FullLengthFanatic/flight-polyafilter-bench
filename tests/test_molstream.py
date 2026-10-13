"""The read-group pass of stage 23 (pafbench.molstream) on the synthetic 10x BAM, against the designed tables of
tests/synth.build_10x, flags.flag_reference, stream.scan_reference and plain counts of the BAM."""
import dataclasses
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pysam
import pytest

from pafbench import molstream, stream
from pafbench.config import IMPACT_SETTINGS, REPO, UNIT_BIN
from pafbench.flags import Setting, flag_reference, setting_tag
from pafbench.intervals import blocks_hit, flatten
from pafbench.molstream import part_meta, read_groups_to_parts, row_index, scan_read_groups
from pafbench.stream import stale_parts
from pafbench.transcripts import load_transcripts
from tests.synth import CELLS_10X, CHROMS, HEADER, SETTINGS_10X, UNCALLED_10X, _sorted_bam
from tests.test_stream import RUNNER
from tests.test_synth10x import CELL, CELL_RECORDS, GENES, GROUPS

TAGS = [setting_tag(s) for s in SETTINGS_10X]
CALLED = sorted(CELLS_10X.values())                 # the rows of barcodes.tsv.gz
GROUP_COLS = ["key", "n_reads", "n_winner_reads", "n_pa", "re_class", "end3", "end3_nonpa"]
GROUP_TYPES = ["uint64", "uint16", "uint16", "uint16", "uint8", "int32", "int32"]      # then uint16 per setting
RECORD_COLS = ["cell", "re", "gx", "total", "with_ub", "units", "units_rg"]
COUNTS = ["records", "primary", "called", "xf1_no_ub", "xf1_no_gene", "in_groups", "read_groups"]


def _kernel_bits(bs, be, br, n_rec, pieces):
    ps, pe, offs = stream._pack(pieces)
    mask = np.zeros(n_rec, np.uint32)
    molstream._mask_hits(bs, be, br, ps, pe, offs, mask)
    return mask


def test_mask_kernel_edges_by_hand():
    # the blocks of test_stream.test_hit_kernel_edges_by_hand against the pieces [10, 20) and [30, 40) as tag 1; tag 0 has no
    # piece and tag 2 the piece [0, 6), which only the first block touches. Records 0 to 5 have one block each.
    bs = np.array([5, 8, 19, 20, 25, 40], np.int64)
    be = np.array([10, 11, 20, 30, 31, 50], np.int64)
    br = np.arange(6, dtype=np.int64)
    none = (np.empty(0, np.int64), np.empty(0, np.int64))
    mask = _kernel_bits(bs, be, br, 6, [none, (np.array([10, 30]), np.array([20, 40])), (np.array([0]), np.array([6]))])
    assert mask.dtype == np.uint32 and mask.tolist() == [4, 2, 2, 0, 2, 0]
    # two blocks of one record: either one sets the bit, and a bit that is set stays
    mask = _kernel_bits(np.array([5, 25]), np.array([10, 31]), np.array([0, 0]), 1, [(np.array([10, 30]), np.array([20, 40]))])
    assert mask.tolist() == [1]
    assert _kernel_bits(bs[:0], be[:0], br[:0], 2, [none]).tolist() == [0, 0]


def test_mask_kernel_matches_blocks_hit_random():
    rng = np.random.default_rng(3)
    n_rec, n_tags = 400, 32
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
        k = int(rng.integers(0, 40))
        s = rng.integers(0, 5600, k)
        pieces.append(flatten(s, s + rng.integers(1, 80, k)))
    order = np.argsort(blk_s, kind="stable")
    got = _kernel_bits(blk_s[order], blk_e[order], blk_r[order], n_rec, pieces)
    want = np.zeros(n_rec, np.int64)
    for t, (s, e) in enumerate(pieces):
        want[np.unique(blk_r[blocks_hit(blk_s, blk_e, s, e)])] |= 1 << t
    assert got.tolist() == want.tolist()
    # all 32 bits are in use, the highest included, and no tag hits every record
    hits = [int(((want >> t) & 1).sum()) for t in range(n_tags)]
    assert all(0 < h < n_rec for h, (s, _) in zip(hits, pieces) if s.size) and hits[31] > 0
    assert sum(s.size > 0 for s, _ in pieces) >= 30


def test_record_counter_matches_a_plain_count_random():
    rng = np.random.default_rng(11)
    n, n_cells, n_cls, n_tags = 4000, 6, 8, 32
    mask = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    prim, has_ub = rng.random(n) < 0.9, rng.random(n) < 0.7
    cls, cell = rng.integers(0, n_cls, n), rng.integers(-1, n_cells, n)
    allc = np.zeros((n_cls, 2 + n_tags), np.int64)
    celc = np.zeros((n_cells, n_cls, 2 + n_tags), np.uint32)
    for part in (slice(0, 1500), slice(1500, n)):      # as the pass calls it: chunk after chunk into the same counters
        molstream._count_records(mask[part], prim[part], cls[part], cell[part], has_ub[part], allc, celc)
    want_all, want_cell = np.zeros_like(allc), np.zeros(celc.shape, np.int64)
    for r in np.flatnonzero(prim):
        row = [1, int(has_ub[r])] + [(int(mask[r]) >> b) & 1 for b in range(n_tags)]
        want_all[cls[r]] += row
        if cell[r] >= 0:
            want_cell[cell[r], cls[r]] += row
    assert np.array_equal(allc, want_all) and np.array_equal(celc, want_cell)
    # every column counts in every class and cell, the one of bit 31 included, and none counts every record
    assert want_cell.min() > 0 and (want_all[:, 1:] < want_all[:, :1]).all() and (want_cell.sum(axis=0) < want_all).all()


def _scan(s, ref, settings=SETTINGS_10X, chunk=1_000_000, bam=None, fasta=None):
    matrix = s.outs / "filtered_feature_bc_matrix"
    return scan_read_groups(bam or s.bam, fasta or s.fasta, ref, load_transcripts(s.gtf), settings,
                            row_index(matrix / "barcodes.tsv.gz"), row_index(matrix / "features.tsv.gz"), chunk)


def _umi(code):
    return "".join("ACGT"[(code >> 2 * (11 - i)) & 3] for i in range(12))


def _group_rows(groups):
    """{(barcode, UMI, gene): the other columns} with the key taken apart: cell 24 bits, UMI 24 bits, gene 16 bits."""
    out = {}
    for r in groups.itertuples(index=False):
        key = int(r.key)
        out[(CALLED[key >> 40], _umi((key >> 16) & 0xFFFFFF), GENES[key & 0xFFFF])] = tuple(int(x) for x in r[1:])
    return out


def _record_rows(frames):
    """{(barcode, None for all barcodes; RE; has GX): (total, with_ub, units, units_rg, removed per tag)} summed over frames."""
    out = Counter()
    for rec in frames:
        for r in rec.itertuples(index=False):
            k = (None if r.cell < 0 else CALLED[r.cell], r.re, bool(r.gx))
            out[k] = tuple(a + int(b) for a, b in zip(out.get(k, (0,) * (len(r) - 3)), r[3:]))
    return dict(out)


def _primary(bam, ref):
    """Primary records of one reference as dicts of their tags plus flag, pos and end, in fetch order."""
    with pysam.AlignmentFile(str(bam)) as fh:
        return [dict(a.get_tags()) | {"flag": a.flag, "pos": a.reference_start, "end": a.reference_end}
                for a in fh.fetch(ref) if not a.flag & 0x900]


def _in_group(r):
    return r.get("CB") in CELL and "UB" in r and r.get("GX") in GENES and r["xf"] & 1


def test_row_index_reads_the_first_column_of_the_matrix_files(synth10x):
    matrix = synth10x.outs / "filtered_feature_bc_matrix"
    assert row_index(matrix / "barcodes.tsv.gz") == {cb: i for i, cb in enumerate(CALLED)}
    assert row_index(matrix / "features.tsv.gz") == {g: i for i, g in enumerate(GENES)}


@pytest.mark.parametrize("chunk", [1, 7, 1000])
def test_read_groups_are_the_designed_ones(synth10x, chunk):
    got, n_groups = {}, {}
    for ref in CHROMS:
        g, _, counts = _scan(synth10x, ref, chunk=chunk)
        assert list(g.columns) == GROUP_COLS + [f"rm_{t}" for t in TAGS]
        assert g.dtypes.astype(str).tolist() == GROUP_TYPES + ["uint16"] * 4
        assert g.key.is_monotonic_increasing and g.key.is_unique
        rows = _group_rows(g)
        assert not set(rows) & set(got)                 # a gene sits on one reference
        got |= rows
        n_groups[ref] = counts["read_groups"]
        assert counts["in_groups"] == int(g.n_reads.sum()) and counts["read_groups"] == len(g)
    t = synth10x.groups
    assert got == {(r.cb, r.ub, r.gene): tuple(int(x) for x in r[4:]) for r in t.itertuples(index=False)}
    case = {(r.cb, r.ub): r.case for r in t.itertuples()}
    designed = defaultdict(dict)
    for (cb, ub, gene), row in got.items():
        designed[case[(cb, ub)]][gene] = row
    assert designed == GROUPS
    assert n_groups == {"chrA": 27, "chrB": 6}        # g1 and g2 on chrA, g3 on chrB


@pytest.mark.parametrize("settings", [SETTINGS_10X, IMPACT_SETTINGS], ids=["test settings", "impact settings"])
@pytest.mark.parametrize("chunk", [3, 1000])
def test_removed_bit_of_every_record_equals_flag_reference(synth10x, monkeypatch, settings, chunk):
    seen = []
    masks = molstream._Chunk.masks

    def spy(self, packed):
        m = masks(self, packed)
        seen.append(m.copy())
        return m

    monkeypatch.setattr(molstream._Chunk, "masks", spy)
    tx = load_transcripts(synth10x.gtf)
    n_removed = 0
    for ref in CHROMS:
        seen.clear()
        _scan(synth10x, ref, settings=settings, chunk=chunk)
        want = flag_reference(synth10x.bam, synth10x.fasta, ref, tx, settings)      # every record, fetch order
        bits = np.concatenate(seen)
        assert bits.dtype == np.uint32 and len(bits) == len(want) and len(seen) >= len(want) / chunk
        for t, st in enumerate(settings):
            assert ((bits >> t) & 1).astype(bool).tolist() == want[setting_tag(st)].tolist(), (ref, setting_tag(st))
        assert int(bits.max()) < 1 << len(settings)
        n_removed += int(want[[setting_tag(st) for st in settings]].to_numpy().sum())
    assert n_removed > 0


@pytest.mark.parametrize("chunk", [1, 7, 1000])
def test_cell_records_are_the_designed_counters_and_units_a_naive_set_count(synth10x, chunk):
    frames = []
    for ref in CHROMS:
        _, rec, counts = _scan(synth10x, ref, chunk=chunk)
        assert list(rec.columns) == RECORD_COLS + TAGS
        assert rec.dtypes.astype(str).tolist() == ["int32", "str", "bool"] + ["uint32"] * (4 + len(TAGS))
        assert (rec.total > 0).all() and not rec.duplicated(["cell", "re", "gx"]).any()
        called = rec[rec.cell >= 0]
        assert counts["called"] == int(called.total.sum()) and counts["primary"] == int(rec[rec.cell < 0].total.sum())
        frames.append(called)
    got = _record_rows(frames)
    assert {(CELL[cb], re_, gx): row[:2] + row[4:] for (cb, re_, gx), row in got.items()} == CELL_RECORDS
    # distinct (cell, UMI, strand, 1 kb bin) among the primary records with a UB of each cell, RE and GX presence, counted
    # with plain sets; the same over the records of read groups
    units, units_rg = defaultdict(set), defaultdict(set)
    for ref in CHROMS:
        for r in _primary(synth10x.bam, ref):
            if r.get("CB") in CELL and "UB" in r:
                k = (r["CB"], r["RE"], "GX" in r)
                unit = (ref, r["UB"], bool(r["flag"] & 16), r["pos"] // 1000)
                units[k].add(unit)
                if _in_group(r):
                    units_rg[k].add(unit)
    assert {k: row[2] for k, row in got.items() if row[2]} == {k: len(v) for k, v in units.items()}
    assert {k: row[3] for k, row in got.items() if row[3]} == {k: len(v) for k, v in units_rg.items()}
    # the fixture tells units from records and from read groups: c1 has 13 exonic records with a UB and a GX in 11 units (two
    # reads of "spliced" and the two of "pa_and_member" share a bin), and its 11 exonic read-group records are 9 units of 7
    # read groups (the multi-gene and the multimapped record are units outside every group)
    c1 = got[(CELLS_10X["c1"], "E", True)]
    assert c1[:4] == (14, 13, 11, 9)
    assert sum(row[2] for row in got.values()) < sum(row[1] for row in got.values())
    assert sum(row[3] for row in got.values()) < sum(row[2] for row in got.values())


@pytest.mark.parametrize("settings", [SETTINGS_10X, IMPACT_SETTINGS], ids=["test settings", "impact settings"])
def test_all_barcodes_rows_and_cell_sums_equal_scan_reference(synth10x, settings):
    """Check A3 of the design on synthetic data: the pass against the engine of stage 13 on the same BAM."""
    tx = load_transcripts(synth10x.gtf)
    tags = [setting_tag(s) for s in settings]
    removed = np.zeros(len(tags), np.int64)
    for ref in CHROMS:
        _, region, cells = stream.scan_reference(synth10x.bam, synth10x.fasta, ref, tx, settings, set(), cells=CALLED)
        _, rec, counts = _scan(synth10x, ref, settings=settings)
        every = rec[rec.cell < 0]
        assert len(every) > 0 and (every.units == 0).all() and (every.units_rg == 0).all()
        by_re = every.groupby("re")[["total"] + tags].sum()
        got = sorted((t, re_, int(by_re.loc[re_, t]), int(by_re.loc[re_, "total"])) for t in tags for re_ in by_re.index)
        assert got == sorted(region[["tag", "re", "removed", "total"]].itertuples(index=False, name=None))
        per_cell = rec[rec.cell >= 0].groupby("cell")[["total"] + tags].sum().reindex(range(len(CALLED)), fill_value=0)
        assert cells.index.tolist() == CALLED
        assert np.array_equal(per_cell.to_numpy(np.int64), cells.to_numpy(np.int64))
        assert counts["records"] == region.attrs["records"]
        # the uncalled barcode and the record without CB are in the all-barcodes rows only
        assert int(every.total.sum()) >= int(per_cell.total.sum()) + 1
        removed += cells[tags].to_numpy(np.int64).sum(axis=0)
    assert (removed > 0).all()                         # every setting removes records of called cells, the last one too


def test_counts_name_the_records_outside_read_groups(synth10x):
    got = {ref: _scan(synth10x, ref)[2] for ref in CHROMS}
    assert all(list(c) == COUNTS for c in got.values())
    for ref in CHROMS:
        recs = _primary(synth10x.bam, ref)
        called = [r for r in recs if r.get("CB") in CELL]
        xf1 = [r for r in called if r["xf"] & 1]
        assert got[ref] == {
            "records": len(recs), "primary": len(recs), "called": len(called),
            "xf1_no_ub": sum("UB" not in r for r in xf1), "xf1_no_gene": 0,
            "in_groups": sum(bool(_in_group(r)) for r in called),
            "read_groups": len({(r["CB"], r["UB"], r["GX"]) for r in called if _in_group(r)}),
        }
    assert got["chrA"]["xf1_no_ub"] == 1 and got["chrA"]["called"] > got["chrA"]["in_groups"] > got["chrA"]["read_groups"]


def _record(hdr, name, ref, pos, flag=0, **tags):
    """A 90M record with its sequence and the given tags. RE gets type A: "RE A Single character indicating the region type
    of this alignment (E = exonic, N = intronic, I = intergenic)." (Cell Ranger 9.0 BAM page,
    https://www.10xgenomics.com/support/software/cell-ranger/9.0/analysis/outputs/cr-outputs-bam; copy in
    design_m3_draft/research/dl/bam.txt, line 102)."""
    a = pysam.AlignedSegment(hdr)
    a.query_name, a.reference_name, a.reference_start, a.cigarstring, a.flag, a.mapping_quality = name, ref, pos, "90M", flag, 255
    a.query_sequence, a.query_qualities = ("ACGT" * 23)[:90], pysam.qualitystring_to_array("I" * 90)
    for k, v in tags.items():
        a.set_tag(k, v, value_type="A" if k == "RE" else None)
    return a


C1, C2, C3 = (CELLS_10X[c] for c in ("c1", "c2", "c3"))
# (name, reference, position, flag, tags) of the records the shared fixture does not hold
PLANTS = [
    # copies of two read-group records as secondary and supplementary alignments, inside a removal window, and a secondary
    # one without UB
    ("secondary", "chrA", 1150, 256, dict(CB=C1, UB="AACCGGTTAACC", GX="g1", RE="E", xf=17)),
    ("supplementary", "chrA", 2150, 2048, dict(CB=C1, UB="ACACACGTGTGT", GX="g1", RE="E", xf=17)),
    ("secondary", "chrA", 1150, 256, dict(CB=C1, GX="g1", RE="E", xf=17)),
    # chrC, a reference without annotation and without a gene of the feature list: intergenic records of a called cell
    # (three with one UB: two forward in one bin, one reverse), one without UB and a secondary copy
    ("intergenic", "chrC", 100, 0, dict(CB=C1, UB="ACGTACGTACGT", RE="I", xf=0)),
    ("intergenic", "chrC", 150, 0, dict(CB=C1, UB="ACGTACGTACGT", RE="I", xf=0)),
    ("intergenic", "chrC", 180, 16, dict(CB=C1, UB="ACGTACGTACGT", RE="I", xf=0)),
    ("intergenic", "chrC", 300, 0, dict(CB=C1, RE="I", xf=0)),
    ("secondary", "chrC", 100, 256, dict(CB=C1, UB="ACGTACGTACGT", RE="I", xf=0)),
    # xf bit 1 and a UB, but no gene of the feature list: a gene id outside it, and two genes
    ("gene_outside", "chrC", 400, 0, dict(CB=C2, UB="TTTTGGGGCCCC", GX="g9", RE="E", xf=25)),
    ("two_genes", "chrC", 450, 0, dict(CB=C2, UB="TTTTGGGGCCCA", GX="g1;g2", RE="E", xf=17)),
    ("no_ub", "chrC", 500, 0, dict(CB=C2, GX="g3", RE="N", xf=17)),
    ("no_re", "chrC", 600, 0, dict(CB=C3, UB="GGGGAAAACCCC", xf=0)),
    ("uncalled", "chrC", 700, 0, dict(CB=UNCALLED_10X, UB="CCCCAAAAGGGG", RE="I", xf=0)),
    # chrD, a reference without a called cell
    ("uncalled", "chrD", 100, 0, dict(CB=UNCALLED_10X, UB="CTAGCTAGGATC", GX="g1", RE="E", xf=25)),
    ("no_cb", "chrD", 200, 0, dict(UB="CAGTCAGTTGAC", RE="I", xf=0)),
]
REFS = ["chrA", "chrB", "chrC", "chrD"]


def _library(d, s, plants):
    """The fixture library with more records and the contigs chrC and chrD: genome.fa and outs/ under d, as a Synth10x."""
    rng = np.random.default_rng(9)
    fasta = d / "genome.fa"
    fasta.write_text(s.fasta.read_text()
                     + "".join(f">{c}\n{''.join(rng.choice(list('ACGT'), 1000))}\n" for c in ("chrC", "chrD")))
    pysam.faidx(str(fasta))
    hdr = pysam.AlignmentHeader.from_dict(HEADER | {"SQ": HEADER["SQ"] + [{"SN": c, "LN": 1000} for c in ("chrC", "chrD")]})
    with pysam.AlignmentFile(str(s.bam)) as fh:
        recs = [pysam.AlignedSegment.from_dict(a.to_dict(), hdr) for a in fh.fetch(until_eof=True)]
    recs += [_record(hdr, name, ref, pos, flag, **tags) for name, ref, pos, flag, tags in plants]
    (d / "outs").mkdir()
    shutil.copytree(s.outs / "filtered_feature_bc_matrix", d / "outs" / "filtered_feature_bc_matrix")
    bam = _sorted_bam(d, hdr, recs, "outs/possorted_genome_bam.bam")
    return dataclasses.replace(s, tmp=d, fasta=fasta, bam=bam, outs=d / "outs")


@pytest.fixture(scope="module")
def planted(synth10x, tmp_path_factory):
    return _library(tmp_path_factory.mktemp("planted"), synth10x, PLANTS)


@pytest.mark.parametrize("chunk", [1, 1000])
def test_secondary_and_supplementary_records_are_fetched_and_not_counted(synth10x, planted, chunk):
    """With a chunk of one record the xf, gene and pa buffers still hold the values of the last primary record of a called
    cell when a secondary record comes: they are not written for it, and not read."""
    for ref in CHROMS:
        groups, rec, counts = _scan(planted, ref, chunk=chunk)
        want_groups, want_rec, want_counts = _scan(synth10x, ref)
        pd.testing.assert_frame_equal(groups, want_groups)
        pd.testing.assert_frame_equal(rec, want_rec)
        extra = sum(p[1] == ref for p in PLANTS)
        assert counts == want_counts | {"records": want_counts["records"] + extra} and extra == 3 * (ref == "chrA")


def test_reference_without_genes_or_without_called_cells_gives_empty_tables(planted):
    empty = _scan(planted, "chrA")[0].iloc[:0]
    groups, rec, counts = _scan(planted, "chrC")
    pd.testing.assert_frame_equal(groups, empty)                       # no row, and the columns and types of a full part
    assert [tuple(r) for r in rec[RECORD_COLS].itertuples(index=False)] == [
        (-1, "E", True, 2, 2, 0, 0), (-1, "N", True, 1, 0, 0, 0), (-1, "I", False, 5, 4, 0, 0), (-1, "", False, 1, 1, 0, 0),
        (0, "I", False, 4, 3, 2, 0), (1, "E", True, 2, 2, 2, 0), (1, "N", True, 1, 0, 0, 0), (2, "", False, 1, 1, 1, 0)]
    assert not rec[TAGS].to_numpy().any()                              # no annotation, nothing to remove
    assert counts == dict(zip(COUNTS, (10, 9, 8, 1, 2, 0, 0)))
    groups, rec, counts = _scan(planted, "chrD")
    pd.testing.assert_frame_equal(groups, empty)
    assert [tuple(r) for r in rec[RECORD_COLS].itertuples(index=False)] == [(-1, "E", True, 1, 1, 0, 0),
                                                                            (-1, "I", False, 1, 1, 0, 0)]
    assert rec.dtypes.astype(str).tolist() == ["int32", "str", "bool"] + ["uint32"] * (4 + len(TAGS))
    assert counts == dict(zip(COUNTS, (2, 2, 0, 0, 0, 0, 0)))


def _same_group(path, hdr, sizes):
    """BAM with one read group per reference: sizes[ref] records of cell c1, one UMI and the gene g1 (chrA) or g3 (chrB)."""
    with pysam.AlignmentFile(str(path), "wb", header=hdr) as out:
        for ref, gene, pos in (("chrA", "g1", 1150), ("chrB", "g3", 600)):
            first, other = (_record(hdr, "r", ref, pos, flag, CB=C1, UB="AACCGGTTAACC", GX=gene, RE="E", xf=xf)
                            for flag, xf in ((0, 25), (1024, 17)))
            out.write(first)
            for _ in range(sizes[ref] - 1):
                out.write(other)
    pysam.index(str(path))
    return path


def test_group_at_the_uint16_limit_stops_the_pass(synth10x, tmp_path):
    hdr = pysam.AlignmentHeader.from_dict(HEADER)
    bam = _same_group(tmp_path / "big.bam", hdr, {"chrA": 65_535, "chrB": 65_534})
    with pytest.raises(ValueError, match="a read group has 65,535 reads"):
        _scan(synth10x, "chrA", bam=bam)
    # one read fewer fits; the controls remove every read of g3 at 600 and the settings none
    groups, rec, counts = _scan(synth10x, "chrB", bam=bam, chunk=20_000)
    assert [int(x) for x in groups.iloc[0, 1:]] == [65_534, 65_534, 0, 1, 689, 689, 0, 0, 65_534, 65_534] and len(groups) == 1
    assert rec[rec.cell >= 0][["total", "with_ub", "units", "units_rg"] + TAGS].to_numpy().tolist() == [
        [65_534, 65_534, 1, 1, 0, 0, 65_534, 65_534]]
    assert counts["in_groups"] == 65_534


@pytest.mark.parametrize("chunk", [1, 1000])
def test_unit_that_starts_outside_the_read_group_and_pa_tag_of_zero(synth10x, tmp_path, chunk):
    """Two cases the fixture does not hold: a unit whose first record is in no read group (xf 0) and whose second is in
    one, and a pa tag of 0, which is no pa read."""
    hdr = pysam.AlignmentHeader.from_dict(HEADER)
    with pysam.AlignmentFile(str(tmp_path / "unit.bam"), "wb", header=hdr) as out:
        out.write(_record(hdr, "r", "chrA", 2150, CB=C1, UB="AACCGGTTAACC", GX="g1", RE="E", xf=0))
        out.write(_record(hdr, "r", "chrA", 2505, CB=C1, UB="AACCGGTTAACC", GX="g1", RE="E", xf=25, pa=0))
    pysam.index(str(tmp_path / "unit.bam"))
    groups, rec, _ = _scan(synth10x, "chrA", bam=tmp_path / "unit.bam", chunk=chunk)
    assert [int(x) for x in groups.iloc[0, 1:7]] == [1, 1, 0, 1, 2594, 2594] and len(groups) == 1
    assert rec[rec.cell >= 0][["total", "with_ub", "units", "units_rg"]].to_numpy().tolist() == [[2, 2, 1, 1]]


def test_end3_of_a_last_read_with_an_intron_or_a_deletion(synth10x, tmp_path):
    """The 3'-most forward read of a group covers reference bases that its sequence does not hold: 400 skipped (N) in one
    group, 3 deleted (D) in the other. In the shared fixture the spliced read is not the last of its group and no record
    of a read group has a deletion."""
    hdr = pysam.AlignmentHeader.from_dict(HEADER)
    with pysam.AlignmentFile(str(tmp_path / "end3.bam"), "wb", header=hdr) as out:
        for pos, cigar, ub in ((2400, "90M", "AACCGGTTAACC"), (2560, "40M400N50M", "AACCGGTTAACC"),
                               (3300, "90M", "ACACACGTGTGT"), (3400, "40M3D50M", "ACACACGTGTGT")):
            a = _record(hdr, "r", "chrA", pos, CB=C1, UB=ub, GX="g1", RE="E", xf=17)
            a.cigarstring = cigar
            out.write(a)
    pysam.index(str(tmp_path / "end3.bam"))
    groups, _, _ = _scan(synth10x, "chrA", bam=tmp_path / "end3.bam")
    # the last aligned base: 2560 + 40 + 400 + 50 - 1 and 3400 + 40 + 3 + 50 - 1
    assert {ub: row[4:6] for (_, ub, _), row in _group_rows(groups).items()} == {"AACCGGTTAACC": (3049, 3049),
                                                                                "ACACACGTGTGT": (3492, 3492)}


@pytest.mark.parametrize("chunk", [1, 1000])
def test_unit_bin_changes_at_a_multiple_of_1000(synth10x, tmp_path, chunk):
    """Two records of one UMI at 1999 and 2000 are two units, two at 3000 and 3999 one."""
    hdr = pysam.AlignmentHeader.from_dict(HEADER)
    with pysam.AlignmentFile(str(tmp_path / "bins.bam"), "wb", header=hdr) as out:
        for pos, cb in ((1999, C1), (2000, C1), (3000, C2), (3999, C2)):
            out.write(_record(hdr, "r", "chrA", pos, CB=cb, UB="AGTCAGTCAGTC", GX="g1", RE="E", xf=17))
    pysam.index(str(tmp_path / "bins.bam"))
    _, rec, _ = _scan(synth10x, "chrA", bam=tmp_path / "bins.bam", chunk=chunk)
    assert rec[rec.cell >= 0][["cell", "total", "units", "units_rg"]].to_numpy().tolist() == [[0, 2, 2, 2], [1, 2, 1, 1]]


def test_unsorted_bam_stops_the_pass(synth10x, tmp_path):
    """Records out of coordinate order behind an index. Such a file cannot be indexed (our observation, pysam 0.24.1:
    pysam.index stops with "[E::hts_idx_push] Unsorted positions on sequence #1: 2151 followed by 1811"), so the index is
    the one of the sorted file with the same records; the fetch below shows that it still returns all four, in file order."""
    hdr = pysam.AlignmentHeader.from_dict(HEADER)
    bam = tmp_path / "swapped.bam"
    for order in ((1150, 1810, 2150, 2505), (1150, 2150, 1810, 2505)):
        with pysam.AlignmentFile(str(bam), "wb", header=hdr) as out:
            for pos in order:
                out.write(_record(hdr, "r", "chrA", pos, CB=C1, UB="AACCGGTTAACC", GX="g1", RE="E", xf=17))
        if order[1] < order[2]:
            pysam.index(str(bam))
            groups, _, _ = _scan(synth10x, "chrA", bam=bam)
            assert groups.n_reads.tolist() == [4]
    os.utime(str(bam) + ".bai")
    with pysam.AlignmentFile(str(bam)) as fh:
        assert [a.reference_start for a in fh.fetch("chrA")] == [1150, 2150, 1810, 2505]
    with pytest.raises(ValueError, match="chrA: record 3 starts at 1810, before its predecessor at 2150"):
        _scan(synth10x, "chrA", bam=bam)


@pytest.mark.parametrize("ub", ["AACCGGTTAACN", "AACCGGTTAAC", "AACCGGTTAACCA", "aaccggttaacc", "AACCGGTT_ACC"])
def test_ub_that_is_not_12_bases_of_acgt_stops_the_pass(synth10x, tmp_path, ub):
    hdr = pysam.AlignmentHeader.from_dict(HEADER)
    with pysam.AlignmentFile(str(tmp_path / "ub.bam"), "wb", header=hdr) as out:
        out.write(_record(hdr, "r", "chrA", 1150, CB=C1, UB="AACCGGTTAACC", GX="g1", RE="E", xf=25))
        # xf 0 and no gene: the UMI of a record outside every read group is read as well, for the units
        out.write(_record(hdr, "r", "chrA", 4050, CB=C1, UB=ub, RE="I", xf=0))
    pysam.index(str(tmp_path / "ub.bam"))
    with pytest.raises(ValueError, match=f"chrA: UB '{ub}' is not 12 bases of A, C, G and T"):
        _scan(synth10x, "chrA", bam=tmp_path / "ub.bam")


def test_re_other_than_e_n_and_i_stops_the_pass(synth10x, tmp_path):
    hdr = pysam.AlignmentHeader.from_dict(HEADER)
    with pysam.AlignmentFile(str(tmp_path / "re.bam"), "wb", header=hdr) as out:
        out.write(_record(hdr, "r", "chrA", 1150, CB=UNCALLED_10X, UB="AACCGGTTAACC", RE="X", xf=0))
    pysam.index(str(tmp_path / "re.bam"))
    with pytest.raises(KeyError, match="X"):
        _scan(synth10x, "chrA", bam=tmp_path / "re.bam")


def _random_records(n_groups, reads_per_group, n_tags):
    """key, removed, flags and end3 of the records of n_groups read groups in random order, as the pass collects them."""
    rng = np.random.default_rng(5)
    key = np.repeat(rng.permutation(n_groups).astype(np.uint64) << np.uint64(16), reads_per_group)
    rng.shuffle(key)
    removed = rng.integers(0, 1 << n_tags, key.size).astype(np.uint32)
    return key, removed, rng.integers(0, 16, key.size).astype(np.uint8), rng.integers(-10**6, 10**6, key.size).astype(np.int32)


def test_reduction_of_random_records_equals_a_groupby():
    tags = [f"t{t}" for t in range(32)]                # every bit of the mask, the highest included
    key, removed, flags, end3 = _random_records(5000, 4, 32)
    got = molstream._reduce_groups(key, removed, flags, end3, tags)
    pa = (flags & 4) != 0
    rec = pd.DataFrame({"key": key, "n_reads": 1, "n_winner_reads": (flags & 8) == 0, "n_pa": pa, "re_class": flags & 3,
                        "end3": end3, "end3_nonpa": pd.Series(end3).where(~pa)}
                       | {f"rm_{tag}": (removed >> t) & 1 for t, tag in enumerate(tags)})
    how = {c: "sum" for c in rec.columns[1:]} | {"re_class": lambda x: np.bitwise_or.reduce(x.to_numpy()), "end3": "max",
                                                 "end3_nonpa": "max"}
    want = rec.groupby("key").agg(how).reset_index()
    want["end3"], want["end3_nonpa"] = want.end3.abs(), want.end3_nonpa.abs().fillna(-1)
    assert list(got.columns) == list(want.columns) and len(got) == 5000
    assert got.dtypes.astype(str).tolist() == GROUP_TYPES + ["uint16"] * 32
    assert np.array_equal(got.to_numpy(np.int64), want.to_numpy(np.int64))
    assert (got.end3_nonpa < 0).any() and (got.re_class == 0).any() and got.rm_t31.sum() > 0


@pytest.mark.parametrize("reads_per_group", [1, 3])
def test_reduction_holds_no_more_than_twice_its_records_and_twice_its_table(reads_per_group):
    """Design 3.1, in-memory state: next to the records it is given (17 bytes each) the reduction holds their sorted copy
    with the sort index, and the table at its row width (51 bytes with 14 settings). The counts of every setting held as
    int64 before the cast would alone be 112 bytes per group."""
    tags = [setting_tag(s) for s in IMPACT_SETTINGS]
    records = _random_records(200_000 // reads_per_group, reads_per_group, len(tags))
    tracemalloc.start()
    try:
        groups = molstream._reduce_groups(*records, tags)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    given, table = sum(x.nbytes for x in records), int(groups.memory_usage(index=False).sum())
    assert given == 17 * records[0].size and table == 51 * len(groups)
    assert peak <= 2 * given + 2 * table


def test_more_settings_cells_or_features_than_the_mask_and_the_key_hold_are_refused(synth10x):
    tx = load_transcripts(synth10x.gtf)
    cells, genes = {cb: i for i, cb in enumerate(CALLED)}, {g: i for i, g in enumerate(GENES)}
    many = [Setting(200, 5 + k, 0, False) for k in range(33)]

    class TooMany(dict):
        def __len__(self):
            return (1 << 24) + 1

    for settings, cell_idx, gene_idx in ((many, cells, genes), (SETTINGS_10X, TooMany(cells), genes),
                                         (SETTINGS_10X, cells, {str(i): i for i in range((1 << 16) + 1)})):
        with pytest.raises(ValueError, match="32 bits"):
            scan_read_groups(synth10x.bam, synth10x.fasta, "chrB", tx, settings, cell_idx, gene_idx)
    groups, _, _ = scan_read_groups(synth10x.bam, synth10x.fasta, "chrB", tx, many[:32], cells, genes)
    assert len(groups) == 6 and groups.shape[1] == len(GROUP_COLS) + 32


def _matrix(s):
    return s.outs / "filtered_feature_bc_matrix"


def _to_parts(s, ref, part_dir, meta, settings=SETTINGS_10X):
    return read_groups_to_parts(s.bam, s.fasta, ref, load_transcripts(s.gtf), settings, row_index(_matrix(s) / "barcodes.tsv.gz"),
                                row_index(_matrix(s) / "features.tsv.gz"), part_dir, meta)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_part_meta_names_the_inputs_the_rule_version_and_the_module(synth10x):
    meta = part_meta(synth10x.bam, _matrix(synth10x), SETTINGS_10X)
    st = synth10x.bam.stat()
    assert meta == {"bam": str(synth10x.bam.resolve()), "bam_size": st.st_size, "bam_mtime_ns": st.st_mtime_ns,
                    "barcodes_sha256": _sha(_matrix(synth10x) / "barcodes.tsv.gz"),
                    "features_sha256": _sha(_matrix(synth10x) / "features.tsv.gz"),
                    "settings": TAGS, "unit_bin": UNIT_BIN, "rule_version": molstream.RULE_VERSION,
                    "module_sha256": _sha(molstream.__file__)}
    assert json.loads(json.dumps(meta)) == meta and type(molstream.RULE_VERSION) is int   # stored and compared as json
    assert Path(molstream.__file__).name == "molstream.py"


def test_parts_hold_the_tables_of_the_scan_and_the_json_names_counts_and_inputs(planted, tmp_path):
    part_dir = tmp_path / "parts"
    meta = part_meta(planted.bam, _matrix(planted), SETTINGS_10X)
    for ref in REFS:
        stats = _to_parts(planted, ref, part_dir, meta)
        groups, rec, counts = _scan(planted, ref)
        pd.testing.assert_frame_equal(pd.read_parquet(part_dir / f"{ref}.read_groups.parquet"), groups)
        pd.testing.assert_frame_equal(pd.read_parquet(part_dir / f"{ref}.cell_records.parquet"), rec)
        stored = json.loads((part_dir / f"{ref}.json").read_text())
        assert stored == stats and stored.pop("seconds") >= 0
        assert stored == {"ref": ref} | counts | meta and list(counts) == COUNTS
    assert sorted(p.name for p in part_dir.iterdir()) == sorted(f"{ref}.{kind}" for ref in REFS for kind in (
        "read_groups.parquet", "cell_records.parquet", "json"))
    # a reference without genes (chrC) or without called cells (chrD) writes a read-group part without rows, with the
    # columns and types of a full one
    full = pq.read_schema(part_dir / "chrA.read_groups.parquet")
    assert [str(t) for t in full.types] == GROUP_TYPES + ["uint16"] * 4
    assert full.names == GROUP_COLS + [f"rm_{t}" for t in TAGS]
    for ref in ("chrC", "chrD"):
        part = pq.read_table(part_dir / f"{ref}.read_groups.parquet")
        assert part.num_rows == 0 and part.schema.equals(full)
    assert pq.read_table(part_dir / "chrA.read_groups.parquet").num_rows == 27
    records = pq.read_schema(part_dir / "chrD.cell_records.parquet")
    assert records.names == RECORD_COLS + TAGS
    assert [str(t) for t in records.types] == ["int32", "large_string", "bool"] + ["uint32"] * 8


def test_finished_reference_is_not_scanned_again(synth10x, tmp_path, monkeypatch):
    part_dir = tmp_path / "parts"
    meta = part_meta(synth10x.bam, _matrix(synth10x), SETTINGS_10X)
    first = {ref: _to_parts(synth10x, ref, part_dir, meta) for ref in CHROMS}
    before = {p.name: p.stat().st_mtime_ns for p in part_dir.iterdir()}

    def boom(*args, **kwargs):
        raise AssertionError("scan_read_groups ran again")

    monkeypatch.setattr(molstream, "scan_read_groups", boom)
    assert {ref: _to_parts(synth10x, ref, part_dir, meta) for ref in CHROMS} == first
    assert {p.name: p.stat().st_mtime_ns for p in part_dir.iterdir()} == before
    (part_dir / "chrB.json").unlink()                 # the parquet parts alone do not make a reference finished
    with pytest.raises(AssertionError, match="ran again"):
        _to_parts(synth10x, "chrB", part_dir, meta)


def test_json_is_written_after_the_parquet_parts(synth10x, tmp_path, monkeypatch):
    part_dir = tmp_path / "parts"
    meta = part_meta(synth10x.bam, _matrix(synth10x), SETTINGS_10X)
    real = pd.DataFrame.to_parquet
    for failing in (".read_groups.parquet", ".cell_records.parquet"):
        def boom(self, path, *a, **k):
            if str(path).endswith(failing):
                raise OSError("disk full")
            return real(self, path, *a, **k)

        with monkeypatch.context() as m:
            m.setattr(pd.DataFrame, "to_parquet", boom)
            with pytest.raises(OSError):
                _to_parts(synth10x, "chrA", part_dir, meta)
        assert not (part_dir / "chrA.json").exists()
    assert (part_dir / "chrA.read_groups.parquet").exists()      # left by the second failure, and written again by the rerun
    _to_parts(synth10x, "chrA", part_dir, meta)
    pd.testing.assert_frame_equal(pd.read_parquet(part_dir / "chrA.read_groups.parquet"), _scan(synth10x, "chrA")[0])
    assert (part_dir / "chrA.json").exists()


def _private_copy(s, d):
    """The library under d with its own BAM and matrix files, which a test may change."""
    shutil.copytree(s.outs, d / "outs")
    return dataclasses.replace(s, tmp=d, bam=d / "outs" / "possorted_genome_bam.bam", outs=d / "outs")


def test_parts_of_other_inputs_another_rule_version_or_another_module_are_stale(synth10x, tmp_path, monkeypatch):
    lib = _private_copy(synth10x, tmp_path)
    part_dir = tmp_path / "parts"
    refs = list(CHROMS)

    def stale(settings=SETTINGS_10X):
        return stale_parts(part_dir, refs, part_meta(lib.bam, _matrix(lib), settings))

    _to_parts(lib, "chrB", part_dir, part_meta(lib.bam, _matrix(lib), SETTINGS_10X))
    assert stale() == []                              # chrA has no part yet
    # another settings list: a setting less, and the same settings in another order (the position of a setting is its bit)
    assert stale(SETTINGS_10X[:3]) == ["chrB"] and stale(SETTINGS_10X[::-1]) == ["chrB"]
    with monkeypatch.context() as m:                  # another bin: the units of cell_records are other units
        m.setattr(molstream, "UNIT_BIN", UNIT_BIN // 10)
        assert stale() == ["chrB"]
    with monkeypatch.context() as m:                  # a change of the read-group rule or of the row format
        m.setattr(molstream, "RULE_VERSION", molstream.RULE_VERSION + 1)
        assert stale() == ["chrB"]
    edited = tmp_path / "molstream.py"                # any edit of the module
    edited.write_bytes(Path(molstream.__file__).read_bytes() + b"\n")
    with monkeypatch.context() as m:
        m.setattr(molstream, "__file__", str(edited))
        assert stale() == ["chrB"]
    assert stale() == []
    # the BAM written again: another mtime, another size, another place
    st = lib.bam.stat()
    os.utime(lib.bam, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    assert stale() == ["chrB"]
    with open(lib.bam, "ab") as fh:
        fh.write(b"\0")
    os.utime(lib.bam, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert stale() == ["chrB"] and lib.bam.stat().st_size == st.st_size + 1
    os.truncate(lib.bam, st.st_size)
    os.utime(lib.bam, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert stale() == []
    assert stale_parts(part_dir, refs, part_meta(synth10x.bam, _matrix(lib), SETTINGS_10X)) == ["chrB"]
    # another cell calling, another feature list
    for name, line in (("barcodes.tsv.gz", "TTTTGGGGCCCCAAAA-1\n"), ("features.tsv.gz", "g5\tG5\tGene Expression\n")):
        path = _matrix(lib) / name
        content = path.read_bytes()
        with gzip.open(path, "at") as fh:
            fh.write(line)
        assert stale() == ["chrB"], name
        path.write_bytes(content)
        assert stale() == []


def _stage23(lib, out, *extra):
    script = REPO / "scripts" / "23_read_groups.py"
    return subprocess.run([sys.executable, "-c", RUNNER, str(lib.fasta), str(lib.gtf), str(lib.bam), str(script), "X",
                           "--workers", "2", "--out", str(out), *extra],
                          capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(REPO)), cwd=REPO)


def _stage23_outputs(out):
    """Per reference the two tables and the stored counts, and the summary without its run times."""
    parts = {ref: (pd.read_parquet(out / "parts" / f"{ref}.read_groups.parquet"),
                   pd.read_parquet(out / "parts" / f"{ref}.cell_records.parquet"),
                   {k: v for k, v in json.loads((out / "parts" / f"{ref}.json").read_text()).items() if k in COUNTS})
             for ref in REFS}
    summary = json.loads((out / "read_groups.json").read_text())
    per_ref = {ref: {k: v for k, v in st.items() if k != "seconds"} for ref, st in summary["per_ref"].items()}
    return parts, {k: v for k, v in summary.items() if k not in ("seconds", "scanned")} | {"per_ref": per_ref}


@pytest.fixture(scope="module")
def stage23_fresh(planted, tmp_path_factory):
    out = tmp_path_factory.mktemp("fresh") / "recount"
    r = _stage23(planted, out, "--refs", *REFS, "chrZ")
    assert r.returncode == 0, r.stderr
    return out, r


def test_stage23_parts_are_the_scan_at_the_impact_settings_and_the_summary_sums_them(planted, stage23_fresh):
    out, r = stage23_fresh
    assert "chrZ" in r.stderr                         # asked for, but not in the BAM index
    tags = [setting_tag(s) for s in IMPACT_SETTINGS]
    meta = part_meta(planted.bam, _matrix(planted), IMPACT_SETTINGS)
    summary = json.loads((out / "read_groups.json").read_text())
    total = Counter()
    for ref in REFS:
        groups, rec, counts = _scan(planted, ref, settings=IMPACT_SETTINGS)
        pd.testing.assert_frame_equal(pd.read_parquet(out / "parts" / f"{ref}.read_groups.parquet"), groups)
        pd.testing.assert_frame_equal(pd.read_parquet(out / "parts" / f"{ref}.cell_records.parquet"), rec)
        stored = json.loads((out / "parts" / f"{ref}.json").read_text())
        assert summary["per_ref"][ref] == counts | {"seconds": stored.pop("seconds")}
        assert stored == {"ref": ref} | counts | meta
        total.update(counts)
    assert len(tags) == 14 and meta["settings"] == tags
    assert list(summary) == ["library", "settings", "workers", "scanned", "seconds", *COUNTS, "per_ref"]
    assert (summary["library"], summary["settings"], summary["workers"], summary["scanned"]) == ("X", tags, 2, 4)
    assert {k: summary[k] for k in COUNTS} == dict(total) and list(summary["per_ref"]) == REFS
    assert json.loads(r.stdout.strip().splitlines()[-1]) == {"library": "X", "settings": 14, "refs": 4} | {
        k: summary[k] for k in ("scanned", "records", "primary", "called", "in_groups", "read_groups", "seconds")}
    # the impact settings remove reads of the synthetic groups, and the references without genes or without called cells
    # have a read-group part without rows
    chrA = pd.read_parquet(out / "parts" / "chrA.read_groups.parquet")
    assert len(chrA) == 27 and (chrA[[f"rm_{t}" for t in tags]].sum() > 0).all()
    assert [pq.read_metadata(out / "parts" / f"{ref}.read_groups.parquet").num_rows for ref in REFS] == [27, 6, 0, 0]


def test_stage23_resume_after_crash_and_stale_parts(planted, stage23_fresh, tmp_path):
    fresh_out, _ = stage23_fresh
    lib = _private_copy(planted, tmp_path)
    out = tmp_path / "recount"
    # a run that finished chrA only, then a crash in chrC after its read-group part was written and before its json
    r = _stage23(lib, out, "--refs", "chrA")
    assert r.returncode == 0, r.stderr
    shutil.copy(out / "parts" / "chrA.read_groups.parquet", out / "parts" / "chrC.read_groups.parquet")
    r = _stage23(lib, out)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["scanned"] == 3          # chrB, chrC and chrD, not chrA again
    got, want = _stage23_outputs(out), _stage23_outputs(fresh_out)
    for ref in REFS:
        pd.testing.assert_frame_equal(got[0][ref][0], want[0][ref][0])
        pd.testing.assert_frame_equal(got[0][ref][1], want[0][ref][1])
        assert got[0][ref][2] == want[0][ref][2]
    assert got[1] == want[1]
    # a second resume scans nothing and writes no part
    before = {p.name: p.stat().st_mtime_ns for p in (out / "parts").iterdir()}
    r = _stage23(lib, out)
    assert r.returncode == 0 and json.loads(r.stdout.strip().splitlines()[-1])["scanned"] == 0
    assert {p.name: p.stat().st_mtime_ns for p in (out / "parts").iterdir()} == before and _stage23_outputs(out)[1] == want[1]
    # the BAM was written again: refused, every finished reference named, nothing rewritten
    before |= {"read_groups.json": (out / "read_groups.json").stat().st_mtime_ns}
    bam_times = {p: p.stat().st_mtime_ns for p in (lib.bam, Path(str(lib.bam) + ".bai"))}
    for p, t in bam_times.items():
        os.utime(p, ns=(t, t + 10**9))
    r = _stage23(lib, out)
    assert r.returncode == 1 and "the parts of chrA, chrB, chrC, chrD " in r.stderr and f"gio trash {out / 'parts'}" in r.stderr
    for p, t in bam_times.items():
        os.utime(p, ns=(t, t))
    # parts written with another rule version, and by another pafbench/molstream.py
    for ref, entry in (("chrB", {"rule_version": molstream.RULE_VERSION - 1}), ("chrD", {"module_sha256": "0" * 64})):
        path = out / "parts" / f"{ref}.json"
        text = path.read_text()
        path.write_text(json.dumps(json.loads(text) | entry))
        r = _stage23(lib, out)
        assert r.returncode == 1 and f"the parts of {ref} " in r.stderr and f"gio trash {out / 'parts'}" in r.stderr
        path.write_text(text)
        os.utime(path, ns=(before[path.name], before[path.name]))
    assert {p.name: p.stat().st_mtime_ns for p in [*(out / "parts").iterdir(), out / "read_groups.json"]} == before
