"""The counting rule of stage 24 (pafbench.recount) on the parts of the synthetic 10x library, against the designed tables
of tests/synth.build_10x and plain counts."""
import dataclasses
import gzip
import json
import os
import shutil
import subprocess
import sys
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
import pytest
import scipy.sparse as sp

from pafbench import matrices, molstream, recount, stream
from pafbench.config import IMPACT_SETTINGS, REPO
from pafbench.flags import setting_tag
from pafbench.grid import published
from pafbench.molstream import part_meta, read_groups_to_parts, row_index
from pafbench.transcripts import load_transcripts
from tests.synth import CELLS_10X, CHROMS, SETTINGS_10X, run_original
from tests.test_synth10x import CELL, CELL_RECORDS, MATRIX, _cell_ranger

TAGS = [setting_tag(s) for s in SETTINGS_10X]
REFS = list(CHROMS)
CALLED = sorted(CELLS_10X.values())                 # the rows of barcodes.tsv.gz
GENES = ["g1", "g2", "g3", "g4"]
GROUP_COLS = ["key", "n_reads", "n_winner_reads", "n_pa", "re_class", "end3", "end3_nonpa"] + [f"rm_{t}" for t in TAGS]
EVENT_COLS = ["counted", "multi", "any_removed", "lost", "gained", "switched"]


def _matrix(s):
    return s.outs / "filtered_feature_bc_matrix"


def _write_parts(s, part_dir, settings=SETTINGS_10X):
    meta = part_meta(s.bam, _matrix(s), settings)
    for ref in REFS:
        read_groups_to_parts(s.bam, s.fasta, ref, load_transcripts(s.gtf), settings, row_index(_matrix(s) / "barcodes.tsv.gz"),
                             row_index(_matrix(s) / "features.tsv.gz"), part_dir, meta)
    return part_dir


@pytest.fixture(scope="module")
def parts(synth10x, tmp_path_factory):
    return _write_parts(synth10x, tmp_path_factory.mktemp("recount") / "parts")


def _umi(code):
    return "".join("ACGT"[(code >> 2 * (11 - i)) & 3] for i in range(12))


def _code(umi):
    return int(umi.translate(str.maketrans("ACGT", "0123")), 4)


def _key(cb, ub, gene):
    return (CALLED.index(cb) << 40) | (_code(ub) << 16) | GENES.index(gene)


def _named(block):
    """The block with barcode, UMI and gene in place of the key."""
    key = block.key.to_numpy()
    return block.assign(cb=[CALLED[k >> 40] for k in key.tolist()], ub=[_umi((k >> 16) & 0xFFFFFF) for k in key.tolist()],
                        gene=[GENES[k & 0xFFFF] for k in key.tolist()])


def _blocks(part_dir, refs=REFS, tags=TAGS):
    multi_cu, n_units, n_dup = recount.multi_units(part_dir, refs)
    multi = recount.rule_on_multi(part_dir, refs, multi_cu, tags)
    return [recount.ref_block(part_dir, refs, r, multi_cu, multi, tags) for r in range(len(refs))], (multi_cu, n_units, n_dup)


def test_counted_group_has_strictly_the_most_reads_of_its_cell_and_umi():
    # one row per group, the groups of a cell and UMI next to each other; each line is one cell and UMI
    units = [([3, 1], [True, False]), ([2, 2], [False, False]), ([0, 1], [False, True]), ([0, 0], [False, False]),
             ([5], [True]), ([0], [False]), ([3, 2, 2], [True, False, False]), ([2, 3, 3], [False, False, False]),
             ([1, 2, 2], [False, False, False]), ([1, 0, 0], [True, False, False]), ([2, 3, 1], [False, True, False]),
             ([-1, 1], [False, True])]         # more reads removed than the group has: it does not compete
    reads = np.array([n for u, _ in units for n in u])
    start = np.cumsum([0] + [len(u) for u, _ in units[:-1]])
    assert recount.counted(reads, start).tolist() == [c for _, want in units for c in want]
    assert recount.counted(reads[:0], start[:0]).tolist() == []


def test_events_are_lost_gained_and_switched_as_step_4_defines_them():
    # five cells and UMIs of two groups: nothing changes, the count goes, a count appears, it moves, no count at all
    start = np.arange(0, 10, 2)
    c0 = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0], bool)
    ct = np.array([1, 0, 0, 0, 0, 1, 0, 1, 0, 0], bool)
    lost, gained, switched = recount.events(c0, ct, start)
    assert lost.tolist() == [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    assert gained.tolist() == [0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
    assert switched.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]       # one loss and one gain, on both rows


def test_cells_and_umis_with_two_or_more_groups_are_found_across_references(synth10x, parts):
    g = synth10x.groups
    n_groups = g.groupby(["cb", "ub"]).size()
    multi_cu, n_units, n_dup = recount.multi_units(parts, REFS)
    want = sorted((CALLED.index(cb) << 24) | _code(ub) for cb, ub in n_groups[n_groups > 1].index)
    assert multi_cu.dtype == np.uint64 and multi_cu.tolist() == want and len(want) == 8
    assert (n_units, n_dup) == (len(n_groups), 0) and n_units == 23
    # the competing gene of three_vs_one and of winner_keeps sits on chrB: on chrA alone they have one group
    case = g.drop_duplicates(["cb", "ub"]).set_index(["cb", "ub"]).case
    alone, n_alone, _ = recount.multi_units(parts, ["chrA"])
    names = {case[(CALLED[cu >> 24], _umi(cu & 0xFFFFFF))] for cu in alone.tolist()}
    assert names == {"leads_by_one", "tie", "tie_broken", "three_genes", "three_vs_two", "chain"}
    assert n_alone == n_units - 2                                   # g3_two and alone_xf19 have their only group on chrB
    assert recount.multi_units(parts, ["chrB"])[0].size == 0


def test_blocks_hold_every_group_once_with_the_columns_of_the_part(synth10x, parts):
    blocks, _ = _blocks(parts)
    for ref, block in zip(REFS, blocks):
        part = pd.read_parquet(parts / f"{ref}.read_groups.parquet")
        assert list(block.columns) == GROUP_COLS + EVENT_COLS
        assert block.dtypes.astype(str).tolist() == part.dtypes.astype(str).tolist() + ["bool", "bool"] + ["uint32"] * 4
        pd.testing.assert_frame_equal(block[GROUP_COLS], part)                       # same rows, sorted by key
    assert [len(b) for b in blocks] == [27, 6]


def test_events_of_every_cell_and_umi_are_the_designed_ones(synth10x, parts):
    blocks, _ = _blocks(parts)
    g = _named(pd.concat(blocks))
    ev = synth10x.events
    assert len(g) == len(synth10x.groups) and g.groupby(["cb", "ub"]).ngroups * len(TAGS) == len(ev)
    before = g[g.counted].set_index(["cb", "ub"]).gene
    assert before.index.is_unique                                   # at most one counted gene per cell and UMI
    want = ev.drop_duplicates(["cb", "ub"]).set_index(["cb", "ub"]).before.dropna()
    assert before.to_dict() == want.to_dict()
    multi = synth10x.groups.groupby(["cb", "ub"]).gene.transform("size") > 1
    assert g.set_index(["cb", "ub", "gene"]).multi.to_dict() == dict(zip(zip(synth10x.groups.cb, synth10x.groups.ub,
                                                                             synth10x.groups.gene), multi))
    for t, tag in enumerate(TAGS):
        bit = lambda col: (g[col].to_numpy() >> t) & 1 == 1
        e = ev[ev.tag == tag].set_index(["cb", "ub"])
        after = g[(g.counted & ~bit("lost")) | bit("gained")].set_index(["cb", "ub"]).gene
        assert after.index.is_unique and after.to_dict() == e.after.dropna().to_dict(), tag
        # the bits sit on the group that loses the UMI and on the group that gains it; a switch is on both
        rows = {(r.cb, r.ub, r.gene): (bool(l), bool(ga), bool(s))
                for r, l, ga, s in zip(g.itertuples(), bit("lost"), bit("gained"), bit("switched")) if l or ga or s}
        designed = {}
        for (cb, ub), r in e.iterrows():
            if r.lost:
                designed[(cb, ub, r.before)] = (True, False, bool(r.switched))
            if r.gained:
                designed[(cb, ub, r.after)] = (False, True, bool(r.switched))
        assert rows == designed, tag
        # a read of the group is removed
        removed = g[f"rm_{tag}"].to_numpy() > 0
        assert (bit("any_removed") == removed).all() and removed.any() and not removed.all()
    assert not (g.lost & g.gained).any() and not g.lost[~g.counted].any() and not g.gained[g.counted].any()


def test_two_gene_cases_give_their_designed_loss_gain_or_switch(synth10x, parts):
    """The four cases of design 3.0 with two genes on one cell and UMI, the three-gene case and 3 reads against 2: the
    gene that holds the UMI before filtering and at each of the four test settings (default mode, -i, and their controls)."""
    blocks, _ = _blocks(parts)
    g = _named(pd.concat(blocks)).merge(synth10x.groups[["cb", "ub", "case"]].drop_duplicates(), on=["cb", "ub"])
    got = {}
    for case, rows in g.groupby("case"):
        held = [rows.gene[rows.counted].tolist()]
        lost, gained = rows.lost.to_numpy(), rows.gained.to_numpy()
        for t in range(len(TAGS)):
            held.append(rows.gene[(rows.counted.to_numpy() & ((lost >> t) & 1 == 0)) | ((gained >> t) & 1 == 1)].tolist())
        got[case] = tuple(h[0] if h else None for h in held)
        assert all(len(h) <= 1 for h in held)
    assert {c: got[c] for c in ("three_vs_one", "leads_by_one", "tie", "tie_broken", "three_genes", "three_vs_two")} == {
        "three_vs_one": ("g1", None, "g3", None, "g3"),         # winner on chrA, loser on chrB: a tie, then a switch
        "leads_by_one": ("g1", None, None, "g2", "g2"),
        "tie": (None, None, None, None, None),
        "tie_broken": (None, None, "g2", None, "g2"),           # a gain
        "three_genes": ("g1", None, "g3", None, "g3"),          # a tie of three in default mode, a switch with -i
        "three_vs_two": ("g1", "g2", "g2", "g2", "g2"),         # a switch while a read remains on g1
    }
    sw = g[g.case == "three_vs_two"].set_index("gene")
    assert sw.switched.to_dict() == {"g1": 15, "g2": 15} and sw.lost.to_dict() == {"g1": 15, "g2": 0}
    assert sw.gained.to_dict() == {"g1": 0, "g2": 15} and (sw.n_reads.g1 - sw[f"rm_{TAGS[0]}"].g1) == 1


def test_blocks_equal_the_rule_on_the_whole_table(synth10x, parts):
    """Only the cells and UMIs with two or more groups go through the rule, and a group that is alone holds the UMI while it
    has a read: the same as the one function on every group of the merged table, before filtering and at each setting."""
    blocks, _ = _blocks(parts)
    g = pd.concat(blocks).sort_values("key")
    key = g.key.to_numpy()
    cu = key >> np.uint64(16)
    start = np.flatnonzero(np.r_[True, cu[1:] != cu[:-1]])
    n = g.n_reads.to_numpy(np.int64)
    c0 = recount.counted(n, start)
    assert g.counted.tolist() == c0.tolist()
    for t, tag in enumerate(TAGS):
        ct = recount.counted(n - g[f"rm_{tag}"].to_numpy(np.int64), start)
        lost, gained, switched = recount.events(c0, ct, start)
        for name, want in (("lost", lost), ("gained", gained), ("switched", switched)):
            assert ((g[name].to_numpy() >> t) & 1 == 1).tolist() == want.tolist(), (tag, name)


def _plant(parts, d, ref, rows):
    """A copy of the parts with rows added to the read groups of one reference."""
    shutil.copytree(parts, d)
    path = d / f"{ref}.read_groups.parquet"
    part = pd.read_parquet(path)
    part = pd.concat([part, rows.astype(part.dtypes.to_dict())]).sort_values("key", ignore_index=True)
    part.to_parquet(path, index=False)
    return d


def test_key_on_two_references_is_summed_and_counted(synth10x, parts, tmp_path):
    """Each gene sits on one reference, so this does not happen (none genome-wide, design 2.3). Planted: the g1 group of
    one_inside (alone in its cell and UMI) and the g1 group of three_vs_one (three reads against one of g3) a second time
    in the part of chrB, with two reads each."""
    g = synth10x.groups.set_index("case")
    keys = [_key(r.cb, r.ub, r.gene) for r in (g.loc["one_inside"], g[g.gene == "g1"].loc["three_vs_one"])]
    a = pd.read_parquet(parts / "chrA.read_groups.parquet").set_index("key").loc[keys]
    planted_rm = dict(zip(TAGS, ([1, 1], [1, 2], [0, 0], [0, 2])))      # reads removed from the two planted rows
    extra = pd.DataFrame({"key": np.array(keys, np.uint64), "n_reads": 2, "n_winner_reads": 2, "n_pa": 1, "re_class": 2,
                          "end3": 700, "end3_nonpa": 650} | {f"rm_{t}": rm for t, rm in planted_rm.items()})
    d = _plant(parts, tmp_path / "parts", "chrB", extra)
    blocks, (multi_cu, n_units, n_dup) = _blocks(d)
    assert n_dup == 2 and n_units == 23 and multi_cu.size == 9         # the cell and UMI of one_inside has two rows now
    assert [len(b) for b in blocks] == [27, 6]                         # the row stays in the block of its first reference
    got = blocks[0].set_index("key").loc[keys]
    assert got.n_reads.tolist() == (a.n_reads + 2).tolist() and got.n_winner_reads.tolist() == (a.n_winner_reads + 2).tolist()
    assert got.n_pa.tolist() == [1, 1] and got.re_class.tolist() == [3, 3]       # E on chrA, N in the planted rows
    assert got.end3.tolist() == [-1, -1] and got.end3_nonpa.tolist() == [-1, -1]  # the 3' ends are missing
    for t in TAGS:
        assert got[f"rm_{t}"].tolist() == (a[f"rm_{t}"] + planted_rm[t]).tolist()
    assert got.dtypes.astype(str).tolist() == blocks[1].set_index("key").dtypes.astype(str).tolist()
    assert got.multi.tolist() == [True, True] and got.counted.tolist() == [True, True]
    # one_inside: 3 reads, the settings remove 2 of them and the controls 1: the UMI stays
    assert got.lost.tolist()[0] == 0 and got.any_removed.tolist()[0] == 15
    # three_vs_one: 5 reads of g1 against 1 of g3; -i and its control remove all 5 and the UMI moves to g3
    assert got.lost.tolist()[1] == 0b1010 and got.switched.tolist()[1] == 0b1010
    assert not set(keys) & set(blocks[1].key.tolist())
    # every other group as without the planted rows
    clean, _ = _blocks(parts)
    for b, c in zip(blocks, clean):
        pd.testing.assert_frame_equal(b[~b.key.isin(keys)].reset_index(drop=True), c[~c.key.isin(keys)].reset_index(drop=True))
    # in the sums the key is one group of g1, and g1 stays a gene of its first reference
    tally, clean_tally = _tally(d), _tally(parts)
    assert tally.gene_ref.tolist() == [0, 0, 1, -1]
    assert np.array_equal(tally.matrix("unfiltered").toarray(), clean_tally.matrix("unfiltered").toarray())
    assert np.array_equal(tally.matrix("xf").toarray(), clean_tally.matrix("xf").toarray())


def _dense(counts):
    """{(barcode, gene): n} as a cells by genes array."""
    m = np.zeros((len(CALLED), len(GENES)), np.int64)
    for (cb, gene), n in counts.items():
        m[CALLED.index(cb), GENES.index(gene)] = n
    return m


def _designed_matrices(s):
    """Dense M0 and, per tag, L, G and the matrix after filtering, counted from the events table of the fixture."""
    ev = s.events
    first = ev[ev.tag == TAGS[0]]
    m0 = _dense(Counter(zip(first.cb[first.before.notna()], first.before.dropna())))
    out = {}
    for tag in TAGS:
        e = ev[ev.tag == tag]
        out[tag] = (_dense(Counter(zip(e.cb[e.lost], e.before[e.lost]))), _dense(Counter(zip(e.cb[e.gained], e.after[e.gained]))),
                    _dense(Counter(zip(e.cb[e.after.notna()], e.after.dropna()))))
    return m0, out


def _tally(part_dir, refs=REFS, tags=TAGS):
    blocks, _ = _blocks(part_dir, refs, tags)
    tally = recount.Tally(len(CALLED), len(GENES), tags)
    for r, block in enumerate(blocks):
        tally.add(block, r)
    return tally


def test_tally_gives_the_designed_matrices(synth10x, parts):
    m0, by_tag = _designed_matrices(synth10x)
    tally = _tally(parts)
    got = tally.matrix("unfiltered")
    assert got.format == "csr" and got.dtype == np.int32 and got.has_canonical_format and (got.data > 0).all()
    assert np.array_equal(got.toarray(), m0) and m0.sum() == 20
    # the groups with a winner read: the matrix Cell Ranger wrote, two entries away from the rule (the source patterns)
    xf = tally.matrix("xf").toarray()
    assert np.array_equal(xf, _dense({(CELLS_10X[c], g): n for (c, g), n in MATRIX.items()}))
    assert {(CALLED[i], GENES[j]): (int(xf[i, j]), int(m0[i, j])) for i, j in np.argwhere(xf != m0)} == {
        (CELLS_10X["c4"], "g3"): (0, 1), (CELLS_10X["c5"], "g1"): (2, 1)}
    for tag, (lost, gained, after) in by_tag.items():
        l, g = tally.matrix(("lost", tag)), tally.matrix(("gained", tag))
        assert np.array_equal(l.toarray(), lost) and np.array_equal(g.toarray(), gained), tag
        assert np.array_equal(matrices.filtered(got, l, g).toarray(), after) and np.array_equal(m0 - lost + gained, after)
    assert by_tag[TAGS[1]][1].sum() == 4 and by_tag[TAGS[0]][0].sum() == 6          # gains with -i, losses in default mode
    # each gene sits on one reference; g4 has no read group of a called cell
    assert tally.gene_ref.tolist() == [0, 0, 1, -1]


def test_tally_of_one_reference_counts_a_group_whose_winner_sits_elsewhere(synth10x, parts):
    """The comparison of check A4 on a subset BAM: on chrB alone the g3 groups of three_vs_one and winner_keeps have no
    partner, so the rule counts them there."""
    m0, _ = _designed_matrices(synth10x)
    alone = _tally(parts, ["chrB"]).matrix("unfiltered").toarray()
    c2, c4, g3 = CALLED.index(CELLS_10X["c2"]), CALLED.index(CELLS_10X["c4"]), GENES.index("g3")
    assert (alone[c2, g3], m0[c2, g3], alone[c4, g3], m0[c4, g3]) == (2, 1, 2, 1)
    assert alone[:, :2].sum() == 0 and alone.sum() == 6            # one UMI per read group of chrB


def test_records_of_the_parts_are_summed_per_cell_region_and_gx(synth10x, parts):
    rec = recount.sum_records(parts, REFS, len(CALLED), TAGS)
    assert rec.shape == (len(CALLED) + 1, 4, 2, 4 + len(TAGS)) and rec.dtype == np.int64
    got = {(CELL[CALLED[c]], "ENI"[re_], bool(gx)): tuple(rec[c, re_, gx, [0, 1, 4, 5, 6, 7]].tolist())
           for c, re_, gx in np.argwhere(rec[:-1, :, :, 0] > 0)}
    assert got == CELL_RECORDS
    frames = pd.concat(pd.read_parquet(parts / f"{ref}.cell_records.parquet") for ref in REFS)
    every = frames[frames.cell < 0].groupby(["re", "gx"])[["total", "with_ub", "units", "units_rg"] + TAGS].sum()
    for (re_, gx), row in every.iterrows():                           # the rows over all barcodes are the last cell
        assert rec[-1, "ENI".index(re_), int(gx)].tolist() == row.tolist()
    assert rec[-1, :, :, 0].sum() == every.total.sum() > rec[:-1, :, :, 0].sum()
    units = frames[frames.cell >= 0].groupby("cell")[["units", "units_rg"]].sum()
    assert rec[:-1, :, :, 2].sum(axis=(1, 2)).tolist() == units.units.tolist()
    assert rec[:-1, :, :, 3].sum(axis=(1, 2)).tolist() == units.units_rg.tolist() and units.units_rg.sum() > 0


def _totals(s, part_dir):
    tally = _tally(part_dir)
    m0 = tally.matrix("unfiltered")
    lost, gained = ({t: tally.matrix((kind, t)) for t in TAGS} for kind in ("lost", "gained"))
    with gzip.open(_matrix(s) / "features.tsv.gz", "rt") as fh:
        features = [line.split("\t")[:2] for line in fh]
    rec = recount.sum_records(part_dir, REFS, len(CALLED), TAGS)
    return recount.totals(tally, m0, lost, gained, rec, CALLED, features, REFS), (m0, lost, gained)


def test_totals_equal_sums_over_the_dense_matrices(synth10x, parts):
    (cell, gene), _ = _totals(synth10x, parts)
    m0, by_tag = _designed_matrices(synth10x)
    names = ["unfiltered"] + TAGS
    assert list(cell.columns) == ["cb", "tag", "umis", "genes", "lost", "gained", "switched", "umis_E", "umis_N", "umis_EN",
                                  "mol_partial", "rec_E", "rec_N", "rec_I", "rec_none", "rec_EN_gx", "units_E", "units_N",
                                  "units_I", "units_none", "units_rg_E", "units_rg_N", "rg_E", "rg_N"]
    assert list(gene.columns) == ["gene_id", "gene_name", "ref", "tag", "umis", "cells", "lost", "gained", "umis_E", "umis_N",
                                  "umis_EN"]
    assert cell.tag.tolist() == [t for t in names for _ in CALLED] and cell.cb.tolist() == CALLED * len(names)
    assert gene.tag.tolist() == [t for t in names for _ in GENES] and gene.gene_id.tolist() == GENES * len(names)
    assert gene.gene_name.tolist() == ["G1", "G2", "G3", "G4"] * len(names)
    assert gene.ref.fillna("").tolist() == ["chrA", "chrA", "chrB", ""] * len(names)
    ev, groups = synth10x.events, synth10x.groups.set_index(["cb", "ub", "gene"])
    for tag in names:
        c, g = cell[cell.tag == tag], gene[gene.tag == tag]
        zero = np.zeros_like(m0)
        lost, gained, m = (zero, zero, m0) if tag == "unfiltered" else by_tag[tag]
        assert c.umis.tolist() == m.sum(axis=1).tolist() and c.genes.tolist() == (m > 0).sum(axis=1).tolist(), tag
        assert g.umis.tolist() == m.sum(axis=0).tolist() and g.cells.tolist() == (m > 0).sum(axis=0).tolist(), tag
        assert c.lost.tolist() == lost.sum(axis=1).tolist() and c.gained.tolist() == gained.sum(axis=1).tolist()
        assert g.lost.tolist() == lost.sum(axis=0).tolist() and g.gained.tolist() == gained.sum(axis=0).tolist()
        # by the region class of the group that holds the UMI at the tag; a switch counts once per cell
        e = ev[ev.tag == TAGS[0] if tag == "unfiltered" else ev.tag == tag]
        held = e.before if tag == "unfiltered" else e.after
        re_class = Counter()
        partial, switched = Counter(), Counter()
        for r, gene_held in zip(e.itertuples(), held):
            if tag != "unfiltered":
                switched[r.cb] += r.switched
            if isinstance(gene_held, str):
                row = groups.loc[(r.cb, r.ub, gene_held)]
                re_class[(r.cb, gene_held, row.re_class)] += 1
                if tag != "unfiltered":
                    partial[r.cb] += int(r.before == r.after and row[f"rm_{tag}"] > 0)
        for k, name in ((1, "umis_E"), (2, "umis_N"), (3, "umis_EN")):
            assert c[name].tolist() == [sum(n for (cb, _, rc), n in re_class.items() if cb == x and rc == k) for x in CALLED]
            assert g[name].tolist() == [sum(n for (_, ge, rc), n in re_class.items() if ge == x and rc == k) for x in GENES]
        assert (c.umis_E + c.umis_N + c.umis_EN).tolist() == c.umis.tolist()
        assert c.switched.tolist() == [switched[x] for x in CALLED] and c.mol_partial.tolist() == [partial[x] for x in CALLED]
        # primary records of the cell kept at the tag, by RE, and those with RE E or N and a GX tag
        t = names.index(tag) - 1
        kept = lambda keys: [sum(n[0] - (n[2 + t] if t >= 0 else 0) for k, n in CELL_RECORDS.items() if k[0] == CELL[x] and keys(k))
                             for x in CALLED]
        assert c.rec_E.tolist() == kept(lambda k: k[1] == "E") and c.rec_N.tolist() == kept(lambda k: k[1] == "N")
        assert c.rec_I.tolist() == kept(lambda k: k[1] == "I") and c.rec_none.tolist() == [0] * len(CALLED)
        assert c.rec_EN_gx.tolist() == kept(lambda k: k[1] in "EN" and k[2])
    # pa_and_member and winner_keeps lose a read at the last control and keep their UMI
    assert cell[cell.tag == TAGS[3]].mol_partial.sum() == 2 and cell.switched.sum() == 10 and cell.umis_EN.sum() > 0
    assert cell[cell.tag == TAGS[1]].rec_EN_gx.sum() < cell[cell.tag == TAGS[1]].rec_E.sum() + cell[cell.tag == TAGS[1]].rec_N.sum()
    # units and read groups by RE are numbers of the unfiltered library: empty at the settings
    unit_cols = ["units_E", "units_N", "units_I", "units_none", "units_rg_E", "units_rg_N", "rg_E", "rg_N"]
    assert cell[cell.tag != "unfiltered"][unit_cols].isna().all().all() and cell[cell.tag == "unfiltered"][unit_cols].notna().all().all()
    u = cell[cell.tag == "unfiltered"].set_index("cb")
    frames = pd.concat(pd.read_parquet(parts / f"{ref}.cell_records.parquet") for ref in REFS)
    by_re = frames[frames.cell >= 0].groupby(["cell", "re"])[["units", "units_rg"]].sum()
    for name, col, re_ in (("units_E", "units", "E"), ("units_N", "units", "N"), ("units_I", "units", "I"),
                           ("units_rg_E", "units_rg", "E"), ("units_rg_N", "units_rg", "N")):
        want = [int(by_re[col].get((i, re_), 0)) for i in range(len(CALLED))]
        assert u[name].tolist() == want and sum(want) > 0, name
    assert u.units_none.tolist() == [0] * len(CALLED)
    rg = synth10x.groups
    assert u.rg_E.tolist() == [int(((rg.cb == x) & (rg.re_class & 1 > 0)).sum()) for x in CALLED]
    assert u.rg_N.tolist() == [int(((rg.cb == x) & (rg.re_class & 2 > 0)).sum()) for x in CALLED]
    assert u.rg_E.sum() + u.rg_N.sum() == len(rg) + (rg.re_class == 3).sum()


def _named_entries(frame):
    return {(CELL[CALLED[r.cell]], GENES[r.gene]): (r.ours, r.theirs) for r in frame.itertuples()}


def test_cell_ranger_matrix_is_read_as_cells_by_genes(synth10x):
    cr = recount.read_cr_matrix(_matrix(synth10x))
    assert cr.format == "csr" and cr.dtype == np.int32 and cr.shape == (len(CALLED), len(GENES))
    assert np.array_equal(cr.toarray(), _dense({(CELLS_10X[c], g): n for (c, g), n in MATRIX.items()}))


def test_a1_is_exact_and_a1b_differs_in_the_two_source_pattern_entries(synth10x, parts):
    tally, cr = _tally(parts), recount.read_cr_matrix(_matrix(synth10x))
    a1 = recount.check_entries(tally.matrix("xf"), cr, 0)
    assert (a1["entries_compared"], a1["entries_equal"], a1["entries_differing"], a1["passed"]) == (9, 9, 0, True)
    assert len(a1["differing"]) == 0 and list(a1["differing"].columns) == ["cell", "gene", "ours", "theirs"]
    m0 = tally.matrix("unfiltered")
    a1b = recount.check_entries(m0, cr, 2)                     # on the synthetic library the gate is a count
    assert (a1b["entries_compared"], a1b["entries_equal"], a1b["entries_differing"], a1b["passed"]) == (10, 8, 2, True)
    assert _named_entries(a1b["differing"]) == {("c4", "g3"): (1, 0), ("c5", "g1"): (1, 2)}
    assert a1b["per_million"] == 200_000 and a1b["gate"] == 2
    assert not recount.check_entries(m0, cr, 1)["passed"]


def test_a1_fails_on_an_off_by_one_and_names_the_entry(synth10x, parts, tmp_path):
    """Planted in a copy of the Cell Ranger matrix: one UMI more at c3, g2 and an entry that only Cell Ranger has."""
    d = tmp_path / "filtered_feature_bc_matrix"
    shutil.copytree(_matrix(synth10x), d)
    with gzip.open(d / "matrix.mtx.gz", "rt") as fh:
        lines = fh.read().splitlines()
    c3, c1 = CALLED.index(CELLS_10X["c3"]) + 1, CALLED.index(CELLS_10X["c1"]) + 1
    assert f"2 {c3} 1" in lines                                  # gene, cell, UMIs, 1-based
    lines[lines.index(f"2 {c3} 1")] = f"2 {c3} 2"
    lines[2] = "4 5 10"
    with gzip.open(d / "matrix.mtx.gz", "wt") as fh:
        fh.write("\n".join(lines + [f"4 {c1} 3"]) + "\n")
    a1 = recount.check_entries(_tally(parts).matrix("xf"), recount.read_cr_matrix(d), 0)
    assert (a1["entries_compared"], a1["entries_equal"], a1["entries_differing"], a1["passed"]) == (10, 8, 2, False)
    assert _named_entries(a1["differing"]) == {("c3", "g2"): (1, 2), ("c1", "g4"): (0, 3)}


def _winners(s):
    """{key: n_reads} of the designed read groups with a winner read."""
    w = s.groups[s.groups.n_winner_reads > 0]
    return {_key(r.cb, r.ub, r.gene): r.n_reads for r in w.itertuples()}


def test_a2_joins_every_group_with_a_winner_read_to_molecule_info(synth10x, parts):
    key, n = recount.winner_groups(parts, REFS)
    assert key.dtype == np.uint64 and dict(zip(key.tolist(), n.tolist())) == _winners(synth10x) and (np.diff(key) > 0).all()
    cells, genes = row_index(_matrix(synth10x) / "barcodes.tsv.gz"), row_index(_matrix(synth10x) / "features.tsv.gz")
    mol_key, count = recount.molecule_info(synth10x.outs / "molecule_info.h5", cells, list(genes))
    # the molecules of the uncalled barcode are not among them, and the rows come back in key order (the file is sorted by
    # barcode, feature and UMI)
    assert dict(zip(mol_key.tolist(), count.tolist())) == _winners(synth10x) and (np.diff(mol_key) > 0).all()
    a2 = recount.check_a2(key, n, mol_key, count)
    assert {k: v for k, v in a2.items() if k != "left_over"} == {
        "read_groups": 20, "molecules": 20, "joined": 20, "only_read_groups": 0, "only_molecule_info": 0,
        "read_count_mismatches": 0, "passed": True}
    assert len(a2["left_over"]) == 0 and list(a2["left_over"].columns) == ["key", "n_reads", "count"]


def test_a2_names_what_is_left_over_on_either_side_and_the_read_counts_that_differ(synth10x, parts):
    key, n = recount.winner_groups(parts, REFS)
    mol_key, count = key.copy(), n.copy()
    count[3] += 1                                               # a molecule with another count
    mol_key, count = np.delete(mol_key, [5, 19]), np.delete(count, [5, 19])       # two molecules Cell Ranger does not have
    extra = np.uint64(_key(CELLS_10X["c5"], "TTTTTTTTTTTT", "g4"))                # and one only Cell Ranger has, the last key
    mol_key, count = np.append(mol_key, extra), np.append(count, 7)
    a2 = recount.check_a2(key, n, mol_key, count)
    assert {k: v for k, v in a2.items() if k != "left_over"} == {
        "read_groups": 20, "molecules": 19, "joined": 18, "only_read_groups": 2, "only_molecule_info": 1,
        "read_count_mismatches": 1, "passed": False}
    left = a2["left_over"].astype(object).where(a2["left_over"].notna(), None)
    assert sorted(left.itertuples(index=False, name=None)) == sorted([
        (int(key[3]), int(n[3]), int(n[3]) + 1), (int(key[5]), int(n[5]), None), (int(key[19]), int(n[19]), None),
        (int(extra), None, 7)])
    # nothing on one side
    none = recount.check_a2(key, n, key[:0], n[:0])
    assert (none["joined"], none["only_read_groups"], none["only_molecule_info"], none["passed"]) == (0, 20, 0, False)
    none = recount.check_a2(key[:0], n[:0], key, n)
    assert (none["joined"], none["only_read_groups"], none["only_molecule_info"], none["passed"]) == (0, 0, 20, False)


def test_winner_groups_sums_a_key_on_two_references(synth10x, parts, tmp_path):
    g = synth10x.groups.set_index("case")
    k = _key(g.loc["one_inside"].cb, g.loc["one_inside"].ub, "g1")
    extra = pd.DataFrame({"key": np.array([k], np.uint64), "n_reads": 2, "n_winner_reads": 2, "n_pa": 0, "re_class": 1,
                          "end3": 700, "end3_nonpa": 700} | {f"rm_{t}": 0 for t in TAGS})
    key, n = recount.winner_groups(_plant(parts, tmp_path / "parts", "chrB", extra), REFS)
    assert dict(zip(key.tolist(), n.tolist())) == _winners(synth10x) | {k: 3} and (np.diff(key) > 0).all()


def test_molecule_info_of_another_feature_list_or_cell_calling_is_refused(synth10x, tmp_path):
    cells, genes = row_index(_matrix(synth10x) / "barcodes.tsv.gz"), row_index(_matrix(synth10x) / "features.tsv.gz")
    h5 = synth10x.outs / "molecule_info.h5"
    with pytest.raises(ValueError, match="features/id is not the first column of features.tsv.gz"):
        recount.molecule_info(h5, cells, list(genes)[::-1])
    fewer = {cb: i for i, cb in enumerate(CALLED[1:])}
    with pytest.raises(ValueError, match="1 pass_filter barcodes are not in barcodes.tsv.gz"):
        recount.molecule_info(h5, fewer, list(genes))


@pytest.fixture(scope="module")
def stage13(synth10x):
    """cell_counts and region_counts as stage 13 writes them, from its engine on the synthetic library."""
    tx = load_transcripts(synth10x.gtf)
    regions, cells = [], None
    for ref in REFS:
        _, region, cell = stream.scan_reference(synth10x.bam, synth10x.fasta, ref, tx, SETTINGS_10X, set(), cells=CALLED)
        regions.append(region)
        cells = cell if cells is None else cells + cell
    return cells, stream.merge_regions(regions, TAGS)


def test_a3_ties_the_record_counters_to_stage_13(synth10x, parts, stage13):
    cells, region = stage13
    rec = recount.sum_records(parts, REFS, len(CALLED), TAGS)
    a3 = recount.check_a3(rec, CALLED, cells, region, TAGS)
    assert {k: v for k, v in a3.items() if k != "differing"} == {
        "cells": 5, "settings": 4, "cell_values_compared": 25, "region_rows_compared": 12, "values_differing": 0, "passed": True}
    # stage 13 holds more settings than this stage reads: its other columns and rows are left out
    wide = recount.check_a3(rec, CALLED, cells.assign(c150_l3_m0_e=1),
                            pd.concat([region, region.assign(tag="c150_l3_m0_e")]), TAGS)
    assert wide["passed"] and wide["region_rows_compared"] == 12
    # the join is on the barcode: stage 13 with its rows in another order
    assert recount.check_a3(rec, CALLED, cells.iloc[::-1], region, TAGS)["passed"]


def test_a3_names_the_cell_the_setting_and_the_region_that_differ(synth10x, parts, stage13):
    cells, region = stage13
    rec = recount.sum_records(parts, REFS, len(CALLED), TAGS)
    off = cells.copy()
    off.loc[CALLED[2], TAGS[1]] += 1
    off.loc[CALLED[4], "total"] -= 1
    reg = region.copy()
    reg.loc[(reg.tag == TAGS[3]) & (reg.re == "N"), "removed"] += 2
    a3 = recount.check_a3(rec, CALLED, off.drop(index=CALLED[0]), reg[~((reg.tag == TAGS[0]) & (reg.re == "I"))], TAGS)
    assert (a3["passed"], a3["values_differing"], a3["cells"]) == (False, 5, 4)
    n = region.set_index(["tag", "re"])
    assert sorted(a3["differing"].astype(object).where(a3["differing"].notna(), None).itertuples(index=False, name=None), key=str) == sorted([
        ("cell only here", CALLED[0], "", int(cells.total[CALLED[0]]), None),
        ("cell", CALLED[2], TAGS[1], int(cells.loc[CALLED[2], TAGS[1]]), int(cells.loc[CALLED[2], TAGS[1]]) + 1),
        ("cell", CALLED[4], "total", int(cells.total[CALLED[4]]), int(cells.total[CALLED[4]]) - 1),
        ("region removed, total", "N", TAGS[3], (int(n.removed[TAGS[3], "N"]), int(n.total[TAGS[3], "N"])),
         (int(n.removed[TAGS[3], "N"]) + 2, int(n.total[TAGS[3], "N"]))),
        ("region removed, total", "I", TAGS[0], (int(n.removed[TAGS[0], "I"]), int(n.total[TAGS[0], "I"])), None)], key=str)
    more = recount.check_a3(rec, CALLED, pd.concat([cells, cells.iloc[:1].rename(index={CALLED[0]: "TTTT-1"})]), region, TAGS)
    assert more["differing"].iloc[0].tolist()[:3] == ["cell only in stage 13", "TTTT-1", ""] and not more["passed"]


def _whole_units(part_dir):
    """The tables check_groups is given: the groups that are alone in their cell and UMI, block by block, and the table of
    the cells and UMIs with two or more groups."""
    multi_cu, _, _ = recount.multi_units(part_dir, REFS)
    multi = recount.rule_on_multi(part_dir, REFS, multi_cu, TAGS)
    blocks = [recount.ref_block(part_dir, REFS, r, multi_cu, multi, TAGS) for r in range(len(REFS))]
    return [b[~b.multi].reset_index(drop=True) for b in blocks], multi


def test_a5_invariants_hold_on_the_synthetic_groups(synth10x, parts):
    alone, multi = _whole_units(parts)
    for table in (*alone, multi):
        v = recount.check_groups(table, SETTINGS_10X)
        assert len(v) == 0 and list(v.columns) == ["invariant", "key", "tag"]
    assert len(recount.check_groups(multi.iloc[:0], SETTINGS_10X)) == 0


def _case_key(s, case, gene):
    r = s.groups[(s.groups.case == case) & (s.groups.gene == gene)].iloc[0]
    return _key(r.cb, r.ub, r.gene)


def _violations(table, key, **values):
    """check_groups on the table with the given columns of one group changed, as (invariant, tag) of that group."""
    t = table.copy()
    row = t.index[t.key == np.uint64(key)]
    assert len(row) == 1
    for col, v in values.items():
        t.loc[row, col] = v
    found = recount.check_groups(t, SETTINGS_10X)
    assert (found.key == np.uint64(key)).all()
    return sorted(zip(found.invariant, found.tag))


def test_a5_each_invariant_of_the_groups_with_a_planted_violation(synth10x, parts):
    (chr_a, chr_b), multi = _whole_units(parts)
    e, i, ce, ci = TAGS                                           # default mode, -i, and the control of each
    # more reads removed than the group has: protected has one read and none removed
    k = _case_key(synth10x, "protected", "g1")
    assert _violations(chr_a, k, **{f"rm_{ci}": 2, "lost": 0b1000}) == [("removed_exceeds_reads", ci)]
    # more reads removed at a setting than at its control: two_one_inside loses 1 of 2 at the settings, 2 at the controls
    k = _case_key(synth10x, "two_one_inside", "g1")
    assert _violations(chr_a, k, **{f"rm_{ce}": 0, "lost": 0b1000}) == [("exceeds_control", e)]
    # a loss where nothing is removed
    k = _case_key(synth10x, "protected", "g1")
    assert _violations(chr_a, k, lost=0b0010) == [("event_without_removal", i), ("single_group_loss", i)]
    # a switch where nothing is removed, in a cell and UMI with two groups: tie has two reads on each gene, none removed
    k = _case_key(synth10x, "tie", "g2")
    assert _violations(multi, k, switched=0b0001) == [("event_without_removal", e)]
    # the group holds the UMI with no read left: one_inside loses its read at every setting
    k = _case_key(synth10x, "one_inside", "g1")
    assert _violations(chr_a, k, lost=0b1110) == [("held_without_reads", e), ("single_group_loss", e)]
    # the same in a cell and UMI with two groups: at -i all three reads of g1 in three_vs_one are removed
    k = _case_key(synth10x, "three_vs_one", "g1")
    assert _violations(multi, k, lost=0b0101) == [("held_without_reads", i), ("held_without_reads", ci)]
    # a gain in a cell and UMI with one group: g3_two on chrB, and two_one_inside where a read is removed and one is left
    k = _case_key(synth10x, "g3_two", "g3")
    assert _violations(chr_b, k, counted=False, lost=0, gained=0b0001) == [("event_without_removal", e), ("gain_in_single_group", e)]
    k = _case_key(synth10x, "two_one_inside", "g1")
    assert _violations(chr_a, k, counted=False, lost=0, gained=0b0001) == [("gain_in_single_group", e)]
    # a group that is alone, keeps a read and is called lost
    k = _case_key(synth10x, "two_one_inside", "g1")
    assert _violations(chr_a, k, lost=0b1101) == [("single_group_loss", e)]
    # the planted violations are the only ones, and a group is named by its key
    assert _violations(chr_a, k) == []


def test_a5_invariants_of_the_matrices_and_totals_with_planted_violations(synth10x, parts):
    (cell, gene), (m0, lost, gained) = _totals(synth10x, parts)
    ok = recount.check_matrices(m0, lost, gained)
    assert len(ok) == 0 and list(ok.columns) == ["invariant", "cell", "gene", "tag", "value"]
    sums = {"unfiltered": int(m0.sum())} | {t: int(matrices.filtered(m0, lost[t], gained[t]).sum()) for t in TAGS}
    assert len(recount.check_totals(cell, gene, sums)) == 0
    # a loss above the unfiltered count, and one where nothing was counted: c1 has 8 UMIs of g1 and none of g2
    c1 = CALLED.index(CELLS_10X["c1"])
    more = lost[TAGS[1]].tolil()
    more[c1, 0], more[c1, 1] = 9, 1
    bad = recount.check_matrices(m0, lost | {TAGS[1]: more.tocsr()}, gained)
    assert sorted(bad.itertuples(index=False, name=None)) == [
        ("loss_above_unfiltered", c1, 0, TAGS[1], -1), ("loss_above_unfiltered", c1, 1, TAGS[1], -1),
        ("negative_entry", c1, 0, TAGS[1], -1), ("negative_entry", c1, 1, TAGS[1], -1)]
    # a gain makes up for the loss in the filtered matrix; the loss matrix alone still shows it
    back = gained[TAGS[1]].tolil()
    back[c1, 0], back[c1, 1] = 1, 1
    bad = recount.check_matrices(m0, lost | {TAGS[1]: more.tocsr()}, gained | {TAGS[1]: back.tocsr()})
    assert sorted(bad.invariant) == ["loss_above_unfiltered"] * 2
    # totals: the UMIs of the cells, of the genes and of the matrix are one number per tag
    off = cell.copy()
    off.loc[(off.tag == TAGS[2]) & (off.cb == CALLED[1]), "umis"] += 1
    assert recount.check_totals(off, gene, sums).to_dict("records") == [
        {"tag": TAGS[2], "cells": sums[TAGS[2]] + 1, "genes": sums[TAGS[2]], "matrix": sums[TAGS[2]]}]
    assert recount.check_totals(cell, gene, sums | {"unfiltered": 19}).to_dict("records") == [
        {"tag": "unfiltered", "cells": 20, "genes": 20, "matrix": 19}]
    off = gene.copy()
    off.loc[(off.tag == TAGS[0]) & (off.gene_id == "g1"), "umis"] -= 2
    assert recount.check_totals(cell, off, sums).tag.tolist() == [TAGS[0]]


def _original(s, d, settings=SETTINGS_10X):
    """The BAMs the original tool writes from the synthetic library, under d as stage 02 names them, each with the time
    file of a finished run. Returns {tag: path}."""
    d.mkdir(parents=True, exist_ok=True)
    bams = {}
    for st in settings:
        run_original(s, st.covlen, st.min_len, st.mism, st.introns)
        bams[setting_tag(st)] = d / f"{setting_tag(st)}.bam"
        shutil.copy(s.tmp / f"orig_{st.covlen}_{st.min_len}_{st.mism}_{int(st.introns)}.bam", bams[setting_tag(st)])
        (d / f"{setting_tag(st)}.time.txt").write_text("\tExit status: 0\n")
    return bams


def _library(s, d, stage13):
    """What stage 24 reads for the synthetic library, under d: recount/parts (stage 23 at the four test settings), grid
    (stage 13) and original (the original tool's BAMs)."""
    _write_parts(s, d / "recount" / "parts")
    (d / "grid").mkdir()
    stage13[0].to_parquet(d / "grid" / "cell_counts.parquet")
    stage13[1].to_parquet(d / "grid" / "region_counts.parquet", index=False)
    return _original(s, d / "original")


GATE = 0.25         # of the 9 entries of the synthetic Cell Ranger matrix: the two designed differences of A1b pass, a third does not


def _run(s, d, bams, gate=GATE, **kwargs):
    with pytest.MonkeyPatch.context() as m:
        m.setattr(recount, "RULE_AGREEMENT_MAX", gate)
        return recount.run("X", s.bam, REFS, SETTINGS_10X, d / "recount", d / "matrices", d / "grid", bams, **kwargs)


@pytest.fixture(scope="module")
def stage24(synth10x, stage13, tmp_path_factory):
    d = tmp_path_factory.mktemp("stage24")
    bams = _library(synth10x, d, stage13)
    return d, bams, _run(synth10x, d, bams)


def test_run_passes_every_check_on_the_synthetic_library(synth10x, stage24):
    d, bams, res = stage24
    assert res["passed"] is True and list(res)[-1] == "passed"
    assert json.loads((d / "recount" / "recount.json").read_text()) == res
    assert list(res) == ["library", "rule_version", "settings", "refs", "records", "primary", "called", "xf1_no_ub", "xf1_no_gene",
                         "in_groups", "read_groups", "seconds", "duplicate_keys", "cell_umis", "cell_umis_two_or_more_groups",
                         "counted_molecules", "other_read_groups", "events", "A1", "A1b", "A3", "A5", "A2", "A4", "A6", "passed"]
    assert (res["library"], res["rule_version"], res["settings"], res["refs"]) == ("X", molstream.RULE_VERSION, TAGS, 2)
    # the counts of stage 23 are summed from the part jsons
    stats = [json.loads((d / "recount" / "parts" / f"{ref}.json").read_text()) for ref in REFS]
    assert {k: res[k] for k in molstream.COUNTS} == {k: sum(st[k] for st in stats) for k in molstream.COUNTS}
    assert (res["read_groups"], res["in_groups"], res["records"]) == (33, 63, len(synth10x.reads))
    assert (res["duplicate_keys"], res["cell_umis"], res["cell_umis_two_or_more_groups"]) == (0, 23, 8)
    assert (res["counted_molecules"], res["other_read_groups"]) == (20, 13)
    ev = synth10x.events
    assert res["events"] == {t: {k: int(ev[ev.tag == t][k].sum()) for k in ("lost", "gained", "switched")} for t in TAGS}
    assert res["events"][TAGS[1]] == {"lost": 7, "gained": 4, "switched": 3}
    assert list(res["seconds"]) == ["rule", "checks", "write", "A2", "A4"]
    a1, a1b = res["A1"], res["A1b"]
    assert (a1["entries_compared"], a1["entries_equal"], a1["entries_differing"], a1["passed"]) == (9, 9, 0, True)
    assert (a1b["entries_compared"], a1b["entries_equal"], a1b["entries_differing"], a1b["passed"]) == (10, 8, 2, True)
    assert a1b["gate"] == GATE * 9 and a1b["differing"] == [
        {"cell": CELLS_10X["c4"], "gene": "g3", "ours": 1, "theirs": 0}, {"cell": CELLS_10X["c5"], "gene": "g1", "ours": 1, "theirs": 2}]
    # the two source patterns, each cell and UMI with all its read groups
    assert (a1b["cell_umis_differing"], a1b["rule_only_molecules"], a1b["xf_only_molecules"]) == (2, 1, 1)
    group = lambda c, umi, gene, n_win, rule, xf: {"n_reads": 1, "n_winner_reads": n_win, "counted_by_rule": rule,
                                                   "counted_by_xf": xf, "cell": CELLS_10X[c], "umi": umi, "gene": gene}
    assert a1b["read_groups"] == [group("c4", "GATCCTGAGGTG", "g3", 0, True, False), group("c5", "CTGTTCTTGTTT", "g1", 1, False, True),
                                  group("c5", "CTGTTCTTGTTT", "g2", 0, False, False), group("c5", "CTGTTCTTGTTT", "g3", 0, False, False)]
    assert "check_script" not in a1b
    assert res["A3"] == {"cells": 5, "settings": 4, "cell_values_compared": 25, "region_rows_compared": 12, "values_differing": 0,
                         "passed": True, "differing": []}
    m0, by_tag = _designed_matrices(synth10x)
    assert res["A5"] == {"umis": {"unfiltered": 20} | {t: int(by_tag[t][2].sum()) for t in TAGS}, "violations": {}, "passed": True,
                         "groups": [], "entries": [], "totals": []}
    assert res["A2"] == {"read_groups": 20, "molecules": 20, "joined": 20, "only_read_groups": 0, "only_molecule_info": 0,
                         "read_count_mismatches": 0, "passed": True, "left_over": []}


def _port_matrix(s, tag):
    """Cell Ranger's counting code on the reads that survive at tag, by the port of tests/test_synth10x: one UMI per read
    that comes out as the UMI count (xf 25)."""
    reads = s.reads[s.reads.cb.isin(CALLED) & s.reads.ub.notna() & s.reads.gx.isin(GENES) & ((s.reads.xf & 1) == 1) & ~s.reads[tag]]
    counts = Counter()
    for cb, rs in reads.groupby("cb"):
        port = _cell_ranger([(r.qname, r.ur, r.gx, r.re == "E") for r in rs.itertuples()])
        counts.update((cb, r.gx) for r, (_, xf, _) in zip(rs.itertuples(), port) if xf == 25)
    return _dense(counts)


def test_run_compares_every_original_tool_bam_with_the_matrices_and_reports_a6(synth10x, stage24):
    d, bams, res = stage24
    a4 = res["A4"]
    assert (a4["bams"], a4["refs"], a4["passed"]) == (4, "all", True) and list(a4["settings"]) == TAGS == list(res["A6"])
    _, by_tag = _designed_matrices(synth10x)
    for tag in TAGS:
        one, kept = a4["settings"][tag], synth10x.reads[~synth10x.reads[tag]]
        after = by_tag[tag][2]
        assert (one["bam"], one["records"], one["passed"], one["differing"]) == (str(bams[tag]), len(kept), True, [])
        assert one["entries_compared"] == one["entries_equal"] == int((after > 0).sum()) and one["entries_differing"] == 0
        # A6: the counting code on the raw UMIs of the surviving reads against the rule on their corrected UMIs
        port, a6 = _port_matrix(synth10x, tag), res["A6"][tag]
        differ = np.argwhere(port != after)
        assert a6["differing"] == [{"cell": CALLED[i], "gene": GENES[j], "ours": int(after[i, j]), "theirs": int(port[i, j])}
                                   for i, j in differ]
        assert (a6["entries_differing"], a6["umis_rule"], a6["umis_code"]) == (len(differ), int(after.sum()), int(port.sum()))
        assert a6["umis_only_rule"] == int(np.clip(after - port, 0, None).sum())
        assert a6["umis_only_code"] == int(np.clip(port - after, 0, None).sum())
        assert a6["entries_compared"] == int(((port > 0) | (after > 0)).sum()) and "passed" not in a6
    # the two source patterns are untouched by the first setting: A6 shows them there, one UMI each way
    assert (res["A6"][TAGS[0]]["entries_differing"], res["A6"][TAGS[0]]["umis_only_rule"], res["A6"][TAGS[0]]["umis_only_code"]) == (2, 1, 1)


def test_run_writes_the_molecule_tables_of_the_output_list(synth10x, stage24):
    d, _, _ = stage24
    g, ev = synth10x.groups, synth10x.events
    before = ev.drop_duplicates(["cb", "ub"]).set_index(["cb", "ub"]).before
    held = np.array([before[(r.cb, r.ub)] == r.gene for r in g.itertuples()])
    bits = lambda rows: sum(int(b) << t for t, b in enumerate(rows))
    for name, event, rows in (("counted_molecules", "lost", g[held]), ("other_read_groups", "gained", g[~held])):
        t = pd.read_parquet(d / "recount" / f"{name}.parquet")
        assert list(t.columns) == ["cell", "gene", "umi", "ref", "n_reads", "n_pa", "re_class", "end3", "end3_nonpa",
                                   "any_removed", event, "switched"]
        assert t.dtypes.astype(str).tolist() == ["int32", "int32", "uint32", "category", "uint16", "uint16", "uint8", "int32",
                                                 "int32", "uint32", "uint32", "uint32"]
        assert pq.read_schema(d / "recount" / f"{name}.parquet").field("ref").type == pa.dictionary(pa.int16(), pa.string())
        # written reference by reference, each block sorted by key: cell, UMI, gene
        order = list(zip(t.ref.map(REFS.index), t.cell, t.umi, t.gene))
        assert order == sorted(order) and set(t.ref) == {"chrA", "chrB"}
        got = {(CALLED[r.cell], _umi(r.umi), GENES[r.gene]): tuple(int(x) for x in r[4:]) for r in t.itertuples(index=False)}
        want = {}
        for r in rows.itertuples():
            e = ev[(ev.cb == r.cb) & (ev.ub == r.ub)].set_index("tag").loc[TAGS]
            mine = (e.before if event == "lost" else e.after) == r.gene
            want[(r.cb, r.ub, r.gene)] = (r.n_reads, r.n_pa, r.re_class, r.end3, r.end3_nonpa,
                                          bits(getattr(r, f"rm_{tag}") > 0 for tag in TAGS), bits(e[event] & mine),
                                          bits(e.switched & mine))
        assert got == want and len(t) == len(rows)
        assert dict(zip(zip(t.cell, t.umi, t.gene), t.ref)) == {
            (CALLED.index(r.cb), _code(r.ub), GENES.index(r.gene)): "chrB" if r.gene == "g3" else "chrA" for r in rows.itertuples()}
    counted = pd.read_parquet(d / "recount" / "counted_molecules.parquet")
    assert len(counted) == 20 and counted.lost.sum() > 0 and counted.switched.sum() > 0
    assert not counted.duplicated(["cell", "umi"]).any()                       # at most one molecule per cell and UMI


def test_run_lists_the_source_patterns_in_rule_vs_xf(synth10x, stage24):
    d, _, _ = stage24
    vs = pd.read_parquet(d / "recount" / "rule_vs_xf.parquet")
    assert list(vs.columns) == ["cell", "umi", "gene", "n_reads", "n_winner_reads", "counted_by_rule", "counted_by_xf"]
    assert vs.dtypes.astype(str).tolist() == ["int32", "uint32", "int32", "uint16", "uint16", "bool", "bool"]
    c4, c5 = CALLED.index(CELLS_10X["c4"]), CALLED.index(CELLS_10X["c5"])
    alone, chain = _code("GATCCTGAGGTG"), _code("CTGTTCTTGTTT")
    assert [tuple(r) for r in vs.itertuples(index=False)] == [
        (c4, alone, 2, 1, 0, True, False),                  # alone at its corrected UMI with xf 19: the rule counts it
        (c5, chain, 0, 1, 1, False, True), (c5, chain, 1, 1, 0, False, False), (c5, chain, 2, 1, 0, False, False)]
    # it marks the row of counted_molecules that molecule_info.h5 lacks and the Cell Ranger molecule in other_read_groups
    key = ["cell", "umi", "gene"]
    counted = pd.read_parquet(d / "recount" / "counted_molecules.parquet")[key]
    other = pd.read_parquet(d / "recount" / "other_read_groups.parquet")[key]
    rule_only, xf_only = vs[vs.counted_by_rule & ~vs.counted_by_xf][key], vs[vs.counted_by_xf & ~vs.counted_by_rule][key]
    assert len(rule_only.merge(counted)) == 1 and len(xf_only.merge(other)) == 1 and len(xf_only.merge(counted)) == 0
    cells, genes = row_index(_matrix(synth10x) / "barcodes.tsv.gz"), row_index(_matrix(synth10x) / "features.tsv.gz")
    mol_key, _ = recount.molecule_info(synth10x.outs / "molecule_info.h5", cells, list(genes))
    packed = lambda t: set(((t.cell.to_numpy(np.int64) << 40) | (t.umi.to_numpy(np.int64) << 16) | t.gene.to_numpy()).tolist())
    assert packed(counted) - set(mol_key.tolist()) == packed(rule_only) and set(mol_key.tolist()) - packed(counted) == packed(xf_only)


def test_run_writes_the_matrices_and_totals(synth10x, stage24, parts):
    d, _, res = stage24
    m0, by_tag = _designed_matrices(synth10x)
    assert np.array_equal(matrices.load("X", "unfiltered", d / "matrices").toarray(), m0)
    for tag, (lost, gained, after) in by_tag.items():
        assert np.array_equal(matrices.load("X", tag, d / "matrices").toarray(), after), tag
        assert np.array_equal(sp.load_npz(d / "matrices" / "lost" / f"{tag}.npz").toarray(), lost)
        assert np.array_equal(sp.load_npz(d / "matrices" / "gained" / f"{tag}.npz").toarray(), gained)
    meta = part_meta(synth10x.bam, _matrix(synth10x), SETTINGS_10X)
    assert json.loads((d / "matrices" / "manifest.json").read_text()) == {"tags": TAGS, "shape": [5, 4], "library": "X"} | {
        k: v for k, v in meta.items() if k != "settings"}
    assert (d / "matrices" / "barcodes.tsv").read_text().split() == CALLED
    assert [line.split("\t")[0] for line in (d / "matrices" / "features.tsv").read_text().splitlines()] == GENES
    (cell, gene), _ = _totals(synth10x, parts)
    pd.testing.assert_frame_equal(pd.read_parquet(d / "matrices" / "cell_totals.parquet"), cell)
    pd.testing.assert_frame_equal(pd.read_parquet(d / "matrices" / "gene_totals.parquet"), gene)
    assert sorted(p.name for p in (d / "matrices").iterdir()) == ["barcodes.tsv", "cell_totals.parquet", "features.tsv", "gained",
                                                                  "gene_totals.parquet", "lost", "manifest.json", "unfiltered.npz"]
    assert sorted(p.name for p in (d / "recount").iterdir()) == ["counted_molecules.parquet", "other_read_groups.parquet", "parts",
                                                                 "recount.json", "rule_vs_xf.parquet"]


def _first_gene_on_a_tie(reads, cu_start):
    """Planted in place of recount.counted: on a tie for the maximum the first gene of the cell and UMI holds the UMI."""
    n = np.diff(np.append(cu_start, reads.size))
    top = (reads == np.repeat(np.maximum.reduceat(reads, cu_start), n)) & (reads > 0)
    unit = np.repeat(np.arange(cu_start.size), n)
    rows = np.flatnonzero(top)
    first = np.zeros(reads.size, np.bool_)
    first[rows[np.r_[True, unit[rows][1:] != unit[rows][:-1]]]] = True
    return first


def _last_gene_on_a_tie(reads, cu_start):
    n = np.diff(np.append(cu_start, reads.size))
    return _first_gene_on_a_tie(reads[::-1], (reads.size - (cu_start + n))[::-1])[::-1]


def test_tie_break_that_prefers_the_first_gene_fails_a4_and_raises_the_a1b_count(synth10x, stage13, tmp_path, monkeypatch, capsys):
    """tie, tie_broken and chain are tied before filtering. With the first gene counted, g1 holds all three: the cells and
    UMIs on which the rule and the xf bits differ go from the designed two to three (the chain now agrees with Cell Ranger,
    the two ties do not), on two matrix entries as before. The original tool's BAMs still hold the ties, and a recount of
    them by the documented rule counts nothing there."""
    assert _first_gene_on_a_tie(np.array([2, 2, 1, 3, 3, 0, 1]), np.array([0, 2, 5])).tolist() == [1, 0, 0, 1, 0, 0, 1]
    assert _last_gene_on_a_tie(np.array([2, 2, 1, 3, 3, 0, 1]), np.array([0, 2, 5])).tolist() == [0, 1, 0, 0, 1, 0, 1]
    bams = _library(synth10x, tmp_path, stage13)
    monkeypatch.setattr(recount, "counted", _first_gene_on_a_tie)
    res = _run(synth10x, tmp_path, bams)
    assert "passed" not in res and "passed" not in json.loads((tmp_path / "recount" / "recount.json").read_text())
    a1b = res["A1b"]
    assert (a1b["cell_umis_differing"], a1b["entries_differing"], a1b["passed"], res["A1"]["passed"]) == (3, 2, True, True)
    assert sorted((e["cell"], e["gene"], e["ours"], e["theirs"]) for e in a1b["differing"]) == [
        (CELLS_10X["c2"], "g1", 4, 2), (CELLS_10X["c4"], "g3", 1, 0)]
    a4 = res["A4"]
    assert not a4["passed"] and not any(one["passed"] for one in a4["settings"].values())
    # default mode leaves four ties in c2 (tie, tie_broken, three_vs_one, leads_by_one), the tie of three_genes in c3 and
    # the chain in c5: the planted rule counts g1 in each (ours), the naive recount of the BAM (theirs) nothing
    assert a4["settings"][TAGS[0]]["differing"] == [
        {"cell": CELLS_10X["c2"], "gene": "g1", "ours": 4, "theirs": 0}, {"cell": CELLS_10X["c3"], "gene": "g1", "ours": 1, "theirs": 0},
        {"cell": CELLS_10X["c5"], "gene": "g1", "ours": 2, "theirs": 1}]
    assert "stage 24, X: A4" in capsys.readouterr().err
    # with the last gene counted the matrix entries of A1b rise as well, to four, and the stage stops there
    monkeypatch.setattr(recount, "counted", _last_gene_on_a_tie)
    res = _run(synth10x, tmp_path, bams)
    a1b = res["A1b"]
    assert (a1b["entries_differing"], a1b["cell_umis_differing"], a1b["passed"]) == (4, 4, False) and "A4" not in res
    assert "passed" not in res


def _private(s, d):
    """The library with its own outs folder under d, which a test may change."""
    shutil.copytree(s.outs, d / "outs")
    return dataclasses.replace(s, tmp=d, bam=d / "outs" / "possorted_genome_bam.bam", outs=d / "outs")


def _stopped(d, res, *checks):
    """A run that ended at a failed check: the checks it holds, no passed, and the same on disk."""
    assert "passed" not in res and [k for k in res if k[0] == "A" or k == "resolution"] == list(checks)
    assert json.loads((d / "recount" / "recount.json").read_text()) == res
    return True


def test_failed_a1_stops_before_the_matrices_and_lists_the_read_groups(synth10x, stage13, tmp_path, capsys):
    """One UMI more in the Cell Ranger matrix at c3, g2, and the molecule that would explain it in molecule_info.h5: the
    resolution step joins the read groups to molecule_info and lists the groups of the cells and UMIs that are left over."""
    lib = _private(synth10x, tmp_path)
    bams = _library(lib, tmp_path, stage13)
    path = _matrix(lib) / "matrix.mtx.gz"
    with gzip.open(path, "rt") as fh:
        text = fh.read()
    c3 = CALLED.index(CELLS_10X["c3"])
    assert f"\n2 {c3 + 1} 1\n" in text
    with gzip.open(path, "wt") as fh:
        fh.write(text.replace(f"\n2 {c3 + 1} 1\n", f"\n2 {c3 + 1} 2\n"))
    with h5py.File(lib.outs / "molecule_info.h5", "r+") as f:
        rows = {k: f[k][:] for k in ("barcode_idx", "feature_idx", "umi", "count", "gem_group")}
        extra = {"barcode_idx": f["barcode_info/pass_filter"][c3, 0], "feature_idx": 1, "umi": _code("TTTTTTTTTTTT"), "count": 5,
                 "gem_group": 1}
        for k, v in rows.items():
            del f[k]
            f[k] = np.append(v, np.array([extra[k]], v.dtype))
    res = _run(lib, tmp_path, bams)
    assert _stopped(tmp_path, res, "A1", "A1b", "A2", "resolution")
    assert (res["A1"]["passed"], res["A1"]["entries_differing"]) == (False, 1)
    assert res["A1"]["differing"] == [{"cell": CELLS_10X["c3"], "gene": "g2", "ours": 1, "theirs": 2}]
    a2 = res["A2"]
    assert (a2["joined"], a2["only_read_groups"], a2["only_molecule_info"], a2["passed"]) == (20, 0, 1, False)
    assert a2["left_over"] == [{"n_reads": None, "count": 5, "cell": CELLS_10X["c3"], "umi": "TTTTTTTTTTTT", "gene": "g2"}]
    # the read groups of the cells and UMIs behind A2 and A1b: none for the planted molecule, four for the source patterns
    assert [(g["cell"], g["umi"], g["gene"], g["ref"], g["n_reads"], g["n_winner_reads"]) for g in res["resolution"]] == [
        (CELLS_10X["c4"], "GATCCTGAGGTG", "g3", "chrB", 1, 0), (CELLS_10X["c5"], "CTGTTCTTGTTT", "g1", "chrA", 1, 1),
        (CELLS_10X["c5"], "CTGTTCTTGTTT", "g2", "chrA", 1, 0), (CELLS_10X["c5"], "CTGTTCTTGTTT", "g3", "chrB", 1, 0)]
    err = capsys.readouterr().err
    assert all(f"stage 24, X: {name}\n" in err for name in ("A1", "A1b", "A2", "resolution")) and '"gene": "g2"' in err
    # the tables written during the pass are there, the matrices are not
    assert not (tmp_path / "matrices").exists() and (tmp_path / "recount" / "counted_molecules.parquet").exists()


def test_a1b_above_its_gate_stops_the_stage(synth10x, stage13, tmp_path):
    bams = _library(synth10x, tmp_path, stage13)
    res = _run(synth10x, tmp_path, bams, gate=0.2)               # 1.8 entries: the two designed differences are one too many
    assert _stopped(tmp_path, res, "A1", "A1b", "A2", "resolution")
    assert (res["A1"]["passed"], res["A1b"]["passed"], res["A1b"]["entries_differing"], res["A2"]["passed"]) == (True, False, 2, True)
    assert len(res["resolution"]) == 4 and not (tmp_path / "matrices").exists()


def test_failed_a3_and_a5_stop_before_the_matrices_and_a_passed_of_an_earlier_run_is_gone(synth10x, stage13, tmp_path, capsys):
    bams = _library(synth10x, tmp_path, stage13)
    assert _run(synth10x, tmp_path, bams)["passed"] is True
    shutil.rmtree(tmp_path / "matrices")
    # stage 13 with another total for one cell
    cells = stage13[0].copy()
    cells.loc[CALLED[3], "total"] += 1
    cells.to_parquet(tmp_path / "grid" / "cell_counts.parquet")
    res = _run(synth10x, tmp_path, bams)
    assert _stopped(tmp_path, res, "A1", "A1b", "A3") and not (tmp_path / "matrices").exists()
    assert res["A3"]["differing"] == [{"what": "cell", "name": CALLED[3], "column": "total", "ours": int(stage13[0].total[CALLED[3]]),
                                       "stage13": int(stage13[0].total[CALLED[3]]) + 1}]
    assert f'"name": "{CALLED[3]}"' in capsys.readouterr().err
    stage13[0].to_parquet(tmp_path / "grid" / "cell_counts.parquet")
    # a part in which a control removes more reads of a group than the group has (protected: one read, none removed)
    path = tmp_path / "recount" / "parts" / "chrA.read_groups.parquet"
    part = pd.read_parquet(path)
    k = _case_key(synth10x, "protected", "g1")
    part.loc[part.key == np.uint64(k), f"rm_{TAGS[3]}"] = 3
    part.to_parquet(path, index=False)
    res = _run(synth10x, tmp_path, bams)
    assert _stopped(tmp_path, res, "A1", "A1b", "A3", "A5") and not (tmp_path / "matrices").exists()
    assert res["A5"]["violations"] == {"removed_exceeds_reads": 1} and not res["A5"]["passed"]
    assert res["A5"]["groups"] == [{"invariant": "removed_exceeds_reads", "tag": TAGS[3], "cell": CELLS_10X["c1"],
                                    "umi": "CATGCATGGTAC", "gene": "g1"}]
    assert '"invariant": "removed_exceeds_reads"' in capsys.readouterr().err


def test_failed_a2_and_a4_leave_the_matrices_without_passed(synth10x, stage13, tmp_path, capsys):
    lib = _private(synth10x, tmp_path)
    bams = _library(lib, tmp_path, stage13)
    # A4: the BAM of another setting under the name of the first
    swapped = bams | {TAGS[0]: bams[TAGS[2]]}
    res = _run(lib, tmp_path, swapped)
    assert _stopped(tmp_path, res, "A1", "A1b", "A3", "A5", "A2", "A4", "A6") and (tmp_path / "matrices" / "unfiltered.npz").exists()
    a4 = res["A4"]
    assert [one["passed"] for one in a4["settings"].values()] == [False, True, True, True] and not a4["passed"]
    _, by_tag = _designed_matrices(synth10x)
    # ours is the matrix of this stage at the first setting, theirs the recount of the BAM, here that of the third
    want = [{"cell": CALLED[i], "gene": GENES[j], "ours": int(by_tag[TAGS[0]][2][i, j]), "theirs": int(by_tag[TAGS[2]][2][i, j])}
            for i, j in np.argwhere(by_tag[TAGS[2]][2] != by_tag[TAGS[0]][2])]
    assert a4["settings"][TAGS[0]]["differing"] == want and len(want) > 0
    assert "stage 24, X: A4\n" in capsys.readouterr().err
    # A2: a molecule of molecule_info.h5 with another read count
    with h5py.File(lib.outs / "molecule_info.h5", "r+") as f:
        f["count"][0] += 1
        first = (int(f["barcode_idx"][0]), int(f["feature_idx"][0]), int(f["umi"][0]), int(f["count"][0]))
        barcode = f["barcodes"][first[0]].decode() + "-1"
    res = _run(lib, tmp_path, bams)
    assert _stopped(tmp_path, res, "A1", "A1b", "A3", "A5", "A2", "resolution") and (tmp_path / "matrices" / "unfiltered.npz").exists()
    a2 = res["A2"]
    assert (a2["joined"], a2["read_count_mismatches"], a2["only_read_groups"], a2["only_molecule_info"]) == (20, 1, 0, 0)
    assert a2["left_over"] == [{"n_reads": first[3] - 1, "count": first[3], "cell": barcode, "umi": _umi(first[2]),
                                "gene": GENES[first[1]]}]
    assert [(g["cell"], g["umi"], g["gene"], g["n_reads"]) for g in res["resolution"]] == [
        (barcode, _umi(first[2]), GENES[first[1]], first[3] - 1)]
    assert "stage 24, X: A2\n" in capsys.readouterr().err


def test_run_without_a_bam_for_a4_does_not_pass(synth10x, stage13, tmp_path):
    _library(synth10x, tmp_path, stage13)
    res = _run(synth10x, tmp_path, {})
    assert _stopped(tmp_path, res, "A1", "A1b", "A3", "A5", "A2", "A4", "A6") and res["A4"] == {
        "bams": 0, "refs": "all", "passed": False, "settings": {}}


def test_incomplete_parts_are_refused_with_the_missing_files_named(synth10x, parts, tmp_path):
    meta = part_meta(synth10x.bam, _matrix(synth10x), SETTINGS_10X)
    recount.require_parts(parts, REFS, meta)                    # complete and current: nothing happens
    d = tmp_path / "parts"
    shutil.copytree(parts, d)
    (d / "chrB.cell_records.parquet").unlink()
    with pytest.raises(SystemExit, match="stage 23 is not complete, missing chrB.cell_records.parquet; run scripts/23_read_groups.py"):
        recount.require_parts(d, REFS, meta)
    (d / "chrA.json").unlink()
    (d / "chrA.read_groups.parquet").unlink()
    with pytest.raises(SystemExit, match="missing chrA.json, chrA.read_groups.parquet, chrB.cell_records.parquet;"):
        recount.require_parts(d, REFS, meta)
    # a reference of the BAM index that stage 23 was not run for (its --refs)
    with pytest.raises(SystemExit, match="missing chrC.json, chrC.read_groups.parquet, chrC.cell_records.parquet;"):
        recount.require_parts(parts, [*REFS, "chrC"], meta)


def test_stale_parts_are_refused_with_the_trash_hint(synth10x, parts, tmp_path):
    meta = part_meta(synth10x.bam, _matrix(synth10x), SETTINGS_10X)
    d = tmp_path / "parts"
    shutil.copytree(parts, d)
    done = d / "chrB.json"
    done.write_text(json.dumps(json.loads(done.read_text()) | {"module_sha256": "0" * 64}))
    with pytest.raises(SystemExit) as stop:
        recount.require_parts(d, REFS, meta)
    assert f"{d}: the parts of chrB were made from other inputs" in str(stop.value) and f"`gio trash {d}`" in str(stop.value)
    # parts of other settings, of another rule version: every reference is named
    for other in (part_meta(synth10x.bam, _matrix(synth10x), SETTINGS_10X[:3]), meta | {"rule_version": meta["rule_version"] + 1}):
        with pytest.raises(SystemExit, match="the parts of chrA, chrB were made from other inputs"):
            recount.require_parts(parts, REFS, other)


def test_run_refuses_incomplete_and_stale_parts_and_writes_nothing(synth10x, stage13, tmp_path):
    bams = _library(synth10x, tmp_path, stage13)
    assert _run(synth10x, tmp_path, bams)["passed"] is True
    files = lambda: {str(p.relative_to(tmp_path)): p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    before = files()
    with pytest.raises(SystemExit, match="missing chrC.json"):       # the references come from the BAM index, here one more
        recount.run("X", synth10x.bam, [*REFS, "chrC"], SETTINGS_10X, tmp_path / "recount", tmp_path / "matrices",
                    tmp_path / "grid", bams)
    with pytest.raises(SystemExit, match="gio trash"):                # the parts were written at four settings, not at these
        recount.run("X", synth10x.bam, REFS, SETTINGS_10X[:2], tmp_path / "recount", tmp_path / "matrices", tmp_path / "grid", bams)
    assert files() == before


def test_later_stages_start_only_after_a_passed_recount(tmp_path, monkeypatch):
    d = tmp_path / "X" / "recount"
    d.mkdir(parents=True)
    with pytest.raises(SystemExit, match=f"{d}/recount.json: stage 24 has not passed its acceptance checks for X"):
        recount.require_passed("X", d)
    for content in ({"library": "X", "A1": {"passed": True}}, {"library": "X", "passed": False}, {"library": "X", "passed": "yes"}):
        (d / "recount.json").write_text(json.dumps(content))
        with pytest.raises(SystemExit, match="scripts/24_recount.py X"):
            recount.require_passed("X", d)
    done = {"library": "X", "A1": {"passed": True}, "passed": True}
    (d / "recount.json").write_text(json.dumps(done))
    assert recount.require_passed("X", d) == done
    monkeypatch.setattr(recount, "OUT", tmp_path)                 # without a directory: OUT/<lib>/recount
    assert recount.require_passed("X") == done
    with pytest.raises(SystemExit, match="for Y"):
        recount.require_passed("Y")


def test_passed_recount_of_the_synthetic_library_lets_a_later_stage_start(stage24):
    d, _, res = stage24
    assert recount.require_passed("X", d / "recount") == res


def test_original_bams_are_the_finished_runs_at_the_given_settings(tmp_path):
    tags = TAGS[:2]
    for name in (f"{tags[0]}.bam", f"{tags[1]}.bam", f"chrA_chrB.{tags[0]}.bam"):
        (tmp_path / name).write_bytes(b"")
        (tmp_path / name).with_suffix(".time.txt").write_text("\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:01.00\n\tExit status: 0\n")
    assert recount.original_bams(tmp_path, "", SETTINGS_10X[:2]) == {t: tmp_path / f"{t}.bam" for t in tags}
    assert recount.original_bams(tmp_path, "chrA_chrB.", SETTINGS_10X[:1]) == {tags[0]: tmp_path / f"chrA_chrB.{tags[0]}.bam"}
    with pytest.raises(SystemExit, match=f"missing or not finished: {tmp_path}/{TAGS[2]}.bam, {tmp_path}/{TAGS[3]}.bam$"):
        recount.original_bams(tmp_path, "", SETTINGS_10X)
    # a run that failed, one that was killed (GNU time then still writes exit status 0), one without its time file
    time_file = tmp_path / f"{tags[1]}.time.txt"
    for text in ("\tExit status: 1\n", "Command terminated by signal 9\n\tExit status: 0\n", None):
        time_file.write_text(text) if text else time_file.unlink()
        with pytest.raises(SystemExit, match=f"not finished: {tmp_path}/{tags[1]}.bam$"):
            recount.original_bams(tmp_path, "", SETTINGS_10X[:2])


def _only(bam, ref, out):
    """The records of one reference of a BAM, as a subset BAM of the original tool holds them."""
    with pysam.AlignmentFile(str(bam)) as fh, pysam.AlignmentFile(str(out), "wb", header=fh.header) as dest:
        for a in fh.fetch(until_eof=True):
            if a.reference_name == ref:
                dest.write(a)
    return out


def test_a4_on_a_reference_subset_applies_the_rule_to_those_references_alone(synth10x, stage24, tmp_path):
    """The 1179 path of check A4: the BAMs of the original tool hold chrB only. There the g3 groups of three_vs_one and
    winner_keeps have no partner and are counted, so the stored matrices are the wrong comparison."""
    d, bams, _ = stage24
    subset = {tag: _only(bam, "chrB", tmp_path / f"chrB.{tag}.bam") for tag, bam in bams.items()}
    alone = recount.rule_matrices(d / "recount" / "parts", ["chrB"], TAGS, (len(CALLED), len(GENES)))
    assert list(alone) == TAGS
    cr = _matrix(synth10x)
    c2, c4, g3 = CALLED.index(CELLS_10X["c2"]), CALLED.index(CELLS_10X["c4"]), GENES.index("g3")
    for tag in TAGS:
        _, a4, a6 = recount.check_a4((tag, subset[tag], cr, alone[tag]))
        assert a4["passed"] and a4["entries_compared"] == alone[tag].nnz > 0, tag
        stored = matrices.load("X", tag, d / "matrices")
        _, wrong, _ = recount.check_a4((tag, subset[tag], cr, stored))
        assert not wrong["passed"]
        # winner_keeps: g1 keeps the UMI at every setting; on chrB alone it is g3's
        assert (alone[tag][c4, g3], stored[c4, g3]) == (2, 1)
    assert (alone[TAGS[0]][c2, g3], matrices.load("X", TAGS[0], d / "matrices")[c2, g3]) == (2, 1)


def test_run_with_subset_bams_passes_and_names_the_references(synth10x, stage13, stage24, tmp_path):
    _, full, _ = stage24
    _library(synth10x, tmp_path, stage13)
    subset = {tag: _only(bam, "chrB", tmp_path / f"chrB.{tag}.bam") for tag, bam in full.items()}
    res = _run(synth10x, tmp_path, subset, a4_refs=("chrB",))
    assert res["passed"] is True and (res["A4"]["bams"], res["A4"]["refs"], res["A4"]["passed"]) == (4, ["chrB"], True)
    assert all(one["entries_differing"] == 0 and one["records"] < 25 for one in res["A4"]["settings"].values())
    # without the references named, the same BAMs are compared with the stored matrices and fail
    res = _run(synth10x, tmp_path, subset)
    assert "passed" not in res and not res["A4"]["passed"]


def test_a1b_next_to_the_check_script_of_the_design(synth10x, stage13, tmp_path):
    """The script of the design review lists the read groups of the differing cells and UMIs under exceptions and the
    number of differing entries under A1b_documented_rule (design_m3_draft/review/v2/a1b_genomewide.py)."""
    bams = _library(synth10x, tmp_path, stage13)
    group = lambda c, umi, gene: {"barcode": CELLS_10X[c], "umi": umi, "gene": gene, "n_reads": 1}
    check = tmp_path / "a1b_genomewide_X.json"
    same = {"exceptions": [group("c4", "GATCCTGAGGTG", "g3"), group("c5", "CTGTTCTTGTTT", "g1"), group("c5", "CTGTTCTTGTTT", "g2"),
                           group("c5", "CTGTTCTTGTTT", "g3")], "A1b_documented_rule": {"entries_differing": 2}}
    check.write_text(json.dumps(same))
    res = _run(synth10x, tmp_path, bams, check_script=check)
    assert res["passed"] is True
    assert res["A1b"]["check_script"] == {"file": str(check), "entries_differing": 2, "same": True, "new": [], "gone": []}
    # the script found another cell and UMI, and not the chain: a difference that stays under the gate does not stop the
    # stage, it is written down with the cells and UMIs that are new
    check.write_text(json.dumps({"exceptions": [group("c4", "GATCCTGAGGTG", "g3"), group("c1", "AACCGGTTAACC", "g1")],
                                 "A1b_documented_rule": {"entries_differing": 2}}))
    res = _run(synth10x, tmp_path, bams, check_script=check)
    assert res["passed"] is True and res["A1b"]["check_script"] == {
        "file": str(check), "entries_differing": 2, "same": False, "new": [[CELLS_10X["c5"], "CTGTTCTTGTTT"]],
        "gone": [[CELLS_10X["c1"], "AACCGGTTAACC"]]}
    check.write_text(json.dumps(same | {"A1b_documented_rule": {"entries_differing": 3}}))
    assert _run(synth10x, tmp_path, bams, check_script=check)["A1b"]["check_script"]["same"] is False


def test_read_groups_of_given_cells_and_umis_come_from_every_reference(synth10x, parts):
    g = synth10x.groups
    units = [(CALLED.index(r.cb) << 24) | _code(r.ub) for r in g[g.case.isin(["three_vs_one", "protected"])].itertuples()]
    got = recount.groups_of_units(parts, REFS, np.array(units, np.uint64))
    assert list(got.columns) == ["key", "n_reads", "n_winner_reads", "n_pa", "re_class", "ref"] and got.key.is_monotonic_increasing
    want = g[g.case.isin(["three_vs_one", "protected"])]
    assert sorted(zip(got.key.tolist(), got.n_reads, got.n_winner_reads, got.n_pa, got.re_class, got.ref)) == sorted(
        (_key(r.cb, r.ub, r.gene), r.n_reads, r.n_winner_reads, r.n_pa, r.re_class, "chrB" if r.gene == "g3" else "chrA")
        for r in want.itertuples())
    assert len(got) == 3 and len(recount.groups_of_units(parts, REFS, np.zeros(0, np.uint64))) == 0


# Runs scripts/24_recount.py with pafbench.config pointed at test inputs: BAM(of library X) OUT RULE_AGREEMENT_MAX SCRIPT ARGS...
RUNNER = """
import runpy, sys
from pathlib import Path
import pafbench.config as c
c.LIBRARIES, c.OUT, c.RULE_AGREEMENT_MAX = {"X": Path(sys.argv[1])}, Path(sys.argv[2]), float(sys.argv[3])
sys.argv = sys.argv[4:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
IMPACT_TAGS = [setting_tag(s) for s in IMPACT_SETTINGS]


def _stage24(lib, out, *args, gate=GATE):
    script = REPO / "scripts" / "24_recount.py"
    return subprocess.run([sys.executable, "-c", RUNNER, str(lib.bam), str(out), str(gate), str(script), "X", *args],
                          capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(REPO)), cwd=REPO)


@pytest.fixture(scope="module")
def script_inputs(synth10x, tmp_path_factory):
    """OUT for library X as the script expects it: recount/parts at the 14 impact settings, the stage 13 tables, the
    original tool's BAMs at the 8 published settings and the json of the design's A1b check."""
    out = tmp_path_factory.mktemp("stage24_script") / "out"
    _write_parts(synth10x, out / "X" / "recount" / "parts", IMPACT_SETTINGS)
    tx = load_transcripts(synth10x.gtf)
    regions, cells = [], None
    for ref in REFS:
        _, region, cell = stream.scan_reference(synth10x.bam, synth10x.fasta, ref, tx, IMPACT_SETTINGS, set(), cells=CALLED)
        regions.append(region)
        cells = cell if cells is None else cells + cell
    (out / "X" / "grid").mkdir()
    cells.to_parquet(out / "X" / "grid" / "cell_counts.parquet")
    stream.merge_regions(regions, IMPACT_TAGS).to_parquet(out / "X" / "grid" / "region_counts.parquet", index=False)
    _original(synth10x, out / "X" / "original", published())
    check = out / "design_m3_draft" / "review" / "v2" / "a1b_genomewide_X.json"
    check.parent.mkdir(parents=True)
    check.write_text(json.dumps({"exceptions": [{"barcode": CELLS_10X["c4"], "umi": "GATCCTGAGGTG"},
                                                {"barcode": CELLS_10X["c5"], "umi": "CTGTTCTTGTTT"}],
                                 "A1b_documented_rule": {"entries_differing": 2}}))
    return out


def test_stage24_script_runs_the_stage_at_the_impact_settings(synth10x, script_inputs):
    out = script_inputs
    r = _stage24(synth10x, out)
    assert r.returncode == 0, r.stderr
    res = json.loads((out / "X" / "recount" / "recount.json").read_text())
    summary = json.loads(r.stdout.strip().splitlines()[-1])
    assert summary == {"library": "X", "refs": 2, "read_groups": 33, "counted_molecules": 20, "other_read_groups": 13,
                       "cell_umis_two_or_more_groups": 8, "seconds": res["seconds"]} | {
        name: True for name in ("A1", "A1b", "A2", "A3", "A4", "A5")} | {"passed": True}
    assert res["passed"] is True and list(res)[-1] == "passed" and res["settings"] == IMPACT_TAGS and len(IMPACT_TAGS) == 14
    # A4 on the BAMs of the 8 published settings, A3 at all 14, A1b next to the check script
    published_tags = [setting_tag(s) for s in published()]
    assert list(res["A4"]["settings"]) == published_tags and (res["A4"]["bams"], res["A4"]["refs"]) == (8, "all")
    assert [one["bam"] for one in res["A4"]["settings"].values()] == [str(out / "X" / "original" / f"{t}.bam") for t in published_tags]
    assert (res["A3"]["settings"], res["A3"]["cells"], res["A3"]["passed"]) == (14, 5, True)
    assert res["A1b"]["check_script"]["same"] is True and res["A1b"]["check_script"]["file"].endswith("v2/a1b_genomewide_X.json")
    assert list(res["events"]) == IMPACT_TAGS and sum(e["lost"] for e in res["events"].values()) > 0
    m = out / "X" / "matrices"
    assert json.loads((m / "manifest.json").read_text())["tags"] == IMPACT_TAGS
    assert sorted(p.stem for p in (m / "lost").iterdir()) == sorted(p.stem for p in (m / "gained").iterdir()) == sorted(IMPACT_TAGS)
    assert pd.read_parquet(m / "cell_totals.parquet").tag.unique().tolist() == ["unfiltered", *IMPACT_TAGS]
    assert recount.require_passed("X", out / "X" / "recount") == res
    # the controls remove what their settings remove, group by group: the invariant of A5 that needs the real settings
    assert res["A5"]["passed"] and res["A5"]["violations"] == {}


def test_stage24_script_takes_the_recount_and_the_matrices_folder(synth10x, script_inputs, tmp_path):
    shutil.copytree(script_inputs / "X" / "recount" / "parts", tmp_path / "recount" / "parts")
    r = _stage24(synth10x, script_inputs, "--out", str(tmp_path / "recount"), "--matrices", str(tmp_path / "m"))
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "recount" / "recount.json").read_text())["passed"] is True
    assert (tmp_path / "m" / "unfiltered.npz").exists() and (tmp_path / "recount" / "counted_molecules.parquet").exists()
    assert np.array_equal(matrices.load("X", IMPACT_TAGS[0], tmp_path / "m").toarray(),
                          matrices.load("X", IMPACT_TAGS[0], script_inputs / "X" / "matrices").toarray())


def test_stage24_script_exits_1_without_passed_when_a_check_fails(synth10x, script_inputs, tmp_path):
    """With the gate of the real libraries, one entry per million, the two source patterns of the synthetic library stop
    the stage at A1b."""
    shutil.copytree(script_inputs / "X" / "recount" / "parts", tmp_path / "recount" / "parts")
    r = _stage24(synth10x, script_inputs, "--out", str(tmp_path / "recount"), "--matrices", str(tmp_path / "m"), gate=1e-6)
    assert r.returncode == 1
    summary = json.loads(r.stdout.strip().splitlines()[-1])
    assert (summary["passed"], summary["A1"], summary["A1b"], summary["A2"]) == (False, True, False, True) and "A4" not in summary
    assert "passed" not in json.loads((tmp_path / "recount" / "recount.json").read_text()) and not (tmp_path / "m").exists()
    assert "stage 24, X: A1b\n" in r.stderr and "stage 24, X: resolution\n" in r.stderr
    with pytest.raises(SystemExit):
        recount.require_passed("X", tmp_path / "recount")


def test_stage24_script_refuses_incomplete_and_stale_parts_and_a_missing_original_bam(synth10x, script_inputs, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(script_inputs, out)
    parts = out / "X" / "recount" / "parts"
    files = lambda: {str(p.relative_to(out)): p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    # an original-tool BAM whose run did not finish
    tag = setting_tag(published()[3])
    time_file = out / "X" / "original" / f"{tag}.time.txt"
    time_file.write_text("\tExit status: 1\n")
    before = files()
    r = _stage24(synth10x, out)
    assert r.returncode == 1 and f"missing or not finished: {out / 'X' / 'original' / tag}.bam\n" in r.stderr and files() == before
    time_file.write_text("\tExit status: 0\n")
    # a part of stage 23 is missing
    (parts / "chrB.read_groups.parquet").rename(parts / "kept")
    before = files()
    r = _stage24(synth10x, out)
    assert r.returncode == 1 and "stage 23 is not complete, missing chrB.read_groups.parquet" in r.stderr and files() == before
    (parts / "kept").rename(parts / "chrB.read_groups.parquet")
    # the parts were written by another pafbench/molstream.py
    done = parts / "chrA.json"
    done.write_text(json.dumps(json.loads(done.read_text()) | {"module_sha256": "0" * 64}))
    before = files()
    r = _stage24(synth10x, out)
    assert r.returncode == 1 and "the parts of chrA were made from other inputs" in r.stderr and f"`gio trash {parts}`" in r.stderr
    assert files() == before
