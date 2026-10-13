"""The counting rule of stage 24 (pafbench.recount) on the parts of the synthetic 10x library, against the designed tables
of tests/synth.build_10x and plain counts."""
import shutil

import numpy as np
import pandas as pd
import pytest

from pafbench import recount
from pafbench.flags import setting_tag
from pafbench.molstream import part_meta, read_groups_to_parts, row_index
from pafbench.transcripts import load_transcripts
from tests.synth import CELLS_10X, CHROMS, SETTINGS_10X

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
