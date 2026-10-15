"""The naive recount of a BAM (pafbench.naive, checks A4 and A6 of stage 24) on the synthetic 10x library: the read groups
it collects, the documented rule on them, and its port of Cell Ranger's UMI correction and low-support step."""
import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pysam
import pytest

from pafbench import naive
from pafbench.flags import setting_tag
from tests.synth import CELLS_10X, HEADER, SETTINGS_10X, UNCALLED_10X, run_original
from tests.test_synth10x import MATRIX, _rule

TAGS = [setting_tag(s) for s in SETTINGS_10X]
CALLED = sorted(CELLS_10X.values())
GENES = ["g1", "g2", "g3", "g4"]


def _code(umi):
    return sum("ACGT".index(b) << 2 * i for i, b in enumerate(reversed(umi)))


def _key(cb, umi, gene):
    return (CALLED.index(cb) << 40) | (_code(umi) << 16) | GENES.index(gene)


def _matrix(s):
    return s.outs / "filtered_feature_bc_matrix"


def _in_groups(reads):
    """The rows of the fixture's read table that are in a read group (design 3.1)."""
    return reads[reads.cb.isin(CALLED) & reads.ub.notna() & reads.gx.isin(GENES) & ((reads.xf & 1) == 1)]


def _dense(counts):
    m = np.zeros((len(CALLED), len(GENES)), np.int64)
    for (cb, gene), n in counts.items():
        m[CALLED.index(cb), GENES.index(gene)] = n
    return m


def test_module_shares_no_code_with_the_pass_or_the_rule():
    """Check A4 compares two counts of the same reads; it says something only while they have no code in common."""
    tree = ast.parse(Path(naive.__file__).read_text())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {(n.module or "pafbench") if n.level == 0 else "pafbench" for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom)}
    assert "pafbench" not in imported and imported <= {"gzip", "array", "itertools", "numpy", "pysam", "scipy"}


def test_read_groups_by_corrected_and_by_raw_umi(synth10x):
    g = _in_groups(synth10x.reads)
    by_ub, by_ur, n = naive.read_groups(synth10x.bam, naive.rows(_matrix(synth10x) / "barcodes.tsv.gz"),
                                        naive.rows(_matrix(synth10x) / "features.tsv.gz"))
    assert n == len(synth10x.reads)
    assert by_ub == dict(Counter(_key(r.cb, r.ub, r.gx) for r in g.itertuples()))
    assert by_ur == dict(Counter(_key(r.cb, r.ur, r.gx) for r in g.itertuples()))
    assert by_ub != by_ur and sum(by_ub.values()) == sum(by_ur.values()) == len(g) and len(by_ub) == 33
    assert all(type(k) is int and type(v) is int for k, v in by_ub.items())


def _bam(path, records):
    """Unsorted, unindexed BAM as the original tool writes it. records: (flag, tags) of 90M records on chrA."""
    hdr = pysam.AlignmentHeader.from_dict(HEADER)
    with pysam.AlignmentFile(str(path), "wb", header=hdr) as out:
        for k, (flag, tags) in enumerate(records):
            a = pysam.AlignedSegment(hdr)
            a.query_name, a.reference_name, a.reference_start, a.cigarstring, a.flag = f"r{k}", "chrA", 1150, "90M", flag
            a.query_sequence, a.query_qualities = "A" * 90, pysam.qualitystring_to_array("I" * 90)
            for tag, v in tags.items():
                a.set_tag(tag, v)
            out.write(a)
    return path


C1 = CALLED[0]
GROUP = dict(CB=C1, UB="AACCGGTTAACC", UR="AACCGGTTAACC", GX="g1", xf=25)


def test_only_primary_xf_bit_1_records_of_called_cells_with_ub_and_one_listed_gene_are_read(synth10x, tmp_path):
    cells, genes = naive.rows(_matrix(synth10x) / "barcodes.tsv.gz"), naive.rows(_matrix(synth10x) / "features.tsv.gz")
    assert cells == {cb: i for i, cb in enumerate(CALLED)} and genes == {g: i for i, g in enumerate(GENES)}
    without = lambda tag: {k: v for k, v in GROUP.items() if k != tag}
    records = [(0, GROUP), (0, GROUP | {"xf": 17}), (0, GROUP | {"xf": 19, "GX": "g2"}),
               (256, GROUP), (2048, GROUP),                                      # secondary, supplementary
               (0, GROUP | {"xf": 0}), (0, GROUP | {"xf": 16}), (0, without("xf")),
               (0, GROUP | {"CB": UNCALLED_10X}), (0, without("CB")), (0, without("UB")), (0, without("GX")),
               (0, GROUP | {"GX": "g1;g2"}), (0, GROUP | {"GX": "g9"}),
               (4, {})]                                                          # unmapped, without tags
    by_ub, by_ur, n = naive.read_groups(_bam(tmp_path / "x.bam", records), cells, genes)
    k = _key(C1, "AACCGGTTAACC", "g1")
    assert n == len(records) and by_ub == by_ur == {k: 2, k + 1: 1}


@pytest.mark.parametrize("tag", ["UB", "UR"])
@pytest.mark.parametrize("umi", ["AACCGGTTAACN", "AACCGGTTAAC", "AACCGGTTAACCA", "aaccggttaacc"])
def test_umi_that_is_not_12_bases_of_acgt_stops_the_recount(synth10x, tmp_path, tag, umi):
    cells, genes = naive.rows(_matrix(synth10x) / "barcodes.tsv.gz"), naive.rows(_matrix(synth10x) / "features.tsv.gz")
    with pytest.raises(ValueError):
        naive.read_groups(_bam(tmp_path / "x.bam", [(0, GROUP | {tag: umi})]), cells, genes)


def test_rule_counts_the_gene_with_strictly_the_most_reads():
    unit = lambda cell, umi: (cell << 40) | (umi << 16)
    reads = {unit(0, 5) | 1: 3, unit(0, 5) | 2: 1,                        # 3 against 1
             unit(0, 6) | 1: 2, unit(0, 6) | 3: 2,                        # a tie counts nothing
             unit(1, 5) | 2: 1,                                           # alone
             unit(1, 7) | 0: 2, unit(1, 7) | 1: 3, unit(1, 7) | 2: 3,     # a tie for the maximum above a third gene
             unit(2, 0) | 3: 1, unit(2, 0) | 0: 4, unit(2, 0) | 1: 4, unit(2, 0) | 2: 5,
             unit(2, (1 << 24) - 1) | 65535: 7}                           # the last UMI code and gene row
    entries = naive.rule_entries(reads)
    assert sorted(entries) == sorted([(0 << 16) | 1, (1 << 16) | 2, (2 << 16) | 2, (2 << 16) | 65535])
    assert len(naive.rule_entries({})) == 0


def test_recount_of_the_unfiltered_bam_equals_the_rule_on_the_designed_groups(synth10x):
    got = naive.recount_bam(synth10x.bam, _matrix(synth10x))
    g = _in_groups(synth10x.reads)
    groups = {}
    for r in g.itertuples():
        groups.setdefault((r.cb, r.ub), Counter())[r.gx] += 1
    want = _dense(Counter((cb, gene) for (cb, _), gene in _rule(groups).items() if gene is not None))
    assert got["rule"].format == "csr" and got["rule"].dtype == np.int32 and got["rule"].shape == (5, 4)
    assert np.array_equal(got["rule"].toarray(), want) and want.sum() == 20
    # by raw UMI the g1 groups of alone_xf19_partner and of chain_end are two groups each
    assert (got["records"], got["in_groups"], got["read_groups"], got["raw_groups"]) == (len(synth10x.reads), len(g), 33, 35)
    assert len(g.groupby(["cb", "ur", "gx"])) == 35
    # on the reads of the unfiltered run the port gives the matrix Cell Ranger wrote: the two source patterns included,
    # where the rule on corrected UMIs differs
    cr = _dense({(CELLS_10X[c], gene): n for (c, gene), n in MATRIX.items()})
    assert np.array_equal(got["port"].toarray(), cr) and got["port"].dtype == np.int32
    assert sorted(zip(*np.nonzero(got["port"].toarray() != want))) == [(3, 2), (4, 0)]      # c4, g3 and c5, g1


@pytest.mark.parametrize("st", SETTINGS_10X, ids=setting_tag)
def test_recount_of_the_original_tools_output_equals_the_rule_on_the_surviving_reads(synth10x, st):
    run_original(synth10x, st.covlen, st.min_len, st.mism, st.introns)
    out = synth10x.tmp / f"orig_{st.covlen}_{st.min_len}_{st.mism}_{int(st.introns)}.bam"
    ev = synth10x.events[synth10x.events.tag == setting_tag(st)]
    want = _dense(Counter(zip(ev.cb[ev.after.notna()], ev.after.dropna())))
    got = naive.recount_bam(out, _matrix(synth10x))
    assert np.array_equal(got["rule"].toarray(), want)
    kept = _in_groups(synth10x.reads[~synth10x.reads[setting_tag(st)]])
    assert got["in_groups"] == len(kept) < len(_in_groups(synth10x.reads))


def test_port_reproduces_the_designed_ub_and_low_support_of_every_read(synth10x):
    """The UB and the xf bit 2 of the fixture are designed by hand from the Cell Ranger source (tests/synth._reads_10x)."""
    g = _in_groups(synth10x.reads)
    n_corrected = n_low = 0
    for cb, reads in g.groupby("cb"):
        counts = dict(Counter((_code(r.ur) << 16) | GENES.index(r.gx) for r in reads.itertuples()))
        corrected, low = naive.correct_and_mark(counts)
        for r in reads.itertuples():
            raw = (_code(r.ur) << 16) | GENES.index(r.gx)
            dest = corrected.get(raw, raw)
            assert dest >> 16 == _code(r.ub) and dest & 0xFFFF == raw & 0xFFFF, r.qname
            assert (dest in low) == bool(r.xf & 2), r.qname
            n_corrected += dest != raw
            n_low += dest in low
    assert n_corrected == 5 and n_low == int(((g.xf & 2) > 0).sum()) > 0


def test_correction_moves_to_the_neighbour_with_more_reads_or_the_larger_umi_at_equal_reads():
    """correct_umis of the source: "If there's a 1-HD UMI w/ greater count, move to that UMI." and "If there's a 1-HD UMI
    w/ equal count, move to the lexicographically larger UMI." (cellranger 9.0.1, lib/rust/tx_annotation/src/mark_dups.rs,
    lines 42 and 43)."""
    k = lambda umi, gene: (_code(umi) << 16) | gene
    counts = {k("AAAAAAAAAAAA", 0): 1, k("AAAAAAAAAAAC", 0): 3, k("TAAAAAAAAAAA", 0): 2,      # to the one with 3 reads
              k("CCCCCCCCCCCC", 0): 2, k("CCCCCCCCCCCG", 0): 2,                               # equal: to the larger UMI
              k("GGGGGGGGGGGG", 0): 2, k("GGGGGGGGGGGT", 1): 5,                               # another gene: stays
              k("ACGTACGTACGT", 2): 1, k("ACGTACGTAATT", 2): 1}                               # two mismatches: stays
    corrected, low = naive.correct_and_mark(counts)
    assert corrected == {k("AAAAAAAAAAAA", 0): k("AAAAAAAAAAAC", 0), k("CCCCCCCCCCCC", 0): k("CCCCCCCCCCCG", 0)}
    assert low == set()
    # low support is decided with one read of each corrected group moved: "Before determining low-support UMI-genes, count
    # one read of each corrected UMI" (lines 226 and 227). Gene 0 has 2 reads at ..AG, which go to ..AT, and gene 1 has 1
    # at ..AG. With one read moved it is 1 against 1 there, a tie, and both are low support; with both moved gene 1 would
    # keep the UMI, with none moved gene 0 would
    counts = {k("AAAAAAAAAAAT", 0): 2, k("AAAAAAAAAAAG", 0): 2, k("AAAAAAAAAAAG", 1): 1}
    corrected, low = naive.correct_and_mark(counts)
    assert corrected == {k("AAAAAAAAAAAG", 0): k("AAAAAAAAAAAT", 0)}
    assert low == {k("AAAAAAAAAAAG", 0), k("AAAAAAAAAAAG", 1)}
    # "Mark as low support umigenes with frequency below the maximum for that UMI." (line 86); a tie marks all
    counts = {k("AAAAAAAAAAAA", 0): 2, k("AAAAAAAAAAAA", 1): 2, k("AAAAAAAAAAAA", 2): 1, k("TTTTTTTTTTTT", 0): 4,
              k("TTTTTTTTTTTT", 3): 1}
    assert naive.correct_and_mark(counts) == ({}, {k("AAAAAAAAAAAA", g) for g in (0, 1, 2)} | {k("TTTTTTTTTTTT", 3)})
