"""Stage 28 (pafbench.selection): selection bias, labeled against unlabeled molecules, on synthetic tables and on the
synthetic 10x library of tests/synth.build_10x."""
import importlib.util
import json
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pafbench import recount
from pafbench.burden import END_BINS, _exon_coord, end_bin, end_distances, gene_ends
from pafbench.config import BIAS_POINTS, DIST_END_BINS, EXPR_BINS, POOL_MIN, REPO
from pafbench.evaluate import resample_counts
from pafbench.exons import load_collapsed_exons
from pafbench.selection import (AXES, AXIS_COLUMNS, FAMILY, GENE_COLUMNS, NONPA, PA_ONLY, PAST, POSITIONS, RECORD_COLUMNS,
                                REGIONS, Tally, acceptance, axis_rows, axis_table, decision, downstream_end, end_index,
                                exon_coord, expr_bins_counted, gene_strands, gene_table, labeled_keys, pool, position_codes,
                                records_table, reweight, rule, tally)
from pafbench.strata import JOIN_COLUMNS, cell_classes, expr_bins, join_counted
from tests.synth import CELLS_10X
from tests.test_matched_sites import GTF, _gtf
from tests.test_recount import stage13, stage24  # noqa: F401  (fixtures: stage 24 on the synthetic 10x library)
from tests.test_strata import _stage27_tree

# gL on chr3 plus: collapsed exons [1000, 2000) and [9000, 10000), one end at 10000; for the two widest bins
LONG = [("chr3", "transcript", 1001, 10000, "+", 'gene_id "gL"; transcript_id "t8";'),
        ("chr3", "exon", 1001, 2000, "+", 'gene_id "gL"; transcript_id "t8";'),
        ("chr3", "exon", 9001, 10000, "+", 'gene_id "gL"; transcript_id "t8";')]
GENES = ["gQ", "gP", "gL", "gM"]  # rows of a features.tsv


def _region(lib, labeled, unlabeled, axis="region", stratum="all"):
    return axis_rows(lib, axis, stratum, list(REGIONS), np.array(labeled), np.array(unlabeled))


def test_axis_rows_give_the_shares_in_each_group_and_their_difference_in_points():
    t = _region("L", [30, 60, 10], [500, 400, 100])
    assert list(t.columns) == AXIS_COLUMNS
    assert t.category.tolist() == list(REGIONS) and set(t.lib) == {"L"} and set(t.axis) == {"region"}
    assert t.n_labeled.tolist() == [30, 60, 10] and t.n_unlabeled.tolist() == [500, 400, 100]
    assert t.share_labeled.tolist() == pytest.approx([0.3, 0.6, 0.1])
    assert t.share_unlabeled.tolist() == pytest.approx([0.5, 0.4, 0.1])
    assert t.diff_points.tolist() == pytest.approx([-20, 20, 0])
    # a group without molecules has no share
    e = _region("L", [0, 0, 0], [5, 5, 0])
    assert e.share_labeled.isna().all() and e.diff_points.isna().all() and e.share_unlabeled.tolist() == [0.5, 0.5, 0]


@pytest.mark.parametrize("labeled,fired", [(551, True), (549, False), (550, False), (449, True), (451, False)])
def test_a_planted_difference_of_5_1_points_fires_and_4_9_does_not(labeled, fired):
    # 55.1% against 50.0% (5.1 points) fires, 54.9% (4.9) does not, and neither does 55.0%: "more than 5 points"
    assert BIAS_POINTS == 5
    t = pd.concat([_region("L", [labeled, 1000 - labeled, 0], [500, 500, 0]),
                   axis_rows("L", "family_size", "all", list(FAMILY), np.array([10, 10, 10]), np.array([10, 10, 10]))])
    r = rule(t, None)
    assert r["axes"]["region"]["fired"] is fired and r["axes"]["family_size"]["fired"] is False
    assert r["fired"] == (["region"] if fired else [])
    assert r["axes"]["region"]["max_abs_diff_points"] == pytest.approx(abs(labeled / 10 - 50))
    # E and N differ by the same points with opposite signs
    assert r["axes"]["region"]["category"] in ("E", "N") and r["axes"]["region"]["stratum"] == "all"


def test_the_rule_reads_the_position_axis_in_the_stratum_with_one_read_without_pa():
    cats = ["0-100", "101-300"]
    t = pd.concat([axis_rows("L", "position", "all", cats, np.array([90, 10]), np.array([50, 50])),
                   axis_rows("L", "position", "1", cats, np.array([52, 48]), np.array([50, 50])),
                   axis_rows("L", "position", "2", cats, np.array([90, 10]), np.array([50, 50]))])
    r = rule(t, None)
    assert r["axes"]["position"] == {"stratum": "1", "fired": False, "max_abs_diff_points": pytest.approx(2.0),
                                     "category": "0-100"}
    assert r["fired"] == []


def test_the_record_table_fires_on_the_unit_share_only():
    rec = pd.DataFrame({"re": ["E", "N", "I"], "diff_points_records": [9.0, -9.0, 0.0], "diff_points_units": [5.1, -4.0, -1.1]})
    r = rule(_region("L", [1, 1, 1], [1, 1, 1]), rec)
    assert r["records"] == {"fired": True, "max_abs_diff_points": pytest.approx(5.1), "re": "E"}
    rec["diff_points_units"] = [4.9, -4.0, -0.9]
    r = rule(_region("L", [1, 1, 1], [1, 1, 1]), rec)
    assert r["records"]["fired"] is False and r["fired"] == []


def test_what_a_fired_axis_decides():
    assert AXES == ("region", "cell_class", "expr_bin", "position", "family_size")
    none = decision([], False)
    assert none == {"wording": False, "axes_named": [], "burden_numbers": ["headline", "counted_unweighted"],
                    "reweight_axes": [], "main_text_accuracy": []}
    # axis 4 or the record table alone change the wording and give no reweighted number
    assert decision(["position"], False) == {"wording": True, "axes_named": ["position"],
                                             "burden_numbers": ["headline", "counted_unweighted"], "reweight_axes": [],
                                             "main_text_accuracy": []}
    assert decision([], True)["axes_named"] == ["records"] and decision([], True)["burden_numbers"] == [
        "headline", "counted_unweighted"]
    d = decision(["family_size", "position", "expr_bin", "region"], True)
    assert d["wording"] is True and d["axes_named"] == ["region", "expr_bin", "position", "family_size", "records"]
    assert d["burden_numbers"] == ["headline", "counted_unweighted", "counted_reweighted"]
    assert d["reweight_axes"] == ["region", "expr_bin", "family_size"]
    # the accuracy tables that move into the main text: region from milestone 2, class and expression from stage 27
    assert [x["axis"] for x in d["main_text_accuracy"]] == ["region", "expr_bin"]
    assert decision(["cell_class"], False)["main_text_accuracy"] == [
        {"axis": "cell_class", "table": "eval/strata_m3.parquet", "stratum_kind": "cell_class"}]
    assert decision(["region"], False)["main_text_accuracy"] == [
        {"axis": "region", "table": "eval/metrics.parquet", "stratum_kind": "region"}]


def _index(tmp_path, genes=GENES):
    gtf = _gtf(tmp_path, GTF + LONG)
    return end_index(genes, load_collapsed_exons(gtf), gene_ends(gtf), gene_strands(gtf))


def test_gene_strands(tmp_path):
    assert gene_strands(_gtf(tmp_path, GTF + LONG)) == {"gP": True, "gM": False, "gQ": True, "gL": True}


def test_exon_coord_is_that_of_burden_for_many_genes_at_once(tmp_path):
    gtf = _gtf(tmp_path, GTF + LONG)
    exons, ix = load_collapsed_exons(gtf), _index(tmp_path)
    rng = np.random.default_rng(0)
    gene = rng.integers(0, len(GENES), 2000)
    x = rng.integers(0, 11_000, 2000)
    want = np.array([_exon_coord(*exons[GENES[g]], np.array([p]))[0] for g, p in zip(gene, x)])
    assert exon_coord(ix, gene, x).tolist() == want.tolist()


# base: the 3'-most aligned base of a read, 0-based. gP (plus) has two annotated ends, 1250 (t2) and 1300 (t1), collapsed
# exons [1000, 1100) and [1200, 1300); gM (minus) ends at 2000 (t3) and 2050 (t4), exons [2000, 2100) and [2200, 2300)
@pytest.mark.parametrize("gene,base,dist,exonic,position", [
    ("gP", 1249, 0, True, "0-100"),        # the last base of t2
    ("gP", 1250, 49, True, "0-100"),       # one base past t2: the end of t1, 49 exon bases on
    ("gP", 1299, 0, True, "0-100"),        # the last base of t1
    ("gP", 1050, 99, True, "0-100"),       # 49 + 50 exon bases to t2, 199 along the genome
    ("gP", 1150, 99, False, "0-100"),      # intronic: along the genome to t2
    ("gP", 999, 250, False, "101-300"),    # upstream of the gene
    ("gP", 1300, None, False, PAST),       # past both ends
    ("gM", 2050, 0, True, "0-100"),        # the last base of t4
    ("gM", 2049, 49, True, "0-100"),       # one base past t4: 49 exon bases to t3
    ("gM", 2250, 100, True, "0-100"),      # 50 + 50 exon bases to t4, 200 along the genome
    ("gM", 2150, 100, False, "0-100"),     # intronic
    ("gM", 2300, 250, False, "101-300"),   # upstream on minus
    ("gM", 1999, None, False, PAST),
    ("gQ", 100, 499, True, "301-1000"),
    ("gL", 1000, 1999, True, "1001-5000"),   # 999 + 1000 exon bases
    ("gL", 5000, 4999, False, "1001-5000"),
    ("gL", 3000, 6999, False, ">5000"),
])
def test_distances_on_a_synthetic_gene_with_two_annotated_ends(tmp_path, gene, base, dist, exonic, position):
    ix = _index(tmp_path)
    g, b = np.array([GENES.index(gene)]), np.array([base])
    d, ex = downstream_end(ix, g, b)
    assert (np.isnan(d[0]) if dist is None else d[0] == dist) and ex[0] == exonic
    assert POSITIONS[position_codes(ix, g, b)[0]] == position


def test_positions_are_the_end_bins_then_past_then_pa_read_only():
    assert POSITIONS == END_BINS[:len(DIST_END_BINS) + 1] + (PAST, PA_ONLY)
    assert POSITIONS == ("0-100", "101-300", "301-1000", "1001-5000", ">5000", "past every annotated end", "pa read only")


def test_a_molecule_whose_reads_all_have_a_pa_tag_lands_in_its_own_category(tmp_path):
    ix = _index(tmp_path)
    code = position_codes(ix, np.array([1, 1, 3]), np.array([-1, 1249, -1]))
    assert [POSITIONS[c] for c in code] == [PA_ONLY, "0-100", PA_ONLY]


def test_downstream_end_against_the_nearest_end_of_stage_31(tmp_path):
    """Where stage 31's end_distances finds its nearest end at or downstream, the two agree in distance and mode; where its
    nearest end lies upstream (d < 0), stage 28 takes the nearest end downstream, which lies no nearer, or none."""
    gtf = _gtf(tmp_path, GTF + LONG)
    ix = _index(tmp_path)
    gene = np.repeat(np.arange(len(GENES)), 3000)
    base = np.tile(np.arange(0, 3000), len(GENES)) + np.repeat([0, 800, 0, 1500], 3000)
    plus = np.array([True, True, True, False])[gene]
    sites = pd.DataFrame({"strand": np.where(plus, "+", "-"), "site": base + plus, "genes": np.array(GENES, object)[gene]})
    ref, mode = end_distances(sites, load_collapsed_exons(gtf), gene_ends(gtf))
    d, exonic = downstream_end(ix, gene, base)
    assert (mode == np.where(exonic, "exon", "genome")).all()
    ahead = ref >= 0
    assert ahead.sum() > 1000 and (~ahead).sum() > 1000
    assert (d[ahead] == ref[ahead]).all()
    behind = d[~ahead]
    assert (np.isnan(behind) | (behind >= -ref[~ahead])).all() and np.isnan(behind).any() and (behind > 0).any()
    # the bins are burden.end_bin's for a distance at or downstream
    code = position_codes(ix, gene, base)
    m = ~np.isnan(d)
    assert (np.array(POSITIONS, object)[code[m]] == end_bin(d[m])).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_expr_bins_are_those_of_stage_27_cut_over_the_counted_molecules(seed):
    rng = np.random.default_rng(seed)
    umis = np.where(rng.random(60) < 0.2, 0, rng.integers(1, 400, 60))
    code, bounds = expr_bins_counted(umis)
    genes = np.array([f"g{i}" for i in range(len(umis))], object)
    # one molecule per counted UMI, carrying its gene's UMIs: the bins of strata.expr_bins over them
    ref, ref_bounds = expr_bins(np.repeat(genes, umis), pd.Series(umis, index=genes))
    assert bounds.tolist() == ref_bounds.tolist() and len(bounds) == EXPR_BINS - 1
    assert [f"q{c + 1}" for c in np.repeat(code, umis)] == ref.tolist()
    assert (code[umis == 0] == -1).all() and (code[umis > 0] >= 0).all()
    # each bin holds about a quarter of the counted molecules
    assert np.bincount(np.repeat(code, umis)).max() < umis.sum() / EXPR_BINS + umis.max()


def test_expr_bins_of_one_gene():
    code, bounds = expr_bins_counted(np.array([0, 5]))
    assert code.tolist() == [-1, 0] and bounds.tolist() == [5.0] * (EXPR_BINS - 1)


COUNTED_SCHEMA = pa.schema([("cell", pa.int32()), ("gene", pa.int32()), ("umi", pa.uint32()), ("ref", pa.string()),
                            ("n_reads", pa.uint16()), ("n_pa", pa.uint16()), ("re_class", pa.uint8()), ("end3", pa.int32()),
                            ("end3_nonpa", pa.int32()), ("any_removed", pa.uint32()), ("lost", pa.uint32()),
                            ("switched", pa.uint32())])


def _counted(path, rows):
    """counted_molecules.parquet from (cell, gene row, umi, n_reads, n_pa, re_class, end3_nonpa) rows."""
    cols = list(zip(*rows))
    t = {"cell": cols[0], "gene": cols[1], "umi": cols[2], "ref": ["chr1"] * len(rows), "n_reads": cols[3], "n_pa": cols[4],
         "re_class": cols[5], "end3": [max(x, 0) for x in cols[6]], "end3_nonpa": cols[6]}
    t |= {k: [0] * len(rows) for k in ("any_removed", "lost", "switched")}
    pq.write_table(pa.table(t, schema=COUNTED_SCHEMA), path, row_group_size=2)
    return path


# cells 0 (rod) and 1 (cone); gene rows of GENES: gQ 0, gP 1, gL 2, gM 3
TOY = [(0, 1, 5, 1, 1, 1, -1),       # labeled, its read has pa: pa read only
       (0, 1, 6, 3, 0, 1, 1249),     # three reads, none with pa; ends on the last base of t2
       (1, 3, 5, 2, 1, 2, 2300),     # labeled, intronic, one read without pa, 250 nt upstream on minus
       (1, 3, 7, 2, 1, 3, 1999),     # n_pa above 0 but not joined: unlabeled; past both ends of gM
       (1, 0, 9, 4, 0, 1, 100)]      # 499 nt from the end of gQ
TOY_JOIN = pd.DataFrame({"lib": "L", "mol": [10, 11, 12, 13], "cell": np.array([0, 1, 1, 0], np.int32),
                         "gene": np.array([1, 3, 0, -1], np.int32), "umi": np.array([5, 5, 9, 8], np.uint32),
                         "joined": [True, True, False, False], "region": "3UTR"})


@pytest.mark.parametrize("batch_rows", [2, 1000])
def test_tally_counts_each_counted_molecule_once_by_axis(tmp_path, batch_rows):
    ix = _index(tmp_path)
    keys = labeled_keys(TOY_JOIN)
    assert len(keys) == 2   # the joined rows only: m12 lies in a counted molecule's key but is not joined
    t = tally(_counted(tmp_path / "c.parquet", TOY), keys, np.array([1, 0]), 2, ix, batch_rows=batch_rows)
    assert t.full.shape == (2, len(REGIONS), 2, len(FAMILY), len(NONPA), len(POSITIONS), 2) and t.full.sum() == 5
    assert t.full.sum(axis=(1, 2, 3, 4, 5)).tolist() == [[2, 1], [0, 2]]   # [labeled][n_pa above 0]
    assert t.by_gene.shape == (len(GENES), len(REGIONS), 2, len(FAMILY))
    assert t.by_gene.sum(axis=(1, 2, 3)).tolist() == [1, 2, 0, 2]
    assert t.labeled.to_dict("list") == {"cell": [0, 1], "gene": [1, 3], "umi": [5, 5], "region": [0, 1], "cell_class": [1, 0],
                                         "family_size": [0, 1]}
    tab = axis_table(t, ["cone", "rod"], np.array([0, 1, 2, 3]), "L")
    assert list(tab.columns) == AXIS_COLUMNS and set(tab.lib) == {"L"}

    def get(axis, stratum="all"):
        x = tab[(tab.axis == axis) & (tab.stratum == stratum)]
        return dict(zip(x.category, zip(x.n_labeled, x.n_unlabeled)))

    assert get("region") == {"E": (1, 2), "N": (1, 0), "EN": (0, 1)}
    assert get("cell_class") == {"cone": (1, 2), "rod": (1, 1)}
    assert get("expr_bin") == {"q1": (0, 1), "q2": (1, 1), "q3": (0, 0), "q4": (1, 1)}
    assert get("family_size") == {"1": (1, 0), "2": (1, 1), "3+": (0, 2)}
    assert get("position") == {"0-100": (0, 1), "101-300": (1, 0), "301-1000": (0, 1), "1001-5000": (0, 0), ">5000": (0, 0),
                               PAST: (0, 1), PA_ONLY: (1, 0)}
    # at equal numbers of reads without pa the pa read only category has no place
    assert get("position", "1") == {"0-100": (0, 0), "101-300": (1, 0), "301-1000": (0, 0), "1001-5000": (0, 0),
                                    ">5000": (0, 0), PAST: (0, 1)}
    assert get("position", "2") == dict.fromkeys(POSITIONS[:-1], (0, 0))
    assert get("position", "3+")["0-100"] == (0, 1) and get("position", "3+")["301-1000"] == (0, 1)
    assert sorted(set(tab[tab.axis == "position"].stratum)) == ["1", "2", "3+", "all"]


def _totals():
    """cell_totals rows at unfiltered of two cells."""
    return pd.DataFrame({"cb": ["A-1", "B-1"], "rec_E": [100, 200], "rec_N": [300, 300], "rec_I": [50, 50], "rec_none": [0, 0],
                         "units_E": [80, 120], "units_N": [200, 200], "units_I": [40, 40], "units_none": [0, 0],
                         "units_rg_E": [70, 70], "units_rg_N": [150, 150], "rg_E": [60, 60], "rg_N": [140, 140]})


def test_records_table_sets_labeled_molecules_against_records_and_duplicate_free_units():
    re = np.array(["E"] * 30 + ["N"] * 60 + ["I"] * 10, object)
    t = records_table(re, _totals(), "L")
    assert list(t.columns) == RECORD_COLUMNS and t.re.tolist() == ["E", "N", "I"] and set(t.lib) == {"L"}
    assert t.labeled.tolist() == [30, 60, 10] and t.records.tolist() == [300, 600, 100] and t.units.tolist() == [200, 400, 80]
    assert t.share_labeled.tolist() == pytest.approx([0.3, 0.6, 0.1])
    assert t.share_records.tolist() == pytest.approx([0.3, 0.6, 0.1])
    assert t.share_units.tolist() == pytest.approx([200 / 680, 400 / 680, 80 / 680])
    assert t.diff_points_records.tolist() == pytest.approx([0, 0, 0])
    assert t.diff_points_units.tolist() == pytest.approx([100 * (x - u / 680) for x, u in ((0.3, 200), (0.6, 400), (0.1, 80))])
    # units per read group with a record of that RE, printed next to the table; RE I is in no read group
    assert t.units_rg.tolist() == [140, 300, 0] and t.read_groups.tolist() == [120, 280, 0]
    assert t.units_per_read_group.tolist()[:2] == pytest.approx([140 / 120, 300 / 280]) and np.isnan(t.units_per_read_group[2])


def test_gene_table_ranks_the_genes_furthest_from_the_library_fraction():
    ids = [f"g{i}" for i in range(7)]
    umis = np.array([100, 60, 40, 200, 500, 0, 80])
    labeled = np.array([10, 30, 40, 0, 0, 0, 8])
    t = gene_table(ids, [f"N{i}" for i in range(7)], umis, labeled, "L", min_umis=50, top=2)
    assert list(t.columns) == GENE_COLUMNS and set(t.lib) == {"L"}
    # g5 has no counted molecule; g2 has fewer than min_umis and is not ranked
    assert t.gene_id.tolist() == ["g0", "g1", "g2", "g3", "g4", "g6"] and t.gene_name.tolist()[0] == "N0"
    assert t.labeled_fraction.tolist() == pytest.approx([0.1, 0.5, 1.0, 0.0, 0.0, 0.1])
    # library fraction 88 / 980: above, g1 and then g0 (g0 and g6 tie, more UMIs first); below, g4 and g3 (a tie at 0)
    assert dict(zip(t.gene_id, t.top)) == {"g0": "above", "g1": "above", "g2": "", "g3": "below", "g4": "below", "g6": ""}


def _accept(**kw):
    t = tally_stub(labeled=2, unlabeled=3, per_gene=[1, 2, 0, 2])
    args = {"t": t, "keys": np.array([1, 2], np.uint64), "m0_sum": 5, "gene_umis": np.array([1, 2, 0, 2]), "join": TOY_JOIN,
            "n_called": 4, "records": pd.Series([700, 750], index=["A-1", "B-1"]),
            "stage13": pd.Series([750, 700], index=["B-1", "A-1"])}
    return acceptance(**(args | kw))


def tally_stub(labeled, unlabeled, per_gene):
    full = np.zeros((2, len(REGIONS), 1, len(FAMILY), len(NONPA), len(POSITIONS), 2), np.int64)
    full[1, 0, 0, 0, 1, 0, 1], full[0, 0, 0, 0, 1, 0, 0] = labeled, unlabeled
    by_gene = np.zeros((len(per_gene), len(REGIONS), 1, len(FAMILY)), np.int64)
    by_gene[:, 0, 0, 0] = per_gene
    return Tally(full, by_gene, pd.DataFrame({"cell": [0] * labeled}))


def test_acceptance():
    a = _accept()
    assert a["passed"] is True
    assert a["molecules"] == {"labeled": 2, "unlabeled": 3, "m0_sum": 5, "joined_keys": 2, "genes_differing": 0, "passed": True}
    assert a["truth"] == {"joined": 2, "not_joined": 2, "labeled_in_called_cells": 4, "passed": True}
    assert a["records"] == {"records": 1450, "stage13": 1450, "cells": 2, "cells_differing": 0, "passed": True}
    assert _accept(m0_sum=6)["molecules"]["passed"] is False and _accept(m0_sum=6)["passed"] is False
    # a joined key that is no counted molecule
    assert _accept(keys=np.array([1, 2, 3], np.uint64))["molecules"]["passed"] is False
    # the gene totals of stage 24 other than the counted molecules per gene
    assert _accept(gene_umis=np.array([1, 2, 1, 1]))["molecules"]["genes_differing"] == 2
    assert _accept(n_called=5)["truth"]["passed"] is False
    # per cell, not only in sum; and a cell missing on one side
    r = _accept(records=pd.Series([750, 700], index=["A-1", "B-1"]))["records"]
    assert r == {"records": 1450, "stage13": 1450, "cells": 2, "cells_differing": 2, "passed": False}
    r = _accept(records=pd.Series([700], index=["A-1"]))["records"]
    assert r["passed"] is False and r["cells_differing"] == 1


def test_pooling_on_a_toy_table():
    assert POOL_MIN == 30
    first = np.array(["a", "a", "a", "a", "b", "b", "b", "c", "d", "d", "e"])
    n_lab = np.array([100, 10, 15, 3, 50, 20, 12, 5, 0, 40, 31])
    g = pool(first, n_lab)
    # a: 10 + 15 + 3 = 28 labeled, still under 30, joins the largest stratum of a; b: 20 + 12 reach 30 and stay a stratum of
    # their own next to 50; c: alone and small, nothing to merge with; d: the empty stratum joins the 40; e: large enough
    assert g.tolist() == [0, 0, 0, 0, 4, 5, 5, 7, 9, 9, 10]
    # the first stratum of a category is not always the largest
    assert pool(np.array(["a", "a", "a"]), np.array([3, 40, 50])).tolist() == [2, 1, 2]
    assert pool(np.array(["a", "a"]), np.array([20, 20]), min_n=10).tolist() == [0, 1]


def _two_strata(counted_e, counted_n):
    counted = pd.DataFrame({"gene": [0, 1], "region": ["E", "N"], "n": [counted_e, counted_n]})
    labeled = pd.DataFrame({"gene": [0] * 40 + [1] * 60, "region": ["E"] * 40 + ["N"] * 60,
                            "templated": [True] * 10 + [False] * 30 + [True] * 45 + [False] * 15})
    return counted, labeled


def test_the_reweighted_share_equals_the_unweighted_one_when_the_mixes_are_equal():
    r = reweight(*_two_strata(400, 600), ["region"], n_boot=20)
    assert r["share_counted_unweighted"] == pytest.approx(0.55) and r["share_counted_reweighted"] == pytest.approx(0.55)
    # E is 80% of the counted molecules and 40% of the labeled ones: 0.8 x 10 / 40 + 0.2 x 45 / 60
    r = reweight(*_two_strata(800, 200), ["region"], n_boot=20)
    assert r["share_counted_unweighted"] == pytest.approx(0.55) and r["share_counted_reweighted"] == pytest.approx(0.35)
    assert r["axes"] == ["region"] and r["strata"] == 2 and r["strata_final"] == 2 and r["strata_pooled"] == 0
    assert r["pooled_share"] == 0 and r["n_boot"] == 20


@pytest.mark.parametrize("axes", [["position"], ["region", "position"], ["records"]])
def test_axis_4_is_refused_as_a_reweighting_axis(axes):
    with pytest.raises(ValueError, match="cannot enter the reweighting"):
        reweight(*_two_strata(400, 600), axes)


def _boot_toy():
    counted = pd.DataFrame([(1, "E", 100), (1, "N", 50), (2, "E", 30), (2, "N", 120), (3, "E", 60), (4, "N", 40)],
                           columns=["gene", "region", "n"])
    rows = [(1, "E", 40, 10), (1, "N", 35, 30), (2, "E", 31, 5), (2, "N", 32, 20), (3, "E", 33, 3), (4, "N", 34, 30)]
    labeled = pd.DataFrame([(g, r, k < t) for g, r, n, t in rows for k in range(n)], columns=["gene", "region", "templated"])
    return counted, labeled


def test_the_gene_bootstrap_recomputes_the_weights_in_every_resample():
    counted, labeled = _boot_toy()
    P = {s: counted[counted.region == s].n.sum() / counted.n.sum() for s in "EN"}
    weight = {s: P[s] / (labeled.region == s).mean() for s in "EN"}   # the weights of the full data
    differs = []
    for seed in range(5):
        r = reweight(counted, labeled, ["region"], n_boot=1, seed=seed)
        w = resample_counts(4, 1, seed)[0]   # genes 1 to 4, sorted
        N = {s: sum(w[g - 1] * n for g, x, n in counted.itertuples(index=False) if x == s) for s in "EN"}
        lab = {s: sum(w[g - 1] for g, x, _ in labeled.itertuples(index=False) if x == s) for s in "EN"}
        tem = {s: sum(w[g - 1] for g, x, t in labeled.itertuples(index=False) if x == s and t) for s in "EN"}
        want = sum(N[s] / (N["E"] + N["N"]) * tem[s] / lab[s] for s in "EN")
        assert r["lo"] == pytest.approx(want) and r["hi"] == pytest.approx(want) and r["n_finite"] == 1
        held = sum(weight[s] * tem[s] for s in "EN") / sum(weight[s] * lab[s] for s in "EN")
        differs.append(abs(held - want) > 1e-6)
    assert any(differs)   # with the weights of the full data the resamples would give other values
    r = reweight(counted, labeled, ["region"], n_boot=1000, seed=0)
    assert r["lo"] < r["share_counted_reweighted"] < r["hi"] and r["n_finite"] <= 1000
    assert reweight(counted, labeled, ["region"], n_boot=1000, seed=0) == r


def test_a_resample_without_labeled_molecules_in_a_stratum_is_left_out():
    # stratum N has its labeled molecules in gene 2 only, and counted molecules in genes 1 and 2
    counted = pd.DataFrame({"gene": [1, 1, 2], "region": ["E", "N", "N"], "n": [50, 50, 50]})
    labeled = pd.DataFrame({"gene": [1] * 40 + [2] * 40, "region": ["E"] * 40 + ["N"] * 40, "templated": [True, False] * 40})
    r = reweight(counted, labeled, ["region"], n_boot=200, seed=1)
    w = resample_counts(2, 200, 1)
    assert r["n_finite"] == int((w[:, 1] > 0).sum()) < 200


def test_reweight_pools_on_the_first_fired_axis_in_the_merge_order():
    # strata (region, family size): E/1 has 40 labeled, E/2 10 and E/3+ 5 (merged, still under 30: they join E/1);
    # N/1 35 labeled; the axes are given out of order
    counted = pd.DataFrame({"gene": [1, 1, 1, 2], "region": ["E", "E", "E", "N"], "family_size": ["1", "2", "3+", "1"],
                            "n": [400, 300, 200, 100]})
    labeled = pd.DataFrame({"gene": [1] * 55 + [2] * 35, "region": ["E"] * 55 + ["N"] * 35,
                            "family_size": ["1"] * 40 + ["2"] * 10 + ["3+"] * 5 + ["1"] * 35,
                            "templated": [True] * 20 + [False] * 20 + [True] * 10 + [True] * 5 + [False] * 35})
    r = reweight(counted, labeled, ["family_size", "region"], n_boot=10)
    assert r["axes"] == ["region", "family_size"]
    assert (r["strata"], r["strata_small"], r["strata_pooled"], r["strata_final"]) == (4, 2, 2, 2)
    assert r["pooled_share"] == pytest.approx(500 / 1000)
    # E pooled: 900 of 1,000 counted molecules, 35 of 55 labeled templated; N: 0 of 35
    assert r["share_counted_reweighted"] == pytest.approx(0.9 * 35 / 55)


def _stage28():
    spec = importlib.util.spec_from_file_location("stage28", REPO / "scripts" / "28_selection_bias.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    return stage


def _stage28_tree(s, d, root):
    """What stage 28 reads for the synthetic library X next to the outputs of stage 24 in d: stage 27's tree (truth tables,
    classes) with the RE of each molecule's pa read and stage 13's cell_counts, and eval/truth_to_counted.parquet from
    stage 27's join. Returns (base, cells template, eval folder, molecules of called cells, the join)."""
    base, cells, mol, tr, _ = _stage27_tree(s, d, root)
    rec = s.reads.assign(qname=s.reads.qname.astype(str)).set_index("qname")
    path = base / "X" / "truth" / "molecules_labeled.parquet"
    allm = pd.read_parquet(path)
    first_pa = tr[tr.role == "pa"].drop_duplicates("mol").set_index("mol").qname
    allm.assign(re=rec.re.reindex(first_pa.reindex(allm.mol)).to_numpy()).to_parquet(path)
    shutil.copy(d / "grid" / "cell_counts.parquet", base / "X" / "grid" / "cell_counts.parquet")
    join = join_counted(mol, tr, pd.read_parquet(base / "X" / "pa_sites.parquet"), sorted(CELLS_10X.values()),
                        np.array(["g1", "g2", "g3", "g4"], object), d / "recount" / "counted_molecules.parquet", "X")
    (root / "eval").mkdir()
    join[JOIN_COLUMNS].to_parquet(root / "eval" / "truth_to_counted.parquet")
    return base, cells, root / "eval", pd.read_parquet(path)[lambda m: m.in_cell].reset_index(drop=True), join


def test_stage28_on_the_synthetic_10x_library(synth10x, stage24, tmp_path, monkeypatch, capsys):
    d, _, _ = stage24
    stage = _stage28()
    monkeypatch.setattr(recount, "LIBRARIES", {"X": synth10x.bam})
    base, cells, ev, mol, join = _stage28_tree(synth10x, d, tmp_path)
    out = tmp_path / "burden"
    argv = ["28", "X", "--base", str(base), "--cells", cells, "--matrices", str(d / "matrices"), "--recount", str(d / "recount"),
            "--eval", str(ev), "--gtf", str(synth10x.gtf), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    stage.main()
    js = json.loads((out / "selection_bias_m3.json").read_text())
    x = js["libraries"]["X"]
    assert x["acceptance"]["passed"] is True and js["acceptance_passed"] is True
    # 20 counted molecules; four truth molecules join one each (A, B, C, D); E lies in a group that is not counted and F
    # in none
    assert x["acceptance"]["molecules"] == {"labeled": 4, "unlabeled": 16, "m0_sum": 20, "joined_keys": 4, "genes_differing": 0,
                                            "passed": True}
    assert x["acceptance"]["truth"] == {"joined": 4, "not_joined": 2, "labeled_in_called_cells": 6, "passed": True}
    assert x["acceptance"]["records"]["passed"] is True and x["acceptance"]["records"]["cells"] == 5

    # the join takes GX of the pa read: B (our gene g2, GX g1) labels the counted molecule of g1
    genes = pd.read_parquet(out / "selection_bias_genes.parquet")
    assert list(genes.columns) == GENE_COLUMNS
    assert dict(zip(genes.gene_id, genes.labeled)) == {"g1": 3, "g2": 1, "g3": 0}
    cm = pd.read_parquet(d / "recount" / "counted_molecules.parquet")
    assert dict(zip(genes.gene_id, genes.umis)) == cm.gene.map({0: "g1", 1: "g2", 2: "g3", 3: "g4"}).value_counts().to_dict()

    # the axis table equals the functions on the same inputs; A, whose only read has pa, has its own category
    tab = pd.read_parquet(out / "selection_bias_m3.parquet")
    assert list(tab.columns) == AXIS_COLUMNS and set(tab.axis) == set(AXES)
    barcodes = sorted(CELLS_10X.values())
    cls = cell_classes(np.array(barcodes, object), pd.read_parquet(tmp_path / "cells" / "X" / "cells.parquet"))
    classes, code = np.unique(cls, return_inverse=True)
    gt = pd.read_parquet(d / "matrices" / "gene_totals.parquet")
    umis = gt[gt.tag == "unfiltered"].set_index("gene_id").umis.reindex(["g1", "g2", "g3", "g4"], fill_value=0).to_numpy()
    gtf = synth10x.gtf
    ix = end_index(["g1", "g2", "g3", "g4"], load_collapsed_exons(gtf), gene_ends(gtf), gene_strands(gtf))
    t = tally(d / "recount" / "counted_molecules.parquet", labeled_keys(join), code, len(classes), ix)
    pd.testing.assert_frame_equal(tab, axis_table(t, classes.tolist(), expr_bins_counted(umis)[0], "X"))
    pa_only = tab[(tab.axis == "position") & (tab.category == PA_ONLY)]
    assert pa_only.stratum.tolist() == ["all"] and pa_only.n_labeled.tolist() == [1] and pa_only.n_unlabeled.tolist() == [0]
    assert set(tab[tab.axis == "cell_class"].category) == {"cone", "rod", "unassigned"}

    rec = pd.read_parquet(out / "selection_bias_records.parquet")
    ct = pd.read_parquet(d / "matrices" / "cell_totals.parquet")
    pd.testing.assert_frame_equal(rec, records_table(mol.re.to_numpy(object), ct[ct.tag == "unfiltered"], "X"))

    # burden numbers: the headline over the six labeled molecules of called cells (A, C, D templated), the unweighted share
    # over the four joined (A, C, D of A to D); the reweighted share only when an axis among 1, 2, 3 and 5 fires
    rw = pd.read_parquet(out / "templated_share_reweighted.parquet")
    assert list(rw.columns) == ["lib", "axes", "share_headline", "share_counted_unweighted", "share_counted_reweighted", "lo",
                                "hi", "strata", "strata_pooled", "pooled_share"]
    assert rw.share_headline.tolist() == [0.5] and rw.share_counted_unweighted.tolist() == [0.75]
    r = rule(tab, rec)
    assert x["rule"] == json.loads(json.dumps(r)) and x["decision"] == decision(r["fired"], r["records"]["fired"])
    fired = x["decision"]["reweight_axes"]
    assert fired and rw["axes"].tolist() == [",".join(fired)]   # rw.axes is the DataFrame attribute
    assert x["reweighting"]["axes"] == fired and rw.strata.tolist() == [x["reweighting"]["strata"]]
    # four labeled molecules: every stratum is small and pools within its region; N has counted molecules (the intronic
    # case) and no labeled one, so the reweighted share has no value
    assert x["reweighting"]["strata_final"] == 3 and np.isnan(rw.share_counted_reweighted[0])
    assert x["reweighting"]["share_counted_reweighted"] is None and x["burden"]["share_counted_reweighted"] is None
    assert x["burden"]["share_headline"] == 0.5 and x["burden"]["share_counted_unweighted"] == 0.75
    assert js["threshold_points"] == BIAS_POINTS and js["pool_min"] == POOL_MIN
    assert x["join_by_region"] == {"3UTR": {"joined": 3, "not_joined": 0}, "CDS": {"joined": 1, "not_joined": 2}}
    assert x["n_pa"] == {"labeled": {"n_pa_above_0": 3, "n_pa_0": 1}, "unlabeled": {"n_pa_above_0": 0, "n_pa_0": 16}}
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["libraries"] == ["X"] and line["acceptance_passed"] is True and "seconds" in line

    # stage 13 with one record more in a cell: every output is written and the stage exits non-zero
    cc = pd.read_parquet(base / "X" / "grid" / "cell_counts.parquet")
    cc.iloc[0, cc.columns.get_loc("total")] += 1
    cc.to_parquet(base / "X" / "grid" / "cell_counts.parquet")
    (out / "selection_bias_m3.json").unlink()
    with pytest.raises(SystemExit, match="acceptance failed for X"):
        stage.main()
    js = json.loads((out / "selection_bias_m3.json").read_text())
    assert js["acceptance_passed"] is False and js["libraries"]["X"]["acceptance"]["records"]["cells_differing"] == 1

    # stage 24 not passed: the stage refuses to start
    monkeypatch.setattr(sys, "argv", argv[:8] + ["--recount", str(tmp_path / "none")] + argv[10:])
    with pytest.raises(SystemExit, match="has not passed its acceptance checks for X"):
        stage.main()
