"""Stage 28 (pafbench.selection): selection bias, labeled against unlabeled molecules, on synthetic tables and on the
synthetic 10x library of tests/synth.build_10x."""
import numpy as np
import pandas as pd
import pytest

import pyarrow as pa
import pyarrow.parquet as pq

from pafbench.burden import END_BINS, _exon_coord, end_bin, end_distances, gene_ends
from pafbench.config import BIAS_POINTS, DIST_END_BINS, EXPR_BINS
from pafbench.exons import load_collapsed_exons
from pafbench.selection import (AXES, AXIS_COLUMNS, FAMILY, NONPA, PA_ONLY, PAST, POSITIONS, REGIONS, axis_rows, axis_table,
                                decision, downstream_end, end_index, exon_coord, expr_bins_counted, gene_strands,
                                labeled_keys, position_codes, rule, tally)
from pafbench.strata import expr_bins
from tests.test_matched_sites import GTF, _gtf

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
