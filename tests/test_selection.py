"""Stage 28 (pafbench.selection): selection bias, labeled against unlabeled molecules, on synthetic tables and on the
synthetic 10x library of tests/synth.build_10x."""
import numpy as np
import pandas as pd
import pytest

from pafbench.config import BIAS_POINTS
from pafbench.selection import AXES, AXIS_COLUMNS, FAMILY, REGIONS, axis_rows, decision, rule


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
