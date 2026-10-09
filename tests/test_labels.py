import pandas as pd

from pafbench.labels import a_band, choose_cutoffs, compare_shares, hexamer_curve, label_molecules, templated_share

RATES = [0.5] * 5 + [0.45, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]


def test_choose_cutoffs():
    curve = pd.DataFrame({"a20": range(21), "n": [1000] * 21, "rate": RATES})
    c = choose_cutoffs(curve)
    assert (c["templated_min"], c["nontemplated_max"]) == (10, 6)


def test_low_count_bins_ignored():
    rates = list(RATES)
    rates[12] = 0.9
    n = [1000] * 21
    n[12] = 50
    c = choose_cutoffs(pd.DataFrame({"a20": range(21), "n": n, "rate": rates}))
    assert c["templated_min"] == 10


def test_hexamer_curve_and_shares():
    mol = pd.DataFrame({"a20": [2, 2, 16, 16, 11], "hexamer": [True, False, False, False, False], "re": ["E", "E", "E", "N", "N"]})
    curve = hexamer_curve(mol)
    assert curve.loc[curve.a20 == 2, "rate"].iloc[0] == 0.5
    t = templated_share(mol, 14, 8)
    assert t.loc["E", "share"] == 1 / 3 and t.loc["N", "share"] == 1.0
    cmp = compare_shares(t, t.assign(share=t.share + 0.1))
    assert cmp.flag.all()


def test_a_band_and_label_molecules():
    assert [a_band(x) for x in (0, 4, 5, 7, 8, 9, 10, 13, 14, 20, 21)] == \
        ["0-4", "0-4", "5-7", "5-7", "8-9", "8-9", "10-13", "10-13", "14-20", "14-20", ""]
    mol = pd.DataFrame({"a20": [3, 4, 5, 7, 8, 20]})
    out = label_molecules(mol)
    assert out.label.tolist() == ["nontemplated", "nontemplated", "ambiguous", "ambiguous", "templated", "templated"]
    assert out.a_band.tolist() == ["0-4", "0-4", "5-7", "5-7", "8-9", "14-20"]
    assert label_molecules(mol, {"templated_min": 14, "nontemplated_max": 8}).label.tolist() == \
        ["nontemplated"] * 5 + ["templated"]
    assert "label" not in mol
