import pytest

from pafbench.flags import Setting, parse_setting, setting_tag
from pafbench.grid import all_settings, grid, published


def test_grid_sizes_and_order():
    g = grid()
    assert len(g) == 392 and len(set(g)) == 392
    assert g[0] == Setting(150, None, 0, False) and g[1] == Setting(150, 3, 0, False)
    assert sum(s.min_len is None for s in g) == 14
    assert {s.mism for s in g if s.min_len is None} == {0}


def test_all_settings_adds_missing_published_once():
    a = all_settings()
    assert len(a) == 396 and len(set(a)) == 396
    assert a[:392] == grid()
    assert set(published()) <= set(a)
    assert [setting_tag(s) for s in a[392:]] == ["c275_l9_m1_e", "c275_l9_m1_i", "c180_l11_m2_e", "c180_l11_m2_i"]


def test_parse_setting_refuses_zero():
    assert parse_setting("300,none,0,i") == Setting(300, None, 0, True)
    assert parse_setting("375,3,2,e") == Setting(375, 3, 2, False)
    with pytest.raises(ValueError):
        parse_setting("300,0,0,i")
