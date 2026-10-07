from pafbench.spikein import module_of, tag_survival


def test_tag_survival():
    before = {"r1": {"CB": "A", "UB": "u", "pa": 25}, "r2": {"CB": "B"}, "r3": {"CB": "C"}}
    after = {"r1": {"CB": "A", "UB": "u", "pa": 25}, "r2": {"CB": "X"}}
    res = tag_survival(before, after)
    assert res["CB"] == {"n": 2, "kept": 1, "fraction": 0.5}
    assert res["pa"]["fraction"] == 1.0 and res["ts"]["n"] == 0


def test_module_of():
    assert module_of("ERCC-00002") == "ERCC" and module_of("SIRV3") == "SIRV_E0" and module_of("SIRV4001") == "long_SIRV"
