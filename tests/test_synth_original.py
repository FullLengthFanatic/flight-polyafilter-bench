from tests.synth import run_original


def test_original_runs_and_removes_something(synth):
    removed = run_original(synth, 300, 8, 1, True)
    assert len(removed) > 0
    # antisense reads in g1 start at 1200, 1500, ... and are never removed
    assert not any(k[1] & 16 and k[2] == "chrA" and 1200 <= k[3] < 3900 and (k[3] - 1200) % 300 == 0 for k in removed)


def test_removed_keys_detects_subsequence(synth):
    removed = run_original(synth, 100, 5, 0, False)
    assert all(len(k) == 5 for k in removed)
