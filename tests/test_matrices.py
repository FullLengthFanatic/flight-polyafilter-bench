"""The matrix files of stage 24 (pafbench.matrices): unfiltered counts, losses and gains per setting, and the loader."""
import gzip
import json

import numpy as np
import pytest
import scipy.sparse as sp

from pafbench import matrices

TAGS = ["c200_l8_m0_e", "c200_l8_m0_i"]
M0 = np.array([[3, 0, 1, 0], [0, 2, 0, 0], [1, 1, 0, 5]])
LOST = {TAGS[0]: np.array([[3, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),
        TAGS[1]: np.array([[1, 0, 1, 0], [0, 2, 0, 0], [0, 1, 0, 2]])}
GAINED = {TAGS[0]: np.array([[0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
          TAGS[1]: np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]])}
META = {"bam": "/x/possorted_genome_bam.bam", "rule_version": 1, "barcodes_sha256": "ab", "features_sha256": "cd"}


def _csr(a):
    return sp.csr_matrix(a, dtype=np.int32)


@pytest.fixture
def saved(tmp_path):
    cr = tmp_path / "filtered_feature_bc_matrix"
    cr.mkdir()
    with gzip.open(cr / "barcodes.tsv.gz", "wt") as fh:
        fh.write("AAAC-1\nAAAG-1\nAAAT-1\n")
    with gzip.open(cr / "features.tsv.gz", "wt") as fh:
        fh.write("".join(f"g{i}\tG{i}\tGene Expression\n" for i in range(1, 5)))
    d = tmp_path / "X" / "matrices"
    matrices.save(d, _csr(M0), {t: _csr(a) for t, a in LOST.items()}, {t: _csr(a) for t, a in GAINED.items()}, cr, META)
    return d


def test_save_writes_the_files_of_the_design(saved):
    assert sorted(str(p.relative_to(saved)) for p in saved.rglob("*") if p.is_file()) == sorted(
        ["unfiltered.npz", "barcodes.tsv", "features.tsv", "manifest.json"]
        + [f"{kind}/{t}.npz" for kind in ("lost", "gained") for t in TAGS])
    assert (saved / "barcodes.tsv").read_text() == "AAAC-1\nAAAG-1\nAAAT-1\n"
    assert (saved / "features.tsv").read_text().splitlines()[1] == "g2\tG2\tGene Expression"
    # tags in bit order, the shape, then what the parts were made from
    assert json.loads((saved / "manifest.json").read_text()) == {"tags": TAGS, "shape": [3, 4]} | META
    for name, want in (("unfiltered", M0), (f"lost/{TAGS[1]}", LOST[TAGS[1]]), (f"gained/{TAGS[0]}", GAINED[TAGS[0]])):
        m = sp.load_npz(saved / f"{name}.npz")
        assert m.format == "csr" and m.dtype == np.int32 and np.array_equal(m.toarray(), want)


def test_load_returns_unfiltered_minus_losses_plus_gains(saved, monkeypatch):
    m = matrices.load("X", "unfiltered", saved)
    assert m.format == "csr" and m.dtype == np.int32 and np.array_equal(m.toarray(), M0)
    for t in TAGS:
        m = matrices.load("X", t, saved)
        want = M0 - LOST[t] + GAINED[t]
        assert m.format == "csr" and m.dtype == np.int32 and np.array_equal(m.toarray(), want)
        # an entry that lost every UMI is no stored zero: nnz is the number of entries with a UMI
        assert m.nnz == (want > 0).sum() and m.has_sorted_indices
    assert (M0 - LOST[TAGS[0]] + GAINED[TAGS[0]])[0, 0] == 0
    # without a directory: OUT/<lib>/matrices
    monkeypatch.setattr(matrices, "OUT", saved.parents[1])
    assert np.array_equal(matrices.load("X", TAGS[1]).toarray(), M0 - LOST[TAGS[1]] + GAINED[TAGS[1]])
    with pytest.raises(FileNotFoundError):
        matrices.load("X", "c300_l10_m1_e", saved)


def test_filtered_is_the_one_combination_of_the_three_matrices():
    m = matrices.filtered(_csr(M0), _csr(LOST[TAGS[1]]), _csr(GAINED[TAGS[1]]))
    assert np.array_equal(m.toarray(), M0 - LOST[TAGS[1]] + GAINED[TAGS[1]]) and (m.data != 0).all()
    # a loss without a count shows as a negative entry, it is not clipped (check A5 of stage 24 looks for it)
    bad = matrices.filtered(_csr(M0), _csr(np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])), _csr(np.zeros((3, 4))))
    assert bad.toarray()[0, 1] == -1
