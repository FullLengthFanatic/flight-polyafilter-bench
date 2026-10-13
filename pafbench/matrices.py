"""The count matrices of stage 24 on disk (design of 2026-10-12, section 3.2, step 5): the unfiltered matrix M0 of the
counting rule and, per setting, the UMIs lost and the UMIs gained. Cells by genes, scipy CSR, int32; rows and columns are
those of barcodes.tsv and features.tsv next to them. The matrix at a setting is M0 minus the losses plus the gains (load)."""
import gzip
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .config import OUT


def filtered(m0, lost, gained):
    """m0 - lost + gained as CSR int32 without stored zeros."""
    m = (m0 - lost + gained).astype(np.int32).tocsr()
    m.eliminate_zeros()
    m.sort_indices()
    return m


def save(matrix_dir, m0, lost, gained, cr_matrix_dir, meta):
    """unfiltered.npz, lost/<tag>.npz and gained/<tag>.npz, barcodes.tsv and features.tsv (the files of cr_matrix_dir,
    Cell Ranger's filtered_feature_bc_matrix, unpacked) and manifest.json: the tags in the order of lost, which is their bit
    order in the tables of stage 24, the shape, and the entries of meta (what the parts were made from)."""
    matrix_dir = Path(matrix_dir)
    for kind, by_tag in (("lost", lost), ("gained", gained)):
        (matrix_dir / kind).mkdir(parents=True, exist_ok=True)
        for tag, m in by_tag.items():
            sp.save_npz(matrix_dir / kind / f"{tag}.npz", m)
    sp.save_npz(matrix_dir / "unfiltered.npz", m0)
    for name in ("barcodes.tsv", "features.tsv"):
        (matrix_dir / name).write_bytes(gzip.decompress((Path(cr_matrix_dir) / f"{name}.gz").read_bytes()))
    (matrix_dir / "manifest.json").write_text(json.dumps({"tags": list(lost), "shape": list(m0.shape)} | meta, indent=1))


def load(lib, tag, matrix_dir=None):
    """The matrix of lib at tag ("unfiltered" or a setting tag), cells by genes: M0, or M0 - L_tag + G_tag.

    matrix_dir: where stage 24 wrote the matrices (default OUT/<lib>/matrices).
    """
    d = Path(matrix_dir or OUT / lib / "matrices")
    m0 = sp.load_npz(d / "unfiltered.npz")
    if tag == "unfiltered":
        return m0
    return filtered(m0, sp.load_npz(d / "lost" / f"{tag}.npz"), sp.load_npz(d / "gained" / f"{tag}.npz"))
