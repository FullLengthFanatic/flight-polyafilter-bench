"""Stage 1b (Pool278): build a per-sample filtered_feature_bc_matrix/ from the three bead-split replicate MEX zips.

The BD cell-label index is deterministic from the cell-label sequence, so a bead that went into one bead-split replicate
cannot appear in another; the three replicate barcode lists are disjoint (observed on 211: 1 to 4 collisions out of ~10k,
consistent with random index collisions at BD's address space). Features differ across replicates (BD drops
zero-expression genes per run), so union by ENSG. Columns of the output matrix are the three replicate columns in order,
rows are the union of features.

Writes merged/BD_nuc_<sample>/filtered_feature_bc_matrix/ with barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz.
"""
import argparse
import gzip
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

POOL = Path("/data/element_runs/Pool278_FLIGHTseq_NucleiHsRetinaDonation")
BD_DIR = POOL / "bd_pipeline"
MERGED_DIR = POOL / "merged"

SAMPLES = {
    "211": [("1488_BD-211a", "Pool278-1488-BD-211a"),
            ("1489_BD-211b", "Pool278-1489-BD-211b"),
            ("1490_BD-211c", "Pool278-1490-BD-211c")],
    "213": [("1492_BD-213a", "Pool278-1492-BD-213a"),
            ("1493_BD-213b", "Pool278-1493-BD-213b"),
            ("1494_BD-213c", "Pool278-1494-BD-213c")],
    "215": [("1496_BD-215a", "Pool278-1496-BD-215a"),
            ("1497_BD-215b", "Pool278-1497-BD-215b"),
            ("1498_BD-215c", "Pool278-1498-BD-215c")],
}


def read_mex(zip_path):
    """Return (features, barcodes, matrix) from a BD RSEC_MolsPerCell_MEX.zip. features: list of (ensg, symbol, type);
    barcodes: list of str; matrix: scipy sparse (genes, cells)."""
    with zipfile.ZipFile(zip_path) as z:
        with z.open(next(n for n in z.namelist() if n.endswith("features.tsv.gz"))) as inner:
            with gzip.open(io.BufferedReader(inner), "rt") as fh:
                features = [tuple(l.rstrip("\n").split("\t")) for l in fh]
        with z.open(next(n for n in z.namelist() if n.endswith("barcodes.tsv.gz"))) as inner:
            with gzip.open(io.BufferedReader(inner), "rt") as fh:
                barcodes = [l.rstrip("\n") for l in fh]
        with z.open(next(n for n in z.namelist() if n.endswith("matrix.mtx.gz"))) as inner:
            with gzip.open(io.BufferedReader(inner), "rb") as fh:
                mat = sio.mmread(io.BytesIO(fh.read())).tocsc()
    assert mat.shape == (len(features), len(barcodes)), f"{zip_path}: shape {mat.shape} vs {len(features)}x{len(barcodes)}"
    return features, barcodes, mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, choices=sorted(SAMPLES))
    args = ap.parse_args()
    t0 = time.monotonic()
    sample = args.sample
    out_dir = MERGED_DIR / f"BD_nuc_{sample}" / "filtered_feature_bc_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BD_nuc_{sample}] reading three MEX zips", file=sys.stderr, flush=True)
    parts = []
    for lib_dir, prefix in SAMPLES[sample]:
        mex = BD_DIR / lib_dir / f"{prefix}_RSEC_MolsPerCell_MEX.zip"
        features, barcodes, mat = read_mex(mex)
        parts.append((lib_dir, features, barcodes, mat))
        print(f"[BD_nuc_{sample}]   {lib_dir}: {mat.shape[0]} features, {mat.shape[1]} barcodes, {mat.nnz} nnz",
              file=sys.stderr, flush=True)
    global_feats = {}
    for _, features, _, _ in parts:
        for row in features:
            if row[0] not in global_feats:
                global_feats[row[0]] = row
    global_feat_list = list(global_feats.values())
    ensg_to_row = {row[0]: i for i, row in enumerate(global_feat_list)}
    print(f"[BD_nuc_{sample}] union features: {len(global_feat_list)}", file=sys.stderr, flush=True)
    bc_cols = []
    sub_mats = []
    for lib_dir, features, barcodes, mat in parts:
        row_map = np.array([ensg_to_row[row[0]] for row in features], dtype=np.int64)
        n_rows = len(global_feat_list)
        coo = mat.tocoo()
        new_rows = row_map[coo.row]
        full_coo = sp.coo_matrix((coo.data, (new_rows, coo.col)), shape=(n_rows, mat.shape[1]))
        sub_mats.append(full_coo.tocsc())
        bc_cols.extend(barcodes)
    merged = sp.hstack(sub_mats, format="csc", dtype=sub_mats[0].dtype)
    print(f"[BD_nuc_{sample}] merged: {merged.shape[0]} features x {merged.shape[1]} barcodes, {merged.nnz} nnz",
          file=sys.stderr, flush=True)
    assert merged.shape[1] == len(bc_cols)
    # BD index-space collisions across bead-split replicates: expected ~1 to 4 per pair at ~10k x 30M-ish address space.
    # When they occur, keep the first replicate's column and sum colliding columns from later replicates into it; this
    # treats those few reads from the later replicates as coming from the first replicate's cell. Lossy by at most a few
    # cells per sample out of ~30k.
    first_idx = {}
    collapse_src = []
    for i, b in enumerate(bc_cols):
        if b in first_idx:
            collapse_src.append((first_idx[b], i))
        else:
            first_idx[b] = i
    if collapse_src:
        print(f"[BD_nuc_{sample}] cross-replicate CB collisions collapsed: {len(collapse_src)}", file=sys.stderr, flush=True)
        merged = merged.tolil()
        for dst, src in collapse_src:
            merged[:, dst] = merged[:, dst] + merged[:, src]
        merged = merged.tocsc()
        keep = sorted(first_idx.values())
        merged = merged[:, keep]
        bc_cols = [bc_cols[i] for i in keep]
    assert len(set(bc_cols)) == len(bc_cols)
    assert merged.shape[1] == len(bc_cols)
    with gzip.open(out_dir / "barcodes.tsv.gz", "wt") as fh:
        for b in bc_cols:
            fh.write(b + "\n")
    with gzip.open(out_dir / "features.tsv.gz", "wt") as fh:
        for row in global_feat_list:
            fh.write("\t".join(row) + "\n")
    buf = io.BytesIO()
    sio.mmwrite(buf, merged, field="integer", symmetry="general")
    with gzip.open(out_dir / "matrix.mtx.gz", "wb") as fh:
        fh.write(buf.getvalue())
    dt = time.monotonic() - t0
    print(f"[BD_nuc_{sample}] wrote {out_dir} in {dt:.1f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
