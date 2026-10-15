"""Stage 33 (pafbench.rerun): the comparison of a Cell Ranger rerun with our matrix, on toy matrices and molecules with
planted differences, and the two commands of the rerun built from a copy of the original run's _cmdline and _invocation."""
import gzip
import importlib.util
import json
import shlex
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
import pytest
import scipy.sparse as sp

from pafbench import matrices, rerun
from pafbench.config import RERUN_DIFF, REPO

BARCODES = ["AAAC-1", "AAAG-1", "AAAT-1", "AACA-1"]                 # the called cells of the original run, our rows
FEATURES = [("g1", "G1"), ("g2", "G2"), ("g3", "G3"), ("g4", "G4")]
# the rerun lists its barcodes and features in another order, holds a barcode that is no original cell (TTTT-1), lacks one
# that is (AACA-1), and has a gene that is not in our feature list (g5) and lacks one that is (g4)
THEIR_BARCODES = ["TTTT-1", "AAAT-1", "AAAC-1", "AAAG-1"]
THEIR_FEATURES = [("g3", "G3"), ("g1", "G1"), ("g2", "G2"), ("g5", "G5")]
OURS = np.array([[3, 2, 0, 1],     # AAAC: g1 equal, g2 only ours, g4 in our feature list only
                 [0, 0, 0, 0],     # AAAG: g3 only theirs
                 [5, 0, 0, 0],     # AAAT: g1 different counts, g5 in their feature list only
                 [0, 1, 0, 0]])    # AACA: not in the rerun's matrix
THEIRS = np.array([[0, 7, 0, 0],   # TTTT: not an original cell, left out
                   [0, 3, 0, 1],
                   [0, 3, 0, 0],
                   [4, 0, 0, 0]])


# the columns (g1 to g4, then g5) of the molecules of one side only: those that make the differing entries, and in the
# equal entry AAAC g1 one molecule each side under another UMI
ONE_SIDED = (np.array([1, 1, 3, 0, 0, 1, 0]),     # AAAC g2 twice, AAAC g4, AAAT g1 twice, AACA g2; AAAC g1
             np.array([2, 2, 2, 2, 4, 0]))        # AAAG g3 four times, AAAT g5; AAAC g1


def _csr(a):
    return sp.csr_matrix(a, dtype=np.int32)


def test_matrices_are_joined_by_barcode_and_gene_id_and_every_kind_of_difference_is_counted():
    got = rerun.compare_matrices(_csr(OURS), BARCODES, FEATURES, _csr(THEIRS), THEIR_BARCODES, THEIR_FEATURES, ONE_SIDED)
    # entries with a count on either side in the original cells: AAAC g1, g2, g4; AAAG g3; AAAT g1, g5; AACA g2
    assert (got["entries_compared"], got["entries_equal"], got["entries_differing"]) == (7, 1, 6)
    assert got["per_million"] == pytest.approx(1e6 * 6 / 7)
    # UMIs of the original cells; the 7 of TTTT-1 are not among them. The excess of a side is summed over the entries
    # where it is higher, so the swap in AAAC g1 is not in it
    assert (got["umis_ours"], got["umis_rerun"], got["umis_excess_ours"], got["umis_excess_rerun"]) == (12, 11, 6, 5)
    assert got["umis_differing_share"] == pytest.approx(11 / 12) and got["rerun_diff"] == RERUN_DIFF
    assert got["above_rerun_diff"] is True
    assert got["cells_absent_from_rerun_matrix"] == ["AACA-1"]
    assert got["genes_only_ours"] == {"genes": 1, "umis": 1, "ids": ["g4"]}
    assert got["genes_only_rerun"] == {"genes": 1, "umis": 1, "ids": ["g5"]}
    assert got["differing"] == [
        {"cell": "AAAC-1", "gene": "g2", "ours": 2, "theirs": 0}, {"cell": "AAAC-1", "gene": "g4", "ours": 1, "theirs": 0},
        {"cell": "AAAG-1", "gene": "g3", "ours": 0, "theirs": 4}, {"cell": "AAAT-1", "gene": "g1", "ours": 5, "theirs": 3},
        {"cell": "AAAT-1", "gene": "g5", "ours": 0, "theirs": 1}, {"cell": "AACA-1", "gene": "g2", "ours": 1, "theirs": 0}]
    # by gene, the genes with a differing entry or a molecule of one side, most such molecules first, then the excess; the
    # molecule columns add up to the molecules of one side (7 and 6), the swap in g1 included
    assert got["by_gene"] == [
        {"gene_id": "g3", "gene_name": "G3", "umis_ours": 0, "umis_rerun": 4, "umis_excess_ours": 0, "umis_excess_rerun": 4,
         "entries_differing": 1, "molecules_only_ours": 0, "molecules_only_rerun": 4},
        {"gene_id": "g1", "gene_name": "G1", "umis_ours": 8, "umis_rerun": 6, "umis_excess_ours": 2, "umis_excess_rerun": 0,
         "entries_differing": 1, "molecules_only_ours": 3, "molecules_only_rerun": 1},
        {"gene_id": "g2", "gene_name": "G2", "umis_ours": 3, "umis_rerun": 0, "umis_excess_ours": 3, "umis_excess_rerun": 0,
         "entries_differing": 2, "molecules_only_ours": 3, "molecules_only_rerun": 0},
        {"gene_id": "g4", "gene_name": "G4", "umis_ours": 1, "umis_rerun": 0, "umis_excess_ours": 1, "umis_excess_rerun": 0,
         "entries_differing": 1, "molecules_only_ours": 1, "molecules_only_rerun": 0},
        {"gene_id": "g5", "gene_name": "G5", "umis_ours": 0, "umis_rerun": 1, "umis_excess_ours": 0, "umis_excess_rerun": 1,
         "entries_differing": 1, "molecules_only_ours": 0, "molecules_only_rerun": 1}]


def test_equal_matrices_in_another_order_differ_nowhere():
    order = [2, 0, 3, 1]
    theirs = np.zeros((5, 4), np.int64)
    theirs[:4] = OURS[order]
    theirs[4] = [9, 9, 9, 9]                         # an uncalled barcode of the rerun
    args = (_csr(OURS), BARCODES, FEATURES, _csr(theirs), [BARCODES[i] for i in order] + ["TTTT-1"], FEATURES)
    got = rerun.compare_matrices(*args, ([], []))
    assert (got["entries_compared"], got["entries_equal"], got["entries_differing"]) == (5, 5, 0)
    assert (got["umis_ours"], got["umis_rerun"], got["umis_excess_ours"], got["umis_excess_rerun"]) == (12, 12, 0, 0)
    assert got["above_rerun_diff"] is False and got["by_gene"] == got["differing"] == []
    assert got["cells_absent_from_rerun_matrix"] == []
    assert got["genes_only_ours"] == got["genes_only_rerun"] == {"genes": 0, "umis": 0, "ids": []}
    # two molecules swapped inside one entry of g1 (same counts, another UMI): equal matrices, one molecule each side
    swap = rerun.compare_matrices(*args, ([0], [0]))
    assert swap.pop("by_gene") == [{"gene_id": "g1", "gene_name": "G1", "umis_ours": 8, "umis_rerun": 8, "umis_excess_ours": 0,
                                    "umis_excess_rerun": 0, "entries_differing": 0, "molecules_only_ours": 1,
                                    "molecules_only_rerun": 1}]
    assert swap == {k: v for k, v in got.items() if k != "by_gene"}


def test_cells_the_rerun_calls_among_the_original_called_cells():
    assert rerun.called_cells(BARCODES, ["AAAC-1", "AAAT-1", "TTTT-1"]) == {
        "original": 4, "rerun": 3, "both": 2, "original_only": 2, "rerun_only": 1,
        "original_only_barcodes": ["AAAG-1", "AACA-1"]}


TAGS = ["tA", "tB"]
COLS = ["cell", "gene", "umi", "n_reads", "any_removed"]
# our molecules: bit 0 is tA, bit 1 tB
COUNTED = pd.DataFrame([(0, 0, 1, 3, 0, 0),         # m1: nothing removed
                        (0, 0, 2, 2, 2, 0),         # m2: one of two reads removed at tB
                        (0, 1, 3, 4, 2, 0),         # m3: a key on two references, 2 and 1 of its 4 reads removed at tB
                        (2, 0, 4, 2, 2, 2),         # m4: lost at tB
                        (2, 0, 5, 1, 1, 1)],        # m5: lost at tA only
                       columns=COLS + ["lost"])
OTHER = pd.DataFrame([(2, 0, 6, 2, 2, 2),           # o1: gained at tB, one read removed there
                      (1, 2, 7, 1, 0, 1)],          # o2: gained at tA only
                     columns=COLS + ["gained"])


def _key(cell, umi, gene):
    return (cell << 40) | (umi << 16) | gene


def _recount_dir(d):
    d.mkdir(parents=True)
    types = {"cell": "int32", "gene": "int32", "umi": "uint32", "n_reads": "uint16", "any_removed": "uint32"}
    COUNTED.astype(types | {"lost": "uint32"}).to_parquet(d / "counted_molecules.parquet", index=False)
    OTHER.astype(types | {"gained": "uint32"}).to_parquet(d / "other_read_groups.parquet", index=False)
    (d / "parts").mkdir()
    rows = {"chrA": [(0, 1, 0, 0, 0), (0, 2, 0, 0, 1), (0, 3, 1, 0, 2), (2, 4, 0, 0, 2), (2, 5, 0, 1, 0), (2, 6, 0, 0, 1)],
            "chrB": [(0, 3, 1, 0, 1), (1, 7, 2, 0, 0)]}
    for ref, r in rows.items():
        pq.write_table(pa.table({"key": pa.array([_key(c, u, g) for c, u, g, _, _ in r], pa.uint64()),
                                 "n_reads": pa.array([9] * len(r), pa.uint16()),
                                 "rm_tA": pa.array([x for *_, x, _ in r], pa.uint16()),
                                 "rm_tB": pa.array([x for *_, x in r], pa.uint16())}),
                       d / "parts" / f"{ref}.read_groups.parquet")
    return d


def test_our_molecules_at_a_tag_with_the_reads_that_survive(tmp_path):
    d = _recount_dir(tmp_path / "recount")
    key, reads = rerun.our_molecules(d, "tB", TAGS)
    # m1, m2, m3, m5 held, m4 lost, o1 gained; reads after removal, m3 summed over its two references
    assert key.tolist() == sorted([_key(0, 1, 0), _key(0, 2, 0), _key(0, 3, 1), _key(2, 5, 0), _key(2, 6, 0)])
    assert dict(zip(key.tolist(), reads.tolist())) == {_key(0, 1, 0): 3, _key(0, 2, 0): 1, _key(0, 3, 1): 1,
                                                       _key(2, 5, 0): 1, _key(2, 6, 0): 1}
    key, reads = rerun.our_molecules(d, "tA", TAGS)
    assert dict(zip(key.tolist(), reads.tolist())) == {_key(0, 1, 0): 3, _key(0, 2, 0): 2, _key(0, 3, 1): 4,
                                                       _key(2, 4, 0): 2, _key(1, 7, 2): 1}
    key, reads = rerun.our_molecules(d, "unfiltered", TAGS)
    assert dict(zip(key.tolist(), reads.tolist())) == {_key(0, 1, 0): 3, _key(0, 2, 0): 2, _key(0, 3, 1): 4,
                                                       _key(2, 4, 0): 2, _key(2, 5, 0): 1}


# the rerun's molecules: barcode (without the "-1", the gem_group), feature, UMI, count
MOLECULES = [("AAAC", "g1", 1, 2),     # m1, with fewer reads
             ("AAAC", "g1", 12, 1),    # m2 under another UMI: the entry AAAC g1 is equal, the molecule on both sides once
             ("AAAT", "g1", 6, 1),     # o1
             ("AAAG", "g3", 9, 4),     # only the rerun
             ("AAAT", "g5", 8, 1),     # only the rerun, a gene not in our feature list
             ("TTTT", "g1", 1, 7)]     # not an original cell
H5_BARCODES = ["TTTT", "AAAT", "AAAC", "AAAG"]
H5_FEATURES = [g for g, _ in THEIR_FEATURES]


def _molecule_info(path, molecules=MOLECULES, gem=1):
    """molecule_info.h5 with the datasets the comparison reads, as Cell Ranger 9.0.1 writes them for one gem group (our
    observation on the outs of 1417, the layout tests/synth._outs_10x writes)."""
    with h5py.File(path, "w") as f:
        f["barcodes"] = np.array(H5_BARCODES, "S")
        f["features/id"] = np.array(H5_FEATURES, "S")
        f["barcode_idx"] = np.array([H5_BARCODES.index(b) for b, *_ in molecules], np.uint64)
        f["feature_idx"] = np.array([H5_FEATURES.index(g) for _, g, *_ in molecules], np.uint32)
        f["umi"] = np.array([u for *_, u, _ in molecules], np.uint32)
        f["count"] = np.array([n for *_, n in molecules], np.uint32)
        f["gem_group"] = np.array([gem] * len(molecules), np.uint16)
    return path


def test_rerun_molecules_are_joined_by_barcode_and_gene_id(tmp_path):
    cell_row = {b: i for i, b in enumerate(BARCODES)}
    gene_col = {g: j for j, g in enumerate(["g1", "g2", "g3", "g4", "g5"])}
    key, count = rerun.rerun_molecules(_molecule_info(tmp_path / "molecule_info.h5"), cell_row, gene_col)
    assert dict(zip(key.tolist(), count.tolist())) == {_key(0, 1, 0): 2, _key(0, 12, 0): 1, _key(2, 6, 0): 1,
                                                       _key(1, 9, 2): 4, _key(2, 8, 4): 1}
    assert key.tolist() == sorted(key.tolist())
    two = _molecule_info(tmp_path / "two.h5")
    with h5py.File(two, "a") as f:
        f["gem_group"][0] = 2
    with pytest.raises(ValueError, match="2 gem groups"):
        rerun.rerun_molecules(two, cell_row, gene_col)


def test_molecules_by_family_size(tmp_path):
    cell_row = {b: i for i, b in enumerate(BARCODES)}
    gene_col = {g: j for j, g in enumerate(["g1", "g2", "g3", "g4", "g5"])}
    got = rerun.by_family(rerun.our_molecules(_recount_dir(tmp_path / "recount"), "tB", TAGS),
                          rerun.rerun_molecules(_molecule_info(tmp_path / "molecule_info.h5"), cell_row, gene_col))
    # both: m1 (3 reads here, 2 in the rerun), o1; only ours: m2, m3 and m5; only the rerun: m2 under another UMI, AAAG g3
    # and AAAT g5
    assert got == {"molecules_ours": 5, "molecules_rerun": 5, "both": 2, "only_ours": 3, "only_rerun": 3,
                   "both_same_reads": 1,
                   "by_family": [{"reads": 1, "only_ours": 3, "only_rerun": 2, "both": 1},
                                 {"reads": 3, "only_ours": 0, "only_rerun": 0, "both": 1},
                                 {"reads": 4, "only_ours": 0, "only_rerun": 1, "both": 0}]}


def _mex(d, barcodes, features, m):
    """A Cell Ranger matrix folder: matrix.mtx.gz genes by cells, barcodes.tsv.gz and features.tsv.gz."""
    d.mkdir(parents=True)
    with gzip.open(d / "barcodes.tsv.gz", "wt") as fh:
        fh.write("".join(f"{b}\n" for b in barcodes))
    with gzip.open(d / "features.tsv.gz", "wt") as fh:
        fh.write("".join(f"{g}\t{n}\tGene Expression\n" for g, n in features))
    c = sp.coo_matrix(m)
    with gzip.open(d / "matrix.mtx.gz", "wt") as fh:
        fh.write("%%MatrixMarket matrix coordinate integer general\n"
                 '%metadata_json: {"software_version": "Cell Ranger cellranger-9.0.1", "format_version": 2}\n'
                 f"{m.shape[1]} {m.shape[0]} {c.nnz}\n" + "".join(f"{j + 1} {i + 1} {n}\n" for i, j, n in
                                                                   sorted(zip(c.row, c.col, c.data))))
    return d


@pytest.fixture
def library(tmp_path):
    """Our side (recount and matrices folders of stage 24 for the molecules above) and a rerun's outs folder made from
    MOLECULES: the raw matrix holds every barcode with a UMI, the filtered one the cells the rerun calls."""
    rec = _recount_dir(tmp_path / "recount")
    cr = _mex(tmp_path / "cr", BARCODES, FEATURES, np.zeros((4, 4), np.int64))
    m0 = _csr([[2, 1, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]])                    # m1 to m5
    lost = {"tA": _csr([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]),             # m5
            "tB": _csr([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])}             # m4
    gained = {"tA": _csr([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),           # o2
              "tB": _csr([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])}           # o1
    mat = tmp_path / "matrices"
    matrices.save(mat, m0, lost, gained, cr, {})
    outs = tmp_path / "outs"
    raw = np.zeros((4, 4), np.int64)
    for b, g, _, _ in MOLECULES:
        raw[H5_BARCODES.index(b), H5_FEATURES.index(g)] += 1
    _mex(outs / "raw_feature_bc_matrix", [f"{b}-1" for b in H5_BARCODES], THEIR_FEATURES, raw)
    _mex(outs / "filtered_feature_bc_matrix", ["AAAC-1", "AAAT-1", "TTTT-1"], THEIR_FEATURES, raw[[2, 1, 0]])
    _molecule_info(outs / "molecule_info.h5")
    return rec, mat, outs


def test_compare_reads_the_outs_by_string_and_writes_every_part_of_the_design(library):
    rec, mat, outs = library
    got = rerun.compare("X", "tB", TAGS, outs, rec, mat)
    assert got.pop("seconds") >= 0
    assert list(got) == ["library", "tag", "outs", "matrix", "entries_compared", "entries_equal", "entries_differing",
                         "per_million", "umis_ours", "umis_rerun", "umis_excess_ours", "umis_excess_rerun",
                         "umis_differing_share", "rerun_diff", "above_rerun_diff", "cells_absent_from_rerun_matrix",
                         "genes_only_ours", "genes_only_rerun", "by_gene", "differing", "cells_called", "molecules"]
    assert (got["library"], got["tag"], got["outs"], got["matrix"]) == ("X", "tB", str(outs), "raw_feature_bc_matrix")
    # ours at tB: AAAC g1 2, g2 1; AAAT g1 2. The rerun: AAAC g1 2; AAAG g3 1; AAAT g1 1, g5 1
    assert (got["entries_compared"], got["entries_equal"], got["entries_differing"]) == (5, 1, 4)
    assert (got["umis_ours"], got["umis_rerun"], got["umis_excess_ours"], got["umis_excess_rerun"]) == (5, 5, 2, 2)
    assert got["cells_called"] == {"original": 4, "rerun": 3, "both": 2, "original_only": 2, "rerun_only": 1,
                                   "original_only_barcodes": ["AAAG-1", "AACA-1"]}
    # the molecules account for the UMIs of both matrices; m2 under another UMI adds one molecule to each side that the
    # excess does not hold
    mol = got["molecules"]
    assert (mol["molecules_ours"], mol["molecules_rerun"], mol["only_ours"], mol["only_rerun"]) == (5, 5, 3, 3)
    # by gene and by family size the molecules of one side add up to the same totals
    assert [(g["gene_id"], g["umis_excess_ours"], g["umis_excess_rerun"], g["molecules_only_ours"], g["molecules_only_rerun"])
            for g in got["by_gene"]] == [("g1", 1, 0, 2, 1), ("g2", 1, 0, 1, 0), ("g3", 0, 1, 0, 1), ("g5", 0, 1, 0, 1)]
    for side in ("only_ours", "only_rerun"):
        assert sum(g[f"molecules_{side}"] for g in got["by_gene"]) == sum(f[side] for f in mol["by_family"]) == mol[side]
    assert json.loads(json.dumps(got)) == got
    # the filtered matrix of the outs in place of the raw one: only the cells the rerun calls, AAAG g3 is gone
    got = rerun.compare("X", "tB", TAGS, outs, rec, mat, "filtered_feature_bc_matrix")
    assert got["matrix"] == "filtered_feature_bc_matrix" and got["cells_absent_from_rerun_matrix"] == ["AAAG-1", "AACA-1"]
    assert (got["entries_compared"], got["umis_rerun"], got["umis_excess_rerun"]) == (4, 4, 1)


# _cmdline and _invocation of the Cell Ranger run of 1417 (/data/element_runs/1417_HsRetina_D28_3k), copied
CMDLINE = ("cellranger count --id=1417_HsRetina_D28_3k --transcriptome=/data/references/refdata-gex-GRCh38-2024-A "
           "--fastqs=/data/cellranger_outputs/HsRetina/fastq_staging/1417_HsRetina_D28_3k --sample=1417_HsRetina_D28_3k "
           "--include-introns=true --create-bam=true --localcores=16 --localmem=64\n")
INVOCATION = """@include "rna/sc_rna_counter_cs.mro"

call SC_RNA_COUNTER_CS(
    sample_id                   = "1417_HsRetina_D28_3k",
    sample_def                  = [
        {
            "fastq_id": null,
            "fastq_mode": "ILMN_BCL2FASTQ",
            "gem_group": null,
            "lanes": null,
            "library_type": "Gene Expression",
            "read_path": "/data/cellranger_outputs/HsRetina/fastq_staging/1417_HsRetina_D28_3k",
            "sample_indices": ["any"],
            "sample_names": ["1417_HsRetina_D28_3k"],
            "subsample_rate": null,
        },
    ],
    target_set                  = null,
    target_set_name             = null,
    sample_desc                 = "",
    reference_path              = "/data/references/refdata-gex-GRCh38-2024-A",
    recovered_cells             = null,
    no_bam                      = false,
    filter_probes               = null,
    no_secondary_analysis       = false,
    no_target_umi_filter        = false,
    force_cells                 = null,
    include_introns             = true,
    check_library_compatibility = true,
    chemistry                   = "auto",
    r1_length                   = null,
    r2_length                   = null,
    targeting_method            = null,
    aligner                     = null,
    trim_polya_min_score        = 20,
    trim_tso_min_score          = 20,
    feature_reference           = null,
    emptydrops_minimum_umis     = null,
    global_minimum_umis         = null,
    max_mito_percent            = null,
    disable_ab_aggregate_detection = false,
    min_crispr_umi_threshold    = 3,
    tenx_cloud_token_path       = null,
    cell_annotation_model       = null,
    skip_cell_annotation        = true,
)
"""
# the @RG line of 1417/original/c375_l3_m2_i.bam (samtools view -H)
RG = {"ID": "1417_HsRetina_D28_3k:0:1:unknow_flowcell:0", "SM": "1417_HsRetina_D28_3k", "LB": "0.1",
      "PU": "1417_HsRetina_D28_3k:0:1:unknow_flowcell:0", "PL": "ILLUMINA"}
FOLDER = "1417_HsRetina_D28_3k_0_1_unknow_flowcell"
HASHES = {"fasta_hash": "b6f131840f9f337e7b858c3d1e89d7ce0321b243", "gtf_hash.gz": "432db3ab308171ef215fac5dc4ca40096099a4c6"}


def _run_dir(d, cmdline=CMDLINE, invocation=INVOCATION):
    (d / "outs").mkdir(parents=True)
    (d / "_cmdline").write_text(cmdline)
    (d / "_invocation").write_text(invocation)
    return d


def _bam(path, rgs=(RG,)):
    hdr = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.4", "SO": "coordinate"}, "SQ": [{"SN": "chrA", "LN": 1000}],
                                           "RG": list(rgs)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with pysam.AlignmentFile(str(path), "wb", header=hdr):
        pass
    return path


def _reference(d, hashes=HASHES):
    d.mkdir(parents=True)
    (d / "reference.json").write_text(json.dumps(hashes | {"genomes": ["GRCh38"], "version": "2024-A"}))
    return d


def _metrics(h5, hashes=HASHES):
    """molecule_info.h5 with the reference entries of its metrics_json as the original run of 1417 has them (h5py)."""
    with h5py.File(h5, "w") as f:
        f["metrics_json"] = json.dumps({"reference_fasta_hash": hashes["fasta_hash"], "reference_gtf_hash": "",
                                        "reference_gtf_hash.gz": hashes["gtf_hash.gz"], "reference_mkref_version": "8.0.0"})
    return h5


def test_original_run_gives_the_flags_of_its_cmdline_checked_against_its_invocation(tmp_path):
    assert rerun.original_run(_run_dir(tmp_path / "run")) == {
        "id": "1417_HsRetina_D28_3k", "transcriptome": "/data/references/refdata-gex-GRCh38-2024-A",
        "fastqs": "/data/cellranger_outputs/HsRetina/fastq_staging/1417_HsRetina_D28_3k", "sample": "1417_HsRetina_D28_3k",
        "include-introns": "true", "create-bam": "true", "localcores": "16", "localmem": "64"}
    for old, new in (("include_introns             = true", "include_introns             = false"),
                     ('reference_path              = "/data/references/refdata-gex-GRCh38-2024-A"',
                      'reference_path              = "/data/references/other"'),
                     ("no_bam                      = false", "no_bam                      = true")):
        with pytest.raises(ValueError, match="_invocation"):
            rerun.original_run(_run_dir(tmp_path / old[:6], invocation=INVOCATION.replace(old, new)))
    with pytest.raises(ValueError, match="cellranger count"):
        rerun.original_run(_run_dir(tmp_path / "multi", cmdline=CMDLINE.replace("cellranger count", "cellranger multi")))


def test_fastq_folder_is_named_from_the_read_group_as_bamtofastq_names_it(tmp_path):
    assert rerun.fastq_folder(_bam(tmp_path / "a.bam")) == FOLDER
    with pytest.raises(ValueError, match="2 read groups"):
        rerun.fastq_folder(_bam(tmp_path / "b.bam", (RG, RG | {"ID": "x:0:1:fc:2", "PU": "x"})))


def test_reference_must_be_the_one_of_the_original_run(tmp_path):
    h5 = _metrics(tmp_path / "molecule_info.h5")
    rerun.check_reference(_reference(tmp_path / "ref"), h5)
    with pytest.raises(ValueError, match="gtf_hash.gz"):
        rerun.check_reference(_reference(tmp_path / "other", HASHES | {"gtf_hash.gz": "0" * 40}), h5)


def test_rerun_folder_is_refused_in_the_repository_and_in_a_cell_ranger_run_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(rerun, "LIBRARIES", {"X": tmp_path / "X_run" / "outs" / "possorted_genome_bam.bam"})
    rerun.check_rerun_dir(tmp_path / "out" / "X" / "rerun" / "t")
    for bad in (REPO / "rerun" / "t", tmp_path / "X_run" / "outs" / "t", tmp_path / "X_run" / "rerun",
                tmp_path / "out" / ".." / "X_run" / "t"):
        with pytest.raises(ValueError, match="must not be under"):
            rerun.check_rerun_dir(bad)


def test_commands_carry_the_values_of_the_original_run(tmp_path, monkeypatch):
    monkeypatch.setattr(rerun, "LIBRARIES", {"X": tmp_path / "X_run" / "outs" / "possorted_genome_bam.bam"})
    run, bam, ref = _run_dir(tmp_path / "X_run"), _bam(tmp_path / "out" / "original" / "c375_l3_m2_i.bam"), tmp_path / "ref"
    folder = tmp_path / "out" / "rerun" / "c375_l3_m2_i"
    b2f, count = rerun.rerun_commands(run, bam, ref, folder)
    assert b2f == [str(rerun.BAMTOFASTQ), "--nthreads=16", str(bam), str(folder / "fastq")]
    assert count == [str(rerun.CELLRANGER), "count", "--id=c375_l3_m2_i", f"--transcriptome={ref}",
                     f"--fastqs={folder / 'fastq' / FOLDER}", "--sample=bamtofastq", "--include-introns=true",
                     "--create-bam=true", "--localcores=16", "--localmem=64", f"--output-dir={folder / 'count'}"]
    assert str(rerun.BAMTOFASTQ) == "/home/simone.picelli/cellranger-9.0.1/lib/bin/bamtofastq"
    assert str(rerun.CELLRANGER) == "/home/simone.picelli/cellranger-9.0.1/cellranger"
    with pytest.raises(ValueError, match="must not be under"):
        rerun.rerun_commands(run, bam, ref, run / "rerun")


def test_commands_hold_absolute_paths_when_given_relative_ones(tmp_path, monkeypatch):
    # run_rerun runs both commands in the rerun folder, so a relative path would resolve below it a second time
    monkeypatch.setattr(rerun, "LIBRARIES", {"X": tmp_path / "X_run" / "outs" / "possorted_genome_bam.bam"})
    run, _ = _run_dir(tmp_path / "X_run"), _bam(tmp_path / "out" / "original" / "c375_l3_m2_i.bam")
    monkeypatch.chdir(tmp_path)
    b2f, count = rerun.rerun_commands(run, Path("out/original/c375_l3_m2_i.bam"), Path("ref"), Path("trial/c375_l3_m2_i"))
    folder = tmp_path / "trial" / "c375_l3_m2_i"
    assert b2f[2:] == [str(tmp_path / "out" / "original" / "c375_l3_m2_i.bam"), str(folder / "fastq")]
    flags = dict(w[2:].split("=", 1) for w in count[2:])
    assert (flags["transcriptome"], flags["fastqs"], flags["output-dir"]) == (
        str(tmp_path / "ref"), str(folder / "fastq" / FOLDER), str(folder / "count"))


def test_run_rerun_runs_both_commands_in_the_rerun_folder_and_skips_one_that_finished(tmp_path, monkeypatch):
    calls = []

    def fake(cmd, log, cwd):
        calls.append((cmd, log, cwd))
        log.write_text("ok\n")

    monkeypatch.setattr(rerun, "run_command", fake)
    folder = tmp_path / "rerun" / "t"
    cmds = (["b2f", "x"], ["cr", "count"])
    got = rerun.run_rerun(cmds, folder)
    assert calls == [(["b2f", "x"], folder / "bamtofastq.log", folder), (["cr", "count"], folder / "cellranger.log", folder)]
    assert set(got) == {"bamtofastq", "cellranger"} and all(s >= 0 for s in got.values())
    for name, cmd in zip(("bamtofastq", "cellranger"), cmds):
        done = json.loads((folder / f"{name}.json").read_text())
        assert done["cmd"] == cmd and done["seconds"] >= 0
    # a second run after a failed count: bamtofastq, whose output folder must not exist, is not run again
    (folder / "cellranger.json").unlink()
    calls.clear()
    assert set(rerun.run_rerun(cmds, folder)) == {"cellranger"} and [c[0] for c in calls] == [["cr", "count"]]
    calls.clear()
    assert rerun.run_rerun(cmds, folder) == {} and calls == []


def test_run_command_sends_the_output_to_the_log_and_stops_on_failure(tmp_path):
    log = tmp_path / "x.log"
    rerun.run_command(["sh", "-c", "pwd; echo err >&2"], log, tmp_path)
    assert log.read_text() == f"{tmp_path}\nerr\n"
    with pytest.raises(Exception, match="exit status 3"):
        rerun.run_command(["sh", "-c", "exit 3"], log, tmp_path)


def _stage33(monkeypatch, tmp_path):
    """The stage script with LIBRARIES, OUT and REF_DIR pointed at a copy of the run folder of 1417, an original-tool BAM
    whose run ended with exit status 0, and a reference with the original run's hashes."""
    spec = importlib.util.spec_from_file_location("stage33", REPO / "scripts" / "33_rerun.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    run, out = _run_dir(tmp_path / "1417_run"), tmp_path / "out"
    _metrics(run / "outs" / "molecule_info.h5")
    bam = _bam(out / "1417" / "original" / "c375_l3_m2_i.bam")
    bam.with_suffix(".time.txt").write_text("\tExit status: 0\n")
    libraries = {"1417": run / "outs" / "possorted_genome_bam.bam"}
    for mod in (stage, rerun):
        monkeypatch.setattr(mod, "LIBRARIES", libraries)
    monkeypatch.setattr(stage, "OUT", out)
    monkeypatch.setattr(stage, "REF_DIR", _reference(tmp_path / "ref"))
    return stage, run, out, bam


def test_stage33_prints_the_commands_and_runs_them_only_when_asked(tmp_path, monkeypatch, capsys):
    stage, run, out, bam = _stage33(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(rerun, "run_command", lambda cmd, log, cwd: calls.append((cmd, cwd)))
    folder = out / "1417" / "rerun" / "c375_l3_m2_i"
    monkeypatch.setattr(sys, "argv", ["33", "run"])
    stage.main()
    lines = capsys.readouterr().out.strip().splitlines()
    b2f, count = rerun.rerun_commands(run, bam, tmp_path / "ref", folder)
    assert lines[:2] == [shlex.join(b2f), shlex.join(count)]
    summary = json.loads(lines[-1])
    assert summary.pop("free_bytes") > 0
    assert summary == {"library": "1417", "tag": "c375_l3_m2_i", "bam": str(bam), "bam_bytes": bam.stat().st_size,
                       "folder": str(folder), "executed": False, "seconds": {}}
    assert calls == [] and not (out / "1417" / "rerun").exists()
    # --execute, into another folder
    other = tmp_path / "trial"
    monkeypatch.setattr(sys, "argv", ["33", "run", "--rerun", str(other), "--execute"])
    stage.main()
    b2f, count = rerun.rerun_commands(run, bam, tmp_path / "ref", other / "c375_l3_m2_i")
    assert calls == [(b2f, other / "c375_l3_m2_i"), (count, other / "c375_l3_m2_i")]
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["executed"] is True and set(summary["seconds"]) == {"bamtofastq", "cellranger"}
    # a reference with other hashes, a BAM whose run did not end with exit status 0
    monkeypatch.setattr(stage, "REF_DIR", _reference(tmp_path / "ref2", HASHES | {"fasta_hash": "1" * 40}))
    with pytest.raises(ValueError, match="fasta_hash"):
        stage.main()
    bam.with_suffix(".time.txt").write_text("\tExit status: 1\n")
    with pytest.raises(SystemExit, match="c375_l3_m2_i.bam"):
        stage.main()


def test_stage33_compares_when_the_outs_exist(tmp_path, monkeypatch, capsys, library):
    stage, *_ = _stage33(monkeypatch, tmp_path)
    rec, mat, outs = library
    asked = []
    monkeypatch.setattr(stage, "require_passed", lambda lib, r, m: asked.append((lib, r, m)) or {"settings": TAGS})
    trial = tmp_path / "trial"
    args = ["33", "compare", "--tag", "tB", "--rerun", str(trial), "--recount", str(rec), "--matrices", str(mat)]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit, match=f"{trial}/tB/count/outs: no raw_feature_bc_matrix"):
        stage.main()
    monkeypatch.setattr(sys, "argv", args + ["--outs", str(outs)])
    stage.main()
    assert asked == [("1417", rec, mat)] * 2
    # a tag other than the one of D8 does not take the place of the D8 comparison, compare.json
    assert not (trial / "compare.json").exists()
    got = json.loads((trial / "compare_tB.json").read_text())
    want = rerun.compare("1417", "tB", TAGS, outs, rec, mat)
    assert got.pop("seconds") >= 0 and want.pop("seconds") >= 0 and got == json.loads(json.dumps(want))
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary == {"library": "1417", "tag": "tB", "compare": str(trial / "compare_tB.json"), "entries_compared": 5,
                       "entries_differing": 4, "umis_excess_ours": 2, "umis_excess_rerun": 2, "above_rerun_diff": True,
                       "cells_called": {"original": 4, "rerun": 3, "both": 2}, "seconds": summary["seconds"]}
    # at the tag of D8 the default is compare.json
    monkeypatch.setattr(rerun, "compare", lambda lib, tag, tags, o, r, m: want | {"tag": tag})
    monkeypatch.setattr(sys, "argv", ["33", "compare", "--rerun", str(trial), "--outs", str(outs)])
    stage.main()
    assert json.loads((trial / "compare.json").read_text())["tag"] == "c375_l3_m2_i"
