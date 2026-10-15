"""Stage 33: one Cell Ranger rerun on a BAM the original tool filtered, compared with our matrix at the same setting (design
of 2026-10-12, section 3.11; decision D8: 1417 at c375_l3_m2_i).

The rerun takes the route of the polyAfilter README: "bamtofastq may need to be used to convert the BAM file back to FASTQ
files, which can be used as input to cellranger count" (vendor/polyAfilter/README.md, line 176, commit 5ce0098). The two
commands are built from the _cmdline and _invocation of the original run (rerun_commands) and run by run_command, which the
tests replace.

bamtofastq v1.4.1 (`bamtofastq --help`: "bamtofastq [options] <bam> <output-path>"; source at the tag v1.4.1 of
https://github.com/10XGenomics/bamtofastq, src/main.rs): it creates <output-path> itself and stops when it exists ("error
creating output directory: {:?}. Does it already exist?", line 1094); it writes one folder per read group of the header,
named after the ID without its last ":" field ("vv.rsplitn(2, ':')", line 321) with ":" replaced by "_" (line 671), and in
it FASTQs named "{}_S1_L{:03}_R1_{:03}.fastq.gz" with the sample name "bamtofastq" (lines 680 and 775). Our observation on
a synthetic BAM with the @RG and @CO lines of 1417/original/c375_l3_m2_i.bam: the folder S1_0_1_unknow_flowcell with
bamtofastq_S1_L000_R1_001.fastq.gz and bamtofastq_S1_L000_R2_001.fastq.gz, and the error above on a second run.

What a Cell Ranger outs folder says about family size (reads per UMI). The matrices hold UMIs only. molecule_info.h5 has a
row per "unique (UMI, cell-barcode, feature) tuple indicating the feature best supported by the reads (i.e., including PCR
duplicates) assigned to that UMI and cell-barcode", and count is the "Number of reads associated with this putative
molecule that were confidently mapped to the assigned feature." It covers "all molecules that contain a valid barcode, a
valid UMI, and were assigned with high confidence to a gene", under the "processed (i.e. corrected) UMI sequence"
(https://www.10xgenomics.com/support/software/cell-ranger/9.0/analysis/outputs/cr-outputs-molecule-info, copy in
design_m3_draft/research/dl/molinfo.txt). So the reads of every counted molecule can be read, in every barcode. Not
readable: the reads of a read group that lost to another gene or tied (its UMI is not counted and has no row; our
observation, check A2 of stage 24: the rows of pass_filter barcodes are exactly our read groups with a winner read), reads
without a valid UMI or not confidently mapped, and reads per raw UMI before correction.
"""
import gzip
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pysam
import scipy.sparse as sp

from . import matrices
from .config import LIBRARIES, REPO, RERUN_DIFF
from .recount import _in_sorted, _listing, check_entries, read_cr_matrix

CELLRANGER_DIR = Path("/home/simone.picelli/cellranger-9.0.1")       # cellranger-9.0.1, the version of the original runs
BAMTOFASTQ = CELLRANGER_DIR / "lib" / "bin" / "bamtofastq"
CELLRANGER = CELLRANGER_DIR / "cellranger"
# the flags of the original call that point to its FASTQs and name its run; the rerun sets its own
_OWN = ("id", "transcriptome", "fastqs", "sample")


def original_run(run_dir):
    """{flag: value} of the `cellranger count` call in run_dir/_cmdline (every flag is written --name=value there), after
    checking that run_dir/_invocation, the pipeline call Cell Ranger made of it, has the same reference, introns and BAM."""
    words = shlex.split((Path(run_dir) / "_cmdline").read_text())
    if words[:2] != ["cellranger", "count"]:
        raise ValueError(f"{run_dir}/_cmdline is no cellranger count call: {' '.join(words[:2])}")
    flags = dict(w[2:].split("=", 1) for w in words[2:])
    inv = dict(re.findall(r"^    (\w+) += (.+),$", (Path(run_dir) / "_invocation").read_text(), re.M))
    seen = {"transcriptome": json.loads(inv["reference_path"]), "include-introns": inv["include_introns"],
            "create-bam": "false" if inv["no_bam"] == "true" else "true"}
    differ = [k for k, v in seen.items() if flags[k] != v]
    if differ:
        raise ValueError(f"{run_dir}: _cmdline and _invocation differ in {', '.join(differ)}")
    return flags


def fastq_folder(bam):
    """The folder bamtofastq writes the FASTQs of bam into (module docstring): the ID of its one read group without the
    last ":" field, ":" replaced by "_"."""
    with pysam.AlignmentFile(str(bam), check_sq=False) as fh:
        rgs = fh.header.to_dict().get("RG", [])
    if len(rgs) != 1:
        raise ValueError(f"{bam}: {len(rgs)} read groups; the rerun is built for the one of the original run")
    return rgs[0]["ID"].rsplit(":", 1)[0].replace(":", "_")


def check_reference(transcriptome, molecule_info_h5):
    """ValueError unless the reference.json of transcriptome has the FASTA and GTF hashes that the metrics_json of the
    original run's molecule_info.h5 records (reference_fasta_hash, reference_gtf_hash.gz; our observation on 1417)."""
    ref = json.loads((Path(transcriptome) / "reference.json").read_text())
    with h5py.File(molecule_info_h5, "r") as f:
        metrics = json.loads(f["metrics_json"][()])
    differ = [k for k in ("fasta_hash", "gtf_hash.gz") if ref.get(k) != metrics.get(f"reference_{k}")]
    if differ:
        raise ValueError(f"{transcriptome} is not the reference of the original run ({molecule_info_h5}): {', '.join(differ)}")


def check_rerun_dir(path):
    """ValueError when path lies in the repository or in the Cell Ranger run folder of a library (the folder above outs):
    FASTQs and Cell Ranger's working files go to neither."""
    p = Path(path).resolve()
    for d in (REPO, *(bam.parent.parent for bam in LIBRARIES.values())):
        d = d.resolve()
        if p == d or d in p.parents:
            raise ValueError(f"the rerun folder {path} must not be under {d}")


def rerun_commands(run_dir, bam, transcriptome, folder):
    """(bamtofastq, cellranger count) as argument lists. bamtofastq writes the FASTQs of bam to folder/fastq; cellranger
    count runs on them with the flags of the original call in run_dir (original_run), its own run id (the name of folder),
    FASTQs and sample, the given transcriptome, and its output in folder/count. bamtofastq reads with as many threads as
    the original run had cores, as the README passes one $NTHREADS to both (lines 185 and 191)."""
    check_rerun_dir(folder)
    folder, flags = Path(folder), original_run(run_dir)
    fastq = folder / "fastq"
    own = {"id": folder.name, "transcriptome": str(transcriptome), "fastqs": str(fastq / fastq_folder(bam)),
           "sample": "bamtofastq"}
    count = [str(CELLRANGER), "count", *(f"--{k}={own[k]}" for k in _OWN),
             *(f"--{k}={v}" for k, v in flags.items() if k not in _OWN), f"--output-dir={folder / 'count'}"]
    return [str(BAMTOFASTQ), f"--nthreads={flags['localcores']}", str(bam), str(fastq)], count


def run_command(cmd, log, cwd):
    """cmd in cwd, its stdout and stderr to the file log; CalledProcessError when it fails. Our observation: cellranger
    count writes __<id>.mro into the directory it runs in (a --dry run on synthetic FASTQs), so cwd is the rerun folder."""
    with open(log, "w") as fh:
        subprocess.run(cmd, cwd=cwd, check=True, stdout=fh, stderr=subprocess.STDOUT)


def run_rerun(commands, folder):
    """The two commands of rerun_commands in folder (created), each logged to folder/<name>.log; after it ends with exit
    status 0, folder/<name>.json records it. A command with that json is not run again: bamtofastq stops on an existing
    FASTQ folder. Returns {name: seconds} of the commands run."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    seconds = {}
    for name, cmd in zip(("bamtofastq", "cellranger"), commands):
        done = folder / f"{name}.json"
        if done.exists():
            continue
        t0 = time.monotonic()
        run_command(cmd, folder / f"{name}.log", folder)
        seconds[name] = round(time.monotonic() - t0, 1)
        done.write_text(json.dumps({"cmd": cmd, "seconds": seconds[name]}, indent=1))
    return seconds


def read_mex(matrix_dir):
    """A Cell Ranger matrix folder: (cells by genes CSR int32, barcodes, features as (id, name))."""
    d = Path(matrix_dir)
    with gzip.open(d / "barcodes.tsv.gz", "rt") as fh:
        barcodes = fh.read().split()
    with gzip.open(d / "features.tsv.gz", "rt") as fh:
        features = [tuple(line.split("\t")[:2]) for line in fh.read().splitlines()]
    return read_cr_matrix(d), barcodes, features


def union_features(ours, theirs):
    """Our features, then those of theirs whose id is not ours, as (id, name)."""
    known = {g for g, _ in ours}
    return list(ours) + [f for f in theirs if f[0] not in known]


def compare_matrices(ours, barcodes, features, theirs, their_barcodes, their_features):
    """Our matrix (cells by genes, rows barcodes, columns features) against theirs, both joined by barcode and gene id;
    the rows of theirs that are not among barcodes are left out, and a gene in one feature list only is a column of zeros
    on the other side.

    Entries compared and equal (recount.check_entries), UMIs on each side, UMIs only we count and only they count (summed
    over the entries where one side has more), their share of our UMIs against RERUN_DIFF (design section 6), our cells
    absent from their matrix, the genes of one list only with their UMIs, by_gene for every gene with a differing entry
    (most differing UMIs first) and the first recount.EXAMPLES differing entries.
    """
    union = union_features(features, their_features)
    col = {g: j for j, (g, _) in enumerate(union)}
    row = {b: i for i, b in enumerate(barcodes)}
    shape = (len(barcodes), len(union))
    t = theirs.tocoo()
    r = np.array([row.get(b, -1) for b in their_barcodes], np.int64)[t.row]
    c = np.array([col[g] for g, _ in their_features], np.int64)[t.col]
    theirs = sp.csr_matrix((t.data[r >= 0], (r[r >= 0], c[r >= 0])), shape=shape, dtype=np.int32)
    ours = sp.csr_matrix((ours.data, ours.indices, ours.indptr), shape=shape, dtype=np.int32)
    e = check_entries(ours, theirs, 0)
    dif = e["differing"]
    d = (dif.ours - dif.theirs).to_numpy()
    sums = {"umis_ours": int(ours.sum()), "umis_rerun": int(theirs.sum()), "umis_only_ours": int(d[d > 0].sum()),
            "umis_only_rerun": int(-d[d < 0].sum())}
    share = (sums["umis_only_ours"] + sums["umis_only_rerun"]) / max(sums["umis_ours"], 1)
    per_gene = pd.DataFrame({"j": dif.gene.to_numpy(), "umis_only_ours": np.maximum(d, 0), "umis_only_rerun": np.maximum(-d, 0),
                             "entries_differing": 1}).groupby("j").sum()
    j = per_gene.index.to_numpy()
    by_gene = per_gene.assign(gene_id=[union[k][0] for k in j], gene_name=[union[k][1] for k in j],
                              umis_ours=np.asarray(ours.sum(axis=0)).ravel()[j],
                              umis_rerun=np.asarray(theirs.sum(axis=0)).ravel()[j],
                              n=per_gene.umis_only_ours + per_gene.umis_only_rerun)
    by_gene = by_gene.sort_values(["n", "gene_id"], ascending=[False, True])[
        ["gene_id", "gene_name", "umis_ours", "umis_rerun", "umis_only_ours", "umis_only_rerun", "entries_differing"]]
    theirs_ids, n_ours = {f[0] for f in their_features}, len(features)
    only_ours = [j for j in range(n_ours) if union[j][0] not in theirs_ids]
    only_theirs = list(range(n_ours, len(union)))
    genes_only = {name: {"genes": len(js), "umis": int(m[:, js].sum()), "ids": [union[j][0] for j in js]}
                  for name, js, m in (("genes_only_ours", only_ours, ours), ("genes_only_rerun", only_theirs, theirs))}
    present = set(their_barcodes)
    return {k: e[k] for k in ("entries_compared", "entries_equal", "entries_differing", "per_million")} | sums | {
        "umis_differing_share": share, "rerun_diff": RERUN_DIFF, "above_rerun_diff": share > RERUN_DIFF,
        "cells_absent_from_rerun_matrix": [b for b in barcodes if b not in present]} | genes_only | {
        "by_gene": json.loads(by_gene.to_json(orient="records")),
        "differing": _listing(dif, barcodes, [g for g, _ in union])}


def called_cells(barcodes, called):
    """The cells the rerun calls (called) among the original called cells (barcodes)."""
    ours, theirs = set(barcodes), set(called)
    return {"original": len(ours), "rerun": len(theirs), "both": len(ours & theirs), "original_only": len(ours - theirs),
            "rerun_only": len(theirs - ours), "original_only_barcodes": sorted(ours - theirs)}


def our_molecules(recount_dir, tag, tags):
    """The molecules our matrix counts at tag ("unfiltered" or one of tags, in bit order), as (key, reads) sorted by key:
    key is cell row << 40 | UMI code << 16 | gene row, the layout of the parts of stage 23, reads the reads of the read
    group that survive at tag.

    At a setting these are the rows of counted_molecules.parquet whose lost bit is unset and the rows of
    other_read_groups.parquet whose gained bit is set (design 3.2, step 4). A molecule with a read removed at tag
    (any_removed) keeps n_reads minus rm_<tag> of its read group in recount_dir/parts, summed over the references that
    hold its key, as stage 24 sums n_reads.
    """
    recount_dir, cols = Path(recount_dir), ["cell", "gene", "umi", "n_reads", "any_removed"]
    held = pd.read_parquet(recount_dir / "counted_molecules.parquet", columns=cols + ["lost"])
    bit = 0 if tag == "unfiltered" else 1 << tags.index(tag)
    if bit:
        gained = pd.read_parquet(recount_dir / "other_read_groups.parquet", columns=cols + ["gained"])
        held = pd.concat([held[(held.lost & bit) == 0], gained[(gained.gained & bit) != 0]], ignore_index=True)
    key = ((held.cell.to_numpy().astype(np.uint64) << np.uint64(40)) | (held.umi.to_numpy().astype(np.uint64) << np.uint64(16))
           | held.gene.to_numpy().astype(np.uint64))
    reads = held.n_reads.to_numpy().astype(np.int64)
    partial = (held.any_removed.to_numpy() & bit) != 0
    if partial.any():
        want, keys, rms = np.unique(key[partial]), [], []
        for part in sorted((recount_dir / "parts").glob("*.read_groups.parquet")):
            t = pq.read_table(part, columns=["key", f"rm_{tag}"])
            k = t.column("key").to_numpy()
            sel = _in_sorted(want, k)
            keys.append(k[sel])
            rms.append(t.column(f"rm_{tag}").to_numpy()[sel].astype(np.int64))
        rm = np.zeros(want.size, np.int64)
        np.add.at(rm, np.searchsorted(want, np.concatenate(keys)), np.concatenate(rms))
        reads[partial] -= rm[np.searchsorted(want, key[partial])]
    o = np.argsort(key, kind="stable")
    return key[o], reads[o]


def rerun_molecules(h5_path, cell_row, gene_col):
    """The molecules of molecule_info.h5 in the cells of cell_row, as (key, count) sorted by key in the layout of
    our_molecules. Joined by string: cell_row maps a barcode to our row, gene_col a feature id to its column. The barcodes
    of the file carry no "-1", which is its gem_group column (our observation, recount.molecule_info); ValueError for a
    file of several gem groups."""
    with h5py.File(h5_path, "r") as f:
        gem = np.unique(f["gem_group"][:])
        if gem.size > 1:
            raise ValueError(f"{h5_path}: {gem.size} gem groups; the join takes the barcodes of one")
        row = np.array([cell_row.get(f"{b}-{gem[0]}", -1) for b in f["barcodes"][:].astype(str)], np.int64)
        col = np.array([gene_col[g] for g in f["features/id"][:].astype(str)], np.int64)
        cell = row[f["barcode_idx"][:]]
        keep = cell >= 0
        key = ((cell[keep].astype(np.uint64) << np.uint64(40)) | (f["umi"][:][keep].astype(np.uint64) << np.uint64(16))
               | col[f["feature_idx"][:][keep]].astype(np.uint64))
        count = f["count"][:][keep].astype(np.int64)
    o = np.argsort(key, kind="stable")
    return key[o], count[o]


def by_family(ours, theirs):
    """Our molecules (our_molecules) joined with the rerun's (rerun_molecules) on cell, UMI and gene: molecules on each
    side, in both, only ours and only the rerun's, those in both with the same reads, and by_family: per family size the
    molecules only ours (by our reads), only the rerun's (by its count) and in both (by our reads)."""
    (ok, reads), (tk, count) = ours, theirs
    in_t, in_o = _in_sorted(tk, ok), _in_sorted(ok, tk)
    sizes = {"only_ours": reads[~in_t], "only_rerun": count[~in_o], "both": reads[in_t]}
    family = np.unique(np.concatenate(list(sizes.values())))
    table = [{"reads": int(f)} | {k: int((v == f).sum()) for k, v in sizes.items()} for f in family]
    return {"molecules_ours": int(ok.size), "molecules_rerun": int(tk.size), "both": int(in_t.sum()),
            "only_ours": int((~in_t).sum()), "only_rerun": int((~in_o).sum()),
            "both_same_reads": int((reads[in_t] == count[in_o]).sum()), "by_family": table}


def compare(lib, tag, tags, outs, recount_dir, matrix_dir, matrix="raw_feature_bc_matrix"):
    """compare.json of design 3.11: the matrix of the Cell Ranger outs folder outs (raw_feature_bc_matrix) restricted to
    the cells our matrix holds, the called cells of the original run, against our matrix at tag (matrices.load; tags:
    the settings of recount.json in bit order), compare_matrices; the cells the rerun calls among them (its
    filtered_feature_bc_matrix), called_cells; and its molecule_info.h5 against our molecules, by_family."""
    t0 = time.monotonic()
    matrix_dir, outs = Path(matrix_dir), Path(outs)
    barcodes = (matrix_dir / "barcodes.tsv").read_text().split()
    features = [tuple(line.split("\t")[:2]) for line in (matrix_dir / "features.tsv").read_text().splitlines()]
    theirs, their_barcodes, their_features = read_mex(outs / matrix)
    res = {"library": lib, "tag": tag, "outs": str(outs), "matrix": matrix} | compare_matrices(
        matrices.load(lib, tag, matrix_dir), barcodes, features, theirs, their_barcodes, their_features)
    with gzip.open(outs / "filtered_feature_bc_matrix" / "barcodes.tsv.gz", "rt") as fh:
        res["cells_called"] = called_cells(barcodes, fh.read().split())
    gene_col = {g: j for j, (g, _) in enumerate(union_features(features, their_features))}
    res["molecules"] = by_family(our_molecules(recount_dir, tag, tags),
                                 rerun_molecules(outs / "molecule_info.h5", {b: i for i, b in enumerate(barcodes)}, gene_col))
    return res | {"seconds": round(time.monotonic() - t0, 1)}
