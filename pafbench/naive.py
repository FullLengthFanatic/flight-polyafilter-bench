"""A recount of one BAM by plain counting, for checks A4 and A6 of stage 24 (design of 2026-10-12, section 3.2).

It shares no code with pafbench.molstream and pafbench.recount, whose matrices it is compared with: the read groups are
one plain dict entry each, with cell, UMI and gene packed into one integer (cell row << 40 | UMI code << 16 | gene row; UMI
code A 0, C 1, G 2, T 3, first base highest), and the rule is a loop over them.

A4: the documented counting rule on the groups by corrected UMI (UB). Cell Ranger 9.0 algorithm page
(https://www.10xgenomics.com/support/software/cell-ranger/9.0/algorithms-overview/cr-gex-algorithm, wording in design 2.3):
"the gene annotation with the most supporting reads is kept for UMI counting, and the other read groups are discarded. In
case of a tie for maximal read support, all read groups are discarded".
A6: Cell Ranger's own counting code on the groups by raw UMI (UR), a port of correct_umis and
determine_low_support_umigenes (cellranger 9.0.1, lib/rust/tx_annotation/src/mark_dups.rs; copy in
design_m3_draft/review/v3/counting_rule/dl_src). It is what a rerun would do with the surviving reads, alignment and cell
calling aside.
"""
import gzip
from array import array
from itertools import groupby

import numpy as np
import pysam
import scipy.sparse as sp

_DIGITS = str.maketrans("ACGT", "0123")


def rows(tsv_gz):
    """{first column: row} of barcodes.tsv.gz or features.tsv.gz."""
    with gzip.open(tsv_gz, "rt") as fh:
        return {line.rstrip("\n").split("\t")[0]: i for i, line in enumerate(fh)}


def _umi_code(umi):
    if len(umi) != 12:
        raise ValueError(f"UMI {umi!r} is not 12 bases")
    return int(umi.translate(_DIGITS), 4)          # ValueError on a letter other than A, C, G and T


def read_groups(bam_path, cells, genes):
    """({key: reads} by corrected UMI, {key: reads} by raw UMI, records read) of a BAM, indexed or not.

    Counted are the primary records with xf bit 1, the CB of a called cell (cells: {barcode: row}), a UB and one gene of
    the feature list in GX (genes: {gene id: row}); the read groups of design 3.1.
    """
    by_ub, by_ur = {}, {}
    n = 0
    with pysam.AlignmentFile(str(bam_path)) as bam:
        for a in bam.fetch(until_eof=True):
            n += 1
            if a.flag & 0x900:
                continue
            try:
                if not a.get_tag("xf") & 1:
                    continue
                cell, gene, ub = cells.get(a.get_tag("CB")), genes.get(a.get_tag("GX")), a.get_tag("UB")
            except KeyError:
                continue
            if cell is None or gene is None:
                continue
            for reads, umi in ((by_ub, ub), (by_ur, a.get_tag("UR"))):
                key = (cell << 40) | (_umi_code(umi) << 16) | gene
                reads[key] = reads.get(key, 0) + 1
    return by_ub, by_ur, n


def rule_entries(reads):
    """The documented rule on {key: reads}: one element (cell row << 16 | gene row) per counted UMI, the gene with strictly
    the most reads of its cell and UMI."""
    out = array("q")
    for unit, keys in groupby(sorted(reads), lambda key: key >> 16):
        by_gene = [(reads[key], key & 0xFFFF) for key in keys]
        top = max(by_gene)[0]
        winners = [gene for n, gene in by_gene if n == top]
        if len(winners) == 1:
            out.append(((unit >> 24) << 16) | winners[0])
    return out


def correct_and_mark(counts):
    """UMI correction and the low-support step of Cell Ranger 9.0.1 on the reads of one barcode.

    counts: {UMI code << 16 | gene row: reads} by raw UMI. Returns ({raw: corrected}, low-support pairs), both in that
    packing. From lib/rust/tx_annotation/src/mark_dups.rs:
    - correct_umis, lines 19 to 59: within a gene, a UMI goes to the UMI at Hamming distance one with more reads, "If
      there's a 1-HD UMI w/ equal count, move to the lexicographically larger UMI." (line 43); the order of UMIs is that
      of their codes.
    - lines 226 to 232: "Before determining low-support UMI-genes, count one read of each corrected UMI" / "to match the
      behaviour of Cell Ranger 3.": one read of each corrected pair is taken from the raw UMI and added to the corrected one.
    - determine_low_support_umigenes, lines 86 to 108: "Mark as low support umigenes with frequency below the maximum for
      that UMI.", and every gene of the UMI when the maximum is tied (`if max_is_tied || count < max_count`, line 101).
    """
    corrected = {}
    for key, n in counts.items():
        best, best_n = key, n
        for shift in range(16, 40, 2):
            base = (key >> shift) & 3
            for other in range(4):
                if other != base:
                    test = key ^ ((base ^ other) << shift)
                    test_n = counts.get(test, 0)
                    if test_n > best_n or (test_n == best_n and test > best):
                        best, best_n = test, test_n
        if best != key:
            corrected[key] = best
    moved = dict(counts)
    for key, dest in corrected.items():
        moved[key] -= 1
        moved[dest] += 1
    low = set()
    for _, keys in groupby(sorted(moved), lambda key: key >> 16):
        keys = list(keys)
        top = max(moved[key] for key in keys)
        tied = sum(moved[key] == top for key in keys) >= 2
        low.update(key for key in keys if tied or moved[key] < top)
    return corrected, low


def port_entries(reads):
    """Cell Ranger's counting code on {key: reads} by raw UMI: one element (cell row << 16 | gene row) per molecule, a
    corrected UMI and gene that is not low support (a read of such a pair is the UMI count, mark_dups.rs lines 301 to 325)."""
    out = array("q")
    for cell, keys in groupby(sorted(reads), lambda key: key >> 40):
        counts = {key & 0xFFFFFFFFFF: reads[key] for key in keys}
        corrected, low = correct_and_mark(counts)
        for pair in {corrected.get(key, key) for key in counts} - low:
            out.append((cell << 16) | (pair & 0xFFFF))
    return out


def recount_bam(bam_path, cr_matrix_dir):
    """The two recounts of a BAM as cells by genes CSR int32, rows and columns those of barcodes.tsv.gz and features.tsv.gz
    in cr_matrix_dir: rule (rule_entries on the groups by UB) and port (port_entries on the groups by UR), with records
    (read), in_groups (records in read groups), read_groups and raw_groups (groups by UB and by UR)."""
    cells, genes = rows(f"{cr_matrix_dir}/barcodes.tsv.gz"), rows(f"{cr_matrix_dir}/features.tsv.gz")
    by_ub, by_ur, n = read_groups(bam_path, cells, genes)
    out = {"records": n, "in_groups": sum(by_ub.values()), "read_groups": len(by_ub), "raw_groups": len(by_ur)}
    for name, entries in (("rule", rule_entries(by_ub)), ("port", port_entries(by_ur))):
        e = np.frombuffer(entries, np.int64)
        m = sp.csr_matrix((np.ones(e.size, np.int32), (e >> 16, e & 0xFFFF)), shape=(len(cells), len(genes)))
        m.sum_duplicates()
        out[name] = m
    return out
