"""Stage 14: spike-in truth sets and grid flags (spec 4.2) from the STAR realignment of unplaced unmapped reads to SIRV Set 4.

Reads <base>/star_Aligned.out.bam (scripts/09_spikein_star.sh) and writes under <out>: star_sorted.bam (+ .bai, kept when
the index exists and is newer than star_Aligned.out.bam), pa_sites.parquet, molecules.parquet, members.parquet,
truth/molecules_labeled.parquet (stage 12 columns plus spike_label, dist_to_end, module), truth/truth_reads.parquet,
grid/truth_flags.parquet, grid/region_counts.parquet. gx and re are "" when the read has no GX or RE tag (none has one
after the realignment). The grid is scanned in full on every run (no per-contig parts to resume), so it always matches
this run's truth reads.
"""
import argparse
import json
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import pysam

from pafbench.config import OUT, SIRV_FASTA, SIRV_GTF
from pafbench.exons import load_collapsed_exons
from pafbench.flags import setting_tag
from pafbench.genes import load_gene_segments
from pafbench.grid import all_settings
from pafbench.molecules import MEMBER_COLUMNS, consolidate_sites, member_rows
from pafbench.regions import load_region_segments
from pafbench.sites import PA_COLUMNS
from pafbench.spikein import add_spike_labels, annotated_ends, merge_scans, spike_pa_row, spike_summary
from pafbench.stream import scan_reference
from pafbench.transcripts import load_transcripts
from pafbench.truth import check_pa_counts, label_truth_molecules, truth_reads

_G = {}  # set in the parent before each pool is forked


def _pa(job):
    bam, ref = job
    with pysam.AlignmentFile(str(bam)) as fh, pysam.FastaFile(str(SIRV_FASTA)) as fa:
        return [spike_pa_row(a, fa, _G["seg"]) for a in fh.fetch(ref)
                if not (a.is_secondary or a.is_supplementary) and a.has_tag("pa")]


def _members(job):
    bam, ref = job
    with pysam.AlignmentFile(str(bam)) as fh:
        return member_rows(fh.fetch(ref), ref, _G["by_k3"], _G["seg"], _G["exons"])


def _grid(job):
    bam, ref = job
    truth, region, _ = scan_reference(bam, SIRV_FASTA, ref, _G["tx"], _G["settings"], _G["truth"])
    return truth, region


def _per_contig(fn, bam, stats, workers):
    """fn over the contigs (largest first), results in BAM header order."""
    jobs = [(bam, s.contig) for s in sorted(stats, key=lambda s: -s.mapped)]
    with get_context("fork").Pool(workers) as pool:
        res = dict(zip([j[1] for j in jobs], pool.map(fn, jobs, chunksize=1)))
    return [res[s.contig] for s in stats]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--base", type=Path, help="dir with star_Aligned.out.bam (default OUT/LIB/spikein)")
    ap.add_argument("--out", type=Path, help="output dir (default: base)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.monotonic()
    base = args.base or OUT / args.library / "spikein"
    out = args.out or base
    out.mkdir(parents=True, exist_ok=True)
    bam, src = out / "star_sorted.bam", base / "star_Aligned.out.bam"
    bai = Path(f"{bam}.bai")
    if not bai.exists() or src.stat().st_mtime_ns > bai.stat().st_mtime_ns:
        pysam.sort("-@", str(args.workers), "-o", str(bam), str(src))
        pysam.index(str(bam))
    with pysam.AlignmentFile(str(bam)) as fh:
        stats = [s for s in fh.get_index_statistics() if s.mapped > 0]

    _G["seg"] = load_gene_segments(SIRV_GTF)
    pa = pd.DataFrame([r for rows in _per_contig(_pa, bam, stats, args.workers) for r in rows], columns=PA_COLUMNS)
    pa[["gx", "re"]] = pa[["gx", "re"]].fillna("").astype("str")
    mol = consolidate_sites(pa)
    by_k3 = defaultdict(list)
    for r in mol.itertuples():
        by_k3[(r.cb, r.ub, r.strand)].append((r.mol, r.ref, r.site, r.genes))
    _G.update(by_k3=dict(by_k3), exons=load_collapsed_exons(SIRV_GTF))
    rows = [r for part in _per_contig(_members, bam, stats, args.workers) for r in part]
    members = pd.DataFrame(rows, columns=MEMBER_COLUMNS)
    members["spliced"] = members["spliced"].astype(bool)

    labeled = add_spike_labels(label_truth_molecules(mol, pa, members, load_region_segments(SIRV_GTF)),
                               annotated_ends(SIRV_GTF))
    reads = truth_reads(mol, pa, members)
    check_pa_counts(mol, reads)

    settings = all_settings()
    _G.update(tx=load_transcripts(SIRV_GTF), settings=settings, truth=set(reads.qname))
    truth_flags, region = merge_scans(_per_contig(_grid, bam, stats, args.workers), [setting_tag(s) for s in settings])

    (out / "truth").mkdir(exist_ok=True)
    (out / "grid").mkdir(exist_ok=True)
    pa.to_parquet(out / "pa_sites.parquet", index=False)
    mol.to_parquet(out / "molecules.parquet", index=False)
    members.to_parquet(out / "members.parquet", index=False)
    labeled.to_parquet(out / "truth" / "molecules_labeled.parquet", index=False)
    reads.to_parquet(out / "truth" / "truth_reads.parquet", index=False)
    truth_flags.to_parquet(out / "grid" / "truth_flags.parquet", index=False)
    region.to_parquet(out / "grid" / "region_counts.parquet", index=False)
    print(json.dumps({"library": args.library} | spike_summary(pa, labeled, members, reads)
                     | {"truth_flag_rows": len(truth_flags), "seconds": round(time.monotonic() - t0, 1)}))


if __name__ == "__main__":
    main()
