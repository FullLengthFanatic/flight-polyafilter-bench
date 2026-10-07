"""Stage 06: pa molecules, member reads, UMI collision background, member cap."""
import argparse
import json
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd
import pysam

from pafbench.config import GTF, LIBRARIES, OUT
from pafbench.genes import genes_at, load_gene_segments
from pafbench.molecules import choose_cap, consolidate_sites, gene_match, member_density, member_distances
from pafbench.sites import last_aligned_base

_SEG = None
_BY_K3 = None


def _init(by_k3):
    global _SEG, _BY_K3
    _SEG = load_gene_segments(GTF)
    _BY_K3 = by_k3


def _work(job):
    bam_path, ref = job
    rows, coll = [], defaultdict(int)
    with pysam.AlignmentFile(str(bam_path)) as bam:
        for a in bam.fetch(ref):
            if a.is_secondary or a.is_supplementary or a.has_tag("pa") or not a.has_tag("CB") or not a.has_tag("UB"):
                continue
            hits = _BY_K3.get((a.get_tag("CB"), a.get_tag("UB"), "-" if a.is_reverse else "+"))
            if not hits:
                continue
            for mol, mref, site, sgenes in hits:
                if mref != ref:
                    coll[mol] += 1
                    continue
                d5, d3 = member_distances(a, site)
                mg = genes_at(_SEG, ref, not a.is_reverse, last_aligned_base(a))
                rows.append([mol, a.query_name, d5, d3, "N" in a.cigarstring, a.is_duplicate, mg, gene_match(sgenes, mg)])
    cols = ["mol", "qname", "dist5", "dist3", "spliced", "dup", "genes", "gene_match"]
    return pd.DataFrame(rows, columns=cols), dict(coll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    base = OUT / args.library
    mol = consolidate_sites(pd.read_parquet(base / "pa_sites.parquet"))
    mol.to_parquet(base / "molecules.parquet", index=False)
    by_k3 = defaultdict(list)
    for r in mol.itertuples():
        by_k3[(r.cb, r.ub, r.strand)].append((r.mol, r.ref, r.site, r.genes))
    bam = LIBRARIES[args.library]
    with pysam.AlignmentFile(str(bam)) as fh:
        stats = [s for s in fh.get_index_statistics() if s.mapped > 0]
        lengths = {s.contig: fh.get_reference_length(s.contig) for s in stats}
    with Pool(args.workers, initializer=_init, initargs=(dict(by_k3),)) as pool:
        parts = pool.map(_work, [(bam, s.contig) for s in stats], chunksize=1)
    members = pd.concat([p[0] for p in parts], ignore_index=True)
    members.to_parquet(base / "members.parquet", index=False)
    coll_total = sum(sum(p[1].values()) for p in parts)
    genome = sum(lengths.values())
    other_bp = float(sum(genome - lengths[r] for r in mol.ref))
    bg_other = coll_total / other_bp if other_bp else 0.0
    curve = member_density(members, len(mol))
    bg_far = float(curve[curve.lo == 100000].density.iloc[0])
    background = max(bg_other, bg_far)
    cap = choose_cap(curve, background)
    out = {"molecules": len(mol), "members": len(members), "collision_reads_other_refs": coll_total,
           "background_other_refs_per_bp": bg_other, "background_far_same_ref_per_bp": bg_far,
           "background_used": background, "cap": cap, "curve": curve.to_dict(orient="records")}
    (base / "collisions.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("molecules", "members", "cap")}))


if __name__ == "__main__":
    main()
