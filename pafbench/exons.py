from collections import defaultdict

import numpy as np


def load_collapsed_exons(gtf):
    """{gene_id: (starts, ends)}: union of the exons of all the gene's transcripts, sorted, 0-based half-open."""
    ex = defaultdict(list)
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if f[2] != "exon":
                continue
            gid = f[8].split('gene_id "', 1)[1].split('"', 1)[0]
            ex[gid].append((int(f[3]) - 1, int(f[4])))
    out = {}
    for gid, lst in ex.items():
        lst.sort()
        merged = []
        for s, e in lst:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        arr = np.asarray(merged, np.int64)
        out[gid] = (arr[:, 0].copy(), arr[:, 1].copy())
    return out


def exonic_bases(exons, a, b):
    """Collapsed-exon bases in [a, b)."""
    if b <= a:
        return 0
    s, e = exons
    return int((np.clip(e, a, b) - np.clip(s, a, b)).sum())


def _signed(exons, a, b):
    return exonic_bases(exons, a, b) if a <= b else -exonic_bases(exons, b, a)


def inside_exons(exons, a, b):
    """True if [a, b) lies inside one collapsed exon."""
    s, e = exons
    i = int(np.searchsorted(s, a, side="right")) - 1
    return i >= 0 and bool(e[i] >= b)


def exon_distances(blocks, is_reverse, site, exons):
    """(dist5, dist3) in collapsed-exon bases, or None if the site or any read block is not exonic."""
    last_cdna = site if is_reverse else site - 1
    if not inside_exons(exons, last_cdna, last_cdna + 1):
        return None
    if not all(inside_exons(exons, s, e) for s, e in blocks):
        return None
    r0, r1 = blocks[0][0], blocks[-1][1]
    if is_reverse:
        return _signed(exons, site, r1), _signed(exons, site, r0)
    return _signed(exons, r0, site), _signed(exons, r1, site)


def member_distance(blocks, is_reverse, site, site_genes, exons_by_gene, genomic):
    """Exon-aware (dist5, dist3, mode): smallest exon distance over the site's genes, else the genomic pair."""
    best = None
    for g in filter(None, site_genes.split(";")):
        ex = exons_by_gene.get(g)
        if ex is None:
            continue
        d = exon_distances(blocks, is_reverse, site, ex)
        if d is not None and (best is None or d[0] < best[0]):
            best = d
    if best is None:
        return genomic[0], genomic[1], "genome"
    return best[0], best[1], "exon"
