import numpy as np
import pandas as pd

from .config import MEMBER_CAP

EDGES = [0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 50000, 100000, 1000000]
MOL_COLUMNS = ["mol", "cb", "ub", "strand", "ref", "site", "n_pa", "spread", "a20", "down20", "up40", "hexamer", "genes", "re", "mapq_min"]


def consolidate_sites(pa, max_spread=10):
    """One row per molecule (cb, ub, strand, ref, own gene label); site = lower median of its pa sites; spread > max_spread dropped; mapq_min over its pa reads."""
    d = pa[pa.cb.notna() & pa.ub.notna()]
    rows = []
    for (cb, ub, strand, ref, genes), g in d.groupby(["cb", "ub", "strand", "ref", "genes"], sort=False):
        sites = np.sort(g.site.to_numpy())
        spread = int(sites[-1] - sites[0])
        if spread > max_spread:
            continue
        site = int(sites[(len(sites) - 1) // 2])
        r = g.iloc[int(np.argmin(np.abs(g.site.to_numpy() - site)))]
        rows.append([len(rows), cb, ub, strand, ref, site, len(g), spread, int(r.a20), r.down20, r.up40, bool(r.hexamer), genes, r.re,
                     int(g.mapq.min())])
    return pd.DataFrame(rows, columns=MOL_COLUMNS)


def member_distances(a, site):
    """(read 5' start to site, read 3' end to site), both in transcript sense; negative dist3 = read passes the site."""
    if a.is_reverse:
        return a.reference_end - site, a.reference_start - site
    return site - a.reference_start, site - a.reference_end


def gene_match(site_genes, member_genes):
    if site_genes == "" and member_genes == "":
        return True
    return bool(set(filter(None, site_genes.split(";"))) & set(filter(None, member_genes.split(";"))))


def member_density(members, n_molecules, edges=EDGES, d5="dist5", d3="dist3", include_spliced=False):
    """Members (d3 >= -25; unspliced only unless include_spliced) per bp per molecule, by d5 bin."""
    m = members[members[d3] >= -25]
    if not include_spliced:
        m = m[~m.spliced]
    counts, _ = np.histogram(m[d5], bins=edges)
    widths = np.diff(edges)
    return pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "n": counts, "density": counts / (widths * n_molecules)})


def choose_cap(curve, background, candidates=(500, 750, 1000, 1500, 2000, 3000, 5000), factor=2.0):
    """Smallest d whose bins in [d, 2d) all have density <= factor * background."""
    for d in candidates:
        sel = curve[(curve.lo >= d) & (curve.hi <= 2 * d)]
        if len(sel) and bool((sel.density <= factor * background).all()):
            return d
    return None


def beyond_cap(members, caps=(500, 1000, 2000), d5="dist5_exon", d3="dist3_exon"):
    """For each cap: members (d3 >= -25) with d5 above the cap, and the molecules they belong to."""
    m = members[members[d3] >= -25]
    out = {"population": int(len(m))}
    for c in caps:
        far = m[m[d5] > c]
        out[str(c)] = {"members": int(len(far)), "molecules": int(far.mol.nunique())}
    return out


def select_members(members, cap=MEMBER_CAP):
    """Members that enter the truth set: same gene label, not past the site by more than 25 nt, 5' start within cap (exon-aware)."""
    return members[members.gene_match & (members.dist3_exon >= -25) & (members.dist5_exon <= cap)]
