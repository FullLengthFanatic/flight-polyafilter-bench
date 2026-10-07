import numpy as np
import pandas as pd

EDGES = [0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 50000, 100000, 1000000]
MOL_COLUMNS = ["mol", "cb", "ub", "strand", "ref", "site", "n_pa", "spread", "a20", "down20", "up40", "hexamer", "genes", "re"]


def consolidate_sites(pa, max_spread=10):
    """One row per molecule (cb, ub, strand, ref, own gene label); site = lower median of its pa sites; spread > max_spread dropped."""
    d = pa[pa.cb.notna() & pa.ub.notna()]
    rows = []
    for (cb, ub, strand, ref, genes), g in d.groupby(["cb", "ub", "strand", "ref", "genes"], sort=False):
        sites = np.sort(g.site.to_numpy())
        spread = int(sites[-1] - sites[0])
        if spread > max_spread:
            continue
        site = int(sites[(len(sites) - 1) // 2])
        r = g.iloc[int(np.argmin(np.abs(g.site.to_numpy() - site)))]
        rows.append([len(rows), cb, ub, strand, ref, site, len(g), spread, int(r.a20), r.down20, r.up40, bool(r.hexamer), genes, r.re])
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


def member_density(members, n_molecules, edges=EDGES):
    """Unspliced members (dist3 >= -25) per bp per molecule, by dist5 bin."""
    m = members[(~members.spliced) & (members.dist3 >= -25)]
    counts, _ = np.histogram(m.dist5, bins=edges)
    widths = np.diff(edges)
    return pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "n": counts, "density": counts / (widths * n_molecules)})


def choose_cap(curve, background, candidates=(500, 750, 1000, 1500, 2000, 3000, 5000), factor=2.0):
    """Smallest d whose bins in [d, 2d) all have density <= factor * background."""
    for d in candidates:
        sel = curve[(curve.lo >= d) & (curve.hi <= 2 * d)]
        if len(sel) and bool((sel.density <= factor * background).all()):
            return d
    return None
