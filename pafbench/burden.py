"""Stage 18: internal-priming burden (templated share of labeled molecules), same-site and same-gene comparisons, selection bias.

Descriptive only (spec 8): tissue, maturity, cell-type mix and the cell/nucleus difference are confounded.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import A_BANDS, LABEL_CUTOFFS
from .evaluate import gene_clusters, interval, resample_counts
from .polyasite import BINS, NULL_LABEL, distance_bin

LABELS = ("templated", "nontemplated", "ambiguous")
TEMPLATED_BANDS = tuple(name for name, lo, _ in A_BANDS if lo >= LABEL_CUTOFFS["templated_min"])


def _stratum(mol, by):
    return np.full(len(mol), "all", object) if by is None else mol[by].astype(str).to_numpy(object)


def share_table(mol, by=None):
    """Label counts per stratum (by None: one stratum 'all'); frac = stratum's share of molecules, share_all = templated /
    all, share_resolved = templated / (templated + nontemplated)."""
    t = pd.crosstab(pd.Series(_stratum(mol, by), name="stratum"), mol.label.to_numpy()).reindex(columns=list(LABELS), fill_value=0)
    t.columns.name = None
    t["total"] = t[list(LABELS)].sum(axis=1)
    t["frac"] = t.total / t.total.sum()
    t["share_all"] = t.templated / t.total
    t["share_resolved"] = t.templated / (t.templated + t.nontemplated)
    return t.reset_index()


def share_ci(mol, by=None, n=1000, seed=0, of="share_all"):
    """Gene-cluster bootstrap 95% interval per stratum (clusters and draws as evaluate.bootstrap) of share_all, or with
    of="frac" of the stratum's share of all molecules (for A bands, where the labels fix share_all)."""
    names, code = np.unique(gene_clusters(mol), return_inverse=True)
    strata, s = np.unique(_stratum(mol, by), return_inverse=True)
    total = np.zeros((len(names), len(strata)))
    templ = np.zeros_like(total)
    np.add.at(total, (code, s), 1)
    t = mol.label.to_numpy() == "templated"
    np.add.at(templ, (code[t], s[t]), 1)
    W = resample_counts(len(names), n, seed).astype(float)
    num, den = W @ templ, W @ total
    if of == "frac":
        num, den = den, den.sum(axis=1, keepdims=True)
    share = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
    ci = [interval(share[:, k])[:2] for k in range(len(strata))]
    return pd.DataFrame({"stratum": strata, "lo": [c[0] for c in ci], "hi": [c[1] for c in ci]})


def site_labels(mol):
    """One row per site (ref, strand, site) labeled with the most common label of its molecules (ties by label name)."""
    c = mol.groupby(["ref", "strand", "site", "label"]).size().reset_index(name="n")
    c = c.sort_values(["ref", "strand", "site", "n", "label"], ascending=[True, True, True, False, True])
    return c.drop_duplicates(["ref", "strand", "site"])[["ref", "strand", "site", "label"]].reset_index(drop=True)


def same_site(a, b, tol=5):
    """Each site of a matched to the nearest site of b on the same ref and strand within tol nt (equal distance: the lower
    coordinate); label pairs counted, label_b 'none' when unmatched."""
    sa, sb = site_labels(a), site_labels(b)
    keys = {k: i for i, k in enumerate(sorted(set(zip(sa.ref, sa.strand)) | set(zip(sb.ref, sb.strand))))}

    def coord(s):
        return np.fromiter((keys[k] for k in zip(s.ref, s.strand)), np.int64, len(s)) * (1 << 40) + s.site.to_numpy(np.int64)

    ca, cb = coord(sa), coord(sb)
    order = np.argsort(cb, kind="stable")
    far = 1 << 62
    cb = np.concatenate([[-far], cb[order], [far]])
    lb = np.concatenate([["none"], sb.label.to_numpy(object)[order], ["none"]])
    i = np.searchsorted(cb, ca)
    dl, dr = ca - cb[i - 1], cb[i] - ca
    near = np.where(dl <= dr, i - 1, i)
    label_b = np.where(np.minimum(dl, dr) <= tol, lb[near], "none")
    pairs = pd.DataFrame({"label_a": sa.label.to_numpy(object), "label_b": label_b})
    return pairs.groupby(["label_a", "label_b"]).size().reset_index(name="n")


def same_gene(a, b, min_n=20):
    """Spearman correlation of share_all across genes with at least min_n labeled molecules in both groups."""
    def per_gene(m):
        g = m[m.gene != ""]
        return g.assign(t=(g.label == "templated").to_numpy()).groupby("gene").t.agg(["sum", "size"])

    j = per_gene(a).join(per_gene(b), lsuffix="_a", rsuffix="_b", how="inner")
    j = j[(j["size_a"] >= min_n) & (j["size_b"] >= min_n)]
    rho, p = spearmanr(j["sum_a"] / j["size_a"], j["sum_b"] / j["size_b"])
    return {"n_genes": len(j), "spearman_rho": float(rho), "p_value": float(p)}


def selection_bias(mol, region_counts):
    """Share of labeled molecules per RE value ('' when absent) against the share of all primary records (records_share)
    and of primary records without the duplicate flag (records_nondup_share); totals of one tag."""
    labeled = mol.re.fillna("").astype(str).value_counts(normalize=True)
    rc = region_counts[region_counts.tag == region_counts.tag.iloc[0]].groupby("re")[["total", "total_nondup"]].sum()
    out = pd.concat([labeled.rename("labeled_share"), (rc.total / rc.total.sum()).rename("records_share"),
                     (rc.total_nondup / rc.total_nondup.sum()).rename("records_nondup_share")], axis=1).fillna(0.0)
    return out.rename_axis("re").sort_index().reset_index()


def proximity_table(mol, dist, null):
    """Molecules per PolyASite distance bin (polyasite.distance_bin; dist and null one value per row of mol): each label
    with a_band 'all', then NULL_LABEL with a_band 'all' (every molecule, binned by null, the distances of the sites moved
    upstream), then templated molecules per templated A band; every bin kept, zeros included."""
    b, label, band = distance_bin(dist), mol.label.to_numpy(), mol.a_band.to_numpy()
    blocks = [(lab, "all", b, label == lab) for lab in LABELS]
    blocks.append((NULL_LABEL, "all", distance_bin(null), np.ones(len(mol), bool)))
    blocks += [("templated", x, b, (label == "templated") & (band == x)) for x in TEMPLATED_BANDS]
    rows = [(lab, x, name, int((bb[m] == name).sum())) for lab, x, bb, m in blocks for name in BINS]
    return pd.DataFrame(rows, columns=["label", "a_band", "bin", "n"])


def near_shares(table):
    """Share of molecules within 10 nt of an atlas site per label (NULL_LABEL included, the chance level) and per templated
    A band of one group's proximity_table (None without molecules)."""
    def share(label, band):
        t = table[(table.label == label) & (table.a_band == band)]
        total = int(t.n.sum())
        return int(t[t.bin == BINS[0]].n.sum()) / total if total else None

    return {"label": {lab: share(lab, "all") for lab in LABELS + (NULL_LABEL,)},
            "templated_a_band": {x: share("templated", x) for x in TEMPLATED_BANDS}}
