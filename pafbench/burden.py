"""Stage 18: internal-priming burden (templated share of labeled molecules), same-site and same-gene comparisons, selection bias.
Stage 29: the templated share corrected for label purity, by region.

Descriptive only (spec 8): tissue, maturity, cell-type mix and the cell/nucleus difference are confounded.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import A_BANDS, LABEL_CUTOFFS, REGION_REF_MIN
from .evaluate import gene_clusters, interval, resample_counts, tail_share
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


def band_corrected_share(purity):
    """Corrected templated share of stage 29 (design 3.7) from a label_purity or tail_share result: the sum over the
    templated A bands of n_b x (1 - tail-primed share of b), divided by the labeled molecules. None without molecules or
    when a templated band with molecules has no estimate. It equals templated_share_corrected when the templated label
    is the templated bands and no band estimate is clipped."""
    if not purity["n_labeled"]:
        return None
    kept = 0.0
    for b in TEMPLATED_BANDS:
        e = purity["bands"][b]
        if e["n"]:
            if e["tail_primed_share"] is None:
                return None
            kept += e["n"] * (1 - e["tail_primed_share"])
    return kept / purity["n_labeled"]


def _reference(mol):
    """[molecules, hexamer rate] of the tail reference (A band 0-4) and then of the internal reference (a20 >= 14), the
    classes label_purity takes its rates from; a rate is None for an empty class."""
    a20, hexa = mol.a20.to_numpy(), mol.hexamer.to_numpy(bool)
    out = []
    for m in (a20 <= A_BANDS[0][2], a20 >= 14):
        out += [int(m.sum()), float(hexa[m].mean()) if m.any() else None]
    return out


def _clipped(rate, ref, tail):
    """Whether (rate - ref) / (tail - ref) lies outside [0, 1], so that the tail-primed share is clipped."""
    return None not in (rate, ref, tail) and ref != tail and not 0 <= (rate - ref) / (tail - ref) <= 1


def headline_version(ref, h_tail_lib, h_ref_lib, ref_min=REGION_REF_MIN):
    """Headline version of each region, by the rule design 3.7 fixed before any number: 'region' (the region's own rates)
    where the region has at least ref_min molecules in each reference class and a library-wide rate lies outside the 95%
    interval of the region's rate of that class, 'library' elsewhere. ref has the columns n_ref_tail, n_ref_internal,
    h_tail_lo, h_tail_hi, h_ref_lo and h_ref_hi; a missing interval (NaN) leaves nothing outside."""
    enough = (ref.n_ref_tail >= ref_min) & (ref.n_ref_internal >= ref_min)
    outside = (h_tail_lib < ref.h_tail_lo) | (h_tail_lib > ref.h_tail_hi) | (h_ref_lib < ref.h_ref_lo) | (h_ref_lib > ref.h_ref_hi)
    return np.where(enough & outside, "region", "library").astype(object)


# columns of the counts behind the bootstrap of purity_by_region: all molecules, the A bands, the two reference classes
_TEMPLATED_COLUMNS = [1 + i for i, (name, _, _) in enumerate(A_BANDS) if name in TEMPLATED_BANDS]


def _purity_draws(N, H):
    """[h_tail, h_ref, corrected share with the library-wide rates (those of region 0), corrected share with each region's
    own rates], each [..., regions], from molecule and hexamer counts [..., regions, columns]; NaN where undefined."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = H / N
        tail, ref = rate[..., -2], rate[..., -1]
        out = [tail, ref]
        for t, r in ((tail[..., :1], ref[..., :1]), (tail, ref)):
            d = t - r
            share = np.clip((rate[..., _TEMPLATED_COLUMNS] - r[..., None]) / np.where(d == 0, np.nan, d)[..., None], 0, 1)
            nb = N[..., _TEMPLATED_COLUMNS]
            out.append(np.where(nb > 0, nb * (1 - share), 0.0).sum(axis=-1) / N[..., 0])
    return out


def purity_by_region(mol, n=1000, seed=0, ref_min=REGION_REF_MIN):
    """Stage 29 (design 3.7) on one group's labeled molecules (columns a20, hexamer, label, region, gene, ref, strand, site),
    over all regions ('all') and per region value found in mol. The tail-primed share per A band (evaluate.tail_share) and
    the corrected templated share (band_corrected_share) in two versions: with the reference rates of the whole group
    (library-wide) and with the region's own (regionref); the headline version of each region by headline_version.
    Intervals: gene-cluster bootstrap with n resamples (clusters and draws as share_ci), every rate, the library-wide
    ones included, recomputed in each resample; resamples in which a value is undefined are left out (evaluate.interval).
    Returns the tables purity_reference_by_region, templated_share_corrected and purity_by_region, without the group
    column."""
    a20, hexa = mol.a20.to_numpy(), mol.hexamer.to_numpy(bool)
    regions, reg = np.unique(mol.region.astype(str).to_numpy(object), return_inverse=True)
    names, code = np.unique(gene_clusters(mol), return_inverse=True)
    columns = [np.ones(len(mol), bool)] + [(a20 >= lo) & (a20 <= hi) for _, lo, hi in A_BANDS] + [a20 <= A_BANDS[0][2], a20 >= 14]
    shape = (len(names), len(regions) + 1)
    counts = np.zeros((2, *shape, len(columns)))  # molecules and molecules with a hexamer per cluster; region 0 is all
    for k, m in enumerate(columns):
        cell = code[m] * shape[1] + reg[m] + 1
        counts[0, ..., k] = np.bincount(cell, minlength=shape[0] * shape[1]).reshape(shape)
        counts[1, ..., k] = np.bincount(cell, weights=hexa[m].astype(float), minlength=shape[0] * shape[1]).reshape(shape)
    counts[:, :, 0] = counts[:, :, 1:].sum(axis=2)
    W = resample_counts(len(names), n, seed).astype(float)
    drawn = (W @ counts.transpose(1, 0, 2, 3).reshape(len(names), -1)).reshape(n, 2, shape[1], len(columns))
    ci = [[interval(d[:, j])[:2] for j in range(shape[1])] for d in _purity_draws(drawn[:, 0], drawn[:, 1])]

    _, h_tail, _, h_ref = _reference(mol)
    refs, shares, bands = [], [], []
    for j, region in enumerate(("all", *regions)):
        sub = mol if j == 0 else mol[reg == j - 1]
        n_tail, own_tail, n_ref, own_ref = _reference(sub)
        lib, own = tail_share(sub, h_ref, h_tail), tail_share(sub, own_ref, own_tail)
        refs.append((region, n_tail, n_ref, own_tail, *ci[0][j], own_ref, *ci[1][j]))
        shares.append((region, lib["n_labeled"], lib["templated_share"], band_corrected_share(lib), *ci[2][j],
                       band_corrected_share(own), *ci[3][j]))
        for b, *_ in A_BANDS:
            rate = lib["bands"][b]["hexamer_rate"]
            bands.append((region, b, lib["bands"][b]["n"], rate, lib["bands"][b]["tail_primed_share"],
                          own["bands"][b]["tail_primed_share"], _clipped(rate, h_ref, h_tail), _clipped(rate, own_ref, own_tail)))
    floats = ["h_tail", "h_tail_lo", "h_tail_hi", "h_ref", "h_ref_lo", "h_ref_hi"]
    ref = pd.DataFrame(refs, columns=["region", "n_ref_tail", "n_ref_internal"] + floats).astype(dict.fromkeys(floats, float))
    ref["headline_version"] = headline_version(ref, np.nan if h_tail is None else h_tail, np.nan if h_ref is None else h_ref, ref_min)
    floats = ["share_label", "share_corrected", "lo", "hi", "share_corrected_regionref", "lo_regionref", "hi_regionref"]
    share = pd.DataFrame(shares, columns=["region", "n_labeled"] + floats).astype(dict.fromkeys(floats, float))
    share["headline_version"] = ref.headline_version
    floats = ["hexamer_rate", "tail_primed_share", "tail_primed_share_regionref"]
    band = pd.DataFrame(bands, columns=["region", "a_band", "n"] + floats + ["clipped", "clipped_regionref"]).astype(
        dict.fromkeys(floats, float))
    return ref, share, band


def purity_acceptance(mol, expected, tol=5e-10):
    """Acceptance of stage 29 (design 3.7) for one library: on its labeled molecules of called cells, the all-regions
    corrected share by the class-level formula of evaluate.label_purity (tail_share at the library's own rates, as
    purity_by_region takes them) equals expected, templated_share_corrected of eval/selection.json, to 9 decimals (tol),
    and so does the per-band sum unless a templated band estimate is clipped."""
    _, h_tail, _, h_ref = _reference(mol)
    p = tail_share(mol, h_ref, h_tail)
    class_level, band_sum = p["templated_share_corrected"], band_corrected_share(p)
    clipped = [b for b in TEMPLATED_BANDS if _clipped(p["bands"][b]["hexamer_rate"], h_ref, h_tail)]

    def equal(x):
        return x is not None and abs(x - expected) < tol

    return {"expected": expected, "class_level": class_level, "band_sum": band_sum, "clipped_bands": clipped,
            "class_level_equal": equal(class_level), "band_sum_equal": equal(band_sum),
            "pass": equal(class_level) and (equal(band_sum) or bool(clipped))}


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
