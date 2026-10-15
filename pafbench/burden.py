"""Stage 18: internal-priming burden (templated share of labeled molecules), same-site and same-gene comparisons, selection bias.
Stage 29: the templated share corrected for label purity, by region.
Stage 31: sites matched between two groups within their shared genes (match_sites onwards; design of 2026-10-12, section 3.9).

Descriptive only (spec 8): tissue, maturity, cell-type mix and the cell/nucleus difference are confounded.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import A_BANDS, DIST_END_BINS, DOWNSAMPLE_DRAWS, LABEL_CUTOFFS, REGION_REF_MIN, SITE_TOL
from .evaluate import gene_clusters, interval, resample_counts, tail_share
from .polyasite import BINS, NULL_LABEL, distance_bin, nearest_signed
from .regions import REGION_NAMES

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
    outside = ((h_tail_lib < ref.h_tail_lo) | (h_tail_lib > ref.h_tail_hi) | (h_ref_lib < ref.h_ref_lo)
               | (h_ref_lib > ref.h_ref_hi))
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


def purity_by_region(mol, n=1000, seed=0):
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
    ref["headline_version"] = headline_version(ref, np.nan if h_tail is None else h_tail, np.nan if h_ref is None else h_ref)
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


SITE_FIELDS = ["label", "a_band", "a20", "down20", "up40", "hexamer", "region", "gene", "genes"]
FEATURES = ["label", "a_band", "a20", "down20", "hexamer", "region", "dist_end", "dist_mode"]
SETS = ("shared", "first_only", "second_only")
_SIZES = tuple(f"{lo}-{hi}" for lo, hi in zip((0,) + tuple(b + 1 for b in DIST_END_BINS[:-1]), DIST_END_BINS)) + (
    f">{DIST_END_BINS[-1]}",)
END_BINS = _SIZES + tuple(f"past {name}" for name in _SIZES) + ("no gene",)


def gene_ends(gtf):
    """{gene_id: sorted unique int64 ndarray} of the 3' ends of the gene's transcripts (GTF transcript lines) in the
    priming_site convention of spikein.annotated_ends: the GTF end on plus, the GTF start - 1 on minus."""
    ends = defaultdict(list)
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if f[2] != "transcript":
                continue
            gid = f[8].split('gene_id "', 1)[1].split('"', 1)[0]
            ends[gid].append(int(f[4]) if f[6] == "+" else int(f[3]) - 1)
    return {g: np.unique(np.asarray(v, np.int64)) for g, v in ends.items()}


def _exon_coord(starts, ends, pos):
    """Collapsed-exon bases before each genomic boundary in pos (exons sorted and disjoint)."""
    j = np.searchsorted(starts, pos, side="right") - 1
    k = np.maximum(j, 0)
    length = ends - starts
    before = np.concatenate([[0], np.cumsum(length)])
    return np.where(j >= 0, before[k] + np.minimum(pos - starts[k], length[k]), 0)


def end_distances(sites, exons, ends):
    """(dist_end, dist_mode) per row of sites (strand '+'/'-', site, genes ';'-separated): the signed distance to the nearest
    3' end of a gene of the site (gene_ends), > 0 when that end lies downstream of the site in transcript sense. Per gene, mode
    'exon' when the last cDNA base (site - 1 on plus, site on minus) lies in a collapsed exon of the gene
    (exons.load_collapsed_exons, the union of the exons of all its transcripts): d counts the collapsed-exon bases between the
    site and the end. Mode 'genome' otherwise: d counts genomic bases. Over the genes the smallest |d| wins, on equal |d| the
    positive one (polyasite.nearest_signed's rule within a gene), then the first gene of genes. Rows without a gene (''): NaN
    and ''."""
    genes = sites.genes.str.split(";").to_numpy(object)
    row = np.repeat(np.arange(len(sites)), [len(g) for g in genes])
    gene = np.asarray([g for gs in genes for g in gs], object)
    strand, pos = sites.strand.to_numpy(object)[row], sites.site.to_numpy(np.int64)[row]
    dist = np.full(len(row), np.nan)
    mode = np.full(len(row), "", object)
    for (g, strd), idx in pd.DataFrame({"gene": gene, "strand": strand}).groupby(["gene", "strand"], sort=False).indices.items():
        if g == "":
            continue
        plus = strd == "+"
        (s, e), x, g_ends = exons[g], pos[idx], ends[g]
        last = x - plus
        k = np.searchsorted(s, last, side="right") - 1
        exonic = (k >= 0) & (last < e[np.maximum(k, 0)])
        along = nearest_signed(_exon_coord(s, e, x), plus, np.unique(_exon_coord(s, e, g_ends)))
        dist[idx] = np.where(exonic, along, nearest_signed(x, plus, g_ends))
        mode[idx] = np.where(exonic, "exon", "genome")
    order = np.lexsort((np.arange(len(row)), -dist, np.abs(dist), row))
    best = order[np.flatnonzero(np.diff(row[order], prepend=-1))]
    return dist[best], mode[best]


def end_bin(d):
    """END_BINS name of each signed distance to the annotated end: |d| binned at DIST_END_BINS, prefixed 'past ' when d < 0
    (the site lies downstream of its nearest end); NaN is 'no gene'."""
    d = np.asarray(d, np.float64)
    k = np.searchsorted(np.asarray(DIST_END_BINS, np.float64), np.abs(np.nan_to_num(d)), side="left")
    names = np.asarray(END_BINS, object)
    out = np.where(d < 0, names[k + len(_SIZES)], names[k])
    out[np.isnan(d)] = END_BINS[-1]
    return out


def site_table(mol, exons, ends):
    """One row per site (ref, strand, site) of labeled molecules, sorted: n (molecules) and the SITE_FIELDS of its first
    molecule (in the labeled molecules of called cells each takes one value per site, our observation in scripts/31), then
    dist_end and dist_mode (end_distances over its genes)."""
    g = mol.groupby(["ref", "strand", "site"], sort=True)
    t = g[SITE_FIELDS].first()
    t.insert(0, "n", g.size())
    t = t.reset_index()
    dist, mode = end_distances(t, exons, ends)
    return t.assign(dist_end=dist, dist_mode=mode)


def usage(sites):
    """Share of each site's molecules (n) among the molecules of its gene in the same site table; NaN without a gene."""
    u = sites.n.to_numpy(np.float64) / sites.groupby("gene").n.transform("sum").to_numpy(np.float64)
    u[sites.gene.to_numpy() == ""] = np.nan
    return u


def shared_genes(a, b, min_n=20):
    """Sorted genes (not '') with at least min_n labeled molecules in both groups: the genes that same_gene correlates."""
    na, nb = (m.gene[m.gene != ""].value_counts() for m in (a, b))
    return sorted(na[na >= min_n].index.intersection(nb[nb >= min_n].index))


def _coords(a, b):
    """One int64 per site of a and of b: the (ref, strand) key of their union times 2**40, plus site (same_site's layout)."""
    keys = {k: i for i, k in enumerate(sorted(set(zip(a.ref, a.strand)) | set(zip(b.ref, b.strand))))}

    def coord(s):
        return np.fromiter((keys[k] for k in zip(s.ref, s.strand)), np.int64, len(s)) * (1 << 40) + s.site.to_numpy(np.int64)

    return coord(a), coord(b)


def _nearest(ca, cb):
    """Index into cb of the nearest coordinate of each one of ca (equal distance: the lower coordinate, as same_site) and its
    distance; index -1 when cb is empty."""
    order = np.argsort(cb, kind="stable")
    far = np.int64(1) << 62
    c = np.concatenate([[-far], cb[order], [far]])
    idx = np.concatenate([[-1], order, [-1]])
    i = np.searchsorted(c, ca)
    dl, dr = ca - c[i - 1], c[i] - ca
    return idx[np.where(dl <= dr, i - 1, i)], np.minimum(dl, dr)


def _mutual(ca, cb, tol):
    jb, d = _nearest(ca, cb)
    ja, _ = _nearest(cb, ca)
    ia = np.flatnonzero((d <= tol) & (jb >= 0))
    ia = ia[ja[jb[ia]] == ia]
    return ia.astype(np.int64), jb[ia]


def _near(ca, cb):
    """Number of coordinates of ca, and of cb, within SITE_TOL nt of one of the other."""
    return int((_nearest(ca, cb)[1] <= SITE_TOL).sum()), int((_nearest(cb, ca)[1] <= SITE_TOL).sum())


def match_sites(a, b, tol=SITE_TOL):
    """Sites of a and b (ref, strand, site) that are each other's nearest on the same ref and strand (equal distance: the
    lower coordinate) and at most tol nt apart, as positional indices (ia ascending, ib). match_sites(b, a) gives the same
    pairs; a site that is the nearest of two sites of the other table pairs with one of them at most."""
    return _mutual(*_coords(a, b), tol)


def site_sets(sa, sb, genes):
    """The sites of genes in two site tables (site_table) in three sets: shared (the match_sites pairs), first_only and
    second_only. One row per pair or unmatched site, sorted by ref, strand and position: gene and FEATURES of the first
    table's site on shared rows, of the site itself on the others; n and usage (usage within the gene in each table) of each
    side, 0 on the side without a site, where site_a or site_b is <NA>."""
    a = sa[sa.gene.isin(genes)].reset_index(drop=True)
    b = sb[sb.gene.isin(genes)].reset_index(drop=True)
    ua, ub = usage(a), usage(b)
    ia, ib = match_sites(a, b)
    oa, ob = np.setdiff1d(np.arange(len(a)), ia), np.setdiff1d(np.arange(len(b)), ib)

    def block(name, t, k, own_usage, other_site, other_n, other_usage, first):
        d = t.iloc[k].reset_index(drop=True)
        own, other = (d.site.to_numpy(), d.n.to_numpy(), own_usage[k]), (other_site, other_n, other_usage)
        (site_a, n_a, u_a), (site_b, n_b, u_b) = (own, other) if first else (other, own)
        return pd.DataFrame({"gene": d.gene, "ref": d.ref, "strand": d.strand, "site_a": pd.array(site_a, "Int64"),
                             "site_b": pd.array(site_b, "Int64"), "set": name, "n_a": n_a, "n_b": n_b, "usage_a": u_a,
                             "usage_b": u_b, **{c: d[c] for c in FEATURES}})

    rows = pd.concat([block("shared", a, ia, ua, b.site.to_numpy()[ib], b.n.to_numpy()[ib], ub[ib], True),
                      block("first_only", a, oa, ua, np.full(len(oa), None), 0, 0.0, True),
                      block("second_only", b, ob, ub, np.full(len(ob), None), 0, 0.0, False)], ignore_index=True)
    rows["n_a"], rows["n_b"] = rows.n_a.astype(np.int64), rows.n_b.astype(np.int64)
    pos = rows.site_a.fillna(rows.site_b)
    return rows.assign(_pos=pos).sort_values(["ref", "strand", "_pos"]).drop(columns="_pos").reset_index(drop=True)


def _onehot(values, bins):
    return np.asarray(values, object)[:, None] == np.asarray(bins, object)[None, :]


def set_summary(rows):
    """Sites and molecules (n_a + n_b, and molecules_a and molecules_b of each group) per set (SETS), feature and bin of
    site_sets rows, every bin kept: label, a_band, a_pos (bin k: an A at position k of the 20 nt after the site, down20),
    hexamer, region and dist_bin (end_bin)."""
    n_a, n_b = rows.n_a.to_numpy(np.int64), rows.n_b.to_numpy(np.int64)
    a_at = np.frombuffer("".join(rows.down20.str.ljust(20, "N").str[:20]).encode(), np.uint8).reshape(-1, 20) == ord("A")
    features = {"label": (LABELS, _onehot(rows.label, LABELS)),
                "a_band": (tuple(name for name, _, _ in A_BANDS), _onehot(rows.a_band, [name for name, _, _ in A_BANDS])),
                "a_pos": (tuple(str(k) for k in range(1, 21)), a_at),
                "hexamer": (("True", "False"), _onehot(rows.hexamer.astype(str), ["True", "False"])),
                "region": (REGION_NAMES, _onehot(rows.region, REGION_NAMES)),
                "dist_bin": (END_BINS, _onehot(end_bin(rows.dist_end), END_BINS))}
    out = []
    for name in SETS:
        m = (rows.set == name).to_numpy()
        for feature, (bins, hit) in features.items():
            h = hit[m]
            out += [(name, feature, b, int(n), int(x + y), int(x), int(y))
                    for b, n, x, y in zip(bins, h.sum(axis=0), n_a[m] @ h, n_b[m] @ h)]
    return pd.DataFrame(out, columns=["set", "feature", "bin", "sites", "molecules", "molecules_a", "molecules_b"])


def _quartiles(x):
    q = [float(v) for v in np.quantile(x, [0.25, 0.5, 0.75])] if len(x) else [None] * 3
    return {"n": len(x), "q25": q[0], "median": q[1], "q75": q[2]}


def usage_comparison(rows):
    """Spearman correlation of usage_a and usage_b over the shared rows of site_sets, and the quartiles of
    log2(usage_a / usage_b) over all of them, by label and by region (every label and region, n 0 and None without rows)."""
    s = rows[rows.set == "shared"]
    r = np.log2(s.usage_a.to_numpy() / s.usage_b.to_numpy())
    rho, p = spearmanr(s.usage_a, s.usage_b) if len(s) > 1 else (np.nan, np.nan)
    label, region = s.label.to_numpy(), s.region.to_numpy()
    return {"n": len(s), "spearman_rho": None if np.isnan(rho) else float(rho), "p_value": None if np.isnan(p) else float(p),
            "log2_ratio": {"all": _quartiles(r), "label": {x: _quartiles(r[label == x]) for x in LABELS},
                           "region": {x: _quartiles(r[region == x]) for x in REGION_NAMES}}}


def set_shares(c):
    """Shares of the shared sites among all sites of the three sets ('all'), among the first group's sites ('first') and
    among the second group's ('second'); c holds shared, first_only and second_only (numbers or columns)."""
    s, f, o = (np.asarray(c[k], np.float64) for k in SETS)
    return {"all": s / (s + f + o), "first": s / (s + f), "second": s / (s + o)}


def set_counts(rows):
    """Sites and molecules of each group per set of site_sets rows, and their set_shares. near (first_only and second_only):
    the sites of the set within SITE_TOL nt of a site of the other group, left unpaired because the match is one to one."""
    out = {}
    for name in SETS:
        m = rows.set == name
        out[name] = {"sites": int(m.sum()), "molecules_first": int(rows.n_a[m].sum()), "molecules_second": int(rows.n_b[m].sum())}
    near = _near(*_coords(*(rows[rows[c].notna()].assign(site=rows[c]) for c in ("site_a", "site_b"))))
    for name, n in zip(SETS[1:], near):
        out[name]["near"] = n - out["shared"]["sites"]
    return out | {"shares": {k: float(v) for k, v in set_shares({k: out[k]["sites"] for k in SETS}).items()}}


def downsampling_summary(draws):
    """Mean and 95% range (evaluate.interval) over the draws of downsample_sets of each set size, of the unpaired sites near
    the other group, of set_shares and, under usage, of spearman_rho and of the median log2 ratio (all, by label, by
    region); over the draws with a value, None without any."""
    def stats(v):
        v = np.asarray(v, np.float64)
        lo, hi, n = interval(v)
        return {"mean": float(v[np.isfinite(v)].mean()), "lo": lo, "hi": hi} if n else {"mean": None, "lo": None, "hi": None}

    cols = {k: draws[k] for k in SETS + ("first_only_near", "second_only_near")} | {
        f"share_{k}": v for k, v in set_shares(draws).items()}
    return {"draws": len(draws), "molecules": int(draws.molecules.iloc[0]), **{k: stats(v) for k, v in cols.items()},
            "usage": {"spearman_rho": stats(draws.spearman_rho),
                      "log2_median": {"all": stats(draws.median_all), "label": {x: stats(draws[f"median_{x}"]) for x in LABELS},
                                      "region": {x: stats(draws[f"median_{x}"]) for x in REGION_NAMES}}}}


def downsample_sets(sa, sb, genes, draws=DOWNSAMPLE_DRAWS, seed=0):
    """Set sizes of site_sets after drawing, in each gene of genes, as many of the second table's molecules as the first table
    has there (all of them where it has fewer), without replacement; draws draws from numpy.random.default_rng(seed). The
    second table's sites of a draw are those with a drawn molecule. One row per draw: draw, molecules (drawn), shared,
    first_only, second_only, first_only_near and second_only_near (set_counts' near), then usage_comparison of the draw, the
    second table's usage taken over the drawn molecules: spearman_rho, median_all (median log2 ratio) and median_<x> for
    every label and region x (NaN without a value)."""
    a = sa[sa.gene.isin(genes)].reset_index(drop=True)
    b = sb[sb.gene.isin(genes)].reset_index(drop=True)
    ca, cb = _coords(a, b)
    ua, label, region = usage(a), a.label.to_numpy(object), a.region.to_numpy(object)
    names, code = np.unique(b.gene.to_numpy(object), return_inverse=True)
    target = a.groupby("gene").n.sum().reindex(names, fill_value=0).to_numpy(np.int64)
    site = np.repeat(np.arange(len(b)), b.n.to_numpy(np.int64))
    gene = code[site]
    rng = np.random.default_rng(seed)
    out = []
    for k in range(draws):
        order = np.lexsort((rng.random(len(site)), gene))
        g = gene[order]
        drawn = order[np.arange(len(g)) - np.searchsorted(g, g) < target[g]]
        n = np.bincount(site[drawn], minlength=len(b))
        kept = np.flatnonzero(n)
        ia, jb = _mutual(ca, cb[kept], SITE_TOL)
        ib = kept[jb]
        near_a, near_b = _near(ca, cb[kept])
        ub = n / np.bincount(code, weights=n)[code]
        u = usage_comparison(pd.DataFrame({"set": "shared", "usage_a": ua[ia], "usage_b": ub[ib], "label": label[ia],
                                           "region": region[ia]}))
        r = u["log2_ratio"]
        out.append((k, len(drawn), len(ia), len(a) - len(ia), len(kept) - len(ia), near_a - len(ia), near_b - len(ia),
                    u["spearman_rho"], r["all"]["median"], *(r["label"][x]["median"] for x in LABELS),
                    *(r["region"][x]["median"] for x in REGION_NAMES)))
    return pd.DataFrame(out, columns=["draw", "molecules", "shared", "first_only", "second_only", "first_only_near",
                                      "second_only_near", "spearman_rho", "median_all",
                                      *(f"median_{x}" for x in LABELS + REGION_NAMES)])


def same_site_check(sa, sb, expected):
    """Acceptance of stage 31: the one-way nearest match of same_site redone on two site tables (site_table, no gene filter)
    against expected, the rows of one pair of burden/same_site.parquet (label_a, label_b, n). It passes when the label-pair
    counts are equal, and with them the number of sites of the first table and of matched ones."""
    jb, d = _nearest(*_coords(sa, sb))
    label_b = np.where((d <= SITE_TOL) & (jb >= 0), sb.label.to_numpy(object)[np.maximum(jb, 0)], "none")
    got = pd.DataFrame({"label_a": sa.label.to_numpy(object), "label_b": label_b}).groupby(["label_a", "label_b"]).size()
    want = expected.groupby(["label_a", "label_b"]).n.sum()
    want = want[want > 0]

    def counts(c):
        return int(c.sum()), int(c[c.index.get_level_values("label_b") != "none"].sum())

    (sites, matched), (e_sites, e_matched) = counts(got), counts(want)
    equal = {k: int(v) for k, v in got.items()} == {k: int(v) for k, v in want.items()}
    return {"sites_a": sites, "matched": matched, "expected_sites_a": e_sites, "expected_matched": e_matched,
            "label_pairs_equal": equal, "pass": equal and (sites, matched) == (e_sites, e_matched)}
