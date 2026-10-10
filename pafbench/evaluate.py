"""Stage 17: grid metrics against the truth labels, setting selection, gene bootstrap, label-free and spike-in checks.

Read units are member reads within the member cap; molecule units are molecules with at least one such member, and
their reads are the pa reads plus those members. A molecule counts as removed when all its reads are removed
(polyAfilter's effect on a UMI count), or when most are (the molecule-level variant, spec 5.3).
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import sparse

from .config import A_BANDS
from .flags import parse_setting, setting_tag
from .grid import grid
from .labels import label_molecules
from .regions import REGION_NAMES

EXPR = ("low", "mid", "high")
MODULES = ("ERCC", "SIRV_E0", "long_SIRV")
LEVELS = ("read", "molecule", "molecule_majority", "molecule_weighted")
MOL_KINDS = ("all", "a_band", "region", "expr_tertile", "has_ts")
READ_KINDS = MOL_KINDS + ("dist_mode", "spliced")
STRATA = {"a_band": tuple(b[0] for b in A_BANDS), "region": REGION_NAMES, "expr_tertile": EXPR + ("no_gene",),
          "has_ts": ("True", "False"), "spliced": ("True", "False"), "module": MODULES}
SETTING_COLUMNS = ["tag", "covlen", "min_len", "mism", "introns"]
COUNT_COLUMNS = ["tp", "fn", "tn", "fp", "n_amb", "amb_removed"]
METRIC_COLUMNS = ["lib", "cap", "cutoffs"] + SETTING_COLUMNS + ["level", "stratum_kind", "stratum"] + COUNT_COLUMNS + \
    ["sens", "spec", "prec"]
READ_COLUMNS = ["qname", "mol", "dist5_exon", "dist_mode", "spliced"]
MOL_ATTRS = ["label", "a_band", "region", "gene", "has_ts", "expr_tertile"]


@dataclass
class Units:
    reads: pd.DataFrame
    R: np.ndarray
    mols: pd.DataFrame
    K: np.ndarray
    tags: list
    pos: str
    neg: str
    amb: str | None
    missing_reads: int
    dropped_mols: int


def tag_setting(tag):
    c, l, m, mode = tag.split("_")
    return parse_setting(f"{c[1:]},{l[1:]},{m[1:]},{mode}")


def group_sum(groups, n_groups, M, weight=None, chunk=64):
    """Column sums of the rows of M per group, [n_groups, cols]; int32 without weights. Sparse product, chunked over columns."""
    w = np.ones(len(groups), np.int32) if weight is None else np.asarray(weight, float)
    S = sparse.csr_matrix((w, (groups, np.arange(len(groups)))), shape=(n_groups, len(groups)))
    out = np.empty((n_groups, M.shape[1]), w.dtype)
    for j in range(0, M.shape[1], chunk):
        block = M[:, j:j + chunk]
        out[:, j:j + chunk] = S @ (block.view(np.uint8) if block.dtype == bool else block)
    return out


def expr_tertiles(mol):
    """low/mid/high tertile of each molecule's gene among genes ranked by molecule count (pd.qcut on the rank, ties by gene
    id; computed with np.quantile so that a single gene works too); 'no_gene' where gene is '' (intergenic or antisense)."""
    gene = mol.gene.to_numpy(object)
    if not len(gene):
        return gene
    counts = pd.Series(gene[gene != ""]).value_counts().sort_index().sort_values(kind="stable")
    rank = np.arange(1, len(counts) + 1)
    tert = pd.Series(np.asarray(EXPR, object)[np.searchsorted(np.quantile(rank, [1 / 3, 2 / 3]), rank)], index=counts.index)
    return pd.Series(gene).map(tert).fillna("no_gene").to_numpy(object)


def units(mol_labeled, truth_reads, truth_flags, tags, cap, label="label", pos="templated", neg="nontemplated",
          amb="ambiguous", all_labeled=False):
    """Read and molecule units at a member cap. `label` names the label column; molecules whose label is not pos, neg or
    amb (amb None: not pos or neg) are left out. Molecules with a read absent from the primary rows of truth_flags drop out.
    With all_labeled=True, include molecules without a member read (their reads are their pa reads only)."""
    classes = [pos, neg] + ([] if amb is None else [amb])
    mol = mol_labeled[mol_labeled[label].isin(classes)]
    mol = mol.assign(label=mol[label].to_numpy(), expr_tertile=expr_tertiles(mol))
    tr = truth_reads[truth_reads.mol.isin(mol.mol)]
    mem = tr[(tr.role == "member") & (tr.dist5_exon <= cap)]
    if all_labeled:
        reads = pd.concat([tr[tr.role == "pa"], mem], ignore_index=True)
    else:
        reads = pd.concat([tr[(tr.role == "pa") & tr.mol.isin(mem.mol)], mem], ignore_index=True)
    fl = truth_flags[(truth_flags.flag.to_numpy() & 0x900) == 0]
    row = pd.Index(fl.qname).get_indexer(reads.qname)
    missing = row < 0
    bad = reads.mol[missing].unique()
    keep = ~reads.mol.isin(bad).to_numpy()
    reads, row = reads[keep].reset_index(drop=True), row[keep]
    mol = mol[mol.mol.isin(reads.mol)].reset_index(drop=True)
    mi = pd.Index(mol.mol).get_indexer(reads.mol)
    removed = fl[list(tags)].to_numpy(bool)[row]
    K = group_sum(mi, len(mol), removed)
    mol["n_reads"] = np.bincount(mi, minlength=len(mol))
    is_mem = (reads.role == "member").to_numpy()
    rd = reads.loc[is_mem, READ_COLUMNS].reset_index(drop=True)
    rd["mi"] = mi[is_mem]
    rd = pd.concat([rd, mol.loc[rd.mi, MOL_ATTRS].reset_index(drop=True)], axis=1)
    return Units(rd, removed[is_mem], mol, K, list(tags), pos, neg, amb, int(missing.sum()), int(len(bad)))


def relabel(u, cutoffs):
    """The same units with labels recomputed from a20 at other cut-offs."""
    lab = label_molecules(u.mols[["a20"]], cutoffs).label.to_numpy(object)
    return replace(u, mols=u.mols.assign(label=lab), reads=u.reads.assign(label=lab[u.reads.mi.to_numpy()]))


def _colsum(M, w):
    return np.count_nonzero(M, axis=0) if w is None else np.einsum("i,ij->j", w, M)


def confusion(removed, label, weight=None, pos="templated", neg="nontemplated"):
    """(tp, fn, tn, fp, n_amb, amb_removed) per column of removed; labels other than pos and neg count as ambiguous."""
    label = np.asarray(label)
    w = None if weight is None else np.asarray(weight, float)
    out = []
    for m in (label == pos, label == neg, (label != pos) & (label != neg)):
        wm = None if w is None else w[m]
        total = m.sum() if w is None else wm.sum()
        rem = _colsum(removed[m], wm)
        out.append((rem, total - rem, total))
    (tp, fn, _), (fp, tn, _), (amb_rem, _, n_amb) = out
    return tp, fn, tn, fp, np.full(removed.shape[1], n_amb), amb_rem


def rates(tp, fn, tn, fp):
    """(sensitivity, specificity, precision), NaN where the denominator is 0."""
    out = []
    for a, d in ((tp, tp + fn), (tn, tn + fp), (tp, tp + fp)):
        a, d = np.asarray(a, float), np.asarray(d, float)
        out.append(np.divide(a, d, out=np.full(a.shape, np.nan), where=d > 0))
    return tuple(out)


def _level(u, level):
    """(frame, removed [n, tags], weights or None) of one level."""
    if level == "read":
        return u.reads, u.R, None
    n = u.mols.n_reads.to_numpy()[:, None]
    if level == "molecule_majority":
        return u.mols, 2 * u.K > n, None
    return u.mols, u.K == n, (1.0 / n[:, 0] if level == "molecule_weighted" else None)


def _strata(u, frame, kinds):
    for kind in kinds:
        if kind == "all":
            yield kind, "all", slice(None)
            continue
        col = frame[kind].to_numpy() if kind in frame else u.mols[kind].to_numpy()[frame.mi.to_numpy()]
        col = col.astype(str)
        for value in STRATA.get(kind) or sorted(set(col)):
            yield kind, value, col == value


def _table(u, tags, levels, kinds_of):
    j = [u.tags.index(t) for t in tags]
    blocks, keys = [], []
    for level in levels:
        frame, removed, w = _level(u, level)
        removed = removed[:, j]
        label = frame.label.to_numpy()
        for kind, value, m in _strata(u, frame, kinds_of(level)):
            c = confusion(removed[m], label[m], None if w is None else w[m], u.pos, u.neg)
            blocks.append(np.column_stack(c).astype(float))
            keys.append((level, kind, value))
    t = len(tags)
    df = pd.DataFrame(np.vstack(blocks), columns=COUNT_COLUMNS)
    settings = [tag_setting(x) for x in tags]
    df.insert(0, "tag", np.tile(np.asarray(tags, object), len(keys)))
    df.insert(1, "covlen", np.tile([s.covlen for s in settings], len(keys)))
    df.insert(2, "min_len", pd.array(np.tile(np.asarray([s.min_len for s in settings], object), len(keys)), dtype="Int64"))
    df.insert(3, "mism", np.tile([s.mism for s in settings], len(keys)))
    df.insert(4, "introns", np.tile([s.introns for s in settings], len(keys)))
    for i, name in enumerate(("level", "stratum_kind", "stratum")):
        df.insert(5 + i, name, np.repeat(np.asarray([k[i] for k in keys], object), t))
    df["sens"], df["spec"], df["prec"] = rates(df.tp.to_numpy(), df.fn.to_numpy(), df.tn.to_numpy(), df.fp.to_numpy())
    return df


def evaluate(u, lib, cap, cutoffs_name="8/4", levels=LEVELS, kinds=None):
    """Metrics of every tag of u per level and stratum (kinds None: all strata of each level)."""
    df = _table(u, u.tags, levels, lambda level: kinds or (READ_KINDS if level == "read" else MOL_KINDS))
    df.insert(0, "lib", lib)
    df.insert(1, "cap", cap)
    df.insert(2, "cutoffs", cutoffs_name)
    return df[METRIC_COLUMNS]


def spikein_metrics(u, tags):
    """Read and molecule metrics overall and per spike-in module (units built on spike_label)."""
    return _table(u, tags, ("read", "molecule"), lambda level: ("all", "module"))


def _num(x):
    return None if x is None or pd.isna(x) else float(x)


def select_setting(metrics, removed_by_tag, introns, lib="1417", cap=1000, cutoffs="8/4", min_spec=0.95):
    """The selection rule of spec 7 over the grid settings of one mode (remove-all control and off-grid published settings
    excluded): highest molecule-level sensitivity with specificity >= min_spec; ties by fewer removed primary records, then
    by tag. Also returns the sensitivity/specificity Pareto front of all candidates."""
    cand = {setting_tag(s) for s in grid() if s.introns == introns and s.min_len is not None}
    m = metrics[(metrics.lib == lib) & (metrics.cap == cap) & (metrics.cutoffs == cutoffs) & (metrics.level == "molecule")
                & (metrics.stratum_kind == "all") & metrics.tag.isin(cand)]
    n_cand = len(m)
    m = m[m.sens.notna() & m.spec.notna()]
    se, sp = m.sens.to_numpy(float), m.spec.to_numpy(float)
    ge = (se[None, :] >= se[:, None]) & (sp[None, :] >= sp[:, None])
    gt = (se[None, :] > se[:, None]) | (sp[None, :] > sp[:, None])
    front = m[~(ge & gt).any(axis=1)].sort_values(["sens", "spec", "tag"], ascending=[False, False, True])
    ok = m[m.spec >= min_spec]
    best = ok.assign(removed=ok.tag.map(removed_by_tag)).sort_values(["sens", "removed", "tag"], ascending=[False, True, True])
    res = {"tag": None, "sens": None, "spec": None, "prec": None}
    if len(best):
        r = best.iloc[0]
        res = {"tag": r.tag, "sens": _num(r.sens), "spec": _num(r.spec), "prec": _num(r.prec)}
    return res | {"n_candidates": n_cand, "n_qualifying": len(ok),
                  "frontier": [[t, float(a), float(b)] for t, a, b in zip(front.tag, front.sens, front.spec)]}


def gene_clusters(mol):
    """Bootstrap cluster of each molecule: its gene, or the 10 kb locus 'ref:strand:site // 10_000' for molecules without a
    gene (intergenic and antisense)."""
    gene = mol.gene.to_numpy(object)
    locus = (mol.ref.astype(str) + ":" + mol.strand.astype(str) + ":" + (mol.site // 10_000).astype(str)).to_numpy(object)
    return np.where(gene != "", gene, locus)


def resample_counts(n_clusters, n, seed):
    """[n, n_clusters] draw counts of a cluster bootstrap (each resample draws n_clusters clusters with replacement)."""
    return np.random.default_rng(seed).multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=n)


def interval(values):
    """(2.5th, 97.5th percentile, number of finite values) of bootstrap values."""
    v = values[np.isfinite(values)]
    lo, hi = np.percentile(v, [2.5, 97.5]) if v.size else (np.nan, np.nan)
    return float(lo), float(hi), int(v.size)


def bootstrap(u, tags, level="molecule", n=1000, seed=0):
    """Gene-cluster bootstrap of sens/spec/prec: per-cluster confusion counts once, then multinomial cluster draws (clusters
    in sorted order) applied by matrix products. Clusters without a positive or negative unit at this level add nothing to
    the counts and are left out of the draws and of n_clusters."""
    frame, removed, w = _level(u, level)
    removed = removed[:, [u.tags.index(t) for t in tags]]
    cl = gene_clusters(u.mols)
    if level == "read":
        cl = cl[frame.mi.to_numpy()]
    names, code = np.unique(cl, return_inverse=True)
    label = frame.label.to_numpy()
    w = np.ones(len(frame)) if w is None else w
    parts = []
    for m in (label == u.pos, label == u.neg):
        rem = group_sum(code[m], len(names), removed[m], w[m])
        total = np.bincount(code[m], weights=w[m], minlength=len(names))
        parts.append((rem, total[:, None] - rem))
    keep = np.bincount(code[(label == u.pos) | (label == u.neg)], minlength=len(names)) > 0
    (tp, fn), (fp, tn) = ((a[keep], b[keep]) for a, b in parts)
    n_clusters = int(keep.sum())
    W = resample_counts(n_clusters, n, seed).astype(float)
    est = rates(*(x.sum(axis=0) for x in (tp, fn, tn, fp)))
    boot = rates(*(W @ x for x in (tp, fn, tn, fp)))
    rows = []
    for k, tag in enumerate(tags):
        for metric, e, b in zip(("sens", "spec", "prec"), est, boot):
            lo, hi, nb = interval(b[:, k])
            rows.append((tag, metric, float(e[k]), lo, hi, nb, n_clusters))
    return pd.DataFrame(rows, columns=["tag", "metric", "estimate", "lo", "hi", "n_boot", "n_clusters"])


def ambiguous_tail_share(mol_labeled):
    """Spec 7: tail-primed share of the ambiguous molecules from AATAAA/ATTAAA rates, a two-component mixture of the
    templated and non-templated rates: (rate_ambiguous - rate_templated) / (rate_nontemplated - rate_templated), clipped to
    [0, 1]. The estimate is None when a class is empty or the two reference rates are equal."""
    lab = mol_labeled.label.to_numpy()
    hexa = mol_labeled.hexamer.to_numpy(bool)
    names = ("templated", "nontemplated", "ambiguous")
    n = {k: int((lab == k).sum()) for k in names}
    rate = {k: float(hexa[lab == k].mean()) if n[k] else None for k in names}
    t, nt, a = (rate[k] for k in names)
    est = None if None in (t, nt, a) or nt == t else float(np.clip((a - t) / (nt - t), 0.0, 1.0))
    return {"estimate": est} | {f"rate_{k}": rate[k] for k in names} | {f"n_{k}": n[k] for k in names}


def _mean(x):
    return float(x.mean()) if x.size else np.nan


def hexamer_check(u, tags):
    """AATAAA/ATTAAA rate among removed and kept molecules (molecule rule), overall and per label."""
    _, removed, _ = _level(u, "molecule")
    removed = removed[:, [u.tags.index(t) for t in tags]]
    hexa = u.mols.hexamer.to_numpy(bool)
    label = u.mols.label.to_numpy()
    rows = []
    for k, tag in enumerate(tags):
        for name in ("all", u.pos, u.neg) + (() if u.amb is None else (u.amb,)):
            m = np.ones(len(label), bool) if name == "all" else label == name
            r, kept = m & removed[:, k], m & ~removed[:, k]
            rows.append((tag, name, int(r.sum()), _mean(hexa[r]), int(kept.sum()), _mean(hexa[kept])))
    return pd.DataFrame(rows, columns=["tag", "label", "removed_n", "removed_hex_rate", "kept_n", "kept_hex_rate"])
