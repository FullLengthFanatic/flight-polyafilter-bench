"""Cell classes from marker genes (stage 25; design of 2026-10-12, section 3.3, decisions D1 and D9).

Every called cell of the unfiltered matrix M0 gets a coarse class: Leiden clusters on the scanpy pipeline of config.CLUSTER,
each cluster named by the panel class (pafbench.markers) with the highest mean score_genes score when that score is above
0 and leads the second by CLASS_MARGIN or more, else "unassigned"; cells inherit the class of their cluster. Stage 26
reuses pipeline (on filtered and thinned matrices), class_scores, name_clusters and the saved model.

Threads. pipeline, class_scores and doublet_scores run with every thread pool that threadpoolctl controls, BLAS and OpenMP,
limited to cluster["blas_threads"] ("Change the maximal number of threads that can be used in thread pools.", docstring of
threadpoolctl.threadpool_limits 3.7.0; user_api None limits all of them). Design 3.0 limits BLAS only. Our observation of
2026-10-15 on the toy library of tests/test_classes.py (900 cells): with BLAS at 1 thread and the OpenMP pools at 8 against
1, the scrublet doublet score of 1 cell differs by 0.0017, and with both at 1 nothing differs. The OpenMP pools of the run
are two libgomp, one of them shipped with scikit-learn (scikit_learn.libs/), whose KNeighborsTransformer scanpy uses for the
neighbour graph of fewer than 8,192 cells with the euclidean metric (scanpy 1.11.5, scanpy/neighbors/__init__.py lines 646
to 651 and 673 to 686), in scrublet and in the pipeline alike.

The scanpy functions, each with the first line of its documentation page (read 2026-10-12, as in
tests/test_scanpy_stack.py, which checks the calls on a toy matrix):
- normalize_total "Normalize counts per cell." https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.normalize_total.html
- log1p "Logarithmize the data matrix." https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.log1p.html
- highly_variable_genes "Annotate highly variable genes"
  https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html
- scale "Scale data to unit variance and zero mean." https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.scale.html;
  it sets "adata.var['mean'] ... Means per gene before scaling." and "adata.var['std'] ... Standard deviations per gene before
  scaling." (scanpy 1.11.5, scanpy/preprocessing/_scale.py, docstring of scale), the mean and sd of the saved model.
- pca "Principal component analysis" https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.pca.html
- neighbors "Compute the nearest neighbors distance matrix and a neighborhood graph of observations"
  https://scanpy.readthedocs.io/en/stable/api/generated/scanpy.pp.neighbors.html
- leiden "Cluster cells into subgroups"; resolution "A parameter value controlling the coarseness of the clustering.";
  random_state "Change the initialization of the optimization."
  https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.leiden.html
- umap "Embed the neighborhood graph using UMAP" https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.umap.html
- score_genes "Score a set of genes"; "The score is the average expression of a set of genes after subtraction by the average
  expression of a reference set of genes. The reference set is randomly sampled from the `gene_pool` for each binned
  expression value." (scanpy/tools/_score_genes.py, docstring)
  https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.score_genes.html
- scrublet "Predict cell doublets using a nearest-neighbor classifier of observed transcriptomes and simulated doublets. Works
  best if the input is a raw (unnormalized) counts matrix"
  https://scanpy.readthedocs.io/en/stable/api/generated/scanpy.pp.scrublet.html
"""
import json
import time
from collections import namedtuple
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from threadpoolctl import threadpool_limits

from . import matrices
from .config import CELLS, CLASS_MARGIN, CLUSTER, MIN_CELLS_GENE, RARE_PCTL
from .markers import AMBIENT_ROD_TRANSCRIPTS, PANEL, ROD_TF

UNASSIGNED = "unassigned"
SEED = 0              # random_state of PCA, neighbour graph, UMAP, score_genes and scrublet; the Leiden seeds are CLUSTER's
FEW_GENES = 2         # a class whose score rests on this many detected panel genes or fewer is marked (3.3, check 1)
WEAK = "two of its genes also mark mature classes (VSX2 bipolar, PAX6 amacrine)"
# "Immune, vascular-associated, and choroidal cell types were not observed" in organoids (Cowan 2020,
# https://pmc.ncbi.nlm.nih.gov/articles/PMC7505495/, design 3.3): a cluster of a library of CELLS named so is flagged
ORGANOID_ABSENT = ("microglia", "vascular")
# "If vascular or microglia cells exist but form no cluster, the report says that the collapse question cannot be asked for
# that class in that library." (3.3, check 6): these classes without a cluster are listed, with their rare candidates
COLLAPSE_CLASSES = ("microglia", "vascular")
# The 13 genes of features.tsv whose names start with "MT-" (MT-ND1 to MT-CYB) are the mitochondrial genes of the
# reference, the same 13 in all three libraries (our observation, 2026-10-15)
MT_PREFIX = "MT-"
# Defaults of scanpy.tl.filter_rank_genes_groups (scanpy 1.11.5, scanpy/tools/_rank_genes_groups.py lines 783 to 785), which
# keeps a gene where "(fraction_in_cluster_matrix > min_in_group_fraction) & (fraction_out_cluster_matrix <
# max_out_group_fraction) & (fold_change_matrix > min_fold_change)" (lines 920 to 922). split_classes applies them to two
# clusters of one class.
MIN_IN_FRACTION, MAX_OUT_FRACTION, MIN_FOLD_CHANGE = 0.25, 0.5, 1

Embedding = namedtuple("Embedding", "norm labels pca umap model")
Classes = namedtuple("Classes", "cells clusters markers table summary embedding")


def normalise(counts, gene_ids, cluster=CLUSTER):
    """AnnData of counts (cells x genes) after normalize_total to cluster["target_sum"] and log1p, all genes, var_names
    gene_ids, obs_names the row numbers."""
    a = ad.AnnData(counts.copy(), obs=pd.DataFrame(index=np.arange(counts.shape[0]).astype(str)),
                   var=pd.DataFrame(index=np.asarray(gene_ids, str)))
    sc.pp.normalize_total(a, target_sum=cluster["target_sum"])
    sc.pp.log1p(a)
    return a


def leiden(b, seed, resolution, cluster=CLUSTER):
    """Leiden labels (int32) on the neighbour graph of b, with the flavour and iterations of cluster and directed=False:
    scanpy 1.11.5 names that call its future default (design 3.0, note of 2026-10-12)."""
    sc.tl.leiden(b, resolution=resolution, random_state=seed, flavor=cluster["leiden_flavor"],
                 n_iterations=cluster["leiden_n_iterations"], directed=False)
    return b.obs["leiden"].to_numpy().astype(np.int32)


def pipeline(counts, gene_ids, seeds=CLUSTER["seeds"][:1], resolutions=(CLUSTER["resolution"],), umap=False, cluster=CLUSTER,
             min_cells=MIN_CELLS_GENE):
    """Steps 1, 2 and 4 of 3.3 on counts (cells x genes, CSR), under cluster["blas_threads"] threads: normalise (all
    genes), then on the genes with a UMI in min_cells cells or more the cluster["n_hvg"] highly variable genes, scale,
    PCA, the neighbour graph and Leiden at every seed and resolution asked for; UMAP when asked. No cell is dropped.

    Returns Embedding: norm (normalise of all genes), labels {(seed, resolution): Leiden labels}, pca (cells x n_pcs), umap
    (cells x 2 or None) and model {genes: column of each variable gene in counts, gene_ids, mean and sd of the scaling,
    loadings: genes x n_pcs}, so that ((norm[:, genes] - mean) / sd) @ loadings gives pca.
    """
    with threadpool_limits(limits=cluster["blas_threads"]):
        norm = normalise(counts, gene_ids, cluster)
        kept = np.flatnonzero(np.asarray((counts > 0).sum(axis=0)).ravel() >= min_cells)
        b = norm[:, kept].copy()
        sc.pp.highly_variable_genes(b, n_top_genes=cluster["n_hvg"])
        hv = kept[b.var.highly_variable.to_numpy()]
        b = b[:, b.var.highly_variable].copy()
        sc.pp.scale(b)
        sc.pp.pca(b, n_comps=cluster["n_pcs"], random_state=SEED)
        sc.pp.neighbors(b, n_neighbors=cluster["k"], n_pcs=cluster["n_pcs"], random_state=SEED)
        labels = {(s, r): leiden(b, s, r, cluster) for r in resolutions for s in seeds}
        if umap:
            sc.tl.umap(b, random_state=SEED)
    model = {"genes": hv.astype(np.int64), "gene_ids": np.asarray(gene_ids, str)[hv], "mean": b.var["mean"].to_numpy(),
             "sd": b.var["std"].to_numpy(), "loadings": b.varm["PCs"]}
    return Embedding(norm, labels, b.obsm["X_pca"], b.obsm["X_umap"] if umap else None, model)


def scored_classes(lib):
    """The panel classes scored in lib: all but the four of 1179 only in a nuclei library."""
    return [c for c in PANEL if lib in c.libraries]


def class_scores(norm, symbols, lib, cluster=CLUSTER):
    """score_genes of every class of scored_classes(lib) on its panel genes in the matrix, and of ROD_TF (column rod_tf), per
    cell of norm (normalise). symbols: the gene symbol of each column. A class without a gene in the matrix scores NaN."""
    column = {s: g for s, g in zip(symbols, norm.var_names)}
    out = {}
    with threadpool_limits(limits=cluster["blas_threads"]):
        for name, genes in [(c.name, c.genes) for c in scored_classes(lib)] + [("rod_tf", ROD_TF)]:
            present = [column[g] for g in genes if g in column]
            if not present:
                out[name] = np.full(norm.n_obs, np.nan)
                continue
            sc.tl.score_genes(norm, present, score_name="_score", random_state=SEED)
            out[name] = norm.obs.pop("_score").to_numpy()
    return pd.DataFrame(out)


def assign(cluster_scores, margin=CLASS_MARGIN):
    """The marker rule of 3.3 step 3 on cluster scores (clusters x classes, NaN for a class not scored): a cluster takes
    the class with the highest score when that score is above 0 and exceeds the second by margin or more, else UNASSIGNED.
    On a tie the first class in column order is class_best. Columns class, score_best, class_second, score_second,
    margin (best minus second) and class_best, indexed like cluster_scores."""
    names = np.asarray(cluster_scores.columns, object)
    v = cluster_scores.to_numpy(np.float64)
    v = np.where(np.isnan(v), -np.inf, v)
    order = np.argsort(-v, axis=1, kind="stable")
    rows = np.arange(len(v))
    best, second = v[rows, order[:, 0]], v[rows, order[:, 1]]
    ok = (best > 0) & (best - second >= margin)
    return pd.DataFrame({"class": np.where(ok, names[order[:, 0]], UNASSIGNED), "score_best": best,
                         "class_second": names[order[:, 1]], "score_second": second, "margin": best - second,
                         "class_best": names[order[:, 0]]}, index=cluster_scores.index)


def name_clusters(scores, labels, margin=CLASS_MARGIN):
    """assign on the cluster scores of labels: the mean cell score of each class (columns of scores) per cluster."""
    return assign(scores.groupby(labels).mean(), margin)


def doublet_scores(counts, cluster=CLUSTER):
    """scanpy.pp.scrublet doublet score of each cell, on the raw counts, seeded (D9, flag b)."""
    with threadpool_limits(limits=cluster["blas_threads"]):
        a = ad.AnnData(counts.copy(), obs=pd.DataFrame(index=np.arange(counts.shape[0]).astype(str)))
        sc.pp.scrublet(a, random_state=SEED)
    return a.obs["doublet_score"].to_numpy()


def _share(counts, symbols, genes):
    """UMIs of genes (symbols) over all UMIs, per cell."""
    wanted = set(genes)
    cols = [i for i, s in enumerate(symbols) if s in wanted]
    return np.asarray(counts[:, cols].sum(axis=1)).ravel() / np.asarray(counts.sum(axis=1)).ravel()


def marker_table(counts, norm, symbols, cell_class, scores, lib):
    """Check 1 of 3.3, one row per scored class and panel gene: in_matrix; fraction of cells with a UMI and mean log
    expression inside and outside the class; ROC AUC of the class score for the class's cells against the rest
    ("Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.",
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html); n_genes_detected, the panel genes
    of the class with a UMI in the library. Inside numbers and AUC are NaN for a class without cells."""
    column = {s: i for i, s in enumerate(symbols)}
    detected = np.asarray((counts > 0).sum(axis=0)).ravel() > 0
    rows = []
    for c in scored_classes(lib):
        inside = cell_class == c.name
        both = inside.any() and not inside.all()
        auc = roc_auc_score(inside, scores[c.name]) if both and np.isfinite(scores[c.name]).all() else np.nan
        n_detected = sum(detected[column[g]] for g in c.genes if g in column)
        for g in c.genes:
            row = {"class": c.name, "gene": g, "in_matrix": g in column, "frac_in": np.nan, "frac_out": np.nan,
                   "mean_in": np.nan, "mean_out": np.nan, "auc_class": auc, "n_genes_detected": n_detected}
            if g in column:
                x = norm.X[:, column[g]].toarray().ravel()
                if inside.any():
                    row |= {"frac_in": (x[inside] > 0).mean(), "mean_in": x[inside].mean()}
                if not inside.all():
                    row |= {"frac_out": (x[~inside] > 0).mean(), "mean_out": x[~inside].mean()}
            rows.append(row)
    return pd.DataFrame(rows).astype({"n_genes_detected": np.int64, "in_matrix": bool})


def read_cell_ranger(analysis_dir, barcodes):
    """Cell Ranger's graph-based cluster of each barcode (outs/analysis/clustering/gene_expression_graphclust/clusters.csv)
    and the log2 fold changes of differential_expression.csv (genes by Feature ID x cluster number)."""
    analysis_dir = Path(analysis_dir)
    cl = pd.read_csv(analysis_dir / "clustering" / "gene_expression_graphclust" / "clusters.csv").set_index("Barcode")
    cr = cl.Cluster.reindex(barcodes)
    if cr.isna().any():
        raise SystemExit(f"{analysis_dir}: {int(cr.isna().sum())} called cells have no Cell Ranger cluster")
    de = pd.read_csv(analysis_dir / "diffexp" / "gene_expression_graphclust" / "differential_expression.csv").set_index(
        "Feature ID")
    lfc = de[[c for c in de.columns if c.endswith(" Log2 fold change")]]
    return cr.to_numpy(np.int32), lfc.set_axis([int(c.split()[1]) for c in lfc.columns], axis=1)


def cell_ranger_check(labels, cell_class, cr, lfc, gene_ids, symbols, lib):
    """Check 3 of 3.3: our clusters against Cell Ranger's graph clusters (contingency, adjusted Rand index, "Rand index
    adjusted for chance.", https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html), the
    most common class of each Cell Ranger cluster and, for the clusters that map to a class, the rank of each of its panel
    genes by log2 fold change in that cluster (1 the highest; ties take the best rank; None when Cell Ranger lists no row
    for the gene)."""
    gene_of = dict(zip(symbols, gene_ids))
    contingency = pd.crosstab(labels, cr)
    out = {"clusters": int(len(np.unique(cr))), "ari": float(adjusted_rand_score(cr, labels)),
           "contingency": {str(k): {str(j): int(n) for j, n in row.items() if n} for k, row in contingency.iterrows()},
           "cr_clusters": [], "panel_gene_ranks": [], "genes_ranked": int(len(lfc))}
    genes = {c.name: c.genes for c in scored_classes(lib)}
    for k in np.unique(cr):
        n = pd.Series(cell_class[cr == k]).value_counts()
        top = sorted(n.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        out["cr_clusters"].append({"cr_cluster": int(k), "cells": int(n.sum()), "class": top[0], "share": top[1] / n.sum()})
        if top[0] == UNASSIGNED:
            continue
        v = lfc[k]
        for g in genes[top[0]]:
            x = v.get(gene_of.get(g))
            out["panel_gene_ranks"].append({"class": top[0], "cr_cluster": int(k), "gene": g,
                                            "log2fc": None if x is None else float(x),
                                            "rank": None if x is None else int((v > x).sum()) + 1})
    return out


def seed_stability(labels, scores, resolution):
    """Check 4 of 3.3. labels: {seed: Leiden labels}; scores: cell scores per class. Pairwise adjusted Rand index of the
    seeds and, per class of the first seed (named by name_clusters at each seed), the lowest Jaccard index of its cells
    against the cells of that class at another seed, with that seed (the first one on a tie)."""
    seeds = sorted(labels)
    pairs = [{"a": a, "b": b, "ari": float(adjusted_rand_score(labels[a], labels[b]))} for a, b in combinations(seeds, 2)]
    named = {s: name_clusters(scores, labels[s])["class"].loc[labels[s]].to_numpy() for s in seeds}
    base = named[seeds[0]]
    per_class = {}
    for c in pd.unique(base):
        a = base == c
        jac = [(a & (named[s] == c)).sum() / (a | (named[s] == c)).sum() for s in seeds[1:]]
        i = int(np.argmin(jac))
        per_class[c] = {"cells": int(a.sum()), "lowest_jaccard": float(jac[i]), "seed": seeds[1:][i]}
    return {"seeds": seeds, "resolution": resolution, "pairwise_ari": pairs, "ari_min": min(p["ari"] for p in pairs),
            "n_clusters": {str(s): int(len(np.unique(labels[s]))) for s in seeds}, "classes": per_class}


def rare_candidates(scores, named, pctl=RARE_PCTL):
    """Check 6 of 3.3, for every class without a cluster: the cells whose highest score is that class's and above its
    pctl percentile over all cells, with the threshold."""
    top = np.asarray(scores.columns)[np.nanargmax(scores.to_numpy(), axis=1)]
    out = {}
    for c in scores.columns:
        if c in set(named) or not np.isfinite(scores[c]).all():
            continue
        threshold = float(np.percentile(scores[c], pctl))
        out[c] = {"cells": int(((top == c) & (scores[c].to_numpy() > threshold)).sum()), "threshold": threshold}
    return out


def split_classes(norm, symbols, labels, named, lib):
    """Classes named on two clusters or more: for each pair of their clusters, the panel genes of the class higher in one
    than in the other by the defaults of scanpy.tl.filter_rank_genes_groups (fraction with a UMI above MIN_IN_FRACTION in
    the one and below MAX_OUT_FRACTION in the other, log2 fold change above MIN_FOLD_CHANGE, computed as scanpy computes it:
    log2((expm1(mean in) + 1e-9) / (expm1(mean out) + 1e-9)), scanpy/tools/_rank_genes_groups.py lines 911 to 914)."""
    column = {s: i for i, s in enumerate(symbols)}
    out = []
    for c in scored_classes(lib):
        clusters = sorted(named.index[named["class"] == c.name])
        if len(clusters) < 2:
            continue
        genes = [g for g in c.genes if g in column]
        x = norm.X[:, [column[g] for g in genes]].toarray()
        frac = {k: (x[labels == k] > 0).mean(axis=0) for k in clusters}
        mean = {k: x[labels == k].mean(axis=0) for k in clusters}
        pairs = []
        for a, b in combinations(clusters, 2):
            lfc = np.log2((np.expm1(mean[a]) + 1e-9) / (np.expm1(mean[b]) + 1e-9))
            higher = [[g for g, fi, fo, f in zip(genes, frac[i], frac[o], s * lfc)
                       if fi > MIN_IN_FRACTION and fo < MAX_OUT_FRACTION and f > MIN_FOLD_CHANGE]
                      for i, o, s in ((a, b, 1), (b, a, -1))]
            pairs.append({"clusters": [int(a), int(b)], "higher_in_first": higher[0], "higher_in_second": higher[1]})
        out.append({"class": c.name, "clusters": [int(k) for k in clusters], "pairs": pairs})
    return out


def flags(lib, clusters, markers):
    """The flags of 3.3: weak classes (the weak mark of the panel, or a score that rests on FEW_GENES detected panel genes or
    fewer), clusters whose name changes with the rod score of ROD_TF, and in a library of CELLS the clusters named
    microglia or vascular."""
    weak = {c.name: [WEAK] for c in PANEL if c.weak and c.name in set(markers["class"])}
    for c, n in markers.groupby("class", sort=False).n_genes_detected.first().items():
        if n <= FEW_GENES:
            weak.setdefault(c, []).append(f"rests on {n} detected panel gene{'' if n == 1 else 's'}")
    renamed = clusters[clusters["class"] != clusters.class_with_rod_tf]
    organoid = clusters[clusters["class"].isin(ORGANOID_ABSENT)] if lib in CELLS else clusters[:0]
    return {"weak_classes": weak,
            "rod_tf_renamed": [{"cluster": int(r.cluster), "class": r["class"], "class_with_rod_tf": r.class_with_rod_tf}
                               for _, r in renamed.iterrows()],
            "organoid_named": [{"cluster": int(r.cluster), "class": r["class"]} for _, r in organoid.iterrows()]}


def classify(lib, counts, barcodes, gene_ids, symbols, cr, lfc, cluster=CLUSTER):
    """Stage 25 on one library: counts is M0 (called cells x genes), barcodes and gene_ids/symbols its rows and columns,
    cr and lfc the Cell Ranger clusters and fold changes (read_cell_ranger). Returns Classes: cells, clusters and markers
    (the tables of the output list of 3.3), table (the class table of the review gate), summary (classes.json without
    inputs) and embedding (pipeline at the seeds of cluster, resolution cluster["resolution"], with UMAP)."""
    seeds, resolution = cluster["seeds"], cluster["resolution"]
    symbols = np.asarray(symbols, str)
    emb = pipeline(counts, gene_ids, seeds, (resolution,), umap=True, cluster=cluster)
    emb.norm.var["symbol"] = symbols
    labels = emb.labels[(seeds[0], resolution)]
    scores = class_scores(emb.norm, symbols, lib, cluster)
    names = [c.name for c in scored_classes(lib)]
    named = name_clusters(scores[names], labels)
    with_tf = name_clusters(scores[names].assign(rod=scores["rod_tf"]), labels)
    cell_class = named["class"].loc[labels].to_numpy()
    umis = np.asarray(counts.sum(axis=1)).ravel()
    cells = pd.DataFrame({"cb": np.asarray(barcodes, str), "umis": umis.astype(np.int64),
                          "genes": np.asarray((counts > 0).sum(axis=1)).ravel().astype(np.int64),
                          "pct_mt": 100 * _share(counts, symbols, [s for s in symbols if s.startswith(MT_PREFIX)]),
                          "cluster": labels, "class": cell_class, "margin": named.margin.loc[labels].to_numpy()})
    for c in names:
        cells[f"score_{c}"] = scores[c].to_numpy()
    cells["rod_tf_score"] = scores["rod_tf"].to_numpy()
    cells["rod_share"] = _share(counts, symbols, AMBIENT_ROD_TRANSCRIPTS)
    cells["doublet_score"] = doublet_scores(counts, cluster)
    cells["cr_cluster"] = cr
    cells["umap1"], cells["umap2"] = emb.umap[:, 0], emb.umap[:, 1]

    by = cells.groupby("cluster")
    clusters = named.rename_axis("cluster").reset_index()
    clusters.insert(1, "n_cells", by.size().to_numpy())
    clusters["class_with_rod_tf"] = with_tf["class"].to_numpy()
    clusters["rod_share_median"] = by.rod_share.median().to_numpy()
    clusters["doublet_score_median"] = by.doublet_score.median().to_numpy()
    markers = marker_table(counts, emb.norm, symbols, cell_class, scores, lib)
    f = flags(lib, clusters, markers)
    splits = split_classes(emb.norm, symbols, labels, named, lib)

    split_of = {k: s["class"] for s in splits for k in s["clusters"]}
    renamed = {r["cluster"] for r in f["rod_tf_renamed"]}
    organoid = {r["cluster"] for r in f["organoid_named"]}
    main_cr = pd.crosstab(labels, cr)
    note = {k: "; ".join([*([f"weak class: {'; '.join(f['weak_classes'][c])}"] if c in f["weak_classes"] else []),
                          *(["named with the full rod score only"] if k in renamed else []),
                          *([f"{c} is not expected in organoids (Cowan 2020)"] if k in organoid else []),
                          *([f"one of several {split_of[k]} clusters (classes.json, split_classes)"] if k in split_of else [])])
            for k, c in zip(clusters.cluster, clusters["class"])}
    table = clusters.drop(columns=["class_with_rod_tf", "rod_share_median", "doublet_score_median"]).assign(
        class_best=named.class_best.to_numpy(), class_with_rod_tf=clusters.class_with_rod_tf,
        rod_tf_renames=clusters.cluster.isin(renamed), score_rod=by["score_rod"].mean().to_numpy(),
        rod_tf_score=by.rod_tf_score.mean().to_numpy(), rod_share_median=clusters.rod_share_median,
        doublet_score_median=clusters.doublet_score_median, pct_mt_median=by.pct_mt.median().to_numpy(),
        umis_median=by.umis.median().to_numpy(), cr_cluster_main=main_cr.idxmax(axis=1).to_numpy(),
        cr_cluster_main_share=(main_cr.max(axis=1) / main_cr.sum(axis=1)).to_numpy(), flags=clusters.cluster.map(note))
    for c in names:
        table[f"score_{c}"] = by[f"score_{c}"].mean().to_numpy()

    per_class = {}
    for c in [*names, UNASSIGNED]:
        sel = cells["class"] == c
        if sel.any():
            per_class[c] = {"clusters": [int(k) for k in clusters.cluster[clusters["class"] == c]], "cells": int(sel.sum()),
                            "rod_share_median": float(cells.rod_share[sel].median()),
                            "doublet_score_median": float(cells.doublet_score[sel].median()),
                            "pct_mt_median": float(cells.pct_mt[sel].median())}
    rare = rare_candidates(scores[names], [c for c in per_class if c != UNASSIGNED])
    summary = {
        "library": lib, "cells": int(counts.shape[0]), "genes": int(counts.shape[1]),
        "genes_clustered": int((np.asarray((counts > 0).sum(axis=0)).ravel() >= MIN_CELLS_GENE).sum()),
        "parameters": {"cluster": cluster, "class_margin": CLASS_MARGIN, "min_cells_gene": MIN_CELLS_GENE,
                       "rare_pctl": RARE_PCTL, "random_state": SEED, "few_genes": FEW_GENES,
                       "split_thresholds": {"min_in_group_fraction": MIN_IN_FRACTION,
                                            "max_out_group_fraction": MAX_OUT_FRACTION, "min_fold_change": MIN_FOLD_CHANGE},
                       "ambient_rod_transcripts": list(AMBIENT_ROD_TRANSCRIPTS), "rod_tf": list(ROD_TF)},
        "seeds": list(seeds), "classes_scored": names, "clusters": int(len(clusters)), "classes": per_class,
        "unassigned": {"cells": int((cell_class == UNASSIGNED).sum()),
                       "clusters": [int(k) for k in clusters.cluster[clusters["class"] == UNASSIGNED]]},
        "cell_ranger": cell_ranger_check(labels, cell_class, cr, lfc, gene_ids, symbols, lib),
        "seed_stability": seed_stability({s: emb.labels[(s, resolution)] for s in seeds}, scores[names], resolution),
        "rare_candidates": rare,
        "collapse_not_askable": [c for c in COLLAPSE_CLASSES if c in rare],
        "panel_genes_missing": list(dict.fromkeys(markers.gene[~markers.in_matrix])),
        **f, "split_classes": splits}
    return Classes(cells, clusters, markers, table, summary, emb)


def marker_plot(path, res, lib):
    """The marker plot of the review gate (3.3, check 7): panel genes by class against clusters labelled with their class;
    dot size the fraction of cells with a UMI, colour the mean log expression (scanpy.pl.dotplot: "Each dot represents two
    values: mean expression within each category (visualized by color) and fraction of cells expressing the `var_name` in
    the category (visualized by the size of the dot)", scanpy/plotting/_dotplot.py, docstring of dotplot)."""
    norm = res.embedding.norm
    column = dict(zip(norm.var.symbol, norm.var_names))
    genes = {c.name: [g for g in c.genes if g in column] for c in scored_classes(lib)}
    sub = norm[:, list(dict.fromkeys(column[g] for v in genes.values() for g in v))].copy()
    label = {k: f"{k}: {c}" for k, c in zip(res.clusters.cluster, res.clusters["class"])}
    sub.obs["cluster"] = pd.Categorical([label[k] for k in res.cells.cluster], categories=list(label.values()))
    fig = sc.pl.dotplot(sub, {c: v for c, v in genes.items() if v}, groupby="cluster", gene_symbols="symbol", dot_min=0.0,
                        dot_max=1.0, title=f"{lib}: panel genes by cluster", show=False, return_fig=True)
    fig.savefig(path, dpi=100)


def write(cells_dir, res, lib):
    """cells.parquet, clusters.parquet, marker_table.parquet, model.npz and classes.json (3.3, outputs), class_table.csv and
    marker_plot.png (review gate) in cells_dir."""
    cells_dir = Path(cells_dir)
    cells_dir.mkdir(parents=True, exist_ok=True)
    res.cells.to_parquet(cells_dir / "cells.parquet", index=False)
    res.clusters[["cluster", "n_cells", "class", "score_best", "class_second", "score_second", "margin", "class_with_rod_tf",
                  "rod_share_median", "doublet_score_median"]].to_parquet(cells_dir / "clusters.parquet", index=False)
    res.markers.to_parquet(cells_dir / "marker_table.parquet", index=False)
    np.savez(cells_dir / "model.npz", **res.embedding.model)
    (cells_dir / "classes.json").write_text(json.dumps(res.summary, indent=1))
    res.table.to_csv(cells_dir / "class_table.csv", index=False)
    marker_plot(cells_dir / "marker_plot.png", res, lib)


def run(lib, matrix_dir, cells_dir, analysis_dir, cluster=CLUSTER):
    """Read M0 with its barcodes and features (stage 24) and Cell Ranger's clusters and fold changes, classify, write.
    Returns the Classes, its summary with the inputs."""
    matrix_dir, analysis_dir = Path(matrix_dir), Path(analysis_dir)
    m0 = matrices.load(lib, "unfiltered", matrix_dir)
    barcodes = (matrix_dir / "barcodes.tsv").read_text().split()
    features = pd.read_csv(matrix_dir / "features.tsv", sep="\t", header=None, dtype=str, keep_default_na=False)
    cr, lfc = read_cell_ranger(analysis_dir, barcodes)
    res = classify(lib, m0, barcodes, features[0].to_numpy(), features[1].to_numpy(), cr, lfc, cluster)
    res.summary["inputs"] = {
        "matrices": str(matrix_dir), "clusters": str(analysis_dir / "clustering" / "gene_expression_graphclust" / "clusters.csv"),
        "differential_expression": str(analysis_dir / "diffexp" / "gene_expression_graphclust" / "differential_expression.csv")}
    write(cells_dir, res, lib)
    return res


def donor_replicate(libs, cells, counts, symbols, cluster=CLUSTER):
    """Check 5 of 3.3 on two libraries with the same features (1417 and 1419): per class, cells and share of the library's
    cells in each, and where both have cells the Pearson correlation of the class-mean log expression over the genes with a
    UMI in the class in either library, with the class means of its panel genes."""
    means, n = [], []
    for frame, m in zip(cells, counts):
        x = normalise(m, np.arange(m.shape[1]).astype(str), cluster).X
        cls = frame["class"].to_numpy()
        means.append({c: np.asarray(x[cls == c].mean(axis=0)).ravel() for c in pd.unique(cls)})
        n.append(len(frame))
    column = {s: i for i, s in enumerate(symbols)}
    panel = {c.name: c.genes for c in PANEL}
    out = {"libraries": list(libs), "cells": n, "classes": []}
    for c in [*[p.name for p in PANEL], UNASSIGNED]:
        k = [int((f["class"] == c).sum()) for f in cells]
        if not any(k):
            continue
        row = {"class": c, "cells": k, "share": [a / b for a, b in zip(k, n)], "pearson_class_mean": None, "genes": None,
               "panel_genes": []}
        if all(k):
            a, b = means[0][c], means[1][c]
            used = (a > 0) | (b > 0)
            row |= {"pearson_class_mean": float(stats.pearsonr(a[used], b[used])[0]), "genes": int(used.sum()),
                    "panel_genes": [{"gene": g, "mean": [float(a[column[g]]), float(b[column[g]])]}
                                    for g in panel.get(c, ()) if g in column]}
        out["classes"].append(row)
    return out
