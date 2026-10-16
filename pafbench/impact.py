"""Impact of filtering on cells and cell classes (stage 26; design of 2026-10-12, section 3.4).

The classes of stage 25 on the unfiltered matrix M0 (cells.parquet) are the fixed labels. Matrices: M0 (kind unfiltered), per
setting M0 - L_tag + G_tag (filtered) and M0 thinned to the filtered cell totals (thinned; THIN_REPLICATES per setting,
THIN_REPLICATES_CHOSEN for each chosen setting). Each matrix goes through analyse_matrix: the stage 25 pipeline from scratch
at every seed of CLUSTER and the three resolutions (steps 5, 6 and 10), the projection onto the unfiltered PCA of stage 25
(model.npz) with a KNN_K vote among the unfiltered cells (step 4), the cross-validated KNN_K classifier of the fixed labels
on its own PCA (step 5), the best-union Jaccard of each class and the name the marker rule of stage 25 gives its cells
(step 7), and for M0 and the filtered matrices the marker genes (step 3). The tables of steps 1 and 2 come from stage 24's
cell_totals.parquet next to the UMIs and genes of each matrix; the collapse call of step 9 from the best-union Jaccards of
all runs (collapse).

Processes. The unfiltered matrix runs first and alone, since it gives the reference cells of the projection, the reference
partitions of the adjusted Rand index and the marker genes; then the other matrices, in a pool of workers forked from a
parent that runs no scanpy code (pool.map_forked; with workers 1 everything runs in the calling process). The reason: scanpy
runs its numba kernels in parallel on numba's OpenMP threading layer (numba.threading_layer() is "omp" in this venv after a
normalize_total, with 32 threads), and a process forked after that ends with "Terminating: fork() called from a process
already using GNU OpenMP, this is unsafe." (our observation of 2026-10-15). Each matrix runs with every thread pool of
threadpoolctl limited to cluster["blas_threads"], as in stage 25, and with numba's threads limited to the same number
(numba.set_num_threads: "Set the number of threads to use for parallel execution.", docstring in numba 0.68.0); on 1417 the
pipeline gives the same labels and model with 32, 4 and 1 numba threads, equal to stage 25's (our observation, 2026-10-15).

Functions of external libraries, with the first line of their documentation:
- sklearn.neighbors.KNeighborsClassifier "Classifier implementing the k-nearest neighbors vote."
  https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html; with uniform weights a tie of
  votes goes to the class first in sorted order, "self.classes_[np.argmax(probabilities, axis=1)]" (scikit-learn 1.9.1,
  sklearn/neighbors/_classification.py line 275, classes_ being the sorted labels).
- sklearn.model_selection.StratifiedKFold "Class-wise stratified K-Fold cross-validator."; "The folds are made by preserving
  the percentage of samples for each class"
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
- sklearn.model_selection.cross_val_predict "Generate cross-validated estimates for each input data point."
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html
- sklearn.metrics.balanced_accuracy_score "It is defined as the average of recall obtained on each class."
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
- sklearn.metrics.adjusted_rand_score "Rand index adjusted for chance."
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
- numpy.random.Generator.multivariate_hypergeometric "Choose ``nsample`` items at random without replacement from a
  collection with ``N`` distinct types." (numpy 2.4.6 docstring)
  https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multivariate_hypergeometric.html
- scanpy.tl.rank_genes_groups "Rank genes for characterizing groups."; "Expects logarithmized data."; "'wilcoxon' uses
  Wilcoxon rank-sum" https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.rank_genes_groups.html. Its fold change is
  log2((expm1(mean in group) + 1e-9) / (expm1(mean in rest) + 1e-9)) (scanpy 1.11.5, scanpy/tools/_rank_genes_groups.py lines
  481 to 486); marker_stats computes the fold changes of step 3 so.
"""
import json
import time
from collections import namedtuple
from functools import partial
from pathlib import Path

import anndata as ad
import numba
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from . import classes, matrices
from .classes import UNASSIGNED
from .config import (CHOSEN_SETTINGS, CLASS_MARGIN, CLUSTER, IMPACT_SETTINGS, JACCARD_DISSOLVED, JACCARD_STABLE, KNN_FOLDS, KNN_K,
                     MARKER_LOSS, THIN_REPLICATES, THIN_REPLICATES_CHOSEN, TOP_GENES)
from .flags import setting_tag
from .pool import map_forked

UNFILTERED = "unfiltered"
KINDS = ("unfiltered", "filtered", "thinned")
TAGS = [setting_tag(s) for s in IMPACT_SETTINGS]
CHOSEN_TAGS = [setting_tag(s) for s in CHOSEN_SETTINGS]
SEED = 0              # entropy of the thinning draws with the setting's bit and the replicate; seed of the folds
# The marker rule keeps a class's name in a run when more than this share of the class's cells sit in clusters it names
# with the class (named_share); our proposal, the design leaves the share open
NAME_KEPT = 0.5
# step 9: the two outcomes and the other reports of a class at a setting and resolution
APART = "cells no longer cluster together"
UNNAMED = "cells cluster together, name lost"
NEITHER = "neither"
DEPTH = "lost at this depth, with or without the filter's choice of reads"
UNCLEAR = "unclear"
# classes without a call
UNSTABLE = "not stable before filtering"
MISSING = "runs missing"
NOT_A_CLASS = "not a class of the marker panel"
NOT_ASKABLE = "no cluster before filtering, the collapse question cannot be asked (stage 25, check 6)"

COLUMNS = {
    "per_cell": ["cb", "class", "tag", "umis0", "umis", "lost", "gained", "genes0", "genes", "intronic0", "intronic",
                 "rec_sense_removed_share", "rec_en_removed_share", "knn_class"],
    "per_class": ["class", "tag", "cells", "umis_median0", "umis_median", "umis_q1", "umis_q3", "umis_q1_0", "umis_q3_0",
                  "genes_median0", "genes_median", "genes_q1", "genes_q3", "genes_q1_0", "genes_q3_0", "umis_lost_share",
                  "umis_gained_share", "intronic0", "intronic", "cells_below_floor"],
    "markers": ["class", "gene", "gene_id", "origin", "tag", "umis0", "umis", "lost_share", "detect0", "detect", "lfc0", "lfc"],
    "knn": ["class", "tag", "kind", "replicate", "retention", "cv_recall", "cv_balanced_accuracy", "cells"],
    "knn_confusion": ["tag", "kind", "replicate", "class_fixed", "class_projected", "cells"],
    "clustering": ["tag", "kind", "replicate", "seed", "resolution", "n_clusters", "ari"],
    "class_jaccard": ["class", "tag", "kind", "replicate", "seed", "resolution", "jaccard", "clusters_in_union", "named_class",
                      "named_share"],
}
TOTALS = ["cb", "tag", "lost", "gained", "umis", "genes", "umis_N", "umis_EN", "rec_E", "rec_N", "rec_EN_gx"]

Job = namedtuple("Job", "kind tag replicate")
# One matrix: cells (umis, genes, knn_class per cell), the rows of clustering, class_jaccard, knn and knn_confusion, markers
# (umis, detect, lfc, lost per row of the marker list; None for a thinned matrix), means (mean class scores of the cells of
# each fixed class) and seconds by step. The unfiltered matrix adds the reference of the projection, its labels at the
# first seed by resolution and the marker list.
Analysis = namedtuple("Analysis", "job cells clustering class_jaccard knn confusion markers means seconds reference labels0 "
                                  "marker_list", defaults=(None, None, None))
Impact = namedtuple("Impact", "summary tables seconds")


def replicates(tag):
    """Thinned replicates of a setting: THIN_REPLICATES_CHOSEN for a chosen setting, else THIN_REPLICATES."""
    return THIN_REPLICATES_CHOSEN if tag in CHOSEN_TAGS else THIN_REPLICATES


def jobs(tags):
    """The matrices of tags: unfiltered, each filtered, then the thinned replicates of each."""
    return ([Job(UNFILTERED, UNFILTERED, 0)] + [Job("filtered", t, 0) for t in tags]
            + [Job("thinned", t, r) for t in tags for r in range(replicates(t))])


def thin_rng(tag, replicate):
    """The generator of a thinning draw, seeded with SEED, the setting's bit (its position in IMPACT_SETTINGS) and the
    replicate, so that a draw does not depend on the other settings of the run."""
    return np.random.default_rng((SEED, TAGS.index(tag), replicate))


def thin(m0, targets, rng):
    """m0 (cells x genes, CSR) with each cell's UMIs drawn without replacement down to its target total, one multivariate
    hypergeometric draw per cell in row order, so the totals are exact; a cell whose target is not below its total is left
    as it is."""
    out = m0.astype(np.int32, copy=True).tocsr()
    totals = np.asarray(out.sum(axis=1)).ravel()
    targets = np.asarray(targets).ravel()
    for i in np.flatnonzero(targets < totals):
        s, e = out.indptr[i], out.indptr[i + 1]
        out.data[s:e] = rng.multivariate_hypergeometric(out.data[s:e], int(targets[i]))
    out.eliminate_zeros()
    return out


def project(norm, model):
    """The cells of norm (classes.normalise of a matrix) on the unfiltered PCA: the variable genes of model, scaled with its
    mean and sd, times its loadings (step 4)."""
    x = norm.X[:, model["genes"]].toarray()
    return ((x - model["mean"]) / model["sd"]) @ model["loadings"]


def knn_vote(reference, labels, query, k=KNN_K):
    """The class of each query cell by a vote of its k nearest reference cells."""
    return KNeighborsClassifier(n_neighbors=k).fit(reference, labels).predict(query)


def cv_predict(pca, labels, k=KNN_K, folds=KNN_FOLDS, seed=SEED):
    """The class of each cell predicted by a k nearest-neighbour classifier fitted on the other folds, folds stratified by
    labels and shuffled with seed."""
    return cross_val_predict(KNeighborsClassifier(n_neighbors=k), pca, labels,
                             cv=StratifiedKFold(folds, shuffle=True, random_state=seed))


def best_union(fixed, labels):
    """Per class of fixed, the Jaccard index between its cells and the best union of clusters of labels (step 7): clusters
    sorted by the share of their cells in the class (then by cluster), the prefix with the highest Jaccard index, the shortest
    on a tie. Exact: with a_k cells of the class among the n_k of cluster k and c cells in the class, a union S has Jaccard
    index sum(a_k) / (c + sum(n_k - a_k)), at least L when sum(a_k - L (n_k - a_k)) >= L c; at the maximum L every cluster
    with a positive term, a share a_k / n_k above L / (1 + L), may join, so all of them form a best union, a prefix of the
    order. Returns class, jaccard and clusters_in_union, by class in sorted order."""
    names, f = np.unique(fixed, return_inverse=True)
    ks, lab = np.unique(labels, return_inverse=True)
    both = np.zeros((len(names), len(ks)), np.int64)
    np.add.at(both, (f, lab), 1)
    size, cells = both.sum(axis=0), both.sum(axis=1)
    rows = []
    for c, a in enumerate(both):
        order = np.lexsort((ks, -a / size))
        jac = np.cumsum(a[order]) / (cells[c] + np.cumsum((size - a)[order]))
        i = int(np.argmax(jac))
        rows.append((names[c], float(jac[i]), i + 1))
    return pd.DataFrame(rows, columns=["class", "jaccard", "clusters_in_union"])


def named(fixed, scores, labels, margin=CLASS_MARGIN):
    """Per class of fixed, what the marker rule of stage 25 (classes.name_clusters on scores) calls its cells in the
    clustering labels: named_share, the share of its cells in clusters named with the class, and named_class, the name held
    by most of its cells (the first in sorted order on a tie)."""
    name = classes.name_clusters(scores, labels, margin)["class"]
    cell_name = name.loc[labels].to_numpy()
    rows = []
    for c in np.unique(fixed):
        mine = cell_name[fixed == c]
        n = pd.Series(mine).value_counts()
        rows.append((c, sorted(n.items(), key=lambda kv: (-kv[1], kv[0]))[0][0], float((mine == c).mean())))
    return pd.DataFrame(rows, columns=["class", "named_class", "named_share"])


def marker_genes(norm, fixed, symbols, lib, n=TOP_GENES):
    """The marker genes of step 3 for every class of fixed but UNASSIGNED: its panel genes in the matrix (origin panel), then
    its n top genes by scanpy's Wilcoxon rank-sum of the class against all other cells on norm (classes.normalise of M0)
    that are not among them (origin ranking). A symbol listed twice in the features takes its last column, as in
    classes.class_scores. Returns class, gene (symbol), gene_id, column and origin."""
    names = [c for c in np.unique(fixed) if c != UNASSIGNED]
    gene_ids = np.asarray(norm.var_names)
    column = {s: i for i, s in enumerate(symbols)}
    a = ad.AnnData(norm.X, obs=pd.DataFrame({"class": pd.Categorical(fixed)}, index=norm.obs_names),
                   var=pd.DataFrame(index=norm.var_names))
    sc.tl.rank_genes_groups(a, "class", groups=names, reference="rest", method="wilcoxon", n_genes=n, use_raw=False)
    ranked = a.uns["rank_genes_groups"]["names"]
    at = {g: i for i, g in enumerate(gene_ids)}
    panel = {c.name: c.genes for c in classes.scored_classes(lib)}
    rows = []
    for c in names:
        own = [(g, column[g]) for g in panel[c] if g in column]
        rows += [(c, g, gene_ids[j], j, "panel") for g, j in own]
        taken = {j for _, j in own}
        rows += [(c, symbols[at[g]], g, at[g], "ranking") for g in ranked[c] if at[g] not in taken]
    return pd.DataFrame(rows, columns=["class", "gene", "gene_id", "column", "origin"])


def marker_stats(counts, norm, lost, fixed, markers):
    """For each row of markers (marker_genes), within the cells of its class: UMIs, the share of the cells with a UMI
    (detect), the log2 fold change of the class against the other cells as scanpy computes it (module docstring) on norm,
    and the UMIs of lost (None: 0)."""
    cols = markers.column.to_numpy()
    x = counts[:, cols].toarray()
    y = norm.X[:, cols].toarray().astype(np.float64)
    gone = lost[:, cols].toarray() if lost is not None else np.zeros_like(x)
    out = np.zeros((len(markers), 4))
    for c, idx in markers.groupby("class", sort=False).indices.items():
        inside = fixed == c
        xi = x[:, idx][inside]
        mean_in, mean_out = y[:, idx][inside].mean(axis=0), y[:, idx][~inside].mean(axis=0)
        out[idx] = np.column_stack([xi.sum(axis=0), (xi > 0).mean(axis=0),
                                    np.log2((np.expm1(mean_in) + 1e-9) / (np.expm1(mean_out) + 1e-9)),
                                    gone[:, idx][inside].sum(axis=0)])
    return pd.DataFrame(out, columns=["umis", "detect", "lfc", "lost"]).astype({"umis": np.int64, "lost": np.int64})


def analyse_matrix(job, counts, lost, fixed, model, gene_ids, symbols, lib, reference=None, reference_labels=None, markers=None,
                   cluster=CLUSTER):
    """Steps 3 to 7 and 10 on one matrix (counts, cells x genes; lost, its L_tag for a filtered matrix). reference: the
    unfiltered cells on the unfiltered PCA (project); reference_labels: {resolution: unfiltered labels at the first seed};
    markers: marker_genes. All three None for the unfiltered matrix, which is its own reference and computes the marker
    genes. Returns Analysis."""
    t0 = time.monotonic()
    seeds, resolutions = cluster["seeds"], (cluster["resolution"], *cluster["resolutions_alt"])
    emb = classes.pipeline(counts, gene_ids, seeds, resolutions, cluster=cluster)
    t1 = time.monotonic()
    scores, _ = classes.class_scores(emb.norm, symbols, lib, cluster)
    t2 = time.monotonic()
    with threadpool_limits(limits=cluster["blas_threads"]):
        coords = project(emb.norm, model)
        own = reference is None
        if own:
            reference = coords
            reference_labels = {r: emb.labels[(seeds[0], r)] for r in resolutions}
        projected = knn_vote(reference, fixed, coords)
        t3 = time.monotonic()
        cv = cv_predict(emb.pca, fixed)
        t4 = time.monotonic()
        if markers is None:
            markers = marker_genes(emb.norm, fixed, symbols, lib)
            marker_list = markers
        else:
            marker_list = None
    key = {"tag": job.tag, "kind": job.kind, "replicate": job.replicate}
    clustering, jaccard = [], []
    for (s, r), lab in emb.labels.items():
        clustering.append(key | {"seed": s, "resolution": r, "n_clusters": int(len(np.unique(lab))),
                                 "ari": float(adjusted_rand_score(reference_labels[r], lab))})
        jaccard.append(best_union(fixed, lab).merge(named(fixed, scores, lab), on="class").assign(**key, seed=s, resolution=r))
    names = np.unique(fixed)
    balanced = float(balanced_accuracy_score(fixed, cv))
    knn = pd.DataFrame([{"class": c, **key, "retention": float((projected[fixed == c] == c).mean()),
                         "cv_recall": float((cv[fixed == c] == c).mean()), "cv_balanced_accuracy": balanced,
                         "cells": int((fixed == c).sum())} for c in names])
    confusion = pd.crosstab(pd.Series(fixed, name="class_fixed"), pd.Series(projected, name="class_projected")).stack()
    confusion = confusion[confusion > 0].rename("cells").reset_index().assign(**key)
    stats = marker_stats(counts, emb.norm, lost, fixed, markers) if job.kind != "thinned" else None
    t5 = time.monotonic()
    cells = pd.DataFrame({"umis": np.asarray(counts.sum(axis=1)).ravel().astype(np.int64),
                          "genes": counts.getnnz(axis=1).astype(np.int64), "knn_class": projected})
    seconds = {"pipeline": t1 - t0, "scores": t2 - t1, "projection": t3 - t2, "cv": t4 - t3, "tables": t5 - t4}
    return Analysis(job, cells, pd.DataFrame(clustering)[COLUMNS["clustering"]],
                    pd.concat(jaccard, ignore_index=True)[COLUMNS["class_jaccard"]], knn[COLUMNS["knn"]],
                    confusion[COLUMNS["knn_confusion"]].astype({"cells": np.int64}), stats, scores.groupby(fixed).mean(),
                    seconds, reference if own else None, reference_labels if own else None, marker_list)


def build(job, m0, load):
    """(the matrix of job, its losses L_tag for a filtered matrix, else None). load(tag): (L_tag, G_tag)."""
    if job.kind == UNFILTERED:
        return m0, None
    lost, gained = load(job.tag)
    f = matrices.filtered(m0, lost, gained)
    if job.kind == "filtered":
        return f, lost
    return thin(m0, np.asarray(f.sum(axis=1)).ravel(), thin_rng(job.tag, job.replicate)), None


_STATE = {}


def _init(state):
    _STATE.clear()
    _STATE.update(state)


def _job(job):
    """analyse_matrix on the matrix of job, from _STATE; the seconds get the time to build the matrix and the total."""
    s = _STATE
    t0 = time.monotonic()
    before = numba.get_num_threads()
    numba.set_num_threads(s["cluster"]["blas_threads"])
    try:
        counts, lost = build(job, s["m0"], s["load"])
        built = time.monotonic() - t0
        res = analyse_matrix(job, counts, lost, s["fixed"], s["model"], s["gene_ids"], s["symbols"], s["lib"], s["reference"],
                             s["reference_labels"], s["markers"], s["cluster"])
    finally:
        numba.set_num_threads(before)
    return res._replace(seconds={"build": built, **res.seconds, "total": time.monotonic() - t0})


def _map(todo, workers, state, fork):
    """_job over todo in job order: in forked workers, or in this process."""
    if fork:
        return map_forked(_job, todo, workers, initializer=_init, initargs=(state,))
    _init(state)
    return [_job(j) for j in todo]


def analyse(lib, m0, load, tags, fixed, model, gene_ids, symbols, workers=1, clusters=None, cluster=CLUSTER):
    """analyse_matrix on every matrix of jobs(tags), the unfiltered one first and alone, the others then in a pool of
    workers (module docstring); with workers 1 all in this process. load(tag): (L_tag, G_tag). clusters: stage 25's
    clusters, which the unfiltered labels at the first seed and cluster["resolution"] must reproduce (else SystemExit: the
    cells folder was made from another matrix or with other code). In that run the marker rule (the panel and CLASS_MARGIN
    of this code) must give every class of fixed to all its cells (else SystemExit: the panel or the margin changed after
    stage 25 ran). Returns [Analysis] in job order."""
    state = {"m0": m0, "load": load, "fixed": fixed, "model": model, "gene_ids": gene_ids, "symbols": symbols, "lib": lib,
             "cluster": cluster, "reference": None, "reference_labels": None, "markers": None}
    first, *rest = jobs(tags)
    (u,) = _map([first], 1, state, workers > 1)
    if clusters is not None and not np.array_equal(u.labels0[cluster["resolution"]], clusters):
        raise SystemExit("the unfiltered matrix does not reproduce the clusters of stage 25 at the first seed and resolution "
                         f"{cluster['resolution']}: the cells folder belongs to another matrix or was made with other code")
    j = u.class_jaccard
    j = j[(j.seed == cluster["seeds"][0]) & (j.resolution == cluster["resolution"])]
    renamed = j[(j.named_class != j["class"]) | (j.named_share != 1.0)]
    if len(renamed):
        raise SystemExit("the marker panel or CLASS_MARGIN differs from the one stage 25 used: on the unfiltered matrix at the "
                         f"first seed and resolution {cluster['resolution']} the marker rule names most cells of "
                         + ", ".join(f"{r['class']} {r.named_class} ({r.named_share:.2f} of them {r['class']})"
                                     for _, r in renamed.iterrows()) + "; rerun stage 25")
    state |= {"reference": u.reference, "reference_labels": u.labels0, "markers": u.marker_list}
    return [u, *_map(rest, workers, state, workers > 1)]


def call(ju, jf, jt, nf, nt):
    """Step 9 for one class at one setting and resolution, from the best-union Jaccard index in the unfiltered seeds (ju),
    the filtered seeds (jf) and every thinned run (jt), and the named share in the filtered seeds (nf) and every thinned run
    (nt). UNSTABLE when an unfiltered seed is at or below JACCARD_STABLE. With every filtered seed at or below
    JACCARD_DISSOLVED: APART when every thinned run stays above JACCARD_STABLE, else DEPTH. With every filtered seed above
    JACCARD_STABLE: UNNAMED when the name is lost (named share at most NAME_KEPT) in every filtered seed and kept in every
    thinned run, NEITHER when it is kept in every filtered seed. UNCLEAR otherwise."""
    if (ju <= JACCARD_STABLE).any():
        return UNSTABLE
    if (jf <= JACCARD_DISSOLVED).all():
        return APART if (jt > JACCARD_STABLE).all() else DEPTH
    if (jf > JACCARD_STABLE).all():
        if (nf <= NAME_KEPT).all() and (nt > NAME_KEPT).all():
            return UNNAMED
        if (nf > NAME_KEPT).all():
            return NEITHER
    return UNCLEAR


def _num(x):
    return None if x is None or np.isnan(x) else float(x)


def _score(means, c):
    """(mean class score of the class's cells for their own class, its lead over the best other class) from a frame of
    scores.groupby(fixed).mean(); (None, None) without one."""
    if means is None or c not in means.index or c not in means.columns:
        return None, None
    row = means.loc[c]
    return _num(row[c]), _num(row[c] - row.drop(c).max())


def _marker_loss(c, tag, markers, means):
    """What the report prints behind a call (step 9): per panel gene of the class the share of its UMIs lost within the
    class, and the class score and margin of the class's cells before and after."""
    m = markers[(markers["class"] == c) & (markers.tag == tag) & (markers.origin == "panel")]
    (s0, m0), (s, m1) = _score(means.get(UNFILTERED), c), _score(means.get(tag), c)
    return {"panel_genes": {g: _num(v) for g, v in zip(m.gene, m.lost_share)}, "score": [s0, s], "margin": [m0, m1]}


def collapse(class_jaccard, expected, markers, means, cluster=CLUSTER):
    """Step 9 at every resolution of class_jaccard, for every class but UNASSIGNED and every tag of expected ({tag: thinned
    replicates}). A call needs every seed of cluster in the unfiltered and in the filtered runs and every replicate at every
    seed in the thinned runs; otherwise the class has no call there (MISSING). markers: the markers table; means: {tag or
    UNFILTERED: scores.groupby(fixed).mean()} (_marker_loss).

    Returns (calls, without). A call holds its numbers: the Jaccard index and named share of every run, the name of the
    filtered runs and the marker loss; at cluster["resolution"] also the call or reason at each other resolution and whether
    the call is the same at both. without: the class and the reason, and for UNSTABLE, listed once per class and resolution,
    the unfiltered Jaccard indices.
    """
    seeds = list(cluster["seeds"])
    calls, without = [], []
    runs = class_jaccard[class_jaccard["class"] != UNASSIGNED]
    for (r, c), j in runs.groupby(["resolution", "class"]):
        u = j[j.kind == UNFILTERED].sort_values("seed")
        if u.seed.tolist() == seeds and (u.jaccard <= JACCARD_STABLE).any():
            without.append({"class": c, "tag": None, "resolution": r, "reason": UNSTABLE,
                            "jaccard_unfiltered": u.jaccard.tolist()})
            continue
        for tag, n in expected.items():
            f = j[(j.kind == "filtered") & (j.tag == tag)].sort_values("seed")
            t = j[(j.kind == "thinned") & (j.tag == tag)].sort_values(["replicate", "seed"])
            if (u.seed.tolist() != seeds or f.seed.tolist() != seeds
                    or list(zip(t.replicate, t.seed)) != [(k, s) for k in range(n) for s in seeds]):
                without.append({"class": c, "tag": tag, "resolution": r, "reason": MISSING,
                                "runs": {"unfiltered": len(u), "filtered": len(f), "thinned": len(t),
                                         "expected": {"unfiltered": len(seeds), "filtered": len(seeds),
                                                      "thinned": n * len(seeds)}}})
                continue
            jt, nt = (t[k].to_numpy().reshape(n, len(seeds)) for k in ("jaccard", "named_share"))
            calls.append({"class": c, "tag": tag, "resolution": r,
                          "call": call(u.jaccard.to_numpy(), f.jaccard.to_numpy(), jt, f.named_share.to_numpy(), nt),
                          "jaccard": {"unfiltered": u.jaccard.tolist(), "filtered": f.jaccard.tolist(), "thinned": jt.tolist()},
                          "named_share": {"unfiltered": u.named_share.tolist(), "filtered": f.named_share.tolist(),
                                          "thinned": nt.tolist()},
                          "named_class_filtered": f.named_class.tolist(), "marker_loss": _marker_loss(c, tag, markers, means)})
    status = {(e["class"], e["tag"], e["resolution"]): e["call"] for e in calls}
    for w in without:
        for tag in (expected if w["tag"] is None else [w["tag"]]):
            status[(w["class"], tag, w["resolution"])] = w["reason"]
    for e in calls:
        if e["resolution"] == cluster["resolution"]:
            e["other_resolutions"] = {str(a): status.get((e["class"], e["tag"], a)) for a in cluster["resolutions_alt"]}
            e["holds_at_both"] = all(v == e["call"] for v in e["other_resolutions"].values())
    return calls, without


def per_cell(cell_totals, barcodes, fixed, results, tags):
    """Step 1: per tag ("unfiltered", then tags) and called cell, the UMIs and genes of the matrix before and after, the
    UMIs lost and gained (stage 24), the intronic share (molecules of re_class N or both over all), the share of the cell's
    sense records (RE E or N with GX) and of its RE E or N records removed, and the class of the projection. Also the
    columns per_class and the acceptance need: intr0 and intr (intronic molecules), ct_umis and ct_genes (cell_totals)."""
    ct = cell_totals.set_index(["tag", "cb"])
    u = ct.loc[UNFILTERED].loc[barcodes]
    mats = {r.job.tag: r.cells for r in results if r.job.kind != "thinned"}
    m0 = mats[UNFILTERED]
    frames = []
    for tag in [UNFILTERED, *tags]:
        t, m = ct.loc[tag].loc[barcodes], mats[tag]
        intr0, intr = (u.umis_N + u.umis_EN).to_numpy(), (t.umis_N + t.umis_EN).to_numpy()
        frames.append(pd.DataFrame({
            "cb": barcodes, "class": fixed, "tag": tag, "umis0": m0.umis.to_numpy(), "umis": m.umis.to_numpy(),
            "lost": t.lost.to_numpy(), "gained": t.gained.to_numpy(), "genes0": m0.genes.to_numpy(), "genes": m.genes.to_numpy(),
            "intronic0": intr0 / u.umis.to_numpy(), "intronic": intr / t.umis.to_numpy(),
            "rec_sense_removed_share": 1 - t.rec_EN_gx.to_numpy() / u.rec_EN_gx.to_numpy(),
            "rec_en_removed_share": 1 - (t.rec_E + t.rec_N).to_numpy() / (u.rec_E + u.rec_N).to_numpy(),
            "knn_class": m.knn_class.to_numpy(), "intr0": intr0, "intr": intr, "ct_umis": t.umis.to_numpy(),
            "ct_genes": t.genes.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def per_class(cells, floor):
    """Step 2 from per_cell, per tag and class: cells; median and quartiles of UMIs and genes before (columns ending in 0)
    and after; pooled shares of UMIs lost and gained and intronic share before and after; cells below floor UMIs after."""
    rows = []
    for tag, frame in cells.groupby("tag", sort=False):
        for c, g in frame.groupby("class"):
            rows.append({"class": c, "tag": tag, "cells": len(g), "umis_median0": g.umis0.median(),
                         "umis_median": g.umis.median(), "umis_q1": g.umis.quantile(0.25), "umis_q3": g.umis.quantile(0.75),
                         "umis_q1_0": g.umis0.quantile(0.25), "umis_q3_0": g.umis0.quantile(0.75),
                         "genes_median0": g.genes0.median(), "genes_median": g.genes.median(),
                         "genes_q1": g.genes.quantile(0.25), "genes_q3": g.genes.quantile(0.75),
                         "genes_q1_0": g.genes0.quantile(0.25), "genes_q3_0": g.genes0.quantile(0.75),
                         "umis_lost_share": g.lost.sum() / g.umis0.sum(),
                         "umis_gained_share": g.gained.sum() / g.umis0.sum(), "intronic0": g.intr0.sum() / g.umis0.sum(),
                         "intronic": g.intr.sum() / g.umis.sum(), "cells_below_floor": int((g.umis < floor).sum())})
    return pd.DataFrame(rows, columns=COLUMNS["per_class"])


def marker_table(results):
    """Step 3: the marker list of the unfiltered matrix with, per filtered matrix, the UMIs within the class before and
    after, the share of them lost, detection and log2 fold change before and after."""
    u = results[0]
    before = u.markers
    frames = []
    for r in results:
        if r.job.kind == "filtered":
            a = r.markers
            frames.append(u.marker_list.drop(columns="column").assign(
                tag=r.job.tag, umis0=before.umis, umis=a.umis, lost_share=a.lost / before.umis, detect0=before.detect,
                detect=a.detect, lfc0=before.lfc, lfc=a.lfc))
    return pd.concat(frames, ignore_index=True)[COLUMNS["markers"]]


def _load(matrix_dir, tag):
    return sp.load_npz(matrix_dir / "lost" / f"{tag}.npz"), sp.load_npz(matrix_dir / "gained" / f"{tag}.npz")


def run(lib, matrix_dir, cells_dir, out_dir, settings=None, workers=1, cluster=CLUSTER):
    """Stage 26 on one library: the matrices of stage 24 in matrix_dir at the tags of its manifest (or those of settings),
    the classes and model of stage 25 in cells_dir; writes the tables of COLUMNS and summary.json in out_dir.

    Stops (SystemExit) when a setting is not in the manifest, when cells.parquet does not list the barcodes of barcodes.tsv
    in their order, when a variable gene of model.npz is not the column of features.tsv it names, and when the unfiltered
    matrix does not reproduce stage 25's clusters or the marker rule its classes (analyse). Returns Impact: summary, the
    tables, and the seconds of every matrix by step (not written, so that a rerun writes the same files).
    """
    matrix_dir, cells_dir, out_dir = Path(matrix_dir), Path(cells_dir), Path(out_dir)
    made = json.loads((matrix_dir / "manifest.json").read_text())["tags"]
    unknown = [t for t in settings or () if t not in made]
    if unknown:
        raise SystemExit(f"{matrix_dir}: {', '.join(unknown)} not in the manifest of stage 24 ({', '.join(made)})")
    tags = [t for t in made if settings is None or t in settings]
    barcodes = (matrix_dir / "barcodes.tsv").read_text().split()
    features = pd.read_csv(matrix_dir / "features.tsv", sep="\t", header=None, dtype=str, keep_default_na=False)
    gene_ids, symbols = features[0].to_numpy(), features[1].to_numpy()
    cells = pd.read_parquet(cells_dir / "cells.parquet", columns=["cb", "cluster", "class"])
    if cells.cb.tolist() != barcodes:
        raise SystemExit(f"{cells_dir / 'cells.parquet'}: its barcodes are not those of {matrix_dir / 'barcodes.tsv'} in that "
                         "order")
    model = dict(np.load(cells_dir / "model.npz"))
    if not np.array_equal(model["gene_ids"], gene_ids[model["genes"]]):
        raise SystemExit(f"{cells_dir / 'model.npz'}: its variable genes are not the columns of {matrix_dir / 'features.tsv'}")
    fixed = cells["class"].to_numpy(dtype=str)
    results = analyse(lib, sp.load_npz(matrix_dir / "unfiltered.npz"), partial(_load, matrix_dir), tags, fixed, model, gene_ids,
                      symbols, workers, cells.cluster.to_numpy(), cluster)

    cell_totals = pd.read_parquet(matrix_dir / "cell_totals.parquet", columns=TOTALS)
    pc = per_cell(cell_totals[cell_totals.tag.isin([UNFILTERED, *tags])], barcodes, fixed, results, tags)
    floor = int(pc.umis0[pc.tag == UNFILTERED].min())
    tables = {"per_cell": pc[COLUMNS["per_cell"]], "per_class": per_class(pc, floor), "markers": marker_table(results)}
    for name, field in (("knn", "knn"), ("knn_confusion", "confusion"), ("clustering", "clustering"),
                        ("class_jaccard", "class_jaccard")):
        tables[name] = pd.concat([getattr(r, field) for r in results], ignore_index=True)[COLUMNS[name]]
    means = {r.job.tag: r.means for r in results if r.job.kind != "thinned"}
    calls, without = collapse(tables["class_jaccard"], {t: replicates(t) for t in tags}, tables["markers"], means, cluster)
    names, n = np.unique(fixed, return_counts=True)
    if UNASSIGNED in names:
        without.append({"class": UNASSIGNED, "tag": None, "resolution": None, "reason": NOT_A_CLASS})
    stage25 = json.loads((cells_dir / "classes.json").read_text())
    without += [{"class": c, "tag": None, "resolution": None, "reason": NOT_ASKABLE} for c in stage25["collapse_not_askable"]]
    mk = tables["markers"]
    lost = mk[mk.lost_share > MARKER_LOSS]
    by_tag = pc.groupby("tag", sort=False)
    sums = cell_totals.groupby("tag").umis.sum()
    acceptance = {
        "umis_sum_per_tag": {t: [int(g.umis.sum()), int(sums[t])] for t, g in by_tag},
        "cells_once_per_tag": all(g.cb.tolist() == barcodes for _, g in by_tag),
        "umis_and_genes_equal_cell_totals": bool((pc.umis == pc.ct_umis).all() and (pc.genes == pc.ct_genes).all()),
        "lost_and_gained_add_up": bool((pc.umis0 - pc.lost + pc.gained == pc.umis).all())}
    acceptance["passed"] = (all(a == b for a, b in acceptance["umis_sum_per_tag"].values())
                            and acceptance["cells_once_per_tag"] and acceptance["umis_and_genes_equal_cell_totals"]
                            and acceptance["lost_and_gained_add_up"])
    kinds = pd.Series([r.job.kind for r in results]).value_counts()
    summary = {
        "library": lib, "cells": len(barcodes), "classes": {c: int(k) for c, k in zip(names, n)}, "settings": tags,
        "parameters": {"cluster": cluster, "knn_k": KNN_K, "knn_folds": KNN_FOLDS, "thin_replicates": THIN_REPLICATES,
                       "thin_replicates_chosen": THIN_REPLICATES_CHOSEN, "chosen_settings": CHOSEN_TAGS, "top_genes": TOP_GENES,
                       "marker_loss": MARKER_LOSS, "jaccard_dissolved": JACCARD_DISSOLVED, "jaccard_stable": JACCARD_STABLE,
                       "name_kept": NAME_KEPT, "class_margin": CLASS_MARGIN, "seed": SEED,
                       "scanpy": classes.SCANPY, "scanpy_n_jobs": sc.settings.n_jobs},
        "inputs": {"matrices": str(matrix_dir), "cells": str(cells_dir)},
        "floor": floor, "matrices": {k: int(kinds.get(k, 0)) for k in KINDS},
        "leiden_runs": len(results) * len(cluster["seeds"]) * (1 + len(cluster["resolutions_alt"])),
        "calls": calls, "without_call": without,
        "marker_loss": [{"class": c, "tag": t, "genes": [{"gene": r.gene, "origin": r.origin, "lost_share": float(r.lost_share)}
                                                        for r in g.itertuples()]}
                        for (c, t), g in lost.groupby(["class", "tag"], sort=False)],
        "acceptance": acceptance}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    seconds = [{"kind": r.job.kind, "tag": r.job.tag, "replicate": r.job.replicate} | r.seconds for r in results]
    return Impact(summary, tables, seconds)
