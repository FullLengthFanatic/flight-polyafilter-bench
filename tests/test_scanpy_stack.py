"""Smoke test of the scanpy stack in the repo venv (design of 2026-10-12, sections 3.0, 3.3 step 2 and 7 step 1): the calls
of stage 25 with the values of config.CLUSTER on a toy matrix with three planted classes. That the same matrix gives the
same PCA, neighbour graph and Leiden labels twice is tested here, not assumed.

The quotes are the wording of the scanpy documentation pages (read 2026-10-12), each with its URL.
"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

from pafbench.config import CLUSTER

N_CELLS, N_GENES, N_CLASSES, N_MARKERS = 600, 3000, 3, 30


def _toy():
    rng = np.random.default_rng(0)
    planted = rng.integers(0, N_CLASSES, N_CELLS)
    lam = np.tile(rng.gamma(0.3, 1.0, size=N_GENES), (N_CELLS, 1))
    for c in range(N_CLASSES):
        lam[planted == c, c * N_MARKERS:(c + 1) * N_MARKERS] += 8.0
    return sp.csr_matrix(rng.poisson(lam).astype(np.int32)), planted


def _leiden(a, seed, resolution):
    # "Cluster cells into subgroups"; resolution: "A parameter value controlling the coarseness of the clustering.";
    # random_state: "Change the initialization of the optimization."
    # https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.leiden.html
    # flavor: "Which package's implementation to use."; n_iterations: "How many iterations of the Leiden clustering
    # algorithm to perform." (same page). The flavour is part of the result, so both come from config.CLUSTER: on the graph
    # of this toy at resolution 2.0 and seed 0, igraph gives 12 clusters and leidenalg 9, adjusted Rand index 0.449 (our
    # observation with scanpy 1.11.5). The values are those scanpy names as its future defaults, in the FutureWarning of
    # its default call: "To achieve the future defaults please pass: flavor="igraph" and n_iterations=2.  directed must
    # also be False to work with igraph's implementation." (scanpy 1.11.5, scanpy/tools/_leiden.py).
    sc.tl.leiden(a, resolution=resolution, random_state=seed, flavor=CLUSTER["leiden_flavor"],
                 n_iterations=CLUSTER["leiden_n_iterations"], directed=False)
    return a.obs["leiden"].to_numpy().astype(str)


def _pipeline(counts):
    a = ad.AnnData(counts.copy(), obs=pd.DataFrame(index=[f"c{i}" for i in range(N_CELLS)]),
                   var=pd.DataFrame(index=[f"g{i}" for i in range(N_GENES)]))
    # "Normalize counts per cell." https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.normalize_total.html
    sc.pp.normalize_total(a, target_sum=CLUSTER["target_sum"])
    # "Logarithmize the data matrix." https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.log1p.html
    sc.pp.log1p(a)
    # "Annotate highly variable genes" https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html
    sc.pp.highly_variable_genes(a, n_top_genes=CLUSTER["n_hvg"])
    b = a[:, a.var.highly_variable].copy()
    # "Scale data to unit variance and zero mean." https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.scale.html
    sc.pp.scale(b)
    # "Principal component analysis"; random_state: "Change to use different initial states for the optimization."
    # https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.pca.html
    sc.pp.pca(b, n_comps=CLUSTER["n_pcs"], random_state=0)
    # "Compute the nearest neighbors distance matrix and a neighborhood graph of observations"; random_state: "A numpy
    # random seed." https://scanpy.readthedocs.io/en/stable/api/generated/scanpy.pp.neighbors.html
    sc.pp.neighbors(b, n_neighbors=CLUSTER["k"], n_pcs=CLUSTER["n_pcs"], random_state=0)
    labels = _leiden(b, CLUSTER["seeds"][0], CLUSTER["resolution"])
    # "Embed the neighborhood graph using UMAP" https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.umap.html
    sc.tl.umap(b, random_state=0)
    # "Score a set of genes"; random_state: "The random seed for sampling."
    # https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.score_genes.html
    for c in range(N_CLASSES):
        sc.tl.score_genes(a, [f"g{i}" for i in range(c * N_MARKERS, (c + 1) * N_MARKERS)], score_name=f"s{c}", random_state=0)
    return b, labels, a.obs[[f"s{c}" for c in range(N_CLASSES)]].to_numpy()


@pytest.fixture(scope="module")
def runs():
    counts, planted = _toy()
    return planted, _pipeline(counts), _pipeline(counts)


def test_same_matrix_twice_gives_the_same_pca_graph_labels_and_scores(runs):
    _, (b1, labels1, scores1), (b2, labels2, scores2) = runs
    assert b1.obsm["X_pca"].shape == (N_CELLS, CLUSTER["n_pcs"])
    assert np.array_equal(b1.obsm["X_pca"], b2.obsm["X_pca"])
    for key in ("connectivities", "distances"):
        assert b1.obsp[key].nnz > 0 and (b1.obsp[key] != b2.obsp[key]).nnz == 0, key
    assert np.array_equal(labels1, labels2)
    assert np.array_equal(scores1, scores2)
    assert b1.obsm["X_umap"].shape == (N_CELLS, 2) and np.isfinite(b1.obsm["X_umap"]).all()


def test_planted_classes_are_recovered(runs):
    planted, (_, labels, scores), _ = runs
    assert len(set(labels)) == N_CLASSES
    assert adjusted_rand_score(planted, labels) == 1.0
    assert np.array_equal(scores.argmax(axis=1), planted)


def test_leiden_seed_is_honoured(runs):
    # at the default resolution every seed finds the three planted classes; at the higher one the partition depends on it
    _, (b, _, _), _ = runs
    resolution = CLUSTER["resolutions_alt"][1]
    first = _leiden(b, CLUSTER["seeds"][0], resolution)
    other = _leiden(b, CLUSTER["seeds"][1], resolution)
    assert np.array_equal(first, _leiden(b, CLUSTER["seeds"][0], resolution))
    assert adjusted_rand_score(first, other) < 1.0
