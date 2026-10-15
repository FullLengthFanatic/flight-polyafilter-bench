"""Stage 25: cell classes from marker genes (pafbench.classes; design of 2026-10-12, section 3.3, decisions D1 and D9).

The toy library has 900 cells in seven planted groups with Poisson counts: rod, cone and bipolar cells with the genes of
their panel class; two macroglia groups, one with RLBP1 and SLC1A3 and one with GFAP and SPARCL1, both with GLUL and CLU;
a group without any panel gene; and rod cells without NRL and NR2E3. Each group has 20 genes of its own outside the
panel, and every cell has 300 housekeeping genes, 2,000 background genes and the 13 mitochondrial genes. GNAT1 (rod panel)
and RHO (not scored) are ambient: present in every cell. Four rod cells carry microglia genes, too few to form a cluster.
"""
import importlib.util
import json
import sys
from collections import namedtuple
from itertools import combinations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from pafbench import classes
from pafbench.classes import UNASSIGNED, assign, class_scores, donor_replicate, doublet_scores, flags, name_clusters, pipeline
from pafbench.classes import scored_classes, seed_stability
from pafbench.config import CELLS, CLASS_MARGIN, CLUSTER, MIN_CELLS_GENE, NUCLEI, REPO
from pafbench.markers import AMBIENT_ROD_TRANSCRIPTS, PANEL, ROD_TF

BY_NAME = {c.name: c for c in PANEL}
MISSING = ("PPEF2", "LNP1")                     # panel symbols the features of the toy lack
MT = ("MT-ND1", "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP8", "MT-ATP6", "MT-CO3", "MT-ND3", "MT-ND4L", "MT-ND4", "MT-ND5",
      "MT-ND6", "MT-CYB")
GROUPS = {"rod": 200, "cone": 150, "bipolar": 150, "muller": 100, "astro": 100, "nosignal": 100, "rod_no_tf": 100}
MARKERS = {"rod": BY_NAME["rod"].genes, "cone": BY_NAME["cone"].genes, "bipolar": BY_NAME["bipolar"].genes,
           "muller": ("RLBP1", "SLC1A3", "GLUL", "CLU"), "astro": ("GFAP", "SPARCL1", "GLUL", "CLU"), "nosignal": (),
           "rod_no_tf": tuple(g for g in BY_NAME["rod"].genes if g not in ROD_TF)}
NAMED = {"rod": "rod", "cone": "cone", "bipolar": "bipolar", "muller": "macroglia", "astro": "macroglia",
         "nosignal": UNASSIGNED, "rod_no_tf": "rod"}
N_HOUSE, N_BACKGROUND, N_OWN, N_RARE = 300, 2000, 20, 4
Toy = namedtuple("Toy", "counts group gene_ids symbols barcodes rare cr diffexp")


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    panel = [g for g in dict.fromkeys(g for c in PANEL for g in c.genes) if g not in MISSING]
    symbols = (panel + ["RHO", *MT] + [f"HK{i}" for i in range(N_HOUSE)] + [f"BG{i}" for i in range(N_BACKGROUND)]
               + [f"OWN_{g}_{i}" for g in GROUPS for i in range(N_OWN)])
    col = {s: i for i, s in enumerate(symbols)}
    group = np.repeat(list(GROUPS), list(GROUPS.values()))
    lam = np.zeros((len(group), len(symbols)))
    lam[:, :len(panel)] = 0.1
    lam[:, [col[g] for g in BY_NAME["horizontal"].genes if g in col]] = 0      # horizontal: VAT1L in two cells only, below
    lam[:, [col[g] for g in MT]] = 1.0
    lam[:, col["HK0"]:col["HK0"] + N_HOUSE] = rng.uniform(2, 30, N_HOUSE)
    lam[:, col["BG0"]:col["BG0"] + N_BACKGROUND] = rng.gamma(0.3, 1.0, N_BACKGROUND)
    lam[:, [col["GNAT1"], col["RHO"]]] += 15                                  # ambient rod transcripts in every cell
    for g, genes in MARKERS.items():
        rows = np.flatnonzero(group == g)
        lam[np.ix_(rows, [col[m] for m in genes if m in col] + [col[f"OWN_{g}_{i}"] for i in range(N_OWN)])] += 8
    rare = np.flatnonzero(group == "rod")[:N_RARE]
    lam[np.ix_(rare, [col[m] for m in BY_NAME["microglia"].genes])] += 4
    counts = rng.poisson(lam).astype(np.int32)
    counts[[0, 300], col["VAT1L"]] = 5
    # Cell Ranger's graph clusters: the planted groups numbered from 1, with the rod cells split in two
    cr = np.array([list(GROUPS).index(g) + 1 for g in group], np.int32)
    cr[np.flatnonzero(group == "rod")[100:]] = len(GROUPS) + 1
    norm = counts / counts.sum(axis=1, keepdims=True) * 1e4
    lfc = {k: np.log2((norm[cr == k].mean(axis=0) + 1) / (norm[cr != k].mean(axis=0) + 1)) for k in np.unique(cr)}
    gene_ids = np.array([f"ENSG{i:011d}" for i in range(len(symbols))])
    detected = counts.sum(axis=0) > 0                                         # Cell Ranger lists genes with a UMI
    diffexp = pd.DataFrame({"Feature ID": gene_ids, "Feature Name": symbols})
    for k, v in lfc.items():
        diffexp[f"Cluster {k} Mean Counts"] = counts[cr == k].mean(axis=0)
        diffexp[f"Cluster {k} Log2 fold change"] = v
        diffexp[f"Cluster {k} Adjusted p value"] = 1.0
    barcodes = np.array([f"AAAC{i:012d}-1" for i in range(len(group))])
    return Toy(sp.csr_matrix(counts), group, gene_ids, np.array(symbols), barcodes, rare, cr, diffexp[detected])


def _inputs(toy, d):
    """The files stage 25 reads, as stage 24 and Cell Ranger write them."""
    m, a = d / "matrices", d / "analysis"
    m.mkdir(parents=True)
    sp.save_npz(m / "unfiltered.npz", toy.counts)
    (m / "barcodes.tsv").write_text("".join(f"{b}\n" for b in toy.barcodes))
    (m / "features.tsv").write_text("".join(f"{i}\t{s}\tGene Expression\n" for i, s in zip(toy.gene_ids, toy.symbols)))
    (a / "clustering" / "gene_expression_graphclust").mkdir(parents=True)
    (a / "diffexp" / "gene_expression_graphclust").mkdir(parents=True)
    pd.DataFrame({"Barcode": toy.barcodes[::-1], "Cluster": toy.cr[::-1]}).to_csv(
        a / "clustering" / "gene_expression_graphclust" / "clusters.csv", index=False)
    toy.diffexp.to_csv(a / "diffexp" / "gene_expression_graphclust" / "differential_expression.csv", index=False)
    return m, a


@pytest.fixture(scope="module")
def toy():
    return _toy()


@pytest.fixture(scope="module")
def trial(toy, tmp_path_factory):
    """classes.run on the toy with 8 threads in every thread pool of the process; the script runs it again with 1."""
    d = tmp_path_factory.mktemp("stage25")
    m, a = _inputs(toy, d)
    with threadpool_limits(limits=8):
        res = classes.run("1417", m, d / "first" / "cells", a)
    return d, m, a, res


def _cluster_of(res, toy):
    """{planted group: the clusters of its cells}"""
    c = res.cells.cluster.to_numpy()
    return {g: sorted(set(c[toy.group == g])) for g in GROUPS}


def test_planted_classes_are_recovered(toy, trial):
    res = trial[3]
    c = res.cells
    assert adjusted_rand_score(toy.group, c.cluster) == 1.0
    named = res.clusters.set_index("cluster")["class"]
    for g, (k,) in _cluster_of(res, toy).items():
        assert named[k] == NAMED[g], g
        assert (c["class"][toy.group == g] == NAMED[g]).all(), g
    assert len(res.clusters) == len(GROUPS)
    assert sorted(res.summary["classes"]) == ["bipolar", "cone", "macroglia", "rod", UNASSIGNED]
    assert res.summary["unassigned"] == {"cells": 100, "clusters": _cluster_of(res, toy)["nosignal"]}


def test_a_cluster_without_marker_signal_is_unassigned_and_the_ambient_gene_does_not_decide_it(toy, trial):
    res = trial[3]
    (k,) = _cluster_of(res, toy)["nosignal"]
    row = res.clusters.set_index("cluster").loc[k]
    assert row["class"] == UNASSIGNED and (row.score_best <= 0 or row.margin < CLASS_MARGIN)
    # without the control genes of score_genes the ambient GNAT1 would decide it: the mean log expression of the rod genes
    # is the highest of all classes in that cluster
    norm = res.embedding.norm
    col = {s: i for i, s in enumerate(toy.symbols)}
    cells = toy.group == "nosignal"
    naive = {cl.name: norm.X[cells][:, [col[g] for g in cl.genes if g in col]].mean() for cl in scored_classes("1417")}
    assert max(naive, key=naive.get) == "rod"
    # no cluster outside the rod groups is named rod, and the rod-transcript share flags the ambient RNA in every cell
    rods = _cluster_of(res, toy)["rod"] + _cluster_of(res, toy)["rod_no_tf"]
    assert set(res.clusters[res.clusters["class"] == "rod"].cluster) == set(rods)
    ambient = [col[g] for g in AMBIENT_ROD_TRANSCRIPTS]
    m = toy.counts.toarray()
    assert np.array_equal(res.cells.rod_share.to_numpy(), m[:, ambient].sum(axis=1) / m.sum(axis=1))
    assert (res.cells.rod_share > 0).all()


def test_two_clusters_of_one_class_are_reported_with_their_differing_panel_genes(toy, trial):
    res = trial[3]
    k = _cluster_of(res, toy)
    split = {s["class"]: s for s in res.summary["split_classes"]}
    assert set(split) == {"macroglia", "rod"}
    (muller,), (astro,) = k["muller"], k["astro"]
    assert split["macroglia"]["clusters"] == sorted([muller, astro])
    (pair,) = split["macroglia"]["pairs"]
    higher = dict(zip(pair["clusters"], (pair["higher_in_first"], pair["higher_in_second"])))
    assert (sorted(higher[muller]), sorted(higher[astro])) == (["RLBP1", "SLC1A3"], ["GFAP", "SPARCL1"])
    (pair,) = split["rod"]["pairs"]
    higher = dict(zip(pair["clusters"], (pair["higher_in_first"], pair["higher_in_second"])))
    assert (higher[k["rod"][0]], higher[k["rod_no_tf"][0]]) == (["NRL", "NR2E3"], [])


def test_rod_score_with_the_transcription_factors_only_marks_the_cluster_whose_name_changes(toy, trial):
    res = trial[3]
    k = _cluster_of(res, toy)
    cl = res.clusters.set_index("cluster")
    assert cl.loc[k["rod"][0], "class_with_rod_tf"] == "rod"
    assert cl.loc[k["rod_no_tf"][0], ["class", "class_with_rod_tf"]].tolist() == ["rod", UNASSIGNED]
    assert res.summary["rod_tf_renamed"] == [{"cluster": k["rod_no_tf"][0], "class": "rod", "class_with_rod_tf": UNASSIGNED}]
    assert (cl.drop(index=k["rod_no_tf"])["class"] == cl.drop(index=k["rod_no_tf"]).class_with_rod_tf).all()
    table = res.table.set_index("cluster")
    assert table.loc[k["rod_no_tf"][0], "rod_tf_renames"] and table.rod_tf_renames.sum() == 1
    assert table.loc[k["rod_no_tf"][0], "rod_tf_score"] < 0 < table.loc[k["rod_no_tf"][0], "score_rod"]


def test_margin_rule_at_its_boundary():
    names = ["rod", "cone", "bipolar"]
    s = pd.DataFrame([[0.1, 0.0, -1.0],                                  # best minus second equals the margin: assigned
                      [np.nextafter(0.1, 0.0), 0.0, -1.0],                 # just below it
                      [0.0, -0.5, -1.0],                                   # best not above 0
                      [5e-324, -0.5, -1.0],                                # just above 0
                      [-0.2, 0.3, 0.3],                                    # a tie for the best score
                      [0.0, 0.9, np.nan]], columns=names, index=[3, 4, 5, 6, 7, 8])
    assert CLASS_MARGIN == 0.1
    a = assign(s)
    assert a["class"].tolist() == ["rod", UNASSIGNED, UNASSIGNED, "rod", UNASSIGNED, "cone"]
    assert a.class_second.tolist() == ["cone", "cone", "cone", "cone", "bipolar", "rod"]
    assert a.margin.tolist() == [0.1, np.nextafter(0.1, 0.0), 0.5, 0.5, 0.0, 0.9]
    assert a.score_best.tolist() == [0.1, np.nextafter(0.1, 0.0), 0.0, 5e-324, 0.3, 0.9]
    assert a.class_best.tolist() == ["rod", "rod", "rod", "rod", "cone", "cone"] and list(a.index) == [3, 4, 5, 6, 7, 8]
    assert assign(s, margin=0.0)["class"].tolist() == ["rod", "rod", UNASSIGNED, "rod", "cone", "cone"]


def test_cells_inherit_the_class_of_their_cluster_and_the_cluster_score_is_the_mean_cell_score(trial):
    res = trial[3]
    c = res.cells
    names = [k.name for k in scored_classes("1417")]
    # the cluster scores computed here, not by name_clusters: the mean of the cell scores of each class per cluster
    mean = c.groupby("cluster")[[f"score_{k}" for k in names]].mean().set_axis(names, axis=1)
    cl = res.clusters.set_index("cluster")
    assert np.array_equal(cl.score_best.to_numpy(), mean.max(axis=1).to_numpy())
    assert np.array_equal(cl.score_second.to_numpy(), [r.nlargest(2).iloc[1] for _, r in mean.iterrows()])
    assert cl.class_best.tolist() == mean.idxmax(axis=1).tolist()
    assert cl.class_second.tolist() == [r.nlargest(2).index[1] for _, r in mean.iterrows()]
    assert np.array_equal(cl.margin.to_numpy(), cl.score_best.to_numpy() - cl.score_second.to_numpy())
    assert (c["class"].to_numpy() == cl["class"].loc[c.cluster].to_numpy()).all()
    assert (c.margin.to_numpy() == cl.margin.loc[c.cluster].to_numpy()).all()


def test_1179_only_classes_are_scored_for_1179_only(trial):
    assert [c.name for c in scored_classes("1179")] == [c.name for c in PANEL]
    for lib in NUCLEI:
        assert [c.name for c in scored_classes(lib)] == [c.name for c in PANEL[:10]]
    norm = trial[3].embedding.norm
    symbols = norm.var.symbol.to_numpy()
    (nuclei, rod_tf), (cells, rod_tf_cells) = class_scores(norm, symbols, "1417"), class_scores(norm, symbols, CELLS[0])
    assert list(nuclei) == [c.name for c in PANEL[:10]]
    assert list(cells) == [c.name for c in PANEL]
    assert nuclei.equals(cells[nuclei.columns]) and np.array_equal(rod_tf, rod_tf_cells)
    assert [c for c in trial[3].cells if c.startswith("score_")] == [f"score_{c.name}" for c in PANEL[:10]]


def test_class_scores_keep_the_rod_tf_score_apart_so_name_clusters_names_panel_classes_only(trial):
    # stage 26 names the clusters of filtered matrices with class_scores and name_clusters: the second rod score must not
    # take part as a class
    res = trial[3]
    norm = res.embedding.norm
    for lib in ("1417", CELLS[0]):
        scores, rod_tf = class_scores(norm, norm.var.symbol.to_numpy(), lib)
        assert list(scores) == [c.name for c in scored_classes(lib)] and rod_tf.shape == (norm.n_obs,)
        named = name_clusters(scores, res.cells.cluster.to_numpy())
        assert set(named["class"]) <= set(scores) | {UNASSIGNED} and set(named.class_second) <= set(scores)
    assert np.array_equal(rod_tf, res.cells.rod_tf_score.to_numpy())


def test_genes_in_fewer_than_min_cells_are_left_out_of_the_clustering_only(toy, trial):
    res = trial[3]
    col = {s: i for i, s in enumerate(toy.symbols)}
    in_cells = (toy.counts > 0).sum(axis=0).A1
    assert MIN_CELLS_GENE == 3 and in_cells[col["VAT1L"]] == 2
    model = res.embedding.model
    assert in_cells[model["genes"]].min() >= MIN_CELLS_GENE and col["VAT1L"] not in model["genes"]
    assert res.summary["genes_clustered"] == int((in_cells >= MIN_CELLS_GENE).sum())
    # VAT1L is still scored, but in 2 cells it does not count as detected: the horizontal class rests on no detected gene
    markers = res.markers.set_index(["class", "gene"])
    assert markers.loc[("horizontal", "VAT1L"), "n_genes_detected"] == 0
    assert markers.loc[("horizontal", "VAT1L"), "frac_out"] == 2 / len(toy.group)
    assert res.summary["weak_classes"]["horizontal"] == [f"rests on 0 panel genes detected in {MIN_CELLS_GENE} cells or more"]
    assert res.summary["parameters"]["detected_min_cells"] == MIN_CELLS_GENE
    assert len(res.cells) == len(toy.group)


def test_saved_model_projects_the_cells_onto_their_pca_coordinates(trial):
    res = trial[3]
    model, norm = res.embedding.model, res.embedding.norm
    assert set(model) == {"genes", "gene_ids", "mean", "sd", "loadings"}
    assert len(model["genes"]) == CLUSTER["n_hvg"] and model["loadings"].shape == (CLUSTER["n_hvg"], CLUSTER["n_pcs"])
    assert np.array_equal(model["gene_ids"], norm.var_names.to_numpy()[model["genes"]])
    x = (norm.X[:, model["genes"]].toarray() - model["mean"]) / model["sd"]
    assert np.allclose(x @ model["loadings"], res.embedding.pca, atol=1e-4)
    saved = np.load(trial[0] / "first" / "cells" / "model.npz")
    assert sorted(saved.files) == sorted(model) and all(np.array_equal(saved[k], model[k]) for k in model)


def test_pipeline_returns_every_seed_and_resolution_asked_for(toy):
    # stage 26 clusters each matrix at the three resolutions of CLUSTER; the toy keeps its seven groups at all three and
    # splits them, differently at each seed, at resolution 4
    seeds, resolutions = CLUSTER["seeds"][:2], (CLUSTER["resolution"], *CLUSTER["resolutions_alt"], 4.0)
    emb = pipeline(toy.counts, toy.gene_ids, seeds, resolutions)
    assert sorted(emb.labels) == sorted((s, r) for s in seeds for r in resolutions)
    assert emb.umap is None and emb.pca.shape == (len(toy.group), CLUSTER["n_pcs"])
    assert all(lab.dtype == np.int32 and len(lab) == len(toy.group) for lab in emb.labels.values())
    for s in seeds:
        assert all(adjusted_rand_score(toy.group, emb.labels[(s, r)]) == 1.0 for r in resolutions[:3])
        assert len(set(emb.labels[(s, 4.0)])) > len(GROUPS)
    assert adjusted_rand_score(emb.labels[(0, 4.0)], emb.labels[(1, 4.0)]) < 1.0


def test_pipeline_and_class_scores_are_the_scanpy_chain_with_every_value_written_out(toy):
    # 3.3 steps 2 and 3 with the values of CLUSTER and the scanpy defaults written out, against classes.pipeline at
    # resolution 4.0, where the toy's groups split and the Leiden flavour or k would change the labels, and class_scores
    emb = pipeline(toy.counts, toy.gene_ids, (0,), (4.0,))
    with threadpool_limits(limits=CLUSTER["blas_threads"]):
        a = ad.AnnData(toy.counts.copy(), obs=pd.DataFrame(index=np.arange(len(toy.group)).astype(str)),
                       var=pd.DataFrame(index=toy.gene_ids))
        sc.pp.normalize_total(a, target_sum=CLUSTER["target_sum"])
        sc.pp.log1p(a)
        kept = np.flatnonzero((toy.counts > 0).sum(axis=0).A1 >= MIN_CELLS_GENE)
        b = a[:, kept].copy()
        sc.pp.highly_variable_genes(b, n_top_genes=CLUSTER["n_hvg"], flavor="seurat")
        hv = kept[b.var.highly_variable.to_numpy()]
        b = b[:, b.var.highly_variable].copy()
        sc.pp.scale(b, zero_center=True, max_value=None)
        sc.pp.pca(b, n_comps=CLUSTER["n_pcs"], svd_solver="arpack", random_state=0)
        sc.pp.neighbors(b, n_neighbors=CLUSTER["k"], n_pcs=CLUSTER["n_pcs"], method="umap", metric="euclidean", random_state=0)
        sc.tl.leiden(b, resolution=4.0, random_state=0, flavor="igraph", n_iterations=2, directed=False)
        column = dict(zip(toy.symbols, toy.gene_ids))
        sc.tl.score_genes(a, [column[g] for g in BY_NAME["cone"].genes], ctrl_as_ref=True, ctrl_size=50, gene_pool=None,
                          n_bins=25, score_name="cone", random_state=0)
    labels = b.obs["leiden"].to_numpy().astype(np.int32)
    assert np.array_equal(class_scores(emb.norm, toy.symbols, "1417")[0]["cone"].to_numpy(), a.obs["cone"].to_numpy())
    assert len(set(labels)) > len(GROUPS) and np.array_equal(emb.labels[(0, 4.0)], labels)
    assert np.array_equal(emb.pca, b.obsm["X_pca"])
    m = emb.model
    assert np.array_equal(m["genes"], hv) and np.array_equal(m["gene_ids"], toy.gene_ids[hv])
    assert np.array_equal(m["mean"], b.var["mean"]) and np.array_equal(m["sd"], b.var["std"])
    assert np.array_equal(m["loadings"], b.varm["PCs"])


def test_doublet_scores_are_seeded_and_on_the_raw_counts(toy, trial):
    # D9: scrublet "Works best if the input is a raw (unnormalized) counts matrix" (scanpy.pp.scrublet documentation); its own
    # seed, not the global numpy state (score_genes seeds that one: np.random.seed(random_state), scanpy/tools/_score_genes.py)
    np.random.seed(1)
    first = doublet_scores(toy.counts)
    np.random.seed(2)
    assert np.array_equal(first, doublet_scores(toy.counts))
    a = ad.AnnData(toy.counts.copy(), obs=pd.DataFrame(index=np.arange(len(toy.group)).astype(str)))
    with threadpool_limits(limits=CLUSTER["blas_threads"]):
        sc.pp.scrublet(a, sim_doublet_ratio=2.0, expected_doublet_rate=0.05, n_prin_comps=30, random_state=0)
    assert np.array_equal(trial[3].cells.doublet_score.to_numpy(), a.obs["doublet_score"].to_numpy())
    assert np.array_equal(first, a.obs["doublet_score"].to_numpy())


def test_seed_stability_on_the_toy(trial):
    s = trial[3].summary["seed_stability"]
    assert s["seeds"] == list(CLUSTER["seeds"]) and s["resolution"] == CLUSTER["resolution"]
    assert [(p["a"], p["b"]) for p in s["pairwise_ari"]] == list(combinations(CLUSTER["seeds"], 2))
    assert all(p["ari"] == 1.0 for p in s["pairwise_ari"]) and s["ari_min"] == 1.0
    assert s["classes"] == {c: {"cells": n, "lowest_jaccard": 1.0, "seed": 1}
                            for c, n in (("rod", 300), ("cone", 150), ("bipolar", 150), ("macroglia", 200), (UNASSIGNED, 100))}


def test_seed_stability_reports_the_lowest_jaccard_of_each_class_against_seed_0():
    # cluster 0 scores rod, cluster 1 cone; at seed 2 one rod cell joins the cone cluster, at seed 1 nothing moves
    scores = pd.DataFrame({"rod": [1, 1, 1, 1, 0, 0, 0, 0.0], "cone": [0, 0, 0, 0, 1, 1, 1, 1.0]})
    labels = {0: np.array([0, 0, 0, 0, 1, 1, 1, 1]), 1: np.array([1, 1, 1, 1, 0, 0, 0, 0]),
              2: np.array([0, 0, 0, 1, 1, 1, 1, 1])}
    s = seed_stability(labels, scores, 1.0)
    assert [p["ari"] for p in s["pairwise_ari"]] == [1.0, adjusted_rand_score(labels[0], labels[2]),
                                                     adjusted_rand_score(labels[1], labels[2])]
    assert s["classes"] == {"rod": {"cells": 4, "lowest_jaccard": 3 / 4, "seed": 2},
                            "cone": {"cells": 4, "lowest_jaccard": 4 / 5, "seed": 2}}
    assert s["n_clusters"] == {"0": 2, "1": 2, "2": 2}


def test_seed_stability_jaccard_is_intersection_over_union_when_a_class_loses_and_gains_cells():
    # at seed 1 rod cell 3 goes to the cone cluster and cone cell 4 to the rod cluster: rod is {0, 1, 2, 3} at seed 0 and
    # {0, 1, 2, 4} at seed 1, 3 cells shared of 5 (intersection over the larger set would give 3 / 4)
    scores = pd.DataFrame({"rod": [1, 1, 1, 1, 0, 0, 0, 0.0], "cone": [0, 0, 0, 0, 1, 1, 1, 1.0]})
    labels = {0: np.array([0, 0, 0, 0, 1, 1, 1, 1]), 1: np.array([0, 0, 0, 1, 0, 1, 1, 1])}
    s = seed_stability(labels, scores, 1.0)
    assert s["classes"] == {"rod": {"cells": 4, "lowest_jaccard": 3 / 5, "seed": 1},
                            "cone": {"cells": 4, "lowest_jaccard": 3 / 5, "seed": 1}}


def test_rare_class_candidates_are_the_planted_microglia_cells(toy, trial):
    res = trial[3]
    rare = res.summary["rare_candidates"]
    assert set(rare) == {c.name for c in scored_classes("1417")} - {"rod", "cone", "bipolar", "macroglia"}
    assert rare["microglia"]["cells"] == N_RARE
    s = res.cells.score_microglia
    assert rare["microglia"]["threshold"] == np.percentile(s, 99)
    top = res.cells[[f"score_{c.name}" for c in scored_classes("1417")]].to_numpy().argmax(axis=1)
    planted = np.zeros(len(toy.group), bool)
    planted[toy.rare] = True
    assert np.array_equal((top == 7) & (s.to_numpy() > rare["microglia"]["threshold"]), planted)
    # next to each count the cells above the class's percentile, about 1 percent of the cells by construction
    assert rare["microglia"]["above_pctl"] == 9
    assert all(v["above_pctl"] == int((res.cells[f"score_{c}"] > v["threshold"]).sum()) for c, v in rare.items())
    assert rare["vascular"]["cells"] > 0 and res.summary["collapse_not_askable"] == ["microglia", "vascular"]


def test_marker_table(toy, trial):
    res = trial[3]
    t = res.markers
    assert list(t) == ["class", "gene", "in_matrix", "frac_in", "frac_out", "mean_in", "mean_out", "auc_class",
                       "n_genes_detected"]
    assert list(t[["class", "gene"]].itertuples(index=False, name=None)) == [
        (c.name, g) for c in scored_classes("1417") for g in c.genes]
    assert t[~t.in_matrix].gene.tolist() == list(MISSING) and t[~t.in_matrix].frac_in.isna().all()
    assert res.summary["panel_genes_missing"] == list(MISSING)
    row = t.set_index(["class", "gene"]).loc[("cone", "ARR3")]
    m = toy.counts.toarray()[:, list(toy.symbols).index("ARR3")]
    cone = toy.group == "cone"
    assert (row.frac_in, row.frac_out) == ((m[cone] > 0).mean(), (m[~cone] > 0).mean())
    x = res.embedding.norm[:, toy.gene_ids[list(toy.symbols).index("ARR3")]].X.toarray().ravel()
    assert (row.mean_in, row.mean_out) == pytest.approx((x[cone].mean(), x[~cone].mean()))
    assert row.auc_class == 1.0 and row.n_genes_detected == 10
    # classes without cells: no inside numbers and no AUC
    assert t[t["class"] == "RPE"][["frac_in", "mean_in", "auc_class"]].isna().all().all()
    assert t[t["class"] == "rod"].n_genes_detected.eq(8).all()


def test_a_panel_gene_counts_as_detected_with_a_umi_in_min_cells_cells_or_more():
    # check 1 counts the panel genes behind a class score as 3.3 step 1 detects a gene: ONECUT1 has a UMI in 3 cells,
    # ONECUT2 in 2, LHX1 in none and VAT1L in 4 (LNP1 is not in the matrix)
    symbols = np.array(["ONECUT1", "ONECUT2", "LHX1", "VAT1L", "HK"])
    counts = sp.csr_matrix(np.array([[1, 1, 0, 1, 5], [1, 1, 0, 1, 5], [1, 0, 0, 1, 5], [0, 0, 0, 1, 5], [0, 0, 0, 0, 5],
                                     [0, 0, 0, 0, 5]], np.int32))
    norm = classes.normalise(counts, np.array([f"G{i}" for i in range(len(symbols))]))
    cell_class = np.array(["horizontal"] * 3 + [UNASSIGNED] * 3)
    scores = pd.DataFrame({c.name: np.zeros(6) for c in scored_classes("1417")})
    t = classes.marker_table(counts, norm, symbols, cell_class, scores, "1417")
    assert MIN_CELLS_GENE == 3
    assert t[t["class"] == "horizontal"].n_genes_detected.eq(2).all()
    assert t[t["class"] != "horizontal"].n_genes_detected.eq(0).all()


def test_cell_ranger_consistency(toy, trial):
    res = trial[3]
    cr = res.summary["cell_ranger"]
    assert res.cells.cr_cluster.tolist() == toy.cr.tolist()
    assert cr["clusters"] == 8 and cr["ari"] == adjusted_rand_score(toy.cr, res.cells.cluster)
    assert sum(n for row in cr["contingency"].values() for n in row.values()) == len(toy.group)
    mapped = {r["cr_cluster"]: r["class"] for r in cr["cr_clusters"]}
    assert mapped == {1: "rod", 2: "cone", 3: "bipolar", 4: "macroglia", 5: "macroglia", 6: UNASSIGNED, 7: "rod", 8: "rod"}
    ranks = pd.DataFrame(cr["panel_gene_ranks"])
    assert sorted(set(ranks.cr_cluster)) == [1, 2, 3, 4, 5, 7, 8] and cr["genes_ranked"] == len(toy.diffexp)
    lfc = toy.diffexp.set_index("Feature Name")["Cluster 2 Log2 fold change"]
    for gene in ("ARR3", "OPN1SW"):
        row = ranks[(ranks.cr_cluster == 2) & (ranks.gene == gene)].iloc[0]
        assert row.log2fc == pytest.approx(lfc[gene]) and row["rank"] == int((lfc > lfc[gene]).sum()) + 1
    assert ranks[ranks.cr_cluster == 2]["rank"].max() <= 10 + N_OWN
    # a panel gene Cell Ranger does not list has no rank
    missing = ranks[(ranks["class"] == "rod") & (ranks.gene == "PPEF2")]
    assert len(missing) == 3 and missing["rank"].isna().all()


def test_d9_flags_and_qc_columns(toy, trial):
    res = trial[3]
    c = res.cells
    m = toy.counts.toarray()
    mt = [list(toy.symbols).index(g) for g in MT]
    assert np.allclose(c.pct_mt, 100 * m[:, mt].sum(axis=1) / m.sum(axis=1))
    assert c.umis.tolist() == m.sum(axis=1).tolist() and c.genes.tolist() == (m > 0).sum(axis=1).tolist()
    assert np.isfinite(c.doublet_score).all() and c.doublet_score.between(0, 1).all() and c.doublet_score.std() > 0
    cl = res.clusters.set_index("cluster")
    for k, frame in c.groupby("cluster"):
        assert cl.loc[k, "doublet_score_median"] == frame.doublet_score.median()
        assert cl.loc[k, "rod_share_median"] == frame.rod_share.median()
        assert cl.loc[k, "n_cells"] == len(frame)
    per_class = res.summary["classes"]["cone"]
    assert per_class["rod_share_median"] == c.rod_share[c["class"] == "cone"].median()
    assert per_class["doublet_score_median"] == c.doublet_score[c["class"] == "cone"].median()


def test_flags_of_the_class_table():
    clusters = pd.DataFrame({"cluster": [0, 1, 2, 3], "class": ["rod", "microglia", "vascular", "rod"],
                             "class_with_rod_tf": ["rod", "microglia", "vascular", UNASSIGNED]})
    markers = pd.DataFrame({"class": ["rod", "rod", "progenitor", "proliferating", "microglia", "vascular"],
                            "n_genes_detected": [9, 9, 3, 1, 2, 3]})
    f = flags(CELLS[0], clusters, markers, {})
    assert f["organoid_named"] == [{"cluster": 1, "class": "microglia"}, {"cluster": 2, "class": "vascular"}]
    assert f["rod_tf_renamed"] == [{"cluster": 3, "class": "rod", "class_with_rod_tf": UNASSIGNED}]
    assert f["weak_classes"] == {"progenitor": ["two of its genes also mark mature classes (VSX2 bipolar, PAX6 amacrine)"],
                                 "proliferating": [f"rests on 1 panel gene detected in {MIN_CELLS_GENE} cells or more"],
                                 "microglia": [f"rests on 2 panel genes detected in {MIN_CELLS_GENE} cells or more"]}
    for lib in NUCLEI:
        assert flags(lib, clusters, markers, {})["organoid_named"] == []
    assert f["collapse_not_askable"] == []
    # check 6: microglia or vascular cells that exist (rare candidates above 0) but form no cluster
    rare = {"microglia": {"cells": 0, "threshold": 0.3, "above_pctl": 9}, "vascular": {"cells": 2, "threshold": 0.2,
                                                                                        "above_pctl": 9},
            "RPE": {"cells": 4, "threshold": 0.4, "above_pctl": 9}}
    assert flags("1417", clusters[clusters.cluster.isin([0, 3])], markers, rare)["collapse_not_askable"] == ["vascular"]


def test_donor_replicate(toy, trial):
    res = trial[3]
    keep = np.arange(len(toy.group)) % 2 == 0                         # half of the cells, thinned to 60% of their UMIs
    thinned = sp.csr_matrix(np.random.default_rng(1).binomial(toy.counts[keep].toarray(), 0.6).astype(np.int32))
    cells_b = res.cells[keep].reset_index(drop=True)
    d = donor_replicate(("1417", "1419"), (res.cells, cells_b), (toy.counts, thinned), toy.symbols)
    by = {c["class"]: c for c in d["classes"]}
    assert list(by) == ["rod", "cone", "bipolar", "macroglia", UNASSIGNED]
    assert (by["rod"]["cells"], by["rod"]["share"]) == ([300, 150], [300 / 900, 150 / 450])
    assert all(0.95 < c["pearson_class_mean"] <= 1 for c in d["classes"])
    cone = {g["gene"]: g for g in by["cone"]["panel_genes"]}
    assert set(cone) == set(BY_NAME["cone"].genes) and cone["ARR3"]["mean"][0] > 2 and cone["ARR3"]["mean"][1] > 2
    same = donor_replicate(("1417", "1417"), (res.cells, res.cells), (toy.counts, toy.counts), toy.symbols)
    assert all(c["pearson_class_mean"] == pytest.approx(1.0) for c in same["classes"])


def test_marker_plot_of_1179_shows_the_genes_of_two_classes_under_both(trial, tmp_path):
    # VSX2, OTX2, FABP7 and DLL3 are panel genes of two classes each; the 1179 plot has a column for each class of a gene
    classes.marker_plot(tmp_path / "plot.png", trial[3], CELLS[0])
    assert (tmp_path / "plot.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cell_ranger_clusters_must_cover_every_called_cell(toy, trial, tmp_path):
    a = trial[2]
    cr, lfc = classes.read_cell_ranger(a, toy.barcodes)
    assert cr.tolist() == toy.cr.tolist() and list(lfc) == list(range(1, len(GROUPS) + 2))
    assert lfc.index.tolist() == toy.diffexp["Feature ID"].tolist()
    with pytest.raises(SystemExit, match="1 called cells have no Cell Ranger cluster"):
        classes.read_cell_ranger(a, [*toy.barcodes, "TTTT-1"])


def test_ambient_list_is_the_list_of_the_design():
    text = (REPO / "docs/specs/2026-10-12-polyafilter-bench-milestone3-design.md").read_text()
    start = text.index("(a) Rod-transcript share: UMIs of ") + len("(a) Rod-transcript share: UMIs of ")
    listed = text[start:text.index(" over the cell's UMIs", start)].replace(" and ", ", ").split(", ")
    assert tuple(listed) == AMBIENT_ROD_TRANSCRIPTS


COLUMNS = {
    "cells.parquet": ["cb", "umis", "genes", "pct_mt", "cluster", "class", "margin", *[f"score_{c.name}" for c in PANEL[:10]],
                      "rod_tf_score", "rod_share", "doublet_score", "cr_cluster", "umap1", "umap2"],
    "clusters.parquet": ["cluster", "n_cells", "class", "score_best", "class_second", "score_second", "margin",
                         "class_with_rod_tf", "rod_share_median", "doublet_score_median"],
    "marker_table.parquet": ["class", "gene", "in_matrix", "frac_in", "frac_out", "mean_in", "mean_out", "auc_class",
                             "n_genes_detected"],
}


def test_stage25_script_writes_the_outputs_and_the_same_seed_gives_the_same_files(toy, trial, monkeypatch, capsys):
    d, m, a, res = trial
    spec = importlib.util.spec_from_file_location("stage25", REPO / "scripts" / "25_cell_classes.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    called = []
    monkeypatch.setattr(stage, "require_passed", lambda *args: called.append(args) or {"passed": True})
    out = d / "second" / "cells"
    monkeypatch.setattr(sys, "argv", ["25", "1417", "--cells", str(out), "--matrices", str(m), "--recount", str(d / "rc"),
                                      "--analysis", str(a)])
    with threadpool_limits(limits=1):
        stage.main()
    assert called == [("1417", d / "rc", m)]
    first = d / "first" / "cells"
    names = sorted(p.name for p in first.iterdir())
    assert names == ["cells.parquet", "class_table.csv", "classes.json", "clusters.parquet", "marker_plot.png",
                     "marker_table.parquet", "model.npz"]
    assert sorted(p.name for p in out.iterdir()) == names
    for name in names:
        assert (out / name).read_bytes() == (first / name).read_bytes(), name
    for name, columns in COLUMNS.items():
        assert list(pd.read_parquet(out / name)) == columns, name
    cells = pd.read_parquet(out / "cells.parquet")
    # every called cell one row, in the order of barcodes.tsv, with one class or "unassigned"
    assert cells.cb.tolist() == toy.barcodes.tolist() and cells["class"].notna().all()
    assert set(cells["class"]) <= {c.name for c in PANEL} | {UNASSIGNED}
    js = json.loads((out / "classes.json").read_text())
    assert js == json.loads(json.dumps(res.summary))
    for key in ("library", "parameters", "seed_stability", "cell_ranger", "unassigned", "rare_candidates",
                "panel_genes_missing", "weak_classes", "classes", "split_classes", "rod_tf_renamed", "organoid_named"):
        assert key in js, key
    assert js["parameters"]["cluster"] == json.loads(json.dumps(CLUSTER)) and js["parameters"]["class_margin"] == CLASS_MARGIN
    assert js["parameters"]["scanpy"] == {"hvg_flavor": "seurat", "svd_solver": "arpack", "zero_center": True, "max_value": None,
                                          "neighbors_method": "umap", "metric": "euclidean", "ctrl_as_ref": True,
                                          "ctrl_size": 50, "n_bins": 25, "gene_pool": None}
    assert js["parameters"]["scrublet"] == {"sim_doublet_ratio": 2.0, "expected_doublet_rate": 0.05, "n_prin_comps": 30}
    assert js["parameters"]["scanpy_n_jobs"] == sc.settings.n_jobs
    assert js["seeds"] == list(CLUSTER["seeds"])
    assert js["inputs"] == {"matrices": str(m), "clusters": str(a / "clustering" / "gene_expression_graphclust" / "clusters.csv"),
                            "differential_expression": str(a / "diffexp" / "gene_expression_graphclust" /
                                                           "differential_expression.csv")}
    table = pd.read_csv(out / "class_table.csv")
    assert table.cluster.tolist() == sorted(cells.cluster.unique()) and "flags" in table and "rod_tf_renames" in table
    assert (out / "marker_plot.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["library"] == "1417" and line["cells"] == len(toy.group) and line["clusters"] == len(GROUPS)
    assert line["unassigned_cells"] == 100 and "seconds" in line


def test_donor_script_writes_the_comparison(toy, trial, tmp_path, monkeypatch, capsys):
    d, m, a, res = trial
    spec = importlib.util.spec_from_file_location("stage25b", REPO / "scripts" / "25b_donor_replicate.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    called = []
    monkeypatch.setattr(stage, "require_passed", lambda *args: called.append(args) or {"passed": True})
    cells = d / "first" / "cells"
    out = tmp_path / "donor.json"
    monkeypatch.setattr(sys, "argv", ["25b", "--cells", str(cells), str(cells), "--matrices", str(m), str(m),
                                      "--recount", str(d / "r1"), str(d / "r2"), "--out", str(out)])
    stage.main()
    assert called == [("1417", d / "r1", m), ("1419", d / "r2", m)]
    js = json.loads(out.read_text())
    assert js == json.loads(json.dumps(donor_replicate(("1417", "1419"), (res.cells, res.cells), (toy.counts, toy.counts),
                                                       toy.symbols)))
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["libraries"] == ["1417", "1419"] and "seconds" in line
