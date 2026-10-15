"""Stage 26: impact of filtering on cells and classes (pafbench.impact; design of 2026-10-12, section 3.4).

The toy library has 400 cells in four planted groups with Poisson counts: rod, cone and bipolar cells with the genes of their
panel class and 40 genes of their own outside the panel, and a group without either, which stage 25 leaves unassigned. Every
cell has 200 housekeeping genes, 600 background genes and every panel gene at a low rate. Stage 25 is run on it as the stage
does (pipeline, class_scores, name_clusters at seed 0 and resolution 1.0). Four filtered matrices, one per setting tag:
- DELETE: the cone panel genes lose every UMI in the cone cells (the cone cells keep their own genes);
- SCRAMBLE: each cone cell is drawn from the expression of a rod or bipolar cell taken at random;
- UNIFORM: every cell drawn without replacement down to its total under DELETE, blind to the gene;
- SHIFT: every cell keeps 1 in 10 of its UMIs, blind to the gene and the class, a change common to all cells.
"""
import importlib.util
import itertools
import json
import os
import subprocess
import sys
from collections import namedtuple
from functools import partial

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from pafbench import classes, impact
from pafbench.config import (CHOSEN_SETTINGS, CLUSTER, IMPACT_SETTINGS, JACCARD_DISSOLVED, JACCARD_STABLE, KNN_FOLDS, KNN_K,
                             MARKER_LOSS, REPO, THIN_REPLICATES, THIN_REPLICATES_CHOSEN, TOP_GENES)
from pafbench.flags import setting_tag
from pafbench.markers import PANEL

BY_NAME = {c.name: c for c in PANEL}
GROUPS = {"rod": 100, "cone": 100, "bipolar": 100, "nosignal": 100}
N_HOUSE, N_BACKGROUND, N_OWN = 200, 600, 40
MARKER_LAM, OWN_LAM = 8, 12
TAGS = [setting_tag(s) for s in IMPACT_SETTINGS]
DELETE, SCRAMBLE, UNIFORM, SHIFT = "c300_l10_m1_e", "c300_l10_m1_i", "c275_l9_m1_e", "c275_l9_m1_i"
CHOSEN = setting_tag(CHOSEN_SETTINGS[0])
RESOLUTIONS = (CLUSTER["resolution"], *CLUSTER["resolutions_alt"])
Toy = namedtuple("Toy", "counts group gene_ids symbols barcodes lam")


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    panel = list(dict.fromkeys(g for c in PANEL for g in c.genes))
    symbols = (panel + [f"HK{i}" for i in range(N_HOUSE)] + [f"BG{i}" for i in range(N_BACKGROUND)]
               + [f"OWN_{g}_{i}" for g in GROUPS for i in range(N_OWN)])
    col = {s: i for i, s in enumerate(symbols)}
    group = np.repeat(list(GROUPS), list(GROUPS.values()))
    lam = np.zeros((len(group), len(symbols)))
    lam[:, :len(panel)] = 0.1
    lam[:, col["HK0"]:col["HK0"] + N_HOUSE] = rng.uniform(2, 30, N_HOUSE)
    lam[:, col["BG0"]:col["BG0"] + N_BACKGROUND] = rng.gamma(0.3, 1.0, N_BACKGROUND)
    for g in ("rod", "cone", "bipolar"):
        rows = np.flatnonzero(group == g)
        lam[np.ix_(rows, [col[m] for m in BY_NAME[g].genes])] += MARKER_LAM
        lam[np.ix_(rows, [col[f"OWN_{g}_{i}"] for i in range(N_OWN)])] += OWN_LAM
    counts = sp.csr_matrix(rng.poisson(lam).astype(np.int32))
    return Toy(counts, group, np.array([f"ENSG{i:011d}" for i in range(len(symbols))]), np.array(symbols),
               np.array([f"AAAC{i:012d}-1" for i in range(len(group))]), lam)


def _scenarios(toy):
    """{tag: filtered matrix} of the module docstring."""
    rng = np.random.default_rng(1)
    col = {s: i for i, s in enumerate(toy.symbols)}
    cone = toy.group == "cone"
    m = toy.counts.toarray()
    delete = m.copy()
    delete[np.ix_(cone, [col[g] for g in BY_NAME["cone"].genes])] = 0
    scramble = m.copy()
    pick = rng.choice(np.flatnonzero(np.isin(toy.group, ["rod", "bipolar"])), cone.sum())
    scramble[cone] = rng.poisson(toy.lam[pick])
    uniform = impact.thin(toy.counts, delete.sum(axis=1), np.random.default_rng(2))
    shift = rng.binomial(m, 0.1)
    return {DELETE: sp.csr_matrix(delete.astype(np.int32)), SCRAMBLE: sp.csr_matrix(scramble.astype(np.int32)),
            UNIFORM: uniform, SHIFT: sp.csr_matrix(shift.astype(np.int32))}


def _cell_totals(toy, tag, m, lost, gained):
    """The columns of stage 24's cell_totals.parquet that stage 26 reads, made up from the matrix: a quarter of the UMIs
    intronic (re_class N), a fortieth both, and records in proportion."""
    umis = np.asarray(m.sum(axis=1)).ravel().astype(np.int64)
    n, en = umis // 4, umis // 40
    e = umis - n - en
    return pd.DataFrame({"cb": toy.barcodes, "tag": tag, "umis": umis, "genes": m.getnnz(axis=1).astype(np.int64),
                         "lost": np.asarray(lost.sum(axis=1)).ravel().astype(np.int64),
                         "gained": np.asarray(gained.sum(axis=1)).ravel().astype(np.int64), "switched": 0,
                         "umis_E": e, "umis_N": n, "umis_EN": en, "mol_partial": 0, "rec_E": 3 * e + 1, "rec_N": 5 * n + en,
                         "rec_I": 100, "rec_none": 7, "rec_EN_gx": 3 * e + 2 * n + 1})


def _stage24(d, toy, filtered):
    """The files of stage 24 for toy and {tag: filtered matrix}: M0, L and G of each tag, barcodes, features, manifest and
    cell_totals."""
    m = d / "matrices"
    (m / "lost").mkdir(parents=True)
    (m / "gained").mkdir()
    sp.save_npz(m / "unfiltered.npz", toy.counts)
    zero = sp.csr_matrix(toy.counts.shape, dtype=np.int32)
    totals = [_cell_totals(toy, "unfiltered", toy.counts, zero, zero)]
    for tag, f in filtered.items():
        diff = (f - toy.counts).toarray()
        lost, gained = (sp.csr_matrix(np.maximum(s * diff, 0).astype(np.int32)) for s in (-1, 1))
        sp.save_npz(m / "lost" / f"{tag}.npz", lost)
        sp.save_npz(m / "gained" / f"{tag}.npz", gained)
        totals.append(_cell_totals(toy, tag, f, lost, gained))
    (m / "barcodes.tsv").write_text("".join(f"{b}\n" for b in toy.barcodes))
    (m / "features.tsv").write_text("".join(f"{i}\t{s}\tGene Expression\n" for i, s in zip(toy.gene_ids, toy.symbols)))
    (m / "manifest.json").write_text(json.dumps({"tags": list(filtered), "shape": list(toy.counts.shape)}))
    pd.concat(totals, ignore_index=True).to_parquet(m / "cell_totals.parquet", index=False)
    return m


def _stage25(d, toy):
    """cells.parquet, model.npz and classes.json as stage 25 writes them for the toy (the columns stage 26 reads)."""
    c = d / "cells"
    c.mkdir(parents=True)
    emb = classes.pipeline(toy.counts, toy.gene_ids)
    labels = emb.labels[(CLUSTER["seeds"][0], CLUSTER["resolution"])]
    scores, _ = classes.class_scores(emb.norm, toy.symbols, "1417")
    named = classes.name_clusters(scores, labels)
    pd.DataFrame({"cb": toy.barcodes, "umis": np.asarray(toy.counts.sum(axis=1)).ravel(), "cluster": labels,
                  "class": named["class"].loc[labels].to_numpy()}).to_parquet(c / "cells.parquet", index=False)
    np.savez(c / "model.npz", **emb.model)
    (c / "classes.json").write_text(json.dumps({"collapse_not_askable": ["vascular"]}))
    return c


@pytest.fixture(scope="module")
def toy():
    return _toy()


@pytest.fixture(scope="module")
def scenarios(toy):
    return _scenarios(toy)


@pytest.fixture(scope="module")
def stage(toy, scenarios, tmp_path_factory):
    """impact.run on the toy and the four scenarios, in this process (workers 1)."""
    d = tmp_path_factory.mktemp("stage26")
    m, c = _stage24(d, toy, scenarios), _stage25(d, toy)
    summary = impact.run("1417", m, c, d / "impact").summary
    out = {name: pd.read_parquet(d / "impact" / f"{name}.parquet") for name in impact.COLUMNS}
    return d, summary, out


def _call(summary, cls, tag, resolution=CLUSTER["resolution"]):
    hits = [e for e in summary["calls"] if (e["class"], e["tag"], e["resolution"]) == (cls, tag, resolution)]
    return hits[0] if hits else None


def test_the_toy_gives_the_planted_classes(toy, stage):
    cells = pd.read_parquet(stage[0] / "cells" / "cells.parquet")
    expected = np.where(toy.group == "nosignal", classes.UNASSIGNED, toy.group)
    assert (cells["class"].to_numpy() == expected).all()


def test_deleting_the_marker_umis_of_a_class_loses_its_name_while_its_cells_stay_together(stage):
    _, summary, out = stage
    e = _call(summary, "cone", DELETE)
    assert e["call"] == impact.UNNAMED != impact.APART
    j = out["class_jaccard"]
    cone = j[(j["class"] == "cone") & (j.tag == DELETE) & (j.resolution == CLUSTER["resolution"])]
    f, t = cone[cone.kind == "filtered"], cone[cone.kind == "thinned"]
    assert len(f) == len(CLUSTER["seeds"]) and (f.jaccard > JACCARD_STABLE).all() and (f.named_share <= impact.NAME_KEPT).all()
    assert (f.named_class != "cone").all() and (t.named_share > impact.NAME_KEPT).all() and (t.named_class == "cone").all()
    assert e["jaccard"]["filtered"] == f.jaccard.tolist() and e["named_share"]["filtered"] == f.named_share.tolist()
    # the marker loss behind the call: every cone panel gene loses all its UMIs in the cone cells, and the class score falls
    assert e["marker_loss"]["panel_genes"] == {g: 1.0 for g in BY_NAME["cone"].genes}
    assert e["marker_loss"]["score"][0] > 1 > 0.5 > e["marker_loss"]["score"][1]
    assert e["marker_loss"]["margin"][0] > 0 > e["marker_loss"]["margin"][1]
    for cls in ("rod", "bipolar"):
        assert _call(summary, cls, DELETE)["call"] == impact.NEITHER


def test_scrambling_the_counts_of_a_class_gives_cells_that_no_longer_cluster_together(stage):
    _, summary, out = stage
    e = _call(summary, "cone", SCRAMBLE)
    assert e["call"] == impact.APART
    assert max(e["jaccard"]["filtered"]) <= JACCARD_DISSOLVED
    assert min(e["jaccard"]["unfiltered"]) > JACCARD_STABLE and min(min(r) for r in e["jaccard"]["thinned"]) > JACCARD_STABLE


def test_uniform_thinning_of_the_same_size_gives_neither(stage, scenarios, toy):
    _, summary, _ = stage
    assert (np.asarray(scenarios[UNIFORM].sum(axis=1)) == np.asarray(scenarios[DELETE].sum(axis=1))).all()
    for cls in ("rod", "cone", "bipolar"):
        assert _call(summary, cls, UNIFORM)["call"] == impact.NEITHER


def test_retention_is_one_for_the_unchanged_matrix(stage):
    _, _, out = stage
    k = out["knn"]
    unfiltered = k[k.kind == "unfiltered"]
    assert unfiltered.tag.eq("unfiltered").all() and len(unfiltered) == 4 and (unfiltered.retention == 1.0).all()
    pc = out["per_cell"]
    u = pc[pc.tag == "unfiltered"]
    assert (u.knn_class == u["class"]).all()


def test_a_common_shift_lowers_retention_in_filtered_and_thinned_alike_and_leaves_the_cross_validated_recall(stage):
    _, _, out = stage
    k = out["knn"]
    s = k[k.tag == SHIFT].pivot_table(index="class", columns="kind", values=["retention", "cv_recall"])
    base = k[k.kind == "unfiltered"].set_index("class")
    assert s[("retention", "filtered")][classes.UNASSIGNED] < 0.5 and s[("retention", "thinned")][classes.UNASSIGNED] < 0.5
    assert (s["retention"]["filtered"] - s["retention"]["thinned"]).abs().max() <= 0.1
    for kind in ("filtered", "thinned"):
        assert (s[("cv_recall", kind)] == base.cv_recall.loc[s.index]).all()
    assert (base.cv_recall == 1.0).all()
    # the confusion table tells the shift apart from one class moving into another: the unassigned cells go to the classes
    c = out["knn_confusion"]
    shifted = c[(c.tag == SHIFT) & (c.kind == "filtered") & (c.class_fixed == classes.UNASSIGNED)]
    assert shifted.cells.sum() == 100 and set(shifted.class_projected) - {classes.UNASSIGNED} <= {"rod", "cone", "bipolar"}


def test_best_union_jaccard_equals_a_brute_force_search_over_all_cluster_subsets():
    rng = np.random.default_rng(0)
    for case in range(30):
        n_clusters = int(rng.integers(1, 8))
        labels = rng.integers(0, n_clusters, 60) * 3 + 5                  # cluster ids need not run from 0
        fixed = rng.choice(np.array(["a", "b", "c"]), 60, p=rng.dirichlet([1, 1, 1]))
        if case % 3 == 0:                                                 # whole clusters of one class, as in a good clustering
            fixed = np.where(labels % 2 == 0, "a", fixed)
        got = impact.best_union(fixed, labels).set_index("class")
        ks = np.unique(labels)
        for c in np.unique(fixed):
            best = 0.0
            for r in range(1, len(ks) + 1):
                for subset in itertools.combinations(ks, r):
                    union = np.isin(labels, subset)
                    best = max(best, (union & (fixed == c)).sum() / (union | (fixed == c)).sum())
            assert got.loc[c, "jaccard"] == pytest.approx(best, abs=1e-12), (case, c)
            # clusters_in_union: the shortest sorted prefix that reaches it
            assert 1 <= got.loc[c, "clusters_in_union"] <= len(ks)


def test_best_union_takes_the_shortest_prefix_and_counts_its_clusters():
    # class a: cluster 7 all a (4 cells), cluster 3 half a (2 of 4), cluster 1 no a; a has 6 cells
    labels = np.array([7, 7, 7, 7, 3, 3, 3, 3, 1, 1])
    fixed = np.array(["a"] * 4 + ["a", "a", "b", "b"] + ["b", "b"])
    got = impact.best_union(fixed, labels).set_index("class")
    # {7}: 4/6; {7, 3}: 6/8 = 0.75; {7, 3, 1}: 6/10
    assert got.loc["a", "jaccard"] == 0.75 and got.loc["a", "clusters_in_union"] == 2
    # class b: {1}: 2/4, {1, 3}: 4/6, {1, 3, 7}: 4/10
    assert got.loc["b", "jaccard"] == pytest.approx(4 / 6) and got.loc["b", "clusters_in_union"] == 2
    assert list(got.index) == ["a", "b"]


def test_thinned_cell_totals_equal_the_targets_exactly(toy):
    targets = np.asarray(toy.counts.sum(axis=1)).ravel() // 3
    targets[:5] += 10 ** 6                                                 # above the cell's total: left as it is
    targets[5] = 0
    got = impact.thin(toy.counts, targets, np.random.default_rng(0))
    totals = np.asarray(toy.counts.sum(axis=1)).ravel()
    assert np.array_equal(np.asarray(got.sum(axis=1)).ravel(), np.minimum(targets, totals))
    assert (got[:5] != toy.counts[:5]).nnz == 0 and got[5].nnz == 0
    assert (toy.counts - got).min() >= 0 and got.dtype == np.int32 and got.has_sorted_indices
    assert np.array_equal(got.toarray(), impact.thin(toy.counts, targets, np.random.default_rng(0)).toarray())


def test_seeds_are_honoured(stage, toy, scenarios):
    d, summary, out = stage
    cl = out["clustering"]
    # every matrix at every seed and resolution
    for (tag, kind, rep), frame in cl.groupby(["tag", "kind", "replicate"]):
        assert sorted(zip(frame.seed, frame.resolution)) == sorted(itertools.product(CLUSTER["seeds"], RESOLUTIONS)), tag
    # the unfiltered matrix: the labels of the pipeline at each seed, against seed 0 at the same resolution
    emb = classes.pipeline(toy.counts, toy.gene_ids, CLUSTER["seeds"], RESOLUTIONS)
    u = cl[cl.kind == "unfiltered"].set_index(["seed", "resolution"])
    for (s, r), lab in emb.labels.items():
        assert u.loc[(s, r), "n_clusters"] == len(np.unique(lab))
        assert u.loc[(s, r), "ari"] == adjusted_rand_score(emb.labels[(0, r)], lab)
    # a filtered matrix whose partitions at resolution 2.0 differ by seed: each seed's labels against the unfiltered seed 0
    emb = classes.pipeline(scenarios[SCRAMBLE], toy.gene_ids, CLUSTER["seeds"], (2.0,))
    base = classes.pipeline(toy.counts, toy.gene_ids, CLUSTER["seeds"][:1], (2.0,)).labels[(0, 2.0)]
    f = cl[(cl.kind == "filtered") & (cl.tag == SCRAMBLE) & (cl.resolution == 2.0)].set_index("seed")
    for s in CLUSTER["seeds"]:
        assert f.loc[s, "ari"] == adjusted_rand_score(base, emb.labels[(s, 2.0)])
    assert f.ari.nunique() > 1
    # a thinned matrix is the draw of its setting and replicate, down to the filtered totals
    m = _stage24(d / "again", toy, {SHIFT: scenarios[SHIFT]})
    load = partial(impact._load, m)
    targets = np.asarray(scenarios[SHIFT].sum(axis=1)).ravel()
    thinned, lost = impact.build(impact.Job("thinned", SHIFT, 0), toy.counts, load)
    assert lost is None and np.array_equal(np.asarray(thinned.sum(axis=1)).ravel(), targets)
    assert (thinned != impact.thin(toy.counts, targets, np.random.default_rng((impact.SEED, TAGS.index(SHIFT), 0)))).nnz == 0
    assert (impact.build(impact.Job("thinned", SHIFT, 1), toy.counts, load)[0] != thinned).nnz > 0
    filtered, lost = impact.build(impact.Job("filtered", SHIFT, 0), toy.counts, load)
    assert (filtered != scenarios[SHIFT]).nnz == 0 and (lost != sp.load_npz(m / "lost" / f"{SHIFT}.npz")).nnz == 0
    # the folds of the cross-validation are those of StratifiedKFold at SEED
    fixed = pd.read_parquet(d / "cells" / "cells.parquet")["class"].to_numpy()
    emb = classes.pipeline(toy.counts, toy.gene_ids)
    pred = np.empty(len(fixed), object)
    for train, test in StratifiedKFold(KNN_FOLDS, shuffle=True, random_state=impact.SEED).split(emb.pca, fixed):
        pred[test] = KNeighborsClassifier(n_neighbors=KNN_K).fit(emb.pca[train], fixed[train]).predict(emb.pca[test])
    assert np.array_equal(impact.cv_predict(emb.pca, fixed), pred)


def test_knn_vote_counts_each_cells_own_unfiltered_copy_among_the_references():
    # reference cells 0 to 3 are class a, 4 to 6 class b; the query is reference cell 3 itself, whose three nearest
    # references are itself and two b cells: with k 3 its own copy is not enough, with k 1 it is
    reference = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2], [5.3]])
    labels = np.array(["a", "a", "a", "a", "b", "b", "b"])
    assert impact.knn_vote(reference, labels, reference[[3]], k=3).tolist() == ["b"]
    assert impact.knn_vote(reference, labels, reference[[3]], k=1).tolist() == ["a"]
    assert impact.knn_vote(reference, labels, reference, k=1).tolist() == labels.tolist()


def test_projection_uses_the_unfiltered_genes_scaling_and_loadings(toy, stage):
    d = stage[0]
    model = dict(np.load(d / "cells" / "model.npz"))
    emb = classes.pipeline(toy.counts, toy.gene_ids)
    assert np.allclose(impact.project(emb.norm, model), emb.pca, atol=1e-4)
    # a filtered matrix: its own normalisation, the unfiltered variable genes, mean, sd and loadings
    f = sp.csr_matrix(np.random.default_rng(3).binomial(toy.counts.toarray(), 0.5).astype(np.int32))
    x = classes.normalise(f, toy.gene_ids).X[:, model["genes"]].toarray()
    assert np.array_equal(impact.project(classes.normalise(f, toy.gene_ids), model),
                          ((x - model["mean"]) / model["sd"]) @ model["loadings"])


def _runs(ju, jf, jt, nf=None, nt=None):
    """class_jaccard rows of one class and tag at one resolution: ju unfiltered and jf filtered per seed, jt thinned per
    replicate and seed; named shares 1 unless given."""
    seeds = CLUSTER["seeds"]
    rows = [("unfiltered", "unfiltered", 0, s, j, 1.0) for s, j in zip(seeds, ju)]
    rows += [(DELETE, "filtered", 0, s, j, n) for s, j, n in zip(seeds, jf, nf if nf is not None else [1.0] * len(jf))]
    nt = nt if nt is not None else [[1.0] * len(r) for r in jt]
    rows += [(DELETE, "thinned", r, s, j, n) for r, (js, ns) in enumerate(zip(jt, nt)) for s, j, n in zip(seeds, js, ns)]
    frame = pd.DataFrame(rows, columns=["tag", "kind", "replicate", "seed", "jaccard", "named_share"])
    return frame.assign(**{"class": "cone", "resolution": CLUSTER["resolution"], "clusters_in_union": 1,
                           "named_class": "cone"})[impact.COLUMNS["class_jaccard"]]


def test_collapse_call_rules_at_their_boundaries():
    n = len(CLUSTER["seeds"])
    one, ten = [1.0] * n, [[1.0] * n]
    stable = [JACCARD_STABLE + 1e-9] * n
    cases = [
        (dict(ju=one, jf=[JACCARD_DISSOLVED] * n, jt=ten), impact.APART),            # at most 0.5 in every filtered seed
        (dict(ju=one, jf=[JACCARD_DISSOLVED] * (n - 1) + [0.51], jt=ten), impact.UNCLEAR),
        (dict(ju=one, jf=[0.2] * n, jt=[[1.0] * (n - 1) + [JACCARD_STABLE]]), impact.DEPTH),
        (dict(ju=[1.0] * (n - 1) + [JACCARD_STABLE], jf=[0.2] * n, jt=ten), impact.UNSTABLE),
        (dict(ju=stable, jf=stable, jt=ten), impact.NEITHER),
        (dict(ju=one, jf=one, jt=ten, nf=[impact.NAME_KEPT] * n), impact.UNNAMED),
        (dict(ju=one, jf=one, jt=ten, nf=[0.0] * n, nt=[[1.0] * (n - 1) + [impact.NAME_KEPT]]), impact.UNCLEAR),
        (dict(ju=one, jf=one, jt=ten, nf=[0.0] * (n - 1) + [0.9]), impact.UNCLEAR),
        (dict(ju=one, jf=[JACCARD_STABLE] + [1.0] * (n - 1), jt=ten, nf=[0.0] * n), impact.UNCLEAR),
        (dict(ju=one, jf=[0.6] * n, jt=ten), impact.UNCLEAR),
    ]
    for kw, expected in cases:
        args = {k: np.asarray(v, float) for k, v in kw.items()}
        args.setdefault("nf", np.ones(n))
        args.setdefault("nt", np.ones_like(args["jt"]))
        assert impact.call(**args) == expected, (kw, expected)


def test_a_class_unstable_before_filtering_gets_no_call_and_is_listed_with_its_numbers():
    n = len(CLUSTER["seeds"])
    runs = _runs([1.0] * (n - 1) + [0.7], [0.2] * n, [[1.0] * n])
    calls, without = impact.collapse(runs, {DELETE: 1}, pd.DataFrame(columns=COLUMNS["markers"]), {})
    assert calls == []
    assert without == [{"class": "cone", "tag": None, "resolution": CLUSTER["resolution"], "reason": impact.UNSTABLE,
                        "jaccard_unfiltered": [1.0] * (n - 1) + [0.7]}]


def test_no_call_is_written_without_every_seed_and_every_thinned_run():
    n = len(CLUSTER["seeds"])
    none = pd.DataFrame(columns=COLUMNS["markers"])
    full = _runs([1.0] * n, [0.2] * n, [[1.0] * n] * 3)
    calls, without = impact.collapse(full, {DELETE: 3}, none, {})
    assert [c["call"] for c in calls] == [impact.APART] and without == []
    f, k = full.seed, full.kind
    for drop in ((k == "filtered") & (f == 4), (k == "unfiltered") & (f == 0),
                 (k == "thinned") & (full.replicate == 2) & (f == 9)):
        calls, without = impact.collapse(full[~drop], {DELETE: 3}, none, {})
        assert calls == [] and [w["reason"] for w in without] == [impact.MISSING]
    # three replicates are expected of a chosen setting: one is not enough
    calls, without = impact.collapse(_runs([1.0] * n, [0.2] * n, [[1.0] * n]), {DELETE: 3}, none, {})
    assert calls == [] and without[0]["reason"] == impact.MISSING and without[0]["tag"] == DELETE


def test_replicates_and_jobs():
    assert impact.replicates(CHOSEN) == THIN_REPLICATES_CHOSEN and impact.replicates(DELETE) == THIN_REPLICATES
    jobs = impact.jobs([CHOSEN, DELETE])
    assert jobs == [impact.Job("unfiltered", "unfiltered", 0), impact.Job("filtered", CHOSEN, 0),
                    impact.Job("filtered", DELETE, 0), *[impact.Job("thinned", CHOSEN, r) for r in range(3)],
                    impact.Job("thinned", DELETE, 0)]
    # all settings: 1 unfiltered, 14 filtered, 18 thinned (design 3.4, run time)
    every = impact.jobs(TAGS)
    assert len(every) == 33 and sum(j.kind == "thinned" for j in every) == 18


def test_per_cell_and_per_class(stage, toy, scenarios):
    d, summary, out = stage
    pc, pk = out["per_cell"], out["per_class"]
    ct = pd.read_parquet(d / "matrices" / "cell_totals.parquet").set_index(["tag", "cb"])
    tags = ["unfiltered", DELETE, SCRAMBLE, UNIFORM, SHIFT]
    assert pc.tag.unique().tolist() == tags
    for tag, frame in pc.groupby("tag", sort=False):
        m = toy.counts if tag == "unfiltered" else scenarios[tag]
        # every called cell once per tag, in the order of barcodes.tsv; UMIs and genes of the matrix
        assert frame.cb.tolist() == toy.barcodes.tolist()
        assert frame.umis.sum() == m.sum() and frame.umis.tolist() == np.asarray(m.sum(axis=1)).ravel().tolist()
        assert frame.genes.tolist() == m.getnnz(axis=1).tolist() and frame.genes0.tolist() == toy.counts.getnnz(axis=1).tolist()
        assert (frame.umis0 - frame.lost + frame.gained == frame.umis).all()
        t, u = ct.loc[tag], ct.loc["unfiltered"]
        assert np.allclose(frame.intronic, ((t.umis_N + t.umis_EN) / t.umis).to_numpy())
        assert np.allclose(frame.intronic0, ((u.umis_N + u.umis_EN) / u.umis).to_numpy())
        assert np.allclose(frame.rec_sense_removed_share, (1 - t.rec_EN_gx / u.rec_EN_gx).to_numpy())
        assert np.allclose(frame.rec_en_removed_share, (1 - (t.rec_E + t.rec_N) / (u.rec_E + u.rec_N)).to_numpy())
    assert summary["acceptance"]["passed"] is True
    floor = int(np.asarray(toy.counts.sum(axis=1)).min())
    assert summary["floor"] == floor
    for (cls, tag), frame in pc.groupby(["class", "tag"]):
        row = pk[(pk["class"] == cls) & (pk.tag == tag)].iloc[0]
        t = ct.loc[tag].loc[frame.cb]
        u = ct.loc["unfiltered"].loc[frame.cb]
        assert row.cells == len(frame) and row.umis_median0 == frame.umis0.median() and row.umis_median == frame.umis.median()
        assert (row.umis_q1, row.umis_q3) == pytest.approx(tuple(frame.umis.quantile([0.25, 0.75])))
        assert (row.genes_median0, row.genes_median) == (frame.genes0.median(), frame.genes.median())
        assert row.umis_lost_share == frame.lost.sum() / frame.umis0.sum()
        assert row.umis_gained_share == frame.gained.sum() / frame.umis0.sum()
        assert row.intronic == pytest.approx((t.umis_N + t.umis_EN).sum() / t.umis.sum())
        assert row.intronic0 == pytest.approx((u.umis_N + u.umis_EN).sum() / u.umis.sum())
        assert row.cells_below_floor == (frame.umis < floor).sum()
    assert pk[pk.tag == SHIFT].cells_below_floor.sum() > 0 and pk[pk.tag == "unfiltered"].cells_below_floor.sum() == 0


def test_marker_genes_are_the_panel_and_the_top_genes_of_the_wilcoxon_ranking(stage, toy, scenarios):
    _, summary, out = stage
    mk = out["markers"]
    assert sorted(mk.tag.unique()) == sorted([DELETE, SCRAMBLE, UNIFORM, SHIFT])
    one = mk[mk.tag == DELETE]
    assert set(one["class"]) == {"rod", "cone", "bipolar"}
    for cls, frame in one.groupby("class"):
        panel = frame[frame.origin == "panel"]
        assert panel.gene.tolist() == list(BY_NAME[cls].genes)
        ranked = frame[frame.origin == "ranking"]
        assert 0 < len(ranked) <= TOP_GENES and not set(ranked.gene) & set(BY_NAME[cls].genes)
        # the class's own planted genes lead the ranking
        assert ranked.gene.str.startswith(f"OWN_{cls}_").mean() > 0.5
    col = {s: i for i, s in enumerate(toy.symbols)}
    cone = toy.group == "cone"
    for gene in ("ARR3", "OPN1SW"):
        row = one[(one["class"] == "cone") & (one.gene == gene)].iloc[0]
        before = toy.counts[:, col[gene]].toarray().ravel()
        assert row.gene_id == toy.gene_ids[col[gene]] and row.umis0 == before[cone].sum() and row.umis == 0
        assert row.lost_share == 1.0 and row.detect0 == (before[cone] > 0).mean() and row.detect == 0.0
        x = classes.normalise(toy.counts, toy.gene_ids).X[:, col[gene]].toarray().ravel()
        lfc = np.log2((np.expm1(x[cone].mean()) + 1e-9) / (np.expm1(x[~cone].mean()) + 1e-9))
        assert row.lfc0 == pytest.approx(lfc, rel=1e-5) and row.lfc < 0 < row.lfc0
    # the genes losing more than MARKER_LOSS of their UMIs in their own class are listed
    listed = {(e["class"], e["tag"]): {g["gene"] for g in e["genes"]} for e in summary["marker_loss"]}
    assert listed[("cone", DELETE)] == set(BY_NAME["cone"].genes)
    assert all(g["lost_share"] > MARKER_LOSS for e in summary["marker_loss"] for g in e["genes"])
    assert set(mk[mk.lost_share > MARKER_LOSS][["class", "tag"]].itertuples(index=False, name=None)) == set(listed)


def test_confusion_and_retention_agree(stage):
    _, _, out = stage
    k, c = out["knn"], out["knn_confusion"]
    assert (c.cells > 0).all()
    for (tag, kind, rep), frame in c.groupby(["tag", "kind", "replicate"]):
        kk = k[(k.tag == tag) & (k.kind == kind) & (k.replicate == rep)].set_index("class")
        n = frame.groupby("class_fixed").cells.sum()
        assert (n == kk.cells.loc[n.index]).all()
        same = frame[frame.class_fixed == frame.class_projected].set_index("class_fixed").cells
        assert np.allclose(same.reindex(kk.index, fill_value=0) / kk.cells, kk.retention)
        assert kk.cv_balanced_accuracy.nunique() == 1 and kk.cv_balanced_accuracy.iloc[0] == pytest.approx(kk.cv_recall.mean())


def test_every_call_names_its_resolution_and_whether_it_holds_at_the_other_two(stage):
    _, summary, _ = stage
    main = [e for e in summary["calls"] if e["resolution"] == CLUSTER["resolution"]]
    assert {(e["class"], e["tag"]) for e in main} == {(c, t) for c in ("rod", "cone", "bipolar") for t in
                                                      (DELETE, SCRAMBLE, UNIFORM, SHIFT)}
    for e in main:
        other = e["other_resolutions"]
        assert sorted(other) == sorted(str(r) for r in CLUSTER["resolutions_alt"])
        assert e["holds_at_both"] == all(v == e["call"] for v in other.values())
        for r, v in other.items():
            hit = _call(summary, e["class"], e["tag"], float(r))
            assert v == (hit["call"] if hit else next(w["reason"] for w in summary["without_call"]
                                                       if w["class"] == e["class"] and w["resolution"] == float(r)))
    without = {(w["class"], w["reason"]) for w in summary["without_call"]}
    assert (classes.UNASSIGNED, impact.NOT_A_CLASS) in without and ("vascular", impact.NOT_ASKABLE) in without


COLUMNS = {
    "per_cell": ["cb", "class", "tag", "umis0", "umis", "lost", "gained", "genes0", "genes", "intronic0", "intronic",
                 "rec_sense_removed_share", "rec_en_removed_share", "knn_class"],
    "per_class": ["class", "tag", "cells", "umis_median0", "umis_median", "umis_q1", "umis_q3", "genes_median0", "genes_median",
                  "umis_lost_share", "umis_gained_share", "intronic0", "intronic", "cells_below_floor"],
    "markers": ["class", "gene", "gene_id", "origin", "tag", "umis0", "umis", "lost_share", "detect0", "detect", "lfc0", "lfc"],
    "knn": ["class", "tag", "kind", "replicate", "retention", "cv_recall", "cv_balanced_accuracy", "cells"],
    "knn_confusion": ["tag", "kind", "replicate", "class_fixed", "class_projected", "cells"],
    "clustering": ["tag", "kind", "replicate", "seed", "resolution", "n_clusters", "ari"],
    "class_jaccard": ["class", "tag", "kind", "replicate", "seed", "resolution", "jaccard", "clusters_in_union", "named_class",
                      "named_share"],
}


def test_output_schemas(stage):
    d, summary, out = stage
    assert impact.COLUMNS == COLUMNS
    for name, columns in COLUMNS.items():
        assert list(out[name]) == columns, name
    assert sorted(p.name for p in (d / "impact").iterdir()) == sorted([*(f"{n}.parquet" for n in COLUMNS), "summary.json"])
    js = json.loads((d / "impact" / "summary.json").read_text())
    assert js == json.loads(json.dumps(summary))
    for key in ("library", "parameters", "inputs", "settings", "cells", "classes", "floor", "matrices", "leiden_runs", "calls",
                "without_call", "marker_loss", "acceptance"):
        assert key in js, key
    p = js["parameters"]
    assert (p["knn_k"], p["knn_folds"], p["thin_replicates"], p["thin_replicates_chosen"], p["top_genes"], p["marker_loss"],
            p["jaccard_dissolved"], p["jaccard_stable"], p["name_kept"], p["seed"]) == (
        KNN_K, KNN_FOLDS, THIN_REPLICATES, THIN_REPLICATES_CHOSEN, TOP_GENES, MARKER_LOSS, JACCARD_DISSOLVED, JACCARD_STABLE,
        impact.NAME_KEPT, impact.SEED)
    assert p["cluster"] == json.loads(json.dumps(CLUSTER))
    assert js["matrices"] == {"unfiltered": 1, "filtered": 4, "thinned": 4} and js["leiden_runs"] == 9 * 30
    k = out["knn"]
    assert set(k.kind) == set(impact.KINDS) and k.groupby(["tag", "kind", "replicate"]).size().eq(4).all()
    j = out["class_jaccard"]
    assert len(j) == 9 * 30 * 4 and j.jaccard.between(0, 1).all()


def test_inputs_must_belong_together(stage, toy, tmp_path):
    d = stage[0]
    other = tmp_path / "cells"
    other.mkdir()
    cells = pd.read_parquet(d / "cells" / "cells.parquet")
    for name in ("model.npz", "classes.json"):
        (other / name).write_bytes((d / "cells" / name).read_bytes())
    cells.iloc[::-1].to_parquet(other / "cells.parquet", index=False)
    with pytest.raises(SystemExit, match="barcodes"):
        impact.run("1417", d / "matrices", other, tmp_path / "out")
    cells.assign(cluster=(cells.cluster + 1) % cells.cluster.nunique()).to_parquet(other / "cells.parquet", index=False)
    with pytest.raises(SystemExit, match="clusters of stage 25"):
        impact.run("1417", d / "matrices", other, tmp_path / "out", settings=[DELETE])
    with pytest.raises(SystemExit, match="not in the manifest"):
        impact.run("1417", d / "matrices", d / "cells", tmp_path / "out", settings=["c180_l11_m2_e"])


def test_stage26_script_and_the_worker_pool_write_the_same_files(toy, scenarios, tmp_path, monkeypatch, capsys):
    # a chosen setting (three thinned replicates) and a published one; the script, in this process with one worker, then
    # impact.run in a fresh interpreter with two workers: the same files byte for byte
    _stage24(tmp_path, toy, {CHOSEN: scenarios[DELETE], DELETE: scenarios[SCRAMBLE], SHIFT: scenarios[SHIFT]})
    _stage25(tmp_path, toy)
    spec = importlib.util.spec_from_file_location("stage26", REPO / "scripts" / "26_impact.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    called = []
    monkeypatch.setattr(stage, "require_passed", lambda *args: called.append(args) or {"passed": True})
    first = tmp_path / "first"
    monkeypatch.setattr(sys, "argv", ["26", "1417", "--cells", str(tmp_path / "cells"), "--matrices", str(tmp_path / "matrices"),
                                      "--recount", str(tmp_path / "rc"), "--out", str(first), "--workers", "1",
                                      "--settings", CHOSEN, DELETE])
    stage.main()
    assert called == [("1417", tmp_path / "rc", tmp_path / "matrices")]
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["library"] == "1417" and line["settings"] == [CHOSEN, DELETE] and line["matrices"] == 7
    assert line["acceptance_passed"] is True and "seconds" in line and "seconds_per_matrix" in line
    js = json.loads((first / "summary.json").read_text())
    assert js["settings"] == [CHOSEN, DELETE] and js["matrices"] == {"unfiltered": 1, "filtered": 2, "thinned": 4}
    assert _call(js, "cone", CHOSEN)["call"] == impact.UNNAMED and _call(js, "cone", DELETE)["call"] == impact.APART
    assert len(_call(js, "cone", CHOSEN)["jaccard"]["thinned"]) == THIN_REPLICATES_CHOSEN
    second = tmp_path / "second"
    code = (f"from pafbench import impact; impact.run('1417', {str(tmp_path / 'matrices')!r}, {str(tmp_path / 'cells')!r}, "
            f"{str(second)!r}, settings={[CHOSEN, DELETE]!r}, workers=2)")
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, cwd=REPO,
                       env=dict(os.environ, PYTHONPATH=str(REPO)))
    assert r.returncode == 0, r.stderr
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(p.name for p in second.iterdir()) and len(names) == 8
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
