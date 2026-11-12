"""Stage 29a: fit the 2x2 interaction model on the RAW (uncorrected) templated share, chemistry (10x, BD) x material
(cells, nuclei), with each library as one replicate within its cell (addendum 2026-10-31, section 3.4).

Rows (7 total): 1417 and 1419 (10x nuclei, chem=10x, mat=nuclei), 1179 (10x cells), BD67 (BD cells), BD_nuc_211/213/215
(BD nuclei). 1416 is in LIBRARIES but not in NUCLEI and does not enter here (addendum decision outside this scope).

Response: logit of the per-library RAW templated share, "all" region, on called cells. The raw share is templated
molecules divided by all uniquely mapped labeled molecules on called cells (truth/molecules_labeled.parquet, in_cell,
mapq_min 255); this is
`share_label` in `burden/templated_share_corrected.parquet`, NOT `share_corrected`. The ambiguous-label correction from
stage 29 is a per-library linear offset in the logit domain and would propagate through the 2x2 fit, but applying it
during the gene-cluster bootstrap requires the per-cluster hexamer-rate inputs stage 29 uses; stage 29a keeps the simpler
raw response and labels it that way. Expect the fitted shares to sit above the corresponding corrected shares in
`templated_share_corrected.parquet` by a small per-library amount.

Design matrix: intercept, chemistry (10x=0, BD=1), material (cells=0, nuclei=1), chemistry x material. OLS per bootstrap
draw: gene-cluster bootstrap per library (evaluate.gene_clusters + resample_counts) with an INDEPENDENT seed per library
derived from --seed via numpy.random.SeedSequence.spawn, so two libraries with identical cluster counts no longer share
their bootstrap draws by array position. Point estimates run on the observed share (no resampling). CIs: 2.5th and 97.5th
percentiles of the N draws.

Share scale is also reported for the three contrasts: the main effects and the interaction expressed as absolute
differences in share. The interaction on the share scale is (nuclei_BD - cells_BD) - (nuclei_10x - cells_10x), averaged
over the within-cell replicates. On the logit scale it is the chemistry:material coefficient.

Writes under --out (default OUT/burden): interaction.parquet (per-draw and point coefficient draws, long-form),
interaction_cells.parquet (per-cell means with CIs on share and logit scales), interaction.json (headline).

Descriptive only. With one BD-cells library (BD67) the BD-cells cell has no within-cell variance; the interaction CI leans
on the three BD-nuclei samples and the two 10x-nuclei libraries (addendum section 3.4). These CIs describe gene-resampling
uncertainty CONDITIONAL on the observed libraries; they do NOT estimate biological-replicate variability across donors or
batches.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pafbench.config import BD_NUC, CELLS, NUCLEI, OUT
from pafbench.evaluate import gene_clusters, interval, resample_counts
from pafbench.truth import in_population, uniquely_mapped

COLUMNS = ["gene", "ref", "strand", "site", "label", "in_cell", "mapq_min"]
ROWS = [
    ("1417", "10x", "nuclei"),
    ("1419", "10x", "nuclei"),
    ("1179", "10x", "cells"),
    ("BD67", "BD", "cells"),
    ("BD_nuc_211", "BD", "nuclei"),
    ("BD_nuc_213", "BD", "nuclei"),
    ("BD_nuc_215", "BD", "nuclei"),
]
N_DRAWS = 1000
SEED = 0


def logit(p):
    return np.log(p / (1.0 - p))


def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def per_library_shares(mol, n_draws, seed):
    """[n_draws + 1] templated shares. Position 0 is the point estimate; 1..n_draws are gene-cluster bootstrap draws."""
    clusters = gene_clusters(mol)
    names, code = np.unique(clusters, return_inverse=True)
    templ_mask = (mol.label.to_numpy() == "templated")
    total = np.bincount(code, minlength=len(names)).astype(float)
    templ = np.bincount(code[templ_mask], minlength=len(names)).astype(float)
    W = resample_counts(len(names), n_draws, seed).astype(float)
    boot_num = W @ templ
    boot_den = W @ total
    point = templ.sum() / total.sum()
    boot = np.divide(boot_num, boot_den, out=np.full_like(boot_num, np.nan), where=boot_den > 0)
    return np.concatenate([[point], boot])


def design_matrix(rows):
    chem = np.array([1.0 if r[1] == "BD" else 0.0 for r in rows])
    mat = np.array([1.0 if r[2] == "nuclei" else 0.0 for r in rows])
    return np.column_stack([np.ones(len(rows)), chem, mat, chem * mat])


def fit_ols(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def cell_predictions(coef):
    """Four fitted logit means for (chem, mat) in {(10x, cells), (10x, nuclei), (BD, cells), (BD, nuclei)}."""
    b0, bc, bm, bcm = coef
    return {
        "10x_cells": b0,
        "10x_nuclei": b0 + bm,
        "BD_cells": b0 + bc,
        "BD_nuclei": b0 + bc + bm + bcm,
    }


def summary_row(name, values, scale):
    v = np.asarray(values, dtype=float)
    point = v[0]
    lo, hi, n = interval(v[1:])
    return {"term": name, "scale": scale, "point": float(point), "lo": lo, "hi": hi, "n_finite": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=OUT)
    ap.add_argument("--out", type=Path, default=OUT / "burden")
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    t0 = time.time()

    allowed = set(NUCLEI + CELLS + BD_NUC)
    rows = [r for r in ROWS if r[0] in allowed]
    missing = [r for r in ROWS if r[0] not in allowed]
    if missing:
        raise SystemExit(f"libraries not registered in config: {missing}")

    # per-library shares: shape (n_libs, n_draws + 1). Independent seed per library so two libraries with equal cluster
    # counts do not share bootstrap draws by array position (addendum section 3.4 review, 2026-11-03).
    ss = np.random.SeedSequence(args.seed)
    child_seeds = ss.spawn(len(rows))
    shares = []
    n_labeled = {}
    for i, (lib, _, _) in enumerate(rows):
        p = args.base / lib / "truth" / "molecules_labeled.parquet"
        if not p.exists():
            raise SystemExit(f"{lib}: missing {p}")
        mol = pd.read_parquet(p, columns=COLUMNS)
        mol = in_population(uniquely_mapped(mol), "cells")
        s = per_library_shares(mol, args.n_draws, child_seeds[i])
        shares.append(s)
        n_labeled[lib] = int(len(mol))
    shares = np.vstack(shares)

    # logit; drop draws where any library hit 0/1 (undefined logit)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_all = logit(shares)
    finite = np.all(np.isfinite(y_all), axis=0)
    n_kept = int(finite.sum())
    X = design_matrix(rows)

    # fit per draw (including point at column 0)
    coefs = np.empty((4, y_all.shape[1]))
    coefs[:] = np.nan
    for k in np.where(finite)[0]:
        coefs[:, k] = fit_ols(X, y_all[:, k])
    cells = {}
    for name in ("10x_cells", "10x_nuclei", "BD_cells", "BD_nuclei"):
        preds = np.array([cell_predictions(coefs[:, k])[name] for k in range(coefs.shape[1])])
        cells[name] = preds

    # share-scale cell means (back-transform the logit prediction) and the three contrasts on both scales
    cells_share = {name: inv_logit(v) for name, v in cells.items()}
    chem_logit = (cells["BD_cells"] + cells["BD_nuclei"]) / 2 - (cells["10x_cells"] + cells["10x_nuclei"]) / 2
    mat_logit = (cells["10x_nuclei"] + cells["BD_nuclei"]) / 2 - (cells["10x_cells"] + cells["BD_cells"]) / 2
    inter_logit = (cells["BD_nuclei"] - cells["BD_cells"]) - (cells["10x_nuclei"] - cells["10x_cells"])
    chem_share = (cells_share["BD_cells"] + cells_share["BD_nuclei"]) / 2 - (cells_share["10x_cells"] + cells_share["10x_nuclei"]) / 2
    mat_share = (cells_share["10x_nuclei"] + cells_share["BD_nuclei"]) / 2 - (cells_share["10x_cells"] + cells_share["BD_cells"]) / 2
    inter_share = (cells_share["BD_nuclei"] - cells_share["BD_cells"]) - (cells_share["10x_nuclei"] - cells_share["10x_cells"])

    # per-draw frame for downstream (one row per draw per term)
    terms = {
        "chemistry": (chem_logit, chem_share),
        "material": (mat_logit, mat_share),
        "chemistry:material": (inter_logit, inter_share),
    }
    frames = []
    for term, (vl, vs) in terms.items():
        frames.append(pd.DataFrame({"term": term, "draw": np.arange(len(vl)), "logit": vl, "share": vs}))
    for name, v in cells.items():
        frames.append(pd.DataFrame({"term": name, "draw": np.arange(len(v)), "logit": v, "share": cells_share[name]}))
    long = pd.concat(frames, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    long.to_parquet(args.out / "interaction.parquet", index=False)

    # per-cell table (observed-share mean of its library replicates + fitted logit cell with CIs)
    cell_table = []
    for cell_name, (chem, mat) in (
        ("10x_cells", ("10x", "cells")),
        ("10x_nuclei", ("10x", "nuclei")),
        ("BD_cells", ("BD", "cells")),
        ("BD_nuclei", ("BD", "nuclei")),
    ):
        members = [r[0] for r in rows if r[1] == chem and r[2] == mat]
        obs = [shares[i, 0] for i, r in enumerate(rows) if r[1] == chem and r[2] == mat]
        d = summary_row(cell_name, cells_share[cell_name], "share")
        d_logit = summary_row(cell_name, cells[cell_name], "logit")
        cell_table.append({
            "cell": cell_name, "chemistry": chem, "material": mat, "members": members, "n_members": len(members),
            "observed_share_mean": float(np.mean(obs)), "observed_shares": [float(x) for x in obs],
            "fitted_share": d["point"], "fitted_share_lo": d["lo"], "fitted_share_hi": d["hi"],
            "fitted_logit": d_logit["point"], "fitted_logit_lo": d_logit["lo"], "fitted_logit_hi": d_logit["hi"],
        })
    pd.DataFrame(cell_table).to_parquet(args.out / "interaction_cells.parquet", index=False)

    headline = {
        "libraries": {r[0]: {"chemistry": r[1], "material": r[2], "n_labeled": n_labeled[r[0]],
                             "share_point": float(shares[i, 0])} for i, r in enumerate(rows)},
        "n_draws": args.n_draws, "n_draws_kept": n_kept, "seed": args.seed,
        "cells": [{"cell": c["cell"], "members": c["members"], "fitted_share": c["fitted_share"],
                   "share_ci": [c["fitted_share_lo"], c["fitted_share_hi"]],
                   "fitted_logit": c["fitted_logit"], "logit_ci": [c["fitted_logit_lo"], c["fitted_logit_hi"]],
                   "observed_shares": c["observed_shares"]} for c in cell_table],
        "contrasts": {
            "chemistry": {"share": summary_row("chemistry", chem_share, "share"),
                          "logit": summary_row("chemistry", chem_logit, "logit")},
            "material": {"share": summary_row("material", mat_share, "share"),
                         "logit": summary_row("material", mat_logit, "logit")},
            "interaction": {"share": summary_row("chemistry:material", inter_share, "share"),
                            "logit": summary_row("chemistry:material", inter_logit, "logit")},
        },
        "seconds": round(time.time() - t0, 2),
    }
    (args.out / "interaction.json").write_text(json.dumps(headline, indent=1))
    print(json.dumps({
        "cells_share_point": {c["cell"]: round(c["fitted_share"], 4) for c in cell_table},
        "chemistry_share_point": round(headline["contrasts"]["chemistry"]["share"]["point"], 4),
        "material_share_point": round(headline["contrasts"]["material"]["share"]["point"], 4),
        "interaction_share_point": round(headline["contrasts"]["interaction"]["share"]["point"], 4),
        "interaction_logit_point": round(headline["contrasts"]["interaction"]["logit"]["point"], 4),
        "interaction_share_ci": [round(headline["contrasts"]["interaction"]["share"]["lo"], 4),
                                 round(headline["contrasts"]["interaction"]["share"]["hi"], 4)],
        "seconds": headline["seconds"],
    }))


if __name__ == "__main__":
    main()
