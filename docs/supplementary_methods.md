# Supplementary Methods

One-page summary for the FLIGHT-seq manuscript supplementary section. For the full narrative see `docs/rationale.md`; for the per-library decision table see `docs/parameter_choices.md`.

## Rationale

polyAfilter (Shulman and Elkon 2019) flags reads whose alignment 3' end falls over a genomic adenine run, under the assumption that oligo-dT primes on such runs more often than on real poly-A tails. The 20 nt downstream window and the 8/4 A-count cutoffs used throughout this work trace to that paper, adapted to this dataset by a hexamer-purity audit on 1417 (ratio of canonical poly-A signal hexamer presence between non-templated and templated sites below 3 at cutoffs looser than 8/4).

## Parameter choice without matched bulk

Svoboda et al. 2022 tune polyAfilter against bulk random-primed RNA-seq. We do not have matched bulk data for the retina organoid libraries. Three substitutes carried the choice:

1. **Hexamer purity ratio > 3x** between non-templated and templated sites at 8/4. All four libraries pass.
2. **Matched long-read corroboration** from two independent PacBio Kinnex runs on the same week-26 cells. Pooled label agreement with 1179 short-read labels: 99.5% (10x_FL, same-prep match), 99.4% (BD67, cross-prep match). Null (1 kb shift upstream) collapses pooled agreement to 44%, confirming the signal is not geometric.
3. **Prep-ordering sanity.** Nuclei are expected to carry more intronic A-stretches (pre-mRNA capture) than cells, and FLIGHT-seq on bead is expected to have lower internal-priming burden than oligo-dT in solution. Measured ordering (82% / 48.5% / 26.8% templated across nuclei, 10x cells, FLIGHT-seq cells) matches.

## Chosen settings

Three settings reported:

| Tag | covlen | lmax | mmax | introns | What it does |
|---|---|---|---|---|---|
| `c250_l7_m1_e` (mode-e) | 250 | 7 | 1 | excluded | Sensitivity pick for cells; moderate specificity gain on nuclei |
| `c200_l10_m2_i` (mode-i) | 200 | 10 | 2 | included | Specificity pick for cells; eats intronic reads on nuclei |
| `c300_l10_m1_e` (paper default) | 300 | 10 | 1 | excluded | Shulman and Elkon's starting point |

Per-library performance in Table S1 (or `docs/parameter_choices.md`). ARI reported as the median across 10 seeds and 3 Leiden resolutions.

## Long-read acceptance

Stage 16 of the benchmark accepts a long-read record if its terminal A run on the aligned sense strand has length >= 10 (one non-A allowed) AND is followed by `MIN_AFTER` or more non-A clipped bases. `MIN_AFTER` is per-library because FLNC conventions differ: BD-style FLIGHT-seq FLNC retains the bead tail (`MIN_AFTER=30`), 10x Kinnex FLNC has the PR1+CB+UMI primer pre-stripped by `isoseq refine` into the CB/XM tags (`MIN_AFTER=0`). On BD67 chr1 this accepts 93% of reads; on 10x_FL chr1 it accepts 84%.

## Dependencies

polyAfilter vendored at commit `5ce0098`. All analysis in Python 3.11 with pinned package versions (`uv.lock`, 90 packages). Scanpy with `leiden_flavor=igraph`, `leiden_n_iterations=2`, `n_pcs=30`, `k=15`, `blas_threads=1` (the thread count affects PCA coordinates).

## Reproduction

Full pipeline from raw BAMs, 12-core / 128 GB machine: approximately 8 h wall-clock. Deterministic given the pinned environment and the per-stage seeds.

## Caveats

Scope limits apply (`docs/rationale.md`): tested on retina organoid, human, oligo-dT 3' chemistries, GRCh38-2024-A reference. Non-human organisms and non-oligo-dT chemistries are outside scope.
