# Supplementary Methods

One-page summary for the supplementary section of the FLIGHT-seq manuscript. The full account is in `docs/rationale.md`, the per-library tables in `docs/parameter_choices.md`.

## Rationale

polyAfilter (Svoboda, Frost and Bosco 2022, NAR Genomics and Bioinformatics) removes reads that lie within `COVLEN` bp upstream of a genome-encoded A stretch of at least `MINSNRLEN` nt (up to `MISM` mismatches) inside an expressed transcript, on the assumption that oligo(dT) primed there. The paper tunes these parameters against random-primed bulk RNA-seq of the same sample. No such data exist for the retina libraries, so the reference was taken from the libraries themselves.

## Reference labels without bulk data

1. **Molecules with a known priming site.** A read whose poly(A) was clipped ends at the priming site of its molecule.
2. **Label from the genome.** Adenines in the 20 nt downstream of the site: 8 or more is templated (internal priming), 4 or fewer is non-templated (tail priming), 5 to 7 is ambiguous. The cut-offs were set on library 1417 from the rate of the poly(A) signal hexamer by adenine count (0.77 at 0 to 4 adenines, 0.18 at 8, 0.02 at 18 to 20) and fixed before any filter result was examined.
3. **Test on the rest of the molecule.** polyAfilter is scored on the other reads of each labelled molecule, at the molecule level, in called cells.

Checks that do not use the adenine count: the hexamer rate per adenine band in every library, spike-ins with known 3' ends in the two 10x nuclei libraries, and long reads. A long-read priming site lies within 10 nt of 87% (BD67 long reads, same cell pool) and 95% (10x_FL, a week-29 organoid 10x library) of the labelled 1179 molecules, against 6% and 17% for sites moved 1 kb upstream.

## Settings

Selection rule, fixed in advance: on 1417, per mode, the most sensitive of 189 grid settings with specificity of at least 0.95. Other libraries are reported without retuning.

| Setting | COVLEN | MINSNRLEN | MISM | Introns | Origin |
|---|---|---|---|---|---|
| `c250_l7_m1_e` (mode-e) | 250 | 7 | 1 | no | selected by the rule |
| `c200_l10_m2_i` (mode-i) | 200 | 10 | 2 | yes | selected by the rule |
| `c300_l10_m1_e` | 300 | 10 | 1 | no | example in the polyAfilter README |

Sensitivity, specificity, UMI cost and ARI per library are in Table S1 (`docs/parameter_choices.md`). ARI is the median over 10 seeds and 3 Leiden resolutions against the unfiltered clustering, reported with its ceiling and a random-thinning control.

## Validation of the reimplementation

All settings were flagged by a reimplementation that reuses polyAfilter's A-stretch scanner (commit `5ce0098`). Its removed records are identical to those of the original tool in 27 runs: ten settings on 1417, eight on 1419, eight on chr21 and chr22 of 1179, one remove-all control.

## Long-read acceptance

A long read is used when its terminal adenine run (aligned plus clipped, one other base allowed) is at least 10 nt and is followed by at least `MIN_AFTER` nt of clipped sequence. `MIN_AFTER` is 30 for BD67, whose FLNC reads keep the bead sequence after the poly(A), and 0 for 10x_FL, where `isoseq refine` has removed it. This accepts 88% of BD67 reads and 84% of 10x_FL reads. A library is analysed only if its non-templated sites carry the hexamer at least 3 times as often as its templated sites (5.3 in BD67, 5.6 in 10x_FL).

## Environment

Python 3.11 with pinned package versions (`uv.lock`). Scanpy with the igraph Leiden flavour, 2 iterations, 30 principal components, 15 neighbours and 1 BLAS thread (the thread count changes the PCA coordinates). Seeds are fixed per stage.

## Caveats

Human retina, GRCh38-2024-A, 10x 3' v4 and FLIGHT-seq. The labelled molecules are a subset of each library. Nuclei are adult donor retina and cells are retinal organoids, so comparisons between them are descriptive. The templated shares count uniquely mapped molecules only (STAR MAPQ 255): multi-mapped reads would raise the share of the FLIGHT-seq libraries (25.9% instead of 19.4% in BD67) without the hexamer profile of internal priming. Sensitivity and specificity are scored on all labelled molecules and change by 0.02 at most without the multi-mapped ones.
