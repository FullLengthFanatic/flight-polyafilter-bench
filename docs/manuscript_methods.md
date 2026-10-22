# Methods paragraph (drop-in for the FLIGHT-seq manuscript)

The three sections below are drafted to be lifted verbatim into the manuscript. Scope limits noted in `docs/rationale.md` apply.

## Short Methods (one paragraph, suitable for main text)

Internal-priming artefacts were quantified and filtered with polyAfilter (Shulman and Elkon 2019), reimplemented in a reproducibility benchmark (Picelli 2026, [10.5281/zenodo.23562409](https://doi.org/10.5281/zenodo.23562409)) that matches the upstream tool byte-for-byte on removed-read sets at three sweep settings. Molecules were labelled templated (internal priming, 8 or more adenines within 20 nt downstream of the alignment 3' end), non-templated (4 or fewer), or ambiguous (5-7), following the convention of Shulman and Elkon 2019 adapted to this dataset. Filter settings were chosen per library population from a 14-setting sweep against hexamer purity (ratio of poly-A signal hexamer occurrence between templated and non-templated sites > 3) and matched long-read corroboration from PacBio Kinnex on the same cells (pooled label agreement 99.5% for 10x_FL vs 1179, 99.4% for BD67 vs 1179). Reported numbers use `c250_l7_m1_e` for the sensitivity setting and `c200_l10_m2_i` for the specificity setting on cells, and the paper default `c300_l10_m1_e` on nuclei to preserve intronic reads. Long-read structural buckets were classified with tecap (Picelli 2026, [github.com/FullLengthFanatic/tecap](https://github.com/FullLengthFanatic/tecap)).

## Long Methods (two paragraphs, suitable for supplementary methods)

**Filter.** polyAfilter (vendored at commit 5ce0098) was run at a 14-setting grid of `covlen` x `lmax` x `mmax` x introns. Molecule labels were assigned by the 20 nt downstream A-count on the reference genome: `templated_min=8`, `nontemplated_max=4`, ambiguous for the 5-7 band. The 20 nt window and the 8/4 cutoffs trace to Shulman and Elkon 2019 adapted to this dataset by a hexamer-purity audit on 1417 (ratio of AATAAA/ATTAAA plus 15 near-canonical variants between non-templated and templated sites falls below 3 at cutoffs looser than 8/4). We did not have matched bulk random-primed RNA-seq, which Svoboda et al. 2022 use as ground truth; three substitutes carried the parameter choice. First, the hexamer-purity ratio above 3x on 1417 and 1419 at 8/4 but not at 14/8. Second, two independent PacBio Kinnex long-read runs on the same week-26 cells as 1179 (10x_FL, the Kinnex 8-fold long-read of the 1179 library itself; BD67, the FLIGHT-seq Rhapsody long-read of the same pool) agreeing with the short-read labels at 99.5% and 99.4% pooled. Third, the expected ordering of templated share across preps (nuclei > cells > FLIGHT-seq cells) matching the measured ordering (82%, 48.5%, 26.8%).

**Impact.** Filter impact was evaluated per setting by sensitivity and specificity against the templated / non-templated truth, UMI cost on called cells (sum of UMIs lost over sum of pre-filter UMIs across cells), and clustering ARI (median across 10 seeds and 3 Leiden resolutions, by scanpy with `leiden_flavor=igraph`, `n_pcs=30`, `k=15`). Full numbers for every (library, setting) combination are archived with the benchmark at Zenodo DOI [10.5281/zenodo.23562409](https://doi.org/10.5281/zenodo.23562409), and every figure digit traces to a source function and column in `report/numbers.json`.

## Data availability

Short-read libraries 1417, 1419, 1416, 1179, and BD67 are deposited at GEO accession GSE_TODO. PacBio Kinnex long-read libraries BD67 long, BD70, 10x_FL_Kinnex, and 10x_SC_Kinnex are deposited at SRA under PRJNA_TODO. Reference annotation is GRCh38-2024-A (10x Genomics) with the PolyASite 2.0 atlas release `GRCh38.GENCODE_42`.

## Code availability

Benchmark code and documentation: [github.com/FullLengthFanatic/flight-polyafilter-bench](https://github.com/FullLengthFanatic/flight-polyafilter-bench), archived at Zenodo DOI [10.5281/zenodo.23562409](https://doi.org/10.5281/zenodo.23562409).

Companion tools: [flightcollapse](https://github.com/FullLengthFanatic/flightcollapse) for long-read collapsing and barcode recovery, [scNoiseMeter](https://github.com/FullLengthFanatic/scNoiseMeter) for single-cell noise quantification, [tecap](https://github.com/FullLengthFanatic/tecap) for 3' end structural bucket classification.
