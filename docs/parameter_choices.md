# Choosing settings for a new library

These numbers come from human retina libraries (10x 3' v4 and FLIGHT-seq, GRCh38-2024-A). Outside that scope treat them as a starting point. How the numbers are produced is in [`rationale.md`](rationale.md).

A setting is written `c<COVLEN>_l<MINSNRLEN>_m<MISM>_<e or i>`:

- `COVLEN`: reads up to this many bp upstream of an A stretch are removed. The last `COVLEN` bp of each transcript are protected.
- `MINSNRLEN`: shortest A stretch that counts.
- `MISM`: mismatches allowed inside the stretch.
- `e` searches exons only (polyAfilter's default). `i` searches introns too (`-i`).

## Where to start

| Your library | Start with | Internal priming removed | UMIs lost | Why |
|---|---|---|---|---|
| Whole cells, 10x 3', counted without introns | `c200_l10_m2_i` | 32% (1179) | 22% | Beats the exon-only pick on every measure in 1179: more removed, more real molecules kept, fewer UMIs lost. Intronic reads are not in the matrix, so removing them costs nothing. |
| Whole cells, FLIGHT-seq | `c200_l10_m2_i`, or `c300_l10_m1_e` for a lighter touch | 19% or 8% (BD67) | 17% or 7% | The burden is already less than half that of 10x, so there is less to gain. |
| Nuclei, counted with introns | `c250_l7_m1_e` | 9% (1417) | 16% | Leaves intronic reads alone. With introns (`c200_l10_m2_i`) the filter removes 44% but takes 55% of the UMIs. |
| Anything, no time to tune | `c300_l10_m1_e` | 3 to 8% | 5 to 8% | The polyAfilter README example. Almost never removes a real molecule, and removes little internal priming. |

Two things to know before filtering.

**No setting removes most of the internal priming.** Among the 189 grid settings per mode that keep at least 95% of the tail-primed molecules, the most sensitive one removes 44% of the internally primed molecules in nuclei, 48% in 10x cells and 27% in FLIGHT-seq cells. The filter looks for A stretches of a fixed minimum length and removes by genomic position, so it misses priming on interrupted A runs, reads of spliced molecules that sit in an upstream exon (the window does not follow splicing), and anything in the protected last `COVLEN` bp of a transcript.

**In nuclei the choice is uncomfortable.** Intronic reads make up about two thirds of the UMIs of a 10x nucleus, and 94% of the labeled intronic molecules are internally primed. Gene counting in nuclei is built on those reads. The exon-only setting leaves them alone and trims 16% of UMIs for a 9% gain. The intron setting halves the matrix. If the analysis is gene-level counting, weigh whether to filter nuclei at all; if it reads 3' ends as poly(A) sites, filtering is needed and the cost has to be accepted.

## Reading the tables

- **Sens**: share of internally primed (templated) molecules removed.
- **Spec**: share of tail-primed (non-templated) molecules kept.
- **UMI cost**: UMIs lost across called cells, as a share of UMIs before filtering.
- **ARI**: agreement between the clusters before and after filtering (1 means identical clusters), median over 10 seeds and 3 resolutions. It says how much the clusters moved, not whether they improved.
- **ARI, thinned**: the same after removing the same number of UMIs per cell at random. If the two ARI columns are close, the clusters moved because the cells got shallower, not because of which reads were removed.

The ARI ceiling, from clustering the unfiltered matrix again with another seed, is 0.69 (1417), 0.68 (1419), 0.70 (1179) and 0.89 (BD67).

## The two selected settings

Both come from one rule, fixed before the results: on 1417, the most sensitive setting with specificity of at least 0.95, one per mode. The other libraries are reported without retuning.

### mode-e: `c250_l7_m1_e` (exons only)

| Library | Sens | Spec | UMI cost | ARI | ARI, thinned |
|---|---|---|---|---|---|
| 1417 (10x nuclei) | 0.093 | 0.956 | 15.7% | 0.39 | 0.41 |
| 1419 (10x nuclei) | 0.094 | 0.962 | 15.8% | 0.48 | 0.49 |
| 1179 (10x cells) | 0.164 | 0.975 | 31.7% | 0.60 | 0.61 |
| BD67 (FLIGHT-seq cells) | 0.193 | 0.968 | 24.0% | 0.65 | 0.65 |

### mode-i: `c200_l10_m2_i` (with introns)

| Library | Sens | Spec | UMI cost | ARI | ARI, thinned |
|---|---|---|---|---|---|
| 1417 (10x nuclei) | 0.435 | 0.951 | 54.7% | 0.36 | 0.32 |
| 1419 (10x nuclei) | 0.440 | 0.958 | 55.3% | 0.42 | 0.42 |
| 1179 (10x cells) | 0.315 | 0.988 | 21.9% | 0.63 | 0.67 |
| BD67 (FLIGHT-seq cells) | 0.190 | 0.985 | 17.5% | 0.64 | 0.70 |

The UMI cost flips between nuclei and cells. The nuclei matrices count intronic reads (about two thirds of the UMIs per nucleus), and mode-i removes many of them. The cell matrices count exons only, so the intronic removals are free; and this setting asks for a longer A stretch than mode-e (10 against 7), so on balance it removes fewer exonic reads.

### The README example: `c300_l10_m1_e`

| Library | Sens | Spec | UMI cost | ARI | ARI, thinned |
|---|---|---|---|---|---|
| 1417 (10x nuclei) | 0.028 | 0.995 | 4.7% | 0.49 | 0.51 |
| 1419 (10x nuclei) | 0.028 | 0.997 | 4.8% | 0.55 | 0.56 |
| 1179 (10x cells) | 0.056 | 0.997 | 8.4% | 0.69 | 0.70 |
| BD67 (FLIGHT-seq cells) | 0.081 | 0.994 | 6.8% | 0.69 | 0.74 |

The three FLIGHT-seq nuclei libraries (BD_nuc_211, 213, 215) are in the burden comparison only. Their sensitivity and specificity are in `eval/selection.json` (0.06 and 0.97 to 0.98 at mode-e, 0.19 to 0.20 and 0.99 at mode-i). Their matrices count exonic reads only, at a median of roughly 100 UMIs per nucleus, which is too shallow for the clustering comparison to mean much.

## Tuning on your own library

The pipeline reads its libraries from `pafbench/config.py`. Add the BAM to `LIBRARIES` and the name to `NUCLEI` or `CELLS`, then:

1. BD Rhapsody only: convert the BAM to 10x-style tags (stage 01 is the template).
2. Stages 05, 06 and 07 find the reads with a known priming site, group them into molecules and write the poly(A) signal curve (`hexamer_curve.tsv`). Look at the curve first: the rate should fall from a high level at 0 to 4 A's to a low one at 14 or more.
3. Stage 12 builds the labeled molecules. Stage 13 flags all 396 settings in one pass.
4. Stage 17 writes `eval/selection.json` with sensitivity and specificity of every setting. The selection library is set at the top of `scripts/17_evaluate.py` (`SEL_LIB`).
5. Stages 23 to 26 add UMI cost and ARI for the 14 settings in `config.IMPACT_SETTINGS`.

The grid pass took 7 minutes on 1417 (97 M reads) and 23 minutes on 1179 (581 M reads) with 8 workers.

## If the poly(A) signal curve looks wrong

A flat curve, or a high rate at 14 or more A's, means the labels cannot be trusted as they are. Causes we know of:

- **Multi-mapped reads.** In the BD libraries, reads placed at loci where the genome encodes a poly(A) signal followed by A's raise the rate at 14 or more A's (0.50 in BD67 against 0.04 to 0.05 in 10x; 0.06 on uniquely mapped molecules of BD67). `hexamer_curve.tsv` includes them, so compute the curve again on uniquely mapped molecules (`mapq_min == 255` in `truth/molecules_labeled.parquet`). The burden stages leave them out already. Sensitivity and specificity barely move without them (`mapq_sensitivity` in `eval/selection.json`).
- **An incomplete annotation or a different reference build.** The label is read from the genome, so reads and labels must use the same build.
- **A library that is not oligo(dT)-primed.** The method does not apply.

## If you have long reads

Stage 16 reads priming sites from PacBio FLNC reads and compares them with the short-read molecules. Set `MIN_AFTER` in `config.py` to match how the reads were processed: BD FLIGHT-seq FLNC reads keep the bead sequence after the poly(A) (`MIN_AFTER = 30`), 10x Kinnex FLNC reads have it removed by `isoseq refine` (`MIN_AFTER = 0`). In our data 87 to 95% of the 1179 molecules have a long-read site within 10 nt, against 6 to 17% for sites moved 1 kb upstream.
