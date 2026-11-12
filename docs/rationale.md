# How the benchmark works, and why

polyAfilter was tuned in its paper against random-primed bulk RNA-seq of the same sample. We had no such data for the retina libraries. This page explains what we used in its place, and the reason for each choice. Numbers are for called cells unless a line says otherwise.

## Internal priming

An oligo(dT) primer is meant to bind the poly(A) tail. It also binds A-rich stretches inside a transcript: in an intron of a pre-mRNA, in a coding exon, in a 3' UTR upstream of the real end. The read that follows carries a barcode and a UMI like any other. It looks like a 3' end and is not one.

## What polyAfilter does

polyAfilter (Svoboda, Frost and Bosco 2022) works from the genome, not from the reads:

1. It finds every stretch of A's in the genome that is at least `MINSNRLEN` long, allowing up to `MISM` mismatches inside it.
2. It draws a window of `COVLEN` bp upstream of each stretch. A read primed at the stretch should start inside that window.
3. It removes every read that touches a window inside an expressed transcript. The last `COVLEN` bp of each transcript are protected, because reads from the real tail sit there.
4. By default only exons are searched. With `-i` the introns are searched too.

A setting is written `c250_l7_m1_e`: `COVLEN` 250, `MINSNRLEN` 7, `MISM` 1, exons only (`e`) or with introns (`i`). The filter never looks at an individual read's 3' end. It removes by position, so it also removes real reads that happen to lie upstream of an A stretch. That is why accuracy has to be measured.

## Where the reference labels come from

Some reads run into the poly(A). Cell Ranger clips those A's and writes their number in the `pa` tag, so for these reads the end of the alignment is the priming site. (BD Rhapsody BAMs have no such tag; stages 01 and 01a add it from the soft clip.)

We then ask one question of the genome: are the A's that the read ran into encoded there?

- **8 or more A's in the 20 nt after the site: templated.** The primer bound A's that are in the genome. Internal priming.
- **4 or fewer: non-templated.** The A's came from a poly(A) tail. A real 3' end.
- **5 to 7: ambiguous.** Reported separately, left out of sensitivity and specificity.

The window is 20 nt, the length of the shortest clipped poly(A) that gets a `pa` tag.

### Why 8 and 4

Real cleavage sites have a poly(A) signal (AATAAA or ATTAAA) a few bases upstream; internal priming sites do not. So the share of sites with a signal 5 to 35 nt upstream tells the two apart without using A content. On library 1417 (all barcodes):

| A's in the 20 nt after the site | Share with a poly(A) signal |
|---|---|
| 0 to 4 | 0.77 |
| 5 | 0.54 |
| 6 | 0.42 |
| 7 | 0.27 |
| 8 | 0.18 |
| 9 | 0.11 |
| 10 to 13 | 0.06 to 0.08 |
| 14 to 20 | 0.01 to 0.06 |

The rate falls from a tail-like level (0.77) to a background level (0.02 at 18 to 20 A's). The cut-offs sit where the curve has covered a quarter of that distance from either end: the rate stays below 0.21 from 8 A's on, and above 0.58 up to 4 A's (`pafbench/labels.py:choose_cutoffs`). They were fixed on 1417 before any polyAfilter result was looked at, and the same pair is used in every library. A sweep (templated from 7 to 10, non-templated from 2 to 6) is in `eval/selection.json`.

### How pure the labels are

In nuclei the templated class is almost all internal priming: about 3% of it are real sites that happen to be followed by A's in the genome. In whole cells real 3' ends are far more common, so the same A count is weaker evidence: about 19% of the templated class of 1179 are real sites, most of them in the 8 to 9 band (66% of that band). The ACTB 3' end, followed by five encoded A's, is one example.

We therefore report two numbers per library: the raw templated share, and a corrected share that subtracts the estimated real sites (stage 29, from the poly(A) signal rate of each A band against the rate at 14 or more A's).

### Multi-mapped reads are left out of the burden

A read that fits several places in the genome still gets one primary alignment, and that place can be the wrong one. In the FLIGHT-seq libraries 13 to 17% of the labeled molecules rest on such reads (4 to 6% in 10x). Most of them carry the templated label, yet they do not look like internal priming: in BD67 the multi-mapped molecules with 14 or more A's have a poly(A) signal in 88% of cases, against 6% for the uniquely mapped ones. We read them as real 3' ends placed at a copy of their locus where the genome supplies the A's.

The burden figures therefore count only molecules whose poly(A) reads map uniquely (`mapq_min` 255 in `truth/molecules_labeled.parquet`; stages 18, 29, 29a, 31, 31b and the figures). With the multi-mapped molecules included, the templated share of BD67 would be 25.9% instead of 19.4%, and that of the FLIGHT-seq nuclei 3 to 4 points higher. The 10x libraries move by 1.3 points at most. The corrected share of BD67 moves further, from 21.9% to 11.7%: the multi-mapped molecules also sat in the band of 14 or more A's, which the correction takes as its reference for pure internal priming (poly(A) signal rate 0.50 with them, 0.06 without).

Accuracy is still scored on all labeled molecules. Leaving the multi-mapped ones out changes sensitivity by 0.02 at most (BD67, mode-e: 0.193 to 0.209) and specificity by 0.001, and the rule picks the same two settings (`mapq_sensitivity` in `eval/selection.json`). The cut-offs do not move either: the rule gives 8 and 4 on the uniquely mapped molecules of 1417 as well.

## How the filter is scored

- **Test reads.** The other reads with the same cell barcode and UMI, up to 1,000 nt upstream of the site (measured along the exons when both lie in exons). polyAfilter judges these without seeing the site. The `pa` read alone would be too easy a test: it ends at the site by construction.
- **Unit.** The molecule. It counts as removed when all its reads are removed, which is what changes a UMI count.
- **Population.** Molecules from barcodes that Cell Ranger called as cells. Other barcodes are reported for comparison only; in the nuclei libraries they are a different population (45% of their labeled molecules lie in 3' UTRs, against 11% in called nuclei of 1417).
- **Sensitivity.** Share of templated molecules removed. **Specificity.** Share of non-templated molecules kept.
- **Uncertainty.** Bootstrap over genes, 1,000 resamples.

### The selection rule

Fixed before the results: on 1417, in each mode, take the setting with the highest sensitivity among those with specificity of at least 0.95. Then report that setting on every other library without retuning. 1419 is a second donor, so it is a donor-level hold-out.

| Mode | Setting | Sensitivity / specificity on 1417 | On 1419 (hold-out) |
|---|---|---|---|
| Exons only | `c250_l7_m1_e` | 0.093 / 0.956 | 0.094 / 0.962 |
| With introns | `c200_l10_m2_i` | 0.435 / 0.951 | 0.440 / 0.958 |

The docs call these mode-e and mode-i. The example in the polyAfilter README, `c300_l10_m1_e`, is carried along as a third row everywhere; it is an example, not a tuned value. The paper's own tuned settings (275/9/1 and 180/11/2 for PBMC, 375/3/2 for mouse cortex nuclei) are part of the grid too.

## Checks that do not rest on the A count

**Poly(A) signal.** Used above to place the cut-offs, and reported per A band and library as a running check on label purity.

**Long reads.** Two PacBio Kinnex libraries give priming sites from other molecules: BD67 long reads (same organoid cell pool as 1179, FLIGHT-seq) and 10x_FL (a 10x library of a week-29 organoid, a different pool). Of the 1179 short-read molecules, 95% have a 10x_FL site within 10 nt and 87% a BD67 site; with the sites moved 1 kb upstream the figures drop to 17% and 6%. So the priming sites recur across libraries and platforms. Where a site is matched the two labels agree in 99.4 to 99.5% of cases, which is expected and not independent evidence: both labels are read from the same genome sequence. Long reads are not ground truth either, since their oligo(dT) can also prime internally. A long-read library enters the analysis only if its tail-primed sites carry the poly(A) signal at least 3 times as often as its internally primed sites (5.3 times in BD67, 5.6 in 10x_FL).

**Spike-ins.** 1417 and 1419 contain SIRV Set 4 and ERCC RNA, whose 3' ends are known. They test the chemistry on free RNA, and their results are in `docs/m2_results.md`.

## The reimplementation

The original tool takes about two hours per setting on 1417, after a one-off setup of about 15 hours, and holds the reads to remove in memory. Testing 396 settings that way was not practical. `pafbench` imports polyAfilter's own A-stretch scanner and flags every record at every setting in one streaming pass: 7 minutes for 1417, 23 for 1179 (8 workers).

It is only useful if it is the same filter. We ran the original at ten settings on 1417, eight on 1419, eight on chr21 and chr22 of 1179 (the full library does not fit in memory) and one remove-all control. The removed records are identical in all 27 runs.

## Measuring what filtering costs

For 14 settings (the two picks, the README example and the paper's settings in both modes, two settings tuned on cells, two remove-all controls) the pipeline recounts UMIs with Cell Ranger's documented counting rule and clusters the cells again.

- **UMI cost.** UMIs lost across called cells, divided by UMIs before filtering.
- **ARI.** The adjusted Rand index between the Leiden clusters of the filtered matrix and those of the unfiltered matrix, median over 10 seeds and 3 resolutions. It measures how much the clusters change, not whether they got better.

Two references are needed to read an ARI. The **ceiling** is what the unfiltered matrix scores against itself with another seed: 0.69 (1417), 0.68 (1419), 0.70 (1179), 0.89 (BD67). The **thinning control** removes the same number of UMIs per cell at random. A filtered ARI close to the thinned one means the clusters moved because the cells got shallower, not because of which reads were taken. In the four libraries with an impact analysis the two differ by 0.06 at most at the three reported settings, so filtering changes the clusters about as much as losing the same number of UMIs at random.

## What this benchmark cannot say

- **The labeled molecules are a subset.** A molecule gets a label only if one of its reads reaches the poly(A). The shares describe that subset. In 1179 it is more exonic than the library as a whole (stage 28).
- **Material and tissue are confounded.** The nuclei are adult donor retina, the cells are retinal organoids. The nuclei against cells comparison is descriptive.
- **Unique mapping does not catch every misplaced read.** A read can score best at a copy of its locus and count as uniquely mapped there. The burden leaves out multi-mapped reads only.
- **No bulk reference.** The labels rest on the A count and the poly(A) signal. A real 3' end that is missing from the annotation and followed by A's is counted as internal priming.
- **One tissue, one genome.** Human retina, GRCh38-2024-A, 10x 3' v4 and FLIGHT-seq. Other organisms and tissues are untested.

The approach should carry over to any oligo(dT)-primed 3' library whose reads can reach the poly(A): other 10x 3' versions, BD Rhapsody, and 3' libraries made from full-length cDNA such as FLIGHT-seq. It does not apply to random-primed libraries.
