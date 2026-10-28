# Why these cutoffs

Short answer: because we do not have the data Svoboda et al. 2022 have, and we had to find a different anchor.

## The problem polyAfilter addresses

Oligo-dT scRNA-seq assays prime on any run of adenines, template or genomic. When they prime on an mRNA poly-A tail, we count a real 3' end. When they prime on an intronic or exonic A-stretch inside the body of the transcript, we count a ghost: the read looks like a 3' end, maps to a legitimate exon, carries a UMI, but represents nothing. The ghost is indistinguishable from a short, poorly-expressed isoform at the read level. It is distinguishable at the site level if we look at the 20 nt downstream of the alignment's 3' end on the reference genome. Internal priming leaves a telltale genomic A-run there; a real poly-A tail does not.

polyAfilter counts those downstream A's across every candidate 3' end in the library, labels sites, filters reads at the sites it does not trust, and recounts.

## Why 20 nt and why 8 and 4

The 20 nt downstream window is polyAfilter's design (Svoboda, Frost, and Bosco 2022, NAR Genomics and Bioinformatics). The tool evaluates genomic adenine content within that window to separate a real poly-A tail from oligo-dT mispriming on a genomic A-stretch, and we kept the window.

The 8 and 4 split (templated at 8 or more, non-templated at 4 or fewer, ambiguous 5-7) comes from a hexamer-purity audit on 1417. At 8 A's downstream the fraction of sites carrying a known mispriming hexamer in the 10 nt upstream drops below one in ten. At 4 A's the hexamer rate is high and the sites behave like real cleavage-polyadenylation signals. 5-7 is a mixed band we keep separate rather than force into either side. In our hands cutoffs of 14/8 (the paper default) are too permissive on nuclei: a lot of mispriming slips into "non-templated" because the upstream signal is already strong. We tightened to 8/4.

## Why we do not have a ground truth and what we use instead

Svoboda et al. 2022 tune their polyAfilter settings against bulk random-primed RNA-seq. Random priming has no 3' bias: a gene with 1,000 real molecules produces about the same number of random-primed reads whether those molecules are poly-adenylated or stripped. If a scRNA-seq filter removes real signal, the gene's per-cell count drops out of correlation with the bulk count. If it removes only artefact, correlation goes up. The sweet spot is a filter aggressive enough to clean up but not so aggressive it eats real molecules.

We did not have matched bulk random-primed RNA-seq for the retina organoid libraries. We needed a different anchor. Three substitutes carried us:

**Hexamer purity.** On the long-read libraries we require that the rate of a canonical poly-A signal hexamer (AATAAA, ATTAAA and the 15 near-canonical variants) within 10 nt upstream of the site be at least three times higher on non-templated sites than on templated ones. If a cutoff setting breaks that ratio, the labels are not splitting biology any more. Three is empirical, picked so that both 1417 and 1419 pass at 8/4 and fail at 14/8 by small margins. The ratio check is a smoke test, not a tune.

**Long-read corroboration.** Two independent PacBio Kinnex long-read runs agree with the 1179 short-read labels: 99.4% pooled for BD67 (FLIGHT-seq Rhapsody long-read on the same week-26 cell pool as 1179, cross-prep comparison) and 99.5% pooled for 10x_FL (Kinnex 8-fold on a week-29 10x retina organoid library, different pool from 1179; cross-platform check). The agreement is near-identical because the 20 nt A-count rule is a reference-sequence property, so cross-platform convergence is expected even without matched cells. This does not tune the cutoffs; it tells us the labels are stable across platforms.

**Known positive and negative controls.** On nuclei the templated share should dominate (pre-mRNA capture favours intronic A-stretches); it does, at 89% pooled across 1417 and 1419. On FLIGHT-seq cells the templated share should be lower (RT-on-bead with less secondary structure); it is, at 26% on BD67. The relative ordering across the four libraries matches the physics.

None of these substitutes is as clean as bulk random-primed data. We tell the reader this up front. We would prefer bulk for a future benchmark and would run it if we had the sample.

## Why these two chosen settings

polyAfilter has three knobs: `covlen` (coverage window inward from the site), `lmax` (minimum uninterrupted A-run length to flag), and `mmax` (allowed mismatches). The paper default is `c300_l10_m1_e` ("e" means introns excluded from the recount). We did not want to pick on vibes, so we ran a 14-setting sweep per library and selected two settings that read the extremes of the operating curve.

- `mode-e: c250_l7_m1_e`, the sensitivity pick. Highest templated recovery at specificity above 0.95 on called cells. UMI cost 15.7% on 1417, 15.1% on 1419, 15.7% on 1179, 24.0% on BD67. ARI 0.42 / 0.49 / 0.66 / 0.65.
- `mode-i: c200_l10_m2_i`, the introns-aware pick. Same spec band, higher specificity. UMI cost 54.7% on 1417, 51.3% on 1419, 21.9% on 1179, 17.5% on BD67. ARI 0.32 / 0.42 / 0.66 / 0.68.

Mode-i inverts between nuclei and cells. On nuclei it eats the intronic reads that are the whole point of a snRNA-seq library. On cells it is cheap and preserves clustering. We present both and let the sample decide, with a decision table in `docs/parameter_choices.md`.

## What this analysis cannot say

The pipeline was tested on four retina organoid libraries and one human reference build (GRCh38-2024-A). It makes no claim about:
- Non-human organisms. The A-run geometry of a mouse genome is different and the 20 nt window may want a different cutoff.
- Non-oligo-dT chemistries (random hexamer RT, pseudo-bulk random-primed). The whole label system is specific to oligo-dT priming geometry, so neither the templated nor the non-templated label is informative without poly-T priming at the RT step.
- Spatial transcriptomics or multiome. The cell type we tune on changes the base rate of each label.
- Datasets where the reference annotation is incomplete. A missed real 3' end looks like internal priming by this rule, because its downstream A-run has no annotation to the contrary.

What the pipeline DOES apply to, including some cases that may look like exceptions:
- 3' chemistries from any oligo-dT RT: 10x 3' v3/v4, GEM-X, Fluent PIP-seq, BD Rhapsody standard. All share the same priming geometry.
- 3' libraries built from SMART-seq-family full-length cDNA. SMART-seq2, FLASH-seq, and FLIGHT-seq (FLASH-seq chemistry done on BD Rhapsody beads) all use oligo-dT RT and inherit the same internal-priming risk as 10x. When the full-length cDNA is sequenced only at its 3' end (BD67's short-read library, this benchmark), polyAfilter applies exactly as to 10x.
- Full-length long-read sequencing of the same cDNA (BD67 long-read, 10x_FL Kinnex). The internal-priming artefact is still there in the molecules, but the manifestation is different: a full-length read misassigned to a short truncated isoform rather than a 3' end at the wrong position. polyAfilter's read-level filter does not directly apply at this scale, but the same 20 nt downstream A-count labelling logic does, and is what this benchmark uses for long-read corroboration in stages 16 and 30.
