# polyAfilter on 10x 3' v4 nuclei and cells without a bulk control: benchmark design

Date: 2026-10-07
Author: Simone Picelli (design drafted with Claude Code)
Status: draft for review

## 1. Questions

polyAfilter (Svoboda, Frost, Bosco 2022, NAR Genomics and Bioinformatics 4:lqac035, PMC9142200; code github.com/MarekSvob/polyAfilter) removes reads that likely come from internal oligo(dT) priming on genome-encoded A stretches. The paper tuned its three parameters per dataset against a random-primed bulk library from the same sample. We have no such library. We do have reads that end exactly at their priming site, SIRV Set 4 spike-ins, and a whole-cell library made with the same chemistry.

**Q1. Burden.** What share of molecules in each library comes from internal priming, by region (3' UTR, CDS, intron, intergenic), in nuclei and in whole cells?

**Q2. Accuracy.** Against in-data truth, how well does polyAfilter separate internally primed molecules from tail-primed ones across its parameter space, in default and intron (`-i`) mode? Do the settings chosen on one library hold on the others?

**Q3. Cost.** What does filtering at the chosen and published settings do to per-cell UMIs, intronic signal and cell types?

Deliverables include the code, so every number in the report traces to the function that computes it.

## 2. Samples and data

All libraries: Cell Ranger 9.0.1, Single Cell 3' v4 (polyA) GEM-X, reference refdata-gex-GRCh38-2024-A (GENCODE v44 filtered), read layout 10-10-28-90. R1 holds only the 16 nt barcode and 12 nt UMI, so the priming site is never read from R1. FASTQs are not on the VM; everything starts from the Cell Ranger BAM.

| Library | Material | Reads | Cells | Exonic | Intronic | Intergenic | Antisense | Mapped to genome | Spike-ins | include-introns | Local path |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1417_HsRetina_D28_3k | adult retina nuclei, donor 28 | 97.4M | 5,571 | 33.9% | 41.1% | 5.3% | 12.2% | 85.4% | SIRV Set 4 | true | `/data/element_runs/1417_HsRetina_D28_3k/outs` |
| 1419_HsRetina_D29_3k | adult retina nuclei, donor 29 | 90.1M | 5,114 | 32.6% | 42.7% | 5.4% | 13.3% | 85.2% | SIRV Set 4 | true | `/data/element_runs/1419_HsRetina_D29_3k/outs` |
| 1179 | retinal organoid week 26, whole cells | 580.6M | 24,378 | 68.5% | 22.2% | 4.1% | 2.1% | 98.8% | none | false | `/data/element_runs/1179_organoid_wk26_10xv4/outs` |

The 3k aliquots stand in for the 20k libraries of the same samples. The user states the chemistry is identical, so the priming bias should match. 1416_HsRetina_D28_20k is already local and was used for the pilot probes below; milestone 1 compares its chr1 templated fraction with 1417 as a cheap check of that assumption.

SIRV Set 4 went into the RT master mix, so spike-ins are free RNA in every GEM. They test the chemistry, not nuclear context. In the 3k libraries, about 85% of a 200k sample of unplaced unmapped reads aligned to SIRV Set 4 (ERCC, SIRV E0, long SIRVs). Combined with the 14.6% to 14.8% unmapped rate, spike-ins are roughly 12% of reads (estimate).

BD67 and BD70 are FLIGHT-seq libraries (BD Rhapsody Enhanced beads, PacBio long reads) made from the same organoid cell pool as 1179 after dissociation; they are two slightly different, comparable versions of the FLIGHT-seq chemistry (user). Their aligned FLNC BAMs are local: `/data/element_runs/BD67/BD67_2x_RT.fltnc.aligned.bam` (35.9 GB) and `/data/element_runs/BD70/BD70.fltnc.aligned.bam` (42.7 GB). Details below were checked on BD67. Files in `gs://sc-ptf/FLIGHT_seq/Long_read/BD67/`; the aligned FLNC BAM is already local at `/data/element_runs/BD67/BD67_2x_RT.fltnc.aligned.bam` (35.9 GB). Reads carry no CB tag; read name to cell barcode (three 9-nt cell labels) and 8-nt UMI is in `BD67_strict_barcodes.tsv`. The gene matrix lists 200,894 barcodes and 26.7M molecules, so cell calling still has to be applied (knee data in `BD67_knee_plot_knee_data.tsv`). Observation on 3,000 reads (chr12:6.53 to 6.54 Mb): the soft clip after the aligned 3' end starts with the bead poly(A) (the first 15 clipped bases are all A in 2,827 reads), followed by UMI and cell label. The poly(A) is not trimmed, so the aligner can absorb genome-encoded A's into the alignment. The priming site is therefore taken as the start of the read's terminal A run mapped back to the genome, which matches how Cell Ranger's pre-alignment trimming defines the `pa` site. Milestone 1 confirms this before it is used.

Other inputs:
- Reference: `~/Documents/Genome_ref/refdata-gex-GRCh38-2024-A` (contigs identical to the BAM headers).
- SIRV Set 4 FASTA and GTF, Lexogen release 20250402 (92 ERCC, 69 SIRV E0 isoforms from 7 genes, 15 long SIRVs).
- PolyASite 3.0 clusters, GRCh38/GENCODE 42: `/data/scNoiseMeter/atlas.clusters.3.0.GRCh38.GENCODE_42.bed.gz`.

## 3. polyAfilter as implemented

Pinned at commit 5ce0098 (2022-12-04). From the code (`BAMfilter.scanBAMfilter`, `findSNRpieces`; `RNAseqAnalysis.getBaselineData`, `getTransEndSensSpec`) and the paper Methods:

1. Expressed transcripts are annotated transcripts with any same-strand coverage (exons only by default, the whole span with `-i`).
2. The last COVLEN bp of each expressed transcript are protected (exon-wise; with `-i` also the absolute last COVLEN bp of the span).
3. A stretches of at least MINSNRLEN bp with up to MISM internal mismatches (none at the first or last base) are found on each strand of the FASTA. Each defines a window of COVLEN bp upstream of its first base, in genomic coordinates (not splice-aware).
4. Windows are intersected with the unprotected transcript body (exons by default, the whole span with `-i`). Every same-strand alignment whose aligned blocks touch a window is removed.

Consequences: removal is per read and deterministic; reads of one molecule can be split; intergenic and antisense reads are never removed; in default mode no intronic read is removed. The removal set is held in RAM as pysam objects.

Published settings (Fig. 4A, COVLEN/MINSNRLEN/MISM): 10X PBMC1 275/9/1, 10X PBMC2 180/11/2, 10X mouse cortex nuclei with intronic counting 375/3/2. The README example 300/10/1 is not a tuned value.

## 4. Truth sets

### 4.1 Primary: molecules with a polyA-trimmed read

Cell Ranger trims poly(A) from the 3' end of R2 before alignment and records the count in `pa:i` (10x BAM docs: "number of poly-A nucleotides trimmed from the 3' end of read 2. Up to 10% mismatches are permitted"; trimmed bases stay in the record as soft clip). For a `pa` read, the alignment 3' end is the priming site, because the bead dT reads as A whatever the template.

- **Molecule.** `UB` is corrected within barcode and gene (10x docs), so a molecule is (CB, UB, strand, gene). The gene comes from our own same-strand overlap with the 2024-A GTF, not from Cell Ranger's GX, because GX depends on include-introns (true for 1417/1419, false for 1179). One rule then serves nuclei and cells. Intergenic reads use a locus window. Milestone 1 checks how often intronic reads in 1179 carry UB.
- **Site.** The alignment 3' end of the `pa` read. Molecules whose `pa` reads disagree by more than 10 nt are dropped (11 of 85,135 on 1416 chr1).
- **Label.** A count in the 20 nt of genome after the site, sense strand: at least 14 is templated (internal priming), at most 8 is non-templated (tail priming), 9 to 13 is ambiguous. These are starting values. On 1417, the hexamer rate is plotted against the A count (0 to 20); the final cut-offs are where it reaches tail-like and background levels, fixed before any polyAfilter result is looked at. A sweep (templated 10 to 16, non-templated 3 to 9) shows whether conclusions depend on them.
- **Members.** The other reads of the molecule, which polyAfilter must judge without knowing the site. Membership requires the same molecule key, a position upstream of the site, and a distance cap. Distance is measured along the gene's collapsed exons when both the read and the site lie in exons, and in the genome otherwise (user decision 2026-10-07): on 1417, members 1 to 50 kb away in the genome were mostly upstream-exon reads of spliced molecules, about 300 nt from the site along the exons, so genomic distance would have discarded real members and hidden polyAfilter's blind spot for spliced reads. The member cap is fixed at 1,000 nt of exon-aware distance (user decision 2026-10-07), with sensitivity reruns at 500 and 2,000 nt; members beyond the cap are excluded from the truth set and counted in the report. The density-based cap rule was dropped: on 1417 it returned no cap because its 'far on the same chromosome' background (1.6e-9 per bp) is structured signal, about 200 times the cross-chromosome chance rate (7.5e-12 per bp), at which chance UMI sharing within 1 kb is about 1e-8 per molecule. On 1417, 76,598 of 81,004 members (95%) fall within 1,000 nt. The counts use candidate members matched on (CB, UB, strand, reference) with dist3 >= -25 along the chosen distance; gene_match is applied in milestone 2. For intergenic molecules the 1,000 nt cap acts as the locus window; no separate window is implemented.
- **Label checks that do not use A content.** AATAAA/ATTAAA within 5 to 35 nt upstream of the site; in the 1416 chr1 pilot it was present for 45.1% of non-templated, 4.1% of templated and 8.0% of ambiguous sites. PolyASite 3.0 proximity is reported separately and not used to validate labels, because the atlas itself screens candidate sites with downstream A content.
- **What the label does and does not test.** The label is the definition of internal priming at the observed site: the oligo(dT) sat on genome-encoded A's. On member reads polyAfilter has to find that site without seeing it. For templated molecules, whether its A-stretch rule catches the site is close to guaranteed at permissive settings (a site with at least 14 A in 20 nt contains a short A-stretch), so sensitivity mainly tests position: COVLEN against the read-to-site distance, protected 3' ends, and genomic windows on spliced reads. Specificity on tail-primed molecules (A-stretches that happen to sit near real ends) is fully informative. The two checks that use no A content at all are the hexamer and the spike-ins, whose labels come from the annotated 3' end. Results are reported with this split.
- **Selection bias.** A molecule enters the truth set only if one of its copies had a short insert (aligned part about 70 nt or less). Bigger UMI families are favoured. Mitigation: weight molecules by inverse family size; compare labeled and unlabeled molecules for region, gene, distance to the annotated 3' end and cell type; report molecules with a `ts` read (short RNAs) separately.

Pilot numbers (1416, chr1; molecule key without gene): 103,708 `pa` reads; 85,135 molecules; 42,146 with at least one member read (85,598 members). Unspliced members start a median of 197 to 215 nt from the site; 85% to 91% start within 300 nt and 90% to 95% within 375 nt (range across the three label classes).

### 4.2 Spike-ins (1417, 1419)

Unplaced unmapped reads are pulled with CB, UB, `pa` and `ts`, clipped where tagged, and realigned splice-aware to SIRV Set 4 with STAR 2.7.10a, keeping tags with `--readFilesSAMattrKeep All` (present in this STAR build per `STAR --help`). In 20,000 unmapped 1416 records, CB was present on 97%, UB on 78%, `ts` on 727 and `pa` on 61. A spot check confirms tags survive realignment. ERCC and SIRV 3' ends are known, so `pa` sites are labeled by distance to the annotated end. The molecule logic of 4.1 then labels member reads exactly. ERCC UMI counts versus input amount give a count-accuracy check like the paper's ERCC analysis (r 0.91 to 0.94 after filtering). Lexogen concentrations and tail lengths are taken from the vendor documentation and cited.

### 4.3 Long reads from the same cell pool (organoid only)

BD67 and BD70 give priming sites for millions of molecules from the same cells, primed by a different oligo(dT) on different beads. Each read's 3' end is labeled with the same downstream A-count rule and hexamer check as in 4.1. Uses: (1) a sample-matched catalogue of 3' ends for 1179, including whole-isoform context (does the end sit in a 3' UTR, CDS or intron of an otherwise complete isoform); (2) a cross-chemistry comparison of which sites attract internal priming in both libraries; Long-read support is not truth by itself, because the BD oligo(dT) can also prime on genomic A's; it is reported next to the 10x labels, not used to define them.

### 4.4 Not used as truth

Unfiltered clusters, PolyASite on its own, and filtered gene counts. They appear only in the descriptive analysis (section 8).

## 5. Filter implementations

1. **Original tool** on 1417 and 1419 in full, at the published settings in both modes. RAM and wall time are recorded. 1179 only if it fits in memory, otherwise by chromosome.
2. **Reimplementation.** A streaming pass per library that imports polyAfilter's own A-stretch scanner (`findSNRpieces`) and its transcript start/end logic, and writes, per alignment record, whether it would be removed at every grid point. It must consider the same records the original fetches (secondary alignments and duplicate-flagged reads included; Cell Ranger flags all but one read per UMI as duplicate, so member reads are mostly duplicates). Analyses then pick records explicitly. Acceptance: the removed-read sets are identical to the original tool on 1417 and 1419 for every setting run in step 1. Any difference is resolved before results are used.
3. **Molecule-level variant.** Same windows, but a molecule is removed only when the majority of its reads are flagged. This is our extension, reported separately from polyAfilter.

## 6. Parameter grid

- COVLEN: 150, 200, 250, 300, 375, 450, 600.
- MINSNRLEN: 0 (remove all non-terminal reads, a control that ignores A content and is reported separately), 3, 5, 7, 9, 10, 11, 13, 15, 20.
- MISM: 0, 1, 2 (not applicable at MINSNRLEN 0).
- Mode: default and `-i`.

That is 7 x (1 + 9 x 3) x 2 = 392 settings, all read off the per-read table without extra BAM passes. Published and README settings are marked on every plot.

## 7. Metrics and parameter choice

Positives are templated molecules, negatives are non-templated. Ambiguous molecules are excluded from the primary metrics and reported on their own, with a hexamer-based estimate of their tail-primed share.

- Read level (members) and molecule level (a molecule counts as removed if all its reads are removed, which is what changes a UMI count).
- Sensitivity, specificity (tail-primed retained), precision, PR and ROC across the grid.
- Strata: region, library, coarse cell type, expression level.
- Label-free check: hexamer rate in removed versus kept labeled molecules.
- Uncertainty: bootstrap by gene, 1,000 resamples.

**Selection rule, fixed before results:** on 1417 (donor 28), in each mode, take the setting with the highest molecule-level sensitivity subject to specificity of at least 0.95; break ties by fewer total reads removed. Report that setting, plus the published settings, on 1419 (donor 29, a donor-level hold-out) and on 1179 without retuning. If no setting reaches 0.95, report the frontier and do not pick one.

## 8. Nuclei versus cells, and downstream impact

**Burden comparison.** Templated share among labeled molecules by region, in 1417, 1419 and 1179. Where possible, compare the same gene and the same site (site sequence, hexamer, distance to the annotated end). Tissue, maturity, cell-type mix and the cell/nucleus difference are confounded, so this stays descriptive.

**Impact (descriptive only).** First, a BAM recount must reproduce the Cell Ranger filtered matrix within a stated tolerance. Starting definition: unique (CB, UB, GX) among reads with `xf` bit 1 set and bit 2 unset (10x BAM docs), adjusted until it matches, since how intronic reads are flagged under include-introns is not yet checked. Then, for the chosen settings and the published ones: per-cell UMI loss, intronic share, per-cell-type loss, cluster preservation and marker genes. The check is whether any cell type collapses, as the endothelial cluster did in the paper. No annotations exist for any of the libraries (user), so cell types are assigned from retinal marker genes in all three.

## 9. Outputs and layout

- Code: `~/polyafilter_bench/`, numbered stage scripts as in `~/kinnex_concat_qc`, Python with pysam, pyranges, pandas and plotly.
- Results: `/data/element_runs/polyafilter_bench/<library>/` (per-read and per-molecule tables as Parquet, JSON metrics) and the final report published as an Artifact page that explains each step in plain terms next to its numbers.
- Draft figure list: truth-set construction; label validation; burden by region in nuclei and cells; read-to-site distance (endogenous versus ERCC) against COVLEN; PR/ROC across the grid with published settings marked, on held-out libraries; impact per cell and per cell type; spike-in results.

## 10. Milestone 1 (about a day)

1. Downloads finish (1417, 1419, 1179); build the gffutils database from the 2024-A GTF.
2. Run the original polyAfilter on 1417 at 300/10/1 and 375/3/2 with `-i`; record RAM and time.
3. Write the reimplementation and pass the exact-match test on 1417.
4. Build the genome-wide `pa` molecule table for 1417; estimate the UMI collision rate; fix the member rule.
5. Check that unmapped records carry `pa`/`ts` and whether STAR 2.7.10a keeps SAM tags.
6. Compare chr1 templated share by region, 1416 versus 1417. If any region differs by more than 5 percentage points, take it back to the user before continuing with the 3k aliquots only.
7. Fix the label cut-offs from the hexamer curve and the molecule member rule from the collision estimate.

Exit: a one-page note with these results and any change to this design.

## 11. Risks

- The original tool may not fit in RAM on 1179 (580M reads). Mitigation: per-chromosome runs for the exact-match test there.
- The `pa` truth set covers only molecules with a short-insert copy. Mitigation in 4.1. If labeled and unlabeled molecules differ strongly, conclusions are stated for the labeled population only.
- The ambiguous class is large (46% of 1416 chr1 `pa` reads). If the cut-offs move results, that is reported, not tuned away.
- The 3' v4 bead oligo design (dT length, anchor bases) is not yet checked. It may shift sites by 1 to 2 nt, which does not change a 20 nt label but matters for exact-position comparisons with the spike-in annotation.
- gffutils database build can take hours (README).

## 12. Open questions

None open. Answered by the user on 2026-10-07: no cell annotations exist; BD67 and BD70 share the 1179 cell pool (BD45 appears in `10x_quick_analysis.R` only because it was the long-read data available at the time); D28 and D29 are donors.
