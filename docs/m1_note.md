# polyAfilter benchmark, milestone 1: exit note

Date: 2026-10-08. Branch `m1-implementation`. Numbers come from `docs/m1_results.md` and the JSON files under `/data/element_runs/polyafilter_bench/` unless a line says otherwise.

Milestone 1 had to show four things before the full benchmark: that polyAfilter can be reproduced exactly and fast enough for a parameter grid, that a truth set can be built without bulk, that the 3k aliquots stand in for the 20k libraries, and that spike-ins and long reads can serve as extra checks.

## Results

**1. Exact reproduction.** On the full 1417 BAM the reimplementation removes the same records as polyAfilter (commit 5ce0098): 17,068,514 at 300/10/1 in intron mode, 46,808,565 at 375/3/2 in intron mode, 3,729,928 at 300/10/1 in exon mode. Zero differences in either direction. An independent count agrees: 97,375,160 input records minus 93,645,232 in the original's exon-mode output is 3,729,928.
The original needed a 5.7 h annotation database, 3.4 h and 5.9 h expression caches, then 1.6 to 2.1 h per setting. Its largest single process peaked at 43.9 GiB (375/3/2), so the whole job used more. The reimplementation ran all three settings in 9 minutes.
Limits: the tested settings cover MISM 1 and 2, MINSNRLEN 3 and 10, COVLEN 300 and 375. 1417 has no secondary, supplementary or QC-fail records, so those flags were only tested on synthetic data.

**2. Truth set.** 167,437 polyA-trimmed reads collapse into 144,474 molecules. The AATAAA/ATTAAA hexamer sits upstream of 77% of sites with at most 4 genomic A's in the 20 nt after the site, and of 2% of sites with 18 to 20. The pre-set rule turned that curve into cut-offs: internal priming at 8 or more A's, real tail at 4 or fewer. With them, 81.9% of molecules are internally primed, 7.5% tail-primed and 10.6% ambiguous.

**3. Members.** 83,317 candidate member reads. Distance runs along the gene's exons when read and site are both exonic (36,943 reads) and in the genome otherwise (46,374). With the fixed 1,000 nt cap, 76,598 of 81,004 members are kept; 4,406 fall beyond it, in 3,160 molecules.

**4. Scope.** On chr1, the internal-priming share by region differs by at most 1.3 points between 1416 (20k) and 1417 (3k): exonic 0.828 vs 0.837, intronic 0.987 vs 0.987, intergenic 0.968 vs 0.955. The 3k aliquots stand in for the 20k libraries.

**5. Spike-ins (1417).** 14.2M unmapped reads realigned to SIRV Set 4, 86.3% uniquely. 12.3M primary alignments: ERCC 4.27M, SIRV E0 4.99M, long SIRVs 3.06M. CB, UB, `pa` and `ts` survive unchanged in a 2,000-read sample.

**6. Organoid cells (1179).** UB is present on 97.1% of intronic and 97.1% of exonic reads (chr1), so intronic molecules from cells can enter the truth set.

**7. Long reads (BD67, BD70): failed the pre-set bar.** Sampled genome-wide, only 61.7% (BD67) and 62.3% (BD70) of reads start their 3' clip with 10 or more A's; the bar was 80%. The hexamer contrast holds: per unique site, 0.34 non-templated vs 0.06 templated in BD67, 0.39 vs 0.07 in BD70. A breakdown of 15,359 BD67 reads shows why the clip test fails:
- 27% have the A run absorbed into the alignment (median 17 aligned A's). That is what priming on a long genomic A stretch looks like, and the site rule already counts aligned A's.
- 10% have no 3' clip and no A run, so they have no usable site.
- 5% have an A-rich clip broken by a mismatch.

Per-read figures include about 2% duplicate reads (BD67 836, BD70 750) because 13 of the 200 random windows overlap; per-site figures are not affected, and the milestone 2 rerun deduplicates reads. The earlier pass (87% of reads, 15% templated) came from a gene-dense 1.3 Mb stretch of chr1. Genome-wide, 25% of BD67 reads are templated at 14/8.

## Decisions needed

**Label cut-offs.** The pre-set rule chose 8/4. What that means for the positive class:

| Genomic A's in 20 nt | Share of molecules | Hexamer rate | Estimated real tails | Has a run of 10 or more A's (1 mismatch) |
|---|---|---|---|---|
| 8 to 9 | 12.9% | 14.3% | 16% | 0.4% |
| 10 to 13 | 41.3% | 6.4% | 6% | 11.8% |
| 14 to 20 | 27.8% | 4.2% | 3% | 76.9% |

About 6.5% of the 8/4 positives are real tails, against 4.6% at 10 and 2.9% at 14. Two thirds of the 8/4 positives have no A stretch that README-type settings (10 A's, 1 mismatch) can detect. At 8/4 the benchmark therefore also measures priming on A-rich but broken sequence, which polyAfilter's definition cannot catch. At 14/8 it mainly measures position, and the internal-priming share of exonic molecules on chr1 drops from 0.84 to 0.29.
Recommendation: keep 8/4 as the primary labels, since they were fixed before any polyAfilter result, and report every metric by A band. The spec's sensitivity sweep then becomes templated 7 to 10 and non-templated 2 to 6, so the two classes never overlap.

**Long-read criterion.** Proposal for milestone 2: accept a read when its whole terminal A run (aligned plus clipped A's) is 10 or longer, drop reads without one (about 12%), and report per unique site. BD67 and BD70 stay out of milestone 2 until this is agreed.

## Carried into the milestone 2 plan (from the reviews)
- Stage 03 streams records instead of holding a chromosome in memory, and stores CB, UB, GX, RE and MAPQ, so the molecule-level variant and the per-cell impact need no extra BAM pass. A 392-setting table on 1179 (580M reads) needs bit-packed or per-cell aggregate storage.
- More exact-match settings before the grid is trusted: MISM 0, COVLEN 600, the remove-all-non-terminal control (in-process, since the CLI cannot express it), and the published 275/9/1 and 180/11/2 in both modes, on 1417 and 1419.
- Member filter applies `gene_match` and the 1,000 nt cap; molecules carry MAPQ; `parse_setting` rejects an integer 0.
- Memory measured for the whole process tree, not the largest process.
