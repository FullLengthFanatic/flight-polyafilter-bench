# Milestone 3 addendum: FLIGHT-seq nuclei (Pool278)

Design, four decisions resolved. Drafted 2026-10-30, decisions complete 2026-10-31. Extends `2026-10-12-polyafilter-bench-milestone3-design.md`.

## 0. What this adds

Pool278 closes the 2×2 that the current benchmark cannot finish.

|            | 10x 3' v4                  | FLIGHT-seq (BD)                              |
|------------|----------------------------|----------------------------------------------|
| Cells      | 1179 (wk26 organoid pool)  | BD67 (same wk26 organoid pool as 1179)       |
| Nuclei     | 1416 (D28), 1417 (D28), 1419 (D29) | BD_nuc_211 (D29), BD_nuc_213 (Graz), BD_nuc_215 (Graz) |

With all four cells filled you can decompose the mispriming burden into a chemistry effect (10x against BD, holding material constant) and a material effect (cells against nuclei, holding chemistry constant), and ask whether the two interact. That interaction is the question the benchmark currently cannot answer.

Two cells carry a biological-match anchor: the organoid pool is shared across 1179 and BD67, and D29 is shared across 1419 and `BD_nuc_211`. Those two pairings reduce donor confounding on both main effects.

## 1. Library handling

Nine BAMs on disk: 3 samples (211, 213, 215) × 3 bead-split replicates (a, b, c). STAR 2.7.10b against `RhapRef_Human_WTA_2023-02`, UCSC-style chromosome naming, same coordinate frame as the benchmark's GRCh38-2024-A primary assembly. Median per-nucleus UMI 98 to 158, aligned read count 68 to 133 M per library.

Pool the three bead-split replicates within each sample. That gives three FLIGHT-nuclei libraries, `BD_nuc_211`, `BD_nuc_213`, `BD_nuc_215`, which matches the scale of the 10x nuclei set (1416, 1417, 1419). Don't pool across samples. The three sample pools become the sibling nuclei libraries of BD67.

Realignment: BD's STAR BAMs were aligned against `RhapRef_Human_WTA_2023-02` with BD-specific parameters, and BD's GTF differs from GRCh38-2024-A. Realigning from FASTQ against GRCh38-2024-A with 10x-style STARsolo parameters is the cleanest apples-to-apples with 1417, 1419, 1416, 1179 (**decided 2026-10-30**, D-pool278-2). Nine R1/R2 pairs are on disk at `fastq/Samples/DefaultProject/<library>/<library>_R{1,2}.fastq.gz`. STARsolo in BD Rhapsody Enhanced v2/v3 CB_UMI_Complex mode (three-segment cell label + 8 nt UMI) with the BD Enhanced v2/v3 whitelist; GRCh38-2024-A genome index; `GeneFull` to count intronic reads the way Cell Ranger does on the 10x libraries; CB and UB tags in the output BAM to drop straight into the pipeline schema.

BD67's short-read BAM is `BD067_as10x.bam`, which was produced by the same pattern (STARsolo retag against GRCh38 with 10x tags). Reuse that script with the BD Enhanced v2/v3 whitelist (Pool278 uses the same bead version per the per-library metrics summary: `Bead version detected: Enhanced V2/V3`). One command invocation per sample, feeding its three bead-split replicates as comma-separated FASTQs.

## 2. Config change

```python
# pafbench/config.py
POOL278 = RUNS / "Pool278_FLIGHTseq_NucleiHsRetinaDonation"
LIBRARIES = {
    "1417": ...,
    "1419": ...,
    "1179": ...,
    "1416": ...,
    "BD67": RUNS / "BD67" / "short_read" / "BD067_as10x.bam",
    "BD_nuc_211": POOL278 / "merged" / "BD_nuc_211_as10x.bam",
    "BD_nuc_213": POOL278 / "merged" / "BD_nuc_213_as10x.bam",
    "BD_nuc_215": POOL278 / "merged" / "BD_nuc_215_as10x.bam",
}
BD_NUC = ("BD_nuc_211", "BD_nuc_213", "BD_nuc_215")
NUCLEI = ("1417", "1419", *BD_NUC)
CELLS = ("1179", "BD67")
FLIGHT = ("BD67", *BD_NUC)   # new, for the chemistry-controlled material comparison
```

Each merged BAM pools its three bead-split replicates (`a + b + c`) within one sample. The three raw BAMs per sample stay addressable through a `BD_NUC_PARTS` dict for a reproducibility inset.

## 3. Stages that need work

The pipeline is already library-pluggable. Stages 1 to 10, 23, 24, 26 to 29 run on a new library with no code change: a new entry in `LIBRARIES` is enough.

### 3.1 Stage 1a: Pool278 realign from FASTQ (new)

Input: 18 FASTQ files (9 R1 + 9 R2) at `fastq/Samples/DefaultProject/<library>/<library>_R{1,2}.fastq.gz`. Output: three STARsolo outputs, one per sample (`BD_nuc_211_as10x.bam`, `BD_nuc_213_as10x.bam`, `BD_nuc_215_as10x.bam`), each the pool of its three bead-split replicates, plus the solo matrices in the pipeline schema.

Steps:

1. For each sample, run STARsolo once against GRCh38-2024-A with the three replicate R1/R2 pairs as comma-separated inputs. Chemistry: BD Rhapsody Enhanced V2/V3, `--soloType CB_UMI_Complex` with the three-segment cell label (9+9+9) and 8 nt UMI, BD Enhanced V2/V3 whitelist, `--soloFeatures GeneFull` to count intronic reads the way Cell Ranger does on the 10x libraries, `--outSAMtype BAM SortedByCoordinate`, `--outSAMattributes NH HI AS nM GX GN CB UB sS sQ sM`.
2. Each replicate carries its own read group so the merged BAM carries replicate identity (`@RG ID:BD-211a SM:BD_nuc_211`, and so on).
3. Index each merged BAM.
4. Write `BD_nuc_<sample>/library_parts.parquet` per sample with per-replicate read counts, median UMI per cell, and the STARsolo cell call so the per-replicate inset in the report does not re-scan BAMs.

Acceptance: `GeneFull` matrix cell count per sample is within 5% of the sum of its three BD per-library cell calls (consistency check against BD's basic cell calling, allowing drift from the whitelist and the knee); every output BAM read has valid `CB` and `UB`; the `@RG` entries of each sample's BAM name only its three replicates.

Tests: on three synthetic BD-style FASTQ pairs (one per replicate) with one valid cell barcode per read, STARsolo produces three reads in the per-sample BAM with correct `CB`/`UB`/`RG`; a replicate from a different sample fed into the wrong invocation is refused.

Run time: STARsolo on ~300 M read pairs per sample (sum of three replicates), with GRCh38-2024-A index. Expect 2 to 4 h per sample on 16 threads. The three samples can run in parallel if disk I/O permits; otherwise serial, ~6 to 12 h total.

Disk: index + 3 output BAMs + solo matrices. Budget ~150 to 250 GB.

### 3.2 Stage 25a: marker set for the FLIGHT nuclei

Stage 25 calls cell classes from a retina marker panel. Reuse the panel the 10x nuclei (1416, 1417, 1419) already use (**decided 2026-10-31**, D-pool278-3). Check per-sample whether BD-nuclei marker signal is strong enough to call classes; if a sample falls below the stage-25 marker-coverage floor, record it as class-unknown and run stage 26 at the library level only for that sample. Don't force class calls that aren't there, and don't swap the panel to improve them.

### 3.3 Stage 31b: FLIGHT-pair matched sites (new instance of stage 31)

The existing stage 31 compares nuclei against cells using 10x libraries (1417 + 1419 against 1179). Add a FLIGHT-pair instance: `BD_nuc_211 + BD_nuc_213 + BD_nuc_215` against `BD67`. Same code, same thresholds. Three samples pool as the nuclei side the way 1417 + 1419 pool on the 10x side. Expected depth gap: smaller than the 10x gap, since BD67 is deeper per cell than Pool278 nuclei but not by the 11-13× factor between 1179 and 1417/1419.

### 3.4 Stage 29a: interaction panel (new)

The 2×2 isn't just four bars next to each other, it's a factorial. Fit the burden model with `chemistry` (10x, BD) and `material` (cells, nuclei) as main effects plus their interaction, treating each library as a replicate within its cell: 1 (BD67) and 3 (211, 213, 215) on the BD side, 1 (1179) and 2 (1417, 1419, per the current `NUCLEI` group) on the 10x side. 1416 is in `LIBRARIES` but not in `NUCLEI`; adding it as a third 10x-nuclei replicate here is a separate choice outside this addendum. Report the four cell means, the two main effects with their CIs, and the interaction coefficient. This is one new table and one figure.

Scope-limit: with one BD-cells sample (BD67) the BD-cells cell has no within-cell variance, so the interaction CI leans on the three BD-nuclei samples and the two 10x-nuclei libraries. State the width, don't bury it.

## 4. Report page additions

- Panel A (burden by library, extended): add three bars for `BD_nuc_211`, `BD_nuc_213`, `BD_nuc_215`, grouped as a FLIGHT-nuclei block next to BD67.
- Panel B (burden by compartment): add the three `BD_nuc_<sample>` rows to the heatmap. Intron and antisense columns are the ones to watch.
- Panel F (new): the 2×2 interaction. Four cells, two main effects, one interaction term; sample points inside each cell.
- Panel G (new): the three BD-nuclei samples side by side, with the three bead-split replicates (a/b/c) as an inset per sample. Shows whether the FLIGHT-nuclei signal is a sample-level effect or a material effect, and whether the bead split adds noise.

Stage 30 is BD67-specific (long reads) and does not get a Pool278 version.

## 5. What this cannot show

- No matched long-read run for Pool278 nuclei. Isoform context for BD-nuclei 3' ends is unavailable; the long-read supplement stays BD67-only.
- Donor coverage. Cells row shares the organoid pool across the two cells (1179 and BD67). Nuclei row shares D29 across 1419 and `BD_nuc_211`; 1417/1416 are D28, and `BD_nuc_213`/`BD_nuc_215` are two further Graz donors not elsewhere in the benchmark. Material and donor are not perfectly orthogonal and 1416 vs 1417 (same donor, two depths) is already in the benchmark, so the design is more balanced than it looks, but the interaction panel should list the shared-donor pairs so readers can see which cells of the 2×2 lean on matched biology and which on biological replication.
- Shallow per-nucleus depth. Fine for rate statistics (which is what polyAfilter is), but not for isoform-level or per-cell-trajectory claims.

## 6. Order of work

1. Stage 1a: STARsolo realignment per sample against GRCh38-2024-A, three merged BAMs out.
2. Stages 1 to 10, 23 to 29 run on each `BD_nuc_<sample>` with no changes.
3. Stage 29a: fit the interaction model, build Panel F.
4. Stage 31b: FLIGHT-pair matched sites.
5. Report: add Panels A/B rows, add Panels F/G.

## 7. Decisions

- D-pool278-1: pool bead-split replicates within each sample, three FLIGHT-nuclei libraries (`BD_nuc_211`, `BD_nuc_213`, `BD_nuc_215`). **Decided 2026-10-30.**
- D-pool278-2: realign from FASTQ with STARsolo against GRCh38-2024-A. **Decided 2026-10-31.**
- D-pool278-3: reuse the 10x nuclei marker panel for the FLIGHT nuclei. **Decided 2026-10-31.**
- D-pool278-4: library names `BD_nuc_211`, `BD_nuc_213`, `BD_nuc_215`; 211 is D29 (shared with 1419), 213 and 215 are two further Graz donors. **Decided 2026-10-31.**

## 8. Rough timing

STARsolo realignment: 2 to 4 h per sample on 16 threads, three samples in parallel or serial depending on I/O (~1 day end-to-end for all three). Stage 1 to 29 reruns on each new library: half a day. Interaction model and Panel F: a day. Stage 31b: half a day. Total: 3 to 4 working days before the first draft of Panel F.
