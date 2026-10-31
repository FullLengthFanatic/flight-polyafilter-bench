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

Reference frame: BD's STAR BAMs are aligned against `RhapRef_Human_WTA_2023-02` (GRCh38 primary assembly, UCSC naming), the same coordinate frame as the benchmark's GRCh38-2024-A primary assembly used by 1417, 1419, 1416, 1179. Mirror BD67's as10x pattern rather than realigning from FASTQ: inspection of `BD067_as10x.bam` headers shows BD67 is itself not a GRCh38-2024-A STARsolo output, it is BD's own STAR BAM with 10x-compatible tags grafted on (CB kept, UB from MR padded to 12 nt, UR = UB, GX from the BD features table, RE from XF, xf = 25, pa from the polyA soft-clip). The benchmark already runs its own region re-annotation on top of the input BAM, so what the input needs is a correct coordinate frame and 10x-style tags, not a 10x aligner. BD67 passes that bar today; Pool278 fits the same mold. Realigning Pool278 to GRCh38-2024-A while leaving BD67 on BD's reference would introduce a cross-reference confound; realigning both is out of scope for this addendum (**decision revised 2026-10-31**, D-pool278-2).

Script: `scripts/01a_pool278_preprocess.py`, structured as `01_bd67_preprocess.py` with per-replicate RG tagging and a per-sample pysam merge at the end. One invocation per sample, over its three bead-split replicates. Output: `/data/element_runs/Pool278_FLIGHTseq_NucleiHsRetinaDonation/merged/BD_nuc_<sample>_as10x.bam` with three `@RG` entries, plus the nine per-replicate part BAMs in `merged/parts/` for the reproducibility inset.

## 2. Config change

```python
# pafbench/config.py
LIBRARIES = {
    "1417": ...,
    "1419": ...,
    "1179": ...,
    "1416": ...,
    "BD67": RUNS / "BD67" / "short_read" / "BD067_as10x.bam",
    "BD_nuc_211": RUNS / "Pool278_FLIGHTseq_NucleiHsRetinaDonation" / "merged" / "BD_nuc_211_as10x.bam",
    "BD_nuc_213": RUNS / "Pool278_FLIGHTseq_NucleiHsRetinaDonation" / "merged" / "BD_nuc_213_as10x.bam",
    "BD_nuc_215": RUNS / "Pool278_FLIGHTseq_NucleiHsRetinaDonation" / "merged" / "BD_nuc_215_as10x.bam",
}
NUCLEI = ("1417", "1419")                            # 10x nuclei, unchanged (stage 31 contrast)
CELLS = ("1179", "BD67")
BD_NUC = ("BD_nuc_211", "BD_nuc_213", "BD_nuc_215")  # FLIGHT nuclei, three samples
FLIGHT = ("BD67", *BD_NUC)                           # chemistry-controlled material comparison
```

`NUCLEI` stays scoped to the 10x nuclei so stage 31 keeps running the way it does (1417 + 1419 against 1179). The BD-nuclei side of stage 31b and the interaction panel use `BD_NUC` and `FLIGHT` directly. Each merged BAM pools its three bead-split replicates (`a + b + c`) within one sample; the three per-replicate part BAMs under `merged/parts/` stay addressable for the reproducibility inset.

## 3. Stages that need work

The pipeline is already library-pluggable. Stages 1 to 10, 23, 24, 26 to 29 run on a new library with no code change: a new entry in `LIBRARIES` is enough.

### 3.1 Stage 1a: retag Pool278 BD BAMs as 10x and merge per sample (new)

Input: 9 BD Bioproduct BAMs at `bd_pipeline/<lib>/<prefix>_Bioproduct.bam` plus their matching `_RSEC_MolsPerCell_MEX.zip` for the gene-symbol → ENSG table. Output: 9 retagged part BAMs under `merged/parts/`, 3 merged BAMs `BD_nuc_<sample>_as10x.bam` under `merged/`, each with 3 `@RG` entries and a `.bai`.

Steps:

1. For each sample, load the sym → ENSG table from the first replicate's `features.tsv.gz` (inside the MEX zip).
2. For each of the three replicates, rewrite the BAM with CB kept, UB = `(MR + 'AAAA')[:12]`, UR = UB, GX from the sym → ENSG lookup, RE in {E, N, I} from XF, xf = 25, pa = polyA soft-clip length (polyAfilter rule applied in place), and RG = replicate label (`BD-<sample>{a,b,c}`). Secondary/supplementary and reads lacking CB or MR are dropped.
3. `pysam.merge -c -p` the three part BAMs per sample into `BD_nuc_<sample>_as10x.bam`.
4. `pysam.index` the merged BAM.
5. Write `BD_nuc_<sample>/library_parts.parquet` per sample with per-replicate kept-read count, pa-tagged count, and gene-lookup-miss count so the per-replicate inset in the report does not re-scan BAMs.

Acceptance: three `@RG` entries per merged BAM named `BD-<sample>{a,b,c}`, SM = `BD_nuc_<sample>`, no cross-sample RG; every kept read carries a valid CB and a 12-nt UB.

Observed on 2026-10-31: `BD_nuc_211` 316.9 M reads total, 316.9 M kept; `BD_nuc_213` 342.5 M reads, 319.9 M kept; `BD_nuc_215` 294.7 M reads, 276.9 M kept. Merged BAMs 19 to 22 GB each, index 5.6 to 6.2 MB.

Run time: ~70 min per sample on 4 threads with all three samples in parallel (observed). Disk: 19 to 22 GB per merged BAM, plus 9 part BAMs of comparable aggregate size under `merged/parts/`.

Note: this is NOT a STARsolo realignment against GRCh38-2024-A. BD's reference is GRCh38 primary assembly with UCSC naming, same coordinate frame as 10x's `refdata-gex-GRCh38-2024-A` primary assembly, so no coordinate conversion is needed. The pipeline's region re-annotation (stages 2 to 10) runs on top of the input BAM, so GX/RE from the BD feature table are recomputed downstream; what the retag preserves is CB/UB (molecule identity) and pa (polyA soft-clip length).

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

1. Stage 1a: retag the nine BD Bioproduct BAMs as 10x and merge to three `BD_nuc_<sample>_as10x.bam`.
2. Stages 1 to 10, 23 to 29 run on each `BD_nuc_<sample>` with no changes.
3. Stage 29a: fit the interaction model, build Panel F.
4. Stage 31b: FLIGHT-pair matched sites.
5. Report: add Panels A/B rows, add Panels F/G.

## 7. Decisions

- D-pool278-1: pool bead-split replicates within each sample, three FLIGHT-nuclei libraries (`BD_nuc_211`, `BD_nuc_213`, `BD_nuc_215`). **Decided 2026-10-30.**
- D-pool278-2: mirror BD67's as10x retag pattern on the nine Pool278 BAMs rather than realigning from FASTQ. Initial decision on 2026-10-31 was to realign; revised the same day after inspecting `BD067_as10x.bam` headers showed BD67 itself is a tag-retrofit of BD's own STAR BAM rather than a GRCh38-2024-A STARsolo realignment. Realigning Pool278 while leaving BD67 on BD's reference would introduce a cross-reference confound; realigning both is out of scope for this addendum. **Revised 2026-10-31.**
- D-pool278-3: reuse the 10x nuclei marker panel for the FLIGHT nuclei. **Decided 2026-10-31.**
- D-pool278-4: library names `BD_nuc_211`, `BD_nuc_213`, `BD_nuc_215`; 211 is D29 (shared with 1419), 213 and 215 are two further Graz donors. **Decided 2026-10-31.**

## 8. Rough timing

Stage 1a retag + merge: ~70 min per sample on 4 threads with all three samples in parallel (observed 2026-10-31). Stage 1 to 29 reruns on each new library: half a day. Interaction model and Panel F: a day. Stage 31b: half a day. Total: 2 to 3 working days before the first draft of Panel F.
