# Data

The benchmark operates on seven libraries from a single human retina organoid differentiation batch. All short-read data were sequenced on Illumina; the long-read data on PacBio Revio with Kinnex concatenation.

**Accessions and sample metadata to fill before submission.** Placeholders below; replace with the actual GEO / SRA accessions once the submission closes.

## Short-read libraries

| Library | Chemistry | Prep | Donor | Age | Cells (called) | Reads | GEO / SRA |
|---|---|---|---|---|---|---|---|
| 1417 | 10x 3' v4 | nuclei | D28 | week 26 | 11,232 | 420 M | GSM_TODO |
| 1419 | 10x 3' v4 | nuclei | D29 | week 26 | 11,784 | 410 M | GSM_TODO |
| 1416 | 10x 3' v4 | nuclei (20k) | D28 | week 26 | 20,145 | 980 M | GSM_TODO |
| 1179 | 10x 3' v4 | dissociated cells | D27 | week 26 | 24,378 | 580 M | GSM_TODO |
| BD67 (short) | FLIGHT-seq (BD Rhapsody) + 10x library kit | dissociated cells | D27 | week 26 | 3,137 | TODO | GSM_TODO |

## Long-read libraries

| Library | Chemistry | Linked to | Reads (FLNC) | Kinnex | GEO / SRA |
|---|---|---|---|---|---|
| BD67 long | FLIGHT-seq + Rhapsody long-read | matches BD67 short + 1179 cells | TODO | 8-fold | GSM_TODO |
| BD70 | FLIGHT-seq + Rhapsody long-read | sibling, same batch, week 28 | TODO | 8-fold | GSM_TODO |
| 10x_FL_Kinnex | Kinnex 8-fold on 1179 10x library | matched prep to 1179 | 79.6 M used | 8-fold (FL) | GSM_TODO |
| 10x_SC_Kinnex | Kinnex 16-fold on 1179 10x library | matched prep to 1179 | TODO | 16-fold (SC) | GSM_TODO |

Cell counts for the short-read libraries are post-Cell Ranger `filtered_feature_bc_matrix` cell calls. BD67's call is low because its cell panel was intentionally shallow for the method comparison, not because the chemistry fails on richer samples.

## Reference

- Genome: GRCh38-2024-A from 10x Genomics ([download](https://cf.10xgenomics.com/supp/cell-exp/refdata-gex-GRCh38-2024-A.tar.gz)).
- PolyASite atlas 2.0: `atlas.clusters.3.0.GRCh38.GENCODE_42.bed.gz` ([download](https://polyasite.unibas.ch/)).
- SIRV Set 4 reference FASTA + GTF: shipped by Lexogen with Set 4 reagent lot; version `20250402`.

## Where the files live on the processing VM

For reference and reproduction, the local layout is:

```
/data/element_runs/
  1417_HsRetina_D28_3k/outs/possorted_genome_bam.bam
  1419_HsRetina_D29_3k/outs/possorted_genome_bam.bam
  1416_HsRetina_D28_20k/outs/possorted_genome_bam.bam
  1179_organoid_wk26_10xv4/outs/possorted_genome_bam.bam
  BD67/
    BD067_Bioproduct.bam                       (BD Rhapsody native format)
    short_read/BD067_as10x.bam                 (preprocessed to 10x-compatible, scripts/01_bd67_preprocess.py)
    BD67_2x_RT.fltnc.aligned.bam               (long-read FLNC)
  BD70/BD70.fltnc.aligned.bam
  10x_FL_Kinnex/10x_FL.fltnc.aligned.bam       (matched long-read of 1179)
  10x_SC_Kinnex/10x_SC.fltnc.aligned.bam
  polyafilter_bench/                            (benchmark outputs, 20 GB)
```

Reviewers cloning the repo will not have these paths. The scripts take explicit `--base` and `--out` arguments; set those to local copies of the GEO downloads.

## Reproduction wall-clock

Fresh run from the FLNC BAMs on a 12-core / 128 GB VM:

- Stage 16 (long-read sites) on 10x_FL: 10-15 min
- Stage 23-26 (grid + recount + impact) on BD67: 45 min
- Stage 30 (long-read context) on BD67 + 10x_FL: ~2 h
- Stage 34 (m3 report): 20 s

Full pipeline from raw BAMs on four libraries: approximately 8 h wall-clock.
