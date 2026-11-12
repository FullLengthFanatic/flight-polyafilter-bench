# Data

Eight short-read libraries and three PacBio Kinnex long-read libraries, all human retina. The pipeline reads two of the long-read libraries. Short reads were sequenced on Element AVITI 24, long reads on PacBio Revio with Kinnex concatenation.

**Accessions are placeholders.** Replace the `GSM_TODO` entries with the GEO / SRA accessions once the submission closes. The three FLIGHT-seq nuclei libraries (BD_nuc_211, 213, 215) are not deposited yet; they will be.

## Short-read libraries

| Library | Chemistry | Material | Donor or pool | Called cells | Reads | Role | GEO / SRA |
|---|---|---|---|---|---|---|---|
| 1417 | 10x 3' v4 | nuclei, adult retina (3k aliquot) | D28 | 5,571 | 97.4 M | settings are selected here | GSM_TODO |
| 1419 | 10x 3' v4 | nuclei, adult retina (3k aliquot) | D29 | 5,114 | 90.1 M | donor hold-out | GSM_TODO |
| 1416 | 10x 3' v4 | nuclei, adult retina (20k aliquot) | D28 | 19,392 | 686.5 M | pilot only | GSM_TODO |
| 1179 | 10x 3' v4 | whole cells, retinal organoid week 26 | organoid pool | 24,378 | 580.6 M | cells hold-out | GSM_TODO |
| BD67 (short) | FLIGHT-seq (BD Rhapsody), 10x 3' library kit | whole cells, retinal organoid week 26 | same pool as 1179 | 3,137 | 61.5 M | FLIGHT-seq cells | GSM_TODO |
| BD_nuc_211 | FLIGHT-seq (BD Rhapsody) | nuclei, adult retina | D29 (as 1419) | 32,089 | 316.9 M | FLIGHT-seq nuclei, burden only | not deposited yet |
| BD_nuc_213 | FLIGHT-seq (BD Rhapsody) | nuclei, adult retina | further donor | 34,545 | 342.5 M | FLIGHT-seq nuclei, burden only | not deposited yet |
| BD_nuc_215 | FLIGHT-seq (BD Rhapsody) | nuclei, adult retina | further donor | 37,775 | 294.7 M | FLIGHT-seq nuclei, burden only | not deposited yet |

The 10x cell calls are Cell Ranger's (`filtered_feature_bc_matrix`); the BD calls are those of the BD Rhapsody pipeline. Each `BD_nuc` library pools the three bead-split replicates (a, b, c) of one sample.

The nuclei are adult donor retina and the cells are retinal organoids. Any nuclei against cells comparison therefore mixes material with tissue.

## Long-read libraries

| Library | What it is | Relation to the short reads | Reads (FLNC) | Kinnex | GEO / SRA |
|---|---|---|---|---|---|
| BD67 long | FLIGHT-seq long reads | same cDNA as BD67 short, same cell pool as 1179 | 61.1 M | 8-fold | GSM_TODO |
| 10x_FL_Kinnex | 10x 3' v4 library of a week-29 retinal organoid | different pool from 1179 | 79.6 M used | 8-fold | GSM_TODO |
| 10x_SC_Kinnex | 10x 3' v4 library of a week-29 retinal organoid | different pool from 1179; listed for completeness, not read by the pipeline | 84.7 M | 16-fold | GSM_TODO |

## Reference

- Genome and annotation: GRCh38-2024-A from 10x Genomics ([download](https://cf.10xgenomics.com/supp/cell-exp/refdata-gex-GRCh38-2024-A.tar.gz)).
- PolyASite atlas 3.0: `atlas.clusters.3.0.GRCh38.GENCODE_42.bed.gz` ([download](https://polyasite.unibas.ch/)).
- SIRV Set 4 FASTA and GTF: Lexogen ([product page](https://www.lexogen.com/store/sirv-set4/)).

The BD libraries were aligned by the BD Rhapsody pipeline to its own GRCh38 reference, which shares the coordinates of GRCh38-2024-A. Stages 01 and 01a add 10x-style tags to those BAMs; they do not realign.

## File layout on the processing VM

```
/data/element_runs/
  1417_HsRetina_D28_3k/outs/possorted_genome_bam.bam
  1419_HsRetina_D29_3k/outs/possorted_genome_bam.bam
  1416_HsRetina_D28_20k/outs/possorted_genome_bam.bam
  1179_organoid_wk26_10xv4/outs/possorted_genome_bam.bam
  BD67/
    BD067_Bioproduct.bam                       (BD Rhapsody format)
    short_read/BD067_as10x.bam                 (10x-style tags, scripts/01_bd67_preprocess.py)
    BD67_2x_RT.fltnc.aligned.bam               (long reads)
  Pool278_FLIGHTseq_NucleiHsRetinaDonation/merged/
    BD_nuc_211/BD_nuc_211_as10x.bam            (scripts/01a_pool278_preprocess.py; same for 213, 215)
  10x_FL_Kinnex/10x_FL.fltnc.aligned.bam
  polyafilter_bench/                           (benchmark outputs)
```

These paths are set in `pafbench/config.py` (`RUNS`, `LIBRARIES`, `REF_DIR`, `POLYASITE`, `OUT`). To run elsewhere, edit that file to point at your copies.

## Run times

On a 12-core, 128 GB machine:

- Stage 13 (396 settings): 7 min on 1417, 23 min on 1179, 8 workers
- Stage 16 (long-read sites) on 10x_FL: 10 to 15 min
- Stages 23 to 26 (read groups, recount, classes, impact) on BD67: 45 min
- Stage 30 (long-read context) on BD67 and 10x_FL: about 2 h
- Stage 34 (report): under a minute

The whole pipeline on the first four libraries takes about 8 hours. That leaves out the original polyAfilter, which is only needed for the validation: about 2 hours per setting on 1417 after roughly 15 hours of one-off setup.
