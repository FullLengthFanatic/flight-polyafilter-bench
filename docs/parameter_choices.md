# Choosing settings for a new library

Scope: tested on retina organoid 10x 3' v4 (cells and nuclei) and FLIGHT-seq (BD Rhapsody) with GRCh38-2024-A. Treat the recommendations below as a starting point outside that scope, and run the hexamer-purity smoke test (`docs/rationale.md`) before trusting any of it.

## Decision table

| Your sample | Chemistry | Starting setting | UMI cost | ARI | Rationale |
|---|---|---|---|---|---|
| Dissociated cells | 10x 3' v4 | `c200_l10_m2_i` (mode-i) | 15.7% on 1179 | 0.66 | Cells read short, filter needs the introns-aware pass to catch internal-priming on short intronic A-stretches. Even if your pipeline does not count introns, the filter decides reads before counting. |
| Nuclei | 10x 3' v4 | `c250_l7_m1_e` (mode-e) for safety | 15.7% / 15.1% on 1417 / 1419 | 0.42 / 0.49 | Nuclei carry real intronic reads from pre-mRNA that mode-i will eat. Mode-e is the compromise: moderate specificity gain, intronic reads kept. If sensitivity matters more than UMI budget, run mode-i and know the UMI cost (54-55% on these libraries). |
| Dissociated cells | FLIGHT-seq (BD Rhapsody) | `c200_l10_m2_i` (mode-i) | 17.5% on BD67 | 0.68 | FLIGHT-seq baseline burden is already lower (27% templated vs 48.5% on 10x cells). The filter still improves clustering. The paper default (`c300_l10_m1_e`) at UMI cost 6.8% is also defensible if you want a lighter touch; ARI 0.71. |
| No bulk, no long reads | any | mode-e as the safer starting point | see per-row | see per-row | Mode-e is harder to over-correct. If the hexamer-purity ratio comes out below 3x on your library, step up to tighter cutoffs or run the sweep. |
| Have matched long reads | any | tune `min_after` per prep | n/a | n/a | BD-style FLIGHT-seq FLNC keeps the bead tail in sequence: `min_after=30`. 10x Kinnex FLNC has primer stripped by `isoseq refine`: `min_after=0`. |

## The two chosen settings, verbatim

### mode-e: `c250_l7_m1_e`

- `covlen=250`: how far inward from the site the coverage window reaches.
- `lmax=7`: minimum uninterrupted A-run length flagged for filtering.
- `mmax=1`: one allowed mismatch inside the A-run.
- `e`: introns excluded from the recount (standard 10x path).

Performance on the four libraries:

| Library | Sens | Spec | UMI cost | ARI |
|---|---|---|---|---|
| 1417 (nuclei) | 0.093 | 0.956 | 15.7% | 0.40 |
| 1419 (nuclei) | 0.094 | 0.962 | 15.1% | 0.49 |
| 1179 (cells) | 0.164 | 0.975 | 15.7% | 0.61 |
| BD67 (cells) | 0.193 | 0.968 | 24.0% | 0.65 |

### mode-i: `c200_l10_m2_i`

- `covlen=200`: narrower coverage window.
- `lmax=10`: longer uninterrupted A-run flagged.
- `mmax=2`: two mismatches allowed.
- `i`: introns included in the recount.

| Library | Sens | Spec | UMI cost | ARI |
|---|---|---|---|---|
| 1417 (nuclei) | 0.435 | 0.951 | 54.7% | 0.32 |
| 1419 (nuclei) | 0.440 | 0.958 | 51.3% | 0.42 |
| 1179 (cells) | 0.315 | 0.988 | 21.9% | 0.66 |
| BD67 (cells) | 0.190 | 0.985 | 17.5% | 0.68 |

The nuclei UMI cost in mode-i is the warning. If your pipeline includes introns and your sample is nuclei, mode-i will eat about half the UMI budget. Not necessarily wrong, but it must be a conscious choice.

### The paper default: `c300_l10_m1_e`

Kept as a sanity row.

| Library | UMI cost | ARI |
|---|---|---|
| 1417 (nuclei) | 4.7% | 0.50 |
| 1419 (nuclei) | 3.5% | 0.55 |
| 1179 (cells) | 7.3% | 0.70 |
| BD67 (cells) | 6.8% | 0.71 |

Light-touch filter. If you have no idea where to start and no long reads to tune against, this is a defensible default.

## The sweep, if you want to tune

`scripts/13_grid.py` runs a 14-setting grid. On a new library:

1. Preprocess the BAM with `scripts/01_bd67_preprocess.py` (BD) or feed Cell Ranger output directly (10x).
2. Run stages 11 to 15 to label molecules and classify cells.
3. Run `scripts/23_read_groups.py` and `scripts/24_recount.py` for every grid setting.
4. Run `scripts/26_impact.py` to get UMI cost and ARI per setting per population.
5. Read `impact/summary.json` and pick the setting that keeps specificity above 0.95 and ARI within one point of the ceiling at the lowest UMI cost.

Reasonable budget per library: about 90 min of wall-clock on 12 cores for the four libraries we tested; your mileage varies by read depth.

## When the hexamer-purity ratio fails

If stage 16 reports `ratio_8_4 < 3` on your library at the 8/4 cutoff, something is off. Either:
- The reference annotation is missing a lot of real 3' ends and real sites look templated.
- The library prep is non-standard (not oligo-dT) and the whole label system does not apply.
- The read depth is too low for a stable per-site A-count.

Fix the first by using a current annotation (GRCh38 2024-A or newer). The second kills polyAfilter entirely; use something else or do not filter. The third asks for more reads or site pooling.

## When long reads disagree with short reads

In our data, 10x_FL agreement with 1179 is 99.5% pooled, BD67 agreement is 99.4%. If you see sub-99% agreement, it is usually one of:
- Long-read primer-trimming settings. BD67 FLNC retains the bead tail; 10x Kinnex FLNC strips it. The `min_after` parameter must match.
- Reference build mismatch. The 20 nt downstream window is reference-dependent.
- Alignment to a different transcriptome (SIRV-only, pre-release GTF). Keep the long reads and short reads on the same reference.
