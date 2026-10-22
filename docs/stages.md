# Pipeline stages

Stages are numbered. Each stage reads from `--base` (default `/data/element_runs/polyafilter_bench`), writes under `--out`, and prints one JSON summary line on exit. Run order is left to right across milestones, library by library.

## Milestone 1: label and reimplementation

| # | Script | Purpose |
|---|---|---|
| 00 | `00_setup.sh` | Clone and build vendored polyAfilter |
| 01 | `01_build_db.sh` | Build the GTF annotation database used downstream |
| 01 | `01_bd67_preprocess.py` | Rewrite `BD067_Bioproduct.bam` to a 10x-compatible BAM (CB kept, UB from MR right-padded, GX from XF via features.tsv, pa:i:N via soft-clip rule) |
| 02 | `02_run_original.sh` | Run upstream polyAfilter at the three sweep settings for the reimpl match test |
| 03 | `03_reimpl_flags.py` | Run the reimplementation at the same settings |
| 04 | `04_compare_removed.py` | Byte-wise compare removed-read sets with upstream |
| 05 | `05_pa_sites.py`, `05b_ub_on_introns.py` | Site table and intronic UB survey |
| 06 | `06_molecules.py` | Collapse records into molecules per library |
| 07 | `07_label_curve.py` | Templated-share curve across cutoffs |
| 08 | `08_scope_check.py` | Scope checks for the milestone 1 note |
| 09 | `09_spikein_star.sh`, `09b_check_tags.py` | SIRV spike-in alignment and tag audit |
| 10 | `10_bd67_ends.py` | BD67 3' end audit (bead tail retention) |
| 11 | `11_m1_summary.py` | Milestone 1 summary |

## Milestone 2: truth, grid, evaluation

| # | Script | Purpose |
|---|---|---|
| 12 | `12_truth_sets.py` | Build the truth molecule tables (templated/non-templated/ambiguous) per library |
| 13 | `13_grid_flags.py` | 14-setting grid of covlen x lmax x mmax x introns |
| 14 | `14_spikein_truth.py` | SIRV truth against the grid |
| 15 | `15_ercc_counts.py` | ERCC counts for mix inference |
| 16 | `16_long_read_sites.py` | Per-site A-count table from long-read libraries; per-library `MIN_AFTER` applied |
| 17 | `17_evaluate.py` | Sensitivity / specificity against truth per setting per population |
| 18 | `18_burden.py` | Templated / non-templated / ambiguous shares per library per compartment |
| 19 | `19_validation_runs.sh`, `20_validation_compare.py` | Replay the paper's three settings and byte-compare removed sets |
| 21 | `21_m2_summary.py` | Milestone 2 summary |
| 22 | `22_polyasite_a_content.py` | PolyASite atlas A-content audit (template, not just rule-of-thumb) |

## Milestone 3: impact

| # | Script | Purpose |
|---|---|---|
| 23 | `23_read_groups.py` | Group reads by removal mask per setting |
| 24 | `24_recount.py` | Rebuild counts matrix per setting, with intron handling |
| 25 | `25_cell_classes.py` | Marker-based cell type assignment |
| 25b | `25b_donor_replicate.py` | Donor replicate robustness |
| 26 | `26_impact.py` | Sens / spec / UMI cost / ARI per setting per cell class |
| 27 | `27_truth_strata.py` | Truth stratified by region and A-band |
| 28 | `28_selection_bias.py` | Labeled vs unlabeled selection bias across axes |
| 29 | `29_purity_by_region.py` | Hexamer purity by region |
| 30 | `30_long_read_context.py` | Long-read 3' end context classification against GTF; vs-1179 agreement; tecap bucket per site |
| 31 | `31_matched_sites.py` | Matched-site table across libraries |
| 32 | `32_ercc_mix.py` | ERCC mix verdict (Mix 1 vs Mix 2) |
| 33 | `33_rerun.py` | Rerun diffs for CI |
| 34 | `34_m3_report.py` | Build `m3_report.html` and `numbers.json` (21 sections) |

## Export

| # | Script | Purpose |
|---|---|---|
| 90 | `90_export_figures.py` | Pull the four SVG figures from `m3_summary.html`, write PNGs at 2x viewBox |
| 92 | `92_main_text_figure.py` | Build the manuscript's two-panel main figure (burden + ARI) from the parquet files |

## Running a subset

Most stages take `--libs` (long-read) or `--lib` (one library) and `--refs` (chromosome subset). Use the subset flags for smoke tests before the full run.
