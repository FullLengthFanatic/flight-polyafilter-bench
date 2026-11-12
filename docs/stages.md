# Pipeline stages

Stages are numbered scripts in `scripts/`, run in order. Python stages run as modules from the repository root, for example `python -m scripts.13_grid_flags 1417`. Each prints one JSON line describing what it wrote.

Three kinds of stage:

- **Per library**: the library name is a positional argument (`1417`, `1419`, `1179`, `BD67`, `BD_nuc_211`, ...). Run once per library.
- **Across libraries**: no library argument, or `--libs`. Run once.
- **Shell scripts** (`.sh`): run the original polyAfilter or STAR.

Input paths are in `pafbench/config.py`. Outputs go under `config.OUT`, one folder per library plus `eval/`, `burden/`, `validation/` and `report/`. Stages that take `--base` or `--out` can read and write elsewhere.

## Milestone 1: is the reimplementation the same filter, and where do labels come from

| Stage | Script | Scope | What it does |
|---|---|---|---|
| 00 | `00_setup.sh` | once | Creates the venv and fetches polyAfilter at the pinned commit |
| 01 | `01_build_db.sh` | once | Builds polyAfilter's annotation database from the GTF |
| 01 | `01_bd67_preprocess.py` | BD67 | Rewrites the BD Rhapsody BAM with 10x-style tags (UB, GX, RE, xf, pa) |
| 01a | `01a_pool278_preprocess.py` | per FLIGHT-seq nuclei sample | Same for the nine Pool278 BAMs, merged to one BAM per sample |
| 01b | `01b_pool278_build_matrix.py` | per FLIGHT-seq nuclei sample | Builds the per-sample barcode and count matrix folder |
| 02 | `02_run_original.sh` | per library and setting | Runs the original polyAfilter |
| 03 | `03_reimpl_flags.py` | per library | Runs the reimplementation at the same settings |
| 04 | `04_compare_removed.py` | per library | Compares the removed records of the two, record by record |
| 05 | `05_pa_sites.py` | per library | Collects the reads with a `pa` tag and the sequence around their priming site |
| 05b | `05b_ub_on_introns.py` | 1179 | Share of exonic and intronic reads with a UMI tag (chr1) |
| 06 | `06_molecules.py` | per library | Groups `pa` reads into molecules, finds the other reads of each molecule, estimates UMI collisions |
| 07 | `07_label_curve.py` | per library | Poly(A) signal rate by A count; the cut-offs it implies |
| 08 | `08_scope_check.py` | once | Checks that the 3k aliquot (1417) matches the 20k library (1416) on chr1 |
| 09 | `09_spikein_star.sh`, `09b_check_tags.py` | 1417, 1419 | Aligns unmapped reads to SIRV Set 4 and checks that the tags survive |
| 10 | `10_bd67_ends.py` | once | Checks how BD67 long reads mark the priming site |
| 11 | `11_m1_summary.py` | once | Writes `docs/m1_results.md` |

## Milestone 2: accuracy across the grid

| Stage | Script | Scope | What it does |
|---|---|---|---|
| 12 | `12_truth_sets.py` | per library | Labels the molecules (templated, non-templated, ambiguous) and lists the reads the filter is scored on |
| 13 | `13_grid_flags.py` | per library | Flags every record at all 396 settings in one pass |
| 14 | `14_spikein_truth.py` | 1417, 1419 | The same labels and flags for the spike-ins, from their annotated 3' ends |
| 15 | `15_ercc_counts.py` | 1417, 1419 | ERCC UMI counts against input amount, before and after filtering |
| 16 | `16_long_read_sites.py` | `--libs` | Priming sites from long reads; accepts a library on its poly(A) signal ratio |
| 17 | `17_evaluate.py` | once | Sensitivity and specificity of every setting, the selection rule, bootstrap; writes `eval/selection.json` |
| 18 | `18_burden.py` | once | Templated share by library, group and region, with bootstrap intervals |
| 19 | `19_validation_runs.sh` | once | Runs the original tool at the validation settings on 1417 and 1419 |
| 20 | `20_validation_compare.py` | once | Compares original and reimplementation on every completed run, plus 1179 on chr21 and chr22 |
| 21 | `21_m2_summary.py` | once | Writes `docs/m2_results.md` |
| 22 | `22_polyasite_a_content.py` | once | Counts how many PolyASite 3.0 sites are followed by 14 or more A's (the atlas is not used as a label check) |

## Milestone 3: what filtering costs

| Stage | Script | Scope | What it does |
|---|---|---|---|
| 23 | `23_read_groups.py` | per library | Groups the reads of called cells by cell, UMI and gene, and records which settings remove each read |
| 24 | `24_recount.py` | per library | Counts UMIs before filtering and at each of the 14 impact settings; checks the recount against Cell Ranger's matrix |
| 25 | `25_cell_classes.py` | per library | Clusters the unfiltered matrix and names the clusters from retinal marker genes |
| 25b | `25b_donor_replicate.py` | once | Compares class proportions between the two donors (1417, 1419) |
| 26 | `26_impact.py` | per library | UMI cost, clustering agreement (ARI) with its thinning control, marker genes and class stability per setting |
| 27 | `27_truth_strata.py` | once (all libraries by default) | Sensitivity and specificity by cell class and expression level |
| 28 | `28_selection_bias.py` | once (all libraries by default) | Compares labeled molecules with all counted molecules |
| 29 | `29_purity_by_region.py` | once | Templated share corrected for label purity, overall and by region |
| 29a | `29a_interaction.py` | once | Chemistry by material model on the seven libraries |
| 30 | `30_long_read_context.py` | `--libs` | Where long-read 3' ends fall in the annotation, how many short-read sites they confirm (with a shifted-site null), and tecap classes |
| 31 | `31_matched_sites.py` | once | Priming sites shared between 10x nuclei and 10x cells |
| 31b | `31b_matched_sites_flightpair.py` | once | The same for FLIGHT-seq nuclei and FLIGHT-seq cells |
| 32 | `32_ercc_mix.py` | 1417, 1419 | Tests which ERCC mix is in the library |
| 33 | `33_rerun.py` | optional | Reruns Cell Ranger on one filtered BAM and compares it with the recount |
| 34 | `34_m3_report.py` | once | Writes the full report (`report/m3_report.html`) and `numbers.json` |

The burden stages (18, 29, 29a, 31, 31b) and the figures count only molecules whose poly(A) reads map uniquely (`mapq_min` 255 in `truth/molecules_labeled.parquet`). The accuracy and selection stages (17, 27, 28) use all labeled molecules; stage 17 also reports the metrics without the multi-mapped ones (`mapq_sensitivity`).

## Figures

| Stage | Script | What it does |
|---|---|---|
| 90 | `90_export_figures.py` | Supplementary figures (burden by library, burden by region, UMI loss by setting) as PNG, SVG and PDF |
| 92 | `92_main_text_figure.py` | The two-panel main figure (burden and ARI) |

## Trying a stage on a subset

Stages 05, 13, 16, 22, 23, 24 and 30 take `--refs` to limit the run to some chromosomes, for example `--refs chr21 chr22`. Use it for a first trial before a full run.
