# flight-polyafilter-bench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23562408.svg)](https://doi.org/10.5281/zenodo.23562408)

A reproducible evaluation of polyAfilter (Svoboda, Frost, and Bosco 2022, NAR Genomics and Bioinformatics) on retina organoid single-cell libraries. Compares three preps on week-26 organoid cultures: nuclei (10x 3' v4, two donors), dissociated cells (10x 3' v4), and FLIGHT-seq (BD Rhapsody) on the same cell pool as the dissociated 10x. Cross-checked with three independent PacBio Kinnex long-read runs: the BD67 FLIGHT-seq long-read on the same week-26 pool, and two week-29 10x Kinnex runs (8-fold FL, 16-fold SC) for cross-platform corroboration of the A-count rule.

Companion to the FLIGHT-seq manuscript. Sister tools:
- [flightcollapse](https://github.com/irinakhven/flightcollapse) — long-read collapsing and barcode recovery (maintained by Irina Khven)
- [scNoiseMeter](https://github.com/FullLengthFanatic/scNoiseMeter) — single-cell noise quantification

## What this benchmark shows

Three claims the four-library comparison supports:

1. **Internal priming is the dominant 3' artefact in oligo-dT scRNA-seq.** Nuclei run 89% templated (internal priming) on the pooled 1417+1419 cohort, 10x cells (1179) 50%, FLIGHT-seq cells (BD67) 26%. The ordering matches the physics of each prep.
2. **polyAfilter is tunable without matched bulk random-primed RNA-seq.** Hexamer-purity ratio above 3x is a cheap, defensible acceptance criterion; cross-prep label agreement on the same week-26 pool (BD67 FLIGHT-seq long-read vs 1179 short-read) is 99.4% pooled, and cross-platform agreement with a week-29 10x Kinnex run is 99.5% pooled, consistent with the A-count rule being a reference-sequence property.
3. **FLIGHT-seq has lower internal-priming burden than matched 10x on the same cells, and more reproducible clustering structure after filtering.** Unfiltered clustering ARI median is 0.89 for BD67 vs 0.71 for 1179 on the same week-26 pool.

## Getting started

```bash
git clone https://github.com/FullLengthFanatic/flight-polyafilter-bench.git
cd flight-polyafilter-bench
uv sync                    # pipeline runtime
uv sync --extra report     # also installs beautifulsoup4 + cairosvg for figure export
```

polyAfilter is vendored at a pinned commit (`5ce0098`) under `vendor/polyAfilter/` as checked-in files, not a git submodule. No submodule init is needed after cloning.

Stage 30 (long-read context classification) calls [`tecap`](https://github.com/FullLengthFanatic/tecap) for 3' end structural buckets. Install it on PATH or run stage 30 with `--no-tecap` to skip that step; the rest of the long-read corroboration still runs.

Reference data: GRCh38-2024-A (10x Genomics), PolyASite 3.0 (GRCh38, GENCODE 42). Download instructions in `data/README.md` once you fetch the sample BAMs from GEO (accessions in the manuscript).

## Running the pipeline

Stages are numbered. Each stage prints one JSON line of what it wrote and exits. Per-library stages take a positional `library` argument (not `--lib`); multi-library stages take `--libs <name> ...`. Script filenames are the module names.

```bash
python -m scripts.01_bd67_preprocess                   # BD67 only; 10x libraries skip this
python -m scripts.11_m1_summary                         # aggregate milestone 1 (all libraries)
python -m scripts.13_grid_flags 1179                    # positional library arg; grid sweeps all 396 settings
python -m scripts.16_long_read_sites --libs BD67 10x_FL # plural --libs
python -m scripts.23_read_groups 1179
python -m scripts.24_recount 1179
python -m scripts.26_impact 1179
python -m scripts.30_long_read_context --libs BD67 10x_FL
python -m scripts.34_m3_report                          # no args
python -m scripts.90_export_figures                    # PNG + SVG + PDF exports of the four figures
```

The impact-analysis pipeline runs on the 14 settings in `pafbench/config.py:IMPACT_SETTINGS` (two chosen + four published settings in both modes + two cells-tuned + two remove-all controls); stage 13 evaluates the full 396-setting grid first.

Full stage inventory: `docs/stages.md` (TODO).

## Documentation

- [`docs/rationale.md`](docs/rationale.md) — why 20 nt, why 8/4 cutoffs, why no bulk, what we use instead
- [`docs/parameter_choices.md`](docs/parameter_choices.md) — decision table for cells vs nuclei × 10x vs FLIGHT-seq, with per-library sens/spec/UMI cost/ARI
- `docs/m1_results.md`, `docs/m2_results.md` — milestone exit notes

## Reproducibility

- `uv.lock` pins 90 packages for byte-identical reruns.
- `report/numbers.json` traces every digit in the figures back to a source function and column.
- The test suite (`tests/`, 607 tests) covers the labelling, filter, recount, and report builders.

## Scope

Tested on retina organoid 10x 3' v4 (cells and nuclei) and FLIGHT-seq (BD Rhapsody) with GRCh38-2024-A. The scope limits in `docs/rationale.md` apply.

## Citation

Zenodo concept DOI (always resolves to the latest version): [10.5281/zenodo.23562408](https://doi.org/10.5281/zenodo.23562408). See `CITATION.cff` for full metadata.

## License

MIT. See `LICENSE`.
