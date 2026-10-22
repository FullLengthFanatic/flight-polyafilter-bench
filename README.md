# flight-polyafilter-bench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23562409.svg)](https://doi.org/10.5281/zenodo.23562409)

A reproducible evaluation of polyAfilter (Shulman and Elkon 2019) on retina organoid single-cell libraries. Compares four preps on the same differentiation batch: nuclei (10x 3' v4), dissociated cells (10x 3' v4), and FLIGHT-seq (BD Rhapsody) on week-26 and week-28 organoids. Cross-checked with two independent PacBio Kinnex long-read runs on the matched week-26 cells.

Companion to the FLIGHT-seq manuscript. Sister tools:
- [flightcollapse](https://github.com/FullLengthFanatic/flightcollapse) — long-read collapsing and barcode recovery
- [scNoiseMeter](https://github.com/FullLengthFanatic/scNoiseMeter) — single-cell noise quantification

## What this benchmark shows

Three claims the four-library comparison supports:

1. **Internal priming is the dominant 3' artefact in oligo-dT scRNA-seq.** Nuclei run 82% templated (internal priming), 10x cells 48.5%, FLIGHT-seq cells 26.8%. The ordering matches the physics of each prep.
2. **polyAfilter is tunable without matched bulk random-primed RNA-seq.** Hexamer-purity ratio above 3x is a cheap, defensible acceptance criterion; two independent long-read runs on the same cells confirm the short-read labels at 99.5% pooled agreement.
3. **FLIGHT-seq has lower internal-priming burden than matched 10x on the same cells, and more reproducible clustering structure after filtering.** Unfiltered clustering ARI ceiling is 0.89 for BD67 vs 0.75 for 1179 on the same week-26 pool.

## Getting started

```bash
git clone https://github.com/FullLengthFanatic/flight-polyafilter-bench.git
cd flight-polyafilter-bench
uv sync                    # pipeline runtime
uv sync --extra report     # also installs beautifulsoup4 + cairosvg for figure export
```

polyAfilter is vendored at a pinned commit (`5ce0098`) in `vendor/`. If you clone without `--recurse-submodules`, run `git submodule update --init` after cloning.

Reference data: GRCh38-2024-A (10x Genomics), PolyASite 2.0 (GRCh38, Gencode v42). Download instructions in `data/README.md` once you fetch the sample BAMs from GEO (accessions in the manuscript).

## Running the pipeline

Stages are numbered. Run in order per library, or pass `--libs <name>` to filter. Each stage prints one JSON line of what it wrote and exits.

```bash
python -m scripts.01_bd67_preprocess                   # BD67 only; 10x libraries skip this
python -m scripts.11_molecules --lib 1179
python -m scripts.13_grid --lib 1179                   # 14-setting sweep
python -m scripts.16_long_read_sites --libs BD67 10x_FL
python -m scripts.23_read_groups --lib 1179
python -m scripts.24_recount --lib 1179
python -m scripts.26_impact --lib 1179
python -m scripts.30_long_read_context --libs BD67 10x_FL
python -m scripts.34_m3_report
python -m scripts.90_export_figures                    # PNG exports of the four SVG figures
```

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

Zenodo DOI: [10.5281/zenodo.23562409](https://doi.org/10.5281/zenodo.23562409). See `CITATION.cff` for full metadata.

## License

MIT. See `LICENSE`.
