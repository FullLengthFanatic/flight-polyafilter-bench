# flight-polyafilter-bench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23562408.svg)](https://doi.org/10.5281/zenodo.23562408)

How much of a 3' single-cell library comes from internal priming, and how well does polyAfilter remove it? This repository measures both on seven human retina libraries: 10x 3' v4 and FLIGHT-seq (BD Rhapsody), whole cells and nuclei. It is a companion to the FLIGHT-seq manuscript.

Sister tools: [flightcollapse](https://github.com/irinakhven/flightcollapse) (long-read collapsing and barcode recovery, maintained by Irina Khven), [scNoiseMeter](https://github.com/FullLengthFanatic/scNoiseMeter) (single-cell noise quantification) and [tecap](https://github.com/FullLengthFanatic/tecap) (3' end classification of long reads).

## The problem

3' protocols start reverse transcription with an oligo(dT) primer that is meant to bind the poly(A) tail. The primer also binds A-rich stretches inside a transcript. That is internal priming. The resulting reads carry a cell barcode and a UMI and are counted like any other, but they do not mark a real 3' end.

[polyAfilter](https://github.com/MarekSvob/polyAfilter) (Svoboda, Frost and Bosco 2022, [doi:10.1093/nargab/lqac035](https://doi.org/10.1093/nargab/lqac035)) removes reads that lie just upstream of an A stretch in the genome. It has three parameters and an intron switch, and the paper tunes them against random-primed bulk RNA-seq of the same sample. We had no bulk data, so this benchmark takes its reference from the libraries themselves.

## How it works

1. **Find reads with a known priming site.** Some reads run into the poly(A). The aligner clips those A's, so the end of the alignment is where the primer sat.
2. **Label the site from the genome.** Count the A's in the 20 nt of genome after the site. With 8 or more, the A's are in the genome and the primer bound inside the transcript: internal priming (`templated` in the code). With 4 or fewer, the A's came from a poly(A) tail: a real 3' end (`non-templated`). Sites with 5 to 7 are ambiguous and stay out of the accuracy scores.
3. **Score the filter on the rest of the molecule.** The other reads with the same cell barcode and UMI belong to the same molecule, and polyAfilter has to judge them without seeing the site. Sensitivity is the share of internally primed molecules it removes. Specificity is the share of tail-primed molecules it keeps.
4. **Sweep the parameters.** A reimplementation flags 396 settings in one pass over the BAM. It removes exactly the same records as the original tool in all 27 runs we compared.
5. **Measure the cost.** At 14 settings the UMIs are recounted per cell and the cells are clustered again.

The reasoning behind each choice is in [`docs/rationale.md`](docs/rationale.md).

## What we found

Share of labeled molecules that come from internal priming, in called cells. Only uniquely mapped molecules are counted:

| Library | Chemistry | Material | Labeled molecules | Internal priming | Corrected |
|---|---|---|---|---|---|
| 1417 | 10x 3' v4 | nuclei, adult retina | 84,889 | 89.5% | 87.1% |
| 1419 | 10x 3' v4 | nuclei, adult retina | 99,626 | 88.6% | 85.9% |
| 1179 | 10x 3' v4 | cells, retinal organoid | 1,122,255 | 48.4% | 39.0% |
| BD67 | FLIGHT-seq | cells, same pool as 1179 | 161,814 | 19.4% | 11.7% |
| BD_nuc_211 | FLIGHT-seq | nuclei, adult retina | 584,311 | 41.0% | 34.6% |
| BD_nuc_213 | FLIGHT-seq | nuclei, adult retina | 919,029 | 34.9% | 27.6% |
| BD_nuc_215 | FLIGHT-seq | nuclei, adult retina | 637,685 | 35.3% | 27.6% |

"Corrected" subtracts the real poly(A) sites that happen to be followed by A's in the genome, estimated from the poly(A) signal rate. Molecules whose poly(A) reads map to several places are left out, because they inflate the share in the FLIGHT-seq libraries (26% instead of 19% in BD67). Both points are explained in the rationale.

- **FLIGHT-seq carries less than half the internal priming of 10x.** On the same organoid cell pool: 19% against 48%. In adult retina nuclei: 35 to 41% against 89%.
- **Nuclei carry more than cells on both chemistries.** The gap is 41 points on 10x and 18 on FLIGHT-seq. Tissue differs too (adult retina against organoid), so this comparison is descriptive.
- **polyAfilter is specific but not sensitive.** The table below gives sensitivity, specificity and the share of UMIs removed at the two settings our rule picked and at the example setting of the polyAfilter README.

| Setting | 10x nuclei (1417) | 10x cells (1179) | FLIGHT-seq cells (BD67) |
|---|---|---|---|
| `c250_l7_m1_e`, exons only | 0.09 / 0.96 / 16% | 0.16 / 0.98 / 32% | 0.19 / 0.97 / 24% |
| `c200_l10_m2_i`, with introns | 0.44 / 0.95 / 55% | 0.32 / 0.99 / 22% | 0.19 / 0.98 / 17% |
| `c300_l10_m1_e`, README example | 0.03 / 0.99 / 5% | 0.06 / 1.00 / 8% | 0.08 / 0.99 / 7% |

A setting name reads `c<COVLEN>_l<MINSNRLEN>_m<MISM>_<e or i>`. In nuclei, catching 44% of the internally primed molecules costs 55% of the UMIs. Which setting to start from, and what each one costs, is in [`docs/parameter_choices.md`](docs/parameter_choices.md).

Three limits to keep in mind. The shares describe labeled molecules (those with a read that reaches the poly(A)), not every molecule in the library. Nuclei are adult donor retina and cells are organoids, so material and tissue cannot be separated. And there is no bulk reference: the labels rest on the A count and the poly(A) signal check.

## Getting started

```bash
git clone https://github.com/FullLengthFanatic/flight-polyafilter-bench.git
cd flight-polyafilter-bench
uv sync                    # pipeline runtime
uv sync --extra report     # also installs beautifulsoup4 + cairosvg for figure export
git clone https://github.com/MarekSvob/polyAfilter.git vendor/polyAfilter
git -C vendor/polyAfilter checkout --quiet 5ce0098
```

The last two lines fetch the original polyAfilter at the commit this benchmark was validated against. It is not part of this repository (`vendor/` is git-ignored), and `tests/test_setup.py` checks the commit.

Use Python 3.11, which `.python-version` pins. The results were produced with it; on 3.12 and later `uv.lock` resolves pandas 3 and seven tests fail on its string dtype.

Stage 30 calls [`tecap`](https://github.com/FullLengthFanatic/tecap). Install it on PATH, or run stage 30 with `--no-tecap`; the rest of the long-read analysis still runs.

Inputs are Cell Ranger BAMs (BD Rhapsody BAMs are converted by stages 01 and 01a), the GRCh38-2024-A reference from 10x Genomics and the PolyASite 3.0 atlas. [`data/README.md`](data/README.md) lists the libraries and where each file comes from. Input paths are set in `pafbench/config.py`.

## Running the pipeline

Stages are numbered scripts. Each prints one JSON line describing what it wrote. Stages that work on one library take its name as a positional argument; stages that work across libraries take `--libs` or no argument.

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
python -m scripts.90_export_figures                    # PNG + SVG + PDF exports of the supplementary figures
```

These are examples, not the full run order. Every stage, in order, is in [`docs/stages.md`](docs/stages.md).

## Documentation

| File | What it answers |
|---|---|
| [`docs/rationale.md`](docs/rationale.md) | How the benchmark works and why each choice was made |
| [`docs/parameter_choices.md`](docs/parameter_choices.md) | Which setting to start from on your own library, with the numbers per library |
| [`docs/stages.md`](docs/stages.md) | Every pipeline stage in run order |
| [`data/README.md`](data/README.md) | Libraries, references, file layout, run times |
| [`docs/index.html`](docs/index.html) | Summary page of the seven-library comparison (chemistry by material) |
| [`docs/manuscript_methods.md`](docs/manuscript_methods.md), [`docs/supplementary_methods.md`](docs/supplementary_methods.md) | Methods text drafted for the manuscript |
| [`docs/m2_results.md`](docs/m2_results.md) | Every milestone 2 table (accuracy by setting, A count and region), written by stage 21 from the current outputs |
| `docs/specs/`, `docs/plans/`, `docs/m1_*.md` | Design documents and milestone 1 notes, kept as the record of each decision |

## Reproducibility

- `uv.lock` pins every dependency.
- The reimplementation is checked against the original tool record for record: ten settings on 1417, eight on 1419, eight on chr21 and chr22 of 1179, and one remove-all control (stage 20).
- The test suite (`tests/`, about 800 tests) runs on synthetic data and needs none of the libraries. `.github/workflows/clean-checkout.yml` runs the install block above in an empty directory, then the tests, on every push and pull request.
- Stage 34 writes the full report (`m3_report.html`) with `numbers.json`, which names the file, column and function behind every number in it.

## Scope

Tested on human retina (adult donor nuclei and organoid cells), 10x 3' v4 and FLIGHT-seq, GRCh38-2024-A. Other tissues, organisms and chemistries are untested; `docs/rationale.md` lists what the method assumes.

## Citation

Zenodo concept DOI (always resolves to the latest version): [10.5281/zenodo.23562408](https://doi.org/10.5281/zenodo.23562408). See `CITATION.cff` for full metadata.

## License

MIT. See `LICENSE`.
