# Internal-priming definition

How this benchmark finds a priming site, labels it and turns the labels into a rate. `docs/definition.json` carries the same values for tecap and scNoiseMeter, which copy it verbatim and name the benchmark version it came from (`source`, `version`). `pafbench/definition.py` builds it from `pafbench/config.py`, and stage 91 (`python -m scripts.91_export_definition`) writes it; `tests/test_definition.py` fails while the file and the code disagree.

| `definition.json` | Value | `pafbench/config.py` |
|---|---|---|
| `site.short_read.clip_min`, `clip_purity_min` | 20, 0.9 | `SHORT_READ_CLIP_MIN`, `SHORT_READ_CLIP_PURITY` |
| `site.long_read.run_min`, `run_tolerance` | 10, 1 | `LONG_READ_RUN_MIN`, `LONG_READ_RUN_TOLERANCE` |
| `site.long_read.min_mapq` | 20 | `LONG_READ_MIN_MAPQ` |
| `site.long_read.min_after` | BD67 30, 10x_FL 0 | `MIN_AFTER` |
| `label.window` | 20 | `SITE_WINDOW` |
| `label.templated_min`, `nontemplated_max` | 8, 4 | `LABEL_CUTOFFS` |
| `rate.unique_mapq` | 255 | `UNIQUE_MAPQ` |

## Site

A read without an observed site gets no label; its 3' end is not a priming site. Short and long reads observe the site differently, so there are two rules.

- **Short reads** (`anchor: alignment_end`). Only reads with a clipped poly(A) of 20 nt or more have a site: the Cell Ranger `pa` tag, or for BD Rhapsody BAMs the tag that stages 01 and 01a derive from the 3' soft clip (20 or more adenines, making up at least 90% of the clip). The site is the alignment end of such a read (`sites.priming_site`, stage 05). The run rule below is not applied to short reads: it would move the site upstream whenever aligned adenines precede the clip, and every short-read label in the benchmark rests on the alignment end.
- **Long reads** (`anchor: a_run_start`). The site is the first base of the read's terminal adenine run, aligned and clipped adenines counted together, with one non-adenine allowed inside the run (`bd67.terminal_a_run`, stages 16 and 30). A read is accepted when the run holds 10 or more adenines and enough clipped sequence follows it: 30 nt of bead sequence for BD67, none for 10x_FL, whose barcode and UMI `isoseq refine` moved into tags. BD67 and 10x_FL long reads keep their poly(A), so the aligner can align the genome-encoded adenines and put the alignment end past them; the run start does not move with that. A run with no aligned base gives the alignment end.

## Coordinates

The site is a 0-based boundary between two reference bases.

- **Plus strand**: the first reference base after the last cDNA base (`reference_end`). A read aligned to [1000, 1100), 1-based 1001 to 1100, has its last cDNA base at 1099 and its site at 1100. The window is [1100, 1120), 1-based 1101 to 1120, read as is.
- **Minus strand**: the last cDNA base is the base at the site (`reference_start`). The same alignment on the minus strand has its last cDNA base at 1000 (1-based 1001) and its site at 1000. The window is the reverse complement of [980, 1000), 1-based 981 to 1000, so its adenines are the thymines of the genome there.

Long reads follow the same convention at the run start. A plus-strand read with six aligned adenines at 1094 to 1099 before its clip has its site at 1094, not 1100; a minus-strand read with six aligned thymines at 1000 to 1005 has its site at 1006. Region and gene are looked up at the last cDNA base: site - 1 on plus, the site itself on minus.

## Label

Count the adenines in the 20 nt of reference after the site, on the transcript strand, as an integer. 8 or more is **templated** (internal priming). 4 or fewer is **non-templated** (tail priming, a real 3' end). 5 to 7 is **ambiguous**. The rule is the same for short and long reads (`labels.label_molecules`). The window is 20 nt because that is the shortest clipped poly(A) that gets a `pa` tag.

## Rate

The unit is the molecule (cell barcode, UMI, strand, gene), and the burden counts only uniquely mapped ones: every `pa` read of the molecule at MAPQ 255 (`mapq_min` in `truth/molecules_labeled.parquet`; stages 18, 29, 29a, 29b, 31, 31b and the figures). Sensitivity and specificity are scored on the templated and non-templated molecules at any MAPQ. Long reads enter as primary records with MAPQ 20 or more (`bd67.scan_long_reads`, `lrcontext.scan_context`).

Two shares, both from `burden.share_table` (stage 18, `burden/templated_share.parquet`):

- `share_all` = templated / (templated + non-templated + ambiguous). The headline, the "Internal priming" column of the README. BD67: 0.194.
- `share_resolved` = templated / (templated + non-templated). BD67: 0.264.

Ambiguous molecules are left out of sensitivity and specificity, not out of `share_all`. A tool that copies this definition divides by all labeled reads, so that its three fractions sum to one and compare with `share_all`.

The corrected share is a benchmark-only quantity. Stage 29 (`evaluate.tail_share`, `burden.band_corrected_share`) subtracts the real poly(A) sites inside the templated class, estimated from the poly(A) signal rate of each adenine band against the rate at 14 or more adenines (README "Corrected" column; BD67: 0.117). tecap and scNoiseMeter do not compute it; their label shares are raw.

## Why 8 and 4

On library 1417, all barcodes, the share of sites with a poly(A) signal (AATAAA or ATTAAA) 5 to 35 nt upstream is 0.77 at 0 to 4 adenines, 0.54 / 0.42 / 0.27 at 5 / 6 / 7, 0.18 at 8, 0.06 to 0.08 at 10 to 13 and 0.02 at 18 to 20. The cut-offs sit where the curve comes within a quarter of that distance of either end (`labels.choose_cutoffs`): from 8 adenines on the rate stays at or below 0.208, and up to 4 adenines it stays at or above 0.584. They were fixed before any filter result was looked at. The full curve is in `docs/rationale.md`.

The 14/8 pair in `sites.classify`, `scripts/16_long_read_sites.py:18` and `scripts/08_scope_check.py:17` is the milestone 1 pair, kept for the old outputs; it is not part of this definition.
