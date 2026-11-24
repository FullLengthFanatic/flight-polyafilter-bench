# Changelog

## 1.2.0 (unreleased)

### Added

- `docs/definition.json` and `docs/definition.md`: the internal-priming definition as values that tecap and scNoiseMeter copy. It holds the site rule for short reads (alignment end of a read tagged pa: 20 or more adenines in its 3' soft clip, making up at least 90% of the clip) and for long reads (start of the terminal adenine run, 10 or more adenines with one non-A allowed, MAPQ 20, followed by 30 nt of clipped bead sequence in BD67 and none in 10x_FL), the 0-based site boundary, the 20 nt label window with the cut-offs 8 and 4, and MAPQ 255 for short-read molecules. Stage 91 (`scripts/91_export_definition.py`) writes the JSON from `pafbench/config.py`; `tests/test_definition.py` fails while the file and the code disagree.
- Stage 29b (`scripts/29b_conventional_rule.py`): the internal-priming share under the conventional rule (6 or more consecutive adenines anywhere in the 20 nt window, or 12 or more of 20) on the uniquely mapped molecules of called cells, written to `burden/conventional_rule.parquet` and `.json` and shown in report section 22 beside the 8/4 share. Conventional share per library: 1417 0.644, 1419 0.639, 1179 0.317, BD67 0.107, BD_nuc_211 0.273, BD_nuc_213 0.228, BD_nuc_215 0.234.
- tecap from PATH with a version check. Stage 30 runs the `tecap` found on PATH instead of a fixed path and, unless run with `--no-tecap`, stops before its scan when `tecap --version` does not print `config.TECAP_VERSION`; `tests/test_setup.py` checks the same.

### Changed

- tecap pin 0.4.0 to 0.5.0. Stage 30 now runs `tecap classify` with `--site-anchor a_run_start` and tecap's default `--strandedness auto`. tecap probes the first 200,000 reads with a gene and calls both libraries stranded (BD67 0.98971 of them on their best-overlap gene's strand, 10x_FL 0.98545). On stranded input a read antisense to its best-overlap gene is classified against the best-overlap gene on its own strand (`antisense_reassigned` in tecap's JSON: BD67 1,440,567 reads, 10x_FL 1,369,941) or, when no gene on its strand overlaps it, skipped and left out of every count below (`antisense_skipped`: BD67 145,386, 10x_FL 337,781). 0.4.0 classified both kinds against the antisense gene. Every classified read is now on its gene's strand. tecap looks for the terminal adenine run at the read's 3' end on that strand and, when the run holds 10 or more adenines, places the read's 3' end at its start, which is the benchmark's site: every classified read has one (tecap's `site_counts` fallback is 0), equal to its stage 30 site key (BD67 44,126,394 reads, 10x_FL 41,440,947). 0.5.0 also fixes an off-by-one on plus-strand genes in the tests against upstream exons and polyA-cluster windows. Rerun with 0.5.0 on the same accepted reads, stage 30's tecap step changes these bucket counts in `tecap_buckets.parquet` (reads, all labels summed):

  | Bucket | BD67, 0.4.0 | BD67, 0.5.0 | 10x_FL, 0.4.0 | 10x_FL, 0.5.0 |
  |---|---|---|---|---|
  | Captured (>=50%) | 20,725,810 | 21,412,000 | 12,536,911 | 13,024,676 |
  | MechA-correct: in TE UTR, at polyA site, too short | 8,653,155 | 8,746,816 | 7,132,918 | 7,227,294 |
  | MechA-internalUTR: in TE UTR, no polyA site | 317,087 | 313,311 | 300,732 | 293,349 |
  | Internal priming in TE CDS (no UTR) | 992,481 | 1,037,641 | 1,591,775 | 1,684,908 |
  | MechA: in TE, non-coding gene | 174,829 | 119,708 | 223,485 | 168,742 |
  | MechB-APA: upstream TE, known polyA site | 4,111,705 | 4,214,920 | 7,982,613 | 8,189,632 |
  | MechB-exon: upstream TE, priming on upstream exon | 6,118,127 | 6,772,666 | 7,179,485 | 7,525,449 |
  | MechB-aspecific: upstream TE, intronic/flanking | 3,173,470 | 1,506,878 | 4,821,506 | 3,317,631 |
  | MechC: downstream of TE | 5,116 | 2,454 | 9,303 | 9,266 |
  | All buckets | 44,271,780 | 44,126,394 | 41,778,728 | 41,440,947 |

  All of the Captured gain and most of the MechB-aspecific drop come from the gene choice: against a 0.5.0 run on the same reads that keeps the either-strand gene, it moves MechB-aspecific by BD67 -1,226,792 and 10x_FL -1,223,057 and Captured by +686,190 and +487,765; the anchor and the plus-strand fix move the rest. The classified reads per label (`per_label` of the `tecap` key in `isoform_context.json`) lose the skipped reads:

  | Label | BD67, 0.4.0 | BD67, 0.5.0 | BD67 reassigned | BD67 skipped | 10x_FL, 0.4.0 | 10x_FL, 0.5.0 | 10x_FL reassigned | 10x_FL skipped |
  |---|---|---|---|---|---|---|---|---|
  | templated | 16,386,559 | 16,285,159 | 408,095 | 101,400 | 23,085,250 | 22,784,826 | 573,134 | 300,424 |
  | nontemplated | 17,981,169 | 17,963,059 | 642,182 | 18,110 | 11,254,216 | 11,240,521 | 471,369 | 13,695 |
  | ambiguous | 9,904,052 | 9,878,176 | 390,290 | 25,876 | 7,439,262 | 7,415,600 | 325,438 | 23,662 |

  The sites with a classified read go from 1,032,858 to 1,001,850 (BD67) and from 2,471,424 to 2,358,613 (10x_FL); the reads written and without a splice stay as they were. No document, report section or `numbers.json` value quotes these tables or counts; only the file sizes in the file list of report section 20 change with them.

No existing number changed.
