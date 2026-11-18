# Changelog

## 1.2.0 (unreleased)

### Added

- `docs/definition.json` and `docs/definition.md`: the internal-priming definition as values that tecap and scNoiseMeter copy. It holds the site rule for short reads (alignment end of a read tagged pa: 20 or more adenines in its 3' soft clip, at least 0.9 of the clip adenine) and for long reads (start of the terminal adenine run, 10 or more adenines with one non-A allowed, MAPQ 20), the 0-based site boundary, the 20 nt label window with the cut-offs 8 and 4, and MAPQ 255 for short-read molecules. Stage 91 (`scripts/91_export_definition.py`) writes the JSON from `pafbench/config.py`; `tests/test_definition.py` fails while the file and the code disagree.
- Stage 29b (`scripts/29b_conventional_rule.py`): the internal-priming share under the conventional rule (6 or more consecutive adenines anywhere in the 20 nt window, or 12 or more of 20) on the uniquely mapped molecules of called cells, written to `burden/conventional_rule.parquet` and `.json` and shown in report section 22 beside the 8/4 share. Conventional share per library: 1417 0.644, 1419 0.639, 1179 0.317, BD67 0.107, BD_nuc_211 0.273, BD_nuc_213 0.228, BD_nuc_215 0.234.
- tecap from PATH with a version check. Stage 30 runs the `tecap` found on PATH instead of a fixed path, and `tests/test_setup.py` requires `tecap --version` to print `config.TECAP_VERSION`.

### Changed

- tecap pin 0.4.0 to 0.5.0. Stage 30 now runs `tecap classify` with `--site-anchor a_run_start`, so tecap takes a long read's 3' end at the benchmark's site, the start of its terminal adenine run. 0.5.0 also fixes an off-by-one at the last base of an exon on plus-strand genes. Rerun with 0.5.0 on the same accepted reads, stage 30's tecap step moves these bucket counts in `tecap_buckets.parquet` (reads, all labels summed):

  | Bucket | BD67, 0.4.0 | BD67, 0.5.0 | 10x_FL, 0.4.0 | 10x_FL, 0.5.0 |
  |---|---|---|---|---|
  | MechA-correct: in TE UTR, at polyA site, too short | 8,653,155 | 8,619,336 | 7,132,918 | 7,074,841 |
  | MechA-internalUTR: in TE UTR, no polyA site | 317,087 | 316,013 | 300,732 | 297,469 |
  | Internal priming in TE CDS (no UTR) | 992,481 | 1,022,903 | 1,591,775 | 1,649,127 |
  | MechA: in TE, non-coding gene | 174,829 | 174,799 | 223,485 | 223,489 |
  | MechB-APA: upstream TE, known polyA site | 4,111,705 | 4,212,487 | 7,982,613 | 8,145,835 |
  | MechB-exon: upstream TE, priming on upstream exon | 6,118,127 | 6,461,748 | 7,179,485 | 7,301,385 |
  | MechB-aspecific: upstream TE, intronic/flanking | 3,173,470 | 2,733,670 | 4,821,506 | 4,540,688 |
  | MechC: downstream of TE | 5,116 | 5,014 | 9,303 | 8,983 |

  Captured (>=50%) (BD67 20,725,810, 10x_FL 12,536,911), the classified reads of every label and every site, and the reads written and without a splice stay as they were. No document or `numbers.json` entry quotes these tables.

No existing number changed.
