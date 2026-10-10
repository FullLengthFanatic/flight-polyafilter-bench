# polyAfilter benchmark, milestone 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer spec Q1 (how much of each library comes from internal priming, by region, in nuclei and in cells) and Q2 (how well polyAfilter separates internally primed from tail-primed molecules across 396 settings in both modes, with the setting chosen on donor 28 and held out on donor 29 and on organoid cells), with spike-in and long-read checks and a wider exact-match test against the original tool.

**Architecture:** Same layout as milestone 1: tested logic in `pafbench/`, numbered stage scripts in `scripts/`, outputs under `/data/element_runs/polyafilter_bench/`. Stage 12 turns the `pa` molecules into labeled truth sets. Stage 13 streams each library once per reference and reproduces polyAfilter's removal decision for all 396 settings at the same time, keeping only what later stages need: per-read flags for truth reads, removed counts by RE region and by called cell. Stages 14 (spike-ins), 15 (ERCC counts) and 16 (long reads) add checks that do not depend on the A-count label. Stages 17 and 18 compute metrics, the parameter choice and the burden tables; 19 and 20 extend the record-for-record comparison with the original tool; 21 writes `docs/m2_results.md`.

**Tech Stack:** Python 3 in `~/polyafilter_bench/.venv` (pysam, numpy, numba, pandas 3, pyarrow, scipy, gffutils), polyAfilter 5ce0098 vendored, samtools, STAR 2.7.10a (stage 09), GNU time, `python -m pafbench.memwatch` for process-tree memory.

**Spec:** `docs/specs/2026-10-07-polyafilter-bench-design.md`. Milestone 1 exit note: `docs/m1_note.md`.

## Global Constraints

- polyAfilter pinned at commit 5ce0098 in `vendor/polyAfilter`; never edited.
- Reference `/home/simone.picelli/Documents/Genome_ref/refdata-gex-GRCh38-2024-A` (`fasta/genome.fa`, `genes/genes.gtf`); gffutils database `/data/element_runs/polyafilter_bench/ref/genes_2024A.db`.
- Libraries: 1417 (donor 28 nuclei), 1419 (donor 29 nuclei), 1179 (retinal organoid week 26 cells); BD67 and BD70 long reads from the 1179 cell pool. Spike-ins exist only in 1417 and 1419.
- Acceptance (spec 5.2): "the removed-read sets are identical to the original tool on 1417 and 1419 for every setting run in step 1. Any difference is resolved before results are used." 1179 does not fit in memory in full (spec 5.1 and 11), so it is checked by chromosome: chr21 and chr22 at the 8 published settings.
- Labels (spec 4.1, user decision 2026-10-08): templated at 8 or more A's in the 20 nt of genome after the site, non-templated at 4 or fewer, ambiguous 5 to 7; every metric by A band 8-9, 10-13, 14-20; sweep templated 7 to 10 and non-templated 2 to 6, non-templated always below templated.
- Members: same molecule key, `gene_match`, `dist3_exon >= -25`, `dist5_exon <= 1000` (exon-aware); sensitivity reruns at 500 and 2,000.
- Grid (spec 6): COVLEN 150, 200, 250, 300, 375, 450, 600; MINSNRLEN none (remove all non-terminal reads, reported separately), 3, 5, 7, 9, 10, 11, 13, 15, 20; MISM 0, 1, 2 (MISM 0 only for none); default and `-i` mode: 392 settings, plus 275/9/1 and 180/11/2 in both modes = 396.
- Selection rule (spec 7, fixed before results): "on 1417 (donor 28), in each mode, take the setting with the highest molecule-level sensitivity subject to specificity of at least 0.95; break ties by fewer total reads removed. Report that setting, plus the published settings, on 1419 (donor 29, a donor-level hold-out) and on 1179 without retuning. If no setting reaches 0.95, report the frontier and do not pick one."
- Long-read acceptance (spec 4.3, user decision 2026-10-08): terminal A run of at least 10 nt counting aligned and clipped A's with one non-A allowed, followed by at least 30 nt of soft clip; site at the run start; statistics per unique site; BD67/BD70 enter later analyses when the hexamer rate at tail-primed sites is at least 3 times the rate at internally primed sites.
- Region of a site: at the last cDNA base (site - 1 on plus, site on minus), priority 3UTR > CDS > 5UTR > ncexon > intron on the site's strand; "antisense" when the site's strand is unannotated but an opposite-strand transcript covers it; "intergenic" otherwise.
- Machine: 117 GB RAM, 32 cores. The largest single original-tool process in milestone 1 peaked at 46.0 GB (375/3/2 `-i` on 1417). Run heavy steps under `.venv/bin/python -m pafbench.memwatch <json> -- <command>`, which records the peak summed over the whole process tree.
- Conventions: `.venv/bin/python`; packages only through `uv pip install --python .venv/bin/python`; commits as `git -c user.name="Simone Picelli" -c user.email="simone.picelli@iob.ch" commit`; no em or en dashes in code, docs or messages; delete with `gio trash`, never `rm -rf`.

## Review Focus

Five inputs the spec implies that are easy to get wrong; each has a test in the task that owns the code:

1. A reference with reads but no expressed transcript on one strand, or with no truth reads (chrM, small contigs, spike-in contigs): stage 13 writes an empty truth part for it but keeps its region totals (Task 3).
2. Stage 13 restarted after a crash: finished references are skipped, the merged tables equal a fresh run, and a part built from an older truth file is refused (Task 3).
3. No grid setting reaches specificity 0.95 in a mode: the selection is null with the frontier reported, and bootstrap, hexamer check, ERCC and the summary still run for the published settings and the control (Tasks 6, 7, 9).
4. A truth read missing from the flag table (for example on a reference that was not scanned): its molecule is dropped and counted, never treated as kept (Task 6).
5. 1179 has no spike-ins: stages 17 and 18 run with spike-in files present for 1417 and 1419 only and say so (Task 6).

## Order and timing

Task 1 starts the original-tool runs in the background (about 22 h for the 1419 chain, 14 h for the 1417 chain) and the 1179 by-chromosome check (about 5.4 h); Tasks 2 to 7 run meanwhile; Task 8 waits for them. Peak memory during the overlap: one 46 GB original-tool process (375/3/2 `-i` on 1419) plus one smaller run, the 1179 check (up to 10 GB) and stage 13 at 8 workers (about 14 GB on 1179), well under 117 GB.

---

### Task 0: Foundation: constants, settings grid, MINSNRLEN 0 guard, molecule and label helpers, region annotator, memory sampler

**Files:**
- Create: `tests/test_grid.py`
- Modify: `tests/test_labels.py`
- Modify: `tests/test_molecules.py`
- Create: `tests/test_regions.py`
- Create: `tests/test_memwatch.py`
- Modify: `pafbench/config.py`
- Create: `pafbench/grid.py`
- Modify: `pafbench/flags.py`
- Modify: `pafbench/labels.py`
- Modify: `pafbench/molecules.py`
- Create: `pafbench/regions.py`
- Create: `pafbench/memwatch.py`

**Interfaces:**
- Consumes: milestone 1 `flags.Setting`, `flags.setting_tag`, `molecules.consolidate_sites`, `labels`, `config`.
- Produces:
  - `pafbench.grid.grid()`
  - `pafbench.grid.published()`
  - `pafbench.grid.all_settings()`
  - `pafbench.flags.parse_setting(text)`
  - `pafbench.molecules.consolidate_sites(pa, max_spread=10)`
  - `pafbench.molecules.select_members(members, cap=MEMBER_CAP)`
  - `pafbench.labels.a_band(a20, bands=A_BANDS)`
  - `pafbench.labels.label_molecules(mol, cutoffs=LABEL_CUTOFFS)`
  - `pafbench.regions.load_region_segments(gtf)`
  - `pafbench.regions.regions_at(segments, ref, strd, positions)`
  - `pafbench.regions.region_at(segments, ref, strd, pos)`
  - `pafbench.memwatch.tree_rss_bytes(pid)`
  - `pafbench.memwatch.run(cmd, interval=2.0)`
- Produces (data): config constants SIRV_DIR, SIRV_FASTA, SIRV_GTF, ERCC_CONC_URL, LABEL_CUTOFFS, A_BANDS, MEMBER_CAP, SENS_CAPS, NUCLEI, CELLS; grid constants COVLENS, MINLENS, MISMS, PUBLISHED; `molecules.MOL_COLUMNS` ends with `mapq_min`; `regions.REGIONS` (priority order) and `regions.REGION_NAMES` (adds "antisense"); `python -m pafbench.memwatch OUT.json -- CMD ...` writes cmd, exit_status, elapsed_s, peak_tree_rss_gib, interval_s.

Everything later builds on these pieces. The region annotator works per strand: a site's region is read at its last cDNA base on the site's own strand, and a position with no annotation on that strand but inside an opposite-strand transcript is "antisense", so antisense molecules are not mixed into "intergenic". `memwatch` exists because GNU time reports only the largest single process, while polyAfilter and the stage scripts fork workers; milestone 1's note asks for the whole tree.

- [ ] **Step 1: Write the failing tests in `tests/test_grid.py`**

```python
import pytest

from pafbench.flags import Setting, parse_setting, setting_tag
from pafbench.grid import all_settings, grid, published


def test_grid_sizes_and_order():
    g = grid()
    assert len(g) == 392 and len(set(g)) == 392
    assert g[0] == Setting(150, None, 0, False) and g[1] == Setting(150, 3, 0, False)
    assert sum(s.min_len is None for s in g) == 14
    assert {s.mism for s in g if s.min_len is None} == {0}


def test_all_settings_adds_missing_published_once():
    a = all_settings()
    assert len(a) == 396 and len(set(a)) == 396
    assert a[:392] == grid()
    assert set(published()) <= set(a)
    assert [setting_tag(s) for s in a[392:]] == ["c275_l9_m1_e", "c275_l9_m1_i", "c180_l11_m2_e", "c180_l11_m2_i"]


def test_parse_setting_refuses_zero():
    assert parse_setting("300,none,0,i") == Setting(300, None, 0, True)
    assert parse_setting("375,3,2,e") == Setting(375, 3, 2, False)
    with pytest.raises(ValueError):
        parse_setting("300,0,0,i")
```

- [ ] **Step 2: Write the failing tests: patch `tests/test_labels.py`** (apply from the repository root)

```bash
git apply <<'PATCH'
diff --git a/tests/test_labels.py b/tests/test_labels.py
index e4ea8e5..7786682 100644
--- a/tests/test_labels.py
+++ b/tests/test_labels.py
@@ -1,6 +1,6 @@
 import pandas as pd
 
-from pafbench.labels import choose_cutoffs, compare_shares, hexamer_curve, templated_share
+from pafbench.labels import a_band, choose_cutoffs, compare_shares, hexamer_curve, label_molecules, templated_share
 
 RATES = [0.5] * 5 + [0.45, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]
 
@@ -28,3 +28,15 @@ def test_hexamer_curve_and_shares():
     assert t.loc["E", "share"] == 1 / 3 and t.loc["N", "share"] == 1.0
     cmp = compare_shares(t, t.assign(share=t.share + 0.1))
     assert cmp.flag.all()
+
+
+def test_a_band_and_label_molecules():
+    assert [a_band(x) for x in (0, 4, 5, 7, 8, 9, 10, 13, 14, 20, 21)] == \
+        ["0-4", "0-4", "5-7", "5-7", "8-9", "8-9", "10-13", "10-13", "14-20", "14-20", ""]
+    mol = pd.DataFrame({"a20": [3, 4, 5, 7, 8, 20]})
+    out = label_molecules(mol)
+    assert out.label.tolist() == ["nontemplated", "nontemplated", "ambiguous", "ambiguous", "templated", "templated"]
+    assert out.a_band.tolist() == ["0-4", "0-4", "5-7", "5-7", "8-9", "14-20"]
+    assert label_molecules(mol, {"templated_min": 14, "nontemplated_max": 8}).label.tolist() == \
+        ["nontemplated"] * 5 + ["templated"]
+    assert "label" not in mol
PATCH
```

- [ ] **Step 3: Write the failing tests: patch `tests/test_molecules.py`** (apply from the repository root)

```bash
git apply <<'PATCH'
diff --git a/tests/test_molecules.py b/tests/test_molecules.py
index 2d0d5c3..b8659c0 100644
--- a/tests/test_molecules.py
+++ b/tests/test_molecules.py
@@ -1,26 +1,26 @@
 import pandas as pd
 import pysam
 
-from pafbench.molecules import beyond_cap, choose_cap, consolidate_sites, gene_match, member_density, member_distances
+from pafbench.molecules import beyond_cap, choose_cap, consolidate_sites, gene_match, member_density, member_distances, select_members
 
 
 def _pa(rows):
-    cols = ["cb", "ub", "strand", "ref", "site", "a20", "down20", "up40", "hexamer", "genes", "re"]
+    cols = ["cb", "ub", "strand", "ref", "site", "a20", "down20", "up40", "hexamer", "genes", "re", "mapq"]
     return pd.DataFrame(rows, columns=cols)
 
 
 def test_consolidate_drops_spread_and_missing_umi():
     pa = _pa([
-        ["c1", "u1", "+", "chr1", 100, 15, "A" * 20, "C" * 40, False, "g", "E"],
-        ["c1", "u1", "+", "chr1", 104, 15, "A" * 20, "C" * 40, False, "g", "E"],
-        ["c1", "u1", "+", "chr1", 109, 15, "A" * 20, "C" * 40, False, "g", "E"],
-        ["c2", "u2", "+", "chr1", 100, 3, "C" * 20, "C" * 40, True, "g", "E"],
-        ["c2", "u2", "+", "chr1", 200, 3, "C" * 20, "C" * 40, True, "g", "E"],
-        ["c3", None, "+", "chr1", 300, 3, "C" * 20, "C" * 40, True, "g", "E"],
+        ["c1", "u1", "+", "chr1", 100, 15, "A" * 20, "C" * 40, False, "g", "E", 255],
+        ["c1", "u1", "+", "chr1", 104, 15, "A" * 20, "C" * 40, False, "g", "E", 3],
+        ["c1", "u1", "+", "chr1", 109, 15, "A" * 20, "C" * 40, False, "g", "E", 255],
+        ["c2", "u2", "+", "chr1", 100, 3, "C" * 20, "C" * 40, True, "g", "E", 255],
+        ["c2", "u2", "+", "chr1", 200, 3, "C" * 20, "C" * 40, True, "g", "E", 255],
+        ["c3", None, "+", "chr1", 300, 3, "C" * 20, "C" * 40, True, "g", "E", 255],
     ])
     mol = consolidate_sites(pa)
     assert len(mol) == 1
-    assert mol.iloc[0].site == 104 and mol.iloc[0].n_pa == 3
+    assert mol.iloc[0].site == 104 and mol.iloc[0].n_pa == 3 and mol.iloc[0].mapq_min == 3
 
 
 def test_member_distances():
@@ -61,3 +61,10 @@ def test_beyond_cap():
     assert r["population"] == 3
     assert r["500"] == {"members": 2, "molecules": 2} and r["1000"] == {"members": 2, "molecules": 2}
     assert r["2000"] == {"members": 1, "molecules": 1}
+
+
+def test_select_members():
+    m = pd.DataFrame({"gene_match": [True, True, False, True, True],
+                      "dist3_exon": [0, -26, 0, -25, 10], "dist5_exon": [1000, 50, 50, 50, 1001]})
+    assert select_members(m).index.tolist() == [0, 3]
+    assert select_members(m, cap=2000).index.tolist() == [0, 3, 4]
PATCH
```

- [ ] **Step 4: Write the failing tests in `tests/test_regions.py`**

```python
from pafbench.regions import REGION_NAMES, REGIONS, load_region_segments, region_at, regions_at

ROWS = [
    # coding transcript on +: exons 101-200, 301-400, 501-600 (1-based); CDS 151-350
    ("transcript", 101, 600, "+", "tc"), ("exon", 101, 200, "+", "tc"), ("exon", 301, 400, "+", "tc"),
    ("exon", 501, 600, "+", "tc"), ("CDS", 151, 200, "+", "tc"), ("CDS", 301, 350, "+", "tc"),
    # non-coding transcript on + whose exon covers part of tc's intron and of its 3' UTR exon
    ("transcript", 251, 560, "+", "tn"), ("exon", 251, 280, "+", "tn"), ("exon", 541, 560, "+", "tn"),
    # coding transcript on -: exons 1001-1100, 1201-1300; CDS 1051-1250
    ("transcript", 1001, 1300, "-", "tm"), ("exon", 1001, 1100, "-", "tm"), ("exon", 1201, 1300, "-", "tm"),
    ("CDS", 1051, 1100, "-", "tm"), ("CDS", 1201, 1250, "-", "tm"),
]


def _gtf(tmp_path):
    p = tmp_path / "r.gtf"
    with open(p, "w") as fh:
        fh.write("#comment\n")
        for ft, s, e, strand, tid in ROWS:
            fh.write(f"chrZ\tt\t{ft}\t{s}\t{e}\t.\t{strand}\t.\tgene_id \"g_{tid}\"; transcript_id \"{tid}\";\n")
    return p


def test_plus_strand_regions(tmp_path):
    seg = load_region_segments(_gtf(tmp_path))
    at = lambda pos: region_at(seg, "chrZ", True, pos)
    assert at(99) == "intergenic" and at(100) == "5UTR" and at(149) == "5UTR" and at(150) == "CDS"
    assert at(199) == "CDS" and at(200) == "intron" and at(260) == "ncexon" and at(300) == "CDS"
    assert at(349) == "CDS" and at(350) == "3UTR" and at(450) == "intron" and at(545) == "3UTR"
    assert at(599) == "3UTR" and at(600) == "intergenic"


def test_minus_strand_mirrors_utrs_and_strands_are_separate(tmp_path):
    seg = load_region_segments(_gtf(tmp_path))
    got = regions_at(seg, "chrZ", False, [1000, 1049, 1050, 1150, 1249, 1250, 1299, 1300])
    assert list(got) == ["3UTR", "3UTR", "CDS", "intron", "CDS", "5UTR", "5UTR", "intergenic"]
    assert region_at(seg, "chrZ", False, 150) == "antisense" and region_at(seg, "chrZ", False, 99) == "intergenic"
    assert region_at(seg, "chrZ", True, 1150) == "antisense" and region_at(seg, "chrZ", True, 1300) == "intergenic"
    assert region_at(seg, "chrQ", True, 150) == "intergenic"
    assert list(regions_at(seg, "chrZ", True, [])) == []


def test_region_names():
    assert REGIONS == ("intergenic", "intron", "ncexon", "5UTR", "CDS", "3UTR")
    assert REGION_NAMES == REGIONS + ("antisense",)
```

- [ ] **Step 5: Write the failing tests in `tests/test_memwatch.py`**

```python
import json
import subprocess
import sys

from pafbench.memwatch import run


def test_peak_counts_children_and_exit_status():
    child = "import time; x = bytearray(300 * 1024 * 1024); time.sleep(2)"
    res = run(["bash", "-c", f"{sys.executable} -c '{child}' & wait; exit 3"], interval=0.2)
    assert res["exit_status"] == 3
    assert res["peak_tree_rss_gib"] > 0.25
    assert res["elapsed_s"] >= 2


def test_cli_writes_json(tmp_path):
    out = tmp_path / "m.json"
    r = subprocess.run([sys.executable, "-m", "pafbench.memwatch", str(out), "--", "true"])
    assert r.returncode == 0 and json.loads(out.read_text())["exit_status"] == 0
```

- [ ] **Step 6: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_grid.py tests/test_labels.py tests/test_molecules.py tests/test_regions.py tests/test_memwatch.py`
Expected: FAIL; the first error is `ModuleNotFoundError: No module named 'pafbench.grid'`.

- [ ] **Step 7: Modify `pafbench/config.py`** (apply from the repository root)

```bash
git apply <<'PATCH'
diff --git a/pafbench/config.py b/pafbench/config.py
index 5bac078..e1e18ab 100644
--- a/pafbench/config.py
+++ b/pafbench/config.py
@@ -6,6 +6,7 @@ POLYAFILTER_COMMIT = "5ce0098"
 REF_DIR = Path("/home/simone.picelli/Documents/Genome_ref/refdata-gex-GRCh38-2024-A")
 FASTA = REF_DIR / "fasta" / "genome.fa"
 GTF = REF_DIR / "genes" / "genes.gtf"
+POLYASITE = Path("/data/scNoiseMeter/atlas.clusters.3.0.GRCh38.GENCODE_42.bed.gz")
 RUNS = Path("/data/element_runs")
 OUT = RUNS / "polyafilter_bench"
 LIBRARIES = {
@@ -17,3 +18,13 @@ LIBRARIES = {
 BD67_BAM = RUNS / "BD67" / "BD67_2x_RT.fltnc.aligned.bam"
 BD70_BAM = RUNS / "BD70" / "BD70.fltnc.aligned.bam"
 LONG_READS = {"BD67": BD67_BAM, "BD70": BD70_BAM}
+SIRV_DIR = OUT / "ref" / "sirv"
+SIRV_FASTA = SIRV_DIR / "SIRV_ERCC_longSIRV_multi-fasta_20250402.fasta"
+SIRV_GTF = SIRV_DIR / "SIRV_ERCC_longSIRV_multi-fasta-annotation_20250402.gtf"
+ERCC_CONC_URL = "https://assets.thermofisher.com/TFS-Assets/LSG/manuals/cms_095046.txt"
+LABEL_CUTOFFS = {"templated_min": 8, "nontemplated_max": 4}
+A_BANDS = (("0-4", 0, 4), ("5-7", 5, 7), ("8-9", 8, 9), ("10-13", 10, 13), ("14-20", 14, 20))
+MEMBER_CAP = 1000
+SENS_CAPS = (500, 1000, 2000)
+NUCLEI = ("1417", "1419")
+CELLS = ("1179",)
PATCH
```

- [ ] **Step 8: Create `pafbench/grid.py`**

```python
from .flags import Setting

COVLENS = (150, 200, 250, 300, 375, 450, 600)
MINLENS = (3, 5, 7, 9, 10, 11, 13, 15, 20)
MISMS = (0, 1, 2)
PUBLISHED = ((300, 10, 1), (275, 9, 1), (180, 11, 2), (375, 3, 2))


def grid():
    """392 settings: per mode and COVLEN, the remove-all control (MINSNRLEN None) then every MINSNRLEN x MISM."""
    out = []
    for introns in (False, True):
        for c in COVLENS:
            out.append(Setting(c, None, 0, introns))
            out.extend(Setting(c, l, m, introns) for l in MINLENS for m in MISMS)
    return out


def published():
    """README example and the paper's Fig. 4A optima, each in both modes."""
    return [Setting(c, l, m, introns) for c, l, m in PUBLISHED for introns in (False, True)]


def all_settings():
    """grid() followed by the published settings it lacks (396)."""
    out = grid()
    seen = set(out)
    out.extend(s for s in published() if s not in seen)
    return out
```

- [ ] **Step 9: Modify `pafbench/flags.py`** (apply from the repository root)

```bash
git apply <<'PATCH'
diff --git a/pafbench/flags.py b/pafbench/flags.py
index 358a998..4324d27 100644
--- a/pafbench/flags.py
+++ b/pafbench/flags.py
@@ -51,7 +51,10 @@ def setting_tag(s):
 
 
 def parse_setting(text):
+    """'COVLEN,MINSNRLEN|none,MISM,i|e'. 'none' is the remove-all-non-terminal control; 0 is refused because the CLI's 0 is not that control."""
     c, l, m, mode = text.split(",")
+    if l != "none" and int(l) == 0:
+        raise ValueError("MINSNRLEN 0 is not the remove-all control; write 'none'")
     return Setting(int(c), None if l == "none" else int(l), int(m), mode == "i")
PATCH
```

- [ ] **Step 10: Modify `pafbench/labels.py`** (apply from the repository root)

```bash
git apply <<'PATCH'
diff --git a/pafbench/labels.py b/pafbench/labels.py
index a520dec..ca7dcd4 100644
--- a/pafbench/labels.py
+++ b/pafbench/labels.py
@@ -1,6 +1,8 @@
 import numpy as np
 import pandas as pd
 
+from .config import A_BANDS, LABEL_CUTOFFS
+
 
 def hexamer_curve(mol):
     g = mol.groupby("a20").hexamer.agg(["size", "mean"])
@@ -43,3 +45,19 @@ def compare_shares(a, b, threshold=0.05):
     j["diff"] = j.share_b - j.share_a
     j["flag"] = j["diff"].abs() > threshold
     return j
+
+
+def a_band(a20, bands=A_BANDS):
+    """Reporting band of a downstream A count ('' outside every band)."""
+    for name, lo, hi in bands:
+        if lo <= a20 <= hi:
+            return name
+    return ""
+
+
+def label_molecules(mol, cutoffs=LABEL_CUTOFFS):
+    """Adds label (templated / nontemplated / ambiguous) and a_band from a20."""
+    a = mol.a20.to_numpy()
+    label = np.where(a >= cutoffs["templated_min"], "templated",
+                     np.where(a <= cutoffs["nontemplated_max"], "nontemplated", "ambiguous"))
+    return mol.assign(label=label, a_band=[a_band(int(x)) for x in a])
PATCH
```

- [ ] **Step 11: Modify `pafbench/molecules.py`** (apply from the repository root)

```bash
git apply <<'PATCH'
diff --git a/pafbench/molecules.py b/pafbench/molecules.py
index b63f0ba..5fcc6c0 100644
--- a/pafbench/molecules.py
+++ b/pafbench/molecules.py
@@ -1,12 +1,14 @@
 import numpy as np
 import pandas as pd
 
+from .config import MEMBER_CAP
+
 EDGES = [0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 50000, 100000, 1000000]
-MOL_COLUMNS = ["mol", "cb", "ub", "strand", "ref", "site", "n_pa", "spread", "a20", "down20", "up40", "hexamer", "genes", "re"]
+MOL_COLUMNS = ["mol", "cb", "ub", "strand", "ref", "site", "n_pa", "spread", "a20", "down20", "up40", "hexamer", "genes", "re", "mapq_min"]
 
 
 def consolidate_sites(pa, max_spread=10):
-    """One row per molecule (cb, ub, strand, ref, own gene label); site = lower median of its pa sites; spread > max_spread dropped."""
+    """One row per molecule (cb, ub, strand, ref, own gene label); site = lower median of its pa sites; spread > max_spread dropped; mapq_min over its pa reads."""
     d = pa[pa.cb.notna() & pa.ub.notna()]
     rows = []
     for (cb, ub, strand, ref, genes), g in d.groupby(["cb", "ub", "strand", "ref", "genes"], sort=False):
@@ -16,7 +18,8 @@ def consolidate_sites(pa, max_spread=10):
             continue
         site = int(sites[(len(sites) - 1) // 2])
         r = g.iloc[int(np.argmin(np.abs(g.site.to_numpy() - site)))]
-        rows.append([len(rows), cb, ub, strand, ref, site, len(g), spread, int(r.a20), r.down20, r.up40, bool(r.hexamer), genes, r.re])
+        rows.append([len(rows), cb, ub, strand, ref, site, len(g), spread, int(r.a20), r.down20, r.up40, bool(r.hexamer), genes, r.re,
+                     int(g.mapq.min())])
     return pd.DataFrame(rows, columns=MOL_COLUMNS)
 
 
@@ -60,3 +63,8 @@ def beyond_cap(members, caps=(500, 1000, 2000), d5="dist5_exon", d3="dist3_exon"
         far = m[m[d5] > c]
         out[str(c)] = {"members": int(len(far)), "molecules": int(far.mol.nunique())}
     return out
+
+
+def select_members(members, cap=MEMBER_CAP):
+    """Members that enter the truth set: same gene label, not past the site by more than 25 nt, 5' start within cap (exon-aware)."""
+    return members[members.gene_match & (members.dist3_exon >= -25) & (members.dist5_exon <= cap)]
PATCH
```

- [ ] **Step 12: Create `pafbench/regions.py`**

```python
from collections import defaultdict

import numpy as np

# Index is the priority: where annotations overlap on a strand, the highest code wins. regions_at adds "antisense".
REGIONS = ("intergenic", "intron", "ncexon", "5UTR", "CDS", "3UTR")
REGION_NAMES = REGIONS + ("antisense",)


def _field(attrs, key):
    return attrs.split(f'{key} "', 1)[1].split('"', 1)[0]


def _paint(bounds, codes, pieces, code):
    if not pieces:
        return
    s, e = np.asarray(pieces, np.int64).T
    diff = np.zeros(bounds.size, np.int64)
    np.add.at(diff, np.searchsorted(bounds, s), 1)
    np.add.at(diff, np.searchsorted(bounds, e), -1)
    codes[np.cumsum(diff)[:-1] > 0] = code


def load_region_segments(gtf):
    """{(ref, strd): (bounds, codes)}: segment [bounds[k], bounds[k+1]) has region REGIONS[codes[k]] on that strand.

    Exon parts before the CDS in transcript sense are 5UTR, after it 3UTR (the stop codon falls in 3UTR, as in
    GENCODE's UTR features); exons of transcripts without CDS are ncexon; the rest of a transcript span is intron.
    """
    tx, exons, cds = {}, defaultdict(list), defaultdict(list)
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] not in ("transcript", "exon", "CDS"):
                continue
            tid = _field(f[8], "transcript_id")
            iv = (int(f[3]) - 1, int(f[4]))
            if f[2] == "transcript":
                tx[tid] = (f[0], f[6] == "+", iv)
            elif f[2] == "exon":
                exons[tid].append(iv)
            else:
                cds[tid].append(iv)
    pieces = defaultdict(lambda: defaultdict(list))
    for tid, (ref, strd, span) in tx.items():
        p = pieces[(ref, strd)]
        p[1].append(span)
        if tid not in cds:
            p[2].extend(exons[tid])
            continue
        cs = min(s for s, _ in cds[tid])
        ce = max(e for _, e in cds[tid])
        left, right = (3, 5) if strd else (5, 3)
        for s, e in exons[tid]:
            if s < cs:
                p[left].append((s, min(e, cs)))
            if e > ce:
                p[right].append((max(s, ce), e))
            if s < ce and e > cs:
                p[4].append((max(s, cs), min(e, ce)))
    out = {}
    for key, by_code in pieces.items():
        bounds = np.unique(np.concatenate([np.asarray(v, np.int64).ravel() for v in by_code.values()]))
        codes = np.zeros(bounds.size - 1, np.int8)
        for code in range(1, len(REGIONS)):
            _paint(bounds, codes, by_code.get(code, []), code)
        out[key] = (bounds, codes)
    return out


def _codes_at(segments, ref, strd, pos):
    c = np.zeros(pos.size, np.int8)
    entry = segments.get((ref, strd))
    if entry is None:
        return c
    bounds, codes = entry
    k = np.searchsorted(bounds, pos, side="right") - 1
    inside = (k >= 0) & (k < codes.size)
    c[inside] = codes[k[inside]]
    return c


def regions_at(segments, ref, strd, positions):
    """Region names at 0-based positions on one strand; unannotated positions inside an opposite-strand transcript are 'antisense'."""
    pos = np.asarray(positions, np.int64)
    names = np.asarray(REGIONS, object)[_codes_at(segments, ref, strd, pos)]
    names[(names == REGIONS[0]) & (_codes_at(segments, ref, not strd, pos) > 0)] = "antisense"
    return names


def region_at(segments, ref, strd, pos):
    return regions_at(segments, ref, strd, [pos])[0]
```

- [ ] **Step 13: Create `pafbench/memwatch.py`**

```python
"""Run a command and record its wall time and the peak RSS summed over its whole process tree.

usage: python -m pafbench.memwatch OUT.json -- CMD [ARG ...]
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _proc_table():
    """{pid: (ppid, rss_kib)} for every process readable in /proc."""
    out = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/status") as fh:
                ppid, rss = None, 0
                for line in fh:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                    elif line.startswith("VmRSS:"):
                        rss = int(line.split()[1])
        except (FileNotFoundError, ProcessLookupError):
            continue
        out[int(name)] = (ppid, rss)
    return out


def tree_rss_bytes(pid):
    table = _proc_table()
    kids = {}
    for p, (pp, _) in table.items():
        kids.setdefault(pp, []).append(p)
    total, stack = 0, [pid]
    while stack:
        p = stack.pop()
        total += table.get(p, (None, 0))[1]
        stack.extend(kids.get(p, []))
    return total * 1024


def run(cmd, interval=2.0):
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd)
    peak = 0
    while True:
        peak = max(peak, tree_rss_bytes(proc.pid))
        try:
            proc.wait(timeout=interval)
            break
        except subprocess.TimeoutExpired:
            pass
    return {"cmd": cmd, "exit_status": proc.returncode, "elapsed_s": time.monotonic() - t0,
            "peak_tree_rss_gib": peak / 1024 ** 3, "interval_s": interval}


def main(argv):
    if len(argv) < 3 or argv[1] != "--":
        raise SystemExit(__doc__)
    res = run(argv[2:])
    Path(argv[0]).write_text(json.dumps(res, indent=1))
    sys.exit(res["exit_status"])


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 14: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `62 passed, 2 warnings`.

- [ ] **Step 15: Check the region annotator against Cell Ranger on 1417 (read-only)**

Run:
```bash
.venv/bin/python - <<'EOF'
import numpy as np, pandas as pd
from pafbench.config import GTF, OUT
from pafbench.regions import load_region_segments, regions_at
seg = load_region_segments(GTF)
mol = pd.read_parquet(OUT / "1417" / "molecules.parquet")
last = np.where(mol.strand == "+", mol.site - 1, mol.site)
region = np.empty(len(mol), object)
for (ref, strand), g in mol.groupby(["ref", "strand"]):
    region[g.index] = regions_at(seg, ref, strand == "+", last[g.index])
print(pd.crosstab(region, mol.re.fillna("none")))
EOF
```
Expected (about 16 s to load the GTF), against Cell Ranger's strand-blind RE tag (E exonic, N intronic, I intergenic):
```
re              E      I      N
3UTR        31072      1     56
5UTR          833      5     24
CDS         13998      6    166
antisense    1452     64   8931
intergenic    319  14149      0
intron       2220     72  60163
ncexon      10866      4     73
```
"intergenic" agrees with I for 14,149 of 14,468 molecules, and every N molecule without a same-strand annotation is antisense.

- [ ] **Step 16: Commit**

```bash
git add tests/test_grid.py tests/test_labels.py tests/test_molecules.py tests/test_regions.py tests/test_memwatch.py pafbench/config.py pafbench/grid.py pafbench/flags.py pafbench/labels.py pafbench/molecules.py pafbench/regions.py pafbench/memwatch.py
git -c user.name="Simone Picelli" -c user.email="simone.picelli@iob.ch" commit -m "Milestone 2 foundation: constants, settings grid, MINSNRLEN 0 guard, mapq_min, member selection, labels and A bands, region annotator with antisense class, process-tree memory sampler"
```

---

### Task 1: Validation tooling and background original-tool runs (stages 19 and 20)

**Files:**
- Create: `tests/test_validation.py`
- Create: `pafbench/validation.py`
- Create: `scripts/19_validation_runs.sh`
- Create: `scripts/20_validation_compare.py`

**Interfaces:**
- Consumes: `scripts/02_run_original.sh LIB COVLEN MINSNRLEN MISM MODE` (milestone 1), `scripts/03_reimpl_flags.py LIB --setting ... --workers N`, `scripts/04_compare_removed.py LIB --setting ...` (writes OUT/LIB/exact_match.json, exit 1 on a difference), `flags.flag_reference`, `flags.removed_keys_by_ref`, `report.parse_gnu_time`, `vendor.polyafilter_modules`, Task 0 `memwatch`.
- Produces:
  - `pafbench.validation.parse_tag(tag)`
  - `pafbench.validation.setting_arg(s)`
  - `pafbench.validation.time_exit_status(tfile)`
  - `pafbench.validation.run_stats(prefix)`
  - `pafbench.validation.discover_settings(orig_dir)`
  - `pafbench.validation.subset_bam(bam, refs, out_bam)`
  - `pafbench.validation.create_trans(db, bam, trans, include_introns, prefix)`
  - `pafbench.validation.run_original(bam, trans_pkl, db, fasta, out_bam, setting, threads=4)`
  - `pafbench.validation.run_original_timed(bam, trans_pkl, db, fasta, out_bam, setting, prefix, threads=4)`
  - `pafbench.validation.compare_subset(bam, out_bam, fasta, transcripts, setting)`
  - `pafbench.validation.none_control(bam, refs, out_dir, db, fasta, gtf, covlen=300, threads=4)`
  - `pafbench.validation.subset_validation(bam, refs, out_dir, db, fasta, gtf, settings, threads=4)`
  - `pafbench.validation.main(argv)`
- Produces (data): `OUT/<lib>/original/<tag>.bam|.time.txt|.log|.mem.json` for each run; `OUT/validation/runs_<lib>.log`; stage 20 writes `OUT/validation/exact_match_m2.json` (per library per tag: identical, n_original, n_reimpl, only_original, only_reimpl, run cost; `none_control`; `1179_subset`; `all_identical`), `OUT/validation/none_control/`, `OUT/validation/subset_1179/`.

Spec 5.2 makes record-for-record identity with the original tool the condition for trusting any grid result. Milestone 1 covered three settings on 1417. This task adds the published settings in both modes on 1417 and 1419, MISM 0, COVLEN 600, the remove-all-non-terminal control, which only an in-process call can express (the CLI's MINSNRLEN 0 is not the control), and 1179 by chromosome (spec 5.1 and 11: the original holds its removal set in RAM, 46.0 GB for 83M records at 375/3/2 `-i`, so 574M records do not fit; chr21 and chr22 of 1179 run in-process at the 8 published settings). The original tool needs hours per setting, so the runs start now and stage 20 compares them in Task 8. Stage 20 is written here because its tests share the fixtures of stage 19.

- [ ] **Step 1: Write the failing tests in `tests/test_validation.py`**

```python
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

import pafbench.validation as validation
from pafbench.config import LIBRARIES, OUT
from pafbench.flags import Setting, parse_setting, record_key, removed_keys, setting_tag
from pafbench.grid import all_settings, published
from pafbench.transcripts import load_transcripts
from pafbench.validation import (
    compare_subset, create_trans, discover_settings, none_control, parse_tag, run_original, run_stats, setting_arg,
    subset_bam, subset_validation,
)
from tests.synth import run_original as serial_original

REPO = Path(__file__).resolve().parents[1]
OK_TIME = "\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03\n\tMaximum resident set size (kbytes): 2097152\n\tExit status: 0\n"
# GNU time on a command killed by a signal (OOM kill, SIGKILL): a "terminated by signal" line, yet "Exit status: 0"
KILLED_TIME = ("Command terminated by signal 9\n\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:10.00\n"
               "\tMaximum resident set size (kbytes): 1048576\n\tExit status: 0\n")
FAILED_TIME = ("Command exited with non-zero status 3\n\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:10.00\n"
               "\tMaximum resident set size (kbytes): 1048576\n\tExit status: 3\n")


def _keys(bam, ref=None):
    with pysam.AlignmentFile(str(bam)) as fh:
        return [record_key(a) for a in fh.fetch(until_eof=True) if ref is None or a.reference_name == ref]


# ---------- stage 19 ----------

def _repo19(tmp_path):
    """A repo layout around a copy of stage 19, so $REPO/.venv is the running venv and pafbench is this checkout."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "19_validation_runs.sh", repo / "scripts")
    (repo / ".venv").symlink_to(Path(sys.executable).parents[1])
    (repo / "pafbench").symlink_to(REPO / "pafbench")
    return repo


def _dryrun(repo, out):
    # the dry run prints the trash command, so a PAFBENCH_TRASH exported by the caller must not reach it
    env = {k: v for k, v in os.environ.items() if k != "PAFBENCH_TRASH"}
    r = subprocess.run(["bash", str(repo / "scripts" / "19_validation_runs.sh")], capture_output=True, text=True,
                       check=True, env=env | {"DRYRUN": "1", "PAFBENCH_OUT": str(out)})
    return r.stdout.splitlines()


def _filter_line(repo, out, lib, c, l, m, mode):
    return (f"{repo}/.venv/bin/python -m pafbench.memwatch {out}/{lib}/original/c{c}_l{l}_m{m}_{mode}.mem.json -- "
            f"{repo}/scripts/02_run_original.sh {lib} {c} {l} {m} {mode}")


def _trans_line(repo, out, mode):
    py, d = f"{repo}/.venv/bin/python", f"{out}/1419/original"
    flag = " -i" if mode == "i" else ""
    return (f"{py} -m pafbench.memwatch {d}/createTRANS_{mode}.mem.json -- /usr/bin/time -v -o {d}/createTRANS_{mode}.time.txt "
            f"{py} {repo}/vendor/polyAfilter/polyAfilter.py createTRANS{flag} {OUT}/ref/genes_2024A.db {LIBRARIES['1419']} "
            f"{d}/trans_{mode}.pkl > {d}/createTRANS_{mode}.log 2>&1 &")


CHAIN_1417 = [(275, 9, 1, "i"), (275, 9, 1, "e"), (180, 11, 2, "i"), (180, 11, 2, "e"), (300, 10, 0, "i"), (600, 10, 1, "i"),
              (375, 3, 2, "e")]
CHAIN_1419 = [(300, 10, 1, "i"), (375, 3, 2, "i"), (300, 10, 1, "e"), (275, 9, 1, "i"), (275, 9, 1, "e"), (180, 11, 2, "i"),
              (180, 11, 2, "e"), (375, 3, 2, "e")]


def test_stage19_dryrun_order(tmp_path):
    repo, out = _repo19(tmp_path), tmp_path / "out"
    out.mkdir()
    lines = _dryrun(repo, out)
    want = (["# chain 1417"] + [_filter_line(repo, out, "1417", *s) for s in CHAIN_1417]
            + ["# chain 1419", _trans_line(repo, out, "i"), _trans_line(repo, out, "e"), "wait"]
            + [_filter_line(repo, out, "1419", *s) for s in CHAIN_1419])
    assert lines == want
    # a dry run writes nothing: no log, no original/ directories
    assert list(out.iterdir()) == []


def test_stage19_skip_rule(tmp_path):
    repo, out = _repo19(tmp_path), tmp_path / "out"
    o17, o19 = out / "1417" / "original", out / "1419" / "original"
    o17.mkdir(parents=True)
    o19.mkdir(parents=True)
    # done: BAM plus a successful time file -> skipped
    (o17 / "c275_l9_m1_i.bam").touch()
    (o17 / "c275_l9_m1_i.time.txt").write_text(OK_TIME)
    # killed by a signal: BAM exists, time file says "Exit status: 0" after a "terminated by signal" line -> stale BAM
    # trashed (BAMfilter returns early on an existing BAM), then rerun
    (o17 / "c275_l9_m1_e.bam").touch()
    (o17 / "c275_l9_m1_e.time.txt").write_text(KILLED_TIME)
    # failed with a non-zero exit: same
    (o17 / "c180_l11_m2_e.bam").touch()
    (o17 / "c180_l11_m2_e.time.txt").write_text(FAILED_TIME)
    # successful time file but no BAM -> rerun
    (o17 / "c180_l11_m2_i.time.txt").write_text(OK_TIME)
    # 1419 createTRANS i done -> skipped; e still runs, and the pair is still followed by wait
    (o19 / "trans_i.pkl").touch()
    (o19 / "createTRANS_i.time.txt").write_text(OK_TIME)
    lines = _dryrun(repo, out)
    want = (["# chain 1417", "# skip 1417 c275_l9_m1_i", f"gio trash {o17}/c275_l9_m1_e.bam"]
            + [_filter_line(repo, out, "1417", *s) for s in CHAIN_1417[1:3]]
            + [f"gio trash {o17}/c180_l11_m2_e.bam"] + [_filter_line(repo, out, "1417", *s) for s in CHAIN_1417[3:]]
            + ["# chain 1419", "# skip 1419 createTRANS_i", _trans_line(repo, out, "e"), "wait"]
            + [_filter_line(repo, out, "1419", *s) for s in CHAIN_1419])
    assert lines == want
    assert (o17 / "c275_l9_m1_e.bam").exists() and (o17 / "c180_l11_m2_e.bam").exists()


FAKE_02 = """#!/usr/bin/env bash
# stand-in for 02_run_original.sh: writes the BAM and a GNU time file under PAFBENCH_OUT; FAIL=LIB/TAG exits 3
set -euo pipefail
TAG=c$2_l$3_m$4_$5
D=$PAFBENCH_OUT/$1/original
RC=0
[ "$1/$TAG" = "${FAIL:-}" ] && RC=3
[ -e "$D/$TAG.bam" ] || echo partial > "$D/$TAG.bam"   # like BAMfilter, leave an existing output alone
printf '\\tExit status: %s\\n' "$RC" > "$D/$TAG.time.txt"
exit "$RC"
"""

FAKE_POLYAFILTER = """import sys, time
time.sleep(1)
open(sys.argv[-1], "w").write(" ".join(sys.argv[1:]))
"""


def _log(path):
    """(event, name, exit) per line of a runs_<lib>.log after the timestamp."""
    out = []
    for line in path.read_text().splitlines():
        words = line.split()
        out.append((words[2], words[3].rstrip(":"), words[5] if words[2] == "end" else None))
    return out


def _repo19_fakes(tmp_path):
    """_repo19 with stand-ins for 02_run_original.sh and polyAfilter.py; env runs it for real with trash into tmp/trash."""
    repo, out = _repo19(tmp_path), tmp_path / "out"
    (repo / "scripts" / "02_run_original.sh").write_text(FAKE_02)
    (repo / "scripts" / "02_run_original.sh").chmod(0o755)
    (repo / "vendor" / "polyAfilter").mkdir(parents=True)
    (repo / "vendor" / "polyAfilter" / "polyAfilter.py").write_text(FAKE_POLYAFILTER)
    (tmp_path / "trash").mkdir()
    env = {**os.environ, "PAFBENCH_OUT": str(out), "PAFBENCH_TRASH": f"mv -t {tmp_path / 'trash'}"}
    env.pop("DRYRUN", None)
    return repo, out, env, ["bash", str(repo / "scripts" / "19_validation_runs.sh")]


def test_stage19_runs_resumes_and_reports_failure(tmp_path):
    repo, out, env, script = _repo19_fakes(tmp_path)
    env["FAIL"] = "1417/c180_l11_m2_e"
    assert subprocess.run(script, env=env).returncode == 1
    o17, o19 = out / "1417" / "original", out / "1419" / "original"
    tags17 = [f"c{c}_l{l}_m{m}_{mode}" for c, l, m, mode in CHAIN_1417]
    tags19 = [f"c{c}_l{l}_m{m}_{mode}" for c, l, m, mode in CHAIN_1419]
    # the failed 1417 step does not stop the chain; both chains ran every step, each under memwatch
    assert all((o17 / f"{t}.bam").exists() and (o17 / f"{t}.mem.json").exists() for t in tags17)
    assert all((o19 / f"{t}.bam").exists() for t in tags19)
    assert json.loads((o17 / "c180_l11_m2_e.mem.json").read_text())["exit_status"] == 3
    # createTRANS got 02_run_original.sh's arguments; its time file and log sit next to the TRANS file
    db_bam = f"{OUT}/ref/genes_2024A.db {LIBRARIES['1419']}"
    assert (o19 / "trans_i.pkl").read_text() == f"createTRANS -i {db_bam} {o19}/trans_i.pkl"
    assert (o19 / "trans_e.pkl").read_text() == f"createTRANS {db_bam} {o19}/trans_e.pkl"
    assert run_stats(o19 / "createTRANS_i")["exit_status"] == 0 and (o19 / "createTRANS_e.log").exists()
    log17 = _log(out / "validation" / "runs_1417.log")
    assert log17 == [e for t in tags17 for e in (("start", t, None), ("end", t, "3" if t == "c180_l11_m2_e" else "0"))]
    log19 = _log(out / "validation" / "runs_1419.log")
    # the createTRANS pair runs in parallel: both start before either ends, and the filter steps follow
    assert {e[0] for e in log19[:2]} == {"start"} and {e[0] for e in log19[2:4]} == {"end"}
    assert {e[1] for e in log19[:4]} == {"createTRANS_i", "createTRANS_e"}
    assert log19[4:] == [e for t in tags19 for e in (("start", t, None), ("end", t, "0"))]
    # resume: completed steps are skipped, the failed one is trashed and rerun
    env.pop("FAIL")
    (o17 / "c180_l11_m2_e.bam").write_text("stale")
    assert subprocess.run(script, env=env).returncode == 0
    assert (o17 / "c180_l11_m2_e.bam").read_text() == "partial\n"
    assert [p.read_text() for p in (tmp_path / "trash").iterdir()] == ["stale"]
    assert _log(out / "validation" / "runs_1417.log")[len(log17):] == (
        [("skip", t, None) for t in tags17[:3]] + [("start", "c180_l11_m2_e", None), ("end", "c180_l11_m2_e", "0")]
        + [("skip", t, None) for t in tags17[4:]])
    assert {e[0] for e in _log(out / "validation" / "runs_1419.log")[len(log19):]} == {"skip"}


def test_stage19_refuses_a_second_concurrent_run(tmp_path):
    repo, out, env, script = _repo19_fakes(tmp_path)
    (out / "validation").mkdir(parents=True)
    with open(out / "validation" / "runs.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)   # a running instance holds the lock
        r = subprocess.run(script, env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "runs.lock" in r.stderr
    # nothing ran: no log, no output, no memwatch file, nothing trashed
    assert sorted(p.name for p in out.rglob("*") if p.is_file()) == ["runs.lock"]
    assert list((tmp_path / "trash").iterdir()) == []
    # once the lock is free the same command runs every step
    assert subprocess.run(script, env=env).returncode == 0
    assert len(list((out / "1417" / "original").glob("*.bam"))) == len(CHAIN_1417)


# ---------- tags and discovery ----------

def test_parse_tag_and_setting_arg_invert_all_settings():
    for s in all_settings():
        assert parse_tag(setting_tag(s)) == s
        assert parse_setting(setting_arg(s)) == s
    assert parse_tag("c300_lnone_m0_i") == Setting(300, None, 0, True)
    assert setting_arg(Setting(375, 3, 2, False)) == "375,3,2,e"
    for bad in ("c300_l0_m0_i", "c300_l10_m1_x", "createTRANS_i", "c300_l10_m1_i.bam_True_chr1"):
        with pytest.raises(ValueError):
            parse_tag(bad)


def test_discover_settings(tmp_path):
    for tag, txt in (("c300_l10_m1_i", OK_TIME), ("c375_l3_m2_i", KILLED_TIME), ("c300_lnone_m0_e", OK_TIME),
                     ("c180_l11_m2_i", FAILED_TIME)):
        (tmp_path / f"{tag}.bam").touch()
        (tmp_path / f"{tag}.time.txt").write_text(txt)
    (tmp_path / "c180_l11_m2_e.bam").touch()                  # still running: no time file yet
    (tmp_path / "c300_l10_m1_i.bam_True_chr1.bam").touch()    # BAMfilter -p temp file
    (tmp_path / "trans_i.pkl").touch()
    (tmp_path / "createTRANS_i.time.txt").write_text(OK_TIME)
    (tmp_path / "c275_l9_m1_e.time.txt").write_text(OK_TIME)  # time file without BAM
    # sorted by file name: "c300_l1..." < "c300_ln..."
    assert discover_settings(tmp_path) == [Setting(300, 10, 1, True), Setting(300, None, 0, False)]
    assert discover_settings(tmp_path / "missing") == []


def test_run_stats(tmp_path):
    (tmp_path / "x.time.txt").write_text(OK_TIME)
    # 1:02:03 = 3723 s; 2097152 kB = 2 GiB
    assert run_stats(tmp_path / "x") == {"elapsed_s": 3723.0, "max_rss_gib": 2.0, "exit_status": 0, "peak_tree_rss_gib": None}
    (tmp_path / "x.mem.json").write_text(json.dumps({"exit_status": 0, "elapsed_s": 3724.5, "peak_tree_rss_gib": 3.5}))
    assert run_stats(tmp_path / "x")["peak_tree_rss_gib"] == 3.5
    # killed by signal 9: GNU time writes "Exit status: 0", the run's status is 128 + 9
    (tmp_path / "k.time.txt").write_text(KILLED_TIME)
    assert run_stats(tmp_path / "k") == {"elapsed_s": 10.0, "max_rss_gib": 1.0, "exit_status": 137, "peak_tree_rss_gib": None}
    (tmp_path / "f.time.txt").write_text(FAILED_TIME)
    assert run_stats(tmp_path / "f")["exit_status"] == 3


def test_killed_fixture_is_what_gnu_time_writes(tmp_path):
    # the premise of KILLED_TIME, checked against this machine's /usr/bin/time: a SIGKILLed command
    r = subprocess.run(["/usr/bin/time", "-v", "-o", str(tmp_path / "k.time.txt"), sys.executable, "-c",
                        "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"])
    text = (tmp_path / "k.time.txt").read_text()
    assert r.returncode == 137
    assert text.startswith("Command terminated by signal 9\n") and text.endswith("\tExit status: 0\n")
    assert run_stats(tmp_path / "k")["exit_status"] == 137
    (tmp_path / "c300_l10_m1_i.bam").touch()
    (tmp_path / "c300_l10_m1_i.time.txt").write_text(text)
    assert discover_settings(tmp_path) == []


# ---------- subset and the remove-all control ----------

def test_subset_bam(synth, tmp_path):
    sub = tmp_path / "b.bam"
    subset_bam(synth.bam, ["chrB"], sub)
    assert _keys(sub) == _keys(synth.bam, "chrB")
    with pysam.AlignmentFile(str(sub)) as fh:
        assert fh.references == ("chrA", "chrB")
        assert sum(1 for _ in fh.fetch("chrA")) == 0
        assert sum(1 for _ in fh.fetch("chrB")) == len(_keys(synth.bam, "chrB"))
    both = tmp_path / "ba.bam"
    subset_bam(synth.bam, ["chrB", "chrA"], both)   # reference order as given is ignored: output stays in file order
    assert _keys(both) == _keys(synth.bam)


# removed-record counts of the original's serial path on the synthetic BAM (tests.synth.run_original), pinned so a
# mode or MISM that is not passed through changes the count: 300/10/1 removes 79 in -i mode and 66 in exon mode
SERIAL_COUNTS = {"c300_lnone_m0_i": 123, "c150_lnone_m0_e": 108, "c300_l10_m1_i": 79, "c300_l10_m1_e": 66}


@pytest.mark.parametrize("st", [Setting(300, None, 0, True), Setting(150, None, 0, False), Setting(300, 10, 1, True),
                                Setting(300, 10, 1, False)], ids=setting_tag)
def test_run_original_matches_reimpl_and_serial_original(synth, tmp_path, st):
    trans = tmp_path / "trans.pkl"
    create_trans(synth.db, synth.bam, trans, st.introns, tmp_path / "createTRANS")
    assert run_stats(tmp_path / "createTRANS")["exit_status"] == 0
    out = tmp_path / "orig.bam"
    run_original(synth.bam, trans, synth.db, synth.fasta, out, st, threads=2)
    res = compare_subset(synth.bam, out, synth.fasta, load_transcripts(synth.gtf), st)
    # independent reference: the original's serial path, fresh tROC, TRANS built in-process
    want = serial_original(synth, st.covlen, st.min_len, st.mism, st.introns)
    assert len(want) == SERIAL_COUNTS[setting_tag(st)]
    assert removed_keys(synth.bam, out) == want
    assert res["setting"] == setting_tag(st) and res["identical"] is True
    assert res["n_original"] == res["n_reimpl"] == len(want)
    assert res["only_original"] == res["only_reimpl"] == 0 and res["examples"] == []
    # the -p temp files next to out_bam were merged and deleted by the tool
    assert sorted(p.name for p in tmp_path.glob("orig.bam*")) == ["orig.bam"]


def test_compare_subset_reports_one_sided_differences(synth):
    # out_bam = the input itself: the "original" removed nothing, so every reimpl removal is only_reimpl
    for st in (Setting(300, None, 0, True), Setting(300, 10, 1, False)):
        want = serial_original(synth, st.covlen, st.min_len, st.mism, st.introns)
        res = compare_subset(synth.bam, synth.bam, synth.fasta, load_transcripts(synth.gtf), st)
        assert res["setting"] == setting_tag(st)
        assert res["identical"] is False and res["n_original"] == 0 and res["only_original"] == 0
        assert res["n_reimpl"] == res["only_reimpl"] == len(want) > 20
        assert len(res["examples"]) == 20 and all(e[0] == "only_reimpl" and tuple(e[1:]) in want for e in res["examples"])


def test_none_control_on_a_reference_subset(synth, tmp_path):
    nc = tmp_path / "nc"
    res = none_control(synth.bam, ["chrA"], nc, synth.db, synth.fasta, synth.gtf, threads=2)
    # the TRANS of a chrA-only BAM has the same chrA transcripts, so the result is the full run restricted to chrA
    want = {k for k in serial_original(synth, 300, None, 0, True) if k[2] == "chrA"}
    assert res["refs"] == ["chrA"] and res["setting"] == "c300_lnone_m0_i"
    assert res["n_records"] == len(_keys(synth.bam, "chrA"))
    assert res["identical"] is True and res["n_original"] == res["n_reimpl"] == len(want) > 0
    assert res["createTRANS"]["exit_status"] == 0 and res["createTRANS"]["peak_tree_rss_gib"] > 0
    assert set(res["seconds"]) == {"subset", "createTRANS", "original", "compare"}
    assert sorted(p.name for p in nc.iterdir()) == [
        "chrA.bam", "chrA.bam.bai", "chrA.c300_lnone_m0_i.bam", "chrA.createTRANS_i.log", "chrA.createTRANS_i.mem.json",
        "chrA.createTRANS_i.time.txt", "chrA.trans_i.pkl",
    ]


# ---------- original tool on a reference subset at given settings (1179 block of stage 20) ----------

def _recorded(monkeypatch):
    """Wrap create_trans and run_original_timed so each call is logged as ("trans", mode) or ("run", tag)."""
    calls = []
    real_trans, real_run = validation.create_trans, validation.run_original_timed

    def trans(db, bam, out, introns, prefix):
        calls.append(("trans", "i" if introns else "e"))
        real_trans(db, bam, out, introns, prefix)

    def run(bam, trans_pkl, db, fasta, out_bam, setting, prefix, threads=4):
        calls.append(("run", setting_tag(setting)))
        real_run(bam, trans_pkl, db, fasta, out_bam, setting, prefix, threads)

    monkeypatch.setattr(validation, "create_trans", trans)
    monkeypatch.setattr(validation, "run_original_timed", run)
    return calls


def _without(res, *keys):
    """res with the given keys dropped from every setting's entry."""
    return {**res, "settings": {t: {k: v for k, v in r.items() if k not in keys} for t, r in res["settings"].items()}}


COST = ("original_elapsed_s", "original_max_rss_gib", "original_peak_tree_rss_gib")


def test_subset_validation_runs_compares_and_resumes(synth, tmp_path, monkeypatch):
    trash = tmp_path / "trash"
    trash.mkdir()
    monkeypatch.setattr(validation, "TRASH", ("mv", "-t", str(trash)))
    calls = _recorded(monkeypatch)
    st_i, st_e = Setting(300, 10, 1, True), Setting(300, 10, 1, False)
    d = tmp_path / "sub"
    res = subset_validation(synth.bam, ["chrA"], d, synth.db, synth.fasta, synth.gtf, [st_i, st_e], threads=2)
    # one createTRANS per mode on the subset, then one original run per setting
    assert sorted(calls[:2]) == [("trans", "e"), ("trans", "i")]
    assert calls[2:] == [("run", "c300_l10_m1_i"), ("run", "c300_l10_m1_e")]
    assert res["refs"] == ["chrA"] and res["n_records"] == len(_keys(synth.bam, "chrA"))
    assert set(res["createTRANS"]) == {"i", "e"} and all(v["exit_status"] == 0 for v in res["createTRANS"].values())
    assert list(res["settings"]) == ["c300_l10_m1_i", "c300_l10_m1_e"]
    for st in (st_i, st_e):
        r = res["settings"][setting_tag(st)]
        # every synthetic removal is on chrA, so the chrA subset removes what the full serial run removes (79 i, 66 e)
        want = serial_original(synth, st.covlen, st.min_len, st.mism, st.introns)
        assert r["identical"] is True and r["n_original"] == r["n_reimpl"] == len(want) == SERIAL_COUNTS[setting_tag(st)]
        assert r["only_original"] == r["only_reimpl"] == 0 and r["examples"] == []
        # the filtered BAM is the original's output for this setting
        assert removed_keys(d / "chrA.bam", d / f"chrA.{setting_tag(st)}.bam") == want
        # run cost of the measured child process: GNU time and memwatch both written, exit 0
        cost = run_stats(d / f"chrA.{setting_tag(st)}")
        assert cost["exit_status"] == 0
        assert (r["original_elapsed_s"], r["original_max_rss_gib"], r["original_peak_tree_rss_gib"]) == (
            cost["elapsed_s"], cost["max_rss_gib"], cost["peak_tree_rss_gib"])
        assert r["original_max_rss_gib"] > 0 and r["original_peak_tree_rss_gib"] > 0 and r["compare_s"] >= 0
        # the child runs this module on the subset with the TRANS file of its own mode (the synthetic TRANS files of the
        # two modes give the same removals, so only the command line shows a swap)
        m = "i" if st.introns else "e"
        assert json.loads((d / f"chrA.{setting_tag(st)}.mem.json").read_text())["cmd"][-9:] == [
            "-m", "pafbench.validation", str(d / "chrA.bam"), str(d / f"chrA.trans_{m}.pkl"), str(synth.db), str(synth.fasta),
            str(d / f"chrA.{setting_tag(st)}.bam"), setting_tag(st), "2"]
        # and THREADS reaches the tool: its log line names 2 pool processes (run_original_timed's default is 4)
        assert ("Identifying alignments 300 bp upstream of non-terminal SNR10+ with up to 1 mismatches "
                f"({'including' if st.introns else 'excluding'} introns) to be removed across 2 parallel processes") in (
            d / f"chrA.{setting_tag(st)}.log").read_text()
        assert json.loads((d / f"chrA.{setting_tag(st)}.json").read_text()) == r
    assert sorted(p.name for p in d.iterdir()) == sorted(
        ["chrA.bam", "chrA.bam.bai"]
        + [f"chrA.trans_{m}.pkl" for m in "ie"] + [f"chrA.createTRANS_{m}.{x}" for m in "ie" for x in ("log", "mem.json", "time.txt")]
        + [f"chrA.{t}.{x}" for t in ("c300_l10_m1_i", "c300_l10_m1_e") for x in ("bam", "json", "log", "mem.json", "time.txt")])

    def again():
        calls.clear()
        return subset_validation(synth.bam, ["chrA"], d, synth.db, synth.fasta, synth.gtf, [st_i, st_e], threads=2)

    # everything done: nothing reruns, the result files are read back
    assert again() == res and calls == []
    # result file gone but the run finished (BAM plus a successful time file): compared again, not rerun
    os.replace(d / "chrA.c300_l10_m1_i.json", trash / "i.json")
    assert _without(again(), "compare_s") == _without(res, "compare_s") and calls == []
    # killed run: a partial BAM next to a time file with a signal line goes to the trash and the setting reruns
    os.replace(d / "chrA.c300_l10_m1_e.json", trash / "e.json")
    (d / "chrA.c300_l10_m1_e.bam").write_text("partial")
    (d / "chrA.c300_l10_m1_e.time.txt").write_text(KILLED_TIME)
    assert _without(again(), "compare_s", *COST) == _without(res, "compare_s", *COST) and calls == [("run", "c300_l10_m1_e")]
    assert (trash / "chrA.c300_l10_m1_e.bam").read_text() == "partial"
    # stopped before the child wrote anything (GNU time truncates its file at start): a partial BAM next to an empty time
    # file goes to the trash and the setting reruns
    os.replace(d / "chrA.c300_l10_m1_i.json", trash / "i2.json")
    (d / "chrA.c300_l10_m1_i.bam").write_text("partial")
    (d / "chrA.c300_l10_m1_i.time.txt").write_text("")
    assert _without(again(), "compare_s", *COST) == _without(res, "compare_s", *COST) and calls == [("run", "c300_l10_m1_i")]
    assert (trash / "chrA.c300_l10_m1_i.bam").read_text() == "partial"
    # every setting has its result file: a failed createTRANS e and a missing createTRANS i time file trigger no rebuild,
    # and the cost entries show the failure and the gap
    (d / "chrA.createTRANS_e.time.txt").write_text(FAILED_TIME)
    os.replace(d / "chrA.createTRANS_i.time.txt", trash / "createTRANS_i.time.txt")
    got = again()
    assert calls == [] and (d / "chrA.trans_e.pkl").exists() and (d / "chrA.trans_i.pkl").exists()
    assert got["createTRANS"]["e"]["exit_status"] == 3 and got["createTRANS"]["i"] is None
    assert {**_without(got, "compare_s", *COST), "createTRANS": None} == {**_without(res, "compare_s", *COST), "createTRANS": None}
    # once an e setting lacks its result file, the failed TRANS file is trashed and rebuilt; the filtered BAM is complete, so
    # the setting is compared again, not rerun, and mode i (all results present) is left alone
    os.replace(d / "chrA.c300_l10_m1_e.json", trash / "e2.json")
    got = again()
    assert calls == [("trans", "e")] and (trash / "chrA.trans_e.pkl").exists() and got["createTRANS"]["e"]["exit_status"] == 0
    assert got["createTRANS"]["i"] is None
    assert {**_without(got, "compare_s", *COST), "createTRANS": None} == {**_without(res, "compare_s", *COST), "createTRANS": None}


# ---------- stage 20 wiring ----------

FAKE_03 = """import os, sys
from pathlib import Path
(Path(os.environ["FAKE_OUT"]) / sys.argv[1] / "args03.txt").write_text(" ".join(sys.argv[2:]))
"""

# stand-in for stage 04: FAKE04=crash dies like a truncated BAM does; otherwise writes FAKE04_RESULT as exact_match.json,
# prints the summary line and exits as stage 04 does
FAKE_04 = """import json, os, sys
from pathlib import Path
d = Path(os.environ["FAKE_OUT"]) / sys.argv[1]
(d / "args04.txt").write_text(" ".join(sys.argv[2:]))
if os.environ["FAKE04"] == "crash":
    raise OSError("no BGZF EOF marker; file may be truncated")
res = json.loads(os.environ["FAKE04_RESULT"])
(d / "exact_match.json").write_text(json.dumps(res))
print(json.dumps({k: v["identical"] for k, v in res.items()}))
sys.exit(0 if all(v["identical"] for v in res.values()) else 1)
"""


def _match(n_o, n_r, only_o, only_r):
    return {"n_original": n_o, "n_reimpl": n_r, "only_original": only_o, "only_reimpl": only_r,
            "identical": only_o == 0 and only_r == 0, "examples": []}


def _stage20(tmp_path, monkeypatch):
    """Stage 20 loaded as a module with REPO and OUT pointed at a stand-in layout (fake stages 03 and 04)."""
    repo, out = tmp_path / "repo", tmp_path / "out"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "03_reimpl_flags.py").write_text(FAKE_03)
    (repo / "scripts" / "04_compare_removed.py").write_text(FAKE_04)
    orig = out / "1417" / "original"
    orig.mkdir(parents=True)
    for tag in ("c300_l10_m1_i", "c300_lnone_m0_e"):
        (orig / f"{tag}.bam").touch()
        (orig / f"{tag}.time.txt").write_text(OK_TIME)
    spec = importlib.util.spec_from_file_location("stage20", REPO / "scripts" / "20_validation_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "REPO", repo)
    monkeypatch.setattr(mod, "OUT", out)
    monkeypatch.setenv("FAKE_OUT", str(out))
    return mod, out / "1417"


def test_compare_library_records_a_difference(tmp_path, monkeypatch):
    mod, d = _stage20(tmp_path, monkeypatch)
    result = {"c300_l10_m1_i": _match(10, 9, 1, 0), "c300_lnone_m0_e": _match(5, 5, 0, 0)}
    monkeypatch.setenv("FAKE04", "ok")
    monkeypatch.setenv("FAKE04_RESULT", json.dumps(result))
    got = mod.compare_library("1417", discover_settings(d / "original"), 7)
    # stage 03 gets every setting once, plus the worker count; stage 04 the same settings
    assert (d / "args03.txt").read_text() == "--setting 300,10,1,i --setting 300,none,0,e --workers 7"
    assert (d / "args04.txt").read_text() == "--setting 300,10,1,i --setting 300,none,0,e"
    # stage 04's exit 1 (a difference) is recorded; run cost from OK_TIME: 1:02:03 = 3723 s, 2 GiB, no mem.json
    cost = {"original_elapsed_s": 3723.0, "original_max_rss_gib": 2.0, "original_peak_tree_rss_gib": None}
    assert got == {tag: {**v, **cost} for tag, v in result.items()}
    assert got["c300_l10_m1_i"]["identical"] is False


def test_compare_library_raises_when_stage04_crashes(tmp_path, monkeypatch):
    mod, d = _stage20(tmp_path, monkeypatch)
    # an earlier run left an all-identical exact_match.json; this run's stage 04 dies with exit 1 and writes nothing
    (d / "exact_match.json").write_text(json.dumps({"c300_l10_m1_i": _match(10, 10, 0, 0),
                                                   "c300_lnone_m0_e": _match(5, 5, 0, 0)}))
    monkeypatch.setenv("FAKE04", "crash")
    with pytest.raises(RuntimeError, match="stage 04 on 1417"):
        mod.compare_library("1417", discover_settings(d / "original"), 7)

def test_stage20_without_libraries_or_control(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "20_validation_compare.py"), "--libs", "--skip-none-control",
                        "--skip-1179", "--out", str(tmp_path)], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0
    assert json.loads((tmp_path / "validation" / "exact_match_m2.json").read_text()) == {"n_compared": 0, "all_identical": True}
    assert json.loads(r.stdout) == {"none_control": None, "1179_subset": None, "n_compared": 0, "all_identical": True}


def test_stage20_1179_block_on_a_stand_in(synth, tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("stage20", REPO / "scripts" / "20_validation_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # the real block: chr21 + chr22 of 1179 at the 8 published settings (4 in each mode)
    assert mod.SUBSET_1179_REFS == ("chr21", "chr22") and mod.LIBRARIES["1179"] == LIBRARIES["1179"]
    assert len(published()) == 8 and sum(s.introns for s in published()) == 4
    monkeypatch.setattr(mod, "LIBRARIES", {**LIBRARIES, "1179": synth.bam})
    monkeypatch.setattr(mod, "SUBSET_1179_REFS", ("chrA", "chrB"))
    monkeypatch.setattr(mod, "DB", synth.db)
    monkeypatch.setattr(mod, "FASTA", synth.fasta)
    monkeypatch.setattr(mod, "GTF", synth.gtf)
    monkeypatch.setattr(sys, "argv", ["20", "--libs", "--skip-none-control", "--out", str(tmp_path)])
    calls = _recorded(monkeypatch)
    with pytest.raises(SystemExit) as ex:
        mod.main()
    assert ex.value.code == 0
    tags = [setting_tag(s) for s in published()]
    assert sorted(calls[:2]) == [("trans", "e"), ("trans", "i")] and calls[2:] == [("run", t) for t in tags]
    em = json.loads((tmp_path / "validation" / "exact_match_m2.json").read_text())
    assert set(em) == {"1179_subset", "n_compared", "all_identical"}
    assert em["n_compared"] == 8 and em["all_identical"] is True
    blk = em["1179_subset"]
    assert blk["refs"] == ["chrA", "chrB"] and blk["n_records"] == len(_keys(synth.bam))
    assert list(blk["settings"]) == tags
    for s in published():
        r = blk["settings"][setting_tag(s)]
        want = serial_original(synth, s.covlen, s.min_len, s.mism, s.introns)
        assert r["identical"] is True and r["n_original"] == r["n_reimpl"] == len(want) > 0
    assert sorted(p.name for p in (tmp_path / "validation" / "subset_1179").glob("*.json")) == sorted(
        [f"chrA_chrB.{t}.json" for t in tags] + [f"chrA_chrB.{t}.mem.json" for t in tags]
        + [f"chrA_chrB.createTRANS_{m}.mem.json" for m in "ie"])
    assert json.loads(capsys.readouterr().out) == {"none_control": None, "1179_subset": {t: True for t in tags},
                                                   "n_compared": 8, "all_identical": True}
    # resume: every setting has its result file, so nothing is subset, built, run or compared again
    for name in ("subset_bam", "compare_subset"):
        monkeypatch.setattr(validation, name, lambda *a, **k: pytest.fail("rerun"))
    calls.clear()
    with pytest.raises(SystemExit) as ex:
        mod.main()
    assert ex.value.code == 0 and calls == []
    assert json.loads((tmp_path / "validation" / "exact_match_m2.json").read_text()) == em
    # a difference in one setting makes the stage fail and is folded into all_identical
    p = tmp_path / "validation" / "subset_1179" / f"chrA_chrB.{tags[0]}.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), "only_reimpl": 1, "identical": False}))
    with pytest.raises(SystemExit) as ex:
        mod.main()
    em = json.loads((tmp_path / "validation" / "exact_match_m2.json").read_text())
    assert ex.value.code == 1 and em["all_identical"] is False and em["n_compared"] == 8
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_validation.py`
Expected: FAIL; the first error is `ModuleNotFoundError: No module named 'pafbench.validation'`.

- [ ] **Step 3: Create `pafbench/validation.py`**

```python
"""Exact-match validation against the original polyAfilter: run bookkeeping, the remove-all-non-terminal control and the
original tool on a reference subset."""
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pysam

from .config import REPO, VENDOR_POLYAFILTER
from .flags import Setting, flag_reference, removed_from_flags, removed_keys_by_ref, setting_tag
from .report import parse_gnu_time
from .transcripts import load_transcripts
from .vendor import polyafilter_modules

_TAG = re.compile(r"c(\d+)_l(none|[1-9]\d*)_m(\d+)_([ie])")
_SIGNAL = re.compile(r"^Command terminated by signal (\d+)$", re.M)
TRASH = ("gio", "trash")


def parse_tag(tag):
    """Inverse of flags.setting_tag; 'lnone' is the remove-all control, 'l0' is refused like parse_setting does."""
    m = _TAG.fullmatch(tag)
    if m is None:
        raise ValueError(f"not a setting tag: {tag}")
    c, l, mism, mode = m.groups()
    return Setting(int(c), None if l == "none" else int(l), int(mism), mode == "i")


def setting_arg(s):
    """Inverse of flags.parse_setting: the --setting argument of stages 03 and 04."""
    return f"{s.covlen},{'none' if s.min_len is None else s.min_len},{s.mism},{'i' if s.introns else 'e'}"


def time_exit_status(tfile):
    """Exit status from a GNU time file; 128 + N for a command killed by signal N, for which GNU time writes 'Exit status: 0'."""
    m = _SIGNAL.search(Path(tfile).read_text())
    return 128 + int(m.group(1)) if m else parse_gnu_time(tfile).get("exit_status")


def run_stats(prefix):
    """Wall time, largest-process max RSS and exit status from <prefix>.time.txt; whole-tree peak from <prefix>.mem.json if present."""
    t = parse_gnu_time(f"{prefix}.time.txt")
    mem = Path(f"{prefix}.mem.json")
    peak = json.loads(mem.read_text())["peak_tree_rss_gib"] if mem.exists() else None
    return {"elapsed_s": t.get("elapsed_s"), "max_rss_gib": t.get("max_rss_gib"),
            "exit_status": time_exit_status(f"{prefix}.time.txt"), "peak_tree_rss_gib": peak}


def discover_settings(orig_dir):
    """Settings of every <tag>.bam in orig_dir whose <tag>.time.txt reports exit status 0 (not killed), in file-name order."""
    out = []
    for bam in sorted(Path(orig_dir).glob("*.bam")):
        tfile = bam.with_suffix(".time.txt")
        if tfile.exists() and time_exit_status(tfile) == 0:
            out.append(parse_tag(bam.stem))
    return out


def subset_bam(bam, refs, out_bam):
    """Records of the given references (samtools view -b -M keeps file order), indexed; the header keeps every reference."""
    pysam.view("-b", "-M", "-o", str(out_bam), str(bam), *refs, catch_stdout=False)
    pysam.index(str(out_bam))


def _timed(cmd, prefix):
    """cmd under memwatch and /usr/bin/time, run from REPO so a child python imports this checkout's pafbench:
    <prefix>.mem.json, .time.txt, .log (stdout and stderr)."""
    full = [sys.executable, "-m", "pafbench.memwatch", f"{prefix}.mem.json", "--",
            "/usr/bin/time", "-v", "-o", f"{prefix}.time.txt", *cmd]
    with open(f"{prefix}.log", "w") as log:
        subprocess.run(full, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)


def create_trans(db, bam, trans, include_introns, prefix):
    """polyAfilter createTRANS with stage 02's command line, under /usr/bin/time and memwatch: <prefix>.time.txt, .mem.json, .log."""
    _timed([sys.executable, str(VENDOR_POLYAFILTER / "polyAfilter.py"), "createTRANS", *(["-i"] if include_introns else []),
            str(db), str(bam), str(trans)], prefix)


def run_original(bam, trans_pkl, db, fasta, out_bam, setting, threads=4):
    """The original scanBAMfilter as the CLI's BAMfilter calls it, at setting (min_len None is the remove-all control), fresh tROC.

    trans_pkl is loaded if it exists, otherwise built from db; the -p temp files go next to out_bam and are deleted by the tool.
    Like the CLI, it does nothing when out_bam already exists.
    """
    BAMfilter, _, _ = polyafilter_modules()
    BAMfilter.scanBAMfilter(setting.covlen, setting.min_len, str(bam), str(fasta), str(trans_pkl), out_db=str(db), base="A",
                            mism=setting.mism, out_bamfile=str(out_bam), tROC={}, includeIntrons=setting.introns, cbFile=None,
                            out_cbFile=None, weightedCov=True, verbose=True, nThreads=threads)


def run_original_timed(bam, trans_pkl, db, fasta, out_bam, setting, prefix, threads=4):
    """run_original in a child process (python -m pafbench.validation) under memwatch and /usr/bin/time, files as create_trans."""
    _timed([sys.executable, "-m", "pafbench.validation", str(bam), str(trans_pkl), str(db), str(fasta), str(out_bam),
            setting_tag(setting), str(threads)], prefix)


def compare_subset(bam, out_bam, fasta, transcripts, setting):
    """Removed records of the original (bam minus out_bam) against flag_reference at setting, per reference."""
    tag = setting_tag(setting)
    n_o = n_r = only_o = only_r = 0
    examples = []
    for ref, want in removed_keys_by_ref(bam, out_bam):
        got = set() if ref is None else removed_from_flags(flag_reference(bam, fasta, ref, transcripts, [setting]), tag)
        a, b = want - got, got - want
        n_o += len(want)
        n_r += len(got)
        only_o += len(a)
        only_r += len(b)
        examples.extend([["only_original", *k] for k in sorted(a, key=str)[:20]]
                        + [["only_reimpl", *k] for k in sorted(b, key=str)[:20]])
    return {"setting": tag, "n_original": n_o, "n_reimpl": n_r, "only_original": only_o, "only_reimpl": only_r,
            "identical": only_o == 0 and only_r == 0, "examples": examples[:20]}


def none_control(bam, refs, out_dir, db, fasta, gtf, covlen=300, threads=4):
    """Remove-all control in intron mode on a reference subset: subset BAM, createTRANS -i, original in-process, comparison.

    Files in out_dir are prefixed with the joined reference names; existing subset, TRANS and filtered BAM are reused.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = "_".join(refs)
    st = Setting(covlen, None, 0, True)
    sub, trans = out_dir / f"{label}.bam", out_dir / f"{label}.trans_i.pkl"
    filtered = out_dir / f"{label}.{setting_tag(st)}.bam"
    secs = {}
    t0 = time.monotonic()
    if not Path(f"{sub}.bai").exists():
        subset_bam(bam, refs, sub)
    secs["subset"] = time.monotonic() - t0
    t0 = time.monotonic()
    if not trans.exists():
        create_trans(db, sub, trans, True, out_dir / f"{label}.createTRANS_i")
    secs["createTRANS"] = time.monotonic() - t0
    t0 = time.monotonic()
    run_original(sub, trans, db, fasta, filtered, st, threads=threads)
    secs["original"] = time.monotonic() - t0
    t0 = time.monotonic()
    res = compare_subset(sub, filtered, fasta, load_transcripts(gtf), st)
    secs["compare"] = time.monotonic() - t0
    with pysam.AlignmentFile(str(sub)) as fh:
        n = sum(s.total for s in fh.get_index_statistics())
    return {"refs": list(refs), "n_records": n, **res, "createTRANS": run_stats(out_dir / f"{label}.createTRANS_i"),
            "seconds": secs}


def _completed(out, prefix):
    """True when out exists and <prefix>.time.txt reports exit 0; an out left by a failed or killed run goes to the trash
    (the original tool and createTRANS return early on an existing output)."""
    out, tfile = Path(out), Path(f"{prefix}.time.txt")
    if out.exists() and tfile.exists() and time_exit_status(tfile) == 0:
        return True
    if out.exists():
        subprocess.run([*TRASH, str(out)], check=True)
    return False


def subset_validation(bam, refs, out_dir, db, fasta, gtf, settings, threads=4):
    """Original tool against the reimplementation on a reference subset: subset BAM, createTRANS per mode, then per setting
    the original in a measured child process and the comparison, saved as <label>.<tag>.json.

    Files in out_dir are prefixed with the joined reference names. Resume: an existing subset (with .bai) is reused, TRANS
    files and filtered BAMs are reused when their time file reports exit 0, and a setting with a result file is not rerun.
    A mode's TRANS file is checked and built only while one of its settings lacks its result file; its createTRANS cost is
    None when the time file is gone.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = "_".join(refs)
    sub = out_dir / f"{label}.bam"
    if not Path(f"{sub}.bai").exists():
        subset_bam(bam, refs, sub)
    modes = {"i" if s.introns else "e": s.introns for s in settings}
    trans = {m: out_dir / f"{label}.trans_{m}.pkl" for m in modes}
    ct = {m: out_dir / f"{label}.createTRANS_{m}" for m in modes}
    need = {"i" if s.introns else "e" for s in settings if not (out_dir / f"{label}.{setting_tag(s)}.json").exists()}
    todo = [m for m in modes if m in need and not _completed(trans[m], ct[m])]
    with ThreadPoolExecutor(max(len(todo), 1)) as ex:
        list(ex.map(lambda m: create_trans(db, sub, trans[m], modes[m], ct[m]), todo))
    transcripts = None
    res = {}
    for st in settings:
        tag = setting_tag(st)
        done = out_dir / f"{label}.{tag}.json"
        if not done.exists():
            filtered, prefix = out_dir / f"{label}.{tag}.bam", out_dir / f"{label}.{tag}"
            if not _completed(filtered, prefix):
                run_original_timed(sub, trans["i" if st.introns else "e"], db, fasta, filtered, st, prefix, threads)
            transcripts = transcripts or load_transcripts(gtf)
            t0 = time.monotonic()
            cmp = compare_subset(sub, filtered, fasta, transcripts, st)
            cost = run_stats(prefix)
            done.write_text(json.dumps({**cmp, "original_elapsed_s": cost["elapsed_s"],
                                        "original_max_rss_gib": cost["max_rss_gib"],
                                        "original_peak_tree_rss_gib": cost["peak_tree_rss_gib"],
                                        "compare_s": time.monotonic() - t0}, indent=1))
        res[tag] = json.loads(done.read_text())
    with pysam.AlignmentFile(str(sub)) as fh:
        n = sum(s.total for s in fh.get_index_statistics())
    return {"refs": list(refs), "n_records": n,
            "createTRANS": {m: run_stats(ct[m]) if Path(f"{ct[m]}.time.txt").exists() else None for m in modes},
            "settings": res}


def main(argv):
    """python -m pafbench.validation BAM TRANS_PKL DB FASTA OUT_BAM TAG THREADS: run_original in this process."""
    bam, trans_pkl, db, fasta, out_bam, tag, threads = argv
    run_original(bam, trans_pkl, db, fasta, out_bam, parse_tag(tag), int(threads))


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 4: Create `scripts/19_validation_runs.sh`**

```bash
#!/usr/bin/env bash
# usage: 19_validation_runs.sh
# Original-tool runs for the milestone 2 exact-match validation: a 1417 chain and a 1419 chain run concurrently, each
# step under memwatch (OUT/LIB/original/<name>.mem.json). Logs: OUT/validation/runs_<lib>.log. Exit 1 if any step failed.
# A step is skipped when its output exists and its .time.txt reports exit status 0 without "Command terminated by signal"
# (GNU time writes "Exit status: 0" for a killed command); a leftover output without that is trashed first, because
# BAMfilter returns without filtering when its output BAM exists.
# One instance at a time: a second launch exits 1 while OUT/validation/runs.lock is held. The lock (fd 9) is held by
# this shell, the two chain subshells and each step's memwatch; memwatch starts the step itself without inherited fds,
# so a step whose memwatch dies keeps running without the lock.
# DRYRUN=1 prints the commands (chains one after the other) and writes nothing.
# PAFBENCH_OUT replaces OUT for this script's own files and checks, PAFBENCH_TRASH replaces "gio trash" (both for tests);
# 02_run_original.sh always writes under /data/element_runs/polyafilter_bench.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY="$REPO/.venv/bin/python"
OUT=${PAFBENCH_OUT:-/data/element_runs/polyafilter_bench}
DB=/data/element_runs/polyafilter_bench/ref/genes_2024A.db
DRY=${DRYRUN:-0}
TRASH=${PAFBENCH_TRASH:-gio trash}
cd "$REPO"
BAM19=$("$PY" -c "from pafbench.config import LIBRARIES; print(LIBRARIES['1419'])")

# step LIB NAME OUTPUT TIMEFILE LOGFILE|- CMD...   (BG=" &" marks a parallel step in the dry-run listing)
step() {
  local lib=$1 name=$2 output=$3 tfile=$4 redir=$5
  shift 5
  local log=$OUT/validation/runs_$lib.log
  local cmd=("$PY" -m pafbench.memwatch "$OUT/$lib/original/$name.mem.json" -- "$@")
  if [ -e "$output" ] && grep -q "Exit status: 0$" "$tfile" 2>/dev/null && ! grep -q "terminated by signal" "$tfile"; then
    if [ "$DRY" = 1 ]; then echo "# skip $lib $name"; else echo "$(date '+%F %T') skip $name" >> "$log"; fi
    return 0
  fi
  if [ "$DRY" = 1 ]; then
    [ -e "$output" ] && echo "$TRASH $output"
    if [ "$redir" = - ]; then echo "${cmd[*]}${BG:-}"; else echo "${cmd[*]} > $redir 2>&1${BG:-}"; fi
    return 0
  fi
  if [ -e "$output" ]; then $TRASH "$output" || return 1; fi
  echo "$(date '+%F %T') start $name: ${cmd[*]}" >> "$log"
  local rc=0
  if [ "$redir" = - ]; then "${cmd[@]}" || rc=$?; else "${cmd[@]}" > "$redir" 2>&1 || rc=$?; fi
  echo "$(date '+%F %T') end $name exit $rc" >> "$log"
  return "$rc"
}

# filter LIB COVLEN MINSNRLEN MISM MODE
filter() {
  local tag=c$2_l$3_m$4_$5 d=$OUT/$1/original
  step "$1" "$tag" "$d/$tag.bam" "$d/$tag.time.txt" - "$REPO/scripts/02_run_original.sh" "$@"
}

# trans MODE: 1419 createTRANS with 02_run_original.sh's command line
trans() {
  local d=$OUT/1419/original flag=()
  [ "$1" = i ] && flag=(-i)
  step 1419 "createTRANS_$1" "$d/trans_$1.pkl" "$d/createTRANS_$1.time.txt" "$d/createTRANS_$1.log" \
    /usr/bin/time -v -o "$d/createTRANS_$1.time.txt" "$PY" "$REPO/vendor/polyAfilter/polyAfilter.py" createTRANS \
    "${flag[@]}" "$DB" "$BAM19" "$d/trans_$1.pkl"
}

chain_1417() {
  local rc=0 s
  [ "$DRY" = 1 ] && echo "# chain 1417"
  for s in "275 9 1 i" "275 9 1 e" "180 11 2 i" "180 11 2 e" "300 10 0 i" "600 10 1 i" "375 3 2 e"; do
    filter 1417 $s || rc=1
  done
  return "$rc"
}

chain_1419() {
  local rc=0 s p1 p2
  if [ "$DRY" = 1 ]; then
    echo "# chain 1419"
    BG=" &" trans i
    BG=" &" trans e
    echo wait
  else
    trans i & p1=$!
    trans e & p2=$!
    wait "$p1" || rc=1
    wait "$p2" || rc=1
    [ "$rc" = 0 ] || return 1
  fi
  for s in "300 10 1 i" "375 3 2 i" "300 10 1 e" "275 9 1 i" "275 9 1 e" "180 11 2 i" "180 11 2 e" "375 3 2 e"; do
    filter 1419 $s || rc=1
  done
  return "$rc"
}

if [ "$DRY" = 1 ]; then
  chain_1417
  chain_1419
  exit 0
fi
mkdir -p "$OUT/validation" "$OUT/1417/original" "$OUT/1419/original"
exec 9> "$OUT/validation/runs.lock"
flock -n 9 || { echo "19_validation_runs.sh: another run holds $OUT/validation/runs.lock" >&2; exit 1; }
chain_1417 & a=$!
chain_1419 & b=$!
rc=0
wait "$a" || rc=1
wait "$b" || rc=1
exit "$rc"
```

- [ ] **Step 5: Create `scripts/20_validation_compare.py`**

```python
"""Stage 20: exact match of the original polyAfilter and the reimplementation on every completed original run, the
remove-all-non-terminal control on 1417 chr21 + chr22, and the original tool at the 8 published settings on 1179
chr21 + chr22 (spec 5.1: 1179 does not fit in memory in full, so it is validated by chromosome).

Original runs and stage 03/04 outputs live under OUT/<lib>/ (stages 03 and 04 have no --out); --out moves only this
stage's own files: validation/exact_match_m2.json, validation/none_control/ and validation/subset_1179/. The 1179 block
resumes: a setting whose result file exists in subset_1179/ is not rerun. Its cost, scaled by record count from a 1179
chr22-only check (8.3M of the 18.4M records): subset about 6 min, createTRANS e about 76 min with i (about 36 min) alongside,
then about 30 min per setting (original 24, comparison 6), about 5.4 h in all; original under 8 GiB, comparison about 10 GiB.
The exact_match_m2.json that stage 21 reads must come from a call without skip flags (each call rewrites it with only the
parts it ran); once the 1179 block has run, such a call only reads the block's result files.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from pafbench.config import FASTA, GTF, LIBRARIES, NUCLEI, OUT, REPO
from pafbench.flags import setting_tag
from pafbench.grid import published
from pafbench.validation import discover_settings, none_control, run_stats, setting_arg, subset_validation

DB = OUT / "ref" / "genes_2024A.db"
SUBSET_1179_REFS = ("chr21", "chr22")


def compare_library(lib, settings, workers):
    """Stage 03 with all settings (it rewrites every reimpl_flags/<ref>.parquet), then stage 04; per-tag match and run cost."""
    args = [a for s in settings for a in ("--setting", setting_arg(s))]
    subprocess.run([sys.executable, str(REPO / "scripts" / "03_reimpl_flags.py"), lib, *args, "--workers", str(workers)],
                   check=True)
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "04_compare_removed.py"), lib, *args], stdout=subprocess.PIPE,
                       text=True)
    # stage 04 prints its summary line only after writing exact_match.json: exit 1 without it is a crash, not a difference
    lines = r.stdout.splitlines()
    if r.returncode not in (0, 1) or not lines or set(json.loads(lines[-1])) != {setting_tag(s) for s in settings}:
        raise RuntimeError(f"stage 04 on {lib} exited {r.returncode} without its summary line")
    match = json.loads((OUT / lib / "exact_match.json").read_text())
    out = {}
    for s in settings:
        tag = setting_tag(s)
        st = run_stats(OUT / lib / "original" / tag)
        out[tag] = {**match[tag], "original_elapsed_s": st["elapsed_s"], "original_max_rss_gib": st["max_rss_gib"],
                    "original_peak_tree_rss_gib": st["peak_tree_rss_gib"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", nargs="*", default=list(NUCLEI))
    ap.add_argument("--skip-none-control", action="store_true")
    ap.add_argument("--skip-1179", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    res = {}
    for lib in args.libs:
        settings = discover_settings(OUT / lib / "original")
        res[lib] = compare_library(lib, settings, args.workers) if settings else {}
    flags = [v["identical"] for lib in args.libs for v in res[lib].values()]
    if not args.skip_none_control:
        res["none_control"] = none_control(LIBRARIES["1417"], ["chr21", "chr22"], args.out / "validation" / "none_control",
                                           DB, FASTA, GTF)
        flags.append(res["none_control"]["identical"])
    if not args.skip_1179:
        res["1179_subset"] = subset_validation(LIBRARIES["1179"], list(SUBSET_1179_REFS),
                                               args.out / "validation" / "subset_1179", DB, FASTA, GTF, published())
        flags.extend(v["identical"] for v in res["1179_subset"]["settings"].values())
    res["n_compared"] = len(flags)
    res["all_identical"] = all(flags)
    dest = args.out / "validation" / "exact_match_m2.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1))
    print(json.dumps({**{lib: {t: v["identical"] for t, v in res[lib].items()} for lib in args.libs},
                      "none_control": res.get("none_control", {}).get("identical"),
                      "1179_subset": ({t: v["identical"] for t, v in res["1179_subset"]["settings"].items()}
                                      if "1179_subset" in res else None),
                      "n_compared": res["n_compared"],
                      "all_identical": res["all_identical"]}))
    sys.exit(0 if res["all_identical"] else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Check the shell syntax**

Run: `chmod +x scripts/19_validation_runs.sh && bash -n scripts/19_validation_runs.sh`
Expected: no output.

- [ ] **Step 7: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `82 passed, 2 warnings`.

- [ ] **Step 8: Dry run against the real OUT**

Run: `DRYRUN=1 scripts/19_validation_runs.sh`
Expected (REPO = this repository, OUT = /data/element_runs/polyafilter_bench), and nothing written:
```
# chain 1417
REPO/.venv/bin/python -m pafbench.memwatch OUT/1417/original/c275_l9_m1_i.mem.json -- REPO/scripts/02_run_original.sh 1417 275 9 1 i
REPO/.venv/bin/python -m pafbench.memwatch OUT/1417/original/c275_l9_m1_e.mem.json -- REPO/scripts/02_run_original.sh 1417 275 9 1 e
REPO/.venv/bin/python -m pafbench.memwatch OUT/1417/original/c180_l11_m2_i.mem.json -- REPO/scripts/02_run_original.sh 1417 180 11 2 i
REPO/.venv/bin/python -m pafbench.memwatch OUT/1417/original/c180_l11_m2_e.mem.json -- REPO/scripts/02_run_original.sh 1417 180 11 2 e
REPO/.venv/bin/python -m pafbench.memwatch OUT/1417/original/c300_l10_m0_i.mem.json -- REPO/scripts/02_run_original.sh 1417 300 10 0 i
REPO/.venv/bin/python -m pafbench.memwatch OUT/1417/original/c600_l10_m1_i.mem.json -- REPO/scripts/02_run_original.sh 1417 600 10 1 i
REPO/.venv/bin/python -m pafbench.memwatch OUT/1417/original/c375_l3_m2_e.mem.json -- REPO/scripts/02_run_original.sh 1417 375 3 2 e
# chain 1419
... createTRANS -i ... trans_i.pkl ... &
... createTRANS ... trans_e.pkl ... &
wait
... 02_run_original.sh 1419 300 10 1 i
... 02_run_original.sh 1419 375 3 2 i
... 02_run_original.sh 1419 300 10 1 e
... 02_run_original.sh 1419 275 9 1 i
... 02_run_original.sh 1419 275 9 1 e
... 02_run_original.sh 1419 180 11 2 i
... 02_run_original.sh 1419 180 11 2 e
... 02_run_original.sh 1419 375 3 2 e
```
(the `...` lines carry the same memwatch prefix; the createTRANS lines use the command line of `02_run_original.sh`). 1417's milestone 1 runs (300/10/1 i and e, 375/3/2 i) and TRANS files are reused.

- [ ] **Step 9: Start the two chains in the background**

Run: `nohup scripts/19_validation_runs.sh > /data/element_runs/polyafilter_bench/validation/runs.out 2>&1 &`
Expected: within a minute `/data/element_runs/polyafilter_bench/validation/runs_1417.log` has a `start c275_l9_m1_i` line and `runs_1419.log` two `start createTRANS` lines; `pgrep -af polyAfilter.py` lists three processes. Timing from milestone 1: createTRANS 3.4 h (`-i`) and 5.9 h (exon mode), each filter run 1.6 to 2.1 h; the 1419 chain ends after about 22 h, the 1417 chain after about 14 h. Peak memory: one 375/3/2 `-i` run (46.0 GB in milestone 1) plus one other run at a time. Leave them running.

- [ ] **Step 10: Start the 1179 by-chromosome check in the background**

Run: `nohup .venv/bin/python scripts/20_validation_compare.py --libs --skip-none-control > /data/element_runs/polyafilter_bench/validation/subset_1179.out 2>&1 &`
(`--libs` with no value selects no full library, so only the 1179 block runs.) It cuts chr21 and chr22 out of the 1179 BAM, builds both TRANS files on that subset and runs the original tool in-process at the 8 published settings, comparing each with the reimplementation: about 5.4 h, under 8 GiB for the original and about 10 GiB for the comparison, about 11 GB of BAMs kept under `/data/element_runs/polyafilter_bench/validation/subset_1179/` as evidence. During the build, 1179 chr22 alone matched exactly at 300/10/1 `-i` (1,265,833 removed), 375/3/2 `-i` (3,383,660), 375/3/2 exon mode (2,023,776), 275/9/1 exon mode (652,544) and 180/11/2 `-i` (1,434,668). Results are stored per setting and reused by the final stage 20 call in Task 8. Go on with Task 2.

- [ ] **Step 11: Commit**

```bash
git add tests/test_validation.py pafbench/validation.py scripts/19_validation_runs.sh scripts/20_validation_compare.py
git -c user.name="Simone Picelli" -c user.email="simone.picelli@iob.ch" commit -m "Stages 19 and 20: original-tool runs for the exact-match validation (published settings in both modes, MISM 0, COVLEN 600, 1419) and their comparison, with the in-process remove-all control"
```

---

### Task 2: Truth sets from labeled pa molecules (stage 12)

**Files:**
- Create: `tests/test_truth.py`
- Create: `pafbench/truth.py`
- Create: `scripts/12_truth_sets.py`

**Interfaces:**
- Consumes: Task 0 `labels.label_molecules`, `molecules.select_members`, `regions.load_region_segments`, `regions.regions_at`, `config.SENS_CAPS`; milestone 1 stage outputs `OUT/<lib>/pa_sites.parquet`, `molecules.parquet` (rerun here for `mapq_min`), `members.parquet`.
- Produces:
  - `pafbench.truth.label_truth_molecules(mol, pa, members, segments, cutoffs=LABEL_CUTOFFS)`
  - `pafbench.truth.truth_reads(mol, pa, members, cap=max(SENS_CAPS))`
  - `pafbench.truth.check_pa_counts(mol, reads)`
  - `pafbench.truth.truth_summary(labeled, reads)`
- Produces (data): `OUT/<lib>/truth/molecules_labeled.parquet` = MOL_COLUMNS + `label, a_band, region, gene, has_ts, n_members`; `OUT/<lib>/truth/truth_reads.parquet` = `qname, mol, role ("pa" or "member"), dist5_exon, dist_mode, spliced` (members up to max(SENS_CAPS) = 2,000 nt); `OUT/<lib>/truth/hexamer_curve.tsv`.

Spec 4.1: a molecule (CB, UB, strand, own gene label) with a polyA-trimmed read has a known priming site; its label comes from the 20 nt of genome after the site; its other reads (members) are what polyAfilter must judge without seeing the site. The pa reads are joined back to their molecule with the same key consolidate_sites used, so molecules dropped for site spread contribute nothing.

- [ ] **Step 1: Write the failing tests in `tests/test_truth.py`**

```python
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest

from pafbench.molecules import MOL_COLUMNS, consolidate_sites
from pafbench.regions import load_region_segments, regions_at
from pafbench.sites import last_aligned_base, priming_site
from pafbench.truth import check_pa_counts, label_truth_molecules, truth_reads, truth_summary

REPO = Path(__file__).resolve().parents[1]

GTF_ROWS = [
    # tc on +: exons 101-200, 301-400, 501-600 (1-based), CDS 151-350; 0-based 199 CDS / 200 intron, 599 3UTR / 600 intergenic
    ("transcript", 101, 600, "+", "tc"), ("exon", 101, 200, "+", "tc"), ("exon", 301, 400, "+", "tc"),
    ("exon", 501, 600, "+", "tc"), ("CDS", 151, 200, "+", "tc"), ("CDS", 301, 350, "+", "tc"),
    # tm on -: exons 1001-1100, 1201-1300, CDS 1051-1250; 0-based 999 intergenic / 1000 3UTR, 1299 5UTR / 1300 intergenic
    ("transcript", 1001, 1300, "-", "tm"), ("exon", 1001, 1100, "-", "tm"), ("exon", 1201, 1300, "-", "tm"),
    ("CDS", 1051, 1100, "-", "tm"), ("CDS", 1201, 1250, "-", "tm"),
]

PA_COLS = ["qname", "cb", "ub", "strand", "ref", "site", "ts", "spliced", "genes", "a20", "down20", "up40", "hexamer", "re", "mapq"]
# Molecules after consolidate_sites (reads r8 without UB and E with spread 20 drop out): A=0, B=1, C=2, D=3, F=4.
PA_ROWS = [
    ["r1", "c1", "u1", "+", "chrZ", 600, 0, False, "g_tc", 15, "A" * 20, "C" * 40, False, "E", 255],          # A
    ["r2", "c1", "u1", "+", "chrZ", 600, 0, False, "g_tc", 15, "A" * 20, "C" * 40, False, "E", 3],            # A
    ["r3", "c1", "u2", "+", "chrZ", 200, 0, False, "g_tc", 3, "C" * 20, "C" * 40, True, "E", 255],            # B
    ["r4", "c2", "u1", "-", "chrZ", 1000, 2, True, "g_tm;g_x", 6, "C" * 20, "C" * 40, True, "E", 255],       # C
    ["r5", "c2", "u3", "-", "chrZ", 1300, 0, False, "", 9, "C" * 20, "C" * 40, False, "N", 255],              # D
    ["r6", "c3", "u1", "+", "chrZ", 300, 4, False, "g_tc", 15, "A" * 20, "C" * 40, False, "E", 255],          # E
    ["r7", "c3", "u1", "+", "chrZ", 320, 4, False, "g_tc", 15, "A" * 20, "C" * 40, False, "E", 255],          # E
    ["r8", "c1", None, "+", "chrZ", 600, 9, False, "g_tc", 15, "A" * 20, "C" * 40, False, "E", 255],          # no UB
    ["r9", "c1", "u1", "+", "chrZ", 560, 7, False, "g_tn", 12, "A" * 20, "C" * 40, False, "E", 255],          # F
]

MEMBER_COLS = ["mol", "qname", "dist5", "dist3", "spliced", "dup", "genes", "gene_match", "dist5_exon", "dist3_exon", "dist_mode"]
# dist5 = dist5_exon + 7 so a function reading the genomic distance gives other answers
MEMBER_ROWS = [
    [0, "m1", 307, 0, False, True, "g_tc", True, 300, 0, "exon"],         # kept at 1000 and 2000
    [0, "m2", 107, 0, False, True, "g_zz", False, 100, 0, "genome"],      # gene mismatch
    [0, "m3", 107, -26, False, True, "g_tc", True, 100, -26, "genome"],   # passes the site by 26
    [0, "m4", 1007, -25, False, True, "g_tc", True, 1000, -25, "genome"], # boundary: kept at both caps
    [0, "m5", 1008, 0, False, True, "g_tc", True, 1001, 0, "exon"],       # only at 2000
    [0, "m6", 2007, 5, False, True, "g_tc", True, 2000, 5, "exon"],       # only at 2000
    [0, "m7", 2008, 5, False, True, "g_tc", True, 2001, 5, "exon"],       # beyond 2000
    [2, "m8", 57, 10, True, False, "g_tm", True, 50, 10, "exon"],         # C
    [4, "m1", 347, 0, False, True, "g_tc", False, 340, 0, "exon"],        # same read, F's gene does not match
]


@pytest.fixture
def data(tmp_path):
    gtf = tmp_path / "t.gtf"
    with open(gtf, "w") as fh:
        for ft, s, e, strand, tid in GTF_ROWS:
            fh.write(f"chrZ\tt\t{ft}\t{s}\t{e}\t.\t{strand}\t.\tgene_id \"g_{tid}\"; transcript_id \"{tid}\";\n")
    pa = pd.DataFrame(PA_ROWS, columns=PA_COLS)
    mol = consolidate_sites(pa)
    members = pd.DataFrame(MEMBER_ROWS, columns=MEMBER_COLS)
    return mol, pa, members, load_region_segments(gtf)


def test_fixture_molecules(data):
    mol = data[0]
    # E (sites 300 and 320, spread 20) is dropped and r8 has no UB
    assert mol.mol.tolist() == [0, 1, 2, 3, 4]
    assert mol.site.tolist() == [600, 200, 1000, 1300, 560] and mol.n_pa.tolist() == [2, 1, 1, 1, 1]


def test_label_columns_and_order(data):
    mol, pa, members, seg = data
    out = label_truth_molecules(mol, pa, members, seg)
    assert out.columns.tolist() == MOL_COLUMNS + ["label", "a_band", "region", "gene", "has_ts", "n_members"]
    pd.testing.assert_frame_equal(out[MOL_COLUMNS], mol)
    # a20 15, 3, 6, 9, 12 at cut-offs 8/4
    assert out.label.tolist() == ["templated", "nontemplated", "ambiguous", "templated", "templated"]
    assert out.a_band.tolist() == ["14-20", "0-4", "5-7", "8-9", "10-13"]
    strict = label_truth_molecules(mol, pa, members, seg, {"templated_min": 14, "nontemplated_max": 8})
    assert strict.label.tolist() == ["templated", "nontemplated", "nontemplated", "ambiguous", "ambiguous"]


def test_region_at_last_cdna_base(data):
    mol, pa, members, seg = data
    out = label_truth_molecules(mol, pa, members, seg)
    # A + site 600 -> base 599 (3UTR; 600 is intergenic); B + site 200 -> 199 (CDS; 200 is intron)
    # C - site 1000 -> 1000 (3UTR; 999 is intergenic); D - site 1300 -> 1300 (intergenic; 1299 is 5UTR); F + 560 -> 559
    assert out.region.tolist() == ["3UTR", "CDS", "3UTR", "intergenic", "3UTR"]
    assert regions_at(seg, "chrZ", True, [600, 200]).tolist() == ["intergenic", "intron"]
    assert regions_at(seg, "chrZ", False, [999, 1299]).tolist() == ["intergenic", "5UTR"]


def test_region_matches_last_aligned_base_of_a_read(data):
    """Independent reference: the region at sites.last_aligned_base of real alignments ending at the molecule site."""
    mol, pa, members, seg = data
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "chrZ", "LN": 6000}]})
    expected = []
    for start, flag in ((570, 0), (170, 0), (1000, 16), (1300, 16), (530, 0)):
        a = pysam.AlignedSegment(hdr)
        a.reference_name, a.reference_start, a.cigarstring, a.flag = "chrZ", start, "30M", flag
        a.query_sequence = "C" * 30
        expected.append((priming_site(a), regions_at(seg, "chrZ", not a.is_reverse, [last_aligned_base(a)])[0]))
    out = label_truth_molecules(mol, pa, members, seg)
    assert list(zip(out.site, out.region)) == expected


def test_gene_has_ts_n_members(data):
    mol, pa, members, seg = data
    out = label_truth_molecules(mol, pa, members, seg)
    assert out.gene.tolist() == ["g_tc", "g_tc", "g_tm", "", "g_tn"]
    # A's reads have ts 0; r9 (ts 7) shares A's CB, UB, strand, ref but not its gene, so only F; r8 (ts 9) has no UB
    assert out.has_ts.tolist() == [False, False, True, False, True]
    # A: m1, m4 (m2 gene, m3 dist3 -26, m5..m7 beyond 1000); C: m8; F: m1 row has gene_match False
    assert out.n_members.tolist() == [2, 0, 1, 0, 0]
    assert out.n_members.dtype == np.int64 and out.has_ts.dtype == bool


def test_truth_reads(data):
    mol, pa, members, seg = data
    reads = truth_reads(mol, pa, members)
    expected = pd.DataFrame({
        "qname": ["r1", "r2", "r3", "r4", "r5", "r9", "m1", "m4", "m5", "m6", "m8"],
        "mol": np.array([0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 2], np.int64),
        "role": ["pa"] * 6 + ["member"] * 5,
        "dist5_exon": [np.nan] * 6 + [300.0, 1000.0, 1001.0, 2000.0, 50.0],
        "dist_mode": [""] * 6 + ["exon", "genome", "exon", "exon", "exon"],
        "spliced": [False, False, False, True, False, False, False, False, False, False, True],
    })
    pd.testing.assert_frame_equal(reads, expected)


def test_truth_reads_cap_1000(data):
    mol, pa, members, seg = data
    reads = truth_reads(mol, pa, members, cap=1000)
    m = reads[reads.role == "member"]
    assert m.qname.tolist() == ["m1", "m4", "m8"] and m.dist5_exon.tolist() == [300.0, 1000.0, 50.0]
    assert reads[reads.role == "pa"].qname.tolist() == ["r1", "r2", "r3", "r4", "r5", "r9"]


def test_pa_count_check(data):
    mol, pa, members, seg = data
    reads = truth_reads(mol, pa, members)
    check_pa_counts(mol, reads)
    with pytest.raises(ValueError, match="n_pa"):
        check_pa_counts(mol, reads.drop(index=1))
    # B loses its only pa read: zero reads for a molecule with n_pa 1
    with pytest.raises(ValueError, match="n_pa"):
        check_pa_counts(mol, truth_reads(mol, pa[pa.qname != "r3"], members))


def test_truth_summary(data):
    mol, pa, members, seg = data
    labeled = label_truth_molecules(mol, pa, members, seg)
    s = truth_summary(labeled, truth_reads(mol, pa, members))
    assert s == {
        "molecules": 5,
        "labels": {"templated": 3, "nontemplated": 1, "ambiguous": 1},
        "regions": {"intergenic": 1, "intron": 0, "ncexon": 0, "5UTR": 0, "CDS": 1, "3UTR": 3, "antisense": 0},
        "has_ts": 2,
        "truth_reads": {"pa": 6, "member": 5},
        # member dist5_exon 300, 1000, 1001, 2000, 50
        "members_within_cap": {"500": 2, "1000": 3, "2000": 5},
        # A (templated) and C (ambiguous) have members at 1000
        "molecules_with_members": {"templated": 1, "nontemplated": 0, "ambiguous": 1},
    }


def test_script_refuses_old_molecules(tmp_path):
    base = tmp_path / "lib"
    base.mkdir()
    pa = pd.DataFrame(PA_ROWS, columns=PA_COLS)
    pa.to_parquet(base / "pa_sites.parquet")
    consolidate_sites(pa).drop(columns="mapq_min").to_parquet(base / "molecules.parquet")
    pd.DataFrame(MEMBER_ROWS, columns=MEMBER_COLS).to_parquet(base / "members.parquet")
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "12_truth_sets.py"), "X1", "--base", str(base),
                        "--out", str(tmp_path / "out")], env=env, capture_output=True, text=True)
    assert r.returncode == 1
    assert "molecules.parquet predates milestone 2; rerun scripts/06_molecules.py X1" in r.stderr
    assert not (tmp_path / "out").exists()


# Cases the main fixture does not separate. After consolidate_sites: G=0 (s1, s2), H=1, I=2, J=3, K=4, L=5.
PA_EXTRA = [
    # G: two pa reads, the first with ts 5 and the last with ts 0, so has_ts is True ("any read", not "all" or "last")
    ["s1", "c5", "u5", "+", "chrZ", 560, 5, False, "g_tc", 15, "A" * 20, "C" * 40, False, "E", 255],
    ["s2", "c5", "u5", "+", "chrZ", 560, 0, False, "g_tc", 15, "A" * 20, "C" * 40, False, "E", 255],
    # H, I: same CB, UB and genes "" on opposite strands; J, K: same CB, UB, strand and genes "" on two references
    ["s3", "c6", "u6", "+", "chrZ", 3000, 0, False, "", 15, "A" * 20, "C" * 40, False, "N", 255],
    ["s4", "c6", "u6", "-", "chrZ", 3500, 0, False, "", 15, "A" * 20, "C" * 40, False, "N", 255],
    ["s5", "c7", "u7", "+", "chrZ", 4000, 0, False, "", 15, "A" * 20, "C" * 40, False, "N", 255],
    ["s6", "c7", "u7", "+", "chrY", 4000, 0, False, "", 15, "A" * 20, "C" * 40, False, "N", 255],
    # L: minus-strand site 1099, last cDNA base 1099 = CDS (tm CDS 1050..1099 0-based); 1100 is intron, 1098 CDS
    ["s7", "c8", "u8", "-", "chrZ", 1099, 0, False, "g_tm", 3, "C" * 20, "C" * 40, True, "E", 255],
]


def test_has_ts_join_key_and_minus_boundary(data):
    seg = data[3]
    pa = pd.DataFrame(PA_EXTRA, columns=PA_COLS)
    mol = consolidate_sites(pa)
    assert mol.mol.tolist() == [0, 1, 2, 3, 4, 5] and mol.n_pa.tolist() == [2, 1, 1, 1, 1, 1]
    members = pd.DataFrame(MEMBER_ROWS, columns=MEMBER_COLS).iloc[:0]
    out = label_truth_molecules(mol, pa, members, seg)
    assert out.has_ts.tolist() == [True, False, False, False, False, False]
    assert out.region.tolist() == ["3UTR", "intergenic", "intergenic", "intergenic", "intergenic", "CDS"]
    assert regions_at(seg, "chrZ", False, [1098, 1099, 1100]).tolist() == ["CDS", "CDS", "intron"]
    reads = truth_reads(mol, pa, members)
    # each pa read joins its own molecule only: the key holds strand and ref
    assert reads.qname.tolist() == ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    assert reads.mol.tolist() == [0, 0, 1, 2, 3, 4, 5] and (reads.role == "pa").all()
    check_pa_counts(mol, reads)
    # a duplicated pa row gives L n_pa + 1 rows
    with pytest.raises(ValueError, match="1 molecules do not receive n_pa pa reads \\(first mol 5\\)"):
        check_pa_counts(mol, pd.concat([reads, reads[reads.qname == "s7"]], ignore_index=True))


# Runs a stage script with pafbench.config.GTF pointed at a test GTF: GTF SCRIPT ARGS...
GTF_RUNNER = """
import runpy, sys
from pathlib import Path
import pafbench.config as c
c.GTF = Path(sys.argv[1])
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def _stage12(tmp_path, mol, *extra):
    base = tmp_path / "lib"
    base.mkdir()
    pa = pd.DataFrame(PA_ROWS, columns=PA_COLS)
    pa.to_parquet(base / "pa_sites.parquet")
    mol.to_parquet(base / "molecules.parquet")
    pd.DataFrame(MEMBER_ROWS, columns=MEMBER_COLS).to_parquet(base / "members.parquet")
    gtf = tmp_path / "t.gtf"
    with open(gtf, "w") as fh:
        for ft, s, e, strand, tid in GTF_ROWS:
            fh.write(f"chrZ\tt\t{ft}\t{s}\t{e}\t.\t{strand}\t.\tgene_id \"g_{tid}\"; transcript_id \"{tid}\";\n")
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, "-c", GTF_RUNNER, str(gtf), str(REPO / "scripts" / "12_truth_sets.py"), "X1",
                        "--base", str(base), *extra], env=env, capture_output=True, text=True)
    return base, r


def test_script_writes_truth_sets_under_base(tmp_path, data):
    mol, pa, members, seg = data
    base, r = _stage12(tmp_path, mol)
    assert r.returncode == 0, r.stderr
    out = base / "truth"  # --base without --out writes to <base>/truth
    labeled = label_truth_molecules(mol, pa, members, seg)
    reads = truth_reads(mol, pa, members)
    pd.testing.assert_frame_equal(pd.read_parquet(out / "molecules_labeled.parquet"), labeled)
    pd.testing.assert_frame_equal(pd.read_parquet(out / "truth_reads.parquet"), reads)
    curve = pd.read_csv(out / "hexamer_curve.tsv", sep="\t")
    assert curve.n.sum() == 5 and curve.set_index("a20").n[[3, 6, 9, 12, 15]].tolist() == [1, 1, 1, 1, 1]
    assert r.stdout.strip().splitlines() == [json.dumps({"library": "X1"} | truth_summary(labeled, reads))]


def test_script_checks_pa_counts_before_writing(tmp_path, data):
    mol = data[0]
    base, r = _stage12(tmp_path, mol.assign(n_pa=mol.n_pa + (mol.mol == 1)), "--out", str(tmp_path / "out"))
    assert r.returncode == 1 and "1 molecules do not receive n_pa pa reads (first mol 1)" in r.stderr
    assert not (tmp_path / "out").exists()
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_truth.py`
Expected: FAIL; the first error is `ModuleNotFoundError: No module named 'pafbench.truth'`.

- [ ] **Step 3: Create `pafbench/truth.py`**

```python
import numpy as np
import pandas as pd

from .config import LABEL_CUTOFFS, MEMBER_CAP, SENS_CAPS
from .labels import label_molecules
from .molecules import select_members
from .regions import REGION_NAMES, regions_at

MOL_KEY = ["cb", "ub", "strand", "ref", "genes"]
TRUTH_COLUMNS = ["qname", "mol", "role", "dist5_exon", "dist_mode", "spliced"]
LABELS = ("templated", "nontemplated", "ambiguous")


def _pa_join(mol, pa):
    """pa reads of each molecule: rows with CB and UB, inner-joined to mol on the molecule key (pa row order kept)."""
    d = pa[pa.cb.notna() & pa.ub.notna()]
    return d.merge(mol[["mol"] + MOL_KEY], on=MOL_KEY, how="inner")


def label_truth_molecules(mol, pa, members, segments, cutoffs=LABEL_CUTOFFS):
    """mol plus label, a_band, region at the last cDNA base, gene, has_ts and n_members (members selected at MEMBER_CAP)."""
    out = label_molecules(mol, cutoffs)
    plus = (out.strand == "+").to_numpy()
    last = out.site.to_numpy() - plus
    region = np.empty(len(out), object)
    for (ref, strand), idx in out.groupby(["ref", "strand"], sort=False).indices.items():
        region[idx] = regions_at(segments, ref, strand == "+", last[idx])
    j = _pa_join(mol, pa)
    has_ts = out.mol.isin(j.mol[j.ts > 0]).to_numpy()
    n = select_members(members, MEMBER_CAP).groupby("mol").size()
    n_members = out.mol.map(n).fillna(0).to_numpy(np.int64)
    gene = out.genes.str.split(";", n=1).str[0].to_numpy(object)
    return out.assign(region=region, gene=gene, has_ts=has_ts, n_members=n_members)


def truth_reads(mol, pa, members, cap=max(SENS_CAPS)):
    """Reads that carry a truth label: every pa read of a molecule (role pa), then members selected at cap (role member)."""
    j = _pa_join(mol, pa)
    p = pd.DataFrame({"qname": j.qname, "mol": j.mol.astype(np.int64), "role": "pa", "dist5_exon": np.nan,
                      "dist_mode": "", "spliced": j.spliced.astype(bool)})
    s = select_members(members, cap)
    m = pd.DataFrame({"qname": s.qname, "mol": s.mol.astype(np.int64), "role": "member",
                      "dist5_exon": s.dist5_exon.astype(np.float64), "dist_mode": s.dist_mode, "spliced": s.spliced.astype(bool)})
    return pd.concat([p, m], ignore_index=True)[TRUTH_COLUMNS]


def check_pa_counts(mol, reads):
    """Raise unless every molecule has exactly n_pa pa rows in reads."""
    got = mol.mol.map(reads[reads.role == "pa"].groupby("mol").size()).fillna(0).to_numpy(np.int64)
    bad = got != mol.n_pa.to_numpy()
    if bad.any():
        raise ValueError(f"{int(bad.sum())} molecules do not receive n_pa pa reads (first mol {int(mol.mol.to_numpy()[bad][0])})")


def truth_summary(labeled, reads):
    """Counts printed by stage 12; members within each sensitivity cap come from the member rows of reads."""
    member = reads[reads.role == "member"]
    with_members = labeled[labeled.n_members > 0].label.value_counts()
    return {
        "molecules": len(labeled),
        "labels": {k: int((labeled.label == k).sum()) for k in LABELS},
        "regions": {k: int((labeled.region == k).sum()) for k in REGION_NAMES},
        "has_ts": int(labeled.has_ts.sum()),
        "truth_reads": {k: int((reads.role == k).sum()) for k in ("pa", "member")},
        "members_within_cap": {str(c): int((member.dist5_exon <= c).sum()) for c in SENS_CAPS},
        "molecules_with_members": {k: int(with_members.get(k, 0)) for k in LABELS},
    }
```

- [ ] **Step 4: Create `scripts/12_truth_sets.py`**

```python
"""Stage 12: labeled molecules and truth reads (pa reads plus selected members)."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from pafbench.config import GTF, OUT
from pafbench.labels import hexamer_curve
from pafbench.regions import load_region_segments
from pafbench.truth import check_pa_counts, label_truth_molecules, truth_reads, truth_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--base", help="directory with molecules, pa_sites and members.parquet (default OUT/LIB)")
    ap.add_argument("--out", help="output directory (default <base>/truth: OUT/LIB/truth without --base, and a redirected "
                                  "--base never writes into OUT)")
    args = ap.parse_args()
    base = Path(args.base) if args.base else OUT / args.library
    out = Path(args.out) if args.out else base / "truth"
    mol = pd.read_parquet(base / "molecules.parquet")
    if "mapq_min" not in mol:
        sys.exit(f"molecules.parquet predates milestone 2; rerun scripts/06_molecules.py {args.library}")
    pa = pd.read_parquet(base / "pa_sites.parquet")
    members = pd.read_parquet(base / "members.parquet")
    labeled = label_truth_molecules(mol, pa, members, load_region_segments(GTF))
    reads = truth_reads(mol, pa, members)
    check_pa_counts(mol, reads)
    out.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(out / "molecules_labeled.parquet", index=False)
    reads.to_parquet(out / "truth_reads.parquet", index=False)
    hexamer_curve(labeled).to_csv(out / "hexamer_curve.tsv", sep="\t", index=False)
    print(json.dumps({"library": args.library} | truth_summary(labeled, reads)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `95 passed, 2 warnings`.

- [ ] **Step 6: Rebuild 1417 molecules with mapq_min, and build 1419 and 1179 from scratch**

Run:
```bash
for L in 1417 1419 1179; do mkdir -p /data/element_runs/polyafilter_bench/$L/runs; done
.venv/bin/python -m pafbench.memwatch /data/element_runs/polyafilter_bench/1417/runs/06_molecules.mem.json -- .venv/bin/python scripts/06_molecules.py 1417
for L in 1419 1179; do
  .venv/bin/python -m pafbench.memwatch /data/element_runs/polyafilter_bench/$L/runs/05_pa_sites.mem.json -- .venv/bin/python scripts/05_pa_sites.py $L
  .venv/bin/python -m pafbench.memwatch /data/element_runs/polyafilter_bench/$L/runs/06_molecules.mem.json -- .venv/bin/python scripts/06_molecules.py $L
done
```
Expected: 1417 prints `"molecules": 144474, "members": 83317` as in milestone 1 (consolidate_sites only gained a column, so molecule ids and members.parquet stay consistent), and `molecules.parquet` now has `mapq_min`. 1419 and 1179 print their pa read, molecule and member counts; 1179 (574M records) is the long one, about an hour for each stage at 16 workers.

- [ ] **Step 7: Stage 12 on the three libraries**

Run: `for L in 1417 1419 1179; do .venv/bin/python -m pafbench.memwatch /data/element_runs/polyafilter_bench/$L/runs/12_truth_sets.mem.json -- .venv/bin/python scripts/12_truth_sets.py $L; done`
Expected for 1417: 144,474 molecules, labels templated 81.87%, nontemplated 7.49%, ambiguous 10.64% (milestone 1: 81.9 / 7.5 / 10.6), 145,629 pa rows, members within 500 / 1,000 / 2,000 nt after gene_match: 73,200 / 76,245 / 77,966; about 20 s and 1 GiB. For 1419 and 1179 note the label shares and region counts; a refusal mentioning `mapq_min` means stage 06 was not rerun.

- [ ] **Step 8: Commit**

```bash
git add tests/test_truth.py pafbench/truth.py scripts/12_truth_sets.py
git -c user.name="Simone Picelli" -c user.email="simone.picelli@iob.ch" commit -m "Stage 12: labeled truth molecules (label, A band, region at the last cDNA base, gene, ts, members) and truth reads"
```

---

### Task 3: Streaming grid engine: all 396 settings in one pass per reference (stage 13)

**Files:**
- Create: `tests/test_stream.py`
- Create: `pafbench/stream.py`
- Create: `scripts/13_grid_flags.py`

**Interfaces:**
- Consumes: milestone 1 `transcripts.coverage_prefix`, `transcripts.expressed`, `transcripts.load_transcripts`, `snr.scan_snrs`, `snr.snr_pieces`, `intervals.flatten`, `intervals.intersect`, `intervals.blocks_hit`, `flags.flag_reference` (the reference the engine must equal), `flags.setting_tag`; Task 0 `grid.all_settings`; Task 2 `truth_reads.parquet`.
- Produces:
  - `pafbench.stream.coverage_prefix_chunked(bam_path, ref, chunk=2000000)`
  - `pafbench.stream.body_pieces(expressed_tx, covlen, include_introns, strd)`
  - `pafbench.stream.reference_pieces(seq, tx_by_strand, prefix, settings)`
  - `pafbench.stream.scan_reference(bam_path, fasta_path, ref, transcripts, settings, truth_qnames, cells=None, chunk=1000000)`
  - `pafbench.stream.scan_to_parts(bam_path, fasta_path, ref, transcripts, settings, truth_qnames, cells, part_dir, chunk=1000000, meta=None)`
  - `pafbench.stream.stale_parts(part_dir, refs, meta)`
  - `pafbench.stream.merge_regions(parts, tags)`
  - `pafbench.stream.merge_parts(part_dir, refs, tags, out_dir)`
- Produces (data): `OUT/<lib>/grid/truth_flags.parquet` (`qname, ref, pos, flag` + one bool column per tag, primary records of truth reads), `region_counts.parquet` (`tag, re, removed, total`), `cell_counts.parquet` (index `cb` = called cells, `total` + one uint32 column per tag), `summary.json`, `parts/` (per reference, for restarts).

`flags.flag_reference` matches the original tool record for record but keeps one Python row per record, which does not scale to 396 settings on 1179 (chr1 alone has 49.9M records). This engine computes the same pieces (expressed transcripts, protected ends, A-stretch windows) once per reference and setting, then streams records in fixed-size numpy chunks and keeps only aggregates plus the rows of truth reads. Every equality test compares it with flag_reference. Measured during the build: 1417 chr21 identical to stage 03 on all 1,837,866 records; 1179 chr1 with 396 settings 740 s at 2.75 GiB per worker.

- [ ] **Step 1: Write the failing tests in `tests/test_stream.py`**

```python
import gzip
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pysam
import pytest

from pafbench import stream
from pafbench.config import REPO
from pafbench.flags import Setting, flag_reference, setting_tag
from pafbench.grid import all_settings
from pafbench.intervals import blocks_hit, flatten, intersect
from pafbench.snr import scan_snrs, snr_pieces
from pafbench.stream import (body_pieces, coverage_prefix_chunked, merge_parts, reference_pieces, scan_reference, scan_to_parts,
                             stale_parts)
from pafbench.transcripts import coverage_prefix, expressed, load_transcripts, start_pieces

SETTINGS = [
    Setting(200, None, 0, True),
    Setting(200, None, 0, False),
    Setting(100, 5, 0, False),
    Setting(300, 8, 1, False),
    Setting(300, 8, 1, True),
    Setting(150, 3, 2, True),
    Setting(150, 3, 2, False),
    Setting(50, 10, 0, True),
    Setting(450, 12, 2, False),
    Setting(600, 9, 1, True),
]
TAGS = [setting_tag(s) for s in SETTINGS]
CELLS = ["GGGTTT-1", "AAACCC-1", "CCCGGG-1"]           # deliberately unsorted
CB_CYCLE = ["AAACCC-1", "CCCGGG-1", "GGGTTT-1", "TTTAAA-1", None]   # TTTAAA-1 is not a called cell
RE_CYCLE = ["E", "N", "I", None]


@pytest.fixture(scope="module")
def tagged(synth, tmp_path_factory):
    """Copy of the synthetic BAM with CB, UB, RE, some supplementary and QC-fail flags, and secondaries sharing a primary's qname."""
    path = tmp_path_factory.mktemp("stream") / "tagged.bam"
    with pysam.AlignmentFile(str(synth.bam)) as fin, pysam.AlignmentFile(str(path), "wb", template=fin) as out:
        prev = None
        for i, a in enumerate(fin.fetch(until_eof=True)):
            cb = CB_CYCLE[i % len(CB_CYCLE)]
            if cb is not None:
                a.set_tag("CB", cb)
            a.set_tag("UB", f"U{i % 7}")
            re_ = "X" if i % 17 == 16 else RE_CYCLE[i % len(RE_CYCLE)]
            if re_ is not None:
                a.set_tag("RE", re_)
            if i % 11 == 5:
                a.flag |= 2048
            if i % 13 == 7:
                a.flag |= 512
            if a.is_secondary and prev is not None:
                a.query_name = prev
            prev = a.query_name
            out.write(a)
    pysam.index(str(path))
    return path


def _expected(bam, fasta, ref, tx, cells):
    """flag_reference rows of primary records, with CB and RE read independently in fetch order."""
    df = flag_reference(bam, fasta, ref, tx, SETTINGS)
    with pysam.AlignmentFile(str(bam)) as fh:
        tags = [(a.get_tag("CB") if a.has_tag("CB") else None, a.get_tag("RE") if a.has_tag("RE") else "") for a in fh.fetch(ref)]
    df["cb"] = [t[0] for t in tags]
    df["re"] = [t[1] for t in tags]
    return df[(df.flag & 0x900) == 0].reset_index(drop=True)


def _region_long(prim):
    rows = []
    for re_, g in prim.groupby("re", sort=False):
        for t in TAGS:
            rows.append((t, re_, int(g[t].sum()), len(g)))
    return sorted(rows)


def _cells_expected(prim, cells):
    sub = prim[prim.cb.isin(cells)]
    idx = sorted(cells)
    want = sub.groupby("cb")[TAGS].sum().reindex(idx, fill_value=0)
    want.insert(0, "total", sub.groupby("cb").size().reindex(idx, fill_value=0))
    return want


def test_coverage_prefix_chunked_equals_coverage_prefix(synth):
    for ref in ("chrA", "chrB"):
        want = coverage_prefix(synth.bam, ref)
        for chunk in (1, 5, 2_000_000):
            got = coverage_prefix_chunked(synth.bam, ref, chunk=chunk)
            for strd in (True, False):
                assert np.array_equal(got[strd], want[strd]), (ref, chunk, strd)


def test_coverage_prefix_chunked_iupac_and_deletions(tmp_path):
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "c", "LN": 60}]})
    path = tmp_path / "x.bam"
    recs = [("a", 5, "4M2D4M", 0, "ACRTACGT"), ("d", 5, "3M", 256, None), ("b", 20, "3M", 16, "ANA"), ("c", 30, "2M10N2M", 0, "ACGT")]
    with pysam.AlignmentFile(str(path), "wb", header=hdr) as out:
        for name, pos, cigar, flag, seq in recs:
            x = pysam.AlignedSegment(hdr)
            x.query_name, x.reference_name, x.reference_start, x.cigarstring, x.flag = name, "c", pos, cigar, flag
            if seq is not None:
                x.query_sequence, x.query_qualities = seq, pysam.qualitystring_to_array("I" * len(seq))
            out.write(x)
    pysam.index(str(path))
    want = coverage_prefix(path, "c")
    for chunk in (1, 2, 100):
        got = coverage_prefix_chunked(path, "c", chunk=chunk)
        for strd in (True, False):
            assert np.array_equal(got[strd], want[strd])
    cov = np.diff(want[True])
    # forward: a covers 5-8 and 11-14 except the R at 7; c covers 30-31 and 42-43; d has no SEQ
    assert np.flatnonzero(cov).tolist() == [5, 6, 8, 11, 12, 13, 14, 30, 31, 42, 43]
    # reverse: b covers 20-22 except the N at 21
    assert np.flatnonzero(np.diff(want[False])).tolist() == [20, 22]


def test_body_pieces_equal_start_pieces(synth):
    tx = load_transcripts(synth.gtf)
    rng = np.random.default_rng(11)
    sizes = []
    for strd in (True, False):
        rand = []
        for _ in range(300):
            pos, exons = int(rng.integers(0, 200_000)), []
            for _ in range(int(rng.integers(1, 6))):
                ln = int(rng.integers(20, 600))
                exons.append((pos, pos + ln))
                pos += ln + int(rng.integers(30, 4000))
            rand.append((exons[0][0], exons[-1][1], sorted(exons, reverse=strd)))
        sets = [rand] + [tx.get((ref, strd), []) for ref in ("chrA", "chrB")]
        for txs in sets:
            for covlen in (50, 150, 300, 600):
                for introns in (False, True):
                    want = start_pieces(txs, covlen, introns, strd)
                    got = body_pieces(txs, covlen, introns, strd)
                    assert got[0].tolist() == want[0].tolist() and got[1].tolist() == want[1].tolist()
                    sizes.append(want[0].size)
    assert min(sizes[:8]) > 50 and max(sizes[8:16]) > 0


def test_reference_pieces_equal_flag_reference_pieces(synth):
    tx = load_transcripts(synth.gtf)
    with pysam.FastaFile(str(synth.fasta)) as fa:
        seqs = {r: fa.fetch(r) for r in ("chrA", "chrB")}
    for ref in ("chrA", "chrB"):
        prefix = coverage_prefix(synth.bam, ref)
        tx_by_strand = {strd: tx.get((ref, strd), []) for strd in (True, False)}
        got = reference_pieces(seqs[ref], tx_by_strand, prefix, SETTINGS)
        assert set(got) == {(t, strd) for t in TAGS for strd in (True, False)}
        n_nonempty = 0
        for st in SETTINGS:
            for strd in (True, False):
                # recomputed exactly as flag_reference does, without caches
                ex = expressed(tx_by_strand[strd], prefix[strd], st.introns)
                bs, be = start_pieces(ex, st.covlen, st.introns, strd)
                if st.min_len is None:
                    ws, we = bs, be
                else:
                    first, last = scan_snrs(seqs[ref], strd, st.mism)
                    fs, fe = flatten(*snr_pieces(first, last, st.min_len, st.covlen, strd, len(seqs[ref])))
                    ws, we = intersect(bs, be, fs, fe)
                ps, pe = got[(setting_tag(st), strd)]
                assert ps.tolist() == ws.tolist() and pe.tolist() == we.tolist(), (ref, st, strd)
                n_nonempty += ps.size > 0
        assert n_nonempty > 0


def test_hit_kernel_edges_by_hand():
    # pieces of one tag: [10, 20) and [30, 40). Blocks (sorted by start) and whether each touches a piece:
    # [5, 10) no (ends where the piece starts); [8, 11) yes; [19, 20) yes; [20, 30) no (touches both, overlaps neither);
    # [25, 31) yes; [40, 50) no.
    bs = np.array([5, 8, 19, 20, 25, 40], np.int64)
    be = np.array([10, 11, 20, 30, 31, 50], np.int64)
    br = np.arange(6, dtype=np.int64)
    ps, pe, offs = stream._pack([(np.array([10, 30]), np.array([20, 40]))])
    tru = _kernel_matrix(bs, be, br, 6, ps, pe, offs)
    assert tru[:, 0].tolist() == [False, True, True, False, True, False]
    assert tru[:, 0].tolist() == blocks_hit(bs, be, np.array([10, 30]), np.array([20, 40])).tolist()


def _kernel_matrix(bs, be, br, n_rec, ps, pe, offs):
    """Per-record hits from the kernel: every record primary, its own truth row, region code 0, no cell."""
    n_tags = offs.size - 1
    tru = np.zeros((n_rec, n_tags), bool)
    reg = np.zeros((n_tags, 1), np.int64)
    cel = np.zeros((n_tags, 0), np.uint32)
    stream._scan_hits(bs, be, br, ps, pe, offs, np.full(n_rec, -1, np.int64), np.ones(n_rec, bool),
                      np.zeros(n_rec, np.int64), np.full(n_rec, -1, np.int64), np.arange(n_rec, dtype=np.int64), reg, cel, tru)
    assert reg[:, 0].tolist() == tru.sum(axis=0).tolist()
    return tru


def test_hit_kernel_matches_blocks_hit_random():
    rng = np.random.default_rng(3)
    n_rec, n_tags = 400, 12
    starts = np.sort(rng.integers(0, 5000, n_rec))
    blk_s, blk_e, blk_r = [], [], []
    for r, s in enumerate(starts):
        pos = int(s)
        for _ in range(rng.integers(1, 4)):
            ln = int(rng.integers(1, 60))
            blk_s.append(pos)
            blk_e.append(pos + ln)
            blk_r.append(r)
            pos += ln + int(rng.integers(1, 300))
    blk_s, blk_e, blk_r = (np.array(x, np.int64) for x in (blk_s, blk_e, blk_r))
    pieces = []
    for t in range(n_tags):
        k = int(rng.integers(0, 60))
        s = rng.integers(0, 5600, k)
        pieces.append(flatten(s, s + rng.integers(1, 80 * (t + 1), k)))
    ps, pe, offs = stream._pack(pieces)
    order = np.argsort(blk_s, kind="stable")
    got = _kernel_matrix(blk_s[order], blk_e[order], blk_r[order], n_rec, ps, pe, offs)
    want = np.zeros((n_rec, n_tags), bool)
    for t, (s, e) in enumerate(pieces):
        want[np.unique(blk_r[blocks_hit(blk_s, blk_e, s, e)]), t] = True
    assert np.array_equal(got, want)
    assert 0 < want.sum() < want.size


@pytest.mark.parametrize("chunk", [3, 1000])
def test_scan_reference_equals_flag_reference(synth, tagged, chunk):
    tx = load_transcripts(synth.gtf)
    with pysam.AlignmentFile(str(tagged)) as fh:
        qnames = {a.query_name for a in fh.fetch(until_eof=True)}
    n_removed = 0
    for ref in ("chrA", "chrB"):
        prim = _expected(tagged, synth.fasta, ref, tx, CELLS)
        truth, region, cells = scan_reference(tagged, synth.fasta, ref, tx, SETTINGS, qnames, cells=CELLS, chunk=chunk)
        assert list(truth.columns) == ["qname", "ref", "pos", "flag"] + TAGS
        assert truth.qname.tolist() == prim.qname.tolist()
        assert (truth.ref == ref).all()
        assert truth.pos.tolist() == prim.pos.tolist() and truth.flag.tolist() == prim.flag.tolist()
        assert np.array_equal(truth[TAGS].to_numpy(), prim[TAGS].to_numpy())
        assert sorted(region[["tag", "re", "removed", "total"]].itertuples(index=False, name=None)) == _region_long(prim)
        want = _cells_expected(prim, CELLS)
        assert cells.index.name == "cb" and cells.index.tolist() == sorted(CELLS)
        assert list(cells.columns) == ["total"] + TAGS
        assert (cells.dtypes == np.uint32).all()
        assert np.array_equal(cells.to_numpy(np.int64), want.to_numpy(np.int64))
        assert region.attrs["records"] == len(flag_reference(tagged, synth.fasta, ref, tx, SETTINGS[:1]))
        n_removed += int(prim[TAGS].to_numpy().sum())
        assert set(region.re) <= {"E", "N", "I", "", "X"}
    assert n_removed > 0


def test_scan_reference_truth_subset_and_no_cells(synth, tagged):
    tx = load_transcripts(synth.gtf)
    full, region_full, _ = scan_reference(tagged, synth.fasta, "chrA", tx, SETTINGS, set(_qnames(tagged, "chrA")), cells=CELLS)
    keep = set(full.qname.iloc[::3]) | {"no_such_read"}
    sub, region_sub, cells = scan_reference(tagged, synth.fasta, "chrA", tx, SETTINGS, keep, chunk=4)
    want = full[full.qname.isin(keep)].reset_index(drop=True)
    pd.testing.assert_frame_equal(sub, want)
    assert 0 < len(sub) < len(full)
    assert cells is None
    pd.testing.assert_frame_equal(region_sub, region_full)


def test_secondary_sharing_a_truth_qname_is_not_a_truth_row(synth, tagged):
    tx = load_transcripts(synth.gtf)
    with pysam.AlignmentFile(str(tagged)) as fh:
        sec = [a for a in fh.fetch("chrA") if a.is_secondary]
        prim_names = {a.query_name for a in fh.fetch("chrA") if not a.is_secondary and not a.is_supplementary}
    shared = {a.query_name for a in sec} & prim_names
    assert shared
    truth, _, _ = scan_reference(tagged, synth.fasta, "chrA", tx, SETTINGS[:2], shared)
    assert sorted(truth.qname) == sorted(shared)
    assert ((truth.flag & 0x900) == 0).all()


def _qnames(bam, ref):
    with pysam.AlignmentFile(str(bam)) as fh:
        return [a.query_name for a in fh.fetch(ref)]


def test_parts_and_merge_equal_flag_reference(synth, tagged, tmp_path):
    tx = load_transcripts(synth.gtf)
    with pysam.AlignmentFile(str(tagged)) as fh:
        qnames = {a.query_name for a in fh.fetch(until_eof=True)}
    part_dir = tmp_path / "parts"
    stats = {ref: scan_to_parts(tagged, synth.fasta, ref, tx, SETTINGS, qnames, CELLS, part_dir, chunk=5) for ref in ("chrB", "chrA")}
    summary = merge_parts(part_dir, ["chrA", "chrB"], TAGS, tmp_path)
    full = {ref: flag_reference(tagged, synth.fasta, ref, tx, SETTINGS) for ref in ("chrA", "chrB")}
    prims = [_expected(tagged, synth.fasta, ref, tx, CELLS) for ref in ("chrA", "chrB")]
    prim = pd.concat(prims, ignore_index=True)

    truth = pd.read_parquet(tmp_path / "truth_flags.parquet")
    assert list(truth.columns) == ["qname", "ref", "pos", "flag"] + TAGS
    assert truth.qname.tolist() == prim.qname.tolist() and truth.ref.tolist() == prim.ref.tolist()
    assert truth.pos.tolist() == prim.pos.tolist() and truth.flag.tolist() == prim.flag.tolist()
    assert np.array_equal(truth[TAGS].to_numpy(), prim[TAGS].to_numpy())

    region = pd.read_parquet(tmp_path / "region_counts.parquet")
    # rows in settings order, then RE in string order
    assert list(region.itertuples(index=False, name=None)) == sorted(_region_long(prim), key=lambda r: (TAGS.index(r[0]), r[1]))

    cells = pd.read_parquet(tmp_path / "cell_counts.parquet")
    assert cells.index.name == "cb" and cells.index.tolist() == sorted(CELLS)
    assert list(cells.columns) == ["total"] + TAGS and (cells.dtypes == np.uint32).all()
    assert np.array_equal(cells.to_numpy(np.int64), _cells_expected(prim, CELLS).to_numpy(np.int64))

    for ref, p, f in zip(("chrA", "chrB"), prims, (full["chrA"], full["chrB"])):
        assert summary["per_ref"][ref]["records"] == stats[ref]["records"] == len(f)
        assert summary["per_ref"][ref]["primary"] == stats[ref]["primary"] == len(p)
    assert summary["records"] == len(full["chrA"]) + len(full["chrB"])
    assert summary["primary"] == len(prim)
    assert summary["removed"] == {t: int(prim[t].sum()) for t in TAGS}
    # chrA has 145 records, 5 of them secondary (flag 256); the tagged copy adds supplementary bits on top
    assert len(full["chrA"]) == 145 and len(prims[0]) < 140


def test_parts_resume_and_no_cells(synth, tagged, tmp_path, monkeypatch):
    tx = load_transcripts(synth.gtf)
    keep = set(_qnames(tagged, "chrA")[::4])
    part_dir = tmp_path / "parts"
    first = {ref: scan_to_parts(tagged, synth.fasta, ref, tx, SETTINGS, keep, None, part_dir) for ref in ("chrA", "chrB")}
    assert not list(part_dir.glob("*.cells.parquet"))

    def boom(*args, **kwargs):
        raise AssertionError("scan_reference ran again")

    monkeypatch.setattr(stream, "scan_reference", boom)
    again = {ref: scan_to_parts(tagged, synth.fasta, ref, tx, SETTINGS, keep, None, part_dir) for ref in ("chrA", "chrB")}
    assert again == first
    summary = merge_parts(part_dir, ["chrA", "chrB"], TAGS, tmp_path)
    assert not (tmp_path / "cell_counts.parquet").exists()
    truth = pd.read_parquet(tmp_path / "truth_flags.parquet")
    want = _expected(tagged, synth.fasta, "chrA", tx, CELLS)
    want = want[want.qname.isin(keep)].reset_index(drop=True)
    # chrB contributes no truth rows; the chrA rows come through with their string and integer types
    assert truth.qname.tolist() == want.qname.tolist() and (truth.ref == "chrA").all()
    assert np.array_equal(truth[TAGS].to_numpy(), want[TAGS].to_numpy())
    assert truth.pos.dtype == np.int64 and truth.flag.dtype == np.int64
    assert summary["primary"] == len(_expected(tagged, synth.fasta, "chrA", tx, CELLS)) + len(_expected(tagged, synth.fasta, "chrB", tx, CELLS))


def _write_bam(path, contigs, recs):
    """Sorted, indexed BAM from (qname, ref, pos, cigar, flag, seq or None, tags) tuples."""
    hdr = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6", "SO": "coordinate"},
                                           "SQ": [{"SN": c, "LN": n} for c, n in contigs.items()]})
    raw = path.with_suffix(".unsorted.bam")
    with pysam.AlignmentFile(str(raw), "wb", header=hdr) as out:
        for name, ref, pos, cigar, flag, seq, tags in recs:
            a = pysam.AlignedSegment(hdr)
            a.query_name, a.reference_name, a.reference_start, a.cigarstring, a.flag = name, ref, pos, cigar, flag
            a.mapping_quality = 255
            if seq is not None:
                a.query_sequence, a.query_qualities = seq, pysam.qualitystring_to_array("I" * len(seq))
            for k, v in tags.items():
                a.set_tag(k, v)
            out.write(a)
    pysam.sort("-o", str(path), str(raw))
    pysam.index(str(path))
    return path


def test_expressed_modes_intron_only_read(tmp_path):
    # One + transcript on contig c: exons [100, 300) and [1000, 1200) (0-based), intron [300, 1000); COVLEN 200.
    # -i: the last 200 nt [1000, 1200) are protected, body [100, 1000). Default mode: the exonic last 200 nt
    # [1000, 1200) are protected, body [100, 300).
    # Forward reads: s0 [150, 210) in exon 1 without SEQ (adds no coverage), i1 [500, 560) inside the intron with SEQ,
    # f2 [2000, 2060) outside the transcript. Only i1 covers bases, so the transcript is expressed with -i (its span is
    # covered) and not in default mode (no exonic base covered).
    rng = np.random.default_rng(5)
    fasta = tmp_path / "c.fa"
    fasta.write_text(">c\n" + "".join(rng.choice(list("ACGT"), 3000)) + "\n")
    pysam.faidx(str(fasta))
    gtf = tmp_path / "c.gtf"
    attr = 'gene_id "gx"; transcript_id "tx";'
    gtf.write_text("".join(f"c\tt\t{ft}\t{s}\t{e}\t.\t+\t.\t{attr}\n"
                           for ft, s, e in (("transcript", 101, 1200), ("exon", 101, 300), ("exon", 1001, 1200))))
    seq = "".join(rng.choice(list("ACGT"), 60))
    bam = _write_bam(tmp_path / "c.bam", {"c": 3000}, [("s0", "c", 150, "60M", 0, None, {}),
                                                         ("i1", "c", 500, "60M", 0, seq, {}),
                                                         ("f2", "c", 2000, "60M", 0, seq, {})])
    tx = load_transcripts(gtf)
    prefix = coverage_prefix(bam, "c")
    assert len(expressed(tx[("c", True)], prefix[True], True)) == 1
    assert expressed(tx[("c", True)], prefix[True], False) == []
    settings = [Setting(200, None, 0, True), Setting(200, None, 0, False), Setting(200, 3, 1, True), Setting(200, 3, 1, False)]
    tags = [setting_tag(s) for s in settings]
    want = flag_reference(bam, fasta, "c", tx, settings)
    for chunk in (1, 1000):
        truth, region, _ = scan_reference(bam, fasta, "c", tx, settings, {"s0", "i1", "f2"}, chunk=chunk)
        assert truth.qname.tolist() == ["s0", "i1", "f2"]
        assert truth["c200_lnone_m0_i"].tolist() == [True, True, False]
        assert truth["c200_lnone_m0_e"].tolist() == [False, False, False]
        for t in tags:
            assert truth[t].tolist() == want[t].tolist(), t
        assert region.set_index("tag").removed.to_dict() == {t: int(want[t].sum()) for t in tags}


@pytest.mark.parametrize("capacity", [1, 2])
def test_scan_reference_with_tiny_block_buffers(synth, tagged, monkeypatch, capacity):
    """Block buffers of 1 or 2 blocks: flushes on a full buffer, and (capacity 1) grows it for a two-block record."""
    caps, flushes = [], []
    init, blocks, hits = stream._Chunk.__init__, stream._Chunk._blocks, stream._scan_hits

    def spy_blocks(self, cap):
        caps.append(cap)
        blocks(self, cap)

    def small(self, chunk):
        init(self, chunk)
        self._blocks(capacity)

    def spy_hits(*args):
        flushes.append(1)
        hits(*args)

    monkeypatch.setattr(stream._Chunk, "_blocks", spy_blocks)
    monkeypatch.setattr(stream._Chunk, "__init__", small)
    monkeypatch.setattr(stream, "_scan_hits", spy_hits)
    tx = load_transcripts(synth.gtf)
    prim = _expected(tagged, synth.fasta, "chrA", tx, CELLS)
    for chunk in (2, 1000):
        caps.clear()
        flushes.clear()
        truth, region, cells = scan_reference(tagged, synth.fasta, "chrA", tx, SETTINGS, set(prim.qname), cells=CELLS,
                                              chunk=chunk)
        assert truth.qname.tolist() == prim.qname.tolist()
        assert np.array_equal(truth[TAGS].to_numpy(), prim[TAGS].to_numpy())
        assert sorted(region[["tag", "re", "removed", "total"]].itertuples(index=False, name=None)) == _region_long(prim)
        assert np.array_equal(cells.to_numpy(np.int64), _cells_expected(prim, CELLS).to_numpy(np.int64))
        # 145 records on chrA; a buffer of at most 4 blocks forces far more flushes (two kernel calls each) than chunk 1000
        assert len(flushes) > 2 * 30
        if capacity == 1:
            assert caps[:2] == [4 * chunk + 64, 1] and 4 in caps[2:]  # spliced and indel records have 2 blocks
        else:
            assert caps == [4 * chunk + 64, 2]


def test_parts_store_their_inputs_and_stale_parts_are_named(synth, tagged, tmp_path):
    tx = load_transcripts(synth.gtf)
    part_dir = tmp_path / "parts"
    meta = {"truth": "/x/truth_reads.parquet", "truth_rows": 12, "settings": len(SETTINGS)}
    st = scan_to_parts(tagged, synth.fasta, "chrB", tx, SETTINGS, set(), None, part_dir, meta=meta)
    stored = json.loads((part_dir / "chrB.json").read_text())
    assert stored == st and {k: stored[k] for k in meta} == meta and stored["ref"] == "chrB"
    # chrA has no part yet, so nothing can be stale there
    assert stale_parts(part_dir, ["chrA", "chrB"], meta) == []
    assert stale_parts(part_dir, ["chrA", "chrB"], meta | {"truth_rows": 13}) == ["chrB"]
    assert stale_parts(part_dir, ["chrB"], meta | {"truth": "/y/truth_reads.parquet"}) == ["chrB"]
    assert stale_parts(part_dir, ["chrB"], meta | {"settings": 396}) == ["chrB"]
    # a part written without the guard (no inputs stored) is stale as well
    scan_to_parts(tagged, synth.fasta, "chrA", tx, SETTINGS, set(), None, part_dir)
    assert stale_parts(part_dir, ["chrA", "chrB"], meta) == ["chrA"]


# Runs a stage script with pafbench.config pointed at test inputs: FASTA GTF BAM(of library X) SCRIPT ARGS...
RUNNER = """
import runpy, sys
from pathlib import Path
import pafbench.config as c
c.FASTA, c.GTF, c.LIBRARIES = Path(sys.argv[1]), Path(sys.argv[2]), {"X": Path(sys.argv[3])}
sys.argv = sys.argv[4:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


@pytest.fixture(scope="module")
def stage13_inputs(synth, tagged, tmp_path_factory):
    """Library X: the tagged synthetic BAM plus chrC (reads, no annotation), its called cells, and a genome with chrC.

    chrB has reads only on the minus strand, so its plus-strand transcript t4 is never expressed.
    """
    d = tmp_path_factory.mktemp("stage13")
    fasta = d / "genome.fa"
    rng = np.random.default_rng(9)
    fasta.write_text(synth.fasta.read_text() + ">chrC\n" + "".join(rng.choice(list("ACGT"), 1000)) + "\n")
    pysam.faidx(str(fasta))
    outs = d / "lib" / "outs"
    (outs / "filtered_feature_bc_matrix").mkdir(parents=True)
    with gzip.open(outs / "filtered_feature_bc_matrix" / "barcodes.tsv.gz", "wt") as fh:
        fh.write("".join(c + "\n" for c in CELLS))
    recs = []
    with pysam.AlignmentFile(str(tagged)) as fh:
        for a in fh.fetch(until_eof=True):
            tags = {k: v for k, v in a.get_tags()}
            recs.append((a.query_name, a.reference_name, a.reference_start, a.cigarstring, a.flag, a.query_sequence, tags))
    seq = "".join(rng.choice(list("ACGT"), 50))
    for i, (pos, flag, re_, cb) in enumerate([(100, 0, "I", "AAACCC-1"), (300, 16, "I", "GGGTTT-1"), (500, 0, None, "TTTAAA-1"),
                                              (700, 1024, "N", "AAACCC-1"), (800, 256, "I", "AAACCC-1")]):
        tags = {"CB": cb} | ({} if re_ is None else {"RE": re_})
        recs.append((f"c{i}", "chrC", pos, "50M", flag, seq, tags))
    bam = _write_bam(outs / "possorted_genome_bam.bam", {"chrA": 6000, "chrB": 2000, "chrC": 1000}, recs)
    chrA = [r[0] for r in recs if r[1] == "chrA" and not r[4] & 0x900]
    truth = d / "truth_reads.parquet"
    pd.DataFrame({"qname": chrA[::2] + ["not_in_bam"]}).to_parquet(truth)
    return fasta, synth.gtf, bam, truth


def _stage13(inputs, out, *extra):
    fasta, gtf, bam, truth = inputs
    script = REPO / "scripts" / "13_grid_flags.py"
    return subprocess.run([sys.executable, "-c", RUNNER, str(fasta), str(gtf), str(bam), str(script), "X", "--workers", "2",
                           "--out", str(out), "--truth", str(truth), *extra],
                          capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(REPO)), cwd=REPO)


def _grid_outputs(out):
    summary = json.loads((out / "summary.json").read_text())
    return (pd.read_parquet(out / "truth_flags.parquet"), pd.read_parquet(out / "region_counts.parquet"),
            pd.read_parquet(out / "cell_counts.parquet"),
            {k: summary[k] for k in ("records", "primary", "removed")} | {"per_ref": {r: {k: v for k, v in s.items() if k != "seconds"}
                                                                                     for r, s in summary["per_ref"].items()}})


@pytest.fixture(scope="module")
def stage13_fresh(stage13_inputs, tmp_path_factory):
    out = tmp_path_factory.mktemp("fresh") / "grid"
    r = _stage13(stage13_inputs, out, "--refs", "chrA", "chrB", "chrC", "chrZ")
    assert r.returncode == 0, r.stderr
    return out, r


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")  # flag_reference adds 396 columns one by one
def test_stage13_empty_parts_keep_their_region_totals(stage13_inputs, stage13_fresh):
    fasta, gtf, bam, truth = stage13_inputs
    out, r = stage13_fresh
    assert "chrZ" in r.stderr  # asked for, but not in the BAM index
    line = json.loads(r.stdout.strip().splitlines()[-1])
    assert (line["library"], line["settings"], line["scanned"], line["refs"]) == ("X", 396, 3, 3)
    settings = all_settings()
    tags = [setting_tag(s) for s in settings]
    tx = load_transcripts(gtf)
    keep = set(pd.read_parquet(truth).qname)
    full = {ref: flag_reference(bam, fasta, ref, tx, settings) for ref in ("chrA", "chrB", "chrC")}
    with pysam.AlignmentFile(str(bam)) as fh:
        for ref, df in full.items():
            recs = list(fh.fetch(ref))
            df["re"] = [a.get_tag("RE") if a.has_tag("RE") else "" for a in recs]
            df["cb"] = [a.get_tag("CB") if a.has_tag("CB") else None for a in recs]
    prim = {ref: df[(df.flag & 0x900) == 0].reset_index(drop=True) for ref, df in full.items()}
    # chrB (no expressed transcript on +, no truth read) and chrC (no annotation, no truth read) give no truth rows
    tf, region, cells, summary = _grid_outputs(out)
    want = prim["chrA"][prim["chrA"].qname.isin(keep)].reset_index(drop=True)
    assert tf.qname.tolist() == want.qname.tolist() and set(tf.ref) == {"chrA"} and len(tf) == len(keep) - 1
    assert np.array_equal(tf[tags].to_numpy(), want[tags].to_numpy())
    # but their primary records stay in the region totals: chrC adds I, I, "" and N (the duplicate counts, the secondary not)
    allp = pd.concat(prim.values(), ignore_index=True)
    want_region = sorted((t, re_, int(g[t].sum()), len(g)) for re_, g in allp.groupby("re") for t in tags)
    assert sorted(region.itertuples(index=False, name=None)) == want_region
    assert prim["chrC"].re.tolist() == ["I", "I", "", "N"] and not prim["chrC"][tags].to_numpy().any()
    # chrB: t4 (+) has no read and t5 (-, 80 nt) is shorter than every COVLEN, so wholly protected: nothing removed
    assert not prim["chrB"][tags].to_numpy().any() and len(prim["chrB"]) > 0
    assert summary["per_ref"]["chrC"] == {"records": 5, "primary": 4} and summary["primary"] == len(allp)
    for ref in ("chrB", "chrC"):
        assert len(pd.read_parquet(out / "parts" / f"{ref}.truth.parquet")) == 0
        part = pd.read_parquet(out / "parts" / f"{ref}.region.parquet")
        assert part[part.tag == tags[0]].total.sum() == len(prim[ref]) and part.removed.sum() == 0
    assert summary["removed"] == {t: int(allp[t].sum()) for t in tags}
    in_cells = allp[allp.cb.isin(CELLS)]
    assert cells.total.to_dict() == in_cells.groupby("cb").size().to_dict()
    assert np.array_equal(cells[tags].to_numpy(np.int64), in_cells.groupby("cb")[tags].sum().reindex(cells.index).to_numpy(np.int64))


def test_stage13_resume_after_crash_and_stale_truth(stage13_inputs, stage13_fresh, tmp_path):
    fresh_out, _ = stage13_fresh
    fasta, gtf, bam, fixture_truth = stage13_inputs
    truth = tmp_path / "truth_reads.parquet"
    shutil.copy(fixture_truth, truth)
    inputs = (fasta, gtf, bam, truth)
    out = tmp_path / "grid"
    # a run that finished chrA only, then a crash in chrC after its truth part was written but before its json
    r = _stage13(inputs, out, "--refs", "chrA")
    assert r.returncode == 0, r.stderr
    pd.read_parquet(out / "parts" / "chrA.truth.parquet").head(3).to_parquet(out / "parts" / "chrC.truth.parquet")
    r = _stage13(inputs, out)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["scanned"] == 2  # chrB and chrC, not chrA again
    got, want = _grid_outputs(out), _grid_outputs(fresh_out)
    pd.testing.assert_frame_equal(got[0], want[0])
    pd.testing.assert_frame_equal(got[1], want[1])
    pd.testing.assert_frame_equal(got[2], want[2])
    assert got[3] == want[3]
    # a second resume scans nothing and changes nothing
    r = _stage13(inputs, out)
    assert r.returncode == 0 and json.loads(r.stdout.strip().splitlines()[-1])["scanned"] == 0
    assert _grid_outputs(out)[3] == want[3]
    # stage 12 reran and rewrote the truth file at the same path with other rows: refused, nothing rewritten
    before = {p.name: p.stat().st_mtime_ns for p in out.iterdir() if p.is_file()}
    pd.concat([pd.read_parquet(truth), pd.DataFrame({"qname": ["extra"]})]).to_parquet(truth)
    r = _stage13(inputs, out)
    assert r.returncode == 1
    assert "chrA, chrB, chrC" in r.stderr and f"gio trash {out / 'parts'}" in r.stderr
    assert {p.name: p.stat().st_mtime_ns for p in out.iterdir() if p.is_file()} == before
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_stream.py`
Expected: FAIL; the first error is `ImportError: cannot import name 'stream' from 'pafbench' (pafbench/__init__.py)`.

- [ ] **Step 3: Create `pafbench/stream.py`**

```python
"""Per-record polyAfilter removal for many settings in two streaming passes per reference (stage 13).

Pass 1 builds the per-strand coverage prefix (only needed to decide which transcripts are expressed).
Pass 2 reads records in chunks and checks each record's blocks against every setting's pieces with a
merge kernel; only aggregates (per RE code, per called cell) and the rows of truth reads are kept.
Results equal flags.flag_reference record for record.
"""
import json
import time
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pysam
from numba import njit

from .flags import setting_tag
from .intervals import flatten, intersect
from .snr import scan_snrs, snr_pieces
from .transcripts import expressed
from .vendor import polyafilter_modules

SECONDARY_OR_SUPPLEMENTARY = 0x900
RE_CODES = ("E", "N", "I", "")


@njit(cache=True)
def _add_blocks(diff_f, diff_r, s, e, fwd, n):
    # Shifted by one: diff[p + 1] holds the depth change at position p. A non-ACGT aligned base at r arrives as
    # (r + 1, r): it undoes its own block's +1 at that one position.
    for i in range(n):
        if fwd[i]:
            diff_f[s[i] + 1] += 1
            diff_f[e[i] + 1] -= 1
        else:
            diff_r[s[i] + 1] += 1
            diff_r[e[i] + 1] -= 1


def coverage_prefix_chunked(bam_path, ref, chunk=2_000_000):
    """Same values as transcripts.coverage_prefix (int32), with blocks buffered and flushed every `chunk` blocks."""
    with pysam.AlignmentFile(str(bam_path)) as bam:
        L = bam.get_reference_length(ref)
        diff_f, diff_r = np.zeros(L + 2, np.int32), np.zeros(L + 2, np.int32)
        bs, be, bf = np.empty(chunk, np.int64), np.empty(chunk, np.int64), np.empty(chunk, np.bool_)
        k = 0
        for a in bam.fetch(ref):
            seq = a.query_sequence
            if seq is None:
                continue
            fwd = not a.is_reverse
            items = a.get_blocks()
            if seq.encode().translate(None, b"ACGT"):
                items = items + [(r + 1, r) for q, r in a.get_aligned_pairs(matches_only=True) if seq[q] not in "ACGT"]
            for s, e in items:
                bs[k] = s
                be[k] = e
                bf[k] = fwd
                k += 1
                if k == chunk:
                    _add_blocks(diff_f, diff_r, bs, be, bf, k)
                    k = 0
        _add_blocks(diff_f, diff_r, bs, be, bf, k)
    out = {}
    for strd, d in ((True, diff_f), (False, diff_r)):
        # in place, no temporaries: d[p + 1] = depth at p, then covered at p, then d[i] = covered positions before i
        np.add.accumulate(d, out=d)
        np.greater(d, 0, out=d)
        np.add.accumulate(d, out=d)
        out[strd] = d[:L + 1]
    return out


def body_pieces(expressed_tx, covlen, include_introns, strd):
    """transcripts.start_pieces, but each transcript's removeOverlaps gets only the protected ends that can touch its span.

    removeOverlaps is a set difference over sorted flattened lists, so the slice gives the same pieces; it turns the
    per-transcript sort and scan of every end on the strand (quadratic in transcripts) into a short list.
    """
    _, RNA, SNR = polyafilter_modules()
    each_start, all_ends = [], []
    for s, e, exons in expressed_tx:
        if include_introns:
            if e - s <= covlen:
                ends, starts = [(s, e)], []
            else:
                ends = RNA.getTransEnding(exons, covlen, strd)
                ends.append((e - covlen, e) if strd else (s, s + covlen))
                ends = SNR.flattenIntervals(ends)
                starts = SNR.removeOverlaps([(s, e)], ends)
        else:
            starts, ends = RNA.getTransStartEnding(exons, covlen, strd)
        all_ends.extend(ends)
        if starts != []:
            each_start.append(tuple(starts))
    all_ends = SNR.flattenIntervals(all_ends)
    end_s = [a for a, _ in all_ends]
    end_e = [b for _, b in all_ends]
    flat = []
    for st in each_start:
        i0 = bisect_right(end_e, min(a for a, _ in st))
        i1 = bisect_left(end_s, max(b for _, b in st))
        flat.extend(SNR.removeOverlaps(list(st), all_ends[i0:i1]))
    flat = SNR.flattenIntervals(flat)
    return np.array([a for a, _ in flat], np.int64), np.array([b for _, b in flat], np.int64)


def reference_pieces(seq, tx_by_strand, prefix, settings):
    """{(tag, strd): (ps, pe)}, the pieces flag_reference intersects blocks with.

    Same caches as flag_reference (bodies by strand, COVLEN and mode; A-stretch scans by strand and MISM). A scan is
    dropped once its settings are done, stretches are pre-filtered by MINSNRLEN (snr_pieces filters again, a no-op),
    and the flattened windows are shared by both modes.
    """
    ref_len = len(seq)
    modes = {s.introns for s in settings}
    ex = {(strd, i): expressed(tx_by_strand[strd], prefix[strd], i) for strd in (True, False) for i in modes}
    del prefix  # scan_reference passes the prefix inline, so this frees it before the A-stretch scans
    out = {}
    for strd in (True, False):
        body = {k: body_pieces(ex[(strd, k[1])], k[0], k[1], strd) for k in sorted({(s.covlen, s.introns) for s in settings})}
        for st in settings:
            if st.min_len is None:
                out[(setting_tag(st), strd)] = body[(st.covlen, st.introns)]
        for mism in sorted({s.mism for s in settings if s.min_len is not None}):
            first, last = scan_snrs(seq, strd, mism)
            group = [s for s in settings if s.min_len is not None and s.mism == mism]
            for min_len in sorted({s.min_len for s in group}):
                keep = (last - first) >= min_len
                first, last = first[keep], last[keep]
                for covlen in sorted({s.covlen for s in group if s.min_len == min_len}):
                    ws, we = flatten(*snr_pieces(first, last, min_len, covlen, strd, ref_len))
                    for st in group:
                        if (st.covlen, st.min_len) == (covlen, min_len):
                            out[(setting_tag(st), strd)] = intersect(*body[(covlen, st.introns)], ws, we)
            del first, last
    return out


def _pack(pairs):
    """Concatenate per-tag (ps, pe) into flat int64 arrays with offsets[t]:offsets[t + 1] per tag."""
    offs = np.zeros(len(pairs) + 1, np.int64)
    offs[1:] = np.cumsum([s.size for s, _ in pairs])
    ps = np.concatenate([np.asarray(s, np.int64) for s, _ in pairs] + [np.empty(0, np.int64)])
    pe = np.concatenate([np.asarray(e, np.int64) for _, e in pairs] + [np.empty(0, np.int64)])
    return ps, pe, offs


@njit(cache=True)
def _scan_hits(bs, be, br, ps, pe, offs, mark, prim, re_code, cell, trow, reg, cel, tru):
    """Blocks of one strand sorted by start against each tag's sorted disjoint pieces; same test as intervals.blocks_hit.

    The first record hit for tag t is marked, and if primary counted once in reg[t, re], cel[t, cell] and tru[row, t].
    """
    nb = bs.size
    if nb == 0:
        return
    for t in range(offs.size - 1):
        lo, hi = offs[t], offs[t + 1]
        if lo == hi:
            continue
        j = lo + np.searchsorted(pe[lo:hi], bs[0], side="right")
        for k in range(nb):
            s = bs[k]
            while j < hi and pe[j] <= s:
                j += 1
            if j == hi:
                break
            if ps[j] < be[k]:
                r = br[k]
                if mark[r] != t:
                    mark[r] = t
                    if prim[r]:
                        reg[t, re_code[r]] += 1
                        if cell[r] >= 0:
                            cel[t, cell[r]] += 1
                        if trow[r] >= 0:
                            tru[trow[r], t] = True


class _Chunk:
    """Fixed numpy buffers for one chunk of records and their blocks."""

    def __init__(self, chunk):
        self.flag = np.empty(chunk, np.int64)
        self.re = np.empty(chunk, np.int64)
        self.cell = np.empty(chunk, np.int64)
        self.trow = np.empty(chunk, np.int64)
        self._blocks(4 * chunk + 64)
        self.nr = self.nb = 0
        self.t_qname, self.t_pos, self.t_flag = [], [], []

    def _blocks(self, cap):
        self.bs, self.be, self.br = np.empty(cap, np.int64), np.empty(cap, np.int64), np.empty(cap, np.int64)

    def fits(self, n_blocks):
        if self.nb + n_blocks <= self.bs.size:
            return True
        if self.nb == 0:
            self._blocks(2 * n_blocks)
            return True
        return False

    def reset(self):
        self.nr = self.nb = 0
        self.t_qname, self.t_pos, self.t_flag = [], [], []


def scan_reference(bam_path, fasta_path, ref, transcripts, settings, truth_qnames, cells=None, chunk=1_000_000):
    """(truth_df, region_df, cell_df) for one reference; region_df.attrs["records"] is the number of records fetched.

    truth_df: primary records whose qname is in truth_qnames, fetch order: qname, ref, pos, flag + one bool per tag.
    region_df: tag, re, removed, total over primary records. cell_df: index cb (sorted cells), total then one
    uint32 per tag over primary records with a called-cell CB; None when cells is None.
    """
    with pysam.FastaFile(str(fasta_path)) as fa:
        seq = fa.fetch(ref)
    tx_by_strand = {strd: transcripts.get((ref, strd), []) for strd in (True, False)}
    pieces = reference_pieces(seq, tx_by_strand, coverage_prefix_chunked(bam_path, ref), settings)
    del seq
    tags = [setting_tag(s) for s in settings]
    packed = {strd: _pack([pieces.pop((t, strd)) for t in tags]) for strd in (True, False)}
    cells_sorted = [] if cells is None else sorted(cells)
    cell_idx = {c: i for i, c in enumerate(cells_sorted)}
    n_tags = len(tags)
    re_idx = {c: i for i, c in enumerate(RE_CODES)}
    reg = np.zeros((n_tags, len(RE_CODES)), np.int64)
    reg_total = np.zeros(len(RE_CODES), np.int64)
    cel = np.zeros((n_tags, len(cells_sorted)), np.uint32)
    cel_total = np.zeros(len(cells_sorted), np.int64)
    truth_parts = []
    n_records = 0

    def flush(c):
        nonlocal reg, reg_total
        if len(re_idx) > reg.shape[1]:
            grow = len(re_idx) - reg.shape[1]
            reg = np.pad(reg, ((0, 0), (0, grow)))
            reg_total = np.pad(reg_total, (0, grow))
        nr, nb = c.nr, c.nb
        flags, re_code, cell, trow = c.flag[:nr], c.re[:nr], c.cell[:nr], c.trow[:nr]
        prim = (flags & SECONDARY_OR_SUPPLEMENTARY) == 0
        reg_total += np.bincount(re_code[prim], minlength=reg.shape[1])
        in_cell = prim & (cell >= 0)
        cel_total[:] += np.bincount(cell[in_cell], minlength=len(cells_sorted))
        tru = np.zeros((len(c.t_qname), n_tags), np.bool_)
        mark = np.full(nr, -1, np.int64)
        bs, be, br = c.bs[:nb], c.be[:nb], c.br[:nb]
        bfwd = (flags[br] & 16) == 0
        for strd in (True, False):
            sel = np.flatnonzero(bfwd == strd)
            o = sel[np.argsort(bs[sel], kind="stable")]
            _scan_hits(bs[o], be[o], br[o], *packed[strd], mark, prim, re_code, cell, trow, reg, cel, tru)
        head = pd.DataFrame({
            "qname": pd.Series(c.t_qname, dtype="str"),
            "ref": pd.Series([ref] * len(c.t_qname), dtype="str"),
            "pos": np.asarray(c.t_pos, np.int64),
            "flag": np.asarray(c.t_flag, np.int64),
        })
        truth_parts.append(pd.concat([head, pd.DataFrame(tru, columns=tags)], axis=1))
        c.reset()

    c = _Chunk(chunk)
    with pysam.AlignmentFile(str(bam_path)) as bam:
        for a in bam.fetch(ref):
            n_records += 1
            blocks = a.get_blocks()
            if c.nr == chunk or not c.fits(len(blocks)):
                flush(c)
                c.fits(len(blocks))
            i = c.nr
            flag = a.flag
            c.flag[i] = flag
            try:
                re_ = a.get_tag("RE")
            except KeyError:
                re_ = ""
            code = re_idx.get(re_)
            if code is None:
                code = re_idx[re_] = len(re_idx)
            c.re[i] = code
            if cell_idx:
                try:
                    c.cell[i] = cell_idx.get(a.get_tag("CB"), -1)
                except KeyError:
                    c.cell[i] = -1
            else:
                c.cell[i] = -1
            if not flag & SECONDARY_OR_SUPPLEMENTARY and a.query_name in truth_qnames:
                c.trow[i] = len(c.t_qname)
                c.t_qname.append(a.query_name)
                c.t_pos.append(a.reference_start)
                c.t_flag.append(flag)
            else:
                c.trow[i] = -1
            nb = c.nb
            for s, e in blocks:
                c.bs[nb] = s
                c.be[nb] = e
                c.br[nb] = i
                nb += 1
            c.nb = nb
            c.nr = i + 1
    flush(c)

    truth_df = pd.concat(truth_parts, ignore_index=True)
    codes = sorted(re_idx, key=re_idx.get)
    seen = [k for k in range(len(codes)) if reg_total[k] > 0]
    region_df = pd.DataFrame({
        "tag": np.repeat(np.array(tags, dtype=object), len(seen)),
        "re": [codes[k] for _ in tags for k in seen],
        "removed": reg[:, seen].reshape(-1),
        "total": np.tile(reg_total[seen], n_tags),
    })
    region_df["tag"] = region_df["tag"].astype("str")
    region_df["re"] = region_df["re"].astype("str")
    region_df.attrs["records"] = n_records
    cell_df = None
    if cells is not None:
        cell_df = pd.DataFrame(cel.T, index=pd.Index(cells_sorted, name="cb", dtype="str"), columns=tags)
        cell_df.insert(0, "total", cel_total.astype(np.uint32))
    return truth_df, region_df, cell_df


def scan_to_parts(bam_path, fasta_path, ref, transcripts, settings, truth_qnames, cells, part_dir, chunk=1_000_000,
                  meta=None):
    """scan_reference written to part_dir as <ref>.truth/.region/.cells.parquet, then <ref>.json (records, primary, seconds,
    plus the entries of meta, which name the inputs; see stale_parts).

    The json is written last, so a reference whose json exists is complete: it is not scanned again and its stored
    stats are returned (resume after a crash). No cells part when cells is None.
    """
    part_dir = Path(part_dir)
    done = part_dir / f"{ref}.json"
    if done.exists():
        return json.loads(done.read_text())
    t0 = time.monotonic()
    truth, region, cell = scan_reference(bam_path, fasta_path, ref, transcripts, settings, truth_qnames, cells, chunk)
    part_dir.mkdir(parents=True, exist_ok=True)
    truth.to_parquet(part_dir / f"{ref}.truth.parquet", index=False)
    region.to_parquet(part_dir / f"{ref}.region.parquet", index=False)
    if cell is not None:
        cell.to_parquet(part_dir / f"{ref}.cells.parquet")
    stats = {"ref": ref, "records": region.attrs["records"], "primary": int(region.drop_duplicates("re").total.sum()),
             "seconds": round(time.monotonic() - t0, 1)} | (meta or {})
    done.write_text(json.dumps(stats))
    return stats


def stale_parts(part_dir, refs, meta):
    """refs whose finished part (<ref>.json) was written with other inputs than meta (or without them), in refs order."""
    stale = []
    for ref in refs:
        done = Path(part_dir) / f"{ref}.json"
        if done.exists():
            stored = json.loads(done.read_text())
            if any(stored.get(k) != v for k, v in meta.items()):
                stale.append(ref)
    return stale


def merge_regions(parts, tags):
    """Region rows (tag, re, removed, total) of several references summed per tag and RE, in tags order then RE."""
    region = pd.concat(parts, ignore_index=True)
    region = region.groupby(["tag", "re"], sort=False, as_index=False)[["removed", "total"]].sum()
    rank = region.tag.map({t: i for i, t in enumerate(tags)})
    return region.iloc[np.lexsort((region.re.to_numpy(), rank.to_numpy()))].reset_index(drop=True)


def merge_parts(part_dir, refs, tags, out_dir):
    """Merge the parts of refs into out_dir and return the summary dict (per-ref stats, totals, removed primary per tag).

    truth_flags.parquet: truth rows in refs order, streamed one reference at a time. region_counts.parquet: summed
    over refs, rows in tags order then RE. cell_counts.parquet: summed over refs, only when every ref has a cells part.
    """
    part_dir, out_dir = Path(part_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("qname", pa.string()), ("ref", pa.string()), ("pos", pa.int64()), ("flag", pa.int64())]
                       + [(t, pa.bool_()) for t in tags])
    with pq.ParquetWriter(out_dir / "truth_flags.parquet", schema) as w:
        for ref in refs:
            w.write_table(pq.read_table(part_dir / f"{ref}.truth.parquet").cast(schema))
    region = merge_regions([pd.read_parquet(part_dir / f"{ref}.region.parquet") for ref in refs], tags)
    region.to_parquet(out_dir / "region_counts.parquet", index=False)
    cell_paths = [part_dir / f"{ref}.cells.parquet" for ref in refs]
    if all(p.exists() for p in cell_paths):
        cell = pd.read_parquet(cell_paths[0])
        for p in cell_paths[1:]:
            cell += pd.read_parquet(p)
        cell.to_parquet(out_dir / "cell_counts.parquet")
    per_ref = {}
    for ref in refs:
        st = json.loads((part_dir / f"{ref}.json").read_text())
        per_ref[ref] = {k: st[k] for k in ("records", "primary", "seconds")}
    removed = region.groupby("tag").removed.sum()
    return {
        "per_ref": per_ref,
        "records": sum(v["records"] for v in per_ref.values()),
        "primary": sum(v["primary"] for v in per_ref.values()),
        "removed": {t: int(removed.get(t, 0)) for t in tags},
    }
```

- [ ] **Step 4: Create `scripts/13_grid_flags.py`**

```python
"""Stage 13: polyAfilter removal at every grid setting, streamed per reference (pafbench.stream).

Writes OUT/LIB/grid/truth_flags.parquet (primary records of truth reads), region_counts.parquet, cell_counts.parquet
and summary.json. Per-reference parts go to grid/parts/; a reference whose json exists is not scanned again. Each part
json names its inputs (truth file path and row count, number of settings); when they differ from this run's, the stage
stops and asks for parts/ to be trashed (after a stage 12 rerun, for example).

Measured on 1417 chr1 (6.96M records, 396 settings, one worker, 2026-10-08): 202 s wall; worker peak RSS 2.71 GiB, of
which 0.6 GiB is shared with the parent through fork (interpreter, GTF transcripts, truth qnames); memwatch tree peak
3.31 GiB. The peak is set by chromosome length, not records: two int32 coverage arrays plus the sequence, about 9 bytes
per bp (chunk buffers are fixed). Measured on 1179 chr1 (49.9M records, 396 settings, one worker, stage 13 review):
740 s wall, worker VmHWM 2.75 GiB, memwatch tree peak 3.36 GiB. Seven 1179 references at once (96.6M records, seven
workers) ran each job 1.2 to 1.35 times slower than alone (concurrent BAM reads). Whole 1179 BAM (574M records, 134
references) at 8 workers: about 22 to 25 min wall, about 14 GiB physical (about 18 GiB as a memwatch sum, which counts
the shared pages once per worker).
"""
import argparse
import gzip
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pysam

from pafbench.config import FASTA, GTF, LIBRARIES, OUT
from pafbench.flags import setting_tag
from pafbench.grid import all_settings
from pafbench.stream import merge_parts, scan_to_parts, stale_parts
from pafbench.transcripts import load_transcripts

_TX = _TRUTH = _CELLS = None


def _init(tx, truth, cells):
    # loaded once in the parent and handed over by fork (no pickling), not reloaded for each reference
    global _TX, _TRUTH, _CELLS
    _TX, _TRUTH, _CELLS = tx, truth, cells


def _work(job):
    bam, ref, settings, part_dir, chunk, meta = job
    return scan_to_parts(bam, FASTA, ref, _TX, settings, _TRUTH, _CELLS, part_dir, chunk, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--refs", nargs="+")
    ap.add_argument("--out", type=Path, help="grid directory (default OUT/LIB/grid)")
    ap.add_argument("--truth", type=Path, help="parquet with a qname column (default OUT/LIB/truth/truth_reads.parquet)")
    ap.add_argument("--chunk", type=int, default=1_000_000)
    args = ap.parse_args()
    t0 = time.monotonic()
    bam = LIBRARIES[args.library]
    out_dir = args.out or OUT / args.library / "grid"
    part_dir = out_dir / "parts"
    truth = args.truth or OUT / args.library / "truth" / "truth_reads.parquet"
    cells = bam.parent / "filtered_feature_bc_matrix" / "barcodes.tsv.gz"
    settings = all_settings()
    with pysam.AlignmentFile(str(bam)) as fh:
        stats = [s for s in fh.get_index_statistics() if s.mapped > 0 and (args.refs is None or s.contig in args.refs)]
    refs = [s.contig for s in stats]
    if args.refs:
        absent = [r for r in args.refs if r not in refs]
        if absent:
            print(f"--refs not in the BAM index or without mapped reads, ignored: {', '.join(absent)}", file=sys.stderr)
    meta = {"truth": str(truth.resolve()), "truth_rows": pq.ParquetFile(truth).metadata.num_rows, "settings": len(settings)}
    stale = stale_parts(part_dir, refs, meta)
    if stale:
        sys.exit(f"{part_dir}: the parts of {', '.join(stale)} were made from another truth file or settings list "
                 f"(this run: {json.dumps(meta)}); trash them with `gio trash {part_dir}` and rerun")
    todo = [s for s in sorted(stats, key=lambda s: -s.mapped) if not (part_dir / f"{s.contig}.json").exists()]
    if todo:
        tx = load_transcripts(GTF)
        truth_qnames = set(pd.read_parquet(truth, columns=["qname"]).qname)
        with gzip.open(cells, "rt") as fh:
            called = [line.strip() for line in fh]
        jobs = [(bam, s.contig, settings, part_dir, args.chunk, meta) for s in todo]
        # fork hands the parent's transcripts, truth set and cells to the workers without pickling (see _init)
        with get_context("fork").Pool(args.workers, initializer=_init, initargs=(tx, truth_qnames, called),
                                      maxtasksperchild=1) as pool:
            pool.map(_work, jobs, chunksize=1)
    summary = merge_parts(part_dir, refs, [setting_tag(s) for s in settings], out_dir)
    summary = {"library": args.library, "settings": len(settings), "workers": args.workers, "scanned": len(todo),
               "seconds": round(time.monotonic() - t0, 1), **summary}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("library", "settings", "scanned", "records", "primary", "seconds")} | {"refs": len(refs)}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `113 passed, 2 warnings`.

- [ ] **Step 6: Stage 13 on 1417 and the match with milestone 1**

Run: `.venv/bin/python -m pafbench.memwatch /data/element_runs/polyafilter_bench/1417/runs/13_grid_flags.mem.json -- .venv/bin/python scripts/13_grid_flags.py 1417 --workers 8`
Then:
```bash
.venv/bin/python - <<'EOF'
import json
s = json.load(open("/data/element_runs/polyafilter_bench/1417/grid/summary.json"))
m = json.load(open("/data/element_runs/polyafilter_bench/1417/exact_match.json"))
for tag in ("c300_l10_m1_i", "c375_l3_m2_i", "c300_l10_m1_e"):
    print(tag, s["removed"][tag], m[tag]["n_reimpl"])
EOF
```
Expected: each line shows two equal numbers, 17,068,514, 46,808,565 and 3,729,928 (1417 has no secondary or supplementary records, so primary counts equal milestone 1's record counts), and `summary.json` reports 83,140,741 primary records (the mapped records in the BAM index).

- [ ] **Step 7: Stage 13 on 1419 and 1179**

Run: `for L in 1419 1179; do .venv/bin/python -m pafbench.memwatch /data/element_runs/polyafilter_bench/$L/runs/13_grid_flags.mem.json -- .venv/bin/python scripts/13_grid_flags.py $L --workers 8; done`
Expected: 1179 takes about 22 to 25 min with about 14 GiB physical (the memwatch tree peak counts shared pages once per worker, so it reads higher). If a run dies, rerun the same command: finished references are skipped. After a stage 12 rerun, `gio trash` the `grid/parts/` directory first (the engine refuses parts built from another truth file).

- [ ] **Step 8: Commit**

```bash
git add tests/test_stream.py pafbench/stream.py scripts/13_grid_flags.py
git -c user.name="Simone Picelli" -c user.email="simone.picelli@iob.ch" commit -m "Stage 13: streaming grid engine (coverage prefix in chunks, per-reference pieces for all settings, chunked hit kernel, per-reference parts, merge and restart guard)"
```

---

### Task 4: Long-read priming sites, BD67 and BD70 (stage 16)

**Files:**
- Create: `tests/test_long_reads.py`
- Modify: `pafbench/sites.py`
- Modify: `pafbench/bd67.py`
- Create: `scripts/16_long_read_sites.py`

**Interfaces:**
- Consumes: milestone 1 `bd67.sense_3p_clip`, `bd67.a_run_site` (the tolerance-0 reference), `sites.priming_site`, `sites.sense_context`, `sites.has_hexamer`, `genes.load_gene_segments`, `genes.genes_at`; Task 0 `regions`, `labels`, `config.LABEL_CUTOFFS`; Task 2 `OUT/1179/truth/molecules_labeled.parquet`.
- Produces:
  - `pafbench.bd67.sense_3p_clip(a)`
  - `pafbench.bd67.leading_a(s)`
  - `pafbench.bd67.a_run_site(a)`
  - `pafbench.bd67.inward_pairs(a)`
  - `pafbench.bd67.terminal_a_run(a, tolerance=1)`
  - `pafbench.bd67.accept_long_read(aligned_a, clip_a, after_len, min_run=10, min_after=30)`
  - `pafbench.bd67.read_outcome(aligned_a, clip_a, after_len, min_run=10, min_after=30)`
  - `pafbench.bd67.scan_long_reads(bam_path, ref, min_mapq=20)`
  - `pafbench.bd67.count_duplicates(digests)`
  - `pafbench.bd67.annotate_sites(sites, ref, seq, region_segments, gene_segments, cutoffs=LABEL_CUTOFFS)`
  - `pafbench.bd67.hexamer_summary(sites, cutoffs)`
  - `pafbench.bd67.nearest_labels(mol, sites, max_dist=10)`
  - `pafbench.bd67.compare_labels(mol, sites, max_dist=10)`
  - `pafbench.sites.context_from_seq(seq, site, is_reverse, down=20, up=40)`
- Produces (data): `OUT/<BD>/long_read_sites.parquet` (`ref, strand, site, n_reads, n_accepted, a20, label, a_band, hexamer, region, gene`), `OUT/<BD>/long_read_acceptance.json` (read outcomes, per-site hexamer rates and ratio, `passes`, the same at 14/8, vs_1179 agreement), `OUT/<BD>/vs_1179.parquet` (`label_1179, label_long, n`).

Spec 4.3 rule, decided 2026-10-08 after the milestone 1 clip test failed genome-wide: the aligner absorbs genome-encoded A's into the alignment, so the run is counted across the aligned part and the clip, with one non-A allowed inside it, and the bead sequence (at least 30 nt) must follow. With tolerance 0 the site equals milestone 1's `a_run_site` (tested). Build-time numbers on chr21 + chr22: 11.1% (BD67) and 7.7% (BD70) of reads dropped, per-site hexamer ratio 5.12 and 4.82 (per read 3.34 and 2.90).

- [ ] **Step 1: Write the failing tests in `tests/test_long_reads.py`**

```python
import random

import numpy as np
import pandas as pd
import pysam
import pytest

from pafbench.bd67 import (accept_long_read, annotate_sites, a_run_site, compare_labels, count_duplicates,
                           hexamer_summary, inward_pairs, read_outcome, scan_long_reads, sense_3p_clip,
                           terminal_a_run)
from pafbench.genes import load_gene_segments
from pafbench.regions import load_region_segments
from pafbench.sites import COMP, context_from_seq, has_hexamer, priming_site, sense_context

HDR = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 100000}]})


def _read(pos, cigar, seq, rev, name="x", extra=0, mapq=60):
    a = pysam.AlignedSegment(HDR)
    a.query_name, a.reference_name, a.reference_start = name, "chr1", pos
    a.cigarstring, a.flag, a.query_sequence, a.mapping_quality = cigar, (16 if rev else 0) | extra, seq, mapq
    return a


# name: (record args, expected at tolerance 1, expected at tolerance 0); expected = (site, aligned_a, clip_a, after_len).
# Forward reads are in sense already. Reverse reads hold the reverse complement of the sense sequence, so the
# 3' clip is the leading soft clip and the aligned base next to it (reference_start) is the sense S[b-1].
CASES = {
    # pure clip run; also the aligned non-A at the run start (G, with C further inward) is not taken
    "pure_clip_fwd": ((100, "10M45S", "C" * 9 + "G" + "A" * 12 + "G" * 33, False), (110, 0, 12, 33), (110, 0, 12, 33)),
    # sense clip A12 G33 -> leading clip C33 T12; sense aligned C9 G -> ref-orientation C G9 (genomic 200 = sense G)
    "pure_clip_rev": ((200, "45S10M", "C" * 33 + "T" * 12 + "C" + "G" * 9, True), (200, 0, 12, 33), (200, 0, 12, 33)),
    # aligned A's 110..114 absorbed plus 8 clip A's: run starts at 110
    "aligned_plus_clip_fwd": ((100, "15M40S", "C" * 10 + "A" * 5 + "A" * 8 + "T" * 32, False), (110, 5, 8, 32), (110, 5, 8, 32)),
    # genomic 200..204 are T (sense A), 205..214 G (sense C); run's first base r = 204 -> site 205
    "aligned_plus_clip_rev": ((200, "40S15M", "A" * 32 + "T" * 8 + "T" * 5 + "G" * 10, True), (205, 5, 8, 32), (205, 5, 8, 32)),
    # clip A5 G A6 C38: the G is taken at tolerance 1 (11 A's), the run stops at it at tolerance 0
    "clip_mismatch_fwd": ((100, "10M50S", "C" * 10 + "AAAAAGAAAAAA" + "C" * 38, False), (110, 0, 11, 38), (110, 0, 5, 45)),
    "clip_mismatch_rev": ((300, "50S10M", "G" * 38 + "TTTTTTCTTTTT" + "G" * 10, True), (300, 0, 11, 38), (300, 0, 5, 45)),
    # aligned C10 AAA G AAAAAA (110..119) + clip A4 C36: tolerance 1 takes G at 113 and the A's at 110..112
    "aligned_mismatch_fwd": ((100, "20M40S", "C" * 10 + "AAAGAAAAAA" + "AAAA" + "C" * 36, False), (110, 9, 4, 36), (114, 6, 4, 36)),
    # genomic 300..305 sense A, 306 sense G, 307..309 sense A, 310..319 sense C
    "aligned_mismatch_rev": ((300, "40S20M", "G" * 36 + "TTTT" + "TTTTTTCTTT" + "G" * 10, True), (310, 9, 4, 36), (306, 6, 4, 36)),
    # aligned C7 G AAAA: the G at 107 has C further inward, so it would start the run and is not taken
    "nonA_run_start_fwd": ((100, "12M40S", "C" * 7 + "G" + "AAAA" + "A" * 8 + "C" * 32, False), (108, 4, 8, 32), (108, 4, 8, 32)),
    # first clip base G after aligned A's: taken at tolerance 1 (aligned base S[b-1] is A)
    "first_clip_nonA_after_A": ((100, "13M41S", "C" * 10 + "AAA" + "G" + "A" * 10 + "C" * 30, False), (110, 3, 10, 30), (110, 3, 0, 41)),
    # first clip base G after aligned C: not taken, the run is empty and the clip is all after-run sequence
    "first_clip_nonA_after_nonA_fwd": ((100, "10M43S", "C" * 10 + "G" + "A" * 12 + "C" * 30, False), (110, 0, 0, 43), (110, 0, 0, 43)),
    "first_clip_nonA_after_nonA_rev": ((200, "43S10M", "G" * 30 + "T" * 12 + "C" + "G" * 10, True), (200, 0, 0, 43), (200, 0, 0, 43)),
    # clip A4 G A4 G A4: the second G finds the budget spent
    "second_nonA_clip": ((100, "10M44S", "C" * 10 + "AAAAGAAAAGAAAA" + "C" * 30, False), (110, 0, 8, 35), (110, 0, 4, 40)),
    # aligned C6 AAGAAA (109..111 A, 108 G) + clip AAAGAAAA C30: the clip G spends the budget, the aligned G stops the run
    "budget_shared": ((100, "12M38S", "C" * 6 + "AAGAAA" + "AAAGAAAA" + "C" * 30, False), (109, 3, 7, 30), (109, 3, 3, 35)),
    # no clip: left growth may still take a non-A (the run has aligned A's on its right)
    "no_clip_aligned_mismatch": ((100, "10M", "CCAAGAAAAA", False), (102, 7, 0, 0), (105, 5, 0, 0)),
    "no_clip_no_a": ((100, "10M", "CCCCCCCCCG", False), (110, 0, 0, 0), (110, 0, 0, 0)),
    # inserted GG inside the aligned run is not an aligned base: run 110..116 (7 A's)
    "insertion_fwd": ((100, "13M2I4M40S", "C" * 10 + "AAA" + "GG" + "AAAA" + "A" * 8 + "T" * 32, False), (110, 7, 8, 32), (110, 7, 8, 32)),
    # sense A at genomic 300..303 and 304..306 (CC inserted between), 307 deleted, 308..317 sense C: r = 306 -> site 307
    "insertion_deletion_rev": ((300, "40S4M2I3M1D10M", "A" * 32 + "T" * 8 + "TTTT" + "CC" + "TTT" + "G" * 10, True), (307, 7, 8, 32), (307, 7, 8, 32)),
    # 5 + 4 = 9 run A's with 30 after: one short of the run threshold
    "run_9": ((100, "15M34S", "C" * 10 + "A" * 5 + "A" * 4 + "C" * 30, False), (110, 5, 4, 30), (110, 5, 4, 30)),
    # 5 + 5 = 10 run A's with 29 after: one short of the clip threshold
    "after_29": ((100, "15M34S", "C" * 10 + "A" * 5 + "A" * 5 + "C" * 29, False), (110, 5, 5, 29), (110, 5, 5, 29)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_terminal_a_run_cases(name):
    args, exp1, exp0 = CASES[name]
    a = _read(*args)
    assert terminal_a_run(a) == exp1
    assert terminal_a_run(a, tolerance=0) == exp0
    assert (exp0[0], exp0[1]) == a_run_site(a)


def test_acceptance_and_outcome():
    assert not accept_long_read(4, 5, 30) and accept_long_read(4, 6, 30)
    assert not accept_long_read(10, 0, 29) and accept_long_read(0, 10, 30)
    outcome = lambda name: read_outcome(*terminal_a_run(_read(*CASES[name][0]))[1:])
    assert outcome("budget_shared") == "accepted"            # 3 + 7 = 10 run A's, 30 after
    assert outcome("run_9") == "short_run"
    assert outcome("after_29") == "short_after"
    assert outcome("no_clip_aligned_mismatch") == "both"     # 7 run A's, 0 after
    assert outcome("first_clip_nonA_after_nonA_fwd") == "short_run"


def _literal_run(a, tol):
    """The brief's rule written out on the full sense string S from get_aligned_pairs."""
    seq, pairs = a.query_sequence, a.get_aligned_pairs(matches_only=True)
    al = [(seq[q].translate(COMP), r) for q, r in reversed(pairs)] if a.is_reverse else [(seq[q], r) for q, r in pairs]
    c = sense_3p_clip(a)
    s, b = "".join(x for x, _ in al) + c, len(al)
    budget, j = tol, b
    while j < len(s):
        if s[j] == "A":
            j += 1
        elif budget and j + 1 < len(s) and s[j + 1] == "A" and (j > b or s[b - 1] == "A"):
            budget, j = budget - 1, j + 1
        else:
            break
    i = b
    while i > 0:
        if s[i - 1] == "A":
            i -= 1
        elif budget and i >= 2 and s[i - 2] == "A" and (i < b or j > b):
            budget, i = budget - 1, i - 1
        else:
            break
    site = priming_site(a) if i == b else (al[i][1] + 1 if a.is_reverse else al[i][1])
    return site, s[i:b].count("A"), s[b:j].count("A"), len(c) - (j - b)


def _random_reads(n, seed=3):
    rng = random.Random(seed)
    for k in range(n):
        rev = rng.random() < 0.5
        ops = [(5, rng.randint(1, 5))] if rng.random() < 0.1 else []
        if rng.random() < 0.7:
            ops.append((4, rng.randint(1, 60)))
        ops.append((rng.choice([0, 7, 8]), rng.randint(1, 30)))
        for _ in range(rng.randint(0, 4)):
            ops += [(rng.choice([1, 2, 3]), rng.randint(1, 4)), (rng.choice([0, 7, 8]), rng.randint(1, 30))]
        if rng.random() < 0.7:
            ops.append((4, rng.randint(1, 60)))
        if rng.random() < 0.1:
            ops.append((5, rng.randint(1, 5)))
        qlen = sum(n for op, n in ops if op in (0, 1, 4, 7, 8))
        a = pysam.AlignedSegment(HDR)
        a.query_name, a.reference_name, a.reference_start = f"q{k}", "chr1", 1000
        a.cigartuples, a.flag = ops, 16 if rev else 0
        a.query_sequence = "".join(rng.choice("TTTTTGCAN" if rev else "AAAAACGTN") for _ in range(qlen))
        yield a


def _expected_inward(a):
    pairs = a.get_aligned_pairs(matches_only=True)
    return pairs if a.is_reverse else pairs[::-1]


def test_random_reads_match_literal_rule_and_a_run_site():
    for a in _random_reads(3000):
        assert list(inward_pairs(a)) == _expected_inward(a)
        assert terminal_a_run(a, 0)[:2] == a_run_site(a)
        for tol in (0, 1, 2):
            assert terminal_a_run(a, tol) == _literal_run(a, tol)


def test_synthetic_bam_matches_a_run_site(synth):
    n = 0
    with pysam.AlignmentFile(str(synth.bam)) as bam:
        for a in bam.fetch():
            if a.query_sequence is None:
                continue
            assert list(inward_pairs(a)) == _expected_inward(a)
            assert terminal_a_run(a, 0)[:2] == a_run_site(a)
            assert terminal_a_run(a) == _literal_run(a, 1)
            n += 1
    assert n == 144     # 149 synthetic records (75 + 5 + 41 + 9 + 5 + 4 + 4 + 2 + 4) minus 5 secondaries without a sequence


def test_context_from_seq_matches_sense_context(synth):
    with pysam.FastaFile(str(synth.fasta)) as fa:
        for ref in fa.references:
            seq, L = fa.fetch(ref).upper(), fa.get_reference_length(ref)
            for site in (0, 1, 19, 20, 21, 39, 40, 41, 1299, 1500, L - 41, L - 40, L - 21, L - 20, L - 1, L):
                for rev in (False, True):
                    assert context_from_seq(seq, site, rev) == sense_context(fa, ref, site, rev)
        seq = fa.fetch("chrA").upper()
        assert context_from_seq(seq, 1299, False)[0].startswith("A" * 12)      # planted A12 at 1299
        assert context_from_seq(seq, 4509, True)[0].startswith("A" * 10)       # planted T10 at 4499..4508


def _bam(tmp_path, reads):
    unsorted, path = tmp_path / "u.bam", tmp_path / "r.bam"
    with pysam.AlignmentFile(str(unsorted), "wb", header=HDR) as out:
        for a in reads:
            out.write(a)
    pysam.sort("-o", str(path), str(unsorted))
    pysam.index(str(path))
    return path


def test_scan_long_reads(tmp_path):
    rec = lambda case, **kw: _read(*CASES[case][0][:4], **kw)
    reads = [
        rec("aligned_plus_clip_fwd", name="f1"),                    # +110 accepted
        rec("aligned_plus_clip_fwd", name="f2", mapq=20),           # +110 accepted, MAPQ 20 is kept
        rec("run_9", name="f3"),                                    # +110 short_run
        rec("after_29", name="f4"),                                 # +110 short_after
        rec("no_clip_aligned_mismatch", name="f5"),                 # +102 both
        rec("aligned_plus_clip_rev", name="r1"),                    # -205 accepted
        rec("aligned_plus_clip_fwd", name="s1", extra=256),         # secondary: skipped
        rec("aligned_plus_clip_rev", name="s2", extra=2048),        # supplementary: skipped
        rec("aligned_plus_clip_fwd", name="lq", mapq=19),           # MAPQ 19: skipped
        rec("first_clip_nonA_after_A", name="f1"),                  # +110 accepted (3 + 10, 30 after), repeated qname
    ]
    sites, counts, digests = scan_long_reads(_bam(tmp_path, reads), "chr1")
    assert sites.strand.tolist() == ["+", "+", "-"] and sites.site.tolist() == [102, 110, 205]
    assert sites.n_reads.tolist() == [1, 5, 1] and sites.n_accepted.tolist() == [0, 3, 1]
    assert counts == {"records": 10, "secondary_supplementary": 2, "low_mapq": 1, "used": 7,
                      "accepted": 4, "short_run": 1, "short_after": 1, "both": 1}
    assert len(digests) == 7 and count_duplicates(digests) == 1


def test_annotate_sites(synth):
    sites = pd.DataFrame({"strand": ["+", "+", "+", "+", "-", "-"], "site": [0, 1299, 1600, 4100, 4509, 6000],
                          "n_reads": [1, 2, 3, 4, 5, 6], "n_accepted": [0, 1, 2, 3, 4, 5]})
    with pysam.FastaFile(str(synth.fasta)) as fa:
        seq = fa.fetch("chrA").upper()
        out = annotate_sites(sites, "chrA", seq, load_region_segments(synth.gtf), load_gene_segments(synth.gtf))
        ctx = [sense_context(fa, "chrA", s, st == "-") for s, st in zip(sites.site, sites.strand)]
    assert list(out.columns) == ["ref", "strand", "site", "n_reads", "n_accepted", "a20", "label", "a_band",
                                 "hexamer", "region", "gene"]
    assert (out.ref == "chrA").all() and out.n_reads.tolist() == [1, 2, 3, 4, 5, 6]
    a20 = [d.count("A") for d, _ in ctx]
    assert out.a20.tolist() == a20 and out.hexamer.tolist() == [has_hexamer(u) for _, u in ctx]
    # planted runs: A6 at 0, A12 at 1299, A10 at 1599 (9 of them after 1600), T10 before 4509 on -, T6 at 5994
    assert a20[0] >= 6 and a20[1] >= 12 and a20[2] >= 9 and a20[4] >= 10 and a20[5] >= 6
    assert out.label.tolist() == ["templated" if x >= 8 else "nontemplated" if x <= 4 else "ambiguous" for x in a20]
    assert out.a_band.iloc[1] == ("14-20" if a20[1] >= 14 else "10-13")
    # last cDNA base: -1 (off contig), 1298 (t1/t2 exon 1, no CDS), 1599 (g1 intron), 4099 (past g1),
    # 4509 on - (t3 exon 4201-4700), 6000 on - (past g2)
    assert out.region.tolist() == ["intergenic", "ncexon", "intron", "intergenic", "ncexon", "intergenic"]
    assert out.gene.tolist() == ["", "g1", "g1", "", "g2", ""]


def test_annotate_sites_last_cdna_base_at_boundaries(synth):
    # 0-based: g1 exon 1 1000..1399 (+); g2 = t3 4200..5799 (-), t3 exon 4200..4699, intron 4700..4999.
    # + 1400: last 1399 exon end (1400 intron); + 1000: last 999 intergenic (1000 exon);
    # - 4200: last 4200 first exon base (4199 intergenic); - 4699: last 4699 exon (4700 intron);
    # - 5800: last 5800 intergenic (5799 g2 exon)
    sites = pd.DataFrame({"strand": ["+", "+", "-", "-", "-"], "site": [1400, 1000, 4200, 4699, 5800],
                          "n_reads": [1] * 5, "n_accepted": [1] * 5})
    with pysam.FastaFile(str(synth.fasta)) as fa:
        out = annotate_sites(sites, "chrA", fa.fetch("chrA").upper(), load_region_segments(synth.gtf),
                             load_gene_segments(synth.gtf))
    assert out.region.tolist() == ["ncexon", "intergenic", "ncexon", "ncexon", "intergenic"]
    assert out.gene.tolist() == ["g1", "", "g2", "g2", ""]


def test_hexamer_summary():
    sites = pd.DataFrame({"a20": [0, 2, 3, 4, 8, 9, 12, 15, 6, 0],
                          "hexamer": [True, True, False, False, True, False, False, False, True, True],
                          "n_accepted": [1, 1, 1, 2, 1, 1, 3, 1, 1, 0]})
    s = hexamer_summary(sites, {"templated_min": 8, "nontemplated_max": 4})
    # last row has no accepted read; non-templated a20 0,2,3,4 -> 2/4; templated 8,9,12,15 -> 1/4; ambiguous 6 -> 1/1
    assert s["n"] == {"templated": 4, "nontemplated": 4, "ambiguous": 1}
    assert s["hexamer_rate"] == {"templated": 0.25, "nontemplated": 0.5, "ambiguous": 1.0}
    assert s["ratio"] == 2.0 and s["passes"] is False
    s = hexamer_summary(sites, {"templated_min": 14, "nontemplated_max": 8})
    # templated only a20 15 (no hexamer): rate 0, ratio undefined
    assert s["n"] == {"templated": 1, "nontemplated": 6, "ambiguous": 2}
    assert s["hexamer_rate"]["nontemplated"] == 4 / 6 and s["ratio"] is None and s["passes"] is False
    sites = pd.DataFrame({"a20": [0, 1, 2, 3, 10, 11, 12, 13], "hexamer": [True] * 3 + [False] + [True] + [False] * 3,
                          "n_accepted": [1] * 8})
    s = hexamer_summary(sites, {"templated_min": 8, "nontemplated_max": 4})
    assert s["ratio"] == 3.0 and s["passes"] is True        # 0.75 / 0.25


def test_compare_labels():
    sites = pd.DataFrame({"ref": ["chr1", "chr1", "chr1", "chr1", "chr2"], "strand": ["+", "+", "+", "-", "+"],
                          "site": [100, 110, 104, 100, 500],
                          "label": ["templated", "nontemplated", "ambiguous", "nontemplated", "templated"],
                          "n_accepted": [1, 2, 0, 1, 1]})
    mol = pd.DataFrame({
        "ref": ["chr1", "chr1", "chr1", "chr1", "chr1", "chr2", "chr3", "chr1", "chr1"],
        "strand": ["+", "+", "+", "+", "-", "-", "+", "+", "+"],
        "site": [105, 107, 121, 120, 100, 500, 10, 90, 200],
        "label": ["templated", "nontemplated", "templated", "ambiguous", "templated", "nontemplated",
                  "nontemplated", "templated", "nontemplated"]})
    table, agree = compare_labels(mol, sites)
    # 105: 100 and 110 both 5 away, the lower site wins (104 is closer but has no accepted read); 107 -> 110;
    # 121 is 11 from 110 (none); 120 is 10 from 110 (matched); chr1 - 100 exact; chr2 - and chr3 have no site;
    # 90 is 10 below 100; 200 is 90 above 110
    exp = {("templated", "templated"): 2, ("nontemplated", "nontemplated"): 1, ("templated", "none"): 1,
           ("ambiguous", "nontemplated"): 1, ("templated", "nontemplated"): 1, ("nontemplated", "none"): 3}
    assert list(table.columns) == ["label_1179", "label_long", "n"]
    assert dict(zip(zip(table.label_1179, table.label_long), table.n)) == exp
    assert agree["molecules"] == 9 and agree["matched"] == 5 and agree["share_matched"] == 5 / 9
    # matched templated: 105 (agree), chr1 - 100 (disagree), 90 (agree); matched non-templated: 107 (agree)
    assert agree["agreement"] == {"templated": 2 / 3, "nontemplated": 1.0, "pooled": 0.75}
    # templated 105, 121, chr1 - 100, 90 -> 3 of 4 matched; non-templated 107, chr2, chr3, 200 -> 1 of 4; ambiguous 120
    assert agree["share_matched_by_label"] == {"templated": 0.75, "nontemplated": 0.25, "ambiguous": 1.0}


def test_count_duplicates():
    assert count_duplicates(np.array([5, 7, 5, 5, 9], np.uint64)) == 2
    assert count_duplicates(np.array([], np.uint64)) == 0
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_long_reads.py`
Expected: FAIL; the first error is `ImportError: cannot import name 'accept_long_read' from 'pafbench.bd67' (pafbench/bd67.py)`.

- [ ] **Step 3: Modify `pafbench/sites.py`** (apply from the repository root)

```bash
git apply <<'PATCH'
diff --git a/pafbench/sites.py b/pafbench/sites.py
index 4adbe0d..32e0ffe 100644
--- a/pafbench/sites.py
+++ b/pafbench/sites.py
@@ -27,6 +27,13 @@ def sense_context(fa, ref, site, is_reverse, down=20, up=40):
     return d, u
 
 
+def context_from_seq(seq, site, is_reverse, down=20, up=40):
+    """sense_context on a preloaded uppercase chromosome sequence (seq = fa.fetch(ref).upper())."""
+    if is_reverse:
+        return seq[max(0, site - down):site].translate(COMP)[::-1], seq[site:site + up].translate(COMP)[::-1]
+    return seq[site:site + down], seq[max(0, site - up):site]
+
+
 def has_hexamer(up40, lo=5, hi=35):
     """AATAAA or ATTAAA fully inside the window lo..hi nt upstream of the site."""
     n = len(up40)
PATCH
```

- [ ] **Step 4: Modify `pafbench/bd67.py`** (apply from the repository root)

```bash
git apply <<'PATCH'
diff --git a/pafbench/bd67.py b/pafbench/bd67.py
index 0c886c9..fd8f880 100644
--- a/pafbench/bd67.py
+++ b/pafbench/bd67.py
@@ -1,4 +1,20 @@
-from .sites import COMP
+import hashlib
+from array import array
+
+import numpy as np
+import pandas as pd
+import pysam
+
+from .config import LABEL_CUTOFFS
+from .genes import genes_at
+from .labels import label_molecules
+from .regions import regions_at
+from .sites import COMP, context_from_seq, has_hexamer, priming_site
+
+OUTCOMES = ("accepted", "short_run", "short_after", "both")
+LABELS = ("templated", "nontemplated", "ambiguous")
+SITE_COLUMNS = ["ref", "strand", "site", "n_reads", "n_accepted", "a20", "label", "a_band", "hexamer", "region", "gene"]
+_ALIGNED, _QUERY, _REF = (0, 7, 8), (1, 4), (2, 3)
 
 
 def sense_3p_clip(a):
@@ -31,3 +47,187 @@ def a_run_site(a):
             break
         k, ref_pos = k + 1, r
     return (ref_pos if k else a.reference_end), k
+
+
+def inward_pairs(a):
+    """(query, ref) pairs of the aligned bases from the transcript 3' end inward: get_aligned_pairs(matches_only=True)
+    reversed on forward reads, as-is on reverse reads, walked lazily from the cigar."""
+    cig = a.cigartuples
+    if a.is_reverse:
+        q, r, step = 0, a.reference_start, 1
+    else:
+        q, r, step, cig = a.query_length - 1, a.reference_end - 1, -1, reversed(cig)
+    for op, n in cig:
+        if op in _ALIGNED:
+            for k in range(n):
+                yield q + step * k, r + step * k
+            q, r = q + step * n, r + step * n
+        elif op in _QUERY:
+            q += step * n
+        elif op in _REF:
+            r += step * n
+
+
+def terminal_a_run(a, tolerance=1):
+    """Terminal A run around the aligned 3' end in transcript sense, with up to `tolerance` interior non-A bases.
+
+    Grows rightwards over the 3' clip, then leftwards over the aligned bases, from one shared non-A budget; a non-A is
+    taken only between A's. Returns (site, aligned_a, clip_a, after_len): the genomic boundary where the run starts
+    (priming_site when the run has no aligned base), the A's in its aligned and clipped parts, and the clipped bases
+    after it.
+    """
+    seq = a.query_sequence.translate(COMP) if a.is_reverse else a.query_sequence
+    clip, pairs, left = sense_3p_clip(a), inward_pairs(a), []
+
+    def aligned(t):
+        """t-th aligned sense base from the 3' end as (base, ref), None past the 5' end."""
+        while len(left) <= t:
+            p = next(pairs, None)
+            if p is None:
+                return None
+            left.append((seq[p[0]], p[1]))
+        return left[t]
+
+    budget, j, n = tolerance, 0, len(clip)
+    while j < n:
+        if clip[j] != "A":
+            if not (budget and j + 1 < n and clip[j + 1] == "A" and (j or aligned(0)[0] == "A")):
+                break
+            budget -= 1
+        j += 1
+    t = aligned_a = 0
+    while (cur := aligned(t)) is not None:
+        if cur[0] == "A":
+            aligned_a += 1
+        else:
+            nxt = aligned(t + 1)
+            if not (budget and nxt is not None and nxt[0] == "A" and (t or j)):
+                break
+            budget -= 1
+        t += 1
+    if t == 0:
+        site = priming_site(a)
+    else:
+        site = left[t - 1][1] + 1 if a.is_reverse else left[t - 1][1]
+    return site, aligned_a, clip[:j].count("A"), n - j
+
+
+def accept_long_read(aligned_a, clip_a, after_len, min_run=10, min_after=30):
+    return aligned_a + clip_a >= min_run and after_len >= min_after
+
+
+def read_outcome(aligned_a, clip_a, after_len, min_run=10, min_after=30):
+    """accepted, short_run (run too short), short_after (too little clip after the run) or both."""
+    if aligned_a + clip_a >= min_run:
+        return "accepted" if after_len >= min_after else "short_after"
+    return "short_run" if after_len >= min_after else "both"
+
+
+def _digest(name):
+    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little")
+
+
+def scan_long_reads(bam_path, ref, min_mapq=20):
+    """Terminal A run (tolerance 1) of every primary record with MAPQ >= min_mapq on one reference.
+
+    Returns the unique sites (strand, site, n_reads, n_accepted), record and read outcome counts, and a 64-bit
+    blake2b digest per read name so that duplicate names can be counted across references.
+    """
+    strand, site, ok, digests = array("b"), array("q"), array("b"), array("Q")
+    counts = {"records": 0, "secondary_supplementary": 0, "low_mapq": 0, "used": 0} | dict.fromkeys(OUTCOMES, 0)
+    with pysam.AlignmentFile(str(bam_path)) as bam:
+        for a in bam.fetch(ref):
+            counts["records"] += 1
+            if a.is_secondary or a.is_supplementary:
+                counts["secondary_supplementary"] += 1
+                continue
+            if a.mapping_quality < min_mapq:
+                counts["low_mapq"] += 1
+                continue
+            s, aligned_a, clip_a, after_len = terminal_a_run(a)
+            outcome = read_outcome(aligned_a, clip_a, after_len)
+            counts[outcome] += 1
+            strand.append(a.is_reverse)
+            site.append(s)
+            ok.append(outcome == "accepted")
+            digests.append(_digest(a.query_name))
+    counts["used"] = len(site)
+    reads = pd.DataFrame({"strand": np.where(np.asarray(strand, bool), "-", "+"), "site": np.asarray(site, np.int64),
+                          "accepted": np.asarray(ok, np.int64)})
+    sites = reads.groupby(["strand", "site"]).accepted.agg(n_reads="size", n_accepted="sum").reset_index()
+    return sites, counts, np.asarray(digests, np.uint64)
+
+
+def count_duplicates(digests):
+    return int(len(digests) - np.unique(digests).size)
+
+
+def annotate_sites(sites, ref, seq, region_segments, gene_segments, cutoffs=LABEL_CUTOFFS):
+    """Adds a20, label, a_band and hexamer from the genome around each site, and region and first gene id at the last
+    cDNA base (site - 1 on '+', site on '-'); seq is the uppercase sequence of ref. Returns SITE_COLUMNS."""
+    rev = (sites.strand == "-").to_numpy()
+    pos = sites.site.to_numpy(np.int64)
+    ctx = [context_from_seq(seq, int(s), bool(r)) for s, r in zip(pos, rev)]
+    last = np.where(rev, pos, pos - 1)
+    region = np.empty(len(sites), object)
+    for strd in (True, False):
+        m = rev != strd
+        region[m] = regions_at(region_segments, ref, strd, last[m])
+    gene = [genes_at(gene_segments, ref, not r, int(p)).split(";")[0] for p, r in zip(last, rev)]
+    out = sites.assign(ref=ref, a20=[d.count("A") for d, _ in ctx], hexamer=[has_hexamer(u) for _, u in ctx],
+                       region=region, gene=gene)
+    return label_molecules(out, cutoffs)[SITE_COLUMNS]
+
+
+def hexamer_summary(sites, cutoffs):
+    """Sites with an accepted read: n and hexamer rate by label at these cut-offs, ratio = rate(nontemplated) /
+    rate(templated) (None when the templated rate is 0 or a class is empty), passes = ratio >= 3."""
+    acc = label_molecules(sites[sites.n_accepted >= 1], cutoffs)
+    g = acc.groupby("label").hexamer
+    n, rate = g.size(), g.mean()
+    rates = {k: float(rate[k]) if k in rate.index else None for k in LABELS}
+    ratio = rates["nontemplated"] / rates["templated"] if rates["templated"] and rates["nontemplated"] is not None else None
+    return {"cutoffs": dict(cutoffs), "n": {k: int(n.get(k, 0)) for k in LABELS}, "hexamer_rate": rates,
+            "ratio": ratio, "passes": ratio is not None and ratio >= 3}
+
+
+def nearest_labels(mol, sites, max_dist=10):
+    """Label of the nearest site with an accepted read on the same ref and strand within max_dist nt of each
+    molecule's site ('none' if there is none); at equal distance the lower site wins."""
+    out = np.full(len(mol), "none", object)
+    groups = dict(list(sites[sites.n_accepted >= 1].groupby(["ref", "strand"])))
+    msite = mol.site.to_numpy(np.int64)
+    for key, idx in mol.groupby(["ref", "strand"]).indices.items():
+        g = groups.get(key)
+        if g is None:
+            continue
+        g = g.sort_values("site")
+        s, lab, x = g.site.to_numpy(np.int64), g.label.to_numpy(object), msite[idx]
+        k = np.searchsorted(s, x)
+        lo, hi = np.maximum(k - 1, 0), np.minimum(k, s.size - 1)
+        dlo = np.where(k > 0, x - s[lo], max_dist + 1)
+        dhi = np.where(k < s.size, s[hi] - x, max_dist + 1)
+        hit = np.minimum(dlo, dhi) <= max_dist
+        out[idx[hit]] = lab[np.where(dlo <= dhi, lo, hi)[hit]]
+    return out
+
+
+def compare_labels(mol, sites, max_dist=10):
+    """Molecules (ref, strand, site, label) against long-read sites: the table (label_1179, label_long, n), the
+    share of molecules matched overall and per 1179 label, and label agreement among matched templated and
+    nontemplated molecules."""
+    long, lab = nearest_labels(mol, sites, max_dist), mol.label.to_numpy(object)
+    table = (pd.DataFrame({"label_1179": lab, "label_long": long}).value_counts().rename("n").reset_index()
+             .sort_values(["label_1179", "label_long"], ignore_index=True))
+    matched = long != "none"
+
+    def share(num, den):
+        return float(num.sum() / den.sum()) if den.sum() else None
+
+    agreement = {k: share(matched & (lab == k) & (long == k), matched & (lab == k)) for k in ("templated", "nontemplated")}
+    both = matched & np.isin(lab, ["templated", "nontemplated"])
+    agreement["pooled"] = share(both & (long == lab), both)
+    return table, {"molecules": len(mol), "matched": int(matched.sum()),
+                   "share_matched": share(matched, np.ones(len(mol), bool)),
+                   "share_matched_by_label": {k: share(matched & (lab == k), lab == k) for k in LABELS},
+                   "agreement": agreement}
PATCH
```

- [ ] **Step 5: Create `scripts/16_long_read_sites.py`**

```python
"""Stage 16: long-read (BD67, BD70) priming sites, read acceptance by terminal A run and bead clip, comparison with 1179."""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from pafbench.bd67 import OUTCOMES, annotate_sites, compare_labels, count_duplicates, hexamer_summary, scan_long_reads
from pafbench.config import FASTA, GTF, LABEL_CUTOFFS, LONG_READS, OUT
from pafbench.genes import load_gene_segments
from pafbench.regions import load_region_segments

REFS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
M1_CUTOFFS = {"templated_min": 14, "nontemplated_max": 8}
RECORD_KEYS = ("records", "secondary_supplementary", "low_mapq", "used")
_SEG = None


def _init(segments):
    global _SEG
    _SEG = segments


def _work(job):
    lib, ref = job
    t0 = time.monotonic()
    with pysam.FastaFile(str(FASTA)) as fa:
        seq = fa.fetch(ref).upper()
    sites, counts, digests = scan_long_reads(LONG_READS[lib], ref)
    sites = annotate_sites(sites, ref, seq, *_SEG)
    return lib, ref, sites, counts | {"seconds": time.monotonic() - t0}, digests


def _summary(lib, parts, sites):
    counts = {k: sum(p[3][k] for p in parts) for k in RECORD_KEYS + OUTCOMES}
    used = counts["used"]
    return {"library": lib, "bam": str(LONG_READS[lib]), "refs": [p[1] for p in parts],
            "records": {k: counts[k] for k in RECORD_KEYS},
            "duplicate_qnames": count_duplicates(np.concatenate([p[4] for p in parts])),
            "outcomes": {k: counts[k] for k in OUTCOMES},
            "outcome_fractions": {k: counts[k] / used for k in OUTCOMES},
            "dropped_fraction": 1 - counts["accepted"] / used,
            "unique_sites": len(sites), "sites_with_accepted_read": int((sites.n_accepted >= 1).sum()),
            "cutoffs_8_4": hexamer_summary(sites, LABEL_CUTOFFS),
            "cutoffs_14_8": hexamer_summary(sites, M1_CUTOFFS),
            "per_ref": {p[1]: p[3] for p in parts}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", nargs="+", default=list(LONG_READS), choices=list(LONG_READS))
    ap.add_argument("--refs", nargs="+", default=REFS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--compare", type=Path, default=OUT / "1179" / "truth" / "molecules_labeled.parquet")
    args = ap.parse_args()
    size = {}
    for lib in args.libs:
        with pysam.AlignmentFile(str(LONG_READS[lib])) as fh:
            size |= {(lib, s.contig): s.mapped for s in fh.get_index_statistics()}
    jobs = sorted(((lib, ref) for lib in args.libs for ref in args.refs), key=lambda j: -size.get(j, 0))
    segments = (load_region_segments(GTF), load_gene_segments(GTF))
    with Pool(min(args.workers, len(jobs)), initializer=_init, initargs=(segments,), maxtasksperchild=1) as pool:
        res = pool.map(_work, jobs, chunksize=1)
    mol, line = None, {"compare": f"skipped, {args.compare} not found", "libraries": []}
    if args.compare.exists():
        mol = pd.read_parquet(args.compare, columns=["ref", "strand", "site", "label"])
        mol = mol[mol.ref.isin(args.refs)].reset_index(drop=True)
        line["compare"] = str(args.compare)
    for lib in args.libs:
        parts = sorted((r for r in res if r[0] == lib), key=lambda r: args.refs.index(r[1]))
        sites = pd.concat([p[2] for p in parts], ignore_index=True)
        out = args.out / lib
        out.mkdir(parents=True, exist_ok=True)
        sites.to_parquet(out / "long_read_sites.parquet", index=False)
        summary = _summary(lib, parts, sites)
        if mol is not None:
            table, agreement = compare_labels(mol, sites)
            table.to_parquet(out / "vs_1179.parquet", index=False)
            summary["vs_1179"] = {"path": str(args.compare)} | agreement
        (out / "long_read_acceptance.json").write_text(json.dumps(summary, indent=1))
        line["libraries"].append({"library": lib, "used": summary["records"]["used"],
                                  "dropped": summary["dropped_fraction"], "duplicate_qnames": summary["duplicate_qnames"],
                                  "accepted_sites": summary["sites_with_accepted_read"],
                                  "ratio_8_4": summary["cutoffs_8_4"]["ratio"], "passes_8_4": summary["cutoffs_8_4"]["passes"],
                                  "ratio_14_8": summary["cutoffs_14_8"]["ratio"]})
    print(json.dumps(line))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `143 passed, 2 warnings`.

- [ ] **Step 7: Stage 16 genome-wide**

Run: `mkdir -p /data/element_runs/polyafilter_bench/runs && .venv/bin/python -m pafbench.memwatch /data/element_runs/polyafilter_bench/runs/16_long_read_sites.mem.json -- .venv/bin/python scripts/16_long_read_sites.py`
Expected: about 11 to 13 min. In each `long_read_acceptance.json`: duplicate qnames 0; dropped reads about 12% (BD67) and 9% (BD70); `passes` true with a per-site ratio near 5 (chr1 in the build: 5.65 and 5.32). If either library does not pass, stop and bring the numbers to the user: by the spec, that library stays out of later analyses. `vs_1179.parquet` exists for both.

- [ ] **Step 8: Commit**

```bash
git add tests/test_long_reads.py pafbench/sites.py pafbench/bd67.py scripts/16_long_read_sites.py
git -c user.name="Simone Picelli" -c user.email="simone.picelli@iob.ch" commit -m "Stage 16: long-read sites from the terminal A run (aligned plus clipped, one non-A allowed), acceptance by bead sequence, per-site labels and hexamer test, comparison with 1179"
```

---

### Task 5: Spike-in truth sets and grid flags (stage 14)

**Files:**
- Create: `tests/test_member_rows.py`
- Create: `tests/test_spikein_truth.py`
- Create: `tests/test_spikein_stage.py`
- Modify: `pafbench/molecules.py`
- Modify: `scripts/06_molecules.py`
- Modify: `pafbench/spikein.py`
- Create: `scripts/14_spikein_truth.py`

**Interfaces:**
- Consumes: Task 2 `truth.label_truth_molecules`, `truth.truth_reads`; Task 3 `stream.scan_reference` and the region-merge helper; Task 4 `bd67.terminal_a_run`; milestone 1 `spikein.module_of`, `exons.member_distance`, `exons.load_collapsed_exons`, `genes.load_gene_segments`; `scripts/09_spikein_star.sh LIB` output `OUT/<lib>/spikein/star_Aligned.out.bam`.
- Produces:
  - `pafbench.spikein.tag_survival(before, after, tags=('CB', 'UB', 'pa', 'ts'))`
  - `pafbench.spikein.module_of(contig)`
  - `pafbench.spikein.annotated_ends(gtf)`
  - `pafbench.spikein.label_spike_site(site, strd, ends, tail_tol=10, internal_min=30)`
  - `pafbench.spikein.spike_pa_row(a, fa, segments)`
  - `pafbench.spikein.add_spike_labels(mol, ends)`
  - `pafbench.spikein.merge_scans(scans, tags)`
  - `pafbench.spikein.spike_summary(pa, labeled, members, reads)`
  - `pafbench.molecules.member_rows(records, ref, by_k3, gene_segments, exons_by_gene, collisions=None)`
- Produces (data): under `OUT/<lib>/spikein/`: `star_sorted.bam(.bai)`, `pa_sites.parquet`, `molecules.parquet`, `members.parquet`, `truth/molecules_labeled.parquet` (stage 12 columns + `spike_label` tail/internal/other, `dist_to_end`, `module`), `truth/truth_reads.parquet`, `grid/truth_flags.parquet`, `grid/region_counts.parquet`.

Spike-in labels come from the annotated SIRV and ERCC 3' ends, not from A content (spec 4.2), so they test the label rule and polyAfilter independently. STAR keeps the reads' poly(A), partly aligned, so the site is the start of the terminal A run, as for long reads. The per-read member logic moves from stage 06 into `molecules.member_rows` so both stages share it; stage 06's output stays byte-identical (checked on 1417 chr21 during the build).

- [ ] **Step 1: Write the failing tests in `tests/test_member_rows.py`**

```python
import importlib.util
from collections import defaultdict

import pandas as pd
import pysam
import pytest

from pafbench.config import REPO
from pafbench.exons import load_collapsed_exons, member_distance
from pafbench.genes import genes_at, load_gene_segments
from pafbench.molecules import MEMBER_COLUMNS, gene_match, member_distances, member_rows
from pafbench.sites import last_aligned_base

BY_K3 = {
    ("c0", "u0", "+"): [(0, "chrA", 3000, "g1"), (1, "chrB", 1500, "g3")],
    ("c1", "u1", "+"): [(2, "chrA", 2600, "g1")],
    ("c2", "u0", "-"): [(3, "chrA", 4300, "g2")],
    ("c0", "u1", "-"): [(4, "chrA", 5000, "")],
    ("c1", "u0", "+"): [(5, "chrB", 1500, "g3")],
}

# (ref, reference_start, cigar, flag) -> tags; every other record gets tags by its index (see _tagged)
OVERRIDE = {
    ("chrA", 950, "90M", 0): {"CB": "c0", "UB": "u0"},
    ("chrA", 1340, "60M400N30M", 0): {"CB": "c1", "UB": "u1"},
    ("chrA", 4396, "90M", 16): {"CB": "c2", "UB": "u0"},
    ("chrA", 5052, "90M", 16): {"CB": "c0", "UB": "u1"},
    ("chrA", 3400, "90M", 1024): {"CB": "c0", "UB": "u0"},
    ("chrA", 1357, "90M", 0): {"CB": "c0", "UB": "u0", "pa": 12},   # pa read: never a member
    ("chrA", 1394, "90M", 0): {"CB": "c0"},                          # no UB
    ("chrA", 1250, "90M", 256): {"CB": "c0", "UB": "u0"},            # secondary
}


@pytest.fixture(scope="module")
def tagged(synth, tmp_path_factory):
    """Copy of the synthetic BAM with CB/UB cycling over the molecule keys, some reads without CB or UB, some pa reads."""
    out = tmp_path_factory.mktemp("members") / "tagged.bam"
    with pysam.AlignmentFile(str(synth.bam)) as fin, pysam.AlignmentFile(str(out), "wb", template=fin) as fout:
        for i, a in enumerate(fin.fetch(until_eof=True)):
            tags = OVERRIDE.get((a.reference_name, a.reference_start, a.cigarstring, a.flag))
            if tags is None:
                tags = {}
                if i % 13 != 12:
                    tags["CB"] = f"c{i % 3}"
                if i % 7 != 6:
                    tags["UB"] = f"u{i % 2}"
                if i % 11 == 10:
                    tags["pa"] = 20
            a.set_tags(list(tags.items()))
            fout.write(a)
    pysam.index(str(out))
    return out


def _stage06_reference(bam_path, ref, by_k3, seg, exons):
    """scripts/06_molecules.py::_work as of e534d16 (milestone 1), with its globals passed in."""
    rows, coll = [], defaultdict(int)
    with pysam.AlignmentFile(str(bam_path)) as bam:
        for a in bam.fetch(ref):
            if a.is_secondary or a.is_supplementary or a.has_tag("pa") or not a.has_tag("CB") or not a.has_tag("UB"):
                continue
            hits = by_k3.get((a.get_tag("CB"), a.get_tag("UB"), "-" if a.is_reverse else "+"))
            if not hits:
                continue
            for mol, mref, site, sgenes in hits:
                if mref != ref:
                    coll[mol] += 1
                    continue
                d5, d3 = member_distances(a, site)
                e5, e3, mode = member_distance(a.get_blocks(), a.is_reverse, site, sgenes, exons, (d5, d3))
                mg = genes_at(seg, ref, not a.is_reverse, last_aligned_base(a))
                rows.append([mol, a.query_name, d5, d3, "N" in a.cigarstring, a.is_duplicate, mg, gene_match(sgenes, mg), e5, e3, mode])
    return rows, dict(coll)


@pytest.fixture(scope="module")
def annot(synth):
    return load_gene_segments(synth.gtf), load_collapsed_exons(synth.gtf)


def _new(bam_path, ref, seg, exons):
    coll = {}
    with pysam.AlignmentFile(str(bam_path)) as bam:
        rows = member_rows(bam.fetch(ref), ref, BY_K3, seg, exons, coll)
    return rows, coll


def test_member_rows_equal_stage06_logic(tagged, annot):
    seg, exons = annot
    total = 0
    for ref in ("chrA", "chrB"):
        rows, coll = _new(tagged, ref, seg, exons)
        ref_rows, ref_coll = _stage06_reference(tagged, ref, BY_K3, seg, exons)
        assert rows == ref_rows
        assert coll == ref_coll
        total += len(rows)
    assert total > 20
    with pysam.AlignmentFile(str(tagged)) as bam:
        assert member_rows(bam.fetch("chrA"), "chrA", BY_K3, seg, exons) == _stage06_reference(tagged, "chrA", BY_K3, seg, exons)[0]


def test_member_rows_by_hand(tagged, annot):
    seg, exons = annot
    rows, coll = _new(tagged, "chrA", seg, exons)
    with pysam.AlignmentFile(str(tagged)) as bam:
        name = {(a.reference_start, a.cigarstring, a.flag): a.query_name for a in bam.fetch("chrA")}
    df = pd.DataFrame(rows, columns=MEMBER_COLUMNS).set_index("qname")
    # 950 90M fwd, mol 0 (site 3000, g1): d5 3000-950, d3 3000-1040; gene at 1039 is g1; the last cDNA base 2999 is
    # intronic in g1 (exons 1000-1400, 1800-2700, 3000-4000), so the genomic pair is kept
    assert df.loc[name[(950, "90M", 0)]].tolist() == [0, 2050, 1960, False, False, "g1", True, 2050, 1960, "genome"]
    # spliced 1340 60M400N30M, mol 2 (site 2600): blocks 1340-1400 and 1800-1830 and 2599 are exonic;
    # exon distances 60 + 800 = 860 and 2600 - 1830 = 770; genomic 1260 and 770
    assert df.loc[name[(1340, "60M400N30M", 0)]].tolist() == [2, 1260, 770, True, False, "g1", True, 860, 770, "exon"]
    # reverse 4396-4486, mol 3 (site 4300, g2 exon 4200-4700): d5 4486-4300, d3 4396-4300, both exonic
    assert df.loc[name[(4396, "90M", 16)]].tolist() == [3, 186, 96, False, False, "g2", True, 186, 96, "exon"]
    # reverse 5052-5142, mol 4 (site 5000, intergenic label ""): read gene g2 does not match, genomic distances
    assert df.loc[name[(5052, "90M", 16)]].tolist() == [4, 142, 52, False, False, "g2", False, 142, 52, "genome"]
    # duplicate 3400-3490, mol 0: past the site (negative distances), dup kept
    assert df.loc[name[(3400, "90M", 1024)]].tolist() == [0, -400, -490, False, True, "g1", True, -400, -490, "genome"]
    # the pa read, the read without UB and the secondary give no row
    for key in ((1357, "90M", 0), (1394, "90M", 0), (1250, "90M", 256)):
        assert name[key] not in df.index
    # mol 1 sits on chrB: c0/u0 forward reads on chrA are collisions, not members
    assert 1 not in set(df.mol) and coll[1] == int((df.mol == 0).sum())


def test_stage06_work_uses_member_rows(tagged, annot):
    spec = importlib.util.spec_from_file_location("stage06", REPO / "scripts" / "06_molecules.py")
    stage06 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage06)
    seg, exons = annot
    stage06._BY_K3, stage06._SEG, stage06._EXONS = BY_K3, seg, exons
    for ref in ("chrA", "chrB"):
        df, coll = stage06._work((tagged, ref))
        ref_rows, ref_coll = _stage06_reference(tagged, ref, BY_K3, seg, exons)
        pd.testing.assert_frame_equal(df, pd.DataFrame(ref_rows, columns=MEMBER_COLUMNS))
        assert coll == ref_coll
```

- [ ] **Step 2: Write the failing tests in `tests/test_spikein_truth.py`**

```python
import numpy as np
import pandas as pd
import pysam
import pytest

from pafbench.bd67 import terminal_a_run
from pafbench.flags import Setting, setting_tag
from pafbench.genes import load_gene_segments
from pafbench.sites import COMP, PA_COLUMNS, has_hexamer, pa_row, priming_site
from pafbench.spikein import (add_spike_labels, annotated_ends, label_spike_site, merge_scans, spike_pa_row,
                              spike_summary)
from pafbench.stream import merge_parts, scan_reference, scan_to_parts
from pafbench.transcripts import load_transcripts

ENDS_GTF = [
    ("X", "gene", 101, 650, "+", 'gene_id "gX";'),
    ("X", "transcript", 101, 500, "+", 'gene_id "gX"; transcript_id "tx1";'),
    ("X", "exon", 101, 200, "+", 'gene_id "gX"; transcript_id "tx1";'),
    ("X", "exon", 301, 500, "+", 'gene_id "gX"; transcript_id "tx1";'),
    ("X", "transcript", 101, 650, "+", 'gene_id "gX"; transcript_id "tx2";'),
    ("X", "transcript", 300, 650, "+", 'gene_id "gX"; transcript_id "tx3";'),
    ("X", "transcript", 701, 900, "-", 'gene_id "gX"; transcript_id "tx4";'),
    ("X", "exon", 701, 900, "-", 'gene_id "gX"; transcript_id "tx4";'),
    ("X", "transcript", 1001, 1100, "-", 'gene_id "gX"; transcript_id "tx5";'),
    ("Y", "gene", 1, 1045, "+", 'gene_id "gY";'),
    ("Y", "transcript", 1, 1045, "+", 'gene_id "gY"; transcript_id "ty1";'),
]


def _write_gtf(path, rows):
    with open(path, "w") as fh:
        for c, ft, s, e, strand, attr in rows:
            fh.write(f"{c}\tt\t{ft}\t{s}\t{e}\t.\t{strand}\t.\t{attr}\n")
    return path


def test_annotated_ends(tmp_path):
    ends = annotated_ends(_write_gtf(tmp_path / "e.gtf", ENDS_GTF))
    # plus: GTF end (tx1 500, tx2 and tx3 both 650); minus: GTF start - 1 (tx4 700, tx5 1000); exon end 200 and gene lines ignored
    assert sorted(ends) == [("X", False), ("X", True), ("Y", True)]
    assert ends[("X", True)].tolist() == [500, 650] and ends[("X", True)].dtype == np.int64
    assert ends[("X", False)].tolist() == [700, 1000]
    assert ends[("Y", True)].tolist() == [1045]


def test_annotated_ends_match_priming_site(tmp_path):
    # a read whose last cDNA base is the transcript's last base has priming_site == the annotated end (d = 0)
    ends = annotated_ends(_write_gtf(tmp_path / "e.gtf", ENDS_GTF))
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "X", "LN": 2000}]})
    fwd, rev = pysam.AlignedSegment(hdr), pysam.AlignedSegment(hdr)
    fwd.reference_name, fwd.reference_start, fwd.cigarstring, fwd.flag = "X", 450, "50M", 0  # covers GTF 451..500
    rev.reference_name, rev.reference_start, rev.cigarstring, rev.flag = "X", 700, "50M", 16  # covers GTF 701..750
    assert label_spike_site(priming_site(fwd), True, ends[("X", True)]) == ("tail", 0)
    assert label_spike_site(priming_site(rev), False, ends[("X", False)]) == ("tail", 0)


@pytest.mark.parametrize("site,expected", [
    (1010, ("tail", -10)),       # d = 1000 - 1010
    (990, ("tail", 10)),
    (989, ("other", 11)),
    (971, ("other", 29)),
    (970, ("internal", 30)),
    (1011, ("other", -11)),
])
def test_label_spike_site_plus(site, expected):
    assert label_spike_site(site, True, np.array([1000], np.int64)) == expected


@pytest.mark.parametrize("site,expected", [
    (530, ("internal", 30)),     # d = site - end on minus: upstream in transcript sense is a larger coordinate
    (529, ("other", 29)),
    (511, ("other", 11)),
    (510, ("tail", 10)),
    (490, ("tail", -10)),
    (489, ("other", -11)),
])
def test_label_spike_site_minus(site, expected):
    assert label_spike_site(site, False, np.array([500], np.int64)) == expected


def test_label_spike_site_several_ends():
    plus = np.array([1000, 1100], np.int64)
    assert label_spike_site(1050, True, plus) == ("internal", 50)   # d -50 and 50: tie on |d|, the positive one
    assert label_spike_site(1105, True, plus) == ("tail", -5)       # d -105, -5
    assert label_spike_site(1200, True, plus) == ("other", -100)    # d -200, -100: past every end
    assert label_spike_site(995, True, plus) == ("tail", 5)         # d 5 and 105: tail wins over internal
    assert label_spike_site(550, False, np.array([500, 600], np.int64)) == ("internal", 50)  # d 50, -50
    assert label_spike_site(550, True, None) == ("other", None)
    assert label_spike_site(550, True, np.array([], np.int64)) == ("other", None)
    assert label_spike_site(1050, True, plus, tail_tol=50) == ("tail", 50)
    assert label_spike_site(1050, True, plus, internal_min=51) == ("other", 50)


# Contig S1 (200 nt) on a background with no A or T; planted runs give each test read a known terminal A run.
def _s1():
    s = list("CG" * 100)
    s[63:69] = "AATAAA"          # hexamer for the plus run read: in the 5..35 nt window of site 98, not of 104
    s[95:105] = "CATAGAAAAC"     # 95 C, 96 A, 97 T, 98 A, 99 G, 100..103 A, 104 C
    s[150:158] = "TTTTCTGG"      # sense on minus (genomic 157 down to 150): C C A G A A A A
    s[184:190] = "TTTATT"        # sense AATAAA for the minus run read: in the window of site 156, not of 150
    return "".join(s)


S1 = _s1()
SEG_GTF = [
    ("S1", "gene", 1, 98, "+", 'gene_id "gp1";'),      # [0, 98)
    ("S1", "gene", 99, 200, "+", 'gene_id "gp2";'),    # [98, 200)
    ("S1", "gene", 1, 156, "-", 'gene_id "gm1";'),     # [0, 156)
    ("S1", "gene", 157, 200, "-", 'gene_id "gm2";'),   # [156, 200)
]


@pytest.fixture
def s1(tmp_path):
    fasta = tmp_path / "s1.fa"
    fasta.write_text(f">S1\n{S1}\n")
    pysam.faidx(str(fasta))
    seg = load_gene_segments(_write_gtf(tmp_path / "s1.gtf", SEG_GTF))
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "S1", "LN": 200}]})
    with pysam.FastaFile(str(fasta)) as fa:
        yield fa, seg, hdr


def _rec(hdr, name, pos, cigar, seq, rev, tags):
    a = pysam.AlignedSegment(hdr)
    a.query_name, a.reference_name, a.reference_start, a.cigarstring = name, "S1", pos, cigar
    a.flag, a.mapping_quality, a.query_sequence = 16 if rev else 0, 255, seq
    a.set_tags(tags)
    return a


def _revcomp(s):
    return s.translate(COMP)[::-1]


def test_spike_pa_row_plus_run(s1):
    fa, seg, hdr = s1
    # aligned 60..103, then 10 clipped A's; the run takes 103..100 (A), 99 (G, the one allowed non-A, A at 98 after it)
    # and 98 (A); 97 is a second single non-A (T, A at 96), so the run starts at 98. A budget of two non-A's would take
    # 97 and 96 and stop at 95 (C, C at 94): site 96
    a = _rec(hdr, "p_run", 60, "44M10S", S1[60:104] + "A" * 10, False,
             [("pa", 10), ("ts", 5), ("CB", "c1"), ("UB", "u1"), ("xf", 25)])
    row = dict(zip(PA_COLUMNS, spike_pa_row(a, fa, seg)))
    assert row["site"] == 98 and row["strand"] == "+"
    assert terminal_a_run(a, tolerance=2)[0] == 96
    assert row["down20"] == S1[98:118] and row["up40"] == S1[58:98]
    assert row["a20"] == 5                       # 98 and 100..103
    assert row["hexamer"] is True                # AATAAA at 63..68 lies in up40[5:35] = S1[63:93]
    assert row["genes"] == "gp1"                 # last cDNA base 97
    assert (row["pa"], row["ts"], row["cb"], row["ub"], row["gx"], row["re"], row["xf"]) == (10, 5, "c1", "u1", None, None, 25)
    assert (row["flag"], row["dup"], row["mapq"], row["aligned_len"], row["spliced"]) == (0, False, 255, 44, False)
    # pa_row keeps the alignment end: site 104, no A downstream, hexamer outside its window, gene gp2 at 103
    old = dict(zip(PA_COLUMNS, pa_row(a, fa, seg)))
    assert (old["site"], old["a20"], old["hexamer"], old["genes"]) == (104, 0, False, "gp2")
    assert {k for k in PA_COLUMNS if row[k] != old[k]} == {"site", "down20", "up40", "a20", "hexamer", "genes"}


def test_spike_pa_row_minus_run(s1):
    fa, seg, hdr = s1
    # reverse read aligned 150..193 with 10 clipped T's on the left (sense A's); sense run over 150..153 (A),
    # 154 (G, then A at 155) and 155 (A); 156 is sense C with sense C at 157, so the run starts at 156
    a = _rec(hdr, "m_run", 150, "10S44M", "T" * 10 + S1[150:194], True, [("pa", 10), ("CB", "c2"), ("UB", "u2")])
    row = dict(zip(PA_COLUMNS, spike_pa_row(a, fa, seg)))
    assert row["site"] == 156 and row["strand"] == "-"
    assert row["down20"] == _revcomp(S1[136:156]) and row["up40"] == _revcomp(S1[156:196])
    assert row["a20"] == 5                       # genomic T's at 150..153 and 155
    assert row["hexamer"] is True                # TTTATT at 184..189 is sense up40[6:12]
    assert row["genes"] == "gm2"                 # last cDNA base 156
    assert (row["ts"], row["xf"]) == (0, None)
    old = dict(zip(PA_COLUMNS, pa_row(a, fa, seg)))
    assert (old["site"], old["a20"], old["hexamer"], old["genes"]) == (150, 0, False, "gm1")


def test_spike_pa_row_without_aligned_run_equals_pa_row(s1):
    fa, seg, hdr = s1
    tags = [("pa", 8), ("CB", "c3"), ("UB", "u3"), ("ts", 2)]
    fwd = _rec(hdr, "p_norun", 20, "30M8S", S1[20:50] + "A" * 8, False, tags)      # 49 G, 48 C: no aligned A run
    rev = _rec(hdr, "m_norun", 120, "8S20M", "T" * 8 + S1[120:140], True, tags)    # 120 C, 121 G: no aligned T run
    for a, site in ((fwd, 50), (rev, 120)):
        row = spike_pa_row(a, fa, seg)
        assert row == pa_row(a, fa, seg)
        assert row[PA_COLUMNS.index("site")] == site


def test_add_spike_labels():
    ends = {("ERCC-00002", True): np.array([1045], np.int64), ("SIRV1", False): np.array([1000, 1006], np.int64),
            ("SIRV4001", True): np.array([3997], np.int64)}
    mol = pd.DataFrame({"mol": [0, 1, 2, 3, 4], "ref": ["ERCC-00002", "ERCC-00002", "SIRV1", "SIRV1", "SIRV4001"],
                        "strand": ["+", "-", "-", "+", "+"], "site": [1040, 800, 1040, 1040, 2000]})
    out = add_spike_labels(mol, ends)
    assert out.columns.tolist() == ["mol", "ref", "strand", "site", "spike_label", "dist_to_end", "module"]
    pd.testing.assert_frame_equal(out[mol.columns], mol)
    # 1045 - 1040 = 5 tail; no minus ends on ERCC-00002; minus d = 1040 - 1006 = 34 internal; no plus ends on SIRV1;
    # 3997 - 2000 = 1997 internal
    assert out.spike_label.tolist() == ["tail", "other", "internal", "other", "internal"]
    assert out.dist_to_end.tolist()[:1] == [5.0] and np.isnan(out.dist_to_end.iloc[1])
    assert out.dist_to_end.iloc[2] == 34.0 and np.isnan(out.dist_to_end.iloc[3]) and out.dist_to_end.iloc[4] == 1997.0
    assert out.dist_to_end.dtype == np.float64
    assert out.module.tolist() == ["ERCC", "ERCC", "SIRV_E0", "SIRV_E0", "long_SIRV"]


def test_merge_scans_equals_merge_parts(synth, tmp_path):
    tx = load_transcripts(synth.gtf)
    settings = [Setting(300, 10, 1, True), Setting(150, 5, 0, False), Setting(150, None, 0, True), Setting(450, 3, 2, False)]
    tags = [setting_tag(s) for s in settings]
    with pysam.AlignmentFile(str(synth.bam)) as fh:
        names = sorted({a.query_name for a in fh.fetch(until_eof=True)})
    truth = set(names[::3])
    refs = ["chrA", "chrB"]
    scans = []
    for ref in refs:
        t, r, _ = scan_reference(synth.bam, synth.fasta, ref, tx, settings, truth)
        scans.append((t, r))
        scan_to_parts(synth.bam, synth.fasta, ref, tx, settings, truth, None, tmp_path / "parts")
    truth_df, region_df = merge_scans(scans, tags)
    merge_parts(tmp_path / "parts", refs, tags, tmp_path / "merged")
    truth_df.to_parquet(tmp_path / "truth_flags.parquet", index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "truth_flags.parquet"),
                                  pd.read_parquet(tmp_path / "merged" / "truth_flags.parquet"))
    pd.testing.assert_frame_equal(region_df, pd.read_parquet(tmp_path / "merged" / "region_counts.parquet"))
    assert len(truth_df) == sum(len(t) for t, _ in scans) > 10
    assert truth_df.ref.tolist() == ["chrA"] * len(scans[0][0]) + ["chrB"] * len(scans[1][0])
    assert region_df.tag.tolist() == tags and set(region_df.re) == {""}
    assert (region_df.total == sum(int(r.total.iloc[0]) for _, r in scans)).all()


def test_spike_summary():
    pa = pd.DataFrame({"qname": list("abcdefg")})
    labeled = pd.DataFrame({"mol": range(5), "module": ["ERCC", "ERCC", "SIRV_E0", "long_SIRV", "SIRV_E0"],
                            "spike_label": ["tail", "internal", "tail", "other", "internal"],
                            "label": ["nontemplated", "templated", "ambiguous", "templated", "templated"]})
    members = pd.DataFrame({"mol": [0, 0, 1, 2]})
    reads = pd.DataFrame({"role": ["pa"] * 6 + ["member"] * 3})
    assert spike_summary(pa, labeled, members, reads) == {
        "pa_reads": 7, "molecules": 5,
        "by_module": {"ERCC": {"tail": 1, "internal": 1, "other": 0}, "SIRV_E0": {"tail": 1, "internal": 1, "other": 0},
                      "long_SIRV": {"tail": 0, "internal": 0, "other": 1}},
        "spike_label_by_label": {"tail": {"templated": 0, "nontemplated": 1, "ambiguous": 1},
                                 "internal": {"templated": 2, "nontemplated": 0, "ambiguous": 0},
                                 "other": {"templated": 1, "nontemplated": 0, "ambiguous": 0}},
        "members": 4, "truth_reads": {"pa": 6, "member": 3}}


def test_merge_scans_order_with_several_re_values():
    def region(rows):
        return pd.DataFrame(rows, columns=["tag", "re", "removed", "total"])
    # contig 1 has E and N, contig 2 has I and E; first appearance order would be t1E t1N t2E t2N t1I t2I
    r1 = region([("t1", "E", 1, 10), ("t1", "N", 2, 20), ("t2", "E", 3, 10), ("t2", "N", 4, 20)])
    r2 = region([("t1", "E", 5, 7), ("t1", "I", 6, 9), ("t2", "E", 0, 7), ("t2", "I", 1, 9)])
    t1 = pd.DataFrame({"qname": ["a"], "ref": ["x"], "pos": [1], "flag": [0], "t1": [True], "t2": [False]})
    t2 = pd.DataFrame({"qname": ["b", "c"], "ref": ["y", "y"], "pos": [5, 9], "flag": [16, 0], "t1": [False, True],
                       "t2": [True, True]})
    truth, reg = merge_scans([(t1, r1), (t2, r2)], ["t1", "t2"])
    assert truth.qname.tolist() == ["a", "b", "c"] and truth.index.tolist() == [0, 1, 2]
    assert list(reg.itertuples(index=False, name=None)) == [
        ("t1", "E", 6, 17), ("t1", "I", 6, 9), ("t1", "N", 2, 20), ("t2", "E", 3, 17), ("t2", "I", 1, 9), ("t2", "N", 4, 20)]
```

- [ ] **Step 3: Write the failing tests in `tests/test_spikein_stage.py`**

```python
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import pysam

from pafbench.config import REPO
from pafbench.exons import load_collapsed_exons
from pafbench.flags import setting_tag
from pafbench.genes import load_gene_segments
from pafbench.grid import all_settings
from pafbench.molecules import MEMBER_COLUMNS, consolidate_sites, member_rows
from pafbench.regions import load_region_segments
from pafbench.sites import PA_COLUMNS
from pafbench.spikein import add_spike_labels, annotated_ends, merge_scans, spike_pa_row
from pafbench.stream import scan_reference
from pafbench.transcripts import load_transcripts
from pafbench.truth import label_truth_molecules, truth_reads


def _stage14(monkeypatch):
    """The stage script as module 'stage14' (registered so that pool workers can unpickle its functions)."""
    spec = importlib.util.spec_from_file_location("stage14", REPO / "scripts" / "14_spikein_truth.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "stage14", mod)
    spec.loader.exec_module(mod)
    return mod


def test_stage14_on_synthetic_spikes(synth, tmp_path, monkeypatch, capsys):
    base, out = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    n_pa = 0
    with pysam.AlignmentFile(str(synth.bam)) as fin, \
            pysam.AlignmentFile(str(base / "star_Aligned.out.bam"), "wb", template=fin) as fout:
        for i, a in enumerate(fin.fetch(until_eof=True)):
            tags = [("CB", "c0"), ("UB", f"u{i // 4}")]  # each pa read (i % 4 == 0) and the next three records share a UMI
            if i % 4 == 0 or a.is_secondary:   # secondaries carry pa too and must not become pa rows
                tags.append(("pa", 20))
                n_pa += not a.is_secondary
            a.set_tags(tags)
            fout.write(a)
    stage = _stage14(monkeypatch)
    monkeypatch.setattr(stage, "SIRV_FASTA", synth.fasta)
    monkeypatch.setattr(stage, "SIRV_GTF", synth.gtf)
    monkeypatch.setattr(sys, "argv", ["14", "SYN", "--base", str(base), "--out", str(out), "--workers", "2"])
    stage.main()
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    # the same steps run sequentially, contigs in header order
    bam = out / "star_sorted.bam"
    seg = load_gene_segments(synth.gtf)
    with pysam.AlignmentFile(str(bam)) as fh, pysam.FastaFile(str(synth.fasta)) as fa:
        refs = [s.contig for s in fh.get_index_statistics() if s.mapped > 0]
        pa = pd.DataFrame([spike_pa_row(a, fa, seg) for r in refs for a in fh.fetch(r)
                           if not a.is_secondary and a.has_tag("pa")], columns=PA_COLUMNS)
        pa[["gx", "re"]] = pa[["gx", "re"]].fillna("").astype("str")
        mol = consolidate_sites(pa)
        by_k3 = defaultdict(list)
        for r in mol.itertuples():
            by_k3[(r.cb, r.ub, r.strand)].append((r.mol, r.ref, r.site, r.genes))
        exons = load_collapsed_exons(synth.gtf)
        members = pd.DataFrame([row for r in refs for row in member_rows(fh.fetch(r), r, by_k3, seg, exons)],
                               columns=MEMBER_COLUMNS)
    labeled = add_spike_labels(label_truth_molecules(mol, pa, members, load_region_segments(synth.gtf)),
                               annotated_ends(synth.gtf))
    reads = truth_reads(mol, pa, members)
    settings = all_settings()
    tx = load_transcripts(synth.gtf)
    flags, region = merge_scans([scan_reference(bam, synth.fasta, r, tx, settings, set(reads.qname))[:2] for r in refs],
                                [setting_tag(s) for s in settings])

    assert refs == ["chrA", "chrB"] and len(pa) == n_pa and len(mol) > 20 and len(members) > 20
    for name, exp in (("pa_sites", pa), ("molecules", mol), ("members", members), ("truth/molecules_labeled", labeled),
                      ("truth/truth_reads", reads), ("grid/truth_flags", flags), ("grid/region_counts", region)):
        pd.testing.assert_frame_equal(pd.read_parquet(out / f"{name}.parquet"), exp, check_dtype=False, obj=name)
    assert len(flags) == reads.qname.nunique() and flags.shape[1] == 4 + 396
    assert printed["pa_reads"] == n_pa and printed["molecules"] == len(mol)
    assert printed["truth_reads"] == {"pa": int((reads.role == "pa").sum()), "member": int((reads.role == "member").sum())}
    assert sum(sum(v.values()) for v in printed["by_module"].values()) == len(mol)
    # the spike BAM has no GX or RE tags: string columns of "", not null-typed parquet columns
    for name, cols in (("pa_sites", ["gx", "re"]), ("molecules", ["re"]), ("truth/molecules_labeled", ["re"])):
        schema, df = pq.read_schema(out / f"{name}.parquet"), pd.read_parquet(out / f"{name}.parquet")
        for c in cols:
            assert schema.field(c).type in (pyarrow.string(), pyarrow.large_string()) and set(df[c]) == {""}, (name, c)

    # the sorted BAM is kept when its index exists and is newer than star_Aligned.out.bam
    mtime = bam.stat().st_mtime_ns
    stage.main()
    assert bam.stat().st_mtime_ns == mtime
    # and sorted again when star_Aligned.out.bam is newer than the index (scripts/09_spikein_star.sh run again)
    src, bai = base / "star_Aligned.out.bam", Path(f"{bam}.bai")
    old = src.stat().st_mtime_ns - 10**9
    os.utime(bai, ns=(old, old))
    stage.main()
    assert bam.stat().st_mtime_ns != mtime and bai.stat().st_mtime_ns > src.stat().st_mtime_ns


def _contig(job):
    return job[1]


def test_per_contig_returns_header_order(monkeypatch):
    stage = _stage14(monkeypatch)
    stats = [SimpleNamespace(contig="a", mapped=1), SimpleNamespace(contig="b", mapped=5), SimpleNamespace(contig="c", mapped=3)]
    assert stage._per_contig(_contig, "x.bam", stats, 2) == ["a", "b", "c"]
```

- [ ] **Step 4: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_member_rows.py tests/test_spikein_truth.py tests/test_spikein_stage.py`
Expected: FAIL; the first error is `ImportError: cannot import name 'MEMBER_COLUMNS' from 'pafbench.molecules' (pafbench/molecules.py)`.

- [ ] **Step 5: Modify `pafbench/molecules.py`** (apply from the repository root)

```bash
git apply <<'PATCH'
diff --git a/pafbench/molecules.py b/pafbench/molecules.py
index 5fcc6c0..773a2c5 100644
--- a/pafbench/molecules.py
+++ b/pafbench/molecules.py
@@ -2,9 +2,13 @@ import numpy as np
 import pandas as pd
 
 from .config import MEMBER_CAP
+from .exons import member_distance
+from .genes import genes_at
+from .sites import last_aligned_base
 
 EDGES = [0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 50000, 100000, 1000000]
 MOL_COLUMNS = ["mol", "cb", "ub", "strand", "ref", "site", "n_pa", "spread", "a20", "down20", "up40", "hexamer", "genes", "re", "mapq_min"]
+MEMBER_COLUMNS = ["mol", "qname", "dist5", "dist3", "spliced", "dup", "genes", "gene_match", "dist5_exon", "dist3_exon", "dist_mode"]
 
 
 def consolidate_sites(pa, max_spread=10):
@@ -36,6 +40,29 @@ def gene_match(site_genes, member_genes):
     return bool(set(filter(None, site_genes.split(";"))) & set(filter(None, member_genes.split(";"))))
 
 
+def member_rows(records, ref, by_k3, gene_segments, exons_by_gene, collisions=None):
+    """Member candidate rows (MEMBER_COLUMNS) from the records of one reference: primary, no pa tag, CB and UB matching a
+    molecule's (cb, ub, strand) key in by_k3 {key: [(mol, ref, site, genes)]}. Hits on molecules of another reference
+    give no row; they are counted per mol in collisions when a dict is passed."""
+    rows = []
+    for a in records:
+        if a.is_secondary or a.is_supplementary or a.has_tag("pa") or not a.has_tag("CB") or not a.has_tag("UB"):
+            continue
+        hits = by_k3.get((a.get_tag("CB"), a.get_tag("UB"), "-" if a.is_reverse else "+"))
+        if not hits:
+            continue
+        for mol, mref, site, sgenes in hits:
+            if mref != ref:
+                if collisions is not None:
+                    collisions[mol] = collisions.get(mol, 0) + 1
+                continue
+            d5, d3 = member_distances(a, site)
+            e5, e3, mode = member_distance(a.get_blocks(), a.is_reverse, site, sgenes, exons_by_gene, (d5, d3))
+            mg = genes_at(gene_segments, ref, not a.is_reverse, last_aligned_base(a))
+            rows.append([mol, a.query_name, d5, d3, "N" in a.cigarstring, a.is_duplicate, mg, gene_match(sgenes, mg), e5, e3, mode])
+    return rows
+
+
 def member_density(members, n_molecules, edges=EDGES, d5="dist5", d3="dist3", include_spliced=False):
     """Members (d3 >= -25; unspliced only unless include_spliced) per bp per molecule, by d5 bin."""
     m = members[members[d3] >= -25]
PATCH
```

- [ ] **Step 6: Modify `scripts/06_molecules.py`** (apply from the repository root)

```bash
git apply <<'PATCH'
diff --git a/scripts/06_molecules.py b/scripts/06_molecules.py
index 7a7ab74..73b8edd 100644
--- a/scripts/06_molecules.py
+++ b/scripts/06_molecules.py
@@ -8,10 +8,9 @@ import pandas as pd
 import pysam
 
 from pafbench.config import GTF, LIBRARIES, OUT
-from pafbench.exons import load_collapsed_exons, member_distance
-from pafbench.genes import genes_at, load_gene_segments
-from pafbench.molecules import choose_cap, consolidate_sites, gene_match, member_density, member_distances
-from pafbench.sites import last_aligned_base
+from pafbench.exons import load_collapsed_exons
+from pafbench.genes import load_gene_segments
+from pafbench.molecules import MEMBER_COLUMNS, choose_cap, consolidate_sites, member_density, member_rows
 
 _SEG = None
 _BY_K3 = None
@@ -27,24 +26,10 @@ def _init(by_k3):
 
 def _work(job):
     bam_path, ref = job
-    rows, coll = [], defaultdict(int)
+    coll = {}
     with pysam.AlignmentFile(str(bam_path)) as bam:
-        for a in bam.fetch(ref):
-            if a.is_secondary or a.is_supplementary or a.has_tag("pa") or not a.has_tag("CB") or not a.has_tag("UB"):
-                continue
-            hits = _BY_K3.get((a.get_tag("CB"), a.get_tag("UB"), "-" if a.is_reverse else "+"))
-            if not hits:
-                continue
-            for mol, mref, site, sgenes in hits:
-                if mref != ref:
-                    coll[mol] += 1
-                    continue
-                d5, d3 = member_distances(a, site)
-                e5, e3, mode = member_distance(a.get_blocks(), a.is_reverse, site, sgenes, _EXONS, (d5, d3))
-                mg = genes_at(_SEG, ref, not a.is_reverse, last_aligned_base(a))
-                rows.append([mol, a.query_name, d5, d3, "N" in a.cigarstring, a.is_duplicate, mg, gene_match(sgenes, mg), e5, e3, mode])
-    cols = ["mol", "qname", "dist5", "dist3", "spliced", "dup", "genes", "gene_match", "dist5_exon", "dist3_exon", "dist_mode"]
-    return pd.DataFrame(rows, columns=cols), dict(coll)
+        rows = member_rows(bam.fetch(ref), ref, _BY_K3, _SEG, _EXONS, coll)
+    return pd.DataFrame(rows, columns=MEMBER_COLUMNS), coll
 
 
 def main():
PATCH
```

- [ ] **Step 7: Modify `pafbench/spikein.py`** (apply from the repository root)

```bash
git apply <<'PATCH'
diff --git a/pafbench/spikein.py b/pafbench/spikein.py
index ef6d4b0..14b1510 100644
--- a/pafbench/spikein.py
+++ b/pafbench/spikein.py
@@ -1,4 +1,17 @@
 import re
+from collections import defaultdict
+
+import numpy as np
+import pandas as pd
+
+from .bd67 import terminal_a_run
+from .genes import genes_at
+from .sites import PA_COLUMNS, has_hexamer, pa_row, sense_context
+from .stream import merge_regions
+from .truth import LABELS
+
+MODULES = ("ERCC", "SIRV_E0", "long_SIRV")
+SPIKE_LABELS = ("tail", "internal", "other")
 
 
 def tag_survival(before, after, tags=("CB", "UB", "pa", "ts")):
@@ -17,3 +30,77 @@ def module_of(contig):
     if re.fullmatch(r"SIRV[1-7]", contig):
         return "SIRV_E0"
     return "long_SIRV"
+
+
+def annotated_ends(gtf):
+    """{(contig, strd): sorted unique int64 ndarray} of transcript 3' ends in the priming_site convention: the GTF end on
+    plus (0-based exclusive end), the GTF start - 1 on minus (0-based last cDNA base)."""
+    ends = defaultdict(list)
+    with open(gtf) as fh:
+        for line in fh:
+            if line.startswith("#"):
+                continue
+            f = line.split("\t")
+            if f[2] != "transcript":
+                continue
+            plus = f[6] == "+"
+            ends[(f[0], plus)].append(int(f[4]) if plus else int(f[3]) - 1)
+    return {k: np.unique(np.asarray(v, np.int64)) for k, v in ends.items()}
+
+
+def label_spike_site(site, strd, ends, tail_tol=10, internal_min=30):
+    """(label, dist) of a site against the annotated ends of its (contig, strand); d > 0 when the site lies upstream of an
+    end in transcript sense. tail: some |d| <= tail_tol; internal: else some d >= internal_min; other: neither, or no ends.
+    dist is the d with the smallest |d| (ties: the positive one), None without ends."""
+    if ends is None or len(ends) == 0:
+        return "other", None
+    d = ends - site if strd else site - ends
+    if (np.abs(d) <= tail_tol).any():
+        label = "tail"
+    elif (d >= internal_min).any():
+        label = "internal"
+    else:
+        label = "other"
+    return label, int(d[np.lexsort((d < 0, np.abs(d)))[0]])
+
+
+def spike_pa_row(a, fa, segments):
+    """sites.pa_row with the site at the start of the read's terminal A run (bd67.terminal_a_run, one non-A allowed,
+    aligned A's included), since spike-in reads keep their poly(A); context, a20, hexamer and genes (at the last cDNA
+    base: site - 1 on plus, site on minus) follow that site."""
+    row = dict(zip(PA_COLUMNS, pa_row(a, fa, segments)))
+    site = terminal_a_run(a, tolerance=1)[0]
+    down, up = sense_context(fa, a.reference_name, site, a.is_reverse)
+    last = site if a.is_reverse else site - 1
+    row.update(site=site, down20=down, up40=up, a20=down.count("A"), hexamer=has_hexamer(up),
+               genes=genes_at(segments, a.reference_name, not a.is_reverse, last))
+    return [row[c] for c in PA_COLUMNS]
+
+
+def add_spike_labels(mol, ends):
+    """mol plus spike_label and dist_to_end (label_spike_site against ends, float with NaN when the strand has no ends)
+    and module (module_of the contig)."""
+    res = [label_spike_site(s, st == "+", ends.get((r, st == "+"))) for r, st, s in zip(mol.ref, mol.strand, mol.site)]
+    dist = np.array([np.nan if d is None else d for _, d in res], np.float64)
+    return mol.assign(spike_label=[lab for lab, _ in res], dist_to_end=dist, module=[module_of(r) for r in mol.ref])
+
+
+def merge_scans(scans, tags):
+    """(truth_flags, region_counts) from per-contig stream.scan_reference results [(truth_df, region_df)], merged as
+    stream.merge_parts does: truth rows in the given contig order; region rows by stream.merge_regions."""
+    return pd.concat([t for t, _ in scans], ignore_index=True), merge_regions([r for _, r in scans], tags)
+
+
+def spike_summary(pa, labeled, members, reads):
+    """Counts printed by stage 14: pa reads, molecules by module and spike_label, spike_label against the A-count label,
+    member candidate rows, truth reads by role."""
+    by_mod = labeled.groupby(["module", "spike_label"]).size()
+    cross = labeled.groupby(["spike_label", "label"]).size()
+    return {
+        "pa_reads": len(pa),
+        "molecules": len(labeled),
+        "by_module": {m: {s: int(by_mod.get((m, s), 0)) for s in SPIKE_LABELS} for m in MODULES},
+        "spike_label_by_label": {s: {k: int(cross.get((s, k), 0)) for k in LABELS} for s in SPIKE_LABELS},
+        "members": len(members),
+        "truth_reads": {k: int((reads.role == k).sum()) for k in ("pa", "member")},
+    }
PATCH
```

- [ ] **Step 8: Create `scripts/14_spikein_truth.py`**

```python
"""Stage 14: spike-in truth sets and grid flags (spec 4.2) from the STAR realignment of unplaced unmapped reads to SIRV Set 4.

Reads <base>/star_Aligned.out.bam (scripts/09_spikein_star.sh) and writes under <out>: star_sorted.bam (+ .bai, kept when
the index exists and is newer than star_Aligned.out.bam), pa_sites.parquet, molecules.parquet, members.parquet,
truth/molecules_labeled.parquet (stage 12 columns plus spike_label, dist_to_end, module), truth/truth_reads.parquet,
grid/truth_flags.parquet, grid/region_counts.parquet. gx and re are "" when the read has no GX or RE tag (none has one
after the realignment). The grid is scanned in full on every run (no per-contig parts to resume), so it always matches
this run's truth reads.
"""
import argparse
import json
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import pysam

from pafbench.config import OUT, SIRV_FASTA, SIRV_GTF
from pafbench.exons import load_collapsed_exons
from pafbench.flags import setting_tag
from pafbench.genes import load_gene_segments
from pafbench.grid import all_settings
from pafbench.molecules import MEMBER_COLUMNS, consolidate_sites, member_rows
from pafbench.regions import load_region_segments
from pafbench.sites import PA_COLUMNS
from pafbench.spikein import add_spike_labels, annotated_ends, merge_scans, spike_pa_row, spike_summary
from pafbench.stream import scan_reference
from pafbench.transcripts import load_transcripts
from pafbench.truth import check_pa_counts, label_truth_molecules, truth_reads

_G = {}  # set in the parent before each pool is forked


def _pa(job):
    bam, ref = job
    with pysam.AlignmentFile(str(bam)) as fh, pysam.FastaFile(str(SIRV_FASTA)) as fa:
        return [spike_pa_row(a, fa, _G["seg"]) for a in fh.fetch(ref)
                if not (a.is_secondary or a.is_supplementary) and a.has_tag("pa")]


def _members(job):
    bam, ref = job
    with pysam.AlignmentFile(str(bam)) as fh:
        return member_rows(fh.fetch(ref), ref, _G["by_k3"], _G["seg"], _G["exons"])


def _grid(job):
    bam, ref = job
    truth, region, _ = scan_reference(bam, SIRV_FASTA, ref, _G["tx"], _G["settings"], _G["truth"])
    return truth, region


def _per_contig(fn, bam, stats, workers):
    """fn over the contigs (largest first), results in BAM header order."""
    jobs = [(bam, s.contig) for s in sorted(stats, key=lambda s: -s.mapped)]
    with get_context("fork").Pool(workers) as pool:
        res = dict(zip([j[1] for j in jobs], pool.map(fn, jobs, chunksize=1)))
    return [res[s.contig] for s in stats]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--base", type=Path, help="dir with star_Aligned.out.bam (default OUT/LIB/spikein)")
    ap.add_argument("--out", type=Path, help="output dir (default: base)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.monotonic()
    base = args.base or OUT / args.library / "spikein"
    out = args.out or base
    out.mkdir(parents=True, exist_ok=True)
    bam, src = out / "star_sorted.bam", base / "star_Aligned.out.bam"
    bai = Path(f"{bam}.bai")
    if not bai.exists() or src.stat().st_mtime_ns > bai.stat().st_mtime_ns:
        pysam.sort("-@", str(args.workers), "-o", str(bam), str(src))
        pysam.index(str(bam))
    with pysam.AlignmentFile(str(bam)) as fh:
        stats = [s for s in fh.get_index_statistics() if s.mapped > 0]

    _G["seg"] = load_gene_segments(SIRV_GTF)
    pa = pd.DataFrame([r for rows in _per_contig(_pa, bam, stats, args.workers) for r in rows], columns=PA_COLUMNS)
    pa[["gx", "re"]] = pa[["gx", "re"]].fillna("").astype("str")
    mol = consolidate_sites(pa)
    by_k3 = defaultdict(list)
    for r in mol.itertuples():
        by_k3[(r.cb, r.ub, r.strand)].append((r.mol, r.ref, r.site, r.genes))
    _G.update(by_k3=dict(by_k3), exons=load_collapsed_exons(SIRV_GTF))
    rows = [r for part in _per_contig(_members, bam, stats, args.workers) for r in part]
    members = pd.DataFrame(rows, columns=MEMBER_COLUMNS)
    members["spliced"] = members["spliced"].astype(bool)

    labeled = add_spike_labels(label_truth_molecules(mol, pa, members, load_region_segments(SIRV_GTF)),
                               annotated_ends(SIRV_GTF))
    reads = truth_reads(mol, pa, members)
    check_pa_counts(mol, reads)

    settings = all_settings()
    _G.update(tx=load_transcripts(SIRV_GTF), settings=settings, truth=set(reads.qname))
    truth_flags, region = merge_scans(_per_contig(_grid, bam, stats, args.workers), [setting_tag(s) for s in settings])

    (out / "truth").mkdir(exist_ok=True)
    (out / "grid").mkdir(exist_ok=True)
    pa.to_parquet(out / "pa_sites.parquet", index=False)
    mol.to_parquet(out / "molecules.parquet", index=False)
    members.to_parquet(out / "members.parquet", index=False)
    labeled.to_parquet(out / "truth" / "molecules_labeled.parquet", index=False)
    reads.to_parquet(out / "truth" / "truth_reads.parquet", index=False)
    truth_flags.to_parquet(out / "grid" / "truth_flags.parquet", index=False)
    region.to_parquet(out / "grid" / "region_counts.parquet", index=False)
    print(json.dumps({"library": args.library} | spike_summary(pa, labeled, members, reads)
                     | {"truth_flag_rows": len(truth_flags), "seconds": round(time.monotonic() - t0, 1)}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 9: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `170 passed, 2 warnings`.

- [ ] **Step 10: Realign 1419's unmapped reads to SIRV Set 4**

Run: `.venv/bin/python -m pafbench.memwatch /data/element_runs/polyafilter_bench/1419/runs/09_spikein_star.mem.json -- scripts/09_spikein_star.sh 1419`
Expected: `/data/element_runs/polyafilter_bench/1419/spikein/star_Log.final.out` with about 85% uniquely mapped reads (1417: 86.3%).

- [ ] **Step 11: Stage 14 on 1417 and 1419**

Run: `for L in 1417 1419; do .venv/bin/python -m pafbench.memwatch /data/element_runs/polyafilter_bench/$L/runs/14_spikein_truth.mem.json -- .venv/bin/python scripts/14_spikein_truth.py $L; done`
Expected for 1417 (about 45 s, 2.7 GiB): `"pa_reads": 17772, "molecules": 15361`; by module ERCC tail 3,922 / internal 1,009 / other 129, SIRV_E0 4,007 / 2,291 / 8, long_SIRV 538 / 3,453 / 4; members 10,705; truth reads pa 15,471 and member 10,419. Against the 8/4 label, internal sites are 88.7% templated; tail sites are mostly non-templated, but at ERCC and long-SIRV tails this is forced because their transcripts end at the contig end (fewer than 20 nt after the site).

- [ ] **Step 12: Commit**

```bash
git add tests/test_member_rows.py tests/test_spikein_truth.py tests/test_spikein_stage.py pafbench/molecules.py scripts/06_molecules.py pafbench/spikein.py scripts/14_spikein_truth.py
git -c user.name="Simone Picelli" -c user.email="simone.picelli@iob.ch" commit -m "Stage 14: spike-in truth sets (site at the terminal A run start, labels by distance to annotated SIRV and ERCC ends) and grid flags; member rows shared with stage 06"
```

---

### Task 6: Evaluation, parameter choice and burden (stages 17 and 18)

**Files:**
- Create: `tests/test_evaluate.py`
- Create: `tests/test_polyasite.py`
- Create: `tests/test_burden.py`
- Create: `pafbench/evaluate.py`
- Create: `pafbench/polyasite.py`
- Create: `pafbench/burden.py`
- Create: `scripts/17_evaluate.py`
- Create: `scripts/18_burden.py`

**Interfaces:**
- Consumes: Task 0 `grid.grid`, `grid.published`, `labels.label_molecules`, `regions.REGION_NAMES`, `config.SENS_CAPS`, `config.NUCLEI`, `config.CELLS`; Task 2 `truth/*`; Task 3 `grid/truth_flags.parquet`, `grid/region_counts.parquet`; Task 5 `spikein/truth/*`, `spikein/grid/truth_flags.parquet`.
- Produces:
  - `pafbench.evaluate.tag_setting(tag)`
  - `pafbench.evaluate.group_sum(groups, n_groups, M, weight=None, chunk=64)`
  - `pafbench.evaluate.expr_tertiles(mol)`
  - `pafbench.evaluate.units(mol_labeled, truth_reads, truth_flags, tags, cap, label='label', pos='templated', neg='nontemplated', amb='ambiguous')`
  - `pafbench.evaluate.relabel(u, cutoffs)`
  - `pafbench.evaluate.confusion(removed, label, weight=None, pos='templated', neg='nontemplated')`
  - `pafbench.evaluate.rates(tp, fn, tn, fp)`
  - `pafbench.evaluate.evaluate(u, lib, cap, cutoffs_name='8/4', levels=LEVELS, kinds=None)`
  - `pafbench.evaluate.spikein_metrics(u, tags)`
  - `pafbench.evaluate.select_setting(metrics, removed_by_tag, introns, lib='1417', cap=1000, cutoffs='8/4', min_spec=0.95)`
  - `pafbench.evaluate.gene_clusters(mol)`
  - `pafbench.evaluate.resample_counts(n_clusters, n, seed)`
  - `pafbench.evaluate.interval(values)`
  - `pafbench.evaluate.bootstrap(u, tags, level='molecule', n=1000, seed=0)`
  - `pafbench.evaluate.ambiguous_tail_share(mol_labeled)`
  - `pafbench.evaluate.hexamer_check(u, tags)`
  - `pafbench.polyasite.load_polyasite(path, min_score=None)`
  - `pafbench.polyasite.nearest_signed(sites, strd, ends)`
  - `pafbench.polyasite.site_distances(mol, atlas, upstream=0)`
  - `pafbench.polyasite.distance_bin(d)`
  - `pafbench.burden.share_table(mol, by=None)`
  - `pafbench.burden.share_ci(mol, by=None, n=1000, seed=0, of='share_all')`
  - `pafbench.burden.site_labels(mol)`
  - `pafbench.burden.same_site(a, b, tol=5)`
  - `pafbench.burden.same_gene(a, b, min_n=20)`
  - `pafbench.burden.selection_bias(mol, region_counts)`
  - `pafbench.burden.proximity_table(mol, dist, null)`
  - `pafbench.burden.near_shares(table)`
- Produces (data): `OUT/eval/metrics.parquet` (`lib, cap, cutoffs, tag, covlen, min_len, mism, introns, level, stratum_kind, stratum, tp, fn, tn, fp, n_amb, amb_removed, sens, spec, prec`), `selection.json` (per mode: chosen tag or null with the frontier, held-out metrics, published settings, cap sensitivity, cut-off sweep, ambiguous tail share), `bootstrap.parquet`, `hexamer_check.parquet`, `spikein_metrics.parquet`; `OUT/burden/templated_share.parquet`, `same_site.parquet`, `same_gene.parquet`, `selection_bias.parquet`, `polyasite.parquet` (group, label, bin, n, with a 1 kb-upstream chance row), `summary.json`.

Spec 7 fixes the units, the metrics and the selection rule before any result is seen; this task implements them as written. Two properties to keep in mind when reading results: polyAfilter never removes intergenic or antisense molecules, which caps overall sensitivity (on the 1417 build mock, 12.5% of templated molecules with members are intergenic), and at 8/4 two thirds of the positives have no A stretch that README-type settings can detect (milestone 1 note), so every metric is also reported by A band. Stage 18 also reports how close labeled sites sit to PolyASite 3.0 clusters (spec 4.1: reported, not used to validate labels), next to the share expected by chance at the atlas's density.

- [ ] **Step 1: Write the failing tests in `tests/test_evaluate.py`**

```python
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pafbench.config import REPO
from pafbench.evaluate import (METRIC_COLUMNS, ambiguous_tail_share, bootstrap, confusion, evaluate, expr_tertiles,
                               gene_clusters, hexamer_check, rates, relabel, resample_counts, select_setting, spikein_metrics,
                               units)
from pafbench.flags import setting_tag
from pafbench.grid import published

T0, T1, T2 = "c300_l10_m1_i", "c375_l3_m2_i", "c300_lnone_m0_i"
TAGS = [T0, T1, T2]

MOL_COLS = ["mol", "label", "a20", "a_band", "region", "gene", "has_ts", "hexamer", "ref", "strand", "site"]
MOLS = [
    (0, "templated", 15, "14-20", "3UTR", "gA", False, False, "chrA", "+", 1000),
    (1, "templated", 9, "8-9", "CDS", "gA", True, True, "chrA", "+", 900),
    (2, "nontemplated", 2, "0-4", "3UTR", "gB", False, True, "chrA", "-", 5000),
    (3, "ambiguous", 6, "5-7", "intron", "gB", False, False, "chrA", "-", 5600),
    (4, "nontemplated", 0, "0-4", "intergenic", "", False, False, "chrA", "+", 12345),
    (5, "templated", 12, "10-13", "3UTR", "gC", False, False, "chrB", "+", 300),
    (6, "templated", 20, "14-20", "3UTR", "gA", False, True, "chrA", "+", 1200),
]
# members: qname, mol, dist5_exon, dist_mode, spliced
MEMBERS = [
    ("r0a", 0, 100, "exon", False), ("r0b", 0, 300, "genome", False), ("r0c", 0, 900, "exon", False),
    ("r1a", 1, 400, "exon", True), ("r1b", 1, 200, "genome", False),
    ("r2a", 2, 150, "exon", False), ("r3a", 3, 250, "genome", False),
    ("r4a", 4, 1500, "genome", False), ("r5a", 5, 100, "exon", False),
]
# primary flags per qname for T0, T1 (T2 = remove-all control, True everywhere); r5a is missing
FLAGS = {"p0": (1, 0), "r0a": (1, 1), "r0b": (1, 0), "r0c": (1, 0), "p1": (0, 0), "r1a": (1, 0), "r1b": (1, 0),
         "p2": (0, 1), "r2a": (0, 1), "p3": (1, 0), "r3a": (1, 0), "p4": (0, 0), "r4a": (1, 0), "p5": (0, 0), "p6": (0, 0)}


def _tables():
    mol = pd.DataFrame(MOLS, columns=MOL_COLS)
    pa = pd.DataFrame({"qname": [f"p{m}" for m in range(7)], "mol": range(7), "role": "pa", "dist5_exon": np.nan,
                       "dist_mode": "", "spliced": False})
    mem = pd.DataFrame([(q, m, "member", float(d), mode, s) for q, m, d, mode, s in MEMBERS], columns=pa.columns)
    tr = pd.concat([pa, mem], ignore_index=True)
    rows = [(q, "chrA", 0, 0, bool(a), bool(b), True) for q, (a, b) in FLAGS.items()]
    rows.append(("r2a", "chrA", 0, 256, True, False, True))  # secondary record of r2a: must be ignored
    fl = pd.DataFrame(rows, columns=["qname", "ref", "pos", "flag"] + TAGS)
    return mol, tr, fl


def _row(m, level, kind, stratum, tag):
    r = m[(m.level == level) & (m.stratum_kind == kind) & (m.stratum == stratum) & (m.tag == tag)]
    assert len(r) == 1
    return r.iloc[0]


def _counts(r):
    return [r.tp, r.fn, r.tn, r.fp, r.n_amb, r.amb_removed]


def test_confusion_and_rates():
    removed = np.array([[1, 0], [0, 0], [1, 1], [0, 1], [1, 1]], bool)
    label = np.array(["templated", "templated", "nontemplated", "nontemplated", "ambiguous"], object)
    tp, fn, tn, fp, n_amb, amb_rem = confusion(removed, label)
    # tag 0: positives removed 1 of 2; negatives removed 1 of 2; the ambiguous one removed
    assert tp.tolist() == [1, 0] and fn.tolist() == [1, 2] and tn.tolist() == [1, 0] and fp.tolist() == [1, 2]
    assert n_amb.tolist() == [1, 1] and amb_rem.tolist() == [1, 1]
    sens, spec, prec = rates(tp, fn, tn, fp)
    assert sens.tolist() == [0.5, 0.0] and spec.tolist() == [0.5, 0.0] and prec.tolist() == [0.5, 0.0]
    w = np.array([0.5, 0.25, 1.0, 1.0, 2.0])
    tp, fn, tn, fp, n_amb, amb_rem = confusion(removed, label, w)
    assert tp.tolist() == [0.5, 0.0] and fn.tolist() == [0.25, 0.75] and n_amb.tolist() == [2.0, 2.0]
    # empty denominators give NaN: no positives, no negatives, nothing removed
    sens, spec, prec = rates(np.array([0]), np.array([0]), np.array([0]), np.array([0]))
    assert math.isnan(sens[0]) and math.isnan(spec[0]) and math.isnan(prec[0])
    sens, spec, prec = rates(np.array([0]), np.array([3]), np.array([2]), np.array([0]))
    assert sens[0] == 0.0 and spec[0] == 1.0 and math.isnan(prec[0])
    # other positive / negative values (spike-ins)
    tp, fn, tn, fp, n_amb, _ = confusion(removed, np.array(["internal", "tail", "tail", "other", "internal"], object),
                                         pos="internal", neg="tail")
    assert tp.tolist() == [2, 1] and fn.tolist() == [0, 1] and tn.tolist() == [1, 1] and fp.tolist() == [1, 1]
    assert n_amb.tolist() == [1, 1]


def test_units_reads_molecules_and_missing():
    mol, tr, fl = _tables()
    u = units(mol, tr, fl, TAGS, cap=1000)
    # m4 has its only member beyond 1000 nt, m6 has no member, m5 loses r5a (absent from the flags) and drops out
    assert u.mols.mol.tolist() == [0, 1, 2, 3]
    assert u.missing_reads == 1 and u.dropped_mols == 1
    assert u.mols.n_reads.tolist() == [4, 3, 2, 2]
    assert u.K[:, 0].tolist() == [4, 2, 0, 2] and u.K[:, 1].tolist() == [1, 0, 2, 0] and u.K[:, 2].tolist() == [4, 3, 2, 2]
    assert sorted(u.reads.qname) == ["r0a", "r0b", "r0c", "r1a", "r1b", "r2a", "r3a"]
    r = dict(zip(u.reads.qname, u.R[:, 0]))
    assert r["r2a"] == False and r["r1a"] == True  # noqa: E712  (primary row of r2a, not the secondary one)
    r1a = u.reads[u.reads.qname == "r1a"].iloc[0]
    assert (r1a.label, r1a.a_band, r1a.region, r1a.gene, r1a.has_ts, r1a.dist_mode, r1a.spliced) == \
        ("templated", "8-9", "CDS", "gA", True, "exon", True)
    u2 = units(mol, tr, fl, TAGS, cap=2000)
    assert u2.mols.mol.tolist() == [0, 1, 2, 3, 4] and u2.mols.n_reads.tolist()[-1] == 2 and u2.K[-1].tolist() == [1, 0, 2]
    # the cap is inclusive: r0c sits at dist5_exon 900
    assert units(mol, tr, fl, TAGS, cap=900).mols.n_reads.tolist()[0] == 4
    assert units(mol, tr, fl, TAGS, cap=899).mols.n_reads.tolist()[0] == 3


def test_expr_tertiles_match_qcut():
    mol, _, _ = _tables()
    # molecules per gene: gA 3, gB 2, gC 1 -> gC low, gB mid, gA high; no gene -> intergenic
    assert expr_tertiles(mol).tolist() == ["high", "high", "mid", "mid", "intergenic", "low", "high"]
    genes = [f"g{i}" for i in range(5) for _ in range(i + 1)]  # g0 x1 ... g4 x5
    t = expr_tertiles(pd.DataFrame({"gene": genes}))
    ref = pd.qcut(pd.Series(np.arange(1, 6)), 3, labels=["low", "mid", "high"]).astype(str).tolist()
    assert [t[genes.index(f"g{i}")] for i in range(5)] == ref == ["low", "low", "mid", "high", "high"]
    assert expr_tertiles(pd.DataFrame({"gene": ["g1", "g1"]})).tolist() == ["low", "low"]


def test_levels_all_reads_versus_majority_and_weighted():
    mol, tr, fl = _tables()
    m = evaluate(units(mol, tr, fl, TAGS, cap=1000), "1417", 1000)
    # molecule rule: m0 removed (4/4), m1 not (2/3), m2 kept, m3 (ambiguous) removed
    assert _counts(_row(m, "molecule", "all", "all", T0)) == [1, 1, 1, 0, 1, 1]
    assert _counts(_row(m, "molecule", "all", "all", T1)) == [0, 2, 0, 1, 1, 0]
    # majority rule: m1 has 2 of 3 reads removed, so it counts as removed
    assert _counts(_row(m, "molecule_majority", "all", "all", T0)) == [2, 0, 1, 0, 1, 1]
    assert _counts(_row(m, "molecule_majority", "all", "all", T1)) == [0, 2, 0, 1, 1, 0]
    # weights 1 / n_reads: tp 1/4 (m0), fn 1/3 (m1), tn 1/2 (m2), ambiguous 1/2 removed
    w = _row(m, "molecule_weighted", "all", "all", T0)
    assert np.allclose(_counts(w), [0.25, 1 / 3, 0.5, 0, 0.5, 0.5]) and w.sens == pytest.approx(3 / 7) and w.spec == 1.0
    r = _row(m, "read", "all", "all", T1)
    assert _counts(r) == [1, 4, 0, 1, 1, 0] and r.sens == 0.2 and r.spec == 0.0 and r.prec == 0.5
    c = _row(m, "molecule", "all", "all", T2)
    assert _counts(c) == [2, 0, 0, 1, 1, 1] and c.prec == pytest.approx(2 / 3)
    m2 = evaluate(units(mol, tr, fl, TAGS, cap=2000), "1417", 2000)
    assert _counts(_row(m2, "molecule", "all", "all", T0))[2:4] == [2, 0]  # m4: 1 of 2 reads removed, kept
    assert _counts(_row(m2, "read", "all", "all", T0))[2:4] == [1, 1]  # r4a removed at read level
    assert _counts(_row(m2, "molecule_majority", "all", "all", T0))[2:4] == [2, 0]  # m4: half its reads is not a majority


def test_strata_and_columns():
    mol, tr, fl = _tables()
    m = evaluate(units(mol, tr, fl, TAGS, cap=1000), "1417", 1000)
    assert list(m.columns) == METRIC_COLUMNS
    assert str(m.min_len.dtype) == "Int64"
    ctrl = m[m.tag == T2].iloc[0]
    assert ctrl.covlen == 300 and pd.isna(ctrl.min_len) and ctrl.mism == 0 and ctrl.introns == True  # noqa: E712
    assert _row(m, "molecule", "all", "all", T0).min_len == 10
    assert set(m.cutoffs) == {"8/4"} and set(m.lib) == {"1417"} and set(m.cap) == {1000}
    assert _counts(_row(m, "molecule", "region", "3UTR", T0)) == [1, 0, 1, 0, 0, 0]
    assert _counts(_row(m, "molecule", "region", "CDS", T0)) == [0, 1, 0, 0, 0, 0]
    assert _counts(_row(m, "molecule", "region", "intron", T0)) == [0, 0, 0, 0, 1, 1]
    assert math.isnan(_row(m, "molecule", "region", "intergenic", T0).sens)
    assert _counts(_row(m, "molecule", "a_band", "8-9", T0)) == [0, 1, 0, 0, 0, 0]
    assert _counts(_row(m, "molecule", "expr_tertile", "high", T0)) == [1, 1, 0, 0, 0, 0]
    assert _counts(_row(m, "molecule", "expr_tertile", "mid", T0)) == [0, 0, 1, 0, 1, 1]
    assert _counts(_row(m, "molecule", "expr_tertile", "low", T0)) == [0] * 6
    assert _counts(_row(m, "molecule", "has_ts", "True", T0)) == [0, 1, 0, 0, 0, 0]
    assert _counts(_row(m, "read", "dist_mode", "exon", T1)) == [1, 2, 0, 1, 0, 0]
    assert _counts(_row(m, "read", "dist_mode", "genome", T1)) == [0, 2, 0, 0, 1, 0]
    assert _counts(_row(m, "read", "spliced", "True", T0)) == [1, 0, 0, 0, 0, 0]
    assert m[(m.level != "read") & m.stratum_kind.isin(["dist_mode", "spliced"])].empty
    per_level = m[m.tag == T0].groupby("level").size().to_dict()
    # all 1 + a_band 5 + region 7 + expr 4 + has_ts 2 = 19; read adds dist_mode 2 + spliced 2
    assert per_level == {"molecule": 19, "molecule_majority": 19, "molecule_weighted": 19, "read": 23}
    only = evaluate(units(mol, tr, fl, TAGS, cap=1000), "1417", 1000, levels=("molecule",), kinds=("all",))
    assert len(only) == 3 and _counts(only[only.tag == T0].iloc[0]) == [1, 1, 1, 0, 1, 1]


def test_relabel_changes_labels_only():
    mol, tr, fl = _tables()
    u = relabel(units(mol, tr, fl, TAGS, cap=1000), {"templated_min": 10, "nontemplated_max": 6})
    # a20 15, 9, 2, 6 -> templated, ambiguous, nontemplated, nontemplated
    assert u.mols.label.tolist() == ["templated", "ambiguous", "nontemplated", "nontemplated"]
    assert u.reads.set_index("qname").label["r1a"] == "ambiguous"
    m = evaluate(u, "1417", 1000, "10/6", levels=("molecule",), kinds=("all",))
    assert _counts(m[m.tag == T0].iloc[0]) == [1, 0, 1, 1, 1, 0]


def _metrics(rows):
    return pd.DataFrame([dict(lib=lib, cap=cap, cutoffs="8/4", level=level, stratum_kind="all", stratum="all", tag=tag,
                              sens=se, spec=sp, prec=0.9) for lib, cap, level, tag, se, sp in rows])


SEL_ROWS = [
    ("1417", 1000, "molecule", "c300_l10_m1_i", 0.6, 0.96),
    ("1417", 1000, "molecule", "c300_l9_m1_i", 0.7, 0.97),
    ("1417", 1000, "molecule", "c450_l9_m1_i", 0.7, 0.99),
    ("1417", 1000, "molecule", "c300_l3_m2_i", 0.9, 0.90),
    ("1417", 1000, "molecule", "c300_lnone_m0_i", 0.99, 0.99),  # control: never a candidate
    ("1417", 1000, "molecule", "c275_l9_m1_i", 0.95, 0.99),  # published but not on the grid
    ("1417", 1000, "molecule", "c300_l9_m1_e", 0.99, 0.99),  # other mode
    ("1419", 1000, "molecule", "c300_l3_m2_i", 0.99, 0.99),  # other library
    ("1417", 500, "molecule", "c300_l3_m2_i", 0.99, 0.99),  # other cap
    ("1417", 1000, "read", "c300_l3_m2_i", 0.99, 0.99),  # other level
]


def test_select_setting():
    m = _metrics(SEL_ROWS)
    removed = {"c300_l9_m1_i": 500, "c450_l9_m1_i": 400, "c300_l10_m1_i": 100, "c300_l3_m2_i": 900}
    s = select_setting(m, removed, introns=True)
    # c300_l9 and c450_l9 tie on sensitivity 0.7; c450 removes fewer records
    assert s["tag"] == "c450_l9_m1_i" and s["sens"] == 0.7 and s["spec"] == 0.99 and s["prec"] == 0.9
    assert s["n_candidates"] == 4 and s["n_qualifying"] == 3
    assert s["frontier"] == [["c300_l3_m2_i", 0.9, 0.90], ["c450_l9_m1_i", 0.7, 0.99]]
    s = select_setting(m, {"c300_l9_m1_i": 400, "c450_l9_m1_i": 400}, introns=True)
    assert s["tag"] == "c300_l9_m1_i"  # same removed count: tag name decides
    s = select_setting(m, removed, introns=True, min_spec=0.995)
    assert s["tag"] is None and s["sens"] is None and s["n_qualifying"] == 0
    assert s["frontier"] == [["c300_l3_m2_i", 0.9, 0.90], ["c450_l9_m1_i", 0.7, 0.99]]
    # specificity exactly at the threshold qualifies
    edge = _metrics(SEL_ROWS + [("1417", 1000, "molecule", "c600_l20_m0_i", 0.8, 0.95)])
    s = select_setting(edge, removed, introns=True)
    assert s["tag"] == "c600_l20_m0_i" and s["n_candidates"] == 5 and s["n_qualifying"] == 4
    assert s["frontier"] == [["c300_l3_m2_i", 0.9, 0.90], ["c600_l20_m0_i", 0.8, 0.95], ["c450_l9_m1_i", 0.7, 0.99]]
    e = select_setting(m, {}, introns=False)
    assert e["tag"] == "c300_l9_m1_e" and e["n_candidates"] == 1


def _boot_units(genes, removed, labels):
    """One pa read and one member per molecule, both removed or both kept, so a molecule is removed iff `removed`."""
    n = len(genes)
    mol = pd.DataFrame({"mol": range(n), "label": labels, "a20": 10, "a_band": "10-13", "region": "3UTR", "gene": genes,
                        "has_ts": False, "hexamer": False, "ref": "chrA", "strand": "+", "site": np.arange(n) * 10_000})
    q = [f"p{i}" for i in range(n)] + [f"r{i}" for i in range(n)]
    tr = pd.DataFrame({"qname": q, "mol": list(range(n)) * 2, "role": ["pa"] * n + ["member"] * n,
                       "dist5_exon": [np.nan] * n + [100.0] * n, "dist_mode": [""] * n + ["exon"] * n, "spliced": False})
    fl = pd.DataFrame({"qname": q, "ref": "chrA", "pos": 0, "flag": 0, T0: list(removed) * 2})
    return units(mol, tr, fl, [T0], cap=1000)


def test_gene_clusters():
    mol = pd.DataFrame({"gene": ["gA", "", ""], "ref": ["chr1", "chr1", "chr2"], "strand": ["+", "-", "+"],
                        "site": [5, 12345, 9999]})
    assert gene_clusters(mol).tolist() == ["gA", "chr1:-:1", "chr2:+:0"]


def test_bootstrap_matches_brute_force_and_is_reproducible():
    rng = np.random.default_rng(1)
    genes = [f"g{i % 7}" for i in range(60)] + ["", ""]
    labels = rng.choice(["templated", "nontemplated", "ambiguous"], 62, p=[0.6, 0.3, 0.1])
    removed = rng.random(62) < 0.5
    u = _boot_units(genes, removed, labels)
    b = bootstrap(u, [T0], n=200, seed=3)
    assert b.equals(bootstrap(u, [T0], n=200, seed=3))
    assert not b.equals(bootstrap(u, [T0], n=200, seed=4))
    assert list(b.columns) == ["tag", "metric", "estimate", "lo", "hi", "n_boot", "n_clusters"]
    assert b.metric.tolist() == ["sens", "spec", "prec"] and set(b.n_clusters) == {9}  # 7 genes + 2 intergenic loci
    # brute force: weight each molecule by the draw count of its cluster (clusters in sorted order)
    cl = gene_clusters(u.mols)
    names = np.unique(cl)
    counts = resample_counts(len(names), 200, 3)
    w = counts[:, np.searchsorted(names, cl)].astype(float)
    rem = (u.K[:, 0] == u.mols.n_reads.to_numpy())
    lab = u.mols.label.to_numpy()
    P, N = lab == "templated", lab == "nontemplated"
    sens = (w * (P & rem)).sum(1) / (w * P).sum(1)
    spec = (w * (N & ~rem)).sum(1) / (w * N).sum(1)
    for metric, vals in (("sens", sens), ("spec", spec)):
        r = b[b.metric == metric].iloc[0]
        assert [r.lo, r.hi] == pytest.approx(np.nanpercentile(vals, [2.5, 97.5]).tolist())
        assert r.lo <= r.estimate <= r.hi
    est = evaluate(u, "x", 1000, levels=("molecule",), kinds=("all",)).iloc[0]
    assert b[b.metric == "sens"].estimate.iloc[0] == est.sens and b[b.metric == "prec"].estimate.iloc[0] == est.prec


def test_bootstrap_single_gene_is_degenerate():
    u = _boot_units(["g1"] * 4, [True, False, True, False], ["templated", "templated", "nontemplated", "nontemplated"])
    b = bootstrap(u, [T0], n=50, seed=0)
    assert (b.lo == b.hi).all() and (b.lo == b.estimate).all() and set(b.n_clusters) == {1}
    assert b.estimate.tolist() == [0.5, 0.5, 0.5]
    r = bootstrap(u, [T0], level="read", n=20, seed=0)
    assert r.estimate.tolist() == [0.5, 0.5, 0.5] and (r.lo == r.hi).all()


def test_hexamer_check():
    mol, tr, fl = _tables()
    h = hexamer_check(units(mol, tr, fl, TAGS, cap=1000), [T0]).set_index("label")
    # T0 removes m0 (hexamer False) and m3 (False); keeps m1 (True) and m2 (True)
    assert list(h.columns) == ["tag", "removed_n", "removed_hex_rate", "kept_n", "kept_hex_rate"]
    assert h.loc["all"].tolist() == [T0, 2, 0.0, 2, 1.0]
    assert h.loc["templated"].tolist() == [T0, 1, 0.0, 1, 1.0]
    assert h.loc["nontemplated", "removed_n"] == 0 and math.isnan(h.loc["nontemplated", "removed_hex_rate"])
    assert h.loc["nontemplated", "kept_hex_rate"] == 1.0
    assert h.loc["ambiguous", "removed_n"] == 1 and h.loc["ambiguous", "kept_n"] == 0


def test_spikein_labels_and_modules():
    mol, tr, fl = _tables()
    mol = mol.assign(spike_label=["internal", "internal", "tail", "other", "tail", "internal", "internal"],
                     module=["ERCC", "SIRV_E0", "ERCC", "ERCC", "ERCC", "ERCC", "ERCC"])
    u = units(mol, tr, fl, TAGS, cap=1000, label="spike_label", pos="internal", neg="tail", amb=None)
    assert u.mols.mol.tolist() == [0, 1, 2]  # m3 ("other") excluded entirely
    s = spikein_metrics(u, [T0, T1])
    assert set(s.level) == {"read", "molecule"} and set(s.stratum_kind) == {"all", "module"}
    assert set(s.stratum) == {"all", "ERCC", "SIRV_E0", "long_SIRV"}
    r = s[(s.level == "molecule") & (s.stratum == "all") & (s.tag == T0)].iloc[0]
    assert _counts(r) == [1, 1, 1, 0, 0, 0]
    r = s[(s.level == "molecule") & (s.stratum == "SIRV_E0") & (s.tag == T0)].iloc[0]
    assert _counts(r) == [0, 1, 0, 0, 0, 0]
    r = s[(s.level == "read") & (s.stratum == "ERCC") & (s.tag == T1)].iloc[0]
    assert _counts(r) == [1, 2, 0, 1, 0, 0]  # r0a removed, r0b, r0c kept; r2a (tail) removed
    assert s[(s.stratum == "long_SIRV")][["tp", "fn", "tn", "fp"]].to_numpy().sum() == 0


def _run_stage(script, base, out):
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / script), "--base", str(base), "--out", str(out)],
                       capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_stage17_end_to_end(tmp_path):
    mol, tr, fl = _tables()
    base, out = tmp_path / "base", tmp_path / "eval"
    for lib in ("1417", "1419"):
        (base / lib / "truth").mkdir(parents=True)
        (base / lib / "grid").mkdir()
        mol.to_parquet(base / lib / "truth" / "molecules_labeled.parquet")
        tr.to_parquet(base / lib / "truth" / "truth_reads.parquet")
        fl.to_parquet(base / lib / "grid" / "truth_flags.parquet")
        pd.DataFrame({"tag": TAGS, "re": "E", "removed": [10, 20, 30], "total": 100}).to_parquet(
            base / lib / "grid" / "region_counts.parquet")
    spike = mol.assign(spike_label=["internal", "internal", "tail", "other", "tail", "internal", "internal"],
                       module=["ERCC", "SIRV_E0", "ERCC", "ERCC", "ERCC", "ERCC", "long_SIRV"])
    (base / "1417" / "spikein" / "truth").mkdir(parents=True)
    (base / "1417" / "spikein" / "grid").mkdir()
    spike.to_parquet(base / "1417" / "spikein" / "truth" / "molecules_labeled.parquet")
    tr.to_parquet(base / "1417" / "spikein" / "truth" / "truth_reads.parquet")
    fl.to_parquet(base / "1417" / "spikein" / "grid" / "truth_flags.parquet")
    stdout = _run_stage("17_evaluate.py", base, out)
    assert "1179: skipped" in stdout and "1419: spike-in metrics skipped" in stdout
    m = pd.read_parquet(out / "metrics.parquet")
    assert set(m.lib) == {"1417", "1419"} and set(m.cap) == {500, 1000, 2000}
    got = m[(m.lib == "1417") & (m.cap == 1000)].reset_index(drop=True)
    # the stage orders tags as all_settings() does: the control precedes the other settings of its COVLEN
    pd.testing.assert_frame_equal(got, evaluate(units(mol, tr, fl, [T2, T0, T1], 1000), "1417", 1000), check_dtype=False)
    sel = json.loads((out / "selection.json").read_text())
    # mode i: T0 (sens 0.5, spec 1.0) qualifies, T1 (spec 0) does not, T2 is the control; mode e has no candidate
    assert sel["modes"]["i"]["selection"]["tag"] == T0 and sel["modes"]["e"]["selection"]["tag"] is None
    assert sel["modes"]["e"]["selection"]["n_candidates"] == 0
    assert sel["modes"]["i"]["metrics"]["1419"]["molecule"]["sens"] == 0.5
    assert sel["modes"]["i"]["metrics"]["1417"]["molecule_majority"]["tp"] == 2.0
    assert sel["published"][T1]["1417"]["read"]["prec"] == 0.5 and sel["published"]["c180_l11_m2_i"]["1417"] == {}
    assert sel["cap_sensitivity"]["2000"]["i"]["tag"] == T0
    assert len(sel["cutoff_sweep"]) == 40
    sw = {(d["templated_min"], d["nontemplated_max"], d["mode"]): d for d in sel["cutoff_sweep"]}
    assert sw[(8, 4, "i")]["tag"] == T0 and sw[(8, 4, "i")]["by_library"]["1419"]["sens"] == 0.5
    assert sw[(10, 6, "i")]["tag"] is None  # T0 spec 0.5 and T1 spec 0.5 at 10/6
    assert sel["units"]["1417"]["1000"] == {"read_units": 7, "molecule_units": 4, "missing_reads": 1, "dropped_molecules": 1}
    assert sel["bootstrapped_tags"][0] == T0 and sel["bootstrapped_tags"][-1] == T2
    b = pd.read_parquet(out / "bootstrap.parquet")
    assert len(b) == 2 * 2 * 3 * 3 and set(b.tag) == {T0, T1, T2}
    r = b[(b.lib == "1417") & (b.level == "molecule") & (b.tag == T0) & (b.metric == "sens")].iloc[0]
    assert r.estimate == 0.5
    h = pd.read_parquet(out / "hexamer_check.parquet")
    assert len(h) == 2 * 3 * 4
    s = pd.read_parquet(out / "spikein_metrics.parquet")
    assert set(s.lib) == {"1417"} and set(s.cap) == {1000}
    ref = spikein_metrics(units(spike, tr, fl, [T2, T0, T1], 1000, label="spike_label", pos="internal", neg="tail", amb=None),
                          [T2, T0, T1])
    pd.testing.assert_frame_equal(s.drop(columns=["lib", "cap"]), ref, check_dtype=False)
    r = s[(s.level == "molecule") & (s.stratum == "all") & (s.tag == T0)].iloc[0]
    assert _counts(r) == [1, 1, 1, 0, 0, 0]  # m0 internal removed, m1 internal kept, m2 tail kept


def test_select_setting_reads_only_stratum_all_and_its_cutoffs():
    m = _metrics(SEL_ROWS)
    # qualifying rows with a higher sensitivity than the choice, in another stratum and at other cut-offs
    extra = _metrics([("1417", 1000, "molecule", "c300_l5_m1_i", 0.95, 0.99), ("1417", 1000, "molecule", "c200_l5_m1_i", 0.96, 0.99)])
    extra.loc[0, ["stratum_kind", "stratum"]] = ["region", "3UTR"]
    extra.loc[1, "cutoffs"] = "7/2"
    m = pd.concat([m, extra], ignore_index=True)
    removed = {"c300_l9_m1_i": 500, "c450_l9_m1_i": 400, "c300_l10_m1_i": 100, "c300_l3_m2_i": 900}
    s = select_setting(m, removed, introns=True)
    assert s["tag"] == "c450_l9_m1_i" and s["n_candidates"] == 4 and s["n_qualifying"] == 3
    s = select_setting(m, removed, introns=True, cutoffs="7/2")
    assert s["tag"] == "c200_l5_m1_i" and s["n_candidates"] == 1 and s["frontier"] == [["c200_l5_m1_i", 0.96, 0.99]]


def test_bootstrap_drops_clusters_without_positive_or_negative_units():
    labels = ["templated", "nontemplated", "templated", "nontemplated", "templated", "ambiguous", "ambiguous", "templated",
              "nontemplated"]
    genes = ["g1", "g1", "g2", "g2", "g3", "g4", "g4", "", ""]
    removed = [True, False, False, True, True, True, False, False, True]
    u = _boot_units(genes, removed, labels)
    # g4 holds only ambiguous molecules: it adds nothing to tp, fn, tn or fp and is left out of the resampled clusters,
    # so the draws equal those of the same units without g4 (g1, g2, g3 and two intergenic loci, in the same sorted order)
    keep = [i for i, g in enumerate(genes) if g != "g4"]
    v = _boot_units([genes[i] for i in keep], [removed[i] for i in keep], [labels[i] for i in keep])
    for level in ("molecule", "read", "molecule_weighted"):
        b = bootstrap(u, [T0], level=level, n=200, seed=7)
        assert set(b.n_clusters) == {5}
        pd.testing.assert_frame_equal(b, bootstrap(v, [T0], level=level, n=200, seed=7))
    # a cluster with only positive units stays: without g1's negative, g1 still counts (5 clusters)
    w = _boot_units(genes[:1] + genes[2:], removed[:1] + removed[2:], labels[:1] + labels[2:])
    assert set(bootstrap(w, [T0], n=50, seed=1).n_clusters) == {5}


def test_ambiguous_tail_share():
    mol = pd.DataFrame({"label": ["templated"] * 10 + ["nontemplated"] * 5 + ["ambiguous"] * 4,
                        "hexamer": [True] + [False] * 9 + [True] * 4 + [False] + [True, True, False, False]})
    # rates: templated 1/10, non-templated 4/5, ambiguous 2/4; (0.5 - 0.1) / (0.8 - 0.1) = 4/7
    s = ambiguous_tail_share(mol)
    assert s == {"estimate": pytest.approx(4 / 7), "rate_templated": 0.1, "rate_nontemplated": 0.8, "rate_ambiguous": 0.5,
                 "n_templated": 10, "n_nontemplated": 5, "n_ambiguous": 4}
    # clipped to [0, 1]: ambiguous rates above the non-templated and below the templated rate
    assert ambiguous_tail_share(mol.assign(hexamer=[True] + [False] * 9 + [True] * 4 + [False] + [True] * 4))["estimate"] == 1.0
    assert ambiguous_tail_share(mol.assign(hexamer=[True] * 2 + [False] * 8 + [True] * 4 + [False] + [False] * 4))["estimate"] == 0.0
    # undefined: no ambiguous molecule, or equal templated and non-templated rates
    s = ambiguous_tail_share(mol[mol.label != "ambiguous"])
    assert s["estimate"] is None and s["rate_ambiguous"] is None and s["n_ambiguous"] == 0
    assert ambiguous_tail_share(mol.assign(hexamer=True))["estimate"] is None


E1, E2, E3, CE, I1, I2 = "c300_l10_m1_e", "c450_l9_m1_e", "c150_l20_m0_e", "c450_lnone_m0_e", "c300_l10_m1_i", "c375_l3_m2_i"
SEL_TAGS = [E1, E2, E3, CE, I1, I2]
# id, label, a20, gene, member dist5_exon (None: no member), hexamer; Tm's member is absent from truth_flags
SEL_MOLS = [(0, "templated", 15, "gA", 100, False), (1, "templated", 15, "gB", 100, False), (2, "templated", 15, "gC", 700, False),
            (3, "templated", 15, "gD", 1500, False), (4, "templated", 15, "gE", 100, False),
            (5, "nontemplated", 2, "gA", 100, True), (6, "nontemplated", 2, "gB", 100, True), (7, "ambiguous", 6, "gC", 100, True),
            (8, "templated", 15, "gF", None, True), (9, "templated", 15, "gF", None, False), (10, "nontemplated", 2, "gF", None, True),
            (11, "nontemplated", 2, "gF", None, False), (12, "ambiguous", 6, "gF", None, False)]
# molecules removed (both reads) per tag: E1 Ta Tc Td; E2 Tb Tc; E3 Ta; CE all; I1 Ta Na; I2 Ta Tb Tc Nb
SEL_REMOVED = {E1: {0, 2, 3}, E2: {1, 2}, E3: {0}, CE: set(range(13)), I1: {0, 5}, I2: {0, 1, 2, 6}}


def _sel_tables():
    mol = pd.DataFrame([(i, lab, a20, "14-20" if lab == "templated" else "0-4" if lab == "nontemplated" else "5-7", "3UTR", g,
                         False, h, "chrA", "+", 1000 * (i + 1)) for i, lab, a20, g, _, h in SEL_MOLS], columns=MOL_COLS)
    rows = [(f"p{i}", i, "pa", np.nan, "", False) for i, *_ in SEL_MOLS]
    rows += [(f"r{i}", i, "member", float(d), "exon", False) for i, _, _, _, d, _ in SEL_MOLS if d is not None]
    tr = pd.DataFrame(rows, columns=["qname", "mol", "role", "dist5_exon", "dist_mode", "spliced"])
    q = [x for x in tr.qname if x != "r4"]
    fl = pd.DataFrame({"qname": q, "ref": "chrA", "pos": 0, "flag": 0}
                      | {t: [int(x[1:]) in SEL_REMOVED[t] for x in q] for t in SEL_TAGS})
    return mol, tr, fl


def _region_counts(removed):
    return pd.DataFrame({"tag": list(removed), "re": "E", "removed": list(removed.values()), "total": 1000})


def test_stage17_selection_without_qualifier_tie_breaks_caps_and_spikeins(tmp_path):
    mol, tr, fl = _sel_tables()
    base, out = tmp_path / "base", tmp_path / "eval"
    # 1417's record counts break the E1/E2 tie at cap 1000 toward E2 and the three-way tie at cap 500 toward E3;
    # 1419's would break them the other way
    counts = {"1417": {E1: 50, E2: 40, E3: 30, CE: 100, I1: 60, I2: 70}, "1419": {E1: 40, E2: 50, E3: 60, CE: 100, I1: 60, I2: 70}}
    counts["1179"] = counts["1417"]
    for lib in ("1417", "1419", "1179"):
        (base / lib / "truth").mkdir(parents=True)
        (base / lib / "grid").mkdir()
        mol.to_parquet(base / lib / "truth" / "molecules_labeled.parquet")
        tr.to_parquet(base / lib / "truth" / "truth_reads.parquet")
        fl.to_parquet(base / lib / "grid" / "truth_flags.parquet")
        _region_counts(counts[lib]).to_parquet(base / lib / "grid" / "region_counts.parquet")
    spike = mol.assign(spike_label=np.where(mol.label == "templated", "internal", "tail"), module="ERCC")
    for sub in ("truth", "grid"):
        (base / "1417" / "spikein" / sub).mkdir(parents=True)
    spike.to_parquet(base / "1417" / "spikein" / "truth" / "molecules_labeled.parquet")
    tr.to_parquet(base / "1417" / "spikein" / "truth" / "truth_reads.parquet")
    fl.to_parquet(base / "1417" / "spikein" / "grid" / "truth_flags.parquet")
    stdout = _run_stage("17_evaluate.py", base, out)
    assert not [line for line in stdout.splitlines() if ": skipped, missing" in line]  # all three libraries used
    assert "1419: spike-in metrics skipped" in stdout and "1179: spike-in" not in stdout
    assert "mode i: no setting chosen" in stdout
    sel = json.loads((out / "selection.json").read_text())
    assert sel["libraries"] == ["1417", "1419", "1179"]
    # mode e at cap 1000: E1 and E2 have sens 2/3, spec 1 (E3 1/3); E2 removes fewer 1417 records
    e = sel["modes"]["e"]
    assert e["selection"]["tag"] == E2 and e["selection"]["sens"] == pytest.approx(2 / 3) and e["selection"]["spec"] == 1.0
    assert e["selection"]["n_candidates"] == 3 and e["selection"]["n_qualifying"] == 3
    # cap 1000 values: Ta fn, Tb and Tc tp, Tm (member missing from truth_flags) dropped, not counted as kept
    assert e["metrics"]["1417"]["molecule"] == {"tp": 2.0, "fn": 1.0, "tn": 2.0, "fp": 0.0, "n_amb": 1.0, "amb_removed": 0.0,
                                                 "sens": pytest.approx(2 / 3), "spec": 1.0, "prec": 1.0}
    assert sel["units"]["1417"]["1000"] == {"read_units": 6, "molecule_units": 6, "missing_reads": 1, "dropped_molecules": 1}
    # mode i: I1 (1/3, 1/2) and I2 (1, 1/2) both miss spec 0.95; the frontier is I2 alone
    i = sel["modes"]["i"]
    assert i["selection"]["tag"] is None and i["selection"]["n_candidates"] == 2 and i["selection"]["n_qualifying"] == 0
    assert i["selection"]["frontier"] == [[I2, 1.0, 0.5]] and i["metrics"] == {}
    # caps 500 (three-way tie at 1/2: E3 removes fewest records) and 2000 (Td enters: E1 3/4) choose differently
    assert {c: sel["cap_sensitivity"][c]["e"]["tag"] for c in ("500", "1000", "2000")} == {"500": E3, "1000": E2, "2000": E1}
    assert sel["cap_sensitivity"]["2000"]["i"]["tag"] is None
    # bootstrap and hexamer check: the chosen tag, the published settings present, the control at the chosen COVLEN
    assert sel["bootstrapped_tags"] == [E2] + [setting_tag(p) for p in published()] + [CE]
    b = pd.read_parquet(out / "bootstrap.parquet")
    assert set(b.tag) == {E2, E1, I1, I2, CE} and set(b.lib) == {"1417", "1419", "1179"} and len(b) == 3 * 2 * 5 * 3
    h = pd.read_parquet(out / "hexamer_check.parquet")
    assert set(h.tag) == {E2, E1, I1, I2, CE} and len(h) == 3 * 5 * 4
    s = pd.read_parquet(out / "spikein_metrics.parquet")
    assert set(s.lib) == {"1417"}
    # hexamer rates over all labeled molecules: templated 1/7, non-templated 3/4, ambiguous 1/2 -> (1/2 - 1/7) / (3/4 - 1/7)
    a = sel["ambiguous_tail_share"]
    assert set(a) == {"1417", "1419", "1179"} and a["1417"]["estimate"] == pytest.approx(10 / 17)
    assert (a["1417"]["n_templated"], a["1417"]["n_nontemplated"], a["1417"]["n_ambiguous"]) == (7, 4, 2)
```

- [ ] **Step 2: Write the failing tests in `tests/test_polyasite.py`**

```python
import gzip

import numpy as np
import pandas as pd
import pysam

from pafbench.polyasite import BINS, distance_bin, load_polyasite, nearest_signed, site_distances
from pafbench.sites import priming_site
from pafbench.spikein import label_spike_site

# PolyASite 3.0 BED rows: chrom, start, end, id chrom:pos:strand (pos 1-based), mean RPM, strand, % tissues, protocols,
# stringency, class, signals. Out of order on purpose. chr2 has a plus site only and chrX a minus site only; chr3, which
# test_site_distances_per_ref_and_strand also uses, is absent.
BED = [
    ("chr1", 1999, 2000, "chr1:2000:+", 2.0, "+", 50.0, 1, 98, "TE", "AUUAAA;AAUAAA"),
    ("chr1", 99, 100, "chr1:100:+", 0.5, "+", 11.1, 1, 47, "TE", "AAUAAA"),
    ("chr1", 480, 520, "chr1:500:+", 0.01, "+", 5.6, 1, 23, "IN", "NaN"),  # cluster span: the id gives the cleavage site
    ("chr1", 199, 200, "chr1:200:-", 0.2, "-", 5.6, 1, 26, "DI", "NaN"),
    ("chr1", 690, 720, "chr1:700:-", 0.001, "-", 5.6, 1, 20, "IN", "NaN"),
    ("chr2", 49, 50, "chr2:50:+", 1.0, "+", 22.2, 1, 36, "UI", "NaN"),
    ("chrX", 9, 10, "chrX:10:-", 0.05, "-", 5.6, 1, 21, "DI", "NaN"),
]


def _bed(path, rows=BED):
    with gzip.open(path, "wt") as fh:
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    return path


def _eq(got, want):
    assert sorted(got) == sorted(want)
    for k in want:
        assert got[k].dtype == np.int64 and got[k].tolist() == want[k], k


def test_load_polyasite_converts_to_the_priming_site_boundary(tmp_path):
    p = _bed(tmp_path / "atlas.bed.gz")
    # plus: the id position itself; minus: id position + 1
    _eq(load_polyasite(p), {("chr1", True): [100, 500, 2000], ("chr1", False): [201, 701], ("chr2", True): [50],
                            ("chrX", False): [11]})
    # score (column 5) at or above min_score: 0.2 is kept, 0.05, 0.01 and 0.001 go, and chrX loses its only site
    _eq(load_polyasite(p, min_score=0.2), {("chr1", True): [100, 2000], ("chr1", False): [201], ("chr2", True): [50]})


def test_conversion_puts_the_atlas_site_on_the_priming_site_of_a_read(tmp_path):
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "chr1", "LN": 5000}]})
    reads = []
    for start, rev in ((90, False), (201, True)):
        a = pysam.AlignedSegment(hdr)
        a.query_name, a.reference_id, a.reference_start, a.cigarstring = f"r{start}", 0, start, "10M"
        a.query_sequence, a.is_reverse = "C" * 10, rev
        reads.append(a)
    # plus read covers 0-based 90..99: last cDNA base 99 = 1-based 100 = chr1:100:+.
    # minus read covers 201..210: last cDNA base 201 = 1-based 202, yet chr1:200:- maps onto it. With the symmetric BED
    # reading, the atlas sits 2 nt further downstream on minus than on plus, both from 1417 tail-primed pa sites (a20 <= 4)
    # and from GENCODE 44 transcript ends, so the minus id is taken 2 nt downstream of the last templated base.
    mol = pd.DataFrame({"ref": "chr1", "strand": ["+", "-"], "site": [priming_site(a) for a in reads]})
    assert mol.site.tolist() == [100, 201]
    assert site_distances(mol, load_polyasite(_bed(tmp_path / "a.bed.gz"))).tolist() == [0.0, 0.0]


def test_nearest_signed_plus():
    ends = np.array([100, 500, 2000], np.int64)
    sites = np.array([50, 100, 101, 300, 299, 301, 2500, 1250])
    # 50: next end 100 downstream (+50); 101: 100 is 1 nt upstream (-1); 300: 100 and 500 both 200 away, the positive wins;
    # 299: -199 beats +201; 301: +199 beats -201; 2500: last end 500 nt upstream; 1250: tie of 750, positive
    got = nearest_signed(sites, True, ends)
    assert got.dtype == np.float64 and got.tolist() == [50, 0, -1, 200, -199, 199, -500, 750]


def test_nearest_signed_minus():
    ends = np.array([201, 701], np.int64)
    # minus: downstream is the lower coordinate, d = site - end. 150: the end at 201 lies upstream (-51); 451: tie of 250,
    # positive; 452: 701 is 249 upstream; 800: 701 lies 99 downstream
    assert nearest_signed(np.array([201, 150, 451, 452, 800]), False, ends).tolist() == [0, -51, 250, -249, 99]


def test_nearest_signed_without_atlas_sites():
    for ends in (None, np.array([], np.int64)):
        got = nearest_signed(np.array([5, 6]), True, ends)
        assert got.shape == (2,) and np.isnan(got).all()


def test_nearest_signed_matches_label_spike_site():
    rng = np.random.default_rng(3)
    ends = np.unique(rng.integers(0, 20_000, 150))
    sites = np.concatenate([rng.integers(-500, 20_500, 3000), ends, ends + 1, ends - 1])
    for strd in (True, False):
        want = [label_spike_site(int(s), strd, ends)[1] for s in sites]
        assert nearest_signed(sites, strd, ends).tolist() == want


def test_site_distances_per_ref_and_strand(tmp_path):
    atlas = load_polyasite(_bed(tmp_path / "atlas.bed.gz"))
    mol = pd.DataFrame([("chr1", "+", 101), ("chr1", "-", 800), ("chr2", "+", 40), ("chr2", "-", 40), ("chr3", "+", 5),
                        ("chr1", "+", 300), ("chrX", "-", 30)], columns=["ref", "strand", "site"])
    # chr1 +: 100 is 1 nt upstream; chr1 -: 701 lies 99 nt downstream; chr2 +: 50 lies 10 downstream; chr2 has no minus
    # site and chr3 no site at all; chr1 + 300: tie, positive; chrX -: 11 is 19 nt downstream
    got = site_distances(mol, atlas)
    assert got[[0, 1, 2, 5, 6]].tolist() == [-1, 99, 10, 200, 19]
    assert np.isnan(got[[3, 4]]).all()


def test_site_distances_moved_upstream():
    atlas = {("chr1", True): np.array([100, 2000], np.int64), ("chr1", False): np.array([3000], np.int64)}
    mol = pd.DataFrame([("chr1", "+", 1100), ("chr1", "-", 1990), ("chr1", "+", 3005), ("chrM", "+", 1100)],
                       columns=["ref", "strand", "site"])
    # 1,000 nt upstream in transcript sense: plus 1100 -> 100 (0); minus 1990 -> 2990 (upstream is the higher coordinate on
    # minus), the atlas site 3000 lies 10 nt upstream (-10); plus 3005 -> 2005, 2000 lies 5 nt upstream (-5); chrM: none.
    # Moved the wrong way the minus site would be 990, 2,010 nt from 3000.
    got = site_distances(mol, atlas, upstream=1000)
    assert got[:3].tolist() == [0, -10, -5] and np.isnan(got[3])
    # unmoved: 1100 is 900 upstream of 2000 and 1,000 downstream of 100 (+900); 1990 against 3000 (-1010); 3005 (-1005)
    got = site_distances(mol, atlas)
    assert got[:3].tolist() == [900, -1010, -1005] and np.isnan(got[3])


def test_distance_bin_edges():
    d = np.array([0, 10, -10, 11, -11, 50, -50, 51, -51, 200, -200, 201, -201, 5000, np.nan])
    want = ["0-10"] * 3 + ["11-50"] * 4 + ["51-200"] * 4 + [">200"] * 3 + ["no atlas site on this ref and strand"]
    assert BINS == ("0-10", "11-50", "51-200", ">200", "no atlas site on this ref and strand")
    assert distance_bin(d).tolist() == want
```

- [ ] **Step 3: Write the failing tests in `tests/test_burden.py`**

```python
import gzip
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from pafbench.config import REPO
from pafbench.burden import (near_shares, proximity_table, same_gene, same_site, selection_bias, share_ci, share_table,
                             site_labels)
from pafbench.evaluate import gene_clusters, resample_counts
from pafbench.polyasite import BINS


def _mol(labels, region=None, gene=None, **cols):
    n = len(labels)
    d = {"label": labels, "region": region or ["3UTR"] * n, "gene": gene or ["g1"] * n, "ref": "chr1", "strand": "+",
         "site": np.arange(n) * 10_000}
    d.update(cols)
    return pd.DataFrame(d)


def test_share_table():
    mol = _mol(["templated", "templated", "nontemplated", "ambiguous", "templated", "templated", "templated"],
               region=["3UTR"] * 4 + ["intron"] * 3)
    a = share_table(mol).iloc[0]
    assert (a.stratum, a.templated, a.nontemplated, a.ambiguous, a.total, a.frac) == ("all", 5, 1, 1, 7, 1.0)
    assert a.share_all == 5 / 7 and a.share_resolved == 5 / 6
    t = share_table(mol, "region").set_index("stratum")
    assert t.loc["3UTR", ["templated", "nontemplated", "ambiguous", "total"]].tolist() == [2, 1, 1, 4]
    assert t.loc["3UTR", "share_all"] == 0.5 and t.loc["3UTR", "share_resolved"] == 2 / 3 and t.loc["3UTR", "frac"] == 4 / 7
    assert t.loc["intron", "share_all"] == 1.0 and t.loc["intron", "share_resolved"] == 1.0
    only_amb = share_table(_mol(["ambiguous"])).iloc[0]
    assert only_amb.share_all == 0.0 and math.isnan(only_amb.share_resolved)


def test_share_ci_matches_brute_force_and_degenerate_case():
    rng = np.random.default_rng(0)
    genes = [f"g{i % 9}" for i in range(80)]
    labels = list(rng.choice(["templated", "nontemplated", "ambiguous"], 80, p=[0.7, 0.2, 0.1]))
    regions = list(rng.choice(["3UTR", "intron"], 80))
    mol = _mol(labels, region=regions, gene=genes)
    ci = share_ci(mol, "region", n=300, seed=5).set_index("stratum")
    assert ci.equals(share_ci(mol, "region", n=300, seed=5).set_index("stratum"))
    names = np.unique(gene_clusters(mol))
    w = resample_counts(len(names), 300, 5)[:, np.searchsorted(names, gene_clusters(mol))].astype(float)
    t = mol.label.to_numpy() == "templated"
    for reg in ("3UTR", "intron"):
        m = mol.region.to_numpy() == reg
        share = (w * (t & m)).sum(1) / (w * m).sum(1)
        assert [ci.loc[reg, "lo"], ci.loc[reg, "hi"]] == pytest.approx(np.percentile(share, [2.5, 97.5]).tolist())
    one = share_ci(_mol(["templated", "nontemplated", "templated", "ambiguous"]), n=50, seed=1).iloc[0]
    assert one.stratum == "all" and one.lo == one.hi == 0.5


def test_site_labels_take_the_most_common_label():
    mol = pd.DataFrame({"ref": ["chr3"] * 4, "strand": "+", "site": [10, 10, 10, 20],
                        "label": ["templated", "nontemplated", "templated", "ambiguous"]})
    assert site_labels(mol).values.tolist() == [["chr3", "+", 10, "templated"], ["chr3", "+", 20, "ambiguous"]]


def test_same_site_nearest_within_tol_and_strand():
    a = pd.DataFrame([
        ("chr1", "+", 100, "templated"), ("chr1", "+", 100, "templated"),  # one site
        ("chr1", "+", 200, "nontemplated"),  # b at 195 (5 nt) and 206 (6 nt): nearest is 195
        ("chr1", "-", 300, "templated"),  # b has 300 only on +: unmatched
        ("chr2", "+", 50, "ambiguous"),  # b at 48 and 52: tie, the lower coordinate wins
        ("chr3", "+", 10, "templated"), ("chr3", "+", 10, "templated"), ("chr3", "+", 10, "nontemplated"),
        ("chr1", "+", 1000, "templated"),  # b at 1006: beyond tol
    ], columns=["ref", "strand", "site", "label"])
    b = pd.DataFrame([
        ("chr1", "+", 104, "templated"), ("chr1", "+", 195, "nontemplated"), ("chr1", "+", 206, "templated"),
        ("chr1", "+", 300, "templated"), ("chr2", "+", 48, "ambiguous"), ("chr2", "+", 52, "templated"),
        ("chr1", "+", 1006, "templated"),
    ], columns=["ref", "strand", "site", "label"])
    s = same_site(a, b)
    assert list(s.columns) == ["label_a", "label_b", "n"]
    got = {(r.label_a, r.label_b): r.n for r in s.itertuples()}
    assert got == {("templated", "templated"): 1, ("nontemplated", "nontemplated"): 1, ("templated", "none"): 3,
                   ("ambiguous", "ambiguous"): 1}
    wide = {(r.label_a, r.label_b): r.n for r in same_site(a, b, tol=6).itertuples()}
    assert wide[("templated", "templated")] == 2 and wide[("templated", "none")] == 2
    assert same_site(a, b.iloc[:0]).n.sum() == 6


def _genes(shares, n, label_prefix="g"):
    rows = []
    for i, (s, k) in enumerate(zip(shares, n)):
        t = round(s * k)
        rows += [(f"{label_prefix}{i}", "templated")] * t + [(f"{label_prefix}{i}", "nontemplated")] * (k - t)
    return pd.DataFrame(rows, columns=["gene", "label"])


def test_same_gene_spearman_and_min_n():
    a = _genes([0.1, 0.2, 0.3, 0.4, 0.9], [20, 20, 20, 20, 19])
    b = _genes([0.15, 0.3, 0.5, 0.9, 0.0], [20, 40, 20, 20, 40])
    a = pd.concat([a, pd.DataFrame({"gene": [""] * 30, "label": "templated"})], ignore_index=True)
    b = pd.concat([b, pd.DataFrame({"gene": [""] * 30, "label": "nontemplated"})], ignore_index=True)
    r = same_gene(a, b)
    # g4 has 19 molecules in a, intergenic molecules never count
    assert r == {"n_genes": 4, "spearman_rho": 1.0, "p_value": pytest.approx(spearmanr([1, 2, 3, 4], [1, 2, 3, 4]).pvalue)}
    r = same_gene(a, b, min_n=19)
    assert r["n_genes"] == 5 and r["spearman_rho"] == pytest.approx(spearmanr([0.1, 0.2, 0.3, 0.4, 0.9],
                                                                               [0.15, 0.3, 0.5, 0.9, 0.0]).statistic)


def test_selection_bias():
    mol = pd.DataFrame({"re": ["E", "E", "N", "I", None]})
    rc = pd.DataFrame({"tag": ["t1"] * 4 + ["t2"] * 4, "re": ["E", "N", "I", ""] * 2, "removed": [5, 1, 1, 0] * 2,
                       "total": [60, 20, 15, 5] * 2})
    s = selection_bias(mol, rc).set_index("re")
    assert list(s.columns) == ["labeled_share", "records_share"]
    assert s.labeled_share.to_dict() == pytest.approx({"": 0.2, "E": 0.4, "I": 0.2, "N": 0.2})
    assert s.records_share.to_dict() == pytest.approx({"": 0.05, "E": 0.6, "I": 0.15, "N": 0.2})
    s = selection_bias(pd.DataFrame({"re": ["E"]}), rc).set_index("re")
    assert s.loc["N", "labeled_share"] == 0.0 and s.loc["E", "labeled_share"] == 1.0


def _lib_mol(shift):
    """60 molecules, genes g0, g1, g2 of 20 each with 5, 10, 15 templated; the rest alternate nontemplated / ambiguous."""
    i = np.arange(60)
    k = i % 20
    lab = np.where(k < 5 * (i // 20 + 1), "templated", np.where(k % 2 == 0, "nontemplated", "ambiguous")).astype(object)
    return pd.DataFrame({"mol": i, "ref": "chr1", "strand": "+", "site": 100 + 50 * i + shift, "label": lab,
                         "a_band": np.where(lab == "templated", "14-20", np.where(lab == "nontemplated", "0-4", "5-7")),
                         "region": np.where(i < 40, "3UTR", "intron"), "gene": [f"g{x}" for x in i // 20],
                         "has_ts": i % 5 == 0, "re": np.where(i < 40, "E", "I")})


def _atlas(tmp_path):
    """PolyASite rows on chr1 plus at 100 and 1105; a minus site at 1150 (on a plus molecule's site) and chr2 do not count."""
    p = tmp_path / "atlas.bed.gz"
    rows = [("chr1", 99, 100, "chr1:100:+", 0.5, "+"), ("chr1", 1104, 1105, "chr1:1105:+", 0.1, "+"),
            ("chr1", 1148, 1149, "chr1:1149:-", 0.1, "-"), ("chr2", 999, 1000, "chr2:1000:+", 0.1, "+")]
    with gzip.open(p, "wt") as fh:
        fh.writelines("\t".join(map(str, r + (11.1, 1, 47, "TE", "NaN"))) + "\n" for r in rows)
    return p


def test_stage18_end_to_end(tmp_path):
    base, out = tmp_path / "base", tmp_path / "burden"
    libs = {"1417": _lib_mol(0), "1419": _lib_mol(0), "1179": _lib_mol(3)}
    for lib, mol in libs.items():
        (base / lib / "truth").mkdir(parents=True)
        mol.to_parquet(base / lib / "truth" / "molecules_labeled.parquet")
    (base / "1417" / "grid").mkdir()
    pd.DataFrame({"tag": "t1", "re": ["E", "I"], "removed": [1, 2], "total": [30, 70]}).to_parquet(
        base / "1417" / "grid" / "region_counts.parquet")
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "18_burden.py"), "--base", str(base), "--out", str(out),
                        "--atlas", str(_atlas(tmp_path))], capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 0, r.stderr
    assert "1419: selection bias skipped" in r.stdout
    t = pd.read_parquet(out / "templated_share.parquet")
    assert set(t.group) == {"1417", "1419", "1179", "nuclei", "cells"}
    a = t[(t.group == "1417") & (t.stratum_kind == "all")].iloc[0]
    # non-templated at even k: g0 k 6..18 (7), g1 k 10..18 (5), g2 k 16, 18 (2); the other 16 ambiguous
    assert (a.templated, a.nontemplated, a.ambiguous, a.total, a.share_all) == (30, 14, 16, 60, 0.5)
    assert t[(t.group == "nuclei") & (t.stratum_kind == "all")].total.iloc[0] == 120
    reg = share_table(libs["1417"], "region").merge(share_ci(libs["1417"], "region"), on="stratum")
    got = t[(t.group == "1417") & (t.stratum_kind == "region")].drop(columns=["group", "stratum_kind"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, reg, check_dtype=False)
    assert set(t.stratum_kind) == {"all", "region", "a_band", "has_ts"}
    s = pd.read_parquet(out / "same_site.parquet")
    same = s[s.pair == "1417_vs_1419"]
    assert (same.label_a == same.label_b).all() and same.n.sum() == 60
    nc = s[s.pair == "nuclei_vs_cells"]
    assert (nc.label_a == nc.label_b).all() and nc.n.sum() == 60  # the 1179 sites lie 3 nt away
    g = pd.read_parquet(out / "same_gene.parquet").set_index("pair")
    assert g.loc["1417_vs_1419", "n_genes"] == 3 and g.loc["1417_vs_1419", "spearman_rho"] == 1.0
    assert g.loc["nuclei_vs_cells", "n_genes"] == 3
    b = pd.read_parquet(out / "selection_bias.parquet")
    assert set(b.lib) == {"1417"} and b.set_index("re").records_share.to_dict() == {"E": 0.3, "I": 0.7}
    summ = json.loads((out / "summary.json").read_text())
    assert summ["groups"]["nuclei"] == ["1417", "1419"] and summ["share"]["1417"]["all"]["share_all"] == 0.5
    assert summ["share"]["cells"]["region"]["3UTR"]["total"] == 40
    assert summ["same_site"]["1417_vs_1419"] == {"sites_a": 60, "matched": 60, "same_label": 60}


def test_share_ci_of_stratum_fraction_matches_brute_force():
    rng = np.random.default_rng(2)
    genes = [f"g{i % 9}" for i in range(80)]
    bands = list(rng.choice(["0-4", "8-9", "14-20"], 80))
    labels = ["nontemplated" if b == "0-4" else "templated" for b in bands]
    mol = _mol(labels, gene=genes, a_band=bands)
    ci = share_ci(mol, "a_band", n=300, seed=5, of="frac").set_index("stratum")
    names = np.unique(gene_clusters(mol))
    w = resample_counts(len(names), 300, 5)[:, np.searchsorted(names, gene_clusters(mol))].astype(float)
    for band in ("0-4", "8-9", "14-20"):
        frac = (w * (mol.a_band.to_numpy() == band)).sum(1) / w.sum(1)
        assert [ci.loc[band, "lo"], ci.loc[band, "hi"]] == pytest.approx(np.percentile(frac, [2.5, 97.5]).tolist())
        assert ci.loc[band, "lo"] < ci.loc[band, "hi"]
    # share_all is still the default, and it is fixed within a band by the labels
    assert share_ci(mol, "a_band", n=300, seed=5).set_index("stratum").loc["14-20"].tolist() == [1.0, 1.0]


def test_stage18_a_band_interval_is_of_the_band_fraction(tmp_path):
    base, out = tmp_path / "base", tmp_path / "burden"
    mol = _lib_mol(0)
    (base / "1417" / "truth").mkdir(parents=True)
    mol.to_parquet(base / "1417" / "truth" / "molecules_labeled.parquet")
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "18_burden.py"), "--base", str(base), "--out", str(out),
                        "--atlas", str(_atlas(tmp_path))], capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 0, r.stderr
    t = pd.read_parquet(out / "templated_share.parquet")
    band = t[(t.group == "1417") & (t.stratum_kind == "a_band")].set_index("stratum")
    want = share_ci(mol, "a_band", of="frac").set_index("stratum")
    assert band[["lo", "hi"]].equals(want[["lo", "hi"]].loc[band.index])
    # 14-20 holds the 30 templated molecules: frac 0.5, with genes of 5, 10 and 15 templated the interval is not a point
    assert band.loc["14-20", "frac"] == 0.5 and band.loc["14-20", "lo"] < 0.5 < band.loc["14-20", "hi"]
    region = t[(t.group == "1417") & (t.stratum_kind == "region")].set_index("stratum")
    assert region[["lo", "hi"]].equals(share_ci(mol, "region").set_index("stratum")[["lo", "hi"]].loc[region.index])


def _prox(t):
    """{(label, a_band): [n per bin in BINS order]} of a proximity table."""
    return {k: g.set_index("bin").n.reindex(list(BINS)).tolist() for k, g in t.groupby(["label", "a_band"], sort=False)}


def test_proximity_table_and_near_shares():
    mol = pd.DataFrame([("templated", "14-20"), ("templated", "14-20"), ("templated", "8-9"), ("templated", "10-13"),
                        ("templated", "10-13"), ("nontemplated", "0-4"), ("nontemplated", "0-4"), ("ambiguous", "5-7")],
                       columns=["label", "a_band"])
    dist = np.array([0, -11, 10, np.nan, 201, -10, 51, 200])
    null = np.array([5, -300, np.nan, 12, -10, 100, 2000, -8])
    t = proximity_table(mol, dist, null)
    assert list(t.columns) == ["label", "a_band", "bin", "n"] and len(t) == 35
    # labels in burden order with every bin (zeros kept), the null over all molecules, then the templated A bands
    assert list(dict.fromkeys(zip(t.label, t.a_band))) == [
        ("templated", "all"), ("nontemplated", "all"), ("ambiguous", "all"), ("null_shift1kb", "all"), ("templated", "8-9"),
        ("templated", "10-13"), ("templated", "14-20")]
    # NaN (no atlas site on the ref and strand) has its own bin, apart from > 200; null: 5, -10, -8 within 10, 12, 100,
    # -300 and 2000 beyond 200, NaN
    assert _prox(t) == {("templated", "all"): [2, 1, 0, 1, 1], ("nontemplated", "all"): [1, 0, 1, 0, 0],
                        ("ambiguous", "all"): [0, 0, 1, 0, 0], ("null_shift1kb", "all"): [3, 1, 1, 2, 1],
                        ("templated", "8-9"): [1, 0, 0, 0, 0], ("templated", "10-13"): [0, 0, 0, 1, 1],
                        ("templated", "14-20"): [1, 1, 0, 0, 0]}
    assert near_shares(t) == {"label": {"templated": 0.4, "nontemplated": 0.5, "ambiguous": 0.0, "null_shift1kb": 3 / 8},
                              "templated_a_band": {"8-9": 1.0, "10-13": 0.0, "14-20": 0.5}}
    # a label without molecules has no share; the null covers the molecules given (5, -10 of the first five)
    s = near_shares(proximity_table(mol[mol.label == "templated"], dist[:5], null[:5]))
    assert s["label"] == {"templated": 0.4, "nontemplated": None, "ambiguous": None, "null_shift1kb": 0.4}


def test_stage18_polyasite(tmp_path):
    base, out = tmp_path / "base", tmp_path / "burden"
    # 1419: sites 100 nt further, i 55 on chr2 minus (chr2 has a plus atlas site only), i 57-59 on chrM (absent from the
    # atlas), i 56 on chr1 minus at 150
    m19 = _lib_mol(100)
    m19.loc[55, ["ref", "strand"]] = ["chr2", "-"]
    m19.loc[56, ["strand", "site"]] = ["-", 150]
    m19.loc[[57, 58, 59], "ref"] = "chrM"
    for lib, mol in {"1417": _lib_mol(0), "1419": m19, "1179": _lib_mol(11)}.items():
        (base / lib / "truth").mkdir(parents=True)
        mol.to_parquet(base / lib / "truth" / "molecules_labeled.parquet")
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "18_burden.py"), "--base", str(base), "--out", str(out),
                        "--atlas", str(_atlas(tmp_path))], capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode == 0, r.stderr
    t = pd.read_parquet(out / "polyasite.parquet")
    assert list(t.columns) == ["group", "label", "a_band", "bin", "n"]
    assert list(dict.fromkeys(t.group)) == ["1417", "1419", "1179", "nuclei", "cells"]
    prox = {g: _prox(t[t.group == g].drop(columns="group")) for g in dict.fromkeys(t.group)}
    # Templated: i 0-4, 20-29, 40-54; non-templated i 6-18, 30-38 even, 56, 58; ambiguous the other i.
    # 1417, sites 100 + 50 i on chr1 plus, atlas at 100 and 1105. 0-10: i 0 (0), i 20 (+5); 11-50: i 1 (-50), i 21 (-45);
    # 51-200: i 2-4 (-100, -150, -200), i 17-19 (+155, +105, +55), i 22-24 (-95, -145, -195); the rest > 200.
    # Null, sites 1,000 nt upstream (-900 + 50 i): 0-10: i 20 (0), i 40 (+5); 11-50: i 19 (+50), i 21 (-50), i 41 (-45);
    # 51-200: i 16-18, 22-24, 37-39, 42-44; the other 43 > 200.
    want = {("templated", "all"): [2, 2, 6, 20, 0], ("nontemplated", "all"): [0, 0, 1, 13, 0],
            ("ambiguous", "all"): [0, 0, 2, 14, 0], ("null_shift1kb", "all"): [2, 3, 12, 43, 0],
            ("templated", "8-9"): [0] * 5, ("templated", "10-13"): [0] * 5, ("templated", "14-20"): [2, 2, 6, 20, 0]}
    assert prox["1417"] == want
    # 1419, sites 200 + 50 i. 0-10: i 18 (+5); 11-50: i 19 (-45); 51-200: i 0-2 (-100, -150, -200), i 15-17 (+155, +105,
    # +55), i 20-22 (-95, -145, -195); i 56 on minus: the minus atlas site 1150 lies 1,000 nt upstream (-1000, > 200);
    # i 55 and 57-59: no atlas site on their ref and strand.
    # Null (-800 + 50 i; i 56 on minus moves up to 1150): 0-10: i 18 (0), i 38 (+5), i 56 (0); 11-50: i 17, 19, 39;
    # 51-200: i 14-16, 20-22, 35-37, 40-42; none: i 55, 57-59; the other 38 > 200.
    assert prox["1419"] == {("templated", "all"): [0, 0, 6, 24, 0], ("nontemplated", "all"): [1, 0, 1, 11, 1],
                            ("ambiguous", "all"): [0, 1, 2, 10, 3], ("null_shift1kb", "all"): [3, 3, 12, 38, 4],
                            ("templated", "8-9"): [0] * 5, ("templated", "10-13"): [0] * 5,
                            ("templated", "14-20"): [0, 0, 6, 24, 0]}
    # 1179, sites 111 + 50 i. 0-10: i 20 (-6); 11-50: i 0 (-11), i 19 (+44); 51-200: i 1-3 (-61, -111, -161), i 16-18
    # (+194, +144, +94), i 21-23 (-56, -106, -156). Null (-889 + 50 i): 0-10: i 40 (-6); 11-50: i 19 (+39), i 20 (-11),
    # i 39 (+44); 51-200: i 16-18, 21-23, 36-38, 41-43; the other 44 > 200.
    assert prox["1179"] == {("templated", "all"): [1, 1, 6, 22, 0], ("nontemplated", "all"): [0, 0, 2, 12, 0],
                            ("ambiguous", "all"): [0, 1, 1, 14, 0], ("null_shift1kb", "all"): [1, 3, 12, 44, 0],
                            ("templated", "8-9"): [0] * 5, ("templated", "10-13"): [0] * 5,
                            ("templated", "14-20"): [1, 1, 6, 22, 0]}
    assert prox["nuclei"] == {k: [a + b for a, b in zip(v, prox["1419"][k])] for k, v in prox["1417"].items()}
    assert prox["cells"] == prox["1179"]
    p = json.loads((out / "summary.json").read_text())["polyasite"]
    assert p["atlas"] == str(_atlas(tmp_path)) and p["atlas_sites"] == 4
    # within 10 nt per group, from the counts above (bands 8-9 and 10-13 hold no molecule)
    within = {"1417": ([2 / 30, 0.0, 0.0, 2 / 60], 2 / 30), "1419": ([0.0, 1 / 14, 0.0, 3 / 60], 0.0),
              "1179": ([1 / 30, 0.0, 0.0, 1 / 60], 1 / 30), "nuclei": ([2 / 60, 1 / 28, 0.0, 5 / 120], 2 / 60),
              "cells": ([1 / 30, 0.0, 0.0, 1 / 60], 1 / 30)}
    assert list(p["within_10nt"]) == list(within)
    for g, (labels, band) in within.items():
        assert p["within_10nt"][g] == {
            "label": dict(zip(("templated", "nontemplated", "ambiguous", "null_shift1kb"), labels)),
            "templated_a_band": {"8-9": None, "10-13": None, "14-20": band}}, g
```

- [ ] **Step 4: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_evaluate.py tests/test_polyasite.py tests/test_burden.py`
Expected: FAIL; the first error is `ModuleNotFoundError: No module named 'pafbench.evaluate'`.

- [ ] **Step 5: Create `pafbench/evaluate.py`**

```python
"""Stage 17: grid metrics against the truth labels, setting selection, gene bootstrap, label-free and spike-in checks.

Read units are member reads within the member cap; molecule units are molecules with at least one such member, and
their reads are the pa reads plus those members. A molecule counts as removed when all its reads are removed
(polyAfilter's effect on a UMI count), or when most are (the molecule-level variant, spec 5.3).
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import sparse

from .config import A_BANDS
from .flags import parse_setting, setting_tag
from .grid import grid
from .labels import label_molecules
from .regions import REGION_NAMES

EXPR = ("low", "mid", "high")
MODULES = ("ERCC", "SIRV_E0", "long_SIRV")
LEVELS = ("read", "molecule", "molecule_majority", "molecule_weighted")
MOL_KINDS = ("all", "a_band", "region", "expr_tertile", "has_ts")
READ_KINDS = MOL_KINDS + ("dist_mode", "spliced")
STRATA = {"a_band": tuple(b[0] for b in A_BANDS), "region": REGION_NAMES, "expr_tertile": EXPR + ("intergenic",),
          "has_ts": ("True", "False"), "spliced": ("True", "False"), "module": MODULES}
SETTING_COLUMNS = ["tag", "covlen", "min_len", "mism", "introns"]
COUNT_COLUMNS = ["tp", "fn", "tn", "fp", "n_amb", "amb_removed"]
METRIC_COLUMNS = ["lib", "cap", "cutoffs"] + SETTING_COLUMNS + ["level", "stratum_kind", "stratum"] + COUNT_COLUMNS + \
    ["sens", "spec", "prec"]
READ_COLUMNS = ["qname", "mol", "dist5_exon", "dist_mode", "spliced"]
MOL_ATTRS = ["label", "a_band", "region", "gene", "has_ts", "expr_tertile"]


@dataclass
class Units:
    reads: pd.DataFrame
    R: np.ndarray
    mols: pd.DataFrame
    K: np.ndarray
    tags: list
    pos: str
    neg: str
    amb: str | None
    missing_reads: int
    dropped_mols: int


def tag_setting(tag):
    c, l, m, mode = tag.split("_")
    return parse_setting(f"{c[1:]},{l[1:]},{m[1:]},{mode}")


def group_sum(groups, n_groups, M, weight=None, chunk=64):
    """Column sums of the rows of M per group, [n_groups, cols]; int32 without weights. Sparse product, chunked over columns."""
    w = np.ones(len(groups), np.int32) if weight is None else np.asarray(weight, float)
    S = sparse.csr_matrix((w, (groups, np.arange(len(groups)))), shape=(n_groups, len(groups)))
    out = np.empty((n_groups, M.shape[1]), w.dtype)
    for j in range(0, M.shape[1], chunk):
        block = M[:, j:j + chunk]
        out[:, j:j + chunk] = S @ (block.view(np.uint8) if block.dtype == bool else block)
    return out


def expr_tertiles(mol):
    """low/mid/high tertile of each molecule's gene among genes ranked by molecule count (pd.qcut on the rank, ties by gene
    id; computed with np.quantile so that a single gene works too); 'intergenic' where gene is ''."""
    gene = mol.gene.to_numpy(object)
    counts = pd.Series(gene[gene != ""]).value_counts().sort_index().sort_values(kind="stable")
    rank = np.arange(1, len(counts) + 1)
    tert = pd.Series(np.asarray(EXPR, object)[np.searchsorted(np.quantile(rank, [1 / 3, 2 / 3]), rank)], index=counts.index)
    return pd.Series(gene).map(tert).fillna("intergenic").to_numpy(object)


def units(mol_labeled, truth_reads, truth_flags, tags, cap, label="label", pos="templated", neg="nontemplated",
          amb="ambiguous"):
    """Read and molecule units at a member cap. `label` names the label column; molecules whose label is not pos, neg or
    amb (amb None: not pos or neg) are left out. Molecules with a read absent from the primary rows of truth_flags drop out."""
    classes = [pos, neg] + ([] if amb is None else [amb])
    mol = mol_labeled[mol_labeled[label].isin(classes)]
    mol = mol.assign(label=mol[label].to_numpy(), expr_tertile=expr_tertiles(mol))
    tr = truth_reads[truth_reads.mol.isin(mol.mol)]
    mem = tr[(tr.role == "member") & (tr.dist5_exon <= cap)]
    reads = pd.concat([tr[(tr.role == "pa") & tr.mol.isin(mem.mol)], mem], ignore_index=True)
    fl = truth_flags[(truth_flags.flag.to_numpy() & 0x900) == 0]
    row = pd.Index(fl.qname).get_indexer(reads.qname)
    missing = row < 0
    bad = reads.mol[missing].unique()
    keep = ~reads.mol.isin(bad).to_numpy()
    reads, row = reads[keep].reset_index(drop=True), row[keep]
    mol = mol[mol.mol.isin(reads.mol)].reset_index(drop=True)
    mi = pd.Index(mol.mol).get_indexer(reads.mol)
    removed = fl[list(tags)].to_numpy(bool)[row]
    K = group_sum(mi, len(mol), removed)
    mol["n_reads"] = np.bincount(mi, minlength=len(mol))
    is_mem = (reads.role == "member").to_numpy()
    rd = reads.loc[is_mem, READ_COLUMNS].reset_index(drop=True)
    rd["mi"] = mi[is_mem]
    rd = pd.concat([rd, mol.loc[rd.mi, MOL_ATTRS].reset_index(drop=True)], axis=1)
    return Units(rd, removed[is_mem], mol, K, list(tags), pos, neg, amb, int(missing.sum()), int(len(bad)))


def relabel(u, cutoffs):
    """The same units with labels recomputed from a20 at other cut-offs."""
    lab = label_molecules(u.mols[["a20"]], cutoffs).label.to_numpy(object)
    return replace(u, mols=u.mols.assign(label=lab), reads=u.reads.assign(label=lab[u.reads.mi.to_numpy()]))


def _colsum(M, w):
    return np.count_nonzero(M, axis=0) if w is None else np.einsum("i,ij->j", w, M)


def confusion(removed, label, weight=None, pos="templated", neg="nontemplated"):
    """(tp, fn, tn, fp, n_amb, amb_removed) per column of removed; labels other than pos and neg count as ambiguous."""
    label = np.asarray(label)
    w = None if weight is None else np.asarray(weight, float)
    out = []
    for m in (label == pos, label == neg, (label != pos) & (label != neg)):
        wm = None if w is None else w[m]
        total = m.sum() if w is None else wm.sum()
        rem = _colsum(removed[m], wm)
        out.append((rem, total - rem, total))
    (tp, fn, _), (fp, tn, _), (amb_rem, _, n_amb) = out
    return tp, fn, tn, fp, np.full(removed.shape[1], n_amb), amb_rem


def rates(tp, fn, tn, fp):
    """(sensitivity, specificity, precision), NaN where the denominator is 0."""
    out = []
    for a, d in ((tp, tp + fn), (tn, tn + fp), (tp, tp + fp)):
        a, d = np.asarray(a, float), np.asarray(d, float)
        out.append(np.divide(a, d, out=np.full(a.shape, np.nan), where=d > 0))
    return tuple(out)


def _level(u, level):
    """(frame, removed [n, tags], weights or None) of one level."""
    if level == "read":
        return u.reads, u.R, None
    n = u.mols.n_reads.to_numpy()[:, None]
    if level == "molecule_majority":
        return u.mols, 2 * u.K > n, None
    return u.mols, u.K == n, (1.0 / n[:, 0] if level == "molecule_weighted" else None)


def _strata(u, frame, kinds):
    for kind in kinds:
        if kind == "all":
            yield kind, "all", slice(None)
            continue
        col = frame[kind].to_numpy() if kind in frame else u.mols[kind].to_numpy()[frame.mi.to_numpy()]
        col = col.astype(str)
        for value in STRATA.get(kind) or sorted(set(col)):
            yield kind, value, col == value


def _table(u, tags, levels, kinds_of):
    j = [u.tags.index(t) for t in tags]
    blocks, keys = [], []
    for level in levels:
        frame, removed, w = _level(u, level)
        removed = removed[:, j]
        label = frame.label.to_numpy()
        for kind, value, m in _strata(u, frame, kinds_of(level)):
            c = confusion(removed[m], label[m], None if w is None else w[m], u.pos, u.neg)
            blocks.append(np.column_stack(c).astype(float))
            keys.append((level, kind, value))
    t = len(tags)
    df = pd.DataFrame(np.vstack(blocks), columns=COUNT_COLUMNS)
    settings = [tag_setting(x) for x in tags]
    df.insert(0, "tag", np.tile(np.asarray(tags, object), len(keys)))
    df.insert(1, "covlen", np.tile([s.covlen for s in settings], len(keys)))
    df.insert(2, "min_len", pd.array(np.tile(np.asarray([s.min_len for s in settings], object), len(keys)), dtype="Int64"))
    df.insert(3, "mism", np.tile([s.mism for s in settings], len(keys)))
    df.insert(4, "introns", np.tile([s.introns for s in settings], len(keys)))
    for i, name in enumerate(("level", "stratum_kind", "stratum")):
        df.insert(5 + i, name, np.repeat(np.asarray([k[i] for k in keys], object), t))
    df["sens"], df["spec"], df["prec"] = rates(df.tp.to_numpy(), df.fn.to_numpy(), df.tn.to_numpy(), df.fp.to_numpy())
    return df


def evaluate(u, lib, cap, cutoffs_name="8/4", levels=LEVELS, kinds=None):
    """Metrics of every tag of u per level and stratum (kinds None: all strata of each level)."""
    df = _table(u, u.tags, levels, lambda level: kinds or (READ_KINDS if level == "read" else MOL_KINDS))
    df.insert(0, "lib", lib)
    df.insert(1, "cap", cap)
    df.insert(2, "cutoffs", cutoffs_name)
    return df[METRIC_COLUMNS]


def spikein_metrics(u, tags):
    """Read and molecule metrics overall and per spike-in module (units built on spike_label)."""
    return _table(u, tags, ("read", "molecule"), lambda level: ("all", "module"))


def _num(x):
    return None if x is None or pd.isna(x) else float(x)


def select_setting(metrics, removed_by_tag, introns, lib="1417", cap=1000, cutoffs="8/4", min_spec=0.95):
    """The selection rule of spec 7 over the grid settings of one mode (remove-all control and off-grid published settings
    excluded): highest molecule-level sensitivity with specificity >= min_spec; ties by fewer removed primary records, then
    by tag. Also returns the sensitivity/specificity Pareto front of all candidates."""
    cand = {setting_tag(s) for s in grid() if s.introns == introns and s.min_len is not None}
    m = metrics[(metrics.lib == lib) & (metrics.cap == cap) & (metrics.cutoffs == cutoffs) & (metrics.level == "molecule")
                & (metrics.stratum_kind == "all") & metrics.tag.isin(cand)]
    n_cand = len(m)
    m = m[m.sens.notna() & m.spec.notna()]
    se, sp = m.sens.to_numpy(float), m.spec.to_numpy(float)
    ge = (se[None, :] >= se[:, None]) & (sp[None, :] >= sp[:, None])
    gt = (se[None, :] > se[:, None]) | (sp[None, :] > sp[:, None])
    front = m[~(ge & gt).any(axis=1)].sort_values(["sens", "spec", "tag"], ascending=[False, False, True])
    ok = m[m.spec >= min_spec]
    best = ok.assign(removed=ok.tag.map(removed_by_tag)).sort_values(["sens", "removed", "tag"], ascending=[False, True, True])
    res = {"tag": None, "sens": None, "spec": None, "prec": None}
    if len(best):
        r = best.iloc[0]
        res = {"tag": r.tag, "sens": _num(r.sens), "spec": _num(r.spec), "prec": _num(r.prec)}
    return res | {"n_candidates": n_cand, "n_qualifying": len(ok),
                  "frontier": [[t, float(a), float(b)] for t, a, b in zip(front.tag, front.sens, front.spec)]}


def gene_clusters(mol):
    """Bootstrap cluster of each molecule: its gene, or 'ref:strand:site // 10_000' when intergenic."""
    gene = mol.gene.to_numpy(object)
    locus = (mol.ref.astype(str) + ":" + mol.strand.astype(str) + ":" + (mol.site // 10_000).astype(str)).to_numpy(object)
    return np.where(gene != "", gene, locus)


def resample_counts(n_clusters, n, seed):
    """[n, n_clusters] draw counts of a cluster bootstrap (each resample draws n_clusters clusters with replacement)."""
    return np.random.default_rng(seed).multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=n)


def interval(values):
    """(2.5th, 97.5th percentile, number of finite values) of bootstrap values."""
    v = values[np.isfinite(values)]
    lo, hi = np.percentile(v, [2.5, 97.5]) if v.size else (np.nan, np.nan)
    return float(lo), float(hi), int(v.size)


def bootstrap(u, tags, level="molecule", n=1000, seed=0):
    """Gene-cluster bootstrap of sens/spec/prec: per-cluster confusion counts once, then multinomial cluster draws (clusters
    in sorted order) applied by matrix products. Clusters without a positive or negative unit at this level add nothing to
    the counts and are left out of the draws and of n_clusters."""
    frame, removed, w = _level(u, level)
    removed = removed[:, [u.tags.index(t) for t in tags]]
    cl = gene_clusters(u.mols)
    if level == "read":
        cl = cl[frame.mi.to_numpy()]
    names, code = np.unique(cl, return_inverse=True)
    label = frame.label.to_numpy()
    w = np.ones(len(frame)) if w is None else w
    parts = []
    for m in (label == u.pos, label == u.neg):
        rem = group_sum(code[m], len(names), removed[m], w[m])
        total = np.bincount(code[m], weights=w[m], minlength=len(names))
        parts.append((rem, total[:, None] - rem))
    keep = np.bincount(code[(label == u.pos) | (label == u.neg)], minlength=len(names)) > 0
    (tp, fn), (fp, tn) = ((a[keep], b[keep]) for a, b in parts)
    n_clusters = int(keep.sum())
    W = resample_counts(n_clusters, n, seed).astype(float)
    est = rates(*(x.sum(axis=0) for x in (tp, fn, tn, fp)))
    boot = rates(*(W @ x for x in (tp, fn, tn, fp)))
    rows = []
    for k, tag in enumerate(tags):
        for metric, e, b in zip(("sens", "spec", "prec"), est, boot):
            lo, hi, nb = interval(b[:, k])
            rows.append((tag, metric, float(e[k]), lo, hi, nb, n_clusters))
    return pd.DataFrame(rows, columns=["tag", "metric", "estimate", "lo", "hi", "n_boot", "n_clusters"])


def ambiguous_tail_share(mol_labeled):
    """Spec 7: tail-primed share of the ambiguous molecules from AATAAA/ATTAAA rates, a two-component mixture of the
    templated and non-templated rates: (rate_ambiguous - rate_templated) / (rate_nontemplated - rate_templated), clipped to
    [0, 1]. The estimate is None when a class is empty or the two reference rates are equal."""
    lab = mol_labeled.label.to_numpy()
    hexa = mol_labeled.hexamer.to_numpy(bool)
    names = ("templated", "nontemplated", "ambiguous")
    n = {k: int((lab == k).sum()) for k in names}
    rate = {k: float(hexa[lab == k].mean()) if n[k] else None for k in names}
    t, nt, a = (rate[k] for k in names)
    est = None if None in (t, nt, a) or nt == t else float(np.clip((a - t) / (nt - t), 0.0, 1.0))
    return {"estimate": est} | {f"rate_{k}": rate[k] for k in names} | {f"n_{k}": n[k] for k in names}


def _mean(x):
    return float(x.mean()) if x.size else np.nan


def hexamer_check(u, tags):
    """AATAAA/ATTAAA rate among removed and kept molecules (molecule rule), overall and per label."""
    _, removed, _ = _level(u, "molecule")
    removed = removed[:, [u.tags.index(t) for t in tags]]
    hexa = u.mols.hexamer.to_numpy(bool)
    label = u.mols.label.to_numpy()
    rows = []
    for k, tag in enumerate(tags):
        for name in ("all", u.pos, u.neg) + (() if u.amb is None else (u.amb,)):
            m = np.ones(len(label), bool) if name == "all" else label == name
            r, kept = m & removed[:, k], m & ~removed[:, k]
            rows.append((tag, name, int(r.sum()), _mean(hexa[r]), int(kept.sum()), _mean(hexa[kept])))
    return pd.DataFrame(rows, columns=["tag", "label", "removed_n", "removed_hex_rate", "kept_n", "kept_hex_rate"])
```

- [ ] **Step 6: Create `pafbench/polyasite.py`**

```python
"""PolyASite 3.0 proximity of priming sites (spec 4.1). Reported next to the labels, never used to validate them: the atlas
comes from 10x 3' reads primed by the same oligo(dT), and its sites are not screened for A-rich genome downstream."""
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv

ATLAS_COLUMNS = ["chrom", "start", "end", "id", "score", "strand", "tissues_pct", "protocols", "stringency", "class", "signals"]
BINS = ("0-10", "11-50", "51-200", ">200", "no atlas site on this ref and strand")
NULL_SHIFT = 1000            # chance level: the same sites moved this many nt upstream in transcript sense
NULL_LABEL = "null_shift1kb"


def load_polyasite(path, min_score=None):
    """{(ref, strd): sorted unique int64 ndarray} of the representative cleavage sites of a PolyASite 3.0 atlas BED (the pos
    of the id chrom:pos:strand, 1-based per the atlas documentation), kept when score (column 5, mean RPM) >= min_score.

    Converted to the priming_site boundary (last cDNA base at site - 1 on plus, at site on minus): pos on plus, pos + 1 on
    minus. With the documented reading on both strands (the base [pos - 1, pos) is the last templated base), the mode of the
    signed distance from tail-primed 1417 pa sites (a20 <= 4) to the atlas is 0 on plus but 2 nt downstream on minus, and from
    GENCODE 44 transcript ends 1 nt upstream on plus but 1 nt downstream on minus; pos + 1 on minus makes the strands agree.
    """
    reader = pcsv.open_csv(path, read_options=pcsv.ReadOptions(column_names=ATLAS_COLUMNS, block_size=1 << 26),
                           parse_options=pcsv.ParseOptions(delimiter="\t"),
                           convert_options=pcsv.ConvertOptions(include_columns=["chrom", "id", "score", "strand"],
                                                               column_types={"chrom": pa.string(), "id": pa.string(),
                                                                             "score": pa.float64(), "strand": pa.string()}))
    parts = defaultdict(list)
    for b in reader:
        if min_score is not None:
            b = b.filter(pc.greater_equal(b.column("score"), min_score))
        pos = pc.cast(pc.list_element(pc.split_pattern(b.column("id"), ":"), 1), pa.int64()).to_numpy()
        key = pc.binary_join_element_wise(b.column("chrom"), b.column("strand"), "\t").dictionary_encode()
        code = key.indices.to_numpy(zero_copy_only=False)
        for k, name in enumerate(key.dictionary.to_pylist()):
            parts[name].append(pos[code == k])
    out = {}
    for name, chunks in parts.items():
        ref, strand = name.split("\t")
        out[(ref, strand == "+")] = np.unique(np.concatenate(chunks) + (strand == "-"))
    return out


def nearest_signed(sites, strd, ends):
    """Signed distance from each site to the nearest of ends (sorted atlas sites of the same ref and strand, strd True on
    plus); d > 0 when that atlas site lies downstream in transcript sense (spikein.label_spike_site's rule; equal |d|: the
    positive one). All NaN when ends is None or empty."""
    sites = np.asarray(sites, np.int64)
    if ends is None or len(ends) == 0:
        return np.full(len(sites), np.nan)
    sign = 1 if strd else -1
    i = np.searchsorted(ends, sites)
    far = np.int64(1) << 62
    right = np.where(i < len(ends), ends[np.minimum(i, len(ends) - 1)] - sites, far) * sign
    left = np.where(i > 0, ends[np.maximum(i - 1, 0)] - sites, -far) * sign
    a, b = np.abs(right), np.abs(left)
    return np.where(a < b, right, np.where(b < a, left, np.maximum(right, left))).astype(np.float64)


def site_distances(mol, atlas, upstream=0):
    """nearest_signed for every row of mol (ref, strand '+'/'-', site) against load_polyasite's dict, in row order, after
    moving each site by upstream nt toward the 5' end in transcript sense (lower coordinate on plus, higher on minus)."""
    d = np.full(len(mol), np.nan)
    site = mol.site.to_numpy(np.int64)
    for (ref, strand), idx in mol.groupby(["ref", "strand"], sort=False).indices.items():
        strd = strand == "+"
        d[idx] = nearest_signed(site[idx] - upstream if strd else site[idx] + upstream, strd, atlas.get((ref, strd)))
    return d


def distance_bin(d):
    """BINS name of each distance by |d|; NaN (no atlas site on the ref and strand, e.g. a contig the atlas lacks) is the
    last bin."""
    a = np.abs(np.asarray(d, np.float64))
    return np.select([a <= 10, a <= 50, a <= 200, a > 200], BINS[:4], BINS[4]).astype(object)
```

- [ ] **Step 7: Create `pafbench/burden.py`**

```python
"""Stage 18: internal-priming burden (templated share of labeled molecules), same-site and same-gene comparisons, selection bias.

Descriptive only (spec 8): tissue, maturity, cell-type mix and the cell/nucleus difference are confounded.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import A_BANDS, LABEL_CUTOFFS
from .evaluate import gene_clusters, interval, resample_counts
from .polyasite import BINS, NULL_LABEL, distance_bin

LABELS = ("templated", "nontemplated", "ambiguous")
TEMPLATED_BANDS = tuple(name for name, lo, _ in A_BANDS if lo >= LABEL_CUTOFFS["templated_min"])


def _stratum(mol, by):
    return np.full(len(mol), "all", object) if by is None else mol[by].astype(str).to_numpy(object)


def share_table(mol, by=None):
    """Label counts per stratum (by None: one stratum 'all'); frac = stratum's share of molecules, share_all = templated /
    all, share_resolved = templated / (templated + nontemplated)."""
    t = pd.crosstab(pd.Series(_stratum(mol, by), name="stratum"), mol.label.to_numpy()).reindex(columns=list(LABELS), fill_value=0)
    t.columns.name = None
    t["total"] = t[list(LABELS)].sum(axis=1)
    t["frac"] = t.total / t.total.sum()
    t["share_all"] = t.templated / t.total
    t["share_resolved"] = t.templated / (t.templated + t.nontemplated)
    return t.reset_index()


def share_ci(mol, by=None, n=1000, seed=0, of="share_all"):
    """Gene-cluster bootstrap 95% interval per stratum (clusters and draws as evaluate.bootstrap) of share_all, or with
    of="frac" of the stratum's share of all molecules (for A bands, where the labels fix share_all)."""
    names, code = np.unique(gene_clusters(mol), return_inverse=True)
    strata, s = np.unique(_stratum(mol, by), return_inverse=True)
    total = np.zeros((len(names), len(strata)))
    templ = np.zeros_like(total)
    np.add.at(total, (code, s), 1)
    t = mol.label.to_numpy() == "templated"
    np.add.at(templ, (code[t], s[t]), 1)
    W = resample_counts(len(names), n, seed).astype(float)
    num, den = W @ templ, W @ total
    if of == "frac":
        num, den = den, den.sum(axis=1, keepdims=True)
    share = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
    ci = [interval(share[:, k])[:2] for k in range(len(strata))]
    return pd.DataFrame({"stratum": strata, "lo": [c[0] for c in ci], "hi": [c[1] for c in ci]})


def site_labels(mol):
    """One row per site (ref, strand, site) labeled with the most common label of its molecules (ties by label name)."""
    c = mol.groupby(["ref", "strand", "site", "label"]).size().reset_index(name="n")
    c = c.sort_values(["ref", "strand", "site", "n", "label"], ascending=[True, True, True, False, True])
    return c.drop_duplicates(["ref", "strand", "site"])[["ref", "strand", "site", "label"]].reset_index(drop=True)


def same_site(a, b, tol=5):
    """Each site of a matched to the nearest site of b on the same ref and strand within tol nt (equal distance: the lower
    coordinate); label pairs counted, label_b 'none' when unmatched."""
    sa, sb = site_labels(a), site_labels(b)
    keys = {k: i for i, k in enumerate(sorted(set(zip(sa.ref, sa.strand)) | set(zip(sb.ref, sb.strand))))}

    def coord(s):
        return np.fromiter((keys[k] for k in zip(s.ref, s.strand)), np.int64, len(s)) * (1 << 40) + s.site.to_numpy(np.int64)

    ca, cb = coord(sa), coord(sb)
    order = np.argsort(cb, kind="stable")
    far = 1 << 62
    cb = np.concatenate([[-far], cb[order], [far]])
    lb = np.concatenate([["none"], sb.label.to_numpy(object)[order], ["none"]])
    i = np.searchsorted(cb, ca)
    dl, dr = ca - cb[i - 1], cb[i] - ca
    near = np.where(dl <= dr, i - 1, i)
    label_b = np.where(np.minimum(dl, dr) <= tol, lb[near], "none")
    pairs = pd.DataFrame({"label_a": sa.label.to_numpy(object), "label_b": label_b})
    return pairs.groupby(["label_a", "label_b"]).size().reset_index(name="n")


def same_gene(a, b, min_n=20):
    """Spearman correlation of share_all across genes with at least min_n labeled molecules in both groups."""
    def per_gene(m):
        g = m[m.gene != ""]
        return g.assign(t=(g.label == "templated").to_numpy()).groupby("gene").t.agg(["sum", "size"])

    j = per_gene(a).join(per_gene(b), lsuffix="_a", rsuffix="_b", how="inner")
    j = j[(j["size_a"] >= min_n) & (j["size_b"] >= min_n)]
    rho, p = spearmanr(j["sum_a"] / j["size_a"], j["sum_b"] / j["size_b"])
    return {"n_genes": len(j), "spearman_rho": float(rho), "p_value": float(p)}


def selection_bias(mol, region_counts):
    """Share of labeled molecules per RE value ('' when absent) against the share of all primary records (totals of one tag)."""
    labeled = mol.re.fillna("").astype(str).value_counts(normalize=True)
    rc = region_counts[region_counts.tag == region_counts.tag.iloc[0]].groupby("re").total.sum()
    out = pd.concat([labeled.rename("labeled_share"), (rc / rc.sum()).rename("records_share")], axis=1).fillna(0.0)
    return out.rename_axis("re").sort_index().reset_index()


def proximity_table(mol, dist, null):
    """Molecules per PolyASite distance bin (polyasite.distance_bin; dist and null one value per row of mol): each label
    with a_band 'all', then NULL_LABEL with a_band 'all' (every molecule, binned by null, the distances of the sites moved
    upstream), then templated molecules per templated A band; every bin kept, zeros included."""
    b, label, band = distance_bin(dist), mol.label.to_numpy(), mol.a_band.to_numpy()
    blocks = [(lab, "all", b, label == lab) for lab in LABELS]
    blocks.append((NULL_LABEL, "all", distance_bin(null), np.ones(len(mol), bool)))
    blocks += [("templated", x, b, (label == "templated") & (band == x)) for x in TEMPLATED_BANDS]
    rows = [(lab, x, name, int((bb[m] == name).sum())) for lab, x, bb, m in blocks for name in BINS]
    return pd.DataFrame(rows, columns=["label", "a_band", "bin", "n"])


def near_shares(table):
    """Share of molecules within 10 nt of an atlas site per label (NULL_LABEL included, the chance level) and per templated
    A band of one group's proximity_table (None without molecules)."""
    def share(label, band):
        t = table[(table.label == label) & (table.a_band == band)]
        total = int(t.n.sum())
        return int(t[t.bin == BINS[0]].n.sum()) / total if total else None

    return {"label": {lab: share(lab, "all") for lab in LABELS + (NULL_LABEL,)},
            "templated_a_band": {x: share("templated", x) for x in TEMPLATED_BANDS}}
```

- [ ] **Step 8: Create `scripts/17_evaluate.py`**

```python
"""Stage 17: grid metrics per library and member cap, setting selection (spec 7), bootstrap, hexamer check, spike-in metrics."""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from pafbench.config import CELLS, MEMBER_CAP, NUCLEI, OUT, SENS_CAPS
from pafbench.evaluate import (COUNT_COLUMNS, ambiguous_tail_share, bootstrap, evaluate, hexamer_check, relabel, select_setting,
                               spikein_metrics, tag_setting, units)
from pafbench.flags import Setting, setting_tag
from pafbench.grid import all_settings, published

RULE = ("On 1417, in each mode, the grid setting (remove-all control and off-grid published settings excluded) with the "
        "highest molecule-level sensitivity subject to specificity >= 0.95, at member cap 1000 nt and 8/4 labels; ties broken "
        "by fewer removed primary records, then by tag name. Reported on 1419 and 1179 without retuning. If no setting "
        "reaches 0.95, the frontier is reported and none is chosen.")
MIN_SPEC = 0.95
SEL_LIB = "1417"
CUTOFFS = "8/4"
MODES = {"e": False, "i": True}
SUMMARY_LEVELS = ("molecule", "molecule_majority", "read")
BOOT_LEVELS = ("molecule", "read")


def _num(x):
    return None if pd.isna(x) else float(x)


def _files(base, lib, sub=""):
    d = base / lib / sub
    return {"mol": d / "truth" / "molecules_labeled.parquet", "reads": d / "truth" / "truth_reads.parquet",
            "flags": d / "grid" / "truth_flags.parquet"}


def _load(f):
    fl = pd.read_parquet(f["flags"])
    tags = [setting_tag(s) for s in all_settings() if setting_tag(s) in fl.columns]
    return pd.read_parquet(f["mol"]), pd.read_parquet(f["reads"]), fl, tags


def _summary(metrics, tag, lib):
    m = metrics[(metrics.tag == tag) & (metrics.lib == lib) & (metrics.cap == MEMBER_CAP) & (metrics.cutoffs == CUTOFFS)
                & (metrics.stratum_kind == "all") & metrics.level.isin(SUMMARY_LEVELS)]
    return {r["level"]: {k: _num(r[k]) for k in COUNT_COLUMNS + ["sens", "spec", "prec"]} for _, r in m.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=OUT)
    ap.add_argument("--out", type=Path, default=OUT / "eval")
    args = ap.parse_args()
    t0 = time.time()
    libs, missing = [], {}
    for lib in NUCLEI + CELLS:
        need = list(_files(args.base, lib).values()) + [args.base / lib / "grid" / "region_counts.parquet"]
        absent = [str(p) for p in need if not p.exists()]
        if absent:
            missing[lib] = absent
            print(f"{lib}: skipped, missing {', '.join(absent)}")
        else:
            libs.append(lib)
    if not libs:
        raise SystemExit("no library has truth and grid files")
    args.out.mkdir(parents=True, exist_ok=True)

    metrics, u_main, removed_by_tag, unit_stats, tail_share = [], {}, {}, {}, {}
    for lib in libs:
        mol, tr, fl, tags = _load(_files(args.base, lib))
        tail_share[lib] = ambiguous_tail_share(mol)
        unit_stats[lib] = {}
        for cap in SENS_CAPS:
            u = units(mol, tr, fl, tags, cap)
            metrics.append(evaluate(u, lib, cap, CUTOFFS))
            unit_stats[lib][str(cap)] = {"read_units": len(u.reads), "molecule_units": len(u.mols),
                                         "missing_reads": u.missing_reads, "dropped_molecules": u.dropped_mols}
            if cap == MEMBER_CAP:
                u_main[lib] = u
        rc = pd.read_parquet(args.base / lib / "grid" / "region_counts.parquet")
        removed_by_tag[lib] = {k: int(v) for k, v in rc.groupby("tag").removed.sum().items()}
        print(f"{lib}: {len(tags)} tags, units {json.dumps(unit_stats[lib])}")
    metrics = pd.concat(metrics, ignore_index=True)
    metrics.to_parquet(args.out / "metrics.parquet", index=False)

    removed_sel = removed_by_tag.get(SEL_LIB, {})
    modes = {}
    for mode, introns in MODES.items():
        s = select_setting(metrics, removed_sel, introns, SEL_LIB, MEMBER_CAP, CUTOFFS, MIN_SPEC)
        modes[mode] = {"introns": introns, "selection": s,
                       "metrics": {lib: _summary(metrics, s["tag"], lib) for lib in libs} if s["tag"] else {}}
    pub = {setting_tag(p): {lib: _summary(metrics, setting_tag(p), lib) for lib in libs} for p in published()}
    cap_sens = {str(cap): {mode: select_setting(metrics, removed_sel, introns, SEL_LIB, cap, CUTOFFS, MIN_SPEC)
                           for mode, introns in MODES.items()} for cap in SENS_CAPS}
    sweep = []
    for tmin in range(7, 11):
        for nmax in range(2, 7):
            if nmax >= tmin:
                continue
            name = f"{tmin}/{nmax}"
            cut = {"templated_min": tmin, "nontemplated_max": nmax}
            m = pd.concat([evaluate(relabel(u_main[lib], cut), lib, MEMBER_CAP, name, levels=("molecule",), kinds=("all",))
                           for lib in libs], ignore_index=True)
            for mode, introns in MODES.items():
                s = select_setting(m, removed_sel, introns, SEL_LIB, MEMBER_CAP, name, MIN_SPEC)
                held = {}
                if s["tag"]:
                    for lib in libs:
                        r = m[(m.lib == lib) & (m.tag == s["tag"])]
                        held[lib] = {k: _num(r[k].iloc[0]) for k in ("sens", "spec", "prec")} if len(r) else None
                sweep.append({"templated_min": tmin, "nontemplated_max": nmax, "mode": mode, "tag": s["tag"],
                              "n_qualifying": s["n_qualifying"], "by_library": held})

    chosen = {mode: d["selection"]["tag"] for mode, d in modes.items()}
    boot_tags = [t for t in chosen.values() if t]
    boot_tags += [setting_tag(p) for p in published()]
    for mode, introns in MODES.items():
        if chosen[mode]:
            boot_tags.append(setting_tag(Setting(tag_setting(chosen[mode]).covlen, None, 0, introns)))
        else:
            print(f"mode {mode}: no setting chosen, so no chosen tag and no remove-all control at its COVLEN are bootstrapped")
    boot_tags = list(dict.fromkeys(boot_tags))
    boots, hexes = [], []
    for lib in libs:
        u = u_main[lib]
        here = [t for t in boot_tags if t in u.tags]
        absent = [t for t in boot_tags if t not in u.tags]
        if absent:
            print(f"{lib}: not in truth_flags, not bootstrapped: {', '.join(absent)}")
        for level in BOOT_LEVELS:
            b = bootstrap(u, here, level=level)
            b.insert(0, "lib", lib)
            b.insert(1, "cap", MEMBER_CAP)
            b.insert(2, "level", level)
            boots.append(b)
        h = hexamer_check(u, here)
        h.insert(0, "lib", lib)
        h.insert(1, "cap", MEMBER_CAP)
        hexes.append(h)
    pd.concat(boots, ignore_index=True).to_parquet(args.out / "bootstrap.parquet", index=False)
    pd.concat(hexes, ignore_index=True).to_parquet(args.out / "hexamer_check.parquet", index=False)

    spikes = []
    for lib in NUCLEI:
        f = _files(args.base, lib, "spikein")
        absent = [str(p) for p in f.values() if not p.exists()]
        if absent:
            print(f"{lib}: spike-in metrics skipped, missing {', '.join(absent)}")
            continue
        mol, tr, fl, tags = _load(f)
        u = units(mol, tr, fl, tags, MEMBER_CAP, label="spike_label", pos="internal", neg="tail", amb=None)
        s = spikein_metrics(u, tags)
        s.insert(0, "lib", lib)
        s.insert(1, "cap", MEMBER_CAP)
        spikes.append(s)
    if spikes:
        pd.concat(spikes, ignore_index=True).to_parquet(args.out / "spikein_metrics.parquet", index=False)

    out = {"rule": RULE, "min_spec": MIN_SPEC, "selection_library": SEL_LIB, "cap": MEMBER_CAP, "cutoffs": CUTOFFS,
           "libraries": libs, "missing": missing, "units": unit_stats, "modes": modes, "published": pub,
           "cap_sensitivity": cap_sens, "cutoff_sweep": sweep, "bootstrapped_tags": boot_tags,
           "ambiguous_tail_share": tail_share}
    (args.out / "selection.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"libraries": libs, "chosen": chosen, "metrics_rows": len(metrics), "bootstrap_tags": len(boot_tags),
                      "spikein_libraries": [s.lib.iloc[0] for s in spikes], "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 9: Create `scripts/18_burden.py`**

```python
"""Stage 18: templated share by library and group with gene-bootstrap CIs, same-site and same-gene comparisons, selection bias,
PolyASite 3.0 proximity of the labeled sites (reported only, not a label check: the atlas is built from 10x 3' reads of the
same oligo(dT) priming and keeps sites with A-rich genome downstream).

In templated_share.parquet, lo and hi bound share_all, except on the a_band rows: there they bound frac (the band's share
of the molecules), because the labels are defined from the A count and fix share_all within a band.

polyasite.parquet (group, label, a_band, bin, n): molecules per bin of |d|, the distance from the site to the nearest atlas
site on the same ref and strand: "0-10", "11-50", "51-200", ">200", "no atlas site on this ref and strand" (contigs the
atlas lacks, such as chrM and unplaced contigs). Per group the label rows (a_band "all"), then label "null_shift1kb"
(a_band "all": every molecule of the group with its site moved 1,000 nt upstream in transcript sense on the same ref and
strand, binned the same way; the share expected by chance at this atlas density), then the templated molecules per
templated A band. summary.json polyasite.within_10nt[group]: the "0-10" share per label, null_shift1kb beside them, and
per templated A band.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from pafbench.burden import near_shares, proximity_table, same_gene, same_site, selection_bias, share_ci, share_table
from pafbench.config import CELLS, NUCLEI, OUT, POLYASITE
from pafbench.polyasite import NULL_SHIFT, load_polyasite, site_distances

KINDS = (None, "region", "a_band", "has_ts")
COLUMNS = ["mol", "ref", "strand", "site", "label", "a_band", "region", "gene", "has_ts", "re"]


def _num(x):
    return None if pd.isna(x) else float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=OUT)
    ap.add_argument("--out", type=Path, default=OUT / "burden")
    ap.add_argument("--atlas", type=Path, default=POLYASITE)
    args = ap.parse_args()
    t0 = time.time()
    mols = {}
    for lib in NUCLEI + CELLS:
        p = args.base / lib / "truth" / "molecules_labeled.parquet"
        if p.exists():
            mols[lib] = pd.read_parquet(p, columns=COLUMNS)
        else:
            print(f"{lib}: skipped, missing {p}")
    if not mols:
        raise SystemExit("no library has truth files")
    args.out.mkdir(parents=True, exist_ok=True)
    groups, members = dict(mols), {lib: [lib] for lib in mols}
    for name, libs in (("nuclei", [lib for lib in NUCLEI if lib in mols]), ("cells", [lib for lib in CELLS if lib in mols])):
        if libs:
            groups[name] = pd.concat([mols[lib] for lib in libs], ignore_index=True)
            members[name] = libs

    tables = []
    for g, mol in groups.items():
        for by in KINDS:
            t = share_table(mol, by).merge(share_ci(mol, by, of="frac" if by == "a_band" else "share_all"), on="stratum")
            t.insert(0, "group", g)
            t.insert(1, "stratum_kind", by or "all")
            tables.append(t)
    share = pd.concat(tables, ignore_index=True)
    share.to_parquet(args.out / "templated_share.parquet", index=False)

    pairs = []
    if "1417" in mols and "1419" in mols:
        pairs.append(("1417_vs_1419", mols["1417"], mols["1419"]))
    if "nuclei" in groups and "cells" in groups:
        pairs.append(("nuclei_vs_cells", groups["nuclei"], groups["cells"]))
    sites, genes = [], []
    for name, a, b in pairs:
        s = same_site(a, b)
        s.insert(0, "pair", name)
        sites.append(s)
        genes.append({"pair": name} | same_gene(a, b))
    site_cols = ["pair", "label_a", "label_b", "n"]
    (pd.concat(sites, ignore_index=True) if sites else pd.DataFrame(columns=site_cols)).to_parquet(
        args.out / "same_site.parquet", index=False)
    pd.DataFrame(genes, columns=["pair", "n_genes", "spearman_rho", "p_value"]).to_parquet(args.out / "same_gene.parquet", index=False)

    bias = []
    for lib, mol in mols.items():
        p = args.base / lib / "grid" / "region_counts.parquet"
        if not p.exists():
            print(f"{lib}: selection bias skipped, missing {p}")
            continue
        b = selection_bias(mol, pd.read_parquet(p))
        b.insert(0, "lib", lib)
        bias.append(b)
    if bias:
        pd.concat(bias, ignore_index=True).to_parquet(args.out / "selection_bias.parquet", index=False)

    t1 = time.time()
    atlas = load_polyasite(args.atlas)
    atlas_seconds = time.time() - t1
    prox = []
    for g, mol in groups.items():
        t = proximity_table(mol, site_distances(mol, atlas), site_distances(mol, atlas, upstream=NULL_SHIFT))
        t.insert(0, "group", g)
        prox.append(t)
    prox = pd.concat(prox, ignore_index=True)
    prox.to_parquet(args.out / "polyasite.parquet", index=False)

    def entry(r):
        return {"share_all": _num(r.share_all), "lo": _num(r.lo), "hi": _num(r.hi), "share_resolved": _num(r.share_resolved),
                "total": int(r.total)}

    headline = {}
    for g in groups:
        d = share[share.group == g]
        headline[g] = {"all": entry(d[d.stratum_kind == "all"].iloc[0])}
        for kind in ("region", "has_ts"):
            headline[g][kind] = {r.stratum: entry(r) for r in d[d.stratum_kind == kind].itertuples()}
    site_summary = {}
    for s in sites:
        n = int(s.n.sum())
        matched = s[s.label_b != "none"]
        site_summary[s.pair.iloc[0]] = {"sites_a": n, "matched": int(matched.n.sum()),
                                        "same_label": int(matched[matched.label_a == matched.label_b].n.sum())}
    out = {"libraries": list(mols), "groups": members, "share": headline,
           "same_gene": {d["pair"]: {"n_genes": d["n_genes"], "spearman_rho": _num(d["spearman_rho"]),
                                     "p_value": _num(d["p_value"])} for d in genes},
           "same_site": site_summary,
           "polyasite": {"atlas": str(args.atlas), "atlas_sites": int(sum(len(v) for v in atlas.values())),
                         "within_10nt": {g: near_shares(prox[prox.group == g]) for g in groups}}}
    (args.out / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"groups": members, "share_all": {g: headline[g]["all"]["share_all"] for g in groups},
                      "pairs": [p[0] for p in pairs], "atlas_seconds": round(atlas_seconds, 1),
                      "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `207 passed, 2 warnings`.

- [ ] **Step 11: Stage 17**

Run: `.venv/bin/python -m pafbench.memwatch /data/element_runs/polyafilter_bench/runs/17_evaluate.mem.json -- .venv/bin/python scripts/17_evaluate.py`
Expected (about a minute): metrics for 1417, 1419 and 1179 at caps 500, 1000 and 2000; `selection.json` has for each mode a tag or `null` plus the frontier, and the chosen tags' metrics on all three libraries; spike-in metrics for 1417 and 1419 only (a printed note says 1179 has none). Do not change the rule or the cut-offs after looking at the result: if no setting reaches specificity 0.95, that is the result.

- [ ] **Step 12: Stage 18**

Run: `.venv/bin/python -m pafbench.memwatch /data/element_runs/polyafilter_bench/runs/18_burden.mem.json -- .venv/bin/python scripts/18_burden.py`
Expected: templated share by region for each library, nuclei and cells, with gene bootstrap intervals; same-site and same-gene tables for 1417 vs 1419 and nuclei vs cells; PolyASite proximity per label (1417 during the build: within 10 nt for 48% of templated, 90% of non-templated and 69% of ambiguous molecules, against about 22% for sites moved 1 kb upstream).

- [ ] **Step 13: Commit**

```bash
git add tests/test_evaluate.py tests/test_polyasite.py tests/test_burden.py pafbench/evaluate.py pafbench/polyasite.py pafbench/burden.py scripts/17_evaluate.py scripts/18_burden.py
git -c user.name="Simone Picelli" -c user.email="simone.picelli@iob.ch" commit -m "Stages 17 and 18: metrics by level and stratum, the fixed selection rule with held-out reporting, bootstrap by gene, hexamer check, cut-off sweep, spike-in metrics; templated share by region in nuclei and cells"
```

---

### Task 7: ERCC counts against input amount (stage 15)

**Files:**
- Create: `tests/test_ercc.py`
- Create: `pafbench/ercc.py`
- Create: `scripts/15_ercc_counts.py`

**Interfaces:**
- Consumes: Task 5 `spikein/star_sorted.bam`; Task 6 `OUT/eval/selection.json`; `flags.flag_reference`, `transcripts.load_transcripts`, `grid.published`.
- Produces:
  - `pafbench.ercc.parse_ercc_conc(text)`
  - `pafbench.ercc.ercc_settings(selection)`
  - `pafbench.ercc.ercc_reads(bam_path, fasta_path, ref, transcripts, settings, cells)`
  - `pafbench.ercc.umi_counts(reads, removed=None)`
  - `pafbench.ercc.ercc_correlation(counts, conc)`
  - `pafbench.ercc.ercc_tables(reads, tags, conc)`
- Produces (data): `OUT/ref/ercc_conc_raw.txt`, `OUT/ref/ercc_conc.tsv` (`ercc_id, mix1_amol_per_ul`), `OUT/<lib>/spikein/ercc_counts.parquet` (`ercc_id, setting, umis`), `OUT/<lib>/spikein/ercc.json` (per setting: pearson on log10, spearman, n, UMIs kept; `provisional: true`).

The paper's count-accuracy check (spec 4.2): ERCC UMIs in called cells against input amount, before and after filtering. Lexogen's ERCC mix identity in SIRV Set 4 is not verified, so the correlations are provisional and the output says so. During the build on 1417: unfiltered pearson 0.955, 0.954 to 0.959 at the published settings, from 126,604 called-cell ERCC reads.

- [ ] **Step 1: Write the failing tests in `tests/test_ercc.py`**

```python
import gzip
import importlib.util
import json
import math
import sys

import numpy as np
import pandas as pd
import pysam
import pytest

from pafbench.config import REPO
from pafbench.ercc import ercc_correlation, ercc_reads, ercc_settings, ercc_tables, parse_ercc_conc, umi_counts
from pafbench.flags import Setting, flag_reference, setting_tag
from pafbench.grid import published
from pafbench.transcripts import load_transcripts

# header and three data lines of cms_095046.txt (ERCC_CONC_URL, fetched 2026-10-08), CRLF as served
EXCERPT = ("Re-sort ID\tERCC ID\tsubgroup\tconcentration in Mix 1 (attomoles/ul)\tconcentration in Mix 2 (attomoles/ul)\t"
           "expected fold-change ratio\tlog2(Mix 1/Mix 2)\r\n"
           "1\tERCC-00130\tA\t30000\t7500\t4\t2\r\n"
           "4\tERCC-00108\tA\t937.5\t234.375\t4\t2\r\n"
           "92\tERCC-00048\tD\t0.01430512\t0.02861023\t0.5\t-1\r\n")


def test_parse_ercc_conc():
    conc = parse_ercc_conc(EXCERPT)
    assert conc.columns.tolist() == ["ercc_id", "mix1_amol_per_ul"]
    assert conc.ercc_id.tolist() == ["ERCC-00130", "ERCC-00108", "ERCC-00048"]
    assert conc.mix1_amol_per_ul.tolist() == [30000.0, 937.5, 0.01430512]
    assert conc.mix1_amol_per_ul.dtype == np.float64


def test_parse_ercc_conc_finds_columns_by_header():
    rows = [line.split("\t") for line in EXCERPT.replace("\r\n", "\n").strip().split("\n")]
    order = [4, 3, 6, 1, 0, 5, 2]  # Mix 2 before Mix 1, ERCC ID after them
    text = "\n".join("\t".join(r[k] for k in order) for r in rows) + "\n"
    pd.testing.assert_frame_equal(parse_ercc_conc(text), parse_ercc_conc(EXCERPT))


def _reads():
    return pd.DataFrame({"ref": ["E1", "E1", "E1", "E1", "E2", "E3"],
                         "cb": ["c1", "c1", "c1", "c2", "c1", "c1"],
                         "ub": ["u1", "u1", "u2", "u1", "u1", "u9"]})


def test_umi_counts():
    reads = _reads()
    # E1: UMIs (c1,u1) with two reads, (c1,u2), (c2,u1); E2: one; E3: one
    assert umi_counts(reads).to_dict() == {"E1": 3, "E2": 1, "E3": 1}
    removed = np.array([True, False, True, True, True, False])
    # (c1,u1) survives through its second read; (c1,u2), (c2,u1) and E2's UMI lose their only read; E2 stays at 0
    c = umi_counts(reads, removed)
    assert c.to_dict() == {"E1": 1, "E2": 0, "E3": 1}
    assert c.index.name == "ercc_id" and c.dtype == np.int64
    assert umi_counts(reads, np.ones(6, bool)).to_dict() == {"E1": 0, "E2": 0, "E3": 0}


def test_ercc_correlation():
    conc = pd.DataFrame({"ercc_id": ["E1", "E2", "E3", "E4", "E7"], "mix1_amol_per_ul": [10.0, 100.0, 1000.0, 10000.0, 5.0]})
    # E6 has no concentration and E7 no count: both left out; log10(UMI + 1) = 1, 2, 3, 4 = log10(conc)
    exact = pd.Series({"E1": 9, "E2": 99, "E3": 999, "E4": 9999, "E6": 5})
    r = ercc_correlation(exact, conc)
    assert r["n"] == 4 and r["pearson_log10"] == pytest.approx(1.0) and r["spearman"] == pytest.approx(1.0)
    # x = 1, 2, 3, 4 and y = 1, 2, 3, 1: pearson 0.5 / sqrt(5 * 2.75) = 0.134840; spearman on ranks
    # (1, 2, 3, 4) and (1.5, 3, 4, 1.5): 0.5 / sqrt(5 * 4.5) = 0.105409
    r = ercc_correlation(pd.Series({"E1": 9, "E2": 99, "E3": 999, "E4": 9}), conc)
    assert r["n"] == 4
    assert r["pearson_log10"] == pytest.approx(0.5 / math.sqrt(13.75)) and r["spearman"] == pytest.approx(0.5 / math.sqrt(22.5))
    # a zero count enters as log10(1) = 0
    r = ercc_correlation(pd.Series({"E1": 0, "E2": 99, "E3": 999}), conc)
    assert r["pearson_log10"] == pytest.approx(np.corrcoef([0, 2, 3], [1, 2, 3])[0, 1])


def test_ercc_correlation_constant_counts_is_none():
    conc = pd.DataFrame({"ercc_id": ["E1", "E2", "E3"], "mix1_amol_per_ul": [1.0, 10.0, 100.0]})
    assert ercc_correlation(pd.Series({"E1": 0, "E2": 0, "E3": 0}), conc) == {"n": 3, "pearson_log10": None, "spearman": None}


def test_ercc_tables():
    reads = _reads().assign(tA=[True, False, True, True, True, False], tB=False)
    conc = pd.DataFrame({"ercc_id": ["E1", "E2", "E3"], "mix1_amol_per_ul": [100.0, 1.0, 10.0]})
    counts, per = ercc_tables(reads, ["tA", "tB"], conc)
    assert counts.columns.tolist() == ["ercc_id", "setting", "umis"]
    assert list(zip(counts.ercc_id, counts.setting, counts.umis)) == [
        ("E1", "unfiltered", 3), ("E2", "unfiltered", 1), ("E3", "unfiltered", 1),
        ("E1", "tA", 1), ("E2", "tA", 0), ("E3", "tA", 1),
        ("E1", "tB", 3), ("E2", "tB", 1), ("E3", "tB", 1)]
    assert per["unfiltered"]["umis"] == 5 and per["unfiltered"]["umis_kept_share"] == 1.0
    assert per["tA"]["umis"] == 2 and per["tA"]["umis_kept_share"] == 0.4
    assert per["tB"] == per["unfiltered"]
    # unfiltered: x = log10(conc) = 2, 0, 1 and y = log10(4), log10(2), log10(2)
    assert per["unfiltered"]["correlation"]["pearson_log10"] == pytest.approx(
        np.corrcoef([2, 0, 1], np.log10([4, 2, 2]))[0, 1])


@pytest.fixture(scope="module")
def tagged(synth, tmp_path_factory):
    out = tmp_path_factory.mktemp("ercc") / "tagged.bam"
    with pysam.AlignmentFile(str(synth.bam)) as fin, pysam.AlignmentFile(str(out), "wb", template=fin) as fout:
        for i, a in enumerate(fin.fetch(until_eof=True)):
            tags = []
            if i % 5 != 4:
                tags.append(("CB", f"c{i % 4}"))   # c3 is not a called cell
            if i % 6 != 5:
                tags.append(("UB", f"u{i % 9}"))
            a.set_tags(tags)
            if i % 11 == 10 and not a.is_secondary:
                a.flag |= 0x800   # supplementary: dropped even with a called CB and a UB
            fout.write(a)
    pysam.index(str(out))
    return out


def test_ercc_reads(synth, tagged):
    tx = load_transcripts(synth.gtf)
    settings = [Setting(300, 10, 1, True), Setting(150, None, 0, False)]
    tags = [setting_tag(s) for s in settings]
    cells = {"c0", "c1", "c2"}
    with pysam.AlignmentFile(str(tagged)) as bam:
        by_name = {a.query_name: a for a in bam.fetch(until_eof=True)}  # qnames are unique in the synthetic BAM
    for ref in ("chrA", "chrB"):
        got = ercc_reads(tagged, synth.fasta, ref, tx, settings, cells)
        fl = flag_reference(tagged, synth.fasta, ref, tx, settings)
        keep = [not (by_name[q].flag & 0x900) and by_name[q].has_tag("UB") and by_name[q].has_tag("CB")
                and by_name[q].get_tag("CB") in cells for q in fl.qname]
        exp = pd.DataFrame({"ref": ref, "cb": [by_name[q].get_tag("CB") for q in fl.qname[keep]],
                            "ub": [by_name[q].get_tag("UB") for q in fl.qname[keep]]})
        exp = pd.concat([exp, fl.loc[keep, tags].reset_index(drop=True)], axis=1)
        pd.testing.assert_frame_equal(got, exp, check_dtype=False)
        assert got.columns.tolist() == ["ref", "cb", "ub"] + tags
        assert 0 < len(got) < len(fl)
    # chrA: secondaries and supplementaries are dropped even with CB and UB; both settings remove some kept reads
    got = ercc_reads(tagged, synth.fasta, "chrA", tx, settings, cells)
    assert got[tags].any().all() and (~got[tags]).any().all()
    with pysam.AlignmentFile(str(tagged)) as bam:
        sup = [a for a in bam.fetch("chrA")
               if a.flag & 0x800 and a.has_tag("UB") and a.has_tag("CB") and a.get_tag("CB") in cells]
    assert len(sup) > 3


def test_ercc_settings(tmp_path):
    pub = [setting_tag(s) for s in published()]
    assert pub == ["c300_l10_m1_e", "c300_l10_m1_i", "c275_l9_m1_e", "c275_l9_m1_i",
                   "c180_l11_m2_e", "c180_l11_m2_i", "c375_l3_m2_e", "c375_l3_m2_i"]
    settings, chosen = ercc_settings(tmp_path / "absent.json")
    assert [setting_tag(s) for s in settings] == pub and chosen is None
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps({"modes": {"e": {"introns": False, "selection": {"tag": "c250_l9_m1_e"}},
                                         "i": {"introns": True, "selection": {"tag": None}}}}))
    settings, chosen = ercc_settings(sel)
    assert chosen == {"e": "c250_l9_m1_e", "i": None}
    assert [setting_tag(s) for s in settings] == pub + ["c250_l9_m1_e"]
    assert settings[-1] == Setting(250, 9, 1, False)
    # a chosen tag that is already a published setting is not added twice
    sel.write_text(json.dumps({"modes": {"e": {"selection": {"tag": "c275_l9_m1_e"}},
                                         "i": {"selection": {"tag": "c150_l20_m0_i"}}}}))
    settings, chosen = ercc_settings(sel)
    assert chosen == {"e": "c275_l9_m1_e", "i": "c150_l20_m0_i"}
    assert [setting_tag(s) for s in settings] == pub + ["c150_l20_m0_i"]


SPIKE_NAMES = {"chrA": "ERCC-00130", "chrB": "ERCC-00108"}


def _spike_set(synth, d):
    """The synthetic genome as spike-in contigs: chrA as ERCC-00130, chrB as ERCC-00108 and again as SIRV1 (not an ERCC,
    so stage 15 must leave it out). Records get CB c((i // 3) % 4) (c3 is not a called cell; none when i % 7 == 6) and
    UB u(i // 3) (none when i % 11 == 10), so most UMIs have several reads. Returns (fasta, gtf, sorted indexed bam)."""
    d.mkdir()
    with pysam.FastaFile(str(synth.fasta)) as fa:
        seqs = {c: fa.fetch(c) for c in SPIKE_NAMES}
    fasta, gtf, bam = d / "spikes.fa", d / "spikes.gtf", d / "star_sorted.bam"
    fasta.write_text("".join(f">{SPIKE_NAMES[c]}\n{s}\n" for c, s in seqs.items()) + f">SIRV1\n{seqs['chrB']}\n")
    pysam.faidx(str(fasta))
    gtf.write_text("".join(SPIKE_NAMES[line[:4]] + line[4:] for line in synth.gtf.read_text().splitlines(True)))
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": SPIKE_NAMES[c], "LN": len(s)} for c, s in seqs.items()]
                                           + [{"SN": "SIRV1", "LN": len(seqs["chrB"])}]})
    sirv = []
    with pysam.AlignmentFile(str(synth.bam)) as fin, pysam.AlignmentFile(str(bam), "wb", header=hdr) as fout:
        for i, a in enumerate(fin.fetch(until_eof=True)):
            tags = [] if i % 7 == 6 else [("CB", f"c{(i // 3) % 4}")]
            if i % 11 != 10:
                tags.append(("UB", f"u{i // 3}"))
            rec = a.to_dict() | {"ref_name": SPIKE_NAMES[a.reference_name]}
            b = pysam.AlignedSegment.from_dict(rec, hdr)
            b.set_tags(tags)
            fout.write(b)
            if a.reference_name == "chrB":
                b = pysam.AlignedSegment.from_dict(rec | {"name": f"{a.query_name}_s", "ref_name": "SIRV1"}, hdr)
                b.set_tags(tags)
                sirv.append(b)
        for b in sirv:
            fout.write(b)
    pysam.index(str(bam))
    return fasta, gtf, bam


def _brute_umis(bam, fasta, gtf, settings, cells):
    """{(ercc_id, setting): UMIs} counted directly: distinct (CB, UB) over primary records with a called-cell CB and a
    UB, all of them ("unfiltered") or those flags.flag_reference keeps; records joined to flags by qname. Also the
    number of such records."""
    tx = load_transcripts(gtf)
    tags = [setting_tag(s) for s in settings]
    umis, n = {}, 0
    with pysam.AlignmentFile(str(bam)) as fh:
        for ref in SPIKE_NAMES.values():
            by_name = {a.query_name: a for a in fh.fetch(ref)}  # qnames are unique in the synthetic BAM
            fl = flag_reference(bam, fasta, ref, tx, settings)
            sets = {t: set() for t in ["unfiltered"] + tags}
            for q, removed in zip(fl.qname, fl[tags].to_numpy()):
                a = by_name[q]
                cb = a.get_tag("CB") if a.has_tag("CB") else None
                if a.flag & 0x900 or cb not in cells or not a.has_tag("UB"):
                    continue
                n += 1
                key = (cb, a.get_tag("UB"))
                sets["unfiltered"].add(key)
                for t, r in zip(tags, removed):
                    if not r:
                        sets[t].add(key)
            umis |= {(ref, t): len(v) for t, v in sets.items()}
    return umis, n


def _stage15(monkeypatch, tmp_path, synth):
    spec = importlib.util.spec_from_file_location("stage15", REPO / "scripts" / "15_ercc_counts.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    fasta, gtf, bam = _spike_set(synth, tmp_path / "base")
    outs = tmp_path / "lib" / "outs"
    (outs / "filtered_feature_bc_matrix").mkdir(parents=True)
    with gzip.open(outs / "filtered_feature_bc_matrix" / "barcodes.tsv.gz", "wt") as fh:
        fh.write("c0\nc1\nc2\n")
    (tmp_path / "ref").mkdir()
    (tmp_path / "ref" / "ercc_conc_raw.txt").write_bytes(EXCERPT.encode())

    def no_network(*args, **kwargs):
        raise AssertionError("stage 15 downloaded the concentration table although it exists")

    monkeypatch.setattr(stage.urllib.request, "urlopen", no_network)
    monkeypatch.setattr(stage, "LIBRARIES", {"SYN": outs / "possorted_genome_bam.bam"})
    monkeypatch.setattr(stage, "SIRV_FASTA", fasta)
    monkeypatch.setattr(stage, "SIRV_GTF", gtf)
    return stage, fasta, gtf, bam


def test_stage15_on_synthetic_spikes(synth, tmp_path, monkeypatch, capsys):
    stage, fasta, gtf, bam = _stage15(monkeypatch, tmp_path, synth)
    out, ref = tmp_path / "out", tmp_path / "ref"
    argv = ["15", "SYN", "--base", str(bam.parent), "--out", str(out), "--ref", str(ref)]
    monkeypatch.setattr(sys, "argv", argv + ["--selection", str(tmp_path / "absent.json")])
    stage.main()
    printed = capsys.readouterr().out.strip().splitlines()
    assert printed[0] == f"{tmp_path / 'absent.json'} absent: published settings only"

    settings = list(published())
    tags = [setting_tag(s) for s in settings]
    umis, n_reads = _brute_umis(bam, fasta, gtf, settings, {"c0", "c1", "c2"})
    ids = sorted(SPIKE_NAMES.values())  # ERCC-00108 then ERCC-00130; SIRV1 is not an ERCC contig
    counts = pd.read_parquet(out / "ercc_counts.parquet")
    assert counts.columns.tolist() == ["ercc_id", "setting", "umis"]
    assert list(counts.itertuples(index=False, name=None)) == [
        (e, t, umis[(e, t)]) for t in ["unfiltered"] + tags for e in ids]
    # chrA holds records 0..144; chrB's 145 (c0, u48), 147 and 148 (c1, u49) are kept, 146 has no CB
    assert umis[("ERCC-00108", "unfiltered")] == 2
    assert umis[("ERCC-00130", "unfiltered")] > 2
    assert any(umis[(e, t)] < umis[(e, "unfiltered")] for e in ids for t in tags)
    pd.testing.assert_frame_equal(pd.read_csv(ref / "ercc_conc.tsv", sep="\t"), parse_ercc_conc(EXCERPT))

    js = json.loads((out / "ercc.json").read_text())
    assert list(js) == ["library", "provisional", "note", "concentration_source", "selection", "ercc_contigs", "reads",
                        "ercc_detected", "ercc_with_concentration", "settings"]
    assert (js["library"], js["provisional"], js["note"]) == ("SYN", True, stage.NOTE)
    assert js["concentration_source"] == stage.ERCC_CONC_URL
    assert js["selection"] is None
    assert (js["ercc_contigs"], js["reads"], js["ercc_detected"], js["ercc_with_concentration"]) == (2, n_reads, 2, 2)
    assert list(js["settings"]) == ["unfiltered"] + tags
    base = umis[("ERCC-00108", "unfiltered")] + umis[("ERCC-00130", "unfiltered")]
    for t in ["unfiltered"] + tags:
        lo, hi = umis[("ERCC-00108", t)], umis[("ERCC-00130", t)]
        # two ERCCs: log10(UMI + 1) against log10(937.5) and log10(30000) correlates +1 or -1, None when the counts tie
        r = None if lo == hi else pytest.approx(1.0 if hi > lo else -1.0)
        assert js["settings"][t]["correlation"] == {"n": 2, "pearson_log10": r, "spearman": r}
        assert js["settings"][t]["umis"] == lo + hi
        assert js["settings"][t]["umis_kept_share"] == pytest.approx((lo + hi) / base)
    assert js["settings"]["unfiltered"]["correlation"]["pearson_log10"] == pytest.approx(1.0)

    # with stage 17's selection: the chosen tag per mode (or None); the chosen tag that is not published is counted too
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps({"modes": {"e": {"introns": False, "selection": {"tag": "c250_l9_m1_e"}},
                                         "i": {"introns": True, "selection": {"tag": None}}}}))
    monkeypatch.setattr(sys, "argv", argv + ["--selection", str(sel)])
    stage.main()
    assert "absent" not in capsys.readouterr().out
    js = json.loads((out / "ercc.json").read_text())
    assert js["selection"] == {"e": "c250_l9_m1_e", "i": None}
    assert list(js["settings"]) == ["unfiltered"] + tags + ["c250_l9_m1_e"]
    extra, _ = _brute_umis(bam, fasta, gtf, [Setting(250, 9, 1, False)], {"c0", "c1", "c2"})
    counts = pd.read_parquet(out / "ercc_counts.parquet")
    assert list(counts[counts.setting == "c250_l9_m1_e"].itertuples(index=False, name=None)) == [
        (e, "c250_l9_m1_e", extra[(e, "c250_l9_m1_e")]) for e in ids]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_ercc.py`
Expected: FAIL; the first error is `ModuleNotFoundError: No module named 'pafbench.ercc'`.

- [ ] **Step 3: Create `pafbench/ercc.py`**

```python
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from scipy import stats

from .evaluate import tag_setting
from .flags import flag_reference, setting_tag
from .grid import published

ID_HEADER = "ERCC ID"
MIX1_HEADER = "concentration in Mix 1 (attomoles/ul)"


def parse_ercc_conc(text):
    """(ercc_id, mix1_amol_per_ul) from Thermo Fisher's ERCC table (ERCC_CONC_URL, tab-separated); columns found by header."""
    t = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    t.columns = [c.strip() for c in t.columns]
    return pd.DataFrame({"ercc_id": t[ID_HEADER].str.strip().to_numpy(),
                         "mix1_amol_per_ul": t[MIX1_HEADER].astype(np.float64).to_numpy()})


def ercc_settings(selection):
    """The 8 published settings, then the tags chosen per mode in stage 17's selection.json that are not among them; also
    the chosen tag per mode ({mode: tag or None}; None when the file is absent)."""
    settings = list(published())
    if not Path(selection).exists():
        return settings, None
    modes = json.loads(Path(selection).read_text())["modes"]
    chosen = {mode: m["selection"]["tag"] for mode, m in modes.items()}
    known = {setting_tag(s) for s in settings}
    return settings + [tag_setting(t) for t in dict.fromkeys(chosen.values()) if t and t not in known], chosen


def ercc_reads(bam_path, fasta_path, ref, transcripts, settings, cells):
    """flags.flag_reference on one contig joined by fetch order to (cb, ub) read in a second pass, keeping primary records
    with a called-cell CB and a UB: ref, cb, ub and one bool per setting tag (True = removed)."""
    fl = flag_reference(bam_path, fasta_path, ref, transcripts, settings)
    qname, cb, ub = [], [], []
    with pysam.AlignmentFile(str(bam_path)) as bam:
        for a in bam.fetch(ref):
            qname.append(a.query_name)
            cb.append(a.get_tag("CB") if a.has_tag("CB") else None)
            ub.append(a.get_tag("UB") if a.has_tag("UB") else None)
    assert qname == fl.qname.tolist(), f"{ref}: second pass is not in flag_reference order"
    keep = (((fl.flag.to_numpy() & 0x900) == 0) & np.array([c in cells for c in cb], bool)
            & np.array([u is not None for u in ub], bool))
    tags = [setting_tag(s) for s in settings]
    out = pd.DataFrame({"ref": ref, "cb": np.asarray(cb, object)[keep], "ub": np.asarray(ub, object)[keep]})
    return pd.concat([out, fl.loc[keep, tags].reset_index(drop=True)], axis=1)


def umi_counts(reads, removed=None):
    """UMIs (ref, cb, ub) per ERCC with at least one read not removed (removed None: unfiltered). Indexed by ercc_id
    over every ref in reads, zeros kept, so a filtered count covers the ERCCs detected unfiltered."""
    kept = reads if removed is None else reads[~np.asarray(removed, bool)]
    n = kept.drop_duplicates(["ref", "cb", "ub"]).groupby("ref").size()
    return n.reindex(np.unique(reads.ref.to_numpy()), fill_value=0).astype(np.int64).rename("umis").rename_axis("ercc_id")


def ercc_correlation(counts, conc):
    """Over the ERCCs of counts (UMIs indexed by ercc_id; from umi_counts, so those with a UMI unfiltered) that have a Mix 1
    concentration: pearson of log10(UMI + 1) against log10(concentration), spearman on raw values (both None when either
    side is constant), n."""
    c = conc.set_index("ercc_id").mix1_amol_per_ul
    ids = [i for i in counts.index if i in c.index]
    x, y = c.loc[ids].to_numpy(np.float64), counts.loc[ids].to_numpy(np.float64)
    const = np.ptp(y) == 0 or np.ptp(x) == 0
    return {"n": len(ids),
            "pearson_log10": None if const else float(stats.pearsonr(np.log10(y + 1), np.log10(x))[0]),
            "spearman": None if const else float(stats.spearmanr(y, x)[0])}


def ercc_tables(reads, tags, conc):
    """ercc_counts rows (ercc_id, setting, umis) for 'unfiltered' then each tag, and per setting the correlation, the UMIs
    and the share of unfiltered UMIs kept."""
    base = umi_counts(reads)
    rows, per = [], {}
    for name in ["unfiltered"] + list(tags):
        c = base if name == "unfiltered" else umi_counts(reads, reads[name].to_numpy())
        rows.append(pd.DataFrame({"ercc_id": c.index.to_numpy(), "setting": name, "umis": c.to_numpy()}))
        per[name] = {"correlation": ercc_correlation(c, conc), "umis": int(c.sum()), "umis_kept_share": float(c.sum() / base.sum())}
    return pd.concat(rows, ignore_index=True), per
```

- [ ] **Step 4: Create `scripts/15_ercc_counts.py`**

```python
"""Stage 15: ERCC UMI counts against ERCC Mix 1 input, unfiltered and after polyAfilter (spec 4.2), provisional.

Per ERCC contig of <base>/star_sorted.bam (stage 14), flags.flag_reference at the 8 published settings plus the tags chosen
in OUT/eval/selection.json (skipped with a note when the file is absent), joined by fetch order to CB and UB. UMIs count
(contig, CB, UB) over primary records with a called-cell CB and a UB. The concentration table is downloaded once to
<ref>/ercc_conc_raw.txt. Writes <out>/ercc_counts.parquet (ercc_id, setting, umis) and <out>/ercc.json.
"""
import argparse
import gzip
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd
import pysam

from pafbench.config import ERCC_CONC_URL, LIBRARIES, OUT, SIRV_FASTA, SIRV_GTF
from pafbench.ercc import ercc_reads, ercc_settings, ercc_tables, parse_ercc_conc
from pafbench.flags import setting_tag
from pafbench.transcripts import load_transcripts

NOTE = ("Lexogen's ERCC mix identity in SIRV Set 4 is not verified; concentrations are ERCC Mix 1 from Thermo Fisher")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--base", type=Path, help="dir with star_sorted.bam (default OUT/LIB/spikein)")
    ap.add_argument("--out", type=Path, help="output dir (default: base)")
    ap.add_argument("--ref", type=Path, default=OUT / "ref", help="dir for the concentration table (default OUT/ref)")
    ap.add_argument("--selection", type=Path, default=OUT / "eval" / "selection.json")
    args = ap.parse_args()
    t0 = time.monotonic()
    base = args.base or OUT / args.library / "spikein"
    out = args.out or base
    raw = args.ref / "ercc_conc_raw.txt"
    if not raw.exists():
        args.ref.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(ERCC_CONC_URL) as r:
            raw.write_bytes(r.read())
    conc = parse_ercc_conc(raw.read_text())
    conc.to_csv(args.ref / "ercc_conc.tsv", sep="\t", index=False)

    settings, chosen = ercc_settings(args.selection)
    if chosen is None:
        print(f"{args.selection} absent: published settings only")
    with gzip.open(LIBRARIES[args.library].parent / "filtered_feature_bc_matrix" / "barcodes.tsv.gz", "rt") as fh:
        cells = {line.strip() for line in fh}
    bam = base / "star_sorted.bam"
    with pysam.AlignmentFile(str(bam)) as fh:
        contigs = [s.contig for s in fh.get_index_statistics() if s.mapped > 0 and s.contig.startswith("ERCC")]
    tx = load_transcripts(SIRV_GTF)
    reads = pd.concat([ercc_reads(bam, SIRV_FASTA, c, tx, settings, cells) for c in contigs], ignore_index=True)
    tags = [setting_tag(s) for s in settings]
    counts, per = ercc_tables(reads, tags, conc)

    out.mkdir(parents=True, exist_ok=True)
    counts.to_parquet(out / "ercc_counts.parquet", index=False)
    summary = {"library": args.library, "provisional": True, "note": NOTE, "concentration_source": ERCC_CONC_URL,
               "selection": chosen, "ercc_contigs": len(contigs),
               "reads": len(reads), "ercc_detected": int((counts.setting == "unfiltered").sum()),
               "ercc_with_concentration": per["unfiltered"]["correlation"]["n"], "settings": per}
    (out / "ercc.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("library", "ercc_contigs", "reads", "ercc_detected", "ercc_with_concentration")}
                     | {t: [per[t]["correlation"]["pearson_log10"], per[t]["umis_kept_share"]] for t in ["unfiltered"] + tags}
                     | {"seconds": round(time.monotonic() - t0, 1)}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `216 passed, 2 warnings`.

- [ ] **Step 6: Stage 15 on 1417 and 1419**

Run: `for L in 1417 1419; do .venv/bin/python -m pafbench.memwatch /data/element_runs/polyafilter_bench/$L/runs/15_ercc_counts.mem.json -- .venv/bin/python scripts/15_ercc_counts.py $L; done`
Expected: the concentration table is downloaded once; for 1417 unfiltered pearson about 0.955; settings = the 8 published ones plus the chosen tags from `selection.json` (none added for a mode whose selection is null).

- [ ] **Step 7: Commit**

```bash
git add tests/test_ercc.py pafbench/ercc.py scripts/15_ercc_counts.py
git -c user.name="Simone Picelli" -c user.email="simone.picelli@iob.ch" commit -m "Stage 15: ERCC UMIs in called cells against ERCC Mix 1 input, unfiltered and at the published and chosen settings (provisional)"
```

---

### Task 8: Exact-match validation (run stage 20 after the background runs)

**Interfaces:**
- Consumes: Task 1 stages 19 and 20.

- [ ] **Step 1: Wait for both chains and the 1179 block**

Run: `tail -2 /data/element_runs/polyafilter_bench/validation/runs_1417.log /data/element_runs/polyafilter_bench/validation/runs_1419.log; cat /data/element_runs/polyafilter_bench/validation/runs.out; tail -3 /data/element_runs/polyafilter_bench/validation/subset_1179.out`
Expected: both logs end with the last step's `end` line, `runs.out` reports no failure, and `subset_1179.out` ends with stage 20's JSON line. A step that failed or was killed is rerun by launching `scripts/19_validation_runs.sh` again (finished steps are skipped).

- [ ] **Step 2: Stage 20**

Run: `.venv/bin/python -m pafbench.memwatch /data/element_runs/polyafilter_bench/runs/20_validation_compare.mem.json -- .venv/bin/python scripts/20_validation_compare.py`
Run it once without skip flags: the file stage 21 reads must cover everything, and the 1179 block started in Task 1 is only read back (its finished settings are not rerun).
Expected: `/data/element_runs/polyafilter_bench/validation/exact_match_m2.json` with `all_identical: true`: every original run on 1417 (10 settings with milestone 1's three) and 1419 (8) identical to the reimplementation, the remove-all control on 1417 chr21 + chr22 identical (chr22 alone during the build: 534,842 of 944,005 records removed on both sides), and the 8 published settings on 1179 chr21 + chr22 identical. Any difference stops the milestone: by spec 5.2 it is resolved before results are used.

---

### Task 9: Milestone 2 results page (stage 21)

**Files:**
- Create: `tests/test_m2report.py`
- Create: `pafbench/m2report.py`
- Create: `scripts/21_m2_summary.py`

**Interfaces:**
- Consumes: every output above; `report.parse_gnu_time`; `validation` helpers for the run table.
- Produces:
  - `pafbench.m2report.runs_section(base)`
  - `pafbench.m2report.truth_section(base)`
  - `pafbench.m2report.grid_section(base)`
  - `pafbench.m2report.selection_section(base)`
  - `pafbench.m2report.strata_section(base)`
  - `pafbench.m2report.hexamer_section(base)`
  - `pafbench.m2report.spikein_section(base)`
  - `pafbench.m2report.ercc_section(base)`
  - `pafbench.m2report.long_read_section(base)`
  - `pafbench.m2report.burden_section(base)`
  - `pafbench.m2report.validation_section(base)`
  - `pafbench.m2report.render(base)`
- Produces (data): `docs/m2_results.md`, eleven sections (runs; truth sets; grid; selection; metrics by A band and region; label-free check; spike-ins; ERCC; long reads; burden; validation); a section whose inputs are missing reads "not run" with the file name.

- [ ] **Step 1: Write the failing tests in `tests/test_m2report.py`**

```python
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pafbench import m2report
from pafbench.evaluate import METRIC_COLUMNS
from pafbench.labels import a_band
from pafbench.molecules import MOL_COLUMNS
from pafbench.validation import parse_tag

REPO = Path(__file__).resolve().parents[1]
I1, I3, E1 = "c300_l10_m1_i", "c375_l3_m2_i", "c300_l10_m1_e"
NOTE = "Lexogen's ERCC mix identity in SIRV Set 4 is not verified; concentrations are ERCC Mix 1 from Thermo Fisher"


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(path, index=False)
    elif isinstance(obj, str):
        path.write_text(obj)
    else:
        path.write_text(json.dumps(obj, indent=1))


def _mol(rows):
    """Labeled molecules with the real columns; rows = (a20, label, region, genes, has_ts, hexamer, n_members)."""
    n = len(rows)
    a20, label, region, genes, has_ts, hexamer, n_members = (list(c) for c in zip(*rows))
    df = pd.DataFrame({"mol": np.arange(n), "cb": "AAAC-1", "ub": [f"U{i}" for i in range(n)], "strand": "+", "ref": "chr1",
                       "site": 1000 + 100 * np.arange(n), "n_pa": 1, "spread": 0, "a20": a20, "down20": "A" * 20,
                       "up40": "C" * 40, "hexamer": hexamer, "genes": genes, "re": "E", "mapq_min": 255})[MOL_COLUMNS]
    return df.assign(label=label, a_band=[a_band(x) for x in a20], region=region, gene=[g.split(";")[0] for g in genes],
                     has_ts=has_ts, n_members=n_members)


def _rates(tp, fn, tn, fp):
    div = lambda a, b: a / b if b else np.nan  # noqa: E731
    return div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)


def _metric(lib, tag, kind, stratum, tp, fn, tn, fp, cap=1000, cutoffs="8/4", level="molecule"):
    s = parse_tag(tag)
    sens, spec, prec = _rates(tp, fn, tn, fp)
    return dict(lib=lib, cap=cap, cutoffs=cutoffs, tag=tag, covlen=s.covlen, min_len=s.min_len, mism=s.mism,
                introns=s.introns, level=level, stratum_kind=kind, stratum=stratum, tp=float(tp), fn=float(fn), tn=float(tn),
                fp=float(fp), n_amb=0.0, amb_removed=0.0, sens=sens, spec=spec, prec=prec)


def _lvl(tp, fn, tn, fp):
    sens, spec, prec = _rates(tp, fn, tn, fp)
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "n_amb": 10, "amb_removed": 1, "sens": sens, "spec": spec, "prec": prec}


def _sel(tag, sens, spec, prec, n_q):
    return {"tag": tag, "sens": sens, "spec": spec, "prec": prec, "n_candidates": 189, "n_qualifying": n_q, "frontier": []}


NONE_E = {"tag": None, "sens": None, "spec": None, "prec": None, "n_candidates": 189, "n_qualifying": 0,
          "frontier": [["c150_l3_m2_e", 0.41, 0.9], [E1, 0.0356, 0.9943]]}
# chosen in -i on 1417: tp 9,795 fn 31,574 tn 2,757 fp 29 -> sens 0.2368, spec 0.9896, prec 0.9970
I1_1417 = {"molecule": _lvl(9795, 31574, 2757, 29), "molecule_majority": {"sens": 0.2587, "spec": 0.9856, "prec": 0.997},
           "read": {"sens": 0.2780, "spec": 0.9752, "prec": 0.9937}}
I1_1179 = {"molecule": _lvl(300, 700, 95, 5), "molecule_majority": {"sens": 0.35, "spec": 0.9, "prec": 0.98},
           "read": {"sens": 0.4, "spec": 0.8, "prec": 0.97}}


def _full_base(tmp_path):
    b = tmp_path / "base"
    # ---- stage 12, 1417 only; 1419 and 1179 absent
    mol = _mol([(15, "templated", "3UTR", "gA", False, False, 2),
                (9, "templated", "intron", "gA;gB", True, False, 1),
                (12, "templated", "intergenic", "", False, True, 0),
                (4, "nontemplated", "3UTR", "gC", False, True, 1),
                (6, "ambiguous", "3UTR", "gC", False, True, 0),
                (18, "templated", "antisense", "", False, False, 0)])
    _write(b / "1417/truth/molecules_labeled.parquet", mol)
    pa = pd.DataFrame({"qname": [f"p{i}" for i in range(6)], "mol": np.arange(6), "role": "pa", "dist5_exon": np.nan,
                       "dist_mode": "", "spliced": False})
    mem = pd.DataFrame({"qname": [f"m{i}" for i in range(6)], "mol": [0, 0, 1, 1, 3, 4], "role": "member",
                        "dist5_exon": [100.0, 700.0, 1000.0, 1500.0, 400.0, 2000.0], "dist_mode": "exon", "spliced": False})
    _write(b / "1417/truth/truth_reads.parquet", pd.concat([pa, mem], ignore_index=True))
    # ---- stage 13
    _write(b / "1417/grid/summary.json", {
        "library": "1417", "settings": 396, "workers": 8, "scanned": 116, "seconds": 1234.4,
        "per_ref": {"chr1": {"records": 6962527, "primary": 6962527, "seconds": 186.0},
                    "chr21": {"records": 1837866, "primary": 1837866, "seconds": 28.6}},
        "records": 85000000, "primary": 83140741,
        "removed": {I1: 17068514, I3: 46808565, E1: 3729928, "c300_lnone_m0_i": 50000000}})
    _write(b / "1417/grid/stage13.mem.json", {
        "cmd": ["/x/.venv/bin/python", "/x/polyafilter_bench/scripts/13_grid_flags.py", "1417", "--workers", "8"],
        "exit_status": 0, "elapsed_s": 1240.2, "peak_tree_rss_gib": 19.123, "interval_s": 2.0})
    # ---- stage 19 original runs: one finished (with memwatch), one killed (GNU time writes Exit status 0)
    _write(b / "1417/original/c275_l9_m1_i.time.txt",
           "\tCommand being timed: \"x\"\n\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:57:00\n"
           "\tMaximum resident set size (kbytes): 16238356\n\tExit status: 0\n")
    _write(b / "1417/original/c275_l9_m1_i.mem.json", {
        "cmd": ["/x/scripts/02_run_original.sh", "1417", "275", "9", "1", "i"], "exit_status": 0, "elapsed_s": 7021.0,
        "peak_tree_rss_gib": 60.5, "interval_s": 2.0})
    _write(b / "1417/original/c180_l11_m2_i.time.txt",
           "Command terminated by signal 9\n\tElapsed (wall clock) time (h:mm:ss or m:ss): 10:00\n"
           "\tMaximum resident set size (kbytes): 1048576\n\tExit status: 0\n")
    _write(b / "1417/original/c300_l10_m0_i.time.txt", "")
    # ---- stage 17
    sel = {
        "rule": "highest molecule-level sensitivity with specificity >= 0.95 on 1417", "min_spec": 0.95,
        "selection_library": "1417", "cap": 1000, "cutoffs": "8/4", "libraries": ["1417", "1179"], "missing": {},
        "units": {"1417": {"500": {"read_units": 73200, "molecule_units": 47402, "missing_reads": 0, "dropped_molecules": 0},
                           "1000": {"read_units": 76245, "molecule_units": 48936, "missing_reads": 0, "dropped_molecules": 0},
                           "2000": {"read_units": 77966, "molecule_units": 49827, "missing_reads": 3, "dropped_molecules": 2}}},
        "modes": {"e": {"introns": False, "selection": NONE_E, "metrics": {}},
                  "i": {"introns": True, "selection": _sel(I1, 0.2368, 0.9896, 0.9970, 120),
                        "metrics": {"1417": I1_1417, "1179": I1_1179}}},
        "published": {I1: {"1417": I1_1417, "1179": I1_1179},
                      I3: {"1417": {"molecule": _lvl(26356, 15013, 2399, 387), "molecule_majority": {"sens": 0.6559, "spec": 0.8482, "prec": 0.98},
                                    "read": {"sens": 0.6771, "spec": 0.828, "prec": 0.9823}},
                           "1179": {}}},
        "cap_sensitivity": {c: {"e": NONE_E, "i": _sel(I1, s, 0.991, 0.99, q)}
                            for c, s, q in (("500", 0.25, 118), ("1000", 0.2368, 120), ("2000", 0.22, 121))},
        "cutoff_sweep": [{"templated_min": 7, "nontemplated_max": 2, "mode": "i", "tag": I1, "n_qualifying": 100,
                          "by_library": {"1417": {"sens": 0.2, "spec": 0.99, "prec": 0.97}, "1179": None}},
                         {"templated_min": 10, "nontemplated_max": 6, "mode": "e", "tag": None, "n_qualifying": 0,
                          "by_library": {}}],
        "bootstrapped_tags": [I1, I3],
        "ambiguous_tail_share": {"1417": {"estimate": 0.458, "rate_templated": 0.069, "rate_nontemplated": 0.772,
                                          "rate_ambiguous": 0.391, "n_templated": 4, "n_nontemplated": 1, "n_ambiguous": 1}},
    }
    _write(b / "eval/selection.json", sel)
    # the read-level row comes first and has its own cluster count (clusters without a unit at a level are left out)
    boot = [("1417", 1000, "read", I1, "sens", 0.278, 0.1, 0.9, 13500),
            ("1417", 1000, "molecule", I1, "sens", 0.2368, 0.2074, 0.2588, 13271),
            ("1417", 1000, "molecule", I1, "spec", 0.9896, 0.9854, 0.9932, 13271),
            ("1417", 1000, "molecule", I1, "prec", 0.9970, 0.9959, 0.9981, 13271),
            ("1417", 1000, "molecule", I3, "sens", 0.6371, 0.5623, 0.6904, 13271)]
    _write(b / "eval/bootstrap.parquet", pd.DataFrame(
        [r[:8] + (1000, r[8]) for r in boot], columns=["lib", "cap", "level", "tag", "metric", "estimate", "lo", "hi", "n_boot", "n_clusters"]))
    rows = []
    for tag in (I1, I3):
        rows += [_metric("1417", tag, "all", "all", 9795, 31574, 2757, 29),
                 _metric("1417", tag, "a_band", "0-4", 0, 0, 2757, 29), _metric("1417", tag, "a_band", "5-7", 0, 0, 0, 0),
                 _metric("1417", tag, "a_band", "8-9", 100, 1900, 0, 0), _metric("1417", tag, "a_band", "10-13", 1000, 5400, 0, 0),
                 _metric("1417", tag, "a_band", "14-20", 8695, 24274, 0, 0),
                 _metric("1417", tag, "region", "intergenic", 0 if tag == I1 else 500, 5182 if tag == I1 else 4682,
                         125 if tag == I1 else 120, 3 if tag == I1 else 8),
                 _metric("1417", tag, "region", "3UTR", 5000, 10000, 2000, 20)]
    # rows the sections must ignore: another cap, other cut-offs, the read level
    rows += [_metric("1417", I1, "region", "intergenic", 99, 5083, 100, 28, cap=500),
             _metric("1417", I1, "region", "intergenic", 77, 5105, 100, 28, cutoffs="7/2"),
             _metric("1417", I1, "a_band", "8-9", 2000, 0, 0, 0, level="read")]
    metrics = pd.DataFrame(rows)[METRIC_COLUMNS]
    metrics["min_len"] = metrics.min_len.astype("Int64")
    _write(b / "eval/metrics.parquet", metrics)
    _write(b / "eval/hexamer_check.parquet", pd.DataFrame(
        [("1417", 1000, I3, "all", 28762, 0.061, 20174, 0.229), ("1417", 1000, I3, "nontemplated", 387, 0.305, 2399, 0.836),
         ("1417", 1000, I3, "ambiguous", 0, np.nan, 10, 0.4), ("1417", 500, I3, "all", 1, 0.9, 1, 0.9)],
        columns=["lib", "cap", "tag", "label", "removed_n", "removed_hex_rate", "kept_n", "kept_hex_rate"]))
    spike = [dict(_metric("1417", I1, "all", "all", 10, 30, 50, 2), level="molecule"),
             dict(_metric("1417", I1, "module", "ERCC", 5, 5, 20, 0)),
             dict(_metric("1417", I1, "all", "all", 9, 1, 0, 0, level="read"))]
    spike = pd.DataFrame(spike).drop(columns=["cutoffs"])
    spike["min_len"] = spike.min_len.astype("Int64")
    _write(b / "eval/spikein_metrics.parquet", spike)
    # ---- stage 14, 1417 only
    sm = _mol([(12, "templated", "ncexon", "ERCC-00002", False, False, 1), (15, "templated", "ncexon", "SIRV1", False, False, 1),
               (1, "nontemplated", "ncexon", "SIRV1", False, True, 1), (0, "nontemplated", "ncexon", "ERCC-00002", False, True, 1),
               (3, "nontemplated", "ncexon", "SIRV4001", False, True, 1), (10, "templated", "ncexon", "SIRV4001", False, False, 1),
               (6, "ambiguous", "ncexon", "ERCC-00002", False, False, 1), (5, "ambiguous", "ncexon", "SIRV1", False, False, 1),
               (7, "ambiguous", "ncexon", "SIRV1", False, False, 1), (9, "templated", "ncexon", "SIRV1", False, False, 1)])
    sm = sm.assign(spike_label=["internal", "internal", "internal", "tail", "tail", "tail", "tail", "other", "internal", "tail"],
                   module=["ERCC", "SIRV_E0", "SIRV_E0", "ERCC", "long_SIRV", "long_SIRV", "ERCC", "SIRV_E0", "SIRV_E0", "SIRV_E0"],
                   dist_to_end=[300, 400, 500, 0, 2, -3, 5, 20, 600, 1])
    _write(b / "1417/spikein/truth/molecules_labeled.parquet", sm)
    # ---- stage 15: 1417 in stage 15's layout (settings -> correlation dict, UMIs, kept share; null correlations for a
    # constant input), 1419 with the settings at the top level
    _write(b / "1417/spikein/ercc.json", {
        "library": "1417", "provisional": True, "note": NOTE, "concentration_source": "https://x",
        "selection": {"e": None, "i": I1}, "ercc_contigs": 92, "reads": 120000, "ercc_detected": 61,
        "ercc_with_concentration": 61, "settings": {
            "unfiltered": {"correlation": {"n": 61, "pearson_log10": 0.9123, "spearman": 0.8876}, "umis": 52000,
                           "umis_kept_share": 1.0},
            I1: {"correlation": {"n": 61, "pearson_log10": 0.9201, "spearman": 0.8899}, "umis": 50502, "umis_kept_share": 0.9712},
            I3: {"correlation": {"n": 3, "pearson_log10": None, "spearman": None}, "umis": 3, "umis_kept_share": 0.0001}}})
    _write(b / "1419/spikein/ercc.json", {"provisional": True, "note": NOTE, "unfiltered": {
        "correlation": {"n": 60, "pearson_log10": 0.9, "spearman": 0.8}, "umis": 4000, "umis_kept_share": 1.0}})
    # ---- stage 16, BD67 only
    used, acc = 1610467, 1432000
    _write(b / "BD67/long_read_acceptance.json", {
        "library": "BD67", "bam": "/x/BD67.bam", "refs": ["chr21", "chr22"],
        "records": {"records": 1735308, "secondary_supplementary": 14097, "low_mapq": 110744, "used": used},
        "duplicate_qnames": 0, "outcomes": {"accepted": acc, "short_run": 77500, "short_after": 113, "both": 100854},
        "outcome_fractions": {}, "dropped_fraction": 1 - acc / used, "unique_sites": 70136, "sites_with_accepted_read": 42170,
        "cutoffs_8_4": {"cutoffs": {"templated_min": 8, "nontemplated_max": 4},
                        "n": {"templated": 28473, "nontemplated": 6469, "ambiguous": 7228},
                        "hexamer_rate": {"templated": 0.106, "nontemplated": 0.541, "ambiguous": 0.443}, "ratio": 5.12, "passes": True},
        "cutoffs_14_8": {"cutoffs": {"templated_min": 14, "nontemplated_max": 8},
                         "n": {"templated": 11551, "nontemplated": 16279, "ambiguous": 14340},
                         "hexamer_rate": {"templated": 0.046, "nontemplated": 0.456, "ambiguous": 0.2}, "ratio": 9.95, "passes": True},
        "per_ref": {},
        "vs_1179": {"path": "/x/1179.parquet", "molecules": 16408, "matched": 10091, "share_matched": 0.615,
                    "share_matched_by_label": {"templated": 0.491, "nontemplated": 0.907, "ambiguous": 0.805},
                    "agreement": {"templated": 0.993, "nontemplated": 0.996, "pooled": 0.994}}})
    sites = pd.DataFrame({"ref": "chr21", "strand": "+", "site": [10, 20, 30, 40, 50, 60, 70, 80, 90],
                          "n_reads": [3, 1, 2, 2, 1, 1, 2, 1, 1], "n_accepted": [3, 1, 2, 2, 0, 0, 2, 1, 1],
                          "a20": [15, 9, 2, 1, 3, 12, 4, 8, 14],
                          "label": ["templated", "templated", "nontemplated", "nontemplated", "nontemplated", "templated",
                                    "nontemplated", "templated", "templated"],
                          "a_band": "", "hexamer": [False, True, True, False, True, True, True, False, True], "region": "3UTR",
                          "gene": "gA"})
    _write(b / "BD67/long_read_sites.parquet", sites)
    # ---- stage 18
    share = pd.DataFrame(
        [("1417", "all", "all", 118278, 10822, 15374, 144474, 1.0, 0.81868, 0.91617, 0.8038, 0.8346),
         ("1417", "region", "3UTR", 15773, 8607, 6749, 31129, 0.2155, 0.5067, 0.6470, 0.479, 0.532),
         ("1417", "a_band", "14-20", 40103, 0, 0, 40103, 0.2776, 1.0, 1.0, 0.2705, 0.2851),
         ("1417", "has_ts", "True", 524, 50, 150, 724, 0.005, 0.7238, 0.9129, 0.638, 0.794),
         ("cells", "all", "all", 900, 50, 50, 1000, 1.0, 0.9, 0.9474, 0.88, 0.92)],
        columns=["group", "stratum_kind", "stratum", "templated", "nontemplated", "ambiguous", "total", "frac", "share_all",
                 "share_resolved", "lo", "hi"])
    _write(b / "burden/templated_share.parquet", share)
    _write(b / "burden/same_site.parquet", pd.DataFrame(
        [("1417_vs_1419", "templated", "templated", 95920), ("1417_vs_1419", "templated", "none", 500),
         ("1417_vs_1419", "nontemplated", "nontemplated", 6261), ("1417_vs_1419", "ambiguous", "ambiguous", 11219),
         ("1417_vs_1419", "ambiguous", "templated", 100)],
        columns=["pair", "label_a", "label_b", "n"]))
    _write(b / "burden/same_gene.parquet", pd.DataFrame([("1417_vs_1419", 1212, 1.0, 0.0)],
                                                        columns=["pair", "n_genes", "spearman_rho", "p_value"]))
    _write(b / "burden/selection_bias.parquet", pd.DataFrame(
        [("1417", "E", 0.421, 0.407), ("1417", "N", 0.48, 0.5), ("1417", "I", 0.099, 0.093)],
        columns=["lib", "re", "labeled_share", "records_share"]))
    prox = {"1417": {("templated", "all"): [9000, 20000, 40000, 44000, 5278],
                     ("nontemplated", "all"): [4000, 2000, 2000, 2681, 141],
                     ("ambiguous", "all"): [3000, 3000, 4000, 3674, 1700],
                     ("null_shift1kb", "all"): [31500, 40000, 50000, 15000, 7974],
                     ("templated", "8-9"): [1000, 3000, 6000, 7577, 1000],
                     ("templated", "10-13"): [5000, 10000, 20000, 22598, 2000],
                     ("templated", "14-20"): [3000, 7000, 14000, 14103, 2000]},
            "cells": {("templated", "all"): [100, 200, 300, 300, 100], ("nontemplated", "all"): [0, 0, 0, 0, 0],
                      ("ambiguous", "all"): [5, 5, 0, 6, 4], ("null_shift1kb", "all"): [255, 255, 255, 153, 102],
                      ("templated", "8-9"): [0, 0, 0, 0, 0], ("templated", "10-13"): [50, 100, 150, 150, 50],
                      ("templated", "14-20"): [50, 100, 150, 150, 50]}}
    _write(b / "burden/polyasite.parquet", pd.DataFrame(
        [(g, lab, band, bin_, n) for g, d in prox.items() for (lab, band), ns in d.items()
         for bin_, n in zip(("0-10", "11-50", "51-200", ">200", "no atlas site on this ref and strand"), ns)],
        columns=["group", "label", "a_band", "bin", "n"]))
    # ---- stage 20
    em = {"n_original": 17000000, "n_reimpl": 17000000, "only_original": 0, "only_reimpl": 0, "identical": True, "examples": []}
    _write(b / "validation/exact_match_m2.json", {
        "1417": {"c275_l9_m1_i": em | {"original_elapsed_s": 7020.0, "original_max_rss_gib": 15.486, "original_peak_tree_rss_gib": 60.5},
                 I1: em | {"n_original": 17068514, "n_reimpl": 17068514, "original_elapsed_s": 7020.0,
                           "original_max_rss_gib": 15.5, "original_peak_tree_rss_gib": None}},
        "1419": {},
        "none_control": {"refs": ["chr21", "chr22"], "n_records": 2781871, "setting": "c300_lnone_m0_i", "n_original": 1894573,
                         "n_reimpl": 1894573, "only_original": 0, "only_reimpl": 0, "identical": True, "examples": [],
                         "createTRANS": {"elapsed_s": 412.0, "max_rss_gib": 0.26, "exit_status": 0, "peak_tree_rss_gib": 0.23},
                         "seconds": {"subset": 20.0, "createTRANS": 412.0, "original": 160.0, "compare": 80.0}},
        "n_compared": 3, "all_identical": True})
    return b


@pytest.fixture
def base(tmp_path):
    return _full_base(tmp_path)


def _sections(text):
    """{section number: its text} from a rendered page."""
    parts = text.split("\n## ")[1:]
    return {int(p.split(".", 1)[0]): p for p in parts}


def test_empty_base_renders_every_section_as_not_run(tmp_path):
    text = m2report.render(tmp_path)
    s = _sections(text)
    assert sorted(s) == list(range(1, 12))
    for k, body in s.items():
        assert "not run" in body, k
    assert "1417: not run (missing 1417/truth/molecules_labeled.parquet, 1417/truth/truth_reads.parquet)" in s[2]
    assert "1179: not run (missing 1179/grid/summary.json)" in s[3]
    assert "not run (missing eval/selection.json)" in s[4]
    assert "not run (missing eval/metrics.parquet)" in s[5]
    assert "not run (missing eval/hexamer_check.parquet)" in s[6]
    assert "1419: not run (missing 1419/spikein/truth/molecules_labeled.parquet)" in s[7]
    assert "1417: not run (missing 1417/spikein/ercc.json)" in s[8]
    assert "BD70: not run (missing BD70/long_read_acceptance.json)" in s[9]
    assert "not run (missing burden/templated_share.parquet)" in s[10]
    assert "not run (missing validation/exact_match_m2.json)" in s[11]
    # a base that does not exist at all behaves the same
    assert m2report.render(tmp_path / "absent") == text.replace(str(tmp_path), str(tmp_path / "absent"))


def test_runs_section(base):
    s = m2report.runs_section(base)
    # memwatch outside original/: command from the script on, paths reduced to names; 1,240.2 s; 19.123 GiB
    assert "| 13_grid_flags.py 1417 --workers 8 | 1417/grid/stage13.mem.json | 1,240 | 19.12 | 0 |" in s
    assert "c275_l9_m1_i.mem.json |" not in s.split("Stage 13")[0]
    # grid: 2 references, 85,000,000 records of which 83,140,741 primary, sum of per-reference seconds 186.0 + 28.6 = 214.6,
    # slowest chr1
    assert "| 1417 | 2 | 85,000,000 | 83,140,741 | 8 | 116 | 1,234 | 215 | chr1 (186 s) |" in s
    assert "1419: not run (missing 1419/grid/summary.json)" in s
    # 1:57:00 = 1.95 h; 16,238,356 KiB = 15.49 GiB; tree peak 60.5 from the mem.json
    assert "| 1417 | c275_l9_m1_i | 1.95 | 15.49 | 60.50 | 0 |" in s
    # killed by signal 9: exit 137 although GNU time wrote 0; 10:00 = 0.17 h; 1,048,576 KiB = 1.00 GiB; no mem.json
    assert "| 1417 | c180_l11_m2_i | 0.17 | 1.00 | n/a | 137 |" in s
    # an empty time file (GNU time killed before writing): no exit status
    assert "| 1417 | c300_l10_m0_i | running | n/a | n/a | n/a |" in s
    assert "validation/none_control/" in s.split("Stage 13")[0]


def test_truth_section(base):
    s = m2report.truth_section(base)
    assert "| Quantity | 1417 |" in s
    assert "| Molecules | 6 |" in s
    # 4 templated, 1 ambiguous, 1 non-templated of 6
    assert "| templated | 4 (66.7%) |" in s
    assert "| ambiguous | 1 (16.7%) |" in s
    assert "| non-templated | 1 (16.7%) |" in s
    assert "| With a ts read (has_ts) | 1 |" in s
    assert "| Multi-gene (gene = alphabetically first id) | 1 |" in s
    assert "| pa reads | 6 |" in s
    # member dist5_exon 100, 700, 1000, 1500, 400, 2000
    assert "| Members within 500 nt | 2 |" in s
    assert "| Members within 1,000 nt | 4 |" in s
    assert "| Members within 2,000 nt | 6 |" in s
    assert "| Molecules with a member within 1,000 nt | 3 |" in s
    # A 0-4: only a20 4 (hexamer); A 18-20: only a20 18 (no hexamer); both on the band edge
    assert "| Hexamer rate at A 0-4 | 1.000 (n 1) |" in s
    assert "| Hexamer rate at A 18-20 | 0.000 (n 1) |" in s
    # region by label: 3UTR holds templated, ambiguous and non-templated, 1 / (1 + 1)
    assert "| 1417 | 3UTR | 1 | 1 | 1 | 3 | 0.500 |" in s
    assert "| 1417 | antisense | 1 | 0 | 0 | 1 | 1.000 |" in s
    assert "| 1417 | CDS | 0 | 0 | 0 | 0 | n/a |" in s
    assert "1179: not run (missing 1179/truth/molecules_labeled.parquet, 1179/truth/truth_reads.parquet)" in s
    assert "| 1417 | 0.458 | 0.069 | 0.772 | 0.391 | 4 | 1 | 1 |" in s
    assert "over all labeled molecules" in s
    assert "| Library | Estimated tail-primed share | Hexamer rate templated | Hexamer rate non-templated | " \
           "Hexamer rate ambiguous | n_templated | n_nontemplated | n_ambiguous |" in s


def test_truth_section_without_ambiguous_estimate(base):
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text())
    del sel["ambiguous_tail_share"]
    p.write_text(json.dumps(sel))
    assert "not run (eval/selection.json has no ambiguous_tail_share)" in m2report.truth_section(base)
    p.unlink()
    assert "Ambiguous molecules" in m2report.truth_section(base)
    assert "not run (missing eval/selection.json)" in m2report.truth_section(base)


def test_grid_section(base):
    s = m2report.grid_section(base)
    assert "| Kind | Setting | 1417 |" in s
    assert "| records | Records fetched | 85,000,000 |" in s
    assert "| records | Primary records | 83,140,741 |" in s
    # over primary records, not the 85,000,000 records (20.08%, 4.39%, 58.82%): 17,068,514 / 83,140,741 = 20.53%;
    # 3,729,928 / 83,140,741 = 4.49%; 50,000,000 / 83,140,741 = 60.14%
    assert "| published | c300_l10_m1_i | 17,068,514 (20.5%) |" in s
    assert "| published | c300_l10_m1_e | 3,729,928 (4.5%) |" in s
    assert "| published | c275_l9_m1_e | n/a |" in s
    assert "| remove-all control | c300_lnone_m0_i | 50,000,000 (60.1%) |" in s
    assert "| remove-all control | c150_lnone_m0_e | n/a |" in s
    assert "1179: not run (missing 1179/grid/summary.json)" in s


def test_selection_section(base):
    s = m2report.selection_section(base)
    assert "| Mode | Chosen setting (on 1417) |" in s
    assert "| -i | c300_l10_m1_i | 0.237 | 0.990 | 0.997 | 120 / 189 |" in s
    assert "| default | none reached specificity 0.95 | n/a | n/a | n/a | 0 / 189 |" in s
    assert "| c150_l3_m2_e | 0.410 | 0.900 |" in s and "| c300_l10_m1_e | 0.036 | 0.994 |" in s
    # chosen and published at once; molecule-level CIs only (the read-level row is not used)
    assert ("| c300_l10_m1_i | chosen (-i), published | 1417 | 41,369 | 2,786 | 0.237 [0.207, 0.259] | "
            "0.990 [0.985, 0.993] | 0.997 [0.996, 0.998] |") in s
    # 1179 has no bootstrap rows; 300 / 1,000 and 95 / 100
    assert "| c300_l10_m1_i | chosen (-i), published | 1179 | 1,000 | 100 | 0.300 [n/a, n/a] | 0.950 [n/a, n/a] |" in s
    # sens 26,356 / 41,369 = 0.637 with its CI; spec and prec without bootstrap rows
    assert "| c375_l3_m2_i | published | 1417 | 41,369 | 2,786 | 0.637 [0.562, 0.690] | 0.861 [n/a, n/a] |" in s
    assert "| c375_l3_m2_i | published | 1179 | n/a | n/a | n/a | n/a | n/a |" in s
    assert "| c300_l10_m1_i | 1417 | 0.237 / 0.990 | 0.259 / 0.986 | 0.278 / 0.975 |" in s
    # molecule-level clusters only (the read-level row, first in the file, has 13,500)
    assert "clusters 1417 13,271." in s
    assert "| 1417 | 2,000 | 77,966 | 49,827 | 3 | 2 |" in s
    assert "| 500 | -i | c300_l10_m1_i | 0.250 | 0.991 | 118 / 189 |" in s
    assert "| 500 | default | none | n/a | n/a | 0 / 189 |" in s
    assert "| Templated min | Non-templated max | Mode | Chosen | Qualifying | 1417 sens / spec | 1179 sens / spec |" in s
    # sens / spec, not prec (0.970)
    assert "| 7 | 2 | -i | c300_l10_m1_i | 100 | 0.200 / 0.990 | n/a |" in s
    assert "| 10 | 6 | default | none | 0 | n/a | n/a |" in s
    # intergenic: 5,182 of 41,369 templated units, at most 500 removed (c375_l3_m2_i; c300_l10_m1_i removes none) at cap
    # 1,000 and 8/4 (the cap 500 and 7/2 rows, with 99 and 77 removed, are ignored); ceiling (41,369 - 5,182 + 500) / 41,369
    # = 0.8868; non-templated tn + fp = 125 + 3 = 128, at most 8 removed
    assert "| 1417 | 41,369 | 5,182 | 500 | 0.887 | 128 | 8 |" in s
    # (0.95 x 2,786 - 128) / (2,786 - 128) = 2,518.7 / 2,658 = 0.9476
    assert ("specificity >= 0.95 over all 2,786 non-templated units on 1417 is at least (0.95 x 2,786 - 128) / 2,658 = "
            "0.948 over the 2,658 genic ones") in s


def test_selection_section_reads_threshold_and_library_from_the_file(base):
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text())
    sel["min_spec"] = 0.9
    p.write_text(json.dumps(sel))
    s = m2report.selection_section(base)
    assert "| default | none reached specificity 0.9 | n/a | n/a | n/a | 0 / 189 |" in s
    # (0.9 x 2,786 - 128) / 2,658 = 2,379.4 / 2,658 = 0.8952
    assert "specificity >= 0.9 over all 2,786 non-templated units on 1417 is at least (0.9 x 2,786 - 128) / 2,658 = 0.895" in s
    # no candidate in a mode: no threshold claim and no frontier; another selection library in the headers
    sel["modes"]["e"]["selection"] |= {"n_candidates": 0, "frontier": []}
    sel["selection_library"] = "1419"
    p.write_text(json.dumps(sel))
    s = m2report.selection_section(base)
    assert "| default | no candidates | n/a | n/a | n/a | 0 / 0 |" in s
    assert "Frontier" not in s
    assert "| Mode | Chosen setting (on 1419) |" in s
    assert "Member cap sensitivity (selection rerun on 1419 at each cap)" in s
    assert "Label cut-off sweep (selection on 1419 at each" in s
    # metrics.parquet has no 1419 rows, so there is no genic-specificity line
    assert "genic ones" not in s


def test_selection_section_without_bootstrap_or_metrics(base):
    (base / "eval/bootstrap.parquet").unlink()
    (base / "eval/metrics.parquet").unlink()
    s = m2report.selection_section(base)
    assert "| -i | c300_l10_m1_i | 0.237 | 0.990 | 0.997 | 120 / 189 |" in s
    assert "| c300_l10_m1_i | chosen (-i), published | 1417 | 41,369 | 2,786 | 0.237 [n/a, n/a] |" in s
    assert "Bootstrap: not run (missing eval/bootstrap.parquet)" in s
    assert "Intergenic molecules: not run (missing eval/metrics.parquet)" in s


def test_selection_section_chosen_setting_outside_the_published_ones(base):
    p = base / "eval/selection.json"
    sel = json.loads(p.read_text())
    sel["modes"]["i"]["selection"]["tag"] = "c450_l9_m1_i"
    sel["modes"]["i"]["metrics"] = {"1417": I1_1417}
    p.write_text(json.dumps(sel))
    s = m2report.selection_section(base)
    # its metrics come from modes, not from published; no bootstrap rows for it
    assert "| c450_l9_m1_i | chosen (-i) | 1417 | 41,369 | 2,786 | 0.237 [n/a, n/a] | 0.990 [n/a, n/a] |" in s
    assert "| c450_l9_m1_i | 1179 | n/a | n/a | n/a |" in s
    assert "| c300_l10_m1_i | published | 1417 | 41,369 | 2,786 | 0.237 [0.207, 0.259] |" in s


def test_strata_section(base):
    s = m2report.strata_section(base)
    # spec 2,757 / 2,786 = 0.990; sens 100 / 2,000, 1,000 / 6,400 = 0.156, 8,695 / 32,969 = 0.264
    assert "| c300_l10_m1_i | 1417 | 0.990 | 0.050 | 0.156 | 0.264 |" in s
    # intergenic 0 / 5,182 and 125 / 128; 3UTR 5,000 / 15,000 and 2,000 / 2,020; other regions absent
    assert "| c300_l10_m1_i | 1417 | 0.000 / 0.977 | n/a | n/a | n/a | n/a | 0.333 / 0.990 | n/a |" in s
    assert "| Setting | Library | intergenic | intron | ncexon | 5UTR | CDS | 3UTR | antisense |" in s
    assert "| 1417 | a_band | 8-9 | 2,000 | 0 |" in s
    assert "| 1417 | region | 3UTR | 15,000 | 2,020 |" in s
    assert "| 1417 | region | intergenic | 5,182 | 128 |" in s
    (base / "eval/selection.json").unlink()
    s = m2report.strata_section(base)
    assert "chosen settings: not run (missing eval/selection.json)" in s
    assert "| c300_l10_m1_i | 1417 | 0.990 | 0.050 | 0.156 | 0.264 |" in s


def test_hexamer_section(base):
    s = m2report.hexamer_section(base)
    assert "| c375_l3_m2_i | 1417 | all | 28,762 | 0.061 | 20,174 | 0.229 |" in s
    assert "| c375_l3_m2_i | 1417 | non-templated | 387 | 0.305 | 2,399 | 0.836 |" in s
    assert "| c375_l3_m2_i | 1417 | ambiguous | 0 | n/a | 10 | 0.400 |" in s
    assert "| 1 | 0.900 |" not in s


def test_spikein_section(base):
    s = m2report.spikein_section(base)
    assert "| 1417 | internal | 2 | 1 | 1 | 4 |" in s
    assert "| 1417 | tail | 2 | 1 | 2 | 5 |" in s
    assert "| 1417 | other | 0 | 1 | 0 | 1 |" in s
    # internal and templated 2 + tail and non-templated 2 over the 7 internal or tail molecules labeled T or NT (the
    # ambiguous ones left out); SIRV_E0 alone: internal T 1 (a20 15), internal NT 1 (a20 1), tail T 1 (a20 9): 1 of 3
    assert "1417: agreement 0.571 (4 of 7), SIRV_E0 only 0.333 (1 of 3)" in s
    assert "forced" in s
    # the ambiguous internal molecules (ERCC none, SIRV_E0 a20 7) count in the denominator, not as templated
    assert "| 1417 | all | 4 | 2 (0.500) | 5 | 2 (0.400) |" in s
    assert "| 1417 | ERCC | 1 | 1 (1.000) | 2 | 1 (0.500) |" in s
    assert "| 1417 | SIRV_E0 | 3 | 1 (0.333) | 1 | 0 (0.000) |" in s
    assert "| 1417 | long_SIRV | 0 | 0 (n/a) | 2 | 1 (0.500) |" in s
    assert "1419: not run (missing 1419/spikein/truth/molecules_labeled.parquet)" in s
    # molecule level only: 10 / 40, 50 / 52; ERCC 5 / 10, 20 / 20; the read-level row (9 / 10) is not used
    assert "| 1417 | c300_l10_m1_i | 0.250 / 0.962 | 0.500 / 1.000 | n/a | n/a |" in s
    (base / "eval/spikein_metrics.parquet").unlink()
    assert "Metrics per module: not run (missing eval/spikein_metrics.parquet)" in m2report.spikein_section(base)


def test_ercc_section(base):
    s = m2report.ercc_section(base)
    assert s.startswith("## 8. ERCC count accuracy (provisional)")
    assert f"Provisional (1417): {NOTE}" in s
    assert "| Library | Setting | n | pearson_log10 | spearman | umis | umis_kept_share |" in s
    assert "| 1417 | unfiltered | 61 | 0.912 | 0.888 | 52,000 | 1.000 |" in s
    assert "| 1417 | c300_l10_m1_i | 61 | 0.920 | 0.890 | 50,502 | 0.971 |" in s
    assert "| 1417 | c375_l3_m2_i | 3 | n/a | n/a | 3 | 0.000 |" in s
    assert "| 1419 | unfiltered | 60 | 0.900 | 0.800 | 4,000 | 1.000 |" in s


def test_long_read_section(base):
    s = m2report.long_read_section(base)
    # dropped 178,467 / 1,610,467 = 11.08%
    assert "| BD67 | 2 | 1,735,308 | 14,097 | 110,744 | 1,610,467 | 1,432,000 | 77,500 | 113 | 100,854 | 11.1% | 0 |" in s
    # per read over sites with an accepted read (a20 3 and 12 have none), cut-offs inclusive. At 8/4, templated a20 15 (3, no
    # hexamer), 9 (1, hexamer), 8 (1, no), 14 (1, hexamer): 2 / 6 = 0.333; non-templated a20 2 (2, hexamer), 1 (2, no), 4 (2,
    # hexamer): 4 / 6 = 0.667; ratio 2.00. At 14/8, templated a20 15 and 14: 1 / 4 = 0.25; non-templated a20 2, 1, 4 and 8:
    # 4 / 7 = 0.571; ratio 2.29. Exclusive cut-offs would give 0.400, 0.500, 0.000 and 0.667.
    assert "| BD67 | 8/4 | 28,473 / 6,469 / 7,228 | 0.106 | 0.541 | 0.443 | 5.12 | True | 0.333 | 0.667 | 2.00 |" in s
    assert "| BD67 | 14/8 | 11,551 / 16,279 / 14,340 | 0.046 | 0.456 | 0.200 | 9.95 | True | 0.250 | 0.571 | 2.29 |" in s
    assert "about 7%" in s
    assert "| BD67 | 16,408 | 0.615 | 0.491 / 0.907 / 0.805 | 0.993 / 0.996 / 0.994 |" in s
    assert "BD70: not run (missing BD70/long_read_acceptance.json)" in s


def test_burden_section(base):
    s = m2report.burden_section(base)
    assert "| Stratum kind | Stratum | 1417 | cells |" in s
    assert "| all | all | 0.819 [0.804, 0.835] (144,474) | 0.900 [0.880, 0.920] (1,000) |" in s
    assert "| region | 3UTR | 0.507 [0.479, 0.532] (31,129) | n/a |" in s
    assert "| region | intron | n/a | n/a |" in s
    assert "| has_ts | True | 0.724 [0.638, 0.794] (724) | n/a |" in s
    assert "| all | all | 0.916 | 0.947 |" in s
    # frac with its own interval
    assert "| 14-20 | 0.278 [0.271, 0.285] | n/a |" in s
    assert "within 5 nt on the same strand" in s
    assert "| 1417_vs_1419 | templated | 95,920 | 0 | 0 | 500 |" in s
    assert "| 1417_vs_1419 | ambiguous | 100 | 11,219 | 0 | 0 |" in s
    # rows in label order, not alphabetical
    assert s.index("| 1417_vs_1419 | templated |") < s.index("| 1417_vs_1419 | ambiguous |") < \
        s.index("| 1417_vs_1419 | non-templated |")
    # 114,000 sites, 113,500 matched (99.56%), 113,400 of them with the same label (99.91%)
    assert "| 1417_vs_1419 | 114,000 | 113,500 (99.6%) | 113,400 (99.9%) |" in s
    assert "| 1417_vs_1419 | 1,212 | 1.000 | 0.00e+00 |" in s
    assert "| 1417 | E | 0.421 | 0.407 |" in s
    (base / "burden/same_gene.parquet").unlink()
    assert "Same gene: not run (missing burden/same_gene.parquet)" in m2report.burden_section(base)


def test_burden_section_without_a_pair(base):
    # stage 18 with one library (1417 alone): empty same-site and same-gene frames
    pd.DataFrame(columns=["pair", "label_a", "label_b", "n"]).to_parquet(base / "burden/same_site.parquet", index=False)
    pd.DataFrame([], columns=["pair", "n_genes", "spearman_rho", "p_value"]).to_parquet(base / "burden/same_gene.parquet",
                                                                                         index=False)
    s = m2report.burden_section(base)
    assert ("Same site: no pair in burden/same_site.parquet (stage 18 needs 1417 and 1419, or nuclei and cells)") in s
    assert ("Same gene: no pair in burden/same_gene.parquet (stage 18 needs 1417 and 1419, or nuclei and cells)") in s
    assert "| Pair |" not in s
    assert "| 1417 | E | 0.421 | 0.407 |" in s
    assert "Same site" in m2report.render(base)


def test_burden_section_polyasite(base):
    s = m2report.burden_section(base)
    assert "does not screen out candidate sites with A-rich genome downstream" in s
    assert ("The null row takes every molecule of the group with its site moved 1,000 nt upstream in transcript sense on the "
            "same ref and strand; its share within 10 nt is the share expected by chance at this atlas density.") in s
    assert "| Label | Molecules | 0-10 | 11-50 | 51-200 | >200 | no atlas site on this ref and strand |" in s
    one, cells = s.split("Group 1417:")[1].split("Group cells:")
    # 9,000 / 118,278 = 0.0761, 20,000 / 118,278 = 0.1691, 40,000 / 118,278 = 0.3382, 44,000 / 118,278 = 0.3720,
    # 5,278 / 118,278 = 0.0446
    assert "| templated | 118,278 | 0.076 | 0.169 | 0.338 | 0.372 | 0.045 |" in one
    # 3,000 / 15,374 = 0.1951, 4,000 / 15,374 = 0.2602, 3,674 / 15,374 = 0.2390, 1,700 / 15,374 = 0.1106
    assert "| ambiguous | 15,374 | 0.195 | 0.195 | 0.260 | 0.239 | 0.111 |" in one
    # 4,000 / 10,822 = 0.3696, 2,000 / 10,822 = 0.1848, 2,681 / 10,822 = 0.2477, 141 / 10,822 = 0.0130
    assert "| non-templated | 10,822 | 0.370 | 0.185 | 0.185 | 0.248 | 0.013 |" in one
    # 144,474 molecules: 31,500 = 0.2180, 40,000 = 0.2769, 50,000 = 0.3461, 15,000 = 0.1038, 7,974 = 0.0552
    assert "| null (sites 1 kb upstream) | 144,474 | 0.218 | 0.277 | 0.346 | 0.104 | 0.055 |" in one
    # 1,000 / 18,577 = 0.0538, 3,000 / 18,577 = 0.1615, 6,000 / 18,577 = 0.3230, 7,577 / 18,577 = 0.4079
    assert "| templated, A band 8-9 | 18,577 | 0.054 | 0.161 | 0.323 | 0.408 | 0.054 |" in one
    # 3,000 / 40,103 = 0.0748, 7,000 = 0.1746, 14,000 = 0.3491, 14,103 = 0.3517, 2,000 = 0.0499
    assert "| templated, A band 14-20 | 40,103 | 0.075 | 0.175 | 0.349 | 0.352 | 0.050 |" in one
    # rows: labels in page order, the null, then the bands
    assert one.index("| templated |") < one.index("| ambiguous |") < one.index("| non-templated |") < \
        one.index("| null (sites 1 kb upstream) |") < one.index("| templated, A band 8-9 |") < \
        one.index("| templated, A band 10-13 |") < one.index("| templated, A band 14-20 |")
    assert "| templated | 1,000 | 0.100 | 0.200 | 0.300 | 0.300 | 0.100 |" in cells
    assert "| non-templated | 0 | n/a | n/a | n/a | n/a | n/a |" in cells
    assert "| ambiguous | 20 | 0.250 | 0.250 | 0.000 | 0.300 | 0.200 |" in cells
    # 255, 255, 255, 153, 102 of 1,020
    assert "| null (sites 1 kb upstream) | 1,020 | 0.250 | 0.250 | 0.250 | 0.150 | 0.100 |" in cells
    assert "| templated, A band 8-9 | 0 | n/a | n/a | n/a | n/a | n/a |" in cells
    (base / "burden/polyasite.parquet").unlink()
    s = m2report.burden_section(base)
    assert "PolyASite proximity: not run (missing burden/polyasite.parquet)" in s and "Group 1417:" not in s


def test_validation_section(base):
    s = m2report.validation_section(base)
    assert "| 1417 | c275_l9_m1_i | 17,000,000 | 17,000,000 | 0 | 0 | True | 1.95 | 15.49 | 60.50 |" in s
    assert "| 1417 | c300_l10_m1_i | 17,068,514 | 17,068,514 | 0 | 0 | True | 1.95 | 15.50 | n/a |" in s
    assert "1419: no completed original run" in s
    assert "1179: not in exact_match_m2.json; the 1179 grid has no original-versus-reimplementation evidence" in s
    # 20 + 412 + 160 + 80 = 672 s
    assert "| c300_lnone_m0_i | chr21, chr22 | 2,781,871 | 1,894,573 | 1,894,573 | 0 | 0 | True | 412 | 0.26 | 672 |" in s
    assert "Compared: 3; all identical: True" in s
    assert "Removed-record sets (every record the original fetches" in s


def test_validation_section_with_the_1179_subset(base):
    p = base / "validation/exact_match_m2.json"
    em = json.loads(p.read_text())
    r = {"setting": I1, "n_original": 2000000, "n_reimpl": 2000000, "only_original": 0, "only_reimpl": 0, "identical": True,
         "examples": [], "original_elapsed_s": 1800.0, "original_max_rss_gib": 3.25, "original_peak_tree_rss_gib": 6.5,
         "compare_s": 300.0}
    em["1179_subset"] = {"refs": ["chr21", "chr22"], "n_records": 18382751,
                         "createTRANS": {m: {"elapsed_s": 900.0, "max_rss_gib": 0.3, "exit_status": 0, "peak_tree_rss_gib": 0.3}
                                         for m in "ie"},
                         "settings": {I1: r, E1: r | {"setting": E1, "only_original": 2, "identical": False}}}
    p.write_text(json.dumps(em))
    s = m2report.validation_section(base)
    # the subset rows join the library table: 1800 s = 0.50 h
    assert "| 1179 (chr21, chr22) | c300_l10_m1_i | 2,000,000 | 2,000,000 | 0 | 0 | True | 0.50 | 3.25 | 6.50 |" in s
    assert "| 1179 (chr21, chr22) | c300_l10_m1_e | 2,000,000 | 2,000,000 | 2 | 0 | False | 0.50 | 3.25 | 6.50 |" in s
    assert "1179: original tool on the chr21, chr22 subset only (18,382,751 records)" in s
    assert "the 1179 grid has no original-versus-reimplementation evidence" not in s
    # the nuclei rows are unchanged
    assert "| 1417 | c275_l9_m1_i | 17,000,000 | 17,000,000 | 0 | 0 | True | 1.95 | 15.49 | 60.50 |" in s


def test_validation_section_on_a_stale_file(base):
    # the schema an earlier draft left in the real OUT: no library keys, no none_control, no totals
    (base / "validation/exact_match_m2.json").write_text(json.dumps({"none_control_chr21_chr22": {"error": "createTRANS failed"}}))
    s = m2report.validation_section(base)
    assert "1417: not in exact_match_m2.json" in s
    assert "Remove-all control: not run (no none_control entry in validation/exact_match_m2.json)" in s
    assert "Compared: n/a; all identical: n/a" in s


def test_render_and_script(base, tmp_path):
    text = m2report.render(base)
    s = _sections(text)
    assert sorted(s) == list(range(1, 12))
    assert text.startswith("# Milestone 2 results (generated by scripts/21_m2_summary.py)")
    assert "not run (missing eval/selection.json)" not in text
    out = tmp_path / "docs" / "m2_results.md"
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "21_m2_summary.py"), "--base", str(base), "--out", str(out)],
                       capture_output=True, text=True, env=env, check=True)
    assert out.read_text() == text
    line = json.loads(r.stdout.strip().splitlines()[-1])
    assert line == {"out": str(out), "sections": 11, "not_run": text.count("not run")}
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_m2report.py`
Expected: FAIL; the first error is `ImportError: cannot import name 'm2report' from 'pafbench' (pafbench/__init__.py)`.

- [ ] **Step 3: Create `pafbench/m2report.py`**

```python
"""Stage 21: the milestone 2 results page. One function per section; each reads its files under a base directory (OUT) and
prints "not run (missing ...)" with the absent file names instead of numbers."""
import json
from pathlib import Path

import pandas as pd

from .config import A_BANDS, CELLS, LABEL_CUTOFFS, LONG_READS, MEMBER_CAP, NUCLEI, SENS_CAPS
from .evaluate import MODULES
from .flags import Setting, setting_tag
from .grid import COVLENS, published
from .polyasite import BINS, NULL_LABEL
from .regions import REGION_NAMES
from .validation import run_stats

LIBS = NUCLEI + CELLS
LABELS = ("templated", "ambiguous", "nontemplated")
LABEL_NAMES = {"templated": "templated", "ambiguous": "ambiguous", "nontemplated": "non-templated", "all": "all"}
MODE_NAMES = {"e": "default", "i": "-i"}
BANDS = tuple(b[0] for b in A_BANDS)
SPIKE_LABELS = ("internal", "tail", "other")
CUTOFFS = "{templated_min}/{nontemplated_max}".format(**LABEL_CUTOFFS)
PUBLISHED_TAGS = [setting_tag(s) for s in published()]
CONTROL_TAGS = [setting_tag(Setting(c, None, 0, introns)) for introns in (False, True) for c in COVLENS]


def _int(x):
    return "n/a" if x is None or pd.isna(x) else f"{int(x):,}"


def _r(x, nd=3):
    return "n/a" if x is None or pd.isna(x) else f"{float(x):.{nd}f}"


def _pct(n, d):
    return "n/a" if not d else f"{100 * n / d:.1f}%"


def _share(n, d):
    return f"{_int(n)} ({_r(n / d if d else None)})"


def _ci(est, lo, hi):
    return "n/a" if est is None or pd.isna(est) else f"{_r(est)} [{_r(lo)}, {_r(hi)}]"


def _sec(x):
    return "n/a" if x is None else f"{x:,.0f}"


def _value(x):
    """A JSON scalar for a table cell: ints with thousands separators, floats to 3 decimals, null as n/a."""
    if isinstance(x, (bool, str)):
        return str(x)
    if isinstance(x, int):
        return _int(x)
    return _r(x)


def _table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def _notes(out, notes):
    """out plus notes as a bullet list, after a blank line (a bare line right after a table would be read as a table row)."""
    return out + ([""] if out[-1] else []) + [f"- {n}" for n in notes] if notes else out


def _rel(base, p):
    return Path(p).relative_to(base).as_posix()


def _not_run(base, paths):
    return f"not run (missing {', '.join(_rel(base, p) for p in paths)})"


def _json(p):
    return json.loads(p.read_text()) if p.exists() else None


def _parquet(p, columns=None):
    return pd.read_parquet(p, columns=columns) if p.exists() else None


def _command(cmd):
    """The command from its script on (or whole), with paths reduced to their last component."""
    i = next((k for k, a in enumerate(cmd) if a.endswith((".py", ".sh"))), 0)
    return " ".join(Path(a).name if "/" in a else a for a in cmd[i:])


def _chosen(sel):
    """{tag: [roles]} of the chosen settings (per mode) and the published ones, chosen first."""
    roles = {}
    for mode, d in sel["modes"].items():
        if d["selection"]["tag"]:
            roles.setdefault(d["selection"]["tag"], []).append(f"chosen ({MODE_NAMES[mode]})")
    for tag in sel["published"]:
        roles.setdefault(tag, []).append("published")
    return roles


def _entry(sel, tag, lib):
    """Per-level metrics of a chosen or published tag on one library ({} when the tag was not in its truth_flags)."""
    for d in sel["modes"].values():
        if d["selection"]["tag"] == tag:
            return d["metrics"].get(lib, {})
    return sel["published"].get(tag, {}).get(lib, {})


def _report_tags(base):
    """Chosen tags from eval/selection.json (when present) then the published tags, plus a note when the selection is absent."""
    p = base / "eval" / "selection.json"
    sel = _json(p)
    if sel is None:
        return list(PUBLISHED_TAGS), [f"chosen settings: {_not_run(base, [p])}; published settings only", ""]
    chosen = [d["selection"]["tag"] for d in sel["modes"].values() if d["selection"]["tag"]]
    return list(dict.fromkeys(chosen + PUBLISHED_TAGS)), []


def runs_section(base):
    base = Path(base)
    out = ["## 1. Runs", "",
           "Stage runs under memwatch (every `*.mem.json` under the base directory outside `*/original/`; a JSON written "
           "elsewhere is not found). This includes stage 20's createTRANS run for the remove-all control "
           "(`validation/none_control/`), an original-tool run, and its createTRANS and original-tool runs on the 1179 subset "
           "(`validation/subset_1179/`). Peak RSS is summed over the process tree, so pages shared by "
           "forked workers count once per process (an upper bound). memwatch samples every 2 s (interval_s), so the peak of a "
           "shorter run reads near zero.", ""]
    mem = [p for p in base.rglob("*.mem.json") if "original" not in p.relative_to(base).parts[:-1]]
    rows = []
    for p in mem:
        d = json.loads(p.read_text())
        rows.append([_command(d["cmd"]), _rel(base, p), _sec(d["elapsed_s"]), _r(d["peak_tree_rss_gib"], 2), d["exit_status"]])
    out += _table(["Command", "File", "Wall (s)", "Peak tree RSS (GiB)", "Exit"], sorted(rows)) if rows else \
        ["not run (missing *.mem.json)"]

    out += ["", "Stage 13 grid scans (`<lib>/grid/summary.json`; wall time is that of the last invocation, so a resumed run "
                "counts only its own references; the per-reference sum covers all parts):", ""]
    rows, notes = [], []
    for lib in LIBS:
        p = base / lib / "grid" / "summary.json"
        d = _json(p)
        if d is None:
            notes.append(f"{lib}: {_not_run(base, [p])}")
            continue
        per = d["per_ref"]
        slow = max(per, key=lambda r: per[r]["seconds"]) if per else None
        rows.append([lib, _int(len(per)), _int(d["records"]), _int(d["primary"]), d["workers"], d["scanned"], _sec(d["seconds"]),
                     _sec(sum(v["seconds"] for v in per.values())), f"{slow} ({_sec(per[slow]['seconds'])} s)" if slow else "n/a"])
    if rows:
        out += _table(["Library", "References", "Records", "Primary records", "Workers", "Scanned in last run", "Wall of last run (s)",
                       "Sum over references (s)", "Slowest reference"], rows)
    out = _notes(out, notes)

    out += ["", "Original polyAfilter runs (`<lib>/original/<step>.time.txt`, `.mem.json`). Max RSS is the largest single "
                "process (GNU time); exit 128 + N means killed by signal N.", ""]
    rows = []
    for p in sorted(base.glob("*/original/*.time.txt")):
        st = run_stats(str(p)[:-len(".time.txt")])
        wall = "running" if st["elapsed_s"] is None else f"{st['elapsed_s'] / 3600:.2f}"
        rows.append([p.parent.parent.name, p.name[:-len(".time.txt")], wall, _r(st["max_rss_gib"], 2),
                     _r(st["peak_tree_rss_gib"], 2), "n/a" if st["exit_status"] is None else st["exit_status"]])
    out += _table(["Library", "Step", "Wall (h)", "Max RSS (GiB)", "Peak tree RSS (GiB)", "Exit"], rows) if rows else \
        ["not run (missing */original/*.time.txt)"]
    return "\n".join(out)


def truth_section(base):
    base = Path(base)
    out = ["## 2. Truth sets", "",
           "Source: `<lib>/truth/molecules_labeled.parquet` and `<lib>/truth/truth_reads.parquet` (stage 12). "
           f"Labels at {CUTOFFS} A's in the 20 nt after the site; members are selected member reads (gene match, "
           "dist3_exon >= -25) by exon-aware distance.", ""]
    data, notes = {}, []
    for lib in LIBS:
        paths = [base / lib / "truth" / "molecules_labeled.parquet", base / lib / "truth" / "truth_reads.parquet"]
        absent = [p for p in paths if not p.exists()]
        if absent:
            notes.append(f"{lib}: {_not_run(base, absent)}")
            continue
        data[lib] = (pd.read_parquet(paths[0], columns=["label", "region", "has_ts", "a20", "hexamer", "genes", "n_members"]),
                     pd.read_parquet(paths[1], columns=["role", "dist5_exon"]))
    if data:
        def row(name, f):
            return [name] + [f(m, r) for m, r in data.values()]

        def hexa(s):
            return f"{_r(s.mean() if len(s) else None)} (n {_int(len(s))})"

        rows = [row("Molecules", lambda m, r: _int(len(m)))]
        rows += [row(LABEL_NAMES[k], lambda m, r, k=k: f"{_int((m.label == k).sum())} ({_pct((m.label == k).sum(), len(m))})")
                 for k in LABELS]
        rows += [row("With a ts read (has_ts)", lambda m, r: _int(m.has_ts.sum())),
                 row("Multi-gene (gene = alphabetically first id)", lambda m, r: _int(m.genes.str.contains(";").sum())),
                 row("pa reads", lambda m, r: _int((r.role == "pa").sum()))]
        rows += [row(f"Members within {cap:,} nt", lambda m, r, cap=cap: _int(((r.role == "member") & (r.dist5_exon <= cap)).sum()))
                 for cap in SENS_CAPS]
        rows += [row(f"Molecules with a member within {MEMBER_CAP:,} nt", lambda m, r: _int((m.n_members > 0).sum())),
                 row("Hexamer rate at A 0-4", lambda m, r: hexa(m.hexamer[m.a20 <= 4])),
                 row("Hexamer rate at A 18-20", lambda m, r: hexa(m.hexamer[m.a20 >= 18]))]
        out += _table(["Quantity"] + list(data), rows)
    out = _notes(out, notes)
    if data:
        out += ["", "Region at the last cDNA base, by label (molecules):", ""]
        rows = []
        for lib, (m, _) in data.items():
            c = pd.crosstab(m.region, m.label).reindex(index=list(REGION_NAMES), columns=list(LABELS), fill_value=0)
            for region, t in c.iterrows():
                resolved = t.templated + t.nontemplated
                rows.append([lib, region, _int(t.templated), _int(t.ambiguous), _int(t.nontemplated), _int(t.sum()),
                             _r(t.templated / resolved if resolved else None)])
        out += _table(["Library", "Region", "templated", "ambiguous", "non-templated", "total",
                       "templated / (templated + non-templated)"], rows)

    out += ["", "Ambiguous molecules: hexamer-based estimate of their tail-primed share over all labeled molecules, not only "
                f"the cap {MEMBER_CAP:,} units the metrics use (`eval/selection.json` ambiguous_tail_share, stage 17; "
                "(rate ambiguous - rate templated) / (rate non-templated - rate templated), clipped to [0, 1]):", ""]
    p = base / "eval" / "selection.json"
    sel = _json(p)
    if sel is None:
        out.append(_not_run(base, [p]))
    elif "ambiguous_tail_share" not in sel:
        out.append("not run (eval/selection.json has no ambiguous_tail_share)")
    else:
        ats = sel["ambiguous_tail_share"]
        nkeys = list(dict.fromkeys(k for d in ats.values() for k in d if k.startswith("n_")))
        rows = [[lib, _r(d.get("estimate")), _r(d.get("rate_templated")), _r(d.get("rate_nontemplated")),
                 _r(d.get("rate_ambiguous"))] + [_int(d.get(k)) for k in nkeys] for lib, d in ats.items()]
        out += _table(["Library", "Estimated tail-primed share", "Hexamer rate templated", "Hexamer rate non-templated",
                       "Hexamer rate ambiguous"] + nkeys, rows)
    return "\n".join(out)


def grid_section(base):
    base = Path(base)
    out = ["## 3. Grid: removed primary records", "",
           "Source: `<lib>/grid/summary.json` (stage 13). Share of primary records in brackets. The remove-all control "
           "(MINSNRLEN none) removes every non-terminal read on an expressed transcript at that COVLEN; it bounds removals "
           "only within its own mode.", ""]
    data, notes = {}, []
    for lib in LIBS:
        p = base / lib / "grid" / "summary.json"
        d = _json(p)
        if d is None:
            notes.append(f"{lib}: {_not_run(base, [p])}")
        else:
            data[lib] = d
    if data:
        rows = [["records", "Records fetched"] + [_int(d["records"]) for d in data.values()],
                ["records", "Primary records"] + [_int(d["primary"]) for d in data.values()]]
        for kind, tags in (("published", PUBLISHED_TAGS), ("remove-all control", CONTROL_TAGS)):
            for tag in tags:
                cells = []
                for d in data.values():
                    v = d["removed"].get(tag)
                    cells.append("n/a" if v is None else f"{_int(v)} ({_pct(v, d['primary'])})")
                rows.append([kind, tag] + cells)
        out += _table(["Kind", "Setting"] + list(data), rows)
    out = _notes(out, notes)
    return "\n".join(out)


def selection_section(base):
    base = Path(base)
    out = ["## 4. Setting selection", ""]
    p = base / "eval" / "selection.json"
    sel = _json(p)
    if sel is None:
        return "\n".join(out + [_not_run(base, [p])])
    libs, sl, min_spec = sel["libraries"], sel["selection_library"], sel["min_spec"]
    out += [f"Source: `eval/selection.json`, `eval/bootstrap.parquet`, `eval/metrics.parquet` (stage 17). Rule: {sel['rule']}", ""]
    rows, fronts = [], []
    for mode, d in sel["modes"].items():
        s = d["selection"]
        none = "no candidates" if s["n_candidates"] == 0 else f"none reached specificity {min_spec:g}"
        rows.append([MODE_NAMES[mode], s["tag"] or none, _r(s["sens"]), _r(s["spec"]), _r(s["prec"]),
                     f"{s['n_qualifying']} / {s['n_candidates']}"])
        if s["tag"] is None and s["n_candidates"]:
            fronts.append((mode, s["frontier"]))
    out += _table(["Mode", f"Chosen setting (on {sl})", "Sens", "Spec", "Prec", "Qualifying / candidates"], rows)
    for mode, front in fronts:
        out += ["", f"Frontier, {MODE_NAMES[mode]} mode (no setting chosen; candidates not dominated in sensitivity and "
                    f"specificity):", ""]
        out += _table(["Setting", "Sens", "Spec"], [[t, _r(a), _r(b)] for t, a, b in front])

    bp = base / "eval" / "bootstrap.parquet"
    boot = _parquet(bp)
    ci = {}
    if boot is not None:
        b = boot[(boot.level == "molecule") & (boot.cap == MEMBER_CAP)]
        ci = {(r.lib, r.tag, r.metric): (r.lo, r.hi) for r in b.itertuples()}
    out += ["", f"Chosen and published settings, molecule level (all reads removed), member cap {MEMBER_CAP:,} nt, {CUTOFFS} "
                f"labels, 95% gene-bootstrap interval in brackets:", ""]
    roles = _chosen(sel)
    rows, rows_lvl = [], []
    for tag, rs in roles.items():
        for lib in libs:
            e = _entry(sel, tag, lib)
            m = e.get("molecule")
            if not m:
                rows.append([tag, ", ".join(rs), lib] + ["n/a"] * 5)
                rows_lvl.append([tag, lib] + ["n/a"] * 3)
                continue
            cells = [_ci(m[k], *ci.get((lib, tag, k), (None, None))) for k in ("sens", "spec", "prec")]
            rows.append([tag, ", ".join(rs), lib, _int(m["tp"] + m["fn"]), _int(m["tn"] + m["fp"])] + cells)
            rows_lvl.append([tag, lib] + [f"{_r(e[lv]['sens'])} / {_r(e[lv]['spec'])}" if lv in e else "n/a"
                                          for lv in ("molecule", "molecule_majority", "read")])
    out += _table(["Setting", "Role", "Library", "Templated", "Non-templated", "Sens", "Spec", "Prec"], rows)
    if boot is None:
        out += ["", f"Bootstrap: {_not_run(base, [bp])}"]
    else:
        clusters = boot[boot.level == "molecule"].groupby("lib", sort=False).n_clusters.first()
        out += ["", "Bootstrap: 2.5th and 97.5th percentiles over gene-cluster resamples (cluster = gene, or a 10 kb locus "
                    "window for intergenic molecules); clusters " + "; ".join(f"{k} {_int(v)}" for k, v in clusters.items()) + "."]
    out += ["", "Majority rule (molecule removed when most of its reads are; our extension, spec 5.3) and read level, "
                "sens / spec:", ""]
    out += _table(["Setting", "Library", "Molecule (all reads)", "Molecule (majority)", "Read"], rows_lvl)

    out += ["", "Units per member cap (molecules with a truth read missing from truth_flags are dropped):", ""]
    out += _table(["Library", "Cap (nt)", "Read units", "Molecule units", "Missing reads", "Dropped molecules"],
                  [[lib, _int(cap), _int(u["read_units"]), _int(u["molecule_units"]), _int(u["missing_reads"]),
                    _int(u["dropped_molecules"])] for lib, caps in sel["units"].items() for cap, u in caps.items()])
    out += ["", f"Member cap sensitivity (selection rerun on {sl} at each cap):", ""]
    out += _table(["Cap (nt)", "Mode", "Chosen", "Sens", "Spec", "Qualifying / candidates"],
                  [[_int(cap), MODE_NAMES[mode], s["tag"] or "none", _r(s["sens"]), _r(s["spec"]),
                    f"{s['n_qualifying']} / {s['n_candidates']}"] for cap, modes in sel["cap_sensitivity"].items()
                   for mode, s in modes.items()])
    out += ["", f"Label cut-off sweep (selection on {sl} at each templated / non-templated pair; molecule level on every "
                "library):", ""]
    rows = []
    for e in sel["cutoff_sweep"]:
        by = e["by_library"]
        rows.append([e["templated_min"], e["nontemplated_max"], MODE_NAMES[e["mode"]], e["tag"] or "none", e["n_qualifying"]]
                    + [f"{_r(by[lib]['sens'])} / {_r(by[lib]['spec'])}" if by.get(lib) else "n/a" for lib in libs])
    out += _table(["Templated min", "Non-templated max", "Mode", "Chosen", "Qualifying"] + [f"{lib} sens / spec" for lib in libs], rows)

    out += ["", "Intergenic molecules: polyAfilter's windows lie on expressed transcripts, so an intergenic molecule is "
                "removed only through reads elsewhere. 'Removed at most' is the largest count over all settings in "
                f"metrics.parquet (molecule level, cap {MEMBER_CAP:,}, {CUTOFFS}); the ceiling is the stratum-all sensitivity if "
                "every other templated molecule were removed. Intergenic non-templated molecules count as retained negatives.", ""]
    mp = base / "eval" / "metrics.parquet"
    metrics = _parquet(mp)
    if metrics is None:
        out[-2:] = [f"Intergenic molecules: {_not_run(base, [mp])}"]
        return "\n".join(out)
    m = metrics[(metrics.cap == MEMBER_CAP) & (metrics.cutoffs == CUTOFFS) & (metrics.level == "molecule")]
    rows, notes = [], []
    for lib in libs:
        a = m[(m.lib == lib) & (m.stratum_kind == "all")]
        g = m[(m.lib == lib) & (m.stratum_kind == "region") & (m.stratum == "intergenic")]
        if a.empty or g.empty:
            continue
        pos, gpos, gneg = a.tp.iloc[0] + a.fn.iloc[0], g.tp.iloc[0] + g.fn.iloc[0], g.tn.iloc[0] + g.fp.iloc[0]
        rows.append([lib, _int(pos), _int(gpos), _int(g.tp.max()), _r((pos - gpos + g.tp.max()) / pos if pos else None),
                     _int(gneg), _int(g.fp.max())])
        neg = a.tn.iloc[0] + a.fp.iloc[0]
        if lib == sl and neg > gneg:
            notes.append(f"While a setting keeps every intergenic non-templated molecule, specificity >= {min_spec:g} over all "
                         f"{_int(neg)} non-templated units on {sl} is at least ({min_spec:g} x {_int(neg)} - {_int(gneg)}) / "
                         f"{_int(neg - gneg)} = {_r((min_spec * neg - gneg) / (neg - gneg))} over the {_int(neg - gneg)} genic "
                         "ones.")
    out += _table(["Library", "Templated units", "Intergenic templated", "Removed at most", "Sensitivity ceiling",
                   "Intergenic non-templated", "Removed at most"], rows)
    return "\n".join(_notes(out, notes))


def strata_section(base):
    base = Path(base)
    out = ["## 5. Metrics by A band and region", ""]
    p = base / "eval" / "metrics.parquet"
    metrics = _parquet(p)
    if metrics is None:
        return "\n".join(out + [_not_run(base, [p])])
    tags, note = _report_tags(base)
    out += [f"Source: `eval/metrics.parquet` (stage 17), molecule level, member cap {MEMBER_CAP:,} nt, {CUTOFFS} labels. "
            f"Specificity exists only in band {BANDS[0]} and sensitivity only in bands {BANDS[2]} and up; band {BANDS[1]} is "
            "ambiguous.", ""] + note
    m = metrics[(metrics.cap == MEMBER_CAP) & (metrics.cutoffs == CUTOFFS) & (metrics.level == "molecule")
                & metrics.tag.isin(tags)]
    libs = list(dict.fromkeys(m.lib))
    look = {(r.tag, r.lib, r.stratum_kind, r.stratum): r for r in m.itertuples()}

    def get(tag, lib, kind, stratum, metric):
        r = look.get((tag, lib, kind, stratum))
        return None if r is None else getattr(r, metric)

    present = [(tag, lib) for tag in tags for lib in libs if (tag, lib, "all", "all") in look]
    out += _table(["Setting", "Library", f"Spec {BANDS[0]}"] + [f"Sens {b}" for b in BANDS[2:]],
                  [[tag, lib, _r(get(tag, lib, "a_band", BANDS[0], "spec"))]
                   + [_r(get(tag, lib, "a_band", b, "sens")) for b in BANDS[2:]] for tag, lib in present])
    out += ["", "By region, sens / spec:", ""]
    out += _table(["Setting", "Library"] + list(REGION_NAMES),
                  [[tag, lib] + [f"{_r(get(tag, lib, 'region', g, 'sens'))} / {_r(get(tag, lib, 'region', g, 'spec'))}"
                                 if (tag, lib, "region", g) in look else "n/a" for g in REGION_NAMES] for tag, lib in present])
    out += ["", "Units per stratum (the same for every setting):", ""]
    rows = []
    for lib in libs:
        first = m[(m.lib == lib) & (m.tag == m[m.lib == lib].tag.iloc[0]) & m.stratum_kind.isin(["a_band", "region"])]
        rows += [[lib, r.stratum_kind, r.stratum, _int(r.tp + r.fn), _int(r.tn + r.fp)] for r in first.itertuples()]
    out += _table(["Library", "Stratum kind", "Stratum", "Templated units", "Non-templated units"], rows)
    return "\n".join(out)


def hexamer_section(base):
    base = Path(base)
    out = ["## 6. Label-free check: hexamer rate in removed and kept molecules", ""]
    p = base / "eval" / "hexamer_check.parquet"
    h = _parquet(p)
    if h is None:
        return "\n".join(out + [_not_run(base, [p])])
    out += [f"Source: `eval/hexamer_check.parquet` (stage 17), molecule level, member cap {MEMBER_CAP:,} nt. Hexamer: "
            "AATAAA/ATTAAA 5 to 35 nt upstream of the site.", ""]
    h = h[h.cap == MEMBER_CAP]
    out += _table(["Setting", "Library", "Label", "Removed", "Removed hexamer rate", "Kept", "Kept hexamer rate"],
                  [[r.tag, r.lib, LABEL_NAMES.get(r.label, r.label), _int(r.removed_n), _r(r.removed_hex_rate), _int(r.kept_n),
                    _r(r.kept_hex_rate)] for r in h.itertuples()])
    return "\n".join(out)


def spikein_section(base):
    base = Path(base)
    out = ["## 7. Spike-ins", "",
           "Source: `<lib>/spikein/truth/molecules_labeled.parquet` (stage 14), `eval/spikein_metrics.parquet` (stage 17). "
           "spike_label comes from the distance to the annotated 3' end; label from the A count after the site.", ""]
    data, notes = {}, []
    for lib in NUCLEI:
        p = base / lib / "spikein" / "truth" / "molecules_labeled.parquet"
        if p.exists():
            data[lib] = pd.read_parquet(p, columns=["label", "spike_label", "module"])
        else:
            notes.append(f"{lib}: {_not_run(base, [p])}")
    if data:
        def counts(m):
            return pd.crosstab(m.spike_label, m.label).reindex(index=list(SPIKE_LABELS), columns=list(LABELS), fill_value=0)

        def agreement(c):
            hit = c.loc["internal", "templated"] + c.loc["tail", "nontemplated"]
            n = c.loc[["internal", "tail"], ["templated", "nontemplated"]].to_numpy().sum()
            return f"{_r(hit / n if n else None)} ({_int(hit)} of {_int(n)})"

        rows, agree = [], []
        for lib, m in data.items():
            c = counts(m)
            rows += [[lib, k, _int(t.templated), _int(t.ambiguous), _int(t.nontemplated), _int(t.sum())] for k, t in c.iterrows()]
            agree.append(f"{lib}: agreement {agreement(c)}, SIRV_E0 only {agreement(counts(m[m.module == 'SIRV_E0']))}")
        out += _table(["Library", "Spike label", "templated", "ambiguous", "non-templated", "total"], rows)
        out += ["", "Agreement = (internal and templated + tail and non-templated) / internal or tail molecules labeled "
                    "templated or non-templated: " + "; ".join(agree) + ".", "",
                "- ERCC and long SIRV transcripts end at the contig end, so at their tail sites the 20 nt after the site are "
                f"truncated and a20 cannot reach {LABEL_CUTOFFS['templated_min']}: their non-templated (or ambiguous) calls are "
                "forced, and they lift the pooled agreement. Only SIRV_E0 tails have full windows.",
                "- A SIRV_E0 tail site can also be internal priming of a longer isoform that runs through a shorter isoform's "
                "annotated end; the annotation cannot separate the two.", ""]
        rows = []
        for lib, m in data.items():
            for module in ("all",) + MODULES:
                g = m if module == "all" else m[m.module == module]
                i, t = g[g.spike_label == "internal"], g[g.spike_label == "tail"]
                rows.append([lib, module, _int(len(i)), _share((i.label == "templated").sum(), len(i)), _int(len(t)),
                             _share((t.label == "nontemplated").sum(), len(t))])
        out += _table(["Library", "Module", "Internal", "Internal and templated", "Tail", "Tail and non-templated"], rows)
    out = _notes(out, notes)

    out += ["", f"Metrics per module (positives: internal, negatives: tail; molecule level, cap {MEMBER_CAP:,} nt), "
                "sens / spec:", ""]
    p = base / "eval" / "spikein_metrics.parquet"
    s = _parquet(p)
    if s is None:
        out[-2:] = [f"Metrics per module: {_not_run(base, [p])}"]
        return "\n".join(out)
    tags, note = _report_tags(base)
    out += note
    s = s[(s.level == "molecule") & (s.cap == MEMBER_CAP)]
    look = {(r.lib, r.tag, r.stratum): r for r in s.itertuples()}
    rows = []
    for lib in dict.fromkeys(s.lib):
        for tag in tags:
            if (lib, tag, "all") in look:
                rows.append([lib, tag] + [f"{_r(look[lib, tag, k].sens)} / {_r(look[lib, tag, k].spec)}"
                                          if (lib, tag, k) in look else "n/a" for k in ("all",) + MODULES])
    out += _table(["Library", "Setting", "all"] + list(MODULES), rows)
    return "\n".join(out)


def _flat(d):
    """Scalar values of a nested dict, keyed by their own name (the first occurrence wins)."""
    out = {}
    for k, v in d.items():
        for kk, vv in (_flat(v).items() if isinstance(v, dict) else [(k, v)]):
            out.setdefault(kk, vv)
    return out


def ercc_section(base):
    base = Path(base)
    out = ["## 8. ERCC count accuracy (provisional)", "",
           "Source: `<lib>/spikein/ercc.json` (stage 15): correlation of ERCC UMI counts with input concentration, "
           "unfiltered and per setting.", ""]
    entries, notes, flags = [], [], []
    for lib in NUCLEI:
        p = base / lib / "spikein" / "ercc.json"
        d = _json(p)
        if d is None:
            notes.append(f"{lib}: {_not_run(base, [p])}")
            continue
        if d.get("provisional"):
            flags.append(f"Provisional ({lib}): {d.get('note', '')}")
        src = d.get("settings", d)
        entries += [(lib, k, _flat(v)) for k, v in src.items() if isinstance(v, dict)]
    out += [f"- {f}" for f in flags] + ([""] if flags else [])
    if entries:
        cols = list(dict.fromkeys(k for _, _, f in entries for k in f))
        out += _table(["Library", "Setting"] + cols,
                      [[lib, k] + [_value(f[c]) if c in f else "n/a" for c in cols] for lib, k, f in entries])
    out = _notes(out, notes)
    return "\n".join(out)


def _per_read(sites, cutoffs):
    """(templated rate, non-templated rate, ratio) of the hexamer weighted by accepted reads, at the given cut-offs."""
    s = sites[sites.n_accepted >= 1]
    w = s.n_accepted.to_numpy(float)
    rate = []
    for m in (s.a20 >= cutoffs["templated_min"], s.a20 <= cutoffs["nontemplated_max"]):
        m = m.to_numpy()
        rate.append((w[m] * s.hexamer.to_numpy()[m]).sum() / w[m].sum() if w[m].sum() else None)
    return rate[0], rate[1], rate[1] / rate[0] if rate[0] and rate[1] is not None else None


def long_read_section(base):
    base = Path(base)
    out = ["## 9. Long reads (BD67, BD70)", "",
           "Source: `<lib>/long_read_acceptance.json` and `<lib>/long_read_sites.parquet` (stage 16). Hexamer rates per unique "
           "site with an accepted read (binding, spec 4.3: passes at ratio >= 3) and per accepted read.", ""]
    data, notes = {}, []
    for lib in LONG_READS:
        p = base / lib / "long_read_acceptance.json"
        d = _json(p)
        if d is None:
            notes.append(f"{lib}: {_not_run(base, [p])}")
        else:
            data[lib] = (d, _parquet(base / lib / "long_read_sites.parquet", ["n_accepted", "a20", "hexamer"]))
    if not data:
        return "\n".join(_notes(out, notes))
    rows = []
    for lib, (d, _) in data.items():
        rec, oc = d["records"], d["outcomes"]
        rows.append([lib, _int(len(d["refs"])), _int(rec["records"]), _int(rec["secondary_supplementary"]), _int(rec["low_mapq"]),
                     _int(rec["used"]), _int(oc["accepted"]), _int(oc["short_run"]), _int(oc["short_after"]), _int(oc["both"]),
                     _pct(d["dropped_fraction"], 1), _int(d["duplicate_qnames"])])
    out += _table(["Library", "References", "Records", "Secondary or supplementary", "MAPQ < 20", "Used", "Accepted", "Short run",
                   "Short after", "Both", "Dropped", "Duplicate qnames"], rows)
    out += ["", "- The spec expects about 7% of reads dropped. BD67 drops more in every build check (11.1% on chr21 and chr22, "
                "12.4% on chr1; BD70 7.7% and 9.1%), from more short-run and both-fail reads, so expect about 12% genome-wide; "
                "that is the library, not a failure of the acceptance rule."]
    out += ["", "Hexamer rates (sites T / NT / amb counted per unique site with an accepted read):", ""]
    rows = []
    for lib, (d, sites) in data.items():
        for key in ("cutoffs_8_4", "cutoffs_14_8"):
            h = d[key]
            c = h["cutoffs"]
            t, nt, ratio = _per_read(sites, c) if sites is not None else (None, None, None)
            rows.append([lib, f"{c['templated_min']}/{c['nontemplated_max']}",
                         " / ".join(_int(h["n"][k]) for k in ("templated", "nontemplated", "ambiguous")),
                         _r(h["hexamer_rate"]["templated"]), _r(h["hexamer_rate"]["nontemplated"]),
                         _r(h["hexamer_rate"]["ambiguous"]), _r(h["ratio"], 2), h["passes"], _r(t), _r(nt), _r(ratio, 2)])
    out += _table(["Library", "Cut-offs", "Sites T / NT / amb", "Rate T", "Rate NT", "Rate amb", "Ratio per site",
                   "Passes (ratio >= 3)", "Per-read rate T", "Per-read rate NT", "Ratio per read"], rows)
    out += ["", "Agreement with 1179 (1179 molecules on the references stage 16 processed; a long-read site within 10 nt on "
                "the same strand; agreement within 10 nt mostly measures co-location, the matched share per label is the "
                "informative number):", ""]
    rows = []
    for lib, (d, _) in data.items():
        v = d.get("vs_1179")
        if v is None:
            notes.append(f"{lib} vs 1179: not run (long_read_acceptance.json has no vs_1179)")
            continue
        by, ag = v.get("share_matched_by_label", {}), v["agreement"]
        rows.append([lib, _int(v["molecules"]), _r(v["share_matched"]),
                     " / ".join(_r(by.get(k)) for k in ("templated", "nontemplated", "ambiguous")),
                     " / ".join(_r(ag[k]) for k in ("templated", "nontemplated", "pooled"))])
    if rows:
        out += _table(["Library", "1179 molecules", "Matched share", "Matched share T / NT / amb", "Agreement T / NT / pooled"], rows)
    return "\n".join(_notes(out, notes))


def burden_section(base):
    base = Path(base)
    out = ["## 10. Burden: templated share of labeled molecules", ""]
    p = base / "burden" / "templated_share.parquet"
    share = _parquet(p)
    if share is None:
        return "\n".join(out + [_not_run(base, [p])])
    out += ["Source: `burden/*.parquet` (stage 18). Descriptive only: tissue, maturity, cell-type mix and the cell/nucleus "
            "difference are confounded. share = templated / all labeled molecules, 95% gene-bootstrap interval in brackets, "
            "molecules in parentheses; resolved = templated / (templated + non-templated).", ""]
    groups = list(dict.fromkeys(share.group))
    look = {(r.group, r.stratum_kind, r.stratum): r for r in share.itertuples()}
    strata = [("all", "all")] + [("region", g) for g in REGION_NAMES] + [("has_ts", "True"), ("has_ts", "False")]

    def cell(g, k, s, f):
        r = look.get((g, k, s))
        return "n/a" if r is None else f(r)

    out += _table(["Stratum kind", "Stratum"] + groups,
                  [[k, s] + [cell(g, k, s, lambda r: f"{_ci(r.share_all, r.lo, r.hi)} ({_int(r.total)})") for g in groups]
                   for k, s in strata])
    out += ["", "Resolved share:", ""]
    out += _table(["Stratum kind", "Stratum"] + groups,
                  [[k, s] + [cell(g, k, s, lambda r: _r(r.share_resolved)) for g in groups] for k, s in strata])
    out += ["", "Distribution over A bands (share of labeled molecules in the band, 95% gene-bootstrap interval in brackets; on "
                "these rows stage 18's lo and hi bound this share, not share_all, because the labels are defined by the band "
                "and fix share_all within it):", ""]
    out += _table(["A band"] + groups, [[b] + [cell(g, "a_band", b, lambda r: _ci(r.frac, r.lo, r.hi)) for g in groups]
                                        for b in BANDS])

    no_pair = "(stage 18 needs 1417 and 1419, or nuclei and cells)"
    p = base / "burden" / "same_site.parquet"
    ss = _parquet(p)
    if ss is None:
        out += ["", f"Same site: {_not_run(base, [p])}"]
    elif ss.empty:
        out += ["", f"Same site: no pair in {_rel(base, p)} {no_pair}"]
    else:
        out += ["", "Same site (each site of the first group matched to the nearest site of the second within 5 nt on the same "
                    "strand; site label = most common label of its molecules):", ""]
        c = ss.pivot_table(index=["pair", "label_a"], columns="label_b", values="n", aggfunc="sum", fill_value=0)
        rows = pd.MultiIndex.from_tuples([(p, k) for p in dict.fromkeys(ss.pair) for k in LABELS if (p, k) in c.index])
        c = c.reindex(index=rows, columns=list(LABELS) + ["none"], fill_value=0)
        out += _table(["Pair", "Label in first", "templated", "ambiguous", "non-templated", "none"],
                      [[pair, LABEL_NAMES.get(la, la)] + [_int(v) for v in t] for (pair, la), t in c.iterrows()])
        rows = []
        for pair, g in ss.groupby("pair", sort=False):
            n, matched = g.n.sum(), g[g.label_b != "none"]
            same = matched[matched.label_a == matched.label_b].n.sum()
            rows.append([pair, _int(n), f"{_int(matched.n.sum())} ({_pct(matched.n.sum(), n)})",
                         f"{_int(same)} ({_pct(same, matched.n.sum())})"])
        out += [""] + _table(["Pair", "Sites in first", "Matched", "Same label (of matched)"], rows)
    p = base / "burden" / "same_gene.parquet"
    sg = _parquet(p)
    if sg is None:
        out += ["", f"Same gene: {_not_run(base, [p])}"]
    elif sg.empty:
        out += ["", f"Same gene: no pair in {_rel(base, p)} {no_pair}"]
    else:
        out += ["", "Same gene (Spearman correlation of the templated share over genes with enough molecules in both):", ""]
        out += _table(["Pair", "Genes", "Spearman rho", "p"],
                      [[r.pair, _int(r.n_genes), _r(r.spearman_rho), "n/a" if pd.isna(r.p_value) else f"{r.p_value:.2e}"]
                       for r in sg.itertuples()])
    p = base / "burden" / "selection_bias.parquet"
    sb = _parquet(p)
    if sb is None:
        out += ["", f"Selection bias: {_not_run(base, [p])}"]
    else:
        out += ["", "Selection bias (share of labeled molecules versus share of primary records, by RE tag):", ""]
        out += _table(["Library", "RE", "Labeled share", "Records share"],
                      [[r.lib, r.re, _r(r.labeled_share), _r(r.records_share)] for r in sb.itertuples()])
    p = base / "burden" / "polyasite.parquet"
    px = _parquet(p)
    if px is None:
        out += ["", f"PolyASite proximity: {_not_run(base, [p])}"]
    else:
        out += ["", "PolyASite 3.0 proximity (share of labeled molecules by the distance in nt from the site to the nearest atlas "
                    "cleavage site on the same strand; templated molecules also by A band). Proximity is reported, not used to "
                    "validate the labels: the atlas is built from 10x 3' single-cell reads, oligo(dT) primed like these "
                    "libraries, and does not screen out candidate sites with A-rich genome downstream, so internally primed "
                    "sites can be in it. The null row takes every molecule of the group with its site moved 1,000 nt upstream "
                    "in transcript sense on the same ref and strand; its share within 10 nt is the share expected by chance "
                    "at this atlas density."]
        for g, d in px.groupby("group", sort=False):
            n = d.groupby(["label", "a_band", "bin"]).n.sum()
            keys = ([(lab, "all") for lab in LABELS] + [(NULL_LABEL, "all")]
                    + [("templated", b) for b in BANDS if ("templated", b) in n.index])
            rows = []
            for lab, band in keys:
                c = n.loc[(lab, band)].reindex(list(BINS), fill_value=0)
                name = ("null (sites 1 kb upstream)" if lab == NULL_LABEL else LABEL_NAMES[lab] if band == "all"
                        else f"templated, A band {band}")
                rows.append([name, _int(c.sum())] + [_r(x / c.sum() if c.sum() else None) for x in c])
            out += ["", f"Group {g}:", ""] + _table(["Label", "Molecules"] + list(BINS), rows)
    return "\n".join(out)


def validation_section(base):
    base = Path(base)
    out = ["## 11. Validation against the original tool", ""]
    p = base / "validation" / "exact_match_m2.json"
    em = _json(p)
    if em is None:
        return "\n".join(out + [_not_run(base, [p])])
    out += ["Source: `validation/exact_match_m2.json` (stage 20). Removed-record sets (every record the original fetches, "
            "secondary and supplementary included) of the original polyAfilter and the reimplementation. Max RSS is the "
            "largest single process; peak tree RSS the memwatch sum.", ""]
    rows, notes = [], []
    for lib in LIBS:
        d, sub = em.get(lib), em.get(f"{lib}_subset")
        if d is None and sub is None:
            notes.append(f"{lib}: not in exact_match_m2.json" + ("; the 1179 grid has no original-versus-reimplementation "
                                                                 "evidence" if lib in CELLS else ""))
        elif d is not None and not d:
            notes.append(f"{lib}: no completed original run")
        if sub is not None:
            notes.append(f"{lib}: original tool on the {', '.join(sub['refs'])} subset only ({_int(sub['n_records'])} records)")
        runs = [(lib, d or {})] + ([(f"{lib} ({', '.join(sub['refs'])})", sub["settings"])] if sub is not None else [])
        for name, by_tag in runs:
            for tag, r in by_tag.items():
                rows.append([name, tag, _int(r["n_original"]), _int(r["n_reimpl"]), _int(r["only_original"]),
                             _int(r["only_reimpl"]), r["identical"],
                             _r(r["original_elapsed_s"] / 3600 if r["original_elapsed_s"] is not None else None, 2),
                             _r(r["original_max_rss_gib"], 2), _r(r["original_peak_tree_rss_gib"], 2)])
    if rows:
        out += _table(["Library", "Setting", "Original removed", "Reimpl removed", "Only original", "Only reimpl", "Identical",
                       "Original wall (h)", "Max RSS (GiB)", "Peak tree RSS (GiB)"], rows)
    out = _notes(out, notes)
    nc = em.get("none_control")
    out += ["", "Remove-all control (MINSNRLEN none) on a reference subset, original run in process:", ""]
    if nc is None:
        out[-2:] = [f"Remove-all control: not run (no none_control entry in {_rel(base, p)})"]
    else:
        ct = nc["createTRANS"]
        out += _table(["Setting", "References", "Records", "Original removed", "Reimpl removed", "Only original", "Only reimpl",
                       "Identical", "createTRANS wall (s)", "createTRANS max RSS (GiB)", "Total (s)"],
                      [[nc["setting"], ", ".join(nc["refs"]), _int(nc["n_records"]), _int(nc["n_original"]), _int(nc["n_reimpl"]),
                        _int(nc["only_original"]), _int(nc["only_reimpl"]), nc["identical"], _sec(ct["elapsed_s"]),
                        _r(ct["max_rss_gib"], 2), _sec(sum(nc["seconds"].values()))]])
    out += ["", f"Compared: {em.get('n_compared', 'n/a')}; all identical: {em.get('all_identical', 'n/a')}"]
    return "\n".join(out)


SECTIONS = (runs_section, truth_section, grid_section, selection_section, strata_section, hexamer_section, spikein_section,
            ercc_section, long_read_section, burden_section, validation_section)


def render(base):
    head = ["# Milestone 2 results (generated by scripts/21_m2_summary.py)", "",
            f"Inputs under `{base}`. Each section names the files its numbers come from; \"not run\" lists the files that "
            "are absent.", ""]
    return "\n".join(head) + "\n" + "\n\n".join(f(base) for f in SECTIONS) + "\n"
```

- [ ] **Step 4: Create `scripts/21_m2_summary.py`**

```python
"""Stage 21: collect milestone 2 numbers into docs/m2_results.md (sections from pafbench.m2report)."""
import argparse
import json
from pathlib import Path

from pafbench.config import OUT, REPO
from pafbench.m2report import SECTIONS, render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=OUT)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "m2_results.md")
    args = ap.parse_args()
    text = render(args.base)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(json.dumps({"out": str(args.out), "sections": len(SECTIONS), "not_run": text.count("not run")}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `237 passed, 2 warnings`.

- [ ] **Step 6: Render and read the results**

Run: `.venv/bin/python -m pafbench.memwatch /data/element_runs/polyafilter_bench/runs/21_m2_summary.mem.json -- .venv/bin/python scripts/21_m2_summary.py`
Expected: `docs/m2_results.md` with no "not run" section. Read it against the JSON files before committing; the selection outcome, the held-out numbers and the burden comparison go to the user before milestone 3 starts.

- [ ] **Step 7: Commit the results page**

```bash
git add docs/m2_results.md
git -c user.name="Simone Picelli" -c user.email="simone.picelli@iob.ch" commit -m "Milestone 2 results"
```

- [ ] **Step 8: Commit**

```bash
git add tests/test_m2report.py pafbench/m2report.py scripts/21_m2_summary.py
git -c user.name="Simone Picelli" -c user.email="simone.picelli@iob.ch" commit -m "Stage 21: milestone 2 results page"
```

---

## After the plan

Milestone 3 (not in this plan): the parts of spec 4.1, 4.3 and 8 deferred on 2026-10-10 (labeled against unlabeled molecules by gene, distance to the annotated 3' end and cell type; whole-isoform context of long-read ends; site sequence, hexamer and distance at matched sites), per-cell UMI matrices after filtering from `grid/cell_counts.parquet` and a BAM recount that reproduces the Cell Ranger matrix, clustering, cell types from retinal marker genes, and the final report as an Artifact page.
