# polyAfilter benchmark, milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** De-risk the benchmark: run the original polyAfilter on 1417, build a reimplementation that reproduces its removed-read sets exactly, build the `pa` molecule truth set for 1417 with a fixed member rule and fixed label cut-offs, check the 3k-only scope against 1416, confirm spike-in tags survive realignment, and confirm how BD67 long reads mark priming sites.

**Architecture:** A small Python package `pafbench` holds tested logic; numbered stage scripts in `scripts/` run it on real data and write Parquet/JSON under `/data/element_runs/polyafilter_bench/`. The reimplementation imports polyAfilter's own interval and transcript-end functions from a pinned, unmodified checkout and replaces only the slow parts (coverage, A-stretch scan, read overlap) with numpy/numba code that is tested against the original on synthetic data and on the full 1417 BAM.

**Tech Stack:** Python 3.11 (uv venv), pysam, numpy, numba, pandas, pyarrow, gffutils, biopython, dill, pytest; STAR 2.7.10a; samtools; GNU time.

**Spec:** `docs/specs/2026-10-07-polyafilter-bench-design.md` (milestone 1 is its section 10).

## Global Constraints

- polyAfilter pinned at commit `5ce0098`, cloned to `vendor/polyAfilter`, never edited.
- Reference: `/home/simone.picelli/Documents/Genome_ref/refdata-gex-GRCh38-2024-A` (`fasta/genome.fa`, `genes/genes.gtf`).
- Libraries: 1417 (`/data/element_runs/1417_HsRetina_D28_3k/outs/possorted_genome_bam.bam`), 1419, 1179; 1416 chr1 only for the scope check; BD67 `/data/element_runs/BD67/BD67_2x_RT.fltnc.aligned.bam`.
- Code in `~/polyafilter_bench`; results in `/data/element_runs/polyafilter_bench/<library>/`.
- Python packages only through `uv` in `~/polyafilter_bench/.venv`; never `pip install --break-system-packages`.
- Exact-match acceptance: the reimplementation's removed-record set equals the original's, record for record, for every setting run.
- The reimplementation considers every alignment record the original fetches: secondary, supplementary, duplicate-flagged and QC-fail records included.
- MINSNRLEN semantics: `None` means "remove all non-terminal reads". The CLI parses `0` as the integer 0, which is NOT `None` (it means A-stretches of length >= 0). Grid "0" is implemented as `None` and labelled `none`.
- Always pass `tROC={}` when calling `scanBAMfilter` in-process: its default argument is a mutable dict that caches transcript starts across calls.
- Starting label cut-offs: templated if at least 14 A in the 20 nt after the site, non-templated if at most 8. Hexamers: AATAAA or ATTAAA inside the 5 to 35 nt window upstream of the site.
- Deletion uses `trash`, never `rm -rf`. No em dashes in any document.

## Review Focus

- Secondary alignments stored without SEQ (`*`): they add no coverage in the original (pysam `count_coverage` counts bases), but they can still be removed. Test in Task 4 and Task 5 (synthetic fixture has them).
- Reads with an `N` base: `count_coverage` does not count `N`. Test in Task 4 (fixture has them).
- A-stretches touching a contig start or end: windows of length zero are skipped by the original. Test in Task 3 (random sequences start and end with runs) and Task 5 (fixture plants runs at both ends of chrA).
- Soft-masked (lowercase) FASTA bases: the original counts lowercase `a`/`t` as stretch bases. Test in Task 3 and Task 5 (fixture has `aaaaaaaaaa`).
- Transcripts shorter than COVLEN in intron mode: the whole transcript is protected. Test in Task 4 and Task 5 (fixture transcript t5 is 80 nt).
- `pa` reads without CB or UB: excluded from molecules. Test in Task 8.

---

### Task 0: Repository scaffolding, environment, pinned polyAfilter

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `scripts/00_setup.sh`, `pafbench/__init__.py`, `pafbench/config.py`, `pafbench/vendor.py`, `tests/__init__.py`, `tests/test_setup.py`

**Interfaces:**
- Produces: `pafbench.config` constants `REPO, VENDOR_POLYAFILTER, POLYAFILTER_COMMIT, REF_DIR, FASTA, GTF, RUNS, OUT, LIBRARIES, BD67_BAM`; `pafbench.vendor.polyafilter_modules() -> (BAMfilter, RNAseqAnalysis, SNRanalysis)`.

- [ ] **Step 1: Write `.gitignore` and `pyproject.toml`**

`.gitignore`:
```
.venv/
vendor/
__pycache__/
*.egg-info/
.pytest_cache/
```

`pyproject.toml`:
```toml
[project]
name = "pafbench"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["pysam", "numpy", "numba", "pandas", "pyarrow", "gffutils", "biopython", "dill", "pytest"]

[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["pafbench"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `scripts/00_setup.sh` and run it**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e .
if [ ! -d vendor/polyAfilter ]; then
  mkdir -p vendor
  git clone https://github.com/MarekSvob/polyAfilter.git vendor/polyAfilter
fi
git -C vendor/polyAfilter checkout --quiet 5ce0098
git -C vendor/polyAfilter rev-parse --short=7 HEAD
```

Run: `bash scripts/00_setup.sh`
Expected: last line prints `5ce0098`.

- [ ] **Step 3: Write `pafbench/config.py`, `pafbench/vendor.py`, empty `pafbench/__init__.py` and `tests/__init__.py`**

`pafbench/config.py`:
```python
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR_POLYAFILTER = REPO / "vendor" / "polyAfilter"
POLYAFILTER_COMMIT = "5ce0098"
REF_DIR = Path("/home/simone.picelli/Documents/Genome_ref/refdata-gex-GRCh38-2024-A")
FASTA = REF_DIR / "fasta" / "genome.fa"
GTF = REF_DIR / "genes" / "genes.gtf"
RUNS = Path("/data/element_runs")
OUT = RUNS / "polyafilter_bench"
LIBRARIES = {
    "1417": RUNS / "1417_HsRetina_D28_3k" / "outs" / "possorted_genome_bam.bam",
    "1419": RUNS / "1419_HsRetina_D29_3k" / "outs" / "possorted_genome_bam.bam",
    "1179": RUNS / "1179_organoid_wk26_10xv4" / "outs" / "possorted_genome_bam.bam",
    "1416": RUNS / "1416_HsRetina_D28_20k" / "outs" / "possorted_genome_bam.bam",
}
BD67_BAM = RUNS / "BD67" / "BD67_2x_RT.fltnc.aligned.bam"
```

`pafbench/vendor.py`:
```python
import sys

from .config import VENDOR_POLYAFILTER


def polyafilter_modules():
    """Import the pinned, unmodified polyAfilter modules."""
    path = str(VENDOR_POLYAFILTER)
    if path not in sys.path:
        sys.path.insert(0, path)
    import BAMfilter
    import RNAseqAnalysis
    import SNRanalysis
    return BAMfilter, RNAseqAnalysis, SNRanalysis
```

- [ ] **Step 4: Write `tests/test_setup.py`**

```python
import subprocess

from pafbench.config import FASTA, GTF, LIBRARIES, POLYAFILTER_COMMIT, VENDOR_POLYAFILTER
from pafbench.vendor import polyafilter_modules


def test_vendor_commit():
    head = subprocess.run(
        ["git", "-C", str(VENDOR_POLYAFILTER), "rev-parse", "--short=7", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == POLYAFILTER_COMMIT


def test_vendor_imports():
    BAMfilter, RNAseqAnalysis, SNRanalysis = polyafilter_modules()
    assert hasattr(BAMfilter, "scanBAMfilter") and hasattr(BAMfilter, "findSNRpieces")
    assert hasattr(RNAseqAnalysis, "getTransStartEnding") and hasattr(RNAseqAnalysis, "getTransEnding")
    assert hasattr(SNRanalysis, "getOverlaps") and hasattr(SNRanalysis, "removeOverlaps")


def test_inputs_exist():
    assert FASTA.exists() and GTF.exists()
    for name in ("1417", "1419", "1179", "1416"):
        assert LIBRARIES[name].exists(), name
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_setup.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add .gitignore pyproject.toml scripts/00_setup.sh pafbench tests
git commit -m "Scaffolding: package, uv env, pinned polyAfilter 5ce0098"
```

---

### Task 1: Synthetic fixture, original tool on it, and original runs on 1417

**Files:**
- Create: `tests/synth.py`, `tests/conftest.py`, `tests/test_synth_original.py`, `pafbench/flags.py` (only `record_key`, `removed_keys_by_ref`, `removed_keys` in this task), `scripts/01_build_db.sh`, `scripts/02_run_original.sh`

**Interfaces:**
- Consumes: `polyafilter_modules()`.
- Produces: `tests.synth.build(tmp: Path) -> Synth` with fields `tmp, fasta, gtf, db, bam`; `tests.synth.run_original(synth, covlen: int, min_len: int | None, mism: int, introns: bool) -> set[tuple]`; `pafbench.flags.record_key(a) -> (qname, flag, ref, pos, cigar)`; `removed_keys_by_ref(in_bam, out_bam) -> Iterator[(ref, set)]`; `removed_keys(in_bam, out_bam) -> set`.

- [ ] **Step 1: Write `pafbench/flags.py` (first part)**

```python
import pysam


def record_key(a):
    return (a.query_name, a.flag, a.reference_name, a.reference_start, a.cigarstring)


def removed_keys_by_ref(in_bam, out_bam):
    """Yield (ref, removed record keys) per reference. out_bam must keep an ordered subsequence of in_bam's records."""
    with pysam.AlignmentFile(str(in_bam)) as fin, pysam.AlignmentFile(str(out_bam)) as fout:
        it = fout.fetch(until_eof=True)
        nxt = next(it, None)
        cur_ref, cur = None, set()
        first = True
        for a in fin.fetch(until_eof=True):
            if first or a.reference_name != cur_ref:
                if not first:
                    yield cur_ref, cur
                cur_ref, cur, first = a.reference_name, set(), False
            if nxt is not None and record_key(a) == record_key(nxt):
                nxt = next(it, None)
            else:
                cur.add(record_key(a))
        if not first:
            yield cur_ref, cur
        if nxt is not None:
            raise ValueError("output BAM is not an ordered subsequence of the input BAM")


def removed_keys(in_bam, out_bam):
    out = set()
    for _, keys in removed_keys_by_ref(in_bam, out_bam):
        out |= keys
    return out
```

- [ ] **Step 2: Write `tests/synth.py`**

```python
"""Small synthetic genome, annotation and BAM for exactness tests against the original polyAfilter."""
import random
from dataclasses import dataclass
from pathlib import Path

import gffutils
import pysam

from pafbench.flags import removed_keys
from pafbench.vendor import polyafilter_modules

CHROMS = {"chrA": 6000, "chrB": 2000}

GTF_ROWS = [
    ("chrA", "gene", 1001, 4000, "+", 'gene_id "g1";'),
    ("chrA", "transcript", 1001, 4000, "+", 'gene_id "g1"; transcript_id "t1";'),
    ("chrA", "exon", 1001, 1400, "+", 'gene_id "g1"; transcript_id "t1";'),
    ("chrA", "exon", 1801, 2600, "+", 'gene_id "g1"; transcript_id "t1";'),
    ("chrA", "exon", 3001, 4000, "+", 'gene_id "g1"; transcript_id "t1";'),
    ("chrA", "transcript", 1001, 2700, "+", 'gene_id "g1"; transcript_id "t2";'),
    ("chrA", "exon", 1001, 1400, "+", 'gene_id "g1"; transcript_id "t2";'),
    ("chrA", "exon", 1801, 2700, "+", 'gene_id "g1"; transcript_id "t2";'),
    ("chrA", "gene", 4201, 5800, "-", 'gene_id "g2";'),
    ("chrA", "transcript", 4201, 5800, "-", 'gene_id "g2"; transcript_id "t3";'),
    ("chrA", "exon", 4201, 4700, "-", 'gene_id "g2"; transcript_id "t3";'),
    ("chrA", "exon", 5001, 5800, "-", 'gene_id "g2"; transcript_id "t3";'),
    ("chrB", "gene", 201, 1500, "+", 'gene_id "g3";'),
    ("chrB", "transcript", 201, 1500, "+", 'gene_id "g3"; transcript_id "t4";'),
    ("chrB", "exon", 201, 1500, "+", 'gene_id "g3"; transcript_id "t4";'),
    ("chrB", "gene", 1601, 1680, "-", 'gene_id "g4";'),
    ("chrB", "transcript", 1601, 1680, "-", 'gene_id "g4"; transcript_id "t5";'),
    ("chrB", "exon", 1601, 1680, "-", 'gene_id "g4"; transcript_id "t5";'),
]

# (contig, 0-based position, sequence) planted over the random background
PLANTS = [
    ("chrA", 0, "AAAAAA"),
    ("chrA", 1299, "A" * 12),
    ("chrA", 1599, "A" * 10),
    ("chrA", 1999, "AAAAAGAAAAAA"),
    ("chrA", 2299, "aaaaaaaaaa"),
    ("chrA", 2549, "A" * 8),
    ("chrA", 3199, "NNNNN"),
    ("chrA", 3499, "A" * 15),
    ("chrA", 4499, "T" * 10),
    ("chrA", 5299, "T" * 12),
    ("chrA", 5994, "TTTTTT"),
    ("chrB", 1640, "T" * 9),
]


@dataclass
class Synth:
    tmp: Path
    fasta: Path
    gtf: Path
    db: Path
    bam: Path


def _reads(rng, hdr):
    reads = []

    def add(ref, pos, cigar, rev, length=90, extra=0, seq=True, n_base=False):
        a = pysam.AlignedSegment(hdr)
        a.query_name = f"r{len(reads)}"
        a.reference_name = ref
        a.reference_start = pos
        a.cigarstring = cigar
        a.flag = (16 if rev else 0) | extra
        a.mapping_quality = 255
        if seq:
            s = "".join(rng.choice("ACGT") for _ in range(length))
            if n_base:
                s = s[:10] + "N" + s[11:]
            a.query_sequence = s
            a.query_qualities = pysam.qualitystring_to_array("I" * length)
        reads.append(a)

    for pos in range(950, 4000, 37):
        add("chrA", pos, "90M", False)
    for pos in range(1340, 1390, 10):
        left = 1400 - pos
        add("chrA", pos, f"{left}M400N{90 - left}M", False)
    for pos in range(4150, 5800, 41):
        add("chrA", pos, "90M", True)
    for pos in range(1200, 3900, 300):
        add("chrA", pos, "90M", True)
    for pos in range(1250, 1550, 60):
        add("chrA", pos, "90M", False, extra=256, seq=False)
    for pos in range(3400, 3600, 50):
        add("chrA", pos, "90M", False, extra=1024)
    for pos in range(2900, 3100, 50):
        add("chrA", pos, "90M", False, n_base=True)
    add("chrA", 1950, "40M2I48M", False)
    add("chrA", 2450, "40M3D50M", False)
    for pos in range(1590, 1660, 20):
        add("chrB", pos, "60M", True, length=60)
    return reads


def build(tmp: Path, seed: int = 7) -> Synth:
    rng = random.Random(seed)
    seqs = {c: [rng.choice("ACGT") for _ in range(n)] for c, n in CHROMS.items()}
    for c, p, s in PLANTS:
        seqs[c][p:p + len(s)] = list(s)
    fasta = tmp / "genome.fa"
    with open(fasta, "w") as fh:
        for c, s in seqs.items():
            s = "".join(s)
            fh.write(f">{c}\n")
            for k in range(0, len(s), 60):
                fh.write(s[k:k + 60] + "\n")
    pysam.faidx(str(fasta))
    gtf = tmp / "genes.gtf"
    with open(gtf, "w") as fh:
        for c, ft, s, e, strand, attr in GTF_ROWS:
            fh.write(f"{c}\tsynth\t{ft}\t{s}\t{e}\t.\t{strand}\t.\t{attr}\n")
    db = tmp / "genes.db"
    gffutils.create_db(str(gtf), str(db), merge_strategy="create_unique")
    hdr = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": c, "LN": n} for c, n in CHROMS.items()]}
    )
    unsorted = tmp / "unsorted.bam"
    with pysam.AlignmentFile(str(unsorted), "wb", header=hdr) as out:
        for a in _reads(rng, hdr):
            out.write(a)
    bam = tmp / "reads.bam"
    pysam.sort("-o", str(bam), str(unsorted))
    pysam.index(str(bam))
    return Synth(tmp, fasta, gtf, db, bam)


def run_original(synth: Synth, covlen, min_len, mism, introns):
    """Removed record keys from the original scanBAMfilter (serial mode, fresh tROC)."""
    BAMfilter, _, _ = polyafilter_modules()
    tag = f"{covlen}_{min_len}_{mism}_{int(introns)}"
    trans = synth.tmp / f"trans_{int(introns)}.pkl"
    out = synth.tmp / f"orig_{tag}.bam"
    BAMfilter.scanBAMfilter(
        covlen, min_len, str(synth.bam), str(synth.fasta), str(trans),
        out_db=str(synth.db), mism=mism, out_bamfile=str(out), tROC={}, includeIntrons=introns,
    )
    return removed_keys(synth.bam, out)
```

- [ ] **Step 3: Write `tests/conftest.py`**

```python
import pytest

from tests.synth import build


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    return build(tmp_path_factory.mktemp("synth"))
```

- [ ] **Step 4: Write `tests/test_synth_original.py`**

```python
from tests.synth import run_original


def test_original_runs_and_removes_something(synth):
    removed = run_original(synth, 300, 8, 1, True)
    assert len(removed) > 0
    # antisense reads in g1 start at 1200, 1500, ... and are never removed
    assert not any(k[1] & 16 and k[2] == "chrA" and 1200 <= k[3] < 3900 and (k[3] - 1200) % 300 == 0 for k in removed)


def test_removed_keys_detects_subsequence(synth):
    removed = run_original(synth, 100, 5, 0, False)
    assert all(len(k) == 5 for k in removed)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_synth_original.py -v`
Expected: 2 passed. If the original fails on numpy 2 or pysam 0.24, read the traceback, pin the failing package to the version the polyAfilter README lists (`uv pip install --python .venv/bin/python 'numpy<2'` etc.), rerun, and record the pin in `pyproject.toml`.

- [ ] **Step 6: Write `scripts/01_build_db.sh` and `scripts/02_run_original.sh`**

`scripts/01_build_db.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=/data/element_runs/polyafilter_bench/ref
mkdir -p "$OUT"
cd "$REPO/vendor/polyAfilter"
/usr/bin/time -v -o "$OUT/createDB.time.txt" "$REPO/.venv/bin/python" polyAfilter.py createDB -v \
  /home/simone.picelli/Documents/Genome_ref/refdata-gex-GRCh38-2024-A/genes/genes.gtf \
  "$OUT/genes_2024A.db" > "$OUT/createDB.log" 2>&1
```

`scripts/02_run_original.sh`:
```bash
#!/usr/bin/env bash
# usage: 02_run_original.sh LIB COVLEN MINSNRLEN MISM MODE   (MODE: i = introns, e = exons)
set -euo pipefail
LIB=$1; COV=$2; L=$3; M=$4; MODE=$5
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY="$REPO/.venv/bin/python"
BAM=$("$PY" -c "from pafbench.config import LIBRARIES; print(LIBRARIES['$LIB'])")
OUT=/data/element_runs/polyafilter_bench/$LIB/original
mkdir -p "$OUT"
DB=/data/element_runs/polyafilter_bench/ref/genes_2024A.db
FA=/home/simone.picelli/Documents/Genome_ref/refdata-gex-GRCh38-2024-A/fasta/genome.fa
FLAG=""; [ "$MODE" = i ] && FLAG="-i"
TRANS=$OUT/trans_$MODE.pkl
TAG=c${COV}_l${L}_m${M}_${MODE}
cd "$REPO/vendor/polyAfilter"
if [ ! -f "$TRANS" ]; then
  /usr/bin/time -v -o "$OUT/createTRANS_$MODE.time.txt" "$PY" polyAfilter.py createTRANS $FLAG "$DB" "$BAM" "$TRANS" > "$OUT/createTRANS_$MODE.log" 2>&1
fi
/usr/bin/time -v -o "$OUT/$TAG.time.txt" "$PY" polyAfilter.py BAMfilter -v -p 8 -m "$M" $FLAG -o "$OUT/$TAG.bam" "$COV" "$L" "$BAM" "$FA" "$TRANS" > "$OUT/$TAG.log" 2>&1
```

- [ ] **Step 7: Start the database build and the three original runs in the background (sequential chain)**

```bash
chmod +x scripts/01_build_db.sh scripts/02_run_original.sh
nohup bash -c 'scripts/01_build_db.sh && scripts/02_run_original.sh 1417 300 10 1 i && scripts/02_run_original.sh 1417 375 3 2 i && scripts/02_run_original.sh 1417 300 10 1 e' \
  > /data/element_runs/polyafilter_bench/original_chain_1417.log 2>&1 &
```
Expected: `ls /data/element_runs/polyafilter_bench/ref` shows `genes_2024A.db` growing. The chain can take many hours; continue with Tasks 2 to 5 meanwhile. If a run is killed for memory (`Exit status: 137` or `Command terminated by signal 9` in its `.time.txt`), record it, then rerun that setting on a chr21+chr22 subset BAM (`samtools view -b -o sub.bam BAM chr21 chr22 && samtools index sub.bam`) and use the subset in Task 6.

- [ ] **Step 8: Commit**

```bash
git add pafbench/flags.py tests/synth.py tests/conftest.py tests/test_synth_original.py scripts/01_build_db.sh scripts/02_run_original.sh
git commit -m "Synthetic fixture, original polyAfilter wrapper, runs on 1417"
```

---

### Task 2: Interval primitives matching polyAfilter

**Files:**
- Create: `pafbench/intervals.py`, `tests/test_intervals.py`

**Interfaces:**
- Produces: `flatten(starts, ends) -> (ndarray, ndarray)`; `intersect(a_s, a_e, b_s, b_e) -> (ndarray, ndarray)`; `blocks_hit(block_s, block_e, piece_s, piece_e) -> ndarray[bool]`. All int64, 0-based half-open.

- [ ] **Step 1: Write the failing tests**

```python
import random

import numpy as np

from pafbench.intervals import blocks_hit, flatten, intersect
from pafbench.vendor import polyafilter_modules

_, _, SNR = polyafilter_modules()


def _random_intervals(rng, n, span=2000, maxlen=60):
    out = []
    for _ in range(n):
        s = rng.randrange(0, span)
        out.append((s, s + rng.randrange(1, maxlen)))
    return out


def test_flatten_matches_original():
    rng = random.Random(1)
    for _ in range(300):
        iv = _random_intervals(rng, rng.randrange(0, 50))
        want = SNR.flattenIntervals(list(iv))
        s, e = flatten([a for a, _ in iv], [b for _, b in iv])
        assert list(zip(s.tolist(), e.tolist())) == want


def test_flatten_merges_touching():
    s, e = flatten([0, 10], [10, 20])
    assert (s.tolist(), e.tolist()) == ([0], [20])


def test_intersect_matches_original():
    rng = random.Random(2)
    for _ in range(300):
        a = SNR.flattenIntervals(_random_intervals(rng, rng.randrange(0, 30)))
        b = SNR.flattenIntervals(_random_intervals(rng, rng.randrange(0, 30)))
        want = sorted(SNR.getOverlaps(list(a), list(b)))
        s, e = intersect(
            np.array([x for x, _ in a], np.int64), np.array([y for _, y in a], np.int64),
            np.array([x for x, _ in b], np.int64), np.array([y for _, y in b], np.int64),
        )
        assert list(zip(s.tolist(), e.tolist())) == want


def test_blocks_hit_matches_original():
    rng = random.Random(3)
    for _ in range(300):
        pieces = SNR.flattenIntervals(_random_intervals(rng, rng.randrange(0, 20)))
        ps = np.array([x for x, _ in pieces], np.int64)
        pe = np.array([y for _, y in pieces], np.int64)
        blocks = _random_intervals(rng, 40)
        got = blocks_hit(np.array([x for x, _ in blocks]), np.array([y for _, y in blocks]), ps, pe)
        want = [any(SNR.getOverlaps([blk], [p]) for p in pieces) for blk in blocks]
        assert got.tolist() == want


def test_adjacent_is_not_a_hit():
    assert blocks_hit(np.array([0]), np.array([10]), np.array([10]), np.array([20])).tolist() == [False]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_intervals.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pafbench.intervals'`.

- [ ] **Step 3: Write `pafbench/intervals.py`**

```python
import numpy as np


def flatten(starts, ends):
    """Merge intervals as polyAfilter's flattenIntervals does (touching intervals merge)."""
    s = np.asarray(starts, dtype=np.int64)
    e = np.asarray(ends, dtype=np.int64)
    if s.size == 0:
        return s, e
    if np.any(s >= e):
        raise ValueError("interval with start >= end")
    order = np.argsort(s, kind="stable")
    s, e = s[order], e[order]
    run_end = np.maximum.accumulate(e)
    new = np.ones(s.size, dtype=bool)
    new[1:] = s[1:] > run_end[:-1]
    idx = np.flatnonzero(new)
    return s[idx], np.maximum.reduceat(e, idx)


def intersect(a_s, a_e, b_s, b_e):
    """Positive-length overlaps of two flattened interval sets, as polyAfilter's getOverlaps."""
    a_s, a_e = np.asarray(a_s, np.int64), np.asarray(a_e, np.int64)
    b_s, b_e = np.asarray(b_s, np.int64), np.asarray(b_e, np.int64)
    if a_s.size == 0 or b_s.size == 0:
        z = np.empty(0, np.int64)
        return z, z.copy()
    lo = np.searchsorted(a_e, b_s, side="right")
    hi = np.searchsorted(a_s, b_e, side="left")
    counts = np.maximum(hi - lo, 0)
    b_idx = np.repeat(np.arange(b_s.size), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    a_idx = np.repeat(lo, counts) + offsets
    s = np.maximum(a_s[a_idx], b_s[b_idx])
    e = np.minimum(a_e[a_idx], b_e[b_idx])
    keep = s < e
    s, e = s[keep], e[keep]
    order = np.argsort(s, kind="stable")
    return s[order], e[order]


def blocks_hit(block_s, block_e, piece_s, piece_e):
    """True for each block overlapping at least one base of any piece (pieces sorted and disjoint)."""
    bs = np.asarray(block_s, np.int64)
    be = np.asarray(block_e, np.int64)
    hit = np.zeros(bs.size, dtype=bool)
    if piece_s.size == 0 or bs.size == 0:
        return hit
    idx = np.searchsorted(piece_e, bs, side="right")
    ok = idx < piece_s.size
    hit[ok] = piece_s[idx[ok]] < be[ok]
    return hit
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_intervals.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pafbench/intervals.py tests/test_intervals.py
git commit -m "Interval primitives equal to polyAfilter flatten/getOverlaps"
```

---

### Task 3: A-stretch scanner equal to findSNRpieces

**Files:**
- Create: `pafbench/snr.py`, `tests/test_snr.py`

**Interfaces:**
- Produces: `scan_snrs(seq: str, strd: bool, mism: int) -> (first: ndarray, last: ndarray)` (every stretch findSNRpieces saves at minSNRlen 1); `snr_pieces(first, last, min_len: int, covlen: int, strd: bool, ref_len: int) -> (starts, ends)` (unflattened, same windows as findSNRpieces).

Why one scan serves every MINSNRLEN: the original only changes whether a candidate stretch is saved; the block bookkeeping is identical. A candidate skipped for being short can only be followed (before a new block) by shorter sub-windows, so the stretches saved at MINSNRLEN L are exactly the minSNRlen-1 stretches of length >= L. The test below checks this against the original for several L.

- [ ] **Step 1: Write the failing tests**

```python
import random

from pafbench.snr import scan_snrs, snr_pieces
from pafbench.vendor import polyafilter_modules

BAMfilter, _, _ = polyafilter_modules()
ALPHABET = "AAAAAAATTTTTTCGNat"


def _seq(rng, n):
    core = "".join(rng.choice(ALPHABET) for _ in range(n))
    return rng.choice(["", "AAAA", "TTT"]) + core + rng.choice(["", "AAAAA", "TTTT"])


def test_matches_original_on_random_sequences():
    rng = random.Random(11)
    for _ in range(200):
        seq = _seq(rng, rng.randrange(1, 400))
        for strd in (True, False):
            for mism in (0, 1, 2):
                first, last = scan_snrs(seq, strd, mism)
                for min_len in (1, 2, 3, 5, 8, 12):
                    for covlen in (1, 7, 50):
                        want = BAMfilter.findSNRpieces(covlen, seq, strd, mism=mism, minSNRlen=min_len)
                        s, e = snr_pieces(first, last, min_len, covlen, strd, len(seq))
                        assert list(zip(s.tolist(), e.tolist())) == want, (seq, strd, mism, min_len, covlen)


def test_lowercase_counts_as_base():
    first, last = scan_snrs("CCaaaaaCC", True, 0)
    assert (first.tolist(), last.tolist()) == ([2], [7])


def test_contig_edges_give_no_zero_length_window():
    first, last = scan_snrs("AAAAACCCC", True, 0)
    s, e = snr_pieces(first, last, 3, 10, True, 9)
    assert s.size == 0
    first, last = scan_snrs("CCCCTTTTT", False, 0)
    s, e = snr_pieces(first, last, 3, 10, False, 9)
    assert s.size == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_snr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pafbench.snr'`.

- [ ] **Step 3: Write `pafbench/snr.py`**

```python
import numpy as np
from numba import njit

_IS_A = np.zeros(256, dtype=np.bool_)
_IS_A[ord("A")] = True
_IS_A[ord("a")] = True
_IS_T = np.zeros(256, dtype=np.bool_)
_IS_T[ord("T")] = True
_IS_T[ord("t")] = True


def base_mask(seq, strd):
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return (_IS_A if strd else _IS_T)[arr]


@njit(cache=True)
def _scan(is_base, mism):
    # Line-by-line port of BAMfilter.findSNRpieces with minSNRlen = 1 (every candidate is saved).
    n = is_base.shape[0]
    out_first = np.empty(n // 2 + 2, np.int64)
    out_last = np.empty(n // 2 + 2, np.int64)
    k = 0
    cap = mism + 3
    bs = np.empty(cap, np.int64)
    be = np.empty(cap, np.int64)
    nb = 0
    n_mism = 0
    blocked = False
    last_saved = True
    start = 0
    for i in range(n):
        if is_base[i]:
            if not blocked:
                start = i
                blocked = True
        else:
            if blocked:
                bs[nb] = start
                be[nb] = i
                nb += 1
                last_saved = False
                blocked = False
            if nb > 0:
                n_mism += 1
                if n_mism > mism:
                    if not last_saved:
                        out_first[k] = bs[0]
                        out_last[k] = be[nb - 1]
                        k += 1
                        last_saved = True
                    if nb > 1:
                        n_mism -= bs[1] - be[0]
                    else:
                        n_mism = 0
                    for j in range(nb - 1):
                        bs[j] = bs[j + 1]
                        be[j] = be[j + 1]
                    nb -= 1
    if blocked:
        bs[nb] = start
        be[nb] = n
        nb += 1
        last_saved = False
    if nb > 0 and not last_saved:
        out_first[k] = bs[0]
        out_last[k] = be[nb - 1]
        k += 1
    return out_first[:k].copy(), out_last[:k].copy()


def scan_snrs(seq, strd, mism):
    """Stretches of A (plus strand) or T (minus strand) saved by findSNRpieces at minSNRlen 1; 0-based, last exclusive."""
    return _scan(base_mask(seq, strd), mism)


def snr_pieces(first, last, min_len, covlen, strd, ref_len):
    """COVLEN windows upstream (in transcript sense) of stretches of length >= min_len, unflattened."""
    keep = (last - first) >= min_len
    f, l = first[keep], last[keep]
    if strd:
        s = np.maximum(0, f - covlen)
        e = f
    else:
        s = l
        e = np.minimum(ref_len, l + covlen)
    ok = s != e
    return s[ok], e[ok]
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_snr.py -v`
Expected: 3 passed. If `test_matches_original_on_random_sequences` fails, print the failing `(seq, strd, mism, min_len, covlen)`, run `BAMfilter.findSNRpieces` and `_scan` by hand on it, and fix the port; do not change the comparison.

- [ ] **Step 5: Commit**

```bash
git add pafbench/snr.py tests/test_snr.py
git commit -m "Numba A-stretch scanner equal to findSNRpieces"
```

---

### Task 4: Expressed transcripts and unprotected transcript body

**Files:**
- Create: `pafbench/transcripts.py`, `tests/test_transcripts.py`

**Interfaces:**
- Consumes: `polyafilter_modules()`; `tests.synth.build`.
- Produces: `load_transcripts(gtf) -> dict[(ref: str, strd: bool), list[(start0, end0, exons)]]`; `coverage_prefix(bam, ref) -> {True: ndarray, False: ndarray}` (int64 prefix sums of "position covered", strand True = forward records); `expressed(tx_list, prefix, include_introns) -> list`; `start_pieces(expressed_tx, covlen, include_introns, strd) -> (starts, ends)`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from pafbench.transcripts import coverage_prefix, expressed, load_transcripts, start_pieces
from pafbench.vendor import polyafilter_modules

_, RNA, SNR = polyafilter_modules()


def test_start_pieces_match_original(synth):
    tx = load_transcripts(synth.gtf)
    for introns in (False, True):
        bl = RNA.getBaselineData(str(synth.tmp / f"bl_{int(introns)}.pkl"), str(synth.db), str(synth.bam), includeIntrons=introns)
        for covlen in (50, 100, 300):
            orig = RNA.getTransEndSensSpec(covlen, str(synth.bam), bl, introns, getSensSpec=False)[4]
            for ref in ("chrA", "chrB"):
                prefix = coverage_prefix(synth.bam, ref)
                for strd in (True, False):
                    ex = expressed(tx.get((ref, strd), []), prefix[strd], introns)
                    s, e = start_pieces(ex, covlen, introns, strd)
                    flat = []
                    for t in orig.get(strd, {}).get(ref, []):
                        flat.extend(t)
                    want = SNR.flattenIntervals(flat)
                    assert list(zip(s.tolist(), e.tolist())) == want, (ref, strd, covlen, introns)


def test_coverage_matches_pysam_count_coverage(tmp_path):
    import pysam
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "c", "LN": 100}]})
    path = tmp_path / "x.bam"
    with pysam.AlignmentFile(str(path), "wb", header=hdr) as out:
        a = pysam.AlignedSegment(hdr)
        a.query_name, a.reference_name, a.reference_start, a.cigarstring, a.flag = "s", "c", 10, "5M", 256  # no SEQ
        out.write(a)
        b = pysam.AlignedSegment(hdr)
        b.query_name, b.reference_name, b.reference_start, b.cigarstring, b.flag = "n", "c", 50, "3M", 0
        b.query_sequence, b.query_qualities = "ANA", pysam.qualitystring_to_array("III")
        out.write(b)
    pysam.index(str(path))
    p = coverage_prefix(path, "c")[True]
    assert p[15] - p[10] == 0          # record without SEQ adds no coverage
    assert p[53] - p[50] == 2          # N is not counted
    with pysam.AlignmentFile(str(path)) as bam:
        cov = np.sum(bam.count_coverage("c", 0, 100, quality_threshold=0, read_callback=lambda r: not r.is_reverse), axis=0)
    assert ((p[1:] - p[:-1]) == (cov > 0)).all()


def test_short_transcript_fully_protected_with_introns(synth):
    tx = load_transcripts(synth.gtf)
    prefix = coverage_prefix(synth.bam, "chrB")
    ex = expressed(tx[("chrB", False)], prefix[False], True)
    s, e = start_pieces(ex, 100, True, False)
    assert s.size == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_transcripts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pafbench.transcripts'`.

- [ ] **Step 3: Write `pafbench/transcripts.py`**

```python
from collections import defaultdict

import numpy as np
import pysam

from .vendor import polyafilter_modules

_ACGT = frozenset("ACGT")


def _attr(field, key):
    for part in field.split(";"):
        part = part.strip()
        if part.startswith(key + " "):
            return part.split(" ", 1)[1].strip('"')
    raise KeyError(key)


def load_transcripts(gtf):
    """{(ref, strd): [(start0, end0, exons sorted reverse=strd)]} from GTF transcript and exon lines."""
    tx, exons = {}, defaultdict(list)
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] not in ("transcript", "exon"):
                continue
            tid = _attr(f[8], "transcript_id")
            if f[2] == "transcript":
                tx[tid] = (f[0], f[6] == "+", int(f[3]) - 1, int(f[4]))
            else:
                exons[tid].append((int(f[3]) - 1, int(f[4])))
    out = defaultdict(list)
    for tid, (ref, strd, s, e) in tx.items():
        out[(ref, strd)].append((s, e, sorted(exons[tid], reverse=strd)))
    return dict(out)


def coverage_prefix(bam_path, ref):
    """Per-strand prefix sums of 'position has an aligned A/C/G/T base', as pysam count_coverage with a callable read_callback sees it.

    Every record counts (secondary, duplicate, QC-fail); records without SEQ add nothing; N calls are not counted.
    Strand True collects forward records (polyAfilter's read_callback r.is_reverse != strd).
    """
    with pysam.AlignmentFile(str(bam_path)) as bam:
        L = bam.get_reference_length(ref)
        starts = {True: [], False: []}
        ends = {True: [], False: []}
        not_acgt = {True: [], False: []}
        for a in bam.fetch(ref):
            seq = a.query_sequence
            if seq is None:
                continue
            strd = not a.is_reverse
            for s, e in a.get_blocks():
                starts[strd].append(s)
                ends[strd].append(e)
            if not set(seq) <= _ACGT:
                for q, r in a.get_aligned_pairs(matches_only=True):
                    if seq[q] not in _ACGT:
                        not_acgt[strd].append(r)
    out = {}
    for strd in (True, False):
        diff = np.zeros(L + 1, np.int64)
        np.add.at(diff, np.asarray(starts[strd], np.int64), 1)
        np.add.at(diff, np.asarray(ends[strd], np.int64), -1)
        depth = np.cumsum(diff[:-1])
        if not_acgt[strd]:
            np.add.at(depth, np.asarray(not_acgt[strd], np.int64), -1)
        prefix = np.zeros(L + 1, np.int64)
        np.cumsum(depth > 0, out=prefix[1:])
        out[strd] = prefix
    return out


def expressed(tx_list, prefix, include_introns):
    """Transcripts with any covered base (whole span with introns, exons otherwise)."""
    out = []
    for s, e, exons in tx_list:
        if include_introns:
            hit = prefix[e] - prefix[s] > 0
        else:
            hit = any(prefix[xe] - prefix[xs] > 0 for xs, xe in exons)
        if hit:
            out.append((s, e, exons))
    return out


def start_pieces(expressed_tx, covlen, include_introns, strd):
    """Flattened unprotected body for one (ref, strand), using polyAfilter's own end and overlap functions."""
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
    flat = []
    for st in each_start:
        flat.extend(SNR.removeOverlaps(list(st), all_ends))
    flat = SNR.flattenIntervals(flat)
    return np.array([a for a, _ in flat], np.int64), np.array([b for _, b in flat], np.int64)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_transcripts.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pafbench/transcripts.py tests/test_transcripts.py
git commit -m "Expressed transcripts and protected ends, equal to polyAfilter"
```

---

### Task 5: Per-record removal flags and end-to-end equality on synthetic data

**Files:**
- Modify: `pafbench/flags.py` (append)
- Create: `tests/test_flags.py`, `scripts/03_reimpl_flags.py`

**Interfaces:**
- Consumes: `flatten, intersect, blocks_hit`; `scan_snrs, snr_pieces`; `coverage_prefix, expressed, start_pieces, load_transcripts`.
- Produces: `Setting(covlen, min_len, mism, introns)` namedtuple (`min_len=None` = all non-terminal); `setting_tag(s) -> str` like `c300_l10_m1_i`; `parse_setting("300,10,1,i") -> Setting`; `flag_reference(bam, fasta, ref, transcripts, settings) -> pandas.DataFrame` with columns `qname, flag, ref, pos, cigar` plus one bool column per setting tag; `removed_from_flags(df, tag) -> set`.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from pafbench.flags import Setting, flag_reference, removed_from_flags, setting_tag
from pafbench.transcripts import load_transcripts
from tests.synth import run_original

SETTINGS = [
    Setting(100, 5, 0, False),
    Setting(300, 8, 1, False),
    Setting(300, 8, 1, True),
    Setting(150, 3, 2, True),
    Setting(50, 10, 0, True),
    Setting(200, None, 0, True),
    Setting(200, None, 0, False),
]


@pytest.mark.parametrize("st", SETTINGS, ids=setting_tag)
def test_flags_match_original(synth, st):
    want = run_original(synth, st.covlen, st.min_len, st.mism, st.introns)
    tx = load_transcripts(synth.gtf)
    got = set()
    for ref in ("chrA", "chrB"):
        df = flag_reference(synth.bam, synth.fasta, ref, tx, [st])
        got |= removed_from_flags(df, setting_tag(st))
    assert got == want


def test_some_setting_removes_reads(synth):
    assert len(run_original(synth, 300, 8, 1, True)) > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_flags.py -v`
Expected: FAIL with `ImportError: cannot import name 'Setting'`.

- [ ] **Step 3: Extend `pafbench/flags.py`**

Put these imports at the top of the file, below `import pysam`, and append the functions after `removed_keys`:

```python
from collections import namedtuple

import numpy as np
import pandas as pd

from .intervals import blocks_hit, flatten, intersect
from .snr import scan_snrs, snr_pieces
from .transcripts import coverage_prefix, expressed, start_pieces

Setting = namedtuple("Setting", "covlen min_len mism introns")


def setting_tag(s):
    lname = "none" if s.min_len is None else str(s.min_len)
    return f"c{s.covlen}_l{lname}_m{s.mism}_{'i' if s.introns else 'e'}"


def parse_setting(text):
    c, l, m, mode = text.split(",")
    return Setting(int(c), None if l == "none" else int(l), int(m), mode == "i")


def flag_reference(bam_path, fasta_path, ref, transcripts, settings):
    """Per-record removal decision on one reference for each setting, in fetch order."""
    prefix = coverage_prefix(bam_path, ref)
    with pysam.FastaFile(str(fasta_path)) as fa:
        seq = fa.fetch(ref)
    ref_len = len(seq)
    rows, rec_fwd, blk_rec, blk_s, blk_e = [], [], [], [], []
    with pysam.AlignmentFile(str(bam_path)) as bam:
        for i, a in enumerate(bam.fetch(ref)):
            rows.append((a.query_name, a.flag, ref, a.reference_start, a.cigarstring))
            rec_fwd.append(not a.is_reverse)
            for s, e in a.get_blocks():
                blk_rec.append(i)
                blk_s.append(s)
                blk_e.append(e)
    df = pd.DataFrame(rows, columns=["qname", "flag", "ref", "pos", "cigar"])
    rec_fwd = np.asarray(rec_fwd, bool)
    blk_rec = np.asarray(blk_rec, np.int64)
    blk_s = np.asarray(blk_s, np.int64)
    blk_e = np.asarray(blk_e, np.int64)
    blk_fwd = rec_fwd[blk_rec] if blk_rec.size else np.zeros(0, bool)
    snr_cache, body_cache = {}, {}
    for st in settings:
        removed = np.zeros(len(df), bool)
        for strd in (True, False):
            bkey = (strd, st.covlen, st.introns)
            if bkey not in body_cache:
                ex = expressed(transcripts.get((ref, strd), []), prefix[strd], st.introns)
                body_cache[bkey] = start_pieces(ex, st.covlen, st.introns, strd)
            body_s, body_e = body_cache[bkey]
            if st.min_len is None:
                ps, pe = body_s, body_e
            else:
                skey = (strd, st.mism)
                if skey not in snr_cache:
                    snr_cache[skey] = scan_snrs(seq, strd, st.mism)
                first, last = snr_cache[skey]
                ws, we = flatten(*snr_pieces(first, last, st.min_len, st.covlen, strd, ref_len))
                ps, pe = intersect(body_s, body_e, ws, we)
            sel = blk_fwd == strd
            hit = blocks_hit(blk_s[sel], blk_e[sel], ps, pe)
            removed[np.unique(blk_rec[sel][hit])] = True
        df[setting_tag(st)] = removed
    return df


def removed_from_flags(df, tag):
    d = df[df[tag]]
    return set(zip(d.qname, d.flag, d.ref, d.pos, d.cigar))
```


- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_flags.py -v`
Expected: 8 passed. If a setting differs, print `got - want` and `want - got`, locate the records on the fixture, and find which stage (body, windows, overlap) disagrees by comparing with the original's intermediate objects; fix the reimplementation, never the expectation.

- [ ] **Step 5: Write `scripts/03_reimpl_flags.py`**

```python
"""Stage 03: per-record removal flags for polyAfilter settings (reimplementation)."""
import argparse
import json
from multiprocessing import Pool

import pysam

from pafbench.config import FASTA, GTF, LIBRARIES, OUT
from pafbench.flags import flag_reference, parse_setting, setting_tag
from pafbench.transcripts import load_transcripts

_TX = None


def _init():
    global _TX
    _TX = load_transcripts(GTF)


def _work(job):
    bam, ref, settings, out_dir = job
    df = flag_reference(bam, FASTA, ref, _TX, settings)
    df.to_parquet(out_dir / f"{ref}.parquet", index=False)
    return ref, len(df), {setting_tag(s): int(df[setting_tag(s)].sum()) for s in settings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--setting", action="append", required=True, help="COVLEN,MINSNRLEN|none,MISM,i|e")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    settings = [parse_setting(s) for s in args.setting]
    bam = LIBRARIES[args.library]
    out_dir = OUT / args.library / "reimpl_flags"
    out_dir.mkdir(parents=True, exist_ok=True)
    with pysam.AlignmentFile(str(bam)) as fh:
        stats = sorted((s for s in fh.get_index_statistics() if s.mapped > 0), key=lambda s: -s.mapped)
    jobs = [(bam, s.contig, settings, out_dir) for s in stats]
    with Pool(args.workers, initializer=_init) as pool:
        res = pool.map(_work, jobs, chunksize=1)
    summary = {
        "library": args.library,
        "settings": [setting_tag(s) for s in settings],
        "per_ref": {ref: {"records": n, "removed": rem} for ref, n, rem in res},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add pafbench/flags.py tests/test_flags.py scripts/03_reimpl_flags.py
git commit -m "Reimplemented removal flags; equal to original on synthetic data"
```

---

### Task 6: Exact match on the full 1417 BAM

**Files:**
- Create: `scripts/04_compare_removed.py`

**Interfaces:**
- Consumes: `removed_keys_by_ref`, `removed_from_flags`, `parse_setting`, `setting_tag`; outputs of Task 1 Step 7 and Task 5 Step 5.
- Produces: `/data/element_runs/polyafilter_bench/1417/exact_match.json` with, per setting, `n_original, n_reimpl, only_original, only_reimpl, identical`.

- [ ] **Step 1: Write `scripts/04_compare_removed.py`**

```python
"""Stage 04: compare removed-record sets, original polyAfilter vs reimplementation."""
import argparse
import json
import sys

import pandas as pd

from pafbench.config import LIBRARIES, OUT
from pafbench.flags import parse_setting, removed_from_flags, removed_keys_by_ref, setting_tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--setting", action="append", required=True)
    args = ap.parse_args()
    bam = LIBRARIES[args.library]
    flags_dir = OUT / args.library / "reimpl_flags"
    results = {}
    for st in (parse_setting(s) for s in args.setting):
        tag = setting_tag(st)
        orig_bam = OUT / args.library / "original" / f"{tag}.bam"
        n_o = n_r = only_o = only_r = 0
        examples = []
        for ref, want in removed_keys_by_ref(bam, orig_bam):
            pq = flags_dir / f"{ref}.parquet"
            got = removed_from_flags(pd.read_parquet(pq), tag) if ref is not None and pq.exists() else set()
            n_o += len(want)
            n_r += len(got)
            a, b = want - got, got - want
            only_o += len(a)
            only_r += len(b)
            examples.extend([["only_original", *k] for k in list(a)[:5]] + [["only_reimpl", *k] for k in list(b)[:5]])
        results[tag] = {"n_original": n_o, "n_reimpl": n_r, "only_original": only_o, "only_reimpl": only_r,
                        "identical": only_o == 0 and only_r == 0, "examples": examples[:50]}
    out = OUT / args.library / "exact_match.json"
    out.write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v["identical"] for k, v in results.items()}))
    sys.exit(0 if all(v["identical"] for v in results.values()) else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the reimplementation on 1417**

```bash
.venv/bin/python scripts/03_reimpl_flags.py 1417 --setting 300,10,1,i --setting 375,3,2,i --setting 300,10,1,e
```
Expected: `/data/element_runs/polyafilter_bench/1417/reimpl_flags/summary.json` and one Parquet per reference with reads.

- [ ] **Step 3: When all three original runs in the chain have finished, compare**

Check: `grep -h "Exit status" /data/element_runs/polyafilter_bench/1417/original/*.time.txt` shows `Exit status: 0` for all three.
Run:
```bash
.venv/bin/python scripts/04_compare_removed.py 1417 --setting 300,10,1,i --setting 375,3,2,i --setting 300,10,1,e
```
Expected: `{"c300_l10_m1_i": true, "c375_l3_m2_i": true, "c300_l10_m1_e": true}` and exit code 0. If any is false: stop, open `exact_match.json`, take the example keys, inspect those records (`samtools view` at the position) and their transcript context, find the rule that differs, add a synthetic case that reproduces it to `tests/synth.py`, fix the reimplementation until `tests/test_flags.py` and this comparison both pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/04_compare_removed.py
git commit -m "Exact-match comparison script; 1417 results in exact_match.json"
```

---

### Task 7: Priming-site context and `pa` read table

**Files:**
- Create: `pafbench/sites.py`, `pafbench/genes.py`, `tests/test_sites.py`, `scripts/05_pa_sites.py`

**Interfaces:**
- Produces: `priming_site(a) -> int`; `last_aligned_base(a) -> int`; `sense_context(fa, ref, site, is_reverse, down=20, up=40) -> (down: str, up: str)`; `has_hexamer(up40, lo=5, hi=35) -> bool`; `classify(a20, templated_min=14, nontemplated_max=8) -> str`; `PA_COLUMNS`; `pa_row(a, fa, segments) -> list`; `load_gene_segments(gtf) -> dict`; `genes_at(segments, ref, strd, pos) -> str` (`;`-joined gene_ids, `""` if none).

- [ ] **Step 1: Write the failing tests**

```python
import pysam

from pafbench.genes import genes_at, load_gene_segments
from pafbench.sites import classify, has_hexamer, priming_site, sense_context


def _read(hdr, pos, cigar, rev):
    a = pysam.AlignedSegment(hdr)
    a.query_name = "x"
    a.reference_name = "chrA"
    a.reference_start = pos
    a.cigarstring = cigar
    a.flag = 16 if rev else 0
    a.query_sequence = "C" * sum(n for op, n in a.cigartuples if op in (0, 1, 4, 7, 8))
    return a


def test_site_and_context(synth):
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "chrA", "LN": 6000}, {"SN": "chrB", "LN": 2000}]})
    with pysam.FastaFile(str(synth.fasta)) as fa:
        fwd = _read(hdr, 1209, "90M", False)        # ends at 1299, where A*12 is planted
        assert priming_site(fwd) == 1299
        down, up = sense_context(fa, "chrA", 1299, False)
        assert down.startswith("A" * 12) and len(down) == 20 and len(up) == 40
        rev = _read(hdr, 4509, "90M", True)          # minus strand; T*10 at 4499..4508 lies downstream
        assert priming_site(rev) == 4509
        down, _ = sense_context(fa, "chrA", 4509, True)
        assert down.startswith("A" * 10)


def test_hexamer_window():
    up = "C" * 20 + "AATAAA" + "C" * 14            # hexamer ends 14 nt before the site
    assert has_hexamer(up)
    assert not has_hexamer("C" * 30 + "AATAAA" + "C" * 4)   # ends 4 nt before the site, too close
    assert not has_hexamer("AATAAA" + "C" * 34)     # further than 35 nt


def test_classify():
    assert classify(14) == "templated" and classify(8) == "nontemplated" and classify(11) == "ambiguous"


def test_gene_segments(synth):
    seg = load_gene_segments(synth.gtf)
    assert genes_at(seg, "chrA", True, 1500) == "g1"
    assert genes_at(seg, "chrA", True, 4100) == ""
    assert genes_at(seg, "chrA", False, 4300) == "g2"
    assert genes_at(seg, "chrB", True, 1700) == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_sites.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pafbench.genes'`.

- [ ] **Step 3: Write `pafbench/genes.py`**

```python
from collections import defaultdict

import numpy as np


def load_gene_segments(gtf):
    """{(ref, strd): (bounds, labels)}: genome position -> ';'-joined gene_ids covering it on that strand."""
    spans = defaultdict(list)
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if f[2] != "gene":
                continue
            gid = f[8].split('gene_id "', 1)[1].split('"', 1)[0]
            spans[(f[0], f[6] == "+")].append((int(f[3]) - 1, int(f[4]), gid))
    out = {}
    for key, lst in spans.items():
        events = sorted([(s, 1, g) for s, _, g in lst] + [(e, -1, g) for _, e, g in lst])
        bounds, labels, active, i = [], [], set(), 0
        while i < len(events):
            pos = events[i][0]
            while i < len(events) and events[i][0] == pos:
                _, kind, g = events[i]
                if kind == 1:
                    active.add(g)
                else:
                    active.discard(g)
                i += 1
            bounds.append(pos)
            labels.append(";".join(sorted(active)))
        out[key] = (np.asarray(bounds, np.int64), labels)
    return out


def genes_at(segments, ref, strd, pos):
    entry = segments.get((ref, strd))
    if entry is None:
        return ""
    bounds, labels = entry
    i = int(np.searchsorted(bounds, pos, side="right")) - 1
    return labels[i] if i >= 0 else ""
```

- [ ] **Step 4: Write `pafbench/sites.py`**

```python
from .genes import genes_at

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
HEXAMERS = ("AATAAA", "ATTAAA")
PA_COLUMNS = ["qname", "ref", "strand", "site", "pa", "ts", "cb", "ub", "gx", "re", "xf", "flag", "dup",
              "mapq", "aligned_len", "spliced", "down20", "up40", "a20", "hexamer", "genes"]


def priming_site(a):
    """Genomic boundary at the read's 3' end in transcript sense (last cDNA base is just upstream of it)."""
    return a.reference_start if a.is_reverse else a.reference_end


def last_aligned_base(a):
    return a.reference_start if a.is_reverse else a.reference_end - 1


def sense_context(fa, ref, site, is_reverse, down=20, up=40):
    """Genome after (down) and before (up) the site, transcript sense, uppercase."""
    L = fa.get_reference_length(ref)
    if is_reverse:
        d = fa.fetch(ref, max(0, site - down), site).upper().translate(COMP)[::-1]
        u = fa.fetch(ref, site, min(L, site + up)).upper().translate(COMP)[::-1]
    else:
        d = fa.fetch(ref, site, min(L, site + down)).upper()
        u = fa.fetch(ref, max(0, site - up), site).upper()
    return d, u


def has_hexamer(up40, lo=5, hi=35):
    """AATAAA or ATTAAA fully inside the window lo..hi nt upstream of the site."""
    n = len(up40)
    win = up40[max(0, n - hi):max(0, n - lo)]
    return any(h in win for h in HEXAMERS)


def classify(a20, templated_min=14, nontemplated_max=8):
    if a20 >= templated_min:
        return "templated"
    if a20 <= nontemplated_max:
        return "nontemplated"
    return "ambiguous"


def _tag(a, t, default=None):
    return a.get_tag(t) if a.has_tag(t) else default


def pa_row(a, fa, segments):
    site = priming_site(a)
    down, up = sense_context(fa, a.reference_name, site, a.is_reverse)
    return [a.query_name, a.reference_name, "-" if a.is_reverse else "+", site, a.get_tag("pa"), _tag(a, "ts", 0),
            _tag(a, "CB"), _tag(a, "UB"), _tag(a, "GX"), _tag(a, "RE"), _tag(a, "xf"), a.flag, a.is_duplicate,
            a.mapping_quality, a.query_alignment_length, "N" in a.cigarstring, down, up, down.count("A"),
            has_hexamer(up), genes_at(segments, a.reference_name, not a.is_reverse, last_aligned_base(a))]
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_sites.py -v`
Expected: 4 passed.

- [ ] **Step 6: Write `scripts/05_pa_sites.py`**

```python
"""Stage 05: primary reads with a Cell Ranger pa tag, with priming-site context."""
import argparse
from multiprocessing import Pool

import pandas as pd
import pysam

from pafbench.config import FASTA, GTF, LIBRARIES, OUT
from pafbench.genes import load_gene_segments
from pafbench.sites import PA_COLUMNS, pa_row

_SEG = None


def _init():
    global _SEG
    _SEG = load_gene_segments(GTF)


def _work(job):
    bam_path, ref = job
    rows = []
    with pysam.AlignmentFile(str(bam_path)) as bam, pysam.FastaFile(str(FASTA)) as fa:
        for a in bam.fetch(ref):
            if a.is_secondary or a.is_supplementary or not a.has_tag("pa"):
                continue
            rows.append(pa_row(a, fa, _SEG))
    return pd.DataFrame(rows, columns=PA_COLUMNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--refs", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    bam = LIBRARIES[args.library]
    with pysam.AlignmentFile(str(bam)) as fh:
        refs = args.refs or [s.contig for s in fh.get_index_statistics() if s.mapped > 0]
    with Pool(args.workers, initializer=_init) as pool:
        parts = pool.map(_work, [(bam, r) for r in refs], chunksize=1)
    df = pd.concat(parts, ignore_index=True)
    out = args.out or (OUT / args.library / "pa_sites.parquet")
    df.to_parquet(out, index=False)
    print(out, len(df))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run on 1417 (all references) and 1416 (chr1 only)**

```bash
mkdir -p /data/element_runs/polyafilter_bench/1416
.venv/bin/python scripts/05_pa_sites.py 1417
.venv/bin/python scripts/05_pa_sites.py 1416 --refs chr1 --out /data/element_runs/polyafilter_bench/1416/pa_sites_chr1.parquet
```
Expected: 1416 chr1 prints 103,708 rows (the pilot count; any difference means the filter differs from the pilot and must be explained before continuing).

- [ ] **Step 8: Check UB on intronic reads in 1179 (spec 4.1)**

`scripts/05b_ub_on_introns.py`:
```python
"""Stage 05b: share of exonic and intronic primary reads carrying UB in 1179 (include-introns false), chr1."""
import json

import pysam

from pafbench.config import LIBRARIES, OUT


def main():
    n = {"E_total": 0, "E_with_UB": 0, "N_total": 0, "N_with_UB": 0}
    with pysam.AlignmentFile(str(LIBRARIES["1179"])) as bam:
        for a in bam.fetch("chr1"):
            if a.is_secondary or a.is_supplementary or not a.has_tag("RE"):
                continue
            region = a.get_tag("RE")
            if region in ("E", "N"):
                n[f"{region}_total"] += 1
                n[f"{region}_with_UB"] += int(a.has_tag("UB"))
    out = OUT / "1179"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ub_on_introns.json").write_text(json.dumps(n, indent=1))
    print(n)


if __name__ == "__main__":
    main()
```
Run: `.venv/bin/python scripts/05b_ub_on_introns.py`
Expected: counts for both regions. If intronic reads in 1179 mostly lack UB, intronic molecules in 1179 cannot enter the truth set; record this for the milestone note (it limits the nuclei versus cells comparison to exonic sites).

- [ ] **Step 9: Commit**

```bash
git add pafbench/sites.py pafbench/genes.py tests/test_sites.py scripts/05_pa_sites.py scripts/05b_ub_on_introns.py
git commit -m "pa read table with priming-site context, own gene assignment, UB on introns in 1179"
```

---

### Task 8: Molecules, members, UMI collision rate, member cap

**Files:**
- Create: `pafbench/molecules.py`, `tests/test_molecules.py`, `scripts/06_molecules.py`

**Interfaces:**
- Consumes: `pa_sites.parquet` from Task 7; `genes_at`, `last_aligned_base`.
- Produces: `consolidate_sites(pa, max_spread=10) -> DataFrame` (one row per molecule key (cb, ub, strand, ref, genes), as spec 4.1; columns `mol, cb, ub, strand, ref, site, n_pa, spread, a20, down20, up40, hexamer, genes, re`); `member_distances(a, site) -> (dist5, dist3)`; `gene_match(site_genes, member_genes) -> bool`; `EDGES`; `member_density(members, n_molecules) -> DataFrame` (`lo, hi, n, density`); `choose_cap(curve, background, candidates=(500, 750, 1000, 1500, 2000, 3000, 5000), factor=2.0) -> int | None`. Files: `molecules.parquet`, `members.parquet`, `collisions.json`.

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd
import pysam

from pafbench.molecules import choose_cap, consolidate_sites, gene_match, member_density, member_distances


def _pa(rows):
    cols = ["cb", "ub", "strand", "ref", "site", "a20", "down20", "up40", "hexamer", "genes", "re"]
    return pd.DataFrame(rows, columns=cols)


def test_consolidate_drops_spread_and_missing_umi():
    pa = _pa([
        ["c1", "u1", "+", "chr1", 100, 15, "A" * 20, "C" * 40, False, "g", "E"],
        ["c1", "u1", "+", "chr1", 104, 15, "A" * 20, "C" * 40, False, "g", "E"],
        ["c1", "u1", "+", "chr1", 109, 15, "A" * 20, "C" * 40, False, "g", "E"],
        ["c2", "u2", "+", "chr1", 100, 3, "C" * 20, "C" * 40, True, "g", "E"],
        ["c2", "u2", "+", "chr1", 200, 3, "C" * 20, "C" * 40, True, "g", "E"],
        ["c3", None, "+", "chr1", 300, 3, "C" * 20, "C" * 40, True, "g", "E"],
    ])
    mol = consolidate_sites(pa)
    assert len(mol) == 1
    assert mol.iloc[0].site == 104 and mol.iloc[0].n_pa == 3


def test_member_distances():
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "chr1", "LN": 10000}]})
    a = pysam.AlignedSegment(hdr)
    a.reference_name, a.reference_start, a.cigarstring, a.flag = "chr1", 700, "90M", 0
    a.query_sequence = "C" * 90
    assert member_distances(a, 1000) == (300, 210)
    a.flag = 16
    assert member_distances(a, 500) == (290, 200)


def test_gene_match():
    assert gene_match("", "") and gene_match("g1;g2", "g2") and not gene_match("g1", "g2") and not gene_match("g1", "")


def test_choose_cap():
    # bins from 750 to 1500 are empty, so d=750 passes; d=500 fails on the 600-nt members
    members = pd.DataFrame({"dist5": [150] * 900 + [600] * 50 + [500000] * 9,
                            "dist3": [0] * 959, "spliced": [False] * 959})
    curve = member_density(members, n_molecules=100)
    background = curve[curve.lo == 100000].density.iloc[0]
    assert choose_cap(curve, background) == 750
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_molecules.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pafbench.molecules'`.

- [ ] **Step 3: Write `pafbench/molecules.py`**

```python
import numpy as np
import pandas as pd

EDGES = [0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 50000, 100000, 1000000]
MOL_COLUMNS = ["mol", "cb", "ub", "strand", "ref", "site", "n_pa", "spread", "a20", "down20", "up40", "hexamer", "genes", "re"]


def consolidate_sites(pa, max_spread=10):
    """One row per molecule (cb, ub, strand, ref, own gene label); site = lower median of its pa sites; spread > max_spread dropped."""
    d = pa[pa.cb.notna() & pa.ub.notna()]
    rows = []
    for (cb, ub, strand, ref, genes), g in d.groupby(["cb", "ub", "strand", "ref", "genes"], sort=False):
        sites = np.sort(g.site.to_numpy())
        spread = int(sites[-1] - sites[0])
        if spread > max_spread:
            continue
        site = int(sites[(len(sites) - 1) // 2])
        r = g.iloc[int(np.argmin(np.abs(g.site.to_numpy() - site)))]
        rows.append([len(rows), cb, ub, strand, ref, site, len(g), spread, int(r.a20), r.down20, r.up40, bool(r.hexamer), genes, r.re])
    return pd.DataFrame(rows, columns=MOL_COLUMNS)


def member_distances(a, site):
    """(read 5' start to site, read 3' end to site), both in transcript sense; negative dist3 = read passes the site."""
    if a.is_reverse:
        return a.reference_end - site, a.reference_start - site
    return site - a.reference_start, site - a.reference_end


def gene_match(site_genes, member_genes):
    if site_genes == "" and member_genes == "":
        return True
    return bool(set(filter(None, site_genes.split(";"))) & set(filter(None, member_genes.split(";"))))


def member_density(members, n_molecules, edges=EDGES):
    """Unspliced members (dist3 >= -25) per bp per molecule, by dist5 bin."""
    m = members[(~members.spliced) & (members.dist3 >= -25)]
    counts, _ = np.histogram(m.dist5, bins=edges)
    widths = np.diff(edges)
    return pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "n": counts, "density": counts / (widths * n_molecules)})


def choose_cap(curve, background, candidates=(500, 750, 1000, 1500, 2000, 3000, 5000), factor=2.0):
    """Smallest d whose bins in [d, 2d) all have density <= factor * background."""
    for d in candidates:
        sel = curve[(curve.lo >= d) & (curve.hi <= 2 * d)]
        if len(sel) and bool((sel.density <= factor * background).all()):
            return d
    return None
```


- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_molecules.py -v`
Expected: 4 passed.

- [ ] **Step 5: Write `scripts/06_molecules.py`**

```python
"""Stage 06: pa molecules, member reads, UMI collision background, member cap."""
import argparse
import json
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd
import pysam

from pafbench.config import GTF, LIBRARIES, OUT
from pafbench.genes import genes_at, load_gene_segments
from pafbench.molecules import choose_cap, consolidate_sites, gene_match, member_density, member_distances
from pafbench.sites import last_aligned_base

_SEG = None
_BY_K3 = None


def _init(by_k3):
    global _SEG, _BY_K3
    _SEG = load_gene_segments(GTF)
    _BY_K3 = by_k3


def _work(job):
    bam_path, ref = job
    rows, coll = [], defaultdict(int)
    with pysam.AlignmentFile(str(bam_path)) as bam:
        for a in bam.fetch(ref):
            if a.is_secondary or a.is_supplementary or a.has_tag("pa") or not a.has_tag("CB") or not a.has_tag("UB"):
                continue
            hits = _BY_K3.get((a.get_tag("CB"), a.get_tag("UB"), "-" if a.is_reverse else "+"))
            if not hits:
                continue
            for mol, mref, site, sgenes in hits:
                if mref != ref:
                    coll[mol] += 1
                    continue
                d5, d3 = member_distances(a, site)
                mg = genes_at(_SEG, ref, not a.is_reverse, last_aligned_base(a))
                rows.append([mol, a.query_name, d5, d3, "N" in a.cigarstring, a.is_duplicate, mg, gene_match(sgenes, mg)])
    cols = ["mol", "qname", "dist5", "dist3", "spliced", "dup", "genes", "gene_match"]
    return pd.DataFrame(rows, columns=cols), dict(coll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    base = OUT / args.library
    mol = consolidate_sites(pd.read_parquet(base / "pa_sites.parquet"))
    mol.to_parquet(base / "molecules.parquet", index=False)
    by_k3 = defaultdict(list)
    for r in mol.itertuples():
        by_k3[(r.cb, r.ub, r.strand)].append((r.mol, r.ref, r.site, r.genes))
    bam = LIBRARIES[args.library]
    with pysam.AlignmentFile(str(bam)) as fh:
        stats = [s for s in fh.get_index_statistics() if s.mapped > 0]
        lengths = {s.contig: fh.get_reference_length(s.contig) for s in stats}
    with Pool(args.workers, initializer=_init, initargs=(dict(by_k3),)) as pool:
        parts = pool.map(_work, [(bam, s.contig) for s in stats], chunksize=1)
    members = pd.concat([p[0] for p in parts], ignore_index=True)
    members.to_parquet(base / "members.parquet", index=False)
    coll_total = sum(sum(p[1].values()) for p in parts)
    genome = sum(lengths.values())
    other_bp = float(sum(genome - lengths[r] for r in mol.ref))
    bg_other = coll_total / other_bp if other_bp else 0.0
    curve = member_density(members, len(mol))
    bg_far = float(curve[curve.lo == 100000].density.iloc[0])
    background = max(bg_other, bg_far)
    cap = choose_cap(curve, background)
    out = {"molecules": len(mol), "members": len(members), "collision_reads_other_refs": coll_total,
           "background_other_refs_per_bp": bg_other, "background_far_same_ref_per_bp": bg_far,
           "background_used": background, "cap": cap, "curve": curve.to_dict(orient="records")}
    (base / "collisions.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("molecules", "members", "cap")}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run on 1417**

Run: `.venv/bin/python scripts/06_molecules.py 1417`
Expected: prints molecule, member counts and a cap from the candidate list. If `cap` is `null`, no candidate up to 5 kb reaches the background: record the curve and bring it to the user before Task 9 (this changes the spec's member rule).

- [ ] **Step 7: Commit**

```bash
git add pafbench/molecules.py tests/test_molecules.py scripts/06_molecules.py
git commit -m "Molecules, members, collision background and member cap"
```

---

### Task 9: Label cut-offs from the hexamer curve

**Files:**
- Create: `pafbench/labels.py`, `tests/test_labels.py`, `scripts/07_label_curve.py`

**Interfaces:**
- Consumes: `molecules.parquet` (columns `a20`, `hexamer`, `re`).
- Produces: `hexamer_curve(mol) -> DataFrame (a20, n, rate)`; `choose_cutoffs(curve, frac=0.25, min_n=200) -> dict(templated_min, nontemplated_max, tail_rate, background_rate, frac, min_n)`; `templated_share(mol, templated_min, nontemplated_max) -> DataFrame` indexed by `re` with `templated, nontemplated, ambiguous, share`; `compare_shares(a, b, threshold=0.05) -> DataFrame (share_a, share_b, diff, flag)`. File: `label_cutoffs.json`, `hexamer_curve.tsv`.

The rule is fixed here, before any polyAfilter metric exists: tail level = pooled hexamer rate at A count 0 to 4; background = pooled rate at 18 to 20; templated from the smallest A count from which every bin (n >= 200) sits within a quarter of the gap above background; non-templated up to the largest A count below which every bin sits within a quarter of the gap below the tail level.

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd

from pafbench.labels import choose_cutoffs, compare_shares, hexamer_curve, templated_share

RATES = [0.5] * 5 + [0.45, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]


def test_choose_cutoffs():
    curve = pd.DataFrame({"a20": range(21), "n": [1000] * 21, "rate": RATES})
    c = choose_cutoffs(curve)
    assert (c["templated_min"], c["nontemplated_max"]) == (10, 6)


def test_low_count_bins_ignored():
    rates = list(RATES)
    rates[12] = 0.9
    n = [1000] * 21
    n[12] = 50
    c = choose_cutoffs(pd.DataFrame({"a20": range(21), "n": n, "rate": rates}))
    assert c["templated_min"] == 10


def test_hexamer_curve_and_shares():
    mol = pd.DataFrame({"a20": [2, 2, 16, 16, 11], "hexamer": [True, False, False, False, False], "re": ["E", "E", "E", "N", "N"]})
    curve = hexamer_curve(mol)
    assert curve.loc[curve.a20 == 2, "rate"].iloc[0] == 0.5
    t = templated_share(mol, 14, 8)
    assert t.loc["E", "share"] == 1 / 3 and t.loc["N", "share"] == 1.0
    cmp = compare_shares(t, t.assign(share=t.share + 0.1))
    assert cmp.flag.all()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pafbench.labels'`.

- [ ] **Step 3: Write `pafbench/labels.py`**

```python
import numpy as np
import pandas as pd


def hexamer_curve(mol):
    g = mol.groupby("a20").hexamer.agg(["size", "mean"])
    g = g.reindex(range(21), fill_value=0)
    return pd.DataFrame({"a20": g.index, "n": g["size"].to_numpy(), "rate": g["mean"].to_numpy()})


def _pooled(c, lo, hi):
    sel = c[(c.a20 >= lo) & (c.a20 <= hi)]
    return float((sel.rate * sel.n).sum() / sel.n.sum())


def choose_cutoffs(curve, frac=0.25, min_n=200):
    tail = _pooled(curve, 0, 4)
    bg = _pooled(curve, 18, 20)
    span = tail - bg
    if span <= 0:
        raise ValueError("hexamer rate is not higher at low A counts; cut-offs cannot be calibrated")
    ok = curve[curve.n >= min_n]
    templ = next(k for k in range(22) if bool((ok[ok.a20 >= k].rate <= bg + frac * span).all()))
    nont = max(k for k in range(-1, 21) if bool((ok[ok.a20 <= k].rate >= tail - frac * span).all()))
    if templ > 20 or nont < 0 or nont >= templ:
        raise ValueError(f"no usable cut-offs (templated_min={templ}, nontemplated_max={nont})")
    return {"templated_min": int(templ), "nontemplated_max": int(nont), "tail_rate": tail,
            "background_rate": bg, "frac": frac, "min_n": min_n}


def templated_share(mol, templated_min, nontemplated_max):
    cls = np.where(mol.a20 >= templated_min, "templated", np.where(mol.a20 <= nontemplated_max, "nontemplated", "ambiguous"))
    t = mol.assign(cls=cls).groupby("re").cls.value_counts().unstack(fill_value=0)
    for col in ("templated", "nontemplated", "ambiguous"):
        if col not in t:
            t[col] = 0
    t["share"] = t.templated / (t.templated + t.nontemplated)
    return t


def compare_shares(a, b, threshold=0.05):
    j = a[["share"]].join(b[["share"]], lsuffix="_a", rsuffix="_b", how="inner")
    j["diff"] = j.share_b - j.share_a
    j["flag"] = j["diff"].abs() > threshold
    return j
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_labels.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write `scripts/07_label_curve.py` and run it on 1417**

```python
"""Stage 07: hexamer rate by downstream A count; fixed label cut-offs."""
import argparse
import json

import pandas as pd

from pafbench.config import OUT
from pafbench.labels import choose_cutoffs, hexamer_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    args = ap.parse_args()
    base = OUT / args.library
    mol = pd.read_parquet(base / "molecules.parquet")
    curve = hexamer_curve(mol)
    curve.to_csv(base / "hexamer_curve.tsv", sep="\t", index=False)
    cut = choose_cutoffs(curve)
    (base / "label_cutoffs.json").write_text(json.dumps(cut, indent=1))
    print(json.dumps(cut))


if __name__ == "__main__":
    main()
```

Run: `.venv/bin/python scripts/07_label_curve.py 1417`
Expected: prints `templated_min` and `nontemplated_max`. A `ValueError` means the curve does not separate: report the curve to the user; do not pick cut-offs by hand.

- [ ] **Step 6: Commit**

```bash
git add pafbench/labels.py tests/test_labels.py scripts/07_label_curve.py
git commit -m "Label cut-offs fixed from the hexamer curve"
```

---

### Task 10: Scope check, 1416 versus 1417 on chr1

**Files:**
- Create: `scripts/08_scope_check.py`

**Interfaces:**
- Consumes: `consolidate_sites`, `templated_share`, `compare_shares`; `1416/pa_sites_chr1.parquet`, `1417/pa_sites.parquet`, `1417/label_cutoffs.json`.
- Produces: `/data/element_runs/polyafilter_bench/scope_check.json` with shares by region for both cut-off sets and `go: true|false`.

- [ ] **Step 1: Write `scripts/08_scope_check.py`**

```python
"""Stage 08: does the 3k aliquot (1417) carry the same priming bias as the 20k library (1416)? chr1, by region."""
import json

import pandas as pd

from pafbench.config import OUT
from pafbench.labels import compare_shares, templated_share
from pafbench.molecules import consolidate_sites


def main():
    a = consolidate_sites(pd.read_parquet(OUT / "1416" / "pa_sites_chr1.parquet"))
    p17 = pd.read_parquet(OUT / "1417" / "pa_sites.parquet")
    b = consolidate_sites(p17[p17.ref == "chr1"])
    cut = json.loads((OUT / "1417" / "label_cutoffs.json").read_text())
    res = {}
    for name, (tmin, nmax) in {"start_14_8": (14, 8), "fixed": (cut["templated_min"], cut["nontemplated_max"])}.items():
        cmp = compare_shares(templated_share(a, tmin, nmax), templated_share(b, tmin, nmax))
        res[name] = cmp.reset_index().to_dict(orient="records")
    go = not any(r["flag"] for rows in res.values() for r in rows)
    res["go"] = go
    res["molecules"] = {"1416_chr1": len(a), "1417_chr1": len(b)}
    (OUT / "scope_check.json").write_text(json.dumps(res, indent=1, default=float))
    print("go" if go else "NO-GO: a region differs by more than 5 points; take it to the user")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python scripts/08_scope_check.py`
Expected: `go`, or the NO-GO line. On NO-GO, stop the milestone after Task 13 and bring `scope_check.json` to the user.

- [ ] **Step 3: Commit**

```bash
git add scripts/08_scope_check.py
git commit -m "Scope check 1416 vs 1417"
```

---

### Task 11: Spike-in realignment keeps the tags

**Files:**
- Create: `pafbench/spikein.py`, `tests/test_spikein.py`, `scripts/09_spikein_star.sh`, `scripts/09b_check_tags.py`

**Interfaces:**
- Produces: `tag_survival(before: dict[str, dict], after: dict[str, dict], tags=("CB", "UB", "pa", "ts")) -> dict`; `module_of(contig) -> "ERCC" | "SIRV_E0" | "long_SIRV"`; `<lib>/spikein/tag_check.json`.

- [ ] **Step 1: Write the failing test**

```python
from pafbench.spikein import module_of, tag_survival


def test_tag_survival():
    before = {"r1": {"CB": "A", "UB": "u", "pa": 25}, "r2": {"CB": "B"}, "r3": {"CB": "C"}}
    after = {"r1": {"CB": "A", "UB": "u", "pa": 25}, "r2": {"CB": "X"}}
    res = tag_survival(before, after)
    assert res["CB"] == {"n": 2, "kept": 1, "fraction": 0.5}
    assert res["pa"]["fraction"] == 1.0 and res["ts"]["n"] == 0


def test_module_of():
    assert module_of("ERCC-00002") == "ERCC" and module_of("SIRV3") == "SIRV_E0" and module_of("SIRV4001") == "long_SIRV"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_spikein.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pafbench.spikein'`.

- [ ] **Step 3: Write `pafbench/spikein.py`**

```python
import re


def tag_survival(before, after, tags=("CB", "UB", "pa", "ts")):
    """Per tag: among reads present in both and carrying the tag before, how many keep the same value."""
    res = {}
    for t in tags:
        has = [q for q, d in before.items() if t in d and q in after]
        kept = sum(1 for q in has if after[q].get(t) == before[q][t])
        res[t] = {"n": len(has), "kept": kept, "fraction": kept / len(has) if has else None}
    return res


def module_of(contig):
    if contig.startswith("ERCC"):
        return "ERCC"
    if re.fullmatch(r"SIRV[1-7]", contig):
        return "SIRV_E0"
    return "long_SIRV"
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_spikein.py -v`
Expected: 2 passed.

- [ ] **Step 5: Write `scripts/09_spikein_star.sh`**

```bash
#!/usr/bin/env bash
# usage: 09_spikein_star.sh LIB
set -euo pipefail
LIB=$1
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY="$REPO/.venv/bin/python"
BAM=$("$PY" -c "from pafbench.config import LIBRARIES; print(LIBRARIES['$LIB'])")
REF=/data/element_runs/polyafilter_bench/ref/sirv
OUT=/data/element_runs/polyafilter_bench/$LIB/spikein
mkdir -p "$REF" "$OUT"
if [ ! -f "$REF/star/SA" ]; then
  curl -fL -o "$REF/SIRV_Set4.zip" "https://www.lexogen.com/wp-content/uploads/2026/01/SIRV_Set4_Norm_Sequences_20250402.zip" \
    || cp /tmp/claude-1001/-data-element-runs/b3824f3c-a825-42b4-8a9f-c3b00896d118/scratchpad/polyafilter_research/sirv/SIRV_Set4.zip "$REF/SIRV_Set4.zip"
  unzip -o -j "$REF/SIRV_Set4.zip" -d "$REF"
  md5sum "$REF"/*.fasta "$REF"/*.gtf > "$REF/md5.txt"
  mkdir -p "$REF/star"
  STAR --runMode genomeGenerate --runThreadN 8 --genomeDir "$REF/star" \
    --genomeFastaFiles "$REF/SIRV_ERCC_longSIRV_multi-fasta_20250402.fasta" \
    --sjdbGTFfile "$REF/SIRV_ERCC_longSIRV_multi-fasta-annotation_20250402.gtf" \
    --sjdbOverhang 89 --genomeSAindexNbases 8 --outFileNamePrefix "$REF/star_"
fi
samtools view -b -o "$OUT/unmapped.bam" "$BAM" '*'
samtools view "$OUT/unmapped.bam" | cut -f2 | sort | uniq -c > "$OUT/unmapped_flags.txt"
STAR --runThreadN 16 --genomeDir "$REF/star" --readFilesIn "$OUT/unmapped.bam" \
  --readFilesCommand samtools view --readFilesType SAM SE --readFilesSAMattrKeep All \
  --outSAMtype BAM Unsorted --outSAMunmapped None --outFileNamePrefix "$OUT/star_"
```

- [ ] **Step 6: Write `scripts/09b_check_tags.py`**

```python
"""Stage 09b: do CB/UB/pa/ts survive STAR realignment of unmapped reads to SIRV Set 4?"""
import argparse
import json
import random
from collections import Counter

import pysam

from pafbench.config import OUT
from pafbench.spikein import module_of, tag_survival

TAGS = ("CB", "UB", "pa", "ts")


def _tags(a):
    return {t: a.get_tag(t) for t in TAGS if a.has_tag(t)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    base = OUT / args.library / "spikein"
    rng = random.Random(0)
    sample, modules, seen = [], Counter(), 0
    with pysam.AlignmentFile(str(base / "star_Aligned.out.bam")) as fh:
        for a in fh.fetch(until_eof=True):
            if a.is_secondary or a.is_supplementary or a.is_unmapped:
                continue
            modules[module_of(a.reference_name)] += 1
            seen += 1
            item = (a.query_name, _tags(a))
            if len(sample) < args.n:
                sample.append(item)
            else:
                j = rng.randrange(seen)
                if j < args.n:
                    sample[j] = item
    after = dict(sample)
    before = {}
    with pysam.AlignmentFile(str(base / "unmapped.bam"), check_sq=False) as fh:
        for a in fh.fetch(until_eof=True):
            if a.query_name in after:
                before[a.query_name] = _tags(a)
    res = {"aligned_primary": seen, "modules": dict(modules), "survival": tag_survival(before, after),
           "log_final": (base / "star_Log.final.out").read_text()}
    (base / "tag_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["survival"]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run on 1417**

```bash
chmod +x scripts/09_spikein_star.sh
scripts/09_spikein_star.sh 1417
.venv/bin/python scripts/09b_check_tags.py 1417
```
If STAR rejects the SAM input, record the error, realign `samtools fastq` output instead, and let 09b reattach tags by read name (it already looks them up by name). Expected: `unmapped_flags.txt` lists the flags present (record whether any unmapped record has flag 16 set, because STAR then sees the read reverse-complemented); every tag in the printed survival has `fraction` 1.0. Anything below 1.0 means tags must be reattached by read name in milestone 2; record it.

- [ ] **Step 8: Commit**

```bash
git add pafbench/spikein.py tests/test_spikein.py scripts/09_spikein_star.sh scripts/09b_check_tags.py
git commit -m "Spike-in realignment with tag carry-over check"
```

---

### Task 12: BD67 priming sites from the terminal A run

**Files:**
- Create: `pafbench/bd67.py`, `tests/test_bd67.py`, `scripts/10_bd67_ends.py`

**Interfaces:**
- Consumes: `sense_context`, `has_hexamer`, `classify`.
- Produces: `sense_3p_clip(a) -> str`; `leading_a(s) -> int`; `a_run_site(a) -> (site: int, aligned_a: int)`; `/data/element_runs/polyafilter_bench/BD67/ends_check.json`.

Acceptance, fixed now: at least 80% of sampled reads start their 3' clip with 10 or more A, and the hexamer rate at non-templated sites is at least 3 times the rate at templated sites (starting cut-offs 14/8).

- [ ] **Step 1: Write the failing tests**

```python
import pysam

from pafbench.bd67 import a_run_site, leading_a, sense_3p_clip

HDR = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "chr1", "LN": 100000}]})


def _read(pos, cigar, seq, rev):
    a = pysam.AlignedSegment(HDR)
    a.query_name, a.reference_name, a.reference_start = "x", "chr1", pos
    a.cigarstring, a.flag, a.query_sequence = cigar, 16 if rev else 0, seq
    return a


def test_forward_read():
    a = _read(100, "10M6S", "CCCCCAAAAA" + "AAAGGG", False)
    assert sense_3p_clip(a) == "AAAGGG" and leading_a("AAAGGG") == 3
    assert a_run_site(a) == (105, 5)


def test_reverse_read():
    a = _read(200, "4S10M", "CCTT" + "TTTGGGGGGG", True)
    assert sense_3p_clip(a) == "AAGG"
    assert a_run_site(a) == (203, 3)


def test_no_a_run():
    a = _read(100, "10M", "CCCCCCCCCG", False)
    assert a_run_site(a) == (110, 0) and sense_3p_clip(a) == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_bd67.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pafbench.bd67'`.

- [ ] **Step 3: Write `pafbench/bd67.py`**

```python
from .sites import COMP


def sense_3p_clip(a):
    """Soft-clipped bases after the aligned 3' end, in transcript sense (FLNC reads are transcript-oriented)."""
    cig, seq = a.cigartuples, a.query_sequence
    if a.is_reverse:
        n = cig[0][1] if cig[0][0] == 4 else 0
        return seq[:n].translate(COMP)[::-1]
    n = cig[-1][1] if cig[-1][0] == 4 else 0
    return seq[len(seq) - n:] if n else ""


def leading_a(s):
    return len(s) - len(s.lstrip("A"))


def a_run_site(a):
    """Genomic boundary where the read's terminal A run starts (aligned A's included), and how many aligned A's it has."""
    seq = a.query_sequence
    pairs = a.get_aligned_pairs(matches_only=True)
    k, ref_pos = 0, None
    if a.is_reverse:
        for q, r in pairs:
            if seq[q] != "T":
                break
            k, ref_pos = k + 1, r
        return (ref_pos + 1 if k else a.reference_start), k
    for q, r in reversed(pairs):
        if seq[q] != "A":
            break
        k, ref_pos = k + 1, r
    return (ref_pos if k else a.reference_end), k
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_bd67.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write `scripts/10_bd67_ends.py` and run it**

```python
"""Stage 10: confirm how BD67 FLNC reads mark priming sites (chr1 sample)."""
import json

import pandas as pd
import pysam

from pafbench.bd67 import a_run_site, leading_a, sense_3p_clip
from pafbench.config import BD67_BAM, FASTA, OUT
from pafbench.sites import classify, has_hexamer, sense_context


def main(n_max=50000):
    rows = []
    with pysam.AlignmentFile(str(BD67_BAM)) as bam, pysam.FastaFile(str(FASTA)) as fa:
        for a in bam.fetch("chr1"):
            if a.is_secondary or a.is_supplementary or a.mapping_quality < 20:
                continue
            site, k = a_run_site(a)
            down, up = sense_context(fa, "chr1", site, a.is_reverse)
            rows.append([leading_a(sense_3p_clip(a)), k, down.count("A"), has_hexamer(up)])
            if len(rows) >= n_max:
                break
    d = pd.DataFrame(rows, columns=["clip_a", "aligned_a", "a20", "hexamer"])
    d["cls"] = d.a20.map(classify)
    rate = d.groupby("cls").hexamer.mean().to_dict()
    frac_clip = float((d.clip_a >= 10).mean())
    ratio = rate.get("nontemplated", 0) / rate["templated"] if rate.get("templated") else None
    res = {"reads": len(d), "frac_clip_a_ge10": frac_clip, "hexamer_rate": rate, "ratio_nontempl_templ": ratio,
           "class_counts": d.cls.value_counts().to_dict(), "aligned_a_quantiles": d.aligned_a.quantile([0.5, 0.9, 0.99]).to_dict(),
           "accept": bool(frac_clip >= 0.8 and ratio is not None and ratio >= 3)}
    out = OUT / "BD67"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ends_check.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("reads", "frac_clip_a_ge10", "ratio_nontempl_templ", "accept")}, default=float))


if __name__ == "__main__":
    main()
```

Run: `.venv/bin/python scripts/10_bd67_ends.py`
Expected: `accept: true`. If false, BD67 sites are not used in milestone 2 until the read structure is understood; record the numbers.

- [ ] **Step 6: Commit**

```bash
git add pafbench/bd67.py tests/test_bd67.py scripts/10_bd67_ends.py
git commit -m "BD67 priming sites from the terminal A run"
```

---

### Task 13: Milestone 1 results table

**Files:**
- Create: `pafbench/report.py`, `tests/test_report.py`, `scripts/11_m1_summary.py`, `docs/m1_results.md` (generated)

**Interfaces:**
- Consumes: every JSON written above and the GNU time files.
- Produces: `parse_gnu_time(path) -> dict(max_rss_gb, elapsed_s, exit_status)`; `docs/m1_results.md`.

- [ ] **Step 1: Write the failing test**

```python
from pafbench.report import parse_gnu_time


def test_parse_gnu_time(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03\n\tMaximum resident set size (kbytes): 2097152\n\tExit status: 0\n")
    r = parse_gnu_time(p)
    assert r == {"elapsed_s": 3723.0, "max_rss_gb": 2.0, "exit_status": 0}
    p.write_text("\tElapsed (wall clock) time (h:mm:ss or m:ss): 2:03.50\n\tMaximum resident set size (kbytes): 1048576\n\tExit status: 137\n")
    assert parse_gnu_time(p) == {"elapsed_s": 123.5, "max_rss_gb": 1.0, "exit_status": 137}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pafbench.report'`.

- [ ] **Step 3: Write `pafbench/report.py`**

```python
def parse_gnu_time(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line.startswith("Elapsed (wall clock) time"):
            parts = [float(x) for x in line.rsplit(" ", 1)[1].split(":")]
            secs = 0.0
            for p in parts:
                secs = secs * 60 + p
            out["elapsed_s"] = secs
        elif line.startswith("Maximum resident set size (kbytes)"):
            out["max_rss_gb"] = int(line.rsplit(" ", 1)[1]) / 1024 ** 2
        elif line.startswith("Exit status"):
            out["exit_status"] = int(line.rsplit(" ", 1)[1])
    return out
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_report.py -v`
Expected: 1 passed.

- [ ] **Step 5: Write `scripts/11_m1_summary.py` and run it**

```python
"""Stage 11: collect milestone 1 numbers into docs/m1_results.md."""
import json

from pafbench.config import OUT, REPO
from pafbench.report import parse_gnu_time


def _j(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    lines = ["# Milestone 1 results (generated by scripts/11_m1_summary.py)", ""]
    lines += ["## Original polyAfilter on 1417", "", "| Step | Wall time (h) | Max RSS (GB) | Exit |", "|---|---|---|---|"]
    for p in sorted((OUT / "1417" / "original").glob("*.time.txt")) + [OUT / "ref" / "createDB.time.txt"]:
        if p.exists():
            t = parse_gnu_time(p)
            lines.append(f"| {p.name.replace('.time.txt', '')} | {t.get('elapsed_s', 0) / 3600:.2f} | {t.get('max_rss_gb', 0):.1f} | {t.get('exit_status')} |")
    em = _j(OUT / "1417" / "exact_match.json") or {}
    lines += ["", "## Exact match, 1417", "", "| Setting | Original removed | Reimpl removed | Only original | Only reimpl | Identical |", "|---|---|---|---|---|---|"]
    for tag, r in em.items():
        lines.append(f"| {tag} | {r['n_original']:,} | {r['n_reimpl']:,} | {r['only_original']:,} | {r['only_reimpl']:,} | {r['identical']} |")
    coll = _j(OUT / "1417" / "collisions.json") or {}
    cut = _j(OUT / "1417" / "label_cutoffs.json") or {}
    scope = _j(OUT / "scope_check.json") or {}
    tags = _j(OUT / "1417" / "spikein" / "tag_check.json") or {}
    bd = _j(OUT / "BD67" / "ends_check.json") or {}
    ub = _j(OUT / "1179" / "ub_on_introns.json") or {}
    lines += ["", "## Truth set, 1417", "",
              f"- Molecules: {coll.get('molecules')}; member reads: {coll.get('members')}; member cap: {coll.get('cap')} nt",
              f"- Collision background per bp: other chromosomes {coll.get('background_other_refs_per_bp')}, far same chromosome {coll.get('background_far_same_ref_per_bp')}",
              f"- Label cut-offs: templated at A >= {cut.get('templated_min')}, non-templated at A <= {cut.get('nontemplated_max')} (tail rate {cut.get('tail_rate')}, background {cut.get('background_rate')})",
              "", "## Scope check (1416 vs 1417, chr1)", "", f"- go: {scope.get('go')}; molecules: {scope.get('molecules')}",
              "", "## Spike-in tags (1417)", "", f"- {json.dumps(tags.get('survival'))}; modules: {tags.get('modules')}",
              "", "## BD67 ends", "", f"- {json.dumps({k: bd.get(k) for k in ('reads', 'frac_clip_a_ge10', 'ratio_nontempl_templ', 'accept')})}",
              "", "## UB on reads in 1179 (chr1)", "", f"- {json.dumps(ub)}"]
    out = REPO / "docs" / "m1_results.md"
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
```

Run: `.venv/bin/python scripts/11_m1_summary.py`
Expected: `docs/m1_results.md` with every section filled; a `None` marks a stage that has not run.

- [ ] **Step 6: Run the whole test suite**

Run: `.venv/bin/python -m pytest -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add pafbench/report.py tests/test_report.py scripts/11_m1_summary.py docs/m1_results.md
git commit -m "Milestone 1 results table"
```

The milestone exit note (one page, in the user's voice, interpreting `docs/m1_results.md` and listing any change to the spec) is written after this task and reviewed with the user before milestone 2 is planned.
