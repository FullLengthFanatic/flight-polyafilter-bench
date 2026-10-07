#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=/data/element_runs/polyafilter_bench/ref
mkdir -p "$OUT"
cd "$REPO/vendor/polyAfilter"
/usr/bin/time -v -o "$OUT/createDB.time.txt" "$REPO/.venv/bin/python" polyAfilter.py createDB -v \
  /home/simone.picelli/Documents/Genome_ref/refdata-gex-GRCh38-2024-A/genes/genes.gtf \
  "$OUT/genes_2024A.db" > "$OUT/createDB.log" 2>&1
