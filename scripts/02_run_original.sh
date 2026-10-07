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
