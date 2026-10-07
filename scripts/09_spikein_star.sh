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
