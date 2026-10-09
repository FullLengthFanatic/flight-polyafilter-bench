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
