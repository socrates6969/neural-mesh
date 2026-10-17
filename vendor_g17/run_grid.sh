#!/bin/bash
# G17 grid: unpatched vs patched hnsw_rs, 20 seeds per cell. Usage: run_grid.sh <mode>
export RAYON_NUM_THREADS=2
H=C:/Users/mariu/nm-target-r7/release/g17_harness.exe
cd /c/Users/mariu/neural-mesh
MODE=${1:-iid}
for cell in "1000 32" "1000 384" "5000 32" "5000 384" "30000 32" "30000 384"; do
  set -- $cell
  out=results/g17_hnsw_${MODE}_n$1_d$2.csv
  [ -s "$out" ] && [ $(wc -l < "$out") -ge 41 ] && continue
  $H --n $1 --dim $2 --mode $MODE --seeds 20 > "$out"
done
echo done > results/g17_grid_${MODE}.done
