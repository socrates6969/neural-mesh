#!/bin/bash
# H1 part 1 grid. patched = repo build (Cargo.toml [patch]); unpatched = copy of the repo under nm-target-r1/copy without [patch]
export RAYON_NUM_THREADS=2
T=C:/Users/mariu/nm-target-r1
cd /c/Users/mariu/neural-mesh
for cell in "2000 32 iid 20" "2000 32 clust 20" "5000 384 iid 8" "5000 128 clust 8"; do
  set -- $cell
  for L in patched unpatched; do
    $T/h1_hnsw_$L.exe --label $L --n $1 --dim $2 --mode $3 --seeds $4 > results/h1_hnsw_${L}_n$1_d$2_$3.csv
  done
done
echo done > results/h1_hnsw.done
