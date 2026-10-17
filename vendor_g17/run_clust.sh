#!/bin/bash
# G17 secondary: clustered (embedding-like) data, unpatched vs patched, 20 seeds, N in {1k,5k} x dim {32,384}
export RAYON_NUM_THREADS=2
H=C:/Users/mariu/nm-target-r7/release/g17_harness.exe
cd /c/Users/mariu/neural-mesh
for cell in "1000 32" "1000 384" "5000 32" "5000 384"; do
  set -- $cell
  $H --n $1 --dim $2 --mode clust --seeds 20 > results/g17_hnsw_clust_n$1_d$2.csv
done
echo done > results/g17_clust.done
