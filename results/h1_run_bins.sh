#!/bin/bash
# re-run the existing HNSW bins in a scratch cwd (they write results/*.json relative to cwd; do not clobber the repo's results)
export RAYON_NUM_THREADS=2
T=C:/Users/mariu/nm-target-r1
for L in patched unpatched; do
  D=$T/run_$L; mkdir -p $D/results; cd $D
  echo "== memory_bench --quick ($L)"; $T/memory_bench_$L.exe --quick > $D/memory_bench.log 2>&1; cp $D/results/memory.json /c/Users/mariu/neural-mesh/results/h1_memory_bench_$L.json
  echo "== r3_store --n 10000 ($L)"; $T/r3_store_$L.exe --n 10000 --out $D/results/r3.json > $D/r3_store.log 2>&1; cp $D/results/r3.json /c/Users/mariu/neural-mesh/results/h1_r3_store_$L.json
  cp $D/memory_bench.log /c/Users/mariu/neural-mesh/results/h1_memory_bench_$L.log; cp $D/r3_store.log /c/Users/mariu/neural-mesh/results/h1_r3_store_$L.log
done
echo done > /c/Users/mariu/neural-mesh/results/h1_bins.done
