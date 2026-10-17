#!/bin/bash
# H8 full run, sequential (one process at a time; RAM-guarded inside the binary). Usage: bash results/h8_run_all.sh
export RAYON_NUM_THREADS=2
cd /c/Users/mariu/neural-mesh
B=${B:-C:/Users/mariu/nm-target-r8/h8_run.exe}
O=results
$B tvmem --n 100000 --out $O > $O/h8_tvmem.log 2>&1
$B sweep --out $O > $O/h8_sweep.log 2>&1
$B confirm --out $O > $O/h8_confirm.log 2>&1
$B cross --dim 384 --out $O > $O/h8_cross_d384.log 2>&1
$B cross --dim 128 --out $O > $O/h8_cross_d128.log 2>&1
$B memory --out $O > $O/h8_memory.log 2>&1
$B scale --sizes 100000,300000,1000000 --out $O > $O/h8_scale.log 2>&1
$B conc --out $O > $O/h8_conc.log 2>&1
touch $O/h8_run_all.done
