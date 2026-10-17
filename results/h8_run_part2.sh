#!/bin/bash
# H8 part 2: bug detector (self-recall ef64/ef512), load-robust interleaved latency, d128 fixed-ef crossover, SSD I/O, concurrent-insert correctness
export RAYON_NUM_THREADS=2
cd /c/Users/mariu/neural-mesh
B=${B:-C:/Users/mariu/nm-target-r8/h8_d.exe}
O=results
$B selfcheck --out $O > $O/h8_selfcheck.log 2>&1
$B lat --out $O > $O/h8_lat.log 2>&1
$B cross --dim 128 --sizes 10000,30000,100000 --recipes 16:200,32:100 --efs-fixed 32,64,128 --tag fixedef --out $O > $O/h8_cross_d128_fixedef.log 2>&1
$B cross --dim 384 --sizes 10000,30000,100000 --recipes 16:200,32:100 --efs-fixed 32,64,128 --tag fixedef --out $O > $O/h8_cross_d384_fixedef.log 2>&1
$B concins --out $O > $O/h8_concins.log 2>&1
$B ssd --out $O > $O/h8_ssd.log 2>&1
touch $O/h8_run_part2.done
