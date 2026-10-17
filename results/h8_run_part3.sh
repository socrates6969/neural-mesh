#!/bin/bash
# H8 part 3 (runs after part 2): fixed-grid load-robust latency matrix with interleaved lib/avx2 kernels.
export RAYON_NUM_THREADS=2
cd /c/Users/mariu/neural-mesh
while [ ! -f results/h8_run_part2.done ]; do sleep 20; done
B=${B:-C:/Users/mariu/nm-target-r8/h8_e.exe}
O=results
$B lat2 --out $O > $O/h8_lat2.log 2>&1
touch $O/h8_run_part3.done
