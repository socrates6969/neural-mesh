#!/bin/bash
# G17 workaround grid on the UNMODIFIED crate (single-thread). Variants: none, flat02, flat05, djit, rjit
export RAYON_NUM_THREADS=2
H=C:/Users/mariu/nm-target-r7/release/g17_wk.exe
cd /c/Users/mariu/neural-mesh
run() { # n dim seeds wks
  out=results/g17_wk_iid_n$1_d$2.csv
  $H --n $1 --dim $2 --mode iid --seeds $3 --wk $4 | grep -E '^(variant|none|flat02|flat05|djit|rjit|dup|reins),' > "$out"
}
run 1000 32 20 none,flat02,flat05,djit,rjit
run 1000 384 20 none,flat02,flat05,djit,rjit
run 5000 32 20 none,flat02,flat05,djit,rjit
run 5000 384 10 none,flat02,flat05,djit,rjit
run 30000 32 5 none,flat02,djit
run 30000 384 3 none,flat02
echo done > results/g17_wk.done
