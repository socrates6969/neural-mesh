#!/bin/bash
# H15 rate grid. usage: run_grid.sh <worker 1|2>   (RAYON/threads: each build is single-threaded; run at most 2 workers)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-C:/Users/mariu/nm-target-r1}
G=$CARGO_TARGET_DIR/release/h15_grid_run.exe
cd /c/Users/mariu/neural-mesh
B=${BUILDS:-3000}
cell() { # tag, then h15_grid args
  tag=$1; shift
  out=results/h15_grid_$tag.csv
  [ -s $out ] && { echo "skip $tag"; return; }
  $G "$@" > $out.tmp && mv $out.tmp $out && echo "done $tag"
}
if [ "$1" = "1" ]; then
  cell base_n500_d32_m16_e200_iid --n 500 --dim 32 --m 16 --efc 200 --mode iid --builds $B --seed0 1000000
  cell n200_d32_m16_e200_iid --n 200 --dim 32 --m 16 --efc 200 --mode iid --builds $B --seed0 2000000
  cell n500_d8_m16_e200_iid --n 500 --dim 8 --m 16 --efc 200 --mode iid --builds $B --seed0 3000000
  cell n500_d128_m16_e200_iid --n 500 --dim 128 --m 16 --efc 200 --mode iid --builds $B --seed0 4000000
  cell n500_d32_m8_e200_iid --n 500 --dim 32 --m 8 --efc 200 --mode iid --builds $B --seed0 5000000
  cell n500_d32_m32_e200_iid --n 500 --dim 32 --m 32 --efc 200 --mode iid --builds $B --seed0 6000000
fi
if [ "$1" = "2" ]; then
  cell n500_d32_m16_e100_iid --n 500 --dim 32 --m 16 --efc 100 --mode iid --builds $B --seed0 7000000
  cell n500_d32_m16_e400_iid --n 500 --dim 32 --m 16 --efc 400 --mode iid --builds $B --seed0 8000000
  cell n500_d32_m16_e200_clust --n 500 --dim 32 --m 16 --efc 200 --mode clust --builds $B --seed0 9000000
  cell n500_d32_m16_e200_dup5 --n 500 --dim 32 --m 16 --efc 200 --mode dup5 --builds $B --seed0 10000000
  cell n500_d32_m16_e200_dup30 --n 500 --dim 32 --m 16 --efc 200 --mode dup30 --builds $B --seed0 11000000
  cell n500_d32_m16_e200_low3 --n 500 --dim 32 --m 16 --efc 200 --mode low3 --builds $B --seed0 12000000
fi
