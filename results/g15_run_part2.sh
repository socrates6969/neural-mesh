#!/usr/bin/env bash
# Part 2 (after tv_build chunk fix to 2048-row adds): rerun quant+scale, add N=3e5 HNSW point, extended M sweep.
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"
export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r6 RAYON_NUM_THREADS=2
cargo build --release --offline --bin g15_vectors || exit 1
B=$CARGO_TARGET_DIR/release/g15_vectors.exe
STEPS=${STEPS:-"quant scale scale300k sweepxl confirmxl"}
for s in $STEPS; do
  case $s in
    quant)      $B quant   --out results > results/g15_quant.md   2> results/g15_quant.log ;;
    scale)      $B scale   --hnsw-build-reps 3 --out results > results/g15_scale.md 2> results/g15_scale.log ;;
    scale300k)  $B scale   --sizes 300000 --hnsw-max 300000 --hnsw-build-reps 1 --tag n300k --out results > results/g15_scale_n300k.md 2> results/g15_scale_n300k.log ;;
    sweepxl)    $B sweep   --m 48,64 --efc 100,200 --tag xl --out results > results/g15_sweep_xl.md 2> results/g15_sweep_xl.log ;;
    confirmxl)  $B confirm --tags xl --out results > results/g15_confirm_xl.md 2> results/g15_confirm_xl.log ;;
  esac
done
echo ALL_DONE > results/g15_run2.done
