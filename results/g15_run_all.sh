#!/usr/bin/env bash
# G15 / W16 full rerun (sequential; ~1.5-2.5 h on a loaded 16 GB box). Run from the crate root.
# Isolation: own target dir, 2 rayon threads. Logs -> results/g15_<mode>.log, JSON -> results/g15_*.json
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"
export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r6 RAYON_NUM_THREADS=2
cargo build --release --offline --bin g15_vectors || exit 1
B=$CARGO_TARGET_DIR/release/g15_vectors.exe
STEPS=${STEPS:-"sweep confirm quant flat conc scale"}
for s in $STEPS; do
  case $s in
    sweep)   $B sweep   --out results > results/g15_sweep.md   2> results/g15_sweep.log ;;
    confirm) $B confirm --out results > results/g15_confirm.md 2> results/g15_confirm.log ;;
    quant)   $B quant   --out results > results/g15_quant.md   2> results/g15_quant.log ;;
    flat)    $B flat    --out results > results/g15_flat.md    2> results/g15_flat.log ;;
    conc)    $B conc    --out results > results/g15_conc.md    2> results/g15_conc.log ;;
    scale)   $B scale   --hnsw-build-reps 3 --out results > results/g15_scale.md 2> results/g15_scale.log ;;
  esac
done
echo ALL_DONE > results/g15_run.done
