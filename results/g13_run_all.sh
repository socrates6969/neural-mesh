#!/bin/bash
# Reproduces every g13 result. Run from the repo root; ~5-10 min total on the shared box.
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"
export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r2 RAYON_NUM_THREADS=2
cd /c/Users/mariu/neural-mesh
cargo build --release --offline --bin g13_tools || exit 1
B=$CARGO_TARGET_DIR/release/g13_tools.exe
$B selfcheck --seeds 1-6 --out results/g13_selfcheck.json | tee results/g13_selfcheck.log
$B tune --seeds 1-8 --out results/g13_tune.json | tee results/g13_tune.log
$B eval --seeds 31-60 --tag held | tee results/g13_held.log
$B controls --seeds 31-60 --tag held | tee results/g13_controls_held.log
$B sens --seeds 31-60 --tag held | tee results/g13_sens_held.log
$B eval --seeds 61-90 --tag confirm61-90 | tee results/g13_confirm61-90.log
$B controls --seeds 61-90 --tag confirm61-90 | tee results/g13_controls_confirm61-90.log
$B sens --seeds 61-90 --tag confirm61-90 | tee results/g13_sens_confirm61-90.log
