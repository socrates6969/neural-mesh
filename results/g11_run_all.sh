#!/bin/bash
# Reruns every G11 result with the final binary (~10-12 min each; deterministic except the ns/decision timings).
# build: export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r4; cargo build --release --offline --bin g11_team
export RAYON_NUM_THREADS=3
cd /c/Users/mariu/neural-mesh
B=C:/Users/mariu/nm-target-r4/release/g11_team.exe
$B --set held    --out results/g11_team_held.json                                   > results/g11_team_held.log 2>&1
$B --set confirm --out results/g11_team_confirm61-90.json                           > results/g11_team_confirm61-90.log 2>&1
$B --set held    --tune perworld --out results/g11_team_held_perworld.json          > results/g11_team_held_perworld.log 2>&1
$B --set confirm --tune perworld --out results/g11_team_confirm61-90_perworld.json  > results/g11_team_confirm61-90_perworld.log 2>&1
$B --set held --noise 0.4 --notime --out results/g11_team_held_noise0.4.json       > results/g11_team_held_noise0.4.log 2>&1
$B --set held --noise 0.1 --notime --out results/g11_team_held_noise0.1.json        > results/g11_team_held_noise0.1.log 2>&1
echo finished > results/g11_run_all.done
