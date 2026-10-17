#!/bin/bash
# W22 (TOOLS round 2) full study. Run ALONE on the box (~15 min). Logs in results/g21_*.log, json in results/g21_*.json
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"
export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r2 RAYON_NUM_THREADS=2
cd /c/Users/mariu/neural-mesh
cargo build --release --offline --bin g21_extract --bin g21_verify --bin g21_headline 2>&1 | tail -2
B=$CARGO_TARGET_DIR/release
# (1) extractor accuracy by phrasing difficulty
$B/g21_extract.exe eval --seeds 1-8   --tag tune          | tee results/g21_extract_tune.log
$B/g21_extract.exe eval --seeds 31-60 --tag held          | tee results/g21_extract_held.log
$B/g21_extract.exe eval --seeds 61-90 --tag confirm61-90  | tee results/g21_extract_confirm61-90.log
# (2) independent verifier: selfcheck + mutation testing of the evaluator
$B/g21_verify.exe selfcheck --seeds 1-6                   | tee results/g21_verify_selfcheck.log
$B/g21_verify.exe mutate --seeds 1-8   --tag tune         | tee results/g21_verify_tune.log
$B/g21_verify.exe mutate --seeds 31-60 --tag held         | tee results/g21_verify_held.log
$B/g21_verify.exe mutate --seeds 61-90 --tag confirm61-90 | tee results/g21_verify_confirm61-90.log
# (3) headline: tune (seeds 1-8), held 31-60, confirm 61-90 (frozen hx() in g21_headline.rs)
$B/g21_headline.exe tune --seeds 1-8                      | tee results/g21_tune.log
for S in "31-60 held" "61-90 confirm61-90"; do
  set -- $S
  $B/g21_headline.exe eval     --seeds $1 --tag $2 | tee results/g21_$2.log
  $B/g21_headline.exe controls --seeds $1 --tag $2 | tee results/g21_controls_$2.log
  $B/g21_headline.exe bugs     --seeds $1 --tag $2 | tee results/g21_bugs_$2.log
  $B/g21_headline.exe sens     --seeds $1 --tag $2 | tee results/g21_sens_$2.log
done
