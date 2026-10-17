#!/bin/bash
# R9-C (h19) full study. Deterministic; ~10-15 min on the shared box. Logs: results/h19_*.log, json: results/h19_*.json
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"
export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r5 RAYON_NUM_THREADS=2
cd /c/Users/mariu/neural-mesh
cargo build --release --offline --bin h19_corpus --bin h19_extract --bin h19_down --bin h19_real 2>&1 | tail -1
B=$CARGO_TARGET_DIR/release
# corpora: determinism check + samples
$B/h19_corpus.exe hash --seeds 1-8   | tee results/h19_corpus_hash.log
$B/h19_corpus.exe hash --seeds 31-60 | tee -a results/h19_corpus_hash.log
$B/h19_corpus.exe dump
# tune (seeds 1-8): abstention tau + repair, frozen to results/h19_tune.json
$B/h19_down.exe tune --seeds 1-8 | tee results/h19_tune.log
$B/h19_extract.exe tau --seeds 1-8 > results/h19_tau.log
for S in "1-8 tune" "31-60 held" "61-90 confirm"; do
  set -- $S
  $B/h19_extract.exe eval   --seeds $1 --tag $2          | tee results/h19_extract_$2.log
  $B/h19_extract.exe eval   --seeds $1 --tag $2n --narrow | tee results/h19_extract_$2n.log
  $B/h19_extract.exe inject --seeds $1 --tag $2          | tee results/h19_inject_$2.log
  $B/h19_extract.exe adv    --seeds $1                   | tee results/h19_adv_$2.log
  $B/h19_down.exe eval      --seeds $1 --tag $2          | tee results/h19_down_$2.log
  $B/h19_down.exe controls  --seeds $1 --tag $2          | tee results/h19_controls_$2.log
done
$B/h19_down.exe sens --seeds 31-60 --tag held | tee results/h19_sens_held.log
$B/h19_real.exe --show 12 | tee results/h19_real.log
