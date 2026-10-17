#!/bin/bash
# H21 full rerun (about 2 min per run on 2 threads; tuning on seeds 1-8 is recomputed inside, deterministic)
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-C:/Users/mariu/nm-target-r7} RAYON_NUM_THREADS=2
cd /c/Users/mariu/neural-mesh
cargo test --release --offline --bin h21_research
cargo build --release --offline --bin h21_research
B=$CARGO_TARGET_DIR/release/h21_research.exe
$B --show > results/h21_world_examples.txt
$B --out results/h21_research.json > results/h21_research.log                                              # held 31-60
$B --held 61-90 --out results/h21_research_confirm61-90.json > results/h21_research_confirm61-90.log         # fresh seeds, frozen tuning
$B --extra --out results/h21_extra.json > results/h21_extra.log                                           # post-hoc extra rare/expensive controls
$B --extra --held 61-90 --out results/h21_extra_confirm61-90.json > results/h21_extra_confirm61-90.log
$B --mu 0.1 --out results/h21_research_mu0.1.json > results/h21_research_mu0.1.log                          # strict exchange rate, retuned
$B --mu 0.1 --held 61-90 --out results/h21_research_mu0.1_confirm61-90.json > results/h21_research_mu0.1_confirm61-90.log
$B --sens --out results/h21_sens.json > results/h21_sens.log                                               # one-at-a-time sensitivity
$B --sens --held 61-90 --out results/h21_sens_confirm61-90.json > results/h21_sens_confirm61-90.log
