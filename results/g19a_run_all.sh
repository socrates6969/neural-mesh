#!/bin/bash
# G19-A (W20) full rerun. Tuning (seeds 1-8) is repeated inside each tuning run and is deterministic (checked: identical JSON on repeat).
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"
export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r8 RAYON_NUM_THREADS=2
cd /c/Users/mariu/neural-mesh
cargo build --release --offline --bin g19_sona_a 2>&1 | tail -1
B=$CARGO_TARGET_DIR/release/g19_sona_a.exe
$B --selfcheck                                                       > results/g19a_selfcheck.log
$B --out results/g19a_sona_a.json                                    > results/g19a_sona_a.log                    # tune 1-8, held 31-60 (~12-15 min)
$B --held 61-90 --out results/g19a_sona_a_confirm61-90.json          > results/g19a_sona_a_confirm61-90.log       # frozen tuning, fresh seeds (~12-15 min)
$B --final --out results/g19a_final_91-120.json                      > results/g19a_final_91-120.log              # no tuning, third fresh block
$B --sens --out results/g19a_sens.json                               > results/g19a_sens.log                      # one-factor sensitivity, 31-60
$B --sens --held 61-90 --out results/g19a_sens_confirm61-90.json     > results/g19a_sens_confirm61-90.log
$B --diag-anchors                                                    > results/g19a_diag_anchors.log
