#!/bin/bash
# h18 (R9-B domain math tools): rerun everything. ~5 min on the shared box, deterministic. Run ALONE.
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-C:/Users/mariu/nm-target-r4} RAYON_NUM_THREADS=2
cd /c/Users/mariu/neural-mesh || exit 1
B=$CARGO_TARGET_DIR/release
cargo test --release --offline --lib fin:: 2>&1 | tee results/h18_unit_tests.log | tail -3
cargo build --release --offline --bin h18_extract --bin h18_verify --bin h18_lift 2>&1 | tail -1
$B/h18_verify.exe selfcheck --seeds 1-6 > results/h18_selfcheck.log 2>&1
for set in "1-8 tune" "31-60 held" "61-90 confirm61-90"; do
  set -- $set
  $B/h18_verify.exe mutate --seeds $1 --tag $2 > results/h18_verify_$2.log 2>&1
  $B/h18_extract.exe --seeds $1 --tag $2 > results/h18_extract_$2.log 2>&1
done
$B/h18_lift.exe tune --seeds 1-8 > results/h18_tune.log 2>&1
for set in "31-60 held" "61-90 confirm61-90"; do
  set -- $set
  for m in eval controls bugs sens; do
    $B/h18_lift.exe $m --seeds $1 --tag $2 > results/h18_${m}_$2.log 2>&1
  done
done
echo done
