#!/bin/bash
# H8 part 4 (after part 3): SSD parallel-I/O retest; then A/B against the UNPATCHED registry hnsw_rs 0.3.4 (private workspace copy so the repo
# Cargo.toml/Cargo.lock are never touched): same h8 code, [patch] section removed.
export RAYON_NUM_THREADS=2
cd /c/Users/mariu/neural-mesh
while [ ! -f results/h8_run_part3.done ]; do sleep 20; done
O=results
C:/Users/mariu/nm-target-r8/h8_f.exe ssd --out $O > $O/h8_ssd.log 2>&1
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"
W=C:/Users/mariu/nm-target-r8/unpatched_ws
cp src/bin/h8_vectors.rs $W/src/bin/h8_vectors.rs; cp src/bin/h8_common/mod.rs $W/src/bin/h8_common/mod.rs
(cd $W && CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r8/unpatched_target CARGO_BUILD_JOBS=3 cargo build --release --offline --bin h8_vectors > $OLDPWD/$O/h8_unpatched_build.log 2>&1)
U=C:/Users/mariu/nm-target-r8/unpatched_target/release/h8_vectors.exe
$U tvmem --n 1000 --out $O/unpatched_tmp > /dev/null 2>&1
$U selfcheck --tag unpatched --out $O > $O/h8_selfcheck_unpatched.log 2>&1
$U cross --dim 384 --sizes 5000,10000,30000,100000 --recipes 16:200,32:100 --tag unpatched --out $O > $O/h8_cross_d384_unpatched.log 2>&1
touch $O/h8_run_part4.done
