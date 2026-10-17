#!/bin/bash
# H8 part 5: N=3e5 HNSW, 3 independent builds per config (1 build each in part 1 -> spread), patched crate then UNPATCHED crate (M16, M32).
export RAYON_NUM_THREADS=2
cd /c/Users/mariu/neural-mesh
O=results
C:/Users/mariu/nm-target-r8/h8_g.exe scale --sizes 300000 --only-hnsw --hnsw-builds 3 --hnsw 16:200:128:f32lib,32:100:32:f32lib,48:100:24:f32lib --tag hnsw3b --out $O > $O/h8_scale_hnsw3b.log 2>&1
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"
W=C:/Users/mariu/nm-target-r8/unpatched_ws
cp src/bin/h8_vectors.rs $W/src/bin/h8_vectors.rs; cp src/bin/h8_common/mod.rs $W/src/bin/h8_common/mod.rs
(cd $W && CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r8/unpatched_target cargo build --release --offline --bin h8_vectors > /c/Users/mariu/neural-mesh/$O/h8_unpatched_build2.log 2>&1)
C:/Users/mariu/nm-target-r8/unpatched_target/release/h8_vectors.exe scale --sizes 300000 --only-hnsw --hnsw-builds 3 --hnsw 16:200:128:f32lib,32:100:32:f32lib,48:100:24:f32lib --tag hnsw3b_unpatched --out $O > $O/h8_scale_hnsw3b_unpatched.log 2>&1
touch $O/h8_run_part5.done
