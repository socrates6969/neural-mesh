#!/bin/bash
# R5-C h11: full reproduction (about 12-15 min on 2 threads). Prelude:
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"
export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r4 RAYON_NUM_THREADS=2
cd /c/Users/mariu/neural-mesh
cargo build --release --offline --bin h11_pipe --bin h11_doc --bin h11_cost 2>&1 | tail -1
B=C:/Users/mariu/nm-target-r4/release
$B/h11_pipe.exe --sens                          > results/h11_pipe.log 2>&1
$B/h11_pipe.exe --sens --held 61-90             > results/h11_pipe_confirm61-90.log 2>&1
$B/h11_pipe.exe --sens --mu 0.1                 > results/h11_pipe_mu0.1.log 2>&1
$B/h11_pipe.exe --sens --mu 0.1 --held 61-90    > results/h11_pipe_mu0.1_confirm61-90.log 2>&1
$B/h11_doc.exe --corpus A --diag                > results/h11_doc_diag.log 2>&1
$B/h11_doc.exe --corpus A                       > results/h11_doc.log 2>&1
$B/h11_doc.exe --corpus B --confirm             > results/h11_doc_confirm61-90.log 2>&1
$B/h11_doc.exe --corpus A --mu 0.1              > results/h11_doc_mu0.1.log 2>&1
$B/h11_doc.exe --corpus B --confirm --mu 0.1    > results/h11_doc_mu0.1_confirm61-90.log 2>&1
$B/h11_cost.exe                                 > results/h11_cost.log 2>&1
echo ALLDONE > results/h11_run_all.done
