#!/usr/bin/env bash
# Part 3: second run of conc + flat crossover (run 1 copies in results/g15_run1/), and sweeps for the confirm of xl.
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r6 RAYON_NUM_THREADS=2
B=$CARGO_TARGET_DIR/release/g15_vectors.exe
$B conc --out results > results/g15_conc.md 2> results/g15_conc.log
$B flat --out results > results/g15_flat.md 2> results/g15_flat.log
echo DONE > results/g15_run3.done
