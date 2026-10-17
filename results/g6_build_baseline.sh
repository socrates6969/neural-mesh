#!/usr/bin/env bash
# G6: build the FROZEN BASELINE `neural-mesh serve` (round-2 HEAD 52f936c, serve.rs untouched) into
# $CARGO_TARGET_DIR/g6/neural-mesh-base.exe, from a git-archive snapshot (no git metadata is modified).
#
# NOTE: this compiles into the same $CARGO_TARGET_DIR as the working tree, so it overwrites
# $CARGO_TARGET_DIR/release/neural-mesh.exe with the HEAD build. Run it FIRST, then build the new
# binary and the harness:
#   bash results/g6_build_baseline.sh
#   cargo build --release --offline --bin neural-mesh --bin g6_serve_ab
#   $CARGO_TARGET_DIR/release/g6_serve_ab equiv
# (needs PATH with cargo and the mingw toolchain, RAYON_NUM_THREADS=3, as in the worker environment)
set -euo pipefail
T="${CARGO_TARGET_DIR:?set CARGO_TARGET_DIR}"
REV="${G6_BASE_REV:-52f936c}"
SNAP="$T/g6/snapshot"
rm -rf "$SNAP"
mkdir -p "$SNAP" "$T/g6"
git archive "$REV" | tar -x -C "$SNAP"
( cd "$SNAP" && cargo build --release --offline --bin neural-mesh )
EXE=neural-mesh
[ -f "$T/release/neural-mesh.exe" ] && EXE=neural-mesh.exe
cp "$T/release/$EXE" "$T/g6/${EXE/neural-mesh/neural-mesh-base}"
rm -rf "$SNAP"
echo "baseline: $T/g6/${EXE/neural-mesh/neural-mesh-base}  (serve.rs sha256 of $REV: $(git show "$REV:src/serve.rs" | sha256sum | cut -c1-16)...)"
