#!/usr/bin/env bash
# H10: build the FROZEN BASELINE `neural-mesh serve` (round-2 HEAD 52f936c) into $CARGO_TARGET_DIR/h10/neural-mesh-base.exe from a git-archive
# snapshot (no git metadata modified).  Same procedure as results/g6_build_baseline.sh but a different output dir.  NOTE: it overwrites
# $CARGO_TARGET_DIR/release/neural-mesh.exe with the HEAD build, so run it FIRST and rebuild the working tree afterwards:
#   bash results/h10_build_baseline.sh && cargo build --release --offline --bin neural-mesh --bin h10_serve_ab
set -euo pipefail
T="${CARGO_TARGET_DIR:?set CARGO_TARGET_DIR}"
REV="${H10_BASE_REV:-52f936c}"
SNAP="$T/h10/snapshot"
rm -rf "$SNAP"; mkdir -p "$SNAP" "$T/h10"
git archive "$REV" | tar -x -C "$SNAP"
( cd "$SNAP" && cargo build --release --offline --bin neural-mesh )
EXE=neural-mesh; [ -f "$T/release/neural-mesh.exe" ] && EXE=neural-mesh.exe
cp "$T/release/$EXE" "$T/h10/${EXE/neural-mesh/neural-mesh-base}"
rm -rf "$SNAP"
echo "baseline: $T/h10/${EXE/neural-mesh/neural-mesh-base} (serve.rs of $REV sha256 $(git show "$REV:src/serve.rs" | sha256sum | cut -c1-16)...)"
