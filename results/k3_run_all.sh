#!/bin/bash
# K3 rerun (python only; no cargo build; no network; ~8-10 min; deterministic: corpus fnv 1744236539a5b5fd must reproduce)
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
python py/k3_main.py all 2>&1 | tee results/k3_all.stdout.txt
