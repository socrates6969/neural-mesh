#!/bin/bash
export RAYON_NUM_THREADS=2
T=C:/Users/mariu/nm-target-r1
cd /c/Users/mariu/neural-mesh
$T/h1_cache_run.exe > results/h1_cache.log 2>&1
$T/h1_embed_run2.exe --part time > results/h1_embed_timing.log 2>&1
echo done > results/h1_timing.done
