# H8 / R4-V (MAX-VECTORS round 2) board entry

Pure retrieval infrastructure; no model-IQ claim. Every number is **synthetic-until-validated-on-real-traffic**: "synth384" = the memory_bench hierarchical Gaussian mixture
(64 topics x 16 subs); "real-mini"/"real-bge" = real MiniLM-384 / BGE-768 embeddings of the g1 corpora, but those corpora are SYNTHETIC TEMPLATED text
(23,153 / 23,453 unique rows after exact-duplicate removal, near-duplicate heavy, recall plateaus at ~1.0). No real corpus >= 1e5 exists in `results/`.
Code (mine): `src/bin/h8_vectors.rs`, `src/bin/h8_common/mod.rs` (copy of g15_common + extensions: counting allocator, streaming data/ground truth, custom `Distance`
on block-fp16 / int8 points, file-backed rerank store, RaBitQ-style codes). Nothing in `src/*.rs`, Cargo.toml or Cargo.lock was edited.
Proposals: `results/h8_proposed_patch.md`. All JSON/MD/logs: `results/h8_*` (tables regenerated with `results/h8_report*.py`, stdlib python).
Run scripts: `results/h8_run_all.sh`, `h8_run_part2.sh` ... `h8_run_part5.sh`.

Setup: release build, default target (AVX2/FMA kernels are runtime-dispatched), own `CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r8`, `RAYON_NUM_THREADS=2`, `--offline`.
The crate is the FIXED hnsw_rs (Cargo `[patch]` -> vendor_g17/hnsw_rs_patched) unless a block says "unpatched" (then a private workspace copy under
`nm-target-r8/unpatched_ws` with the `[patch]` section removed was built, so the repo Cargo files were never touched).
Latency: one query at a time on ONE thread (turbovec / RaBitQ inside a 1-thread rayon pool); recall@10 = id overlap with exact f32 brute force on the same normalised vectors.
Bytes/vector = delta of a counting global allocator (exact heap) around the build, `shrink_to_fit` applied to my own Vecs; not RSS.
HNSW recipe as in `src/memory.rs` (cosine = 1-dot, `parallel_insert_slice` in 2048-row batches on 2 threads; build is NON-deterministic: recall spread over builds is reported).

## MACHINE-LOAD CAVEAT / load probe (read first)
16-core box shared with other workers. Each JSON stores `calib_ms_fixed_scan` (fixed in-cache 20x4000x384 AVX2 scan, median of 15; quiet = 1.2-2.0 ms).
Values at the END of each run: sweeps 1.25-1.36 (start 1.96-2.06), **confirm 5.0 / 5.9 / 2.3 (3-4x slow: latencies of `h8_hnsw_confirm_*` are NOT usable, only their recall/self-recall)**,
cross d384 2.0 / d128 2.0 / fixed-ef 1.2-1.95, memory 1.23-1.29, scale 1.26 (start 1.25-1.28), conc 1.55 -> 2.05, lat/lat2 1.2-1.8, ssd 1.26-1.62. To make the final latency numbers load-robust I added
`lat2`: 3 builds per config, lib/avx2 kernels interleaved pass by pass, latency = **p50/p99 of the per-query MINIMUM over 7 passes** (a preemption spike rarely hits the same query
in all passes; deterministic slow queries remain). These "p99(min-of-7)" values are what the headline uses; raw per-pass p99 is also stored (`p99_raw_mean_us`).
Recall, bytes/vector and ratios are load-independent; absolute latencies elsewhere carry +-30-50 %.

## HEADLINE (all synthetic / templated-corpus; 1 thread; d=384 unless stated)
1. **Fixed crate, HNSW**: recall now saturates at 1.000 (old unpatched ceiling 0.995-0.999 on the same data) and the hnsw_rs floor `ef >= k = 10` is the knee.
   recall@10 >= 0.99 at **p99 ~ 120-180 us, p50 ~ 65-97 us** (MiniLM N=23k: M24/efc100/ef16 recall 0.993 p99 121 us; BGE-768 N=23k: M32/efc200/ef10 0.994 p99 145 us; synth N=30k: M16/efc200/ef16 0.996 p99 180 us),
   vs g15 (unpatched, one build, loaded box) 0.23-0.5 ms for the same target. Memory 5.9-8.8 KB/vector.
2. **Self-recall check passes on the patched crate and fails on the unpatched one**: at ef=512 (k=10) 0 of 20-30k items are unfindable by their own vector for every efc>2M config on synthetic and MiniLM data
   (3 builds each) vs 0.4-1.7 % lost unpatched (same code, A/B). Residual on real BGE: 16-19 / 8-9 of 23,453 (0.07 % / 0.04 %) at efc 100 (M16 / M32); 0 at efc 200. **New trap: efc <= 2M breaks the graph (3-42 % unreachable)**.
3. **Flat vs HNSW**: d=384 exact flat wins/ties only up to N ~ 5k; HNSW 2.5-3.4x faster at 10k, 7-10x at 30k, 7-18x at 1e5 (recall >= 0.99). d=128: flat wins to ~10k, HNSW 3-6x at 30k.
   The old "flat fastest at 30k" is NOT reproduced by anyone: it is a misreading of H2 (see crossover block); H2's own numbers (flat 340 us, HNSW 1.3-2.5x faster at ef 32-128, d=128, N=30k) reproduce exactly (flat 341 us; HNSW 2.4-2.6x / 1.5x / 0.9-1.07x at ef 32/64/128).
4. **N=3e5 recall with larger M** (3 builds each): M16 efc200 stays bad (0.76-0.89 at ef 32-256), M32/efc100 reaches 0.964 / 0.993 at ef 128 / 256, M48/efc100 0.986 / 0.996 at ef 128 / 256
   (p99 1.2 ms / 1.7-2.0 ms, 6.9-7.3 KB/vec = 2.1-2.2 GB for 3e5). Larger M (with ef >= 128) fixes it; the fixed crate alone does not (M16 patched vs unpatched: see block 4).
5. **Max vectors per GB, MEASURED** (index heap; f32 rerank copy counted if in RAM): turbovec 2-bit 100 B/vec (9.96 M/GB), turbovec **3-bit = 196 B/vec (same as 4-bit; g15's nominal 148 B was wrong)**, 4-bit 196 B (5.1 M/GB),
   int8 flat 388 B (2.58), block-fp16 flat 816 B (1.23, exact recall 1.000), f32 flat 1536 B (0.65), HNSW 5.1-8.7 KB (0.11-0.20 M/GB). With the exact f32 rows on SSD: turbovec 4-bit + x4 rerank = 196-207 B RAM/vec (4.8-5.1 M/GB) at recall 0.999-1.000 up to N=1e6,
   but p50 5.7-6.2 ms (N=3e5, 1 I/O thread; 3.9 ms with 4 I/O handles) / 14.6 ms (N=1e6) vs 0.9-3 ms with the rows in RAM.
6. HNSW with block-fp16 *stored* points (custom `Distance<i16>`): -10..-18 % bytes (6.1 vs 6.9 KB at 1e5), same recall, same/faster; int8 points cap recall at 0.95-0.97 (do not use).
7. Multi-bit RaBitQ-style codes (my std-only implementation): **no gain over turbovec per code bit** (e.g. N=1e6 4-bit 0.755 vs 0.772; +rerank x4 0.997 vs 0.999) and a slower scan; its only advantage is genuine 3-bit packing (148.6 vs 207 B/vec). Negative result.
8. Concurrency: HNSW (shared, no lock) scales 3.9x on 4 reader threads (4,657 -> 18,349 qps, p50 flat ~200 us) and one concurrent inserter (2.2-2.3k inserts/s) does not slow readers; 2 writer threads inserting 6,000 vectors during reads lose 0 of them (own-vector top-1 check) and keep recall@10 0.993.

---
### VERIFY REQUEST — H8 — HNSW M x efc x ef on the fixed crate: Pareto and best recall at fixed latency
Rerun (build ~1 min; sweep ~50 min, lat2 ~35 min; fixtures from `results/`, override `G15_FIXTURES`):
```
cd C:\Users\mariu\neural-mesh ; export PATH=... (rust+mingw) CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r8 RAYON_NUM_THREADS=2
cargo build --release --offline --bin h8_vectors ; B=$CARGO_TARGET_DIR/release/h8_vectors.exe
$B selftest                                   # stream==synth_ds, custom dots, rabitq estimator, file store, self-recall sanity (1 min)
$B sweep --out results/h8_verify              # M{16,24,32,48,64} x efc{100,200,400} x ef{10..256}, 3 datasets, 1 build per cell, + self-recall per build
$B lat2  --out results/h8_verify              # 3 builds per (M{16,24,32,48}, efc{100,200} with efc>=2M), ef{10,16,24,32}, lib vs avx2 kernel interleaved, per-query-min latency
python results/h8_report.py sweep ; python results/h8_report_lat.py lat2
```
Files: `h8_hnsw_sweep_{synth384-n30000,real-mini,real-bge}.json` + `h8_sweep.md`; `h8_lat2_*.json` + `h8_lat2.md`; `h8_hnsw_confirm_*.json` + `h8_confirm.md` (3 rebuilds of the sweep picks; latency contaminated by load, recall valid).
**Minimum p99 at a recall target (lib kernel, mean over 3 builds; p99 = per-query-min stat)**:
| dataset | recall >= | min p99 us | config M/efc/ef | p50 us | recall | heap B/vec |
|---|---|---|---|---|---|---|
| synth30k | 0.95 | 143 | 24/100/10 | 70 | 0.953 | 6503 |
| synth30k | 0.98 | 167 | 24/200/10 | 101 | 0.981 | 6686 |
| synth30k | 0.99 / 0.995 | 180 | 16/200/16 | 97 | 0.996 | 6115 |
| mini23k | 0.95 / 0.98 | 93 | 24/100/10 | 50 | 0.980 | 6284 |
| mini23k | 0.99 | 121 | 24/100/16 | 65 | 0.993 | 6284 |
| mini23k | 0.995 | 145 | 32/200/16 | 83 | 0.995 | 6695 |
| bge23k (768-d) | 0.95 / 0.98 | 134 | 32/100/10 | 71 | 0.983 | 8113 |
| bge23k | 0.99 | 145 | 32/200/10 | 87 | 0.994 | 8302 |
| bge23k | 0.995 | 186 | 48/100/16 | 100 | 0.995 | 8571 |
**Best recall at a fixed p99 budget**: <=150 us: synth30k 0.969 (16/200/10), mini 0.996 (24/100/24, p99 149), bge 0.994 (32/200/10, p99 145); <=200 us: 0.998 (32/200/16) / 0.998 (24/100/32) / 0.999 (32/200/16);
<=300 us: 1.000 / 0.999 / 1.000. (Requiring all three builds >= target changes the pick only slightly, see `h8_lat2.md` last column.)
**Pareto frontiers of the single-build sweep** (`h8_sweep.md`; p99 = mean of 3 passes under the then-load, noisy +-20 %):
synth30k 0.983 @245 us (24/200/10), 0.988 @279 (48/100/16), 0.998 @299 (24/200/16), 1.000 @343 (48/100/24);
mini 0.978 @105 (48/100/10), 0.989 @145 (32/200/10), 0.994 @158 (32/100/16), 0.999 @241 (32/100/32), 1.000 @303 (32/100/48);
bge 0.970 @178 (16/100/10), 0.995 @189 (64/400/10), 0.999 @247 (64/400/16), 1.000 @355 (24/100/32).
Findings (measured): (1) ef=10 is the hnsw_rs floor and already gives 0.95-0.995; ef 16-24 reaches 0.99-0.999; ef >= 32 only buys latency (+25-40 us p50 per +8 ef). (2) M matters less than before: with the fix M16/efc200 reaches 0.996 (synth) / 0.994 (mini) at ef 16;
M24-32 is the sweet spot; M48 costs +12 % bytes and +5-15 % p50 for <= +0.002 recall (M64 more). (3) efc 200 beats 100 at the smallest ef (recall at ef 10 +0.006..+0.074, largest for M16 on synth30k: 0.895 -> 0.969) at 2.3-2.6x build time; efc 400 appears on the single-build frontier only for M24/M64 on mini/bge (4-8x build cost) and was not retested in lat2. (4) AVX2 kernel (g15 P1, still not in `memory::dot`): paired avx2/lib p50 ratio 0.82-0.92 (median 0.87), p99 -10..-20 %, recall unchanged.
(5) recall plateau is 1.000 (g15: 0.995-0.999) and recall at ef 64/efc 200 is 0.999-1.000 for every M>=16 on all three sets.
Negatives: single-build sweep points are noisy (rebuild spread of recall at ef 10 is +-0.01-0.02, p99 +-20 %); sweep p99 of the `confirm` rebuilds (3.7 ms etc.) are load artefacts (calib 5-6 ms) - use `lat2`.
Vs g15 (unpatched, same data, one loaded build): recall >= 0.99 needed M32-48 / efc200 / ef16 at p99 0.23-0.5 ms; now M16-24 at 0.12-0.18 ms. Part of the gain is load (g15 box was at 100 % CPU), but the recall-vs-ef curve moved up by itself (ef 10: 0.96-0.97 -> 0.97-0.995 for M>=24).

---
### VERIFY REQUEST — H8 — self-recall (own vector must be its own top-1) on patched vs unpatched crate; the efc <= 2M trap
Rerun (~25 min patched, + ~7 min first build of the unpatched workspace):
```
$B selfcheck --out results/h8_verify                      # M16/efc100, M16/efc200, M32/efc100; synth30k, mini, bge; 3 builds; ef 64 and ef 512; EVERY item queried
$B selfcheck --recipes 64:100,64:128,32:50,32:64,48:80,48:96,24:48 --builds 2 --data synth384:30000,mini --tag efc_lt_2m --out results/h8_verify
# unpatched A/B (private copy, repo untouched): mkdir nm-target-r8/unpatched_ws; copy Cargo.toml (delete the [patch.crates-io] block at the end), Cargo.lock, src/*.rs, src/bin/h8_vectors.rs, src/bin/h8_common/;
#   cd there; CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r8/unpatched_target cargo build --release --offline --bin h8_vectors ; run the same two selfcheck commands with --tag unpatched
python results/h8_report.py selfcheck ; python results/h8_report.py selfcheck _unpatched ; python results/h8_report.py selfcheck _efc_lt_2m     (all in results/h8_selfcheck.md)
```
Items not their own top-1 (k=10, n = every stored vector; per build):
| dataset | M/efc | patched ef64 | **patched ef512** | UNPATCHED ef64 | **UNPATCHED ef512** |
|---|---|---|---|---|---|
| synth30k | 16/100 | 27, 28, 18 | **0, 0, 0** | 493, 434, 508 | 465, 414, 483 (1.4-1.6 %) |
| synth30k | 16/200 | 0, 0, 0 | **0, 0, 0** | 478, 514, 482 | 468, 500, 475 (1.6 %) |
| synth30k | 32/100 | 1, 1, 0 | **0, 0, 0** | 192, 143, 162 | 181, 133, 154 (0.5 %) |
| mini23k | 16/100 | 3, 6, 3 | **0, 0, 0** | 390, 413, 347 | 368, 389, 332 (1.5 %) |
| mini23k | 16/200 | 0, 1, 0 | **0, 0, 0** | 366, 353, 382 | 347, 340, 370 (1.5 %) |
| mini23k | 32/100 | 4, 3, 2 | **0, 0, 0** | 140, 159, 157 | 134, 142, 146 (0.6 %) |
| bge23k | 16/100 | 54, 23, 43 | **19, 17, 16** | 412, 357, 395 | 366, 324, 356 (1.5 %) |
| bge23k | 16/200 | 10, 13, 9 | **0, 0, 0** | 398, 365, 405 | 362, 335, 371 (1.5 %) |
| bge23k | 32/100 | 24, 31, 27 | **9, 8, 9** | 174, 182, 209 | 151, 146, 169 (0.6-0.7 %) |
Reading: ef-64 misses of the patched crate are ordinary ANN approximation (all but 0-0.08 % vanish at ef 512, as g17 found). The bug signature (misses that survive ef 512, "absent from top-10" == "not top-1") is
1.4-1.6 % at M16 (6 % level>=1 points, 25 % of them unreachable) and absent in the patched crate at every efc > 2M on synthetic and MiniLM data. Residual on the real BGE set at efc 100 is 0.04-0.08 % (not the reverse-edge bug - unpatched is 1.5 %; efc 200 gives 0).
**Trap (new, independent of the patch)**: configs with `efc <= 2M` (hnsw_rs level-0 lists hold 2M entries) leave huge parts of the graph unreachable *even at ef 512*, worst exactly at efc = 2M:
synth30k M24/efc48 12,197 / 30,000 lost (41 %), M32/efc64 8,707-10,934 (29-36 %), M32/efc50 4,639 (15 %), M48/efc96 9,846-12,519 (33-42 %), M48/efc80 3,133-4,253 (10-14 %), M64/efc128 11,241-12,652 (37-42 %), M64/efc100 1,398-1,628 (5 %);
MiniLM the same configs lose 0.01-6 % at ef 512 (e.g. M24/efc48 603-1,379; M32/efc64 214-305). All shipped defaults satisfy efc > 2M (`HnswParams` M16/efc200; tiered test M8/efc50), but any tuning to "M64/efc100" or "M32/efc64" silently loses items. Proposal Q1: clamp `efc >= 3M`.
Negatives: 3 builds per cell, so rare-event rates (0.04 %) have wide intervals; the selfcheck uses cosine on unit vectors only; I did not look into hnsw_rs to explain the efc<=2M behaviour (empirical only).

---
### VERIFY REQUEST — H8 — flat-vs-HNSW crossover (d=384 and d=128), why the R3 result differed from "flat fastest at 30k"
Rerun (~10 min each):
```
$B cross --dim 384 --sizes 5000,10000,30000,100000 --recipes 16:200,32:100 --tag calm --out results/h8_verify
$B cross --dim 128 --sizes 10000,30000,100000 --recipes 16:200,32:100 --efs-fixed 32,64,128 --tag fixedef --out results/h8_verify
$B cross --dim 384 --sizes 10000,30000,100000 --recipes 16:200,32:100 --efs-fixed 32,64,128 --tag fixedef --out results/h8_verify
(unpatched binary, same command with --tag unpatched, for the unpatched column)
```
Files `h8_cross_d384_calm.json` (quiet box, calib 1.2-2 ms), `h8_cross_d384_unpatched.json`, `h8_cross_d128*.json`, `h8_cross_d384_fixedef.json`, logs `h8_cross_*.log`.
p50 us, mean of 3 passes, one HNSW graph grown through the N list, HNSW = smallest ef (from 10,16,24,32,48,64,96,128,...512) reaching the recall target (ef in parentheses):
d=384, fixed crate, quiet box
| N | flat f32 AVX2 | flat f32 lib kernel | flat block-fp16 | int8 scan+f32 rerank x10 | turbovec4+rerank x4 | M16/efc200 r>=.95 | M16/efc200 r>=.99 | M32/efc100 r>=.95 | M32/efc100 r>=.99 |
|---|---|---|---|---|---|---|---|---|---|
| 5,000 | 77 | 140 | 73 | 66 | 95 | 58 (10) | 71 (16) | 56 (10) | 76 (16) |
| 10,000 | 183 | 294 | 158 | 121 | 156 | 71 (10) | 72 (10) | 54 (10) | 55 (10) |
| 30,000 | 731 | 915 | 650 | 295 | 389 | 76 (10) | 100 (16) | 71 (10) | 90 (16) |
| 100,000 | 3187 | 3526 | 2335 | 941 | 655 | 216 (48) | 462 (128) | 172 (32) | 311 (64) |
d=384, UNPATCHED crate (same code/box state): M16/efc200 never reaches 0.99 (ef <= 512) at any N; M32/efc100 r>=.99 needs ef 16 / 16 / 32 / 96 and p50 82 / 98 / 147 / 370 us at N = 5k / 10k / 30k / 100k
(patched: ef 16 / 10 / 16 / 64 and 76 / 55 / 90 / 311 us) => at a fixed recall target of 0.99 the fix makes HNSW 1.1-1.8x faster (and M16 usable at all); r>=.95: 63 / 90 / 111 / 205 (M16) and 66 / 72 / 74 / 164 (M32) us unpatched vs 58 / 71 / 76 / 216 and 56 / 54 / 71 / 172 patched (no change, as expected).
d=128 (quiet box, `h8_cross_d128_fixedef`): flat AVX2 p50 58 / 341 / 1548 us at N = 10k / 30k / 100k; HNSW at fixed ef (M32/efc100, recall 1.000 at 10-30k, 0.975-1.000 at 1e5):
ef 32: 170 / 134 / 225 us, ef 64: 288 / 227 / 334 us, ef 128: 471 / 371 / 565 us => flat/HNSW = 0.34 / 2.56 / 6.9 (ef 32), 0.20 / 1.50 / 4.6 (ef 64), 0.12 / 0.92 / 2.7 (ef 128).
Min-ef rows (r>=.95 / r>=.99, p50 us): N=10k 84 (ef 10) / 93; 30k 64 (10) / 90 (16); 1e5 158 (24) / 391 (96) vs flat 58 / 341 / 1548.
d=384 fixed ef (`h8_cross_d384_fixedef`): flat AVX2 316 / 850 / 3365 us (10k / 30k / 1e5); M32/efc100 ef 32: 170 / 190 / 188 us, ef 64: 259 / 315 / 258 us, ef 128: 490 / 482 / 453 us.
**Crossover (measured): d=384: exact flat (AVX2 f32 or block-fp16) wins or ties up to N ~ 5k (77 vs 56-76 us), HNSW wins from ~8-10k (2.5-3.4x at 10k, 7-10x at 30k, 7-18x at 1e5 at recall >= .95-.99); the int8-scan + f32-rerank flat
(recall 1.000) is the best exact option between 5k and ~10k (121 us at 10k); d=128: flat wins to ~10-15k (58 vs 84-137 us at 10k), HNSW from ~20k (2.0-2.5x at 20k, 3.8-5.9x at 30k at ef 10-16).**
**Why R3 / H8 do not show "flat fastest at 30k"**: (1) It is not an H2 result. H2 (research_board line 56) says *HNSW is 1.3-2.5x faster* than flat at N=30k, d=128, ef 32-128, recall .96-.99, with 11x memory; line 79 and `g8_triage.md` line 47 turned that into "flat is the better default / flat is fastest at our N",
a memory/simplicity judgement (11x bytes), not a speed measurement. (2) I reproduce H2's numbers: flat AVX2 d=128 N=30k p50 341 us (H2: 340 us) and HNSW at ef 32/64/128 = 134-142 / 227-232 / 319-371 us = 2.4-2.6x / 1.5x / 0.9-1.07x faster (H2: 1.3-2.5x).
(3) R3 and H8 use the hnsw_rs floor ef = 10-16 where HNSW is 4-6x (d=128) to 7-10x (d=384) faster; H2 tuned ef to 32-128 (needed for recall .96-.99 on the buggy crate) which eats most of the gap. (4) d=128/30k flat data (15 MB) is cache-friendly, d=384/30k (46 MB) is memory-bound, so the crossover moves from ~5k (d=384) to ~15k (d=128).
(5) The reverse-edge bug does not change speed per ef but caps recall: unpatched M16 cannot reach 0.99 at all and M32 needs 1-2x the ef (30k: 147 vs 90 us p50). (6) AVX2: lib-kernel flat is 1.2-2.4x slower than AVX2 flat at N<=30k, which makes H2-style (lib kernel) HNSW-vs-flat gaps look different; I did not find a thread effect (all single thread).
Negatives: synthetic data; one graph per recipe per N (grown incrementally, not 3 rebuilds); the d=384 `h8_cross_d384.json` first run was on a loaded box (flat 523 us at 10k vs 183 us quiet) - use the `_calm` file.

---
### VERIFY REQUEST — H8 — scale N = 1e5 / 3e5 / 1e6 (streamed, never holding the f32 base for the indexes), and HNSW M32/M48 at 3e5
Rerun (~60 min incl. 3e5 HNSW builds; guard: pauses while free RAM < 3 GB, projected HNSW > 2.5 GB is skipped):
```
$B scale --sizes 100000,300000,1000000 --out results/h8_verify                                           # everything (h8_scale.json / h8_scale.md)
$B scale --sizes 300000 --only-hnsw --hnsw-builds 3 --hnsw 16:200:128:f32lib,32:100:32:f32lib,48:100:24:f32lib --tag hnsw3b --out results/h8_verify   # 3 builds each
python results/h8_report.py scale ; python results/h8_report_3e5.py hnsw3b
```
Peak process working set: 452 MB (1e5), 1,238 MB (3e5), **3,154 MB at 1e6 (above my 2.5 GB cap**: transient Vec doubling in the chunk-merged stores before `shrink_to_fit` + the 1.5 GB f32 reference scan; free RAM stayed > 5 GB, no paging).
Selected rows (p50 / p99 us, mean of 3 passes; RAM B/vec = index heap, SSD = exact f32 rows):
| N | config | RAM B/vec | SSD B/vec | recall@10 | p50 | p99 |
|---|---|---|---|---|---|---|
| 1e5 | flat f32 AVX2 (g15 ref 4503) | 1536 | - | 1.000 | 3852 | 5912 |
| 1e5 | flat block-fp16 | 816 | - | 1.000 | 2705 | 3808 |
| 1e5 | flat int8 | 388 | - | 0.974 | 1122 | 1819 |
| 1e5 | turbovec 4-bit + x4 rerank, f32 in RAM | 196 (+1536) | - | 1.000 | 928 | 1470 |
| 1e5 | HNSW M16 efc200 ef64 / ef128 | 6168 | - | 0.985 / 1.000 | 319 / 489 | 685 / 883 |
| 1e5 | HNSW M32 efc100 ef64 / ef128 | 6898 | - | 0.985 / 1.000 | 263 / 451 | 521 / 821 |
| 1e5 | HNSW M48 efc100 ef32 / ef64 | 7267 | - | 0.995 / 1.000 | 192 / 302 | 378 / 599 |
| 1e5 | HNSW M32 efc100 ef128, block-fp16 stored | 6117 | - | 1.000 | 368 | 576 |
| 3e5 | flat f32 AVX2 | 1536 | - | 1.000 | 10862 | 15548 |
| 3e5 | flat block-fp16 / int8 | 816 / 388 | - | 1.000 / 0.964 | 9234 / 5681 | 10755 / 22578 |
| 3e5 | turbovec 4-bit + x4 rerank, f32 in RAM / SSD cold | 196 (+1536 RAM) / 217 | - / 1536 | 1.000 / 1.000 | 2944 / 5705 | 3774 / 7056 |
| 3e5 | HNSW (3 builds, table below) | 6.2-7.3 KB | - | 0.76-0.996 | 240-826 | 530-2261 |
| 1e6 | flat f32 AVX2 / block-fp16 / int8 | 1536 / 816 / 388 | - | 1.000 / 1.000 / 0.963 | 35225 / 32710 / 11637 | 43992 / 52483 / 17060 |
| 1e6 | turbovec 4-bit + x4 rerank, f32 in RAM / SSD cold | 196 (+1536 RAM) / 207 | - / 1536 | 0.999 / 0.999 | 10167 / 14626 | 14924 / 16264 |
| 1e6 | turbovec 2-bit + x16 rerank, SSD cold | 107 | 1536 | 0.969 | 18508 | 22966 |
| 1e6 | turbovec 3-bit + x8 rerank, SSD cold | 207 | 1536 | 0.995 | 16741 | 20111 |
| 1e6 | HNSW (any) | **SKIPPED**: 6.2-7.3 KB/vec measured at 3e5 -> 6.2-7.3 GB > 2.5 GB cap | | | | |
(all other rows incl. rabitq/turbovec 2/3/4-bit without rerank, int8+SSD rerank: `results/h8_scale.md`.) turbovec heap reads 232 / 217 / 207 B at 1e5 / 3e5 / 1e6 (+31 / +10 / +3 B/vec of retained 2048-row-add scratch, g15 P4; 196 B nominal).
**HNSW at N=3e5, 3 independent builds per config (fixed crate), recall@10 mean [min-max]**
| M | efc | ef | recall@10 | p50 us | p99 us | heap B/vec | build s |
|---|---|---|---|---|---|---|---|
| 16 | 200 | 32 / 64 / 128 / 256 | 0.757 [0.742-0.779] / 0.796 [0.760-0.820] / 0.844 [0.825-0.857] / 0.894 [0.868-0.911] | 240 / 367 / 546 / 793 | 577 / 860 / 1503 / 2261 | 6227 | 149 |
| 32 | 100 | 32 / 64 / 128 / 256 | 0.834 [0.773-0.880] / 0.916 [0.878-0.942] / 0.964 [0.954-0.977] / 0.993 [0.992-0.994] | 243 / 362 / 488 / 823 | 591 / 894 / 1206 / 1981 | 6878 | 85 |
| 48 | 100 | 24 / 32 / 64 / 128 / 256 | 0.890 / 0.922 / 0.957 [0.950-0.964] / 0.986 [0.983-0.990] / 0.996 [0.993-1.000] | 264 / 277 / 396 / 546 / 826 | 530 / 588 / 884 / 1205 / 1666 | 7299 | 82 |
**Same 3e5 benchmark on the UNPATCHED crate** (private workspace, same code, 3 builds, `h8_scale_hnsw3b_unpatched.json`): M16/efc200 ef 32/64/128/256: 0.745 / 0.790 / 0.843 / 0.897 (patched 0.757 / 0.796 / 0.844 / 0.894 = identical within the +-0.03 build spread);
M32/efc100: 0.822 / 0.903 / 0.961 / 0.986 (patched 0.834 / 0.916 / 0.964 / 0.993, +0.003..+0.013); M48/efc100 ef 24/32/64/128/256: 0.826 / 0.858 / 0.937 / 0.975 / 0.993 (patched 0.890 / 0.922 / 0.957 / 0.986 / 0.996, +0.003..+0.064). Build 69-177 s unpatched vs 82-149 s patched (no systematic difference).
=> at N=3e5 the reverse-edge fix is worth <= +0.06 recall (largest at small ef and large M) and does not by itself repair M16; at 1e5 (R3 unpatched M16 ef64/128 0.978 / 0.990 vs 0.985 / 1.000 now) it lifts the ceiling to 1.000.
Answer to "does the fixed crate / larger M fix R3's 0.84-0.92 at 3e5": **larger M does, the fixed crate alone does not** (M32 ef256 0.993, M48 ef128 0.986 / ef256 0.996; 7.3 KB/vec, p99 1.2-1.7 ms, 82 s build); M16 stays at 0.76-0.89 with or without the patch (build spread +-0.03-0.04; R3's single-build 0.84-0.92 is at/above the top of that spread - not reproduced, 3 builds here give 0.79 / 0.84 / 0.90 at ef 64 / 128 / 256).
Why: the synthetic mixture is noise-heavy (many near-equidistant neighbours), recall at fixed ef falls with N, and M16 level-0 lists (32 entries) cannot cover it; ef must scale roughly with N (1e5: ef 64-128, 3e5: ef 128-256).
Negatives: HNSW at 1e6 was not run (RAM); at 3e5 the Windows working set (1.1 GB) was below the 1.9-2.2 GB heap (idle pages trimmed), no paging observed; these are synthetic-mixture numbers, real embeddings at >= 1e5 are unmeasured; 3 builds at 3e5 only for the three f32 recipes (single build for bfp16/int8-stored).

---
### VERIFY REQUEST — H8 — max-vectors-per-GB with MEASURED heap bytes (decimal GB; RAM = index heap + f32 rerank copy if it lives in RAM)
Rerun (~30 min): `$B memory --out results/h8_verify ; $B tvmem --n 100000 --out results/h8_verify ; python results/h8_report_mem.py` ; files `h8_memory_{real-mini,real-bge,synth384-n100000}.json`, `h8_tvmem.json`, `h8_memory.md` (every row), `h8_pareto_mem.md` (3-D Pareto RAM bytes x recall x p99, RAM-only and SSD-rerank views for all N incl. scale rows).
| config | real-mini d=384 B/vec : M vec/GB : recall : p50/p99 us | real-bge d=768 B/vec : M/GB : recall | synth 1e5 B/vec : M/GB : recall : p50/p99 | synth 1e6 B/vec : recall |
|---|---|---|---|---|
| flat f32 exact | 1536 : 0.65 : 1.000 : 1151/3413 | 3072 : 0.33 : 1.000 | 1536 : 0.65 : 1.000 : 3225/4717 | 1536 : 1.000 |
| flat f16 | 768 : 1.30 : 1.000 | 1536 : 0.65 : 1.000 | 768 : 1.30 : 0.999 | - |
| flat block-fp16 | 816 : 1.23 : 1.000 : 1070/2703 | 1632 : 0.61 : 1.000 | 816 : 1.23 : 1.000 : 2540/3240 | 816 : 1.000 |
| flat int8 | 388 : 2.58 : 0.997 : 560/2584 | 772 : 1.30 : 0.993 | 388 : 2.58 : 0.976 : 2100/2235 | 388 : 0.963 |
| turbovec 2-bit | 100 : 9.96 : 0.924 : 84/303 | 197 : 5.08 : 0.931 | 100 : 9.99 : 0.557 | 107 : 0.361 |
| turbovec 3-bit (**measured 196 B = the 4-bit footprint**) | 196 : 5.09 : 0.960 : 125/152 | 389 : 2.57 : 0.958 | 196 : 5.10 : 0.726 | 207 : 0.606 |
| turbovec 4-bit | 196 : 5.09 : 0.977 : 341/380 | 389 : 2.57 : 0.976 | 196 : 5.10 : 0.847 : 765/993 | 207 : 0.772 |
| turbovec 4-bit + x4 rerank, f32 in RAM | 1732 : 0.58 : 1.000 : 372/411 | - | 1732 : 0.58 : 1.000 : 811/969 | 1732 : 0.999 |
| turbovec 4-bit + x2..x4 rerank, f32 on SSD (cold direct) | 196 (+1536 SSD) : 5.09 : 1.000 : 2019/2941 (x2) | 389 : 2.57 : 1.000 : 2455/2804 (x2) | 196 : 5.10 : 1.000 : 4415/5082 (x4) | 207 : 0.999 : 14626/16264 |
| turbovec 2-bit + x4 rerank, f32 on SSD | 100 : 9.96 : 1.000 : 4117/10218 | 197 : 5.08 : 1.000 : 4524/6546 | 100 : 9.99 : 0.957 (x8: 1.000 : 6656/8205) | 107 : 0.969 (x16) |
| RaBitQ-style 2 / 3 / 4-bit (no rerank) | 126 / 174 / 222 : 0.929 / 0.959 / 0.972 | 297 / 393 / 489 : 0.923 / 0.955 / 0.975 | 106 / 154 / 202 : 0.554 / 0.720 / 0.842 | 101 / 149 / 197 : 0.343 / 0.587 / 0.755 |
| HNSW M32 efc100 ef24, f32 / bfp16 / int8 stored | 6562 / 5816 / 5432 : 0.15-0.18 : 0.992 / 0.994 / 0.988 : 127/232 | 8165 / 6696 / 5874 : 0.996 / 0.996 / 0.993 | 6843 / 6130 / 5684 : 0.936 / 0.942 / 0.927 | (see scale) |
Reading (measured, d=384): the exact options are block-fp16 (816 B, recall 1.000, 1.23 M/GB) and f16 (768 B); int8 (388 B) loses 0.003-0.037 recall; turbovec without rerank is only usable on the easy real sets (0.92-0.98) and collapses with N (4-bit: 0.977 real, 0.847 synth 1e5, 0.772 synth 1e6);
exact rerank restores 0.995-1.000 but then the f32 rows are either RAM (1.7 KB/vec: **no saving over flat f32**) or SSD (see SSD block). HNSW is 4-8x the bytes of flat f32 (graph ~ 5 KB/vec) for 3-18x lower latency.
**Best configs per goal** (all measured, synthetic/templated): max M vec/GB with nothing but the index in RAM and recall >= 0.95: turbovec 3-bit/4-bit on easy real data (5.09 M/GB mini, 2.57 M/GB BGE-768), int8 flat (2.58 / 1.30 M/GB) when recall >= 0.97 is needed on harder data; recall 1.000 exact: block-fp16 flat 1.23 M/GB;
with f32 rows on SSD: turbovec 2-bit + x4..x8 rerank (100 B, 9.96-9.99 M/GB, recall 1.000 up to 1e5, 0.969 at 1e6 needs x16) or 4-bit + x4 (196-207 B, 4.8-5.1 M/GB, recall 0.999-1.000 up to 1e6) at 2-16 ms per query.
Findings: (1) **turbovec 3-bit is stored with the 4-bit footprint** (counting allocator, single add, N=1e5: 2-bit 100.1, 3-bit 196.1, 4-bit 196.1 B/vec; nominal 3-bit would be 148): g15's "3-bit 6.8 M/GB" was nominal, measured 5.1.
(2) block-fp16 loss vs f32: 0.000 on all sets (recall 1.000), f16 0.000-0.001. (3) HNSW graph bytes dominate (5.1-8.7 KB/vec): block-fp16 points save 10-18 % (bge d=768: 8.2 -> 6.7 KB), int8 points 17-28 %.
Negatives: heap bytes exclude allocator slack/page rounding (RSS is larger); at N=2.3e4 the fixed turbovec scratch / RaBitQ rotation matrix (590 KB) inflate B/vec by 2-25 B (RaBitQ rows include it); `SSD buffered` rows are not RAM savings (the OS page cache holds the file) and are excluded from the Pareto.

---
### VERIFY REQUEST — H8 — file-backed (SSD) exact-rerank store: measured cost
Rerun (~5 min): `$B ssd --out results/h8_verify` (writes a 460 MB f32 file under `nm-target-r8/h8_tmp`, deleted afterwards); file `h8_ssd.json`. Std-only `FileRows` = `seek_read` positional reads; direct = `FILE_FLAG_NO_BUFFERING|RANDOM_ACCESS` (every row read hits the device), buffered = OS page cache.
Random 1,536-byte row reads (N=3e5): cold direct **70 us/row single-threaded (14.2k rows/s)**; 8 threads sharing ONE handle stay at 13.8k rows/s (serialised); **one handle per thread: 28k / 57k / 113k rows/s with 2 / 4 / 8 threads (35 / 17 / 8.9 us/row)**; warm page cache ~6-7 us/row.
turbovec 4-bit + x4 rerank (40 rows/query), N=3e5, exact f32 rows on SSD with direct I/O: recall 1.000, p50 / p99 = **6.2 / 8.5 ms (1 I/O thread), 4.4 / 5.2 ms (2), 3.9 / 5.4 ms (4)**, of which ~2.6 ms is the scan; same codes with the rows in RAM: 2.9 / 3.8 ms; warm OS page cache (file buffered) costs ~+0.2 ms over RAM at N=23k (MiniLM 543 vs 372 us p50, `h8_memory.md` "SSD buffered").
At 1e6: 4-bit x4 SSD 14.6 / 16.3 ms vs RAM 10.2 / 14.9 ms; 2-bit x16 18.5 / 23.0 ms.
=> the rerank copy off RAM costs +3.5-5 ms per query (40 random rows) single-threaded, ~+1.3 ms with 4 I/O handles; it buys 196-207 B RAM/vec (4.8-5.1 M vec/GB) at recall 0.999-1.000 up to 1e6 (synthetic). Not a latency optimisation.
Negatives: one SSD, Windows direct I/O, no true mmap (page-fault path unmeasured), queue depth <= 8 handles, a per-query thread::scope (~100 us) is included in the multi-handle p50; the OS page cache can still serve "direct" reads from the SSD's own cache (device DRAM), so cold-after-reboot cost may be higher.

---
### VERIFY REQUEST — H8 — HNSW with quantised stored vectors (custom `Distance` on block-fp16 / int8 points)
Part of `$B memory` and `$B scale` (rows "HNSW ... [bfp16 stored]" / "[int8 stored]"): `Hnsw<'static, i16, BfpDist>` (32-value blocks, i16 mantissa + f32 scale per block = 816 B at d=384, query quantised the same way, `madd_epi16` per block) and `Hnsw<i8, I8Dist>` (per-vector scale, 388 B). `selftest` checks dot accuracy (block-fp16 0.065435 vs f32 0.065437; int8 0.065309).
| dataset | M/efc/ef | heap B/vec f32 / bfp16 / int8 | recall f32 / bfp16 / int8 | p50 us f32 / bfp16 / int8 |
|---|---|---|---|---|
| real-mini | 32/100/24 | 6562 / 5816 / 5432 | 0.992 / 0.994 / 0.988 | 127 / 120 / 103 |
| real-bge | 32/100/24 | 8165 / 6696 / 5874 | 0.996 / 0.996 / 0.993 | 162 / 128 / 95 |
| synth 1e5 | 32/100/24 | 6843 / 6130 / 5684 | 0.936 / 0.942 / 0.927 | 165 / 109 / 152 |
| synth 1e5 (scale) | 32/100/ef128 | 6898 / 6117 / 5703 | 1.000 / 1.000 / 0.969 | 451 / 368 / 324 |
| synth 3e5 | 32/100/ef256 | 6879 / 6182 / 5719 | 0.993 / 0.987 / 0.958 | 662 / 545 / 499 |
Result: block-fp16 points save 10-18 % of the HNSW bytes (graph = ~5 KB/vec stays) at equal recall (-0.006 at 3e5/ef256, within the build spread) and equal or lower latency; int8 points save 17-28 % but cap recall (0.958-0.974 at 1e5-3e5, same ceiling as flat int8). Recommend block-fp16 only if memory matters (proposal Q3).
Negatives: single build per quantised config; latencies from the (partly loaded) memory/scale runs.

---
### VERIFY REQUEST — H8 — RaBitQ-style multi-bit codes + rerank (std only)
Implementation (`h8_common::RaBitQ`, ~150 lines): random orthogonal rotation (Gram-Schmidt), B-bit uniform grid per coordinate with a per-vector grid scale picked to maximise cos(code, vector), unbiased-style rescale g = 1/<code-mid, o'>, scan over B bit-planes with an int8-quantised rotated query (AVX2 `maddubs`), top-(k x rerank) exact rerank (RAM or SSD). `selftest` shows corr(estimate, true dot) 0.897 / 0.971 / 0.992 / 0.998 for 1 / 2 / 3 / 4 bits, slope 0.99-1.00.
(An earlier selftest exposed a bug of mine - the rotation seed equalled the data-generator seed, so the first rows of the rotation were the cluster centroids (corr 0.54-0.95); fixed before any reported run.)
Recall@10 without rerank (B/vec incl. the 590 KB rotation) vs turbovec at the same bits: mini 0.867 / 0.929 / 0.959 / 0.972 (1-4 bit; 78 / 126 / 174 / 222 B) vs turbovec 2/3/4-bit 0.924 / 0.960 / 0.977 (100 / 196 / 196 B); synth 1e5 0.347 / 0.554 / 0.720 / 0.842 vs 0.557 / 0.726 / 0.847; 3e5 0.227 / 0.466 / 0.655 / 0.794 vs 0.461 / 0.651 / 0.803;
1e6 0.139 / 0.343 / 0.587 / 0.755 vs 0.361 / 0.606 / 0.772. With rerank x4 at 1e6 (SSD): 4-bit 0.997 vs turbovec 4-bit 0.999; 3-bit x4 0.955; 2-bit x8 0.865 vs turbovec 2-bit x16 0.969.
**Verdict (the science report's expectation that multi-bit RaBitQ matters at 1e5-1e6 is not supported here):** per code bit it is equal to or slightly below turbovec on this data; the only advantage is true 3-bit packing (148.6 B vs turbovec's 207 B at 1e6, 6.7 vs 4.8 M vec/GB) and its scan is 1.5-3x slower (4-bit: 3.6 ms vs 1.5 ms at 1e5; 32.6 vs 11.8 ms at 1e6), unoptimised (no FastScan LUTs). Not recommended.
Negatives: my implementation (no learned/optimised rotation, no data-centred residual, int8 query), 14-step scale search; the true RaBitQ library might do better.

---
### VERIFY REQUEST — H8 — concurrent readers on ONE shared index (1/2/4), with a writer; concurrent-insert correctness
Rerun (~10 min): `$B conc --out results/h8_verify ; $B concins --out results/h8_verify` ; files `h8_conc.json`, `h8_concins.json`. N=1e5 synthetic d=384; HNSW M32/efc100/ef32 (recall 0.969 single thread) shared as `Arc<dyn Ann>` with NO lock; 4 s windows x 3 reps; max 4 reader threads (+1 writer); box shared (calib 1.55-2.05).
| index | scenario | threads | qps mean [min-max] | p50 / p99 us | writer ops/s |
|---|---|---|---|---|---|
| HNSW M32 efc100 ef32 | read-only | 1 / 2 / 4 | 4657 [4255-4888] / 9150 [8614-10124] / 18349 [16905-19461] | 198/455, 204/442, 203/438 | - |
| HNSW | readers + 1 insert thread (insert_slice on &self) | 1 / 2 | 4772 [4228-5109] / 9748 [9057-10230] | 194/444, 191/421 | 2248 / 2311 |
| flat int8 + x4 f32 rerank (RAM) | read-only | 1 / 2 / 4 | 748 / 1254 / 1522 | 1195/2274, 1548/2533, 2503/5005 | - |
| turbovec 4-bit + x4 rerank (RwLock, per-reader 1-thread pool) | read-only | 1 / 2 / 4 | 948 / 1780 / 3051 | 1022/1810, 1039/2346, 1027/4002 | - |
| turbovec 4-bit + x4 rerank | 2 readers + writer adding 64 vec / 50 ms | 2 | 1557 [1350-1810] | 1155 / 4144 | 1173 vec/s |
HNSW read-only scales 3.9x on 4 threads (flat p50), one inserter running flat out (~2.3k inserts/s at efc100) does not measurably reduce reader qps/p99 (4772 vs 4657, 9748 vs 9150 qps, inside the 10 % rep spread); flat int8 scales 2.0x (memory-bound), turbovec 3.2x; each turbovec `add` invalidates its blocked-code cache (p99 2.3 -> 4.1 ms with a writer).
**Concurrent-insert correctness** (`concins`): HNSW M32/efc100 with 30,000 base vectors, 2 writer threads inserting 6,000 more via `insert_slice` while 2 readers query (15,233 reads in 1.2 s): afterwards 0/6,000 inserted vectors fail to be their own top-1 (ef 64), 0/6,000 sampled base items fail, recall@10 (ef 32) over all 36,000 vectors 0.993.
Negatives: 16-core shared box, 4 s windows; max 4 reader threads; concurrent-insert check is one run, 1.2 s; insert-heavy workloads (efc 200+) not measured; the turbovec writer row measures lock/repack cost (new vectors are skipped by the rerank filter).

---
### SIZE-GATING RULE (which backend by N; measured crossovers; synthetic / templated data, d=384 unless stated, 1 thread, recall target >= 0.99)
| N (vectors) | choose | why (measured) |
|---|---|---|
| <= ~5,000 | exact flat (f32 AVX2, or block-fp16 flat to halve bytes) | flat 77 us vs HNSW 56-76 us at 5k (tie), exact, no build, 1.5 KB/vec (0.8 KB block-fp16); at d=128 flat wins to ~10-15k |
| ~5,000-10,000 | int8-scan + f32-rerank x10 (exact, 388 B + f32) or HNSW M24/efc100/ef16 | 121 us vs flat 183 us at 10k; HNSW 55-72 us (2.5-3.4x), 6.3 KB/vec |
| 10,000 - ~3.5e5 (RAM budget ~2.5 GB, 6.9-7.3 KB/vec) | **HNSW M24-32, efc 100-200 (efc >= 3M), ef 16-24 (<=3e4); ef 64-128 at 1e5; M48/efc100 ef 128-256 at 3e5** | 7-10x faster than flat at 30k, 7-18x at 1e5; recall 0.99-1.0; block-fp16 points save 10-18 % bytes |
| RAM-bound (> ~3.5e5 per 2.5 GB, or bytes/vec <= ~1 KB wanted) and latency 3-15 ms acceptable | turbovec 4-bit codes (196 B) + exact f32 rerank x4 (rows in RAM if RAM allows: 1.7 KB/vec, 2.9 ms at 3e5, 10 ms at 1e6; rows on SSD: 4.8-5.1 M vec/GB RAM, 5.7 ms at 3e5, 14.6 ms at 1e6; recall 0.999-1.000) | HNSW 1e6 needs 6-7 GB (skipped) |
| bytes-extreme | turbovec 2-bit + x16 rerank (SSD): 100-107 B RAM, 9.3-10 M vec/GB, recall 0.969-1.000, 18 ms at 1e6 | measured |
| exact, no rerank, tight RAM | block-fp16 flat 816 B (1.23 M/GB), 9-33 ms at 3e5-1e6 | recall 1.000 measured |
Switch points are noisy +-30 % (load, one graph); at d=128 move the first switch to ~10-15k.

---
### Not done / not measured (blunt list)
* Any real corpus >= 1e5 (none exists); real-data results are 23k-vector templated corpora (recall saturates, near-duplicate heavy). HNSW at N=1e6 (6-7 GB). Real mmap (page-fault path); only direct I/O vs OS-cached `seek_read` on one Windows SSD.
* Deletes, filtered search, persistence/load time, `parallel_search`, > 4 reader threads, insert-heavy workloads, recall under churn.
* HNSW at 3e5 with block-fp16/int8 points was one build; rare-event rates (0.04 %) from 3 builds; RaBitQ is my own implementation (no FastScan), so its latency is a lower bound on engineering effort not on the method.
* `h8_hnsw_confirm_*` latencies (box 3-4x slow); use `h8_lat2.md`. The first `h8_cross_d384.json` is a loaded run; use `_calm`.
* Peak working set at N=1e6 was 3.15 GB (cap 2.5 GB), no paging (free RAM > 5 GB).
* No claim about routing quality or model IQ; this is infrastructure.

Proposed patches (not applied): `results/h8_proposed_patch.md` (Q1 HNSW defaults + efc > 2M guard, Q2 AVX2 dot, Q3 block-fp16 HNSW points, Q4 size gating, Q5 turbovec 3-bit accounting, Q6 SSD rerank store, Q7 RaBitQ not recommended).
