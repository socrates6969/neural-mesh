# G15 / W16 (MAX-VECTORS) board entry

Pure retrieval infrastructure; no model-IQ claim. Code: `src/bin/g15_vectors.rs` + `src/bin/g15_common/mod.rs` (new, nothing in `src/*.rs`
edited; proposals in `results/g15_proposed_patch.md`). All JSON/MD/logs: `results/g15_*`. First (heavily loaded) runs of quant/scale/conc/flat are kept in
`results/g15_run1/`; the files directly under `results/` for those four modes are the second, quieter run. Run scripts: `results/g15_run_all.sh`,
`g15_run_part2.sh`, `g15_run_part3.sh`.

Setup common to every block: release build, default target (no `-C target-cpu`; AVX2/FMA/F16C kernels are runtime-dispatched `#[target_feature]`
functions), own `CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r6`, `RAYON_NUM_THREADS=2`, `--offline`. Latency = one query at a time on ONE thread
(turbovec inside a 1-thread rayon pool, like memory_bench), 300 queries (200 at scale), 1 untimed warm pass then 3 timed passes -> "mean [min-max]" over
the 3 reps (p50 / p99 each rep). Recall@10 = id-set overlap with exact f32 brute force on the same L2-normalised vectors (AVX2 flat scan, deterministic).
Bytes/vector = delta of a counting global allocator around the build (exact heap), cross-checked with working-set deltas. HNSW = `hnsw_rs` 0.3.4 as in
`src/memory.rs` (cosine = 1-dot), built with `parallel_insert_slice` in 2048-row batches on 2 threads (graph is NON-deterministic -> rebuild spread reported).

Data: synthetic = `memory::synth::Clustered::hierarchical(384, 64 topics x 16 subs, 0.8, 0.6)` (same as memory_bench), queries = fresh draws.
real-mini = results/g1_emb_[A-D]_mini.f32 concatenated (32,476 rows -> **23,453 unique after dropping exact duplicates, 300 held out as queries -> 23,153 base rows**; no MiniLM
query embeddings are stored). real-bge = g1_emb_[A-D]_bge.f32 (768-d, 23,453 unique rows), queries = 300 real `g1_emb_*_bgei.f32` instruction-prefixed
query embeddings. Both real sets come from SYNTHETIC TEMPLATED corpora (near-duplicate heavy; recall plateaus at 0.995-0.999 because of near-ties).
`f6_minilm_fixture.json` (2,424 vectors) / `f8_bge_embeddings.f32` (2,424) were judged too small and not used.

## MACHINE-LOAD CAVEAT (read first)
The box was at ~100 % total CPU from other workers (g12/g16/g17 jobs, cargo builds) during the first runs and free RAM swung 2.5-6.3 GB. Each JSON stores
`calib_ms_fixed_scan` (a fixed in-cache 20x4000x384 AVX2 scan, median of 15) at start/end: it ranged 1.4-2.7 ms normally and 4.3-5.8 ms (2-3x slower) during
run 1 of scale N=1e5/1e6. Same code: HNSW N=1e5 M16/efc200 build 116.5 s (loaded) vs 56.5 s (quiet); flat exact N=1e5 p50 9.9 ms vs 4.5 ms. **Absolute
latencies/build times are +-50-100 % load dependent; trust ratios, recall (+-0.01-0.02 from graph nondeterminism) and bytes/vector (+-2 %).** p99 is the most
load-sensitive number (single outliers of 5-40 ms appear in loaded runs). The sweep Pareto is from ONE build per config and so is noisy; the `confirm`
blocks rebuild 3x.

---
### VERIFY REQUEST — W16 — HNSW M x ef_construction x ef_search sweep (recall@10 vs latency Pareto)
Rerun (≈35 min; fixtures from `results/`, override with `G15_FIXTURES`):
```
cd C:\Users\mariu\neural-mesh ; export PATH=... (rust+mingw) CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r6 RAYON_NUM_THREADS=2
cargo build --release --offline --bin g15_vectors
B=$CARGO_TARGET_DIR/release/g15_vectors.exe
$B sweep --out results/g15_verify                                   # M{8,16,32} x efc{100,200,400} x ef{10..512}; 3 datasets
$B sweep --m 48,64 --efc 100,200 --tag xl --out results/g15_verify  # extended M
```
Result files: `g15_hnsw_sweep_{synth384-n30000,real-mini,real-bge}[_xl].json` (every config/ef with per-rep p50/p99/mean + build s + heap B/vec), `g15_sweep.md`, `g15_sweep_xl.md`.
Pareto frontier (M 8..64, single build each; p50 / p99 mean us over 3 reps; heap = bytes/vec of the graph incl. its own f32 copy):

synth384, N=30,000 (d=384)
| recall@10 | p50 | p99 [min-max] | M | efc | ef | heap B/vec |
|---|---|---|---|---|---|---|
| 0.939 | 140 | 272 [262-284] | 48 | 100 | 10 | 7222 |
| 0.961 | 134 | 296 [286-304] | 32 | 100 | 10 | 6778 |
| 0.983 | 177 | 321 [312-334] | 48 | 100 | 16 | 7222 |
| 0.992 | 210 | 354 [339-362] | 48 | 200 | 16 | 7500 |
| 0.995 | 311 | 458 [452-467] | 48 | 200 | 24 | 7500 |
| 0.997 | 544 | 846 [832-863] | 64 | 200 | 48 | 7817 |

real-mini, N=23,153 (d=384)
| 0.973 | 76 | 149 [141-154] | 32 | 100 | 10 | 6539 |
|---|---|---|---|---|---|---|
| 0.982 | 103 | 219 [202-243] | 32 | 100 | 16 | 6539 |
| 0.991 | 151 | 260 [238-284] | 48 | 200 | 16 | 7133 |
| 0.993 | 195 | 342 [316-385] | 32 | 200 | 32 | 6719 |
| 0.995 | 256 | 429 [405-473] | 32 | 200 | 48 | 6719 |
| 0.997 | 367 | 657 [624-676] | 64 | 200 | 48 | 7427 |

real-bge, N=23,453 (d=768)
| 0.850 | 87 | 206 [202-211] | 8 | 100 | 10 | 6674 |
|---|---|---|---|---|---|---|
| 0.973 | 114 | 220 [198-258] | 48 | 100 | 10 | 8552 |
| 0.992 | 205 | 350 [342-365] | 48 | 200 | 16 | 8747 |
| 0.997 | 362 | 565 [540-590] | 48 | 200 | 32 | 8747 |
| 0.998 | 482 | 727 [709-740] | 48 | 200 | 48 | 8747 |

Findings (measured): (1) the frontier lives at the SMALL-ef edge: hnsw_rs raises ef below k to k, so ef=10 is the floor and already gives recall 0.94-0.98 with
M>=32; raising ef beyond 24-32 buys <=0.005 recall for 2-10x p99. (2) M matters more than efc: recall at ef=64/efc=200 by M = 8: 0.95, 16: 0.985, 32: 0.993,
48: 0.996 (synth); M=8 appears only as the lowest-recall corner (0.85 on real-bge, 0.57-0.82 on synth). (3) efc=100 suffices up to recall ~0.98 and builds 2-3x faster (3.9 s vs 11.8 s vs 26.8 s for efc 100/200/400 on real-mini M32); efc=200 is needed to reach >=0.99 at ef<=16; efc=400 never appears on the merged frontier. (4) M64 shows no gain over M48 and its efc=100 builds retain 12.3-13.8 KB/vec (vs 7.8-9.0 KB at efc=200) - an hnsw_rs
quirk I did not chase. (5) recall saturates at 0.995-0.999 on real data (near-duplicate ties), higher ef only costs latency (ef=384: p99 4-8 ms).
Negatives: single-build frontier points are noisy (+-20 % on p99); the exact ordering of near-equal frontier points (M32 vs M48 at ef=10) is not significant.

---
### VERIFY REQUEST — W16 — best recall-at-fixed-latency HNSW configs (3 rebuilds, library kernel vs AVX2 kernel)
Rerun (~25 min): `$B confirm --out results/g15_verify` and `$B confirm --tags xl --out results/g15_verify` (needs the sweep files in the same `--out`; copy them there or rerun the sweeps).
Files: `g15_hnsw_confirm_{synth384-n30000,real-mini,real-bge}.json` (+ `_xl.json`), `g15_confirm.md`, `g15_confirm_xl.md`. 3 independent graph builds per row, 3 timed passes each
(mean of build means [min-max over builds]). "lib" = `memory::CosineDist` autovectorised kernel (as in repo), "avx2" = same graph recipe with AVX2+FMA dot (proposal P1).
| dataset | target | cfg | kernel | recall@10 | p50 us | p99 us | build s | heap B/vec |
|---|---|---|---|---|---|---|---|---|
| synth30k | >=0.95 | M32 efc100 ef10 | lib | 0.956 [0.941-0.967] | 127 [121-131] | 367 [270-550] | 12.3 | 6791 |
| synth30k | >=0.95 | M32 efc100 ef10 | avx2 | 0.959 [0.953-0.962] | 108 [102-111] | 643 [403-820]* | 13.6 | 6814 |
| synth30k | >=0.99 | M32 efc100 ef24 | lib | 0.990 [0.985-0.994] | 184 [179-187] | 439 [404-476] | 14.4 | 6795 |
| synth30k | >=0.99 | M32 efc100 ef24 | avx2 | 0.992 [0.990-0.994] | 162 [153-169] | 402 [347-470] | 12.6 | 6837 |
| synth30k | >=0.99 | M48 efc200 ef16 | lib/avx2 | 0.993 / 0.994 | 245 / 224 | 489 / 396 | 28 / 26 | 7498 |
| mini23k | >=0.95 | M32 efc100 ef10 | lib | 0.961 [0.955-0.968] | 93 [84-101] | 298 [258-341] | 4.9 | 6574 |
| mini23k | >=0.95 | M32 efc100 ef10 | avx2 | 0.971 [0.969-0.972] | 79 [75-82] | 214 [195-245] | 3.8 | 6544 |
| mini23k | >=0.99 | M48 efc200 ef16 | lib | 0.989 [0.988-0.991] | 161 [159-162] | 305 [266-356] | 10.9 | 7161 |
| mini23k | >=0.99 | M48 efc200 ef16 | avx2 | 0.990 [0.987-0.993] | 133 [123-138] | 233 [211-255] | 8.4 | 7135 |
| bge23k | >=0.95 | M16 efc100 ef10 | lib | 0.956 [0.938-0.969] | 108 [106-112] | 378 [367-394] | 7.8 | 7413 |
| bge23k | >=0.95 | M16 efc100 ef10 | avx2 | 0.958 [0.954-0.960] | 95 [94-96] | 216 [208-225] | 6.4 | 7409 |
| bge23k | >=0.99 | M48 efc200 ef16 | lib | 0.992 [0.990-0.994] | 202 [200-208] | 342 [330-349] | 16.3 | 8698 |
| bge23k | >=0.99 | M48 efc200 ef16 | avx2 | 0.992 [0.991-0.993] | 183 [176-190] | 317 [294-333] | 13.4 | 8695 |
(*) loaded outlier; second independent confirm run of the same config (xl file) gave 261 us. Second-run duplicates of M32 efc100 ef10 in `_xl` files:
mini lib p99 540 [184-1186] / avx2 166; synth lib 263 / avx2 261; bge M48 efc100 ef10 lib 286 / avx2 231 - i.e. p99 scatter between runs is up to 3x.
**Headline: recall@10 >= 0.95 at p99 ~ 0.2-0.4 ms (p50 ~ 0.08-0.13 ms) and >= 0.99 at p99 ~ 0.23-0.5 ms (p50 0.13-0.25 ms) for N ~ 23-30k, single thread.**
AVX2 kernel (no build flags): p50 -9 to -17 %, p99 -7 to -43 % (noisy), build +10 to -22 % (noisy), recall unchanged (+0.00 to +0.01). Negative: on synth30k the avx2 p99 was worse in one run (outlier).

---
### VERIFY REQUEST — W16 — quantisation: turbovec 2/3/4-bit (+exact rerank), int8, PQ-lite, f16/bf16/block-fp16 — memory vs recall
Rerun (~12 min): `$B quant --out results/g15_verify` ; files `g15_quant_{synth384-n100000,real-mini,real-bge}.json`, `g15_quant.md`.
turbovec API used: `turbovec::TurboQuantIndex::{new,add,prepare,search}` (data-oblivious TurboQuant, 2/3/4 bit, no training). turbovec has NO int8, PQ or float16 mode, so
I hand-rolled (std only, `g15_common`): int8 with per-vector scale + AVX2 `madd_epi16` scan; PQ-lite (sub-vector k-means, 256 centroids, 8-iter, ADC table scan;
m = d/8 or d/4); f16 (F16C), bf16, block-fp16 (32-value blocks, i16 mantissa + f32 shared scale = 816 B/vec at d=384). "+rerank Nx" = fetch N*10 candidates from the
compressed scan, rescore exactly from an f32 copy.
Excerpt (nominal B/vec is the codes+scales; turbovec measured heap has a FIXED ~3-7 MB retained scratch, so heap B/vec is inflated at N~23k - see tvmem block; p50/p99 mean us, 1 thread):
| config (d=384) | nominal B/vec | M vec/GB (index only) | recall mini23k | recall synth1e5 | p50/p99 mini23k | p50/p99 synth1e5 |
|---|---|---|---|---|---|---|
| flat f32 exact AVX2 | 1536 | 0.65 | 1.000 | 1.000 | 1132 / 1713 | 4481 / 6143 |
| flat f32 exact lib kernel | 1536 | 0.65 | 1.000 | 1.000 | 1635 / 2808 | 5175 / 8218 |
| flat f16 | 768 | 1.30 | 1.000 | 0.999 | 798 / 1054 | 3534 / 4724 |
| flat bf16 | 768 | 1.30 | 0.998 | 0.991 | 829 / 1432 | 3499 / 4371 |
| **flat block-fp16 (32, i16+f32 scale)** | 816 | 1.23 | **1.000** | **1.000** | 836 / 1054 | 3466 / 5876 |
| flat int8 | 388 | 2.58 | 0.997 | 0.976 | 326 / 399 | 1498 / 1921 |
| flat int8 + f32 rerank 4x | 388 (+1536) | 0.52 | 1.000 | 1.000 | 352 / 438 | 1466 / 1750 |
| turbovec 4-bit | 196 | 5.1 | 0.977 | 0.847 | 353 / 991 | 1085 / 1325 |
| turbovec 3-bit | 148 | 6.8 | 0.960 | 0.726 | 131 / 155 | 1186 / 1355 |
| turbovec 2-bit | 100 | 10.0 | 0.924 | 0.557 | 81 / 101 | 569 / 758 |
| turbovec 4-bit + f32 rerank 2x | 196 (+1536) | 0.58 | 1.000 | 0.995 | 361 / 405 | 1042 / 1205 |
| turbovec 2-bit + f32 rerank 4x | 100 (+1536) | 0.61 | 1.000 | 0.957 | 107 / 130 | 599 / 829 |
| PQ-lite m=48 | 65 / 52 | 15-19 | 0.805 | **0.309** | 761 / 824 | 3249 / 3763 |
| PQ-lite m=96 | 113 / 100 | 9-10 | 0.889 | 0.452 | 1521 / 2163 | 6425 / 7606 |
d=768 (real-bge): flat f16 1.000, bf16 0.997, block-fp16 1.000 (1632 B vs 3072), int8 0.993 (772 B), turbovec 4/3/2-bit 0.976/0.958/0.931 at 388/292/196 B/vec;
rerank 2x -> 1.000 for 4 and 3 bit, 0.998 for 2 bit.
**Block-fp16 recall loss vs f32: 0.000 measured (0 of 3,000 top-10 slots differ) on all three sets, at 53 % of the bytes and 19-26 % faster scan (memory-traffic
bound); f16 -0.000 to -0.001 (1 miss in 3,000 slots); bf16 -0.002 to -0.009; int8 -0.003 to -0.024 (-0.037 at N=1e6 synthetic).** Resolution caveat: nq=300, so differences < 0.001 are not resolvable.
Negatives: (a) turbovec WITHOUT rerank is only usable on the real, easy sets (0.92-0.98); on harder synthetic data recall drops with N (4-bit: 0.965 @1e4,
0.848 @1e5, 0.803 @3e5, 0.772 @1e6; 2-bit 0.888 -> 0.361). (b) Exact rerank restores 0.995-1.000 but needs the f32 vectors resident: RAM per vector is then
1.6-1.9 KB, i.e. NO memory saving over flat f32 unless the f32 copy lives on SSD/mmap (not measured here). (c) My PQ-lite is dominated by turbovec 2-bit (higher recall at
equal or fewer bytes, 5x faster scan, 50-100x faster build); it is a negative result, not a recommendation. (d) f16/bf16/block-fp16 scans are only ~20-25 % faster
than the AVX2 f32 scan; int8 is ~3x.

---
### VERIFY REQUEST — W16 — scaling N = 1e4, 1e5, 3e5 (HNSW only), 1e6 (no HNSW) — build time, bytes/vector, recall, p50/p99
Rerun (~35 min incl. 3x HNSW builds at 1e5): `$B scale --hnsw-build-reps 3 --out results/g15_verify` and `$B scale --sizes 300000 --hnsw-max 300000 --hnsw-build-reps 1 --tag n300k --out results/g15_verify`.
Files: `g15_scale.json`, `g15_scale_n300k.json`, `.md`, run-1 copies in `g15_run1/`. RAM rule applied by the code: N>=5e5 only if free RAM > 4 GB at start (5.3 GB at the time -> ran); stop if < 2 GB.
Peak process working set: 66 MB (1e4), 531 MB (1e5), 1,602 MB (3e5), 1,856 MB (1e6) - under the 3 GB cap.
(run 2 = quiet-ish; p50/p99 mean us [min-max of 3 passes]; build s = mean of 3 builds except where noted)
| N | config | B/vec (nominal for flat/int8/turbovec - measured heap for HNSW) | recall@10 | p50 | p99 | build s |
|---|---|---|---|---|---|---|
| 1e4 | flat f32 AVX2 | 1536 | 1.000 | 436 [370-479] | 821 | 0 |
| 1e4 | int8 flat | 388 | 0.997 | 123 [96-137] | 179 | 0.01 |
| 1e4 | turbovec 4-bit + rerank 4x | 196 (+1536 f32) | 1.000 | 173 [170-175] | 244 | 0.00(add) |
| 1e4 | turbovec 2-bit + rerank 4x | 100 (+1536) | 0.998 | 54 [54-54] | 72 | 0.02 |
| 1e4 | HNSW M16 efc200 ef32 / ef64 (lib) | 6508 | 0.982 / 0.985 | 266 / 481 | 349 / 641 | 5.6 [5.5-5.7] |
| 1e5 | flat f32 AVX2 | 1536 | 1.000 | 4503 [4414-4582] | 5825 | 0 |
| 1e5 | int8 flat (+rerank 10x) | 388 | 0.974 (1.000) | 1487 [1435-1527] | 1834 | 0.12 |
| 1e5 | turbovec 4-bit, no rerank | 196 | 0.848 | 1030 [1021-1045] | 1398 | 0.13 |
| 1e5 | turbovec 4-bit + rerank 4x, 1 thr / 2 thr | 196 (+1536) | 1.000 | 1103 / 561 | 1534 / 751 | 0.13 |
| 1e5 | HNSW M16 efc200 ef64 / ef128 (lib) | 6211 | 0.978 / 0.990 | 429 / 753 | 969 / 1411 | 56.5 [54.9-57.3] |
| 1e5 | HNSW M16 efc200 ef64 / ef128 (avx2, 1 build) | 6177 | 0.969 / 0.984 | 354 / 684 | 728 / 1250 | 50.6 |
| 3e5 | flat f32 AVX2 | 1536 | 1.000 | 13075 [11890-13982] | 18073 | 0 |
| 3e5 | int8 flat (+rerank 10x) | 388 | 0.964 (1.000) | 4124 (4446) | 4831 (5265) | 0.30 |
| 3e5 | turbovec 4-bit + rerank 4x, 1 thr / 2 thr | 196 (+1536) | 1.000 | 3341 / 2165 | 4630 / 2945 | 0.46 |
| 3e5 | HNSW M16 efc200 ef64/128/256 (lib) | 6231 | **0.841 / 0.882 / 0.923** | 493 / 757 / 1185 | 1265 / 1808 / 3135 | **188 (1 build only)** |
| 1e6 | flat f32 AVX2 | 1536 | 1.000 | 45430 [44404-46260] | 64559 | 0 |
| 1e6 | int8 flat (+rerank 10x) | 388 | 0.963 (1.000) | 14682 (14590) | 17798 (17344) | 1.19 (1 build) |
| 1e6 | turbovec 4-bit, no rerank | 196 (+~12 MB scratch) | 0.772 | 11421 | 13135 | 1.45 (1 build) |
| 1e6 | turbovec 4-bit + rerank 4x, 1 thr / 2 thr | 196 (+1536) | 0.999 | 10843 / 6312 | 12170 / 8241 | - |
| 1e6 | turbovec 2-bit + rerank 16x, 1 thr | 100 (+1536) | 0.969 | 8035 | 9998 | 1.02 (1 build) |
| 1e6 | HNSW | **SKIPPED**: 6.2 KB/vec measured at 1e5/3e5 => ~6.3 GB graph + 1.5 GB data (projected 7.7 GB) > 3 GB cap | | | | |
Run-1 (loaded) numbers for the same rows are 1.5-2.5x slower in p50 and 3-10x worse in p99 (see `g15_run1/g15_scale.md`) - recall and bytes/vector were identical.
Negatives: HNSW at M16/efc200 **loses recall with N on this synthetic mixture (0.99 @1e5 -> 0.78-0.92 @3e5 for ef 32-256)** and was not re-tuned (M32/M48 not run at 3e5); the exact-rerank
flat-quantised scan (turbovec 4-bit + 4x rerank, recall 1.000 up to 3e5, 0.999 at 1e6) beats it at that scale on recall while being 2-6x slower per query. Synthetic noise-heavy clusters
make 1e6 neighbours nearly equidistant, so recall numbers at large N are pessimistic for approximate methods on real embeddings (not verified: no real fixture >= 1e5). Only ONE HNSW build at 3e5 and
one build rep for 1e6 quantisers (RAM/time rule), so no build spread there. Bytes at 1e4 turbovec 479-636 B/vec are the fixed-scratch artefact, see tvmem.

---
### VERIFY REQUEST — W16 — max-vectors-per-GB (index bytes only; decimal GB; d=384 unless stated)
Source: `g15_quant_*.json`, `g15_scale*.json` nominal/heap bytes. "needs f32" = exact-rerank copy NOT counted (it would add 1536 B/vec of RAM, SSD cost unmeasured).
| rank | config | B/vec | M vectors/GB | recall@10 (mini23k / synth1e5 / synth1e6) |
|---|---|---|---|---|
| 1 | turbovec 2-bit (+16x rerank, needs f32) | 100 | 10.0 | 0.924 / 0.557 / 0.361 (rerank16x: 1.000 / 1.000 / 0.969) |
| 2 | turbovec 3-bit | 148 | 6.8 | 0.960 / 0.726 / n.m. |
| 3 | turbovec 4-bit (+4x rerank, needs f32) | 196 | 5.1 | 0.977 / 0.848 / 0.772 (rerank4x: 1.000 / 1.000 / 0.999) |
| 4 | int8 flat (no f32 needed) | 388 | 2.58 | 0.997 / 0.976 / 0.963 |
| 5 | block-fp16 / f16 flat | 816 / 768 | 1.23 / 1.30 | 1.000 / 1.000 / n.m. ; 1.000 / 0.999 |
| 6 | flat f32 | 1536 | 0.65 | 1.000 |
| 7 | HNSW M16 (graph+data) | 6.2-6.5 KB | 0.15-0.16 | 0.98-0.99 (1e5), 0.78-0.92 (3e5) |
| 7' | HNSW M32 / M48 | 6.8 / 7.1-8.7 KB | 0.12-0.15 | 0.96-0.99 |
d=768: flat f32 3072 B (0.33 M/GB), int8 772 B (1.30 M/GB, 0.993), turbovec 4/3/2-bit 388/292/196 B (2.6/3.4/5.1 M/GB, 0.976/0.958/0.931), HNSW M16 7.4 KB (0.13 M/GB), M48 8.7 KB.
**Best max-vectors/GB at recall >= 0.95 with nothing but the index resident**: turbovec 3-bit on the easy real sets (6.8 M/GB mini, 3.4 M/GB bge), int8 flat (2.58 M/GB) on the harder synthetic 1e5-1e6 sets (the only
index-only config >= 0.96 there). **With f32 on cheap storage**: turbovec 4-bit codes + 4x rerank gives recall 0.999-1.000 at 196 B/vec RAM = 5.1 M vectors/GB RAM up to N=1e6 (the SSD random-read cost of 40 rows/query is unmeasured).
HNSW is the worst per GB (~4x flat f32), which is the price of its low latency at N >= 1e4.

---
### VERIFY REQUEST — W16 — concurrent query throughput on ONE shared index (+ concurrent inserts)
Rerun (~8 min): `$B conc --out results/g15_verify` ; files `g15_conc.json`, `g15_conc.md` (run 1: `g15_run1/`). N=1e5 synthetic d=384, 4 s windows x 3 reps (interleaved HNSW arms so drift hits both). HNSW M16 efc200 ef128 (recall 0.99 measured single-thread)
shared as `Arc<Hn>` with NO lock (hnsw_rs `search`/`insert_slice` take `&self`); turbovec 4-bit + 4x rerank behind `RwLock` (turbovec `add` needs `&mut`), each reader has its own 1-thread rayon pool; max 4 reader threads (+1 writer).
| index | scenario | threads | qps [min-max] run 2 | p50 / p99 us | run-1 qps |
|---|---|---|---|---|---|
| HNSW | read-only | 1 / 2 / 4 | 1122 [1033-1182] / 2302 [2079-2481] / 4673 [4343-4865] | 859/1712, 832/1668, 811/1641 | 1229 / 2402 / 4367 |
| HNSW | readers + 1 insert thread | 1 / 2 | 1148 [1098-1200] / 2273 [2203-2380] (writer 693-697 inserts/s) | 836/1620, 832/1675 | 1270 / 2497 (writer 752-762/s) |
| turbovec4+rr4 | read-only (RwLock) | 1 / 2 / 4 | 685 / 1492 / 2780 [2340-3263] | 1414/2409, 1183/3232, 1091/6944 | 799 / 1767 / 2204 |
| turbovec4+rr4 | 2 readers + writer adding 64 vec / 50 ms | 2 | 1077 [641-1603] (writer 1,087 vec/s) | 1303 / **18,106** | 647 (p99 11,973) |
| flat f32 AVX2 | read-only | 1 / 2 / 4 | 178 / 300 / 419 | 5421/8601, 6292/12324, 8129/33597 | 195 / 222 / 284 |
Findings: HNSW read-only scales 4.2x on 4 threads (3.6x in the loaded run) with flat p50 and no lock; one concurrent inserter running flat out (~700 inserts/s at efc200) does NOT measurably reduce reader qps or p99
(1148 vs 1122 qps, within the 10 % rep spread). Flat scan scales only 2.4x (memory bandwidth bound). turbovec scales 4.1x, but inserts are expensive: every `add` invalidates the blocked-code cache, so the first search after each
batch repacks the index -> p99 18 ms and -28 % qps with a writer. Also vectors added after the rerank f32 store was built are invisible to rerank in my harness (ids >= N skipped) - the 2-reader+writer row measures lock/repack cost, not correctness.
Caveats: 16-core box shared with other jobs, so 4-thread numbers include contention; each window is only 4 s; insert recall during growth not measured.

---
### VERIFY REQUEST — W16 — flat-scan vs HNSW crossover
Rerun (~12 min): `$B flat --out results/g15_verify` ; files `g15_flat_crossover.json`, `g15_flat.md` (run 1 in `g15_run1/`). d=384 synthetic, one graph (M16 efc200) grown through N = 1e3..1e5; HNSW row = smallest ef in {16..512} reaching recall@10 >= 0.95; p50 us, mean of 3 passes.
| N | flat f32 AVX2 | flat f32 lib | int8 scan + rerank 10x | turbovec 4b + rerank 4x (1 thr) | HNSW (ef) |
|---|---|---|---|---|---|
| 1,000 | 25 | 45 | 36 | 68 | 83 (16) |
| 2,000 | 49 | 84 | 65 | 92 | 138 (16) |
| 5,000 | **131** | 218 | 117 | 134 | 213 (16) |
| 10,000 | 503 | 574 | **151** | 222 | 212 (16) |
| 20,000 | 1020 | 1245 | 362 | 377 | **199** (16) |
| 30,000 | 1738 | 1818 | 511 | 555 | **186** (16) |
| 50,000 | 2744 | 3005 | 893 | 494 | **199** (16) |
| 100,000 | 5450 | 6352 | 1768 | 1333 | **406** (48) |
(run 1: flat 27/53/133/379/822/1307/2220/4552 vs HNSW 75/94/176/174/151/158/178/350; both runs: crossover HNSW < flat-f32-AVX2 at N=10,000.)
**Crossover: exact flat wins up to ~5k, HNSW wins from ~10k (2-3x at 10k, 8-9x at 30k), int8-scan+rerank (recall 1.000) ties HNSW at ~10k and is the best exact option below that.** This does NOT reproduce the earlier
"flat fastest at 30k": here the AVX2 flat scan is 9x slower than HNSW at 30k with HNSW recall 0.97-0.98 (ef 16-24). The earlier observation likely used higher-ef/recall HNSW or the slower library kernel plus a different data distribution;
I did not find which. HNSW M16 recall plateaus ~0.98 on this data, so at a >= 0.99 target the HNSW row needs M32+ (see sweep) - it is still >= 5x faster than flat at 30k.

---
### VERIFY REQUEST — W16 — turbovec memory anatomy (counting allocator) [side finding]
Rerun (~1 min): `$B tvmem --n 100000 --out results/g15_verify` ; files `g15_tvmem_n100000.json`, `g15_tvmem_n300000.json`. After `add` in chunks, turbovec 1.0 keeps ~chunk_rows x dim x 4 bytes of scratch alive:
384-d, 4-bit, N=1e5: single add of all rows 196 B/vec (exact nominal 192 + 4); 50k-row chunks 964 B/vec (= +77 MB retained; at N=3e5 +96 MB, 4-bit) ; 10k-row chunks +22-27 MB; 1k-row chunks +3-5 MB (2-bit: 100 -> 868 / 312 / 122 B/vec at 1e5). All
`g15_scale`/`g15_quant` numbers use 2048-row adds (fixed overhead ~3.1 MB at d=384, ~6 MB at d=768, which is why B/vec for N<=2.3e4 turbovec rows reads 2x-5x nominal). `src/memory.rs::TurbovecIndex::bytes()` is fine for per-vector cost; the proposal (P4) is to bound batch size.

---
### Not done / not measured (blunt list)
* HNSW at N=1e6 (memory), HNSW at 3e5 with M32/M48 or more than one build, any real-embedding corpus >= 1e5 (none exists in `results/`).
* SSD/mmap rerank, persistence/load time, deletes, filtered search, insert recall, HNSW `parallel_search`, concurrency > 4 threads.
* Single run per mode under shared-box load: recall/bytes reproduce tightly; latencies carry +-50 % (run-1 vs run-2 shows up to 2-3x). The sweep frontier uses one build per cell.
* PQ-lite uses plain k-means (8 iters, 20k sample), no OPQ / no SIMD ADC; a tuned PQ (FAISS-style fast-scan) could be much better than my negative result.
* Real-set queries: mini = held-out base rows (optimistic: near duplicates of other base rows exist), bge = real instruction queries; both corpora are templated synthetic text.

Proposed patches (not applied): `results/g15_proposed_patch.md` (P1 AVX2 dot, P2 HNSW defaults M32/efc100/ef24 or M48/efc200/ef16 for >=0.99, P3 auto flat->HNSW switch at ~8k, P4 turbovec batch size <= 2048 rows, P5 int8 flat backend).
