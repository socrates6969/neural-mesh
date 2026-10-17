# H8 / R4-V proposed patches (NOT applied; src/*.rs untouched)

Reference code for every item: `src/bin/h8_common/mod.rs` (kernels, `BfpDist`/`I8Dist`, `FileRows`, `RaBitQ`) and `src/bin/h8_vectors.rs`.
Numbers: `results/h8_board_entry.md`, `results/h8_lat2.md`, `results/h8_memory.md`, `results/h8_scale.md`.
All data are synthetic-or-templated (see board entry); latency numbers are the load-robust "p99 of per-query minimum over 7 interleaved passes".

## Q1. HnswParams defaults on the FIXED crate: M32 / efc200 / ef_search 16 (quality) or M24 / efc100 / ef 16 (build speed)
With the reverse-edge fix the recall ceiling is gone (1.000 reachable) and the hnsw_rs floor `ef >= k = 10` is the knee.
On three datasets (synthetic 384-d N=30k, real MiniLM N=23k, real BGE-768 N=23k; 3 builds each) `M24..32, efc 100..200, ef 16`
gives recall@10 0.993-0.999 at p99 ~ 0.17-0.20 ms; ef 10 gives 0.97-0.995 at p99 ~ 0.13-0.17 ms.
Current `HnswParams` defaults (M16/efc200/ef64) cost ~2x the latency of ef16 for +0.001..0.005 recall.
Guard (measured, `h8_selfcheck.md`): **efc must be > 2*M, use efc >= 3*M** (hnsw_rs level-0 lists hold 2M entries). With efc <= 2M items become unreachable *even at ef 512*, worst at efc = 2M:
synthetic N=30k: M24/efc48 34-41 % lost, M32/efc64 29-36 %, M32/efc50 15 %, M48/efc96 33-42 %, M48/efc80 10-14 %, M64/efc128 37-42 %, M64/efc100 5 %; MiniLM 0.01-6 %. Clamp `efc = max(efc, 3*M)` in `HnswIndex::new` (all current defaults already satisfy it).
Also keep efc >= 200 for BGE-like templated data: efc 100 leaves 0.04-0.08 % of items unreachable even at ef 512 (efc 200: 0).

## Q2. Keep P1 (AVX2+FMA runtime-dispatched dot) from g15 - re-measured
Paired, interleaved, 3 builds x 3 datasets (`h8_lat2`): avx2/lib p50 ratio 0.82-0.92 (median 0.87), p99 -10..-20 %, recall unchanged. Still not applied in `memory::dot`.

## Q3. Optional `HnswIndex` storage: block-fp16 points (custom `Distance<i16>`), NOT int8
`Hnsw<i16, BfpDist>` with the g15 block-fp16 layout (32-value blocks, i16 mantissa + f32 scale = 816 B at d=384) and a quantised query:
* measured heap/vector 10-18 % lower than f32 points (6.1 vs 6.9 KB at N=1e5 M32; 5.8 vs 6.6 KB MiniLM; 6.7 vs 8.2 KB BGE-768),
  graph edges dominate (~5 KB/vec of the 6-8 KB) so it cannot do better than ~-18 %;
* recall equal to f32 points (N=1e5 M32 efc100 ef128: 1.000 vs 1.000; 3e5 ef256 0.987 vs 0.993), p50 the same or 10-30 % faster (less memory traffic);
* int8 points are cheaper (-17..-28 %) but cap recall at the int8 quantisation ceiling (0.95-0.974 at N=1e5, 0.958 at 3e5): do not use.
Implementation = 60 lines (`encode_bfp`, `bfp_dot_avx2`, `BfpDist`); the vendored hnsw_rs needs no change (`Hnsw<'static, i16, BfpDist>`).

## Q4. Size gating in `tiered.rs` / `memory.rs` (measured crossovers, d=384 unless noted; see board entry "size-gating rule")
```
N <= 5_000                : FlatIndex (exact f32 AVX2; or block-fp16 flat if bytes matter)
5_000 < N <= 10_000       : flat block-fp16 / int8+f32-rerank (HNSW only ties)
N >= 10_000 (d=384) / >= 20_000 (d=128): HnswIndex M24-32, efc 100-200 (>= 3M), ef 16-24 (2.5-10x faster than flat, 6-7 KB/vec)
bytes-bound (>= ~5e5 vectors per GB RAM needed): turbovec 4-bit codes (196 B/vec) + exact f32 rerank x4 with the f32 rows on SSD
N >= 3e5 and HNSW wanted: M48 efc100 ef >= 128-256 (M16/M32 fall to recall 0.75-0.96 on this hard synthetic set)
```
Switch `LongKind::Auto { switch_at }` should use `switch_at = 10_000` for d>=384 and `20_000` for d<=128 (the old 8_000 proposal is fine for d=384).

## Q5. turbovec accounting fix: 3-bit costs the same as 4-bit
`TurboQuantIndex` 3-bit is stored with the 4-bit footprint (measured with the counting allocator, single add, N=1e5, d=384: 2-bit 100.1, 3-bit **196.1**, 4-bit 196.1 B/vec; nominal 3-bit would be 148).
`TurbovecIndex::bytes()` (and any capacity planning built on "3-bit = 148 B") should use the measured value, or prefer 2-bit or 4-bit. (g15 reported 3-bit at 148 B nominal; that was not measured.)

## Q6. File-backed (SSD) exact-rerank store for `TurbovecIndex` (measured cost, honest)
`FileRows` (std only, `seek_read`, optional `FILE_FLAG_NO_BUFFERING`): f32 rows kept on disk, codes in RAM. Measured on this box (NVMe-class SSD, Windows):
cold direct I/O 70 us/row single-threaded (a x4 rerank = 40 rows = +2.8..4 ms per query), 28k/57k/113k rows/s with 2/4/8 **separate handles**
(a single shared handle serialises at 14k rows/s); warm OS page cache 6 us/row (but then the f32 file occupies RAM as page cache - it is not saved, only evictable).
=> turbovec 4-bit + x4 rerank with SSD rows: 196 B RAM/vec (5.1 M vectors/GB), recall 0.999-1.000 up to 1e6, p50 5.7-6.2 ms (N=3e5, 1 I/O thread) / 3.9 ms (4 I/O threads) vs 0.9-2.9 ms with the f32 rows in RAM.
Worth it only when bytes matter more than latency; the rerank factor needed grows with N (2-bit: x16 at 1e6).

## Q7. Not recommended: RaBitQ-style multi-bit codes (my std-only implementation)
Equal to or slightly below turbovec per code bit (4-bit 0.755 vs 0.772 at 1e6 synthetic; +rerank x4 0.997 vs 0.999), its only win is a real 3-bit packing (148.6 vs 207 B/vec measured), and my AVX2 bit-plane scan is 1.5-3x slower than turbovec's. See board entry.
