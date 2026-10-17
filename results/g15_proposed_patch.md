# G15 / W16 proposed patches (NOT applied; src/*.rs untouched)

All numbers are from `results/g15_*.json` (see `results/g15_board_entry.md` for rerun commands and the load caveats).
The reference implementation of every kernel below is `src/bin/g15_common/mod.rs`.

## P1. AVX2+FMA runtime-dispatched dot in `memory.rs` (CosineDist + FlatIndex + rerank)  -- recommended

`memory::dot` is an 8-lane scalar loop that the default (no `-C target-cpu`) build autovectorises to SSE2. A
`#[target_feature(enable="avx2,fma")]` kernel selected by `is_x86_feature_detected!` needs no build flags:

```rust
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return unsafe { dot_avx2(a.as_ptr(), b.as_ptr(), a.len().min(b.len())) };
    }
    dot_scalar(a, b) // current body
}
// dot_avx2: 2x _mm256_fmadd_ps accumulators, 16 floats/iter, horizontal sum, scalar tail (g15_common::dot_avx2)
```

Measured (same graph recipe, kernel is the only difference; 3 rebuilds each; loaded box, so read ratios not absolutes):

| setting | lib kernel | AVX2 kernel |
|---|---|---|
| HNSW build, N=1e5, M16 efc200 (2 threads) | 116.5 s (loaded run) / 56.5 s (quiet run) | 87.5 s / 50.6 s |
| HNSW query p50, N=1e5, ef128 | 753-963 us | 677-684 us |
| HNSW query p99, N=1e5, ef128 | 1.4-3.3 ms | 1.2-1.3 ms |
| confirm real-bge M16 efc100 ef10 p99 (3 builds) | 316-378 us | 201-216 us |
| confirm real-mini M32 efc100 ef10 p50 | 81-93 us | 75-79 us |
| flat exact scan p50, N=1e3 / 5e3 / 1e4 | 45-51 / 218-229 / 504-574 us | 25-27 / 131-133 / 379-503 us |
| flat exact scan p50, N=1e5 (memory-bound) | 4.8-6.4 ms | 4.5-5.5 ms (only 7-16 % faster) |

Recall is unchanged within build noise (+-0.01). It is a free 10-30 % on p50 and a larger gain on p99.
Risk: `unsafe` + x86_64 gating; keep the scalar fallback for non-x86 / non-AVX2.

## P2. `HnswParams` defaults: M16/efc200/ef64 -> M32/efc100/ef24 (or M48/efc200/ef16 for recall >= 0.99)

On all three datasets (synthetic 384-d N=30k, real MiniLM N=23k, real BGE-768 N=23k; see `g15_hnsw_confirm_*.json`)
the lowest-p99 configs reaching recall@10 >= 0.95 / 0.99 were M32 (or M48) with SMALL ef and small efc. ef_search
below k is silently raised to k by hnsw_rs, so ef=10 is the floor; at ef=10 M32 already gives recall 0.955-0.97.
Cost: ~+10 % bytes/vector (6.8-8.2 KB vs 6.0-7.5 KB) and efc100 builds are 2-3x faster than efc200.
Do not pick M8: recall 0.57-0.85 at ef=10 and no latency win. Keep `ef_search` = 24 for >= 0.99 (M32 efc100 ef24 -> 0.990).

## P3. Auto flat/HNSW switch in `tiered.rs` (`LongKind::Flat` default)

Crossover (dim 384, synthetic, single thread, recall >= 0.95 HNSW vs exact AVX2 flat, `g15_flat_crossover.json`):
flat wins up to N~5k (131-133 us vs 176-213 us), HNSW wins from N=10k (174-212 us vs 379-503 us) and by 8-9x at N=30k
(158-186 us vs 1.3-1.7 ms; two independent runs, `results/g15_run1/` = first run). An int8-scan + f32-rerank flat
store (recall 1.000) is about as fast as HNSW at N=10k (151-206 us vs 174-212 us) and the fastest option at 5k (111-117 us).
HNSW recall plateaus at ~0.98 for M16 (needs M32+ for >= 0.99), flat/int8+rerank are exact.
Suggest `LongKind::Auto { switch_at: 8_000 }`: Flat below, HNSW(M32,efc100,ef24) above.
Caveat: the earlier "flat was fastest at 30k" observation does not reproduce here with M16/efc200/ef16-48.

## P4. `TurbovecIndex::add_batch`: bound the batch size (<= 2048 rows)

turbovec 1.0 retains a scratch buffer of ~(rows of the LAST multi-add batch) x dim x 4 bytes after `add`
(measured with the counting allocator, `g15_tvmem_n100000.json`/`..._n300000.json`: add in 50k-row chunks leaves
+77-96 MB at dim 384 (4-bit, N=1e5/3e5); 10k-row chunks +22-27 MB; 1k-row chunks +3-5 MB; a single add of all N leaves none). Feed
`inner.add_with_ids` in 2048-row slices (or a single call when N is known) to avoid a hidden ~chunk-size f32 footprint.
Also `TurbovecIndex::bytes()` is accurate to nominal (codes + 4 B scale) once that scratch is not retained; HNSW
`bytes()` calibration (150 B/edge) matches the counting-allocator measurement (6.2-6.5 KB/vec at 384-d, M16).

## P5. Optional new backend: `Int8FlatIndex` (+ f32 rerank)

Per-vector-scaled int8 (`g15_common::LowStore::Int8`) with an AVX2 `madd_epi16` scan: 388 B/vec at 384-d (4x less
than f32), recall@10 0.997 (real MiniLM), 0.993 (real BGE), 0.974-0.976 (synthetic 1e5), 0.963 (synthetic 1e6);
scan ~3x faster than f32 at 1e5-1e6 (14.7 ms vs 45.4 ms at N=1e6; 1.5 ms vs 4.5 ms at 1e5). Simple, no training, exact after a 4x f32 rerank
(recall 1.000 everywhere measured) if the f32 copy is kept. It dominates the hand-rolled PQ-lite on every axis
measured, and is the right fallback where turbovec's recall without rerank is too low (synthetic N >= 1e5).

## Not recommended
* Block-fp16 / f16 / bf16 flat storage for speed: scan is only ~19-26 % faster than f32 AVX2 (memory-traffic bound);
  the 1.9x byte saving is the real benefit (recall@10 1.000 / 0.999-1.000 / 0.991-0.998, see the board entry).
* PQ-lite (256 centroids, sub-dim 8 or 4): recall without rerank 0.31-0.45 (synthetic 1e5), 0.81-0.89 (real MiniLM),
  with 30x slower build than turbovec and a slower scan; strictly dominated by turbovec 2-bit at equal bytes.
