# H6 proposed patch (NOT applied; no src/*.rs edited by R4-B3)

All numbers: SYNTHETIC-until-validated-on-real-traffic (g1 templated corpora, g16 seeded worlds, g15 synthetic mixture). Opt-in only, defaults unchanged.

## P1. `FlatIndex` storage tier `Precision::{F32 (default), BlockFp16, Int8}`  (src/memory.rs)
Evidence: block-fp16 gave bit-identical routing decisions in the g16 closed loop (both policies, corpus A and B: diff exactly 0.0000 in every one of 8 worlds), 0 of 3,000 top-10 slots differ (g15), 0 recall loss and F1@3/hit@1 diff 0.0000 on 8 doc/chunk sets (h6_docq); 816 B/vec vs 1536 at d=384 (-47 %), scan 20-35 % faster (memory bound).
int8 (per-vector scale, f32 query ok): TOST-equivalent (+-0.01, H2000) in all 4 (policy x corpus) loop cells, worst world diff -0.0022; doc recall@10 0.995-0.999; but id-set recall@10 falls to 0.961 at N=1e6 synthetic -> int8 alone is the compressed tier only below that, and with an exact rerank (f32 in RAM) above.
Sketch (block-fp16 row: 32-value blocks, i16 mantissa, f32 scale):
```rust
pub enum Precision { F32, BlockFp16, Int8 }
// FlatIndex::with_precision(dim, p); add() quantises on insert, search() scans the codes with the AVX2 kernel (g15_common::scan_bfp / scan_i8 are drop-in)
```

## P2. size-gating function (opt-in `StoreConfig::auto`)  (src/memory.rs or src/knowledge.rs)
Rule (validated 27/27 against the measured grid at N=1e4,1e5,1e6 in results/h6_gate_rule.md; p50 single-thread, box was ~1.4x loaded at 1e6 -> on a quiet box thresholds move ~30 % higher):
```rust
/// least-compressed backend whose RAM <= ram_budget, modelled p50 <= lat_budget_us and recall >= 0.99
pub fn auto(n: usize, dim: usize, ram_budget: usize, lat_budget_us: f64) -> Backend {
    // measured ns/vector at d=384, 1 thread (results/h6_gate_grid_*.json); scale linearly in dim/384
    // f32 AVX2 flat 47-66 (p50(1e5)=5.1 ms)   block-fp16 30-44   int8+f32rerank4x(RAM) 16-18   tv4+f32rerank4x(RAM) 11-14
    // f32 rerank copy ON DISK costs +4.6 ms/query (40 unbuffered QD1 reads x ~115 us) -> only when RAM binds
    let cands = [FlatF32, FlatBlockFp16, Int8RerankRam, Tv4RerankRam, Int8RerankDisk, Tv4RerankDisk];
    cands.into_iter().find(|c| c.ram(n, dim) <= ram_budget && c.p50_us(n, dim) <= lat_budget_us && c.recall(n) >= 0.99)
         .unwrap_or(NeedsAnnGraph)   // HNSW (g15: ~6.2 KB/vec, 0.4 ms p50 @1e5) / not measured here beyond 1e5
}
```
Practical consequence at the sizes this repo's knowledge store actually has: N <= ~2e4 -> flat f32 (p50 <= 1 ms); 2e4-1e5 -> block-fp16 if p50 <= 5 ms is required, else f32; > 1e5 the latency budget, not RAM, forces a change (RAM at 1e6 is 1.5 GB f32). Compression below int8 is never chosen by the rule at N <= 1e6 for budgets >= 128 MB.

## P3. exact-duplicate suppression on ingest  (src/knowledge.rs)
Keep one row per identical query vector (hash of the normalised f32 bits; count++). Evidence: closed loop rows 2000 -> 209 (-90 %), H2000 diff within -0.0085..+0.0025 (R0 A/B and FINAL A TOST-equivalent; FINAL on corpus B sib16 -0.0085 CI [-0.018,+0.001] inconclusive); doc-level (30 % re-ingested exact duplicates) F1@2 +0.035 [A], +0.025 [B], context chars unchanged at fixed k. Ship opt-in `KnowledgeConfig::dedup_exact: bool`.

## P4. NOT recommended (measured negative; no patch)
turbovec 2/3/4-bit without rerank as the knowledge-store scan; anchor-summary rows; adaptive-k "to save bytes" in the routing vote; a RaBitQ-style 1-bit scan (own implementation: no faster than f32 flat at 1e5-1e6, recall 0.18 raw / 0.93 with 32x rerank at 1e6).
