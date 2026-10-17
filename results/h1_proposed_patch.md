# H1 proposed patches (files I do not own; nothing below is applied)

All additions I made live in `src/embedding.rs` (`MemoEmbedding`, `Embedded`) and `src/memory.rs`
(`CachedIndex`, `impl MemoryIndex for Box<T>`, `hnsw_integrity_probe`). Everything is opt-in; defaults unchanged.
The wiring below needs files owned by others.

## 1. casebased.rs / shortlab-style compositions: NO source change needed
`MemoEmbedding::task_embedder()` returns exactly the `casebased::Embedder` closure type:

```rust
let memo = Arc::new(MemoEmbedding::new(real_encoder, 4096));
let case = CaseRouter::new(a, dim, cfg, memo.task_embedder());          // select/update/local_mass share one encode
let kn   = Knowledge::new_shared(kc).with_embedder(memo.clone());       // knowledge retrieval shares it too
```
Measured on the shortlab-style `Warm<CaseRouter>` + `Knowledge` composition: 3.64 encoder calls/decision -> 0.98 (per-decision memo, cap 4)
-> 0.40 (cap 4096 over the repeat-heavy f5/f6 worlds), decisions bit-identical (results/h1_embed_ident.json).

## 2. research.rs (ResearchSession): optional cross-decision memo for repeated task texts
`ResearchSession::run_task` already encodes ONCE per task (`self.emb.embed_task(task)`), so the within-decision saving is 0 there.
The only saving is across decisions when the same task text recurs. Opt-in, `cfg.embed_memo: usize` (serde default 0):

```rust
// ResearchConfig: #[serde(default)] pub embed_memo: usize,     // 0 = off (default)
// ResearchSession::new, after build_or_hash:
let emb: Arc<dyn Embedding> = if cfg.embed_memo > 0 { Arc::new(MemoEmbedding::new(emb, cfg.embed_memo)) } else { emb };
```
(`EmbeddingConfig::Real{cache:true}` already does the same through `CachedEmbedding`; MemoEmbedding is the same semantics with an
allocation-free exact key and an encoder-call counter, so this is a convenience, not a new capability.)

## 3. serve.rs / tiered.rs: result cache in front of the long-term index
`CachedIndex<I>` implements `MemoryIndex` and `RemovableIndex` (every write/remove/compact clears the cache). To put it under a
`TieredMemory` hot tier add a `LongKind::Cached(Box<LongKind>, usize)` variant (tiered.rs, not mine) building
`Box::new(CachedIndex::new(inner_removable, cap))`; searches through `search_ro`/`search_tiered` are `&self`, which `CachedIndex`
supports (interior Mutex). Only worthwhile when the same query vector recurs (see h1_cache table); with no repeats it costs a hash
and a lock per query.

## 4. Cargo.toml / dependents
`[patch.crates-io]` is honoured only in the workspace root. Anything that depends on `neural-mesh` as a library gets the UNPATCHED
hnsw_rs silently. Either repeat the `[patch]` in the dependent root, or call `neural_mesh::memory::hnsw_integrity_probe(2000, 32, 1)`
once at startup (0 = fixed crate; about 14 of 2000 lost with the unpatched crate; ~0.3 s).
Defence in depth that I did NOT enable: `Hnsw::modify_level_scale(0.2)` in `HnswIndex::new` (g17: 0 misses on the unmodified crate up to 30k points,
recall/latency equal) because it prints two lines to stdout per index and is untested above 30k points.

## 5. tiered.rs (HnswRm) / workmem Store: call the audit after a bulk build (opt-in)
The fixed hnsw_rs still produces a rare build (about 0.5 % at N=2000 d32) in which 8-37 % of the items are unreachable (results/h1_board_entry.md, H1a).
`HnswIndex::audit_and_repair(sample, max_lost, max_rebuilds)` (memory.rs, new) detects it (8/8, 0 false alarms in 1992 builds) and one rebuild repaired 8/8.
Proposed call sites (not mine):
```rust
// tiered.rs HnswRm::compact(), after `idx.add_batch(&ids, &flat);`
idx.audit_and_repair(200, 3, 2);          // ~5 % of the build time; 0 effect on a healthy graph
// workmem.rs Store::open_with(), after the snapshot/replay rebuild of `hnsw`
if let Some(h) = s.hnsw.as_mut() { h.audit_and_repair(200, 3, 2); }
```
Per-add `HnswIndex::add` cannot be audited at build time; call `audit` periodically (e.g. every 10k adds) if the index is long-lived.
