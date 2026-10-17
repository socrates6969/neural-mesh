# h4 proposed patches (files I do NOT own; nothing applied)

Context: `src/tiered.rs` (owned) gained opt-in `ForgetPolicy {Value,Fifo,Lru}`, `merge_sim` (importance-merge/absorb), `cold_kind {Turbovec,F16,Int8}`,
`hard_budget`, `ItemStats.weight`, `TieredStats.merged`. All defaults are unchanged (Value, no merge, turbovec cold, budget enforced only at consolidate).

## 1. src/knowledge.rs : make the "knowledge layer" bounded (it is the only component whose memory grows with traffic)
Problem: `recs: Vec<Rec>` is indexed by record id and never shrinks, so even with a tiered backend + byte budget the per-record bookkeeping (and `bytes()` which adds `recs.len()*size_of::<Rec>()`) grows without bound.
Proposal (opt-in via `KnowledgeConfig`, default off):
  * `recs: HashMap<u64, Rec>`; every N remembers sweep `recs.retain(|id,_| tiered.contains(*id))` (Backend::contains).
  * meta is already `a{anchor}`; set `TieredConfig.merge_sim = Some(0.8)` so near-duplicate records of the same anchor are absorbed (weights add; the vote in `retrieve` should multiply `c` by `item_stats(id).weight`).
  * `hard_budget = true`, `forgetting = true`, `byte_budget` = knob. Prefer `ForgetPolicy::Fifo` or merge (see results: value-by-reward did NOT beat FIFO).
  * If LRU is wanted, `retrieve` must call the counting `search` (currently `search_ro`, so `last_access`/`retrievals` never update).

## 2. src/lessons.rs
`LessonConfig.memory` is a plain TieredConfig, so callers can already set `forgetting/hard_budget/policy/merge_sim/cold_kind`. Proposal: expose a `LessonConfig::budgeted(dim, byte_budget)` helper = {forgetting, hard_budget, policy Fifo, merge_sim Some(0.8)} and keep the default unchanged. Lessons have no per-id side tables outside TieredMemory, so no other change is needed.

## 3. src/memory.rs (SharedMemory/CachedIndex are being edited by another worker; no change proposed)
`bytes()` accounting for HnswRm counts the f32 shadow copy; with hnsw the hot per-item cost is ~4x flat, so byte budgets must be set relative to the HNSW baseline (done in h4_memory `hnsw` stage).

## 4. Defaults
Do NOT change any default. Recommended opt-in presets (measured, synthetic):
  * lossless compression only: `compression=Some(Compression{bit_width:4,hot_cap: 0.15*budget/item_bytes}), cold_kind=Int8, forgetting=true, byte_budget=<current full bytes>`  (41% fewer bytes at D=64, 57% projected at D=384, EQ +-0.005 in both worlds)
  * 50% budget: Fifo/Lru/merge/tiered-int8 (EQ +-0.01 in both worlds)
