# K1 proposed patch (NOT applied; I own no src files). All opt-in, defaults unchanged. Evidence: results/k1_combined.md, k1_board_entry.md.

Framing: knowledge ACCESS/MEMORY layer only (what the store keeps and returns); nothing here adds knowledge to the frozen model. All numbers synthetic-until-validated (real text/structure, synthetic edits/arrivals/outcomes).

## P1. Structural-key supersession in the ingest path (src/dedup.rs or a new `src/supersede.rs`)
What the data supports: a row carries `(key: u64 hash of source path + section/paragraph ordinal, rev: u32 revision stamp (commit id / mtime / version counter))`. On ingest:
* a row with the same key and a higher `rev` already retained -> drop the incoming record (late/stale arrival), count it;
* same key, equal `rev` -> merge into the existing row (multiplicity+1; this also swallows exact/near duplicates of the same source);
* same key, lower `rev` rows retained -> delete them, then store the incoming row.
Measured (k1_growth `supdel`; worlds A and B, held and confirm): bytes -35.2 % (rows -35.3 %, 64.7-64.8 % kept), top-1 superseded share 8.4-9.8 % -> 0.0 %, hit@1 +0.014..+0.017, MRR +0.029..+0.031 (Holm-significant gains in both stages in both worlds, in a store where 16 % of retained rows are superseded versions; the gain scales with that share, see the dose-response table). Without revision stamps (arrival order only) the 10 % late-arriving old records poison it: see the post-hoc `supdel-arrival` row.
Interface sketch: `Superseder::check(key: u64, rev: u32) -> Verdict{New, Duplicate{of}, Stale{newer}, Supersedes{old: Vec<u64>}}` with the same `forget` / `to_json` contract as `Deduper`; `ingest` removes the `old` ids from the `FlatIndex` (tombstone + `compact`). Soft variant: keep superseded rows and subtract a penalty (lambda >= 2 z-units on the fused score) -- same quality, no byte saving.
Do NOT implement it as a cosine rule (see negatives in the board entry): `cosnew` (merge into the nearest row if cos >= theta, keep the newest) only works at theta .75-.85 where it also merges unrelated siblings and it HURT by -0.0095 hit@1 on a stream without revisions.

## P2. Age tiering with an int8 cold tier in `flatq` (src/flatq.rs / tiered.rs)
`TieredFlat { hot: FlatIndex(F32), warm: FlatIndex(BlockFp16), cold: FlatIndex(Int8) }` with a demotion pass by arrival age (newest 20 % hot / next 30 % warm / rest cold) or, if usage is tracked, by recency of use; search = score all three (or only the hot+warm with an exact rerank of cold hits), merge by score. Measured (`tier.2/.3`): -46.7 % total bytes (-51.4 % dense), hit@1 / MRR diffs <= +-0.0003 (EQ +-0.01 AND +-0.005 in both stages, both worlds), closed loop 10/10 cells EQ in both stages. All-int8 (`tier-int8`): -67.9 % total bytes, EQ in both stages both worlds. All-block-fp16: -42.5 %, bit-identical retrieval. Note the 8 % sparse BM25 postings and 24 B/row metadata are untouched, so the total saving is below the dense saving.

## P3. Optional exact-dup suppression stays as is (src/dedup.rs)
No change; reproduced: -11.0 % rows/bytes, hit@1 +-0.0002 (EQ), MRR +0.0053..+0.0058 (gain, because duplicate rows no longer crowd the top-10). It is a no-op without duplicates.

## NOT proposed (measured negatives)
* byte-budget eviction FIFO / LRU / usage-value (kept at 25-50 % bytes): every variant loses 0.007..0.12 hit@1 on uniform queries; usage-value is the least bad and beats the unbounded store only under popularity-weighted queries with informative feedback (conditional on the synthetic Zipf popularity; the uninformative-feedback control loses -0.071).
* anchor-summary rows keyed by structural section: -0.004..-0.0075 hit@1 (Holm-significant) for -20 % bytes only; stale versions inflate in the summaries unless supersession runs first.
* cosine-threshold merge keep-first (`cosfirst`): quality drops monotonically with theta, and it never removes staleness.
