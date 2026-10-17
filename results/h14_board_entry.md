# H14 / R6-B: productionised opt-in library types (popguard, flatq, dedup)

All numbers SYNTHETIC-until-validated-on-real-traffic. Opt-in only; defaults unchanged; Cargo.toml untouched. Only edit outside owned files: three lines appended to `src/lib.rs` (`pub mod popguard; pub mod flatq; pub mod dedup;`).
New files: `src/popguard.rs`, `src/flatq.rs`, `src/dedup.rs`, `src/bin/h14_popguard.rs`, `src/bin/h14_dedup.rs`; outputs `results/h14_popguard.{json,log}`, `results/h14_dedup.{json,log}`.
Tests: full `cargo test --release --offline --lib` = **286 passed, 0 failed** (was 249 + R3/R4 additions; mine: popguard 5, flatq 8, dedup 5 own + 1 pre-existing `lessons::dedup_*` matches the filter).

## What was built
1. `popguard`: `PoolLog` (record/serde), `FitCfg`/`fit_prior` (hierarchical shrinkage, cov_min), `TrustCfg`, `PopGuarded` (implements `Router`), `recommended_warm_router(map, n, &pool)` with the frozen H9 pick (M16, m1=1, n_min 5, cov .5, inf 8, w0 .8, cf 1, cg .5, delta .97). Not ported (as the H9 patch says): PopTs, PopMap, kw-cal.
2. `flatq`: `FlatIndex` with `Tier::{F32, BlockFp16, Int8}`, Dot/Cosine, optional exact rerank (f32 copy in RAM, top factor*k re-scored), add (upsert)/remove (tombstone)/compact, exact `bytes()`, JSON persistence, and `suggest_tier(n, dim, latency_ms, ram_bytes) -> Advice` (port of the H6 gate rule). Std-only, no SIMD (portable; H6's AVX2 kernels in g15_common are faster).
3. `dedup`: `Deduper` (normalised-text hash and/or L2-normalised vector hash, 2x64-bit FNV, configurable case folding / punctuation / mantissa-drop bits), multiplicity counts, `forget`, serde, and `ingest(dd, flatindex, ..)`/`remove` helpers.

## Divergences from the source bins (disclosed)
- popguard: `PoolLog::record` clamps reward to [0,1] (h9 `log_dep` did not; identical for the 0/1 rewards used). `PopPrior` drops the unused `kw_acc`/`agent_mu` fields. Decision/update code is a copy; no re-tuning.
- flatq: scores computed with plain f32 loops (summation order differs from AVX2 kernels, so scores differ at ~1e-7); int8 query stays f32 (h6 `Int8` variant), the int8-query variant is not offered. turbovec / RaBitQ / disk-rerank exist only as `Advice` values (advice only, `flat_config()` returns None); `suggest_tier` p50 points are the rounded values printed in results/h6_gate_rule.md and latency/RAM are scaled linearly in dim (model output for dim != 384).
- dedup: library decides from hashes, h6 used labels it attached (oracle). Result differs by exactly the natural duplicates in the corpus (see below).

---
### VERIFY REQUEST - R6-B - popguard parity with the h9 bin and h9_priors_confirm61-90.json
```
export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r6 RAYON_NUM_THREADS=2
cargo test --release --offline --lib popguard
cargo build --release --offline --bin h14_popguard && $CARGO_TARGET_DIR/release/h14_popguard.exe   # ~6 s, writes results/h14_popguard.json, exit 1 on FAIL
```
Result: ALL 14 h9 worlds x seeds 61-90 (5040 world/seed/metric values): library `fit_prior` cells bit-equal to h9's; library `PopGuarded` success curves (f32) and 12 metrics **bit-equal** to `h9_common::PopGuarded` in-process; max |dev| vs the numbers stored in `results/h9_priors_confirm61-90.json` (policy `pop-guard+conf[M16,m1=1,cov0.5,inf=8][w0=0.8,cg=0.5]`) = 5.7e-14 (1 ulp of a ~100 value; serde_json's default float parser is not bit-exact, so the json comparison cannot be stricter). Empty-prior library PopGuarded == `defaults::recommended_router` curves bit-identical in all 14x30 runs (and in the unit test over 3 seeds x trust on/off x inf {0,2,8}).
Means (success, seeds 61-90): A1-sim-fixed s50/s100/s250/s500/s2000 = .8262/.8262/.8250/.8202/.7802; B1-md-related .7858/.7933/.8045/.8192/.8344; B6-md-wrong (wrong prior) .7641/.7780/.7912/.8073/.8356; B7-md-equal .7000 flat. These equal the h9 log lines (A1 .8262 .8262 .8250 .8202 .7802). The h9 conclusions stand unchanged and are NOT re-derived here: early gain +0.08 at H50 fading by ~2000, wrong-prior loss bounded (-0.035 worst at A5 H50 vs best baseline), loses to best baseline in several control worlds at early horizon (h9 log); opt-in conditional on K>=2 related prior deployments.

### VERIFY REQUEST - R6-B - flatq correctness (recall/bytes/remove/compact/gating)
```
cargo test --release --offline --lib flatq -- --nocapture
```
8 tests: F32 == independent brute force; **block-fp16 recall@10 vs f32 = 1.00000 and identical ordered top-10 in 200/200 queries** (n=3000, d=96, gaussian, seeded; one dataset, not a proof); **int8 recall@10 raw 0.984, with rerank 4x 1.000** (>=0.99 asserted; raw int8 here is 0.984 < 0.99 on this data, consistent with H6 saying int8 alone is only for small N); byte accounting exact for 6 tier/rerank configs (`bytes() == n*row_bytes` incl. tombstones until compact; d=384: 1536 / 816 / 388 B), dim not multiple of 32 and zero vectors; remove/compact results identical before/after compact and match brute force (F32 exact; BlockFp16/Int8+rerank recall >=0.99); replace-on-add; Dot metric; error cases; JSON round trip; `suggest_tier` reproduces 30 cells of the H6 rule table (N 1e4..1e6, 1/5/20 ms, 128MB/512MB/2GB).
Not verified: speed (no benchmark; do not quote latency from this module), real embeddings, N>=1e5 behaviour of the library scan.

### VERIFY REQUEST - R6-B - dedup unit tests + doc-level exact-duplicate effect on F1
```
cargo test --release --offline --lib dedup::
cargo build --release --offline --bin h14_dedup && $CARGO_TARGET_DIR/release/h14_dedup.exe   # needs results/g1_* fixtures, ~1 min, writes results/h14_dedup.json
```
Same stream as `h6_docq ctx` (30 % of docs re-ingested, same Rng seed), MiniLM+spell. Library keeps 779 (A) / 780 (B) of 1002 rows vs the label oracle's 780: in corpus A one natural duplicate text exists, so the library removes 1 row more than h6. The no-dedup and oracle F1 recomputed here match `results/h6_ctx.json` exactly (48 values, max dev 0). Library F1 at fixed k=2: A .6840 -> .7195 (+0.0355, CI [+0.026,+0.045]), B .6871 -> .7117 (+0.0247, CI [+0.016,+0.034]) = h6's +0.0355/+0.025. Text-only, vector-only and combined give identical kept sets on this stream (every duplicate is both text- and vector-identical).
**Caveat (blunt):** F1 is better only at k=2 (the knee by construction: 2 relevant docs). At k=3,4 F1 FALLS with dedup (A -0.040, -0.057; B -0.043, -0.052) because the metric counts duplicate rows as relevant hits in precision, so the no-dedup store is inflated there. The cleaner measure, distinct-relevant-doc recall, improves at every k>=2 (A k=2 .642 -> .720, k=3 .754 -> .789, k=4 .795 -> .823) and k=1 is unchanged. Context chars unchanged at fixed k. Closed-loop vote-store result from H6 (rows -90 %, one of 4 TOST cells inconclusive) was not rerun here.

## Final summary
popguard: faithful, bit-parity proven (14 worlds x 30 seeds); flatq: tiers/bytes/remove/compact/gating tested, no speed claims; dedup: reproduces h6's k=2 F1 gain, with the F1-metric caveat above. 286 lib tests green. Nothing is wired into defaults.
