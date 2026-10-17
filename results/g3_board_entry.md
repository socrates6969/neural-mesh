### VERIFY REQUEST — W3 — compact smallest-viable-brain as a library type (CompactRouter)

What it is: `src/compact.rs` (new, opt-in; one added line `pub mod compact;` in src/lib.rs; no default, config or CLI change). `CompactRouter` implements `router::Router` and is the SAME decision rule as `defaults::recommended_router` (= guarded(hebbian-robust)) with a different state encoding: interned (optionally shared, `SharedVocab`) or 32-bit-hashed feature ids, ONE sparse row per feature (Hebbian weights + baseline + guard counts), u16 weights, Q16 (1/1024) counts. Save/load via serde (`snapshot()`/`from_snapshot()`, `to_json()/from_json()`, `save()/load()`), exact byte accounting (`bytes()` -> `ByteReport`), `fork_blank()` for many brains over one shared vocab/keyword table. Generic core also gives `CompactRouterF32` and `CompactRouterF64`; the f64 instance is decision- AND rng-identical to the shipped router (unit test + 0 diff in all 5 worlds x 220 seeds), so every difference of the u16 instance is representation error, not logic.
Files: src/compact.rs, src/bin/g3_equiv.rs, src/bin/g3_perf.rs, src/bin/g3_common/mod.rs (worlds/stats snapshot of f3_common + TOST helpers), results/g3_equiv.{json,log}, results/g3_perf.{json,log}, results/g3_board_entry.md. std + existing serde/serde_json only; no new crate/package/model.

rerun (machine dependent: only ns and, slightly, commit charge; equivalence and bytes tables are deterministic):
  export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"; export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r3 RAYON_NUM_THREADS=3 G3_THREADS=3; cd /c/Users/mariu/neural-mesh
  cargo test --release --offline --lib compact                      # 11 tests: f64 identity, determinism, serde roundtrip, trait conformance, bytes, limits, shared vocab
  cargo run --release --offline --bin g3_equiv -- --out results/g3_equiv.json     # ~11 s, deterministic (--quick = 4/8/8/10 seeds)
  cargo run --release --offline --bin g3_perf  -- --out results/g3_perf.json      # ~5.5 min (waits <=60 s per world for the box to be quiet; --quiet-wait 0 disables; --quick ~6 s)

### VERIFY REQUEST — W3 — equivalence (TOST) of CompactRouter u16/q16 vs the f64 recommended router
Pre-declared protocol (header of g3_equiv.rs, nothing tuned: `CompactConfig::default()` == defaults::recommended_guard + recommended_inner; seeds 1-8 are only a smoke run): metric = full-run mean EXPECTED success; paired by seed; margin +-0.005 (primary) and +-0.003 (strict, on the 160-seed split); TOST alpha .05 (= 90% CI inside the margin), 95% CI also given; Holm over the 5 worlds per (subject, split, margin); "CERTIFIED" iff Holm p < .05 AND 90% CI inside margin. Splits: held 31-60, confirm 61-90, POOLED 31-90 (60 seeds, primary: this is the extra power the 32-agent cases needed), power 91-250 (160 fresh seeds). Worlds (all mine, synthetic): sim Fixed / Random / NoTraps (6 agents, 2000 tasks), big-32agents (32 agents, 232 features, 4000 tasks), big-32agents-longtail (1838 features).

PRIMARY (pooled seeds 31-90, n=60, margin +-0.005, diff = compact - f64):
  sim-fixed              -0.0001  90%CI [-0.0003,+0.0001]  95%CI [-0.0003,+0.0001]  Holm p=.0000  CERTIFIED
  sim-random             +0.0004  90%CI [-0.0007,+0.0015]  95%CI [-0.0010,+0.0017]  Holm p=.0000  CERTIFIED
  sim-no-traps           +0.0001  90%CI [-0.0001,+0.0004]  95%CI [-0.0001,+0.0004]  Holm p=.0000  CERTIFIED
  big-32agents           +0.0005  90%CI [-0.0023,+0.0033]  95%CI [-0.0028,+0.0038]  Holm p=.0046  CERTIFIED   <- the case round 3 (f3) left inconclusive
  big-32agents-longtail  +0.0003  90%CI [-0.0013,+0.0019]  95%CI [-0.0016,+0.0022]  Holm p=.0000  CERTIFIED
Frozen confirm 61-90 (n=30): all five CERTIFIED at +-0.005 (big-32agents +0.0011 90%CI [-0.0023,+0.0044], Holm p=.028; long-tail +0.0003 [-0.0022,+0.0027]). Held 31-60 (n=30): all five CERTIFIED at +-0.005, big-32agents only just (-0.0000 90%CI [-0.0047,+0.0046], Holm p=.039).
Power 91-250 (n=160): all five CERTIFIED at +-0.005 AND at +-0.003: big-32agents +0.0002 90%CI [-0.0021,+0.0025] (Holm p=.045 at +-0.003), long-tail -0.0006 [-0.0027,+0.0015] (Holm p=.045 at +-0.003), the three sims within +-0.0009.
Secondary subjects (same tests): compact-u16-hash32 (no vocabulary at all) is numerically identical to the table mode in these worlds (no hash collisions among 232..1838 features); compact-f32 and compact-f64 have diff exactly 0.0000 with zero variance in every cell (max |mean diff| over all cells = 0e0).
Absolute success (pooled): sim-fixed .7671 vs .7670, sim-random .7736 vs .7740, no-traps .7918 vs .7919, big-32 .7041 vs .7046, long-tail .6972 vs .6975 (f64 vs compact).
NEGATIVES / caveats (equivalence):
 1. With only the smoke 8 seeds (1-8) the verdicts are inconclusive in 4/5 worlds (90% CI up to +-0.015); the certification needs the 60+ seeds. At 30 seeds the 32-agent worlds only just certify at +-0.005 (Holm .028-.039) and could NOT certify +-0.003 (pooled 95% CI of big-32agents is [-0.0028,+0.0038], i.e. it reaches beyond 0.003). +-0.003 is certified only with the 160-seed split and then with Holm p=.045 (borderline).
 2. Equivalence is on EXPECTED success in my own synthetic worlds; not on realised outcomes and not on real traffic. No claim beyond "within +-0.005 (CI) of the f64 router in these 5 worlds"; the mean differences are +-0.0006 or smaller.
 3. u16/q16 does not reproduce f64 decision-for-decision (it diverges after a few hundred tasks; unit test only requires the trajectory quality to match); only f64/f32 instances are bit-/decision-identical.
 4. Hash32 ids can collide in an open token stream (p ~ n^2/2^33): equal here only because no collision occurred.

### VERIFY REQUEST — W3 — agents-per-GiB and ns/decision of CompactRouter vs the f64 recommended router
Method: counting global allocator (exact live bytes of K independently trained brains per world, each trained on its own world seed 1000+k, world data freed), library accounting cross-check, Windows commit charge column in the json (UNRELIABLE for small K: heap-granularity noise, do not quote). ns = min (and median) over 31 INTERLEAVED repetitions of route+feedback per decision, fresh router each rep ("fresh") and a second pass over a trained router ("warm"), same precomputed outcome table, same rng seed; compact ns includes string->id lookups. Declared criterion: compact must not exceed the reference by more than +10% ns (it did not).
                               reference f64        compact private (vocab+kw per brain)   compact SHARED vocab+kw          agents/GiB ref -> private | shared
  sim-fixed (6 ag, 32 feat)    16,068 B             5,669 B  (2.8x)                         2,389 B  (6.7x)                  66.8 k -> 189 k | 449 k
  sim-random                   16,068 B             5,618 B  (2.9x)                         2,339 B  (6.9x)                  66.8 k -> 191 k | 459 k
  sim-no-traps                 16,068 B             5,626 B  (2.9x)                         2,346 B  (6.8x)                  66.8 k -> 191 k | 458 k
  big-32agents (232 feat)      246,128 B            44,702 B (5.5x)                         20,854 B (11.8x)                 4.4 k -> 24.0 k | 51.5 k
  big-32agents-longtail (1838) 1,877,858 B          250,659 B (7.5x)                        75,930 B (24.7x)                 572 -> 4.3 k | 14.1 k
  (library accounting `bytes()` agrees with the allocator within 2-7% for the shared variant, 5% for private; 1 trained brain: owned 2,266 B + vocab ~2,862 B (estimate) + keywords 256 B at sim-fixed; owned 17,260 B at big-32agents; 66,850 B at long-tail.)
  ns/decision min fresh (warm):  ref -> compact (ratio fresh | warm)
  sim-fixed      1180 (1167) -> 601 (571)    x0.51 | x0.49     allocations/decision 12.8 -> 0.00
  sim-random     1162 (1116) -> 552 (532)    x0.48 | x0.48     12.8 -> 0.00
  sim-no-traps   1489 (1535) -> 635 (685)    x0.43 | x0.45     11.9 -> 0.00
  big-32agents   3477 (3658) -> 1265 (1457)  x0.36 | x0.40     14.0 -> 0.00
  big-32agents-longtail 4721 (4484) -> 1302 (1434)  x0.28 | x0.32   15.5 -> 0.00
  select-only (warm): 607 -> 246, 568 -> 230, 663 -> 293, 2544 -> 645, 3249 -> 671 ns.
NEGATIVES / caveats (bytes, latency):
 1. NO latency or accuracy regression, but the box was shared: background CPU during the full run was 50-81% (bg% per world in the json/log); absolute ns are inflated and +-30% noisy, only the interleaved ratios (0.28-0.51) are meaningful. A quick run at 41-48% background gave 0.33-0.51, same picture.
 2. The shared-vocab figure only holds when many brains share ONE vocabulary and keyword table (`SharedVocab`, `fork_blank`); a self-contained brain (private vocab + keyword map, also what save/load gives you) is 2.8x (6 agents) .. 7.5x (long tail). The vocabulary is unbounded for an open token stream (long-tail: ~173 KB estimated for 1838 features, no eviction; the earlier row-cap/admission experiments in f3 showed such caps cost accuracy, not implemented here).
 3. Sparse rows lose against dense rows at 6 agents (f3 measured 1.8 KB dense vs 2.3 KB sparse brain-only); this library type uses sparse only (24 B fixed per row incl. index + 6 B per touched cell), so the 6-agent shared figure is 449 k/GiB, below f3's bin-local dense 526 k/GiB. The win grows with agents (5.5x-11.8x at 32 agents).
 4. Persistence: JSON state is text (4 KB / 35 KB / 124 KB for sim / big / long-tail, larger than the in-memory brain); no binary format was added. It is still a self-contained snapshot (feature strings stored once).
 5. Limits (enforced): <= 64 agents; plain Hebbian rule only (meta/weighting/ucb/agg rejected by `CompactConfig::from_configs`); agent count fixed (updates for agent >= n and non-finite rewards are ignored, candidates must be < n); Q16 counts saturate at 63.99 (the recommended discount .98 keeps counts < 50); no team routing/mesh/knowledge layer. Not wired into the CLI/serve, so the serve.rs hot spot found in f3 is unchanged.
 6. `Vocab::bytes_estimate` is an estimate of the hashbrown layout; use the counting allocator for exact numbers. Commit-charge column is noisy and should be ignored.
 7. Worlds are mine; agents-per-GiB counts routing brains in one process (memory only), not CPU or concurrency (a CompactRouter is not internally synchronised).

lib tests: `cargo test --release --offline --lib` -> 228 passed, 0 failed (11 of them are compact::tests; 3 consecutive reruns of the test exe all green, ~1.1 s). One earlier foreground `cargo test --lib` run of mine did not return for 30 min (test exe spinning at 1,772 CPU-s on a box shared with other workers who were editing src/serve.rs and src/spatial.rs at the time); after killing it, a fresh build ran green in 1 s and 3 reruns were green. I could not attribute the stall (not a compact test: those finish in 0.02 s); worth one rerun by the director.
