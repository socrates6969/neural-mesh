# G19-A (W20) proposed patch -- NOT applied (src/*.rs is read-only this round)

Everything below already exists, tested and measured, as bin-local code in `src/bin/g19_common/keyed.rs` (+ `Credited` / `KPol` in `src/bin/g19_common/mod.rs`). The proposal is to move the pieces into the crate as OPT-IN additions that leave every default untouched (the same pattern D7 used for `SonaPlus`). All evidence is SYNTHETIC until validated on real traffic (see results/g19a_board_entry.md).

## P1 (recommended, opt-in): evidence-gated set-level pattern memory behind the unchanged guard
Where: `src/sona.rs` (SonaPlus section, after `PatternLearner`), new learner variant `PlusLearner::Hybrid`.

* Add `KeyedMemory` (keyed.rs lines ~230-740): unigram token tables + exact key-set patterns (`HashMap<Vec<u32>, Pat>`) + k-nearest-pattern blend (Jaccard similarity^gamma x W/(W+kappa)) + optional learned key selection, neighbour credit spread and bounded capacity. Port the u32 token interning from `Anchors::intern` (features -> ids) so it works on `Task.features` without the anchor machinery; `AnchorCfg::off()` reproduces raw-feature tokens.
* Add `HybridInner { heb: HebbianRouter(robust), mem: KeyedMemory, min_w }` implementing `Router`: update() trains BOTH on every outcome; select() uses `mem` iff `mem.set_support(task) >= min_w`, else `heb`. `set_support` = weight of the exact key-set pattern + sum over the k nearest patterns of similarity x weight.
* `SonaPlusConfig` new fields, all `#[serde(default)]`, defaults reproduce today's behaviour exactly (learner stays `Hebbian`): `hyb_min_w: f64 = 8.0`, `mem_knn_k = 3`, `mem_s_min = 0.3`, `mem_gamma = 3.0`, `mem_kappa = 6.0`, `mem_lam_u = 0.3`, `mem_init = 0.6`, `mem_temp = 0.03`, `mem_c0 = 1.0`, `mem_cap = 0` (unlimited). The guard stays `GuardConfig` of `defaults::recommended_guard()`.
* Recommended `hyb_min_w`: 8. The seed-1-8 tuning chose 3 (frozen "K-full"), but min_w=8 is better in the additive worlds on every fresh block (31-60/61-90/91-120) while keeping the conjunction gains; this is a POST-HOC one-factor choice (see board entry).
* Tests to add (all already pass in the bin; names in `g19_sona_a.rs`): identity (`identity_guard_matches_guarded_safe`: guard + hebbian inner reproduces `defaults::recommended_router` bit-for-bit), capacity respected (`capacity_is_respected_and_merge_conserves_weight`), and a conjunction regression: on `conj-clean` seed 31-40 `Hybrid` must beat `Hebbian` at H=2000 by >= 0.02.
* Expected effect (synthetic): +0.02..+0.07 at H2000 over guarded-safe where the best agent depends on feature conjunctions (conj-clean/-noisy, conjT-clean, shortmod-traps/-clean; Holm-significant in 3 independent seed blocks); no Holm loss vs the best baseline in 6 control worlds; costs 0.0..0.5 pts (min_w=8) in additive many-surface-form worlds (var-clean -0.0053 on 91-120, Holm LOSS). Early horizons in trap-keyword worlds are the guard's weakness (same for guarded-safe), not the memory's.
* Do NOT ship: the ungated memory (`PlusLearner::Patterns`-style, memory decides every hand-over choice): larger gains in conjunction worlds but Holm losses in correct-keywords/sim-notraps.

## P2 (optional, opt-in, low priority): learned anchor keys
Where: `src/prior.rs` (new type next to `Guarded`, do not touch `Guarded`): `AnchorGuard<R>` = `Guarded` with outcome counts keyed by anchor tokens; `Anchors` = merge-only agglomerative clustering from the task history (token-pair count >= 3, lift >= lift_min, Poisson upper tail <= alpha; recomputed every 25 tasks; counts/memory migrate on a merge).
* Evidence: neutral in normal feedback (K-full - K[no anchors]: 1 W/0 L and 3 W/1 L), +0.017..+0.028 at H2000 under sparse (reuse-only) feedback in 3 worlds (4 W/0 L direct, both blocks); clustering purity 1.00 on the author-built generators, topics recovered by t~500 with permissive thresholds (lift 3, alpha 1e-2), but permissive thresholds cost 0.002-0.006 at H2000 in the control worlds. The tuned conservative thresholds (lift 8, alpha 1e-4) recover only 20-65% of the features.
* Ship only with `AnchorCfg::off()` as the default. The ORACLE-anchor arm shows headroom (+0.04 at H500 in var-traps, +0.19 in var-single) for SEMANTIC anchors; the existing `knowledge.rs` anchors need real embeddings and were not testable with the synthetic hash embeddings (g19_sona_a does not exercise them).
* Known limitation: Hebbian inner rows are not migrated on a merge (the pattern memory's are).

## P3 (recommended, generic, NOT SONA-specific): trajectory-level credit wrapper for session-level outcomes
Where: new `src/credit.rs` (or `src/prior.rs`): `Credited<R>` wrapping any `Router` whose caller can only report a session outcome (all steps ok / not).
```
on session end(steps = [(task_i, agent_i)], ok):
    if ok: update(task_i, agent_i, 1.0) for all i
    else:  p_i = clamp(P_hat(task_i, agent_i), .02, .98)            // plain per-feature mean estimator, same as UniEst in g19_common/mod.rs
           q_i = (1 - p_i) / (1 - prod_j p_j)                       // posterior P(step i failed | session failed)
           update(task_i, agent_i, 1 - q_i)
```
* Evidence (session worlds, series system of 3 steps, only the shared bit is reported): H2000 expected success, broadcast -> credited: guarded-safe 0.535 -> 0.712 (sess-traps), 0.500 -> 0.689 (sess-var-traps), 0.562 -> 0.740 (sess-sim-random); K-full credit - broadcast 15 W / 0 L in both seed blocks. It helps every router; the SONA pattern-memory estimator as the source of p_i is WORSE than the plain estimator (0 W / 10-12 L), so keep the plain estimator.
* Tests: credits sum to >= 1 on a failed session, success credits all steps 1, single-step session reduces to the ordinary update.

## P4 (informational): consolidation of the pattern store
`mem_cap > 0` with importance = evidence x (best-agent mean - mean over agents), victim merged into its nearest pattern: at 16 patterns (vs 1161 uncapped) it keeps ~45% of the memory's gain over guarded-safe at H2000 in memT-conj48 (0.5018 vs 0.5325 uncapped, guarded-safe 0.4765) while LRU/FIFO keep none (0.476-0.479). It never beats uncapped; ship it only if memory is a constraint.

## What NOT to change
`SonaRouter`, `SonaPlusConfig::default()`, `defaults.rs`, `GuardConfig::default()` stay as they are. No claim that SonaPlus now beats the best baseline in general: it ties guarded-safe in the controls and wins only in worlds with conjunction structure.
