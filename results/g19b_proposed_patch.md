# G19b proposed patch (W21): NONE recommended for merge

Nothing in src/ was edited. After two rounds (results/g19b_board_entry.md) no mechanism beat the best fair baseline held-out + Holm in any world without a Holm loss to some comparator, so there is no patch to propose for SonaPlus.

For the record, the one mechanism with a reproducible positive effect vs the retuned (discount-free) guarded(hebr) is the CUSUM drift reset ("detect"). To reproduce it inside the library without re-implementing Guarded it would need an additive, default-off API:

```rust
// prior.rs (sketch, opt-in; NOT proposed for merge)
impl<R: Router> Guarded<R> {
    /// scale the evidence of a feature row (drift-alarm response); keep in [0,1]
    pub fn shrink_row(&mut self, feature: &str, keep: f64) { /* counts.get_mut(feature): every (s,f) *= keep */ }
}
// sona.rs: SonaPlusConfig { det_delta, det_lambda, det_keep } (all 0 = off), per-(feature,agent) CUSUM:
//   g = max(0, g + (m - r) - det_delta), m = s/(s+f) of the chosen agent's cell BEFORE the update (needs s+f >= 4);
//   alarm when g > det_lambda -> Guarded::shrink_row(feature, det_keep) for every feature of the task, g := 0.
```

Evidence against merging: it equals the retuned guard in stationary worlds only when the alarm is rare; it loses to warm(hebr)/hebbian-robust in drift-abrupt (-0.013 | -0.017 at H2000 vs best baseline, Holm LOSS on 61-90); the round-2 variant that closes that gap (bypass the keyword rule for 200 tasks after an alarm) costs -0.011 in correct-keywords, -0.004..-0.007 in traps(main) and many-topics (Holm LOSS on several seed sets). Reference implementation: `SonaB` in src/bin/g20_common/mod.rs; its all-off mode is asserted bit-identical to prior::Guarded<HebbianRouter> by `g20_sona_b --selftest`.
