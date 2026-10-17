# H7 proposed patch (NOT applied; I own no src files)

Nothing in src/ is needed for the measured result. If the director wants an opt-in building block later, the only reusable piece is the
candidate selector of the best-of-k arm. Sketch (pure function, no state, default behaviour of every router unchanged, to live in a new
opt-in module, e.g. `src/creative.rs`, not wired into `defaults::recommended_router`):

```rust
/// Pick one of k candidate outputs for a creative-tagged task.
/// `emb[i]`: embedding of candidate i, `qhat[i]`: judged quality in [0,1] (verifier call, charged by the caller),
/// `recent`: embeddings of the last <=10 DELIVERED outputs of this domain, `theta`: euclid dedup radius (unit-norm: cos = 1 - theta^2/2),
/// `w`: novelty weight. Returns the index to deliver.
pub fn select_diverse(emb: &[Vec<f32>], qhat: &[f32], recent: &[Vec<f32>], theta: f32, w: f32) -> usize { /* H7: engine.rs `pick`, Sel::Dedup */ }
```
Tuned values in the H7 worlds (pooled seeds 1-8): W4-type worlds theta .15 / w .75 (k=3); new generators theta 0 / w .5. They are NOT portable:
they must be tuned on a deployer's own graded outputs. Pre-conditions for any wiring: (1) human `creative` tag per task type, (2) a judge/verifier
whose calls are accounted as cost, (3) a real embedding of outputs. None of this is validated on real traffic.

No patch is proposed for `Router`, `Gate` or the nbucb/nbreg arms: the H7 result is that they do not generalise (see board entry), so they stay
lab-only (g4_common / h7_common).
