# K5 proposed patch (NOT applied; no src edit, defaults unchanged, opt-in only)

Status: both items below are hypotheses from a synthetic-until-validated simulator. Item 1 is supported by a frozen confirm (A 61-90, B 61-90, `results/k5_board_entry.md` K5-B/D); item 2 comes from POST-HOC ablations (K5-E) and needs a frozen run on fresh seeds before anything is adopted.

## 1. `PipelineMeta`: Thompson selection between complete routing pipelines (opt-in, e.g. src/popguard.rs sibling or new src/pipemeta.rs)
Purpose: pick the *front* (topic-keyed router with retrieval anchor vs a feature-blind global bandit) per deployment, from realised rewards, instead of fixing it. In the K5 worlds it brings the topic-independent control to TOST-equivalence with the best fair baseline (global-ts) (ctl-irrel -.004 / -.003, both text worlds) at a dependent-world price of ~.01-.02 (the price of discovering the world type online; it cannot be removed by a cleverer off-policy estimator because the keyed policy is deterministic given the keywords).
```rust
pub struct PipelineMeta { n0: f64 /*10*/, s: Vec<f64>, n: Vec<f64> /* len = #pipelines */ }
impl PipelineMeta {
    pub fn new(k: usize, n0: f64) -> Self { /* s = n = 0 */ }
    /// Thompson draw: Beta(n0/2 + s_i, n0/2 + n_i - s_i); returns argmax (ties -> lowest index)
    pub fn choose(&self, rng: &mut Rng) -> usize { ... }
    /// credit the realised reward in [0,1] of the pipeline that played (ALL tasks incl. vote-played ones);
    /// discount 1.0 (tuned: .995/.98 were worse)
    pub fn credit(&mut self, i: usize, reward: f64) { s[i] += reward; n[i] += 1.0; }
}
```
Wiring: pipelines = {warm(hebr)+retrieval-anchor feature, global-ts}; EVERY front (warm, guarded, global-ts) receives `update(task, played_agent, reward)` on every task (shared learning); only the chosen pipeline's `select` runs. Credit goes to the chosen pipeline on every task. Unit tests to add: (a) k=2 with rewards .6/.5 chooses the better arm >85% of the last 500 of 2000 draws; (b) all-equal rewards keep selection near uniform; (c) deterministic given the seed; (d) with one pipeline it is the identity (no RNG consumed).
Cost: 3x router updates per task (2-10 us/task measured here), no extra model/verifier call, no extra retrieval.

## 2. Retrieval anchor as a plain router feature (hypothesis; needs a frozen fresh-seed run)
POST-HOC (K5-E): `feat|warm` (retrieval top-1 anchor injected as feature `kn:<anchor>` into warm(hebr), never an anchor vote, no gate, no paragraph context) was the strongest arm: vs the best fair baseline T12 +.042*|+.034*, T12-drift +.041*|+.033*, T48 +.017*|+.011 (A conf|B conf), ctl-oneoff +.025*|+.030*, but still -.058|-.059 below global-ts in ctl-irrel and -.038|-.032 in ctl-noinfo (front problem, see 1). Retrieval value over the same front is harmless in the controls. The g16 shrinkage/post gates, the anchor-local Bayes test and the online A/B holdout all lose most of that gain (K5-C): do NOT port them.
Implementation sketch: in the existing `kn-direct` router path (src/hybrid.rs consumer), when the top-1 admitted paragraph (dense cosine >= .30) exists add `format!("kn:{cluster_of(doc)}")` to `task.features`; no vote, no window gate, no paragraph injection (the real top-1 precision .12-.35 is below the break-even of the assumed context model; keep context injection a separate, A/B-gated, user-visible switch).

## Not proposed
- confidence gates on the vote (post / shrinkage / local Bayes / A/B holdout) and the injection-cosine gate: no gain over the ungated feature in these worlds;
- front-level meta (Thompson on non-vote tasks): worse than pipeline-level credit (g1|meta3 vs P2w+ in controls -.012 vs -.004 ctl-irrel in A conf).
