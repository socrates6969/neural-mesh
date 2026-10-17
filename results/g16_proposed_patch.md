# g16 proposed patch (W17) — NOT applied (src/*.rs untouched). Proposal only; apply only if the director's rerun confirms results/g16_board_entry.md.

Scope: `src/knowledge.rs` (`Knowledge::vote`, `Knowledge::credit`, `KnowledgeRouter::select`). Everything below is what the simulation arm `FINAL[mini-hybrid]` (= `G-shrink+g1(inj-gated)`) does; the crate arm it modifies is the existing `kn-direct+vote`.
Evidence: results/g16_route.json, results/g16_route_confirm.json. Honest status: improves on the current vote policy in 6/8 worlds, does NOT make the control worlds harmless (see board entry); the safer `G-post+g1` variant is also described.

## 1. Hierarchical shrinkage of the anchor evidence toward the agent's GLOBAL success rate (the part that matters)
Current `vote`: `m = (prior.0 + s) / (prior.0 + prior.1 + n)` with a flat Beta(1,1) prior and `m >= tau (.7)`, `n >= min_n`. With 6 agents and 2-3 tries, a lucky streak of a globally mediocre agent qualifies (winner's curse): in the control worlds the vote-played tasks had mean expected success .46 vs .75 for front-played tasks (g16 `--diag`).
Change: keep per-agent GLOBAL counts and use them as the prior centre.
```rust
// in Knowledge: pub gagent: Vec<(f64 /*sum reward*/, f64 /*n*/)>   (updated in credit(): gagent[agent].0 += r; .1 += 1.0)
// in vote():
let g = (1.0 + gs) / (2.0 + gn);                 // agent a's success rate over ALL anchors
let (al, be) = (kappa * g + s, kappa * (1.0 - g) + (n - s));   // kappa = 3 in the tuned pick
let m = al / (al + be);                          // shrunk posterior mean; candidate = argmax m over agents with n >= min_n (min_n = 1)
```
Candidate also needs the floor `m >= 0.6` (tuned) and the confidence test below.

## 2. Confidence test against the front's own recent success (replaces the flat `tau`)
Keep `front_ref` = EMA (alpha = max(1/(n+1), .03)) of the reward of tasks decided by the FRONT (not by the vote). Take the vote only if
`P(Beta(al, be) > front_ref + delta) >= gamma` with delta = 0.0, gamma = 0.5 (tuned: `inc_beta` from `stats`).

## 3. Keep the existing window gate (`vote_gate_open`) AND-ed in
`G-post+g1` / `G-shrink+g1` are the arms with `and_g1 = true`; the window gate alone (current crate) is what protects the controls; the confidence gates ALONE (no window gate) were worse in every control world (`G-post`, `G-shrink`: results/g16_route.log). Do not remove the window gate.

## 4. Inject `kn:<anchor>` only when the vote is open (FeatMode::Direct)
Current: `routed = Some(self.kn.enrich(task,&r))` whenever any neighbour passes `sim_min`. Proposed: build `routed` only when `vote(...)` returned a candidate that passed 1.+2.+3. (the front then sees the anchor feature on exactly the tasks where the vote is trusted). In the simulation this alone moved ctl-irrel from -0.0200 (inject always) to -0.0127 vs the same front without retrieval (confirm) at a cost of ~0.007 H2000 in the recurring worlds.

## 5. Not recommended (measured negative or null; see board entry)
- Recency decay of anchor evidence (`decay` .997/.99): -0.03..-0.07 in the stationary recurring worlds, and NO gain in the drift world.
- `reset after 3 consecutive failures`: neutral in stationary worlds, +0.006 H2000 / +0.010 late in the drift world on FINAL (small, see table).
- Adaptive k (score-drop cut-off, kmax 8): +0.001..+0.003 H2000 on R0 downstream (n.s. after Holm); doc-level it helps hybrid rankings (+0.008..+0.021 F1) but is beaten by fixed k=2 on dense-only rankings.
- Exact-duplicate suppression (cosine >= .999) is worth +0.02 F1@3 in a store with 30% re-ingested duplicates; embedding-cosine dedup does NOT detect "superseded revision" pairs (paraphrase pair cosine < .85).
- Partitioned indices: a tag-partition is a large win when the project tag is known (hit@1 +.44) and degrades with wrong tags; a soft boost (+0.1 cosine for same-project docs) is the robust choice under 20% tag noise. Topic-domain centroid routing is a loss (route-1: hit@1 -0.084).

## Tests to add if applied
- unit: shrinkage reduces a 2/2-lucky agent whose global rate is .3 below the floor; plain Beta(1,1) accepts it.
- unit: with every agent identical (global == anchor rates) the vote never opens after warm-up.
- regression: `g16_route --check` replicates the g1_route mini-hybrid arm bit-exactly (max|d| 5.6e-17) — keep the `Gate::G1` default path unchanged.
