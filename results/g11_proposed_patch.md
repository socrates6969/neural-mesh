# G11 proposed lib patch (W12) - NOT applied; src/*.rs untouched

Scope: add ONE small opt-in module, `src/team.rs`, exporting an **additive** learned team model with greedy selection
(`TeamModel`). Do NOT merge the pairwise term, the pair gate or the controller gate (see `g11_board_entry.md`: they
add nothing measurable outside worlds that are pairwise by construction, and the controller gate costs 1-3 points where
the learner is useful). Nothing existing changes; no default changes; `defaults::recommended_router` stays as is.

## Why this and only this
* The reference implementation is `src/bin/g11_common/learn.rs` (`Model` with `pair = false`, `Learner` with `Mode::Add`,
  `greedy_team(.., None)`). Pooled-tuned config on seeds 1-8 over all 16 G11 worlds:
  `lr 0.1, l2 5e-4, eps 0.02, c1 0.1, rho 0.3, boost 4.0` (the boost = count-based step size is what removed the separable-control
  regression; it was found on dev seeds 101-130, not on held/confirm).
* Cost measured in `g11_team` (single thread, policy only): 0.4-1.2 us/decision (policy only, measured), 5.7-15.8 KB live heap after 2500-3500 decisions
  (N 9..24, K 3..5; includes a cloned keyword map and the feature interner). `NeuralMesh(guarded-safe, team_size=K)`: 2.9-8 us/decision, 11-24 KB; sequential routers 5-70 us/decision (g11_team held logs).

## Proposed API (src/team.rs)
```rust
use crate::router::{Rng, Task};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TeamModelConfig {
    pub lr: f64,    // NLMS step, default 0.1
    pub l2: f64,    // weight decay, default 5e-4
    pub eps: f64,   // prob. of swapping one member for a random candidate, default 0.02
    pub c1: f64,    // optimism c1/sqrt(1+visits(agent)), default 0.1
    pub rho: f64,   // keyword prior mass on H[feature][lead agent], default 0.3
    pub boost: f64, // step *= 1 + boost/(1 + n_cell/30), default 4.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeamModel {
    cfg: TeamModelConfig,
    n_agents: usize,
    team_size: usize,
    intern: HashMap<String, usize>, // feature -> row
    b0: f64,
    g: Vec<f64>,       // n_agents
    hf: Vec<f64>,      // rows * n_agents   (feature x agent additive credit)
    hcnt: Vec<f64>,    // rows * n_agents
    nvis: Vec<f64>,    // n_agents
}

impl TeamModel {
    pub fn new(n_agents: usize, team_size: usize, keyword_map: &HashMap<String, usize>, cfg: TeamModelConfig) -> Self;
    /// distinct `team_size` members of `candidates`: greedy top-K of h_a = g_a + mean_f H[f][a] + c1/sqrt(1+nvis_a), then eps swap
    pub fn select_team(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> Vec<usize>;
    /// one normalised-LMS step on the TEAM reward (clamped to [0,1]): pred = b0 + sum_{a in team}(g_a + mean_f H[f][a])
    pub fn feedback(&mut self, task: &Task, team: &[usize], reward: f64);
    pub fn state_bytes(&self) -> usize;
}
```
Update rule (identical to `Model::update` with `pair = false`): `err = r - pred; step = lr*err/(1+2K);`
for every member `a`: `g_a = g_a*(1-l2) + step*(1 + boost/(1+nvis_a/30))`, and for every task feature `f`:
`H[f][a] = H[f][a]*(1-l2) + step*(1 + boost/(1+hcnt[f][a]/30))`, `hcnt[f][a] += 1`; `nvis_a += 1`.
Because the model is additive, "greedy" is EXACT (top-K by h), so no polish loop is needed.

## Tests to ship with it (all offline, deterministic)
1. `fits_repeated_sample`: 200 identical `(task, team, r)` feedbacks drive `pred` within 0.02 of `r`.
2. `team_is_distinct_and_inside_candidates` for random candidate subsets, K = 1..|candidates|.
3. `keyword_prior_is_followed_first`: with `rho > 0` and no feedback the lead of a keyword feature is selected.
4. `separable_world_beats_random`: 8-agent additive world, 600 steps, mean value > random-team mean + 0.1.
5. `serde_roundtrip_continues_identically`.
6. `no_nan_on_extreme_rewards` (rewards outside [0,1] are clamped).

## Docs wording that the evidence supports (and no more)
"Opt-in learned team model. Online regression of team reward on (agent, task-feature) indicators, greedy selection.
Measured on simulated worlds only: beats the repo's sequential routers and `NeuralMesh(team_size=K)` where the keyword
table is stale or wrong, is 0.5-1.5 points below a PERFECT hand-authored keyword table, ~1 us/decision. No pairwise term; the gain over the mesh is +0.05..+0.30 (v units), over the best per-world-tuned baseline it is -0.015..+0.16 depending on keyword-table quality.
No evidence on real traces."

## Explicitly NOT proposed
* pairwise `J` terms (vs L-add, held | confirm: LAT +0.001 | +0.008 (Holm better only on confirm), COMP +0.020 | +0.012 n.s., COMP-X +0.015 | +0.016 n.s.; RC family -0.007..-0.016 Holm WORSE; ROLE -0.011 worse on held);
* the prequential pair gate (`L-gate` is inert: |L-gate - L-add| <= 0.0012 in every world on both sets);
* the randomized controller gate (`L-full`; needs ~250 randomized decisions per arm per 0.02 resolvable gap, costs 1-3 points where the learner is useful);
* any change to `NeuralMesh`, `defaults`, or the router grid.
