# H9 proposed patch (NOT applied; no src edits were made)

Goal: make the one surviving H9 mechanism, `pop-guard+conf` (pool-seeded, trust-weighted `Guarded`), available as an OPT-IN type. Defaults unchanged. All numbers behind this are SYNTHETIC-until-validated-on-real-traffic (see results/h9_board_entry.md).

Reference implementation (tested, deterministic): `src/bin/h9_common/mod.rs` — `Dep`, `log_dep`, `PopPrior`, `FitCfg`, `fit_prior`, `TrustCfg`, `PopGuarded`.

## New module `src/transfer.rs` (opt-in; add `pub mod transfer;` to lib.rs)

```rust
/// What survives of one previous deployment: per-feature per-agent (success mass, trials) of its LOGGED outcomes.
pub struct PoolLog { pub cells: HashMap<String, Vec<(f64, f64)>> }
impl PoolLog { pub fn record(&mut self, task: &Task, agent: usize, reward: f64, n_agents: usize); }

pub struct FitCfg { pub m_max: f64 /*16*/, pub m1: f64 /*1*/, pub n_min: f64 /*5*/, pub shrink: bool /*true*/, pub cov_min: f64 /*0.5*/ }
pub struct Cell { pub s: f64, pub n: f64, pub mu: f64, pub m: f64 }   // m = prior pseudo-trials (0 = no information)
pub struct PopPrior { pub cells: HashMap<String, Vec<Cell>> }
pub fn fit_prior(pool: &[PoolLog], n_agents: usize, cfg: FitCfg) -> PopPrior;   // see below
pub struct TrustCfg { pub w0: f64 /*0.8*/, pub cf: f64 /*1*/, pub cg: f64 /*0.5*/, pub delta: f64 /*0.97*/ }

/// prior::Guarded<HebbianRouter> (decision rule, discount, inner learner EXACTLY as defaults::recommended_router)
/// with per-(feature,agent) counts initialised from the pool prior and an "informed fallback".
pub struct PopGuarded { .. }
impl PopGuarded { pub fn new(map: HashMap<String,usize>, n_agents: usize, prior: PopPrior, trust: Option<TrustCfg>) -> Self;
                  pub fn with_informed(self, inf_min: f64 /*8*/) -> Self; }
impl Router for PopGuarded { .. }
```

### fit_prior (what the pool contributes)
For each (feature f, agent a): over deployments with >= 3 trials of that cell: `mu = (sum s + .5)/(sum n + 1)`, coverage `cov = K'/K`;
skip if `sum n < n_min` or `cov < cov_min`; strength `m = min(m_max, mu(1-mu)/tau^2 - 1, sum n) * cov` where `tau^2` is the
between-deployment dispersion estimated by weighted method of moments (`tau^2 = max(0, (sum n_k (r_k-mu)^2 - (K'-1) mu(1-mu)) / (N - sum n_k^2/N))`;
`m = m_max` if `tau^2 ~ 0`; `m = min(m1, N)` if K' = 1). (Ablation: shrinkage vs plain `min(N, m_max)` is within +-0.001, see board entry; keep the simple plain rule if you prefer.)

### PopGuarded semantics
* per-feature row stores (data counts, prior counts); pooled counts for the Guard bounds = mean over the task's features of `data + w_f * prior`, `w_f` = trust weight.
* trust `w_f = sigmoid(logit(w0) + cf*(Lprior_f - Ldata_f) + cg*(Lprior_g - Ldata_g))` where L are `delta`-discounted prequential log scores of the observed reward under
  (prior-only mean of the cell) vs (data-only Beta(1,1) mean of the cell). A wrong prior therefore loses weight within a handful of outcomes; a right one keeps it while the data is thin and fades as the data-only predictor catches up.
* decision: Guarded's rule unchanged, PLUS: when the keyword rule is absent or "proven poor" (UCB < tau), follow the candidate with the highest LCB among those carrying >= `inf_min` (trust-weighted) pool pseudo-trials if its LCB beats the keyword UCB (or .6 if no keyword vote); otherwise ask the (cold) inner Hebbian as before.
* with an empty prior (or `inf_min` never met) it is BIT-IDENTICAL to `defaults::recommended_router` (selftest in `h9_priors.rs`, 3 worlds x {trust on/off}).

## Suggested API on top
`defaults::recommended_warm_router(keyword_map, n_agents, pool: &[PoolLog]) -> PopGuarded` using the frozen pick
`FitCfg{m_max:16,m1:1,n_min:5,shrink:true,cov_min:.5}`, `TrustCfg{w0:.8,cf:1,cg:.5,delta:.97}`, `inf_min:8`; plus a `PoolLog` serializer so a deployment can export its log (feature, agent, reward) and a new deployment can import K >= 2 of them (K=1 gives no consistency estimate and no measurable effect, see K sensitivity).

## Tests to add (library)
1. empty prior => identical decisions to `recommended_router` on 3 worlds (copy of h9 selftest 1b).
2. `fit_prior`: consistent cells => m = cap, inconsistent cells => ~0, cell seen in 1 of 4 deployments dropped at cov_min .5, thin cells ignored.
3. trust: wrong prior (prior .85, truth .3) => w < .2 within 40 outcomes; right prior keeps w > .5 after 8.
4. defaults unchanged: `GuardConfig::default()`, `HebbianConfig::default()` asserts already in defaults.rs.

## Do NOT port
`PopTs` (Thompson + prior pseudo-counts), `PopMap` (pool-corrected keyword map) and `kw-cal`: see board entry (dominated / unsafe / no effect).
