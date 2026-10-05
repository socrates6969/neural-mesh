//! Opt-in warm-start prior for the recommended router: `PopGuarded` (R5-A / H9 `pop-guard+conf`).
//!
//! A faithful library port of `src/bin/h9_common/mod.rs` (`Dep`, `fit_prior`, `PopPrior`, `TrustCfg`,
//! `PopGuarded`); see `results/h9_proposed_patch.md` and `results/h14_board_entry.md`.
//!
//! * [`PoolLog`] = what survives of one previous deployment: per-feature per-agent (success mass, trials).
//! * [`fit_prior`] fits a population prior from K >= 2 such logs (hierarchical shrinkage: the prior strength of a
//!   (feature, agent) cell comes from the between-deployment consistency of its success rate).
//! * [`PopGuarded`] = `prior::Guarded<HebbianRouter>` (decision rule, discount and inner learner exactly as
//!   `defaults::recommended_router`) whose per-(feature, agent) counts are seeded from the prior; the prior part is
//!   scaled by a per-feature trust weight learned online (prequential prior-vs-data log score), so a WRONG prior
//!   loses weight within a handful of outcomes.
//!
//! With an EMPTY prior (or `inf_min == 0`) the decisions are bit-identical to `defaults::recommended_router`
//! (unit test). Nothing here changes any default. All measured gains are SYNTHETIC-until-validated-on-real-traffic
//! (early gains fade by ~2000 tasks; a wrong prior costs a bounded amount; see the board entry).
//!
//! Do NOT port (measured dominated / unsafe / no effect in H9): `PopTs`, `PopMap`, `kw-cal`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::defaults::{recommended_guard, recommended_inner};
use crate::hebbian::HebbianRouter;
use crate::prior::{GuardConfig, KeywordPrior};
use crate::router::{Rng, Router, Task};

// ------------------------------------------------------------------------------------------------ pool log
/// Logged outcomes of one previous deployment: feature -> per-agent (success mass, trials).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct PoolLog {
    pub cells: HashMap<String, Vec<(f64, f64)>>,
}

impl PoolLog {
    pub fn new() -> Self {
        Self::default()
    }
    /// Log one graded outcome of `agent` on `task` (reward clamped to [0,1]).
    pub fn record(&mut self, task: &Task, agent: usize, reward: f64, n_agents: usize) {
        let r = reward.clamp(0.0, 1.0);
        let n = n_agents.max(agent + 1);
        for f in &task.features {
            let row = self.cells.entry(f.clone()).or_insert_with(|| vec![(0.0, 0.0); n]);
            if row.len() < n {
                row.resize(n, (0.0, 0.0));
            }
            row[agent].0 += r;
            row[agent].1 += 1.0;
        }
    }
    /// Total logged trials (each task counts once per feature).
    pub fn total_trials(&self) -> f64 {
        self.cells.values().flat_map(|r| r.iter()).map(|c| c.1).sum()
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("PoolLog serialises")
    }
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

// ------------------------------------------------------------------------------------------------ prior
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Cell {
    pub s: f64,
    pub n: f64,
    pub mu: f64,
    /// prior strength in pseudo-trials (0 = no information)
    pub m: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct FitCfg {
    /// cap on the pseudo-count of one (feature, agent) cell
    pub m_max: f64,
    /// pseudo-count when a cell was seen in a single deployment only
    pub m1: f64,
    /// minimum pooled trials for a cell to carry any prior
    pub n_min: f64,
    /// true = hierarchical shrinkage (strength from between-deployment consistency); false = plain `min(N, m_max)`
    pub shrink: bool,
    /// a cell must be observed (>= 3 trials) in at least this fraction of the pool's deployments, and its strength
    /// is scaled by that fraction (selection-bias guard)
    pub cov_min: f64,
}

impl Default for FitCfg {
    /// The frozen H9 pick: M16, m1=1, n_min 5, shrink, cov .5.
    fn default() -> Self {
        FitCfg { m_max: 16.0, m1: 1.0, n_min: 5.0, shrink: true, cov_min: 0.5 }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct PopPrior {
    pub cells: HashMap<String, Vec<Cell>>,
    pub n_deps: usize,
}

impl PopPrior {
    /// Cell with information (m > 0) for (feature, agent).
    pub fn cell(&self, f: &str, a: usize) -> Option<&Cell> {
        self.cells.get(f).and_then(|v| v.get(a)).filter(|c| c.m > 0.0)
    }
    /// Empty prior (== no information).
    pub fn empty() -> Self {
        PopPrior { cells: HashMap::new(), n_deps: 0 }
    }
    pub fn is_empty(&self) -> bool {
        self.cells.values().all(|r| r.iter().all(|c| c.m <= 0.0))
    }
    /// Number of (feature, agent) cells carrying information.
    pub fn n_informative(&self) -> usize {
        self.cells.values().flat_map(|r| r.iter()).filter(|c| c.m > 0.0).count()
    }
}

/// Fit a population prior from the logs of previous deployments (see module docs / h9_proposed_patch.md).
pub fn fit_prior(pool: &[PoolLog], n_agents: usize, cfg: FitCfg) -> PopPrior {
    let mut feats: Vec<&String> = pool.iter().flat_map(|d| d.cells.keys()).collect();
    feats.sort();
    feats.dedup();
    let mut cells = HashMap::new();
    for f in feats {
        let mut row = vec![Cell::default(); n_agents];
        for a in 0..n_agents {
            let obs: Vec<(f64, f64)> = pool
                .iter()
                .filter_map(|d| d.cells.get(f).and_then(|v| v.get(a)).copied())
                .filter(|&(_, n)| n >= 3.0)
                .collect();
            let ntot: f64 = obs.iter().map(|x| x.1).sum();
            let cov = obs.len() as f64 / pool.len().max(1) as f64;
            if ntot < cfg.n_min || obs.is_empty() || cov < cfg.cov_min {
                continue;
            }
            let stot: f64 = obs.iter().map(|x| x.0).sum();
            let mu = (stot + 0.5) / (ntot + 1.0);
            let k = obs.len() as f64;
            let m = if !cfg.shrink {
                ntot.min(cfg.m_max)
            } else if obs.len() < 2 {
                cfg.m1.min(ntot)
            } else {
                let ss: f64 = obs.iter().map(|&(s, n)| n * (s / n - mu) * (s / n - mu)).sum();
                let n2: f64 = obs.iter().map(|x| x.1 * x.1).sum();
                let denom = ntot - n2 / ntot;
                let tau2 = ((ss - (k - 1.0) * mu * (1.0 - mu)) / denom.max(1e-9)).max(0.0);
                let m0 = if tau2 < 1e-9 { cfg.m_max } else { (mu * (1.0 - mu) / tau2 - 1.0).clamp(0.0, cfg.m_max) };
                m0.min(ntot)
            };
            let m = m * cov;
            row[a] = Cell { s: stot, n: ntot, mu, m };
        }
        cells.insert(f.clone(), row);
    }
    PopPrior { cells, n_deps: pool.len() }
}

// ------------------------------------------------------------------------------------------------ trust
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct TrustCfg {
    /// initial trust in the prior
    pub w0: f64,
    /// weight of the per-feature evidence
    pub cf: f64,
    /// weight of the global (all-features) evidence
    pub cg: f64,
    /// discount of the prequential log scores
    pub delta: f64,
}

impl Default for TrustCfg {
    /// The frozen H9 pick: w0 .8, cf 1, cg .5, delta .97.
    fn default() -> Self {
        TrustCfg { w0: 0.8, cf: 1.0, cg: 0.5, delta: 0.97 }
    }
}

fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-4, 1.0 - 1e-4);
    (p / (1.0 - p)).ln()
}
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
}
fn bounds(a: f64, b: f64, z: f64) -> (f64, f64) {
    let n = a + b;
    let m = a / n;
    let sd = (m * (1.0 - m) / (n + 1.0)).sqrt();
    (m - z * sd, m + z * sd)
}

// ------------------------------------------------------------------------------------------------ PopGuarded
/// `prior::Guarded<HebbianRouter>` (the recommended default) with its per-(feature, agent) counts initialised from a
/// pool prior (pseudo-counts m*mu, m*(1-mu); prior part scaled by an online trust weight when `trust` is set).
/// Decision rule, discounting and inner learner are exactly Guarded's; with an empty prior it is bit-identical to
/// `defaults::recommended_router`.
pub struct PopGuarded {
    pub inner: HebbianRouter,
    prior_kw: KeywordPrior,
    pop: PopPrior,
    cfg: GuardConfig,
    n: usize,
    /// feature -> (data counts, prior counts) per agent
    rows: HashMap<String, (Vec<(f64, f64)>, Vec<(f64, f64)>)>,
    trust: Option<TrustCfg>,
    lp: HashMap<String, (f64, f64)>,
    g: (f64, f64),
    /// when the keyword rule is abandoned / absent, follow the best candidate whose (trust-weighted) evidence holds at
    /// least `inf_min` pool pseudo-trials (and whose LCB beats the keyword UCB) instead of the cold inner learner.
    /// 0 = off (decisions are then exactly Guarded's).
    inf_min: f64,
    /// decision counters: [follow keyword, informed/challenge, inner learner]
    pub decisions: [u64; 3],
}

impl PopGuarded {
    pub fn new(map: HashMap<String, usize>, n: usize, pop: PopPrior, trust: Option<TrustCfg>) -> Self {
        PopGuarded {
            inner: HebbianRouter::new(n, recommended_inner()),
            prior_kw: KeywordPrior::new(map, n),
            pop,
            cfg: recommended_guard(),
            n,
            rows: HashMap::new(),
            trust,
            lp: HashMap::new(),
            g: (0.0, 0.0),
            inf_min: 0.0,
            decisions: [0; 3],
        }
    }
    pub fn with_informed(mut self, inf_min: f64) -> Self {
        self.inf_min = inf_min;
        self
    }

    /// Best candidate by LCB among those carrying >= inf_min (trust-weighted) pool pseudo-trials, if its LCB > floor.
    fn informed(&self, task: &Task, candidates: &[usize], floor: f64) -> Option<usize> {
        if self.inf_min <= 0.0 || task.features.is_empty() {
            return None;
        }
        let nf = task.features.len() as f64;
        let mut best: Option<(usize, f64)> = None;
        for &a in candidates {
            let mut mass = 0.0;
            for k in &task.features {
                let w = self.weight(k);
                let p = match self.rows.get(k) {
                    Some((_, pr)) => pr.get(a).copied().unwrap_or((0.0, 0.0)),
                    None => self.seed(k, a),
                };
                mass += w * (p.0 + p.1);
            }
            if mass / nf < self.inf_min {
                continue;
            }
            let (s, f) = self.pooled(task, a);
            let (lcb, _) = bounds(1.0 + s, 1.0 + f, self.cfg.z);
            if best.is_none_or(|(_, v)| lcb > v) {
                best = Some((a, lcb));
            }
        }
        best.filter(|&(_, l)| l > floor).map(|(a, _)| a)
    }
    fn seed(&self, f: &str, a: usize) -> (f64, f64) {
        match self.pop.cell(f, a) {
            Some(c) => (c.m * c.mu, c.m * (1.0 - c.mu)),
            None => (0.0, 0.0),
        }
    }
    /// Trust weight of the prior for feature `f` (1.0 if trust learning is off).
    pub fn weight(&self, f: &str) -> f64 {
        match &self.trust {
            None => 1.0,
            Some(t) => {
                let (lp, ld) = self.lp.get(f).copied().unwrap_or((0.0, 0.0));
                sigmoid(logit(t.w0) + t.cf * (lp - ld) + t.cg * (self.g.0 - self.g.1))
            }
        }
    }
    /// Pooled (successes, failures) of `agent` over the task's features (mean), prior part weighted by trust.
    fn pooled(&self, task: &Task, agent: usize) -> (f64, f64) {
        if task.features.is_empty() {
            return (0.0, 0.0);
        }
        let (mut s, mut f) = (0.0, 0.0);
        for k in &task.features {
            let w = self.weight(k);
            let (d, p) = match self.rows.get(k) {
                Some((dr, pr)) => (dr.get(agent).copied().unwrap_or((0.0, 0.0)), pr.get(agent).copied().unwrap_or((0.0, 0.0))),
                None => ((0.0, 0.0), self.seed(k, agent)),
            };
            s += d.0 + w * p.0;
            f += d.1 + w * p.1;
        }
        let n = task.features.len() as f64;
        (s / n, f / n)
    }
    fn decide(&self, task: &Task, candidates: &[usize]) -> (u8, usize) {
        let votes = self.prior_kw.votes(task, candidates);
        let Some(k) = votes
            .iter()
            .fold(None, |b: Option<(usize, usize)>, &(a, v)| match b {
                Some((_, bv)) if bv >= v => b,
                _ => Some((a, v)),
            })
            .map(|(a, _)| a)
        else {
            return match self.informed(task, candidates, 0.6) {
                Some(a) => (1, a),
                None => (2, 0),
            };
        };
        let (s, f) = self.pooled(task, k);
        let (_, ucb_k) = bounds(self.cfg.kw_a + s, self.cfg.kw_b + f, self.cfg.z);
        if ucb_k < self.cfg.tau {
            return match self.informed(task, candidates, ucb_k) {
                Some(a) => (1, a),
                None => (2, 0),
            };
        }
        if self.cfg.challenge {
            let mut best = (ucb_k, usize::MAX);
            for &a in candidates {
                if a == k {
                    continue;
                }
                let (s, f) = self.pooled(task, a);
                let (lcb, _) = bounds(1.0 + s, 1.0 + f, self.cfg.z);
                if lcb > best.0 {
                    best = (lcb, a);
                }
            }
            if best.1 != usize::MAX {
                return (1, best.1);
            }
        }
        (0, k)
    }
}

impl Router for PopGuarded {
    fn name(&self) -> String {
        "pop-guard".into()
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        let (d, a) = self.decide(task, candidates);
        self.decisions[d as usize] += 1;
        if d == 2 { self.inner.select(task, candidates, rng) } else { a }
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let r = reward.clamp(0.0, 1.0);
        let n = self.n.max(agent + 1);
        if let Some(t) = self.trust {
            let (mut gl, mut gd, mut cnt) = (0.0, 0.0, 0.0);
            for f in &task.features {
                if let Some(cell) = self.pop.cell(f, agent).copied() {
                    let (ds, df) = self.rows.get(f).and_then(|(dr, _)| dr.get(agent)).copied().unwrap_or((0.0, 0.0));
                    let pd = ((1.0 + ds) / (2.0 + ds + df)).clamp(0.02, 0.98);
                    let pp = cell.mu.clamp(0.02, 0.98);
                    let sp = r * pp.ln() + (1.0 - r) * (1.0 - pp).ln();
                    let sd = r * pd.ln() + (1.0 - r) * (1.0 - pd).ln();
                    let e = self.lp.entry(f.clone()).or_insert((0.0, 0.0));
                    e.0 = e.0 * t.delta + sp;
                    e.1 = e.1 * t.delta + sd;
                    gl += sp;
                    gd += sd;
                    cnt += 1.0;
                }
            }
            if cnt > 0.0 {
                self.g.0 = self.g.0 * t.delta + gl / cnt;
                self.g.1 = self.g.1 * t.delta + gd / cnt;
            }
        }
        for k in &task.features {
            if !self.rows.contains_key(k) {
                let pri: Vec<(f64, f64)> = (0..n).map(|a| self.seed(k, a)).collect();
                self.rows.insert(k.clone(), (vec![(0.0, 0.0); n], pri));
            }
            let (dr, pr) = self.rows.get_mut(k).unwrap();
            if dr.len() < n {
                dr.resize(n, (0.0, 0.0));
                pr.resize(n, (0.0, 0.0));
            }
            if self.cfg.discount < 1.0 {
                for c in dr.iter_mut().chain(pr.iter_mut()) {
                    c.0 *= self.cfg.discount;
                    c.1 *= self.cfg.discount;
                }
            }
            dr[agent].0 += r;
            dr[agent].1 += 1.0 - r;
        }
        self.inner.update(task, agent, reward);
    }
}

// ------------------------------------------------------------------------------------------------ frozen pick
/// Frozen H9 configuration (`pop-guard+conf[M16,m1=1,cov0.5,inf=8][w0=0.8,cg=0.5]`).
pub const RECOMMENDED_INF_MIN: f64 = 8.0;

/// Opt-in warm router from a pool of previous deployments' logs (K >= 2 recommended: K = 1 gives no consistency
/// estimate and no measurable effect in H9). Uses the frozen H9 pick. Empty/uninformative pool => behaves as
/// `defaults::recommended_router`.
pub fn recommended_warm_router(keyword_map: HashMap<String, usize>, n_agents: usize, pool: &[PoolLog]) -> PopGuarded {
    let prior = fit_prior(pool, n_agents, FitCfg::default());
    PopGuarded::new(keyword_map, n_agents, prior, Some(TrustCfg::default())).with_informed(RECOMMENDED_INF_MIN)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults::recommended_router;

    fn task(id: u64, f: &[&str]) -> Task {
        Task { id, task_type: None, features: f.iter().map(|s| s.to_string()).collect() }
    }

    /// Tiny deterministic world: 6 agents, 4 skills, noisy keyword map, drift at the midpoint.
    fn world(seed: u64, n: usize) -> (Vec<Task>, Vec<Vec<f64>>, HashMap<String, usize>) {
        let mut r = Rng::new(seed);
        let names = ["sql", "rust", "copy", "seo"];
        let mut p = vec![vec![0.0; 6]; 4];
        for s in 0..4 {
            for a in 0..6 {
                p[s][a] = 0.25 + 0.3 * r.f64();
            }
            p[s][(s + 1) % 6] = 0.9;
        }
        let mut kw = HashMap::new();
        for (s, nm) in names.iter().enumerate() {
            kw.insert(nm.to_string(), if s == 2 { 5 } else { (s + 1) % 6 });
        }
        let mut tasks = vec![];
        for id in 0..n {
            let s = r.below(4);
            let mut f = vec![names[s].to_string()];
            if r.bernoulli(0.5) {
                f.push("urgent".into());
            }
            tasks.push(Task { id: id as u64, task_type: None, features: f });
        }
        (tasks, p, kw)
    }
    fn skill(t: &Task) -> usize {
        match t.features[0].as_str() {
            "sql" => 0,
            "rust" => 1,
            "copy" => 2,
            _ => 3,
        }
    }
    fn run(r: &mut dyn Router, tasks: &[Task], p: &[Vec<f64>], seed: u64) -> Vec<usize> {
        let cands: Vec<usize> = (0..6).collect();
        let mut rng = Rng::new(seed);
        let mut env = Rng::new(seed ^ 0xABCD);
        let mut out = vec![];
        for t in tasks {
            let a = r.select(t, &cands, &mut rng);
            out.push(a);
            let rew = if env.f64() < p[skill(t)][a] { 1.0 } else { 0.0 };
            r.update(t, a, rew);
        }
        out
    }

    #[test]
    fn empty_prior_is_bit_identical_to_recommended_router() {
        for seed in [1u64, 2, 3] {
            let (tasks, p, kw) = world(seed, 600);
            let mut g = recommended_router(kw.clone(), 6);
            let base = run(&mut g, &tasks, &p, seed);
            for tr in [None, Some(TrustCfg { w0: 0.5, cf: 1.0, cg: 0.5, delta: 0.97 })] {
                for inf in [0.0, 2.0, 8.0] {
                    let mut pg = PopGuarded::new(kw.clone(), 6, PopPrior::empty(), tr).with_informed(inf);
                    assert_eq!(base, run(&mut pg, &tasks, &p, seed), "seed {seed} trust {tr:?} inf {inf}");
                }
            }
            // an empty/too-thin pool through the convenience constructor is also identical
            let mut w = recommended_warm_router(kw.clone(), 6, &[]);
            assert_eq!(base, run(&mut w, &tasks, &p, seed));
            let mut w = recommended_warm_router(kw.clone(), 6, &[PoolLog::new(), PoolLog::new()]);
            assert_eq!(base, run(&mut w, &tasks, &p, seed));
        }
    }

    fn mk(rates: &[f64]) -> Vec<PoolLog> {
        rates
            .iter()
            .map(|&r| {
                let mut d = PoolLog::new();
                d.cells.insert("f".into(), vec![(100.0 * r, 100.0), (0.0, 0.0)]);
                d
            })
            .collect()
    }

    #[test]
    fn fit_prior_shrinkage_coverage_and_thin_cells() {
        let fit = FitCfg { m_max: 16.0, m1: 2.0, n_min: 5.0, shrink: true, cov_min: 0.0 };
        let p = fit_prior(&mk(&[0.8, 0.8, 0.8, 0.8]), 2, fit);
        assert!(p.cells["f"][0].m > 12.0 && p.cells["f"][1].m == 0.0, "consistent: {:?}", p.cells["f"][0]);
        let p = fit_prior(&mk(&[0.1, 0.9, 0.2, 0.8]), 2, fit);
        assert!(p.cells["f"][0].m < 2.0, "inconsistent: {:?}", p.cells["f"][0]);
        let p = fit_prior(&mk(&[0.8]), 2, fit);
        assert!((p.cells["f"][0].m - 2.0).abs() < 1e-9);
        let p = fit_prior(&mk(&[0.8, 0.8]), 2, FitCfg { shrink: false, ..fit });
        assert!((p.cells["f"][0].m - 16.0).abs() < 1e-9);
        // a cell seen in 1 of 4 deployments is dropped at cov_min .5, kept (scaled by 1/4) at 0
        let mut d4 = mk(&[0.8, 0.8, 0.8, 0.8]);
        for d in d4.iter_mut().skip(1) {
            d.cells.insert("f".into(), vec![(0.0, 0.0), (0.0, 0.0)]);
        }
        assert!(fit_prior(&d4, 2, FitCfg { cov_min: 0.5, ..fit }).cells["f"][0].m == 0.0);
        assert!((fit_prior(&d4, 2, FitCfg { cov_min: 0.0, ..fit }).cells["f"][0].m - 0.5).abs() < 1e-9);
        // thin cells (< 3 trials per deployment) carry nothing
        let mut thin = PoolLog::new();
        thin.cells.insert("g".into(), vec![(2.0, 2.0)]);
        let p = fit_prior(&[thin.clone(), thin], 1, fit);
        assert_eq!(p.n_informative(), 0);
        assert!(p.is_empty());
    }

    #[test]
    fn trust_wrong_prior_loses_weight_right_prior_keeps_it() {
        let mut cells = HashMap::new();
        cells.insert("f".to_string(), vec![Cell { s: 0.0, n: 100.0, mu: 0.85, m: 8.0 }]);
        let prior = PopPrior { cells, n_deps: 4 };
        let tc = TrustCfg { w0: 0.8, cf: 1.0, cg: 0.0, delta: 0.97 };
        let t = task(0, &["f"]);
        let mut rng = Rng::new(5);
        let mut r = PopGuarded::new(HashMap::new(), 1, prior.clone(), Some(tc));
        for _ in 0..40 {
            r.update(&t, 0, if rng.bernoulli(0.3) { 1.0 } else { 0.0 });
        }
        assert!(r.weight("f") < 0.2, "wrong prior trust {}", r.weight("f"));
        let mut r = PopGuarded::new(HashMap::new(), 1, prior, Some(tc));
        for _ in 0..8 {
            r.update(&t, 0, if rng.bernoulli(0.85) { 1.0 } else { 0.0 });
        }
        assert!(r.weight("f") > 0.5, "right prior trust {}", r.weight("f"));
    }

    #[test]
    fn informed_prior_changes_cold_decisions_and_pool_log_roundtrips() {
        // pool says agent 3 is best on "novel"; no keyword for it => informed rule follows the pool, cold router would not
        let mut logs = vec![];
        for _ in 0..4 {
            let mut l = PoolLog::new();
            let t = task(0, &["novel"]);
            for i in 0..40 {
                l.record(&t, 3, if i % 10 < 9 { 1.0 } else { 0.0 }, 5);
                l.record(&t, 1, if i % 10 < 3 { 1.0 } else { 0.0 }, 5);
            }
            logs.push(l);
        }
        let j = logs[0].to_json();
        assert_eq!(PoolLog::from_json(&j).unwrap(), logs[0]);
        let prior = fit_prior(&logs, 5, FitCfg::default());
        assert!(prior.cell("novel", 3).unwrap().m > 8.0);
        let mut r = recommended_warm_router(HashMap::new(), 5, &logs);
        let mut rng = Rng::new(9);
        let t = task(1, &["novel"]);
        assert_eq!(r.select(&t, &[0, 1, 2, 3, 4], &mut rng), 3);
        assert_eq!(r.decisions, [0, 1, 0]);
        // serde roundtrip of the prior
        let pj = serde_json::to_string(&prior).unwrap();
        assert_eq!(serde_json::from_str::<PopPrior>(&pj).unwrap(), prior);
    }

    #[test]
    fn defaults_unchanged() {
        use crate::hebbian::HebbianConfig;
        assert_eq!(GuardConfig::default().tau, 0.6);
        assert_eq!(HebbianConfig::default().eta, 0.5);
    }
}
