//! Baseline routers the Hebbian router / mesh must beat.
//!
//! * [`KeywordRouter`]: static keyword -> agent map, majority vote.
//! * [`ThompsonRouter`]: Beta-Bernoulli Thompson sampling per
//!   (context, agent), context = task type, else first feature.
//!   NOTE: an independent audit showed this keying is a weak baseline.
//! * [`PooledThompsonRouter`]: fair Thompson — Beta counts per (feature, agent)
//!   pooled over all active features, with optional discounting.
//! * [`EpsGreedyRouter`]: epsilon-greedy over per-(feature, agent) discounted
//!   mean rewards, averaged over the task's features. Untuned reference that
//!   any learned router must beat.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::router::{Rng, Router, Task};

/// Static map from keyword to agent. Votes are cast by every feature of the
/// task and by the raw `task_type` (if any) that appear in the map and whose
/// agent is a candidate. The agent with most votes wins; ties go to the lowest
/// agent index. With no votes the fallback is deterministic:
/// `candidates[task.id % candidates.len()]`. Never consumes randomness and
/// never learns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KeywordRouter {
    pub map: HashMap<String, usize>,
    pub n_agents: usize,
}

impl KeywordRouter {
    pub fn new(map: HashMap<String, usize>, n_agents: usize) -> Self {
        KeywordRouter { map, n_agents }
    }
}

impl Router for KeywordRouter {
    fn name(&self) -> String {
        "keyword".into()
    }

    fn select(&mut self, task: &Task, candidates: &[usize], _rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "KeywordRouter::select: no candidates");
        let mut votes: BTreeMap<usize, usize> = BTreeMap::new();
        let keys = task.features.iter().chain(task.task_type.iter());
        for k in keys {
            if let Some(&a) = self.map.get(k)
                && candidates.contains(&a)
            {
                *votes.entry(a).or_insert(0) += 1;
            }
        }
        // BTreeMap iterates in ascending agent order; strict '>' keeps lowest on ties.
        let mut best: Option<(usize, usize)> = None;
        for (&a, &v) in &votes {
            if best.is_none_or(|(_, bv)| v > bv) {
                best = Some((a, v));
            }
        }
        match best {
            Some((a, _)) => a,
            None => candidates[(task.id % candidates.len() as u64) as usize],
        }
    }

    fn update(&mut self, _task: &Task, _agent: usize, _reward: f64) {}
}

/// Standard normal via Box-Muller (consumes two uniforms).
fn std_normal(rng: &mut Rng) -> f64 {
    let u1 = 1.0 - rng.f64(); // (0, 1]
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Gamma(shape, 1) via Marsaglia & Tsang (2000); shape < 1 handled with the
/// boost `Gamma(a) = Gamma(a + 1) * U^(1/a)`.
pub fn sample_gamma(shape: f64, rng: &mut Rng) -> f64 {
    if shape.is_nan() || shape <= 0.0 {
        return 0.0;
    }
    if shape < 1.0 {
        let u = 1.0 - rng.f64();
        return sample_gamma(shape + 1.0, rng) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = std_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = 1.0 - rng.f64();
        if u < 1.0 - 0.0331 * x * x * x * x || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Beta(a, b) as X / (X + Y), X ~ Gamma(a), Y ~ Gamma(b).
pub fn sample_beta(a: f64, b: f64, rng: &mut Rng) -> f64 {
    let x = sample_gamma(a, rng);
    let y = sample_gamma(b, rng);
    if x + y <= 0.0 { 0.5 } else { x / (x + y) }
}

/// Thompson sampling with a Beta(1,1) prior per (context, agent).
/// Rewards in [0,1] are treated as fractional successes:
/// `alpha += r`, `beta += 1 - r`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThompsonRouter {
    pub n_agents: usize,
    /// context -> per-agent (alpha, beta).
    pub posteriors: BTreeMap<String, Vec<(f64, f64)>>,
}

impl ThompsonRouter {
    pub fn new(n_agents: usize) -> Self {
        ThompsonRouter { n_agents, posteriors: BTreeMap::new() }
    }

    /// Context key: `task_type`, else first feature, else "".
    pub fn context(task: &Task) -> String {
        task.task_type.clone().or_else(|| task.features.first().cloned()).unwrap_or_default()
    }

    /// Posterior (alpha, beta) for an agent in a context (prior (1,1) if unseen).
    pub fn posterior(&self, ctx: &str, agent: usize) -> (f64, f64) {
        self.posteriors.get(ctx).and_then(|r| r.get(agent).copied()).unwrap_or((1.0, 1.0))
    }

    /// Posterior mean success probability.
    pub fn mean(&self, ctx: &str, agent: usize) -> f64 {
        let (a, b) = self.posterior(ctx, agent);
        a / (a + b)
    }
}

impl Router for ThompsonRouter {
    fn name(&self) -> String {
        "thompson".into()
    }

    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "ThompsonRouter::select: no candidates");
        let ctx = Self::context(task);
        let mut best = candidates[0];
        let mut best_v = f64::NEG_INFINITY;
        for &c in candidates {
            let (a, b) = self.posterior(&ctx, c);
            let v = sample_beta(a, b, rng);
            if v > best_v {
                best_v = v;
                best = c;
            }
        }
        best
    }

    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        if !reward.is_finite() {
            return;
        }
        let r = reward.clamp(0.0, 1.0);
        self.n_agents = self.n_agents.max(agent + 1);
        let n = self.n_agents;
        let row = self.posteriors.entry(Self::context(task)).or_insert_with(|| vec![(1.0, 1.0); n]);
        if row.len() < n {
            row.resize(n, (1.0, 1.0));
        }
        row[agent].0 += r;
        row[agent].1 += 1.0 - r;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(id: u64, ty: Option<&str>, f: &[&str]) -> Task {
        Task { id, task_type: ty.map(Into::into), features: f.iter().map(|s| s.to_string()).collect() }
    }

    #[test]
    fn keyword_majority_vote_and_fallback() {
        let map: HashMap<String, usize> =
            [("sql", 1), ("db", 1), ("css", 2), ("ui", 2), ("api", 0)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
        let mut r = KeywordRouter::new(map, 3);
        let mut rng = Rng::new(1);
        assert_eq!(r.select(&t(0, None, &["sql", "db", "css"]), &[0, 1, 2], &mut rng), 1);
        // tie 1 vs 2 -> lowest index
        assert_eq!(r.select(&t(0, None, &["sql", "css"]), &[0, 1, 2], &mut rng), 1);
        // agent 1 not a candidate -> its votes are ignored
        assert_eq!(r.select(&t(0, None, &["sql", "db", "css"]), &[0, 2], &mut rng), 2);
        // no votes -> deterministic id-based fallback
        assert_eq!(r.select(&t(5, None, &["zzz"]), &[0, 1, 2], &mut rng), 2);
        assert_eq!(r.select(&t(5, None, &["zzz"]), &[0, 1, 2], &mut rng), 2);
        // task_type votes too
        assert_eq!(r.select(&t(0, Some("api"), &[]), &[0, 1, 2], &mut rng), 0);
    }

    #[test]
    fn beta_sampler_moments() {
        let mut rng = Rng::new(11);
        for &(a, b) in &[(1.0, 1.0), (2.0, 5.0), (30.0, 10.0), (0.5, 0.5)] {
            let n = 20000;
            let xs: Vec<f64> = (0..n).map(|_| sample_beta(a, b, &mut rng)).collect();
            assert!(xs.iter().all(|x| (0.0..=1.0).contains(x)));
            let m = xs.iter().sum::<f64>() / n as f64;
            let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n as f64;
            let em = a / (a + b);
            let ev = a * b / ((a + b) * (a + b) * (a + b + 1.0));
            assert!((m - em).abs() < 0.01, "Beta({a},{b}) mean {m} vs {em}");
            assert!((var - ev).abs() < 0.1 * ev + 0.002, "Beta({a},{b}) var {var} vs {ev}");
        }
    }

    #[test]
    fn thompson_posterior_sanity_and_convergence() {
        let mut r = ThompsonRouter::new(2);
        let task = t(0, Some("x"), &["k"]);
        assert_eq!(r.posterior("x", 0), (1.0, 1.0));
        r.update(&task, 0, 1.0);
        r.update(&task, 0, 0.0);
        r.update(&task, 0, 1.0);
        assert_eq!(r.posterior("x", 0), (3.0, 2.0));
        assert_eq!(r.posterior("x", 1), (1.0, 1.0));
        // context falls back to first feature when no task type
        r.update(&t(0, None, &["k"]), 1, 1.0);
        assert_eq!(r.posterior("k", 1), (2.0, 1.0));

        let mut r = ThompsonRouter::new(2);
        let mut rng = Rng::new(5);
        let p = [0.3, 0.8];
        let mut best = 0;
        for i in 0..2000 {
            let a = r.select(&task, &[0, 1], &mut rng);
            r.update(&task, a, if rng.bernoulli(p[a]) { 1.0 } else { 0.0 });
            if i >= 1500 && a == 1 {
                best += 1;
            }
        }
        assert!(best > 450, "thompson picked best {best}/500");
        assert!((r.mean("x", 1) - 0.8).abs() < 0.05);
    }
}


fn task_keys(task: &Task) -> Vec<String> {
    let mut k: Vec<String> = task.features.clone();
    if let Some(t) = &task.task_type {
        k.push(format!("type:{t}"));
    }
    if k.is_empty() {
        k.push("no-keyword".into());
    }
    k
}

/// Thompson sampling with Beta counts kept per (feature, agent) and pooled over
/// the task's active features: alpha = 1 + sum successes, beta = 1 + sum failures.
/// `gamma` < 1 discounts old counts (for drift); 1.0 = no discounting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PooledThompsonRouter {
    pub gamma: f64,
    n_agents: usize,
    /// feature -> per-agent (successes, failures)
    counts: BTreeMap<String, Vec<(f64, f64)>>,
}

impl PooledThompsonRouter {
    pub fn new(n_agents: usize, gamma: f64) -> Self {
        PooledThompsonRouter { gamma, n_agents, counts: BTreeMap::new() }
    }
    fn grow(&mut self, n: usize) {
        if n > self.n_agents {
            self.n_agents = n;
            for v in self.counts.values_mut() {
                v.resize(n, (0.0, 0.0));
            }
        }
    }
    /// Pooled posterior (alpha, beta) for `agent` on `task`.
    pub fn posterior(&self, task: &Task, agent: usize) -> (f64, f64) {
        let (mut s, mut f) = (0.0, 0.0);
        for k in task_keys(task) {
            if let Some((a, b)) = self.counts.get(&k).and_then(|v| v.get(agent)) {
                s += a;
                f += b;
            }
        }
        (1.0 + s, 1.0 + f)
    }
}

impl Router for PooledThompsonRouter {
    fn name(&self) -> String {
        "pooled-thompson".into()
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "PooledThompsonRouter::select: no candidates");
        let mut best = candidates[0];
        let mut best_v = f64::NEG_INFINITY;
        for &c in candidates {
            let (a, b) = self.posterior(task, c);
            let v = sample_beta(a, b, rng);
            if v > best_v {
                best_v = v;
                best = c;
            }
        }
        best
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        self.grow(agent + 1);
        let r = reward.clamp(0.0, 1.0);
        let n = self.n_agents;
        let g = self.gamma;
        for k in task_keys(task) {
            let row = self.counts.entry(k).or_insert_with(|| vec![(0.0, 0.0); n]);
            if g < 1.0 {
                for c in row.iter_mut() {
                    c.0 *= g;
                    c.1 *= g;
                }
            }
            row[agent].0 += r;
            row[agent].1 += 1.0 - r;
        }
    }
}

/// Epsilon-greedy over discounted per-(feature, agent) mean rewards, averaged
/// across the task's features. Untried cells score an optimistic 1.0.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpsGreedyRouter {
    pub gamma: f64,
    pub epsilon: f64,
    n_agents: usize,
    /// feature -> per-agent (discounted reward sum, discounted count)
    stats: BTreeMap<String, Vec<(f64, f64)>>,
}

impl EpsGreedyRouter {
    pub fn new(n_agents: usize, gamma: f64, epsilon: f64) -> Self {
        EpsGreedyRouter { gamma, epsilon, n_agents, stats: BTreeMap::new() }
    }
    pub fn value(&self, task: &Task, agent: usize) -> f64 {
        let keys = task_keys(task);
        let mut tot = 0.0;
        for k in &keys {
            tot += match self.stats.get(k).and_then(|v| v.get(agent)) {
                Some(&(s, n)) if n > 1e-9 => s / n,
                _ => 1.0,
            };
        }
        tot / keys.len() as f64
    }
}

impl Router for EpsGreedyRouter {
    fn name(&self) -> String {
        "eps-greedy".into()
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "EpsGreedyRouter::select: no candidates");
        if rng.bernoulli(self.epsilon) {
            return candidates[rng.below(candidates.len())];
        }
        let mut best = candidates[0];
        let mut best_v = f64::NEG_INFINITY;
        for &c in candidates {
            let v = self.value(task, c);
            if v > best_v {
                best_v = v;
                best = c;
            }
        }
        best
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        if agent + 1 > self.n_agents {
            self.n_agents = agent + 1;
            for v in self.stats.values_mut() {
                v.resize(self.n_agents, (0.0, 0.0));
            }
        }
        let n = self.n_agents;
        let g = self.gamma;
        for k in task_keys(task) {
            let row = self.stats.entry(k).or_insert_with(|| vec![(0.0, 0.0); n]);
            let c = &mut row[agent];
            c.0 = c.0 * g + reward.clamp(0.0, 1.0);
            c.1 = c.1 * g + 1.0;
        }
    }
}

#[cfg(test)]
mod fair_baseline_tests {
    use super::*;

    fn learns_best<R: Router>(mut r: R) -> f64 {
        let mut rng = Rng::new(7);
        let t = Task { id: 0, task_type: None, features: vec!["sql".into(), "db".into()] };
        let mut hits = 0;
        for i in 0..2000 {
            let a = r.select(&t, &[0, 1, 2], &mut rng);
            let p = [0.3, 0.85, 0.5][a];
            r.update(&t, a, if rng.bernoulli(p) { 1.0 } else { 0.0 });
            if i >= 1500 && a == 1 {
                hits += 1;
            }
        }
        hits as f64 / 500.0
    }

    #[test]
    fn pooled_thompson_converges() {
        assert!(learns_best(PooledThompsonRouter::new(3, 1.0)) > 0.85);
        assert!(learns_best(PooledThompsonRouter::new(3, 0.99)) > 0.7);
    }

    #[test]
    fn eps_greedy_converges() {
        assert!(learns_best(EpsGreedyRouter::new(3, 0.99, 0.02)) > 0.85);
    }

    #[test]
    fn pooled_thompson_pools_features() {
        let mut r = PooledThompsonRouter::new(2, 1.0);
        let t = Task { id: 0, task_type: None, features: vec!["a".into(), "b".into()] };
        r.update(&t, 1, 1.0);
        let only_a = Task { id: 1, task_type: None, features: vec!["a".into()] };
        assert_eq!(r.posterior(&only_a, 1), (2.0, 1.0));
        assert_eq!(r.posterior(&t, 1), (3.0, 1.0));
    }
}
