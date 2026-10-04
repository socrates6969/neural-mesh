//! Baseline routers the Hebbian router / mesh must beat.
//!
//! * [`KeywordRouter`]: static keyword -> agent map, majority vote.
//! * [`ThompsonRouter`]: Beta-Bernoulli Thompson sampling per
//!   (context, agent), context = task type, else first feature.

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
