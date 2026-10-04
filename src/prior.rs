//! Warm-start prior: a static keyword map that hands control to a learned
//! router as evidence accumulates.
//!
//! * [`KeywordPrior`]: the static keyword map (same voting rule as
//!   `baselines::KeywordRouter`) plus access to the raw votes so other
//!   components (e.g. `casebased::CaseRouter`) can use it as pseudo-counts.
//! * [`Warm<R>`]: wraps any [`Router`]. On each `select` it follows the keyword
//!   map with probability `w = k / (k + n)` where `n` is the amount of evidence
//!   for this task (see [`Evidence`]), otherwise it asks the inner router.
//!   The inner router is trained on every outcome (including those produced by
//!   following the keyword map). At `n = 0` the wrapper is *exactly* the keyword
//!   router (w = 1 and `rng.f64() < 1` always holds), so it cannot do worse than
//!   keyword before any outcome has been seen.

use std::collections::{BTreeMap, HashMap};

use crate::baselines::KeywordRouter;
use crate::router::{Rng, Router, Task};

/// Static keyword -> agent prior.
#[derive(Clone, Debug)]
pub struct KeywordPrior {
    router: KeywordRouter,
}

impl KeywordPrior {
    pub fn new(map: HashMap<String, usize>, n_agents: usize) -> Self {
        KeywordPrior { router: KeywordRouter::new(map, n_agents) }
    }

    pub fn n_agents(&self) -> usize {
        self.router.n_agents
    }

    /// Vote counts per candidate agent (features + raw task_type), ascending agent order.
    pub fn votes(&self, task: &Task, candidates: &[usize]) -> Vec<(usize, usize)> {
        let mut v: BTreeMap<usize, usize> = BTreeMap::new();
        for k in task.features.iter().chain(task.task_type.iter()) {
            if let Some(&a) = self.router.map.get(k)
                && candidates.contains(&a)
            {
                *v.entry(a).or_insert(0) += 1;
            }
        }
        v.into_iter().collect()
    }

    /// Exactly what `KeywordRouter::select` returns (deterministic, incl. fallback).
    pub fn choose(&mut self, task: &Task, candidates: &[usize]) -> usize {
        let mut dummy = Rng::new(0);
        self.router.select(task, candidates, &mut dummy)
    }
}

/// How much evidence has accumulated for a task (drives the hand-over).
#[derive(Clone, Copy)]
pub enum Evidence<R> {
    /// Number of outcomes seen for this exact (sorted) feature set.
    FeatureSet,
    /// Mean over the task's features of the number of outcomes involving that feature.
    MeanFeature,
    /// Total number of outcomes seen, regardless of task.
    Global,
    /// Evidence AGAINST the keyword map: accumulated failure mass (1 - reward) of
    /// outcomes where the keyword agent was the one that ran, per exact feature set.
    /// Keeps trusting keywords where they work, hands over where they fail.
    SetFailures,
    /// As `SetFailures`, averaged over the task's features.
    FeatureFailures,
    /// Supplied by the inner router, e.g. kernel mass of nearby cases.
    Inner(fn(&R, &Task) -> f64),
}

/// Keyword-prior wrapper; see module docs.
pub struct Warm<R: Router> {
    pub inner: R,
    pub prior: KeywordPrior,
    /// Hand-over scale: w = k / (k + n). Larger = trust keywords longer.
    pub k: f64,
    pub evidence: Evidence<R>,
    set_n: HashMap<String, f64>,
    feat_n: HashMap<String, f64>,
    fail_set: HashMap<String, f64>,
    fail_feat: HashMap<String, f64>,
    total: f64,
}

impl<R: Router> Warm<R> {
    pub fn new(inner: R, map: HashMap<String, usize>, n_agents: usize, k: f64) -> Self {
        Warm {
            inner,
            prior: KeywordPrior::new(map, n_agents),
            k: k.max(0.0),
            evidence: Evidence::FeatureSet,
            set_n: HashMap::new(),
            feat_n: HashMap::new(),
            fail_set: HashMap::new(),
            fail_feat: HashMap::new(),
            total: 0.0,
        }
    }

    pub fn with_evidence(mut self, e: Evidence<R>) -> Self {
        self.evidence = e;
        self
    }

    fn set_key(task: &Task) -> String {
        let mut f: Vec<&str> = task.features.iter().map(|s| s.as_str()).collect();
        f.sort_unstable();
        f.dedup();
        let mut s = f.join("|");
        if let Some(t) = &task.task_type {
            s.push_str("#");
            s.push_str(t);
        }
        s
    }

    /// Evidence count n for `task`.
    pub fn n_evidence(&self, task: &Task) -> f64 {
        match self.evidence {
            Evidence::FeatureSet => self.set_n.get(&Self::set_key(task)).copied().unwrap_or(0.0),
            Evidence::MeanFeature => {
                let keys: Vec<&String> = task.features.iter().collect();
                if keys.is_empty() {
                    return self.set_n.get(&Self::set_key(task)).copied().unwrap_or(0.0);
                }
                keys.iter().map(|k| self.feat_n.get(*k).copied().unwrap_or(0.0)).sum::<f64>() / keys.len() as f64
            }
            Evidence::Global => self.total,
            Evidence::SetFailures => self.fail_set.get(&Self::set_key(task)).copied().unwrap_or(0.0),
            Evidence::FeatureFailures => {
                if task.features.is_empty() {
                    return self.fail_set.get(&Self::set_key(task)).copied().unwrap_or(0.0);
                }
                task.features.iter().map(|k| self.fail_feat.get(k).copied().unwrap_or(0.0)).sum::<f64>()
                    / task.features.len() as f64
            }
            Evidence::Inner(f) => f(&self.inner, task),
        }
    }

    /// Probability of following the keyword map for `task` right now.
    pub fn keyword_weight(&self, task: &Task) -> f64 {
        let n = self.n_evidence(task).max(0.0);
        if self.k + n <= 0.0 { 0.0 } else { self.k / (self.k + n) }
    }
}

impl<R: Router> Router for Warm<R> {
    fn name(&self) -> String {
        format!("warm({},k={})", self.inner.name(), self.k)
    }

    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "Warm::select: no candidates");
        let w = self.keyword_weight(task);
        let u = rng.f64(); // < 1 always, so w == 1 => keyword
        if u < w {
            self.prior.choose(task, candidates)
        } else {
            self.inner.select(task, candidates, rng)
        }
    }

    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        *self.set_n.entry(Self::set_key(task)).or_insert(0.0) += 1.0;
        for f in &task.features {
            *self.feat_n.entry(f.clone()).or_insert(0.0) += 1.0;
        }
        self.total += 1.0;
        let all: Vec<usize> = (0..self.prior.n_agents().max(agent + 1)).collect();
        if self.prior.choose(task, &all) == agent {
            let fail = 1.0 - reward.clamp(0.0, 1.0);
            *self.fail_set.entry(Self::set_key(task)).or_insert(0.0) += fail;
            for f in &task.features {
                *self.fail_feat.entry(f.clone()).or_insert(0.0) += fail;
            }
        }
        self.inner.update(task, agent, reward);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baselines::{EpsGreedyRouter, PooledThompsonRouter};

    fn t(id: u64, f: &[&str]) -> Task {
        Task { id, task_type: None, features: f.iter().map(|s| s.to_string()).collect() }
    }
    fn map() -> HashMap<String, usize> {
        [("sql", 1), ("db", 1), ("css", 2), ("api", 0)].iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn identical_to_keyword_at_zero_evidence() {
        let mut w = Warm::new(EpsGreedyRouter::new(3, 0.99, 0.5), map(), 3, 10.0);
        let mut kw = KeywordRouter::new(map(), 3);
        let mut rng = Rng::new(3);
        let mut rng2 = Rng::new(3);
        for i in 0..200u64 {
            let fs: [&[&str]; 4] = [&["sql"], &["css", "api"], &["zzz"], &["db", "css", "sql"]];
            let task = t(i, fs[(i % 4) as usize]);
            assert_eq!(w.select(&task, &[0, 1, 2], &mut rng), kw.select(&task, &[0, 1, 2], &mut rng2));
        }
        assert_eq!(w.keyword_weight(&t(0, &["sql"])), 1.0);
    }

    #[test]
    fn weight_decays_with_observations() {
        let mut w = Warm::new(PooledThompsonRouter::new(3, 1.0), map(), 3, 5.0);
        let task = t(0, &["sql"]);
        assert!((w.keyword_weight(&task) - 1.0).abs() < 1e-12);
        for _ in 0..5 {
            w.update(&task, 1, 1.0);
        }
        assert!((w.keyword_weight(&task) - 0.5).abs() < 1e-12);
        // other feature sets are untouched
        assert_eq!(w.keyword_weight(&t(0, &["css"])), 1.0);
        let g = Warm::new(PooledThompsonRouter::new(3, 1.0), map(), 3, 5.0).with_evidence(Evidence::Global);
        assert_eq!(g.keyword_weight(&task), 1.0);
    }

    #[test]
    fn hands_over_when_keyword_is_wrong() {
        // keyword says agent 1 for "sql" but agent 2 is truly best.
        let p = [0.2, 0.3, 0.9];
        let mut w = Warm::new(EpsGreedyRouter::new(3, 1.0, 0.02), map(), 3, 8.0);
        let mut rng = Rng::new(9);
        let task = t(0, &["sql"]);
        let mut late_best = 0;
        for i in 0..1500 {
            let a = w.select(&task, &[0, 1, 2], &mut rng);
            w.update(&task, a, if rng.bernoulli(p[a]) { 1.0 } else { 0.0 });
            if i >= 1000 && a == 2 {
                late_best += 1;
            }
        }
        assert!(late_best > 400, "late picks of best agent {late_best}/500");
    }

    #[test]
    fn failure_evidence_only_counts_keyword_failures() {
        let mut w = Warm::new(PooledThompsonRouter::new(3, 1.0), map(), 3, 2.0).with_evidence(Evidence::SetFailures);
        let task = t(0, &["sql"]); // keyword agent = 1
        w.update(&task, 1, 1.0); // keyword success: no evidence against
        w.update(&task, 0, 0.0); // non-keyword failure: ignored
        assert_eq!(w.keyword_weight(&task), 1.0);
        w.update(&task, 1, 0.0);
        w.update(&task, 1, 0.0);
        assert!((w.keyword_weight(&task) - 0.5).abs() < 1e-12); // k=2, fails=2
    }

    #[test]
    fn prior_votes_and_choose() {
        let mut p = KeywordPrior::new(map(), 3);
        assert_eq!(p.votes(&t(0, &["sql", "db", "css"]), &[0, 1, 2]), vec![(1, 2), (2, 1)]);
        assert_eq!(p.votes(&t(0, &["sql", "db", "css"]), &[0, 2]), vec![(2, 1)]);
        assert_eq!(p.choose(&t(0, &["sql", "db", "css"]), &[0, 1, 2]), 1);
    }
}
