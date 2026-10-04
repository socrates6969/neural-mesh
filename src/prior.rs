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

/// Per-feature success/failure counts of the two arms of [`ArmWarm`].
#[derive(Clone, Copy, Debug, Default)]
struct ArmCounts {
    kw_s: f64,
    kw_f: f64,
    in_s: f64,
    in_f: f64,
}

/// Evidence-adaptive warm start (opt-in, new type; [`Warm`] is unchanged).
///
/// A two-arm Thompson meta-bandit decides, per task, between "follow the keyword
/// map" and "ask the learned router". Each arm has a Beta posterior over the
/// reward it earned on the task's features (counts pooled as the mean over the
/// task's features, times `pool`). The keyword arm starts optimistic
/// (`Beta(kw_mean*k, (1-kw_mean)*k)`), the learned arm neutral
/// (`Beta(in_mean*k_in, ..)`), so the keyword map is trusted until its own
/// outcomes show it losing to the learner, and is kept for good where it works.
/// The inner router is trained on every outcome.
pub struct ArmWarm<R: Router> {
    pub inner: R,
    pub prior: KeywordPrior,
    /// Prior strength of the keyword arm (pseudo-trials).
    pub k: f64,
    /// Prior mean reward of the keyword arm.
    pub kw_mean: f64,
    /// Prior strength / mean of the learned arm.
    pub k_in: f64,
    pub in_mean: f64,
    /// Multiplier on the pooled counts (>1 = more confident, hands over faster).
    pub pool: f64,
    /// Per-update discount of old counts of a feature (1 = none).
    pub discount: f64,
    counts: HashMap<String, ArmCounts>,
    pending: Option<(u64, bool)>,
}

impl<R: Router> ArmWarm<R> {
    pub fn new(inner: R, map: HashMap<String, usize>, n_agents: usize) -> Self {
        ArmWarm {
            inner,
            prior: KeywordPrior::new(map, n_agents),
            k: 4.0,
            kw_mean: 0.8,
            k_in: 2.0,
            in_mean: 0.5,
            pool: 1.0,
            discount: 1.0,
            counts: HashMap::new(),
            pending: None,
        }
    }

    pub fn with_params(mut self, k: f64, kw_mean: f64, k_in: f64, in_mean: f64, pool: f64, discount: f64) -> Self {
        self.k = k.max(1e-6);
        self.kw_mean = kw_mean.clamp(0.0, 1.0);
        self.k_in = k_in.max(1e-6);
        self.in_mean = in_mean.clamp(0.0, 1.0);
        self.pool = pool.max(0.0);
        self.discount = discount.clamp(0.0, 1.0);
        self
    }

    fn pooled(&self, task: &Task) -> ArmCounts {
        let mut a = ArmCounts::default();
        let mut n = 0.0;
        for f in &task.features {
            if let Some(c) = self.counts.get(f) {
                a.kw_s += c.kw_s;
                a.kw_f += c.kw_f;
                a.in_s += c.in_s;
                a.in_f += c.in_f;
            }
            n += 1.0;
        }
        if n > 0.0 {
            let m = self.pool / n;
            a.kw_s *= m;
            a.kw_f *= m;
            a.in_s *= m;
            a.in_f *= m;
        }
        a
    }

    /// Posterior mean rewards (keyword arm, learned arm) for `task`.
    pub fn arm_means(&self, task: &Task) -> (f64, f64) {
        let c = self.pooled(task);
        (
            (self.kw_mean * self.k + c.kw_s) / (self.k + c.kw_s + c.kw_f),
            (self.in_mean * self.k_in + c.in_s) / (self.k_in + c.in_s + c.in_f),
        )
    }
}

impl<R: Router> Router for ArmWarm<R> {
    fn name(&self) -> String {
        format!("armwarm({},k={},m={})", self.inner.name(), self.k, self.kw_mean)
    }

    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "ArmWarm::select: no candidates");
        let c = self.pooled(task);
        let (ka, kb) = (self.kw_mean * self.k + c.kw_s, (1.0 - self.kw_mean) * self.k + c.kw_f);
        let (ia, ib) = (self.in_mean * self.k_in + c.in_s, (1.0 - self.in_mean) * self.k_in + c.in_f);
        let pk = crate::baselines::sample_beta(ka.max(1e-3), kb.max(1e-3), rng);
        let pi = crate::baselines::sample_beta(ia.max(1e-3), ib.max(1e-3), rng);
        let use_kw = pk >= pi;
        self.pending = Some((task.id, use_kw));
        if use_kw { self.prior.choose(task, candidates) } else { self.inner.select(task, candidates, rng) }
    }

    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let r = reward.clamp(0.0, 1.0);
        let arm_kw = match self.pending.take() {
            Some((id, a)) if id == task.id => a,
            _ => {
                let all: Vec<usize> = (0..self.prior.n_agents().max(agent + 1)).collect();
                self.prior.choose(task, &all) == agent
            }
        };
        for f in &task.features {
            let c = self.counts.entry(f.clone()).or_default();
            if self.discount < 1.0 {
                c.kw_s *= self.discount;
                c.kw_f *= self.discount;
                c.in_s *= self.discount;
                c.in_f *= self.discount;
            }
            if arm_kw {
                c.kw_s += r;
                c.kw_f += 1.0 - r;
            } else {
                c.in_s += r;
                c.in_f += 1.0 - r;
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
    fn armwarm_keeps_good_keyword_and_abandons_bad() {
        use crate::hebbian::{HebbianConfig, HebbianRouter};
        let heb = || HebbianRouter::new(3, HebbianConfig { eta: 0.05, temperature: 0.02, epsilon: 0.02, ..HebbianConfig::default() });
        // keyword -> agent 1 for "sql". Case A: keyword agent is best.
        let mut rng = Rng::new(5);
        let mut w = ArmWarm::new(heb(), map(), 3);
        let task = t(0, &["sql"]);
        let p = [0.2, 0.9, 0.3];
        let mut kw_picks = 0;
        for i in 0..600 {
            let a = w.select(&task, &[0, 1, 2], &mut rng);
            if i >= 300 && a == 1 { kw_picks += 1; }
            w.update(&task, a, if rng.bernoulli(p[a]) { 1.0 } else { 0.0 });
        }
        assert!(kw_picks > 250, "kept keyword {kw_picks}/300");
        // Case B: keyword agent is bad, agent 2 is best.
        let mut w = ArmWarm::new(heb(), map(), 3);
        let p = [0.2, 0.15, 0.9];
        let mut best = 0;
        for i in 0..1500 {
            let a = w.select(&task, &[0, 1, 2], &mut rng);
            if i >= 1000 && a == 2 { best += 1; }
            w.update(&task, a, if rng.bernoulli(p[a]) { 1.0 } else { 0.0 });
        }
        assert!(best > 350, "late best picks {best}/500");
        // at zero evidence it is exactly the keyword router
        let mut w = ArmWarm::new(heb(), map(), 3).with_params(1e6, 1.0, 1.0, 0.0, 1.0, 1.0);
        assert_eq!(w.select(&task, &[0, 1, 2], &mut rng), 1);
    }

    #[test]
    fn prior_votes_and_choose() {
        let mut p = KeywordPrior::new(map(), 3);
        assert_eq!(p.votes(&t(0, &["sql", "db", "css"]), &[0, 1, 2]), vec![(1, 2), (2, 1)]);
        assert_eq!(p.votes(&t(0, &["sql", "db", "css"]), &[0, 2]), vec![(2, 1)]);
        assert_eq!(p.choose(&t(0, &["sql", "db", "css"]), &[0, 1, 2]), 1);
    }
}
