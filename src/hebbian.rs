//! Reward-modulated three-factor Hebbian router.
//!
//! Implemented independently from the *published algorithm description* only
//! (see DESIGN.md); no third-party code was consulted or copied.
//!
//! Model
//! -----
//! * Synapses `w[feature][agent]` in `[0, 1]`. Active features of a task are
//!   `task.features` plus `"type:<task_type>"` when a task type is present
//!   (prefixed so a type never collides with a keyword of the same name).
//!   Duplicate features are counted once.
//! * `score(agent) = sum_{f active} w[f][agent]`.
//! * `select`: epsilon-greedy over a softmax of `score / temperature`
//!   restricted to the candidate set.
//! * `update` (three factors: pre = feature active (1.0), post = agent was the
//!   one chosen, modulator = `reward - baseline`): for every active feature row
//!     - every weight in the row decays: `w *= 1 - decay`
//!     - the chosen agent gets `delta = eta * pre * (reward - baseline_f)`
//!     - weights are clamped to `[0, 1]`
//!     - if the row sum exceeds `row_cap` the row is rescaled to sum to `row_cap`
//!     - `baseline_f <- baseline_f + baseline_alpha * (reward - baseline_f)`
//!       (the *pre-update* baseline is used in `delta`).
//!
//! Interpretation choices (documented deviations from the one-line spec):
//! * The EMA baseline is kept **per feature** rather than globally, so a
//!   feature whose best achievable success rate is low is not dragged to zero
//!   by a global baseline dominated by easy features.
//! * Decay is applied to the whole active row (not only the chosen agent), so
//!   the chosen agent is not penalised relative to its competitors simply for
//!   being used; the effect is a slow flattening of stale preferences.
//! * Unknown features start at [`INIT_WEIGHT`] for every agent (neutral: it
//!   adds the same constant to every agent's score), new baselines at
//!   [`INIT_BASELINE`].

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::router::{Rng, Router, Task};

/// Initial synaptic weight for a (feature, agent) pair never seen before.
pub const INIT_WEIGHT: f64 = 0.1;
/// Initial reward baseline for a feature never seen before.
pub const INIT_BASELINE: f64 = 0.5;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HebbianConfig {
    /// Learning rate of the reward-modulated term.
    pub eta: f64,
    /// Probability of picking a uniformly random candidate.
    pub epsilon: f64,
    /// Softmax temperature (scores are divided by this). `<= 0` means argmax.
    pub temperature: f64,
    /// Multiplicative weight decay per update of a row.
    pub decay: f64,
    /// EMA rate of the per-feature reward baseline.
    pub baseline_alpha: f64,
    /// Maximum allowed sum of a feature row; larger rows are rescaled.
    pub row_cap: f64,
}

impl Default for HebbianConfig {
    fn default() -> Self {
        HebbianConfig {
            eta: 0.5,
            epsilon: 0.02,
            temperature: 0.05,
            decay: 0.01,
            baseline_alpha: 0.1,
            row_cap: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HebbianRouter {
    pub cfg: HebbianConfig,
    n_agents: usize,
    /// feature -> per-agent weights (len == n_agents).
    weights: BTreeMap<String, Vec<f64>>,
    /// feature -> EMA reward baseline.
    baselines: BTreeMap<String, f64>,
    /// Number of `update` calls seen.
    updates: u64,
}

/// Active features of a task: features (deduplicated, order kept) plus
/// `type:<task_type>` if present.
pub fn active_features(task: &Task) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(task.features.len() + 1);
    for f in &task.features {
        if !out.iter().any(|g| g == f) {
            out.push(f.clone());
        }
    }
    if let Some(t) = &task.task_type {
        let k = format!("type:{t}");
        if !out.contains(&k) {
            out.push(k);
        }
    }
    out
}

/// Sample an index from `logits` with a numerically stable softmax.
/// Consumes exactly one `rng.f64()`.
pub(crate) fn softmax_sample(logits: &[f64], rng: &mut Rng) -> usize {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits
        .iter()
        .map(|&l| if l.is_finite() { (l - max).exp() } else { 0.0 })
        .collect();
    let total: f64 = exps.iter().sum();
    let u = rng.f64();
    if !(total.is_finite() && total > 0.0) {
        return 0;
    }
    let mut acc = 0.0;
    let target = u * total;
    for (i, e) in exps.iter().enumerate() {
        acc += e;
        if target < acc {
            return i;
        }
    }
    exps.len() - 1
}

impl HebbianRouter {
    pub fn new(n_agents: usize, cfg: HebbianConfig) -> Self {
        HebbianRouter { cfg, n_agents, weights: BTreeMap::new(), baselines: BTreeMap::new(), updates: 0 }
    }

    pub fn n_agents(&self) -> usize {
        self.n_agents
    }

    /// Grow to at least `n` agents (new agents get [`INIT_WEIGHT`] in every row).
    pub fn ensure_agents(&mut self, n: usize) {
        if n > self.n_agents {
            for row in self.weights.values_mut() {
                row.resize(n, INIT_WEIGHT);
            }
            self.n_agents = n;
        }
    }

    /// Weight of (feature, agent); unknown pairs report [`INIT_WEIGHT`].
    pub fn weight(&self, feature: &str, agent: usize) -> f64 {
        self.weights.get(feature).and_then(|r| r.get(agent).copied()).unwrap_or(INIT_WEIGHT)
    }

    /// Reward baseline of a feature ([`INIT_BASELINE`] if unseen).
    pub fn baseline(&self, feature: &str) -> f64 {
        self.baselines.get(feature).copied().unwrap_or(INIT_BASELINE)
    }

    /// All learned rows (feature -> per-agent weights).
    pub fn rows(&self) -> &BTreeMap<String, Vec<f64>> {
        &self.weights
    }

    pub fn updates(&self) -> u64 {
        self.updates
    }

    fn score_of(&self, feats: &[String], agent: usize) -> f64 {
        feats.iter().map(|f| self.weight(f, agent)).sum()
    }

    /// One score per agent (`0..n_agents`).
    pub fn scores(&self, task: &Task) -> Vec<f64> {
        let feats = active_features(task);
        (0..self.n_agents).map(|a| self.score_of(&feats, a)).collect()
    }
}

impl Router for HebbianRouter {
    fn name(&self) -> String {
        "hebbian".into()
    }

    /// Panics if `candidates` is empty (contract: all candidates eligible, >= 1).
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "HebbianRouter::select: no candidates");
        // Always consume one draw for the epsilon test, then exactly one more.
        if rng.f64() < self.cfg.epsilon {
            return candidates[rng.below(candidates.len())];
        }
        let feats = active_features(task);
        let scores: Vec<f64> = candidates.iter().map(|&c| self.score_of(&feats, c)).collect();
        if self.cfg.temperature <= 0.0 {
            let _ = rng.f64(); // keep rng consumption identical to the softmax path
            let mut best = 0;
            for i in 1..scores.len() {
                if scores[i] > scores[best] {
                    best = i;
                }
            }
            return candidates[best];
        }
        let logits: Vec<f64> = scores.iter().map(|s| s / self.cfg.temperature).collect();
        candidates[softmax_sample(&logits, rng)]
    }

    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let reward = if reward.is_finite() { reward.clamp(0.0, 1.0) } else { return };
        self.ensure_agents(agent + 1);
        self.updates += 1;
        let n = self.n_agents;
        let cfg = self.cfg.clone();
        for f in active_features(task) {
            let b = self.baselines.get(&f).copied().unwrap_or(INIT_BASELINE);
            let row = self.weights.entry(f.clone()).or_insert_with(|| vec![INIT_WEIGHT; n]);
            let delta = cfg.eta * 1.0 * (reward - b);
            for w in row.iter_mut() {
                *w *= 1.0 - cfg.decay;
            }
            row[agent] += delta;
            for w in row.iter_mut() {
                *w = w.clamp(0.0, 1.0);
            }
            let sum: f64 = row.iter().sum();
            if cfg.row_cap > 0.0 && sum > cfg.row_cap {
                let k = cfg.row_cap / sum;
                for w in row.iter_mut() {
                    *w *= k;
                }
            }
            self.baselines.insert(f, b + cfg.baseline_alpha * (reward - b));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(t: &str) -> Task {
        Task { id: 0, task_type: Some(t.into()), features: vec![format!("kw-{t}")] }
    }

    /// Run `steps` rounds with success probabilities `p[agent]`; returns the
    /// fraction of picks of `track` in the last `window` rounds.
    fn run(r: &mut HebbianRouter, rng: &mut Rng, p: &[f64], steps: usize, track: usize, window: usize) -> f64 {
        let t = task("a");
        let cands: Vec<usize> = (0..p.len()).collect();
        let mut hits = 0;
        for i in 0..steps {
            let a = r.select(&t, &cands, rng);
            let rew = if rng.bernoulli(p[a]) { 1.0 } else { 0.0 };
            r.update(&t, a, rew);
            if i >= steps - window && a == track {
                hits += 1;
            }
        }
        hits as f64 / window as f64
    }

    #[test]
    fn converges_to_best_agent_stationary() {
        for seed in 1..=5 {
            let mut r = HebbianRouter::new(2, HebbianConfig::default());
            let mut rng = Rng::new(seed);
            let frac = run(&mut r, &mut rng, &[0.2, 0.9], 2000, 1, 500);
            assert!(frac > 0.85, "seed {seed}: best picked only {frac}");
            let s = r.scores(&task("a"));
            assert!(s[1] > s[0], "seed {seed}: scores {s:?}");
        }
    }

    #[test]
    fn decay_and_baseline_allow_switch_after_drift() {
        for seed in 1..=5 {
            let mut r = HebbianRouter::new(2, HebbianConfig::default());
            let mut rng = Rng::new(seed);
            let before = run(&mut r, &mut rng, &[0.9, 0.2], 1500, 0, 300);
            assert!(before > 0.85, "seed {seed}: pre-drift {before}");
            let after = run(&mut r, &mut rng, &[0.2, 0.9], 1500, 1, 300);
            assert!(after > 0.85, "seed {seed}: post-drift {after}");
        }
    }

    #[test]
    fn failure_lowers_weight_success_raises_it() {
        let mut r = HebbianRouter::new(3, HebbianConfig::default());
        let t = task("x");
        // Establish a baseline of ~0.5 then fail: weight must go below the start.
        let w0 = r.weight("kw-x", 1);
        r.update(&t, 1, 0.0);
        assert!(r.weight("kw-x", 1) < w0);
        assert!(r.baseline("kw-x") < INIT_BASELINE);
        let w1 = r.weight("kw-x", 2);
        r.update(&t, 2, 1.0);
        assert!(r.weight("kw-x", 2) > w1);
        // Once baseline is high (agent always succeeds), a failure is a large
        // negative surprise even though reward 0 would never increase a weight.
        let mut r = HebbianRouter::new(2, HebbianConfig::default());
        for _ in 0..50 {
            r.update(&t, 0, 1.0);
        }
        let hi = r.weight("kw-x", 0);
        assert!(r.baseline("kw-x") > 0.9);
        r.update(&t, 0, 0.0);
        assert!(r.weight("kw-x", 0) < hi - 0.3, "baseline-relative failure should cut weight");
    }

    #[test]
    fn row_cap_holds_and_weights_bounded() {
        let cfg = HebbianConfig { row_cap: 0.8, ..HebbianConfig::default() };
        let mut r = HebbianRouter::new(4, cfg);
        let t = task("y");
        let mut rng = Rng::new(7);
        for i in 0..2000 {
            let a = i % 4;
            r.update(&t, a, if rng.bernoulli(0.7) { 1.0 } else { 0.0 });
            for (f, row) in r.rows() {
                let s: f64 = row.iter().sum();
                assert!(s <= 0.8 + 1e-9, "row {f} sum {s}");
                assert!(row.iter().all(|w| (0.0..=1.0).contains(w)));
            }
        }
    }

    #[test]
    fn unknown_features_neutral_and_agents_grow() {
        let mut r = HebbianRouter::new(2, HebbianConfig::default());
        let s = r.scores(&task("new"));
        assert_eq!(s, vec![2.0 * INIT_WEIGHT; 2]);
        r.update(&task("new"), 4, 1.0);
        assert_eq!(r.n_agents(), 5);
        assert_eq!(r.scores(&task("new")).len(), 5);
    }

    #[test]
    fn serde_roundtrip_and_determinism() {
        let mut r = HebbianRouter::new(3, HebbianConfig::default());
        let mut rng = Rng::new(3);
        run(&mut r, &mut rng, &[0.3, 0.6, 0.9], 300, 2, 10);
        let js = serde_json::to_string(&r).unwrap();
        let back: HebbianRouter = serde_json::from_str(&js).unwrap();
        // serde_json may differ in the last ulp of a float; compare with tolerance.
        assert_eq!(back.n_agents(), r.n_agents());
        assert_eq!(back.updates(), r.updates());
        for (f, row) in r.rows() {
            for (a, w) in row.iter().enumerate() {
                assert!((back.weight(f, a) - w).abs() < 1e-12);
            }
            assert!((back.baseline(f) - r.baseline(f)).abs() < 1e-12);
        }
        let (mut a, mut b) = (r.clone(), back);
        let (mut ra, mut rb) = (Rng::new(9), Rng::new(9));
        for _ in 0..100 {
            assert_eq!(a.select(&task("a"), &[0, 1, 2], &mut ra), b.select(&task("a"), &[0, 1, 2], &mut rb));
        }
        a.update(&task("a"), 0, 1.0);
    }

    #[test]
    fn task_type_is_a_feature() {
        let t = Task { id: 1, task_type: Some("k".into()), features: vec!["k".into(), "k".into()] };
        assert_eq!(active_features(&t), vec!["k".to_string(), "type:k".to_string()]);
    }
}
