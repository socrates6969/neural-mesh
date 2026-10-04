//! Case-based router: tasks carry an embedding; every outcome is stored as a
//! case `(embedding, agent, reward)` in a [`MemoryIndex`]. To route a new task
//! it retrieves the `k` nearest cases and, per candidate agent, forms a
//! kernel-weighted Beta posterior
//!
//! ```text
//! alpha_a = a0 + wscale * sum_i w_i * r_i      [cases i of agent a]
//! beta_a  = b0 + wscale * sum_i w_i * (1-r_i)
//! w_i     = exp(-(1 - cos_i) / bandwidth)
//! ```
//!
//! and picks the agent with the best Thompson draw. Unlike per-keyword
//! learners, one outcome informs every task near it in embedding space.
//! An optional keyword prior adds pseudo-successes to the keyword-voted
//! agent (see [`crate::prior::KeywordPrior`]). Wrap in
//! [`crate::prior::Warm`] (with `Evidence::Inner(CaseRouter::local_mass)`) for
//! a hard hand-over from keywords.
//!
//! The router has no access to embeddings through [`Task`], so the caller
//! supplies an [`Embedder`] closure (e.g. a lookup keyed on `task.id`).

use std::sync::Arc;

use crate::baselines::sample_beta;
use crate::memory::{FlatIndex, MemoryIndex};
use crate::prior::KeywordPrior;
use crate::router::{Rng, Router, Task};

/// Maps a task to its embedding (None if unavailable).
pub type Embedder = Arc<dyn Fn(&Task) -> Option<Vec<f32>> + Send + Sync>;

#[derive(Clone, Debug)]
pub struct CaseConfig {
    /// Neighbours retrieved per query.
    pub k: usize,
    /// Kernel bandwidth on (1 - cosine). Smaller = more local.
    pub bandwidth: f64,
    /// Beta prior pseudo-successes / pseudo-failures.
    pub a0: f64,
    pub b0: f64,
    /// Multiplier on kernel weights (higher = tighter posteriors).
    pub wscale: f64,
    /// Total pseudo-observations added to keyword-voted agents (needs a prior). 0 = off.
    pub kw_pseudo: f64,
    /// Assumed success rate of the keyword agent in those pseudo-observations
    /// (1.0 = all successes; 0.7 = Beta(0.7m, 0.3m), easier to overturn).
    pub kw_mean: f64,
}

impl Default for CaseConfig {
    fn default() -> Self {
        CaseConfig { k: 30, bandwidth: 0.15, a0: 1.0, b0: 1.0, wscale: 2.0, kw_pseudo: 0.0, kw_mean: 1.0 }
    }
}

pub struct CaseRouter {
    pub cfg: CaseConfig,
    n_agents: usize,
    index: Box<dyn MemoryIndex>,
    cases: Vec<(usize, f32)>,
    embed: Embedder,
    prior: Option<KeywordPrior>,
}

impl CaseRouter {
    /// Exact (flat) index of dimension `dim`.
    pub fn new(n_agents: usize, dim: usize, cfg: CaseConfig, embed: Embedder) -> Self {
        Self::with_index(n_agents, Box::new(FlatIndex::new(dim)), cfg, embed)
    }

    /// Any backend, e.g. `HnswIndex` for large case bases.
    pub fn with_index(n_agents: usize, index: Box<dyn MemoryIndex>, cfg: CaseConfig, embed: Embedder) -> Self {
        CaseRouter { cfg, n_agents, index, cases: Vec::new(), embed, prior: None }
    }

    pub fn with_keyword_prior(mut self, prior: KeywordPrior) -> Self {
        self.prior = Some(prior);
        self
    }

    pub fn n_cases(&self) -> usize {
        self.cases.len()
    }

    fn weight(&self, cos: f32) -> f64 {
        (-(1.0 - cos as f64).max(0.0) / self.cfg.bandwidth.max(1e-9)).exp()
    }

    /// Kernel-weighted (successes, failures) per agent among the k nearest cases
    /// (before `wscale`, prior and keyword pseudo-counts) and the total kernel mass.
    pub fn neighbour_counts(&self, emb: &[f32]) -> (Vec<(f64, f64)>, f64) {
        let mut c = vec![(0.0, 0.0); self.n_agents];
        let mut mass = 0.0;
        if self.index.is_empty() {
            return (c, 0.0);
        }
        for (id, cos) in self.index.search(emb, self.cfg.k) {
            let Some(&(agent, r)) = self.cases.get(id as usize) else { continue };
            let w = self.weight(cos);
            if agent < c.len() {
                c[agent].0 += w * r as f64;
                c[agent].1 += w * (1.0 - r as f64);
            }
            mass += w;
        }
        (c, mass)
    }

    /// Kernel mass of the neighbourhood of `task` (a soft "how many similar cases").
    /// Use as `Evidence::Inner(CaseRouter::local_mass)` in [`crate::prior::Warm`].
    pub fn local_mass(&self, task: &Task) -> f64 {
        match (self.embed)(task) {
            Some(e) => self.neighbour_counts(&e).1,
            None => 0.0,
        }
    }

    /// Posterior (alpha, beta) per agent for `task` (without sampling).
    pub fn posteriors(&self, task: &Task, candidates: &[usize]) -> Vec<(f64, f64)> {
        let counts = match (self.embed)(task) {
            Some(e) => self.neighbour_counts(&e).0,
            None => vec![(0.0, 0.0); self.n_agents],
        };
        let mut post: Vec<(f64, f64)> = counts
            .iter()
            .map(|&(s, f)| (self.cfg.a0 + self.cfg.wscale * s, self.cfg.b0 + self.cfg.wscale * f))
            .collect();
        if self.cfg.kw_pseudo > 0.0
            && let Some(p) = &self.prior
        {
            let votes = p.votes(task, candidates);
            let tot: usize = votes.iter().map(|v| v.1).sum();
            for (a, n) in votes {
                if a < post.len() && tot > 0 {
                    let m = self.cfg.kw_pseudo * n as f64 / tot as f64;
                    post[a].0 += m * self.cfg.kw_mean;
                    post[a].1 += m * (1.0 - self.cfg.kw_mean);
                }
            }
        }
        post
    }
}

impl Router for CaseRouter {
    fn name(&self) -> String {
        format!("case[{}](k={},h={},w={})", self.index.name(), self.cfg.k, self.cfg.bandwidth, self.cfg.wscale)
    }

    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "CaseRouter::select: no candidates");
        let post = self.posteriors(task, candidates);
        let mut best = candidates[0];
        let mut best_v = f64::NEG_INFINITY;
        for &c in candidates {
            let (a, b) = post.get(c).copied().unwrap_or((self.cfg.a0, self.cfg.b0));
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
        let Some(e) = (self.embed)(task) else { return };
        self.n_agents = self.n_agents.max(agent + 1);
        let id = self.cases.len() as u64;
        self.index.add(id, &e);
        self.cases.push((agent, reward.clamp(0.0, 1.0) as f32));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{HnswIndex, HnswParams, normalize_in_place, synth};

    const DIM: usize = 16;

    /// Topic-clustered embeddings, id -> (topic, vector).
    struct Data {
        topics: Vec<usize>,
        embs: Vec<Vec<f32>>,
    }
    fn make(n: usize, n_topics: usize, noise: f32, seed: u64) -> Data {
        let mut rng = synth::Rng::new(seed);
        let cents: Vec<Vec<f32>> = (0..n_topics)
            .map(|_| {
                let mut c: Vec<f32> = (0..DIM).map(|_| rng.normal()).collect();
                normalize_in_place(&mut c);
                c
            })
            .collect();
        let mut topics = vec![];
        let mut embs = vec![];
        for _ in 0..n {
            let t = rng.below(n_topics);
            let mut v: Vec<f32> = cents[t].iter().map(|x| x + noise / (DIM as f32).sqrt() * rng.normal()).collect();
            normalize_in_place(&mut v);
            topics.push(t);
            embs.push(v);
        }
        Data { topics, embs }
    }
    fn embedder(d: &Arc<Data>) -> Embedder {
        let d = d.clone();
        Arc::new(move |t: &Task| d.embs.get(t.id as usize).cloned())
    }
    fn task(id: u64) -> Task {
        Task { id, task_type: None, features: vec![] }
    }

    fn play(mut r: CaseRouter, d: &Data, n: usize) -> f64 {
        // best agent for topic t is t % 3; others bad.
        let mut rng = Rng::new(21);
        let mut good = 0;
        for i in 0..n {
            let t = task(i as u64);
            let a = r.select(&t, &[0, 1, 2], &mut rng);
            let best = d.topics[i] % 3;
            let p = if a == best { 0.9 } else { 0.15 };
            r.update(&t, a, if rng.bernoulli(p) { 1.0 } else { 0.0 });
            if i >= n - 200 && a == best {
                good += 1;
            }
        }
        good as f64 / 200.0
    }

    #[test]
    fn learns_topic_to_agent_map() {
        let d = Arc::new(make(1200, 6, 0.5, 5));
        let r = CaseRouter::new(3, DIM, CaseConfig::default(), embedder(&d));
        let acc = play(r, &d, 1200);
        assert!(acc > 0.8, "late accuracy {acc}");
    }

    #[test]
    fn hnsw_backend_matches_flat_quality() {
        let d = Arc::new(make(800, 5, 0.5, 8));
        let r = CaseRouter::with_index(
            3,
            Box::new(HnswIndex::new(DIM, HnswParams { max_elements: 4096, ..Default::default() })),
            CaseConfig::default(),
            embedder(&d),
        );
        let acc = play(r, &d, 800);
        assert!(acc > 0.75, "hnsw late accuracy {acc}");
    }

    #[test]
    fn local_mass_grows_near_cases_and_no_embedding_is_safe() {
        let d = Arc::new(make(100, 3, 0.3, 2));
        let mut r = CaseRouter::new(3, DIM, CaseConfig::default(), embedder(&d));
        assert_eq!(r.local_mass(&task(0)), 0.0);
        for i in 0..50 {
            r.update(&task(i), 0, 1.0);
        }
        assert!(r.local_mass(&task(60)) > 1.0);
        // task without embedding: select works (prior-only), update ignored
        let mut rng = Rng::new(1);
        let none = task(10_000);
        assert!(r.select(&none, &[0, 1, 2], &mut rng) < 3);
        r.update(&none, 1, 1.0);
        assert_eq!(r.n_cases(), 50);
    }

    #[test]
    fn keyword_pseudo_counts_bias_unseen_regions() {
        let d = Arc::new(make(10, 2, 0.3, 4));
        let map = [("kw".to_string(), 2usize)].into_iter().collect();
        let cfg = CaseConfig { kw_pseudo: 20.0, ..Default::default() };
        let r = CaseRouter::new(3, DIM, cfg, embedder(&d)).with_keyword_prior(KeywordPrior::new(map, 3));
        let t = Task { id: 0, task_type: None, features: vec!["kw".into()] };
        let p = r.posteriors(&t, &[0, 1, 2]);
        assert!(p[2].0 > p[0].0 + 15.0 && p[0] == (1.0, 1.0));
    }
}
