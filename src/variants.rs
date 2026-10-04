//! Research-pass variants of the leader router and of the mesh co-activation rule.
//!
//! * [`PgActorRouter`] — policy-gradient three-factor actor (inspired by
//!   `ferromorphic::reinforce::Actor`, re-implemented natively because that actor is
//!   tabular over discrete states while our "state" is a bag of features).
//! * [`RStdpRouter`] — reward-modulated STDP with a cross-task eligibility trace,
//!   driven by `oxicuda_snn::plasticity::r_stdp::r_stdp_step` (crate code, not a copy).
//! * [`TrustEdges`] — hatcher-style asymmetric trust + plasticity edge learning
//!   (equations from `hatcher-neural` trust.rs / equations.rs, re-implemented natively
//!   on plain `Vec`s), with `petgraph` bridge / articulation-point fragility analysis.
//!
//! ## Swapping the mesh co-activation rule for `TrustEdges`
//! `NeuralMesh` (src/mesh.rs) has exactly two touch points on its edge matrix:
//! 1. `NeuralMesh::feedback` — the block guarded by `team.len() >= 2` that decays
//!    `self.edges` and adds `edge_eta * (reward - baseline)` to every team pair.
//!    Replace that whole block with `trust_edges.record_team(team, reward)`.
//! 2. `NeuralMesh::route_k` — helper ranking `rest.sort_by(|&a,&b| self.edge(leader,b)
//!    .total_cmp(&self.edge(leader,a)))`, i.e. `NeuralMesh::edge`. Make `edge(a,b)`
//!    return `trust_edges.edge(a,b)` (or call `trust_edges.top_partners(leader, k-1, &rest)`).
//! `ensure_agents(n)` should forward to `TrustEdges::ensure_agents(n)`.
//! Note TrustEdges weights live in `[0,1]` (not `[-1,1]`), unused edges decay by
//! `psi*cost` per call, and trust only grows for pairs whose success rate exceeds
//! `mu / (lambda + mu)` (= 0.714 with defaults).

use std::collections::HashMap;

use oxicuda_snn::plasticity::r_stdp::{r_stdp_step, RStdpConfig as OxRStdpConfig, RStdpState};
use oxicuda_snn::plasticity::stdp::{StdpConfig, StdpTraces};
use petgraph::graph::{NodeIndex, UnGraph};
use serde::{Deserialize, Serialize};

use crate::router::{Rng, Router, Task};

/// All features of a task plus `type:<task_type>` if present.
fn task_keys(task: &Task) -> Vec<String> {
    let mut keys: Vec<String> = task.features.clone();
    if let Some(t) = &task.task_type {
        keys.push(format!("type:{t}"));
    }
    keys
}

/// Numerically stable softmax over `logits / temperature`.
fn softmax(logits: &[f64], temperature: f64) -> Vec<f64> {
    let t = temperature.max(1e-9);
    let top = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| ((l - top) / t).exp()).collect();
    let z: f64 = exps.iter().sum();
    exps.iter().map(|e| e / z).collect()
}

/// Sample an index from a probability vector.
fn sample(p: &[f64], rng: &mut Rng) -> usize {
    let u = rng.f64();
    let mut acc = 0.0;
    for (k, &pk) in p.iter().enumerate() {
        acc += pk;
        if u < acc {
            return k;
        }
    }
    p.len() - 1
}

// =====================================================================================
// 1. Policy-gradient actor
// =====================================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PgActorConfig {
    /// Learning rate on the preferences.
    pub eta: f64,
    /// EMA rate of the per-feature reward baseline.
    pub baseline_alpha: f64,
    /// Initial baseline for unseen features.
    pub baseline_init: f64,
    /// |theta| clip so the policy can still move after skill drift (0 = no clip).
    pub theta_clip: f64,
    /// Uniform exploration mixed into the sampling distribution. Needed after drift:
    /// a saturated softmax has a vanishing (1 - pi) gradient on the stale favourite,
    /// so the new best agent is only found by exploration.
    #[serde(default)]
    pub epsilon: f64,
    /// Multiplicative decay of a touched preference row per update. Bounds the
    /// softmax gap so a stale favourite's (1 - pi) gradient does not vanish.
    #[serde(default)]
    pub decay: f64,
}

impl Default for PgActorConfig {
    fn default() -> Self {
        Self { eta: 0.5, baseline_alpha: 0.1, baseline_init: 0.5, theta_clip: 3.0, epsilon: 0.05, decay: 0.01 }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct PgLast {
    task_id: u64,
    candidates: Vec<usize>,
    probs: Vec<f64>,
}

/// Softmax actor: logit(a) = sum over task keys of theta[key][a]; softmax over the
/// candidate set only; REINFORCE three-factor update with a per-feature EMA baseline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PgActorRouter {
    pub cfg: PgActorConfig,
    n_agents: usize,
    theta: HashMap<String, Vec<f64>>,
    baseline: HashMap<String, f64>,
    last: Option<PgLast>,
}

impl PgActorRouter {
    pub fn new(n_agents: usize, cfg: PgActorConfig) -> Self {
        Self { cfg, n_agents, theta: HashMap::new(), baseline: HashMap::new(), last: None }
    }

    fn theta_at(&self, key: &str, a: usize) -> f64 {
        self.theta.get(key).and_then(|r| r.get(a)).copied().unwrap_or(0.0)
    }

    /// Logits for every agent `0..n_agents`.
    pub fn scores(&self, task: &Task) -> Vec<f64> {
        let keys = task_keys(task);
        (0..self.n_agents).map(|a| keys.iter().map(|k| self.theta_at(k, a)).sum()).collect()
    }

    /// Policy over `candidates` (same order).
    pub fn policy(&self, task: &Task, candidates: &[usize]) -> Vec<f64> {
        let keys = task_keys(task);
        let logits: Vec<f64> =
            candidates.iter().map(|&a| keys.iter().map(|k| self.theta_at(k, a)).sum()).collect();
        softmax(&logits, 1.0)
    }
}

impl Router for PgActorRouter {
    fn name(&self) -> String {
        "pg-actor".into()
    }

    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        if let Some(&m) = candidates.iter().max() {
            self.n_agents = self.n_agents.max(m + 1);
        }
        let probs = self.policy(task, candidates);
        let k = if rng.f64() < self.cfg.epsilon {
            rng.below(candidates.len())
        } else {
            sample(&probs, rng)
        };
        self.last = Some(PgLast { task_id: task.id, candidates: candidates.to_vec(), probs });
        candidates[k]
    }

    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        self.n_agents = self.n_agents.max(agent + 1);
        // Use the stored candidate set / probabilities from `select` when they match;
        // otherwise recompute over the stored set (or all agents).
        let (cands, probs) = match self.last.take() {
            Some(l) if l.task_id == task.id && l.candidates.contains(&agent) => (l.candidates, l.probs),
            Some(l) if l.candidates.contains(&agent) => {
                let p = self.policy(task, &l.candidates);
                (l.candidates, p)
            }
            _ => {
                let c: Vec<usize> = (0..self.n_agents).collect();
                let p = self.policy(task, &c);
                (c, p)
            }
        };
        let n = self.n_agents;
        let decay = self.cfg.decay;
        let (eta, alpha, b0, clip) =
            (self.cfg.eta, self.cfg.baseline_alpha, self.cfg.baseline_init, self.cfg.theta_clip);
        for key in task_keys(task) {
            let b = self.baseline.entry(key.clone()).or_insert(b0);
            let delta = reward - *b;
            *b += alpha * (reward - *b);
            let row = self.theta.entry(key).or_default();
            if row.len() < n {
                row.resize(n, 0.0);
            }
            if decay > 0.0 {
                for v in row.iter_mut() {
                    *v *= 1.0 - decay;
                }
            }
            for (&c, &p) in cands.iter().zip(&probs) {
                let g = if c == agent { 1.0 - p } else { -p };
                let mut v = row[c] + eta * delta * g;
                if clip > 0.0 {
                    v = v.clamp(-clip, clip);
                }
                row[c] = v;
            }
        }
    }
}

// =====================================================================================
// 2. Reward-modulated STDP router
// =====================================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RStdpRouterConfig {
    /// Hashed feature rows (pre-synaptic neurons).
    pub n_pre: usize,
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    /// Eligibility time constant in protocol steps (3 steps per task).
    pub tau_e: f32,
    pub eta: f32,
    pub w_init: f32,
    pub temperature: f64,
    pub epsilon: f64,
    pub baseline_alpha: f64,
    pub baseline_init: f64,
}

impl Default for RStdpRouterConfig {
    fn default() -> Self {
        Self {
            n_pre: 64,
            a_plus: 1.0,
            a_minus: 1.0,
            tau_plus: 2.0,
            tau_minus: 2.0,
            tau_e: 2.0,
            eta: 1.0,
            w_init: 0.5,
            temperature: 0.1,
            epsilon: 0.02,
            baseline_alpha: 0.1,
            baseline_init: 0.5,
        }
    }
}

impl RStdpRouterConfig {
    fn ox(&self) -> OxRStdpConfig {
        OxRStdpConfig {
            stdp: StdpConfig {
                a_plus: self.a_plus,
                a_minus: self.a_minus,
                tau_plus: self.tau_plus,
                tau_minus: self.tau_minus,
                dt: 1.0,
                w_min: 0.0,
                w_max: 1.0,
            },
            tau_e: self.tau_e,
            eta: self.eta,
        }
    }
}

/// Features hashed (FNV-1a) to `n_pre` rows; agents are the `n_post` columns.
/// Per task, a 3-step protocol is run through `oxicuda_snn::r_stdp_step`:
///   1. pre = feature rows, post = 0, reward 0
///   2. pre = 0, post = onehot(chosen agent), reward 0   (pre-before-post => LTP into e)
///   3. pre = 0, post = 0, reward = reward - baseline    (weights move by eta * r * e)
/// The eligibility matrix persists across tasks (decaying with `tau_e`); the fast
/// spike traces (x_pre / y_post) are reset at the start of every task so spikes of
/// the previous task do not cause spurious LTD.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RStdpRouter {
    pub cfg: RStdpRouterConfig,
    n_post: usize,
    /// Row-major `[n_pre x n_post]`.
    weights: Vec<f32>,
    eligibility: Vec<f32>,
    baseline: Vec<f64>,
}

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

impl RStdpRouter {
    pub fn new(n_agents: usize, cfg: RStdpRouterConfig) -> Self {
        let n_pre = cfg.n_pre.max(1);
        let n_post = n_agents.max(1);
        Self {
            weights: vec![cfg.w_init; n_pre * n_post],
            eligibility: vec![0.0; n_pre * n_post],
            baseline: vec![cfg.baseline_init; n_pre],
            n_post,
            cfg,
        }
    }

    fn n_pre(&self) -> usize {
        self.baseline.len()
    }

    /// Distinct hashed rows active for this task.
    fn rows(&self, task: &Task) -> Vec<usize> {
        let mut r: Vec<usize> =
            task_keys(task).iter().map(|k| (fnv1a(k) % self.n_pre() as u64) as usize).collect();
        r.sort_unstable();
        r.dedup();
        r
    }

    /// Grow the agent dimension (re-layout the row-major matrices).
    pub fn ensure_agents(&mut self, n: usize) {
        if n <= self.n_post {
            return;
        }
        let (np, old) = (self.n_pre(), self.n_post);
        let mut w = vec![self.cfg.w_init; np * n];
        let mut e = vec![0.0; np * n];
        for i in 0..np {
            w[i * n..i * n + old].copy_from_slice(&self.weights[i * old..(i + 1) * old]);
            e[i * n..i * n + old].copy_from_slice(&self.eligibility[i * old..(i + 1) * old]);
        }
        self.weights = w;
        self.eligibility = e;
        self.n_post = n;
    }

    /// Weight-row sums over the task's active rows, one per agent.
    pub fn scores(&self, task: &Task) -> Vec<f64> {
        let rows = self.rows(task);
        (0..self.n_post)
            .map(|a| rows.iter().map(|&i| self.weights[i * self.n_post + a] as f64).sum())
            .collect()
    }

    pub fn weight(&self, row: usize, agent: usize) -> f32 {
        self.weights[row * self.n_post + agent]
    }

    fn step(&mut self, st: &mut RStdpState, pre: &[f32], post: &[f32], reward: f32, cfg: &OxRStdpConfig) {
        let (np, nq) = (self.n_pre(), self.n_post);
        r_stdp_step(&mut self.weights, st, pre, post, np, nq, reward, cfg)
            .expect("r_stdp_step: shapes/config validated by construction");
    }
}

impl Router for RStdpRouter {
    fn name(&self) -> String {
        "r-stdp".into()
    }

    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        if let Some(&m) = candidates.iter().max() {
            self.ensure_agents(m + 1);
        }
        if rng.f64() < self.cfg.epsilon {
            return candidates[rng.below(candidates.len())];
        }
        let s = self.scores(task);
        let logits: Vec<f64> = candidates.iter().map(|&a| s[a]).collect();
        let p = softmax(&logits, self.cfg.temperature);
        candidates[sample(&p, rng)]
    }

    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        self.ensure_agents(agent + 1);
        let (np, nq) = (self.n_pre(), self.n_post);
        let rows = self.rows(task);
        let b = if rows.is_empty() {
            self.cfg.baseline_init
        } else {
            rows.iter().map(|&i| self.baseline[i]).sum::<f64>() / rows.len() as f64
        };
        let signal = (reward - b) as f32;
        for &i in &rows {
            self.baseline[i] += self.cfg.baseline_alpha * (reward - self.baseline[i]);
        }

        let cfg = self.cfg.ox();
        let mut st = RStdpState {
            eligibility: std::mem::take(&mut self.eligibility),
            stdp_traces: StdpTraces::new(np, nq), // fresh spike traces per task
        };
        let mut pre = vec![0.0f32; np];
        for &i in &rows {
            pre[i] = 1.0;
        }
        let mut post = vec![0.0f32; nq];
        let zeros_pre = vec![0.0f32; np];
        let zeros_post = vec![0.0f32; nq];
        self.step(&mut st, &pre, &zeros_post, 0.0, &cfg);
        post[agent] = 1.0;
        self.step(&mut st, &zeros_pre, &post, 0.0, &cfg);
        self.step(&mut st, &zeros_pre, &zeros_post, signal, &cfg);
        self.eligibility = st.eligibility;
    }
}

// =====================================================================================
// 3. Hatcher-style trust edges
// =====================================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrustEdgesConfig {
    /// Trust gain per success (`λ`).
    pub lambda: f64,
    /// Trust loss per failure (`μ`, > λ: failures hurt more).
    pub mu: f64,
    /// Plasticity gain (`φ`).
    pub phi: f64,
    /// Cost penalty rate (`ψ`).
    pub psi: f64,
    /// Normalised per-link cost / latency.
    pub cost: f64,
    pub initial_trust: f64,
    pub initial_weight: f64,
}

impl Default for TrustEdgesConfig {
    fn default() -> Self {
        // lambda/mu/phi/psi = hatcher MeshCoefficients::default(); cost = INITIAL_LATENCY.
        Self { lambda: 0.10, mu: 0.25, phi: 0.15, psi: 0.05, cost: 0.20, initial_trust: 0.5, initial_weight: 0.10 }
    }
}

/// Symmetric per-pair trust `T` and edge weight `w`, both in `[0,1]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrustEdges {
    pub cfg: TrustEdgesConfig,
    n: usize,
    trust: Vec<Vec<f64>>,
    weight: Vec<Vec<f64>>,
}

impl TrustEdges {
    pub fn new(n_agents: usize, cfg: TrustEdgesConfig) -> Self {
        let mut t = Self { cfg, n: 0, trust: Vec::new(), weight: Vec::new() };
        t.ensure_agents(n_agents);
        t
    }

    pub fn n_agents(&self) -> usize {
        self.n
    }

    pub fn ensure_agents(&mut self, n: usize) {
        if n <= self.n {
            return;
        }
        let (t0, w0) = (self.cfg.initial_trust, self.cfg.initial_weight);
        for r in self.trust.iter_mut() {
            r.resize(n, t0);
        }
        for r in self.weight.iter_mut() {
            r.resize(n, w0);
        }
        self.trust.resize(n, vec![t0; n]);
        self.weight.resize(n, vec![w0; n]);
        self.n = n;
    }

    /// One team outcome. Every pair in the team: successes = reward, failures = 1 - reward,
    /// `T += λ·succ − μ·fail`, then `w += φ·T·(1−w) − ψ·cost`. Every other pair:
    /// `w −= ψ·cost`. Teams of size < 2 carry no pair traffic but still decay edges.
    pub fn record_team(&mut self, team: &[usize], reward: f64) {
        if let Some(&m) = team.iter().max() {
            self.ensure_agents(m + 1);
        }
        let r = reward.clamp(0.0, 1.0);
        let c = &self.cfg;
        let mut member = vec![false; self.n];
        for &a in team {
            member[a] = true;
        }
        for a in 0..self.n {
            for b in (a + 1)..self.n {
                let w = self.weight[a][b];
                let nw = if member[a] && member[b] {
                    let t = (self.trust[a][b] + c.lambda * r - c.mu * (1.0 - r)).clamp(0.0, 1.0);
                    self.trust[a][b] = t;
                    self.trust[b][a] = t;
                    w + c.phi * t * (1.0 - w) - c.psi * c.cost
                } else {
                    w - c.psi * c.cost
                }
                .clamp(0.0, 1.0);
                self.weight[a][b] = nw;
                self.weight[b][a] = nw;
            }
        }
    }

    /// Edge weight (0 for a == b or unknown agents).
    pub fn edge(&self, a: usize, b: usize) -> f64 {
        if a == b {
            return 0.0;
        }
        self.weight.get(a).and_then(|r| r.get(b)).copied().unwrap_or(0.0)
    }

    pub fn trust(&self, a: usize, b: usize) -> f64 {
        if a == b {
            return 0.0;
        }
        self.trust.get(a).and_then(|r| r.get(b)).copied().unwrap_or(self.cfg.initial_trust)
    }

    /// Up to `k` candidates (excluding `a`) by descending edge weight; ties by index.
    pub fn top_partners(&self, a: usize, k: usize, candidates: &[usize]) -> Vec<usize> {
        let mut c: Vec<usize> = candidates.iter().copied().filter(|&b| b != a).collect();
        c.sort_unstable();
        c.dedup();
        c.sort_by(|&x, &y| self.edge(a, y).total_cmp(&self.edge(a, x)));
        c.truncate(k);
        c
    }

    /// Build the undirected graph of edges with `w >= threshold`.
    pub fn graph(&self, threshold: f64) -> UnGraph<(), f64> {
        let mut g = UnGraph::<(), f64>::with_capacity(self.n, 0);
        for _ in 0..self.n {
            g.add_node(());
        }
        for a in 0..self.n {
            for b in (a + 1)..self.n {
                let w = self.weight[a][b];
                if w >= threshold {
                    g.add_edge(NodeIndex::new(a), NodeIndex::new(b), w);
                }
            }
        }
        g
    }

    /// `(bridges, articulation_points)` of the thresholded edge graph.
    pub fn fragility(&self, threshold: f64) -> (usize, usize) {
        let g = self.graph(threshold);
        let bridges = petgraph::algo::bridges(&g).count();
        let aps = petgraph::algo::articulation_points::articulation_points(&g).len();
        (bridges, aps)
    }

    /// Test/setup helper: set a symmetric edge weight directly.
    pub fn set_edge(&mut self, a: usize, b: usize, w: f64) {
        self.ensure_agents(a.max(b) + 1);
        self.weight[a][b] = w.clamp(0.0, 1.0);
        self.weight[b][a] = w.clamp(0.0, 1.0);
    }
}

// =====================================================================================
// Tests
// =====================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn task(i: u64, ty: &str) -> Task {
        Task { id: i, task_type: Some(ty.into()), features: vec![format!("kw-{ty}"), "common".into()] }
    }

    /// Run `n` tasks with success prob `p(t, a)`; return chosen-best fraction over last `tail`.
    fn run<R: Router>(
        r: &mut R,
        rng: &mut Rng,
        start: u64,
        n: u64,
        tail: u64,
        best: &dyn Fn(&str) -> usize,
        p: &dyn Fn(&str, usize) -> f64,
    ) -> f64 {
        let cands: Vec<usize> = (0..5).collect();
        let types = ["a", "b", "c"];
        let mut hits = 0;
        for i in start..start + n {
            let ty = types[(i % 3) as usize];
            let t = task(i, ty);
            let a = r.select(&t, &cands, rng);
            let rew = if rng.bernoulli(p(ty, a)) { 1.0 } else { 0.0 };
            r.update(&t, a, rew);
            if i >= start + n - tail && a == best(ty) {
                hits += 1;
            }
        }
        hits as f64 / tail as f64
    }

    fn best0(ty: &str) -> usize {
        match ty {
            "a" => 1,
            "b" => 3,
            _ => 4,
        }
    }
    fn p0(ty: &str, a: usize) -> f64 {
        if a == best0(ty) { 0.9 } else { 0.2 }
    }
    fn best1(ty: &str) -> usize {
        match ty {
            "a" => 2,
            "b" => 0,
            _ => 4,
        }
    }
    fn p1(ty: &str, a: usize) -> f64 {
        if a == best1(ty) { 0.9 } else { 0.2 }
    }

    #[test]
    fn pg_actor_converges_to_best_agent() {
        let mut r = PgActorRouter::new(5, PgActorConfig::default());
        let mut rng = Rng::new(7);
        let frac = run(&mut r, &mut rng, 0, 3000, 600, &best0, &p0);
        assert!(frac > 0.8, "pg best-agent fraction {frac}");
    }

    #[test]
    fn pg_actor_adapts_after_drift() {
        let mut r = PgActorRouter::new(5, PgActorConfig::default());
        let mut rng = Rng::new(11);
        let pre = run(&mut r, &mut rng, 0, 3000, 600, &best0, &p0);
        assert!(pre > 0.8, "pre-drift {pre}");
        let post = run(&mut r, &mut rng, 3000, 4000, 600, &best1, &p1);
        assert!(post > 0.7, "post-drift best fraction {post}");
    }

    #[test]
    fn pg_actor_policy_only_over_candidates_and_serde() {
        let mut r = PgActorRouter::new(5, PgActorConfig::default());
        let mut rng = Rng::new(1);
        let t = task(0, "a");
        for _ in 0..50 {
            let a = r.select(&t, &[2, 4], &mut rng);
            assert!(a == 2 || a == 4);
            r.update(&t, a, 1.0);
        }
        let s = serde_json::to_string(&r).unwrap();
        let r2: PgActorRouter = serde_json::from_str(&s).unwrap();
        for (x, y) in r.scores(&t).iter().zip(r2.scores(&t)) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
    }

    #[test]
    fn rstdp_converges_to_best_agent() {
        let mut r = RStdpRouter::new(5, RStdpRouterConfig::default());
        let mut rng = Rng::new(3);
        let frac = run(&mut r, &mut rng, 0, 3000, 600, &best0, &p0);
        assert!(frac > 0.75, "r-stdp best-agent fraction {frac}");
    }

    #[test]
    fn rstdp_reward_sign_moves_weights() {
        let mut r = RStdpRouter::new(3, RStdpRouterConfig::default());
        let t = task(0, "a");
        let s0 = r.scores(&t);
        r.update(&t, 1, 1.0); // above baseline 0.5 -> LTP for agent 1
        let s1 = r.scores(&t);
        assert!(s1[1] > s0[1], "{s0:?} -> {s1:?}");
        assert_eq!(s1[0], s0[0]);
        r.update(&task(1, "a"), 2, 0.0); // below baseline -> LTD for agent 2
        assert!(r.scores(&t)[2] < s0[2]);
        let js = serde_json::to_string(&r).unwrap();
        let r2: RStdpRouter = serde_json::from_str(&js).unwrap();
        assert_eq!(r.scores(&t), r2.scores(&t));
    }

    #[test]
    fn trust_is_asymmetric() {
        let mut te = TrustEdges::new(3, TrustEdgesConfig::default());
        te.record_team(&[0, 1], 1.0);
        let up = te.trust(0, 1) - 0.5;
        let mut te2 = TrustEdges::new(3, TrustEdgesConfig::default());
        te2.record_team(&[0, 1], 0.0);
        let down = 0.5 - te2.trust(0, 1);
        assert!((up - 0.10).abs() < 1e-12 && (down - 0.25).abs() < 1e-12);
        assert!(down > up);
        // Alternating success/failure loses trust net.
        let mut te3 = TrustEdges::new(2, TrustEdgesConfig::default());
        for i in 0..10 {
            te3.record_team(&[0, 1], (i % 2) as f64);
        }
        assert!(te3.trust(0, 1) < 0.5);
    }

    #[test]
    fn trust_edges_grow_for_reliable_pair_and_decay_unused() {
        let mut te = TrustEdges::new(4, TrustEdgesConfig::default());
        for _ in 0..50 {
            te.record_team(&[0, 1], 1.0);
        }
        assert!(te.edge(0, 1) > 0.5, "{}", te.edge(0, 1));
        assert_eq!(te.edge(2, 3), 0.0);
        assert_eq!(te.edge(0, 1), te.edge(1, 0));
        assert_eq!(te.top_partners(0, 2, &[0, 1, 2, 3]), vec![1, 2]);
        let mut bad = TrustEdges::new(2, TrustEdgesConfig::default());
        for _ in 0..50 {
            bad.record_team(&[0, 1], 0.0);
        }
        assert_eq!(bad.edge(0, 1), 0.0);
    }

    #[test]
    fn fragility_on_hand_built_graph() {
        // Triangle 0-1-2, bridge 2-3, path 3-4; node 5 isolated; weak edge 4-5 below threshold.
        let mut te = TrustEdges::new(6, TrustEdgesConfig { initial_weight: 0.0, ..Default::default() });
        for (a, b) in [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4)] {
            te.set_edge(a, b, 0.8);
        }
        te.set_edge(4, 5, 0.2);
        // Bridges: 2-3, 3-4. Articulation points: 2, 3.
        assert_eq!(te.fragility(0.5), (2, 2));
        // Lower threshold includes 4-5: bridges 2-3,3-4,4-5; APs 2,3,4.
        assert_eq!(te.fragility(0.1), (3, 3));
    }
}
