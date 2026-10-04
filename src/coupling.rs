//! Coupled brain: router <-> mesh <-> shared memory in one closed learning loop.
//!
//! Opt-in wrapper (`CoupledBrain`) that composes only existing public APIs:
//! [`Warm`]`<`[`HebbianRouter`]`>` (keyword warm start + Hebbian learner), a
//! re-implementation of the mesh's per-node health EMA / relative bench threshold
//! (same rule as `mesh::NeuralMesh`, so the coupling hooks can see and steer it),
//! and a [`FlatIndex`] shared experience memory. Nothing else in the crate changes;
//! with every option off a `CoupledBrain` makes exactly the choices of the bare
//! `Warm<HebbianRouter>` it wraps (tested).
//!
//! The three couplings studied (each independently switchable):
//!
//! 1. **Router <-> mesh signal exchange** (`health`, `shield`, `probe_gate`)
//!    * mesh -> router: benched agents are removed from the router's candidates
//!      (the "health prior"), and with `shield` a failure the mesh classifies as an
//!      *outage* (the agent's health is below the bench threshold, or it is on a
//!      probation probe) is NOT fed to the router: the router learns skills, the
//!      mesh learns availability, so an outage does not erase skill weights.
//!    * router -> mesh: with `probe_gate` a benched agent is only put on probation
//!      when the router currently ranks it first for the task at hand (its health
//!      evidence is needed exactly where the router would use it; elsewhere the
//!      probe would be a wasted, risky task).
//! 2. **Shared experience replay** (`replay_n > 0`): every outcome is stored as
//!    (hashed feature embedding, task, agent, reward). Before routing a task, the
//!    `replay_n` most similar (cosine >= `replay_min_sim`; ties -> most recent) past
//!    outcomes are re-applied to the inner learner (consolidation). Replays do not
//!    count as warm-start evidence.
//! 3. **Reward shaping** ([`RewardShaper`]): turns the weak hook signal (tool
//!    succeeded or not, ~always 1) plus latency into a graded reward. Latency is
//!    normalised per feature (running log-latency mean/sd), so task-type speed does
//!    not leak into the reward. Delayed signals (downstream reuse) are delivered
//!    with [`CoupledBrain::feedback_delayed`].
//!
//! The health tracker and the router can be fed DIFFERENT signals
//! ([`CoupledBrain::feedback_split`]): the raw hook bit is exactly outage
//! information (health), the shaped reward is skill information (router).
//!
//! Memory accounting helpers ([`StateBytes`]) estimate the resident bytes of the
//! combined state, and of the proposed compact representations (f32 / sparse rows).

use std::collections::HashMap;

use crate::hebbian::{HebbianConfig, HebbianRouter};
use crate::memory::{dot, normalized, FlatIndex, MemoryIndex};
use crate::prior::{Evidence, GuardConfig, Guarded, Warm};
use crate::router::{Rng, Router, Task};

/// The "hebbian-robust" configuration used across the crate's labs.
pub fn heb_robust_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

/// Warm-start evidence kind (mirror of [`Evidence`] without the `Inner` variant,
/// so it can be `Copy`/`Debug` in configs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarmEvidence {
    FeatureSet,
    MeanFeature,
    Global,
    SetFailures,
}

impl WarmEvidence {
    fn to_evidence<R>(self) -> Evidence<R> {
        match self {
            WarmEvidence::FeatureSet => Evidence::FeatureSet,
            WarmEvidence::MeanFeature => Evidence::MeanFeature,
            WarmEvidence::Global => Evidence::Global,
            WarmEvidence::SetFailures => Evidence::SetFailures,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CouplingConfig {
    /// Inner learner config.
    pub hebbian: HebbianConfig,
    /// Keyword front-end: `None` = `Warm` (k / evidence below), `Some` = router-thinker's
    /// confidence-gated `Guarded` hand-over (composition test).
    pub guard: Option<GuardConfig>,
    /// Warm-start hand-over scale k (0 = never follow the keyword map).
    pub warm_k: f64,
    pub warm_evidence: WarmEvidence,

    // ---- 1. router <-> mesh ------------------------------------------------------------
    /// Track per-agent health and bench agents (mesh rule).
    pub health: bool,
    pub health_alpha: f64,
    /// Relative bench threshold (fraction of the healthiest candidate's health).
    pub health_threshold: f64,
    /// Routes on the bench before a probation probe is due.
    pub retry_every: u64,
    /// Do not feed outage failures to the router.
    pub shield: bool,
    /// Only probe a benched agent when the router ranks it first for this task.
    pub probe_gate: bool,

    // ---- 2. shared experience replay ----------------------------------------------------
    /// Past outcomes replayed per route (0 = off).
    pub replay_n: usize,
    /// Minimum cosine similarity of a replayed outcome.
    pub replay_min_sim: f64,
    /// Embedding dimension of the hashed feature vectors.
    pub replay_dim: usize,
    /// FIFO capacity of the experience memory (0 = unbounded).
    pub replay_cap: usize,
}

impl Default for CouplingConfig {
    /// Every coupling OFF: behaves exactly like `Warm<HebbianRouter>` (k=1, MeanFeature).
    fn default() -> Self {
        CouplingConfig {
            hebbian: heb_robust_cfg(),
            guard: None,
            warm_k: 1.0,
            warm_evidence: WarmEvidence::MeanFeature,
            health: false,
            health_alpha: 0.2,
            health_threshold: 0.3,
            retry_every: 20,
            shield: false,
            probe_gate: false,
            replay_n: 0,
            replay_min_sim: 0.99,
            replay_dim: 64,
            replay_cap: 0,
        }
    }
}

// =============================================================================================
// Hashed feature embedding

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Signed feature hashing of the task's (deduplicated) features and task type
/// into `dim` buckets. Not normalised (the index normalises).
pub fn embed(task: &Task, dim: usize) -> Vec<f32> {
    let dim = dim.max(1);
    let mut v = vec![0f32; dim];
    let mut seen: Vec<&str> = Vec::with_capacity(task.features.len() + 1);
    for f in task.features.iter().map(|s| s.as_str()).chain(task.task_type.as_deref()) {
        if seen.contains(&f) {
            continue;
        }
        seen.push(f);
        let h = fnv1a(f);
        let i = (h % dim as u64) as usize;
        v[i] += if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
    }
    v
}

// =============================================================================================
// Experience memory (FIFO window materialised as a FlatIndex, newest first)

#[derive(Clone, Debug)]
struct Experience {
    task: Task,
    agent: usize,
    reward: f64,
}

/// Shared experience memory. A `FlatIndex` holds the window newest-first (rebuilt
/// every `REBUILD_EVERY` additions) so that exact-score ties resolve to the most
/// recent outcome; outcomes added since the last rebuild are scanned exactly on
/// the side. Ids are global sequence numbers.
pub struct ExperienceMemory {
    dim: usize,
    cap: usize,
    index: FlatIndex,
    /// all retained items: seq -> (normalised vector, experience)
    items: HashMap<u64, (Vec<f32>, Experience)>,
    /// retained seqs in insertion order
    order: std::collections::VecDeque<u64>,
    /// seqs added after the last rebuild (not in `index`)
    pending: Vec<u64>,
    next: u64,
}

const REBUILD_EVERY: usize = 16;

impl ExperienceMemory {
    pub fn new(dim: usize, cap: usize) -> Self {
        ExperienceMemory {
            dim: dim.max(1),
            cap,
            index: FlatIndex::new(dim.max(1)),
            items: HashMap::new(),
            order: Default::default(),
            pending: Vec::new(),
            next: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn rebuild(&mut self) {
        let mut idx = FlatIndex::with_capacity(self.dim, self.order.len());
        for &s in self.order.iter().rev() {
            if let Some((v, _)) = self.items.get(&s) {
                idx.add(s, v);
            }
        }
        self.index = idx;
        self.pending.clear();
    }

    pub fn add(&mut self, task: &Task, agent: usize, reward: f64) {
        let v = normalized(&embed(task, self.dim));
        let s = self.next;
        self.next += 1;
        self.items.insert(s, (v, Experience { task: task.clone(), agent, reward }));
        self.order.push_back(s);
        self.pending.push(s);
        let mut evicted = false;
        if self.cap > 0 {
            while self.order.len() > self.cap {
                if let Some(old) = self.order.pop_front() {
                    self.items.remove(&old);
                    evicted = true;
                }
            }
        }
        if evicted || self.pending.len() >= REBUILD_EVERY {
            self.rebuild();
        }
    }

    /// Up to `n` most similar outcomes with cosine >= `min_sim` (ties: newest first).
    fn similar(&self, task: &Task, n: usize, min_sim: f64) -> Vec<Experience> {
        if n == 0 || self.items.is_empty() {
            return Vec::new();
        }
        let q = normalized(&embed(task, self.dim));
        // pending items are the newest: scan them exactly, newest first
        let mut cand: Vec<(f32, u64)> = Vec::with_capacity(n * 2);
        for &s in self.pending.iter().rev() {
            if let Some((v, _)) = self.items.get(&s) {
                cand.push((dot(&q, v), s));
            }
        }
        for (s, sc) in self.index.search(&q, n) {
            if self.items.contains_key(&s) {
                cand.push((sc, s));
            }
        }
        // score ties (up to float noise) -> newest (largest seq) first
        cand.sort_by(|a, b| {
            let (qa, qb) = ((a.0 as f64 * 1e5).round(), (b.0 as f64 * 1e5).round());
            qb.total_cmp(&qa).then(b.1.cmp(&a.1))
        });
        cand.into_iter()
            .filter(|(sc, _)| *sc as f64 >= min_sim - 1e-6)
            .take(n)
            .filter_map(|(_, s)| self.items.get(&s).map(|(_, e)| e.clone()))
            .collect()
    }

    /// Approximate resident bytes (vectors + ids in the index + retained copies +
    /// per-item task/meta table).
    pub fn bytes(&self) -> usize {
        let per_item_meta: usize = self
            .items
            .values()
            .map(|(_, e)| 8 + 8 + 8 + 24 + e.task.features.iter().map(|f| 24 + f.len()).sum::<usize>())
            .sum();
        self.index.bytes() + self.items.len() * (self.dim * 4 + 24 + 16) + per_item_meta
    }
}

// =============================================================================================
// Reward shaping

#[derive(Clone, Debug, PartialEq)]
pub struct ShapingConfig {
    /// Use latency to grade a successful (raw_ok) outcome.
    pub use_latency: bool,
    /// Logistic slope on the per-feature latency z-score.
    pub lat_slope: f64,
    /// EMA rate of the per-feature log-latency mean / variance.
    pub lat_alpha: f64,
    /// Observations of a feature before its latency stats are trusted (before that
    /// a successful outcome gets `neutral`).
    pub lat_warmup: u32,
    /// Reward of a raw success when latency is not (yet) informative.
    pub neutral: f64,
    /// When latency is off: send an immediate reward only on raw failure (the
    /// graded signal comes later via `feedback_delayed`).
    pub immediate_only_failures: bool,
    /// Remove a per-agent speed offset (two-way context x agent model of log-latency),
    /// so an agent that is merely fast/slow everywhere is not rewarded/punished.
    pub debias_agent: bool,
}

impl Default for ShapingConfig {
    fn default() -> Self {
        ShapingConfig {
            use_latency: true,
            lat_slope: 1.5,
            lat_alpha: 0.05,
            lat_warmup: 3,
            neutral: 0.5,
            immediate_only_failures: false,
            debias_agent: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct LatStat {
    n: u32,
    mean: f64,
    var: f64,
}

/// Turns weak hook observations (`raw_ok`, optional latency) into a graded reward.
///
/// Log-latency model: `ln lat = mu_context + beta_agent + noise (+ slower when bad)`.
/// `mu` is tracked per feature (EMA); with `debias_agent` the agent offset `beta` is
/// the running mean residual of that agent over ALL contexts (an agent's average
/// quality effect is absorbed too, but routing only needs within-context ranking).
#[derive(Clone, Debug)]
pub struct RewardShaper {
    pub cfg: ShapingConfig,
    stats: HashMap<String, LatStat>,
    /// per agent (n, mean residual)
    agent_off: Vec<(u32, f64)>,
}

impl RewardShaper {
    pub fn new(cfg: ShapingConfig) -> Self {
        RewardShaper { cfg, stats: HashMap::new(), agent_off: Vec::new() }
    }

    fn offset(&self, agent: usize) -> f64 {
        if !self.cfg.debias_agent {
            return 0.0;
        }
        match self.agent_off.get(agent) {
            Some(&(n, m)) if n >= self.cfg.lat_warmup => m,
            _ => 0.0,
        }
    }

    /// Mean over the task's warmed-up features of the context mean / sd, or None.
    fn context(&self, task: &Task) -> Option<(f64, f64)> {
        let (mut m, mut sd, mut n) = (0.0, 0.0, 0.0);
        for f in &task.features {
            if let Some(st) = self.stats.get(f)
                && st.n >= self.cfg.lat_warmup
            {
                m += st.mean;
                sd += st.var.max(1e-4).sqrt();
                n += 1.0;
            }
        }
        if n > 0.0 { Some((m / n, sd / n)) } else { None }
    }

    /// z-score of `ln(latency)` (minus the agent offset) against the task's context
    /// (None if no feature has enough observations). Does not update the statistics.
    pub fn latency_z(&self, task: &Task, agent: usize, latency: f64) -> Option<f64> {
        let x = latency.max(1e-9).ln() - self.offset(agent);
        self.context(task).map(|(m, sd)| (x - m) / sd)
    }

    fn observe_latency(&mut self, task: &Task, agent: usize, latency: f64) {
        let raw = latency.max(1e-9).ln();
        // agent offset update uses the pre-update context mean
        if self.cfg.debias_agent
            && let Some((m, _)) = self.context(task)
        {
            if self.agent_off.len() <= agent {
                self.agent_off.resize(agent + 1, (0, 0.0));
            }
            let e = &mut self.agent_off[agent];
            e.0 = e.0.saturating_add(1);
            let alpha = self.cfg.lat_alpha.max(1.0 / e.0 as f64);
            e.1 += alpha * ((raw - m) - e.1);
        }
        let x = raw - self.offset(agent);
        let a = self.cfg.lat_alpha;
        for f in &task.features {
            let st = self.stats.entry(f.clone()).or_default();
            st.n = st.n.saturating_add(1);
            if st.n == 1 {
                st.mean = x;
                st.var = 0.25;
            } else {
                let alpha = a.max(1.0 / st.n as f64);
                let d = x - st.mean;
                st.mean += alpha * d;
                st.var = (1.0 - alpha) * (st.var + alpha * d * d);
            }
        }
    }

    /// Immediate reward for one observation of `agent` on `task`, or `None` when
    /// nothing informative should be sent to the learner now.
    pub fn immediate(&mut self, task: &Task, agent: usize, raw_ok: bool, latency: Option<f64>) -> Option<f64> {
        if !raw_ok {
            return Some(0.0);
        }
        if !self.cfg.use_latency {
            return if self.cfg.immediate_only_failures { None } else { Some(1.0) };
        }
        let r = match latency {
            Some(l) => {
                let z = self.latency_z(task, agent, l);
                self.observe_latency(task, agent, l);
                match z {
                    Some(z) => 1.0 / (1.0 + (self.cfg.lat_slope * z).exp()),
                    None => self.cfg.neutral,
                }
            }
            None => self.cfg.neutral,
        };
        Some(r)
    }

    pub fn bytes(&self) -> usize {
        self.stats.keys().map(|k| k.len() + 24 + 4 + 16 + 16 + 32).sum::<usize>() + self.agent_off.len() * 16
    }
}

// =============================================================================================
// The coupled brain

/// Keyword front-end around the Hebbian learner.
pub enum Front {
    Warm(Warm<HebbianRouter>),
    Guarded(Guarded<HebbianRouter>),
}

impl Front {
    pub fn inner(&self) -> &HebbianRouter {
        match self {
            Front::Warm(w) => &w.inner,
            Front::Guarded(g) => &g.inner,
        }
    }
    pub fn inner_mut(&mut self) -> &mut HebbianRouter {
        match self {
            Front::Warm(w) => &mut w.inner,
            Front::Guarded(g) => &mut g.inner,
        }
    }
    pub fn select(&mut self, task: &Task, c: &[usize], rng: &mut Rng) -> usize {
        match self {
            Front::Warm(w) => w.select(task, c, rng),
            Front::Guarded(g) => g.select(task, c, rng),
        }
    }
    pub fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        match self {
            Front::Warm(w) => w.update(task, agent, reward),
            Front::Guarded(g) => g.update(task, agent, reward),
        }
    }
}

pub struct CoupledBrain {
    pub cfg: CouplingConfig,
    pub router: Front,
    n_agents: usize,
    health: Vec<f64>,
    benched_since: Vec<Option<u64>>,
    probation: Vec<bool>,
    routes: u64,
    memory: Option<ExperienceMemory>,
    /// Counters for diagnostics.
    pub shielded: u64,
    pub replays: u64,
    pub probes: u64,
    pub probes_deferred: u64,
}

impl CoupledBrain {
    pub fn new(n_agents: usize, keyword_map: HashMap<String, usize>, cfg: CouplingConfig) -> Self {
        let inner = HebbianRouter::new(n_agents, cfg.hebbian.clone());
        let router = match &cfg.guard {
            None => Front::Warm(Warm::new(inner, keyword_map, n_agents, cfg.warm_k).with_evidence(cfg.warm_evidence.to_evidence())),
            Some(g) => Front::Guarded(Guarded::new(inner, keyword_map, n_agents, g.clone())),
        };
        let memory = if cfg.replay_n > 0 { Some(ExperienceMemory::new(cfg.replay_dim, cfg.replay_cap)) } else { None };
        CoupledBrain {
            n_agents,
            health: vec![1.0; n_agents],
            benched_since: vec![None; n_agents],
            probation: vec![false; n_agents],
            routes: 0,
            memory,
            cfg,
            router,
            shielded: 0,
            replays: 0,
            probes: 0,
            probes_deferred: 0,
        }
    }

    fn ensure(&mut self, n: usize) {
        if n > self.n_agents {
            self.health.resize(n, 1.0);
            self.benched_since.resize(n, None);
            self.probation.resize(n, false);
            self.n_agents = n;
        }
    }

    pub fn health(&self, a: usize) -> f64 {
        self.health.get(a).copied().unwrap_or(1.0)
    }
    pub fn is_benched(&self, a: usize) -> bool {
        self.benched_since.get(a).is_some_and(|b| b.is_some())
    }
    pub fn memory_len(&self) -> usize {
        self.memory.as_ref().map_or(0, |m| m.len())
    }

    /// Router's top-scored candidate (inner learner's exploit scores).
    fn router_top(&self, task: &Task, cands: &[usize]) -> usize {
        let s = self.router.inner().scores(task);
        let mut best = cands[0];
        for &c in cands {
            if s.get(c).copied().unwrap_or(0.0) > s.get(best).copied().unwrap_or(0.0) {
                best = c;
            }
        }
        best
    }

    fn eligible(&mut self, task: &Task, cands: &[usize]) -> Vec<usize> {
        let now = self.routes;
        let best_h = cands.iter().map(|&c| self.health[c]).fold(0.0_f64, f64::max);
        let thr = self.cfg.health_threshold * best_h;
        let top = if self.cfg.probe_gate { Some(self.router_top(task, cands)) } else { None };
        let mut el = Vec::with_capacity(cands.len());
        for &c in cands {
            if self.health[c] >= thr {
                self.benched_since[c] = None;
                el.push(c);
                continue;
            }
            match self.benched_since[c] {
                None => self.benched_since[c] = Some(now),
                Some(since) if now.saturating_sub(since) >= self.cfg.retry_every => {
                    if top.is_some_and(|t| t != c) {
                        self.probes_deferred += 1;
                        continue; // keep benched: its evidence is not needed for this task
                    }
                    self.probation[c] = true;
                    self.health[c] = thr;
                    self.benched_since[c] = None;
                    self.probes += 1;
                    el.push(c);
                }
                Some(_) => {}
            }
        }
        if el.is_empty() {
            let best = cands.iter().map(|&c| self.health[c]).fold(f64::NEG_INFINITY, f64::max);
            el = cands.iter().copied().filter(|&c| self.health[c] >= best).collect();
        }
        el
    }

    /// Route one task (single agent). Replays similar past outcomes first, then
    /// applies the health gate, then asks the warm router.
    pub fn route(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "CoupledBrain::route: no candidates");
        if let Some(&mx) = candidates.iter().max() {
            self.ensure(mx + 1);
        }
        if let Some(mem) = &self.memory {
            let ex = mem.similar(task, self.cfg.replay_n, self.cfg.replay_min_sim);
            for e in ex {
                self.router.inner_mut().update(&e.task, e.agent, e.reward);
                self.replays += 1;
            }
        }
        let el = if self.cfg.health { self.eligible(task, candidates) } else { candidates.to_vec() };
        self.routes += 1;
        self.router.select(task, &el, rng)
    }

    /// Feed the health tracker `health_signal` and the router `router_reward`
    /// (None = no router update now). Returns whether the router update was shielded.
    pub fn feedback_split(&mut self, task: &Task, agent: usize, health_signal: f64, router_reward: Option<f64>) -> bool {
        self.ensure(agent + 1);
        let mut outage = false;
        if self.cfg.health {
            let h = health_signal.clamp(0.0, 1.0);
            self.health[agent] += self.cfg.health_alpha * (h - self.health[agent]);
            let best = self.health.iter().cloned().fold(0.0_f64, f64::max);
            let below = self.health[agent] < self.cfg.health_threshold * best;
            outage = h < 0.5 && (self.probation[agent] || below);
            self.probation[agent] = false;
        }
        if let Some(r) = router_reward {
            if self.cfg.shield && outage {
                self.shielded += 1;
                return true;
            }
            self.router.update(task, agent, r);
            if let Some(mem) = self.memory.as_mut() {
                mem.add(task, agent, r.clamp(0.0, 1.0));
            }
        }
        false
    }

    /// Ordinary feedback: same graded reward for health and router.
    pub fn feedback(&mut self, task: &Task, agent: usize, reward: f64) {
        self.feedback_split(task, agent, reward, Some(reward));
    }

    /// Delayed graded signal (e.g. downstream reuse) for an earlier routing.
    /// Goes to the router (and the experience memory) only.
    pub fn feedback_delayed(&mut self, task: &Task, agent: usize, reward: f64) {
        self.router.update(task, agent, reward);
        if let Some(mem) = self.memory.as_mut() {
            mem.add(task, agent, reward.clamp(0.0, 1.0));
        }
    }

    /// Byte breakdown of the combined state.
    pub fn state_bytes(&self) -> StateBytes {
        let rb = router_bytes(self.router.inner());
        StateBytes {
            router_f64: rb.dense_f64,
            router_f32: rb.dense_f32,
            router_sparse_f32: rb.sparse_f32,
            health: self.n_agents * (8 + 16 + 1),
            memory: self.memory.as_ref().map_or(0, |m| m.bytes()),
            n_rows: rb.n_rows,
            n_cells: rb.n_cells,
            n_touched: rb.n_touched,
        }
    }
}

impl Router for CoupledBrain {
    fn name(&self) -> String {
        format!(
            "coupled(health={},shield={},probe_gate={},replay={})",
            self.cfg.health, self.cfg.shield, self.cfg.probe_gate, self.cfg.replay_n
        )
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        self.route(task, candidates, rng)
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        self.feedback(task, agent, reward)
    }
}

// =============================================================================================
// Memory accounting

/// Combined-state byte estimate (heap payload + main side tables; BTreeMap/HashMap
/// node overhead approximated at 32 B/entry).
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct StateBytes {
    pub router_f64: usize,
    pub router_f32: usize,
    pub router_sparse_f32: usize,
    pub health: usize,
    pub memory: usize,
    pub n_rows: usize,
    pub n_cells: usize,
    pub n_touched: usize,
}

#[derive(Clone, Debug, Default)]
pub struct RouterBytes {
    pub dense_f64: usize,
    pub dense_f32: usize,
    /// row = (scale f32, count u32, baseline f32) + touched cells (u16 agent, f32 w)
    pub sparse_f32: usize,
    pub n_rows: usize,
    pub n_cells: usize,
    pub n_touched: usize,
}

const NODE: usize = 32;

/// Byte estimates of a Hebbian router's weights + baselines in three layouts.
/// "Touched" cells are those whose weight differs from the value an untouched cell
/// of the same row would have (all cells of a row are scaled together by decay /
/// row-cap, so an exact sparse layout stores one row scale plus the touched cells).
pub fn router_bytes(r: &HebbianRouter) -> RouterBytes {
    let mut out = RouterBytes::default();
    for (k, row) in r.rows() {
        out.n_rows += 1;
        out.n_cells += row.len();
        let key = 24 + k.len() + NODE;
        out.dense_f64 += key + 24 + row.len() * 8 + (key + 8); // weights row + baseline entry
        out.dense_f32 += key + 24 + row.len() * 4 + (key + 4);
        // untouched cells share the row's minimum value (they only ever decay/rescale)
        let base = row.iter().cloned().fold(f64::INFINITY, f64::min);
        let touched = row.iter().filter(|&&w| (w - base).abs() > 1e-12).count();
        out.n_touched += touched;
        out.sparse_f32 += key + 4 + 4 + 4 + 24 + touched * 6;
    }
    out
}

// =============================================================================================
// Quantisation (memory-budget study): round every weight/baseline of a router to a grid.

/// Precision of a quantised router snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    F64,
    F32,
    /// 16-bit fixed point on [0,1] (weights and baselines live in [0,1]).
    U16,
    U8,
}

fn q(x: f64, p: Precision) -> f64 {
    match p {
        Precision::F64 => x,
        Precision::F32 => x as f32 as f64,
        Precision::U16 => (x.clamp(0.0, 1.0) * 65535.0).round() / 65535.0,
        Precision::U8 => (x.clamp(0.0, 1.0) * 255.0).round() / 255.0,
    }
}

fn quantise_value(v: &mut serde_json::Value, p: Precision) {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(x) = n.as_f64()
                && !n.is_u64()
                && !n.is_i64()
                && let Some(m) = serde_json::Number::from_f64(q(x, p))
            {
                *n = m;
            }
        }
        serde_json::Value::Array(a) => a.iter_mut().for_each(|x| quantise_value(x, p)),
        serde_json::Value::Object(o) => o.iter_mut().for_each(|(_, x)| quantise_value(x, p)),
        _ => {}
    }
}

/// Round the learned state (weights + baselines, NOT the config) of `r` to `p`.
pub fn quantise_router(r: &mut HebbianRouter, p: Precision) {
    if p == Precision::F64 {
        return;
    }
    let mut v = serde_json::to_value(&*r).expect("serialise router");
    for key in ["weights", "baselines"] {
        if let Some(x) = v.get_mut(key) {
            quantise_value(x, p);
        }
    }
    *r = serde_json::from_value(v).expect("deserialise router");
}

// =============================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn kw() -> HashMap<String, usize> {
        let mut m = HashMap::new();
        m.insert("a".to_string(), 0);
        m.insert("b".to_string(), 1);
        m.insert("c".to_string(), 2);
        m
    }

    fn task(i: u64) -> Task {
        let f = ["a", "b", "c"][(i % 3) as usize];
        Task { id: i, task_type: None, features: vec![f.to_string(), format!("x{}", i % 2)] }
    }

    /// Best agent per feature: a -> 1 (keyword says 0: a trap), b -> 1, c -> 2.
    fn p(t: &Task, a: usize) -> f64 {
        let best = match t.features[0].as_str() {
            "a" => 1,
            "b" => 1,
            _ => 2,
        };
        if a == best { 0.9 } else { 0.3 }
    }

    #[test]
    fn all_off_equals_bare_warm() {
        let cfg = CouplingConfig::default();
        let mut brain = CoupledBrain::new(3, kw(), cfg.clone());
        let mut bare = Warm::new(HebbianRouter::new(3, cfg.hebbian.clone()), kw(), 3, cfg.warm_k)
            .with_evidence(Evidence::MeanFeature);
        let (mut r1, mut r2, mut env) = (Rng::new(5), Rng::new(5), Rng::new(9));
        let c = [0, 1, 2];
        for i in 0..2000 {
            let t = task(i);
            let a = brain.route(&t, &c, &mut r1);
            let b = bare.select(&t, &c, &mut r2);
            assert_eq!(a, b, "diverged at {i}");
            let r = if env.bernoulli(p(&t, a)) { 1.0 } else { 0.0 };
            brain.feedback(&t, a, r);
            bare.update(&t, b, r);
        }
        assert_eq!(*brain.router.inner(), bare.inner);
    }

    #[test]
    fn health_benches_dead_agent_and_shield_protects_weights() {
        let cfg = CouplingConfig { health: true, shield: true, warm_k: 0.0, ..CouplingConfig::default() };
        let mut brain = CoupledBrain::new(3, kw(), cfg);
        let t = Task { id: 0, task_type: None, features: vec!["c".into()] };
        // learn (router only, health untouched) that agent 2 is good on "c"
        for _ in 0..200 {
            brain.router.update(&t, 2, 1.0);
            brain.router.update(&t, 0, 0.0);
        }
        // agents 0 and 1 are healthy (they work fine elsewhere)
        brain.feedback_split(&t, 0, 1.0, None);
        brain.feedback_split(&t, 1, 1.0, None);
        let w_before = brain.router.inner().weight("c", 2);
        // agent 2 goes down: repeated hard failures
        for _ in 0..30 {
            brain.feedback(&t, 2, 0.0);
        }
        assert!(brain.health(2) < 0.3 * brain.health(1));
        assert!(brain.shielded > 0, "outage failures should be shielded");
        // the first failures (before the bench threshold) still reach the router,
        // but the weight must not have collapsed
        assert!(brain.router.inner().weight("c", 2) > 0.5 * w_before);
        let mut rng = Rng::new(1);
        let a = brain.route(&t, &[0, 1, 2], &mut rng);
        assert_ne!(a, 2, "a benched agent must not be routed");
    }

    #[test]
    fn probe_gate_defers_probe_when_router_prefers_other() {
        let cfg = CouplingConfig { health: true, probe_gate: true, retry_every: 2, warm_k: 0.0, ..CouplingConfig::default() };
        let mut brain = CoupledBrain::new(3, kw(), cfg);
        let t = Task { id: 0, task_type: None, features: vec!["c".into()] };
        for _ in 0..200 {
            brain.feedback(&t, 1, 1.0); // router prefers 1 on "c"
        }
        for _ in 0..20 {
            brain.feedback(&t, 0, 0.0); // agent 0 is down
        }
        let mut rng = Rng::new(3);
        for _ in 0..10 {
            let a = brain.route(&t, &[0, 1, 2], &mut rng);
            assert_ne!(a, 0);
        }
        assert!(brain.probes_deferred > 0);
        assert_eq!(brain.probes, 0);
    }

    #[test]
    fn replay_returns_most_recent_identical_outcomes() {
        let mut m = ExperienceMemory::new(32, 0);
        let t = Task { id: 0, task_type: None, features: vec!["a".into(), "b".into()] };
        for i in 0..50 {
            m.add(&t, i % 3, i as f64 / 100.0);
        }
        let other = Task { id: 1, task_type: None, features: vec!["zzz".into()] };
        m.add(&other, 0, 1.0);
        let got = m.similar(&t, 3, 0.99);
        assert_eq!(got.len(), 3);
        let rw: Vec<f64> = got.iter().map(|e| e.reward).collect();
        assert_eq!(rw, vec![0.49, 0.48, 0.47]);
        // capacity evicts the oldest
        let mut c = ExperienceMemory::new(32, 10);
        for i in 0..40 {
            c.add(&t, 0, i as f64);
        }
        assert_eq!(c.len(), 10);
        let got = c.similar(&t, 20, 0.99);
        assert_eq!(got.len(), 10);
        assert_eq!(got[0].reward, 39.0);
    }

    #[test]
    fn replay_brain_runs_and_replays() {
        let mk = |n| CoupledBrain::new(3, kw(), CouplingConfig { replay_n: n, ..CouplingConfig::default() });
        let run = |mut b: CoupledBrain| {
            let (mut rng, mut env) = (Rng::new(2), Rng::new(4));
            let mut good = 0.0;
            for i in 0..300 {
                let t = task(i);
                let a = b.route(&t, &[0, 1, 2], &mut rng);
                good += p(&t, a);
                let r = if env.bernoulli(p(&t, a)) { 1.0 } else { 0.0 };
                b.feedback(&t, a, r);
            }
            good / 300.0
        };
        let (base, rep) = (run(mk(0)), run(mk(4)));
        // sanity only (the effect size is measured by bin x3_coupling, not asserted here)
        assert!(base > 0.3 && rep > 0.3, "replay {rep} vs base {base}");
        let mut b = mk(4);
        let mut rng = Rng::new(1);
        for i in 0..20 {
            let t = task(i);
            let a = b.route(&t, &[0, 1, 2], &mut rng);
            b.feedback(&t, a, 1.0);
        }
        assert!(b.replays > 0 && b.memory_len() == 20);
    }

    #[test]
    fn shaper_grades_by_latency_within_context() {
        let mut s = RewardShaper::new(ShapingConfig::default());
        let fast = Task { id: 0, task_type: None, features: vec!["slow-type".into()] };
        for i in 0..50 {
            s.immediate(&fast, 0, true, Some(10.0 * (1.0 + 0.1 * ((i % 5) as f64 - 2.0))));
        }
        let quick = s.immediate(&fast, 0, true, Some(6.0)).unwrap();
        let slow = s.immediate(&fast, 0, true, Some(18.0)).unwrap();
        assert!(quick > 0.8 && slow < 0.2, "quick {quick} slow {slow}");
        assert_eq!(s.immediate(&fast, 0, false, Some(1.0)), Some(0.0));
        // a different context with a different absolute speed is normalised separately
        let other = Task { id: 1, task_type: None, features: vec!["fast-type".into()] };
        for _ in 0..50 {
            s.immediate(&other, 0, true, Some(1.0));
        }
        let mid = s.immediate(&other, 0, true, Some(1.0)).unwrap();
        assert!((mid - 0.5).abs() < 0.2, "mid {mid}");
        let off = RewardShaper::new(ShapingConfig { use_latency: false, immediate_only_failures: true, ..Default::default() });
        let mut off = off;
        assert_eq!(off.immediate(&fast, 0, true, Some(3.0)), None);
    }

    #[test]
    fn debias_removes_agent_speed_confound() {
        // agent 1 is 2x slower everywhere but equally good; agent 0 normal.
        let mk = |d| RewardShaper::new(ShapingConfig { debias_agent: d, ..Default::default() });
        let ctx: Vec<Task> = (0..4).map(|i| Task { id: i, task_type: None, features: vec![format!("c{i}")] }).collect();
        let mut rs = [mk(false), mk(true)];
        let mut env = Rng::new(11);
        let mut acc = [[0.0; 2]; 2];
        for i in 0..400 {
            let t = &ctx[i % 4];
            for (j, s) in rs.iter_mut().enumerate() {
                for a in 0..2 {
                    let lat = (1.0 + a as f64) * (0.3 * (env.f64() - 0.5)).exp();
                    let r = s.immediate(t, a, true, Some(lat)).unwrap();
                    if i >= 300 {
                        acc[j][a] += r / 100.0;
                    }
                }
            }
        }
        // without debiasing the slow agent is punished; with debiasing both are ~neutral
        assert!(acc[0][1] < 0.25 && acc[0][0] > 0.75, "biased {:?}", acc[0]);
        assert!((acc[1][1] - 0.5).abs() < 0.2 && (acc[1][0] - 0.5).abs() < 0.2, "debiased {:?}", acc[1]);
    }

    #[test]
    fn guarded_front_composes() {
        let cfg = CouplingConfig { guard: Some(GuardConfig::default()), health: true, ..CouplingConfig::default() };
        let mut brain = CoupledBrain::new(3, kw(), cfg);
        let (mut rng, mut env) = (Rng::new(2), Rng::new(4));
        let mut late = 0.0;
        for i in 0..1500 {
            let t = task(i);
            let a = brain.route(&t, &[0, 1, 2], &mut rng);
            if i >= 1200 {
                late += p(&t, a);
            }
            let r = if env.bernoulli(p(&t, a)) { 1.0 } else { 0.0 };
            brain.feedback(&t, a, r);
        }
        // the trap ("a": keyword 0, best 1) must be escaped by the guard
        assert!(late / 300.0 > 0.75, "late {}", late / 300.0);
    }

    #[test]
    fn quantise_and_bytes() {
        let mut r = HebbianRouter::new(3, heb_robust_cfg());
        for i in 0..100 {
            r.update(&task(i), (i % 3) as usize, (i % 2) as f64);
        }
        let b = router_bytes(&r);
        assert!(b.dense_f32 < b.dense_f64 && b.n_rows > 0 && b.n_touched <= b.n_cells);
        let mut q8 = r.clone();
        quantise_router(&mut q8, Precision::U8);
        for (k, row) in q8.rows() {
            for (a, w) in row.iter().enumerate() {
                assert!(((w * 255.0) - (w * 255.0).round()).abs() < 1e-9);
                assert!((w - r.weight(k, a)).abs() <= 0.5 / 255.0 + 1e-12);
            }
        }
        let mut q64 = r.clone();
        quantise_router(&mut q64, Precision::F64);
        assert_eq!(q64, r);
    }
}
